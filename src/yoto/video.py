"""Video overlay rendering for cleaned tracking data.

Reads a video file alongside its cleaned tracking DataFrame and produces
an output video with per-tag labels, coloured trails, and frame
counters.  Uses a 3-thread pipeline (ffmpeg decoder -> main drawer ->
ffmpeg encoder) with YUV420 pipes for ~2-3x throughput over a
single-threaded ``cv2.VideoCapture``/``cv2.VideoWriter`` loop.
"""

from __future__ import annotations

import logging
import os
import queue
import select
import subprocess
import threading
import time
from typing import Any

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from yoto._types import FloatArray
from yoto.constants import (
    COL_CENTER_X,
    COL_CENTER_Y,
    DEFAULT_TRAIL_LENGTH,
    DEFAULT_TRAIL_SKIP,
    TAG_COLOR_SEED,
)
from yoto.exceptions import EncoderError, VideoReadError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _open_ffmpeg_writer(
    output_path: str,
    frame_width: int,
    frame_height: int,
    fps: float,
    codec: str = "auto",
) -> subprocess.Popen[bytes]:
    """Open an ffmpeg subprocess pipe fed with YUV420 frames.

    Default priority (``codec="auto"``): hevc_nvenc -> h264_nvenc ->
    libx264.  Each candidate is validated by sending a short burst of
    blank frames; NVENC fails lazily so a single-frame probe is not
    enough.

    Parameters
    ----------
    output_path : str
        Path for the output video file.
    frame_width, frame_height : int
        Output frame size in pixels.  Must be even (YUV420 requirement).
    fps : float
        Output frames per second.
    codec : {"auto", "hevc", "h264"}
        Encoder family to try.  ``"h264"`` is recommended for smooth
        VLC playback, ``"hevc"`` for smaller files.

    Returns
    -------
    subprocess.Popen
        ffmpeg process with stdin open for raw YUV420 frames.

    Raises
    ------
    EncoderError
        If no working encoder is found.
    """
    fps_str = str(fps)
    if "/" in fps_str:
        parts = fps_str.split("/")
        keyframe_interval = max(1, int(float(parts[0]) * 2 / float(parts[-1])))
    else:
        keyframe_interval = max(1, int(float(fps_str) * 2))

    base_cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "yuv420p",
        "-s",
        f"{frame_width}x{frame_height}",
        "-r",
        fps_str,
        "-i",
        "pipe:0",
        "-vsync",
        "cfr",
    ]
    from yoto import __version__

    compat = [
        "-g",
        str(keyframe_interval),
        "-movflags",
        "+faststart",
        "-metadata",
        f"comment=yoto={__version__}",
    ]
    hevc_nvenc = (
        "hevc_nvenc (GPU)",
        base_cmd
        + [
            "-vcodec",
            "hevc_nvenc",
            "-preset",
            "p1",
            "-rc",
            "vbr",
            "-cq",
            "28",
            "-pix_fmt",
            "yuv420p",
        ]
        + compat
        + [output_path],
    )
    h264_nvenc = (
        "h264_nvenc (GPU)",
        base_cmd
        + [
            "-vcodec",
            "h264_nvenc",
            "-preset",
            "p1",
            "-rc",
            "vbr",
            "-cq",
            "28",
            "-pix_fmt",
            "yuv420p",
        ]
        + compat
        + [output_path],
    )
    libx264 = (
        "libx264 ultrafast",
        base_cmd
        + [
            "-vcodec",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
        ]
        + compat
        + [output_path],
    )

    if codec == "h264":
        candidates = [h264_nvenc, libx264]
    elif codec == "hevc":
        candidates = [hevc_nvenc, libx264]
    elif codec == "auto":
        candidates = [hevc_nvenc, h264_nvenc, libx264]
    else:
        raise ValueError(
            f"Unknown codec {codec!r}; expected 'auto', 'hevc', or 'h264'."
        )

    # YUV420 black frame: Y=0, U=V=128 (neutral chroma).
    y_size = frame_width * frame_height
    uv_size = (frame_width // 2) * (frame_height // 2)
    blank = (
        np.zeros(y_size, dtype=np.uint8).tobytes()
        + np.full(uv_size * 2, 128, dtype=np.uint8).tobytes()
    )

    # NVENC checks device capabilities lazily (after the first frame is
    # buffered) so a single-frame probe can accept an encoder that will
    # later die with "No capable devices found".  Write a burst and give
    # ffmpeg ~1s to fail before declaring success.
    probe_frames = 4
    probe_timeout = 1.0

    for name, cmd in candidates:
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            assert proc.stdin is not None and proc.stderr is not None
            os.set_blocking(proc.stderr.fileno(), False)
            try:
                for _ in range(probe_frames):
                    proc.stdin.write(blank)
                proc.stdin.flush()
            except BrokenPipeError:
                proc.wait()
                continue

            start = time.monotonic()
            stderr_bytes = b""
            while time.monotonic() - start < probe_timeout:
                if proc.poll() is not None:
                    break
                try:
                    chunk = proc.stderr.read()
                except Exception:
                    chunk = None
                if chunk:
                    stderr_bytes += chunk
                    break
                select.select([proc.stderr], [], [], 0.1)

            if proc.poll() is not None or stderr_bytes:
                try:
                    remainder = proc.stderr.read()
                    if remainder:
                        stderr_bytes += remainder
                except Exception:
                    pass
                proc.wait()
                logger.debug(
                    "Encoder %s failed probe: %s",
                    name,
                    stderr_bytes.decode(errors="replace").strip() or "no stderr",
                )
                continue

            # Passed the probe — restore blocking and drain stderr in the
            # background so the pipe buffer never fills on long videos.
            os.set_blocking(proc.stderr.fileno(), True)
            threading.Thread(target=proc.stderr.read, daemon=True).start()
            logger.info("Encoder: %s", name)
            return proc
        except Exception:
            continue

    raise EncoderError()


def _open_ffmpeg_reader(
    video_path: str, frame_width: int, frame_height: int
) -> tuple[subprocess.Popen[bytes] | None, bytes | None]:
    """Open an ffmpeg decoder pipe, trying CUDA hwaccel then CPU.

    Returns ``(proc, first_frame_bytes)`` on success, or ``(None, None)``
    if every candidate fails.  The first frame is consumed as part of
    validation and must be replayed downstream.
    """
    byte_size = frame_width * frame_height * 3 // 2  # YUV420: 1.5 B/pixel
    candidates = [
        ("cuda (GPU)", ["-hwaccel", "cuda"]),
        ("CPU", []),
    ]
    for name, hw_args in candidates:
        cmd = (
            ["ffmpeg"]
            + hw_args
            + [
                "-i",
                video_path,
                "-f",
                "rawvideo",
                "-pix_fmt",
                "yuv420p",
                "-v",
                "quiet",
                "pipe:1",
            ]
        )
        proc: subprocess.Popen[bytes] | None = None
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=byte_size * 4,
            )
            assert proc.stdout is not None
            test = proc.stdout.read(byte_size)
            if len(test) == byte_size:
                logger.info("Decoder: ffmpeg %s", name)
                return proc, test
            proc.kill()
            proc.wait()
        except Exception:
            if proc is not None:
                try:
                    proc.kill()
                    proc.wait()
                except Exception:
                    pass
    return None, None


def _reader_worker_ffmpeg(
    proc: subprocess.Popen[bytes],
    byte_size: int,
    first_frame_bytes: bytes,
    frame_h: int,
    frame_w: int,
    short_mode: bool,
    read_q: "queue.Queue[Any]",
) -> None:
    """Decode frames from ffmpeg stdout into ``read_q``.

    Pushes ``(frame_num, yuv_ndarray)`` tuples, then a terminal ``None``.
    The reader does NOT convert YUV->BGR so this stays as fast as
    possible — conversion happens on the main thread.
    """
    assert proc.stdout is not None
    yuv = (
        np.frombuffer(first_frame_bytes, dtype=np.uint8)
        .reshape(frame_h * 3 // 2, frame_w)
        .copy()
    )
    read_q.put((0, yuv))
    fn = 1
    while True:
        if short_mode and fn > 2000:
            break
        raw = proc.stdout.read(byte_size)
        if len(raw) != byte_size:
            break
        yuv = (
            np.frombuffer(raw, dtype=np.uint8).reshape(frame_h * 3 // 2, frame_w).copy()
        )
        read_q.put((fn, yuv))
        fn += 1
    read_q.put(None)


def _reader_worker_cv2(
    cap: cv2.VideoCapture,
    do_resize: bool,
    out_w: int,
    out_h: int,
    short_mode: bool,
    read_q: "queue.Queue[Any]",
) -> None:
    """Fallback reader using ``cv2.VideoCapture`` (BGR output)."""
    fn = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if short_mode and fn > 2000:
            break
        if do_resize:
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        read_q.put((fn, frame))
        fn += 1
    read_q.put(None)


def _writer_worker(
    write_q: "queue.Queue[Any]",
    ffmpeg_proc: subprocess.Popen[bytes],
    error_box: dict[str, BaseException],
) -> None:
    """Drain ``write_q`` and feed YUV420 frames to ffmpeg stdin.

    Pure I/O — releases the GIL so encode overlaps with the main
    thread's drawing work.  Errors are recorded in *error_box* so the
    main thread can raise a proper :class:`EncoderError`.
    """
    assert ffmpeg_proc.stdin is not None
    try:
        while True:
            item = write_q.get()
            if item is None:
                break
            ffmpeg_proc.stdin.write(item.tobytes())
    except BrokenPipeError as exc:
        error_box["error"] = exc
    finally:
        try:
            ffmpeg_proc.stdin.close()
        except Exception:
            pass


def _build_numpy_cache(
    frame_data: pd.DataFrame,
    id_list: np.ndarray[Any, np.dtype[Any]],
) -> tuple[FloatArray, FloatArray, set[int], np.ndarray[Any, np.dtype[np.int64]], int]:
    """Pre-extract tracking coordinates into numpy arrays.

    Eliminates per-frame pandas ``.loc`` in the hot rendering loop.
    """
    n_ids = len(id_list)
    max_frame = int(frame_data.index.max())
    cx = np.full((max_frame + 1, n_ids), np.nan, dtype=np.float32)
    cy = np.full((max_frame + 1, n_ids), np.nan, dtype=np.float32)

    for i, tag_id in enumerate(id_list):
        col_x = frame_data[(tag_id, COL_CENTER_X)]
        col_y = frame_data[(tag_id, COL_CENTER_Y)]
        valid = col_x.notna()
        idxs = col_x.index[valid].to_numpy()
        cx[idxs, i] = col_x[valid].to_numpy(dtype=np.float32)
        cy[idxs, i] = col_y[valid].to_numpy(dtype=np.float32)

    nb_ids_per_frame = (~np.isnan(cx)).sum(axis=1)
    frame_set = set(frame_data.index.tolist())
    return cx, cy, frame_set, nb_ids_per_frame, max_frame


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_overlay_video(
    video_path: str,
    frame_data: pd.DataFrame,
    id_list: np.ndarray[Any, np.dtype[Any]],
    output_path: str | None = None,
    data_name: str = "apriltagDetect14",
    scale: float = 1.0,
    short: bool = False,
    trail_length: int = DEFAULT_TRAIL_LENGTH,
    trail_skip: int = DEFAULT_TRAIL_SKIP,
    draw_trails: bool = True,
    text_scale_factor: float = 1.0,
    codec: str = "auto",
) -> str:
    """Render a video with tracking overlays via a 3-thread pipeline.

    The hot loop runs ``ffmpeg decode -> draw -> ffmpeg encode`` across
    three threads connected by bounded queues.  YUV420 frames flow
    through the pipes (1.5 B/pixel vs 3 for BGR24) and YUV<->BGR
    conversion happens on the main thread so the writer's pipe I/O
    fully overlaps with drawing work.

    Parameters
    ----------
    video_path, frame_data, id_list, output_path, data_name, scale,
    short, trail_length, trail_skip, draw_trails, text_scale_factor,
    codec
        See module docs / CLI.

    Returns
    -------
    str
        Path to the written output video.

    Raises
    ------
    VideoReadError
        If the input video cannot be opened.
    EncoderError
        If no working ffmpeg encoder is found or the encoder dies
        mid-render.
    """
    if output_path is None:
        video_folder = os.path.dirname(video_path)
        basename = os.path.basename(video_path).rsplit(".", 1)[0]
        if scale != 1.0:
            suffix = f"_{data_name}_scaled{scale:.2f}.mp4"
        else:
            suffix = f"_{data_name}.mp4"
        output_path = os.path.join(video_folder, "video_output", basename + suffix)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Probe once via cv2 for size / fps / frame count metadata.
    cap_probe = cv2.VideoCapture(video_path)
    if not cap_probe.isOpened():
        raise VideoReadError(video_path)
    frame_width = int(cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap_probe.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_probe.release()

    # Encoders (and YUV420) require even dimensions.
    out_w = int(frame_width * scale) & ~1
    out_h = int(frame_height * scale) & ~1
    do_resize = out_w != frame_width or out_h != frame_height
    if do_resize:
        logger.info("Output resolution: %d x %d (scale=%.2f)", out_w, out_h, scale)

    nb_ids = len(id_list)

    logger.info("Pre-extracting data arrays...")
    cx, cy, frame_set, nb_ids_per_frame, max_frame = _build_numpy_cache(
        frame_data, id_list
    )
    if do_resize:
        cx = cx * scale
        cy = cy * scale

    rng = np.random.default_rng(TAG_COLOR_SEED)
    tag_colors = [tuple(int(c) for c in rng.integers(0, 255, 3)) for _ in id_list]

    thickness = np.array([1] * 5 + [1] * 5 + [2] * 10 + [3] * 30, dtype=np.int32)

    # Open decoder (ffmpeg + cuda hwaccel, then CPU, then cv2 fallback).
    ffmpeg_reader, first_frame_bytes = _open_ffmpeg_reader(
        video_path, frame_width, frame_height
    )
    use_ffmpeg_reader = ffmpeg_reader is not None
    cap_reader: cv2.VideoCapture | None = None
    if not use_ffmpeg_reader:
        logger.info("Decoder: OpenCV (CPU fallback)")
        cap_reader = cv2.VideoCapture(video_path)
        if not cap_reader.isOpened():
            raise VideoReadError(video_path)

    ffmpeg_proc = _open_ffmpeg_writer(output_path, out_w, out_h, fps, codec=codec)

    text_scale = max(1.0, 3 * scale) * text_scale_factor
    text_thick = max(1, int(12 * scale * text_scale_factor))

    from yoto._progress import make_status_updater

    short_limit = 2000
    status_total = min(short_limit + 1, total_frames) if short else total_frames
    status_update = make_status_updater(video_path, status_total)

    read_q: "queue.Queue[Any]" = queue.Queue(maxsize=16)
    write_q: "queue.Queue[Any]" = queue.Queue(maxsize=16)
    writer_error: dict[str, BaseException] = {}

    if use_ffmpeg_reader:
        assert ffmpeg_reader is not None and first_frame_bytes is not None
        reader_thread = threading.Thread(
            target=_reader_worker_ffmpeg,
            args=(
                ffmpeg_reader,
                frame_width * frame_height * 3 // 2,
                first_frame_bytes,
                frame_height,
                frame_width,
                short,
                read_q,
            ),
            daemon=True,
        )
    else:
        assert cap_reader is not None
        reader_thread = threading.Thread(
            target=_reader_worker_cv2,
            args=(cap_reader, do_resize, out_w, out_h, short, read_q),
            daemon=True,
        )
    writer_thread = threading.Thread(
        target=_writer_worker,
        args=(write_q, ffmpeg_proc, writer_error),
        daemon=True,
    )
    reader_thread.start()
    writer_thread.start()

    frame_num = 0
    try:
        with tqdm(
            total=total_frames,
            desc=f"Rendering {os.path.basename(video_path)}",
            disable=bool(os.environ.get("YOTO_NO_PROGRESS")),
        ) as pbar:
            while True:
                item = read_q.get()
                if item is None:
                    break
                fn, raw_frame = item

                if use_ffmpeg_reader:
                    frame = cv2.cvtColor(raw_frame, cv2.COLOR_YUV2BGR_I420)
                    if do_resize:
                        frame = cv2.resize(
                            frame,
                            (out_w, out_h),
                            interpolation=cv2.INTER_LINEAR,
                        )
                else:
                    frame = raw_frame

                for i, (tag_id, color) in enumerate(zip(id_list, tag_colors)):
                    x = cx[fn, i] if fn <= max_frame else np.nan
                    y = cy[fn, i] if fn <= max_frame else np.nan
                    if not np.isnan(x):
                        cv2.putText(
                            frame,
                            str(tag_id),
                            (int(x) + int(40 * scale), int(y) + int(40 * scale)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            text_scale,
                            (0, 0, 0),
                            text_thick,
                            cv2.LINE_AA,
                        )

                    if (
                        draw_trails
                        and fn > trail_length + trail_skip
                        and fn <= max_frame
                    ):
                        t0 = fn - trail_length
                        t1 = fn - trail_skip + 1
                        xs = cx[t0:t1, i]
                        ys = cy[t0:t1, i]
                        valid = ~(np.isnan(xs) | np.isnan(ys))
                        if valid.any():
                            th_slice = thickness[: len(xs)]
                            for th_val in np.unique(th_slice):
                                seg_mask = valid & (th_slice == th_val)
                                if not seg_mask.any():
                                    continue
                                seg_pts = np.column_stack(
                                    [xs[seg_mask], ys[seg_mask]]
                                ).astype(np.int32)
                                cv2.polylines(
                                    frame,
                                    [seg_pts.reshape(-1, 1, 2)],
                                    isClosed=False,
                                    color=color,
                                    thickness=int(th_val),
                                )

                n_visible = int(nb_ids_per_frame[fn]) if fn <= max_frame else 0
                cv2.putText(
                    frame,
                    str(fn),
                    (10, int(80 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    text_scale,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"{n_visible}/{nb_ids}",
                    (10, int(180 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    text_scale,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

                # BGR->YUV on the main thread so the writer only does I/O.
                yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
                write_q.put(yuv)

                frame_num += 1
                if frame_num % 100 == 0:
                    pbar.update(100)
                    status_update(frame_num)

                if writer_error:
                    break

            pbar.update(max(0, pbar.total - pbar.n))
    finally:
        write_q.put(None)
        writer_thread.join()
        reader_thread.join()
        ffmpeg_proc.wait()
        if use_ffmpeg_reader:
            assert ffmpeg_reader is not None
            try:
                if ffmpeg_reader.stdout is not None:
                    ffmpeg_reader.stdout.close()
            except Exception:
                pass
            ffmpeg_reader.wait()
        elif cap_reader is not None:
            cap_reader.release()

    if writer_error:
        raise EncoderError(
            f"ffmpeg died while writing frame {frame_num} "
            f"(exit={ffmpeg_proc.returncode}): {writer_error['error']!r}"
        ) from writer_error["error"]

    logger.info("Output: %s", output_path)
    return output_path
