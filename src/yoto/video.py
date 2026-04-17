"""Video overlay rendering for cleaned tracking data.

Reads a video file alongside its cleaned tracking DataFrame and produces
an output video with per-tag labels, coloured trails, and frame
counters.  Encoding uses ffmpeg via a subprocess pipe, preferring GPU
encoders (NVENC) with automatic CPU fallback.
"""

from __future__ import annotations

import logging
import os
import select
import subprocess
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
) -> subprocess.Popen[bytes]:
    """Open an ffmpeg subprocess pipe, trying GPU encoders first.

    Priority: hevc_nvenc -> h264_nvenc -> libx264 ultrafast.
    Each candidate is validated by sending one blank frame.

    Parameters
    ----------
    output_path : str
        Path for the output video file.
    frame_width : int
        Output frame width in pixels.
    frame_height : int
        Output frame height in pixels.
    fps : float
        Output frames per second.

    Returns
    -------
    subprocess.Popen
        ffmpeg process with stdin ready for raw BGR frames.

    Raises
    ------
    EncoderError
        If no working encoder is found.
    """
    fps_str = str(fps)
    # Compute keyframe interval (one keyframe every ~2 seconds)
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
        "bgr24",
        "-s",
        f"{frame_width}x{frame_height}",
        "-r",
        fps_str,
        "-i",
        "pipe:0",
        "-vsync",
        "cfr",
    ]
    compat = [
        "-g",
        str(keyframe_interval),
        "-movflags",
        "+faststart",
    ]
    candidates = [
        (
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
            ]
            + compat
            + [output_path],
        ),
        (
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
            ]
            + compat
            + [output_path],
        ),
        (
            "libx264 ultrafast",
            base_cmd
            + ["-vcodec", "libx264", "-preset", "ultrafast", "-crf", "23"]
            + compat
            + [output_path],
        ),
    ]

    blank = np.zeros(
        (frame_height, frame_width, 3), dtype=np.uint8
    ).tobytes()

    for name, cmd in candidates:
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            assert proc.stdin is not None
            proc.stdin.write(blank)
            proc.stdin.flush()
            # Brief check that the encoder didn't immediately crash
            ready, _, _ = select.select([proc.stderr], [], [], 0.5)
            if proc.poll() is not None:
                proc.stderr.read()  # type: ignore[union-attr]
                continue
            logger.info("Encoder: %s", name)
            return proc
        except Exception:
            continue

    raise EncoderError()


def _build_numpy_cache(
    frame_data: pd.DataFrame,
    id_list: np.ndarray[Any, np.dtype[Any]],
) -> tuple[FloatArray, FloatArray, set[int], np.ndarray[Any, np.dtype[np.int64]], int]:
    """Pre-extract tracking coordinates into numpy arrays.

    Eliminates per-frame pandas ``.loc`` in the hot rendering loop.

    Parameters
    ----------
    frame_data : pd.DataFrame
        Cleaned tracking DataFrame.
    id_list : ndarray
        Array of unique tag IDs.

    Returns
    -------
    tuple
        ``(cx, cy, frame_set, nb_ids_per_frame, max_frame)``
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
) -> str:
    """Render a video with tracking overlays (labels, coloured trails).

    Parameters
    ----------
    video_path : str
        Path to the original input video.
    frame_data : pd.DataFrame
        Cleaned tracking DataFrame (output of :func:`clean_tracking_data`).
    id_list : ndarray
        Array of unique tag IDs present in *frame_data*.
    output_path : str | None
        Full path for the output video.  When ``None`` a default path is
        generated in a ``video_output/`` subdirectory next to the input.
    data_name : str
        Label used in the default output filename.
    scale : float
        Output resolution scale factor (e.g. 0.5 = half size).
    short : bool
        When True, process only the first 2000 frames (for quick previews).
    trail_length : int
        Number of past frames to include in the trail.
    trail_skip : int
        Skip the most recent *trail_skip* frames (avoids clutter at the
        tag's current position).

    Returns
    -------
    str
        Path to the written output video file.

    Raises
    ------
    VideoReadError
        If the input video cannot be opened.
    EncoderError
        If no working ffmpeg encoder is found.

    Examples
    --------
    >>> from yoto.cleaning import clean_tracking_data
    >>> import pandas as pd
    >>> df = pd.read_pickle("tracking.pkl")  # doctest: +SKIP
    >>> cleaned, ids, _ = clean_tracking_data(df)  # doctest: +SKIP
    >>> render_overlay_video("input.mp4", cleaned, ids)  # doctest: +SKIP
    """
    # Resolve output path
    if output_path is None:
        video_folder = os.path.dirname(video_path)
        basename = os.path.basename(video_path).rsplit(".", 1)[0]
        if scale != 1.0:
            suffix = f"_{data_name}_scaled{scale:.2f}.mp4"
        else:
            suffix = f"_{data_name}.mp4"
        output_path = os.path.join(
            video_folder, "video_output", basename + suffix
        )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise VideoReadError(video_path)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Apply scale — ensure even dimensions (required by most encoders)
    out_w = int(frame_width * scale) & ~1
    out_h = int(frame_height * scale) & ~1
    do_resize = out_w != frame_width or out_h != frame_height
    if do_resize:
        logger.info("Output resolution: %d x %d (scale=%.2f)", out_w, out_h, scale)

    nb_ids = len(id_list)

    # Pre-extract data into numpy arrays
    logger.info("Pre-extracting data arrays...")
    cx, cy, frame_set, nb_ids_per_frame, max_frame = _build_numpy_cache(
        frame_data, id_list
    )
    if do_resize:
        cx = cx * scale
        cy = cy * scale

    # Deterministic per-tag colours
    rng = np.random.default_rng(TAG_COLOR_SEED)
    tag_colors = [
        tuple(int(c) for c in rng.integers(0, 255, 3)) for _ in id_list
    ]

    # Trail thickness ramp (oldest -> newest)
    thickness = np.array(
        [1] * 5 + [1] * 5 + [2] * 10 + [3] * 30, dtype=np.int32
    )

    # Open encoder
    ffmpeg_proc = _open_ffmpeg_writer(output_path, out_w, out_h, fps)
    assert ffmpeg_proc.stdin is not None

    text_scale = max(1.0, 3 * scale) * text_scale_factor
    text_thick = max(1, int(12 * scale * text_scale_factor))

    from yoto._progress import make_status_updater

    frame_num = 0
    short_limit = 2000
    status_total = short_limit + 1 if short else total_frames
    status_update = make_status_updater(video_path, status_total)
    with tqdm(
        total=total_frames,
        desc=f"Rendering {os.path.basename(video_path)}",
        disable=bool(os.environ.get("YOTO_NO_PROGRESS")),
    ) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if short and frame_num > short_limit:
                break

            fn = frame_num

            if do_resize:
                frame = cv2.resize(
                    frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR
                )

            # Draw labels and trails
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

                if draw_trails and fn > trail_length + trail_skip and fn <= max_frame:
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

            # Frame counter and visible-ID count
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

            ffmpeg_proc.stdin.write(frame.tobytes())

            frame_num += 1
            if frame_num % 100 == 0:
                pbar.update(100)
                status_update(frame_num)

    cap.release()
    ffmpeg_proc.stdin.close()
    ffmpeg_proc.wait()
    logger.info("Output: %s", output_path)
    return output_path
