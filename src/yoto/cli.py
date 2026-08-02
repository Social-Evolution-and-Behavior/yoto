"""Command-line interface for the YOTO package.

Provides three sub-commands:

* ``yoto detect``  — run the YOLO + AprilTag detection pipeline
* ``yoto clean``   — clean and interpolate raw tracking data
* ``yoto render``  — produce a video overlay from cleaned data

Each sub-command mirrors the corresponding public API function.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
from typing import Any, Callable

import pandas as pd

from yoto.constants import (
    DEFAULT_WEIGHTS,
    IMAGE_EXTENSIONS,
    TRACKING_DIR,
    TRACKING_SUBDIRS,
    VIDEO_EXTENSIONS,
)


def _tracking_layout(recording_dir: str) -> dict[str, str]:
    """Return the standard ``tracking/`` sub-paths for a recording folder.

    Paths are not created — that happens at write time.
    """
    base = os.path.join(os.path.abspath(recording_dir), TRACKING_DIR)
    return {k: os.path.join(base, v) for k, v in TRACKING_SUBDIRS.items()}


def _recording_dir_for_video(video_path: str) -> str:
    """Recording dir for a video = its parent directory."""
    return os.path.dirname(os.path.abspath(video_path))


def _recording_dir_for_pickle(pkl_path: str) -> str:
    """Recording dir for a pickle.

    If the pickle is inside ``tracking/raw_data/`` or ``tracking/clean_data/``
    the recording dir is two levels up.  Otherwise it is the pickle's parent.
    """
    abs_pkl = os.path.abspath(pkl_path)
    parent = os.path.dirname(abs_pkl)
    grandparent = os.path.dirname(parent)
    if (
        os.path.basename(parent) in TRACKING_SUBDIRS.values()
        and os.path.basename(grandparent) == TRACKING_DIR
    ):
        return os.path.dirname(grandparent)
    return parent


def _normalize_dataname(name: str) -> str:
    """Ensure the dataname suffix begins with an underscore so it is
    visually separated from the video stem in output filenames."""
    if not name:
        return name
    return name if name.startswith("_") else "_" + name


def _parse_index_spec(spec: str, n: int) -> list[int]:
    """Parse a ``--video-nb`` spec into a sorted list of unique indices.

    Accepts comma-separated 0-based indices and inclusive ranges, e.g.
    ``"3"``, ``"0,2,5"``, ``"0-9"``, ``"0-4,10,20-25"``.  Every index is
    validated against ``[0, n)``.  Raises :class:`ValueError` on malformed
    tokens or out-of-range indices.
    """
    indices: set[int] = set()
    for raw in spec.split(","):
        token = raw.strip()
        if not token:
            continue
        if "-" in token:
            lo_s, _, hi_s = token.partition("-")
            try:
                lo, hi = int(lo_s), int(hi_s)
            except ValueError:
                raise ValueError(f"--video-nb: invalid range {token!r}")
            if lo > hi:
                raise ValueError(f"--video-nb: reversed range {token!r} (lo > hi)")
            indices.update(range(lo, hi + 1))
        else:
            try:
                indices.add(int(token))
            except ValueError:
                raise ValueError(f"--video-nb: invalid index {token!r}")
    if not indices:
        raise ValueError("--video-nb: no indices given")
    lo, hi = min(indices), max(indices)
    if lo < 0 or hi >= n:
        raise ValueError(
            f"--video-nb index out of range: got {lo}..{hi}, valid "
            f"indices 0..{n - 1} ({n} item(s) found)"
        )
    return sorted(indices)


def _apply_video_nb(video_paths: list[str], spec: str | None) -> list[str]:
    """Filter *video_paths* by a ``--video-nb`` spec (or return all).

    Prints the selection and exits with a clear message on a bad spec.
    """
    if spec is None:
        return video_paths
    try:
        indices = _parse_index_spec(spec, len(video_paths))
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    selected = [video_paths[i] for i in indices]
    if len(selected) == 1:
        print(f"Selected video [{indices[0]}]: {selected[0]}")
    else:
        print(
            f"Selected {len(selected)} video(s) by --video-nb {spec!r}: "
            f"indices {indices}"
        )
    return selected


def _str_to_bool(s: str) -> bool:
    """argparse type converter — accepts ``True/False`` (any case),
    plus ``1/0``, ``yes/no``, ``on/off``.  Raises a friendly error
    otherwise.  Used for ``--flag True|False`` style toggles so the
    CLI presents all booleans uniformly."""
    if s.lower() in ("true", "1", "yes", "on"):
        return True
    if s.lower() in ("false", "0", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected (True/False), got {s!r}")


#: Named colors in BGR (OpenCV native) for the render highlight flag.
_NAMED_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
    "cyan": (255, 255, 0),
    "magenta": (255, 0, 255),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
}


def _parse_id_list(
    tokens: list[str] | None, flag_name: str = "--highlight-ids"
) -> list[int]:
    """Parse a space- or comma-separated list of tag IDs.

    argparse hands us ``nargs="+"`` tokens; each token may itself be a
    comma-separated group (``"42,87"``).  Flatten, split, strip, and
    convert to ``int``.  Empty input returns ``[]``.  ``flag_name`` is
    used only in error messages so the right flag gets the blame.
    """
    if not tokens:
        return []
    ids: list[int] = []
    for token in tokens:
        for part in token.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                ids.append(int(part))
            except ValueError as exc:
                raise argparse.ArgumentTypeError(
                    f"{flag_name}: expected integer IDs, got {part!r}"
                ) from exc
    return ids


def _parse_color_bgr(s: str) -> tuple[int, int, int]:
    """Parse a color spec into a BGR triple.

    Accepts a name from :data:`_NAMED_COLORS_BGR` (case-insensitive) or
    a comma-separated ``R,G,B`` triple in 0-255.  The R,G,B form is
    user-facing (matches what people read off color pickers); the
    returned tuple is BGR because OpenCV draws in BGR.
    """
    key = s.strip().lower()
    if key in _NAMED_COLORS_BGR:
        return _NAMED_COLORS_BGR[key]
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"--highlight-color: expected a named color or 'R,G,B', got {s!r}"
        )
    try:
        r, g, b = (int(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--highlight-color: R,G,B values must be integers, got {s!r}"
        ) from exc
    for v in (r, g, b):
        if not 0 <= v <= 255:
            raise argparse.ArgumentTypeError(
                f"--highlight-color: each channel must be 0-255, got {s!r}"
            )
    return (b, g, r)


def _resolve_video_paths(path: str) -> list[str]:
    """Resolve *path* to a list of video file paths.

    Handles three cases:

    1. *path* is a video file → ``[path]``
    2. *path* is a directory that directly contains video files → sorted list
    3. *path* is a directory whose immediate children are sub-directories,
       each containing video files → sorted list across all sub-dirs
    """
    if os.path.isfile(path):
        return [path]

    if not os.path.isdir(path):
        return [path]  # let downstream raise a clear error

    # Look for video files directly inside the directory.  Hidden files
    # (leading '.') are skipped — this filters out macOS AppleDouble
    # stubs like ``._000002.mp4`` that crash the NVDEC decoder.
    direct_videos = sorted(
        os.path.join(path, f)
        for f in os.listdir(path)
        if not f.startswith(".")
        and os.path.isfile(os.path.join(path, f))
        and os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS
    )
    if direct_videos:
        return direct_videos

    # Otherwise look one level deeper (sub-directories of recordings)
    videos: list[str] = []
    for entry in sorted(os.listdir(path)):
        if entry.startswith("."):
            continue
        subdir = os.path.join(path, entry)
        if not os.path.isdir(subdir):
            continue
        videos.extend(
            sorted(
                os.path.join(subdir, f)
                for f in os.listdir(subdir)
                if not f.startswith(".")
                and os.path.isfile(os.path.join(subdir, f))
                and os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS
            )
        )
    return videos


def _resolve_pickle_paths(path: str, data_suffix: str = "") -> list[str]:
    """Resolve *path* to a list of raw-detection pickle files.

    For a directory, prefers ``{path}/tracking/raw_data/`` when it exists;
    otherwise looks at pickles in the directory itself; otherwise descends
    one level into sub-folders (each treated as its own recording dir).
    Files ending ``_clean.pkl`` are always excluded.
    """

    def _is_raw(f: str) -> bool:
        # Exclude sidecars: <stem>_quads.pkl and <stem>_yolo.pkl are
        # produced by `yoto detect` and are not cleanable input.
        # When data_suffix is set, also drop pickles from other detect
        # runs in the same dir (e.g. `_8nNewSet_IR` alongside `_yoto_0.10.x`).
        if (
            f.startswith(".")
            or not f.endswith(".pkl")
            or f.endswith("_clean.pkl")
            or f.endswith("_quads.pkl")
            or f.endswith("_yolo.pkl")
        ):
            return False
        if data_suffix and not f[: -len(".pkl")].endswith(data_suffix):
            return False
        return True

    def _pickles_in(d: str) -> list[str]:
        return sorted(
            os.path.join(d, f)
            for f in os.listdir(d)
            if os.path.isfile(os.path.join(d, f)) and _is_raw(f)
        )

    def _find_in_recording(d: str) -> list[str]:
        tracking_raw = _tracking_layout(d)["raw_data"]
        if os.path.isdir(tracking_raw):
            return _pickles_in(tracking_raw)
        return _pickles_in(d)

    if os.path.isfile(path):
        # If a video file was passed, find its raw-detection pickle.
        if os.path.splitext(path)[1].lower() in VIDEO_EXTENSIONS:
            pkl = _find_pickle_for_video(path, data_suffix, raw_only=True)
            if pkl is None:
                return [path]  # let downstream raise a clear error
            return [pkl]
        return [path]
    if not os.path.isdir(path):
        return [path]  # let downstream raise a clear error

    direct = _find_in_recording(path)
    if direct:
        return direct

    pkls: list[str] = []
    for entry in sorted(os.listdir(path)):
        subdir = os.path.join(path, entry)
        if not os.path.isdir(subdir):
            continue
        pkls.extend(_find_in_recording(subdir))
    return pkls


def _find_pickle_for_video(
    video_path: str,
    data_suffix: str,
    raw_only: bool = False,
) -> str | None:
    """Locate the tracking pickle for *video_path*.

    Default preference order:

    1. ``<recording>/tracking/clean_data/<stem><data_suffix>_clean.pkl``
    2. ``<recording>/tracking/raw_data/<stem><data_suffix>.pkl``
    3. legacy path next to the video: ``<recording>/<stem><data_suffix>.pkl``

    When *raw_only* is True the clean_data candidate is skipped so the
    caller gets the un-interpolated raw detections.

    Returns the first existing path, or ``None`` if none are found.
    """
    recording = _recording_dir_for_video(video_path)
    stem = os.path.splitext(os.path.basename(video_path))[0]
    layout = _tracking_layout(recording)
    candidates = [
        os.path.join(layout["raw_data"], f"{stem}{data_suffix}.pkl"),
        os.path.join(recording, f"{stem}{data_suffix}.pkl"),
    ]
    if not raw_only:
        candidates.insert(
            0,
            os.path.join(layout["clean_data"], f"{stem}{data_suffix}_clean.pkl"),
        )
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def _configure_logging(debug: bool = False) -> None:
    """Set up root logger for CLI usage.

    Parameters
    ----------
    debug : bool
        When True, set level to DEBUG; otherwise INFO.
    """
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Sub-command: detect
# ---------------------------------------------------------------------------


def _add_detect_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``detect`` sub-command."""
    p = subparsers.add_parser(
        "detect",
        help="Run YOLO + AprilTag detection on a video",
    )
    p.add_argument(
        "video",
        help="Path to a video file, a directory of videos, "
        "or a directory of recording sub-folders",
    )
    p.add_argument(
        "output_dir",
        nargs="?",
        default=None,
        help="Output directory (default: same as video)",
    )
    p.add_argument(
        "--yoloweights",
        default=DEFAULT_WEIGHTS,
        help=f"Path to YOLO weights (default: {DEFAULT_WEIGHTS})",
    )
    p.add_argument(
        "--dataname",
        default="_apriltagDetect14",
        help="Suffix for output data file",
    )
    p.add_argument(
        "--use-nvdec",
        type=_str_to_bool,
        default=True,
        metavar="BOOL",
        help="Use the NVDEC + GPU-resident preprocessing pipeline for "
        "maximum throughput (default: True; requires PyNvVideoCodec and "
        "an NVIDIA GPU with NVDEC support).  Set to False to fall back "
        "to the portable Ultralytics streaming pipeline, which works on "
        "any CUDA-capable machine but is ~30 percent slower.",
    )
    p.add_argument(
        "--parallel",
        type=int,
        default=None,
        metavar="N",
        help="Process N videos concurrently in separate worker processes "
        "(no default — must be set explicitly)",
    )
    p.add_argument(
        "--video-nb",
        type=str,
        default=None,
        metavar="SPEC",
        help="Process only the video(s) at these 0-based indices in the "
        "resolved list. Accepts a single index, a comma list, and/or "
        "inclusive ranges: e.g. '3', '0,2,5', '0-9', '0-4,10,20-25' "
        "(useful for re-running failed videos).",
    )
    p.add_argument(
        "--apriltag-preset",
        default=None,
        metavar="NAME_OR_PATH",
        help="AprilTag preset to apply on top of the pipeline defaults. "
        "Either a built-in preset name (e.g. 'ir') or a path to a JSON "
        "file (e.g. an Optuna best_params_*.json). Defaults stay untouched "
        "when omitted.",
    )
    from yoto.constants import (
        DEFAULT_CONF_THRESHOLD,
        DEFAULT_IOU_THRESHOLD,
        DEFAULT_MAX_TAG_OFFSET_RATIO,
        DEFAULT_PAD_RATIO,
        DEFAULT_TAG_FAMILY,
    )

    p.add_argument(
        "--tag-family",
        default=DEFAULT_TAG_FAMILY,
        metavar="NAME",
        help=f"AprilTag family passed to the decoder "
        f"(default: {DEFAULT_TAG_FAMILY}). Change this to decode a "
        "different tag family (e.g. 'tag25h9', 'tag36h11').",
    )
    p.add_argument(
        "--max-tag-id",
        type=int,
        default=None,
        metavar="N",
        help=f"Maximum tag ID to keep; decodes with higher IDs are "
        f"discarded as out-of-family misdecodes. Default depends on "
        f"--tag-family: 237 for {DEFAULT_TAG_FAMILY!r} (ARTag), 999 for "
        "any other family.",
    )
    p.add_argument(
        "--silence-ids",
        nargs="+",
        default=None,
        metavar="ID",
        help="Tag IDs to drop unconditionally (in addition to the "
        "--max-tag-id cutoff). Use this for IDs known to be prone to "
        "misdecode in your setup. Space- or comma-separated, mixable: "
        "'--silence-ids 12 45', '--silence-ids 12,45,67', or "
        "'--silence-ids 12,45 67'.",
    )
    p.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONF_THRESHOLD,
        metavar="FLOAT",
        help=f"YOLO confidence threshold (default: {DEFAULT_CONF_THRESHOLD})",
    )
    p.add_argument(
        "--pad-ratio",
        type=float,
        default=DEFAULT_PAD_RATIO,
        metavar="FLOAT",
        help=f"Per-axis padding ratio added around each YOLO box before "
        f"cropping for AprilTag decoding (default: {DEFAULT_PAD_RATIO}). "
        "Each side grows by ratio * box_dim, so padding scales with "
        "apparent tag size — the same value works across camera heights. "
        "Larger values give the decoder more context but slow it down and "
        "risk neighbouring tags landing in the same crop.",
    )
    p.add_argument(
        "--max-tag-offset-ratio",
        type=float,
        default=DEFAULT_MAX_TAG_OFFSET_RATIO,
        metavar="FLOAT",
        help=f"Drop AprilTag decodes whose center is farther from the source "
        f"YOLO box center than `ratio * min(box_w, box_h)` "
        f"(default: {DEFAULT_MAX_TAG_OFFSET_RATIO}). Catches misdecodes "
        "where AprilTag found a quad in the padding region rather than the "
        "actual tag. The dropped box stays in the undecoded pool for "
        "yoto clean to pick up via YOLO-fill.",
    )
    p.add_argument(
        "--tag-offset-filter",
        type=_str_to_bool,
        default=True,
        metavar="BOOL",
        help="Apply the box-center offset filter that drops AprilTag "
        "decodes farther than `--max-tag-offset-ratio * min(box_w, "
        "box_h)` from their source YOLO box center (default: True). "
        "Set to False to accept every decode regardless of distance — "
        "useful for diagnostics, equivalent to "
        "`--max-tag-offset-ratio inf`.",
    )
    p.add_argument(
        "--iou",
        type=float,
        default=DEFAULT_IOU_THRESHOLD,
        metavar="FLOAT",
        help=f"YOLO NMS IoU threshold (default: {DEFAULT_IOU_THRESHOLD}). "
        "Lower = more aggressive duplicate suppression.",
    )
    from yoto.constants import DEFAULT_BATCH_SIZE

    p.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        metavar="N",
        help=f"Frames per GPU batch for the fast pipeline "
        f"(default: {DEFAULT_BATCH_SIZE}). Larger batches amortise NVDEC "
        "fetch, NMS dispatch and Python overhead but cost more VRAM. "
        "Ignored when --use-nvdec False.",
    )
    p.add_argument(
        "--save-yolo",
        type=_str_to_bool,
        default=True,
        metavar="BOOL",
        help="Write the _yolo.pkl sidecar (default: True). Set False to skip "
        "the GPU→CPU confidence transfer and reduce disk I/O — safe when "
        "you will not use `yoto clean --yolo-fill True`.",
    )
    p.add_argument(
        "--save-quads",
        type=_str_to_bool,
        default=False,
        metavar="BOOL",
        help="Write the _quads.pkl sidecar with raw AprilTag quads "
        "(default: False). Only needed for `yoto render --quads`.",
    )
    p.add_argument(
        "--no-yolo",
        action="store_true",
        help="Skip YOLO and run AprilTag on the full frame (simple/portable "
        "pipeline only; --use-nvdec is ignored when this flag is set)",
    )
    p.add_argument(
        "--no-enhance",
        action="store_true",
        help="Skip all image enhancement (sharpening, contrast, and any "
        "preset pre-stages) and run the AprilTag decoder on the raw "
        "grayscale crop/frame. Useful for diagnostics or already-clean "
        "input.",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug profiling output",
    )
    p.set_defaults(func=_run_detect)


def _run_single_video(
    vpath: str,
    use_nvdec: bool,
    output_dir: str | None,
    yolo_weights: str,
    data_suffix: str,
    debug: bool,
    preset: str | None = None,
    conf: float | None = None,
    iou: float | None = None,
    pad_ratio: float | None = None,
    max_tag_offset_ratio: float | None = None,
    save_yolo: bool = True,
    save_quads: bool = False,
    tag_family: str | None = None,
    batch_size: int | None = None,
    max_tag_id: int | None = None,
    silence_ids: list[int] | None = None,
    no_yolo: bool = False,
    no_enhance: bool = False,
) -> tuple[str, str | None]:
    """Run detection on one video. Returns ``(vpath, None)`` on success,
    or ``(vpath, traceback_text)`` on failure."""
    import traceback

    from yoto.constants import (
        DEFAULT_BATCH_SIZE,
        DEFAULT_CONF_THRESHOLD,
        DEFAULT_IOU_THRESHOLD,
        DEFAULT_MAX_TAG_OFFSET_RATIO,
        DEFAULT_PAD_RATIO,
        DEFAULT_TAG_FAMILY,
    )

    if conf is None:
        conf = DEFAULT_CONF_THRESHOLD
    if iou is None:
        iou = DEFAULT_IOU_THRESHOLD
    if pad_ratio is None:
        pad_ratio = DEFAULT_PAD_RATIO
    if max_tag_offset_ratio is None:
        max_tag_offset_ratio = DEFAULT_MAX_TAG_OFFSET_RATIO
    if tag_family is None:
        tag_family = DEFAULT_TAG_FAMILY
    if batch_size is None:
        batch_size = DEFAULT_BATCH_SIZE

    if output_dir is None:
        output_dir = _tracking_layout(_recording_dir_for_video(vpath))["raw_data"]

    try:
        if no_yolo:
            from yoto.detection import run_detection_simple

            run_detection_simple(
                video_path=vpath,
                output_path=output_dir,
                yolo_weights=yolo_weights,
                data_suffix=data_suffix,
                preset=preset,
                conf_threshold=conf,
                iou_threshold=iou,
                pad_ratio=pad_ratio,
                max_offset_ratio=max_tag_offset_ratio,
                save_yolo=False,
                save_quads=False,
                tag_family=tag_family,
                max_tag_id=max_tag_id,
                silence_ids=silence_ids,
                no_yolo=True,
                no_enhance=no_enhance,
            )
        elif use_nvdec:
            from yoto.detection import run_detection_fast

            run_detection_fast(
                video_path=vpath,
                output_path=output_dir,
                yolo_weights=yolo_weights,
                data_suffix=data_suffix,
                debug=debug,
                preset=preset,
                conf_threshold=conf,
                iou_threshold=iou,
                pad_ratio=pad_ratio,
                max_offset_ratio=max_tag_offset_ratio,
                save_yolo=save_yolo,
                save_quads=save_quads,
                tag_family=tag_family,
                batch_size=batch_size,
                max_tag_id=max_tag_id,
                silence_ids=silence_ids,
                no_enhance=no_enhance,
            )
        else:
            from yoto.detection import run_detection_simple

            run_detection_simple(
                video_path=vpath,
                output_path=output_dir,
                yolo_weights=yolo_weights,
                data_suffix=data_suffix,
                preset=preset,
                conf_threshold=conf,
                iou_threshold=iou,
                pad_ratio=pad_ratio,
                max_offset_ratio=max_tag_offset_ratio,
                save_yolo=save_yolo,
                save_quads=save_quads,
                tag_family=tag_family,
                max_tag_id=max_tag_id,
                silence_ids=silence_ids,
                no_enhance=no_enhance,
            )
        return (vpath, None)
    except Exception:
        return (vpath, traceback.format_exc())


def _build_worker_cmd(
    vpath: str,
    args: argparse.Namespace,
) -> list[str]:
    """Build a ``yoto detect`` command for a single video (used by GNU parallel)."""
    cmd = [sys.executable, "-m", "yoto.cli", "detect", vpath]
    if args.output_dir:
        cmd.append(args.output_dir)
    cmd.extend(["--yoloweights", args.yoloweights])
    cmd.extend(["--dataname", args.dataname])
    cmd.extend(["--use-nvdec", str(args.use_nvdec)])
    if args.debug:
        cmd.append("--debug")
    if getattr(args, "apriltag_preset", None):
        cmd.extend(["--apriltag-preset", args.apriltag_preset])
    if getattr(args, "conf", None) is not None:
        cmd.extend(["--conf", str(args.conf)])
    if getattr(args, "iou", None) is not None:
        cmd.extend(["--iou", str(args.iou)])
    if getattr(args, "pad_ratio", None) is not None:
        cmd.extend(["--pad-ratio", str(args.pad_ratio)])
    if getattr(args, "max_tag_offset_ratio", None) is not None:
        cmd.extend(["--max-tag-offset-ratio", str(args.max_tag_offset_ratio)])
    if getattr(args, "tag_family", None):
        cmd.extend(["--tag-family", args.tag_family])
    if getattr(args, "batch_size", None) is not None:
        cmd.extend(["--batch-size", str(args.batch_size)])
    if getattr(args, "max_tag_id", None) is not None:
        cmd.extend(["--max-tag-id", str(args.max_tag_id)])
    silence = _parse_id_list(getattr(args, "silence_ids", None), "--silence-ids")
    if silence:
        cmd.extend(["--silence-ids", ",".join(str(i) for i in silence)])
    cmd.extend(["--save-yolo", str(args.save_yolo)])
    cmd.extend(["--save-quads", str(args.save_quads)])
    return cmd


def _format_duration(seconds: float) -> str:
    """Format a duration in seconds as a short human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def _format_tqdm_line(current: int, total: int, elapsed: float) -> str:
    """Format a tqdm-like progress string:
    ``'164/18000 [00:03<07:04, 42.06it/s]'``."""

    def _hms(s: float) -> str:
        s_int = max(0, int(s))
        h, rem = divmod(s_int, 3600)
        m, sec = divmod(rem, 60)
        if h:
            return f"{h:02d}:{m:02d}:{sec:02d}"
        return f"{m:02d}:{sec:02d}"

    rate = current / elapsed if elapsed > 0 else 0.0
    remaining = (total - current) / rate if rate > 0 and total > current else 0.0
    return f"{current}/{total} " f"[{_hms(elapsed)}<{_hms(remaining)}, {rate:.2f}it/s]"


def _write_progress_txt(
    progress_path: str,
    video_paths: list[str],
    joblog_path: str,
    start_time: float,
    jobs: int,
    status_dir: str | None = None,
    final_wall_time: float | None = None,
) -> None:
    """Write a simple human-readable progress summary based on the joblog.

    Refreshed every few seconds during a parallel run so users can check
    progress with ``cat`` or ``tail -f`` without parsing a TSV.
    """
    import datetime
    import time as _time

    completed: list[tuple[int, str, float, str]] = []
    if os.path.isfile(joblog_path):
        try:
            with open(joblog_path) as f:
                for line in f.readlines()[1:]:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 7:
                        continue
                    try:
                        seq = int(parts[0])
                    except ValueError:
                        continue
                    try:
                        runtime = float(parts[3])
                    except ValueError:
                        runtime = 0.0
                    exitval = parts[6]
                    if 1 <= seq <= len(video_paths):
                        completed.append((seq, video_paths[seq - 1], runtime, exitval))
        except OSError:
            pass

    done_seqs = {s for s, *_ in completed}
    ok_count = sum(1 for _, _, _, e in completed if e == "0")
    fail_count = len(completed) - ok_count
    elapsed = (
        final_wall_time if final_wall_time is not None else _time.time() - start_time
    )
    status = "FINISHED" if final_wall_time is not None else "RUNNING"

    lines: list[str] = []
    lines.append(f"YOTO parallel run — {len(video_paths)} video(s), {jobs} worker(s)")
    lines.append(
        f"Started:  "
        f"{datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}"
    )
    lines.append(f"Updated:  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Elapsed:  {_format_duration(elapsed)}")
    lines.append(f"Status:   {status}")
    lines.append("")
    lines.append(
        f"Progress: {len(completed)}/{len(video_paths)} complete "
        f"({ok_count} ok, {fail_count} failed)"
    )

    if completed:
        lines.append("")
        lines.append("Completed (in finish order):")
        # Sort by completion time = starttime + runtime, ascending.
        for seq, vpath, runtime, exitval in sorted(completed, key=lambda x: x[0]):
            marker = "OK  " if exitval == "0" else f"FAIL{exitval:>2}"
            lines.append(
                f"  [{seq:>3} {marker}] {_format_duration(runtime):>10}  "
                f"{os.path.basename(vpath)}"
            )

    pending = [
        (i + 1, video_paths[i])
        for i in range(len(video_paths))
        if (i + 1) not in done_seqs
    ]
    if pending and final_wall_time is None:
        from yoto._progress import read_status

        active: list[tuple[int, str, str]] = []
        queued: list[tuple[int, str]] = []
        for seq, vpath in pending:
            state = read_status(status_dir, vpath) if status_dir is not None else None
            if state is not None:
                current, total, p_start, updated = state
                # Treat a stale status file (>15s since last write) as queued.
                if _time.time() - updated <= 15.0:
                    active.append(
                        (
                            seq,
                            vpath,
                            _format_tqdm_line(current, total, updated - p_start),
                        )
                    )
                    continue
            queued.append((seq, vpath))

        lines.append("")
        lines.append(f"Remaining: {len(active)} running, {len(queued)} queued")
        name_w = max((len(os.path.basename(p)) for _, p, _ in active), default=0)
        for seq, vpath, tq in active:
            lines.append(f"  [{seq:>3}] {os.path.basename(vpath):<{name_w}}  {tq}")
        for seq, vpath in queued[:10]:
            lines.append(f"  [{seq:>3}] {os.path.basename(vpath)}  (queued)")
        if len(queued) > 10:
            lines.append(f"  ... and {len(queued) - 10} more queued")

    try:
        with open(progress_path, "w") as f:
            f.write("\n".join(lines) + "\n")
    except OSError:
        pass


def _run_parallel_gnu(
    video_paths: list[str],
    worker_tmpl: list[str],
    jobs: int,
    input_root: str,
    results_root_tmpl: str = "{//}",
    recording_dir_for: Callable[[str], str] = _recording_dir_for_video,
) -> tuple[list[tuple[str, str | None]], dict[str, float], float]:
    """Dispatch one subprocess per input item via GNU parallel.

    ``worker_tmpl`` is the full ``yoto <sub>`` command with ``{}`` as the
    item placeholder (a video path for ``detect``/``render``, a raw pickle
    for ``clean``).  Each worker is a fully independent OS process, so a
    crash in one has no effect on the others.  Exit codes are captured via
    ``--joblog`` and mapped back to items.  A human-readable
    ``progress.txt`` is refreshed every 3 seconds alongside the joblog.

    ``results_root_tmpl`` is a GNU-parallel template that expands to each
    item's *recording root* (where its ``tracking/logs/`` lives).  It is
    ``"{//}"`` (the item's own dir) for videos, whose dir *is* the
    recording root; ``clean`` passes ``"{//}/../.."`` because its pickles
    sit two levels down in ``tracking/raw_data/``.  ``recording_dir_for``
    is the Python equivalent, used to reconstruct log paths for failures.
    """
    import datetime
    import shutil
    import subprocess
    import threading
    import time

    if shutil.which("parallel") is None:
        print(
            "Error: --parallel requires GNU parallel.\n"
            "Install on Ubuntu with: sudo apt install parallel"
        )
        sys.exit(1)

    # Everything for this run lives under a single dated folder.
    logs_dir = _tracking_layout(input_root)["logs"]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(logs_dir, f"yoto-parallel-{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    joblog = os.path.join(run_dir, "joblog.tsv")
    progress_txt = os.path.join(run_dir, "progress.txt")
    status_dir = os.path.join(run_dir, "status")
    os.makedirs(status_dir, exist_ok=True)

    # Per-worker stdout/stderr: each video's logs land under its OWN
    # recording dir (tracking/logs/yoto-parallel-<timestamp>/<seq>-<stem>).
    # GNU parallel's {//} placeholder expands to the video's dirname.
    worker_results = os.path.join(
        results_root_tmpl,
        TRACKING_DIR,
        TRACKING_SUBDIRS["logs"],
        f"yoto-parallel-{timestamp}",
        "{#}-{/.}",
    )

    parallel_cmd = [
        "parallel",
        "--jobs",
        str(jobs),
        "--joblog",
        joblog,
        "--results",
        worker_results,
        "-q",
        *worker_tmpl,
        ":::",
        *video_paths,
    ]

    print(f"Launching {jobs} GNU parallel workers " f"for {len(video_paths)} video(s).")
    print("")
    print("Check progress at any time with:")
    print(f"  cat {progress_txt}")
    print(f"  watch -n 5 cat {progress_txt}")
    print("")
    print(f"Run folder: {run_dir}/")
    print("  progress.txt — human-readable summary (refreshed every 3s)")
    print("  joblog.tsv   — GNU parallel joblog (one line per finished video)")
    print(
        "  <recording>/tracking/logs/"
        f"yoto-parallel-{timestamp}/<seq>-<stem>/stdout  "
        "— per-worker output"
    )
    print("")

    # Disable in-worker tqdm so per-video log files stay readable; workers
    # publish their progress via YOTO_STATUS_DIR status files instead.
    worker_env = os.environ.copy()
    worker_env["YOTO_NO_PROGRESS"] = "1"
    worker_env["YOTO_STATUS_DIR"] = status_dir

    wall_start = time.time()

    # Seed progress.txt before parallel starts so `cat` works immediately.
    _write_progress_txt(
        progress_txt,
        video_paths,
        joblog,
        wall_start,
        jobs,
        status_dir=status_dir,
    )

    # Background thread refreshes progress.txt every 3s while parallel runs.
    stop_event = threading.Event()

    def _watch() -> None:
        while not stop_event.wait(3.0):
            _write_progress_txt(
                progress_txt,
                video_paths,
                joblog,
                wall_start,
                jobs,
                status_dir=status_dir,
            )

    watcher = threading.Thread(target=_watch, daemon=True)
    watcher.start()

    try:
        subprocess.run(parallel_cmd, env=worker_env)
    finally:
        stop_event.set()
        watcher.join(timeout=2.0)

    wall_time = time.time() - wall_start
    _write_progress_txt(
        progress_txt,
        video_paths,
        joblog,
        wall_start,
        jobs,
        status_dir=status_dir,
        final_wall_time=wall_time,
    )

    # Parse the joblog to build success/failure map. GNU parallel joblog:
    # Seq  Host  Starttime  JobRuntime  Send  Receive  Exitval  Signal  Command
    results: list[tuple[str, str | None]] = []
    runtimes: dict[str, float] = {}
    try:
        with open(joblog) as f:
            lines = f.readlines()
        # Skip header
        for line in lines[1:]:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 7:
                continue
            seq = int(parts[0])
            try:
                runtime = float(parts[3])
            except ValueError:
                runtime = 0.0
            exitval = parts[6]
            signal = parts[7] if len(parts) > 7 else "0"
            vpath = video_paths[seq - 1]
            runtimes[vpath] = runtime
            if exitval == "0":
                results.append((vpath, None))
            else:
                err = f"exit code {exitval}"
                if signal not in ("0", ""):
                    err += f" (signal {signal})"
                stem = os.path.splitext(os.path.basename(vpath))[0]
                worker_log = os.path.join(
                    _tracking_layout(recording_dir_for(vpath))["logs"],
                    f"yoto-parallel-{timestamp}",
                    f"{seq}-{stem}",
                )
                err += f" — see log dir: {worker_log}"
                results.append((vpath, err))
    except FileNotFoundError:
        print(f"Warning: GNU parallel did not produce a joblog at {joblog}")

    # Mark any videos missing from the joblog as failures (parallel crashed)
    seen = {p for p, _ in results}
    for vpath in video_paths:
        if vpath not in seen:
            results.append((vpath, "no joblog entry — worker never ran"))

    return results, runtimes, wall_time


def _is_image_input(path: str) -> bool:
    """Return True if *path* is an image file or a directory of images (no videos)."""
    if os.path.isfile(path):
        return os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS
    if os.path.isdir(path):
        entries = [f for f in os.listdir(path) if not f.startswith(".")]
        has_images = any(
            os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS for f in entries
        )
        has_videos = any(
            os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS for f in entries
        )
        return has_images and not has_videos
    return False


def _run_detect(args: argparse.Namespace) -> None:
    """Execute the detect sub-command."""
    _configure_logging(args.debug)
    args.dataname = _normalize_dataname(args.dataname)
    if not args.tag_offset_filter:
        args.max_tag_offset_ratio = float("inf")

    # Route image files / image-only folders to the image pipeline.
    if _is_image_input(args.video):
        from yoto.detection import run_detection_images

        output_root = args.output_dir or None
        try:
            dfs = run_detection_images(
                image_path=args.video,
                output_root=output_root,
                yolo_weights=args.yoloweights,
                data_suffix=args.dataname,
                conf_threshold=args.conf,
                pad_ratio=args.pad_ratio,
                max_offset_ratio=args.max_tag_offset_ratio,
                preset=args.apriltag_preset,
                tag_family=args.tag_family,
                max_tag_id=args.max_tag_id,
                no_yolo=args.no_yolo,
                no_enhance=args.no_enhance,
            )
        except FileNotFoundError as exc:
            print(f"Error: {exc}")
            sys.exit(1)
        if dfs and "tag_id" in dfs[0].columns:
            n_detections = sum(len(df) for df in dfs)
            n_ids = sum(int(df["tag_id"].nunique()) for df in dfs)
        else:
            n_detections = sum(
                len(df.columns.get_level_values(0).unique()) for df in dfs
            )
            n_ids = n_detections
        tag_summary = (
            f"{n_detections} detection(s) across {n_ids} unique ID(s)"
            if n_detections != n_ids
            else f"{n_detections} detection(s)"
        )
        print(
            f"Processed {len(dfs)} image(s), {tag_summary}. "
            f"Overlays → tracking/image_output/  |  Pickles → tracking/data/"
        )
        return

    video_paths = _resolve_video_paths(args.video)
    if not video_paths:
        print(f"No video files found in: {args.video}")
        sys.exit(1)

    video_paths = _apply_video_nb(video_paths, args.video_nb)

    if args.parallel is not None and args.parallel < 1:
        print("Error: --parallel must be >= 1")
        sys.exit(1)

    if args.use_nvdec and args.parallel and args.parallel > 1:
        print(
            f"WARNING: --use-nvdec True with --parallel {args.parallel} runs "
            "multiple NVDEC sessions on one GPU. The RTX A6000 has a limited "
            "number of NVDEC engines (2–3); oversubscription may hurt "
            "throughput and contend on VRAM. The Ultralytics streaming "
            "pipeline (--use-nvdec False) parallelises more cleanly at high "
            "worker counts."
        )

    use_parallel = (
        args.parallel is not None and args.parallel > 1 and len(video_paths) > 1
    )

    if len(video_paths) > 1:
        mode = (
            f"parallel ({args.parallel} GNU parallel workers)"
            if use_parallel
            else "sequentially"
        )
        print(f"Processing {len(video_paths)} video(s) {mode}")

    results: list[tuple[str, str | None]]
    runtimes: dict[str, float] = {}
    wall_time: float = 0.0
    if use_parallel:
        worker_tmpl = [sys.executable, "-m", "yoto.cli", "detect", "{}"]
        if args.output_dir:
            worker_tmpl.append(args.output_dir)
        worker_tmpl.extend(["--yoloweights", args.yoloweights])
        worker_tmpl.extend(["--dataname", args.dataname])
        worker_tmpl.extend(["--use-nvdec", str(args.use_nvdec)])
        if args.debug:
            worker_tmpl.append("--debug")
        if args.apriltag_preset:
            worker_tmpl.extend(["--apriltag-preset", args.apriltag_preset])
        worker_tmpl.extend(["--conf", str(args.conf)])
        worker_tmpl.extend(["--iou", str(args.iou)])
        worker_tmpl.extend(["--pad-ratio", str(args.pad_ratio)])
        worker_tmpl.extend(["--max-tag-offset-ratio", str(args.max_tag_offset_ratio)])
        worker_tmpl.extend(["--tag-family", args.tag_family])
        worker_tmpl.extend(["--batch-size", str(args.batch_size)])
        if args.max_tag_id is not None:
            worker_tmpl.extend(["--max-tag-id", str(args.max_tag_id)])
        worker_silence = _parse_id_list(args.silence_ids, "--silence-ids")
        if worker_silence:
            worker_tmpl.extend(
                ["--silence-ids", ",".join(str(i) for i in worker_silence)]
            )
        worker_tmpl.extend(["--save-yolo", str(args.save_yolo)])
        worker_tmpl.extend(["--save-quads", str(args.save_quads)])
        if args.no_yolo:
            worker_tmpl.append("--no-yolo")
        if args.no_enhance:
            worker_tmpl.append("--no-enhance")
        input_root = (
            args.video
            if os.path.isdir(args.video)
            else os.path.dirname(os.path.abspath(args.video))
        )
        results, runtimes, wall_time = _run_parallel_gnu(
            video_paths=video_paths,
            worker_tmpl=worker_tmpl,
            jobs=args.parallel,
            input_root=input_root,
        )
    else:
        results = []
        for idx, vpath in enumerate(video_paths, start=1):
            if len(video_paths) > 1:
                print(f"\n[{idx}/{len(video_paths)}] {vpath}")
            result = _run_single_video(
                vpath=vpath,
                use_nvdec=args.use_nvdec,
                output_dir=args.output_dir,
                yolo_weights=args.yoloweights,
                data_suffix=args.dataname,
                debug=args.debug,
                preset=args.apriltag_preset,
                conf=args.conf,
                iou=args.iou,
                pad_ratio=args.pad_ratio,
                max_tag_offset_ratio=args.max_tag_offset_ratio,
                save_yolo=args.save_yolo,
                save_quads=args.save_quads,
                tag_family=args.tag_family,
                batch_size=args.batch_size,
                max_tag_id=args.max_tag_id,
                silence_ids=_parse_id_list(args.silence_ids, "--silence-ids"),
                no_yolo=args.no_yolo,
                no_enhance=args.no_enhance,
            )
            if result[1] is not None:
                logging.getLogger(__name__).error(
                    "Detection failed for %s:\n%s", vpath, result[1]
                )
            results.append(result)

    failures = [(p, e) for p, e in results if e is not None]
    successes = len(results) - len(failures)

    if len(results) > 1 or failures:
        print("")
        print("─" * 60)
        print(f"Summary: {successes} succeeded, {len(failures)} failed")
        print("─" * 60)

        if use_parallel and runtimes:
            total_cpu_time = sum(runtimes.values())
            print(f"Total wall time:     {_format_duration(wall_time)}")
            print(
                f"Per videos time: {_format_duration(total_cpu_time/len(video_paths))} "
            )
            print("")
            print("Per-video runtime (longest first):")
            errors = {p: e for p, e in results}
            for vpath, rt in sorted(runtimes.items(), key=lambda x: x[1], reverse=True):
                status = "OK  " if errors.get(vpath) is None else "FAIL"
                name = os.path.basename(vpath)
                print(f"  [{status}] {_format_duration(rt):>12}  {name}")
            print("")

        if failures:
            print("Failed videos (retry individually with --video-nb):")
            for path, err in failures:
                summary_line = err.strip().splitlines()[-1] if err else ""
                print(f"  - {path}")
                print(f"      {summary_line}")
            sys.exit(1)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Sub-command: clean
# ---------------------------------------------------------------------------


def _add_clean_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``clean`` sub-command."""
    p = subparsers.add_parser(
        "clean",
        help="Clean and interpolate raw tracking data",
    )
    p.add_argument(
        "input_pkl",
        help="Path to a raw-detection pickle, a video file (the matching "
        "raw pickle is looked up via --dataname), a directory of pickles, "
        "or a directory of recording sub-folders. Files ending "
        "'_clean.pkl' are skipped.",
    )
    p.add_argument(
        "--dataname",
        default="_apriltagDetect14",
        help="Suffix used to locate the raw pickle when a video file is "
        "passed as input (default: _apriltagDetect14)",
    )
    p.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output pickle path (default: <input>_clean.pkl). "
        "Only valid when input_pkl is a single file.",
    )
    p.add_argument(
        "--min-detections",
        type=int,
        default=100,
        help="Minimum detections to keep a tag ID (default: 100)",
    )
    p.add_argument(
        "--interp-limit",
        type=int,
        default=5,
        help="Max gap length for interpolation (default: 5)",
    )
    p.add_argument(
        "--max-jump",
        type=float,
        default=100.0,
        help="Max pixel distance before jump deletion (default: 100)",
    )
    p.add_argument(
        "--video-nb",
        type=str,
        default=None,
        metavar="SPEC",
        help="When input is a recording directory, clean only the pickle(s) "
        "for the video(s) at these 0-based indices in the resolved video "
        "list. Accepts a single index, a comma list, and/or inclusive "
        "ranges: e.g. '3', '0,2,5', '0-9'.",
    )
    p.add_argument(
        "--csv",
        action="store_true",
        help="Also write CSV copies alongside the pickles: a "
        "<stem>_clean.csv next to each cleaned pickle, and a "
        "<stem>.csv next to the input raw pickle.",
    )
    from yoto.constants import (
        DEFAULT_MAX_CONSECUTIVE_MISSES,
        DEFAULT_TAG_SIZE_MULTIPLIER,
        DEFAULT_YOLO_FILL_LIMIT,
    )

    p.add_argument(
        "--yolo-fill",
        type=_str_to_bool,
        default=True,
        metavar="BOOL",
        help="Run the YOLO-fill pass that bridges gaps using the "
        "undecoded YOLO boxes from <stem>_yolo.pkl (default: True). "
        "Set to False to skip step 6 entirely and rely only on linear "
        "interpolation.",
    )
    p.add_argument(
        "--yolo-fill-limit",
        type=int,
        default=DEFAULT_YOLO_FILL_LIMIT,
        metavar="N",
        help=f"Hard cap on frames since last anchor refresh before the "
        f"chain breaks regardless of misses (default: "
        f"{DEFAULT_YOLO_FILL_LIMIT}; 0 disables the cap). Normally "
        "--max-consecutive-misses is the right knob; this is a safety "
        "net for pathological cases.",
    )
    p.add_argument(
        "--tag-size-multiplier",
        type=float,
        default=DEFAULT_TAG_SIZE_MULTIPLIER,
        metavar="FLOAT",
        help=f"Multiplier on the median tag side (px) that sets the maximum "
        f"distance a YOLO box can be from a tag's anchor to be accepted "
        f"(default: {DEFAULT_TAG_SIZE_MULTIPLIER}). Constant cap — does NOT grow with gap length.",
    )
    p.add_argument(
        "--max-consecutive-misses",
        type=int,
        default=DEFAULT_MAX_CONSECUTIVE_MISSES,
        metavar="N",
        help=f"Chain breaks after this many consecutive frames where no "
        f"YOLO box is within the search radius (default: "
        f"{DEFAULT_MAX_CONSECUTIVE_MISSES}). Tolerates short bursts of "
        "bad YOLO detection while still cutting the chain when the tag "
        "has truly left.",
    )
    from yoto.constants import (
        DEFAULT_FINAL_JUMP_PASS as _FINAL_JUMP_PASS_DEFAULT,
        DEFAULT_MIN_GAP_RECOVERY_FRAMES as _MIN_GAP_RECOVERY_FRAMES_DEFAULT,
        DEFAULT_RECHAIN_AFFECTED_ONLY as _RECHAIN_AFFECTED_ONLY_DEFAULT,
        DEFAULT_RECOVER_LONG_GAPS as _RECOVER_LONG_GAPS_DEFAULT,
    )

    p.add_argument(
        "--rechain-affected-only",
        type=_str_to_bool,
        default=_RECHAIN_AFFECTED_ONLY_DEFAULT,
        metavar="BOOL",
        help=f"In the re-chain pass (step 6d), restrict candidate tags "
        f"to only those whose YOLO fills were pruned in step 6c "
        f"(default: {_RECHAIN_AFFECTED_ONLY_DEFAULT}). False (default) "
        f"lets every tag compete for the freed YOLO boxes.",
    )
    p.add_argument(
        "--recover-long-gaps",
        type=_str_to_bool,
        default=_RECOVER_LONG_GAPS_DEFAULT,
        metavar="BOOL",
        help=f"Experimental.  After step 7, for each tag with a NaN gap "
        f"longer than --min-gap-recovery-frames, match each gap frame to "
        f"the closest unclaimed YOLO box within median_tag_side * "
        f"--reach-multiplier of the gap's leading or trailing anchor "
        f"(anchors are fixed — they don't drift) "
        f"(default: {_RECOVER_LONG_GAPS_DEFAULT}).",
    )
    p.add_argument(
        "--min-gap-recovery-frames",
        type=int,
        default=_MIN_GAP_RECOVERY_FRAMES_DEFAULT,
        metavar="N",
        help=f"Minimum gap length (frames) for --recover-long-gaps to "
        f"fire on a given gap "
        f"(default: {_MIN_GAP_RECOVERY_FRAMES_DEFAULT}).",
    )
    p.add_argument(
        "--final-jump-pass",
        type=_str_to_bool,
        default=_FINAL_JUMP_PASS_DEFAULT,
        metavar="BOOL",
        help=f"Experimental.  Run an extra _delete_jump_blocks round at "
        f"step 9 on the fully-recovered data "
        f"(default: {_FINAL_JUMP_PASS_DEFAULT}).",
    )
    p.add_argument(
        "--debug-snapshots",
        type=_str_to_bool,
        default=False,
        metavar="BOOL",
        help="Debug.  Pickle the frame_data at each major cleaning "
        "checkpoint to "
        "<clean_dir>/<stem>_snapshots/step{6a,6c,6d,6f,7,8}_*.pkl. "
        "Inspect them in a notebook to see exactly what each step "
        "did.  Off by default.",
    )
    from yoto.constants import DEFAULT_TAG_SIZE_MM as _TAG_SIZE_MM_DEFAULT

    p.add_argument(
        "--tag-size",
        type=float,
        default=_TAG_SIZE_MM_DEFAULT,
        metavar="MM",
        help=f"Physical side length of one AprilTag border, in "
        f"millimetres (default: {_TAG_SIZE_MM_DEFAULT}). Used to "
        f"compute a mm_per_px scale from decoded tag corners; stored "
        f"on the clean pkl as the 'yoto_mm_per_px' attr.",
    )
    p.add_argument(
        "--parallel",
        type=int,
        default=None,
        metavar="N",
        help="Clean N pickles concurrently in separate worker processes "
        "via GNU parallel (no default — must be set explicitly). Cleaning "
        "is pure CPU/pandas, so workers scale near-linearly across cores.",
    )
    p.set_defaults(func=_run_clean)


def _write_csv(df: pd.DataFrame, path: str) -> None:
    """Write *df* to CSV with a flattened single-row header.

    Drops the per-detection ``corners`` column (each cell is a 4x2
    ndarray that bloats the CSV and dominates write time). The pickle
    keeps it.
    """
    keep = [c for c in df.columns if not (isinstance(c, tuple) and c[1] == "corners")]
    flat = df[keep].reset_index()
    flat.columns = [
        "_".join(str(p) for p in c if p != "") if isinstance(c, tuple) else str(c)
        for c in flat.columns
    ]
    flat.to_csv(path, index=False)


def _clean_one_pickle(
    pkl_path: str,
    output_path: str | None,
    min_detections: int,
    interp_limit: int,
    max_jump: float,
    write_csv: bool = False,
    yolo_fill: bool = True,
    yolo_fill_limit: int | None = None,
    tag_size_multiplier: float | None = None,
    max_consecutive_misses: int | None = None,
    rechain_affected_only: bool | None = None,
    recover_long_gaps: bool | None = None,
    min_gap_recovery_frames: int | None = None,
    final_jump_pass: bool | None = None,
    tag_size_mm: float | None = None,
    debug_snapshots: bool = False,
) -> tuple[str, str | None]:
    """Clean a single pickle. Returns ``(pkl_path, None)`` on success,
    or ``(pkl_path, error_message)`` on failure."""
    import traceback

    from yoto.cleaning import clean_tracking_data
    from yoto.exceptions import EmptyTrackingError
    from yoto.constants import (
        DEFAULT_FINAL_JUMP_PASS,
        DEFAULT_MAX_CONSECUTIVE_MISSES,
        DEFAULT_MIN_GAP_RECOVERY_FRAMES,
        DEFAULT_RECHAIN_AFFECTED_ONLY,
        DEFAULT_RECOVER_LONG_GAPS,
        DEFAULT_TAG_SIZE_MULTIPLIER,
        DEFAULT_TAG_SIZE_MM,
        DEFAULT_YOLO_FILL_LIMIT,
    )

    if yolo_fill_limit is None:
        yolo_fill_limit = DEFAULT_YOLO_FILL_LIMIT
    if tag_size_multiplier is None:
        tag_size_multiplier = DEFAULT_TAG_SIZE_MULTIPLIER
    if max_consecutive_misses is None:
        max_consecutive_misses = DEFAULT_MAX_CONSECUTIVE_MISSES
    if rechain_affected_only is None:
        rechain_affected_only = DEFAULT_RECHAIN_AFFECTED_ONLY
    if recover_long_gaps is None:
        recover_long_gaps = DEFAULT_RECOVER_LONG_GAPS
    if min_gap_recovery_frames is None:
        min_gap_recovery_frames = DEFAULT_MIN_GAP_RECOVERY_FRAMES
    if final_jump_pass is None:
        final_jump_pass = DEFAULT_FINAL_JUMP_PASS
    if tag_size_mm is None:
        tag_size_mm = DEFAULT_TAG_SIZE_MM

    try:
        frame_data = pd.read_pickle(pkl_path)

        # No detections at all → nothing to clean.  Warn and skip; do not
        # write a clean pickle (matches user expectation that an empty
        # raw pickle short-circuits the whole pipeline).
        if frame_data.empty or frame_data.shape[1] == 0:
            logging.getLogger(__name__).warning(
                "No tag detections in %s — skipping (no clean pickle written).",
                pkl_path,
            )
            return (pkl_path, None)

        # Write the raw CSV *before* clean_tracking_data, which mutates the
        # input frame in place.
        if write_csv:
            raw_csv = os.path.splitext(pkl_path)[0] + ".csv"
            _write_csv(frame_data, raw_csv)

        # Auto-discover the YOLO-fill source next to the raw pickle.
        # The _yolo.pkl sidecar is the single source of YOLO boxes; we
        # filter to the undecoded subset (decoded==False) for the fill.
        undecoded_df = None
        if yolo_fill and pkl_path.endswith(".pkl"):
            yolo_path = pkl_path[:-4] + "_yolo.pkl"
            if os.path.isfile(yolo_path):
                yolo_df = pd.read_pickle(yolo_path)
                undecoded_df = yolo_df[~yolo_df["decoded"].astype(bool)].copy()
                print(
                    f"YOLO-fill source: {os.path.basename(yolo_path)} "
                    f"(filtered to {len(undecoded_df)} undecoded boxes)"
                )

        # Resolve output_path up-front so debug snapshots land in the
        # same directory.  When --output is set, the dir is the parent
        # of that path; otherwise it's tracking/clean_data next to the
        # input pickle.
        if output_path is None:
            clean_dir = _tracking_layout(_recording_dir_for_pickle(pkl_path))[
                "clean_data"
            ]
            os.makedirs(clean_dir, exist_ok=True)
            stem = os.path.splitext(os.path.basename(pkl_path))[0]
            output_path = os.path.join(clean_dir, f"{stem}_clean.pkl")

        debug_snapshot_dir: str | None = None
        if debug_snapshots:
            stem = os.path.splitext(os.path.basename(output_path))[0]
            debug_snapshot_dir = os.path.join(
                os.path.dirname(output_path), f"{stem}_snapshots"
            )

        cleaned, _id_list, metrics = clean_tracking_data(
            frame_data,
            min_detections=min_detections,
            interpolation_limit=interp_limit,
            max_jump_distance=max_jump,
            undecoded_df=undecoded_df,
            yolo_fill_limit=yolo_fill_limit,
            tag_size_multiplier=tag_size_multiplier,
            max_consecutive_misses=max_consecutive_misses,
            rechain_affected_only=rechain_affected_only,
            recover_long_gaps=recover_long_gaps,
            min_gap_recovery_frames=min_gap_recovery_frames,
            final_jump_pass=final_jump_pass,
            tag_size_mm=tag_size_mm,
            debug_snapshot_dir=debug_snapshot_dir,
        )
        if debug_snapshot_dir is not None:
            print(f"Debug snapshots: {debug_snapshot_dir}")
        cleaned.to_pickle(output_path)
        print(f"Cleaned: {output_path}")
        _mm_per_px = cleaned.attrs.get("yoto_mm_per_px", float("nan"))
        _side_px = cleaned.attrs.get("yoto_median_tag_side_px", float("nan"))
        _n_scale = cleaned.attrs.get("yoto_scale_sample_count", 0)
        if _n_scale:
            print(
                f"  scale: {_mm_per_px:.5f} mm/px "
                f"(median tag side {_side_px:.2f} px, n={_n_scale})"
            )
        if write_csv:
            clean_csv = os.path.splitext(output_path)[0] + ".csv"
            _write_csv(cleaned, clean_csv)
            print(f"  CSV:   {raw_csv}")
            print(f"  CSV:   {clean_csv}")
        _n = metrics["total_samples"]
        _raw = metrics["total_detections"]
        _final = metrics["final_count"]
        _raw_pct = 100.0 * _raw / _n if _n else 0.0
        _final_pct = 100.0 * _final / _n if _n else 0.0
        print(
            f"  raw: {_raw}/{_n} ({_raw_pct:.1f}%)"
            f"  →  final: {_final}/{_n} ({_final_pct:.1f}%)"
        )
        print(
            f"    errors={metrics['original_bad_count']} ({metrics['error_pct']:.2f}%)"
            f" | interp={metrics['filled_count']}"
            f" | yolo={metrics['yolo_inferred_count']}/{metrics['total_gaps']} gaps"
            f" ({metrics['yolo_inferred_pct_of_gaps']:.1f}%)"
            f" | pruned={metrics['yolo_pruned_count']}"
            f" | rechained={metrics['yolo_rechained_count']}"
            f" | recovered={metrics['long_gap_recovered_count']}"
            f" | jump_del={metrics['final_jump_deleted_count']}"
            f" | max_move={metrics['max_move_px']:.1f}px"
        )
        return (pkl_path, None)
    except EmptyTrackingError as exc:
        # No usable tracks (empty input, or every tag below min_detections).
        # Skip cleanly, same shape as the early empty-pickle short-circuit.
        logging.getLogger(__name__).warning(
            "%s — skipping %s (no clean pickle written).", exc, pkl_path
        )
        return (pkl_path, None)
    except Exception:
        return (pkl_path, traceback.format_exc())


def _run_clean(args: argparse.Namespace) -> None:
    """Execute the clean sub-command."""
    _configure_logging(False)
    args.dataname = _normalize_dataname(args.dataname)

    if args.parallel is not None and args.parallel < 1:
        print("Error: --parallel must be >= 1")
        sys.exit(1)

    if args.video_nb is not None:
        # Resolve videos in the directory, pick the selected one(s), then
        # look up each raw pickle by --dataname suffix.
        video_paths = _resolve_video_paths(args.input_pkl)
        if not video_paths:
            print(f"No video files found in: {args.input_pkl}")
            sys.exit(1)
        pkl_paths = []
        for selected in _apply_video_nb(video_paths, args.video_nb):
            pkl = _find_pickle_for_video(selected, args.dataname, raw_only=True)
            if pkl is None:
                print(f"No raw pickle found for {selected} (suffix {args.dataname!r})")
                sys.exit(1)
            pkl_paths.append(pkl)
    else:
        pkl_paths = _resolve_pickle_paths(args.input_pkl, args.dataname)

    if not pkl_paths:
        print(f"No raw-detection pickle files found in: {args.input_pkl}")
        sys.exit(1)

    if args.output is not None and len(pkl_paths) > 1:
        print(
            "Error: --output can only be used with a single input pickle. "
            f"Found {len(pkl_paths)} pickles; outputs will be written "
            "next to each input by default."
        )
        sys.exit(1)

    use_parallel = (
        args.parallel is not None and args.parallel > 1 and len(pkl_paths) > 1
    )

    if len(pkl_paths) > 1:
        mode = (
            f"parallel ({args.parallel} GNU parallel workers)"
            if use_parallel
            else "sequentially"
        )
        print(f"Processing {len(pkl_paths)} pickle file(s) {mode}")

    results: list[tuple[str, str | None]] = []
    runtimes: dict[str, float] = {}
    wall_time: float = 0.0

    if use_parallel:
        worker_tmpl = [sys.executable, "-m", "yoto.cli", "clean", "{}"]
        worker_tmpl.extend(["--dataname", args.dataname])
        worker_tmpl.extend(["--min-detections", str(args.min_detections)])
        worker_tmpl.extend(["--interp-limit", str(args.interp_limit)])
        worker_tmpl.extend(["--max-jump", str(args.max_jump)])
        if args.csv:
            worker_tmpl.append("--csv")
        worker_tmpl.extend(["--yolo-fill", str(args.yolo_fill)])
        worker_tmpl.extend(["--yolo-fill-limit", str(args.yolo_fill_limit)])
        worker_tmpl.extend(["--tag-size-multiplier", str(args.tag_size_multiplier)])
        worker_tmpl.extend(
            ["--max-consecutive-misses", str(args.max_consecutive_misses)]
        )
        worker_tmpl.extend(["--rechain-affected-only", str(args.rechain_affected_only)])
        worker_tmpl.extend(["--recover-long-gaps", str(args.recover_long_gaps)])
        worker_tmpl.extend(
            ["--min-gap-recovery-frames", str(args.min_gap_recovery_frames)]
        )
        worker_tmpl.extend(["--final-jump-pass", str(args.final_jump_pass)])
        worker_tmpl.extend(["--tag-size", str(args.tag_size)])
        worker_tmpl.extend(["--debug-snapshots", str(args.debug_snapshots)])
        input_root = (
            args.input_pkl
            if os.path.isdir(args.input_pkl)
            else os.path.dirname(os.path.abspath(args.input_pkl))
        )
        results, runtimes, wall_time = _run_parallel_gnu(
            video_paths=pkl_paths,
            worker_tmpl=worker_tmpl,
            jobs=args.parallel,
            input_root=input_root,
            results_root_tmpl=os.path.join("{//}", "..", ".."),
            recording_dir_for=_recording_dir_for_pickle,
        )
    else:
        for idx, pkl in enumerate(pkl_paths, start=1):
            if len(pkl_paths) > 1:
                print(f"\n[{idx}/{len(pkl_paths)}] {pkl}")
            result = _clean_one_pickle(
                pkl_path=pkl,
                output_path=args.output if len(pkl_paths) == 1 else None,
                min_detections=args.min_detections,
                interp_limit=args.interp_limit,
                max_jump=args.max_jump,
                write_csv=args.csv,
                yolo_fill=args.yolo_fill,
                yolo_fill_limit=args.yolo_fill_limit,
                tag_size_multiplier=args.tag_size_multiplier,
                max_consecutive_misses=args.max_consecutive_misses,
                rechain_affected_only=args.rechain_affected_only,
                recover_long_gaps=args.recover_long_gaps,
                min_gap_recovery_frames=args.min_gap_recovery_frames,
                final_jump_pass=args.final_jump_pass,
                tag_size_mm=args.tag_size,
                debug_snapshots=args.debug_snapshots,
            )
            if result[1] is not None:
                logging.getLogger(__name__).error(
                    "Cleaning failed for %s:\n%s", pkl, result[1]
                )
            results.append(result)

    failures = [(p, e) for p, e in results if e is not None]
    successes = len(results) - len(failures)

    if len(results) > 1 or failures:
        print("")
        print("─" * 60)
        print(f"Summary: {successes} succeeded, {len(failures)} failed")
        print("─" * 60)
        if use_parallel and runtimes:
            total_cpu_time = sum(runtimes.values())
            print(f"Total wall time:     {_format_duration(wall_time)}")
            print(
                f"Sum of worker time:  {_format_duration(total_cpu_time)} "
                f"across {len(pkl_paths)} pickle(s)"
            )
        if failures:
            print("Failed pickles:")
            for path, err in failures:
                summary_line = err.strip().splitlines()[-1] if err else ""
                print(f"  - {path}")
                print(f"      {summary_line}")
            sys.exit(1)


# ---------------------------------------------------------------------------
# Sub-command: render
# ---------------------------------------------------------------------------


def _add_render_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``render`` sub-command."""
    p = subparsers.add_parser(
        "render",
        help="Render tracking overlay video",
    )
    p.add_argument(
        "video_path",
        help="Path to a video file, a directory of videos, "
        "or a directory of recording sub-folders",
    )
    p.add_argument(
        "--dataname",
        default="_apriltagDetect14",
        help="Suffix used to locate the tracking pickle and name the "
        "output video (default: _apriltagDetect14)",
    )
    p.add_argument(
        "--pkl",
        default=None,
        help="Explicit path to pickle file (single-video mode only; "
        "overrides --dataname pickle lookup)",
    )
    p.add_argument(
        "--short",
        action="store_true",
        help="Process only first 2000 frames",
    )
    p.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Output resolution scale factor (default: 1.0)",
    )
    p.add_argument(
        "--no-trails",
        action="store_true",
        help="Disable per-tag motion trails in the overlay",
    )
    p.add_argument(
        "--raw",
        action="store_true",
        help="Use the raw (un-interpolated) detection pickle and skip the "
        "cleaning pass. Implies --no-trails since raw data is gap-heavy.",
    )
    p.add_argument(
        "--quads",
        action="store_true",
        help="Overlay undecoded raw quads (white polygons) from the "
        "<stem>_quads.pkl sidecar, for visually verifying which gaps "
        "have a candidate AprilTag quad nearby.",
    )
    p.add_argument(
        "--undecoded",
        action="store_true",
        help="Overlay YOLO boxes (yellow rectangles) that AprilTag "
        "failed to decode, from the <stem>_yolo.pkl sidecar filtered "
        "to decoded==False. Coarser than --quads but more reliable.",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Debug overlay: every YOLO box from <stem>_yolo.pkl coloured "
        "by post-Phase-2 state (green=decoded original, yellow=matched by "
        "YOLO-fill, red=still unrecovered) plus AprilTag decoded quads "
        "(green polygons) for ORIGINAL tags.",
    )
    p.add_argument(
        "--text-scale",
        type=float,
        default=1.0,
        metavar="FACTOR",
        help="Multiplier for label/counter text size (default: 1.0; "
        "use <1 for smaller text, e.g. 0.5)",
    )
    p.add_argument(
        "--codec",
        choices=("auto", "h264", "hevc"),
        default="auto",
        help="Video encoder family (default: auto = hevc_nvenc first, "
        "then h264_nvenc, then libx264). Use 'h264' for smooth VLC "
        "playback; 'hevc' for smaller files.",
    )
    p.add_argument(
        "--highlight-ids",
        nargs="+",
        default=None,
        metavar="IDS",
        help="Tag ID(s) to visually highlight in the overlay. Accepts "
        "space-separated (--highlight-ids 42 87 103), comma-separated "
        "(--highlight-ids 42,87,103), or any mix (--highlight-ids "
        "42,87 103). Highlighted tags get the --highlight-color label "
        "and, when --highlight-bold True, a thicker label stroke.",
    )
    p.add_argument(
        "--highlight-color",
        default="red",
        metavar="NAME_OR_RGB",
        help="Color for highlighted tag labels. Either a named color "
        "(red, green, blue, yellow, cyan, magenta, white, black) or a "
        "comma-separated 'R,G,B' triple in 0-255 (default: red).",
    )
    p.add_argument(
        "--highlight-bold",
        type=_str_to_bool,
        default=True,
        metavar="BOOL",
        help="Draw highlighted tag labels with a thicker stroke "
        "(default: True). Set False to keep normal weight and change "
        "only the color.",
    )
    p.add_argument(
        "--parallel",
        type=int,
        default=None,
        metavar="N",
        help="Render N videos concurrently in separate worker processes "
        "(no default — must be set explicitly)",
    )
    p.add_argument(
        "--video-nb",
        type=int,
        default=None,
        metavar="INDEX",
        help="Render only the video at this 0-based index in the resolved "
        "list (useful for re-running a single failed video)",
    )
    p.set_defaults(func=_run_render)


def _render_single_video(
    vpath: str,
    pkl_override: str | None,
    data_suffix: str,
    scale: float,
    short: bool,
    draw_trails: bool,
    text_scale_factor: float,
    raw: bool,
    codec: str,
    overlay_quads: bool = False,
    overlay_undecoded: bool = False,
    debug: bool = False,
    highlight_ids: set[int] | None = None,
    highlight_color: tuple[int, int, int] = (0, 0, 255),
    highlight_bold: bool = True,
) -> tuple[str, str | None]:
    """Render one video. Returns ``(vpath, None)`` on success, or
    ``(vpath, traceback_text)`` on failure."""
    import traceback

    import numpy as np

    from yoto.cleaning import clean_tracking_data
    from yoto.constants import COL_ASS_TYPE
    from yoto.video import render_overlay_video

    try:
        if pkl_override is not None:
            pkl_path: str | None = pkl_override
        else:
            pkl_path = _find_pickle_for_video(vpath, data_suffix, raw_only=raw)
        if pkl_path is None or not os.path.isfile(pkl_path):
            where = (
                "tracking/raw_data/"
                if raw
                else "tracking/clean_data/ or tracking/raw_data/"
            )
            return (
                vpath,
                f"No pickle found for {vpath} "
                f"(expected under {where} with suffix {data_suffix!r})",
            )

        frame_data = pd.read_pickle(pkl_path)

        from yoto.cleaning import _ensure_wide

        frame_data = _ensure_wide(frame_data)

        # If the pickle was produced by `yoto clean`, it already contains
        # `ass_type` columns; re-running clean_tracking_data on it would
        # duplicate those columns and break .loc indexing.  Detect by
        # column content rather than filename to be robust to renames.
        already_cleaned = COL_ASS_TYPE in frame_data.columns.get_level_values(1)

        if raw:
            cleaned = frame_data
            id_list = np.unique(cleaned.columns.get_level_values(0))
            print(f"Using raw pickle (no interpolation): {os.path.basename(pkl_path)}")
        elif already_cleaned:
            cleaned = frame_data
            id_list = np.unique(cleaned.columns.get_level_values(0))
            print(f"Using pre-cleaned pickle: {os.path.basename(pkl_path)}")
        else:
            cleaned, id_list, metrics = clean_tracking_data(frame_data)
            print(
                f"Data quality ({os.path.basename(pkl_path)}): "
                f"detections={metrics['total_detections']}"
                f"/{metrics['total_samples']} | "
                f"errors={metrics['original_bad_count']} "
                f"({metrics['error_pct']:.2f}%) | "
                f"filled={metrics['filled_count']}/{metrics['total_gaps']} gaps "
                f"({metrics['filled_pct_of_gaps']:.2f}% recovered)"
            )

        # Place the output under <recording>/tracking/video_output/.
        video_out_dir = _tracking_layout(_recording_dir_for_video(vpath))[
            "video_output"
        ]
        os.makedirs(video_out_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(vpath))[0]
        raw_tag = "_raw" if raw else ""
        short_tag = "_short" if short else ""
        debug_tag = "_debug" if debug else ""
        if scale != 1.0:
            out_name = f"{stem}{data_suffix}{raw_tag}{short_tag}{debug_tag}_scaled{scale:.2f}.mp4"
        else:
            out_name = f"{stem}{data_suffix}{raw_tag}{short_tag}{debug_tag}.mp4"
        output_path = os.path.join(video_out_dir, out_name)

        quads_data = None
        if overlay_quads:
            quads_path = _find_pickle_for_video(
                vpath, data_suffix + "_quads", raw_only=True
            )
            if quads_path is None or not os.path.isfile(quads_path):
                print(
                    f"WARNING: --quads requested but no _quads.pkl sidecar "
                    f"found for {vpath}; rendering without it."
                )
            else:
                quads_data = pd.read_pickle(quads_path)
                print(f"Overlaying quads from: {quads_path}")

        undecoded_data = None
        if overlay_undecoded:
            # Read _yolo.pkl and filter to undecoded boxes.
            yolo_path_for_und = _find_pickle_for_video(
                vpath, data_suffix + "_yolo", raw_only=True
            )
            if yolo_path_for_und is not None and os.path.isfile(yolo_path_for_und):
                yolo_df = pd.read_pickle(yolo_path_for_und)
                undecoded_data = yolo_df[~yolo_df["decoded"].astype(bool)].copy()
                print(
                    f"Overlaying undecoded boxes from: {yolo_path_for_und} "
                    f"({len(undecoded_data)} rows)"
                )
            else:
                print(
                    f"WARNING: --undecoded requested but no _yolo.pkl "
                    f"sidecar found for {vpath}; rendering without it."
                )

        yolo_data = None
        if debug:
            yolo_path = _find_pickle_for_video(
                vpath, data_suffix + "_yolo", raw_only=True
            )
            if yolo_path is None or not os.path.isfile(yolo_path):
                print(
                    f"WARNING: --debug requested but no _yolo.pkl sidecar "
                    f"found for {vpath}; debug boxes will be skipped "
                    "(quads will still draw if available)."
                )
            else:
                yolo_data = pd.read_pickle(yolo_path)
                print(f"Debug YOLO source: {yolo_path}")

        output = render_overlay_video(
            video_path=vpath,
            frame_data=cleaned,
            id_list=id_list,
            output_path=output_path,
            data_name=data_suffix,
            scale=scale,
            short=short,
            draw_trails=draw_trails,
            text_scale_factor=text_scale_factor,
            codec=codec,
            quads_data=quads_data,
            undecoded_data=undecoded_data,
            debug=debug,
            yolo_data=yolo_data,
            highlight_ids=highlight_ids,
            highlight_color=highlight_color,
            highlight_bold=highlight_bold,
        )
        print(f"Output: {output}")
        return (vpath, None)
    except Exception:
        return (vpath, traceback.format_exc())


def _run_render(args: argparse.Namespace) -> None:
    """Execute the render sub-command."""
    _configure_logging(False)
    args.dataname = _normalize_dataname(args.dataname)

    highlight_color_bgr = _parse_color_bgr(args.highlight_color)
    highlight_ids_list = _parse_id_list(args.highlight_ids)
    highlight_ids_set: set[int] | None = (
        set(highlight_ids_list) if highlight_ids_list else None
    )

    video_paths = _resolve_video_paths(args.video_path)
    if not video_paths:
        print(f"No video files found in: {args.video_path}")
        sys.exit(1)

    if args.video_nb is not None:
        if args.video_nb < 0 or args.video_nb >= len(video_paths):
            print(
                f"Error: --video-nb {args.video_nb} out of range "
                f"(found {len(video_paths)} video(s), valid indices "
                f"0..{len(video_paths) - 1})"
            )
            sys.exit(1)
        selected = video_paths[args.video_nb]
        print(f"Selected video [{args.video_nb}]: {selected}")
        video_paths = [selected]

    if args.pkl is not None and len(video_paths) > 1:
        print(
            "Error: --pkl can only be used with a single input video. "
            f"Found {len(video_paths)} videos; remove --pkl so each pickle "
            "is looked up per-video via --dataname."
        )
        sys.exit(1)

    if args.parallel is not None and args.parallel < 1:
        print("Error: --parallel must be >= 1")
        sys.exit(1)

    use_parallel = (
        args.parallel is not None and args.parallel > 1 and len(video_paths) > 1
    )

    if len(video_paths) > 1:
        mode = (
            f"parallel ({args.parallel} GNU parallel workers)"
            if use_parallel
            else "sequentially"
        )
        print(f"Rendering {len(video_paths)} video(s) {mode}")

    results: list[tuple[str, str | None]]
    runtimes: dict[str, float] = {}
    wall_time: float = 0.0
    if use_parallel:
        worker_tmpl = [sys.executable, "-m", "yoto.cli", "render", "{}"]
        worker_tmpl.extend(["--dataname", args.dataname])
        worker_tmpl.extend(["--scale", str(args.scale)])
        worker_tmpl.extend(["--text-scale", str(args.text_scale)])
        worker_tmpl.extend(["--codec", args.codec])
        if args.short:
            worker_tmpl.append("--short")
        if args.no_trails:
            worker_tmpl.append("--no-trails")
        if args.raw:
            worker_tmpl.append("--raw")
        if args.quads:
            worker_tmpl.append("--quads")
        if args.undecoded:
            worker_tmpl.append("--undecoded")
        if args.debug:
            worker_tmpl.append("--debug")
        if highlight_ids_list:
            worker_tmpl.extend(
                ["--highlight-ids", ",".join(str(i) for i in highlight_ids_list)]
            )
        worker_tmpl.extend(["--highlight-color", args.highlight_color])
        worker_tmpl.extend(["--highlight-bold", str(args.highlight_bold)])
        input_root = (
            args.video_path
            if os.path.isdir(args.video_path)
            else os.path.dirname(os.path.abspath(args.video_path))
        )
        results, runtimes, wall_time = _run_parallel_gnu(
            video_paths=video_paths,
            worker_tmpl=worker_tmpl,
            jobs=args.parallel,
            input_root=input_root,
        )
    else:
        results = []
        for idx, vpath in enumerate(video_paths, start=1):
            if len(video_paths) > 1:
                print(f"\n[{idx}/{len(video_paths)}] {vpath}")
            result = _render_single_video(
                vpath=vpath,
                pkl_override=args.pkl if len(video_paths) == 1 else None,
                data_suffix=args.dataname,
                scale=args.scale,
                short=args.short,
                draw_trails=(not args.no_trails) and (not args.raw),
                text_scale_factor=args.text_scale,
                raw=args.raw,
                codec=args.codec,
                overlay_quads=args.quads,
                overlay_undecoded=args.undecoded,
                debug=args.debug,
                highlight_ids=highlight_ids_set,
                highlight_color=highlight_color_bgr,
                highlight_bold=args.highlight_bold,
            )
            if result[1] is not None:
                logging.getLogger(__name__).error(
                    "Rendering failed for %s:\n%s", vpath, result[1]
                )
            results.append(result)

    failures = [(p, e) for p, e in results if e is not None]
    successes = len(results) - len(failures)

    if len(results) > 1 or failures:
        print("")
        print("─" * 60)
        print(f"Summary: {successes} succeeded, {len(failures)} failed")
        print("─" * 60)

        if use_parallel and runtimes:
            total_cpu_time = sum(runtimes.values())
            print(f"Total wall time:     {_format_duration(wall_time)}")
            print(
                "Per videos time: "
                f"{_format_duration(total_cpu_time / len(video_paths))} "
            )
            print("")
            print("Per-video runtime (longest first):")
            errors = {p: e for p, e in results}
            for vpath, rt in sorted(runtimes.items(), key=lambda x: x[1], reverse=True):
                status = "OK  " if errors.get(vpath) is None else "FAIL"
                name = os.path.basename(vpath)
                print(f"  [{status}] {_format_duration(rt):>12}  {name}")
            print("")

        if failures:
            print("Failed videos (retry individually with --video-nb):")
            for path, err in failures:
                summary_line = err.strip().splitlines()[-1] if err else ""
                print(f"  - {path}")
                print(f"      {summary_line}")
            sys.exit(1)


# ---------------------------------------------------------------------------
# Sub-command: train
# ---------------------------------------------------------------------------


def _add_build_testset_parser(subparsers: argparse._SubParsersAction) -> None:
    from yoto.constants import (
        DEFAULT_BATCH_SIZE,
        DEFAULT_CONF_THRESHOLD,
        DEFAULT_PAD_RATIO,
    )

    p = subparsers.add_parser(
        "build-testset",
        help=(
            "Sample frames from videos, run YOLO, and save composite strips "
            "with ground-truth tag positions for preset optimisation"
        ),
    )
    p.add_argument(
        "videos",
        nargs="+",
        help="One or more video files",
    )
    p.add_argument(
        "--pickles",
        nargs="*",
        default=None,
        help=(
            "YOTO clean pickle per video (must match the resolved video "
            "count). Default: located via --dataname under each recording's "
            "tracking/clean_data/"
        ),
    )
    p.add_argument(
        "--dataname",
        default="_apriltagDetect14",
        help="Suffix used to locate the clean pickle per video, i.e. "
        "tracking/clean_data/<stem><dataname>_clean.pkl "
        "(default: _apriltagDetect14)",
    )
    p.add_argument(
        "--yoloweights",
        default=DEFAULT_WEIGHTS,
        help=f"YOLO weights file used to locate tag regions "
        f"(default: {DEFAULT_WEIGHTS})",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Root testset directory (default: <first_video_parent>/apriltag_testset)",
    )
    p.add_argument(
        "--gt-source",
        choices=["auto", "clean", "yolo"],
        default="auto",
        help="Ground-truth source. 'clean' = decoded clean pickle (needs "
        "detection to already work). 'yolo' = build from the _yolo.pkl "
        "sidecar with NO ground truth (cold-start preset tuning when nothing "
        "decodes). 'auto' (default) = clean pickle if present, else fall back "
        "to the YOLO sidecar.",
    )
    p.add_argument(
        "--sample-per-video",
        type=int,
        default=50,
        help="Frames to sample per video (default: 50)",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=1000,
        help="Pool size ranked by tag visibility (default: 1000)",
    )
    p.add_argument(
        "--min-detection-frac",
        type=float,
        default=0.8,
        help="Minimum fraction of known tags visible for a frame to enter the pool (default: 0.8)",
    )
    p.add_argument(
        "--pad-ratio",
        type=float,
        default=DEFAULT_PAD_RATIO,
        help=f"Per-axis crop padding — must match yoto detect (default: {DEFAULT_PAD_RATIO})",
    )
    p.add_argument(
        "--conf-thresh",
        type=float,
        default=DEFAULT_CONF_THRESHOLD,
        help=f"YOLO confidence threshold (default: {DEFAULT_CONF_THRESHOLD})",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"YOLO inference batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if a manifest already exists for this video",
    )
    p.add_argument(
        "--sample-strategy",
        choices=["top", "uniform", "mixed"],
        default="mixed",
        help=(
            "Frame sampling strategy: 'top' = highest-visibility only (biased easy), "
            "'uniform' = evenly spaced across the full video, "
            "'mixed' = half top + half uniform (default)"
        ),
    )
    p.add_argument(
        "--video-nb",
        type=str,
        default=None,
        metavar="SPEC",
        help="Process only the video(s) at these 0-based indices in the "
        "resolved list. Accepts a single index, a comma list, and/or "
        "inclusive ranges: e.g. '3', '0,2,5', '0-9'.",
    )
    p.add_argument(
        "--parallel",
        type=int,
        default=None,
        metavar="N",
        help="Run N videos concurrently via GNU parallel. Each worker runs "
        "YOLO on the GPU, so high N may contend on VRAM. Cannot be combined "
        "with explicit --pickles.",
    )
    p.set_defaults(func=_run_build_testset)


def _run_build_testset(args: argparse.Namespace) -> None:
    from pathlib import Path

    from yoto.tuning import build_testset

    dataname = _normalize_dataname(args.dataname)

    # Expand any directories in the positional args to actual video files,
    # matching how detect/clean/render accept a recording folder.
    videos: list[str] = []
    for item in args.videos:
        videos.extend(_resolve_video_paths(item))
    if not videos:
        raise SystemExit(f"No video files found in: {', '.join(args.videos)}")

    pickles = args.pickles

    if pickles is not None and len(pickles) != len(videos):
        raise SystemExit(
            f"--pickles count ({len(pickles)}) must match the resolved "
            f"video count ({len(videos)})"
        )

    # gt_modes[i] is "clean" or "yolo" — how build_testset should read pickles[i].
    gt_modes: list[str] = []
    if pickles is not None:
        # Explicit pickles are always treated as decoded clean pickles.
        gt_modes = ["clean"] * len(pickles)
    else:
        pickles = []
        for v in videos:
            stem = Path(v).with_suffix("").name
            layout = _tracking_layout(_recording_dir_for_video(v))
            clean = Path(layout["clean_data"]) / f"{stem}{dataname}_clean.pkl"
            sidecar = Path(layout["raw_data"]) / f"{stem}{dataname}_yolo.pkl"

            if args.gt_source == "clean":
                if not clean.exists():
                    raise SystemExit(
                        f"Cannot find clean pickle for {v}: tried {clean} "
                        f"(check --dataname, currently {dataname!r})"
                    )
                pickles.append(str(clean))
                gt_modes.append("clean")
            elif args.gt_source == "yolo":
                if not sidecar.exists():
                    raise SystemExit(
                        f"Cannot find YOLO sidecar for {v}: tried {sidecar} "
                        f"(run 'yoto detect' first; check --dataname, "
                        f"currently {dataname!r})"
                    )
                pickles.append(str(sidecar))
                gt_modes.append("yolo")
            else:  # auto: clean pickle if present, else YOLO sidecar
                if clean.exists():
                    pickles.append(str(clean))
                    gt_modes.append("clean")
                elif sidecar.exists():
                    pickles.append(str(sidecar))
                    gt_modes.append("yolo")
                else:
                    raise SystemExit(
                        f"No ground truth for {v}: found neither clean pickle "
                        f"({clean}) nor YOLO sidecar ({sidecar}). Run "
                        f"'yoto detect' first (check --dataname, currently "
                        f"{dataname!r})."
                    )

    if "yolo" in gt_modes:
        n_yolo = gt_modes.count("yolo")
        print(
            "\n"
            + "!" * 64
            + f"\n!!  YOLO MODE for {n_yolo}/{len(gt_modes)} video(s): building testset from\n"
            "!!  YOLO boxes with NO GROUND TRUTH (nothing has decoded yet).\n"
            "!!  optimize-preset will target raw decode YIELD, not accuracy.\n"
            + "!" * 64
            + "\n"
        )

    # --video-nb selects a subset of the (video, pickle) pairs.
    if args.video_nb is not None:
        try:
            indices = _parse_index_spec(args.video_nb, len(videos))
        except ValueError as exc:
            raise SystemExit(f"Error: {exc}")
        videos = [videos[i] for i in indices]
        pickles = [pickles[i] for i in indices]
        gt_modes = [gt_modes[i] for i in indices]
        print(
            f"Selected {len(videos)} video(s) by --video-nb "
            f"{args.video_nb!r}: indices {indices}"
        )

    if args.parallel is not None and args.parallel < 1:
        raise SystemExit("Error: --parallel must be >= 1")

    out_dir = args.out_dir or str(
        Path(videos[0]).parent / "tracking" / "apriltag_testset"
    )

    use_parallel = args.parallel is not None and args.parallel > 1 and len(videos) > 1
    if use_parallel and args.pickles is not None:
        raise SystemExit(
            "--parallel cannot be combined with explicit --pickles; omit "
            "--pickles so each worker resolves its own via --dataname."
        )

    if use_parallel:
        worker_tmpl = [
            sys.executable,
            "-m",
            "yoto.cli",
            "train",
            "build-testset",
            "{}",
            "--yoloweights",
            args.yoloweights,
            "--dataname",
            dataname,
            "--out-dir",
            out_dir,
            "--sample-per-video",
            str(args.sample_per_video),
            "--top-n",
            str(args.top_n),
            "--min-detection-frac",
            str(args.min_detection_frac),
            "--pad-ratio",
            str(args.pad_ratio),
            "--conf-thresh",
            str(args.conf_thresh),
            "--batch-size",
            str(args.batch_size),
            "--sample-strategy",
            args.sample_strategy,
            "--gt-source",
            args.gt_source,
        ]
        if args.force:
            worker_tmpl.append("--force")
        input_root = os.path.dirname(os.path.abspath(videos[0]))
        print(
            f"Processing {len(videos)} video(s) parallel "
            f"({args.parallel} GNU parallel workers)"
        )
        results, runtimes, wall_time = _run_parallel_gnu(
            video_paths=videos,
            worker_tmpl=worker_tmpl,
            jobs=args.parallel,
            input_root=input_root,
        )
        failures = [(p, e) for p, e in results if e is not None]
        print("")
        print("─" * 60)
        print(
            f"Summary: {len(results) - len(failures)} succeeded, "
            f"{len(failures)} failed"
        )
        print("─" * 60)
        if runtimes:
            print(f"Total wall time:    {_format_duration(wall_time)}")
            print(
                f"Sum of worker time: {_format_duration(sum(runtimes.values()))} "
                f"across {len(videos)} video(s)"
            )
        if failures:
            print("Failed videos (retry with --video-nb <index>):")
            for path, err in failures:
                tail = err.strip().splitlines()[-1] if err else ""
                print(f"  - {path}")
                print(f"      {tail}")
            sys.exit(1)
    else:
        for i, (vid, pkl, mode) in enumerate(zip(videos, pickles, gt_modes)):
            print(f"\n{'='*60}")
            print(f"Video {i+1}/{len(videos)}: {vid}  [gt-source: {mode}]")
            build_testset(
                vid,
                pkl,
                out_dir,
                args.yoloweights,
                sample_per_video=args.sample_per_video,
                top_n=args.top_n,
                min_detection_frac=args.min_detection_frac,
                pad_ratio=args.pad_ratio,
                conf_threshold=args.conf_thresh,
                batch_size=args.batch_size,
                force=args.force,
                sample_strategy=args.sample_strategy,
                gt_source=mode,
            )

    from pathlib import Path as _Path
    import json as _json

    all_manifests = sorted(_Path(out_dir).glob("*/manifest.json"))
    total_composites = total_crops = 0
    for mp in all_manifests:
        with open(mp) as f:
            doc = _json.load(f)
        total_composites += len(doc["frames"])
        total_crops += sum(len(e["crops"]) for e in doc["frames"])
    print(f"\n{'='*60}")
    print(f"GLOBAL SUMMARY ({len(all_manifests)} videos in {out_dir})")
    print(f"{'='*60}")
    print(f"Total video folders:  {len(all_manifests)}")
    print(f"Total composites:     {total_composites}")
    print(f"Total crops:          {total_crops}")


def _resolve_testset_dir(path: str | None) -> str:
    """Resolve *path* to a testset directory (``*/manifest.json`` inside).

    *path* may be the testset dir itself, a recording folder (uses its
    ``tracking/apriltag_testset``), or ``None`` (the current directory).
    Returns the best candidate; the caller checks it actually has manifests.
    """
    base = path or "."
    if glob.glob(os.path.join(base, "*", "manifest.json")):
        return base
    return os.path.join(base, TRACKING_DIR, "apriltag_testset")


def _add_subsample_testset_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "subsample-testset",
        help=(
            "Pick a compact, ID-diverse subset of a testset (greedy set "
            "cover weighted by tag difficulty) for faster optimize-preset runs"
        ),
    )
    p.add_argument(
        "recording",
        nargs="?",
        default=None,
        help="Recording folder (uses its tracking/apriltag_testset) or a "
        "testset directory directly. Defaults to the current directory.",
    )
    p.add_argument(
        "--testset-dir",
        default=None,
        help="Explicit testset directory (overrides the positional lookup)",
    )
    p.add_argument(
        "--target",
        type=int,
        default=50,
        metavar="N",
        help="Target number of composites to keep (default: 50)",
    )
    p.add_argument(
        "--min-appearances",
        type=int,
        default=3,
        metavar="N",
        help="Cover each tag ID at least N times (default: 3)",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output manifest path (default: <testset-dir>/subset_manifest.json)",
    )
    p.set_defaults(func=_run_subsample_testset)


def _run_subsample_testset(args: argparse.Namespace) -> None:
    from yoto.tuning import subsample_testset

    testset_dir = args.testset_dir or _resolve_testset_dir(args.recording)
    if not glob.glob(os.path.join(testset_dir, "*", "manifest.json")):
        raise SystemExit(
            f"No testset manifests found under {testset_dir}/*/manifest.json. "
            "Run 'yoto train build-testset' first, or pass --testset-dir."
        )
    print(f"Testset: {testset_dir}")
    subsample_testset(
        testset_dir,
        target=args.target,
        min_appearances=args.min_appearances,
        output=args.output,
    )


def _add_compare_presets_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "compare-presets",
        help=(
            "Compare AprilTag preset JSON(s) against each other and/or the "
            "yoto detect defaults, as a parameter diff table"
        ),
    )
    p.add_argument(
        "presets",
        nargs="+",
        help="Preset JSON path(s) or built-in preset name(s) to compare "
        "(e.g. a best_params_*.json from optimize-preset)",
    )
    p.add_argument(
        "--pipeline",
        choices=["fast", "simple"],
        default="fast",
        help="Base defaults for the 'default' column; yoto detect uses "
        "'fast' by default (default: fast)",
    )
    p.add_argument(
        "--no-default",
        action="store_true",
        help="Omit the yoto detect default column",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Show every parameter, not just rows that differ",
    )
    p.add_argument(
        "--raw",
        action="store_true",
        help="Show only the values present in each JSON (unmerged); by "
        "default presets are merged onto the pipeline defaults so each "
        "column shows the params yoto detect would actually use",
    )
    p.add_argument(
        "--testset-dir",
        default=None,
        help="Testset directory (built by build-testset) to also compare "
        "decode performance on. Auto-detected from the preset path when a "
        "preset lives inside a testset dir (e.g. a best_params_*.json).",
    )
    p.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip the on-dataset performance comparison (params table only)",
    )
    p.set_defaults(func=_run_compare_presets)


def _run_compare_presets(args: argparse.Namespace) -> None:
    from yoto.apriltag_presets import load_preset, resolve_preset
    from yoto.detection import (
        _build_apriltag_params_fast,
        _build_apriltag_params_simple,
    )

    base = (
        _build_apriltag_params_fast()
        if args.pipeline == "fast"
        else _build_apriltag_params_simple()
    )

    # Each column: (label, values_dict, keys_set_explicitly_in_source).
    columns: list[tuple[str, dict, set]] = []
    if not args.no_default:
        columns.append((f"default[{args.pipeline}]", dict(base), set()))
    for src in args.presets:
        try:
            raw = load_preset(src)
            label = os.path.splitext(os.path.basename(resolve_preset(src)))[0]
        except FileNotFoundError as exc:
            print(f"Error: {exc}")
            sys.exit(1)
        eff = raw if args.raw else {**base, **raw}
        columns.append((label, eff, set(raw)))

    def fmt(v: Any) -> str:
        if v is None:
            return ""
        return f"{v:g}" if isinstance(v, float) else str(v)

    # Skip underscore-prefixed metadata keys (e.g. "_description").
    keys = sorted({k for _, d, _ in columns for k in d if not k.startswith("_")})
    rows: list[tuple[str, list[str], bool]] = []
    for k in keys:
        cells = [fmt(d.get(k)) for _, d, _ in columns]
        differ = len(set(cells)) > 1
        if differ or args.all:
            marked = [
                cell + ("*" if k in rawset else "")
                for (_, _, rawset), cell in zip(columns, cells)
            ]
            rows.append((k, marked, differ))

    if not rows:
        print("No parameters." if args.all else "No differing parameters.")
        return

    labels = [lab for lab, _, _ in columns]
    key_w = max([len("parameter")] + [len(r[0]) for r in rows])
    col_w = [
        max([len(labels[i])] + [len(r[1][i]) for r in rows]) for i in range(len(labels))
    ]
    header = (
        "  "
        + "parameter".ljust(key_w)
        + "  "
        + "  ".join(labels[i].ljust(col_w[i]) for i in range(len(labels)))
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for k, marked, differ in rows:
        prefix = "> " if differ else "  "
        print(
            prefix
            + k.ljust(key_w)
            + "  "
            + "  ".join(marked[i].ljust(col_w[i]) for i in range(len(labels)))
        )
    print("")
    print("  (leading > = differs across columns;  trailing * = set explicitly)")

    # On-dataset performance comparison.
    if args.no_eval:
        return
    testset_dir = args.testset_dir
    if testset_dir is None:
        for src in args.presets:
            try:
                cand = os.path.dirname(os.path.abspath(resolve_preset(src)))
            except FileNotFoundError:
                continue
            if glob.glob(os.path.join(cand, "*", "manifest.json")):
                testset_dir = cand
                break
    if testset_dir is None:
        print(
            "\n  (pass --testset-dir to also compare decode performance "
            "on the dataset)"
        )
        return

    from yoto.tuning.optimize import _evaluate, load_manifests

    print(f"\nEvaluating on testset: {testset_dir}")
    entries, valid_ids, _, detected_family, yield_mode = load_manifests(testset_dir)
    family = detected_family or "tag36ARTag"
    print(f"  {len(entries)} composites, {len(valid_ids)} IDs, family={family}")
    if yield_mode:
        print(
            "  [YOLO MODE] no ground truth — 'recall' below is decode YIELD "
            "(fraction of YOLO boxes that decode), not accuracy."
        )
    print("")

    metric_cols = []
    for lab, eff, _ in columns:
        m = _evaluate(entries, valid_ids, eff, eff, family, yield_mode=yield_mode)
        metric_cols.append((lab, m))

    metric_rows = [
        ("recall", lambda m: f"{m.get('individual_recall', 0.0):.4f}", "max"),
        ("detection_rate", lambda m: f"{m.get('detection_rate', 0.0):.4f}", "max"),
        ("fp_rate", lambda m: f"{m.get('false_positive_rate', 0.0):.4f}", "min"),
        (
            "found/total",
            lambda m: f"{m.get('individual_found', 0)}/{m.get('total_gt_ids', 0)}",
            None,
        ),
        ("avg_ms/comp", lambda m: f"{m.get('avg_total_ms', 0.0):.1f}", "min"),
    ]
    mkey_w = max([len("metric")] + [len(r[0]) for r in metric_rows])
    mcol_w = [
        max([len(lab)] + [len(f(m)) for _, f, _ in metric_rows])
        for lab, m in metric_cols
    ]
    mheader = (
        "  "
        + "metric".ljust(mkey_w)
        + "  "
        + "  ".join(metric_cols[i][0].ljust(mcol_w[i]) for i in range(len(metric_cols)))
    )
    print(mheader)
    print("  " + "-" * (len(mheader) - 2))
    for name, f, better in metric_rows:
        vals = [f(m) for _, m in metric_cols]
        best_i = -1
        if better and len(metric_cols) > 1:
            nums = [float(v) for v in vals]
            best_i = nums.index(max(nums) if better == "max" else min(nums))
        cells = [(v + " *" if i == best_i else v) for i, v in enumerate(vals)]
        cw = [max(mcol_w[i], len(cells[i])) for i in range(len(cells))]
        print(
            "  "
            + name.ljust(mkey_w)
            + "  "
            + "  ".join(cells[i].ljust(cw[i]) for i in range(len(cells)))
        )
    print("")
    print("  (* = best column for that metric)")


def _add_optimize_preset_parser(subparsers: argparse._SubParsersAction) -> None:
    from yoto.constants import DEFAULT_TAG_FAMILY

    p = subparsers.add_parser(
        "optimize-preset",
        help=(
            "Run Optuna to find the best AprilTag preprocessing preset "
            "for a testset built with build-testset"
        ),
    )
    p.add_argument(
        "--testset-dir",
        required=True,
        help="Testset directory built by build-testset",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Where to write best_params_<study-name>.json (default: --testset-dir)",
    )
    p.add_argument(
        "--search-space",
        choices=["apriltag-only", "minimal", "standard-lite", "standard", "full"],
        default="standard",
        help=(
            "Parameter search space: apriltag-only (the 5 AprilTag detector "
            "params on the raw crop, no image enhancement -- like detect "
            "--no-yolo but still on crops), minimal (decoder + upscale + "
            "contrast, no cv2), standard-lite (+ cv2/unsharp, no "
            "tone-map/wiener; can reproduce the detect default), standard "
            "(+ tone-map/wiener), full (+ invert/bilateral/median/gamma/"
            "adaptive). Default: standard"
        ),
    )
    p.add_argument(
        "--n-trials",
        type=int,
        default=500,
        help="Total Optuna trials (default: 500)",
    )
    p.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help=(
            "Number of parallel workers (default: 1). "
            "With n_jobs > 1, spawns real OS processes sharing a SQLite study — "
            "a DB is auto-created next to the testset unless --storage is given."
        ),
    )
    p.add_argument(
        "--tag-family",
        default=DEFAULT_TAG_FAMILY,
        help=f"AprilTag family string (default: {DEFAULT_TAG_FAMILY})",
    )
    p.add_argument(
        "--study-name",
        default="yoto_preset",
        help="Optuna study name, also used in the output filename (default: yoto_preset)",
    )
    p.add_argument(
        "--storage",
        default=None,
        help="Optuna storage URI for persistent studies, e.g. sqlite:///study.db",
    )
    p.add_argument(
        "--seed-params",
        default=None,
        help="JSON preset file to enqueue as the first trial",
    )
    p.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Limit evaluation to the first N composites (for quick tests)",
    )
    p.add_argument(
        "--subset-manifest",
        default=None,
        help="Path to a subset_manifest.json from 'yoto train "
        "subsample-testset'; evaluates exactly those composites "
        "(takes precedence over --subset)",
    )
    p.add_argument(
        "--pruner-startup-trials",
        type=int,
        default=40,
        help="Trials before the MedianPruner activates (default: 40)",
    )
    p.add_argument(
        "--pruner-warmup-steps",
        type=int,
        default=200,
        help="Composites evaluated before pruning within a trial (default: 200)",
    )
    p.add_argument(
        "--prune-eval-interval",
        type=int,
        default=50,
        help="Report interim score to pruner every N composites (default: 50)",
    )
    p.add_argument(
        "--speed-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for speed penalty in the score. 0 = recall-only; "
            "set > 0 once recall exceeds --speed-floor-recall (default: 0.0)"
        ),
    )
    p.add_argument(
        "--speed-floor-recall",
        type=float,
        default=0.05,
        help="Recall threshold below which speed penalty is suppressed (default: 0.05)",
    )
    p.add_argument(
        "--synthetic-blur",
        action="store_true",
        help="Apply disk-kernel blur before preprocessing (brightfield-style augmentation)",
    )
    p.add_argument(
        "--max-tag-id",
        type=int,
        default=9999,
        help="Drop decoded IDs above this value (e.g. 512 for tagBCH64). Default: 9999 (no cap)",
    )
    p.add_argument(
        "--silence-ids",
        default=None,
        help="Comma-separated tag IDs to unconditionally discard (e.g. 341)",
    )
    p.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip end-of-run visualisation",
    )
    p.add_argument(
        "--viz-samples",
        type=int,
        default=8,
        help="Number of composites to visualise (default: 8)",
    )
    p.add_argument(
        "--viz-dir",
        default=None,
        help="Where to write viz PNGs (default: <testset-dir>/viz_<study-name>/)",
    )
    p.add_argument(
        "--export-trial",
        type=int,
        default=None,
        metavar="N",
        help="Don't optimize — build a preset JSON from trial N of an "
        "existing trials_<study-name>.csv and exit. Reads "
        "<out-dir-or-testset-dir>/trials_<study-name>.csv unless "
        "--trials-csv is given.",
    )
    p.add_argument(
        "--trials-csv",
        default=None,
        help="Explicit trials CSV to read for --export-trial",
    )
    p.add_argument(
        "-o",
        "--out",
        default=None,
        help="Output preset path for --export-trial "
        "(default: <dir>/preset_trial<N>_<study-name>.json)",
    )
    p.set_defaults(func=_run_optimize_preset)


def _run_optimize_preset_parallel(
    args: argparse.Namespace, storage: str, silence_ids: frozenset[int]
) -> None:
    """Spawn n_jobs-1 background workers + run the main worker in-process."""
    import subprocess
    import time

    from yoto.tuning import optimize_preset

    n_jobs = args.n_jobs
    per_worker = args.n_trials // n_jobs
    main_trials = args.n_trials - per_worker * (n_jobs - 1)

    out_dir = args.out_dir or args.testset_dir
    os.makedirs(out_dir, exist_ok=True)
    log_dir = os.path.join(out_dir, f"worker_logs_{args.study_name}")
    os.makedirs(log_dir, exist_ok=True)

    # Build the base subprocess command (background workers: no-viz, n_jobs=1)
    base_cmd = [
        sys.executable,
        "-m",
        "yoto.cli",
        "train",
        "optimize-preset",
        "--testset-dir",
        args.testset_dir,
        "--n-trials",
        str(per_worker),
        "--n-jobs",
        "1",
        "--search-space",
        args.search_space,
        "--tag-family",
        args.tag_family,
        "--study-name",
        args.study_name,
        "--storage",
        storage,
        "--pruner-startup-trials",
        str(args.pruner_startup_trials),
        "--pruner-warmup-steps",
        str(args.pruner_warmup_steps),
        "--prune-eval-interval",
        str(args.prune_eval_interval),
        "--speed-weight",
        str(args.speed_weight),
        "--speed-floor-recall",
        str(args.speed_floor_recall),
        "--max-tag-id",
        str(args.max_tag_id),
        "--no-viz",
    ]
    if args.out_dir:
        base_cmd += ["--out-dir", args.out_dir]
    if args.subset is not None:
        base_cmd += ["--subset", str(args.subset)]
    if args.subset_manifest:
        base_cmd += ["--subset-manifest", args.subset_manifest]
    if args.synthetic_blur:
        base_cmd.append("--synthetic-blur")
    if args.silence_ids:
        base_cmd += ["--silence-ids", args.silence_ids]
    # seed_params only on main worker (first trial, not duplicated across workers)

    procs: list[tuple[subprocess.Popen[bytes], str]] = []
    for i in range(n_jobs - 1):
        log_path = os.path.join(log_dir, f"worker_{i + 1}.log")
        log_fh = open(log_path, "wb")
        p = subprocess.Popen(base_cmd, stdout=log_fh, stderr=log_fh)
        procs.append((p, log_path))

    print(
        f"  {n_jobs - 1} background worker(s) started "
        f"({per_worker} trials each) — logs: {log_dir}"
    )
    print(f"  Main worker running {main_trials} trials with live display...\n")

    optimize_preset(
        args.testset_dir,
        out_dir=args.out_dir,
        search_space=args.search_space,
        n_trials=main_trials,
        n_jobs=1,
        tag_family=args.tag_family,
        study_name=args.study_name,
        storage=storage,
        seed_params=args.seed_params,
        subset=args.subset,
        subset_manifest=args.subset_manifest,
        pruner_startup_trials=args.pruner_startup_trials,
        pruner_warmup_steps=args.pruner_warmup_steps,
        prune_eval_interval=args.prune_eval_interval,
        speed_weight=args.speed_weight,
        speed_floor_recall=args.speed_floor_recall,
        synthetic_blur=args.synthetic_blur,
        max_tag_id=args.max_tag_id,
        silence_ids=silence_ids,
        no_viz=True,
        viz_samples=args.viz_samples,
        viz_dir=args.viz_dir,
    )

    print(f"\n  Main worker done. Waiting for {n_jobs - 1} background worker(s)...")
    failed = []
    for i, (p, log_path) in enumerate(procs):
        p.wait()
        if p.returncode != 0:
            failed.append((i + 1, log_path))

    if failed:
        for idx, log_path in failed:
            print(f"  [WARN] Worker {idx} exited non-zero — see {log_path}")

    print(f"  All {n_jobs} workers done. Running final output + viz...")

    # Re-run with 0 new trials just to write the consolidated output and viz.
    optimize_preset(
        args.testset_dir,
        out_dir=args.out_dir,
        search_space=args.search_space,
        n_trials=0,
        n_jobs=1,
        tag_family=args.tag_family,
        study_name=args.study_name,
        storage=storage,
        seed_params=None,
        subset=args.subset,
        subset_manifest=args.subset_manifest,
        pruner_startup_trials=args.pruner_startup_trials,
        pruner_warmup_steps=args.pruner_warmup_steps,
        prune_eval_interval=args.prune_eval_interval,
        speed_weight=args.speed_weight,
        speed_floor_recall=args.speed_floor_recall,
        synthetic_blur=args.synthetic_blur,
        max_tag_id=args.max_tag_id,
        silence_ids=silence_ids,
        no_viz=args.no_viz,
        viz_samples=args.viz_samples,
        viz_dir=args.viz_dir,
    )


def _export_preset_from_trial(args: argparse.Namespace) -> None:
    """Build a preset JSON from one row of a trials_<study>.csv."""
    import json

    import pandas as pd

    out_dir = args.out_dir or args.testset_dir
    csv_path = args.trials_csv or os.path.join(out_dir, f"trials_{args.study_name}.csv")
    if not os.path.isfile(csv_path):
        raise SystemExit(
            f"Trials CSV not found: {csv_path} "
            "(run optimize-preset first, or pass --trials-csv)"
        )
    df = pd.read_csv(csv_path)
    match = df[df["number"] == args.export_trial]
    if match.empty:
        raise SystemExit(f"Trial {args.export_trial} not found in {csv_path}")
    row = match.iloc[0]

    def _clean(prefix: str) -> dict:
        out = {}
        for col in df.columns:
            if col.startswith(prefix) and not pd.isna(row[col]):
                v = row[col]
                v = v.item() if hasattr(v, "item") else v
                if isinstance(v, float) and v.is_integer():
                    v = int(v)  # 11.0 -> 11 (pandas floats integer columns)
                out[col[len(prefix) :]] = v
        return out

    # The CSV records only each trial's *active* params (conditional keys are
    # absent when their branch was off), which is exactly what a preset needs:
    # merged onto the pipeline defaults, inactive keys stay at their defaults.
    params = _clean("params_")
    score = None if pd.isna(row.get("value")) else float(row["value"])
    doc = {
        "score": score,
        "params": params,
        "metrics": _clean("user_attrs_"),
        "_source": {"trials_csv": csv_path, "trial": int(args.export_trial)},
    }
    out_path = args.out or os.path.join(
        out_dir, f"preset_trial{args.export_trial}_{args.study_name}.json"
    )
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"Wrote preset from trial {args.export_trial} (score={score}) to:")
    print(f"  {out_path}")
    print(f"  params: {params}")


def _run_optimize_preset(args: argparse.Namespace) -> None:
    from yoto.tuning import optimize_preset

    # Allow --testset-dir to point directly at a subset manifest file
    # (e.g. subset_manifest.json): treat it as --subset-manifest and use its
    # parent as the testset dir.
    if os.path.isfile(args.testset_dir):
        if args.subset_manifest is None:
            args.subset_manifest = args.testset_dir
        args.testset_dir = os.path.dirname(os.path.abspath(args.testset_dir))
        print(
            f"  [info] --testset-dir is a file; using it as --subset-manifest, "
            f"testset dir = {args.testset_dir}"
        )

    if args.export_trial is not None:
        _export_preset_from_trial(args)
        return

    silence_ids: frozenset[int] = frozenset()
    if args.silence_ids:
        silence_ids = frozenset(
            int(x) for x in args.silence_ids.replace(",", " ").split()
        )

    if args.n_jobs > 1:
        out_dir = args.out_dir or args.testset_dir
        os.makedirs(out_dir, exist_ok=True)
        if args.storage:
            storage = args.storage
        else:
            sqlite_path = os.path.join(
                os.path.abspath(out_dir), f"study_{args.study_name}.db"
            )
            storage = f"sqlite:///{sqlite_path}"
            print(f"  Auto-created study DB: {sqlite_path}")
        # Pre-create the study before spawning workers so they only attach
        # (avoids the cold-start "table studies already exists" race).
        from yoto.tuning.optimize import ensure_study

        ensure_study(args.study_name, storage)
        _run_optimize_preset_parallel(args, storage, silence_ids)
        return

    optimize_preset(
        args.testset_dir,
        out_dir=args.out_dir,
        search_space=args.search_space,
        n_trials=args.n_trials,
        n_jobs=1,
        tag_family=args.tag_family,
        study_name=args.study_name,
        storage=args.storage,
        seed_params=args.seed_params,
        subset=args.subset,
        subset_manifest=args.subset_manifest,
        pruner_startup_trials=args.pruner_startup_trials,
        pruner_warmup_steps=args.pruner_warmup_steps,
        prune_eval_interval=args.prune_eval_interval,
        speed_weight=args.speed_weight,
        speed_floor_recall=args.speed_floor_recall,
        synthetic_blur=args.synthetic_blur,
        max_tag_id=args.max_tag_id,
        silence_ids=silence_ids,
        no_viz=args.no_viz,
        viz_samples=args.viz_samples,
        viz_dir=args.viz_dir,
    )


def _add_build_yolo_dataset_parser(subparsers: argparse._SubParsersAction) -> None:
    from yoto.constants import DEFAULT_TAG_FAMILY

    p = subparsers.add_parser(
        "build-yolo-dataset",
        help=(
            "Curate full-frame AprilTag candidates into a YOLO training set "
            "via a browser GUI"
        ),
    )
    p.add_argument(
        "recordings",
        nargs="*",
        default=[],
        help="Recording dir(s) to auto-discover video+pkl from "
        "(positional; same as --recording)",
    )
    p.add_argument(
        "--recording",
        action="append",
        default=[],
        help="Recording dir(s) to auto-discover video+pkl from (repeatable)",
    )
    p.add_argument(
        "--experiment",
        action="append",
        default=[],
        help="Explicit '<pkl>:<video>' pair (repeatable); pkl may be empty",
    )
    p.add_argument(
        "--pkl-suffix",
        default=None,
        help="Disambiguate when several pkls match a video",
    )
    p.add_argument(
        "--no-pkl",
        type=_str_to_bool,
        default=False,
        help="Ignore pkls; even-stride frames, self-seeded thresholds",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Output/cache directory "
        "(default: <recording>/tracking/training/yolo_dataset)",
    )
    p.add_argument("--total-frames", type=int, default=40)
    p.add_argument("--best-fraction", type=float, default=1.0 / 3.0)
    p.add_argument(
        "--frame-select", choices=["auto", "best-worst", "stride"], default="auto"
    )
    p.add_argument("--family", default=DEFAULT_TAG_FAMILY)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--open", type=_str_to_bool, default=True)
    p.add_argument("--precompute-only", type=_str_to_bool, default=False)
    p.set_defaults(func=_run_build_yolo_dataset)


def _run_build_yolo_dataset(args: argparse.Namespace) -> None:
    from pathlib import Path

    from yoto.detection import _build_apriltag_params_fast, _create_detector
    from yoto.tuning.yolo_dataset import builder, server

    def _chooser(video: Path, pkls: list[Path]) -> Path:
        print(f"Multiple pkls for {video.name}:")
        for i, pk in enumerate(pkls):
            print(f"  [{i}] {pk.name}")
        return pkls[int(input("Select index: "))]

    recordings = list(args.recordings) + list(args.recording)
    exps = builder.resolve_experiments(
        recordings,
        args.experiment,
        args.pkl_suffix,
        args.no_pkl,
        _chooser if sys.stdin.isatty() else None,
    )
    if not exps:
        print("No experiments resolved; pass a recording dir or --experiment.")
        sys.exit(1)

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else builder.default_out_dir(exps[0].video.parent)
    )
    print(f"Output directory: {out_dir}")

    params = _build_apriltag_params_fast()
    detector = _create_detector(params, family=args.family)
    for exp in exps:
        print(f"Precomputing {exp.name} ...")
        builder.precompute_experiment(
            exp,
            out_dir,
            args.total_frames,
            args.best_fraction,
            args.frame_select,
            params,
            detector,
            args.family,
        )

    if args.precompute_only:
        print(f"Cache written to {out_dir}.")
        print("Re-run without --precompute-only to review.")
        return
    server.run_server(out_dir, args.host, args.port, args.open)


def _add_train_parser(subparsers: argparse._SubParsersAction) -> None:
    train_p = subparsers.add_parser(
        "train",
        help=(
            "Preset and model training tools (build-testset, "
            "subsample-testset, optimize-preset, compare-presets, "
            "build-yolo-dataset)"
        ),
    )
    train_sub = train_p.add_subparsers(dest="train_command")
    _add_build_testset_parser(train_sub)
    _add_subsample_testset_parser(train_sub)
    _add_optimize_preset_parser(train_sub)
    _add_compare_presets_parser(train_sub)
    _add_build_yolo_dataset_parser(train_sub)
    train_p.set_defaults(func=_run_train)


def _run_train(args: argparse.Namespace) -> None:
    if not hasattr(args, "train_command") or args.train_command is None:
        import sys as _sys

        print(
            "yoto train — available sub-commands: build-testset, "
            "subsample-testset, optimize-preset, compare-presets"
        )
        print("Run 'yoto train <sub-command> --help' for details.")
        _sys.exit(1)
    args.func(args)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse arguments and dispatch to the appropriate sub-command."""
    parser = argparse.ArgumentParser(
        prog="yoto",
        description=(
            "YOTO — GPU-accelerated AprilTag tracking pipeline for ant "
            "behavioral studies"
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_get_version()}",
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    _add_detect_parser(subparsers)
    _add_clean_parser(subparsers)
    _add_render_parser(subparsers)
    _add_train_parser(subparsers)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    print(f"yoto {_get_version()} — {args.command}")
    args.func(args)


def _get_version() -> str:
    """Return the package version string."""
    from yoto import __version__

    return __version__


if __name__ == "__main__":
    main()
