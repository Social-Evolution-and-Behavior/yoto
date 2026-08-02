"""I/O helpers for loading YOTO output files."""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from yoto.constants import DEFAULT_DATANAME


# Attribute keys that legitimately vary from one video to the next.
_EXPECTED_VARYING_KEYS = frozenset(
    {
        "yoto_video",
        "yoto_video_path",
        "yoto_video_stem",
        "yoto_frame_count",
        "yoto_fps",
        "yoto_width",
        "yoto_height",
        "yoto_duration_s",
        "yoto_start_time",
        "yoto_end_time",
        "yoto_processed_at",
        "yoto_n_frames_processed",
        "yoto_created_utc",
        "yoto_cleaned_utc",
        "yoto_scale_sample_count",
        "yoto_yolo_filled_count",
        "yoto_yolo_pruned_count",
        "yoto_yolo_rechained_count",
    }
)


def _values_equal(a: Any, b: Any) -> bool:
    try:
        if a is b:
            return True
        if hasattr(a, "equals") and hasattr(b, "equals"):
            return bool(a.equals(b))
        return bool(a == b)
    except Exception:
        return False


def _merge_attrs(
    attrs_list: list[dict[str, Any]],
    sources: list[str],
) -> dict[str, Any]:
    all_keys: set[str] = set()
    for a in attrs_list:
        all_keys.update(a.keys())

    merged: dict[str, Any] = {}
    for key in sorted(all_keys):
        values = [a.get(key) for a in attrs_list]
        first = values[0]
        if all(_values_equal(first, v) for v in values[1:]):
            merged[key] = first
            continue

        merged[key] = values
        if key not in _EXPECTED_VARYING_KEYS:
            sample = ", ".join(f"{s}={v!r}" for s, v in list(zip(sources, values))[:3])
            warnings.warn(
                f"Attribute {key!r} differs across pickles "
                f"(stored as list of {len(values)}). Sample: {sample}",
                UserWarning,
                stacklevel=3,
            )

    return merged


def load_data(
    path: str | os.PathLike[str],
    dataname: str = DEFAULT_DATANAME,
) -> pd.DataFrame:
    """Load and concatenate clean tracking pickles produced by ``yoto clean``.

    *path* can be either a **recording folder** (loads every matching pickle
    inside ``<folder>/tracking/clean_data/``) or a **single video file**
    (loads only that video's pickle from the same sub-directory).

    Parameters
    ----------
    path : str or path-like
        Recording folder *or* a video file path.  When a file is given its
        parent directory is used as the recording folder and only the pickle
        matching that video stem is loaded.
    dataname : str
        The ``--dataname`` suffix used at detect/clean time.  Follows the
        same leading-underscore convention as the CLI
        (e.g. ``"_apriltagDetect14"``).  Defaults to
        :data:`~yoto.constants.DEFAULT_DATANAME`.

    Returns
    -------
    pandas.DataFrame
        All rows concatenated vertically with a ``(frame, source,
        video_frame)`` row MultiIndex — ``frame`` is a global monotonic
        counter, ``source`` is the video stem, ``video_frame`` is the
        original per-video frame number.  Original ``(tag_id, metric)``
        column MultiIndex is preserved.  Merged ``.attrs`` are attached,
        including ``scale`` (median mm/px across all pickles).

    Raises
    ------
    FileNotFoundError
        If the ``tracking/clean_data/`` directory does not exist or no
        matching pickle is found.
    """
    p = Path(path)

    # Resolve folder and optional stem filter.
    if p.is_file():
        folder = p.parent
        stem_filter: str | None = p.stem
    else:
        folder = p
        stem_filter = None

    clean_dir = folder / "tracking" / "clean_data"
    if not clean_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {clean_dir}")

    # Normalise dataname: ensure leading underscore for non-empty values.
    norm = dataname if (not dataname or dataname.startswith("_")) else f"_{dataname}"
    suffix = f"{norm}_clean.pkl" if norm else "_clean.pkl"

    pickles = sorted(
        p
        for p in clean_dir.iterdir()
        if p.is_file()
        and not p.name.startswith(".")
        and p.name.endswith(suffix)
        and (stem_filter is None or p.name == f"{stem_filter}{suffix}")
    )
    if not pickles:
        hint = f" for video {stem_filter!r}" if stem_filter else ""
        raise FileNotFoundError(
            f"No '*{suffix}' files in {clean_dir}{hint}. "
            "Check the --dataname argument."
        )

    frames: list[pd.DataFrame] = []
    sources: list[str] = []
    attrs_list: list[dict[str, Any]] = []

    for pkl in pickles:
        df = pd.read_pickle(pkl)
        source = pkl.name[: -len(suffix)]
        frames.append(df)
        sources.append(source)
        attrs_list.append(dict(df.attrs))

    merged = pd.concat(frames, keys=sources, names=["source", "video_frame"])
    merged.index = pd.MultiIndex.from_arrays(
        [
            np.arange(len(merged), dtype=np.int64),
            merged.index.get_level_values("source"),
            merged.index.get_level_values("video_frame"),
        ],
        names=["frame", "source", "video_frame"],
    )

    merged.attrs = _merge_attrs(attrs_list, sources)

    mm_per_px = [
        a.get("yoto_mm_per_px")
        for a in attrs_list
        if a.get("yoto_mm_per_px") is not None
    ]
    if mm_per_px:
        merged.attrs["scale"] = float(np.median(mm_per_px))

    return merged
