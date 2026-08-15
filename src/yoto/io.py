"""I/O helpers for loading YOTO output files."""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from yoto.constants import (
    CORNER_COLS,
    CORNER_DTYPE,
    COL_CORNERS,
    DEFAULT_DATANAME,
    PICKLE_EXTS,
)

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


def strip_pickle_ext(path: str) -> str:
    """Strip a trailing ``.pkl.zst`` or ``.pkl`` from *path*.

    Returns *path* unchanged if it ends with neither.
    """
    for ext in PICKLE_EXTS:
        if path.endswith(ext):
            return path[: -len(ext)]
    return path


def is_pickle(path: str) -> bool:
    """Return whether *path* names a pickle in any recognised format."""
    return path.endswith(PICKLE_EXTS)


def find_pickle(base: str) -> str | None:
    """Return the existing pickle for *base*, an extension-less path.

    Prefers the compressed form, then falls back to a plain ``.pkl`` so
    recordings processed before compression keep resolving.  Returns
    ``None`` when neither exists.
    """
    for ext in PICKLE_EXTS:
        candidate = f"{base}{ext}"
        if os.path.isfile(candidate):
            return candidate
    return None


def corner_row(corners: Any) -> dict[str, float]:
    """Expand a ``(4, 2)`` corner array into the flat column mapping.

    Used when building detection rows, so the resulting DataFrame column is
    plain float rather than an object column boxing one ndarray per row.

    Parameters
    ----------
    corners : array-like
        Four ``(x, y)`` pairs in ``lb-rb-rt-lt`` order.

    Returns
    -------
    dict
        Maps each name in :data:`~yoto.constants.CORNER_COLS` to a float.
    """
    flat = np.asarray(corners, dtype=np.float64).reshape(8)
    return {name: float(v) for name, v in zip(CORNER_COLS, flat)}


def has_corners(df: pd.DataFrame) -> bool:
    """Return whether *df* carries corner coordinates in either format."""
    level = -1 if isinstance(df.columns, pd.MultiIndex) else 0
    names = set(df.columns.get_level_values(level))
    return CORNER_COLS[0] in names or COL_CORNERS in names


def load_corners(df: pd.DataFrame, tag_id: Any = None) -> np.ndarray[Any, Any]:
    """Read tag corners out of a detection or tracking DataFrame.

    Accepts both storage formats transparently: the flat float32 columns
    written by current versions, and the legacy object-dtype ``corners``
    column written by older ones.  Rows with no detection come back as NaN.

    Parameters
    ----------
    df : pandas.DataFrame
        A raw (long-format) detection frame, or a cleaned frame with a
        ``(tag_id, metric)`` column MultiIndex.
    tag_id : optional
        For a cleaned frame, the tag to extract.  Ignored for a raw frame,
        which stores one detection per row.

    Returns
    -------
    numpy.ndarray
        ``(n_rows, 4, 2)`` for a raw frame or a single tag; ``(n_frames,
        n_tags, 4, 2)`` for a cleaned frame with *tag_id* omitted, with tags
        in column order (see :func:`corner_tag_ids`).

    Raises
    ------
    KeyError
        If *df* carries no corners, or *tag_id* is not present.
    """
    wide = isinstance(df.columns, pd.MultiIndex)

    if wide and tag_id is None:
        ids = corner_tag_ids(df)
        if not ids:
            raise KeyError("DataFrame has no corner columns.")
        return np.stack([load_corners(df, t) for t in ids], axis=1)

    if wide:
        sub = df[tag_id]
        if CORNER_COLS[0] in sub.columns:
            flat = sub.loc[:, list(CORNER_COLS)].to_numpy(dtype=CORNER_DTYPE)
            return flat.reshape(len(sub), 4, 2)
        if COL_CORNERS in sub.columns:
            return _corners_from_object(sub[COL_CORNERS])
        raise KeyError(f"Tag {tag_id!r} has no corner columns.")

    if CORNER_COLS[0] in df.columns:
        flat = df.loc[:, list(CORNER_COLS)].to_numpy(dtype=CORNER_DTYPE)
        return flat.reshape(len(df), 4, 2)
    if COL_CORNERS in df.columns:
        return _corners_from_object(df[COL_CORNERS])
    raise KeyError("DataFrame has no corner columns.")


def corner_tag_ids(df: pd.DataFrame) -> list[Any]:
    """Return the tag IDs in *df* that carry corners, in column order."""
    if not isinstance(df.columns, pd.MultiIndex):
        return []
    wanted = {CORNER_COLS[0], COL_CORNERS}
    seen: list[Any] = []
    for tag, metric in df.columns:
        if metric in wanted and tag not in seen:
            seen.append(tag)
    return seen


def _corners_from_object(col: pd.Series) -> np.ndarray[Any, Any]:
    """Stack a legacy object-dtype corners column into a ``(n, 4, 2)`` array."""
    out = np.full((len(col), 4, 2), np.nan, dtype=CORNER_DTYPE)
    for i, v in enumerate(col.to_numpy()):
        if v is None:
            continue
        arr = np.asarray(v, dtype=np.float64)
        if arr.shape == (4, 2):
            out[i] = arr
    return out


def load_data(
    path: str | os.PathLike[str],
    dataname: str = DEFAULT_DATANAME,
    video_nb: int | None = None,
    corners: bool = True,
) -> pd.DataFrame:
    """Load and concatenate clean tracking pickles produced by ``yoto clean``.

    *path* can be either a **recording folder** (loads every matching pickle
    inside ``<folder>/tracking/clean_data/``) or a **single video file**
    (loads only that video's pickle from the same sub-directory).

    A whole recording is often far too large to hold in memory — see
    ``video_nb`` to load one video at a time.

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
    video_nb : int, optional
        Load a single video instead of the whole recording.  This is a
        **position** in the sorted list of matching pickles, not a filename
        number: ``0`` is the first video that has a clean pickle for this
        *dataname*.  The two coincide for the usual ``000000.mp4``,
        ``000001.mp4`` … layout, but they drift apart if a video was skipped
        at clean time.  Negative values index from the end.  The returned
        frame still carries the full ``(frame, source, video_frame)``
        MultiIndex, so ``df.index.get_level_values("source")[0]`` tells you
        which video you actually got.
    corners : bool
        Keep the per-tag corner coordinates — the eight
        :data:`~yoto.constants.CORNER_COLS` metrics, or the legacy
        ``corners`` column on older pickles.  That is eight numbers per tag
        per frame against four for ``center_x``, ``center_y``, ``ass_type``
        and ``distance`` combined, so dropping them roughly halves a loaded
        frame.  Pass ``False`` when you only need trajectories.  Corners are
        dropped per video as each pickle is read, so a multi-video load never
        holds more than one video's worth at a time; the pickle itself is
        still read in full, since that is how pandas pickles work.  Use
        :func:`load_corners` to get corners back as a ``(n, 4, 2)`` array.

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
    IndexError
        If *video_nb* is out of range for the matching pickles.
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
    stem_suffix = f"{norm}_clean" if norm else "_clean"

    # Compressed and uncompressed pickles coexist: a recording cleaned before
    # compression keeps its plain '.pkl' files, and both are loaded the same
    # way since pandas infers the codec from the extension.
    suffixes = tuple(f"{stem_suffix}{ext}" for ext in PICKLE_EXTS)

    def _matches(name: str) -> bool:
        for suf in suffixes:
            if name.endswith(suf):
                return stem_filter is None or name == f"{stem_filter}{suf}"
        return False

    pickles = sorted(
        p
        for p in clean_dir.iterdir()
        if p.is_file() and not p.name.startswith(".") and _matches(p.name)
    )
    if not pickles:
        hint = f" for video {stem_filter!r}" if stem_filter else ""
        raise FileNotFoundError(
            f"No '*{stem_suffix}[{'|'.join(PICKLE_EXTS)}]' files in "
            f"{clean_dir}{hint}. Check the --dataname argument."
        )

    if video_nb is not None:
        if not -len(pickles) <= video_nb < len(pickles):
            raise IndexError(
                f"video_nb={video_nb} is out of range: {len(pickles)} video(s) "
                f"match '*{stem_suffix}' in {clean_dir}."
            )
        pickles = [pickles[video_nb]]

    frames: list[pd.DataFrame] = []
    sources: list[str] = []
    attrs_list: list[dict[str, Any]] = []

    for pkl in pickles:
        df = pd.read_pickle(pkl)
        if not corners:
            present = set(df.columns.get_level_values(-1))
            drop = [c for c in (*CORNER_COLS, COL_CORNERS) if c in present]
            if drop:
                df = df.drop(columns=drop, level=-1)
        source = next(pkl.name[: -len(s)] for s in suffixes if pkl.name.endswith(s))
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
