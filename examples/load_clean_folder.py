"""Load every cleaned tracking pickle in a folder into one DataFrame.

Standalone helper for sharing.  Depends only on ``pandas``.

Each ``<stem>_clean.pkl`` written by ``yoto clean`` contains a DataFrame
with a MultiIndex of ``(tag_id, metric)`` columns indexed by frame number,
plus run metadata stored in ``df.attrs`` (yoto version, model weights,
tag family, …).

``load_clean_folder`` finds every clean pickle in *folder* whose stem
ends with ``_<dataname>`` (e.g. ``--dataname yoto_0.10.x`` →
``*_yoto_0.10.x_clean.pkl``), concatenates them vertically into one
DataFrame with a row MultiIndex of ``(source_video, frame)``, and merges
the per-pickle ``.attrs`` dictionaries:

* If every pickle has the same value for an attribute, it is preserved
  as a scalar.
* If values differ, the attribute becomes a list ordered the same as
  the input pickles.  A warning is emitted unless the attribute is in
  ``_EXPECTED_VARYING_KEYS`` (paths, per-video counters, …) where
  per-video variation is normal.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# Attribute keys that legitimately vary from one video to the next.
# Listing them here keeps the merge silent for the expected cases and
# loud for everything else (e.g. a yoto version mismatch).
_EXPECTED_VARYING_KEYS = frozenset(
    {
        # Per-video identity / paths.
        "yoto_video",
        "yoto_video_path",
        "yoto_video_stem",
        # Per-video timing / shape.
        "yoto_frame_count",
        "yoto_fps",
        "yoto_width",
        "yoto_height",
        "yoto_duration_s",
        "yoto_start_time",
        "yoto_end_time",
        "yoto_processed_at",
        "yoto_n_frames_processed",
        # Per-video timestamps stamped by detect/clean.
        "yoto_created_utc",
        "yoto_cleaned_utc",
        # Per-video counters from the clean pass.
        "yoto_scale_sample_count",
        "yoto_yolo_filled_count",
        "yoto_yolo_pruned_count",
        "yoto_yolo_rechained_count",
    }
)


def _values_equal(a: Any, b: Any) -> bool:
    """Equality that works for scalars, lists, tuples and numpy arrays."""
    try:
        if a is b:
            return True
        # numpy / pandas objects expose .equals
        if hasattr(a, "equals") and hasattr(b, "equals"):
            return bool(a.equals(b))
        return bool(a == b)
    except Exception:
        return False


def _merge_attrs(
    attrs_list: list[dict[str, Any]],
    sources: list[str],
) -> dict[str, Any]:
    """Merge a list of ``df.attrs`` dicts into one.

    Same value across all inputs → scalar.  Differing values → list in
    input order.  Differences outside ``_EXPECTED_VARYING_KEYS`` emit a
    ``UserWarning``.
    """
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
                stacklevel=2,
            )

    return merged


def load_clean_folder(folder: str | os.PathLike[str], dataname: str) -> pd.DataFrame:
    """Load and concatenate every clean pickle in *folder* matching *dataname*.

    Parameters
    ----------
    folder : str or path-like
        Directory containing ``<video_stem>_<dataname>_clean.pkl`` files
        written by ``yoto clean``.
    dataname : str
        The ``--dataname`` suffix used at detect/clean time, without the
        leading underscore (e.g. ``"yoto_0.10.x"``).

    Returns
    -------
    pandas.DataFrame
        All rows from every matching pickle, concatenated vertically with
        a ``(source, frame)`` row MultiIndex.  Original ``(tag_id, metric)``
        column MultiIndex is preserved.  Merged ``.attrs`` are attached
        to the returned DataFrame.

    Raises
    ------
    FileNotFoundError
        If *folder* does not exist or no matching pickle is found.
    """
    folder = Path(folder)
    clean_dir = folder / "tracking" / "clean_data"
    if not clean_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {clean_dir}")

    suffix = f"_{dataname}_clean.pkl" if dataname else "_clean.pkl"
    pickles = sorted(
        p
        for p in clean_dir.iterdir()
        if p.is_file() and not p.name.startswith(".") and p.name.endswith(suffix)
    )
    if not pickles:
        raise FileNotFoundError(
            f"No '*{suffix}' files in {clean_dir}. " "Check the --dataname argument."
        )

    frames: list[pd.DataFrame] = []
    sources: list[str] = []
    attrs_list: list[dict[str, Any]] = []

    for path in pickles:
        df = pd.read_pickle(path)
        # Strip the suffix so the source key is the video stem.
        source = path.name[: -len(suffix)]
        frames.append(df)
        sources.append(source)
        attrs_list.append(dict(df.attrs))

    # Concatenate first under (source, video_frame), then prepend a
    # continuous global ``frame`` level so time-series analysis over the
    # whole folder can use a single monotonic index.
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

    # Median mm-per-px across pickles, exposed as ``scale`` for downstream
    # code that wants a single conversion factor for the merged frame.
    mm_per_px = [
        a.get("yoto_mm_per_px")
        for a in attrs_list
        if a.get("yoto_mm_per_px") is not None
    ]
    if mm_per_px:
        merged.attrs["scale"] = float(np.median(mm_per_px))

    return merged


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", help="Folder containing *_clean.pkl files.")
    parser.add_argument(
        "--dataname",
        required=True,
        help="Detect/clean run suffix, e.g. 'yoto_0.10.x'.",
    )
    args = parser.parse_args()

    df = load_clean_folder(args.folder, args.dataname)
    print(
        f"Loaded {df.index.get_level_values('source').nunique()} videos, "
        f"{len(df)} rows, {df.shape[1]} columns."
    )
    print("Merged attrs:")
    for k, v in df.attrs.items():
        if isinstance(v, list):
            print(f"  {k}: <list of {len(v)}>")
        else:
            print(f"  {k}: {v!r}")
