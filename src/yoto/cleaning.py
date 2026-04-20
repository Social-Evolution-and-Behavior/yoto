"""Data cleaning and interpolation for AprilTag tracking data.

This module post-processes the raw detection DataFrame produced by the
detection pipeline.  It removes spurious IDs, fills short gaps via
linear interpolation, detects and deletes tracking jumps, and computes
quality metrics.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from yoto.constants import (
    ASS_TYPE_INTERPOLATED,
    ASS_TYPE_NONE,
    ASS_TYPE_ORIGINAL,
    COL_ASS_TYPE,
    COL_CENTER_X,
    COL_CENTER_Y,
    COL_DISTANCE,
    DEFAULT_INTERPOLATION_LIMIT,
    DEFAULT_MAX_JUMP_DISTANCE,
    MIN_DETECTIONS_PER_ID,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Quality metrics type
# ---------------------------------------------------------------------------

CleaningMetrics = dict[str, int | float]
"""Dictionary of quality metrics returned by :func:`clean_tracking_data`."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _interpolate_data(
    frame_data: pd.DataFrame,
    limit: int = DEFAULT_INTERPOLATION_LIMIT,
) -> pd.DataFrame:
    """Linearly interpolate short gaps in all numeric columns.

    Parameters
    ----------
    frame_data : pd.DataFrame
        Tracking DataFrame with MultiIndex columns.
    limit : int
        Maximum number of consecutive NaN frames to bridge.

    Returns
    -------
    pd.DataFrame
        DataFrame with gaps of at most *limit* frames filled.
    """
    for col in frame_data.columns:
        if pd.api.types.is_numeric_dtype(frame_data[col]):
            frame_data[col] = frame_data[col].interpolate(
                method="linear",
                limit=limit,
                limit_direction="both",
                limit_area="inside",
            )
    return frame_data


def _delete_jump_blocks(
    frame_data: pd.DataFrame,
    tag_id: Any,
    max_distance: float = DEFAULT_MAX_JUMP_DISTANCE,
) -> tuple[pd.DataFrame, pd.Index]:
    """Remove contiguous blocks of frames where a tag jumped unrealistically.

    Within each contiguous block of ``distance > max_distance`` frames,
    the last *original* detection is preserved (so subsequent
    interpolation has a valid anchor) and everything else is cleared.

    Parameters
    ----------
    frame_data : pd.DataFrame
        Tracking DataFrame.
    tag_id : Any
        Column-level tag identifier.
    max_distance : float
        Pixel-distance threshold above which a detection is considered a
        tracking jump.

    Returns
    -------
    tuple[pd.DataFrame, pd.Index]
        Updated DataFrame and the index of deleted rows.
    """
    delete_cols = (COL_CENTER_X, COL_CENTER_Y, COL_DISTANCE)

    sub = frame_data[tag_id]
    jump = sub[COL_DISTANCE].gt(max_distance)
    grp = jump.ne(jump.shift(fill_value=False)).cumsum()
    grp = grp.where(jump)

    to_delete: list[pd.Index] = []

    for g, idx in grp.groupby(grp).groups.items():
        if pd.isna(g):
            continue

        rows = sub.index[list(idx)]
        real_rows = rows[sub.loc[rows, COL_ASS_TYPE].eq(ASS_TYPE_ORIGINAL)]

        # Keep only the last real detection in the block
        if len(real_rows):
            keep = real_rows[-1]
            rows = rows[rows != keep]

        if sub.loc[idx[0] - 1, COL_ASS_TYPE] == ASS_TYPE_INTERPOLATED:
            to_delete.append(pd.Index([idx[0] - 1]))
        to_delete.append(rows)

    if not to_delete:
        return frame_data, pd.Index([])

    del_idx = to_delete[0].append(to_delete[1:]) if len(to_delete) > 1 else to_delete[0]

    # Clear only this tag's data on the affected frames
    frame_data.loc[del_idx, (tag_id, list(delete_cols))] = np.nan
    frame_data.loc[del_idx, (tag_id, COL_ASS_TYPE)] = ASS_TYPE_NONE

    return frame_data, del_idx


def _compute_distances(
    frame_data: pd.DataFrame,
    id_list: np.ndarray[Any, np.dtype[Any]],
) -> pd.DataFrame:
    """Compute frame-to-frame Euclidean distances for every tag.

    Parameters
    ----------
    frame_data : pd.DataFrame
        Tracking DataFrame.
    id_list : ndarray
        Array of unique tag IDs.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``(tag_id, "distance")`` columns appended.
    """
    x_vals = frame_data.xs(COL_CENTER_X, axis=1, level=1)[id_list].to_numpy()
    y_vals = frame_data.xs(COL_CENTER_Y, axis=1, level=1)[id_list].to_numpy()

    dx = np.diff(x_vals, axis=0)
    dy = np.diff(y_vals, axis=0)
    distances = np.sqrt(dx * dx + dy * dy)

    out = np.full((x_vals.shape[0], x_vals.shape[1]), np.nan, dtype=float)
    out[1:, :] = distances

    dist_cols = pd.MultiIndex.from_product(
        [id_list, [COL_DISTANCE]],
        names=frame_data.columns.names,
    )
    dist_df = pd.DataFrame(out, index=frame_data.index, columns=dist_cols)
    return frame_data.join(dist_df)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def clean_tracking_data(
    frame_data: pd.DataFrame,
    min_detections: int = MIN_DETECTIONS_PER_ID,
    interpolation_limit: int = DEFAULT_INTERPOLATION_LIMIT,
    max_jump_distance: float = DEFAULT_MAX_JUMP_DISTANCE,
) -> tuple[pd.DataFrame, np.ndarray[Any, np.dtype[Any]], CleaningMetrics]:
    """Clean, interpolate, and validate raw AprilTag tracking data.

    Processing steps
    ----------------
    1. Remove tag IDs that have fewer than *min_detections* valid frames.
    2. Create an ``ass_type`` column to track data provenance (original
       vs. interpolated).
    3. Fill short gaps (up to *interpolation_limit* frames) via linear
       interpolation.
    4. Compute frame-to-frame distances.
    5. Delete unrealistic jumps (distance > *max_jump_distance*).
    6. Re-interpolate and recompute distances after deletion.

    Parameters
    ----------
    frame_data : pd.DataFrame
        Raw tracking DataFrame produced by the detection pipeline.
    min_detections : int
        Minimum non-NaN detections to keep a tag ID.
    interpolation_limit : int
        Maximum gap length (frames) for linear interpolation.
    max_jump_distance : float
        Pixel-distance threshold for jump detection.

    Returns
    -------
    tuple[pd.DataFrame, ndarray, CleaningMetrics]
        ``(cleaned_dataframe, id_list, metrics_dict)``

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.read_pickle("raw_tracking.pkl")  # doctest: +SKIP
    >>> cleaned, ids, metrics = clean_tracking_data(df)  # doctest: +SKIP
    >>> print(f"Error rate: {metrics['error_pct']:.2f}%")  # doctest: +SKIP
    """
    id_list = np.unique(frame_data.columns.get_level_values(0))

    # --- Step 1: remove low-count IDs ---
    for tag_id in id_list:
        non_na_x = frame_data[tag_id][COL_CENTER_X].notna().sum()
        if non_na_x < min_detections:
            columns_to_drop = [col for col in frame_data.columns if col[0] == tag_id]
            frame_data = frame_data.drop(columns=columns_to_drop)

    frame_data.columns = frame_data.columns.remove_unused_levels()
    id_list = np.unique(frame_data.columns.get_level_values(0))

    # Baseline metrics
    total_samples = int(len(frame_data.index) * len(id_list))
    original_non_na = frame_data.xs(COL_CENTER_X, axis=1, level=1)[id_list].notna()
    original_good_count = int(original_non_na.values.sum())
    original_missing_count = total_samples - original_good_count

    # --- Step 2: create ass_type column ---
    new_cols = pd.MultiIndex.from_tuples(
        [(tag_id, COL_ASS_TYPE) for tag_id in id_list],
        names=frame_data.columns.names,
    )
    new_data = pd.DataFrame(
        ASS_TYPE_NONE,
        index=frame_data.index,
        columns=new_cols,
    )
    frame_data = pd.concat([frame_data, new_data], axis=1)

    for tag_id in id_list:
        frame_data.loc[
            frame_data[tag_id][COL_CENTER_X].notna(),
            (tag_id, COL_ASS_TYPE),
        ] = ASS_TYPE_ORIGINAL

    # --- Step 3: first interpolation pass ---
    frame_data = _interpolate_data(frame_data, limit=interpolation_limit)
    for tag_id in id_list:
        mask = (frame_data[(tag_id, COL_ASS_TYPE)] == ASS_TYPE_NONE) & frame_data[
            (tag_id, COL_CENTER_X)
        ].notna()
        frame_data.loc[mask, (tag_id, COL_ASS_TYPE)] = ASS_TYPE_INTERPOLATED

    # --- Step 4: compute distances ---
    frame_data = _compute_distances(frame_data, id_list)

    # --- Step 5: delete jumps ---
    before_non_na = frame_data.xs(COL_CENTER_X, axis=1, level=1)[id_list].notna()
    for tag_id in id_list:
        frame_data, _ = _delete_jump_blocks(
            frame_data, tag_id, max_distance=max_jump_distance
        )
    after_non_na = frame_data.xs(COL_CENTER_X, axis=1, level=1)[id_list].notna()
    jump_deleted_count = int((before_non_na & ~after_non_na).values.sum())

    # --- Step 6: re-interpolate after deletion ---
    frame_data = _interpolate_data(frame_data, limit=interpolation_limit)
    for tag_id in id_list:
        mask = (frame_data[(tag_id, COL_ASS_TYPE)] == ASS_TYPE_NONE) & frame_data[
            (tag_id, COL_CENTER_X)
        ].notna()
        frame_data.loc[mask, (tag_id, COL_ASS_TYPE)] = ASS_TYPE_INTERPOLATED

        # Recompute per-ID distances
        x = frame_data.loc[:, (tag_id, COL_CENTER_X)]
        y = frame_data.loc[:, (tag_id, COL_CENTER_Y)]
        distances = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
        frame_data.loc[:, (tag_id, COL_DISTANCE)] = np.nan
        frame_data.loc[1:, (tag_id, COL_DISTANCE)] = distances

    filled_count = int(
        (
            frame_data.xs(COL_ASS_TYPE, axis=1, level=1)[id_list]
            == ASS_TYPE_INTERPOLATED
        ).values.sum()
    )

    total_gaps = original_missing_count + jump_deleted_count
    total_detections = original_good_count + jump_deleted_count

    metrics: CleaningMetrics = {
        "total_samples": total_samples,
        "total_detections": total_detections,
        "original_good_count": original_good_count,
        "original_bad_count": jump_deleted_count,
        "original_missing_count": original_missing_count,
        "total_gaps": total_gaps,
        "error_pct": (
            (100.0 * jump_deleted_count / total_detections) if total_detections else 0.0
        ),
        "original_bad_pct": (
            (100.0 * jump_deleted_count / total_samples) if total_samples else 0.0
        ),
        "original_missing_pct": (
            (100.0 * original_missing_count / total_samples) if total_samples else 0.0
        ),
        "filled_count": filled_count,
        "filled_pct_of_total": (
            (100.0 * filled_count / total_samples) if total_samples else 0.0
        ),
        "filled_pct_of_gaps": (
            (100.0 * filled_count / total_gaps) if total_gaps else 0.0
        ),
    }

    logger.info(
        "Cleaning complete: detections=%d/%d (%.2f%%), errors=%d (%.2f%%), "
        "filled=%d/%d gaps (%.2f%% recovered)",
        metrics["total_detections"],
        metrics["total_samples"],
        (
            metrics["total_detections"] / metrics["total_samples"] * 100
            if metrics["total_samples"]
            else 0
        ),
        metrics["original_bad_count"],
        metrics["error_pct"],
        metrics["filled_count"],
        metrics["total_gaps"],
        metrics["filled_pct_of_gaps"],
    )

    return frame_data, id_list, metrics
