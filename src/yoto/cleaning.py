"""Data cleaning and interpolation for AprilTag tracking data.

This module post-processes the raw detection DataFrame produced by the
detection pipeline.  It removes spurious IDs, fills short gaps via
linear interpolation, detects and deletes tracking jumps, and computes
quality metrics.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from yoto.exceptions import EmptyTrackingError
from yoto.constants import (
    ASS_TYPE_INTERPOLATED,
    ASS_TYPE_NONE,
    ASS_TYPE_ORIGINAL,
    ASS_TYPE_YOLO_INFERRED,
    COL_ASS_TYPE,
    COL_CENTER_X,
    COL_CENTER_Y,
    COL_CORNERS,
    COL_DISTANCE,
    DEFAULT_FINAL_JUMP_PASS,
    DEFAULT_GOOD_TRACK_PERCENTILE,
    DEFAULT_INTERPOLATION_LIMIT,
    DEFAULT_MAX_CONSECUTIVE_MISSES,
    DEFAULT_MAX_JUMP_DISTANCE,
    DEFAULT_MIN_GAP_RECOVERY_FRAMES,
    DEFAULT_RECHAIN_AFFECTED_ONLY,
    DEFAULT_RECOVER_LONG_GAPS,
    DEFAULT_SCALE_SAMPLE_SIZE,
    DEFAULT_SNAP_MULTIPLIER,
    DEFAULT_TAG_SIZE_MM,
    DEFAULT_YOLO_FILL_LIMIT,
    MIN_DETECTIONS_PER_ID,
)

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
    preserve_last_original: bool = True,
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

        # ``idx`` is a pd.Index of *labels* (frame numbers).  Use it
        # directly rather than ``sub.index[list(idx)]`` which would
        # treat the labels as positional offsets — only correct when
        # the index happens to be ``0..n-1``.  For sliced inputs
        # (notebook windows, partial videos) the labels don't start
        # at 0, so positional indexing crashes or grabs the wrong rows.
        rows = pd.Index(idx)
        if preserve_last_original:
            real_rows = rows[sub.loc[rows, COL_ASS_TYPE].eq(ASS_TYPE_ORIGINAL)]
            # Keep only the last real detection in the block so subsequent
            # interpolation has an anchor.  Step 7b skips this: after
            # YOLO-fill, YOLO_INFERRED data provides continuity and an
            # isolated misdecoded ORIGINAL must be removed entirely.
            if len(real_rows):
                keep = real_rows[-1]
                rows = rows[rows != keep]

        # The frame just before this jump block — guarded against the
        # edge case where the block starts at the very first row of
        # the slice (no prior frame to inspect).
        prev_label = idx[0] - 1
        if prev_label in sub.index and (
            sub.loc[prev_label, COL_ASS_TYPE] == ASS_TYPE_INTERPOLATED
        ):
            to_delete.append(pd.Index([prev_label]))
        to_delete.append(rows)

    if not to_delete:
        return frame_data, pd.Index([])

    del_idx = to_delete[0].append(to_delete[1:]) if len(to_delete) > 1 else to_delete[0]

    # Clear only this tag's data on the affected frames
    frame_data.loc[del_idx, (tag_id, list(delete_cols))] = np.nan
    frame_data.loc[del_idx, (tag_id, COL_ASS_TYPE)] = ASS_TYPE_NONE

    return frame_data, del_idx


def _compute_snap_threshold(
    frame_data: pd.DataFrame,
    id_list: np.ndarray[Any, np.dtype[Any]],
    percentile: float = DEFAULT_GOOD_TRACK_PERCENTILE,
) -> float:
    """Per-frame movement threshold derived from "good" track segments.

    For each tag, collect frame-to-frame Euclidean distances between
    pairs of *consecutive* frames where both endpoints have
    ``ass_type == ORIGINAL``.  Consecutive in *frame index* — not just
    in the row count — so gaps don't bias the distribution.  Return
    the *percentile*-th value across all collected distances.

    Used by :func:`_fill_via_yolo` as the maximum distance an undecoded
    YOLO box can be from a tag's last-known position to be accepted as
    that tag's position on a gap frame.

    Returns
    -------
    float
        The threshold in pixels.  ``inf`` when no good consecutive
        pairs exist (e.g., on a synthetic single-frame DataFrame),
        which effectively disables snapping.
    """
    distances: list[float] = []
    for tag_id in id_list:
        x = frame_data[(tag_id, COL_CENTER_X)].to_numpy()
        y = frame_data[(tag_id, COL_CENTER_Y)].to_numpy()
        types = frame_data[(tag_id, COL_ASS_TYPE)].to_numpy()
        frames = frame_data.index.to_numpy()

        both_original = (types[:-1] == ASS_TYPE_ORIGINAL) & (
            types[1:] == ASS_TYPE_ORIGINAL
        )
        consecutive = (frames[1:] - frames[:-1]) == 1
        keep = both_original & consecutive
        if not keep.any():
            continue
        dx = x[1:][keep] - x[:-1][keep]
        dy = y[1:][keep] - y[:-1][keep]
        distances.extend(np.sqrt(dx * dx + dy * dy).tolist())

    if not distances:
        return float("inf")
    return float(np.percentile(distances, percentile))


def _chain_pass(
    x_arr: np.ndarray[Any, np.dtype[np.float64]],
    y_arr: np.ndarray[Any, np.dtype[np.float64]],
    t_arr: np.ndarray[Any, np.dtype[np.int64]],
    frames: np.ndarray[Any, np.dtype[Any]],
    by_frame: dict[Any, tuple[np.ndarray, np.ndarray]],
    snap_max: float,
    max_consecutive_misses: int,
    yolo_fill_limit: int,
    forward: bool,
    per_tag_forbidden: dict[int, set[int]] | None = None,
    candidate_tag_indices: set[int] | None = None,
    claimed_per_frame: dict[int, set[int]] | None = None,
    pass_label: str = "forward",
) -> tuple[int, list[dict[str, Any]]]:
    """Single forward or backward chain pass over the frame range.

    For each tag, walks frames in the requested direction.  Tracks an
    anchor (last known position) per tag plus a consecutive-miss
    counter and a since-last-anchor age.  An ORIGINAL or already-filled
    YOLO_INFERRED cell refreshes the anchor.  On a non-anchor frame
    with an active anchor:

    * If the closest undecoded YOLO box on this frame is within
      ``snap_max`` of the anchor → snap; reset misses and age.
    * Else → increment misses; clear anchor after
      ``max_consecutive_misses`` consecutive non-snaps.

    The optional ``yolo_fill_limit`` (> 0) clears the anchor when it's
    been more than that many frames since the last refresh, regardless
    of misses.  ``0`` disables the cap.

    Mutates ``x_arr / y_arr / t_arr`` in place.  Returns ``(filled,
    claims)`` where *claims* is a list of dicts ``{frame_idx, frame,
    box_id, tag_idx, pass, distance}``.

    ``per_tag_forbidden`` maps tag_idx → set of box_ids that tag may not
    snap to.  ``candidate_tag_indices`` restricts which tags are active
    candidates (others still refresh anchors from existing data but don't
    snap to new boxes).  ``claimed_per_frame`` maps frame_label → set of
    box_ids that are already in use at that frame and must not be
    snapped to by *any* tag; the dict is mutated in place so a forward
    pass populates it and a subsequent backward pass inherits its
    claims.  Pre-populating it before the forward call also blocks
    boxes that existing YOLO_INFERRED cells already use.
    """
    n_frames, n_tags = x_arr.shape

    anchor_x = np.full(n_tags, np.nan, dtype=float)
    anchor_y = np.full(n_tags, np.nan, dtype=float)
    misses = np.zeros(n_tags, dtype=int)
    age = np.zeros(n_tags, dtype=int)

    step = 1 if forward else -1
    start = 0 if forward else n_frames - 1
    stop = n_frames if forward else -1

    filled = 0
    claims: list[dict[str, Any]] = []
    for i in range(start, stop, step):
        entry = by_frame.get(frames[i])
        if entry is None:
            boxes = None
            box_ids = None
        else:
            boxes, box_ids = entry

        # Refresh anchors from any cell that has *real* evidence at
        # this frame — ORIGINAL detections, or YOLO_INFERRED cells from
        # a previous pass / earlier in this pass.  Linear-INTERPOLATED
        # cells deliberately don't refresh: they're a guess, and we
        # want to overwrite them with YOLO evidence where possible.
        refresh = (t_arr[i] == ASS_TYPE_ORIGINAL) | (t_arr[i] == ASS_TYPE_YOLO_INFERRED)
        if refresh.any():
            anchor_x[refresh] = x_arr[i, refresh]
            anchor_y[refresh] = y_arr[i, refresh]
            misses[refresh] = 0
            age[refresh] = 0

        # Candidates: tags with an active anchor that don't already
        # have ORIGINAL or YOLO_INFERRED evidence on this frame.
        # INTERPOLATED cells *are* candidates — a successful snap
        # upgrades them to YOLO_INFERRED.
        has_anchor = ~np.isnan(anchor_x)
        candidate_idx = np.where(has_anchor & ~refresh)[0]
        if candidate_tag_indices is not None and len(candidate_idx):
            candidate_idx = candidate_idx[
                np.isin(candidate_idx, list(candidate_tag_indices))
            ]
        if len(candidate_idx) == 0:
            # Still age out anchors that aren't being refreshed.
            non_refresh = has_anchor & ~refresh
            age[non_refresh] += 1
            if yolo_fill_limit > 0:
                stale = age > yolo_fill_limit
                anchor_x[stale] = np.nan
                anchor_y[stale] = np.nan
                misses[stale] = 0
                age[stale] = 0
            continue

        # If no YOLO boxes on this frame, every candidate misses.
        if boxes is None or box_ids is None or len(boxes) == 0:
            misses[candidate_idx] += 1
            age[candidate_idx] += 1
            broken = misses >= max_consecutive_misses
            if broken.any():
                anchor_x[broken] = np.nan
                anchor_y[broken] = np.nan
                misses[broken] = 0
                age[broken] = 0
            if yolo_fill_limit > 0:
                stale = age > yolo_fill_limit
                anchor_x[stale] = np.nan
                anchor_y[stale] = np.nan
                misses[stale] = 0
                age[stale] = 0
            continue

        # Build the (candidates × boxes) distance matrix and greedily
        # assign closest pairs within snap_max.  Greedy per-frame
        # matching avoids first-tag-grabs-the-box bias.
        tag_pos = np.column_stack([anchor_x[candidate_idx], anchor_y[candidate_idx]])
        diff = tag_pos[:, None, :] - boxes[None, :, :]
        dists = np.sqrt((diff * diff).sum(axis=-1))
        dists_masked = np.where(dists > snap_max, np.inf, dists)
        if per_tag_forbidden:
            for row_j, tag_j in enumerate(candidate_idx):
                forbidden_j = per_tag_forbidden.get(int(tag_j))
                if forbidden_j:
                    for col_k, bid in enumerate(box_ids):
                        if int(bid) in forbidden_j:
                            dists_masked[row_j, col_k] = np.inf

        # Mask boxes already claimed (by a prior pass at this frame, or
        # by a pre-existing YOLO_INFERRED cell that was registered before
        # this call).  Applies to all candidate tags uniformly.
        frame_label_i = int(frames[i])
        if claimed_per_frame is not None:
            claimed_here = claimed_per_frame.get(frame_label_i)
            if claimed_here:
                for col_k in range(len(box_ids)):
                    if int(box_ids[col_k]) in claimed_here:
                        dists_masked[:, col_k] = np.inf

        snapped_any = np.zeros(len(candidate_idx), dtype=bool)
        while np.isfinite(dists_masked).any():
            flat = int(np.argmin(dists_masked))
            row, col = divmod(flat, dists_masked.shape[1])
            if not np.isfinite(dists_masked[row, col]):
                break
            tag_idx = int(candidate_idx[row])
            bx, by_ = float(boxes[col, 0]), float(boxes[col, 1])
            box_id = int(box_ids[col])
            x_arr[i, tag_idx] = bx
            y_arr[i, tag_idx] = by_
            t_arr[i, tag_idx] = ASS_TYPE_YOLO_INFERRED
            anchor_x[tag_idx] = bx
            anchor_y[tag_idx] = by_
            misses[tag_idx] = 0
            age[tag_idx] = 0
            snapped_any[row] = True
            filled += 1
            claims.append(
                {
                    "frame_idx": int(i),
                    "frame": frame_label_i,
                    "box_id": box_id,
                    "tag_idx": tag_idx,
                    "pass": pass_label,
                    "distance": float(dists[row, col]),
                }
            )
            if claimed_per_frame is not None:
                claimed_per_frame.setdefault(frame_label_i, set()).add(box_id)
            dists_masked[row, :] = np.inf
            dists_masked[:, col] = np.inf

        # Unmatched candidates miss.
        unmatched = candidate_idx[~snapped_any]
        if len(unmatched):
            misses[unmatched] += 1
            age[unmatched] += 1
            broken = misses[unmatched] >= max_consecutive_misses
            if broken.any():
                idx = unmatched[broken]
                anchor_x[idx] = np.nan
                anchor_y[idx] = np.nan
                misses[idx] = 0
                age[idx] = 0
            if yolo_fill_limit > 0:
                stale_mask = age[unmatched] > yolo_fill_limit
                if stale_mask.any():
                    idx = unmatched[stale_mask]
                    anchor_x[idx] = np.nan
                    anchor_y[idx] = np.nan
                    misses[idx] = 0
                    age[idx] = 0

    return filled, claims


def _fill_via_yolo(
    frame_data: pd.DataFrame,
    id_list: np.ndarray[Any, np.dtype[Any]],
    undecoded_df: pd.DataFrame,
    snap_threshold: float,
    snap_multiplier: float = DEFAULT_SNAP_MULTIPLIER,
    yolo_fill_limit: int = DEFAULT_YOLO_FILL_LIMIT,
    max_consecutive_misses: int = DEFAULT_MAX_CONSECUTIVE_MISSES,
) -> tuple[int, list[dict[str, Any]]]:
    """Bridge per-tag gaps using undecoded YOLO box centers.

    Forward-then-backward chain matching with a constant snap distance.

    For each frame in order, every tag with an active anchor (last
    known position from ORIGINAL or earlier YOLO_INFERRED fill) is
    paired greedily with the closest undecoded YOLO box on that frame.
    A pair only counts as a match when the distance is within
    ``snap_threshold * snap_multiplier`` — constant, not age-scaled, so
    long stretches stay tight against nearby ants.

    A tag's chain breaks (anchor cleared) after
    ``max_consecutive_misses`` consecutive frames where no box passes
    the snap test.  ``yolo_fill_limit > 0`` also clears the anchor
    after that many frames without a refresh; ``0`` disables that cap.

    After the forward pass, a backward pass repeats the same matching
    in reverse, anchoring on ORIGINAL detections and on forward fills.
    The backward pass recovers leading-edge gaps (tag's first ORIGINAL
    is mid-video, prior YOLO boxes get filled now) and frames the
    forward chain abandoned too early.

    Mutates *frame_data* in place.  Returns ``(filled, claims)`` where
    *claims* is the merged list of per-snap records from both passes
    (see :func:`_chain_pass` for the record schema).  ``box_id`` in the
    records is the integer row position in *undecoded_df*; the loader
    builds an explicit array so the same id is stable across calls.
    """
    if undecoded_df is None or len(id_list) == 0:
        return 0, []
    if COL_CENTER_X not in undecoded_df.columns:
        return 0, []
    if len(undecoded_df) == 0:
        return 0, []

    n_frames = len(frame_data.index)
    n_tags = len(id_list)

    x_arr = np.full((n_frames, n_tags), np.nan, dtype=float)
    y_arr = np.full((n_frames, n_tags), np.nan, dtype=float)
    t_arr = np.full((n_frames, n_tags), ASS_TYPE_NONE, dtype=np.int64)
    for j, tag_id in enumerate(id_list):
        x_arr[:, j] = frame_data[(tag_id, COL_CENTER_X)].to_numpy()
        y_arr[:, j] = frame_data[(tag_id, COL_CENTER_Y)].to_numpy()
        t_arr[:, j] = frame_data[(tag_id, COL_ASS_TYPE)].to_numpy()

    # by_frame[frame_label] = (positions (N, 2), box_ids (N,)).  box_id
    # is the integer position of the row in undecoded_df — stable
    # across calls so the prune step can use it as a per-tag key.
    pos_arr = np.column_stack(
        [
            undecoded_df[COL_CENTER_X].to_numpy(),
            undecoded_df[COL_CENTER_Y].to_numpy(),
        ]
    )
    frame_index = undecoded_df.index.to_numpy()
    pos_by_frame: dict[Any, list[int]] = {}
    for pos, fval in enumerate(frame_index):
        pos_by_frame.setdefault(fval, []).append(pos)
    by_frame: dict[Any, tuple[np.ndarray, np.ndarray]] = {}
    for fval, positions_list in pos_by_frame.items():
        ids = np.asarray(positions_list, dtype=np.int64)
        by_frame[fval] = (pos_arr[ids], ids)

    frames = frame_data.index.to_numpy()
    snap_max = snap_threshold * snap_multiplier

    # Deliberately do NOT share a claimed_per_frame between forward and
    # backward here: cross-pass collisions are the *signal*
    # ``_prune_ambiguous_tracklets`` (step 6c) uses to detect drift
    # (e.g. tag A walking over tag B and the chains swapping IDs).
    # Removing them here breaks the detection.  The re-chain (step 6d)
    # is where we DO share the claim set, so it doesn't re-introduce
    # duplicates after the prune.
    filled_fwd, claims_fwd = _chain_pass(
        x_arr,
        y_arr,
        t_arr,
        frames,
        by_frame,
        snap_max=snap_max,
        max_consecutive_misses=max_consecutive_misses,
        yolo_fill_limit=yolo_fill_limit,
        forward=True,
        pass_label="forward",
    )
    filled_bwd, claims_bwd = _chain_pass(
        x_arr,
        y_arr,
        t_arr,
        frames,
        by_frame,
        snap_max=snap_max,
        max_consecutive_misses=max_consecutive_misses,
        yolo_fill_limit=yolo_fill_limit,
        forward=False,
        pass_label="backward",
    )
    filled = filled_fwd + filled_bwd
    claims = claims_fwd + claims_bwd

    if filled == 0:
        return 0, []

    for j, tag_id in enumerate(id_list):
        if not (t_arr[:, j] == ASS_TYPE_YOLO_INFERRED).any():
            continue
        frame_data[(tag_id, COL_CENTER_X)] = x_arr[:, j]
        frame_data[(tag_id, COL_CENTER_Y)] = y_arr[:, j]
        frame_data[(tag_id, COL_ASS_TYPE)] = t_arr[:, j]

    return filled, claims


# ---------------------------------------------------------------------------
# Collision resolution helpers for the YOLO-fill pipeline
# ---------------------------------------------------------------------------


def _extract_tag_arrays(
    frame_data: pd.DataFrame,
    id_list: np.ndarray[Any, np.dtype[Any]],
) -> tuple[
    np.ndarray[Any, np.dtype[np.float64]],
    np.ndarray[Any, np.dtype[np.float64]],
    np.ndarray[Any, np.dtype[np.int64]],
]:
    """Pull per-tag center_x / center_y / ass_type out of *frame_data* into
    contiguous numpy arrays of shape ``(n_frames, n_tags)``.  Used by the
    re-chain pass so it can iterate without touching pandas."""
    n_frames = len(frame_data.index)
    n_tags = len(id_list)
    x_arr = np.full((n_frames, n_tags), np.nan, dtype=float)
    y_arr = np.full((n_frames, n_tags), np.nan, dtype=float)
    t_arr = np.full((n_frames, n_tags), ASS_TYPE_NONE, dtype=np.int64)
    for j, tag_id in enumerate(id_list):
        x_arr[:, j] = frame_data[(tag_id, COL_CENTER_X)].to_numpy()
        y_arr[:, j] = frame_data[(tag_id, COL_CENTER_Y)].to_numpy()
        t_arr[:, j] = frame_data[(tag_id, COL_ASS_TYPE)].to_numpy()
    return x_arr, y_arr, t_arr


def _writeback_tag_arrays(
    frame_data: pd.DataFrame,
    id_list: np.ndarray[Any, np.dtype[Any]],
    x_arr: np.ndarray[Any, np.dtype[np.float64]],
    y_arr: np.ndarray[Any, np.dtype[np.float64]],
    t_arr: np.ndarray[Any, np.dtype[np.int64]],
) -> None:
    """Bulk-write per-tag x/y/ass_type arrays back into *frame_data*."""
    for j, tag_id in enumerate(id_list):
        frame_data[(tag_id, COL_CENTER_X)] = x_arr[:, j]
        frame_data[(tag_id, COL_CENTER_Y)] = y_arr[:, j]
        frame_data[(tag_id, COL_ASS_TYPE)] = t_arr[:, j]


def _build_box_lookup(
    undecoded_df: pd.DataFrame,
) -> dict[Any, tuple[np.ndarray, np.ndarray]]:
    """``{frame_label: (positions (N, 2), box_ids (N,))}`` for fast lookup.

    ``box_id`` is the row's integer position in *undecoded_df* — stable
    across calls so the prune step can use it as a per-tag key.
    """
    pos_arr = np.column_stack(
        [
            undecoded_df[COL_CENTER_X].to_numpy(),
            undecoded_df[COL_CENTER_Y].to_numpy(),
        ]
    )
    frame_index = undecoded_df.index.to_numpy()
    pos_by_frame: dict[Any, list[int]] = {}
    for pos, fval in enumerate(frame_index):
        pos_by_frame.setdefault(fval, []).append(pos)
    out: dict[Any, tuple[np.ndarray, np.ndarray]] = {}
    for fval, positions_list in pos_by_frame.items():
        ids = np.asarray(positions_list, dtype=np.int64)
        out[fval] = (pos_arr[ids], ids)
    return out


def _run_chain_pair(
    x_arr: np.ndarray[Any, np.dtype[np.float64]],
    y_arr: np.ndarray[Any, np.dtype[np.float64]],
    t_arr: np.ndarray[Any, np.dtype[np.int64]],
    frames: np.ndarray[Any, np.dtype[Any]],
    by_frame: dict[Any, tuple[np.ndarray, np.ndarray]],
    snap_max: float,
    max_consecutive_misses: int,
    yolo_fill_limit: int,
    per_tag_forbidden: dict[int, set[int]] | None = None,
    candidate_tag_indices: set[int] | None = None,
    claimed_per_frame: dict[int, set[int]] | None = None,
) -> tuple[int, list[dict[str, Any]]]:
    """Forward then backward :func:`_chain_pass`, on pre-extracted arrays.

    ``claimed_per_frame`` is shared (mutated in place) by the two passes
    so backward can't grab a box forward already used.  If None, a fresh
    empty dict is created and shared between the two calls; pass a
    pre-populated dict from the caller to also forbid boxes used by
    existing YOLO_INFERRED cells (so the re-chain doesn't snap onto a
    cell another tag already occupies).
    """
    if claimed_per_frame is None:
        claimed_per_frame = {}
    f_filled, f_claims = _chain_pass(
        x_arr,
        y_arr,
        t_arr,
        frames,
        by_frame,
        snap_max=snap_max,
        max_consecutive_misses=max_consecutive_misses,
        yolo_fill_limit=yolo_fill_limit,
        forward=True,
        per_tag_forbidden=per_tag_forbidden,
        candidate_tag_indices=candidate_tag_indices,
        claimed_per_frame=claimed_per_frame,
        pass_label="forward",
    )
    b_filled, b_claims = _chain_pass(
        x_arr,
        y_arr,
        t_arr,
        frames,
        by_frame,
        snap_max=snap_max,
        max_consecutive_misses=max_consecutive_misses,
        yolo_fill_limit=yolo_fill_limit,
        forward=False,
        per_tag_forbidden=per_tag_forbidden,
        candidate_tag_indices=candidate_tag_indices,
        claimed_per_frame=claimed_per_frame,
        pass_label="backward",
    )
    return f_filled + b_filled, f_claims + b_claims


def _prune_ambiguous_tracklets(
    frame_data: pd.DataFrame,
    id_list: np.ndarray[Any, np.dtype[Any]],
    claims: list[dict[str, Any]],
) -> tuple[dict[int, set[int]], int, set[int]]:
    """Prune tracklets bounded by ambiguity events.

    An *ambiguity* is a ``(frame, box_id)`` claimed by two or more different
    tags in the merged forward+backward claims list — almost always a
    cross-pass collision where the forward chain of one tag and the backward
    chain of another both snapped to the same undecoded YOLO box, producing
    identical centroids on that frame.  For each tag involved, every
    non-ORIGINAL cell between the surrounding ORIGINALs is cleared and the
    ambiguous ``box_id`` is added to that tag's forbidden set so the
    subsequent re-chain pass doesn't re-create the same fill.

    Bounding by ORIGINALs (instead of a threshold around the ambiguity
    frame) lets the re-chain rebuild the whole tracklet from scratch with
    the bad box excluded — the *good* half of the tracklet re-snaps to its
    correct boxes, the *bad* half is forbidden the only entry point onto
    the wrong path.

    Returns
    -------
    per_tag_forbidden : dict[int, set[int]]
        tag_idx → set of forbidden box_ids.
    pruned_count : int
        Number of non-ORIGINAL cells cleared.
    affected_tag_indices : set[int]
        Tags whose tracklets were pruned (used by ``rechain_affected_only``).
    """
    # Group claims by (frame, box_id) → set of tag_idxs.
    by_box: dict[tuple[int, int], set[int]] = {}
    for c in claims:
        key = (int(c["frame"]), int(c["box_id"]))
        by_box.setdefault(key, set()).add(int(c["tag_idx"]))

    # Per-tag list of (frame, box_id) ambiguity events.
    per_tag_events: dict[int, list[tuple[int, int]]] = {}
    for (frame, box_id), tag_idxs in by_box.items():
        if len(tag_idxs) < 2:
            continue
        for tag_idx in tag_idxs:
            per_tag_events.setdefault(tag_idx, []).append((frame, box_id))

    if not per_tag_events:
        return {}, 0, set()

    frames_arr = frame_data.index.to_numpy()
    frame_to_pos = {int(f): i for i, f in enumerate(frames_arr)}

    per_tag_forbidden: dict[int, set[int]] = {}
    pruned_count = 0
    affected: set[int] = set()

    for tag_idx, events in per_tag_events.items():
        tag_id = id_list[tag_idx]
        ass = frame_data[(tag_id, COL_ASS_TYPE)].to_numpy()
        is_orig = ass == ASS_TYPE_ORIGINAL

        clear_positions: set[int] = set()
        forbidden_boxes: set[int] = set()

        for frame, box_id in events:
            pos = frame_to_pos.get(int(frame))
            if pos is None:
                continue
            forbidden_boxes.add(int(box_id))

            # Previous ORIGINAL position (or -1 if none).
            orig_before = np.where(is_orig[:pos])[0]
            r_prev = int(orig_before[-1]) if len(orig_before) else -1

            # Next ORIGINAL position (or len(ass) if none).
            orig_after = np.where(is_orig[pos + 1 :])[0]
            r_next = int(orig_after[0]) + pos + 1 if len(orig_after) else len(ass)

            # Clear all non-ORIGINAL positions in (r_prev, r_next).
            for k in range(r_prev + 1, r_next):
                if not is_orig[k]:
                    clear_positions.add(k)

        if not clear_positions:
            continue

        labels = [int(frames_arr[k]) for k in clear_positions]
        frame_data.loc[labels, (tag_id, COL_CENTER_X)] = np.nan
        frame_data.loc[labels, (tag_id, COL_CENTER_Y)] = np.nan
        if (tag_id, COL_DISTANCE) in frame_data.columns:
            frame_data.loc[labels, (tag_id, COL_DISTANCE)] = np.nan
        frame_data.loc[labels, (tag_id, COL_ASS_TYPE)] = ASS_TYPE_NONE

        pruned_count += len(clear_positions)
        per_tag_forbidden[tag_idx] = forbidden_boxes
        affected.add(tag_idx)

    return per_tag_forbidden, pruned_count, affected


def _maybe_save_snapshot(
    frame_data: pd.DataFrame,
    snapshot_dir: str | None,
    label: str,
) -> None:
    """Pickle ``frame_data`` to ``<snapshot_dir>/<label>.pkl`` when set.

    Used for end-to-end debugging — load the snapshot in a notebook to
    inspect the per-step state of cleaning without re-running the whole
    pipeline.  ``pd.to_pickle`` serializes the current state, so the
    snapshot is not affected by subsequent in-place mutations.
    """
    if snapshot_dir is None:
        return
    import os

    os.makedirs(snapshot_dir, exist_ok=True)
    frame_data.to_pickle(os.path.join(snapshot_dir, f"{label}.pkl"))


def _recover_long_gaps(
    frame_data: pd.DataFrame,
    id_list: np.ndarray[Any, np.dtype[Any]],
    undecoded_df: pd.DataFrame,
    snap_threshold: float,
    snap_multiplier: float,
    min_gap_frames: int,
) -> int:
    """Experimental: fill long NaN gaps from YOLO boxes near the anchors.

    For each tag, find contiguous NaN runs longer than *min_gap_frames*
    that are bounded by known frames on both sides.  Within each gap,
    run two **chained, velocity-aware** walks that meet in the middle:

    * **Forward** from the leading anchor (last known position before
      the gap).  At each gap frame, look at every undecoded YOLO box
      within ``snap_threshold * snap_multiplier`` of the *current*
      forward anchor — the cutoff — and among those, pick the one
      closest to the *predicted* position ``anchor + velocity``.  The
      anchor advances to that box, the per-frame velocity updates.
    * **Backward** from the trailing anchor, same logic in reverse.
      Backward skips frames the forward pass already filled (and
      re-anchors onto them so the two halves meet smoothly).

    Chained anchors let the recovery follow a moving ant across the
    gap; the velocity-prediction tiebreak picks the *trajectory-
    consistent* box when multiple candidates are within snap_max — for
    instance when a spurious YOLO detection appears near the real ant
    on a single frame.

    Tracks **box IDs** (not positions) to avoid re-claiming a YOLO box
    already used by another tag in steps 6a/6d/6f.  Boxes claimed by
    the forward walk are added to ``claimed`` so the backward walk
    can't grab the same box for the same tag.

    Returns the number of cells filled.
    """
    if undecoded_df is None or len(undecoded_df) == 0:
        return 0
    if COL_CENTER_X not in undecoded_df.columns:
        return 0

    snap_max = snap_threshold * snap_multiplier
    by_frame = _build_box_lookup(undecoded_df)
    frames_arr = frame_data.index.to_numpy()
    n_frames = len(frames_arr)

    # Per-frame set of already-claimed box_ids.
    claimed: dict[int, set[int]] = {}
    for tag_id in id_list:
        ass = frame_data[(tag_id, COL_ASS_TYPE)].to_numpy()
        x = frame_data[(tag_id, COL_CENTER_X)].to_numpy()
        y = frame_data[(tag_id, COL_CENTER_Y)].to_numpy()
        for i in range(n_frames):
            if ass[i] != ASS_TYPE_YOLO_INFERRED:
                continue
            f = int(frames_arr[i])
            entry = by_frame.get(f)
            if entry is None:
                continue
            boxes, box_ids = entry
            match = np.where((boxes[:, 0] == x[i]) & (boxes[:, 1] == y[i]))[0]
            for m in match:
                claimed.setdefault(f, set()).add(int(box_ids[m]))

    filled_count = 0
    for tag_id in id_list:
        x = frame_data[(tag_id, COL_CENTER_X)].to_numpy()
        y = frame_data[(tag_id, COL_CENTER_Y)].to_numpy()
        known = ~np.isnan(x)

        i = 0
        while i < n_frames:
            if known[i]:
                i += 1
                continue
            gs = i
            while i < n_frames and not known[i]:
                i += 1
            ge = i - 1
            # Need known frames on both sides; leading / trailing open
            # gaps have nothing to anchor against.
            if gs == 0 or ge == n_frames - 1:
                continue
            if (ge - gs + 1) <= min_gap_frames:
                continue

            # Track which gap frames have been filled so the backward
            # walk doesn't overwrite the forward walk's results.
            filled_in_gap: set[int] = set()

            def _snap_at(
                k: int,
                anchor_x: float,
                anchor_y: float,
                pred_x: float,
                pred_y: float,
            ) -> tuple[float, float] | None:
                """Try to snap tag at position k.  Cutoff: distance from
                anchor must be ≤ snap_max.  Selection: among candidates
                that pass the cutoff and aren't claimed, pick the one
                closest to (pred_x, pred_y).  Returns the new
                (anchor_x, anchor_y) on success, None on miss."""
                f = int(frames_arr[k])
                entry = by_frame.get(f)
                if entry is None:
                    return None
                boxes, box_ids = entry
                if len(boxes) == 0:
                    return None
                d_anchor = np.sqrt(
                    (boxes[:, 0] - anchor_x) ** 2 + (boxes[:, 1] - anchor_y) ** 2
                )
                valid = d_anchor <= snap_max
                claimed_ids = claimed.get(f, set())
                for b in range(len(boxes)):
                    if int(box_ids[b]) in claimed_ids:
                        valid[b] = False
                n_valid = int(valid.sum())
                if n_valid == 0:
                    return None
                if n_valid == 1:
                    # Unambiguous — skip the velocity-prediction work.
                    b_min = int(np.argmax(valid))
                else:
                    # 2+ candidates: tiebreak by closeness to predicted
                    # position to follow the trajectory rather than just
                    # grab the nearest box (which may be a spurious
                    # detection adjacent to the real one).
                    d_pred = np.sqrt(
                        (boxes[:, 0] - pred_x) ** 2 + (boxes[:, 1] - pred_y) ** 2
                    )
                    d_pred_masked = np.where(valid, d_pred, np.inf)
                    b_min = int(np.argmin(d_pred_masked))
                bx_v, by_v = float(boxes[b_min, 0]), float(boxes[b_min, 1])
                frame_data.loc[f, (tag_id, COL_CENTER_X)] = bx_v
                frame_data.loc[f, (tag_id, COL_CENTER_Y)] = by_v
                frame_data.loc[f, (tag_id, COL_ASS_TYPE)] = ASS_TYPE_YOLO_INFERRED
                claimed.setdefault(f, set()).add(int(box_ids[b_min]))
                filled_in_gap.add(k)
                return bx_v, by_v

            # Forward walk.  Seed velocity from the two frames preceding
            # the gap; falls back to zero (predict = anchor) when those
            # aren't available.
            anchor_x = float(x[gs - 1])
            anchor_y = float(y[gs - 1])
            if gs >= 2 and not np.isnan(x[gs - 2]):
                vx = anchor_x - float(x[gs - 2])
                vy = anchor_y - float(y[gs - 2])
            else:
                vx, vy = 0.0, 0.0
            for k in range(gs, ge + 1):
                result = _snap_at(k, anchor_x, anchor_y, anchor_x + vx, anchor_y + vy)
                if result is not None:
                    new_x, new_y = result
                    vx = new_x - anchor_x
                    vy = new_y - anchor_y
                    anchor_x, anchor_y = new_x, new_y
                    filled_count += 1

            # Backward walk.  Velocity in this direction points
            # "backward in time" so we subtract the *next* frame's
            # position from the current anchor.
            anchor_x = float(x[ge + 1])
            anchor_y = float(y[ge + 1])
            if ge + 2 < n_frames and not np.isnan(x[ge + 2]):
                vx = anchor_x - float(x[ge + 2])
                vy = anchor_y - float(y[ge + 2])
            else:
                vx, vy = 0.0, 0.0
            for k in range(ge, gs - 1, -1):
                if k in filled_in_gap:
                    f = int(frames_arr[k])
                    new_x = float(frame_data.at[f, (tag_id, COL_CENTER_X)])
                    new_y = float(frame_data.at[f, (tag_id, COL_CENTER_Y)])
                    vx = anchor_x - new_x
                    vy = anchor_y - new_y
                    anchor_x, anchor_y = new_x, new_y
                    continue
                result = _snap_at(k, anchor_x, anchor_y, anchor_x + vx, anchor_y + vy)
                if result is not None:
                    new_x, new_y = result
                    vx = anchor_x - new_x
                    vy = anchor_y - new_y
                    anchor_x, anchor_y = new_x, new_y
                    filled_count += 1

    return filled_count


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


def compute_pixel_scale(
    frame_data: pd.DataFrame,
    tag_size_mm: float = DEFAULT_TAG_SIZE_MM,
    sample_size: int = DEFAULT_SCALE_SAMPLE_SIZE,
) -> tuple[float, float, int]:
    """Estimate the pixel-to-millimetre scale from decoded tag corners.

    AprilTags have a known physical side length (``tag_size_mm``).  By
    measuring the side length of decoded tags in pixels we can recover
    the imaging scale of the recording.  The four corners of each
    decoded tag are stored as ``(lb, rb, rt, lt)`` 4x2 arrays in the
    ``COL_CORNERS`` column; we measure all four sides and take the
    median over a stratified sample.

    The sample is stratified across both axes so the estimate is robust
    to (a) per-ID bias (a single tag with consistently misdecoded
    corners) and (b) temporal drift (focus or camera height changing
    mid-recording).  For each tag we pick at most
    ``ceil(sample_size / n_tags)`` rows with non-NaN corners, evenly
    spaced through that tag's valid frames via ``np.linspace`` integer
    indices — no RNG needed and the time coverage is uniform.

    Parameters
    ----------
    frame_data : pd.DataFrame
        Raw or partially-cleaned detection DataFrame with the
        ``COL_CORNERS`` sub-column for each tag.
    tag_size_mm : float
        Physical side length of one AprilTag border, in millimetres.
    sample_size : int
        Approximate total number of corner rows to sample across all
        tags.  ``0`` or negative means use every available row.

    Returns
    -------
    tuple[float, float, int]
        ``(median_side_px, mm_per_px, n_samples)``.  Returns
        ``(nan, nan, 0)`` if no usable corners are present.
    """
    if COL_CORNERS not in frame_data.columns.get_level_values(1):
        return float("nan"), float("nan"), 0

    id_list = np.unique(
        frame_data.columns[
            frame_data.columns.get_level_values(1) == COL_CORNERS
        ].get_level_values(0)
    )
    if len(id_list) == 0:
        return float("nan"), float("nan"), 0

    per_tag = (
        max(1, int(np.ceil(sample_size / len(id_list)))) if sample_size > 0 else -1
    )

    sides: list[float] = []
    n_samples = 0
    for tag_id in id_list:
        col = frame_data[(tag_id, COL_CORNERS)]
        valid_idx = np.flatnonzero(col.notna().to_numpy())
        if len(valid_idx) == 0:
            continue
        if per_tag > 0 and len(valid_idx) > per_tag:
            pick = np.linspace(0, len(valid_idx) - 1, per_tag).astype(int)
            valid_idx = valid_idx[pick]
        for i in valid_idx:
            corners = col.iloc[int(i)]
            if corners is None:
                continue
            arr = np.asarray(corners, dtype=float)
            if arr.shape != (4, 2) or not np.all(np.isfinite(arr)):
                continue
            # 4 side lengths: lb→rb, rb→rt, rt→lt, lt→lb
            diffs = np.diff(arr[[0, 1, 2, 3, 0]], axis=0)
            sides.extend(np.sqrt((diffs * diffs).sum(axis=1)).tolist())
            n_samples += 1

    if not sides:
        return float("nan"), float("nan"), 0

    median_side_px = float(np.median(sides))
    mm_per_px = tag_size_mm / median_side_px if median_side_px > 0 else float("nan")
    return median_side_px, mm_per_px, n_samples


def clean_tracking_data(
    frame_data: pd.DataFrame,
    min_detections: int = MIN_DETECTIONS_PER_ID,
    interpolation_limit: int = DEFAULT_INTERPOLATION_LIMIT,
    max_jump_distance: float = DEFAULT_MAX_JUMP_DISTANCE,
    undecoded_df: pd.DataFrame | None = None,
    yolo_fill_limit: int = DEFAULT_YOLO_FILL_LIMIT,
    snap_percentile: float = DEFAULT_GOOD_TRACK_PERCENTILE,
    snap_multiplier: float = DEFAULT_SNAP_MULTIPLIER,
    max_consecutive_misses: int = DEFAULT_MAX_CONSECUTIVE_MISSES,
    rechain_affected_only: bool = DEFAULT_RECHAIN_AFFECTED_ONLY,
    recover_long_gaps: bool = DEFAULT_RECOVER_LONG_GAPS,
    min_gap_recovery_frames: int = DEFAULT_MIN_GAP_RECOVERY_FRAMES,
    final_jump_pass: bool = DEFAULT_FINAL_JUMP_PASS,
    tag_size_mm: float = DEFAULT_TAG_SIZE_MM,
    debug_snapshot_dir: str | None = None,
) -> tuple[pd.DataFrame, np.ndarray[Any, np.dtype[Any]], CleaningMetrics]:
    """Clean, interpolate, and validate raw AprilTag tracking data.

    Processing steps
    ----------------
    1. Remove tag IDs with fewer than *min_detections* valid frames.
    2. Stamp ``ass_type`` provenance codes (ORIGINAL / NONE).
    3. Fill short gaps via linear interpolation.
    4. Compute frame-to-frame distances.
    5. Delete unrealistic jumps (distance > *max_jump_distance*).
    6. *(Optional, when undecoded_df provided)*:
       a. YOLO-fill — forward+backward chain matching to undecoded boxes.
       c. Detect *ambiguity* events (same ``(frame, box_id)`` claimed by
          two or more different tags — typically forward+backward
          cross-pass collisions where both chains snapped to the same
          shared YOLO box).  For each tag involved, clear every
          non-ORIGINAL cell between the surrounding ORIGINALs and add
          the ambiguous box to that tag's forbidden set.
       d. Re-chain on the pruned state with per-tag forbidden sets;
          *rechain_affected_only* restricts candidates to tags that
          had cells pruned in step c.
       e. Recompute distances.
       f. Final jump deletion (safety net for residual mistakes).
    7. *(Optional, experimental — gated by* ``recover_long_gaps`` *)*:
       For each tag with a NaN gap longer than *min_gap_recovery_frames*,
       snap each gap frame to the closest unclaimed undecoded YOLO box
       within ``snap_threshold * snap_multiplier`` of the gap's leading
       OR trailing anchor (fixed anchors — no drift).  Targets ants
       whose AprilTag was temporarily un-decodable but YOLO still saw
       the ant.  Operates on raw NaN gaps before interpolation.
    8. *(Optional, experimental — gated by* ``final_jump_pass`` *)*:
       One more ``_delete_jump_blocks`` round on YOLO-only data before
       interpolation, for A/B comparison.
    9. Final re-interpolation of remaining short gaps and distance
       recompute.  Always runs last so all gaps created by steps 7–8
       are bridged and the output distances are fresh.

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
    undecoded_df : pd.DataFrame | None
        Undecoded YOLO box sidecar.  Columns must include ``center_x``
        and ``center_y``; index is frame number.
    yolo_fill_limit : int
        Hard cap on chain length without anchor refresh.  ``0`` disables.
    snap_percentile : float
        Percentile of original-to-original distances for snap threshold.
    snap_multiplier : float
        Multiplier on snap threshold to get the maximum snap distance.
    max_consecutive_misses : int
        Chain breaks after this many consecutive missed frames.
    rechain_affected_only : bool
        When True, restrict re-chain candidates (step 6d) to tags that
        had cells pruned in step 6c.  Default False (all tags compete).
    recover_long_gaps : bool
        Experimental.  When True, run the step-8 long-gap recovery.
    min_gap_recovery_frames : int
        Minimum gap length (frames) for step 8 to fire on a given gap.
    final_jump_pass : bool
        Experimental.  When True, run an extra ``_delete_jump_blocks``
        round at step 9 on the fully-recovered data.

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
    input_attrs = dict(frame_data.attrs)

    if frame_data.empty or frame_data.shape[1] == 0:
        raise EmptyTrackingError("Input tracking DataFrame is empty (no detections).")

    # Measure imaging scale from decoded tag corners *before* any tags
    # are filtered out — broader sample, more robust median.
    median_tag_side_px, mm_per_px, scale_sample_count = compute_pixel_scale(
        frame_data, tag_size_mm=tag_size_mm
    )

    id_list = np.unique(frame_data.columns.get_level_values(0))
    n_input_ids = len(id_list)

    # --- Step 1: remove low-count IDs ---
    for tag_id in id_list:
        non_na_x = frame_data[tag_id][COL_CENTER_X].notna().sum()
        if non_na_x < min_detections:
            columns_to_drop = [col for col in frame_data.columns if col[0] == tag_id]
            frame_data = frame_data.drop(columns=columns_to_drop)

    frame_data.columns = frame_data.columns.remove_unused_levels()
    id_list = np.unique(frame_data.columns.get_level_values(0))

    if len(id_list) == 0:
        raise EmptyTrackingError(
            f"No tag reached the min_detections={min_detections} threshold "
            f"({n_input_ids} raw ID(s) in input)."
        )

    # Baseline metrics (before any cleaning).
    total_samples = int(len(frame_data.index) * len(id_list))
    original_non_na = frame_data.xs(COL_CENTER_X, axis=1, level=1)[id_list].notna()
    original_good_count = int(original_non_na.values.sum())
    original_missing_count = total_samples - original_good_count

    # --- Step 2: stamp ass_type column ---
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

    # --- Step 3: interpolate short gaps ---
    frame_data = _interpolate_data(frame_data, limit=interpolation_limit)
    for tag_id in id_list:
        mask = (frame_data[(tag_id, COL_ASS_TYPE)] == ASS_TYPE_NONE) & frame_data[
            (tag_id, COL_CENTER_X)
        ].notna()
        frame_data.loc[mask, (tag_id, COL_ASS_TYPE)] = ASS_TYPE_INTERPOLATED

    # --- Step 4: compute distances ---
    frame_data = _compute_distances(frame_data, id_list)

    # --- Step 5: delete jump blocks (pre-YOLO) ---
    jump_deleted_count = 0
    before_5 = (
        frame_data.xs(COL_CENTER_X, axis=1, level=1)[id_list].notna().values.copy()
    )
    for tag_id in id_list:
        frame_data, _ = _delete_jump_blocks(
            frame_data, tag_id, max_distance=max_jump_distance
        )
    after_5 = frame_data.xs(COL_CENTER_X, axis=1, level=1)[id_list].notna().values
    jump_deleted_count += int((before_5 & ~after_5).sum())

    snap_threshold: float = float("inf")
    yolo_filled_count = 0
    yolo_pruned_count = 0
    yolo_rechained_count = 0

    if undecoded_df is not None:
        snap_threshold = _compute_snap_threshold(frame_data, id_list, snap_percentile)
        if np.isfinite(snap_threshold):
            snap_max = snap_threshold * snap_multiplier
            by_frame = _build_box_lookup(undecoded_df)

            # --- Step 6a: initial YOLO-fill (all tags, no forbidden) ---
            yolo_filled_count, claims = _fill_via_yolo(
                frame_data,
                id_list,
                undecoded_df,
                snap_threshold=snap_threshold,
                snap_multiplier=snap_multiplier,
                yolo_fill_limit=yolo_fill_limit,
                max_consecutive_misses=max_consecutive_misses,
            )
            _maybe_save_snapshot(frame_data, debug_snapshot_dir, "step6a_after_fill")

            # --- Step 6c: prune ambiguous tracklets ---
            # Detect (frame, box_id) entries claimed by 2+ tags in the
            # merged forward+backward claims (forward gives box B to tag
            # A; backward independently gives the same box to tag B —
            # both end up at the same exact centroid).  For each tag
            # involved, clear every non-ORIGINAL cell between the
            # surrounding ORIGINALs and forbid the ambiguous box from
            # the next chain pass.
            per_tag_forbidden, ambig_pruned, affected_tag_indices = (
                _prune_ambiguous_tracklets(frame_data, id_list, claims)
            )
            yolo_pruned_count += ambig_pruned
            _maybe_save_snapshot(frame_data, debug_snapshot_dir, "step6c_after_prune")

            # --- Step 6d: re-chain on current state (no reset) ---
            # Keeps all existing YOLO_INFERRED; only fills new gaps and
            # excludes the per-tag forbidden boxes from step 6c.
            candidate_tag_indices: set[int] | None = (
                affected_tag_indices if rechain_affected_only else None
            )
            x_arr, y_arr, t_arr = _extract_tag_arrays(frame_data, id_list)
            frames_arr = frame_data.index.to_numpy()

            # Pre-populate claimed_per_frame from existing YOLO_INFERRED
            # cells so the re-chain can't snap a tag onto a position
            # another tag already occupies.  A YOLO_INFERRED centroid is
            # bit-for-bit equal to its source box's centroid in
            # by_frame, so the position match is exact.
            claimed_per_frame: dict[int, set[int]] = {}
            yolo_mask = t_arr == ASS_TYPE_YOLO_INFERRED
            for i_pos in range(t_arr.shape[0]):
                if not yolo_mask[i_pos].any():
                    continue
                f_label = int(frames_arr[i_pos])
                entry = by_frame.get(f_label)
                if entry is None:
                    continue
                boxes_f, box_ids_f = entry
                for j_tag in np.where(yolo_mask[i_pos])[0]:
                    match = np.where(
                        (boxes_f[:, 0] == x_arr[i_pos, j_tag])
                        & (boxes_f[:, 1] == y_arr[i_pos, j_tag])
                    )[0]
                    for m in match:
                        claimed_per_frame.setdefault(f_label, set()).add(
                            int(box_ids_f[m])
                        )

            yolo_rechained_count, _ = _run_chain_pair(
                x_arr,
                y_arr,
                t_arr,
                frames_arr,
                by_frame,
                snap_max=snap_max,
                max_consecutive_misses=max_consecutive_misses,
                yolo_fill_limit=yolo_fill_limit,
                per_tag_forbidden=per_tag_forbidden if per_tag_forbidden else None,
                candidate_tag_indices=candidate_tag_indices,
                claimed_per_frame=claimed_per_frame,
            )
            _writeback_tag_arrays(frame_data, id_list, x_arr, y_arr, t_arr)
            _maybe_save_snapshot(frame_data, debug_snapshot_dir, "step6d_after_rechain")

            # --- Step 6e: recompute distances after re-chain ---
            existing_dist = [
                (tid, COL_DISTANCE)
                for tid in id_list
                if (tid, COL_DISTANCE) in frame_data.columns
            ]
            if existing_dist:
                frame_data = frame_data.drop(columns=existing_dist)
            frame_data = _compute_distances(frame_data, id_list)

            # --- Step 6f: final jump deletion ---
            before_6f = (
                frame_data.xs(COL_CENTER_X, axis=1, level=1)[id_list]
                .notna()
                .values.copy()
            )
            for tag_id in id_list:
                frame_data, _ = _delete_jump_blocks(
                    frame_data, tag_id, max_distance=max_jump_distance
                )
            after_6f = (
                frame_data.xs(COL_CENTER_X, axis=1, level=1)[id_list].notna().values
            )
            jump_deleted_count += int((before_6f & ~after_6f).sum())
            _maybe_save_snapshot(frame_data, debug_snapshot_dir, "step6f_after_jump")

    # --- Step 7 (experimental): long-gap recovery ---
    # Runs on raw NaN gaps (no prior interpolation), so the gap boundaries
    # are the genuine ORIGINAL/YOLO_INFERRED cells produced by step 6 —
    # not linear-interp guesses.
    long_gap_recovered_count = 0
    if recover_long_gaps and undecoded_df is not None and np.isfinite(snap_threshold):
        long_gap_recovered_count = _recover_long_gaps(
            frame_data,
            id_list,
            undecoded_df,
            snap_threshold=snap_threshold,
            snap_multiplier=snap_multiplier,
            min_gap_frames=min_gap_recovery_frames,
        )
        _maybe_save_snapshot(frame_data, debug_snapshot_dir, "step7_after_recovery")

    # --- Step 8 (experimental): one more jump-block pass ---
    # Always recompute distances first so the jump check sees the
    # post-recovery state.  Operates on YOLO_INFERRED / ORIGINAL only —
    # no INTERPOLATED cells exist yet at this point.
    final_jump_deleted_count = 0
    if final_jump_pass:
        existing_dist = [
            (tid, COL_DISTANCE)
            for tid in id_list
            if (tid, COL_DISTANCE) in frame_data.columns
        ]
        if existing_dist:
            frame_data = frame_data.drop(columns=existing_dist)
        frame_data = _compute_distances(frame_data, id_list)

        before = (
            frame_data.xs(COL_CENTER_X, axis=1, level=1)[id_list].notna().values.copy()
        )
        for tag_id in id_list:
            frame_data, _ = _delete_jump_blocks(
                frame_data, tag_id, max_distance=max_jump_distance
            )
        after = frame_data.xs(COL_CENTER_X, axis=1, level=1)[id_list].notna().values
        final_jump_deleted_count = int((before & ~after).sum())
        jump_deleted_count += final_jump_deleted_count
        _maybe_save_snapshot(frame_data, debug_snapshot_dir, "step8_after_final_jump")

    # --- Step 9: final interpolation and distance recompute ---
    # Always runs last so interpolation fills any gaps created by step 8
    # and ALL distances are fresh in the output.
    frame_data = _interpolate_data(frame_data, limit=interpolation_limit)
    for tag_id in id_list:
        mask = (frame_data[(tag_id, COL_ASS_TYPE)] == ASS_TYPE_NONE) & frame_data[
            (tag_id, COL_CENTER_X)
        ].notna()
        frame_data.loc[mask, (tag_id, COL_ASS_TYPE)] = ASS_TYPE_INTERPOLATED

    existing_dist = [
        (tid, COL_DISTANCE)
        for tid in id_list
        if (tid, COL_DISTANCE) in frame_data.columns
    ]
    if existing_dist:
        frame_data = frame_data.drop(columns=existing_dist)
    frame_data = _compute_distances(frame_data, id_list)

    # --- Metrics ---
    ass_types_after = frame_data.xs(COL_ASS_TYPE, axis=1, level=1)[id_list]
    filled_count = int((ass_types_after == ASS_TYPE_INTERPOLATED).values.sum())
    yolo_inferred_count = int((ass_types_after == ASS_TYPE_YOLO_INFERRED).values.sum())
    final_count = int((ass_types_after != ASS_TYPE_NONE).values.sum())

    total_gaps = original_missing_count + jump_deleted_count
    total_detections = original_good_count + jump_deleted_count

    metrics: CleaningMetrics = {
        "total_samples": total_samples,
        "total_detections": total_detections,
        "final_count": final_count,
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
        "yolo_inferred_count": yolo_inferred_count,
        "yolo_inferred_pct_of_gaps": (
            (100.0 * yolo_inferred_count / total_gaps) if total_gaps else 0.0
        ),
        "yolo_pruned_count": yolo_pruned_count,
        "yolo_rechained_count": yolo_rechained_count,
        "long_gap_recovered_count": long_gap_recovered_count,
        "final_jump_deleted_count": final_jump_deleted_count,
        "snap_threshold_px": snap_threshold,
    }

    from datetime import datetime, timezone

    from yoto import __version__

    # Restore detect-stage provenance (lost by .drop/.join) and overlay
    # clean-stage fields on top.
    frame_data.attrs.update(input_attrs)
    detect_version = input_attrs.get("yoto_version")
    if detect_version is not None:
        frame_data.attrs["yoto_detect_version"] = detect_version
    frame_data.attrs["yoto_version"] = __version__
    frame_data.attrs["yoto_stage"] = "clean"
    frame_data.attrs["yoto_cleaned_utc"] = datetime.now(timezone.utc).isoformat()
    frame_data.attrs["yoto_snap_threshold_px"] = snap_threshold
    frame_data.attrs["yoto_snap_multiplier"] = snap_multiplier
    frame_data.attrs["yoto_yolo_fill_limit"] = yolo_fill_limit
    frame_data.attrs["yoto_max_consecutive_misses"] = max_consecutive_misses
    frame_data.attrs["yoto_yolo_filled_count"] = yolo_filled_count
    frame_data.attrs["yoto_yolo_pruned_count"] = yolo_pruned_count
    frame_data.attrs["yoto_yolo_rechained_count"] = yolo_rechained_count
    frame_data.attrs["yoto_rechain_affected_only"] = rechain_affected_only
    frame_data.attrs["yoto_recover_long_gaps"] = recover_long_gaps
    frame_data.attrs["yoto_min_gap_recovery_frames"] = min_gap_recovery_frames
    frame_data.attrs["yoto_long_gap_recovered_count"] = long_gap_recovered_count
    frame_data.attrs["yoto_final_jump_pass"] = final_jump_pass
    frame_data.attrs["yoto_final_jump_deleted_count"] = final_jump_deleted_count
    frame_data.attrs["yoto_tag_size_mm"] = tag_size_mm
    frame_data.attrs["yoto_median_tag_side_px"] = median_tag_side_px
    frame_data.attrs["yoto_mm_per_px"] = mm_per_px
    frame_data.attrs["yoto_scale_sample_count"] = scale_sample_count

    return frame_data, id_list, metrics
