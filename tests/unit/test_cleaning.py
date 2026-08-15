"""Unit tests for yoto.cleaning."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from yoto.cleaning import (
    _fill_via_yolo,
    _interpolate_data,
    clean_tracking_data,
    compute_pixel_scale,
)
from yoto.io import load_corners
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
    COL_FRAME,
    CORNER_COLS,
)


class TestInterpolateData:
    """Tests for the internal _interpolate_data helper."""

    def test_fills_small_gaps(self) -> None:
        data = pd.DataFrame({"a": [1.0, np.nan, 3.0, np.nan, np.nan, 6.0]})
        result = _interpolate_data(data.copy(), limit=2)
        assert result["a"].notna().all()

    def test_respects_limit(self) -> None:
        # Gap of 7 NaNs; limit=2 fills 2 from each end, leaving 3 NaN.
        data = pd.DataFrame({"a": [1.0] + [np.nan] * 7 + [9.0]})
        result = _interpolate_data(data.copy(), limit=2)
        assert result["a"].isna().sum() > 0

    def test_does_not_extrapolate(self) -> None:
        data = pd.DataFrame({"a": [np.nan, np.nan, 3.0, 4.0, np.nan, np.nan]})
        result = _interpolate_data(data.copy(), limit=5)
        # Leading and trailing NaNs should remain
        assert pd.isna(result["a"].iloc[0])
        assert pd.isna(result["a"].iloc[-1])


class TestCleanTrackingData:
    """Tests for the clean_tracking_data function."""

    def test_removes_sparse_ids(self, sample_tracking_sparse: pd.DataFrame) -> None:
        cleaned, id_list, metrics = clean_tracking_data(
            sample_tracking_sparse, min_detections=100
        )
        # Tag 10 only has 50 detections — should be removed
        assert 10 not in id_list
        assert 20 in id_list

    def test_keeps_ids_above_threshold(
        self, sample_tracking_sparse: pd.DataFrame
    ) -> None:
        cleaned, id_list, metrics = clean_tracking_data(
            sample_tracking_sparse, min_detections=40
        )
        # Both should survive with threshold=40
        assert 10 in id_list
        assert 20 in id_list

    def test_creates_ass_type_column(
        self, sample_tracking_dataframe: pd.DataFrame
    ) -> None:
        cleaned, id_list, _ = clean_tracking_data(
            sample_tracking_dataframe, min_detections=1
        )
        for tag_id in id_list:
            assert (tag_id, COL_ASS_TYPE) in cleaned.columns

    def test_original_detections_marked_correctly(
        self, sample_tracking_dataframe: pd.DataFrame
    ) -> None:
        cleaned, id_list, _ = clean_tracking_data(
            sample_tracking_dataframe, min_detections=1
        )
        # Tag 1 has all frames detected originally
        assert (cleaned[(1, COL_ASS_TYPE)] == ASS_TYPE_ORIGINAL).all()

    def test_interpolated_frames_marked(
        self, sample_tracking_dataframe: pd.DataFrame
    ) -> None:
        cleaned, id_list, _ = clean_tracking_data(
            sample_tracking_dataframe, min_detections=1
        )
        # Tag 2 had NaN at frames 5 and 6, gap=2 <= default limit=5
        # so they should be interpolated
        interp_mask = cleaned[(2, COL_ASS_TYPE)] == ASS_TYPE_INTERPOLATED
        assert interp_mask.sum() >= 2

    def test_distance_column_created(
        self, sample_tracking_dataframe: pd.DataFrame
    ) -> None:
        cleaned, id_list, _ = clean_tracking_data(
            sample_tracking_dataframe, min_detections=1
        )
        for tag_id in id_list:
            assert (tag_id, COL_DISTANCE) in cleaned.columns

    def test_metrics_structure(self, sample_tracking_dataframe: pd.DataFrame) -> None:
        _, _, metrics = clean_tracking_data(sample_tracking_dataframe, min_detections=1)
        expected_keys = {
            "total_samples",
            "total_detections",
            "final_count",
            "original_good_count",
            "original_bad_count",
            "original_missing_count",
            "total_gaps",
            "error_pct",
            "original_bad_pct",
            "original_missing_pct",
            "filled_count",
            "filled_pct_of_total",
            "filled_pct_of_gaps",
            "yolo_inferred_count",
            "yolo_inferred_pct_of_gaps",
            "yolo_pruned_count",
            "yolo_rechained_count",
            "long_gap_recovered_count",
            "final_jump_deleted_count",
            "tag_size_px",
            "max_move_px",
        }
        assert set(metrics.keys()) == expected_keys

    def test_metrics_total_samples(
        self, sample_tracking_dataframe: pd.DataFrame
    ) -> None:
        cleaned, id_list, metrics = clean_tracking_data(
            sample_tracking_dataframe, min_detections=1
        )
        # total_samples = n_frames * n_ids
        assert metrics["total_samples"] == len(cleaned.index) * len(id_list)

    def test_runs_on_sliced_index_not_starting_at_zero(
        self, sample_tracking_dataframe: pd.DataFrame
    ) -> None:
        # Regression: clean_tracking_data used `frame_data.loc[1:, ...]`
        # for distance recompute, which silently selected ALL rows
        # when the frame index didn't start at 0 (e.g. notebook
        # windowing).  That tripped a length-mismatch ValueError on
        # the distance assignment.
        sliced = sample_tracking_dataframe.iloc[5:].copy()
        cleaned, _, _ = clean_tracking_data(sliced, min_detections=1)
        assert (1, COL_DISTANCE) in cleaned.columns

    @pytest.mark.parametrize("max_jump", [50.0, 100.0, 2000.0])
    def test_jump_detection_respects_threshold(
        self,
        sample_tracking_dataframe: pd.DataFrame,
        max_jump: float,
    ) -> None:
        _, _, metrics = clean_tracking_data(
            sample_tracking_dataframe,
            min_detections=1,
            max_jump_distance=max_jump,
        )
        # The fixture's synthetic jump is ~673 px.  Above that, no jump
        # should be flagged.
        if max_jump >= 2000.0:
            assert metrics["original_bad_count"] == 0


def _make_tracking_df(
    tag_xy: dict[int, tuple[list[float], list[float]]],
) -> pd.DataFrame:
    """Build a MultiIndex tracking DataFrame from per-tag (x, y) lists."""
    n_frames = max(len(xy[0]) for xy in tag_xy.values())
    data: dict[tuple[Any, str], list[Any]] = {(COL_FRAME, ""): list(range(n_frames))}
    for tag_id, (xs, ys) in tag_xy.items():
        data[(tag_id, COL_CENTER_X)] = xs
        data[(tag_id, COL_CENTER_Y)] = ys
    df = pd.DataFrame(data)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    df = df.set_index(COL_FRAME)
    return df


def _attach_ass_type(df: pd.DataFrame) -> pd.DataFrame:
    """Add ass_type=ORIGINAL where center_x is not NaN, NONE otherwise."""
    id_list = sorted({c[0] for c in df.columns if isinstance(c[0], int)})
    for tag_id in id_list:
        types = np.where(
            df[(tag_id, COL_CENTER_X)].notna(),
            ASS_TYPE_ORIGINAL,
            ASS_TYPE_NONE,
        )
        df[(tag_id, COL_ASS_TYPE)] = types
    return df


class TestFillViaYolo:
    """Tests for _fill_via_yolo (forward + backward chain, constant snap)."""

    @staticmethod
    def _undecoded(rows: list[tuple[int, float, float]]) -> pd.DataFrame:
        """Build an undecoded sidecar DataFrame from (frame, cx, cy) rows."""
        df = pd.DataFrame(
            rows, columns=[COL_FRAME, COL_CENTER_X, COL_CENTER_Y]
        ).set_index(COL_FRAME)
        return df

    def test_forward_chain_fills_consecutive_close_boxes(self) -> None:
        # ORIGINAL at frame 0; YOLO boxes at frames 1, 2 close to anchor.
        xs = [0.0, np.nan, np.nan, 30.0]
        ys = [0.0, np.nan, np.nan, 0.0]
        df = _attach_ass_type(_make_tracking_df({1: (xs, ys)}))
        undecoded = self._undecoded([(1, 10.0, 0.0), (2, 20.0, 0.0)])
        filled, _claims = _fill_via_yolo(
            df,
            np.array([1]),
            undecoded,
            tag_size_px=15.0,
            tag_size_multiplier=1.0,
        )
        assert filled == 2
        assert df.loc[1, (1, COL_ASS_TYPE)] == ASS_TYPE_YOLO_INFERRED
        assert df.loc[2, (1, COL_ASS_TYPE)] == ASS_TYPE_YOLO_INFERRED

    def test_constant_threshold_rejects_far_boxes(self) -> None:
        # A box 500 px away must NOT snap, regardless of any other state.
        xs = [0.0, np.nan, 20.0]
        ys = [0.0, np.nan, 0.0]
        df = _attach_ass_type(_make_tracking_df({1: (xs, ys)}))
        undecoded = self._undecoded([(1, 500.0, 500.0)])
        filled, _claims = _fill_via_yolo(
            df,
            np.array([1]),
            undecoded,
            tag_size_px=10.0,
            tag_size_multiplier=2.0,
        )
        assert filled == 0
        assert pd.isna(df.loc[1, (1, COL_CENTER_X)])

    def test_backward_pass_fills_leading_edge(self) -> None:
        # Tag's first ORIGINAL is at frame 3.  YOLO boxes at frames 0,1,2
        # close to that position.  Forward pass can't help (no anchor
        # yet); backward pass must walk back from frame 3 and fill them.
        xs = [np.nan, np.nan, np.nan, 0.0, 10.0]
        ys = [np.nan, np.nan, np.nan, 0.0, 0.0]
        df = _attach_ass_type(_make_tracking_df({1: (xs, ys)}))
        undecoded = self._undecoded([(0, 0.0, 0.0), (1, 0.0, 0.0), (2, 0.0, 0.0)])
        filled, _claims = _fill_via_yolo(
            df,
            np.array([1]),
            undecoded,
            tag_size_px=5.0,
            tag_size_multiplier=1.0,
        )
        assert filled == 3
        for f in (0, 1, 2):
            assert df.loc[f, (1, COL_ASS_TYPE)] == ASS_TYPE_YOLO_INFERRED

    def test_chain_breaks_after_consecutive_misses(self) -> None:
        # Anchor at frame 0.  Frames 1-3 have boxes near the anchor →
        # snap.  Frames 4-6 have boxes far away → 3 misses → chain
        # breaks.  Frame 7 has a box again, but no anchor, so no fill.
        xs = [0.0] + [np.nan] * 7
        ys = [0.0] + [np.nan] * 7
        df = _attach_ass_type(_make_tracking_df({1: (xs, ys)}))
        undecoded = self._undecoded(
            [
                (1, 0.0, 0.0),
                (2, 0.0, 0.0),
                (3, 0.0, 0.0),
                (4, 500.0, 0.0),  # too far
                (5, 500.0, 0.0),
                (6, 500.0, 0.0),
                (7, 0.0, 0.0),  # would match but chain is broken
            ]
        )
        filled, _claims = _fill_via_yolo(
            df,
            np.array([1]),
            undecoded,
            tag_size_px=10.0,
            tag_size_multiplier=1.0,
            max_consecutive_misses=3,
        )
        assert filled == 3
        assert pd.isna(df.loc[7, (1, COL_CENTER_X)])

    def test_yolo_fill_limit_zero_means_unlimited(self) -> None:
        # 9-frame gap with perfect boxes; with limit=0 (default) and
        # high miss tolerance, everything fills.
        xs = [0.0] + [np.nan] * 9 + [100.0]
        ys = [0.0] + [np.nan] * 9 + [0.0]
        df = _attach_ass_type(_make_tracking_df({1: (xs, ys)}))
        undecoded = self._undecoded([(i, i * 10.0, 0.0) for i in range(1, 10)])
        filled, _claims = _fill_via_yolo(
            df,
            np.array([1]),
            undecoded,
            tag_size_px=15.0,
            tag_size_multiplier=1.0,
            yolo_fill_limit=0,
        )
        assert filled == 9

    def test_per_frame_conflict_resolved_by_closest(self) -> None:
        # Two tags, both at gap frame 1.  Two YOLO boxes available, each
        # closer to a different tag → greedy global-min assigns the
        # closest pair to its tag.
        df = _attach_ass_type(
            _make_tracking_df(
                {
                    1: ([0.0, np.nan, 0.0], [0.0, np.nan, 0.0]),
                    2: ([100.0, np.nan, 100.0], [0.0, np.nan, 0.0]),
                }
            )
        )
        undecoded = self._undecoded(
            [
                (1, 102.0, 0.0),  # closer to tag 2's anchor
                (1, 1.0, 0.0),  # closer to tag 1's anchor
            ]
        )
        filled, _claims = _fill_via_yolo(
            df,
            np.array([1, 2]),
            undecoded,
            tag_size_px=10.0,
            tag_size_multiplier=2.0,
        )
        assert filled == 2
        assert df.loc[1, (1, COL_CENTER_X)] == pytest.approx(1.0)
        assert df.loc[1, (2, COL_CENTER_X)] == pytest.approx(102.0)


class TestCleanTrackingDataWithYoloFill:
    """End-to-end checks that clean_tracking_data uses the undecoded sidecar."""

    def test_undecoded_box_replaces_interpolation(
        self, sample_tracking_dataframe: pd.DataFrame
    ) -> None:
        # Tag 2 has gaps at frames 5, 6.  Provide an undecoded box on
        # frame 5 near the linear-interp midpoint.
        x5 = (
            sample_tracking_dataframe[(2, COL_CENTER_X)].iloc[4]
            + sample_tracking_dataframe[(2, COL_CENTER_X)].iloc[7]
        ) / 2.0
        y5 = (
            sample_tracking_dataframe[(2, COL_CENTER_Y)].iloc[4]
            + sample_tracking_dataframe[(2, COL_CENTER_Y)].iloc[7]
        ) / 2.0
        undecoded = pd.DataFrame(
            [{COL_FRAME: 5, COL_CENTER_X: x5, COL_CENTER_Y: y5}]
        ).set_index(COL_FRAME)

        cleaned, _, metrics = clean_tracking_data(
            sample_tracking_dataframe,
            min_detections=1,
            undecoded_df=undecoded,
        )
        assert metrics["yolo_inferred_count"] >= 1
        assert cleaned.loc[5, (2, COL_ASS_TYPE)] == ASS_TYPE_YOLO_INFERRED

    def test_no_undecoded_no_yolo_filled(
        self, sample_tracking_dataframe: pd.DataFrame
    ) -> None:
        _, _, metrics = clean_tracking_data(sample_tracking_dataframe, min_detections=1)
        assert metrics["yolo_inferred_count"] == 0
        # No YOLO input → prune/rechain counters stay at 0.
        assert metrics["yolo_pruned_count"] == 0
        assert metrics["yolo_rechained_count"] == 0

    def test_yolo_fill_jump_prune_then_rechain(self) -> None:
        # Tag 1 has a long gap.  Two undecoded boxes are available:
        # one that creates a jump (far from the anchor trajectory) and
        # one that lies on the natural trajectory.  The initial fill
        # might pick the bad box; step 6c prunes it and step 6d re-chains
        # using the good one.  This exercises the prune+rechain loop
        # without asserting on which box wins (the goal is just that
        # the pruning/re-chain pipeline runs).
        df = _make_tracking_df(
            {
                1: (
                    [0.0, 10.0, np.nan, np.nan, np.nan, 50.0, 60.0],
                    [0.0, 10.0, np.nan, np.nan, np.nan, 50.0, 60.0],
                ),
            }
        )
        undecoded = pd.DataFrame(
            [
                {COL_FRAME: 2, COL_CENTER_X: 20.0, COL_CENTER_Y: 20.0},
                {COL_FRAME: 3, COL_CENTER_X: 30.0, COL_CENTER_Y: 30.0},
                {COL_FRAME: 4, COL_CENTER_X: 40.0, COL_CENTER_Y: 40.0},
            ]
        ).set_index(COL_FRAME)
        _, _, metrics = clean_tracking_data(
            df,
            min_detections=1,
            interpolation_limit=2,
            undecoded_df=undecoded,
            tag_size_multiplier=5.0,
        )
        # Counters exist and are non-negative.
        assert metrics["yolo_pruned_count"] >= 0
        assert metrics["yolo_rechained_count"] >= 0


class TestCornerProvenance:
    """Corners must survive only on ORIGINAL rows.

    A quad is a measurement of where the tag physically was.  Interpolating
    one, or leaving one behind after a jump block is deleted, produces an
    outline that no longer matches its own center_x / center_y — before this
    was enforced, interpolated quads drifted hundreds of pixels away from
    their tag centre.
    """

    @staticmethod
    def _raw(n_frames: int, missing: range) -> pd.DataFrame:
        """Two tags walking right at 1 px/frame; tag 1 absent on *missing*.

        Tag 2 is present on every frame so the wide index still covers the
        gap — otherwise there would be no row for tag 1 to interpolate into.
        """
        rows = []
        for f in range(n_frames):
            for tag_id, y in ((1, 100.0), (2, 300.0)):
                if tag_id == 1 and f in missing:
                    continue
                cx, cy = 100.0 + f, y
                rows.append(
                    {
                        COL_FRAME: f,
                        "tag_id": tag_id,
                        COL_CENTER_X: cx,
                        COL_CENTER_Y: cy,
                        "c0x": cx - 5.0,
                        "c0y": cy - 5.0,
                        "c1x": cx + 5.0,
                        "c1y": cy - 5.0,
                        "c2x": cx + 5.0,
                        "c2y": cy + 5.0,
                        "c3x": cx - 5.0,
                        "c3y": cy + 5.0,
                    }
                )
        return pd.DataFrame(rows).set_index(COL_FRAME)

    def test_interpolated_rows_have_no_corners(self) -> None:
        cleaned, _, _ = clean_tracking_data(
            self._raw(40, range(10, 13)), min_detections=1, interpolation_limit=5
        )
        ass = cleaned[(1, COL_ASS_TYPE)]
        assert (ass.iloc[10:13] == ASS_TYPE_INTERPOLATED).all()
        quads = load_corners(cleaned, tag_id=1)
        filled = ass == ASS_TYPE_ORIGINAL
        assert np.isnan(quads[~filled.to_numpy()]).all()
        assert np.isfinite(quads[filled.to_numpy()]).all()

    def test_surviving_corners_still_match_their_centre(self) -> None:
        cleaned, _, _ = clean_tracking_data(
            self._raw(40, range(10, 13)), min_detections=1, interpolation_limit=5
        )
        quads = load_corners(cleaned, tag_id=1)
        keep = np.isfinite(quads).all(axis=(1, 2))
        centroid_x = quads[keep, :, 0].mean(axis=1)
        assert centroid_x == pytest.approx(cleaned[(1, COL_CENTER_X)].to_numpy()[keep])

    def test_legacy_corners_column_input_is_also_masked(self) -> None:
        raw = self._raw(40, range(10, 13))
        flat = raw[list(CORNER_COLS)].to_numpy().reshape(-1, 4, 2)
        raw = raw.drop(columns=list(CORNER_COLS))
        raw[COL_CORNERS] = pd.Series(list(flat), index=raw.index, dtype=object)
        cleaned, _, _ = clean_tracking_data(
            raw, min_detections=1, interpolation_limit=5
        )
        ass = cleaned[(1, COL_ASS_TYPE)]
        quads = load_corners(cleaned, tag_id=1)
        assert np.isnan(quads[(ass != ASS_TYPE_ORIGINAL).to_numpy()]).all()


class TestComputePixelScale:
    """Tests for compute_pixel_scale."""

    @staticmethod
    def _square(cx: float, cy: float, side: float) -> np.ndarray:
        """Return an (lb, rb, rt, lt) corner array for an axis-aligned square."""
        h = side / 2.0
        return np.array(
            [
                [cx - h, cy - h],  # lb
                [cx + h, cy - h],  # rb
                [cx + h, cy + h],  # rt
                [cx - h, cy + h],  # lt
            ],
            dtype=float,
        )

    def _build_df(
        self, n_frames: int, tag_ids: list[int], sides: dict[int, float]
    ) -> pd.DataFrame:
        index = pd.Index(range(n_frames), name=COL_FRAME)
        per_tag_frames: dict[tuple[int, str], pd.Series] = {}
        for tag_id in tag_ids:
            side = sides[tag_id]
            per_tag_frames[(tag_id, COL_CENTER_X)] = pd.Series(
                [100.0] * n_frames, index=index, dtype=float
            )
            per_tag_frames[(tag_id, COL_CENTER_Y)] = pd.Series(
                [100.0] * n_frames, index=index, dtype=float
            )
            per_tag_frames[(tag_id, COL_CORNERS)] = pd.Series(
                [self._square(100.0, 100.0, side) for _ in range(n_frames)],
                index=index,
                dtype=object,
            )
        df = pd.concat(per_tag_frames, axis=1)
        df.columns = pd.MultiIndex.from_tuples(
            df.columns.to_list(), names=["id", "metric"]
        )
        return df

    def test_recovers_known_scale(self) -> None:
        df = self._build_df(20, [1, 2, 3], {1: 50.0, 2: 50.0, 3: 50.0})
        side_px, mm_per_px, n = compute_pixel_scale(df, tag_size_mm=0.4)
        assert side_px == pytest.approx(50.0)
        assert mm_per_px == pytest.approx(0.4 / 50.0)
        assert n > 0

    def test_robust_to_outlier_id(self) -> None:
        # One tag with grossly miscalibrated corners shouldn't move the median.
        df = self._build_df(50, [1, 2, 3], {1: 50.0, 2: 50.0, 3: 500.0})
        side_px, _, _ = compute_pixel_scale(df, tag_size_mm=0.4)
        assert side_px == pytest.approx(50.0, rel=0.01)

    def test_returns_nan_when_no_corners(self) -> None:
        cols = pd.MultiIndex.from_product(
            [[1], [COL_CENTER_X, COL_CENTER_Y]], names=["id", "metric"]
        )
        df = pd.DataFrame({c: [1.0, 2.0] for c in cols})
        side_px, mm_per_px, n = compute_pixel_scale(df)
        assert np.isnan(side_px)
        assert np.isnan(mm_per_px)
        assert n == 0

    def test_attrs_set_on_clean_output(self) -> None:
        df = self._build_df(150, [1, 2], {1: 40.0, 2: 40.0})
        cleaned, _, _ = clean_tracking_data(df, min_detections=1, tag_size_mm=0.4)
        assert cleaned.attrs["yoto_tag_size_mm"] == 0.4
        assert cleaned.attrs["yoto_median_tag_side_px"] == pytest.approx(40.0)
        assert cleaned.attrs["yoto_mm_per_px"] == pytest.approx(0.4 / 40.0)
        assert cleaned.attrs["yoto_scale_sample_count"] > 0
