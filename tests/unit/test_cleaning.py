"""Unit tests for yoto.cleaning."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from yoto.cleaning import clean_tracking_data, _interpolate_data
from yoto.constants import (
    ASS_TYPE_INTERPOLATED,
    ASS_TYPE_ORIGINAL,
    COL_ASS_TYPE,
    COL_CENTER_X,
    COL_CENTER_Y,
    COL_DISTANCE,
)


class TestInterpolateData:
    """Tests for the internal _interpolate_data helper."""

    def test_fills_small_gaps(self) -> None:
        data = pd.DataFrame({"a": [1.0, np.nan, 3.0, np.nan, np.nan, 6.0]})
        result = _interpolate_data(data.copy(), limit=2)
        assert result["a"].notna().all()

    def test_respects_limit(self) -> None:
        data = pd.DataFrame(
            {"a": [1.0, np.nan, np.nan, np.nan, np.nan, 6.0]}
        )
        result = _interpolate_data(data.copy(), limit=2)
        # Gap of 4 > limit of 2, so middle values stay NaN
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

    def test_metrics_structure(
        self, sample_tracking_dataframe: pd.DataFrame
    ) -> None:
        _, _, metrics = clean_tracking_data(
            sample_tracking_dataframe, min_detections=1
        )
        expected_keys = {
            "total_samples",
            "total_detections",
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

    @pytest.mark.parametrize("max_jump", [50.0, 100.0, 500.0])
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
        # With a very large threshold, no jumps should be detected
        if max_jump >= 500.0:
            assert metrics["original_bad_count"] == 0
