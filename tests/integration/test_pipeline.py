"""Integration tests for the full YOTO pipeline.

These tests exercise the cleaning and video rendering modules end-to-end
using synthetic data. Detection tests are marked slow because they
require GPU and model weights.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from yoto.cleaning import clean_tracking_data
from yoto.constants import (
    COL_ASS_TYPE,
    COL_CENTER_X,
    COL_CENTER_Y,
    COL_FRAME,
)


@pytest.mark.integration
class TestCleaningPipeline:
    """End-to-end tests for the cleaning pipeline."""

    def _make_large_tracking_df(
        self, n_frames: int = 500, n_ids: int = 10
    ) -> pd.DataFrame:
        """Build a synthetic tracking DataFrame with realistic patterns."""
        rng = np.random.default_rng(123)
        frames = list(range(n_frames))

        data: dict[tuple[int | str, str], list[Any]] = {
            (COL_FRAME, ""): frames,
        }

        for tag_id in range(1, n_ids + 1):
            x = np.cumsum(rng.normal(0, 2, n_frames)) + 500
            y = np.cumsum(rng.normal(0, 2, n_frames)) + 500

            # Introduce random gaps (~5% of frames)
            gap_mask = rng.random(n_frames) < 0.05
            x[gap_mask] = np.nan
            y[gap_mask] = np.nan

            # Introduce a jump for every other tag
            if tag_id % 2 == 0:
                lo = min(100, max(1, n_frames // 4))
                hi = max(lo + 1, n_frames)
                jump_frame = int(rng.integers(lo, hi))
                x[jump_frame] = x[jump_frame - 1] + 500
                y[jump_frame] = y[jump_frame - 1] + 500

            data[(tag_id, COL_CENTER_X)] = x.tolist()
            data[(tag_id, COL_CENTER_Y)] = y.tolist()

        df = pd.DataFrame(data)
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        df = df.set_index(COL_FRAME)
        return df

    def test_full_cleaning_cycle(self) -> None:
        df = self._make_large_tracking_df()
        cleaned, id_list, metrics = clean_tracking_data(df, min_detections=10)

        # All 10 IDs should survive (each has ~475 detections)
        assert len(id_list) == 10

        # ass_type column exists for every ID
        for tag_id in id_list:
            assert (tag_id, COL_ASS_TYPE) in cleaned.columns

        # Some interpolation should have happened
        assert metrics["filled_count"] > 0

        # Jumps should have been detected in the even IDs
        assert metrics["original_bad_count"] > 0

    def test_cleaning_preserves_frame_count(self) -> None:
        df = self._make_large_tracking_df(n_frames=200)
        cleaned, _, _ = clean_tracking_data(df, min_detections=10)
        assert len(cleaned.index) == 200

    def test_metrics_percentages_valid(self) -> None:
        df = self._make_large_tracking_df()
        _, _, metrics = clean_tracking_data(df, min_detections=10)
        assert 0.0 <= metrics["error_pct"] <= 100.0
        assert 0.0 <= metrics["filled_pct_of_gaps"] <= 100.0
        assert 0.0 <= metrics["original_missing_pct"] <= 100.0


@pytest.mark.integration
@pytest.mark.slow
class TestVideoRender:
    """Integration tests for video rendering (requires ffmpeg)."""

    def test_render_tiny_video(self, tmp_video: str, tmp_path: Any) -> None:
        """Render overlay on a tiny synthetic video."""
        from yoto.video import render_overlay_video

        # Build minimal tracking data matching the 10-frame video
        n_frames = 10
        frames = list(range(n_frames))
        data: dict[tuple[int | str, str], list[Any]] = {
            (COL_FRAME, ""): frames,
            (1, COL_CENTER_X): [32.0] * n_frames,
            (1, COL_CENTER_Y): [32.0] * n_frames,
        }
        df = pd.DataFrame(data)
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        df = df.set_index(COL_FRAME)

        # Clean (even though it's already clean)
        cleaned, id_list, _ = clean_tracking_data(df, min_detections=1)

        output = str(tmp_path / "output.mp4")
        result = render_overlay_video(
            video_path=tmp_video,
            frame_data=cleaned,
            id_list=id_list,
            output_path=output,
        )
        assert result == output
