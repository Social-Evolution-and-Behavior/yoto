"""Shared test fixtures for the YOTO test suite."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from yoto.constants import COL_CENTER_X, COL_CENTER_Y, COL_FRAME


@pytest.fixture()
def sample_gray_image() -> np.ndarray[Any, np.dtype[np.uint8]]:
    """A small synthetic grayscale image for image-processing tests."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, (200, 300), dtype=np.uint8)


@pytest.fixture()
def sample_bgr_image() -> np.ndarray[Any, np.dtype[np.uint8]]:
    """A small synthetic BGR image."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, (200, 300, 3), dtype=np.uint8)


@pytest.fixture()
def sample_bboxes() -> np.ndarray[Any, np.dtype[np.float64]]:
    """A small array of bounding boxes in xyxy format."""
    return np.array(
        [
            [10.0, 20.0, 50.0, 60.0],
            [100.0, 110.0, 160.0, 170.0],
        ],
        dtype=np.float64,
    )


@pytest.fixture()
def sample_tracking_dataframe() -> pd.DataFrame:
    """A minimal tracking DataFrame with two tag IDs over 20 frames.

    Tag 1 has a smooth trajectory; tag 2 has a jump and some missing frames.
    """
    n_frames = 20
    frames = list(range(n_frames))

    # Tag 1: smooth diagonal trajectory
    tag1_x = np.linspace(100, 200, n_frames).tolist()
    tag1_y = np.linspace(100, 200, n_frames).tolist()

    # Tag 2: trajectory with a jump at frame 10 and gaps at frames 5-6
    tag2_x = np.linspace(300, 350, n_frames).tolist()
    tag2_y = np.linspace(300, 350, n_frames).tolist()
    tag2_x[10] = 800.0  # introduce a jump
    tag2_y[10] = 800.0
    tag2_x[5] = np.nan  # introduce gaps
    tag2_y[5] = np.nan
    tag2_x[6] = np.nan
    tag2_y[6] = np.nan

    data: dict[tuple[int | str, str], list[Any]] = {
        (COL_FRAME, ""): frames,
        (1, COL_CENTER_X): tag1_x,
        (1, COL_CENTER_Y): tag1_y,
        (2, COL_CENTER_X): tag2_x,
        (2, COL_CENTER_Y): tag2_y,
    }

    df = pd.DataFrame(data)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    df = df.set_index(COL_FRAME)
    return df


@pytest.fixture()
def sample_tracking_sparse() -> pd.DataFrame:
    """A tracking DataFrame where one tag has fewer than 100 detections.

    Useful for testing the minimum-detection filter in cleaning.
    """
    n_frames = 200
    frames = list(range(n_frames))

    # Tag 10: only 50 detections
    tag10_x: list[Any] = [np.nan] * n_frames
    tag10_y: list[Any] = [np.nan] * n_frames
    for i in range(50):
        tag10_x[i * 4] = 100.0 + i
        tag10_y[i * 4] = 100.0 + i

    # Tag 20: 200 detections (all frames)
    tag20_x = np.linspace(200, 400, n_frames).tolist()
    tag20_y = np.linspace(200, 400, n_frames).tolist()

    data: dict[tuple[int | str, str], list[Any]] = {
        (COL_FRAME, ""): frames,
        (10, COL_CENTER_X): tag10_x,
        (10, COL_CENTER_Y): tag10_y,
        (20, COL_CENTER_X): tag20_x,
        (20, COL_CENTER_Y): tag20_y,
    }

    df = pd.DataFrame(data)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    df = df.set_index(COL_FRAME)
    return df


@pytest.fixture(scope="session")
def tmp_video(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Create a tiny synthetic video file for integration tests.

    Returns the path to a 10-frame, 64x64 video.
    """
    import cv2

    path = str(tmp_path_factory.mktemp("video") / "test.mp4")
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 10.0, (64, 64))

    rng = np.random.default_rng(0)
    for _ in range(10):
        frame = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path
