"""Unit tests for yoto.detection internals."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from yoto.constants import COL_CENTER_X, COL_CENTER_Y, COL_CORNERS, COL_FRAME
from yoto.detection import (
    _build_apriltag_params_fast,
    _build_apriltag_params_simple,
    _crop_and_pack,
    _process_frame_cpu,
    _reproject_tags,
)


class TestBuildApriltagParams:
    """Tests for parameter builder functions."""

    def test_simple_params_has_required_keys(self) -> None:
        params = _build_apriltag_params_simple()
        required = {
            "threads",
            "decimate",
            "blur",
            "refine_edges",
            "decode_sharpening",
            "max_hamming",
            "kernel_size",
            "sigma",
            "amount",
            "contrast_factor",
        }
        assert required.issubset(set(params.keys()))

    def test_fast_params_has_contrast_method(self) -> None:
        params = _build_apriltag_params_fast()
        assert "contrast_method" in params
        assert params["contrast_method"] == "cv2"

    def test_fast_params_has_cv2_keys(self) -> None:
        params = _build_apriltag_params_fast()
        assert "cv2_alpha" in params
        assert "cv2_beta" in params


class TestCropAndPack:
    """Tests for the _crop_and_pack function."""

    def test_empty_boxes_returns_none_composite(
        self, sample_gray_image: np.ndarray[Any, np.dtype[np.uint8]]
    ) -> None:
        empty_boxes = np.empty((0, 4), dtype=np.float64)
        crops, offsets, composite, x_offsets = _crop_and_pack(
            sample_gray_image, empty_boxes, pad_pixels=10
        )
        assert len(crops) == 0
        assert composite is None

    def test_single_box_crop(
        self, sample_gray_image: np.ndarray[Any, np.dtype[np.uint8]]
    ) -> None:
        boxes = np.array([[20.0, 30.0, 80.0, 90.0]], dtype=np.float64)
        crops, offsets, composite, x_offsets = _crop_and_pack(
            sample_gray_image, boxes, pad_pixels=5
        )
        assert len(crops) == 1
        assert composite is not None
        assert composite.ndim == 2  # grayscale input -> grayscale composite

    def test_multiple_boxes_packed_horizontally(
        self,
        sample_gray_image: np.ndarray[Any, np.dtype[np.uint8]],
        sample_bboxes: np.ndarray[Any, np.dtype[np.float64]],
    ) -> None:
        crops, offsets, composite, x_offsets = _crop_and_pack(
            sample_gray_image, sample_bboxes, pad_pixels=5
        )
        assert len(crops) == 2
        assert composite is not None
        # Strip width = sum of crop widths
        total_w = sum(c.shape[1] for c in crops)
        assert composite.shape[1] == total_w

    def test_bgr_input_returns_grayscale_composite(
        self, sample_bgr_image: np.ndarray[Any, np.dtype[np.uint8]]
    ) -> None:
        boxes = np.array([[10.0, 10.0, 50.0, 50.0]], dtype=np.float64)
        _, _, composite, _ = _crop_and_pack(sample_bgr_image, boxes, pad_pixels=0)
        assert composite is not None
        assert composite.ndim == 2  # should be grayscale

    def test_padding_clamps_to_frame_bounds(
        self, sample_gray_image: np.ndarray[Any, np.dtype[np.uint8]]
    ) -> None:
        # Box at the edge — padding shouldn't go negative
        boxes = np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float64)
        crops, offsets, _, _ = _crop_and_pack(sample_gray_image, boxes, pad_pixels=20)
        assert len(crops) == 1
        x_off, y_off = offsets[0]
        assert x_off >= 0
        assert y_off >= 0


class TestReprojectTags:
    """Tests for the _reproject_tags function."""

    def test_valid_tag_added_to_dict(self) -> None:
        crops = [np.zeros((50, 60), dtype=np.uint8)]
        canvas_x_offsets = [0]
        offsets_xy = [(100, 200)]
        frame_dict: dict[tuple[Any, str], Any] = {}

        tags = [
            {
                "id": 5,
                "center": (30.0, 25.0),
                "lb-rb-rt-lt": np.array(
                    [[25, 20], [35, 20], [35, 30], [25, 30]], dtype=np.float64
                ),
            }
        ]

        _reproject_tags(tags, crops, canvas_x_offsets, offsets_xy, frame_dict)

        assert (5, COL_CENTER_X) in frame_dict
        assert (5, COL_CENTER_Y) in frame_dict
        assert (5, COL_CORNERS) in frame_dict
        # Absolute position: x_off + (cx - crop_x0) = 100 + (30 - 0) = 130
        assert frame_dict[(5, COL_CENTER_X)] == pytest.approx(130.0)
        assert frame_dict[(5, COL_CENTER_Y)] == pytest.approx(225.0)

    def test_tag_above_max_id_ignored(self) -> None:
        crops = [np.zeros((50, 60), dtype=np.uint8)]
        canvas_x_offsets = [0]
        offsets_xy = [(0, 0)]
        frame_dict: dict[tuple[Any, str], Any] = {}

        tags = [
            {
                "id": 999,  # above MAX_VALID_TAG_ID
                "center": (30.0, 25.0),
                "lb-rb-rt-lt": np.array(
                    [[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64
                ),
            }
        ]

        _reproject_tags(tags, crops, canvas_x_offsets, offsets_xy, frame_dict)
        assert len(frame_dict) == 0


class TestProcessFrameCpu:
    """Tests for the full CPU frame-processing function."""

    def test_empty_detections(
        self, sample_gray_image: np.ndarray[Any, np.dtype[np.uint8]]
    ) -> None:
        empty_boxes = np.empty((0, 4), dtype=np.float64)
        params = _build_apriltag_params_simple()
        mock_detector = MagicMock()

        result = _process_frame_cpu(
            frame_idx=0,
            frame=sample_gray_image,
            boxes_np=empty_boxes,
            frame_width=sample_gray_image.shape[1],
            frame_height=sample_gray_image.shape[0],
            pad_pixels=10,
            apriltag_params=params,
            detector=mock_detector,
        )

        assert result[(COL_FRAME, "")] == 0
        # No detections, so only the frame key
        assert len(result) == 1
        mock_detector.detect.assert_not_called()

    def test_with_detections_calls_detector(
        self, sample_gray_image: np.ndarray[Any, np.dtype[np.uint8]]
    ) -> None:
        boxes = np.array([[10.0, 10.0, 50.0, 50.0]], dtype=np.float64)
        params = _build_apriltag_params_simple()
        mock_detector = MagicMock()
        mock_detector.detect.return_value = []

        result = _process_frame_cpu(
            frame_idx=42,
            frame=sample_gray_image,
            boxes_np=boxes,
            frame_width=sample_gray_image.shape[1],
            frame_height=sample_gray_image.shape[0],
            pad_pixels=10,
            apriltag_params=params,
            detector=mock_detector,
        )

        assert result[(COL_FRAME, "")] == 42
        mock_detector.detect.assert_called_once()
