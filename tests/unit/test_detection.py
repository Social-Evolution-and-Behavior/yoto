"""Unit tests for yoto.detection internals."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from yoto.constants import COL_CENTER_X, COL_CENTER_Y
from yoto.detection import (
    _build_apriltag_params_fast,
    _build_apriltag_params_simple,
    _crop_and_pack,
    _enhance_and_detect,
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

    @pytest.mark.parametrize(
        "builder", [_build_apriltag_params_simple, _build_apriltag_params_fast]
    )
    def test_default_preset_applies_unsharp(self, builder: Any) -> None:
        """Regression guard: the default presets must enable the unsharp mask.

        Both builders set ``kernel_size``/``sigma``/``amount`` but historically
        relied on ``use_unsharp`` defaulting to True. When that code default was
        flipped to False the built-in presets silently stopped sharpening,
        dropping the raw AprilTag decode rate (~58% -> ~42% on the IUSSI
        recording). This asserts the enhancement actually reaches the detector.
        """
        params = builder()
        assert params.get("use_unsharp") is True

        # A textured image so unsharp_mask actually changes pixels.
        yy, xx = np.mgrid[0:64, 0:64]
        img = ((xx * 4 + yy * 2) % 256).astype(np.uint8)

        captured: list[np.ndarray[Any, np.dtype[np.uint8]]] = []

        class _CaptureDetector:
            def detect(self, image: Any) -> list[dict[str, Any]]:
                captured.append(np.array(image))
                return []

        _enhance_and_detect(img, dict(params), _CaptureDetector())
        _enhance_and_detect(img, {**params, "use_unsharp": False}, _CaptureDetector())

        with_unsharp, without_unsharp = captured
        assert not np.array_equal(with_unsharp, without_unsharp)


class TestCropAndPack:
    """Tests for the _crop_and_pack function."""

    def test_empty_boxes_returns_none_composite(
        self, sample_gray_image: np.ndarray[Any, np.dtype[np.uint8]]
    ) -> None:
        empty_boxes = np.empty((0, 4), dtype=np.float64)
        crops, offsets, composite, x_offsets = _crop_and_pack(
            sample_gray_image, empty_boxes, pad_ratio=0.3
        )
        assert len(crops) == 0
        assert composite is None

    def test_single_box_crop(
        self, sample_gray_image: np.ndarray[Any, np.dtype[np.uint8]]
    ) -> None:
        boxes = np.array([[20.0, 30.0, 80.0, 90.0]], dtype=np.float64)
        crops, offsets, composite, x_offsets = _crop_and_pack(
            sample_gray_image, boxes, pad_ratio=0.1
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
            sample_gray_image, sample_bboxes, pad_ratio=0.1
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
        _, _, composite, _ = _crop_and_pack(sample_bgr_image, boxes, pad_ratio=0.0)
        assert composite is not None
        assert composite.ndim == 2  # should be grayscale

    def test_padding_clamps_to_frame_bounds(
        self, sample_gray_image: np.ndarray[Any, np.dtype[np.uint8]]
    ) -> None:
        # Box at the edge — padding shouldn't go negative
        boxes = np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float64)
        # pad_ratio=2.0 would extend 20px past each side; the clamp keeps
        # the crop inside frame bounds.
        crops, offsets, _, _ = _crop_and_pack(sample_gray_image, boxes, pad_ratio=2.0)
        assert len(crops) == 1
        x_off, y_off = offsets[0]
        assert x_off >= 0
        assert y_off >= 0

    def test_pad_ratio_grows_box_proportionally(
        self, sample_gray_image: np.ndarray[Any, np.dtype[np.uint8]]
    ) -> None:
        # 30px box at ratio=0.5 → 15px pad each side → 60px crop.
        boxes = np.array([[40.0, 40.0, 70.0, 70.0]], dtype=np.float64)
        crops, _, _, _ = _crop_and_pack(sample_gray_image, boxes, pad_ratio=0.5)
        assert crops[0].shape[0] == 60
        assert crops[0].shape[1] == 60

    def test_pad_ratio_handles_non_square_box(
        self, sample_gray_image: np.ndarray[Any, np.dtype[np.uint8]]
    ) -> None:
        # 20x40 box at ratio=0.5 → pad_x=10, pad_y=20 → 40x80 crop.
        boxes = np.array([[40.0, 40.0, 60.0, 80.0]], dtype=np.float64)
        crops, _, _, _ = _crop_and_pack(sample_gray_image, boxes, pad_ratio=0.5)
        assert crops[0].shape == (80, 40)


class TestReprojectTags:
    """Tests for the _reproject_tags function."""

    def test_valid_tag_added_to_dict(self) -> None:
        # Crop: 60×50 placed in composite at x=0, anchored at frame
        # (100, 200).  Original YOLO box: 30 px wide × 25 px tall
        # centred at frame (130, 225) — i.e. matches the tag center
        # below so the box-offset check passes trivially.
        crop_shapes = [(50, 60)]
        canvas_x_offsets = [0]
        offsets_xy = [(100, 200)]
        boxes_np = np.array([[115.0, 213.0, 145.0, 238.0]], dtype=np.float64)
        tags = [
            {
                "id": 5,
                "center": (30.0, 25.0),
                "lb-rb-rt-lt": np.array(
                    [[25, 20], [35, 20], [35, 30], [25, 30]], dtype=np.float64
                ),
            }
        ]

        decoded, rows = _reproject_tags(
            tags, crop_shapes, canvas_x_offsets, offsets_xy, boxes_np, frame_idx=0
        )

        assert decoded == {0: 5}
        assert len(rows) == 1
        assert rows[0]["tag_id"] == 5
        # Absolute position: x_off + (cx - crop_x0) = 100 + (30 - 0) = 130
        assert rows[0][COL_CENTER_X] == pytest.approx(130.0)
        assert rows[0][COL_CENTER_Y] == pytest.approx(225.0)

    def test_tag_above_max_id_ignored(self) -> None:
        crop_shapes = [(50, 60)]
        canvas_x_offsets = [0]
        offsets_xy = [(0, 0)]
        boxes_np = np.array([[0.0, 0.0, 60.0, 50.0]], dtype=np.float64)
        tags = [
            {
                "id": 999,  # above MAX_VALID_TAG_ID
                "center": (30.0, 25.0),
                "lb-rb-rt-lt": np.array(
                    [[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64
                ),
            }
        ]

        decoded, rows = _reproject_tags(
            tags, crop_shapes, canvas_x_offsets, offsets_xy, boxes_np, frame_idx=0
        )
        assert decoded == {}
        assert rows == []

    def test_tag_far_from_box_center_dropped(self) -> None:
        # YOLO box at frame (100, 200)–(140, 240): 40×40, centre (120, 220).
        # Default max_offset_ratio=0.6 → threshold = 0.6*40 = 24 px.
        # AprilTag decoded a quad sitting in the padding region: tag
        # composite-center is (5, 5) → frame (95, 195), which is
        # sqrt(25**2 + 25**2) ≈ 35 px from the box center.  Should be
        # rejected.
        crop_shapes = [(60, 60)]
        canvas_x_offsets = [0]
        offsets_xy = [(90, 190)]
        boxes_np = np.array([[100.0, 200.0, 140.0, 240.0]], dtype=np.float64)
        tags = [
            {
                "id": 7,
                "center": (5.0, 5.0),  # frame (95, 195)
                "lb-rb-rt-lt": np.array(
                    [[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64
                ),
            }
        ]

        decoded, rows = _reproject_tags(
            tags, crop_shapes, canvas_x_offsets, offsets_xy, boxes_np, frame_idx=0
        )
        assert decoded == {}
        assert rows == []

    def test_tag_near_box_center_accepted(self) -> None:
        # Same setup but tag at composite-center (30, 30) → frame (120, 220),
        # exactly the box center.
        crop_shapes = [(60, 60)]
        canvas_x_offsets = [0]
        offsets_xy = [(90, 190)]
        boxes_np = np.array([[100.0, 200.0, 140.0, 240.0]], dtype=np.float64)
        tags = [
            {
                "id": 7,
                "center": (30.0, 30.0),
                "lb-rb-rt-lt": np.array(
                    [[25, 25], [35, 25], [35, 35], [25, 35]], dtype=np.float64
                ),
            }
        ]

        decoded, rows = _reproject_tags(
            tags, crop_shapes, canvas_x_offsets, offsets_xy, boxes_np, frame_idx=0
        )
        assert decoded == {0: 7}
        assert len(rows) == 1
        assert rows[0][COL_CENTER_X] == pytest.approx(120.0)
        assert rows[0][COL_CENTER_Y] == pytest.approx(220.0)


class TestProcessFrameCpu:
    """Tests for the full CPU frame-processing function."""

    def test_empty_detections(
        self, sample_gray_image: np.ndarray[Any, np.dtype[np.uint8]]
    ) -> None:
        empty_boxes = np.empty((0, 4), dtype=np.float64)
        params = _build_apriltag_params_simple()
        mock_detector = MagicMock()

        detection_rows, quad_rows, yolo_rows = _process_frame_cpu(
            frame_idx=0,
            frame=sample_gray_image,
            boxes_np=empty_boxes,
            frame_width=sample_gray_image.shape[1],
            frame_height=sample_gray_image.shape[0],
            pad_ratio=0.3,
            apriltag_params=params,
            detector=mock_detector,
        )

        assert detection_rows == []
        assert quad_rows == []
        assert yolo_rows == []
        mock_detector.detect.assert_not_called()

    def test_with_detections_calls_detector(
        self, sample_gray_image: np.ndarray[Any, np.dtype[np.uint8]]
    ) -> None:
        boxes = np.array([[10.0, 10.0, 50.0, 50.0]], dtype=np.float64)
        params = _build_apriltag_params_simple()
        mock_detector = MagicMock()
        mock_detector.detect.return_value = []

        mock_detector.raw_quads.return_value = []

        detection_rows, quad_rows, yolo_rows = _process_frame_cpu(
            frame_idx=42,
            frame=sample_gray_image,
            boxes_np=boxes,
            frame_width=sample_gray_image.shape[1],
            frame_height=sample_gray_image.shape[0],
            pad_ratio=0.3,
            apriltag_params=params,
            detector=mock_detector,
            confs_np=np.array([0.91], dtype=np.float32),
        )

        assert detection_rows == []
        assert quad_rows == []
        # All-YOLO sidecar should always have one row per YOLO box; the
        # single box decoded nothing, so it carries decoded=False and
        # tag_id=-1 — the YOLO-fill source consumed by `yoto clean`.
        assert len(yolo_rows) == 1
        assert yolo_rows[0]["decoded"] is False
        assert yolo_rows[0]["tag_id"] == -1
        assert yolo_rows[0]["confidence"] == pytest.approx(0.91, rel=1e-3)
        mock_detector.detect.assert_called_once()
