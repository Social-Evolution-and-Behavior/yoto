"""Unit tests for yoto.image_processing."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from yoto.image_processing import (
    contrast_enhance_cv2,
    contrast_enhance_pil,
    unsharp_mask,
)


class TestUnsharpMask:
    """Tests for the unsharp_mask function."""

    def test_output_shape_matches_input(
        self, sample_gray_image: np.ndarray[Any, np.dtype[np.uint8]]
    ) -> None:
        result = unsharp_mask(sample_gray_image)
        assert result.shape == sample_gray_image.shape

    def test_output_dtype_is_uint8(
        self, sample_gray_image: np.ndarray[Any, np.dtype[np.uint8]]
    ) -> None:
        result = unsharp_mask(sample_gray_image)
        assert result.dtype == np.uint8

    def test_zero_amount_returns_original(self) -> None:
        img = np.full((50, 50), 128, dtype=np.uint8)
        result = unsharp_mask(img, amount=0.0)
        np.testing.assert_array_equal(result, img)

    @pytest.mark.parametrize(
        "kernel_size,sigma,amount",
        [
            ((3, 3), 0.5, 1.0),
            ((5, 5), 1.0, 1.5),
            ((11, 11), 3.0, 5.0),
        ],
    )
    def test_various_parameters(
        self,
        sample_gray_image: np.ndarray[Any, np.dtype[np.uint8]],
        kernel_size: tuple[int, int],
        sigma: float,
        amount: float,
    ) -> None:
        result = unsharp_mask(
            sample_gray_image,
            kernel_size=kernel_size,
            sigma=sigma,
            amount=amount,
        )
        assert result.shape == sample_gray_image.shape
        assert result.dtype == np.uint8

    @given(
        img=arrays(
            dtype=np.uint8,
            shape=st.tuples(
                st.integers(min_value=10, max_value=100),
                st.integers(min_value=10, max_value=100),
            ),
        )
    )
    @settings(max_examples=20, deadline=None)
    def test_property_output_bounds(
        self, img: np.ndarray[Any, np.dtype[np.uint8]]
    ) -> None:
        result = unsharp_mask(img, kernel_size=(3, 3), sigma=1.0, amount=1.0)
        assert result.min() >= 0
        assert result.max() <= 255


class TestContrastEnhancePil:
    """Tests for PIL-based contrast enhancement."""

    def test_output_shape_matches_input(
        self, sample_gray_image: np.ndarray[Any, np.dtype[np.uint8]]
    ) -> None:
        result = contrast_enhance_pil(sample_gray_image)
        assert result.shape == sample_gray_image.shape

    def test_factor_one_preserves_image(self) -> None:
        img = np.full((50, 50), 128, dtype=np.uint8)
        result = contrast_enhance_pil(img, factor=1.0)
        np.testing.assert_array_equal(result, img)

    @pytest.mark.parametrize("factor", [0.5, 1.0, 1.5, 2.0, 3.0])
    def test_various_factors(
        self,
        sample_gray_image: np.ndarray[Any, np.dtype[np.uint8]],
        factor: float,
    ) -> None:
        result = contrast_enhance_pil(sample_gray_image, factor=factor)
        assert result.dtype == np.uint8


class TestContrastEnhanceCv2:
    """Tests for OpenCV-based contrast enhancement."""

    def test_output_shape_matches_input(
        self, sample_gray_image: np.ndarray[Any, np.dtype[np.uint8]]
    ) -> None:
        result = contrast_enhance_cv2(sample_gray_image)
        assert result.shape == sample_gray_image.shape

    def test_identity_transform(self) -> None:
        img = np.full((50, 50), 128, dtype=np.uint8)
        result = contrast_enhance_cv2(img, alpha=1.0, beta=0.0)
        np.testing.assert_array_equal(result, img)

    @pytest.mark.parametrize(
        "alpha,beta",
        [(1.0, 0.0), (1.5, -40.0), (2.0, -80.0)],
    )
    def test_various_parameters(
        self,
        sample_gray_image: np.ndarray[Any, np.dtype[np.uint8]],
        alpha: float,
        beta: float,
    ) -> None:
        result = contrast_enhance_cv2(sample_gray_image, alpha=alpha, beta=beta)
        assert result.dtype == np.uint8

    @given(
        img=arrays(
            dtype=np.uint8,
            shape=st.tuples(
                st.integers(min_value=10, max_value=100),
                st.integers(min_value=10, max_value=100),
            ),
        ),
        alpha=st.floats(min_value=0.5, max_value=3.0),
        beta=st.floats(min_value=-100.0, max_value=100.0),
    )
    @settings(max_examples=20, deadline=None)
    def test_property_output_is_uint8(
        self,
        img: np.ndarray[Any, np.dtype[np.uint8]],
        alpha: float,
        beta: float,
    ) -> None:
        result = contrast_enhance_cv2(img, alpha=alpha, beta=beta)
        assert result.dtype == np.uint8
