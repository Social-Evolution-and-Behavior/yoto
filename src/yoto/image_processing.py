"""Image pre-processing utilities for the AprilTag detection pipeline.

Functions in this module sharpen and enhance contrast of cropped tag
regions before they are passed to the AprilTag decoder.  All operations
are pure (no side effects) and work on NumPy arrays.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from PIL import Image, ImageEnhance

from yoto._types import GrayImage
from yoto.constants import (
    DEFAULT_CONTRAST_FACTOR,
    DEFAULT_UNSHARP_AMOUNT,
    DEFAULT_UNSHARP_KERNEL,
    DEFAULT_UNSHARP_SIGMA,
)

logger = logging.getLogger(__name__)


def unsharp_mask(
    image: GrayImage,
    kernel_size: tuple[int, int] = DEFAULT_UNSHARP_KERNEL,
    sigma: float = DEFAULT_UNSHARP_SIGMA,
    amount: float = DEFAULT_UNSHARP_AMOUNT,
) -> GrayImage:
    """Apply an unsharp mask to sharpen an image.

    The technique subtracts a Gaussian-blurred copy from the original
    and amplifies the difference, emphasising edges and fine detail.

    Parameters
    ----------
    image : GrayImage
        Input grayscale uint8 image.
    kernel_size : tuple[int, int]
        Size of the Gaussian kernel.
    sigma : float
        Standard deviation of the Gaussian kernel.
    amount : float
        Strength of the sharpening effect (1.0 = double the high-freq
        component).

    Returns
    -------
    GrayImage
        Sharpened uint8 image of the same shape as *image*.

    Examples
    --------
    >>> import numpy as np
    >>> img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    >>> out = unsharp_mask(img, kernel_size=(5, 5), sigma=1.0, amount=1.5)
    >>> out.shape == img.shape
    True
    """
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened: GrayImage = cv2.addWeighted(
        image, 1 + amount, blurred, -amount, 0
    )
    return sharpened


def contrast_enhance_pil(
    image: GrayImage,
    factor: float = DEFAULT_CONTRAST_FACTOR,
) -> GrayImage:
    """Enhance contrast using PIL's ``ImageEnhance.Contrast``.

    Parameters
    ----------
    image : GrayImage
        Input grayscale uint8 image.
    factor : float
        Contrast multiplier.  1.0 = unchanged, >1.0 = higher contrast.

    Returns
    -------
    GrayImage
        Contrast-enhanced uint8 image.

    Examples
    --------
    >>> import numpy as np
    >>> img = np.full((50, 50), 128, dtype=np.uint8)
    >>> out = contrast_enhance_pil(img, factor=2.0)
    >>> out.shape == img.shape
    True
    """
    pil_img = Image.fromarray(image)
    enhancer = ImageEnhance.Contrast(pil_img)
    enhanced = enhancer.enhance(factor)
    return np.array(enhanced, dtype=np.uint8)


def contrast_enhance_cv2(
    image: GrayImage,
    alpha: float = 2.0,
    beta: float = -80.0,
) -> GrayImage:
    """Enhance contrast using OpenCV's ``convertScaleAbs``.

    This method is faster than the PIL path and avoids a colour-space
    round-trip, making it preferable in the fast pipeline.

    Parameters
    ----------
    image : GrayImage
        Input grayscale uint8 image.
    alpha : float
        Gain (contrast multiplier).
    beta : float
        Bias (brightness offset).

    Returns
    -------
    GrayImage
        Contrast-enhanced uint8 image.

    Examples
    --------
    >>> import numpy as np
    >>> img = np.full((50, 50), 128, dtype=np.uint8)
    >>> out = contrast_enhance_cv2(img, alpha=1.5, beta=-40)
    >>> out.shape == img.shape
    True
    """
    enhanced: GrayImage = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return enhanced
