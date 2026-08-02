from __future__ import annotations

"""Unified AprilTag composite preprocessing for preset optimisation.

All parameters are driven by a ``params`` dict so Optuna can sample them
directly.  :func:`preprocess_composite` is the single entry-point used by the
optimizer's objective function.
"""

from typing import Any

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Primitive helpers
# ---------------------------------------------------------------------------


def _disk_blur(image: np.ndarray, radius: int = 3) -> np.ndarray:
    """Disk-kernel convolution — simulates defocus / circle of confusion."""
    ys, xs = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    kernel = (xs**2 + ys**2 <= radius**2).astype(np.float32)
    kernel /= kernel.sum()
    return cv2.filter2D(image, -1, kernel)


def _unsharp_mask(
    image: np.ndarray,
    kernel_size: tuple[int, int] = (5, 5),
    sigma: float = 1.0,
    amount: float = 1.5,
) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    return cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)


def _contrast_simple(image: np.ndarray, factor: float = 1.5) -> np.ndarray:
    from PIL import Image, ImageEnhance

    pil = Image.fromarray(image)
    return np.array(ImageEnhance.Contrast(pil).enhance(factor))


def _contrast_cv2(
    image: np.ndarray, alpha: float = 1.75, beta: float = -96.0
) -> np.ndarray:
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def _contrast_clahe(
    image: np.ndarray, clip_limit: float = 2.0, grid_size: int = 8
) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    return clahe.apply(image)


def _contrast_adaptive(
    image: np.ndarray,
    block_size: int = 21,
    C: float = 5.0,
    gaussian: bool = True,
) -> np.ndarray:
    method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C if gaussian else cv2.ADAPTIVE_THRESH_MEAN_C
    bs = block_size if block_size % 2 == 1 else block_size + 1
    return cv2.adaptiveThreshold(image, 255, method, cv2.THRESH_BINARY, bs, int(C))


def _gamma(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    inv = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(image, table)


def _tone_map(image: np.ndarray, method: str = "none") -> np.ndarray:
    if method == "none":
        return image
    f = image.astype(np.float32) / 255.0
    if method == "log":
        out = np.log1p(f * 9.0) / np.log1p(9.0)
    elif method == "sqrt":
        out = np.sqrt(f)
    elif method == "sigmoid":
        out = 1.0 / (1.0 + np.exp(-10.0 * (f - 0.5)))
    else:
        return image
    return np.clip(out * 255, 0, 255).astype(np.uint8)


def _wiener(
    image: np.ndarray, psf_radius: float = 2.0, noise_level: float = 0.01
) -> np.ndarray:
    """FFT Wiener deconvolution with a disk PSF."""
    H, W = image.shape
    f = image.astype(np.float32) / 255.0
    cy, cx = H // 2, W // 2
    ys = np.arange(H, dtype=np.float32) - cy
    xs = np.arange(W, dtype=np.float32) - cx
    xx, yy = np.meshgrid(xs, ys)
    disk = (xx**2 + yy**2 <= psf_radius**2).astype(np.float32)
    disk /= disk.sum()
    psf = np.roll(disk, (-cy, -cx), axis=(0, 1))
    PSF_F = np.fft.rfft2(psf)
    F_F = np.fft.rfft2(f)
    denom = np.abs(PSF_F) ** 2 + noise_level
    result = np.fft.irfft2(F_F * PSF_F.conj() / denom, s=(H, W))
    return np.clip(result * 255, 0, 255).astype(np.uint8)


_INTERP_MAP: dict[str, int] = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def preprocess_composite(
    composite_bgr: np.ndarray, params: dict[str, Any]
) -> np.ndarray:
    """Apply the full preprocessing pipeline to a BGR composite strip.

    Pipeline order
    --------------
    0. Optional invert (BGR, before grayscale)
    1. Optional upscale
    2. Grayscale
    3. Optional median blur
    4. Optional bilateral filter
    5. Optional Wiener deconvolution
    6. Optional tone mapping (log / sqrt / sigmoid)
    7. Optional unsharp mask
    8. Contrast: clahe / cv2 / simple / adaptive / none
    9. Optional gamma correction

    Parameters
    ----------
    composite_bgr:
        Raw BGR composite strip from ``build_testset`` (uint8).
    params:
        Parameter dict — Optuna trial params or a preset JSON blob.

    Returns
    -------
    np.ndarray
        Preprocessed grayscale uint8 image, same spatial extent as input
        (possibly upscaled).
    """
    # 0. Invert
    if params.get("invert", False):
        composite_bgr = 255 - composite_bgr

    # 1. Upscale
    upscale = float(params.get("upscale", 1.0))
    if upscale != 1.0:
        interp = _INTERP_MAP.get(
            params.get("upscale_interp", "lanczos"), cv2.INTER_LANCZOS4
        )
        h, w = composite_bgr.shape[:2]
        composite_bgr = cv2.resize(
            composite_bgr,
            (int(w * upscale), int(h * upscale)),
            interpolation=interp,
        )

    # 2. Grayscale
    gray = cv2.cvtColor(composite_bgr, cv2.COLOR_BGR2GRAY)

    # 3. Median blur
    if params.get("use_median_blur", False):
        gray = cv2.medianBlur(gray, int(params.get("median_ksize", 3)))

    # 4. Bilateral filter
    if params.get("use_bilateral", False):
        gray = cv2.bilateralFilter(
            gray,
            int(params.get("bilateral_d", 5)),
            float(params.get("bilateral_sigma_color", 50.0)),
            float(params.get("bilateral_sigma_space", 50.0)),
        )

    # 5. Wiener deconvolution
    if params.get("use_wiener", False):
        gray = _wiener(
            gray,
            psf_radius=float(params.get("wiener_psf_radius", 2.0)),
            noise_level=float(params.get("wiener_noise_level", 0.01)),
        )

    # 6. Tone mapping
    tone = params.get("tone_map", "none")
    if tone != "none":
        gray = _tone_map(gray, str(tone))

    # 7. Unsharp mask
    if params.get("use_unsharp", False):
        ks = int(params.get("kernel_size", 5))
        gray = _unsharp_mask(
            gray,
            kernel_size=(ks, ks),
            sigma=float(params.get("sigma", 1.0)),
            amount=float(params.get("amount", 1.5)),
        )

    # 8. Contrast
    method = params.get("contrast_method", "clahe")
    if method == "clahe":
        gray = _contrast_clahe(
            gray,
            clip_limit=float(params.get("clahe_clip", 2.0)),
            grid_size=int(params.get("clahe_grid", 8)),
        )
    elif method == "cv2":
        gray = _contrast_cv2(
            gray,
            alpha=float(params.get("cv2_alpha", 1.75)),
            beta=float(params.get("cv2_beta", -96.0)),
        )
    elif method == "simple":
        gray = _contrast_simple(gray, factor=float(params.get("contrast_factor", 1.5)))
    elif method == "adaptive":
        gray = _contrast_adaptive(
            gray,
            block_size=int(params.get("adapt_block", 21)),
            C=float(params.get("adapt_C", 5.0)),
            gaussian=bool(params.get("adapt_gaussian", True)),
        )
    # method == "none": pass through

    # 9. Gamma
    if params.get("use_gamma", False):
        gray = _gamma(gray, gamma=float(params.get("gamma", 1.0)))

    return gray


def disk_blur_augment(composite_bgr: np.ndarray, radius: int = 3) -> np.ndarray:
    """Synthetic defocus augmentation (brightfield-style)."""
    return _disk_blur(composite_bgr, radius)
