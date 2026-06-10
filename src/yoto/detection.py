"""YOLO + AprilTag detection pipelines.

This module provides two pipeline variants for detecting AprilTag
barcodes on ants in video frames:

* :func:`run_detection_simple` — portable pipeline using the standard
  Ultralytics ``model.predict(stream=True)`` path.  Easy to understand
  and debug; works on any machine with a CUDA-capable GPU.

* :func:`run_detection_fast` — high-throughput pipeline using NVDEC
  hardware decoding (via ``PyNvVideoCodec``) and GPU-resident
  pre-processing.  Significantly faster but requires an NVIDIA GPU with
  NVDEC support and the ``PyNvVideoCodec`` package.

Both pipelines produce identical output: a pandas ``DataFrame`` with a
``MultiIndex`` of ``(tag_id, metric)`` columns indexed by frame number.
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterator

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

from yoto._types import BBoxArray, GrayImage, Image as ImageType
from yoto.constants import (
    COL_BOX_X1,
    COL_BOX_X2,
    COL_BOX_Y1,
    COL_BOX_Y2,
    COL_CENTER_X,
    COL_CENTER_Y,
    COL_CORNERS,
    COL_FRAME,
    DEFAULT_APRILTAG_THREADS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_BLUR,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_CONTRAST_FACTOR,
    DEFAULT_DECODE_SHARPENING,
    DEFAULT_DECIMATE,
    DEFAULT_DETECTOR_THREADS,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_MAX_DETECTIONS,
    DEFAULT_MAX_HAMMING,
    DEFAULT_MAX_TAG_OFFSET_RATIO,
    DEFAULT_PAD_RATIO,
    DEFAULT_QTP_DEGLITCH,
    DEFAULT_REFINE_EDGES,
    DEFAULT_TAG_FAMILY,
    DEFAULT_TARGET_SIZE,
    DEFAULT_UNSHARP_AMOUNT,
    DEFAULT_UNSHARP_KERNEL,
    DEFAULT_UNSHARP_SIGMA,
    FAST_CONTRAST_FACTOR,
    FAST_CV2_ALPHA,
    FAST_CV2_BETA,
    FAST_DECODE_SHARPENING,
    FAST_DECIMATE,
    FAST_UNSHARP_AMOUNT,
    FAST_UNSHARP_KERNEL,
    FAST_UNSHARP_SIGMA,
    MAX_VALID_TAG_ID,
)
from yoto.exceptions import ModelLoadError
from yoto.image_processing import (
    contrast_enhance_cv2,
    contrast_enhance_pil,
    unsharp_mask,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_apriltag_params_simple() -> dict[str, Any]:
    """Return the default parameter dict for the *simple* pipeline."""
    return {
        "threads": DEFAULT_APRILTAG_THREADS,
        "decimate": DEFAULT_DECIMATE,
        "blur": DEFAULT_BLUR,
        "refine_edges": DEFAULT_REFINE_EDGES,
        "decode_sharpening": DEFAULT_DECODE_SHARPENING,
        "max_hamming": DEFAULT_MAX_HAMMING,
        "kernel_size": DEFAULT_UNSHARP_KERNEL,
        "sigma": DEFAULT_UNSHARP_SIGMA,
        "amount": DEFAULT_UNSHARP_AMOUNT,
        "contrast_factor": DEFAULT_CONTRAST_FACTOR,
    }


def _build_apriltag_params_fast() -> dict[str, Any]:
    """Return the default parameter dict for the *fast* pipeline."""
    return {
        "threads": DEFAULT_APRILTAG_THREADS,
        "decimate": FAST_DECIMATE,
        "blur": DEFAULT_BLUR,
        "refine_edges": DEFAULT_REFINE_EDGES,
        "decode_sharpening": FAST_DECODE_SHARPENING,
        "max_hamming": DEFAULT_MAX_HAMMING,
        "kernel_size": FAST_UNSHARP_KERNEL,
        "sigma": FAST_UNSHARP_SIGMA,
        "amount": FAST_UNSHARP_AMOUNT,
        "contrast_method": "cv2",
        "contrast_factor": FAST_CONTRAST_FACTOR,
        "cv2_alpha": FAST_CV2_ALPHA,
        "cv2_beta": FAST_CV2_BETA,
    }


def _create_detector(
    apriltag_params: dict[str, Any],
    family: str = DEFAULT_TAG_FAMILY,
) -> Any:
    """Instantiate the AprilTag detector from the SEBLab fork.

    Parameters
    ----------
    apriltag_params : dict[str, Any]
        Detection parameters (``max_hamming``, ``decimate``, etc.).
    family : str
        Tag family string.

    Returns
    -------
    apriltag.apriltag
        Configured detector instance.
    """
    import apriltag  # type: ignore[import-untyped]

    return apriltag.apriltag(
        family=family,
        threads=DEFAULT_DETECTOR_THREADS,
        maxhamming=apriltag_params["max_hamming"],
        decimate=apriltag_params["decimate"],
        blur=apriltag_params["blur"],
        refine_edges=apriltag_params["refine_edges"],
        qtp_deglitch=DEFAULT_QTP_DEGLITCH,
        decode_sharpening=apriltag_params["decode_sharpening"],
    )


def _crop_and_pack(
    frame: ImageType,
    boxes_np: BBoxArray,
    pad_ratio: float,
) -> tuple[
    list[np.ndarray[Any, np.dtype[np.uint8]]],
    list[tuple[int, int]],
    np.ndarray[Any, np.dtype[np.uint8]] | None,
    list[int],
]:
    """Crop detected regions and pack them into a single wide strip.

    Parameters
    ----------
    frame : Image
        Full video frame (grayscale or BGR).
    boxes_np : BBoxArray
        YOLO bounding boxes in ``xyxy`` format, shape ``(N, 4)``.
    pad_ratio : float
        Per-axis padding ratio.  Each side grows by ``pad_ratio * w`` in
        x and ``pad_ratio * h`` in y, where ``w`` and ``h`` are this
        box's own width and height.  Scales padding with tag size, so
        the same value works across camera heights.

    Returns
    -------
    tuple
        ``(crops, offsets_xy, composite_gray, canvas_x_offsets)`` where
        *composite_gray* is ``None`` when there are no detections.
    """
    frame_height, frame_width = frame.shape[:2]
    crops: list[np.ndarray[Any, np.dtype[np.uint8]]] = []
    offsets_xy: list[tuple[int, int]] = []

    for box in boxes_np:
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        pad_x = int(round(pad_ratio * (x2 - x1)))
        pad_y = int(round(pad_ratio * (y2 - y1)))
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(frame_width, x2 + pad_x)
        y2 = min(frame_height, y2 + pad_y)
        crops.append(frame[y1:y2, x1:x2])
        offsets_xy.append((x1, y1))

    if not crops:
        return crops, offsets_xy, None, []

    # Pack into a horizontal strip
    strip_height = max(c.shape[0] for c in crops)
    strip_width = sum(c.shape[1] for c in crops)
    is_gray = frame.ndim == 2

    if is_gray:
        composite = np.zeros((strip_height, strip_width), dtype=np.uint8)
    else:
        composite = np.zeros((strip_height, strip_width, 3), dtype=np.uint8)

    x_cursor = 0
    canvas_x_offsets: list[int] = []
    for crop in crops:
        h, w = crop.shape[:2]
        composite[0:h, x_cursor : x_cursor + w] = crop
        canvas_x_offsets.append(x_cursor)
        x_cursor += w

    # Convert to grayscale for AprilTag
    if is_gray:
        composite_gray = composite
    else:
        composite_gray = cv2.cvtColor(composite, cv2.COLOR_BGR2GRAY)

    return crops, offsets_xy, composite_gray, canvas_x_offsets


def _enhance_and_detect(
    composite_gray: GrayImage,
    apriltag_params: dict[str, Any],
    detector: Any,
    save_quads: bool = False,
) -> tuple[list[dict[str, Any]], list[np.ndarray[Any, np.dtype[np.float64]]]]:
    """Sharpen, enhance contrast, and run AprilTag detection.

    Returns
    -------
    tuple
        ``(tags, raw_quads)``.  Tags are the decoded detections from the
        AprilTag library.  Raw quads are 4x2 corner arrays for every
        quad the detector found, decoded or not — used downstream to
        recover missing detections in the cleaning step.
    """
    img = composite_gray

    # Optional pre-stages (no-ops unless the preset enables them)
    if apriltag_params.get("invert", False):
        from yoto.image_processing import invert as _invert

        img = _invert(img)

    upscale_factor = float(apriltag_params.get("upscale", 1.0))
    if upscale_factor != 1.0:
        from yoto.image_processing import upscale as _upscale

        img = _upscale(
            img,
            factor=upscale_factor,
            interp=apriltag_params.get("upscale_interp", "lanczos"),
        )

    tone = apriltag_params.get("tone_map", "none")
    if tone and tone != "none":
        from yoto.image_processing import tone_map as _tone_map

        img = _tone_map(img, method=tone)

    if apriltag_params.get("use_gamma", False):
        from yoto.image_processing import gamma_correct

        img = gamma_correct(img, gamma=float(apriltag_params.get("gamma", 1.0)))

    if apriltag_params.get("use_median_blur", False):
        k = int(apriltag_params.get("median_kernel", 3))
        if k % 2 == 0:
            k += 1
        img = cv2.medianBlur(img, k)

    if apriltag_params.get("use_bilateral", False):
        img = cv2.bilateralFilter(
            img,
            d=int(apriltag_params.get("bilateral_d", 5)),
            sigmaColor=float(apriltag_params.get("bilateral_sigma_color", 50.0)),
            sigmaSpace=float(apriltag_params.get("bilateral_sigma_space", 50.0)),
        )

    if apriltag_params.get("use_wiener", False):
        raise NotImplementedError(
            "use_wiener is not yet implemented in yoto.image_processing"
        )

    if apriltag_params.get("use_unsharp", True):
        ks = apriltag_params["kernel_size"]
        if isinstance(ks, int):
            ks = (ks, ks)
        sharp = unsharp_mask(
            img,
            kernel_size=ks,
            sigma=apriltag_params["sigma"],
            amount=apriltag_params["amount"],
        )
    else:
        sharp = img

    contrast_method = apriltag_params.get("contrast_method", "simple")
    if contrast_method == "cv2":
        enhanced = contrast_enhance_cv2(
            sharp,
            alpha=apriltag_params["cv2_alpha"],
            beta=apriltag_params["cv2_beta"],
        )
    else:
        enhanced = contrast_enhance_pil(
            sharp, factor=apriltag_params["contrast_factor"]
        )

    tags: list[dict[str, Any]] = detector.detect(enhanced)

    raw_quads: list[np.ndarray[Any, np.dtype[np.float64]]] = []
    if save_quads:
        # LSEB AprilTag fork exposes `raw_quads()`; older forks silently degrade.
        try:
            rq = detector.raw_quads()
        except (AttributeError, NotImplementedError):
            rq = None
        if rq:
            for q in rq:
                arr = np.asarray(q, dtype=float)
                if arr.shape == (4, 2):
                    raw_quads.append(arr)

    if upscale_factor != 1.0:
        for tag in tags:
            cx, cy = tag["center"]
            tag["center"] = (cx / upscale_factor, cy / upscale_factor)
            tag["lb-rb-rt-lt"] = tag["lb-rb-rt-lt"] / upscale_factor
        if save_quads:
            raw_quads = [q / upscale_factor for q in raw_quads]

    return tags, raw_quads


def _reproject_tags(
    tags: list[dict[str, Any]],
    crops: list[np.ndarray[Any, np.dtype[np.uint8]]],
    canvas_x_offsets: list[int],
    offsets_xy: list[tuple[int, int]],
    boxes_np: BBoxArray,
    frame_dict: dict[tuple[Any, str], Any],
    max_offset_ratio: float = DEFAULT_MAX_TAG_OFFSET_RATIO,
) -> dict[int, int]:
    """Map tag coordinates from the composite strip back to the full frame.

    For each decoded tag, bucket it back to its source YOLO crop via the
    composite x-coordinate, reproject the tag center and corners into
    full-frame coordinates, and stamp them into ``frame_dict``.

    Reject the tag (don't stamp anything) if:
      * ``tag_id > MAX_VALID_TAG_ID`` — outside the family range.
      * The reprojected center is farther from the source YOLO box's
        center than ``max_offset_ratio * min(box_w, box_h)``.  This
        catches AprilTag misdecodes that lock onto a quad in the
        padding region rather than the actual tag — the resulting
        ``tag_id`` is unreliable and would corrupt the trajectory.

    Parameters
    ----------
    tags : list[dict[str, Any]]
        Raw detections from AprilTag.
    crops : list[ndarray]
        Individual padded crop arrays (one per YOLO box, in input order).
    canvas_x_offsets : list[int]
        X-offset of each crop in the composite strip.
    offsets_xy : list[tuple[int, int]]
        Top-left corner of each crop in the original frame.
    boxes_np : BBoxArray
        Original (unpadded) YOLO boxes ``(N, 4)`` xyxy in frame coords —
        used for the box-center distance check.
    frame_dict : dict
        Mutable dict accumulating results for this frame.
    max_offset_ratio : float
        Maximum allowed tag-to-box-center distance, as a fraction of
        ``min(box_w, box_h)``.

    Returns
    -------
    dict[int, int]
        Map ``crop_idx -> tag_id`` for accepted decodes only.  Used
        downstream to (a) filter raw quads and (b) stamp ``tag_id`` /
        ``decoded`` columns on the ``_yolo.pkl`` sidecar.
    """
    if not tags or not crops:
        return {}

    n_crops = len(crops)
    cum_widths = np.cumsum([c.shape[1] for c in crops])

    # Stack the tag list into numpy arrays once.  Everything that's a
    # uniform per-tag scalar (center, id) becomes a vector; everything
    # else stays per-row.  All branches below operate on the whole batch.
    centers = np.array([t["center"] for t in tags], dtype=np.float64)  # (N, 2)
    tag_ids_arr = np.array([t["id"] for t in tags], dtype=np.int64)  # (N,)

    # Vectorized crop-bucket assignment.  ``searchsorted`` over the
    # whole batch in one call replaces N scalar ``np.searchsorted``
    # calls in the old per-tag loop.
    crop_idxs = np.searchsorted(cum_widths, centers[:, 0], side="right")

    # Mask: keep tags whose center landed inside a crop and whose id is
    # in the valid family range.  Out-of-bounds indices get clamped to
    # 0 for safe array lookups below — their ``valid`` slot stays False
    # so they never reach ``frame_dict``.
    valid = (crop_idxs < n_crops) & (tag_ids_arr <= MAX_VALID_TAG_ID)
    if not valid.any():
        return {}
    safe_idxs = np.where(valid, crop_idxs, 0)

    # Vectorized center reprojection to frame coordinates.
    canvas_x_arr = np.asarray(canvas_x_offsets, dtype=np.float64)
    offsets_x_arr = np.asarray([o[0] for o in offsets_xy], dtype=np.float64)
    offsets_y_arr = np.asarray([o[1] for o in offsets_xy], dtype=np.float64)
    abs_xs = offsets_x_arr[safe_idxs] + (centers[:, 0] - canvas_x_arr[safe_idxs])
    abs_ys = offsets_y_arr[safe_idxs] + centers[:, 1]

    # Vectorized box-center offset filter.  Skipped entirely when
    # disabled (``max_offset_ratio == inf`` from ``--no-tag-offset-filter``).
    if np.isfinite(max_offset_ratio):
        widths = boxes_np[:, 2] - boxes_np[:, 0]
        heights = boxes_np[:, 3] - boxes_np[:, 1]
        box_cx = (boxes_np[:, 0] + boxes_np[:, 2]) * 0.5
        box_cy = (boxes_np[:, 1] + boxes_np[:, 3]) * 0.5
        thresh_sq = (max_offset_ratio * np.minimum(widths, heights)) ** 2
        dx = abs_xs - box_cx[safe_idxs]
        dy = abs_ys - box_cy[safe_idxs]
        valid &= (dx * dx + dy * dy) <= thresh_sq[safe_idxs]
        if not valid.any():
            return {}

    # Vectorized corner reprojection.  Stack to (N, 4, 2), then add the
    # per-tag x/y offset broadcast across the 4 corners.
    all_corners = np.stack([t["lb-rb-rt-lt"] for t in tags], axis=0).astype(
        np.float64, copy=True
    )
    x_offs = offsets_x_arr[safe_idxs] - canvas_x_arr[safe_idxs]
    y_offs = offsets_y_arr[safe_idxs]
    all_corners[:, :, 0] += x_offs[:, np.newaxis]
    all_corners[:, :, 1] += y_offs[:, np.newaxis]

    # Final write loop.  Dict assignment is inherently per-key, but
    # every scalar is pre-converted via ``.tolist()`` so the inner body
    # has zero numpy overhead.
    decoded: dict[int, int] = {}
    abs_xs_list = abs_xs.tolist()
    abs_ys_list = abs_ys.tolist()
    crop_idxs_list = safe_idxs.tolist()
    tag_ids_list = tag_ids_arr.tolist()
    accepted = np.flatnonzero(valid).tolist()
    for i in accepted:
        crop_idx = crop_idxs_list[i]
        tag_id = tag_ids_list[i]
        frame_dict[(tag_id, COL_CENTER_X)] = abs_xs_list[i]
        frame_dict[(tag_id, COL_CENTER_Y)] = abs_ys_list[i]
        frame_dict[(tag_id, COL_CORNERS)] = all_corners[i]
        decoded[crop_idx] = tag_id

    return decoded


def _filter_and_reproject_quads(
    raw_quads: list[np.ndarray[Any, np.dtype[np.float64]]],
    decoded_crops: set[int] | dict[int, int],
    crops: list[np.ndarray[Any, np.dtype[np.uint8]]],
    cum_widths: np.ndarray[Any, np.dtype[np.int64]],
    canvas_x_offsets: list[int],
    offsets_xy: list[tuple[int, int]],
    frame_idx: int,
) -> list[dict[str, Any]]:
    """Map undecoded raw quads back to original-frame coordinates.

    A crop's quad is kept only if (a) no decoded tag came from that
    crop and (b) exactly one quad fell inside it — both conditions
    keep ambiguous cases out of downstream cleaning.
    """
    if not raw_quads or not crops:
        return []

    by_crop: dict[int, list[np.ndarray[Any, np.dtype[np.float64]]]] = {}
    for q in raw_quads:
        qcx = float(q[:, 0].mean())
        idx = int(np.searchsorted(cum_widths, qcx, side="right"))
        if idx >= len(crops):
            continue
        by_crop.setdefault(idx, []).append(q)

    rows: list[dict[str, Any]] = []
    for idx, quads in by_crop.items():
        if idx in decoded_crops or len(quads) != 1:
            continue
        q = quads[0]
        crop_x0 = canvas_x_offsets[idx]
        x_off, y_off = offsets_xy[idx]
        abs_corners = q.copy()
        abs_corners[:, 0] += x_off - crop_x0
        abs_corners[:, 1] += y_off
        rows.append(
            {
                COL_FRAME: frame_idx,
                COL_CENTER_X: float(abs_corners[:, 0].mean()),
                COL_CENTER_Y: float(abs_corners[:, 1].mean()),
                COL_CORNERS: abs_corners,
            }
        )
    return rows


def _collect_all_yolo_boxes(
    boxes_np: BBoxArray,
    confs_np: np.ndarray[Any, np.dtype[np.float32]] | None,
    decoded_tag_ids: dict[int, int],
    frame_idx: int,
) -> list[dict[str, Any]]:
    """Return one row per YOLO box (decoded or not).

    Each row carries the box geometry, confidence, a ``decoded`` flag
    and the ``tag_id`` produced when decoding succeeded (``-1`` when
    undecoded).  This sidecar is a strict superset of the main
    decoded-tags pickle: rows with ``decoded == False`` are the YOLO-fill
    source consumed by ``yoto clean``, and rows with ``decoded == True``
    document which YOLO box produced each entry in the main pickle.
    """
    rows: list[dict[str, Any]] = []
    for i, box in enumerate(boxes_np):
        x1, y1, x2, y2 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
        tag_id = decoded_tag_ids.get(i, -1)
        rows.append(
            {
                COL_FRAME: frame_idx,
                COL_BOX_X1: x1,
                COL_BOX_Y1: y1,
                COL_BOX_X2: x2,
                COL_BOX_Y2: y2,
                COL_CENTER_X: 0.5 * (x1 + x2),
                COL_CENTER_Y: 0.5 * (y1 + y2),
                "confidence": (
                    float(confs_np[i]) if confs_np is not None else float("nan")
                ),
                "decoded": tag_id != -1,
                "tag_id": tag_id,
            }
        )
    return rows


def _process_frame_cpu(
    frame_idx: int,
    frame: ImageType,
    boxes_np: BBoxArray,
    frame_width: int,
    frame_height: int,
    pad_ratio: float,
    apriltag_params: dict[str, Any],
    detector: Any,
    confs_np: np.ndarray[Any, np.dtype[np.float32]] | None = None,
    max_offset_ratio: float = DEFAULT_MAX_TAG_OFFSET_RATIO,
    save_yolo: bool = True,
    save_quads: bool = False,
) -> tuple[
    dict[tuple[Any, str], Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Full CPU-side work for one frame: crop, pack, enhance, detect, reproject.

    Designed to run in a worker thread so it can overlap with the next
    frame's GPU inference.

    Returns
    -------
    tuple
        ``(frame_dict, quad_rows, all_yolo_rows)``.  ``frame_dict`` has
        ``(tag_id, metric)`` keys for decoded tags; ``quad_rows`` lists
        surviving raw AprilTag quads from undecoded crops;
        ``all_yolo_rows`` lists *every* YOLO box with its confidence,
        a ``decoded`` flag, and the resulting ``tag_id`` (``-1`` when
        undecoded).  The undecoded subset of ``all_yolo_rows`` is the
        YOLO-fill source consumed by ``yoto clean``.
    """
    frame_dict: dict[tuple[Any, str], Any] = {(COL_FRAME, ""): frame_idx}

    crops, offsets_xy, composite_gray, canvas_x_offsets = _crop_and_pack(
        frame, boxes_np, pad_ratio
    )

    if composite_gray is None:
        return frame_dict, [], []

    tags, raw_quads = _enhance_and_detect(
        composite_gray, apriltag_params, detector, save_quads=save_quads
    )
    decoded_tag_ids = _reproject_tags(
        tags,
        crops,
        canvas_x_offsets,
        offsets_xy,
        boxes_np,
        frame_dict,
        max_offset_ratio=max_offset_ratio,
    )

    quad_rows: list[dict[str, Any]] = []
    if save_quads:
        cum_widths = np.cumsum([c.shape[1] for c in crops])
        quad_rows = _filter_and_reproject_quads(
            raw_quads,
            decoded_tag_ids,
            crops,
            cum_widths,
            canvas_x_offsets,
            offsets_xy,
            frame_idx,
        )

    all_yolo_rows: list[dict[str, Any]] = []
    if save_yolo:
        all_yolo_rows = _collect_all_yolo_boxes(
            boxes_np, confs_np, decoded_tag_ids, frame_idx
        )

    return frame_dict, quad_rows, all_yolo_rows


# ---------------------------------------------------------------------------
# NVDEC helpers (fast pipeline only)
# ---------------------------------------------------------------------------


class _NvdecResult:
    """Minimal stand-in for an Ultralytics ``Results`` object.

    Exposes ``.orig_img`` (NV12 Y plane, grayscale), ``.boxes.xyxy``
    and ``.boxes.conf`` (CPU torch tensors), and a ``.speed`` dict.
    """

    __slots__ = ("orig_img", "boxes", "speed")

    def __init__(
        self,
        orig_img: GrayImage,
        xyxy: Any,
        conf: Any,
        speed: dict[str, float],
    ) -> None:
        self.orig_img = orig_img
        self.boxes = type("_Boxes", (), {"xyxy": xyxy, "conf": conf})()
        self.speed = speed


def _nv12_to_model_input(
    nv12: Any,
    h: int,
    w: int,
    target: int,
) -> Any:
    """Convert a ``(3*H/2, W)`` NV12 CUDA tensor to a ``(3, T, T)`` float16 tensor.

    Uses BT.601 colour conversion.  Aspect ratio is preserved only when
    the source is square (which is the case for 4512x4512 ant videos).

    Parameters
    ----------
    nv12 : torch.Tensor
        Raw NV12 frame on GPU.
    h : int
        Frame height.
    w : int
        Frame width.
    target : int
        Square side length for the model input.

    Returns
    -------
    torch.Tensor
        ``(3, target, target)`` float16 RGB tensor in ``[0, 1]``.
    """
    import torch
    import torch.nn.functional as F

    y = nv12[:h, :].float().mul_(1.0 / 255.0)
    uv = nv12[h:, :].view(h // 2, w // 2, 2).float().mul_(1.0 / 255.0)

    y_ = F.interpolate(
        y[None, None],
        size=(target, target),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    uv_ = F.interpolate(
        uv.permute(2, 0, 1)[None],
        size=(target, target),
        mode="bilinear",
        align_corners=False,
    )[0]

    u = uv_[0].sub_(0.5)
    v = uv_[1].sub_(0.5)

    r = y_ + 1.402 * v
    g = y_ - 0.344 * u - 0.714 * v
    b = y_ + 1.772 * u

    return torch.stack([r, g, b], dim=0).clamp_(0, 1)


class _TRTModule:
    """Direct TensorRT engine loader for ``.engine`` / ``.trt`` weights.

    Ultralytics' ``AutoBackend`` manages internal CUDA binding addresses
    that become corrupted when the backbone is called directly as
    ``model.model(tensor)`` outside its ``predict`` loop.  That surfaces
    as ``CUDA_ERROR_ILLEGAL_ADDRESS`` inside TensorRT's ``executeV2`` and
    poisons the CUDA context — which in turn crashes ``PyNvVideoCodec``'s
    NVDEC decoder.

    This class bypasses ultralytics entirely: it strips ultralytics'
    metadata prefix (4-byte LE length + UTF-8 JSON) from the engine
    file, deserializes the raw TRT engine, and exposes a
    ``__call__(tensor) -> tensor`` interface that matches the YOLO
    backbone, so ``_pynvdec_predict`` works unchanged for both paths.

    The engine must be exported with ``dynamic=True`` to accept variable
    batch sizes:

    .. code-block:: bash

        yolo export model=yolo.pt format=engine imgsz=1024 \\
            half=True batch=20 dynamic=True
    """

    def __init__(self, engine_path: str) -> None:
        import json
        import struct

        import tensorrt as trt  # type: ignore[import-not-found]
        import torch

        with open(engine_path, "rb") as f:
            meta_len = struct.unpack("<I", f.read(4))[0]
            meta_json = f.read(meta_len).decode("utf-8")
            engine_data = f.read()
        self._meta = json.loads(meta_json)
        self.names: dict[int, str] = self._meta.get("names", {})
        self.nc: int = len(self.names) or 1

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        if self.engine is None:
            # TensorRT writes the real reason to its Logger (see stderr).
            # The most common cause is a TRT version mismatch between the
            # engine and the installed tensorrt; engines are not portable
            # across TRT versions or GPUs.
            raise RuntimeError(
                f"TensorRT failed to deserialize {engine_path} "
                f"(installed tensorrt {trt.__version__}). "
                "See the [TRT] [E] line above for details. If it says "
                "'engine plan file is not compatible', re-export on this "
                "machine: `yolo export model=<MODEL>.pt format=engine "
                "imgsz=1024 half=True batch=20 dynamic=True`."
            )
        self.context = self.engine.create_execution_context()
        # Dedicated non-default stream avoids TRT's extra
        # cudaStreamSynchronize overhead on the default stream.
        self._stream = torch.cuda.Stream()

    def __call__(self, x: Any) -> Any:
        import torch

        # The caller produces ``x`` on the current (usually default) CUDA
        # stream.  TRT runs on self._stream, and any half->float cast
        # must also run on self._stream so the cast + TRT read share one
        # stream.  We therefore:
        #   1. make self._stream wait for the current stream (so the
        #      original x is fully produced before we read it),
        #   2. do the cast + TRT enqueue on self._stream,
        #   3. CPU-synchronize on self._stream so the caller can read
        #      ``output`` on whatever stream it likes.
        # Without step (1), TRT sporadically read half-written input
        # memory and produced a batch of zero detections — the cause of
        # the "all IDs disappear for a few frames" bug.
        current = torch.cuda.current_stream(x.device)
        self._stream.wait_stream(current)
        with torch.cuda.stream(self._stream):
            if x.dtype != torch.float32:
                x = x.float()
            x = x.contiguous()
            B, C, H, W = x.shape
            self.context.set_input_shape("images", (B, C, H, W))
            out_shape = self.context.get_tensor_shape("output0")
            output = torch.empty(tuple(out_shape), dtype=torch.float32, device=x.device)
            self.context.set_tensor_address("images", x.data_ptr())
            self.context.set_tensor_address("output0", output.data_ptr())
            self.context.execute_async_v3(self._stream.cuda_stream)
        self._stream.synchronize()
        return output


class _ONNXModule:
    """Direct ONNX Runtime loader for ``.onnx`` weights.

    Uses the CUDA execution provider with ``IOBinding`` so the input
    tensor stays on GPU (zero-copy via ``data_ptr()``).  Only the much
    smaller output is copied back through CPU.  Exposes the same
    ``__call__(tensor) -> tensor`` interface as ``_TRTModule``.
    """

    def __init__(self, onnx_path: str) -> None:
        import ast

        import onnxruntime as ort  # type: ignore[import-not-found]

        providers = [
            ("CUDAExecutionProvider", {"device_id": 0}),
            "CPUExecutionProvider",
        ]
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        if "CUDAExecutionProvider" not in self.session.get_providers():
            raise RuntimeError(
                "onnxruntime could not use CUDAExecutionProvider. Install "
                "`onnxruntime-gpu` and ensure CUDA/cuDNN versions match."
            )

        inp = self.session.get_inputs()[0]
        out = self.session.get_outputs()[0]
        self._input_name = inp.name
        self._output_name = out.name
        self._fp16 = inp.type == "tensor(float16)"

        # Ultralytics embeds class names in the ONNX custom metadata map.
        meta = self.session.get_modelmeta().custom_metadata_map
        self.names: dict[int, str] = {}
        if "names" in meta:
            try:
                self.names = ast.literal_eval(meta["names"])
            except (ValueError, SyntaxError):
                pass
        self.nc: int = len(self.names) or 1

    def __call__(self, x: Any) -> Any:
        import numpy as np
        import torch

        x = (x.half() if self._fp16 else x.float()).contiguous()
        np_dtype = np.float16 if self._fp16 else np.float32

        io = self.session.io_binding()
        io.bind_input(
            name=self._input_name,
            device_type="cuda",
            device_id=0,
            element_type=np_dtype,
            shape=tuple(x.shape),
            buffer_ptr=x.data_ptr(),
        )
        io.bind_output(self._output_name, device_type="cuda", device_id=0)
        self.session.run_with_iobinding(io)
        # Output is tiny compared to the image input; CPU hop is cheap.
        out_np = io.get_outputs()[0].numpy()
        return torch.from_numpy(out_np).to(x.device)


def _fuse_overlapping_boxes(dets: Any, iou_thres: float) -> Any:
    """Greedy-cluster boxes by IoU and replace each cluster with the union.

    Drop-in replacement for the suppression half of NMS.  Standard NMS
    keeps the most-confident box of each overlap cluster and discards the
    rest; this function instead returns one fused box per cluster, with
    corners equal to the min/max of all member corners.  Confidence and
    class are taken from the highest-confidence member.

    "Cluster" is connected-components on the IoU adjacency graph (edge
    iff ``IoU >= iou_thres``), so chains like A–B–C with B overlapping
    both A and C all collapse into one box even when A and C don't
    directly overlap.

    Parameters
    ----------
    dets : Tensor
        Shape ``(N, 6)`` with columns ``[x1, y1, x2, y2, conf, cls]``,
        already conf-filtered and in xyxy format.
    iou_thres : float
        Boxes with pairwise ``IoU >= iou_thres`` are placed in the same
        cluster.

    Returns
    -------
    Tensor
        Shape ``(M, 6)`` with ``M <= N``, same column layout, sorted by
        descending confidence.
    """
    import torch
    from torchvision.ops import box_iou  # type: ignore[import-untyped]

    if dets.numel() == 0:
        return dets

    n = int(dets.shape[0])
    adj = box_iou(dets[:, :4], dets[:, :4]) >= iou_thres

    # Connected components via iterative BFS.  ``component[i] == -1`` means
    # box ``i`` hasn't been assigned to a cluster yet.
    component = [-1] * n
    out_rows: list[Any] = []

    for seed in range(n):
        if component[seed] != -1:
            continue
        frontier = [seed]
        members: list[int] = []
        while frontier:
            u = frontier.pop()
            if component[u] != -1:
                continue
            component[u] = seed
            members.append(u)
            neighbours = adj[u].nonzero(as_tuple=True)[0].tolist()
            for v in neighbours:
                if component[v] == -1:
                    frontier.append(v)

        cluster = dets[members]
        x1 = cluster[:, 0].min()
        y1 = cluster[:, 1].min()
        x2 = cluster[:, 2].max()
        y2 = cluster[:, 3].max()
        max_idx = int(cluster[:, 4].argmax())
        out_rows.append(
            torch.stack([x1, y1, x2, y2, cluster[max_idx, 4], cluster[max_idx, 5]])
        )

    fused = torch.stack(out_rows)
    # Sort by descending confidence to match ultralytics NMS output order.
    order = fused[:, 4].argsort(descending=True)
    return fused[order]


def _fuse_nms(
    preds: Any,
    conf_thres: float,
    iou_thres: float,
    max_det: int,
    nc: int,
) -> list[Any]:
    """NMS-shaped wrapper that fuses overlapping boxes instead of suppressing.

    Signature mirrors :func:`ultralytics.utils.nms.non_max_suppression`
    so it is a drop-in replacement at the call site.  Internally it
    calls ultralytics NMS once with ``iou_thres=1.0`` (effectively no
    suppression of distinct boxes) purely to reuse its conf filtering,
    xywh→xyxy conversion, class-aware grouping and ``max_det`` capping,
    then runs :func:`_fuse_overlapping_boxes` on each per-image result.
    """
    from ultralytics.utils.nms import non_max_suppression

    pre = non_max_suppression(
        preds,
        conf_thres=conf_thres,
        iou_thres=1.0,
        max_det=max_det,
        nc=nc,
    )
    return [_fuse_overlapping_boxes(d, iou_thres) for d in pre]


def _pynvdec_predict(
    video_path: str,
    model_module: Any,
    conf_thres: float,
    iou_thres: float,
    batch_size: int,
    target_size: int = DEFAULT_TARGET_SIZE,
    max_det: int = DEFAULT_MAX_DETECTIONS,
    nc: int = 1,
    nms_mode: str = "suppress",
    debug: bool = False,
) -> Iterator[_NvdecResult]:
    """Drop-in generator replacement for ``model.predict(stream=True)``.

    Uses NVDEC hardware decoding and keeps the full BGR frame on the GPU.
    Only the grayscale Y plane is copied to CPU for the AprilTag path.

    Parameters
    ----------
    video_path : str
        Path to the input video file.
    model_module : torch.nn.Module
        Raw YOLO backbone, already ``.eval().half().cuda()``.
    conf_thres : float
        Confidence threshold for NMS.
    iou_thres : float
        IoU threshold for NMS.
    batch_size : int
        Number of frames decoded per batch.
    target_size : int
        Model input resolution.
    max_det : int
        Maximum detections per frame.
    nc : int
        Number of object classes.
    nms_mode : str
        ``"suppress"`` (default) — standard NMS that drops lower-conf
        boxes whose IoU with a kept box exceeds the threshold.
        ``"fuse"`` — :func:`_fuse_nms` replaces each overlap cluster
        with the min/max-corner union (see its docstring for details).
    debug : bool
        When True, populate ``speed`` dicts with real timings.

    Yields
    ------
    _NvdecResult
        One result per decoded frame.
    """
    import torch
    import PyNvVideoCodec as nvc  # type: ignore[import-untyped]
    from ultralytics.utils.nms import non_max_suppression

    if nms_mode not in ("suppress", "fuse"):
        raise ValueError(f"nms_mode must be 'suppress' or 'fuse', got {nms_mode!r}")
    nms_fn = _fuse_nms if nms_mode == "fuse" else non_max_suppression

    dec = nvc.ThreadedDecoder(
        video_path,
        buffer_size=max(batch_size * 2, 8),
        gpu_id=0,
        use_device_memory=True,
    )
    md = dec.get_stream_metadata()
    frame_h, frame_w = md.height, md.width
    total = md.num_frames
    scale_x = frame_w / target_size
    scale_y = frame_h / target_size

    yielded = 0
    while yielded < total:
        bs = min(batch_size, total - yielded)

        if debug:
            t0 = time.perf_counter()
        frames = dec.get_batch_frames(bs)
        if not frames:
            break
        actual_bs = len(frames)
        nv12s = [torch.from_dlpack(f) for f in frames]
        if debug:
            torch.cuda.synchronize()
            t_decode = time.perf_counter() - t0
            t0 = time.perf_counter()

        inputs = torch.stack(
            [_nv12_to_model_input(t, frame_h, frame_w, target_size) for t in nv12s],
            dim=0,
        ).half()
        if debug:
            torch.cuda.synchronize()
            t_preproc = time.perf_counter() - t0
            t0 = time.perf_counter()

        with torch.no_grad():
            preds = model_module(inputs)
        if debug:
            torch.cuda.synchronize()
            t_infer = time.perf_counter() - t0
            t0 = time.perf_counter()

        if isinstance(preds, (tuple, list)):
            preds = preds[0]
        nms_out = nms_fn(
            preds,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            max_det=max_det,
            nc=nc,
        )
        # Scale boxes back to original frame coordinates
        for dets in nms_out:
            if dets.numel():
                dets[:, 0] *= scale_x
                dets[:, 1] *= scale_y
                dets[:, 2] *= scale_x
                dets[:, 3] *= scale_y

        # Copy only the Y plane (grayscale) to CPU for AprilTag
        y_cpu_list = [t[:frame_h, :].cpu().numpy() for t in nv12s]
        if debug:
            t_postproc = time.perf_counter() - t0

        if debug:
            speed = {
                "preprocess": (t_decode + t_preproc) * 1000.0 / actual_bs,
                "inference": t_infer * 1000.0 / actual_bs,
                "postprocess": t_postproc * 1000.0 / actual_bs,
            }
        else:
            speed = {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}

        for y_np, dets in zip(y_cpu_list, nms_out):
            yield _NvdecResult(
                y_np,
                dets[:, :4].cpu(),
                dets[:, 4].cpu(),
                speed,
            )
            yielded += 1
            if yielded >= total:
                return


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _stamp_attrs(
    df: pd.DataFrame,
    video_path: str,
    pipeline: str,
    extra: dict[str, Any] | None = None,
) -> None:
    """Embed yoto provenance into ``df.attrs`` (preserved through pickle)."""
    from datetime import datetime, timezone

    from yoto import __version__

    df.attrs["yoto_version"] = __version__
    df.attrs["yoto_stage"] = "detect"
    df.attrs["yoto_pipeline"] = pipeline
    df.attrs["yoto_video"] = os.path.abspath(video_path)
    df.attrs["yoto_created_utc"] = datetime.now(timezone.utc).isoformat()
    if extra:
        df.attrs.update(extra)


DEFAULT_WEIGHTS = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "models", "detect14.pt")
)


def _save_sidecar(
    rows: list[dict[str, Any]],
    main_pkl: str,
    video_path: str,
    pipeline: str,
    suffix: str,
    kind: str,
    empty_columns: list[str],
) -> None:
    """Write a per-frame long-format sidecar next to the main pickle.

    Output path: ``<main_pkl_stem><suffix>.pkl``.  Indexed by frame
    (non-unique).  ``empty_columns`` defines the schema for the
    no-data case so downstream code can read the file unconditionally.
    """
    out_path = (
        main_pkl[:-4] + suffix + ".pkl"
        if main_pkl.endswith(".pkl")
        else main_pkl + suffix + ".pkl"
    )
    if rows:
        df = pd.DataFrame(rows).set_index(COL_FRAME)
    else:
        df = pd.DataFrame(columns=empty_columns).rename_axis(COL_FRAME)
    _stamp_attrs(df, video_path, pipeline=pipeline, extra={"yoto_kind": kind})
    df.to_pickle(out_path)


def _probe_frame_count(video_path: str, fallback: int = 18000) -> int:
    """Return the video's frame count via OpenCV, or *fallback* on error."""
    cap = cv2.VideoCapture(video_path)
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    return n if n > 0 else fallback


def run_detection_simple(
    video_path: str,
    output_path: str | None = None,
    yolo_weights: str = DEFAULT_WEIGHTS,
    data_suffix: str = "_apriltagDetect14",
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    pad_ratio: float = DEFAULT_PAD_RATIO,
    max_offset_ratio: float = DEFAULT_MAX_TAG_OFFSET_RATIO,
    num_frames: int | None = None,
    apriltag_params: dict[str, Any] | None = None,
    preset: str | None = None,
    save_yolo: bool = True,
    save_quads: bool = False,
    tag_family: str = DEFAULT_TAG_FAMILY,
) -> pd.DataFrame:
    """Run the simple (portable) YOLO + AprilTag detection pipeline.

    This is the easier-to-read variant that uses the standard Ultralytics
    ``model.predict(stream=True)`` path.  It works on any CUDA-capable
    machine but is slower than :func:`run_detection_fast`.

    Parameters
    ----------
    video_path : str
        Path to the input video file.
    output_path : str | None
        Directory for the output pickle.  When ``None`` the pickle is
        written next to the input video.
    yolo_weights : str
        Path to YOLO weights (``.engine`` or ``.pt``).
    data_suffix : str
        Suffix appended to the video basename for the output filename.
    conf_threshold : float
        YOLO confidence threshold.
    pad_ratio : float
        Per-axis padding ratio applied to each YOLO box before cropping
        (see :func:`_crop_and_pack`).  Each side grows by
        ``pad_ratio * box_dim``, so the padding scales with apparent
        tag size.
    num_frames : int
        Expected total frame count (used for the progress bar only).
    apriltag_params : dict[str, Any] | None
        Override default AprilTag / image-processing parameters.  When
        ``None``, :func:`_build_apriltag_params_simple` defaults are used.

    Returns
    -------
    pd.DataFrame
        MultiIndex DataFrame with ``(tag_id, metric)`` columns indexed
        by frame number.

    Raises
    ------
    ModelLoadError
        If YOLO weights cannot be loaded.
    """
    if apriltag_params is None:
        apriltag_params = _build_apriltag_params_simple()
    if preset is not None:
        from yoto.apriltag_presets import load_preset, merge_preset

        apriltag_params = merge_preset(apriltag_params, load_preset(preset))

    if num_frames is None:
        num_frames = _probe_frame_count(video_path)

    # Resolve output path
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        video_basename = os.path.basename(video_path).rsplit(".", 1)[0]
        out_pkl = os.path.join(output_path, video_basename + data_suffix + ".pkl")
    else:
        out_pkl = video_path.rsplit(".", 1)[0] + data_suffix + ".pkl"

    start_time = time.time()

    # Load models
    try:
        seg_model = YOLO(yolo_weights, task="detect")
    except Exception as exc:
        raise ModelLoadError(yolo_weights, str(exc)) from exc

    detector = _create_detector(apriltag_params, family=tag_family)

    results = seg_model.predict(
        source=video_path,
        conf=conf_threshold,
        iou=iou_threshold,
        stream=True,
        batch=2,
        verbose=False,
        half=True,
        task="detect",
    )
    results_tag: list[dict[tuple[Any, str], Any]] = []
    quads_all: list[dict[str, Any]] = []
    yolo_all: list[dict[str, Any]] = []

    from yoto._progress import make_status_updater

    disable_tqdm = bool(os.environ.get("YOTO_NO_PROGRESS"))
    status_update = make_status_updater(video_path, num_frames)
    for i, result in enumerate(
        tqdm(
            results,
            desc="Processing frames",
            total=num_frames,
            miniters=20,
            disable=disable_tqdm,
        )
    ):
        frame = result.orig_img
        boxes_np = result.boxes.xyxy.cpu().numpy()
        confs_np = result.boxes.conf.cpu().numpy() if save_yolo else None
        frame_height, frame_width = frame.shape[0], frame.shape[1]

        frame_dict, quad_rows, yolo_rows = _process_frame_cpu(
            i,
            frame,
            boxes_np,
            frame_width,
            frame_height,
            pad_ratio,
            apriltag_params,
            detector,
            confs_np=confs_np,
            max_offset_ratio=max_offset_ratio,
            save_yolo=save_yolo,
            save_quads=save_quads,
        )
        results_tag.append(frame_dict)
        if save_quads:
            quads_all.extend(quad_rows)
        if save_yolo:
            yolo_all.extend(yolo_rows)
        status_update(i + 1)

    results.close()

    # Build DataFrame
    df = pd.DataFrame(results_tag)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    df = df.set_index(COL_FRAME)
    _stamp_attrs(
        df,
        video_path,
        pipeline="simple",
        extra={
            "yoto_yolo_weights": yolo_weights,
            "yoto_preset": preset,
            "yoto_pad_ratio": pad_ratio,
        },
    )
    df.to_pickle(out_pkl)
    if save_quads:
        _save_sidecar(
            quads_all,
            out_pkl,
            video_path,
            "simple",
            suffix="_quads",
            kind="quads",
            empty_columns=[COL_CENTER_X, COL_CENTER_Y, COL_CORNERS],
        )
    if save_yolo:
        _save_sidecar(
            yolo_all,
            out_pkl,
            video_path,
            "simple",
            suffix="_yolo",
            kind="yolo_boxes",
            empty_columns=[
                COL_BOX_X1,
                COL_BOX_Y1,
                COL_BOX_X2,
                COL_BOX_Y2,
                COL_CENTER_X,
                COL_CENTER_Y,
                "confidence",
                "decoded",
                "tag_id",
            ],
        )

    total_time = time.time() - start_time
    m, s = divmod(int(total_time), 60)
    logger.info("Total processing time for %s: %dm %02ds", video_path, m, s)

    return df


def run_detection_fast(
    video_path: str,
    output_path: str | None = None,
    yolo_weights: str = DEFAULT_WEIGHTS,
    data_suffix: str = "_apriltagDetect14",
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    pad_ratio: float = DEFAULT_PAD_RATIO,
    max_offset_ratio: float = DEFAULT_MAX_TAG_OFFSET_RATIO,
    batch_size: int = DEFAULT_BATCH_SIZE,
    target_size: int = DEFAULT_TARGET_SIZE,
    num_frames: int | None = None,
    debug: bool = False,
    apriltag_params: dict[str, Any] | None = None,
    preset: str | None = None,
    nms_mode: str = "suppress",
    save_yolo: bool = True,
    save_quads: bool = False,
    tag_family: str = DEFAULT_TAG_FAMILY,
) -> pd.DataFrame:
    """Run the fast (NVDEC) YOLO + AprilTag detection pipeline.

    Uses hardware video decoding and GPU-resident pre-processing for
    maximum throughput.  Requires ``PyNvVideoCodec`` and an NVIDIA GPU
    with NVDEC support.

    Parameters
    ----------
    video_path : str
        Path to the input video file.
    output_path : str | None
        Directory for the output pickle.  When ``None`` the pickle is
        written next to the input video.
    yolo_weights : str
        Path to YOLO weights (``.engine`` or ``.pt``).
    data_suffix : str
        Suffix appended to the video basename for the output filename.
    conf_threshold : float
        YOLO confidence threshold.
    iou_threshold : float
        IoU threshold for non-maximum suppression.
    pad_ratio : float
        Per-axis padding ratio applied to each YOLO box before cropping
        (see :func:`_crop_and_pack`).  Each side grows by
        ``pad_ratio * box_dim``, so the padding scales with apparent
        tag size.
    batch_size : int
        Number of frames decoded per GPU batch.
    target_size : int
        YOLO input resolution (square side).
    num_frames : int
        Expected total frame count (for the progress bar).
    debug : bool
        When True, collect and print per-stage profiling information.
    apriltag_params : dict[str, Any] | None
        Override default AprilTag / image-processing parameters.
    nms_mode : str
        ``"suppress"`` (default) keeps standard NMS; ``"fuse"`` replaces
        each overlap cluster with the union of its boxes (see
        :func:`_fuse_overlapping_boxes`).

    Returns
    -------
    pd.DataFrame
        MultiIndex DataFrame with ``(tag_id, metric)`` columns indexed
        by frame number.

    Raises
    ------
    ModelLoadError
        If YOLO weights cannot be loaded.
    """
    if apriltag_params is None:
        apriltag_params = _build_apriltag_params_fast()
    if preset is not None:
        from yoto.apriltag_presets import load_preset, merge_preset

        apriltag_params = merge_preset(apriltag_params, load_preset(preset))

    if num_frames is None:
        num_frames = _probe_frame_count(video_path)

    # Resolve output path
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        video_basename = os.path.basename(video_path).rsplit(".", 1)[0]
        out_pkl = os.path.join(output_path, video_basename + data_suffix + ".pkl")
    else:
        out_pkl = video_path.rsplit(".", 1)[0] + data_suffix + ".pkl"

    start_time = time.time()

    # Load models.  For TensorRT engines we bypass ultralytics entirely
    # (see _TRTModule docstring for why); .onnx files go through ONNX
    # Runtime with the CUDA EP; .pt weights use the existing YOLO path.
    seg_module: Any
    try:
        if yolo_weights.endswith((".engine", ".trt")):
            seg_module = _TRTModule(yolo_weights)
            yolo_nc = seg_module.nc
        elif yolo_weights.endswith(".onnx"):
            seg_module = _ONNXModule(yolo_weights)
            yolo_nc = seg_module.nc
        else:
            seg_model = YOLO(yolo_weights, task="detect")
            seg_module = seg_model.model.eval().half().cuda()
            yolo_nc = getattr(seg_model.model, "nc", None) or 1
    except Exception as exc:
        raise ModelLoadError(yolo_weights, str(exc)) from exc

    detector = _create_detector(apriltag_params, family=tag_family)

    debug_frame_limit = 500

    results = _pynvdec_predict(
        video_path,
        seg_module,
        conf_thres=conf_threshold,
        iou_thres=iou_threshold,
        batch_size=batch_size,
        target_size=target_size,
        max_det=DEFAULT_MAX_DETECTIONS,
        nc=yolo_nc,
        nms_mode=nms_mode,
        debug=debug,
    )
    results_tag: list[dict[tuple[Any, str], Any]] = []
    quads_all: list[dict[str, Any]] = []
    yolo_all: list[dict[str, Any]] = []

    effective_frames = debug_frame_limit if debug else num_frames

    # Overlap CPU work with GPU inference using a single worker thread
    max_inflight = batch_size + 4
    executor = ThreadPoolExecutor(max_workers=1)
    pending: deque[Any] = deque()

    from yoto._progress import make_status_updater

    disable_tqdm = bool(os.environ.get("YOTO_NO_PROGRESS"))
    status_update = make_status_updater(video_path, effective_frames)
    for i, result in enumerate(
        tqdm(
            results,
            desc="Processing frames",
            total=effective_frames,
            miniters=20,
            disable=disable_tqdm,
        )
    ):
        if debug and i >= debug_frame_limit:
            break

        frame = result.orig_img
        frame_height, frame_width = frame.shape[0], frame.shape[1]
        boxes_np = result.boxes.xyxy.cpu().numpy()
        confs_np = result.boxes.conf.cpu().numpy() if save_yolo else None

        fut = executor.submit(
            _process_frame_cpu,
            i,
            frame,
            boxes_np,
            frame_width,
            frame_height,
            pad_ratio,
            apriltag_params,
            detector,
            confs_np,
            max_offset_ratio,
            save_yolo,
            save_quads,
        )
        pending.append(fut)

        # Back-pressure: drain oldest futures before queueing more
        while len(pending) >= max_inflight:
            fd, qr, yr = pending.popleft().result()
            results_tag.append(fd)
            if save_quads:
                quads_all.extend(qr)
            if save_yolo:
                yolo_all.extend(yr)
        status_update(i + 1)

    # Drain remaining futures
    while pending:
        fd, qr, yr = pending.popleft().result()
        results_tag.append(fd)
        if save_quads:
            quads_all.extend(qr)
        if save_yolo:
            yolo_all.extend(yr)
    executor.shutdown()
    results.close()

    # Build DataFrame
    df = pd.DataFrame(results_tag)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    df = df.set_index(COL_FRAME)
    _stamp_attrs(
        df,
        video_path,
        pipeline="fast",
        extra={
            "yoto_yolo_weights": yolo_weights,
            "yoto_preset": preset,
            "yoto_nms_mode": nms_mode,
            "yoto_pad_ratio": pad_ratio,
        },
    )
    df.to_pickle(out_pkl)
    if save_quads:
        _save_sidecar(
            quads_all,
            out_pkl,
            video_path,
            "fast",
            suffix="_quads",
            kind="quads",
            empty_columns=[COL_CENTER_X, COL_CENTER_Y, COL_CORNERS],
        )
    if save_yolo:
        _save_sidecar(
            yolo_all,
            out_pkl,
            video_path,
            "fast",
            suffix="_yolo",
            kind="yolo_boxes",
            empty_columns=[
                COL_BOX_X1,
                COL_BOX_Y1,
                COL_BOX_X2,
                COL_BOX_Y2,
                COL_CENTER_X,
                COL_CENTER_Y,
                "confidence",
                "decoded",
                "tag_id",
            ],
        )

    total_time = time.time() - start_time
    m, s = divmod(int(total_time), 60)
    logger.info("Total processing time for %s: %dm %02ds", video_path, m, s)

    return df
