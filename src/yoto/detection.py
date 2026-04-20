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
    DEFAULT_PAD_PIXELS,
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
    pad_pixels: int,
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
    pad_pixels : int
        Padding added around each box before cropping.

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
        x1 = max(0, x1 - pad_pixels)
        y1 = max(0, y1 - pad_pixels)
        x2 = min(frame_width, x2 + pad_pixels)
        y2 = min(frame_height, y2 + pad_pixels)
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
) -> list[dict[str, Any]]:
    """Sharpen, enhance contrast, and run AprilTag detection.

    Parameters
    ----------
    composite_gray : GrayImage
        Grayscale composite strip of all crops.
    apriltag_params : dict[str, Any]
        Image-processing and detector parameters.
    detector : AprilTagDetectorProtocol
        Instantiated AprilTag detector.

    Returns
    -------
    list[dict[str, Any]]
        Raw tag detections from the AprilTag library.
    """
    sharp = unsharp_mask(
        composite_gray,
        kernel_size=apriltag_params["kernel_size"],
        sigma=apriltag_params["sigma"],
        amount=apriltag_params["amount"],
    )

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
    return tags


def _reproject_tags(
    tags: list[dict[str, Any]],
    crops: list[np.ndarray[Any, np.dtype[np.uint8]]],
    canvas_x_offsets: list[int],
    offsets_xy: list[tuple[int, int]],
    frame_dict: dict[tuple[Any, str], Any],
) -> None:
    """Map tag coordinates from the composite strip back to the full frame.

    Parameters
    ----------
    tags : list[dict[str, Any]]
        Raw detections from AprilTag.
    crops : list[ndarray]
        Individual crop arrays.
    canvas_x_offsets : list[int]
        X-offset of each crop in the composite strip.
    offsets_xy : list[tuple[int, int]]
        Top-left corner of each crop in the original frame.
    frame_dict : dict
        Mutable dict accumulating results for this frame.
    """
    cum_widths = np.cumsum([c.shape[1] for c in crops])

    for tag in tags:
        cx, cy = tag["center"]
        crop_idx = int(np.searchsorted(cum_widths, cx, side="right"))
        if crop_idx >= len(crops):
            continue

        crop_x0 = canvas_x_offsets[crop_idx]
        x_off, y_off = offsets_xy[crop_idx]

        abs_x = x_off + (cx - crop_x0)
        abs_y = y_off + cy

        tag_id: int = tag["id"]
        if tag_id > MAX_VALID_TAG_ID:
            continue

        abs_corners = tag["lb-rb-rt-lt"].copy()
        abs_corners[:, 0] += x_off - crop_x0
        abs_corners[:, 1] += y_off

        frame_dict[(tag_id, COL_CENTER_X)] = abs_x
        frame_dict[(tag_id, COL_CENTER_Y)] = abs_y
        frame_dict[(tag_id, COL_CORNERS)] = abs_corners


def _process_frame_cpu(
    frame_idx: int,
    frame: ImageType,
    boxes_np: BBoxArray,
    frame_width: int,
    frame_height: int,
    pad_pixels: int,
    apriltag_params: dict[str, Any],
    detector: Any,
) -> dict[tuple[Any, str], Any]:
    """Full CPU-side work for one frame: crop, pack, enhance, detect, reproject.

    This function is designed to run in a worker thread so it can
    overlap with the next frame's GPU inference.

    Parameters
    ----------
    frame_idx : int
        Zero-based frame index.
    frame : Image
        Full video frame (grayscale or BGR).
    boxes_np : BBoxArray
        YOLO bounding boxes in ``xyxy`` format.
    frame_width : int
        Width of the original frame in pixels.
    frame_height : int
        Height of the original frame in pixels.
    pad_pixels : int
        Padding around each detection box.
    apriltag_params : dict[str, Any]
        Image-processing and detector configuration.
    detector : AprilTagDetectorProtocol
        Instantiated AprilTag detector.

    Returns
    -------
    dict[tuple[Any, str], Any]
        Frame result dict with ``(tag_id, metric)`` keys.
    """
    frame_dict: dict[tuple[Any, str], Any] = {(COL_FRAME, ""): frame_idx}

    crops, offsets_xy, composite_gray, canvas_x_offsets = _crop_and_pack(
        frame, boxes_np, pad_pixels
    )

    if composite_gray is None:
        return frame_dict

    tags = _enhance_and_detect(composite_gray, apriltag_params, detector)
    _reproject_tags(tags, crops, canvas_x_offsets, offsets_xy, frame_dict)

    return frame_dict


# ---------------------------------------------------------------------------
# NVDEC helpers (fast pipeline only)
# ---------------------------------------------------------------------------


class _NvdecResult:
    """Minimal stand-in for an Ultralytics ``Results`` object.

    Exposes just the attributes the processing loop needs:

    * ``.orig_img`` — 2-D uint8 array (the NV12 Y plane, i.e. grayscale).
    * ``.boxes.xyxy`` — CPU torch tensor of shape ``(N, 4)``.
    * ``.speed`` — dict with ``preprocess`` / ``inference`` /
      ``postprocess`` timings in milliseconds.
    """

    __slots__ = ("orig_img", "boxes", "speed")

    def __init__(
        self,
        orig_img: GrayImage,
        xyxy: Any,
        speed: dict[str, float],
    ) -> None:
        self.orig_img = orig_img
        self.boxes = type("_Boxes", (), {"xyxy": xyxy})()
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

        yolo export model=detect14.pt format=engine imgsz=1024 \\
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


def _pynvdec_predict(
    video_path: str,
    model_module: Any,
    conf_thres: float,
    iou_thres: float,
    batch_size: int,
    target_size: int = DEFAULT_TARGET_SIZE,
    max_det: int = DEFAULT_MAX_DETECTIONS,
    nc: int = 1,
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
        nms_out = non_max_suppression(
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
            yield _NvdecResult(y_np, dets[:, :4].cpu(), speed)
            yielded += 1
            if yielded >= total:
                return


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_detection_simple(
    video_path: str,
    output_path: str | None = None,
    yolo_weights: str = "detect14.engine",
    data_suffix: str = "_apriltagDetect14",
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    pad_pixels: int = DEFAULT_PAD_PIXELS,
    num_frames: int = 18000,
    apriltag_params: dict[str, Any] | None = None,
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
    pad_pixels : int
        Pixels of padding around each detected bounding box.
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

    detector = _create_detector(apriltag_params)

    results = seg_model.predict(
        source=video_path,
        conf=conf_threshold,
        stream=True,
        batch=2,
        verbose=False,
        half=True,
        task="detect",
    )
    results_tag: list[dict[tuple[Any, str], Any]] = []

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
        frame_height, frame_width = frame.shape[0], frame.shape[1]

        frame_dict = _process_frame_cpu(
            i,
            frame,
            boxes_np,
            frame_width,
            frame_height,
            pad_pixels,
            apriltag_params,
            detector,
        )
        results_tag.append(frame_dict)
        status_update(i + 1)

    results.close()

    # Build DataFrame
    df = pd.DataFrame(results_tag)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    df = df.set_index(COL_FRAME)
    df.to_pickle(out_pkl)

    total_time = time.time() - start_time
    m, s = divmod(int(total_time), 60)
    logger.info("Total processing time for %s: %dm %02ds", video_path, m, s)

    return df


def run_detection_fast(
    video_path: str,
    output_path: str | None = None,
    yolo_weights: str = "detect14.engine",
    data_suffix: str = "_apriltagDetect14",
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    pad_pixels: int = DEFAULT_PAD_PIXELS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    target_size: int = DEFAULT_TARGET_SIZE,
    num_frames: int = 18000,
    debug: bool = False,
    apriltag_params: dict[str, Any] | None = None,
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
    pad_pixels : int
        Pixels of padding around each detected bounding box.
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

    detector = _create_detector(apriltag_params)

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
        debug=debug,
    )
    results_tag: list[dict[tuple[Any, str], Any]] = []

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

        fut = executor.submit(
            _process_frame_cpu,
            i,
            frame,
            boxes_np,
            frame_width,
            frame_height,
            pad_pixels,
            apriltag_params,
            detector,
        )
        pending.append(fut)

        # Back-pressure: drain oldest futures before queueing more
        while len(pending) >= max_inflight:
            results_tag.append(pending.popleft().result())
        status_update(i + 1)

    # Drain remaining futures
    while pending:
        results_tag.append(pending.popleft().result())
    executor.shutdown()
    results.close()

    # Build DataFrame
    df = pd.DataFrame(results_tag)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    df = df.set_index(COL_FRAME)
    df.to_pickle(out_pkl)

    total_time = time.time() - start_time
    m, s = divmod(int(total_time), 60)
    logger.info("Total processing time for %s: %dm %02ds", video_path, m, s)

    return df
