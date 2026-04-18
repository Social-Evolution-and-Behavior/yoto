# Detection Pipeline

YOTO provides two detection pipeline variants that produce identical output.

## Simple Pipeline

The portable variant uses Ultralytics' `model.predict(stream=True)`. It
works on any CUDA-capable machine.

```python
from yoto import run_detection_simple

df = run_detection_simple(
    "video.mp4",
    yolo_weights="detect14.engine",
    conf_threshold=0.1,
    pad_pixels=10,
)
```

## Fast Pipeline

The high-throughput variant uses NVDEC hardware decoding and GPU-resident
pre-processing. It requires `PyNvVideoCodec`, `tensorrt`,
`onnxruntime-gpu`, and an NVIDIA GPU with NVDEC.

Weights can be `.pt`, `.engine` / `.trt`, or `.onnx` — the loader picks
the right backend automatically based on the file extension.

```python
from yoto import run_detection_fast

df = run_detection_fast(
    "video.mp4",
    yolo_weights="detect14.engine",
    batch_size=20,
    target_size=1024,
    debug=True,  # prints per-stage profiling
)
```

### TensorRT engine export

For `.engine` / `.trt` weights the fast pipeline loads the engine directly
via the `tensorrt` Python API (bypassing Ultralytics' `AutoBackend`, which
corrupts TRT binding addresses when called outside its `predict` loop and
crashes NVDEC via CUDA-context poisoning).

Export with `dynamic=True` so the engine accepts variable batch sizes:

```bash
yolo export model=detect14.pt format=engine imgsz=1024 \
    half=True batch=20 dynamic=True
```

Without `dynamic=True` the engine has a fixed batch size and will crash
when fed a smaller trailing batch. `.pt` weights still go through the
normal YOLO path.

### ONNX export

For `.onnx` weights the fast pipeline uses ONNX Runtime's
`CUDAExecutionProvider` with `IOBinding`, so the input tensor stays on
GPU (zero-copy via `data_ptr()`). Portable across GPUs, no version lock.

```bash
yolo export model=detect14.pt format=onnx imgsz=1024 \
    half=True dynamic=True simplify=True
```

Slightly slower than a matched TensorRT engine but far more portable — a
good drop-in when you don't want to rebuild a TRT engine per machine.

## How It Works

1. **YOLO detection** — a trained YOLOv8 model locates ant bounding boxes
2. **Cropping** — detected regions are padded and cropped from the frame
3. **Composite strip** — crops are packed side-by-side into one wide image
4. **Image enhancement** — unsharp masking and contrast boosting
5. **AprilTag decoding** — the SEBLab detector runs once on the composite
6. **Reprojection** — tag coordinates are mapped back to the original frame

## Output Format

Both pipelines produce a pandas DataFrame with a MultiIndex:

- **Index**: frame number
- **Columns**: `(tag_id, "center_x")`, `(tag_id, "center_y")`, `(tag_id, "corners")`

The DataFrame is saved as a pickle file alongside the input video.
