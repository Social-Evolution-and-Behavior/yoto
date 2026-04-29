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

## AprilTag Presets

The image-processing and AprilTag-decoder parameters used by both
pipelines can be overridden by a **preset** — a JSON file of parameter
values that are merged on top of the built-in defaults. Unspecified
keys keep their defaults, so presets are non-destructive and trivially
reversible (omit the flag to revert).

```bash
# Built-in preset (looked up in src/yoto/presets/)
yoto detect video.mp4 --fast --apriltag-preset ir

# Or a JSON file on disk (e.g. an Optuna best-params dump)
yoto detect video.mp4 --fast --apriltag-preset /path/to/best_params.json
```

From Python:

```python
run_detection_fast(
    "video.mp4",
    yolo_weights="detect14.engine",
    preset="ir",  # or "/path/to/best_params.json"
)
```

### Preset file format

Two on-disk shapes are accepted:

1. **Flat dict** of parameters (the format used by built-in presets):

   ```json
   {
     "upscale": 1.5,
     "tone_map": "sqrt",
     "decode_sharpening": 11.5,
     "max_hamming": 3
   }
   ```

2. **Optuna-style dump** with a top-level `"params"` key (also typically
   containing `"score"` / `"metrics"`); only the `"params"` sub-dict is
   read.

### Recognised keys

| Key                                                  | Default          | Effect when set                           |
| ---------------------------------------------------- | ---------------- | ----------------------------------------- |
| `invert`                                             | `false`          | Photographic negative before processing   |
| `upscale`, `upscale_interp`                          | `1.0`, `lanczos` | Resize composite before detection         |
| `tone_map`                                           | `none`           | `sqrt` or `log` per-pixel curve           |
| `use_gamma`, `gamma`                                 | `false`, `1.0`   | Gamma correction                          |
| `use_median_blur`, `median_kernel`                   | `false`, `3`     | Median blur (kernel forced odd)           |
| `use_bilateral`, `bilateral_d`/`sigma_color`/`sigma_space` | `false`    | Edge-preserving smoothing                 |
| `use_unsharp`, `kernel_size`, `sigma`, `amount`      | `true`, …        | Unsharp-mask sharpening                   |
| `contrast_method`                                    | `simple`         | `simple` (PIL) or `cv2` contrast          |
| `contrast_factor` (simple) / `cv2_alpha`, `cv2_beta` | …                | Contrast strength                         |
| `decimate`, `blur`, `refine_edges`, `decode_sharpening`, `max_hamming` | …      | Forwarded to the AprilTag detector |

The shipped `ir.json` preset was produced by an Optuna sweep on
infrared-illuminated footage; new presets can be dropped into
`src/yoto/presets/` to be discovered by name.

## How It Works

1. **YOLO detection** — a trained YOLOv8 model locates ant bounding boxes
2. **Cropping** — detected regions are padded and cropped from the frame
3. **Composite strip** — crops are packed side-by-side into one wide image
4. **Image enhancement** — unsharp masking and contrast boosting
5. **AprilTag decoding** — the SEBLab detector runs once on the composite (parameters configurable via a preset; see above)
6. **Reprojection** — tag coordinates are mapped back to the original frame

## Output Format

Both pipelines produce a pandas DataFrame with a MultiIndex:

- **Index**: frame number
- **Columns**: `(tag_id, "center_x")`, `(tag_id, "center_y")`, `(tag_id, "corners")`

The DataFrame is saved as a pickle file alongside the input video.
