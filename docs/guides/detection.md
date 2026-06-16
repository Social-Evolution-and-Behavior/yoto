# Detection Pipeline

YOTO provides two detection pipeline variants that produce identical output.

## Simple Pipeline

The portable variant uses Ultralytics' `model.predict(stream=True)`. It
works on any CUDA-capable machine.

```python
from yoto import run_detection_simple

df = run_detection_simple(
    "video.mp4",
    yolo_weights="yolo.pt",
    conf_threshold=0.1,
    pad_ratio=0.34,
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
    yolo_weights="yolo.pt",
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
yolo export model=yolo.pt format=engine imgsz=1024 \
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
yolo export model=yolo.pt format=onnx imgsz=1024 \
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
yoto detect video.mp4 --yoloweights yolo.pt --apriltag-preset ir

# Or a JSON file on disk (e.g. an Optuna best-params dump)
yoto detect video.mp4 --yoloweights yolo.pt --apriltag-preset /path/to/best_params.json
```

From Python:

```python
run_detection_fast(
    "video.mp4",
    yolo_weights="yolo.pt",
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

## Key Flags

| Flag | Default | Effect |
|------|---------|--------|
| `--pad-ratio` | `0.34` | Per-axis padding added around each YOLO box before cropping (scales with tag size) |
| `--tag-offset-filter` | `True` | Drop AprilTag decodes whose center is too far from the source YOLO box center — catches misdecodes in padding regions |
| `--max-tag-offset-ratio` | `0.6` | Distance threshold for `--tag-offset-filter`, as a fraction of `min(box_w, box_h)` |
| `--tag-family` | `tag36ARTag` | AprilTag family passed to the decoder. Swap to decode a different family (e.g. `tag25h9`, `tag36h11`) without recompiling |
| `--save-yolo` | `True` | Write the `_yolo.pkl` sidecar (disable if not using YOLO-fill in clean) |
| `--save-quads` | `False` | Write the `_quads.pkl` sidecar (only needed for `render --quads` debug overlay) |
| `--use-nvdec` | `True` | Use NVDEC hardware decoding (fast pipeline); `False` falls back to the portable Ultralytics pipeline |

## How It Works

1. **YOLO detection** — a trained YOLOv8 model locates ant bounding boxes
2. **Cropping** — detected regions are padded by `pad_ratio × box_dim` on each side (scales with apparent tag size) and cropped from the frame
3. **Composite strip** — crops are packed side-by-side into one wide image
4. **Image enhancement** — unsharp masking and contrast boosting
5. **AprilTag decoding** — the SEBLab detector runs once on the composite (parameters configurable via a preset; see above)
6. **Reprojection + offset filter** — tag coordinates are mapped back to the original frame; decodes whose center is farther than `max_tag_offset_ratio × min(box_w, box_h)` from the source YOLO box center are dropped (catches misdecodes in padding regions). Disable with `--tag-offset-filter False`.

## Output Format

Both pipelines produce a pandas DataFrame with a MultiIndex:

- **Index**: frame number
- **Columns**: `(tag_id, "center_x")`, `(tag_id, "center_y")`, `(tag_id, "corners")`

The DataFrame is saved as `<stem><dataname>.pkl` in `tracking/raw_data/`.

Two optional sidecars are written next to the main pickle:

| File | Default | Content |
|------|---------|---------|
| `<stem>_yolo.pkl` | always (disable with `--save-yolo False`) | Every YOLO box per frame with columns `box_x1/y1/x2/y2`, `center_x/y`, `confidence`, `decoded`, `tag_id` (`-1` when undecoded). Used by `yoto clean --yolo-fill True` and `yoto render --undecoded`. |
| `<stem>_quads.pkl` | off by default (`--save-quads True` to enable) | Raw AprilTag quads (4×2 corner arrays) for every quad found, decoded or not. Used by `yoto render --quads` for debug overlays. |
