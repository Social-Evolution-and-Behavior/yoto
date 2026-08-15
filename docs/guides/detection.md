# Detection Pipeline

`yoto detect` runs the two-stage pipeline: YOLO locates tag regions in each
frame, then AprilTag decodes IDs from the cropped regions. It accepts a single
video, a recording directory, a tree of recordings, or images.

There are two pipeline variants behind one flag. Both produce identical output:

- **Fast** (`--use-nvdec True`, the default) — NVDEC hardware decoding and
  GPU-resident pre-processing. Needs an NVIDIA GPU with NVDEC.
- **Portable** (`--use-nvdec False`) — Ultralytics' `model.predict(stream=True)`.
  Works on any CUDA-capable machine, roughly 30% slower.

## Usage

```bash
# Default: fast pipeline, bundled weights
yoto detect /path/to/experiment.mp4

# Portable pipeline
yoto detect /path/to/experiment.mp4 --use-nvdec False

# Your own weights (.pt, .engine / .trt, or .onnx)
yoto detect /path/to/experiment.mp4 --yoloweights /path/to/yolo.pt

# Apply an AprilTag preset: built-in name, or a JSON file
yoto detect /path/to/experiment.mp4 --apriltag-preset ir

# Decode a different tag family
yoto detect /path/to/experiment.mp4 --tag-family tag25h9

# Tune the YOLO stage
yoto detect /path/to/experiment.mp4 --conf 0.1 --iou 0.4

# Batch a whole tree, 3 videos at a time
yoto detect /path/to/recordings/ --parallel 3

# Re-run only some videos (0-based indices, ranges allowed)
yoto detect /path/to/recordings/ --video-nb 0,2,5-9
```

`--yoloweights` is optional: the default weights ship inside the package.
Override the lookup globally with the `YOTO_WEIGHTS` environment variable.

Outputs go to `tracking/raw_data/` next to the input — see
[Output Format](#output-format).

## Choosing a pipeline

The fast pipeline is the default and needs nothing extra installed —
`pynvvideocodec` is a core dependency. Use `--use-nvdec False` on machines
without NVDEC.

Weights can be `.pt`, `.engine` / `.trt`, or `.onnx`; the loader picks the
backend from the file extension. `.pt` works out of the box. The other two
formats need the `engines` extra (`pip install ".[engines]"`), which adds
`tensorrt` and `onnxruntime-gpu`.

### TensorRT engine export

For `.engine` / `.trt` weights the fast pipeline loads the engine directly via
the `tensorrt` Python API, bypassing Ultralytics' `AutoBackend` — that
corrupts TRT binding addresses when called outside its `predict` loop and
crashes NVDEC via CUDA-context poisoning.

Export with `dynamic=True` so the engine accepts variable batch sizes:

```bash
yolo export model=yolo.pt format=engine imgsz=1024 \
    half=True batch=20 dynamic=True
```

Without `dynamic=True` the engine has a fixed batch size and will crash when
fed a smaller trailing batch.

### ONNX export

For `.onnx` weights the fast pipeline uses ONNX Runtime's
`CUDAExecutionProvider` with `IOBinding`, so the input tensor stays on GPU
(zero-copy via `data_ptr()`).

```bash
yolo export model=yolo.pt format=onnx imgsz=1024 \
    half=True dynamic=True simplify=True
```

Slightly slower than a matched TensorRT engine but far more portable — a good
drop-in when you don't want to rebuild a TRT engine per machine.

## AprilTag Presets

The image-processing and AprilTag-decoder parameters used by both pipelines
can be overridden by a **preset** — a JSON file of parameter values merged on
top of the built-in defaults. Unspecified keys keep their defaults, so presets
are non-destructive and trivially reversible (omit the flag to revert).

```bash
# Built-in preset (looked up in src/yoto/presets/)
yoto detect video.mp4 --apriltag-preset ir

# Or a JSON file on disk (e.g. an Optuna best-params dump)
yoto detect video.mp4 --apriltag-preset /path/to/best_params.json
```

`yoto train optimize-preset` generates these — see
[Training and Tuning](training.md).

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
   containing `"score"` / `"metrics"`); only the `"params"` sub-dict is read.

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

`--no-enhance` skips every enhancement stage, including a preset's, and
decodes the raw grayscale crop. Useful for diagnostics and already-clean
input.

## Image Detection

`yoto detect` accepts a single image file or a directory of images (jpg, jpeg,
png, tif, tiff, bmp) in addition to videos. The same YOLO + AprilTag pipeline
runs; outputs go into `tracking/image_output/` (annotated overlays) and
`tracking/data/` (pickles) next to the input.

```bash
# Single image
yoto detect /path/to/frame.jpg

# Folder of images
yoto detect /path/to/frames/

# Skip YOLO — run AprilTag directly on the full image (useful for testing)
yoto detect /path/to/frame.jpg --no-yolo
```

## Key Flags

| Flag | Default | Effect |
|------|---------|--------|
| `--yoloweights` | bundled `detect34.pt` | YOLO weights. `.pt`, `.engine` / `.trt`, or `.onnx`. Override the default globally with the `YOTO_WEIGHTS` env var. |
| `--use-nvdec` | `True` | Use NVDEC hardware decoding (fast pipeline); `False` falls back to the portable Ultralytics pipeline |
| `--dataname` | `_apriltagDetect14` | Suffix on the output pickle name |
| `--parallel` | unset | Process N videos concurrently via GNU parallel |
| `--video-nb` | all | Restrict to 0-based indices in the resolved video list: `3`, `0,2,5`, `0-9`, `0-4,10,20-25` |
| `--conf` | `0.1` | YOLO confidence threshold |
| `--iou` | `0.4` | YOLO NMS IoU threshold. Lower = more aggressive duplicate suppression |
| `--pad-ratio` | `0.34` | Per-axis padding added around each YOLO box before cropping (scales with tag size). Capped at `pad_ratio × median(box_dim)` across the batch so one unusually large box can't produce an oversized crop. |
| `--tag-offset-filter` | `True` | Drop AprilTag decodes whose center is too far from the source YOLO box center — catches misdecodes in padding regions. The dropped box stays in the undecoded pool for `clean` to pick up via YOLO-fill. |
| `--max-tag-offset-ratio` | `0.6` | Distance threshold for `--tag-offset-filter`, as a fraction of `min(box_w, box_h)` |
| `--tag-family` | `tag36ARTag` | AprilTag family passed to the decoder. Swap to decode a different family (e.g. `tag25h9`, `tag36h11`) without recompiling |
| `--max-tag-id` | family-dependent | Drop any decode whose tag ID exceeds this value. 237 for the ARTag family (to reject IDs outside its valid range), 999 for all other families. |
| `--silence-ids` | none | Drop specific tag IDs unconditionally — for IDs known to be misdecode-prone in your setup. Space- or comma-separated: `--silence-ids 12 45` or `--silence-ids 12,45,67`. |
| `--apriltag-preset` | none | Built-in preset name or path to a JSON file, merged over the defaults |
| `--batch-size` | `2` | Frames per GPU batch for the fast pipeline. Larger batches amortise NVDEC fetch, NMS dispatch and Python overhead but cost more VRAM. Ignored when `--use-nvdec False`. |
| `--save-yolo` | `True` | Write the `_yolo` sidecar (disable if not using YOLO-fill in `clean`) |
| `--save-quads` | `False` | Write the `_quads` sidecar (only needed for `render --quads` debug overlay) |
| `--no-yolo` | off | Skip YOLO entirely and run AprilTag on the full image. Portable pipeline only — `--use-nvdec` is ignored. Takes no value. |
| `--no-enhance` | off | Skip all image enhancement and decode the raw grayscale crop. Takes no value. |
| `--debug` | off | Per-stage profiling output |

## How It Works

1. **YOLO detection** — the trained YOLO model locates ant bounding boxes
2. **Crop layout** — padding bounds are computed for each box (`pad_ratio × box_dim`, capped at `pad_ratio × median(box_dim)` to prevent outliers from inflating crops)
3. **GPU composite** — in the fast pipeline, crops are assembled into a single wide strip directly on GPU (zero CPU copy); the portable pipeline does this on CPU
4. **Image enhancement** — unsharp masking and contrast boosting
5. **AprilTag decoding** — the apriltag detector runs once on the composite (parameters configurable via a preset; see above)
6. **Reprojection + offset filter** — tag coordinates are mapped back to the original frame; decodes whose center is farther than `max_tag_offset_ratio × min(box_w, box_h)` from the source YOLO box center are dropped. Disable with `--tag-offset-filter False`.

## Output Format

Both pipelines produce a long-format pandas DataFrame — one row per decoded
detection, indexed by frame number:

| Column | Meaning |
|--------|---------|
| `tag_id` | Decoded AprilTag ID |
| `center_x`, `center_y` | Tag centre in original-frame pixels |
| `c0x, c0y … c3x, c3y` | The four corners in `lb-rb-rt-lt` order, float32 |

The DataFrame is saved as `<stem><dataname>.pkl.zst` in `tracking/raw_data/`.

Two sidecars are written next to the main pickle:

| File | Default | Content |
|------|---------|---------|
| `<stem>_yolo.pkl.zst` | always (disable with `--save-yolo False`) | Every YOLO box per frame with columns `box_x1/y1/x2/y2`, `center_x/y`, `confidence`, `decoded`, `tag_id` (`-1` when undecoded). Used by `yoto clean --yolo-fill True` and `yoto render --undecoded`. |
| `<stem>_quads.pkl.zst` | off by default (`--save-quads True` to enable) | Raw AprilTag quads for every quad found, decoded or not, in the same corner columns. Used by `yoto render --quads` for debug overlays. |

### Reading corners

Read corners with [`load_corners`](../api/io.md) rather than indexing the
eight columns yourself. It returns an `(n, 4, 2)` array:

```python
import pandas as pd
from yoto import load_corners

raw = pd.read_pickle("tracking/raw_data/000000_myrun.pkl.zst")
quads = load_corners(raw)          # (n_detections, 4, 2)
```

### Compression

Pipeline pickles are zstd-compressed. pandas infers the codec from the `.zst`
extension, so `pd.read_pickle(path)` needs no extra argument. Pickles written
by earlier yoto versions still load unchanged.

## Python API

Both variants are importable. They take the same parameters as the CLI flags
and return the DataFrame described above.

```python
from yoto import run_detection_fast, run_detection_simple

# Fast pipeline (NVDEC + GPU-resident preprocessing)
df = run_detection_fast(
    "video.mp4",
    batch_size=2,
    target_size=1024,
    preset="ir",       # or "/path/to/best_params.json"
    debug=True,        # per-stage profiling
)

# Portable pipeline
df = run_detection_simple(
    "video.mp4",
    yolo_weights="yolo.pt",
    conf_threshold=0.1,
    pad_ratio=0.34,
)
```

See the [Detection API reference](../api/detection.md) for the full
signatures.
