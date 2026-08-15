# Quickstart

## Command-Line Usage

`detect`, `clean`, and `render` all accept a single video file, a recording directory, or a tree of recordings.

### 1. Detect tags

```bash
# Fast (NVDEC + TensorRT) pipeline — default
yoto detect /path/to/experiment.mp4 --yoloweights /path/to/yolo.pt

# Portable pipeline (any CUDA machine, no NVDEC required)
yoto detect /path/to/experiment.mp4 --yoloweights /path/to/yolo.pt --use-nvdec False

# Apply an AprilTag preset (built-in name, or path to a JSON file)
yoto detect /path/to/experiment.mp4 --yoloweights /path/to/yolo.pt --apriltag-preset ir
yoto detect /path/to/experiment.mp4 --yoloweights /path/to/yolo.pt --apriltag-preset /path/to/best_params.json

# Decode a different AprilTag family (default is tag36ARTag)
yoto detect /path/to/experiment.mp4 --yoloweights /path/to/yolo.pt --tag-family tag25h9

# Drop IDs outside the valid family range, or silence specific known-misdecode IDs
yoto detect /path/to/experiment.mp4 --yoloweights /path/to/yolo.pt --max-tag-id 200
yoto detect /path/to/experiment.mp4 --yoloweights /path/to/yolo.pt --silence-ids 12,45

# Tune the YOLO stage: confidence threshold and NMS IoU
yoto detect /path/to/experiment.mp4 --yoloweights /path/to/yolo.pt --conf 0.1 --iou 0.4

# Run on a single image or folder of images (same pipeline, outputs to tracking/)
yoto detect /path/to/frame.jpg --yoloweights /path/to/yolo.pt
yoto detect /path/to/frames/  --yoloweights /path/to/yolo.pt
```

See [Detection → AprilTag Presets](detection.md#apriltag-presets) for the
full list of recognised keys and how to add your own preset.

### 2. Clean the raw tracking data

```bash
# Pass either the video, the raw pickle, or the recording directory.
yoto clean /path/to/experiment.mp4
yoto clean /path/to/recordings/

# Compute and store an mm/px scale (one tag border = 0.4 mm by default).
yoto clean /path/to/experiment.mp4 --tag-size 0.4
```

The clean pickle's `df.attrs["yoto_mm_per_px"]` lets you convert pixel
distances to millimetres. See
[Cleaning → Pickle Attributes](cleaning.md#pickle-attributes) for the
full list of metadata stamped on the output.

### 3. Render the overlay video

```bash
yoto render /path/to/experiment.mp4
yoto render /path/to/experiment.mp4 --scale 0.5 --codec h264

# Highlight specific tags (bold red labels). Space- or comma-separated.
yoto render /path/to/experiment.mp4 --highlight-ids 42,87,103
yoto render /path/to/experiment.mp4 --highlight-ids 42 --highlight-color yellow
```

### 4. Batch process an entire experiment

```bash
# Process every video under /path/to/recordings/ in parallel (3 workers)
yoto detect /path/to/recordings/ --yoloweights /path/to/yolo.pt --parallel 3
yoto clean  /path/to/recordings/ --parallel 3
yoto render /path/to/recordings/ --parallel 3
```

To re-run only some videos of a recording — after a few workers failed, or to
test settings on one video first — use `--video-nb`. It takes 0-based indices
into the resolved video list, as a single index, a comma list, and/or inclusive
ranges, and works on `detect`, `clean`, `render` and `train build-testset`:

```bash
yoto detect /path/to/recordings/ --yoloweights /path/to/yolo.pt --video-nb 0
yoto clean  /path/to/recordings/ --video-nb 0,2,5
yoto render /path/to/recordings/ --video-nb 0-4,10
```

This creates a standard `tracking/` directory next to each video:

```
<recording>/tracking/
├── raw_data/       # detect outputs
├── clean_data/     # clean outputs
├── video_output/   # render outputs
├── training/       # `yoto train` testsets and datasets
└── logs/           # parallel worker logs
```

### 5. Retrain the models (optional)

When decoding degrades on new footage, `yoto train` rebuilds the two things
`detect` depends on — the AprilTag preset and the YOLO weights:

```bash
yoto train build-testset /path/to/recordings/ --dataname myrun
yoto train optimize-preset --testset-dir <recording>/tracking/training/apriltag_testset
yoto train build-yolo-dataset /path/to/recordings/
```

See [Training and Tuning](training.md) for the full workflow.

## Python API

The same three stages are available as importable functions:

```python
from yoto import run_detection_simple, clean_tracking_data, render_overlay_video

df = run_detection_simple(
    "/path/to/experiment.mp4",
    yolo_weights="/path/to/yolo.pt",
)
cleaned, ids, metrics = clean_tracking_data(df)
render_overlay_video("/path/to/experiment.mp4", cleaned, ids, scale=0.5)
```

### Loading a whole experiment

After `clean`, use [`load_data`](../api/io.md) to load an entire experiment's
clean pickles as one DataFrame — no need to read and concatenate each
`*_clean.pkl` by hand:

```python
from yoto import load_data

# A recording folder -> every *_clean.pkl under tracking/clean_data/, concatenated.
df = load_data("/path/to/recordings/")

# ...or a single video's clean pickle.
df = load_data("/path/to/experiment.mp4")

# ...or the Nth video of a recording, by position.
df = load_data("/path/to/recordings/", dataname="myrun", video_nb=0)
```

Rows carry a `(frame, source, video_frame)` MultiIndex (`frame` is a global
monotonic counter, `source` is the video stem, `video_frame` is the original
per-video frame number); the original `(tag_id, metric)` columns are preserved.
Merged metadata is attached to `df.attrs`, including `df.attrs["scale"]` — the
median mm/px across all loaded pickles. Pass `dataname=` if you used a
`--dataname` suffix at detect/clean time.

Corner coordinates are eight `float32` metrics per tag (`c0x` … `c3y`). Read
them with [`load_corners`](../api/io.md), which returns an `(n, 4, 2)` array
and accepts pickles from any yoto version:

```python
from yoto import load_corners

quads = load_corners(df, tag_id="42")   # (n_frames, 4, 2)
```

They roughly halve a loaded frame, so pass `corners=False` when you only need
trajectories — a whole recording rarely fits in RAM otherwise:

```python
df = load_data(folder, dataname="myrun", corners=False)
```
