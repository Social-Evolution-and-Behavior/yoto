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
yoto clean  /path/to/recordings/
yoto render /path/to/recordings/ --parallel 3
```

This creates a standard `tracking/` directory next to each video:

```
<recording>/tracking/
├── raw_data/       # detect outputs
├── clean_data/     # clean outputs
├── video_output/   # render outputs
└── logs/           # parallel worker logs
```

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
