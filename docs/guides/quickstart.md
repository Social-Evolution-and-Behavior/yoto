# Quickstart

## Command-Line Usage

### 1. Detect tags in a video

```bash
# Simple (portable) pipeline
yoto detect experiment.mp4

# Fast (NVDEC) pipeline
yoto detect experiment.mp4 --fast

# Custom weights and output directory
yoto detect experiment.mp4 output/ --yoloweights best.engine --fast

# Apply an AprilTag preset (built-in name, or path to a JSON file)
yoto detect experiment.mp4 --fast --apriltag-preset ir
yoto detect experiment.mp4 --fast --apriltag-preset /path/to/best_params.json
```

See [Detection → AprilTag Presets](detection.md#apriltag-presets) for the
full list of recognised keys and how to add your own preset.

### 2. Clean the raw tracking data

```bash
yoto clean output/experiment_apriltagDetect14.pkl
```

### 3. Render overlay video

```bash
yoto render experiment.mp4
```

### 4. Batch process an entire experiment

```bash
# Process all videos in all sub-folders of /data/experiments/
bash scripts/batch_track.sh /data/experiments/ --fast
```

This creates a `tracking/` directory in each sub-folder:

```
sub-folder/tracking/
├── raw/            # raw detection pickles
├── clean/          # cleaned pickles
├── video_output/   # overlay videos
└── logs/           # per-video logs
```

## Python API

```python
from yoto import run_detection_simple, clean_tracking_data, render_overlay_video

# Detect
df = run_detection_simple("video.mp4", yolo_weights="detect14.engine")

# Clean
cleaned, ids, metrics = clean_tracking_data(df)

# Render
render_overlay_video("video.mp4", cleaned, ids, scale=0.5)
```
