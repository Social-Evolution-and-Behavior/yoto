# YOTO

**GPU-accelerated AprilTag tracking pipeline for ant behavioral studies.**

YOTO provides a complete workflow for detecting AprilTag barcodes on ants
using YOLO object detection, with tools for data cleaning, interpolation,
and video overlay visualization.

## Features

- **Two detection pipelines**: a portable simple mode and a high-throughput
  NVDEC-accelerated fast mode
- **Automated data cleaning**: removes spurious IDs, fills short tracking
  gaps via interpolation, detects and deletes unrealistic jumps
- **Video overlay rendering**: produces annotated videos with per-tag
  labels, coloured motion trails, and frame counters
- **Batch processing**: bash script to process entire experiment folders
  with structured output directories
- **CLI and Python API**: use from the command line or import into your
  own analysis scripts

## Quick Example

```python
from yoto import run_detection_simple, clean_tracking_data, render_overlay_video

# Step 1: Detect tags
df = run_detection_simple("experiment.mp4", yolo_weights="detect14.engine")

# Step 2: Clean and interpolate
cleaned, ids, metrics = clean_tracking_data(df)
print(f"Error rate: {metrics['error_pct']:.2f}%")

# Step 3: Render overlay video
render_overlay_video("experiment.mp4", cleaned, ids)
```
