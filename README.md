[![CI](https://github.com/ychemtob/yoto/actions/workflows/ci.yml/badge.svg)](https://github.com/ychemtob/yoto/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# YOTO

GPU-accelerated AprilTag tracking pipeline for ant behavioral studies.

## Features

- **Two detection pipelines**: portable simple mode and high-throughput NVDEC-accelerated fast mode
- **Automated data cleaning**: removes spurious IDs, fills short gaps via interpolation, deletes tracking jumps
- **Video overlay rendering**: annotated videos with per-tag labels, coloured motion trails, and frame counters
- **Batch processing**: `--parallel N` on `detect` and `render` dispatches one worker per video via GNU parallel, with a live human-readable `progress.txt`
- **CLI and Python API**: use from the command line or import into your analysis scripts

## Installation

`pyproject.toml` is the single source of truth. Inside any Python ≥ 3.10 env:

```bash
git clone https://github.com/Social-Evolution-and-Behavior/yoto.git
cd yoto
pip install -e ".[dev,fast]"     # add ,docs,lint if needed
pre-commit install
```

### With conda

```bash
conda create -n yoto python=3.11 -y
conda activate yoto
pip install -e ".[dev,fast]"
```

The `fast` extra pulls in `torch` and `pynvvideocodec` for the NVDEC pipeline.

## Quick Example

```python
from yoto import run_detection_simple, clean_tracking_data, render_overlay_video

# Detect AprilTags in a video
df = run_detection_simple("experiment.mp4", yolo_weights="detect14.engine")

# Clean and interpolate the tracking data
cleaned, ids, metrics = clean_tracking_data(df)
print(f"Error rate: {metrics['error_pct']:.2f}%")

# Render an overlay video
render_overlay_video("experiment.mp4", cleaned, ids, scale=0.5)
```

## Command Line

```bash
# Detect tags (single file, directory of videos, or directory of recording sub-folders)
yoto detect experiment.mp4 --fast
yoto detect /data/recordings/ --parallel 3

# Clean raw data
yoto clean raw_output.pkl
yoto clean /data/recordings/

# Render overlay
yoto render experiment.mp4
yoto render /data/recordings/ --scale 0.5 --parallel 3
yoto render /data/recordings/ --raw --no-trails      # raw, uninterpolated
```

Outputs land in a standard layout next to each video:

```
<recording>/tracking/raw_data/       # detect outputs
<recording>/tracking/clean_data/     # clean outputs
<recording>/tracking/video_output/   # render outputs
<recording>/tracking/logs/           # parallel worker logs
```

### Checking parallel progress

Each parallel run creates a dated folder under `tracking/logs/`:

```
<input_root>/tracking/logs/yoto-parallel-<timestamp>/
├── progress.txt     # human-readable summary, refreshed every 3s
├── joblog.tsv       # GNU parallel joblog (one line per finished video)
└── <seq>-<stem>/    # per-worker stdout/stderr, one dir per video
```

While the run is live:

```bash
cat /path/to/progress.txt          # snapshot
watch -n 5 cat /path/to/progress.txt  # poll every 5s
```

The launch banner prints the exact path.

## Requirements

- Python ≥ 3.10
- NVIDIA GPU with CUDA support
- ffmpeg (for video rendering)
- [AprilTag](https://github.com/Social-Evolution-and-Behavior/apriltag) (LSEB fork)

## License

[MIT](LICENSE)
