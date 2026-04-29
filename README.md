[![CI](https://github.com/Social-Evolution-and-Behavior/yoto/actions/workflows/ci.yml/badge.svg)](https://github.com/Social-Evolution-and-Behavior/yoto/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# YOTO

GPU-accelerated AprilTag tracking pipeline for ant behavioral studies.

YOTO is a hybrid pipeline for fast and reliable tracking of fiducial-marked insects. Running AprilTag directly on a high-resolution video frame is slow, because the decoder scans the whole image looking for quads. YOTO splits the work into two stages and only runs the expensive decoder on the parts of the frame that matter:

1. **Stage 1 — YOLO detection.** A lightweight YOLO model trained on a small (auto-generated) dataset locates marker regions in the full frame. YOLO is fast and accurate at finding small tags in large images.
2. **Stage 2 — AprilTag decoding.** Detected regions are cropped and packed into a compact composite, and the AprilTag library decodes tag IDs from that composite. Restricting AprilTag to small cropped regions preserves its decoding accuracy while drastically reducing its cost.

In a reference benchmark, 7 days of 10 fps 4512×4512 footage of 100 clonal raider ants was processed in under 9 hours — roughly a week's worth of compute on a classical full-frame AprilTag pipeline.

## Features

- **Two-stage detection** (YOLO → AprilTag): an order-of-magnitude faster than running AprilTag on full frames, with fewer false positives in busy scenes
- **End-to-end pipeline**: `detect → clean → render` — raw decode, automatic gap-filling and jump-detection on trajectories, and overlay-video rendering
- **GPU end-to-end**: NVDEC decode → TensorRT / ONNX YOLO inference → AprilTag → NVENC encode; threaded ffmpeg render ~2–3× faster than single-threaded `cv2.VideoWriter`
- **AprilTag presets**: swap detection / image-processing parameters non-destructively via `--apriltag-preset` (built-in `ir` preset, plus support for any JSON file including Optuna `best_params*.json` dumps)
- **Two detection backends**: portable simple mode (any CUDA box) and NVDEC-accelerated fast mode for production runs
- **Batch processing**: `--parallel N` on `detect` / `render` dispatches one worker per video via GNU parallel, with a live human-readable `progress.txt`
- **CLI and Python API**: use from the command line or import into your analysis scripts

## Installation

`pyproject.toml` is the single source of truth. Use whichever Python ≥ 3.10 environment you prefer.

### With conda (recommended)

```bash
conda create -n yoto python=3.11 -y
conda activate yoto
git clone https://github.com/Social-Evolution-and-Behavior/yoto.git
cd yoto
pip install -e ".[dev,fast]"     # add ,docs,lint if needed
pre-commit install
```

### Plain pip / venv

```bash
git clone https://github.com/Social-Evolution-and-Behavior/yoto.git
cd yoto
pip install -e ".[dev,fast]"
pre-commit install
```

The `fast` extra pulls in `torch`, `pynvvideocodec`, `tensorrt`, and `onnxruntime-gpu` for the NVDEC detection pipeline. `.pt` weights work without any of these — the extras are only needed for `.engine`/`.trt`/`.onnx` weights.

## Quick Example

```bash
# Step 1: Detect tags (NVDEC fast pipeline + TensorRT engine)
yoto detect /path/to/experiment.mp4 --yoloweights /path/to/detect14.engine --fast

# Step 2: Clean and interpolate the raw tracking data
yoto clean /path/to/experiment.mp4

# Step 3: Render the overlay video
yoto render /path/to/experiment.mp4
```

## Command Line

`detect`, `clean`, and `render` all accept a single video file, a recording directory, or a tree of recordings.

```bash
# Detection
yoto detect /path/to/experiment.mp4 --fast
yoto detect /path/to/experiment.mp4 --fast --apriltag-preset ir   # IR-illuminated footage
yoto detect /path/to/recordings/ --parallel 3                     # batch over a tree

# Cleaning
yoto clean /path/to/experiment.mp4
yoto clean /path/to/recordings/

# Overlay rendering
yoto render /path/to/experiment.mp4
yoto render /path/to/recordings/ --scale 0.5 --parallel 3
yoto render /path/to/recordings/ --raw --no-trails                # raw, uninterpolated
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

## Documentation

Full guides live under `docs/guides/` (detection, cleaning, video rendering, installation, quickstart). They render as plain Markdown on GitHub, but the project also ships an MkDocs site:

```bash
pip install -e ".[docs]"   # one-time
mkdocs serve               # live preview at http://127.0.0.1:8000
mkdocs build               # static site under ./site/
```

## Requirements

- Python ≥ 3.10
- NVIDIA GPU with CUDA support
- ffmpeg — built with NVENC encoders (`hevc_nvenc`, `h264_nvenc`) and `-hwaccel cuda` for the GPU render fast path; `libx264` + OpenCV decode are used as automatic fallbacks
- [AprilTag](https://github.com/Social-Evolution-and-Behavior/apriltag) (LSEB fork)

## License

[MIT](LICENSE)
