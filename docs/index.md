# YOTO

**GPU-accelerated AprilTag tracking pipeline for ant behavioral studies.**

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

## Quick Example

```bash
# Step 1: Detect tags (NVDEC fast pipeline + TensorRT engine)
yoto detect /path/to/recording --yoloweights models/detect14.pt --fast --parallel 5

# Step 2: Clean and interpolate the raw tracking data
yoto clean /path/to/recording

# Step 3: Render the overlay video
yoto render /path/to/recording --short --scale 0.5
```

You can also point any of the commands at a recording directory or a tree of recordings; see the [Quickstart](guides/quickstart.md) for details.
