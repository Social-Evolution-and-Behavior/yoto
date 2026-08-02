<p align="center">
  <img src="assets/YOTO-LOGO.png#only-light" alt="YOTO logo" width="260">
  <img src="assets/YOTO-LOGO-W.png#only-dark" alt="YOTO logo" width="260">
</p>

# YOTO

**GPU-accelerated AprilTag tracking pipeline for ant behavioral studies.**

YOTO is a hybrid pipeline for fast and reliable tracking of fiducial-marked insects. Running AprilTag directly on a high-resolution video frame is slow, because the decoder scans the whole image looking for quads. YOTO splits the work into two stages and only runs the expensive decoder on the parts of the frame that matter:

1. **Stage 1 — YOLO detection.** A lightweight YOLO model trained on a small (auto-generated) dataset locates marker regions in the full frame. YOLO is fast and accurate at finding small tags in large images.
2. **Stage 2 — AprilTag decoding.** Detected regions are cropped and packed into a compact composite, and the AprilTag library decodes tag IDs from that composite. Restricting AprilTag to small cropped regions preserves its decoding accuracy while drastically reducing its cost.

In a reference benchmark, 7 days of 10 fps 4512×4512 footage of 100 clonal raider ants was processed in under 9 hours — roughly a week's worth of compute on a classical full-frame AprilTag pipeline.

## Features

- **Two-stage detection** (YOLO → AprilTag): an order-of-magnitude faster than running AprilTag on full frames, with fewer false positives in busy scenes
- **End-to-end pipeline**: `detect → clean → render` — raw decode, automatic gap-filling and jump-detection on trajectories, and overlay-video rendering
- **GPU end-to-end**: NVDEC decode → TensorRT / ONNX YOLO inference → GPU composite → AprilTag → NVENC encode; threaded ffmpeg render ~2–3× faster than single-threaded `cv2.VideoWriter`
- **Image detection**: `yoto detect` accepts a single image or a folder of images in addition to videos — same YOLO + AprilTag pipeline, outputs to `tracking/`
- **AprilTag presets**: swap detection / image-processing parameters non-destructively via `--apriltag-preset` (built-in `ir` preset, plus support for any JSON file including Optuna `best_params*.json` dumps)
- **Two detection backends**: portable simple mode (any CUDA box) and NVDEC-accelerated fast mode for production runs
- **Configurable tag family**: `yoto detect --tag-family` swaps the AprilTag family (default `tag36ARTag`) without recompiling the C decoder
- **Tag ID filtering**: `--max-tag-id` drops out-of-family IDs (default 237 for ARTag); `--silence-ids` blacklists specific misdecode-prone IDs
- **YOLO-fill gap recovery**: undecoded YOLO boxes are stitched into each tag's trajectory via a forward+backward chain matcher with collision resolution — recovers tag positions on frames where AprilTag failed to decode
- **Auto px→mm calibration**: `yoto clean --tag-size` measures the median AprilTag side length from decoded corners and stores a `mm_per_px` scale on the clean pickle
- **Self-describing output**: every clean pickle carries a `df.attrs` block with the YOTO version, all knobs used, per-step counts, and the imaging scale — see [Pickle Attributes](guides/cleaning.md#pickle-attributes)
- **Tag highlighting in overlays**: `yoto render --highlight-ids 42,87` calls out specific tags with a bold colored label
- **Batch processing**: `--parallel N` on `detect` / `render` dispatches one worker per video via GNU parallel, with a live human-readable `progress.txt`
- **CLI and Python API**: use from the command line or import into your analysis scripts

## Quick Example

```bash
# Step 1: Detect tags (NVDEC fast pipeline is the default)
yoto detect /path/to/recording --yoloweights /path/to/yolo.pt --parallel 5

# Step 2: Clean and interpolate the raw tracking data (auto px→mm calibration)
yoto clean /path/to/recording --tag-size 0.4

# Step 3: Render the overlay video, highlighting two tags of interest
yoto render /path/to/recording --short --scale 0.5 --highlight-ids 42,87
```

You can also point any of the commands at a recording directory or a tree of recordings; see the [Quickstart](guides/quickstart.md) for details.
