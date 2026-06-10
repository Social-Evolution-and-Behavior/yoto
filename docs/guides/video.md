# Video Overlay

Render annotated videos showing tag labels and motion trails.

## Usage

### Command line

```bash
# Single video — pickle is auto-resolved from <recording>/tracking/clean_data/
yoto render /path/to/experiment.mp4

# Half-resolution H.264 preview
yoto render /path/to/experiment.mp4 --scale 0.5 --codec h264

# Batch over a directory tree, 3 workers in parallel
yoto render /path/to/recordings/ --parallel 3

# Render the raw (uninterpolated) detections
yoto render /path/to/experiment.mp4 --raw --no-trails

# Highlight specific tags (bold red labels by default).
# IDs can be space- or comma-separated; both forms can be mixed.
yoto render /path/to/experiment.mp4 --highlight-ids 42,87,103
yoto render /path/to/experiment.mp4 --highlight-ids 42 87
yoto render /path/to/experiment.mp4 --highlight-ids 42 --highlight-color yellow
yoto render /path/to/experiment.mp4 --highlight-ids 42 --highlight-bold False
```

### Python API

```python
from yoto.video import render_overlay_video

render_overlay_video(
    "/path/to/experiment.mp4",
    cleaned_dataframe,
    id_list,
    scale=0.5,                  # half-resolution output
    short=True,                 # first 2000 frames only
    codec="auto",               # "auto" (HEVC first), "hevc", or "h264"
    highlight_ids={42, 87},     # set / list / iterable of tag IDs
    highlight_color=(0, 0, 255),  # BGR triple (default: red)
    highlight_bold=True,        # thicker label stroke for highlighted tags
)
```

## Features

- **Per-tag coloured trails** with thickness ramp (thin = old, thick = recent)
- **Tag ID labels** drawn at each detection position
- **Tag highlighting** — `--highlight-ids` redraws a subset of labels with a configurable color and an optional bold stroke, useful for calling out individuals of interest in a busy colony
- **Frame counter** and visible-tag count overlay
- **3-thread pipeline** (ffmpeg decode → draw → ffmpeg encode) with
  YUV420 pipes — ~2-3× faster than a single-threaded
  `cv2.VideoWriter` loop
- **GPU-accelerated** decode (`-hwaccel cuda`) and encode
  (`hevc_nvenc` / `h264_nvenc`), with automatic CPU fallbacks
- **Scalable output** — reduce resolution for quick previews

## Highlight Flags

| Flag | Default | Effect |
|------|---------|--------|
| `--highlight-ids` | *(none)* | Tag IDs to highlight. Space- or comma-separated, mixable: `--highlight-ids 42 87`, `--highlight-ids 42,87,103`, `--highlight-ids 42,87 103` |
| `--highlight-color` | `red` | Label color for highlighted tags. Named (`red`, `green`, `blue`, `yellow`, `cyan`, `magenta`, `white`, `black`) or `R,G,B` triple in 0–255 (e.g. `255,128,0`) |
| `--highlight-bold` | `True` | Draw highlighted labels with a thicker stroke (~1.6× the regular weight). Set `False` to keep normal weight and change only the color |

## Architecture

```
reader thread          main thread              writer thread
(ffmpeg CUDA decode)   (YUV↔BGR + drawing)      (pipe I/O, GIL-free)
         │                      │                       │
         └── YUV420 ndarray ────▶ cv2 overlays ────────▶ ffmpeg encoder
```

YUV420 halves pipe bandwidth vs BGR24. Conversions are pinned to the
main thread so the writer's pipe write fully overlaps with drawing.

## Codec Selection

The `codec` argument (CLI: `--codec`) picks the encoder family:

| Value   | Candidate order                              | Notes                               |
|---------|----------------------------------------------|-------------------------------------|
| `auto`  | `hevc_nvenc` → `h264_nvenc` → `libx264`      | Default. Smallest files.            |
| `hevc`  | `hevc_nvenc` → `libx264`                     | Same as auto minus h264_nvenc.      |
| `h264`  | `h264_nvenc` → `libx264`                     | Smooth VLC playback; ≤4096 px wide. |

Each candidate is validated with a short burst of blank frames before
the real render starts — NVENC device-capability failures manifest
only after the first buffered frame, so a single-frame probe would let
a broken encoder through.

## Requirements

ffmpeg must be installed with NVENC encoders and `-hwaccel cuda` for
the GPU fast path. Without them the renderer transparently falls back
to `libx264` for encode and OpenCV for decode.
