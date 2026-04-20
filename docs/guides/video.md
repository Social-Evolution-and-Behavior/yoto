# Video Overlay

Render annotated videos showing tag labels and motion trails.

## Usage

```python
from yoto.video import render_overlay_video

render_overlay_video(
    "input.mp4",
    cleaned_dataframe,
    id_list,
    scale=0.5,        # half-resolution output
    short=True,       # first 2000 frames only
    codec="auto",     # "auto" (HEVC first), "hevc", or "h264"
)
```

From the CLI:

```bash
yoto render experiment.mp4 --scale 0.5 --codec h264
```

## Features

- **Per-tag coloured trails** with thickness ramp (thin = old, thick = recent)
- **Tag ID labels** drawn at each detection position
- **Frame counter** and visible-tag count overlay
- **3-thread pipeline** (ffmpeg decode → draw → ffmpeg encode) with
  YUV420 pipes — ~2-3× faster than a single-threaded
  `cv2.VideoWriter` loop
- **GPU-accelerated** decode (`-hwaccel cuda`) and encode
  (`hevc_nvenc` / `h264_nvenc`), with automatic CPU fallbacks
- **Scalable output** — reduce resolution for quick previews

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
