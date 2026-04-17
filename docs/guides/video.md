# Video Overlay

Render annotated videos showing tag labels and motion trails.

## Usage

```python
from yoto.video import render_overlay_video

render_overlay_video(
    "input.mp4",
    cleaned_dataframe,
    id_list,
    scale=0.5,       # half-resolution output
    short=True,      # first 2000 frames only
)
```

## Features

- **Per-tag coloured trails** with thickness ramp (thin = old, thick = recent)
- **Tag ID labels** drawn at each detection position
- **Frame counter** and visible-tag count overlay
- **GPU-accelerated encoding** via NVENC (automatic CPU fallback)
- **Scalable output** — reduce resolution for quick previews

## Encoder Priority

The renderer tries encoders in this order:

1. `hevc_nvenc` (GPU, no 4096-width limit)
2. `h264_nvenc` (GPU)
3. `libx264 ultrafast` (CPU fallback)
