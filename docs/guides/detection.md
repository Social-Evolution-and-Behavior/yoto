# Detection Pipeline

YOTO provides two detection pipeline variants that produce identical output.

## Simple Pipeline

The portable variant uses Ultralytics' `model.predict(stream=True)`. It
works on any CUDA-capable machine.

```python
from yoto import run_detection_simple

df = run_detection_simple(
    "video.mp4",
    yolo_weights="detect14.engine",
    conf_threshold=0.1,
    pad_pixels=10,
)
```

## Fast Pipeline

The high-throughput variant uses NVDEC hardware decoding and GPU-resident
pre-processing. It requires `PyNvVideoCodec` and an NVIDIA GPU with NVDEC.

```python
from yoto import run_detection_fast

df = run_detection_fast(
    "video.mp4",
    yolo_weights="detect14.engine",
    batch_size=20,
    target_size=1024,
    debug=True,  # prints per-stage profiling
)
```

## How It Works

1. **YOLO detection** — a trained YOLOv8 model locates ant bounding boxes
2. **Cropping** — detected regions are padded and cropped from the frame
3. **Composite strip** — crops are packed side-by-side into one wide image
4. **Image enhancement** — unsharp masking and contrast boosting
5. **AprilTag decoding** — the SEBLab detector runs once on the composite
6. **Reprojection** — tag coordinates are mapped back to the original frame

## Output Format

Both pipelines produce a pandas DataFrame with a MultiIndex:

- **Index**: frame number
- **Columns**: `(tag_id, "center_x")`, `(tag_id, "center_y")`, `(tag_id, "corners")`

The DataFrame is saved as a pickle file alongside the input video.
