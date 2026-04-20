# Installation

## pip (recommended)

```bash
pip install yoto
```

For the fast NVDEC detection pipeline (requires NVIDIA GPU):

```bash
pip install "yoto[fast]"
```

For development:

```bash
git clone https://github.com/Social-Evolution-and-Behavior/yoto.git
cd yoto
pip install -e ".[dev,docs,lint]"
pre-commit install
```

## conda

```bash
conda create -n yoto python=3.11 -y
conda activate yoto
pip install -e ".[dev,fast]"
```

## Requirements

- Python >= 3.10
- NVIDIA GPU with CUDA support (for YOLO inference)
- ffmpeg — **built with NVENC encoders and CUDA hwaccel** for the fast
  rendering path (`hevc_nvenc`, `h264_nvenc`, `-hwaccel cuda`). Without
  NVENC the renderer silently falls back to `libx264` (slower) and
  without `-hwaccel cuda` it falls back to CPU decode via OpenCV.
- [AprilTag library](https://github.com/Social-Evolution-and-Behavior/apriltag)
  (SEBLab fork)

### Optional (fast detection pipeline)

Pulled in by the `[fast]` extra:

- `torch >= 2.0` (with CUDA)
- `pynvvideocodec >= 2.0` — NVDEC hardware decoding
- `tensorrt >= 8.6` — TensorRT backend for `.engine` / `.trt` weights
- `onnxruntime-gpu >= 1.17` — ONNX Runtime backend for `.onnx` weights

`.pt` weights work without any of these — they go through the standard
Ultralytics path.
