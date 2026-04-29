# Installation

YOTO is installed from source. `pyproject.toml` is the single source of
truth for dependencies; pick one of the environments below and use the
relevant `[extras]` for your use case.

## With conda (recommended)

```bash
conda create -n yoto python=3.11 -y
conda activate yoto
git clone https://github.com/Social-Evolution-and-Behavior/yoto.git
cd yoto
pip install -e ".[dev,fast]"     # add ,docs,lint if needed
pre-commit install
```

## Plain pip / venv

```bash
git clone https://github.com/Social-Evolution-and-Behavior/yoto.git
cd yoto
pip install -e ".[dev,fast]"
pre-commit install
```

## Available extras

| Extra   | What it pulls in                                   | When you need it                |
| ------- | -------------------------------------------------- | ------------------------------- |
| `fast`  | `torch`, `pynvvideocodec`, `tensorrt`, `onnxruntime-gpu` | NVDEC detection / `.engine` / `.onnx` weights |
| `dev`   | `pytest`, `pytest-cov`, `pytest-mock`, `hypothesis` | Running the test suite          |
| `docs`  | `mkdocs-material`, `mkdocstrings[python]`, …       | Building these docs locally     |
| `lint`  | `black`, `flake8`, `mypy`, `vulture`               | Local linting / pre-commit      |

`.pt` weights work without the `fast` extra — they go through the standard
Ultralytics path.

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
