# Installation

## pip (recommended)

```bash
pip install yoto
```

For the fast NVDEC pipeline (requires NVIDIA GPU):

```bash
pip install yoto[fast]
```

For development:

```bash
git clone https://github.com/ychemtob/yoto.git
cd yoto
pip install -e ".[dev,docs,lint]"
pre-commit install
```

## conda

```bash
conda env create -f environment.yml
conda activate yoto
```

## Requirements

- Python >= 3.10
- NVIDIA GPU with CUDA support (for YOLO inference)
- ffmpeg (for video overlay rendering)
- [AprilTag library](https://github.com/Social-Evolution-and-Behavior/apriltag) (SEBLab fork)

### Optional (fast pipeline)

- PyNvVideoCodec >= 2.0
- PyTorch >= 2.0 with CUDA
