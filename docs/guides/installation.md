# Installation

YOTO needs Python 3.10 or newer, and is installed from a clone of the
repository. Pick either recipe below — conda if you already use it, plain pip
otherwise.

## With conda (recommended)

```bash
conda create -n yoto python=3.11 -y
conda activate yoto
git clone https://github.com/Social-Evolution-and-Behavior/yoto.git
cd yoto
pip install .
```

## Plain pip / venv

```bash
git clone https://github.com/Social-Evolution-and-Behavior/yoto.git
cd yoto
pip install .
```

`pip install .` pulls in every Python dependency `detect`, `clean`, `render`
and `train` need at runtime — no extras required — and the default YOLO
weights ship inside the package, so `--yoloweights` is optional.

Two things are not pip-installable: **ffmpeg** (see
[Requirements](#requirements)) and the **AprilTag library**, which is built
from source — see
[Installing the AprilTag library](#installing-the-apriltag-library).

Check it worked:

```bash
yoto --version
yoto --help
```

## Installing the AprilTag library

YOTO decodes tags through the [LSEB fork of
AprilTag](https://github.com/Social-Evolution-and-Behavior/apriltag). It is not
on PyPI, so it is built from source. The build produces a C shared library
(`libapriltag.so`) and a Python extension module (`apriltag.*.so`).

Build it with the YOTO environment activated, and install into that
environment:

```bash
conda activate yoto
sudo apt install build-essential cmake

git clone https://github.com/Social-Evolution-and-Behavior/apriltag.git
cd apriltag

cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" \
  -DPython3_EXECUTABLE="$CONDA_PREFIX/bin/python"

cmake --build build -j"$(nproc)"

cmake --install build
```

Check it:

```bash
python -c "from apriltag import apriltag; print('apriltag ok')"
```

## Optional extras

Nothing here is needed to run the pipeline — every runtime dependency is
already in the plain install. There is exactly one extra a user might want:

| Extra     | What it pulls in                    | When you need it                          |
| --------- | ----------------------------------- | ----------------------------------------- |
| `engines` | `tensorrt`, `onnxruntime-gpu`       | `.engine` / `.trt` / `.onnx` YOLO weights |

```bash
pip install ".[engines]"
```

The quotes matter in zsh, which otherwise reads the brackets as a glob.

The bundled default weights are `.pt` and go through the standard Ultralytics
path, so most people never need this. It is only for running a TensorRT- or
ONNX-converted model.

The remaining extras — `dev`, `docs`, `lint` — are for working on YOTO
itself, not for using it. See [Developing YOTO](#developing-yoto).

## Developing YOTO

Only needed if you are changing YOTO's own source. Two extra pieces:

```bash
pip install -e ".[dev,lint,docs]"
pre-commit install
```

`-e` (`--editable`) installs the checkout in place instead of copying it into
`site-packages`. Python then imports the source files you are editing, so a
change takes effect on the next run with no reinstall. Without `-e` you would
have to re-run `pip install .` after every edit. Plain users do not need it.

`pre-commit install` writes a git hook into `.git/hooks/`. From then on,
`git commit` first runs the checks configured in `.pre-commit-config.yaml`
(trailing whitespace, end-of-file fixer, YAML validation, `black`, `flake8`)
and aborts the commit if any fails, so formatting problems are caught before
they reach CI. It is a one-time, per-clone step and affects nothing outside
this repository. `mypy` is deliberately not a hook — run it with
`make typecheck`.

## Requirements

- Python >= 3.10
- NVIDIA GPU with CUDA support (for YOLO inference)
- ffmpeg — **built with NVENC encoders and CUDA hwaccel** for the fast
  rendering path (`hevc_nvenc`, `h264_nvenc`, `-hwaccel cuda`). Without
  NVENC the renderer silently falls back to `libx264` (slower) and
  without `-hwaccel cuda` it falls back to CPU decode via OpenCV.
- [AprilTag library](https://github.com/Social-Evolution-and-Behavior/apriltag)
  (LSEB fork) — built from source, see
  [Installing the AprilTag library](#installing-the-apriltag-library)

### Optional (alternative weight formats)

Pulled in by the `[engines]` extra:

- `tensorrt >= 8.6` — TensorRT backend for `.engine` / `.trt` weights
- `onnxruntime-gpu >= 1.17` — ONNX Runtime backend for `.onnx` weights

`.pt` weights work without either of these — they go through the standard
Ultralytics path.
