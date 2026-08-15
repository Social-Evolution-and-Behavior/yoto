"""YOTO — GPU-accelerated AprilTag tracking pipeline for ant behavioral studies.

Provides a complete workflow for detecting AprilTag barcodes on ants using
YOLO object detection, with tools for data cleaning, interpolation, and
video overlay visualization.
"""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("yoto")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

from yoto.detection import run_detection_simple, run_detection_fast
from yoto.cleaning import clean_tracking_data, clean_video
from yoto.video import render_overlay_video
from yoto.io import load_corners, load_data

__all__ = [
    "__version__",
    "run_detection_simple",
    "run_detection_fast",
    "clean_tracking_data",
    "clean_video",
    "render_overlay_video",
    "load_data",
    "load_corners",
]
