"""YOTO — GPU-accelerated AprilTag tracking pipeline for ant behavioral studies.

Provides a complete workflow for detecting AprilTag barcodes on ants using
YOLO object detection, with tools for data cleaning, interpolation, and
video overlay visualization.
"""

__version__ = "0.8.1"

from yoto.detection import run_detection_simple, run_detection_fast
from yoto.cleaning import clean_tracking_data
from yoto.video import render_overlay_video

__all__ = [
    "__version__",
    "run_detection_simple",
    "run_detection_fast",
    "clean_tracking_data",
    "render_overlay_video",
]
