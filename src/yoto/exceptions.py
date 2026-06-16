"""Custom exception hierarchy for the YOTO package."""

from __future__ import annotations


class YotoError(Exception):
    """Base exception for all YOTO errors."""


class VideoReadError(YotoError):
    """Raised when a video file cannot be opened or read."""

    def __init__(self, path: str) -> None:
        self.path = path
        super().__init__(f"Cannot open video file: {path}")


class ModelLoadError(YotoError):
    """Raised when YOLO weights fail to load."""

    def __init__(self, weights_path: str, reason: str = "") -> None:
        self.weights_path = weights_path
        msg = f"Failed to load YOLO weights: {weights_path}"
        if reason:
            msg += f" ({reason})"
        super().__init__(msg)


class EmptyTrackingError(YotoError):
    """Raised when a tracking DataFrame has no usable tags to clean.

    Triggered by :func:`yoto.cleaning.clean_tracking_data` when the input
    is empty, has no tag columns, or has no tag whose detection count
    reaches the ``min_detections`` threshold.  Lets the CLI distinguish
    "nothing to do" from a real failure and skip without writing output.
    """


class EncoderError(YotoError):
    """Raised when no working ffmpeg encoder can be found or when ffmpeg
    dies mid-encode."""

    def __init__(self, message: str | None = None) -> None:
        super().__init__(
            message
            or (
                "No working ffmpeg encoder found. "
                "Install ffmpeg with nvenc or libx264 support."
            )
        )
