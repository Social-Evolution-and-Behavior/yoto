"""Named constants used throughout the YOTO package.

All magic numbers and default configuration values live here so they can
be referenced, overridden, and documented in a single place.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# YOLO detection defaults
# ---------------------------------------------------------------------------

#: Minimum confidence score for a YOLO bounding-box to be kept.
DEFAULT_CONF_THRESHOLD: float = 0.1

#: Pixels added around each detected bounding box before cropping.
DEFAULT_PAD_PIXELS: int = 10

#: IoU threshold for non-maximum suppression.
DEFAULT_IOU_THRESHOLD: float = 0.7

#: YOLO input resolution (square side length).
DEFAULT_TARGET_SIZE: int = 1024

#: Maximum detections per frame passed to NMS.
DEFAULT_MAX_DETECTIONS: int = 300

#: Default batch size for the fast (NVDEC) pipeline.
DEFAULT_BATCH_SIZE: int = 2

# ---------------------------------------------------------------------------
# AprilTag detection defaults
# ---------------------------------------------------------------------------

#: Default AprilTag family (Social Evolution & Behavior fork).
DEFAULT_TAG_FAMILY: str = "tag36ARTag"

#: Maximum Hamming-distance correction allowed during decoding.
DEFAULT_MAX_HAMMING: int = 3

#: Decimation factor for the AprilTag quad detector.
DEFAULT_DECIMATE: float = 1.5

#: Gaussian blur sigma applied before decoding.
DEFAULT_BLUR: float = 0.7

#: Whether to refine detected edges for sub-pixel accuracy (0 = off).
DEFAULT_REFINE_EDGES: int = 0

#: Sharpening applied during decoding.
DEFAULT_DECODE_SHARPENING: float = 1.0

#: Number of threads used inside a single ``detect()`` call.
DEFAULT_APRILTAG_THREADS: int = 4

#: Number of threads passed to the apriltag C library constructor.
DEFAULT_DETECTOR_THREADS: int = 6

#: Deglitch parameter for the quad-tree phase.
DEFAULT_QTP_DEGLITCH: int = 1

# ---------------------------------------------------------------------------
# Image pre-processing defaults (simple pipeline)
# ---------------------------------------------------------------------------

#: Kernel size for the unsharp-mask Gaussian blur.
DEFAULT_UNSHARP_KERNEL: tuple[int, int] = (11, 11)

#: Sigma for the unsharp-mask Gaussian blur.
DEFAULT_UNSHARP_SIGMA: float = 3.0

#: Strength multiplier for the unsharp mask.
DEFAULT_UNSHARP_AMOUNT: float = 5.0

#: Contrast enhancement factor (PIL mode).
DEFAULT_CONTRAST_FACTOR: float = 1.75

# ---------------------------------------------------------------------------
# Image pre-processing defaults (fast pipeline — cv2 mode)
# ---------------------------------------------------------------------------

#: Kernel size for unsharp mask in the fast pipeline.
FAST_UNSHARP_KERNEL: tuple[int, int] = (9, 9)

#: Sigma for unsharp mask in the fast pipeline.
FAST_UNSHARP_SIGMA: float = 2.75

#: Strength multiplier for unsharp mask in the fast pipeline.
FAST_UNSHARP_AMOUNT: float = 5.0

#: OpenCV ``convertScaleAbs`` alpha (gain) for contrast.
FAST_CV2_ALPHA: float = 2.0

#: OpenCV ``convertScaleAbs`` beta (bias) for contrast.
FAST_CV2_BETA: float = -80.0

#: Contrast factor for PIL-based enhancement in the fast pipeline.
FAST_CONTRAST_FACTOR: float = 1.75

#: Decimation factor for the fast pipeline (slightly more aggressive).
FAST_DECIMATE: float = 2.0

#: Decode sharpening for the fast pipeline.
FAST_DECODE_SHARPENING: float = 0.5

# ---------------------------------------------------------------------------
# Data cleaning defaults
# ---------------------------------------------------------------------------

#: Minimum number of non-NaN detections required to keep a tag ID.
MIN_DETECTIONS_PER_ID: int = 100

#: Maximum gap length (frames) that linear interpolation will bridge.
DEFAULT_INTERPOLATION_LIMIT: int = 5

#: Maximum distance (pixels) between consecutive frames before a
#: detection is considered a tracking jump and deleted.
DEFAULT_MAX_JUMP_DISTANCE: float = 100.0

# ---------------------------------------------------------------------------
# Tag ID filtering
# ---------------------------------------------------------------------------

#: Tags with IDs above this value are discarded (family-dependent).
MAX_VALID_TAG_ID: int = 237

# ---------------------------------------------------------------------------
# Video overlay defaults
# ---------------------------------------------------------------------------

#: Length of the trail drawn behind each tag (in frames).
DEFAULT_TRAIL_LENGTH: int = 50

#: How many recent frames to skip (no trail segment drawn for these).
DEFAULT_TRAIL_SKIP: int = 4

#: Random seed used to generate per-tag colors (deterministic palette).
TAG_COLOR_SEED: int = 42

# ---------------------------------------------------------------------------
# Data column names (used in the MultiIndex DataFrame)
# ---------------------------------------------------------------------------

COL_FRAME: str = "frame"
COL_CENTER_X: str = "center_x"
COL_CENTER_Y: str = "center_y"
COL_CORNERS: str = "corners"
COL_ASS_TYPE: str = "ass_type"
COL_DISTANCE: str = "distance"

# ---------------------------------------------------------------------------
# Assignment type codes (stored in the ``ass_type`` column)
# ---------------------------------------------------------------------------

#: No data available for this tag on this frame.
ASS_TYPE_NONE: int = 0

#: Original detection from the AprilTag detector.
ASS_TYPE_ORIGINAL: int = 1

#: Value filled by short-range linear interpolation.
ASS_TYPE_INTERPOLATED: int = 2
