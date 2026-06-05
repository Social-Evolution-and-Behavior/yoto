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

#: Per-axis padding ratio added around each YOLO bounding box before
#: cropping for AprilTag decoding.  Each side grows by
#: ``pad_ratio * box_dimension`` (independently in x and y), so the
#: padding scales with the apparent tag size — useful when the camera
#: height changes between recordings.  Default ``0.34`` matches the
#: empirical sweet spot for the CleanSlab/IR dataset where the median
#: box dimension is ~35 px and 12 px each side decodes best.
DEFAULT_PAD_RATIO: float = 0.34

#: IoU threshold for non-maximum suppression.
DEFAULT_IOU_THRESHOLD: float = 0.4

#: YOLO input resolution (square side length).
DEFAULT_TARGET_SIZE: int = 1024

#: Maximum detections per frame passed to NMS.
DEFAULT_MAX_DETECTIONS: int = 300

#: Default batch size for the fast (NVDEC) pipeline.
DEFAULT_BATCH_SIZE: int = 2

#: Maximum allowed distance between an AprilTag decode's center and the
#: center of its source YOLO box, expressed as a fraction of the box's
#: shorter side: ``threshold = max_offset_ratio * min(box_w, box_h)``.
#: Tags decoded outside that radius are dropped — almost always
#: AprilTag finding a quad in the padding region rather than the actual
#: tag.  ``0.6`` permits a tag center to fall in the central 1.2 *
#: min(box_w, box_h) square around the box center, generous enough that
#: it doesn't fire on slightly off-center genuine decodes.
DEFAULT_MAX_TAG_OFFSET_RATIO: float = 0.6

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

#: Maximum gap length (frames) that the YOLO-fill pass will bridge.
#: ``0`` (default) disables the cap — gap length is controlled instead
#: by ``DEFAULT_MAX_CONSECUTIVE_MISSES`` so chains die on their own when
#: the evidence runs out.  Set to a positive integer to add a hard cap.
DEFAULT_YOLO_FILL_LIMIT: int = 0

#: Percentile of the "good track" frame-to-frame distance distribution
#: used as the per-frame snap threshold for YOLO fill.  A high value
#: tolerates the rare fast movements without admitting unrealistic jumps.
DEFAULT_GOOD_TRACK_PERCENTILE: float = 99.0

#: Multiplier applied to ``snap_threshold`` to obtain the maximum
#: distance between a YOLO box and the linear-interp prior for the
#: YOLO-fill snap to be accepted.  Constant — does NOT scale with gap
#: age, so long gaps stay tightly bounded.
DEFAULT_SNAP_MULTIPLIER: float = 2.0

#: Maximum number of consecutive frames inside a chain where no YOLO
#: box passes the snap test before the chain is broken (anchor cleared).
#: Lets short stretches of bad YOLO detection survive without letting
#: the chain drift off when the tag really has left the scene.
DEFAULT_MAX_CONSECUTIVE_MISSES: int = 10

#: When re-chaining after a prune pass, restrict candidates to only the
#: tags whose YOLO fills were pruned (``True``), rather than letting all
#: tags compete for the freed boxes (``False``, default).
DEFAULT_RECHAIN_AFFECTED_ONLY: bool = False

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
COL_BOX_X1: str = "box_x1"
COL_BOX_Y1: str = "box_y1"
COL_BOX_X2: str = "box_x2"
COL_BOX_Y2: str = "box_y2"
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

#: Value inferred from an undecoded YOLO box in the same frame.
ASS_TYPE_YOLO_INFERRED: int = 3
