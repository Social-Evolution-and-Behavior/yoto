# Data Cleaning

The cleaning module post-processes raw detection data to produce
reliable trajectories.

## Pipeline Steps

1. **Filter sparse IDs** — tags with fewer than `min_detections` raw detections are dropped
2. **Stamp provenance** — every frame/tag pair is assigned an `ass_type` code (`ORIGINAL`, `NONE`)
3. **Short-gap interpolation** — linear interpolation bridges gaps up to `interp_limit` frames
4. **Distance computation** — frame-to-frame Euclidean distances per tag
5. **Jump deletion** — consecutive detections farther than `max_jump_distance` px are removed as a block
6. **YOLO-fill** — if a `_yolo.pkl` sidecar exists, undecoded YOLO boxes are used to recover missing tag positions via a forward+backward chain matcher; each tag's anchor is snapped to the nearest undecoded box within `snap_threshold × snap_multiplier`. A collision-resolution loop then resolves boxes claimed by multiple tags and optionally prunes isolated ORIGINAL detections that sit on another tag's continuous trajectory (likely AprilTag misdecodes in the padding region).
7. **Re-interpolation** — short gaps created by step 6 are filled by linear interpolation

## Usage

### Command line

```bash
# Pass a video, a raw pickle, or a directory tree of recordings.
yoto clean /path/to/experiment.mp4
yoto clean /path/to/recordings/

# Override defaults
yoto clean /path/to/experiment.mp4 \
    --min-detections 100 \
    --interp-limit 5 \
    --max-jump 100

# YOLO-fill options
yoto clean /path/to/experiment.mp4 \
    --yolo-fill True \
    --snap-multiplier 2.0 \
    --max-consecutive-misses 10 \
    --collision-max-iters 5 \
    --prune-cross-track-originals True \
    --cross-track-neighbor-frames 5

# Disable YOLO-fill entirely
yoto clean /path/to/experiment.mp4 --yolo-fill False
```

When a video file is passed, `yoto clean` finds the matching raw pickle in
`<recording>/tracking/raw_data/` (using `--dataname` for the suffix, default
`_apriltagDetect14`) and auto-discovers the `_yolo.pkl` sidecar in the same
directory.

### Python API

```python
from yoto.cleaning import clean_tracking_data
import pandas as pd

df = pd.read_pickle("/path/to/raw_tracking.pkl")
yolo_df = pd.read_pickle("/path/to/raw_tracking_yolo.pkl")

cleaned, ids, metrics = clean_tracking_data(
    df,
    min_detections=100,
    interpolation_limit=5,
    max_jump_distance=100.0,
    yolo_fill=True,
    yolo_df=yolo_df,
    snap_multiplier=2.0,
    max_consecutive_misses=10,
    collision_max_iters=5,
    prune_cross_track_originals=True,
    cross_track_neighbor_frames=5,
)

print(f"Error rate: {metrics['error_pct']:.2f}%")
print(f"Gaps filled: {metrics['filled_pct_of_gaps']:.1f}%")
```

## Quality Metrics

The `metrics` dict contains:

| Key | Description |
|-----|-------------|
| `total_samples` | frames × IDs |
| `total_detections` | all raw detections (including jumps) |
| `original_good_count` | valid raw detections |
| `original_bad_count` | detections deleted as jumps |
| `error_pct` | jump rate as % of detections |
| `filled_count` | gaps filled by interpolation |
| `filled_pct_of_gaps` | % of all gaps successfully filled |
| `yolo_filled_count` | positions recovered by YOLO-fill |
| `yolo_cross_track_originals_pruned` | isolated ORIGINAL detections removed as misdecodes |

## Assignment Types

Each frame/tag pair has an `ass_type` code:

| Code | Constant | Meaning |
|------|----------|---------|
| 0 | `ASS_TYPE_NONE` | No data |
| 1 | `ASS_TYPE_ORIGINAL` | Original AprilTag detection |
| 2 | `ASS_TYPE_INTERPOLATED` | Linear interpolation |
| 3 | `ASS_TYPE_YOLO_INFERRED` | Position inferred from an undecoded YOLO box |
