# Data Cleaning

The cleaning module post-processes raw detection data to produce
reliable trajectories.

## Pipeline Steps

1. **Filter sparse IDs** — tags with fewer than 100 detections are removed
2. **Short-gap interpolation** — linear interpolation bridges gaps up to 5 frames
3. **Distance computation** — frame-to-frame Euclidean distances per tag
4. **Jump deletion** — detections with distance > 100px are flagged as jumps
5. **Re-interpolation** — gaps created by jump deletion are re-filled

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
```

When a video file is passed, `yoto clean` finds the matching raw pickle in
`<recording>/tracking/raw_data/` (using `--dataname` for the suffix, default
`_apriltagDetect14`).

### Python API

```python
from yoto.cleaning import clean_tracking_data
import pandas as pd

df = pd.read_pickle("/path/to/raw_tracking.pkl")
cleaned, ids, metrics = clean_tracking_data(
    df,
    min_detections=100,
    interpolation_limit=5,
    max_jump_distance=100.0,
)

print(f"Error rate: {metrics['error_pct']:.2f}%")
print(f"Gaps filled: {metrics['filled_pct_of_gaps']:.1f}%")
```

## Quality Metrics

The `metrics` dict contains:

| Key | Description |
|-----|-------------|
| `total_samples` | frames x IDs |
| `total_detections` | all raw detections (including jumps) |
| `original_good_count` | valid raw detections |
| `original_bad_count` | detections deleted as jumps |
| `error_pct` | jump rate as % of detections |
| `filled_count` | gaps filled by interpolation |
| `filled_pct_of_gaps` | % of all gaps successfully filled |

## Assignment Types

Each frame/tag pair has an `ass_type` code:

| Code | Meaning |
|------|---------|
| 0 | No data |
| 1 | Original detection |
| 2 | Interpolated value |
