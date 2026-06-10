# Data Cleaning

The cleaning module post-processes raw detection data to produce
reliable trajectories.

## Pipeline Steps

1. **Filter sparse IDs** — tags with fewer than `min_detections` raw detections are dropped.
2. **Stamp provenance** — every frame/tag pair is assigned an `ass_type` code (`ORIGINAL`, `NONE`).
3. **Short-gap interpolation** — linear interpolation bridges gaps up to `interp_limit` frames.
4. **Distance computation** — frame-to-frame Euclidean distances per tag.
5. **Jump deletion** — consecutive detections farther than `max_jump_distance` px are removed as a block.
6. **YOLO-fill** — if a `_yolo.pkl` sidecar exists, undecoded YOLO boxes are used to recover missing tag positions via a forward + backward chain matcher; each tag's anchor is snapped to the nearest undecoded box within `snap_threshold × snap_multiplier`. A collision-resolution loop then resolves boxes claimed by multiple tags. Sub-steps:
    - **6a** — initial chain fill.
    - **6c** — prune ambiguous tracklets (boxes claimed by 2+ tags) and re-chain.
    - **6d** — re-chain on the post-prune state, with cross-frame collision tracking so tags can't re-claim a box another tag already occupies.
    - **6f** — extra `_delete_jump_blocks` pass to clean up any jumps the re-chain introduced.
7. **Long-gap recovery** *(experimental, gated by `--recover-long-gaps`)* — for each tag with a NaN gap longer than `min_gap_recovery_frames`, run a chained velocity-aware walk from the bounding anchors that snaps each gap frame to the closest unclaimed YOLO box within the snap radius. Targets ants whose AprilTag was temporarily un-decodable but YOLO still saw the body.
8. **Final jump pass** *(experimental, gated by `--final-jump-pass`)* — one more `_delete_jump_blocks` round on the fully-recovered YOLO-only data, before interpolation. Safety net for A/B comparisons.
9. **Re-interpolation** — short gaps created by steps 6–8 are filled by linear interpolation and frame-to-frame distances are recomputed. Always runs last so all output distances are fresh.

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

# YOLO-fill options (step 6)
yoto clean /path/to/experiment.mp4 \
    --yolo-fill True \
    --snap-multiplier 2.0 \
    --max-consecutive-misses 10 \
    --yolo-fill-limit 0 \
    --rechain-affected-only False

# Disable YOLO-fill entirely
yoto clean /path/to/experiment.mp4 --yolo-fill False

# Experimental: long-gap recovery (step 7) + extra final jump pass (step 8)
yoto clean /path/to/experiment.mp4 \
    --recover-long-gaps True \
    --min-gap-recovery-frames 10 \
    --final-jump-pass True

# px → mm calibration: measure the tag side length and stamp mm_per_px on the pkl
yoto clean /path/to/experiment.mp4 --tag-size 0.4

# Also write a flattened CSV next to each clean pickle
yoto clean /path/to/experiment.mp4 --csv

# Pickle per-step snapshots into <clean_dir>/<stem>_snapshots/ for notebook debugging
yoto clean /path/to/experiment.mp4 --debug-snapshots True
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
undecoded_df = yolo_df[~yolo_df["decoded"].astype(bool)].copy()

cleaned, ids, metrics = clean_tracking_data(
    df,
    min_detections=100,
    interpolation_limit=5,
    max_jump_distance=100.0,
    undecoded_df=undecoded_df,        # enables step 6 (YOLO-fill)
    snap_multiplier=2.0,
    max_consecutive_misses=10,
    yolo_fill_limit=0,                # 0 = uncapped; misses control chain death
    rechain_affected_only=False,
    recover_long_gaps=False,          # step 7 (experimental)
    min_gap_recovery_frames=10,
    final_jump_pass=False,            # step 8 (experimental)
    tag_size_mm=0.4,                  # measures px→mm scale on the output
)

print(f"Error rate: {metrics['error_pct']:.2f}%")
print(f"Gaps filled: {metrics['filled_pct_of_gaps']:.1f}%")
print(f"Scale: {cleaned.attrs['yoto_mm_per_px']:.5f} mm/px")
```

The imaging scale is computed for you inside `clean_tracking_data`; the
underlying helper is also importable if you want to measure scale
without running the full cleaning pipeline:

```python
from yoto.cleaning import compute_pixel_scale

median_side_px, mm_per_px, n_samples = compute_pixel_scale(
    df, tag_size_mm=0.4, sample_size=5000
)
```

## Quality Metrics

The `metrics` dict returned alongside the cleaned DataFrame contains:

| Key | Description |
|-----|-------------|
| `total_samples` | `n_frames × n_IDs` |
| `total_detections` | All raw detections (good + jumps) |
| `original_good_count` | Raw detections that survived the jump filter |
| `original_bad_count` | Raw detections deleted as jumps |
| `original_missing_count` | Frame/tag cells that were already NaN in the input |
| `total_gaps` | `original_bad_count + original_missing_count` |
| `error_pct` | Jump-rate as % of all detections |
| `original_bad_pct` | Jumps as % of total samples |
| `original_missing_pct` | Missing cells as % of total samples |
| `filled_count` | Cells filled by step-3/step-9 linear interpolation |
| `filled_pct_of_total` | Interpolated cells as % of total samples |
| `filled_pct_of_gaps` | Interpolated cells as % of all gaps |
| `yolo_inferred_count` | Cells recovered by YOLO-fill (`ASS_TYPE_YOLO_INFERRED`) |
| `yolo_inferred_pct_of_gaps` | YOLO-fill recoveries as % of all gaps |
| `yolo_pruned_count` | YOLO_INFERRED cells dropped by collision resolution |
| `yolo_rechained_count` | Cells re-filled by step 6d after pruning |
| `long_gap_recovered_count` | Cells filled by step 7 long-gap recovery (0 unless enabled) |
| `final_jump_deleted_count` | Cells removed by step 8 final-jump-pass (0 unless enabled) |
| `snap_threshold_px` | Per-video snap radius computed from the good-track distance percentile |

## Assignment Types

Each frame/tag pair has an `ass_type` code:

| Code | Constant | Meaning |
|------|----------|---------|
| 0 | `ASS_TYPE_NONE` | No data |
| 1 | `ASS_TYPE_ORIGINAL` | Original AprilTag detection |
| 2 | `ASS_TYPE_INTERPOLATED` | Linear interpolation |
| 3 | `ASS_TYPE_YOLO_INFERRED` | Position inferred from an undecoded YOLO box |

## Pickle Attributes

Every clean pickle carries a small metadata block on `frame_data.attrs`
(a plain `dict` that survives `pd.read_pickle`).  It records the YOTO
version, the knobs the run used, count summaries for each cleaning
pass, and the imaging scale.  This makes a pickle self-describing — you
can reload one months later and recover the exact parameters and a
mm-per-pixel conversion without re-running anything.

```python
import pandas as pd

df = pd.read_pickle("recording_clean.pkl")
print(df.attrs["yoto_mm_per_px"])         # e.g. 0.00803
print(df.attrs["yoto_yolo_filled_count"]) # e.g. 12_437
mm = df[(42, "center_x")].diff().abs() * df.attrs["yoto_mm_per_px"]
```

### Provenance

| Attribute | Type | Meaning |
|-----------|------|---------|
| `yoto_version` | str | YOTO version that wrote this clean pickle |
| `yoto_detect_version` | str | YOTO version that wrote the input detect pickle |
| `yoto_stage` | str | `"clean"` for cleaned pickles; `"detect"` on raw pickles |
| `yoto_cleaned_utc` | str | ISO-8601 UTC timestamp of this clean run |

### Detect-stage attributes (carried over)

The cleaner copies every attribute it found on the input detect pickle
onto its output, then overlays the clean-stage fields on top.  These
are the ones the `yoto detect` run stamps — they survive into the
clean pickle so the full pipeline configuration is recoverable from a
single file.

| Attribute | Type | Meaning |
|-----------|------|---------|
| `yoto_pipeline` | str | `"simple"` (portable Ultralytics) or `"fast"` (NVDEC + GPU-resident) |
| `yoto_video` | str | Absolute path to the source video |
| `yoto_created_utc` | str | ISO-8601 UTC timestamp of the detect run |
| `yoto_yolo_weights` | str | Path to the YOLO weights used for detection |
| `yoto_preset` | str \| None | AprilTag preset (built-in name or JSON path), or `None` for defaults |
| `yoto_pad_ratio` | float | Per-axis padding ratio applied around each YOLO box before AprilTag decoding |
| `yoto_nms_mode` | str | `"suppress"` (standard NMS) or `"fuse"` (replace overlap clusters with union). Fast pipeline only |

### YOLO-fill knobs and counts

| Attribute | Type | Meaning |
|-----------|------|---------|
| `yoto_snap_threshold_px` | float | Per-frame snap radius computed from the good-track distance percentile |
| `yoto_snap_multiplier` | float | Multiplier applied to `snap_threshold_px` to obtain the snap cap |
| `yoto_yolo_fill_limit` | int | Hard cap on YOLO-fill gap length (0 = unlimited; chains die via `max_consecutive_misses`) |
| `yoto_max_consecutive_misses` | int | Consecutive miss frames before a chain anchor is cleared |
| `yoto_rechain_affected_only` | bool | Whether the step-6d re-chain restricted candidates to pruned tags |
| `yoto_yolo_filled_count` | int | Cells filled by YOLO-fill (`ASS_TYPE_YOLO_INFERRED`) |
| `yoto_yolo_pruned_count` | int | YOLO_INFERRED cells dropped by collision resolution / cross-track pruning |
| `yoto_yolo_rechained_count` | int | Cells re-filled by step-6d after pruning |

### Long-gap recovery (experimental)

| Attribute | Type | Meaning |
|-----------|------|---------|
| `yoto_recover_long_gaps` | bool | Whether step 7 (`--recover-long-gaps`) ran |
| `yoto_min_gap_recovery_frames` | int | Minimum gap length step 7 attempts to fill |
| `yoto_long_gap_recovered_count` | int | Cells filled by the long-gap recovery walk |

### Final jump pass (experimental)

| Attribute | Type | Meaning |
|-----------|------|---------|
| `yoto_final_jump_pass` | bool | Whether the extra step-8 jump-block pass ran |
| `yoto_final_jump_deleted_count` | int | Cells deleted by the step-8 pass |

### Imaging scale

Measured at the *start* of cleaning, from decoded AprilTag corners
(`COL_CORNERS`) in the input pickle — before any tag filtering, so the
sample is as broad as possible.  See
[`compute_pixel_scale`](../api/cleaning.md#yoto.cleaning.compute_pixel_scale)
for the stratified sampling logic.

| Attribute | Type | Meaning |
|-----------|------|---------|
| `yoto_tag_size_mm` | float | Physical side length of one tag border, supplied via `--tag-size` (default 0.4) |
| `yoto_median_tag_side_px` | float | Median measured side length in pixels (`nan` if no corners) |
| `yoto_mm_per_px` | float | `tag_size_mm / median_tag_side_px` — multiply pixel distances by this to get mm |
| `yoto_scale_sample_count` | int | Number of tag instances sampled to compute the median |
