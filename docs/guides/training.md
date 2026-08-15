# Training and Tuning

`yoto detect` depends on two things you can improve without touching the
pipeline: the **YOLO weights** that find tag regions, and the **AprilTag
preset** that decodes them. `yoto train` builds the datasets and runs the
searches for both.

```bash
yoto train --help            # list the sub-commands
yoto train <sub> --help      # full flags for one
```

Everything writes under `<recording>/tracking/training/`, alongside the
`raw_data/`, `clean_data/` and `video_output/` that `detect`/`clean`/`render`
produce.

| Sub-command | Improves | Needs |
| ----------- | -------- | ----- |
| `build-testset` | — | videos (+ clean pickles, optional) |
| `subsample-testset` | — | a testset |
| `optimize-preset` | AprilTag preset | a testset |
| `compare-presets` | — | one or more preset JSONs |
| `build-yolo-dataset` | YOLO weights | videos |
| `build-crop-dataset` | tag classifier | a testset |

Nothing here needs an extra install — `optuna` and `plotly`, which
`optimize-preset` runs on, come with the base `pip install .`.

## Tuning the AprilTag preset

### 1. Build a testset

Samples frames, runs YOLO, and writes the cropped tag composites plus a
per-video `manifest.json` recording which tag IDs are actually present.
That manifest is the ground truth every later step scores against.

```bash
yoto train build-testset /path/to/recordings/ --dataname myrun
```

Ground truth comes from `--gt-source`:

- `clean` — from the decoded clean pickle. Accurate, but needs decoding to
  already work well enough to produce one.
- `yolo` — from the `_yolo.pkl` sidecar, with **no** ground-truth IDs. This
  is the cold-start path: use it when nothing decodes yet, so there is no
  clean pickle to learn from.
- `auto` (default) — clean pickle if present, else the YOLO sidecar.

`--sample-strategy mixed` (the default) takes half the frames from the
highest-visibility ones and half evenly spaced across the video. Pure `top`
biases the testset toward easy frames, which makes a preset look better than
it is on real footage.

### 2. Subsample it (optional)

A full testset is slow to score, and Optuna scores it once per trial. This
picks a compact subset that still covers every tag ID:

```bash
yoto train subsample-testset /path/to/recording --target 50 --min-appearances 3
```

It writes a `subset_manifest.json` that `optimize-preset` accepts via
`--subset-manifest`.

### 3. Optimize

Runs Optuna over AprilTag decoder and image-enhancement parameters, maximising
the number of distinct ground-truth tags decoded while penalising false
positives.

```bash
yoto train optimize-preset \
    --testset-dir /path/to/recording/tracking/training/apriltag_testset \
    --n-trials 500 \
    --search-space standard
```

The search spaces trade breadth against the trials needed to explore them:

| `--search-space` | Covers | Trials |
| ---------------- | ------ | ------ |
| `apriltag-only` | the 5 AprilTag decoder params, no enhancement | ~100 |
| `minimal` | decoder + upscale + contrast | ~100 |
| `standard-lite` | + cv2 / unsharp (can reproduce the `detect` default) | 300–600 |
| `standard` (default) | + tone mapping, Wiener deconvolution | 300–600 |
| `full` | + invert, bilateral, median, gamma, adaptive threshold | 800+ |

The output is `best_params_<study-name>.json`, which `yoto detect` consumes
directly:

```bash
yoto detect /path/to/experiment.mp4 \
    --yoloweights /path/to/yolo.pt \
    --apriltag-preset /path/to/best_params_yoto_preset.json
```

Useful flags:

- `--n-jobs N` spawns real OS worker processes sharing one SQLite study. A DB
  is created next to the testset unless you pass `--storage`.
- `--seed-params preset.json` enqueues an existing preset as trial 0, so the
  search starts from something known to work rather than from noise.
- `--export-trial N` skips optimisation entirely and rebuilds a preset JSON
  from row `N` of an earlier run's `trials_<study-name>.csv` — for when the
  best-scoring trial is not the one you want.
- `--testset-dir` also accepts a single image or a folder of images. With no
  manifests there is no ground truth, so the objective switches to maximising
  the count of *distinct* tags decoded.

### 4. Compare

```bash
yoto train compare-presets best_params_a.json best_params_b.json ir
```

Prints a parameter table (only differing rows unless `--all`) plus a decode
comparison on a testset. By default each column shows the parameters
`yoto detect` would *actually* use — presets are merged onto the pipeline
defaults first. `--raw` shows only what each JSON literally contains.

## Retraining the YOLO detector

`build-yolo-dataset` runs AprilTag over full frames to propose tag candidates,
then opens a browser UI for you to accept or reject them. Accepted boxes
export as a YOLO training set.

```bash
yoto train build-yolo-dataset /path/to/recordings/
```

Frames are chosen by `--frame-select`: `best-worst` splits them between the
best- and worst-decoding frames so the set is not all easy cases, `stride`
spaces them evenly, and `auto` (the default) picks best-worst when a clean
pickle is available and stride otherwise.

On a headless machine, precompute first and review later:

```bash
yoto train build-yolo-dataset /path/to/recordings/ --precompute-only True
yoto train build-yolo-dataset /path/to/recordings/   # serves the cache
```

## Building a tag-classifier dataset

`build-crop-dataset` reorganises the crops written by `build-testset` into an
ImageFolder / Ultralytics-classification layout — one directory per tag ID —
for training a classifier that reads tags YOLO found but AprilTag could not
decode.

```bash
yoto train build-crop-dataset /path/to/recording --val-frac 0.2
```

Two details matter for getting an honest number out of it:

- The split is **by experiment**, not by crop. Crops from one recording are
  highly correlated, so a random per-crop split leaks the validation set into
  training and inflates accuracy.
- `--ass-types all` (train default) includes YOLO-inferred crops — the
  undecoded ones this model exists to recover. `--val-ass-types original`
  (val default) excludes them, because an inferred label is partly a product
  of the cleaning chain matcher, so scoring against it measures the heuristic
  rather than the classifier.

`--symlink True` avoids copying entirely. Crops usually live on NFS, where
each copy is several round-trips on a ~2 KB file — that latency, not
bandwidth or `--jobs`, is what makes a large build slow.
