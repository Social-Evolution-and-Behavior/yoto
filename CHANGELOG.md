# Changelog

## v1.0.0 — 2026-08-15

First stable release. It closes the two remaining TODOs from the 0.x line —
a semi-automatic retraining pipeline and a documented analysis API — and
changes the on-disk storage format for every pipeline pickle.

Every pickle written by earlier versions still loads. Nothing needs to be
re-detected or re-cleaned.

### Highlights

- **`yoto train`** — a tool suite for retraining the two things `detect`
  depends on: the AprilTag preset and the YOLO weights.
- **`yoto.io`** — a documented loading API (`load_data`, `load_corners`) so
  analysis code never touches the storage layout directly.
- **Smaller, faster pickles** — zstd compression plus flat float32 corner
  columns. A raw pickle drops from 126 MB to 46 MB and its write from 4.9 s
  to 0.06 s.
- **pretrained yolo model ship with the package** — `--yoloweights` is now optional.

### Breaking changes

- **Raw detection pickles are long format.** `detect` now writes one row per
  decoded detection, with plain `tag_id` / `center_x` / `center_y` / corner
  columns indexed by frame number, instead of a wide `(tag_id, metric)`
  MultiIndex. `clean` pivots to wide internally (`_ensure_wide`) and accepts
  either shape, so old raw pickles still clean. `clean` output is unchanged
  in shape: still wide `(tag_id, metric)`.
- **Corners are eight flat `float32` columns** (`c0x` … `c3y`, in the
  AprilTag `lb-rb-rt-lt` order) rather than one object-dtype `corners` column
  holding a `(4, 2)` array per cell. The object column boxed a separate small
  ndarray per detection, costing ~2.6x the bytes of the coordinates and
  dominating read/write time. `float32` resolves a 4512 px coordinate to
  ~0.00024 px, far finer than AprilTag's sub-pixel accuracy.
  **Read corners with `yoto.io.load_corners(df, tag_id=None)`**, which returns
  `(n, 4, 2)` from either format. `constants.COL_CORNERS` still exists for
  reading legacy pickles but nothing writes it any more.
- **Pipeline pickles are zstd-compressed** and written as `.pkl.zst`
  (`PICKLE_EXT`). pandas infers the codec from the extension, so
  `pd.read_pickle(path)` needs no extra argument. Every reader accepts both
  `.pkl.zst` and plain `.pkl` (`PICKLE_EXTS`), and old files are never
  rewritten. Use `yoto.io.strip_pickle_ext()` rather than
  `os.path.splitext()` on a pickle path, and `yoto.io.find_pickle(base)` to
  resolve a path across extensions.
- **`clean --snap-multiplier` is now `--tag-size-multiplier`** (Python:
  `snap_multiplier` → `tag_size_multiplier`), and its meaning changed. The
  YOLO-fill search radius is `median_tag_side_px * tag_size_multiplier` — a
  radius derived from the measured tag size in pixels, not from a percentile
  of how far ants move between frames. `DEFAULT_SNAP_MULTIPLIER` and
  `DEFAULT_GOOD_TRACK_PERCENTILE` are gone, replaced by
  `DEFAULT_TAG_SIZE_MULTIPLIER`. The former `_compute_snap_threshold` is now
  `_compute_max_move_px`, kept only as an informational metric.
- **Step-6 sub-steps relabelled contiguously.** The letters skipped `6b`, a
  leftover from a removed step. They are now `6a`–`6e`, which renames the
  `--debug-snapshots` files: `step6c_after_prune` → `step6b_after_prune`,
  `step6d_after_rechain` → `step6c_after_rechain`, `step6f_after_jump` →
  `step6e_after_jump`. Anything reading snapshots by filename needs updating.
- **`examples/load_clean_folder.py` row index changed** from
  `(source, frame)` to `(frame, source, video_frame)`, matching `load_data`:
  `frame` is a global monotonic counter across the folder, `source` is the
  video stem, `video_frame` is the original per-video frame number.

### Added

#### `yoto train` — retraining and tuning

A new sub-command group. Everything writes under
`<recording>/tracking/training/`, alongside `raw_data/`, `clean_data/` and
`video_output/`. Full workflow in `docs/guides/training.md`.

- `train build-testset` — samples frames, runs YOLO, writes cropped tag
  composites plus a per-video `manifest.json` of the tag IDs actually
  present, which is the ground truth every later step scores against.
  `--gt-source clean|yolo|auto` selects where that ground truth comes from;
  `yolo` is the cold-start path for footage where nothing decodes yet.
  `--sample-strategy mixed` (default) splits frames between the
  highest-visibility ones and an even spread, so a preset cannot look good
  purely by being scored on easy frames.
- `train subsample-testset` — cuts a testset down so an Optuna trial, which
  scores the whole set once per trial, stays affordable.
- `train optimize-preset` — Optuna search over AprilTag decoder parameters,
  with pruning, a speed/recall trade-off (`--speed-weight`,
  `--speed-floor-recall`), resumable studies (`--storage`, `--study-name`),
  seedable starting points (`--seed-params`), and trial export
  (`--export-trial`, `--trials-csv`). Requires the new `tune` extra.
- `train compare-presets` — parameter table plus a decode comparison across
  preset JSONs, showing by default what `yoto detect` would actually use
  (presets merged onto pipeline defaults) rather than the raw JSON.
- `train build-yolo-dataset` — runs AprilTag over full frames to propose tag
  candidates, then serves a browser review UI (FastAPI + uvicorn) to accept
  or reject them and export a YOLO training set. `--frame-select best-worst`
  keeps the set from being all easy cases; `--precompute-only True` lets a
  headless machine do the work and a workstation do the review.
- `train build-crop-dataset` — reorganises `build-testset` crops into an
  ImageFolder / Ultralytics-classification layout for training a classifier
  that reads tags YOLO found but AprilTag could not decode. Splits **by
  experiment**, not by crop, because crops from one recording are correlated
  enough that a per-crop split leaks into training. `--symlink True` avoids
  copying entirely, which matters on NFS where per-file latency dominates.

New modules behind it: `yoto/tuning/{testset,optimize,viz,crop_dataset,preprocess}.py`
and `yoto/tuning/yolo_dataset/` (builder, FastAPI server, static review UI).

#### Data loading API

- New `yoto.io` module, exported from the package root: `load_data`,
  `load_corners`, `corner_row`, `corner_tag_ids`, `has_corners`,
  `find_pickle`, `is_pickle`, `strip_pickle_ext`.
- `load_data(path, dataname, video_nb, corners)` loads a whole recording's
  clean pickles as one DataFrame, or a single video's, with a
  `(frame, source, video_frame)` row MultiIndex and merged `.attrs`
  (including `attrs["scale"]`, the median mm/px). `corners=False` drops the
  corner metrics, which roughly halves the loaded frame — a whole recording
  rarely fits in RAM otherwise. `video_nb` loads one video at a time.
- `clean_video()` is now public and exported: cleans a single raw pickle (or
  a video path, resolving the pickle itself), discovers the `_yolo` sidecar,
  and writes to `tracking/clean_data/` — the Python mirror of `yoto clean`.

#### CLI

- `detect --no-enhance` — skip all image enhancement, including preset
  pre-stages, and decode the raw grayscale crop. For diagnostics and
  already-clean input.
- `clean --parallel N` — GNU parallel batch cleaning, matching `detect` and
  `render`.
- `--yoloweights` is now optional everywhere. Default weights resolve, in
  order, from the `YOTO_WEIGHTS` env var, the copy bundled in the installed
  package, the source-tree `models/`, and `./models/` (`DEFAULT_WEIGHTS`).
  The wheel force-includes `models/detect34.pt` at
  `<site-packages>/yoto/models/detect34.pt`, so an installed `yoto` finds it
  from any working directory.
- `--video-nb` accepts a single index, a comma list, and/or inclusive ranges
  (`0-4,10`) on `detect`, `clean`, `render` and `train build-testset`.

#### Constants

Filesystem layout (`TRACKING_DIR`, `TRACKING_SUBDIRS`, `TRAINING_SUBDIR`,
`TESTSET_SUBDIR`), file discovery (`VIDEO_EXTENSIONS`, `IMAGE_EXTENSIONS`),
storage (`CORNER_COLS`, `CORNER_DTYPE`, `PICKLE_COMPRESSION`, `PICKLE_EXT`,
`PICKLE_EXTS`), `DEFAULT_DATANAME`, and `DEFAULT_WEIGHTS`.

### Fixed

- **Duplicate `(frame, tag_id)` decodes are resolved by trajectory
  continuity.** A physical ant carries one tag, so two decodes of the same ID
  in a frame mean at most one is genuine. `_resolve_duplicate_ids` keeps the
  candidate closest to the position interpolated from frames where that tag
  decoded exactly once, instead of blindly keeping the first row. All
  occurrences are still preserved in the raw pickle.
- `render --quads` and the debug quad overlay read corners through
  `load_corners`, so they work with both storage formats, and quad
  extraction is vectorised rather than per-cell.

### Documentation

- New **Training and Tuning** guide (`docs/guides/training.md`) and **Tuning
  API** reference (`docs/api/tuning.md`).
- New **Data Loading API** reference (`docs/api/io.md`).
- Detection guide documents the long-format output, corner storage, and
  compression, with the measured size/time numbers.
- Cleaning guide documents the output format and the ORIGINAL-only corner
  invariant.
- Quickstart covers `--video-nb`, `--conf` / `--iou`, the retraining step,
  and loading a whole experiment with `load_data` / `load_corners`.
- Logo added to the README, the docs index, and the MkDocs header/favicon;
  header logo sizing fixed so it no longer overhangs the tab bar on scroll.
- MkDocs excludes working documents (`*/specs/`, `*/plans/`) from the built
  site.
- Install instructions corrected: the user-facing install is `pip install .`,
  with no extras needed to run the pipeline. `-e` and `pre-commit install`
  moved into a "Developing YOTO" section that explains what each actually
  does. Fixed two broken links in the docs index that pointed at
  `docs/guides/…` from inside `docs/`.
- New **Installing the AprilTag library** section: the build-from-source
  steps, including the `CMAKE_INSTALL_PREFIX` and `Python3_EXECUTABLE` flags
  that install the wrapper into the active environment instead of
  `/usr/local`, where it lands outside the environment's `sys.path` and
  `import apriltag` fails despite a successful build.

### Packaging

- **`pip install .` is now the complete runtime install.** `fastapi`,
  `uvicorn[standard]`, `zstandard`, `torch`, `pynvvideocodec`, `optuna` and
  `plotly` are all core dependencies. Previously `detect` crashed on a plain
  install with `ModuleNotFoundError: PyNvVideoCodec` — it defaults to
  `--use-nvdec True` and imports the module unconditionally — and
  `train optimize-preset` needed a separate extra.
- The `fast` extra is now **`engines`** and holds only `tensorrt` and
  `onnxruntime-gpu`, for `.engine` / `.trt` / `.onnx` weights. The old name
  also collided with the `fast` *pipeline* (`--pipeline fast`,
  `run_detection_fast`), which is core and always available. The `tune` extra
  is gone.
- Wheels bundle the default YOLO weights.
- mypy ignores missing imports for `pandas.*` and `tqdm.*`; the stub packages
  surface hundreds of errors under a non-strict config for no real safety
  win.
- Trove classifier raised to `Development Status :: 5 - Production/Stable`.
- `.gitignore` covers `runs/` and repo-root `*.txt` scratch files.

### Tests

Six new unit-test modules — `test_io.py`, `test_corner_storage.py`,
`test_crop_dataset.py`, `test_yolo_dataset.py`, `test_testset_yolo_mode.py`,
`test_optimize_image.py` — plus corner-provenance coverage in
`test_cleaning.py` and format updates in `test_detection.py`. 233 tests pass.

## v0.10.1 and earlier

See the [release history](https://github.com/Social-Evolution-and-Behavior/yoto/releases).
