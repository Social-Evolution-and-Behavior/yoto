"""Tests for `yoto train build-crop-dataset`.

Builds an image-classification dataset from the crops + manifests that
``build-testset`` already wrote.  Everything here runs on synthetic manifests
and tiny generated jpgs — no video decode, no GPU, no model.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any
from unittest import mock

import cv2
import numpy as np
import pytest

from yoto.tuning import crop_dataset as cd

# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------

ORIGINAL = 1
INTERPOLATED = 2
YOLO_INFERRED = 3


def _write_experiment(
    testset_dir: Path,
    exp: str,
    crops: list[tuple[int, int, list[int], list[int]]],
    *,
    manifest: bool = True,
) -> Path:
    """Create one experiment dir with crops + manifest.

    *crops* is a list of ``(frame, crop_idx, gt_ids, gt_ass_types)``.
    """
    exp_dir = testset_dir / f"video__{exp}"
    (exp_dir / "crops").mkdir(parents=True, exist_ok=True)

    frames: dict[int, list[dict]] = {}
    for frame, idx, gt_ids, ass_types in crops:
        crop_file = f"f{frame:06d}_c{idx:03d}.jpg"
        img = np.full((8, 8, 3), (idx * 7) % 256, dtype=np.uint8)
        cv2.imwrite(str(exp_dir / "crops" / crop_file), img)
        frames.setdefault(frame, []).append(
            {
                "crop_idx": idx,
                "crop_file": crop_file,
                "bbox_xyxy": [idx, frame, idx + 8, frame + 8],
                "crop_shape": [8, 8],
                "gt_ids": gt_ids,
                "gt_ass_types": ass_types,
            }
        )

    if manifest:
        doc = {
            "video_key": f"video__{exp}",
            "frames": [
                {"frame_idx": f, "composite_file": f"f{f:06d}.png", "crops": cs}
                for f, cs in sorted(frames.items())
            ],
        }
        (exp_dir / "manifest.json").write_text(json.dumps(doc))
    return exp_dir


def _simple_testset(tmp_path: Path, n_exp: int = 4, n_frames: int = 30) -> Path:
    """Two tag IDs present in every experiment, all ORIGINAL."""
    ts = tmp_path / "apriltag_testset"
    for e in range(n_exp):
        crops = []
        for f in range(n_frames):
            crops.append((f, 0, [10], [ORIGINAL]))
            crops.append((f, 1, [20], [ORIGINAL]))
        _write_experiment(ts, f"{e:06d}", crops)
    return ts


def _classes(split_dir: Path) -> dict[str, int]:
    return {
        d.name: len(list(d.glob("*.jpg")))
        for d in sorted(split_dir.iterdir())
        if d.is_dir()
    }


# ---------------------------------------------------------------------------
# discovery / robustness
# ---------------------------------------------------------------------------


def test_discovers_experiments_by_manifest(tmp_path):
    ts = _simple_testset(tmp_path, n_exp=3)
    assert len(cd.discover_experiments(ts)) == 3


def test_skips_experiment_dir_without_manifest(tmp_path):
    """build-testset makes the dir (and empty crops/) before writing the
    manifest, so an in-progress or interrupted run must not break the build."""
    ts = _simple_testset(tmp_path, n_exp=2)
    (ts / "video__000009" / "crops").mkdir(parents=True)  # in-progress, no manifest

    assert len(cd.discover_experiments(ts)) == 2


def test_skips_experiment_with_unparseable_manifest(tmp_path):
    ts = _simple_testset(tmp_path, n_exp=2)
    bad = ts / "video__000009"
    (bad / "crops").mkdir(parents=True)
    (bad / "manifest.json").write_text("{not json")

    res = cd.build_crop_dataset(ts, tmp_path / "out", min_count=1)
    assert "video__000009" in res["skipped_experiments"]


# ---------------------------------------------------------------------------
# label selection
# ---------------------------------------------------------------------------


def test_keeps_only_single_gt_id_crops(tmp_path):
    ts = tmp_path / "apriltag_testset"
    _write_experiment(
        ts,
        "000000",
        [
            (0, 0, [10], [ORIGINAL]),  # keep
            (0, 1, [], []),  # no tag centred -> drop
            (0, 2, [10, 20], [ORIGINAL, ORIGINAL]),  # ambiguous -> drop
        ],
    )
    recs = cd.load_crop_records(ts / "video__000000")

    assert len(recs) == 1
    assert recs[0]["tag_id"] == 10


def test_record_carries_provenance_and_geometry(tmp_path):
    ts = tmp_path / "apriltag_testset"
    _write_experiment(ts, "000000", [(7, 3, [42], [YOLO_INFERRED])])
    rec = cd.load_crop_records(ts / "video__000000")[0]

    assert rec["tag_id"] == 42
    assert rec["ass_type"] == YOLO_INFERRED
    assert rec["frame"] == 7
    assert rec["crop_idx"] == 3
    assert rec["bbox"] == [3, 7, 11, 15]
    assert Path(rec["src_path"]).exists()


# ---------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------


def test_split_holds_out_whole_experiments(tmp_path):
    ts = _simple_testset(tmp_path, n_exp=8)
    out = tmp_path / "out"
    res = cd.build_crop_dataset(ts, out, val_frac=0.2, min_count=1)

    # round(0.2 * 8) == 2 held out, taken from the end
    assert res["val_experiments"] == ["video__000006", "video__000007"]
    assert len(res["train_experiments"]) == 6

    rows = list(csv.DictReader((out / "manifest.csv").open()))
    val_exps = {r["experiment"] for r in rows if r["split"] == "val"}
    train_exps = {r["experiment"] for r in rows if r["split"] == "train"}
    assert not (val_exps & train_exps), "an experiment leaked across splits"


def test_every_class_present_in_both_splits(tmp_path):
    ts = _simple_testset(tmp_path, n_exp=8)
    out = tmp_path / "out"
    cd.build_crop_dataset(ts, out, val_frac=0.2, min_count=1)

    assert set(_classes(out / "train")) == {"10", "20"}
    assert set(_classes(out / "val")) == {"10", "20"}


def test_single_experiment_falls_back_to_tail_split(tmp_path):
    """max(1, round(0.2*1)) would hold out the only experiment and leave the
    training set empty; n == 1 must split by frame order instead."""
    ts = _simple_testset(tmp_path, n_exp=1, n_frames=10)
    out = tmp_path / "out"
    res = cd.build_crop_dataset(ts, out, val_frac=0.2, min_count=1)

    assert res["val_experiments"] == []
    assert _classes(out / "train") == {"10": 8, "20": 8}
    assert _classes(out / "val") == {"10": 2, "20": 2}


def test_val_frac_never_empties_train(tmp_path):
    ts = _simple_testset(tmp_path, n_exp=3)
    out = tmp_path / "out"
    res = cd.build_crop_dataset(ts, out, val_frac=1.0, min_count=1)

    assert len(res["train_experiments"]) >= 1


def test_class_missing_from_val_is_tail_split(tmp_path):
    """A tag ID living only in training experiments must still reach val."""
    ts = _simple_testset(tmp_path, n_exp=4, n_frames=20)
    # ID 99 appears only in the first experiment, which lands in train.
    _write_experiment(
        ts,
        "000000",
        [(f, 5, [99], [ORIGINAL]) for f in range(20)],
    )
    out = tmp_path / "out"
    res = cd.build_crop_dataset(ts, out, val_frac=0.25, min_count=1)

    assert "99" in _classes(out / "val")
    assert 99 in res["classes_tail_split"]


# ---------------------------------------------------------------------------
# provenance filtering
# ---------------------------------------------------------------------------


def _mixed_testset(tmp_path: Path, n_exp: int = 4) -> Path:
    ts = tmp_path / "apriltag_testset"
    for e in range(n_exp):
        crops = []
        for f in range(20):
            at = ORIGINAL if f % 2 == 0 else YOLO_INFERRED
            crops.append((f, 0, [10], [at]))
            crops.append((f, 1, [20], [at]))
        _write_experiment(ts, f"{e:06d}", crops)
    return ts


def test_val_is_original_only_by_default(tmp_path):
    """Scoring against trajectory-inferred labels measures agreement with the
    chaining heuristic, not correctness — val defaults to decoded tags."""
    ts = _mixed_testset(tmp_path)
    out = tmp_path / "out"
    cd.build_crop_dataset(ts, out, val_frac=0.25, min_count=1)

    rows = list(csv.DictReader((out / "manifest.csv").open()))
    val_types = {r["ass_type"] for r in rows if r["split"] == "val"}
    train_types = {r["ass_type"] for r in rows if r["split"] == "train"}

    assert val_types == {str(ORIGINAL)}
    assert train_types == {str(ORIGINAL), str(YOLO_INFERRED)}


def test_ass_types_original_excludes_inferred_from_train(tmp_path):
    ts = _mixed_testset(tmp_path)
    out = tmp_path / "out"
    cd.build_crop_dataset(ts, out, ass_types="original", val_frac=0.25, min_count=1)

    rows = list(csv.DictReader((out / "manifest.csv").open()))
    assert {r["ass_type"] for r in rows} == {str(ORIGINAL)}


def test_val_ass_types_all_admits_inferred(tmp_path):
    ts = _mixed_testset(tmp_path)
    out = tmp_path / "out"
    cd.build_crop_dataset(ts, out, val_ass_types="all", val_frac=0.25, min_count=1)

    rows = list(csv.DictReader((out / "manifest.csv").open()))
    val_types = {r["ass_type"] for r in rows if r["split"] == "val"}
    assert val_types == {str(ORIGINAL), str(YOLO_INFERRED)}


# ---------------------------------------------------------------------------
# class filtering / caps
# ---------------------------------------------------------------------------


def test_min_count_drops_undersized_classes(tmp_path):
    ts = _simple_testset(tmp_path, n_exp=4, n_frames=30)
    # Rewrite one experiment with its usual crops plus a single rare-class one.
    crops = [(f, i, [10 * (i + 1)], [ORIGINAL]) for f in range(30) for i in (0, 1)]
    _write_experiment(ts, "000000", crops + [(0, 5, [99], [ORIGINAL])])
    out = tmp_path / "out"
    res = cd.build_crop_dataset(ts, out, min_count=100)

    assert "99" not in _classes(out / "train")
    assert 99 in res["dropped_classes"]


def test_max_per_class_caps_each_split(tmp_path):
    ts = _simple_testset(tmp_path, n_exp=4, n_frames=30)
    out = tmp_path / "out"
    cd.build_crop_dataset(ts, out, val_frac=0.25, min_count=1, max_per_class=10)

    assert all(v <= 10 for v in _classes(out / "train").values())
    assert all(v <= 10 for v in _classes(out / "val").values())


def test_max_per_class_strides_rather_than_truncates(tmp_path):
    """Capping must preserve temporal coverage, not keep one contiguous run."""
    ts = _simple_testset(tmp_path, n_exp=1, n_frames=40)
    out = tmp_path / "out"
    cd.build_crop_dataset(ts, out, val_frac=0.2, min_count=1, max_per_class=4)

    rows = [
        r
        for r in csv.DictReader((out / "manifest.csv").open())
        if r["split"] == "train" and r["tag_id"] == "10"
    ]
    frames = sorted(int(r["frame"]) for r in rows)
    assert len(frames) == 4
    assert max(frames) - min(frames) > 10, f"truncated instead of strided: {frames}"


# ---------------------------------------------------------------------------
# output layout
# ---------------------------------------------------------------------------


def test_writes_imagefolder_tree_with_copies(tmp_path):
    ts = _simple_testset(tmp_path, n_exp=4)
    out = tmp_path / "out"
    cd.build_crop_dataset(ts, out, val_frac=0.25, min_count=1)

    jpg = next((out / "train" / "10").glob("*.jpg"))
    assert not jpg.is_symlink()
    assert cv2.imread(str(jpg)) is not None


def test_symlink_mode_does_not_copy(tmp_path):
    ts = _simple_testset(tmp_path, n_exp=4)
    out = tmp_path / "out"
    cd.build_crop_dataset(ts, out, val_frac=0.25, min_count=1, symlink=True)

    assert next((out / "train" / "10").glob("*.jpg")).is_symlink()


def test_parallel_and_serial_copies_agree(tmp_path):
    """Threads only overlap NFS waits — the tree must be byte-identical."""
    ts = _simple_testset(tmp_path, n_exp=4)

    cd.build_crop_dataset(ts, tmp_path / "par", val_frac=0.25, min_count=1, jobs=8)
    cd.build_crop_dataset(ts, tmp_path / "ser", val_frac=0.25, min_count=1, jobs=1)

    par = sorted(
        p.relative_to(tmp_path / "par") for p in (tmp_path / "par").rglob("*.jpg")
    )
    ser = sorted(
        p.relative_to(tmp_path / "ser") for p in (tmp_path / "ser").rglob("*.jpg")
    )
    assert par == ser
    assert (tmp_path / "par" / "manifest.csv").read_text() == (
        tmp_path / "ser" / "manifest.csv"
    ).read_text()


def _count_mkdirs(testset: Path, out: Path) -> int:
    real_mkdir = Path.mkdir
    calls: list[Any] = []

    def counting_mkdir(self, *a, **kw):
        calls.append(self)
        return real_mkdir(self, *a, **kw)

    with mock.patch.object(Path, "mkdir", counting_mkdir):
        cd.build_crop_dataset(testset, out, val_frac=0.5, min_count=1)
    return len(calls)


def test_mkdir_count_does_not_scale_with_crops(tmp_path):
    """A mkdir per crop costs an extra NFS round-trip (~3 ms) on top of copying
    a ~2 KB jpg, so directory creation must scale with classes, not files."""
    small = _count_mkdirs(
        _simple_testset(tmp_path / "a", n_exp=2, n_frames=25), tmp_path / "out_a"
    )
    large = _count_mkdirs(
        _simple_testset(tmp_path / "b", n_exp=2, n_frames=200), tmp_path / "out_b"
    )

    assert small == large, f"mkdir calls grew {small} -> {large} with 8x the crops"


def test_manifest_rows_match_emitted_files(tmp_path):
    ts = _simple_testset(tmp_path, n_exp=4)
    out = tmp_path / "out"
    cd.build_crop_dataset(ts, out, val_frac=0.25, min_count=1)

    rows = list(csv.DictReader((out / "manifest.csv").open()))
    on_disk = sum(len(list((out / s).rglob("*.jpg"))) for s in ("train", "val"))

    assert len(rows) == on_disk
    assert set(rows[0]) >= {
        "crop_path",
        "tag_id",
        "experiment",
        "frame",
        "crop_idx",
        "ass_type",
        "split",
        "src_path",
    }
    for r in rows:
        assert (out / r["crop_path"]).exists()


def test_dataset_json_reports_counts_and_provenance(tmp_path):
    ts = _mixed_testset(tmp_path)
    out = tmp_path / "out"
    cd.build_crop_dataset(ts, out, val_frac=0.25, min_count=1)

    doc = json.loads((out / "dataset.json").read_text())
    assert doc["n_classes"] == 2
    assert doc["counts"]["train"]["10"] > 0
    assert doc["provenance"]["train"][str(ORIGINAL)] > 0
    assert doc["params"]["val_frac"] == 0.25


def test_force_rebuilds_instead_of_merging(tmp_path):
    """The source pool grows as more videos are processed, so a rebuild must
    not leave crops from a previous, larger run behind."""
    ts = _simple_testset(tmp_path, n_exp=4)
    out = tmp_path / "out"
    cd.build_crop_dataset(ts, out, val_frac=0.25, min_count=1)
    stale = out / "train" / "10" / "stale_leftover.jpg"
    stale.write_bytes(b"x")

    cd.build_crop_dataset(ts, out, val_frac=0.25, min_count=1, force=True)
    assert not stale.exists()


def test_force_reports_progress_while_clearing(tmp_path):
    """Clearing a previous dataset unlinks tens of thousands of files; doing it
    silently reads as a hang before the copy bar appears."""
    ts = _simple_testset(tmp_path, n_exp=4)
    out = tmp_path / "out"
    cd.build_crop_dataset(ts, out, val_frac=0.25, min_count=1)

    with mock.patch.object(cd, "tqdm", wraps=cd.tqdm) as bar:
        cd.build_crop_dataset(ts, out, val_frac=0.25, min_count=1, force=True)

    descs = [c.kwargs.get("desc") for c in bar.call_args_list]
    assert "Removing old dataset" in descs


def test_refuses_existing_output_without_force(tmp_path):
    ts = _simple_testset(tmp_path, n_exp=4)
    out = tmp_path / "out"
    cd.build_crop_dataset(ts, out, val_frac=0.25, min_count=1)

    with pytest.raises(FileExistsError):
        cd.build_crop_dataset(ts, out, val_frac=0.25, min_count=1)


def test_empty_testset_raises(tmp_path):
    empty = tmp_path / "apriltag_testset"
    empty.mkdir()
    with pytest.raises(ValueError, match="[Nn]o experiments"):
        cd.build_crop_dataset(empty, tmp_path / "out")
