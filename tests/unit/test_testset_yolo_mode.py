"""Tests for cold-start (no-ground-truth) YOLO-mode testset + preset tuning.

Covers building a testset from the ``_yolo.pkl`` sidecar when nothing has
decoded yet, and the decode-yield objective that lets ``optimize-preset`` climb
out of a zero-detection cold start.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pandas as pd
import pytest

from yoto.tuning import testset as ts
from yoto.tuning import optimize as opt

# ---------------------------------------------------------------------------
# _select_frames_by_box_count
# ---------------------------------------------------------------------------


def _counts(mapping: dict[int, int]) -> pd.Series:
    s = pd.Series(mapping)
    s.index.name = "frame"
    return s


def test_select_by_box_count_top_prefers_highest():
    counts = _counts({0: 1, 1: 9, 2: 2, 3: 8})
    # "top" strides through the highest-count pool (mirrors _select_frames);
    # the single busiest frame must be selected and lower ones excluded.
    sel = ts._select_frames_by_box_count(
        counts, sample_n=2, top_n=1000, sample_strategy="top"
    )
    assert len(sel) == 2
    assert 1 in sel  # frame with the most boxes
    assert 0 not in sel  # frame with the fewest is never picked


def test_select_by_box_count_uniform_spreads():
    counts = _counts({i: 5 for i in range(10)})
    sel = ts._select_frames_by_box_count(counts, sample_n=5, sample_strategy="uniform")
    assert len(sel) == 5
    assert sel == sorted(sel)


def test_select_by_box_count_empty():
    assert ts._select_frames_by_box_count(_counts({}), sample_n=5) == []


# ---------------------------------------------------------------------------
# _build_testset_yolo
# ---------------------------------------------------------------------------


def _make_sidecar(path, frames, boxes_per_frame=3, size=200):
    rows = []
    for f in frames:
        for k in range(boxes_per_frame):
            x1 = 10 + k * 40
            y1 = 20
            rows.append(
                {
                    "box_x1": float(x1),
                    "box_y1": float(y1),
                    "box_x2": float(x1 + 20),
                    "box_y2": float(y1 + 20),
                    "center_x": float(x1 + 10),
                    "center_y": float(y1 + 10),
                    "confidence": 0.9,
                    "decoded": False,
                    "tag_id": -1,
                }
            )
    df = pd.DataFrame(rows)
    df.index = pd.Index(
        [f for f in frames for _ in range(boxes_per_frame)], name="frame"
    )
    df.to_pickle(str(path))
    return df


def test_build_testset_yolo_writes_manifest_without_gt(tmp_path, monkeypatch):
    frames = [0, 5, 10]
    sidecar = tmp_path / "000000_apriltagDetect14_yolo.pkl"
    _make_sidecar(sidecar, frames)

    # Avoid touching a real video: hand _grab_frames synthetic frames.
    monkeypatch.setattr(
        ts,
        "_grab_frames",
        lambda video, indices, **kw: {
            f: np.full((200, 200, 3), 127, np.uint8) for f in indices
        },
    )

    out_dir = tmp_path / "testset"
    doc = ts.build_testset(
        tmp_path / "000000.mp4",
        sidecar,
        out_dir,
        "ignored_weights.pt",
        sample_per_video=10,
        gt_source="yolo",
    )

    assert doc is not None
    assert doc["gt_mode"] == "yolo"
    assert doc["tag_family"] == "unknown"  # no sibling raw pkl in tmp
    assert doc["frames"], "expected at least one composite"
    # Every crop must carry an empty ground truth.
    for entry in doc["frames"]:
        assert entry["all_frame_gt_ids"] == []
        assert entry["crops"]
        for crop in entry["crops"]:
            assert crop["gt_ids"] == []

    manifest_on_disk = out_dir / doc["video_key"] / "manifest.json"
    assert manifest_on_disk.exists()
    assert json.loads(manifest_on_disk.read_text())["gt_mode"] == "yolo"


def test_build_testset_yolo_reads_tag_family_from_raw_pkl(tmp_path, monkeypatch):
    frames = [0, 1]
    sidecar = tmp_path / "000000_apriltagDetect14_yolo.pkl"
    _make_sidecar(sidecar, frames)
    # Sibling raw pkl carries the family.
    raw = tmp_path / "000000_apriltagDetect14.pkl"
    raw_df = pd.DataFrame({"tag_id": []})
    raw_df.attrs["yoto_tag_family"] = "tagStandard41h12"
    raw_df.to_pickle(str(raw))

    monkeypatch.setattr(
        ts,
        "_grab_frames",
        lambda video, indices, **kw: {
            f: np.full((200, 200, 3), 127, np.uint8) for f in indices
        },
    )

    doc = ts.build_testset(
        tmp_path / "000000.mp4",
        sidecar,
        tmp_path / "testset",
        "ignored.pt",
        gt_source="yolo",
    )
    assert doc["tag_family"] == "tagStandard41h12"


# ---------------------------------------------------------------------------
# load_manifests yield-mode detection
# ---------------------------------------------------------------------------


def _write_manifest(testset_dir, video_key, frames, gt_mode=None):
    vdir = testset_dir / video_key
    (vdir / "composites").mkdir(parents=True, exist_ok=True)
    doc = {"video_key": video_key, "tag_family": "tag36h11", "frames": frames}
    if gt_mode:
        doc["gt_mode"] = gt_mode
    (vdir / "manifest.json").write_text(json.dumps(doc))


def _frame_entry(gt_ids):
    return {
        "composite_file": "f000000.png",
        "all_frame_gt_ids": list(gt_ids),
        "crops": [
            {
                "crop_idx": 0,
                "canvas_x_offset": 0,
                "crop_shape": [20, 20],
                "gt_ids": list(gt_ids),
            }
        ],
    }


def test_load_manifests_yield_mode_true_when_no_gt(tmp_path):
    _write_manifest(tmp_path, "v1", [_frame_entry([])], gt_mode="yolo")
    _, _, _, _, yield_mode = opt.load_manifests(tmp_path)
    assert yield_mode is True


def test_load_manifests_yield_mode_false_with_gt(tmp_path):
    _write_manifest(tmp_path, "v1", [_frame_entry([3, 7])])
    _, valid_ids, _, _, yield_mode = opt.load_manifests(tmp_path)
    assert yield_mode is False
    assert valid_ids == {3, 7}


# ---------------------------------------------------------------------------
# yield scoring
# ---------------------------------------------------------------------------


def test_yield_score_rewards_hits_penalises_multi():
    # All boxes decode once → perfect yield.
    assert opt._yield_score(10, 0, 10) == pytest.approx(1.0)
    # Nothing decodes → zero.
    assert opt._yield_score(0, 0, 10) == pytest.approx(0.0)
    # More hits scores higher.
    assert opt._yield_score(6, 0, 10) > opt._yield_score(3, 0, 10)
    # Spurious multi-decodes are penalised.
    assert opt._yield_score(10, 5, 10) < opt._yield_score(10, 0, 10)


def test_evaluate_yield_mode_is_not_degenerate(tmp_path):
    # A blank composite decodes nothing, but yield mode must still return the
    # yield metrics (not the total_with_gt==0 early bail-out that would make
    # every trial score identically).
    # _evaluate runs the real decoder, so this needs the AprilTag C extension —
    # built from source, absent on CI.
    pytest.importorskip("apriltag")
    comp_dir = tmp_path / "composites"
    comp_dir.mkdir()
    cv2.imwrite(str(comp_dir / "f000000.png"), np.zeros((40, 60, 3), np.uint8))
    entry = {
        "composite_file": "f000000.png",
        "_composites_dir": str(comp_dir),
        "all_frame_gt_ids": [],
        "crops": [
            {"crop_idx": 0, "canvas_x_offset": 0, "crop_shape": [40, 20], "gt_ids": []},
            {
                "crop_idx": 1,
                "canvas_x_offset": 20,
                "crop_shape": [40, 20],
                "gt_ids": [],
            },
        ],
    }
    params = {
        "max_hamming": 1,
        "decimate": 1.0,
        "blur": 0.0,
        "refine_edges": 1,
        "decode_sharpening": 0.25,
        "upscale": 1.0,
    }
    m = opt._evaluate([entry], set(), params, params, "tag36h11", yield_mode=True)
    assert "yield_rate" in m
    assert m["total_yield_crops"] == 2
    assert m["yield_rate"] == pytest.approx(0.0)
