from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from yoto.tuning.yolo_dataset import builder as bd

UNIT_SQUARE = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])


def test_polygon_area_square():
    assert bd.polygon_area(UNIT_SQUARE) == pytest.approx(100.0)


def test_corner_angles_square_all_90():
    assert np.allclose(bd.corner_angles(UNIT_SQUARE), 90.0, atol=1e-6)


def test_side_lengths_and_ratio_square():
    assert np.allclose(bd.side_lengths(UNIT_SQUARE), 10.0)
    assert bd.min_side_ratio(UNIT_SQUARE) == pytest.approx(1.0)


def test_classify_ok_for_square_in_range():
    thr = bd.Thresholds(
        area_min=50,
        area_max=200,
        angle_min=80,
        angle_max=100,
        side_min=5,
        side_max=15,
        ratio_min=0.6,
        dedup_px=10.0,
    )
    assert bd.classify(UNIT_SQUARE, thr, None) == "ok"


def test_classify_rejects_on_area_then_angle_then_dedup():
    base = dict(
        area_min=50,
        area_max=200,
        angle_min=80,
        angle_max=100,
        side_min=5,
        side_max=15,
        ratio_min=0.6,
        dedup_px=10.0,
    )
    assert (
        bd.classify(UNIT_SQUARE, bd.Thresholds(**{**base, "area_max": 50}), None)
        == "area"
    )
    # area 100 (in range) but a long thin rectangle -> fails on side, not area
    sliver = np.array([[0.0, 0.0], [20.0, 0.0], [20.0, 5.0], [0.0, 5.0]])
    assert bd.classify(sliver, bd.Thresholds(**base), None) in ("angle", "side")
    centers = np.array([[5.0, 5.0]])
    assert bd.classify(UNIT_SQUARE, bd.Thresholds(**base), centers) == "dedup"


def test_seed_thresholds_from_squares():
    corners = [UNIT_SQUARE * s for s in (0.9, 1.0, 1.1)]
    thr = bd.seed_thresholds(corners)
    assert thr["area_min"] <= 100.0 <= thr["area_max"]
    assert thr["ratio_min"] == bd.DEFAULT_RATIO_MIN
    assert thr["dedup_px"] == bd.DEFAULT_DEDUP_PX


def _wide_pkl(tmp_path):
    corners = np.zeros((4, 2)).tolist()
    cols = pd.MultiIndex.from_tuples(
        [(1, "center_x"), (1, "corners"), (2, "center_x"), (2, "corners")]
    )
    df = pd.DataFrame(
        {
            (1, "center_x"): [1.0, np.nan, 3.0],
            (1, "corners"): [corners, None, corners],
            (2, "center_x"): [1.0, np.nan, np.nan],
            (2, "corners"): [corners, None, None],
        },
        index=[0, 1, 2],
    )
    df.columns = cols
    p = tmp_path / "wide.pkl"
    df.to_pickle(p)
    return p


def _long_pkl(tmp_path):
    corners = np.zeros((4, 2)).tolist()
    df = pd.DataFrame(
        {"tag_id": [1, 2, 1], "center_x": [1.0, 1.0, 3.0], "corners": [corners] * 3},
        index=[0, 0, 2],
    )
    df.index.name = "frame"
    p = tmp_path / "long.pkl"
    df.to_pickle(p)
    return p


def test_read_pkl_stats_wide(tmp_path):
    counts, corners = bd.read_pkl_stats(_wide_pkl(tmp_path))
    assert counts.loc[0] == 2 and counts.loc[1] == 0 and counts.loc[2] == 1
    assert len(corners) == 3


def test_read_pkl_stats_long(tmp_path):
    counts, corners = bd.read_pkl_stats(_long_pkl(tmp_path))
    assert counts.loc[0] == 2 and counts.loc[2] == 1
    assert len(corners) == 3


def test_select_best_worst():
    counts = pd.Series({0: 10, 1: 0, 2: 5, 3: 1, 4: 8, 5: 2})
    picked = bd.select_best_worst(counts, total=3, best_fraction=1 / 3.0)
    assert 0 in picked and 3 in picked and 5 in picked
    assert 1 not in picked


def test_select_stride():
    assert bd.select_stride(n_frames=100, total=5) == [0, 20, 40, 60, 80]


def _make_recording(tmp_path, stem="000028", suffixes=("apriltagDetect14",)):
    rec = tmp_path / "rec1"
    (rec / "tracking" / "raw_data").mkdir(parents=True)
    (rec / f"{stem}.mp4").write_bytes(b"x")
    for suf in suffixes:
        (rec / "tracking" / "raw_data" / f"{stem}_{suf}.pkl").write_bytes(b"x")
    return rec


def test_find_pkls_for_video(tmp_path):
    rec = _make_recording(tmp_path, suffixes=("a", "b"))
    assert len(bd.find_pkls_for_video(rec / "000028.mp4")) == 2


def test_resolve_single_pkl(tmp_path):
    rec = _make_recording(tmp_path)
    exps = bd.resolve_experiments([str(rec)], [], None, False, None)
    assert len(exps) == 1 and exps[0].pkl is not None and exps[0].video.exists()


def test_resolve_multi_pkl_uses_suffix(tmp_path):
    rec = _make_recording(tmp_path, suffixes=("apriltagDetect14", "test_run"))
    exps = bd.resolve_experiments([str(rec)], [], "test_run", False, None)
    assert exps[0].pkl.name.endswith("test_run.pkl")


def test_resolve_multi_pkl_ambiguous_raises(tmp_path):
    rec = _make_recording(tmp_path, suffixes=("a", "b"))
    with pytest.raises(ValueError):
        bd.resolve_experiments([str(rec)], [], None, False, None)


def test_resolve_no_pkl(tmp_path):
    rec = _make_recording(tmp_path, suffixes=("a", "b"))
    exps = bd.resolve_experiments([str(rec)], [], None, True, None)
    assert exps[0].pkl is None


def _thr():
    return bd.Thresholds(
        area_min=50,
        area_max=200,
        angle_min=80,
        angle_max=100,
        side_min=5,
        side_max=15,
        ratio_min=0.6,
        dedup_px=10.0,
    )


def test_session_roundtrip(tmp_path):
    s = bd.default_session(_thr())
    bd.set_decision(s, 7, "accepted", {3: "include"})
    p = tmp_path / "session.json"
    bd.save_session(p, s)
    loaded = bd.load_session(p)
    assert loaded["frames"]["7"]["status"] == "accepted"
    assert loaded["frames"]["7"]["overrides"]["3"] == "include"


def test_set_decision_persists_enhance():
    s = bd.default_session(_thr())
    bd.set_decision(s, 4, "accepted", {}, enhance=True)
    assert s["frames"]["4"]["enhance"] is True
    bd.set_decision(s, 5, "accepted", {})
    assert s["frames"]["5"]["enhance"] is False


def test_detect_full_frame_enhance_flag(monkeypatch):
    import yoto.detection as det

    captured: dict = {}

    def fake(gray, params, detector, save_quads=False):
        captured["params"] = params
        return [], []

    monkeypatch.setattr(det, "_enhance_and_detect", fake)
    gray = np.zeros((16, 16), dtype=np.uint8)

    bd.detect_full_frame(gray, {"_family": "tag36ARTag"}, object(), enhance=False)
    assert captured["params"].get("no_enhance") is True

    bd.detect_full_frame(gray, {"_family": "tag36ARTag"}, object(), enhance=True)
    assert not captured["params"].get("no_enhance")


def test_run_export_uses_redetect_for_enhanced_frames(tmp_path):
    cv2 = __import__("cv2")
    exp_dir = tmp_path / "rec1"
    (exp_dir / "images").mkdir(parents=True)
    cv2.imwrite(
        str(exp_dir / "images" / "frame_000000.jpg"),
        np.zeros((32, 32, 3), dtype=np.uint8),
    )
    tiny = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    cache = {
        "rec1": {
            "df": pd.DataFrame([{"frame": 0, "quad_idx": 0, "corners": tiny}]),
            "meta": {"image_width": 32, "image_height": 32},
            "dir": exp_dir,
        }
    }
    session = bd.default_session(_thr())
    bd.set_decision(session, 0, "accepted", {0: "include"}, enhance=True)

    calls = []

    def redetect(exp, frame):
        calls.append((exp, frame))
        return [{"quad_idx": 0, "corners": UNIT_SQUARE}]

    n = bd.run_export(
        cache, session, _thr(), tmp_path / "out", ["yolo_obb"], 0.0, redetect=redetect
    )
    assert calls == [("rec1", 0)] and n == 1
    # Exported label is the re-detected UNIT_SQUARE (side 10 -> 10/32), not `tiny`.
    lbl = (
        tmp_path / "out" / "dataset" / "train" / "labels" / "rec1_000000.txt"
    ).read_text()
    assert "0.312500" in lbl


def test_final_quad_indices_with_overrides():
    good = UNIT_SQUARE
    bad = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    cands = [{"quad_idx": 0, "corners": good}, {"quad_idx": 1, "corners": bad}]
    thr = _thr()
    assert bd.final_quad_indices(cands, thr, None, {}) == [0]
    got = bd.final_quad_indices(cands, thr, None, {1: "include", 0: "exclude"})
    assert got == [1]


def test_xanylabeling_shape():
    s = bd.xanylabeling_shape(UNIT_SQUARE)
    assert s["label"] == "apriltag" and s["shape_type"] == "rotation"
    assert len(s["points"]) == 4
    # UNIT_SQUARE's first edge (0,0)->(10,0) is horizontal -> direction 0.
    assert s["direction"] == pytest.approx(0.0)


def test_xanylabeling_shape_direction_from_first_edge():
    # First edge points straight up: (0,0)->(0,10) -> pi/2.
    rot = np.array([[0.0, 0.0], [0.0, 10.0], [-10.0, 10.0], [-10.0, 0.0]])
    assert bd.xanylabeling_shape(rot)["direction"] == pytest.approx(np.pi / 2)


def test_yolo_obb_line_normalized():
    parts = bd.yolo_obb_line(UNIT_SQUARE, w=100, h=100, cls=0).split()
    assert parts[0] == "0" and len(parts) == 9
    assert all(0.0 <= float(v) <= 1.0 for v in parts[1:])


def test_yolo_axis_line_center_and_size():
    cls, cx, cy, bw, bh = bd.yolo_axis_line(UNIT_SQUARE, w=100, h=100).split()
    assert cls == "0"
    assert float(cx) == pytest.approx(0.05) and float(bw) == pytest.approx(0.10)


def test_split_train_val_deterministic():
    names = [f"f{i}" for i in range(10)]
    tr, va = bd.split_train_val(names, val_fraction=0.2, seed=0)
    assert len(va) == 2 and len(tr) == 8
    assert bd.split_train_val(names, 0.2, 0) == (tr, va)


def test_write_data_yaml(tmp_path):
    p = tmp_path / "data.yaml"
    bd.write_data_yaml(p)
    text = p.read_text()
    assert "apriltag" in text and "names" in text


def test_build_candidates_frame_marks_valid():
    quads = [UNIT_SQUARE, UNIT_SQUARE + 100.0]
    valid_centers = np.array([[5.0, 5.0]])
    rows = bd.build_candidates_frame(quads, valid_centers)
    assert rows[0]["is_valid"] is True and rows[1]["is_valid"] is False
    assert rows[0]["area"] == pytest.approx(100.0)


def test_precompute_experiment_writes_cache(tmp_path, monkeypatch):
    def reader(video, frame_no):
        return np.zeros((64, 64, 3), dtype=np.uint8)

    def fake_detect(gray, params, detector):
        tag = {"id": 1, "center": (5.0, 5.0), "lb-rb-rt-lt": UNIT_SQUARE}
        return [tag], [UNIT_SQUARE, UNIT_SQUARE + 20.0]

    monkeypatch.setattr(bd, "detect_full_frame", fake_detect)

    exp = bd.Experiment(name="rec1", video=tmp_path / "v.mp4", pkl=None)
    out = bd.precompute_experiment(
        exp,
        tmp_path / "out",
        total_frames=3,
        best_fraction=1 / 3.0,
        frame_select="stride",
        params={},
        detector=object(),
        family="tag36ARTag",
        video_reader=reader,
        n_frames_hint=3,
    )
    assert (out / "candidates.pkl").exists() and (out / "meta.json").exists()
    df = pd.read_pickle(out / "candidates.pkl")
    assert {"quad_idx", "corners", "area", "is_valid"}.issubset(df.columns)


from fastapi.testclient import TestClient  # noqa: E402

from yoto.tuning.yolo_dataset import server as srv  # noqa: E402


def _seed_cache(tmp_path):
    exp_out = tmp_path / "rec1"
    (exp_out / "images").mkdir(parents=True)
    cv2 = __import__("cv2")
    cv2.imwrite(
        str(exp_out / "images" / "frame_000000.jpg"),
        np.zeros((32, 32, 3), dtype=np.uint8),
    )
    pd.DataFrame(
        [
            {
                "frame": 0,
                "quad_idx": 0,
                "corners": UNIT_SQUARE,
                "area": 100.0,
                "angle_min": 90.0,
                "angle_max": 90.0,
                "side_min": 10.0,
                "side_max": 10.0,
                "ratio": 1.0,
                "is_valid": False,
            }
        ]
    ).to_pickle(exp_out / "candidates.pkl")
    (exp_out / "meta.json").write_text(
        json.dumps(
            {
                "image_width": 32,
                "image_height": 32,
                "thresholds": {
                    "area_min": 50,
                    "area_max": 200,
                    "angle_min": 80,
                    "angle_max": 100,
                    "side_min": 5,
                    "side_max": 15,
                    "ratio_min": 0.6,
                    "dedup_px": 10.0,
                },
                "frame_stats": {"0": 0},
                "frames": [0],
            }
        )
    )
    return tmp_path


def test_server_frames_and_decision(tmp_path):
    out = _seed_cache(tmp_path)
    client = TestClient(srv.build_app(out))
    frames = client.get("/api/frames").json()
    assert frames[0]["exp"] == "rec1"
    r = client.put(
        "/api/frame/rec1/0/decision",
        json={"status": "accepted", "overrides": {"0": "include"}},
    )
    assert r.status_code == 200 and (out / "session.json").exists()


def test_server_reseeds_thresholds_from_meta_over_stale_session(tmp_path):
    out = _seed_cache(tmp_path)  # meta thresholds have area_min=50, side_min=5
    # A stale session.json with wild thresholds but a real frame decision.
    (out / "session.json").write_text(
        json.dumps(
            {
                "thresholds": {
                    "area_min": 300,
                    "area_max": 726,
                    "angle_min": 30,
                    "angle_max": 150,
                    "side_min": 72,
                    "side_max": 120,
                    "ratio_min": 0,
                    "dedup_px": 0,
                },
                "per_frame_thresholds": {},
                "frames": {"0": {"status": "accepted", "overrides": {}}},
                "export": {},
            }
        )
    )
    client = TestClient(srv.build_app(out))
    thr = client.get("/api/thresholds").json()
    # Sliders default to the pkl-measured bounds, not the stale session values.
    assert thr["side_min"] == 5 and thr["area_min"] == 50
    # ... but the accept decision from the session is preserved.
    frames = client.get("/api/frames").json()
    assert frames[0]["status"] == "accepted"


def test_server_thresholds_default_returns_pkl_bounds(tmp_path):
    out = _seed_cache(tmp_path)  # meta thresholds: area_min=50, side_min=5
    client = TestClient(srv.build_app(out))
    # Tune the live thresholds away from the pkl bounds.
    client.put("/api/thresholds", json={"side_min": 40})
    assert client.get("/api/thresholds").json()["side_min"] == 40
    # The default endpoint still reports the pkl-measured bounds (Reset target).
    assert client.get("/api/thresholds/default").json()["side_min"] == 5


def test_server_export_writes_dataset(tmp_path):
    out = _seed_cache(tmp_path)
    client = TestClient(srv.build_app(out))
    client.put(
        "/api/frame/rec1/0/decision",
        json={"status": "accepted", "overrides": {"0": "include"}},
    )
    r = client.post("/api/export", json={"formats": ["yolo_obb"], "val_fraction": 0.0})
    assert r.status_code == 200
    labels = list((out / "dataset").rglob("labels/*.txt"))
    assert labels and labels[0].read_text().split()[0] == "0"


import argparse  # noqa: E402

from yoto import cli  # noqa: E402


def test_cli_registers_build_yolo_dataset():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    cli._add_train_parser(sub)
    args = parser.parse_args(
        [
            "train",
            "build-yolo-dataset",
            "--recording",
            "/tmp/rec",
            "--out-dir",
            "/tmp/out",
            "--precompute-only",
            "True",
        ]
    )
    assert args.train_command == "build-yolo-dataset"
    assert args.recording == ["/tmp/rec"]
    assert args.precompute_only is True


def test_default_out_dir():
    from pathlib import Path

    assert bd.default_out_dir(Path("/data/rec")) == Path(
        "/data/rec/tracking/training/yolo_dataset"
    )


def test_cli_build_yolo_dataset_positional_and_default_outdir():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    cli._add_train_parser(sub)
    args = parser.parse_args(["train", "build-yolo-dataset", "/data/rec"])
    assert args.recordings == ["/data/rec"]
    assert args.out_dir is None


def test_find_pkls_excludes_sidecars(tmp_path):
    from pathlib import Path

    rec = tmp_path / "rec2"
    raw = rec / "tracking" / "raw_data"
    raw.mkdir(parents=True)
    (rec / "000000.mp4").write_bytes(b"x")
    for n in ["000000_run.pkl", "000000_run_yolo.pkl", "000000_run_quads.pkl"]:
        (raw / n).write_bytes(b"x")
    pkls = bd.find_pkls_for_video(Path(rec) / "000000.mp4")
    assert [p.name for p in pkls] == ["000000_run.pkl"]


def test_server_accept_all(tmp_path):
    out = _seed_cache(tmp_path)
    client = TestClient(srv.build_app(out))
    r = client.post("/api/accept-all")
    assert r.status_code == 200 and r.json()["accepted"] == 1
    frames = client.get("/api/frames").json()
    assert all(f["status"] == "accepted" for f in frames)
