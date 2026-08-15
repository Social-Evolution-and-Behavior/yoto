"""Backend for `yoto train build-yolo-dataset`.

One module, organised into labelled sections:
  geometry · pkl/frame selection · discovery · session · export · precompute.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypedDict

import cv2
import numpy as np
import pandas as pd

from yoto.io import has_corners, load_corners
from yoto.constants import (
    PICKLE_EXTS,
    TRACKING_DIR,
    TRAINING_SUBDIR,
    default_max_tag_id_for_family,
)

#: This tool's sub-directory under ``tracking/training/``.
OUT_SUBDIR = "yolo_dataset"


def default_out_dir(recording_dir: Path) -> Path:
    """Default output/cache dir for a recording, under ``tracking/training/``.

    Mirrors the rest of yoto's layout (``<recording>/tracking/raw_data`` etc.);
    all training tools share the ``tracking/training/`` base.
    """
    return Path(recording_dir) / TRACKING_DIR / TRAINING_SUBDIR / OUT_SUBDIR


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #

AREA_PCTL = (1.0, 99.0)
ANGLE_PCTL = (1.0, 99.0)
SIDE_PCTL = (0.1, 99.9)
DEFAULT_RATIO_MIN = 0.6
DEFAULT_DEDUP_PX = 10.0


class Thresholds(TypedDict):
    area_min: float
    area_max: float
    angle_min: float
    angle_max: float
    side_min: float
    side_max: float
    ratio_min: float
    dedup_px: float


def polygon_area(quad: np.ndarray) -> float:
    pts = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    x, y = pts[:, 0], pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def corner_angles(quad: np.ndarray) -> np.ndarray:
    pts = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    angles = []
    for i in range(4):
        v1 = pts[i - 1] - pts[i]
        v2 = pts[(i + 1) % 4] - pts[i]
        cos_t = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
        angles.append(np.degrees(np.arccos(np.clip(cos_t, -1.0, 1.0))))
    return np.asarray(angles, dtype=np.float64)


def side_lengths(quad: np.ndarray) -> np.ndarray:
    pts = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    return np.linalg.norm(np.diff(np.vstack([pts, pts[:1]]), axis=0), axis=1)


def min_side_ratio(quad: np.ndarray) -> float:
    s = side_lengths(quad)
    return float(s.min() / max(s.max(), 1e-9))


def classify(
    quad: np.ndarray, thr: Thresholds, valid_centers: np.ndarray | None
) -> str:
    a = polygon_area(quad)
    if not (thr["area_min"] <= a <= thr["area_max"]):
        return "area"
    angs = corner_angles(quad)
    if not np.all((angs >= thr["angle_min"]) & (angs <= thr["angle_max"])):
        return "angle"
    s = side_lengths(quad)
    if not (s.min() >= thr["side_min"] and s.max() <= thr["side_max"]):
        return "side"
    if min_side_ratio(quad) < thr["ratio_min"]:
        return "side"
    if valid_centers is not None and len(valid_centers) > 0:
        c = np.asarray(quad, dtype=np.float64).reshape(4, 2).mean(axis=0)
        if np.linalg.norm(valid_centers - c, axis=1).min() < thr["dedup_px"]:
            return "dedup"
    return "ok"


def seed_thresholds(corners: list[np.ndarray]) -> Thresholds:
    areas, angles, sides = [], [], []
    for q in corners:
        arr = np.asarray(q, dtype=np.float64).reshape(4, 2)
        if not np.isfinite(arr).all():
            continue
        areas.append(polygon_area(arr))
        angles.extend(corner_angles(arr).tolist())
        sides.extend(side_lengths(arr).tolist())
    if not areas:
        raise ValueError("seed_thresholds: no valid corner sets provided")
    a = np.percentile(areas, AREA_PCTL)
    an = np.percentile(angles, ANGLE_PCTL)
    sd = np.percentile(sides, SIDE_PCTL)
    return Thresholds(
        area_min=float(a[0]),
        area_max=float(a[1]),
        angle_min=float(an[0]),
        angle_max=float(an[1]),
        side_min=float(sd[0]),
        side_max=float(sd[1]),
        ratio_min=DEFAULT_RATIO_MIN,
        dedup_px=DEFAULT_DEDUP_PX,
    )


# --------------------------------------------------------------------------- #
# pkl stats + frame selection
# --------------------------------------------------------------------------- #


def read_pkl_stats(pkl_path: str | Path) -> tuple[pd.Series, list[np.ndarray]]:
    """Return (tag-count-per-frame, list-of-corner-arrays).

    Supports the wide MultiIndex ``(tag_id, metric)`` schema and the long
    one-row-per-detection schema produced by newer yoto versions.
    """
    df = pd.read_pickle(pkl_path)

    if isinstance(df.columns, pd.MultiIndex):
        counts = df.loc[:, (slice(None), "center_x")].notna().sum(axis=1)
    else:
        counts = df.groupby(level=0).size()

    if has_corners(df):
        quads = load_corners(df).astype(np.float64).reshape(-1, 4, 2)
        quads = quads[np.isfinite(quads).all(axis=(1, 2))]
        corners = list(quads)
    else:
        corners = []

    return counts.astype(int), corners


def select_best_worst(counts: pd.Series, total: int, best_fraction: float) -> list[int]:
    n_best = int(round(total * best_fraction))
    n_worst = total - n_best
    best = counts.nlargest(n_best).index.tolist()
    worst = counts[counts > 0].nsmallest(n_worst).index.tolist()
    return sorted({int(f) for f in (*best, *worst)})


def select_stride(n_frames: int, total: int) -> list[int]:
    if total <= 0 or n_frames <= 0:
        return []
    step = max(1, n_frames // total)
    return list(range(0, n_frames, step))[:total]


# --------------------------------------------------------------------------- #
# experiment discovery
# --------------------------------------------------------------------------- #


@dataclass
class Experiment:
    name: str
    video: Path
    pkl: Path | None


#: Sidecar pkls that are not the decoded-tags pkl (skipped in discovery).
#: Listed for every pickle extension, since compressed and plain pickles
#: coexist across yoto versions.
_SIDECAR_SUFFIXES = tuple(
    f"{kind}{ext}" for kind in ("_yolo", "_quads") for ext in PICKLE_EXTS
)


def find_pkls_for_video(video: Path) -> list[Path]:
    raw = video.parent / "tracking" / "raw_data"
    if not raw.is_dir():
        return []
    return sorted(
        p
        for ext in PICKLE_EXTS
        for p in raw.glob(f"{video.stem}*{ext}")
        if not p.name.endswith(_SIDECAR_SUFFIXES)
    )


def _find_video(recording: Path) -> Path:
    vids = sorted(p for ext in ("*.mp4", "*.avi", "*.mkv") for p in recording.glob(ext))
    if not vids:
        raise ValueError(f"No video found in recording dir: {recording}")
    return vids[0]


def _pick_pkl(
    video: Path,
    pkls: list[Path],
    pkl_suffix: str | None,
    no_pkl: bool,
    chooser: Callable[[Path, list[Path]], Path] | None,
) -> Path | None:
    if no_pkl or not pkls:
        return None
    if len(pkls) == 1:
        return pkls[0]
    if pkl_suffix:
        matches = [p for p in pkls if pkl_suffix in p.stem]
        if len(matches) == 1:
            return matches[0]
        raise ValueError(
            f"--pkl-suffix {pkl_suffix!r} matched {len(matches)} pkls for "
            f"{video.name}: {[p.name for p in matches]}"
        )
    if chooser is not None:
        return chooser(video, pkls)
    raise ValueError(
        f"{len(pkls)} pkls match {video.name}; pass --pkl-suffix to choose: "
        f"{[p.name for p in pkls]}"
    )


def resolve_experiments(
    recordings: list[str],
    explicit: list[str],
    pkl_suffix: str | None,
    no_pkl: bool,
    chooser: Callable[[Path, list[Path]], Path] | None,
) -> list[Experiment]:
    exps: list[Experiment] = []
    for spec in explicit:
        pkl_str, _, vid_str = spec.partition(":")
        video = Path(vid_str)
        exps.append(
            Experiment(
                name=video.parent.name or video.stem,
                video=video,
                pkl=None if no_pkl or not pkl_str else Path(pkl_str),
            )
        )
    for rec_str in recordings:
        recording = Path(rec_str)
        video = _find_video(recording)
        pkl = _pick_pkl(video, find_pkls_for_video(video), pkl_suffix, no_pkl, chooser)
        exps.append(Experiment(name=recording.name, video=video, pkl=pkl))
    return exps


# --------------------------------------------------------------------------- #
# review session
# --------------------------------------------------------------------------- #


def default_session(thresholds: Thresholds) -> dict:
    return {
        "thresholds": dict(thresholds),
        "per_frame_thresholds": {},
        "frames": {},
        "export": {},
    }


def load_session(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def save_session(path: Path, session: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(session, indent=2))


def set_decision(
    session: dict,
    frame: int,
    status: str,
    overrides: dict[int, str],
    *,
    enhance: bool = False,
) -> None:
    session["frames"][str(frame)] = {
        "status": status,
        "overrides": {str(k): v for k, v in overrides.items()},
        "enhance": bool(enhance),
    }


def final_quad_indices(
    candidates_frame: list[dict],
    thr: Thresholds,
    valid_centers: np.ndarray | None,
    overrides: dict[int, str],
) -> list[int]:
    keep: set[int] = set()
    for cand in candidates_frame:
        idx = int(cand["quad_idx"])
        ov = overrides.get(idx)
        if ov == "exclude":
            continue
        if ov == "include":
            keep.add(idx)
            continue
        if classify(np.asarray(cand["corners"]), thr, valid_centers) == "ok":
            keep.add(idx)
    return sorted(keep)


# --------------------------------------------------------------------------- #
# export writers
# --------------------------------------------------------------------------- #


def xanylabeling_shape(
    corners: np.ndarray, label: str = "apriltag", score: float = 1.0
) -> dict:
    pts = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    # X-AnyLabeling rotation shapes carry a `direction`: the heading of the
    # points[0]->points[1] edge in radians, normalised to [0, 2*pi). Without it
    # the box loads flat (direction 0) and its rotation handle is wrong.
    (x1, y1), (x2, y2) = pts[0], pts[1]
    direction = float(np.arctan2(y2 - y1, x2 - x1)) % (2.0 * np.pi)
    return {
        "label": label,
        "score": float(score),
        "points": pts.tolist(),
        "group_id": None,
        "description": None,
        "difficult": False,
        "shape_type": "rotation",
        "direction": direction,
        "flags": {},
        "attributes": {},
    }


def yolo_obb_line(corners: np.ndarray, w: int, h: int, cls: int = 0) -> str:
    pts = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    norm = pts / np.array([w, h], dtype=np.float64)
    coords = " ".join(f"{v:.6f}" for v in norm.reshape(-1))
    return f"{cls} {coords}"


def yolo_axis_line(corners: np.ndarray, w: int, h: int, cls: int = 0) -> str:
    pts = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    cx = (x1 + x2) / 2 / w
    cy = (y1 + y2) / 2 / h
    return f"{cls} {cx:.6f} {cy:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"


def split_train_val(
    names: list[str], val_fraction: float, seed: int = 0
) -> tuple[list[str], list[str]]:
    shuffled = list(names)
    random.Random(seed).shuffle(shuffled)
    n_val = int(round(len(shuffled) * val_fraction))
    return sorted(shuffled[n_val:]), sorted(shuffled[:n_val])


def write_data_yaml(path: Path, class_name: str = "apriltag") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        "train: train/images\n" "val: val/images\n" "names:\n" f"  0: {class_name}\n"
    )


# --------------------------------------------------------------------------- #
# precompute + export runner
# --------------------------------------------------------------------------- #


def detect_full_frame(
    gray: np.ndarray,
    params: dict[str, Any],
    detector: Any,
    *,
    enhance: bool = False,
) -> tuple[list[dict], list[np.ndarray]]:
    """Detect tags + raw quads on a full frame.

    ``enhance=False`` (the default) runs the AprilTag detector on the raw
    grayscale frame, matching what you get testing the detector by hand.
    ``enhance=True`` applies the fast pipeline's sharpen/contrast pre-stages
    (``_enhance_and_detect``) first — more candidate quads, but the picture
    diverges from the raw frame.
    """
    from yoto.detection import _enhance_and_detect

    max_tag_id = default_max_tag_id_for_family(params.get("_family", "tag36ARTag"))
    det_params = params if enhance else {**params, "no_enhance": True}
    tags, quads = _enhance_and_detect(gray, det_params, detector, save_quads=True)
    return [t for t in tags if t["id"] <= max_tag_id], quads


def build_candidates_frame(
    quads: list[np.ndarray], valid_centers: np.ndarray
) -> list[dict]:
    rows: list[dict] = []
    for i, q in enumerate(quads):
        arr = np.asarray(q, dtype=np.float64).reshape(4, 2)
        if not np.isfinite(arr).all():
            continue
        angs = corner_angles(arr)
        sides = side_lengths(arr)
        center = arr.mean(axis=0)
        is_valid = bool(
            len(valid_centers) > 0
            and np.linalg.norm(valid_centers - center, axis=1).min() < 5.0
        )
        rows.append(
            {
                "quad_idx": i,
                "corners": arr,
                "area": polygon_area(arr),
                "angle_min": float(angs.min()),
                "angle_max": float(angs.max()),
                "side_min": float(sides.min()),
                "side_max": float(sides.max()),
                "ratio": min_side_ratio(arr),
                "is_valid": is_valid,
            }
        )
    return rows


def _default_reader(video: Path) -> Callable[[Path, int], np.ndarray | None]:
    cap = cv2.VideoCapture(str(video))

    def read(_video: Path, frame_no: int) -> np.ndarray | None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ok, frame = cap.read()
        return frame if ok else None

    return read


def precompute_experiment(
    exp: Experiment,
    out_dir: Path,
    total_frames: int,
    best_fraction: float,
    frame_select: str,
    params: dict[str, Any],
    detector: Any,
    family: str,
    *,
    video_reader: Callable[[Path, int], np.ndarray | None] | None = None,
    n_frames_hint: int | None = None,
) -> Path:
    exp_out = Path(out_dir) / exp.name
    (exp_out / "images").mkdir(parents=True, exist_ok=True)

    seed_corners: list[np.ndarray] = []
    if exp.pkl is not None and frame_select in ("auto", "best-worst"):
        counts, seed_corners = read_pkl_stats(exp.pkl)
        frame_ids = select_best_worst(counts, total_frames, best_fraction)
    else:
        n = n_frames_hint
        if n is None:
            cap = cv2.VideoCapture(str(exp.video))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        frame_ids = select_stride(n, total_frames)

    reader = video_reader or _default_reader(exp.video)
    params = {**params, "_family": family}

    all_rows: list[dict] = []
    frame_stats: dict[str, int] = {}
    img_h = img_w = 0
    valid_corner_pool: list[np.ndarray] = []
    for fnum in frame_ids:
        frame = reader(exp.video, fnum)
        if frame is None:
            continue
        img_h, img_w = frame.shape[:2]
        cv2.imwrite(str(exp_out / "images" / f"frame_{fnum:06d}.jpg"), frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        valid, quads = detect_full_frame(gray, params, detector)
        valid_centers = (
            np.array([t["center"] for t in valid], dtype=np.float64)
            if valid
            else np.zeros((0, 2))
        )
        valid_corner_pool.extend(
            np.asarray(t["lb-rb-rt-lt"], dtype=np.float64).reshape(4, 2) for t in valid
        )
        rows = build_candidates_frame(quads, valid_centers)
        for r in rows:
            r["frame"] = int(fnum)
        all_rows.extend(rows)
        frame_stats[str(fnum)] = len(valid)

    pd.DataFrame(all_rows).to_pickle(exp_out / "candidates.pkl")

    corner_source = seed_corners if seed_corners else valid_corner_pool
    try:
        thr = seed_thresholds(corner_source)
    except ValueError:
        thr = Thresholds(
            area_min=100.0,
            area_max=1000.0,
            angle_min=70.0,
            angle_max=110.0,
            side_min=8.0,
            side_max=40.0,
            ratio_min=DEFAULT_RATIO_MIN,
            dedup_px=DEFAULT_DEDUP_PX,
        )
    (exp_out / "meta.json").write_text(
        json.dumps(
            {
                "image_width": img_w,
                "image_height": img_h,
                "family": family,
                "thresholds": dict(thr),
                "frame_stats": frame_stats,
                "frames": [int(f) for f in frame_ids],
            },
            indent=2,
        )
    )
    return exp_out


def run_export(
    cache: dict[str, dict],
    session: dict,
    thr: Thresholds,
    out_dir: Path,
    formats: list[str],
    val_fraction: float,
    *,
    redetect: Callable[[str, int], list[dict]] | None = None,
) -> int:
    """Write the accepted frames as a YOLO dataset.

    ``redetect`` — when a frame's decision was made with the *Enhance* toggle
    on, its cached (raw) candidates no longer match what the reviewer saw. If
    provided, ``redetect(exp, frame)`` re-detects that frame with enhancement
    and returns candidate dicts (``quad_idx`` + ``corners``) so the exported
    labels are the enhanced quads the reviewer actually curated.
    """
    ds = Path(out_dir) / "dataset"
    accepted: list[tuple[str, int, dict]] = []
    for exp in cache:
        for f, dec in session["frames"].items():
            if dec.get("status") == "accepted":
                accepted.append((exp, int(f), dec))

    names = [f"{e}_{f:06d}" for e, f, _ in accepted]
    _, val = split_train_val(names, val_fraction)
    written = 0
    for exp, frame, dec in accepted:
        c = cache[exp]
        if dec.get("enhance") and redetect is not None:
            cands = redetect(exp, frame)
        else:
            sub = c["df"][c["df"]["frame"] == frame]
            cands = [
                {"quad_idx": int(r.quad_idx), "corners": np.asarray(r.corners)}
                for r in sub.itertuples()
            ]
        overrides = {int(k): v for k, v in dec.get("overrides", {}).items()}
        keep = final_quad_indices(cands, thr, None, overrides)
        quads = [
            np.asarray(cands[i]["corners"]).reshape(4, 2)
            for i in range(len(cands))
            if cands[i]["quad_idx"] in keep
        ]
        w, h = c["meta"]["image_width"], c["meta"]["image_height"]
        name = f"{exp}_{frame:06d}"
        split = "val" if name in val else "train"
        img_dir = ds / split / "images"
        lbl_dir = ds / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        src_img = c["dir"] / "images" / f"frame_{frame:06d}.jpg"
        (img_dir / f"{name}.jpg").write_bytes(Path(src_img).read_bytes())
        if "yolo_obb" in formats:
            (lbl_dir / f"{name}.txt").write_text(
                "\n".join(yolo_obb_line(q, w, h) for q in quads) + "\n"
            )
        elif "yolo_axis" in formats:
            (lbl_dir / f"{name}.txt").write_text(
                "\n".join(yolo_axis_line(q, w, h) for q in quads) + "\n"
            )
        if "xanylabeling" in formats:
            (img_dir / f"{name}.json").write_text(
                json.dumps(
                    {
                        "version": "3.3.5",
                        "flags": {},
                        "shapes": [xanylabeling_shape(q) for q in quads],
                        "imagePath": f"{name}.jpg",
                        "imageData": None,
                        "imageHeight": h,
                        "imageWidth": w,
                    },
                    indent=2,
                )
            )
        written += 1
    write_data_yaml(ds / "data.yaml")
    return written
