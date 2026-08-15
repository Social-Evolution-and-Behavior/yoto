"""Build an AprilTag optimisation testset from cleaned YOTO pickles.

Composites and crops are produced with the same layout logic as the
detection pipeline (:func:`yoto.detection._compute_crop_layout` +
``pad_ratio``), so images in the testset are pixel-identical to what
``yoto detect`` would produce on the same frames.

Output layout (one folder per video, never overwritten unless ``force=True``)::

    out_dir/
      <experiment>__<video_stem>/
        manifest.json
        composites/          # one BGR PNG per sampled frame (lossless)
          f000127.png
          ...
        crops/               # individual crops (for inspection)
          f000127_c000.jpg
          ...
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from yoto.constants import DEFAULT_BATCH_SIZE, DEFAULT_CONF_THRESHOLD, DEFAULT_PAD_RATIO
from yoto.detection import _compute_crop_layout
from yoto.io import find_pickle, strip_pickle_ext

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_video_key(video_path: str | Path) -> str:
    vp = Path(video_path).resolve()
    return f"{vp.parent.name}__{vp.stem}"


def _select_frames(
    frame_data: pd.DataFrame,
    id_list: list[Any],
    top_n: int = 1000,
    sample_n: int = 50,
    min_detection_frac: float = 0.8,
    sample_strategy: str = "mixed",
) -> list[int]:
    """Pick a sample of frames to include in the testset.

    Parameters
    ----------
    sample_strategy:
        ``"top"``    — only the highest-visibility frames (original behaviour;
                       biases the testset toward easy cases).
        ``"uniform"`` — uniformly spaced across the full video; sees all
                        difficulty levels.
        ``"mixed"``  — half top-visibility + half uniform (default); covers
                       easy frames while also forcing the optimizer to handle
                       harder ones.
    """
    totaltag = frame_data.loc[:, (slice(None), "center_x")].notna().sum(axis=1)
    all_indices = list(frame_data.index)

    if sample_strategy == "top":
        min_count = int(len(id_list) * min_detection_frac)
        good = totaltag[totaltag >= min_count]
        if len(good) < sample_n:
            good = totaltag.nlargest(min(top_n, len(totaltag)))
        top = good.nlargest(min(top_n, len(good))).index.tolist()
        if len(top) <= sample_n:
            return sorted(top)
        step = max(1, len(top) // sample_n)
        return sorted(top[::step][:sample_n])

    elif sample_strategy == "uniform":
        step = max(1, len(all_indices) // sample_n)
        return sorted(all_indices[::step][:sample_n])

    else:  # "mixed": half top-visibility, half uniform
        n_top = sample_n // 2
        n_uniform = sample_n - n_top

        min_count = int(len(id_list) * min_detection_frac)
        good = totaltag[totaltag >= min_count]
        if len(good) < n_top:
            good = totaltag.nlargest(min(top_n, len(totaltag)))
        top = good.nlargest(min(top_n, len(good))).index.tolist()
        step = max(1, len(top) // n_top)
        top_sample = set(top[::step][:n_top])

        # Uniform sample from the rest of the video
        remaining = [i for i in all_indices if i not in top_sample]
        if remaining:
            step2 = max(1, len(remaining) // n_uniform)
            uniform_sample = set(remaining[::step2][:n_uniform])
        else:
            uniform_sample = set()

        return sorted(top_sample | uniform_sample)


def _grab_frames(
    video_path: str | Path,
    indices: list[int],
    seek_gap: int = 30,
) -> dict[int, np.ndarray]:
    """Grab specific frames from a video, seeking when gaps are large."""
    indices = sorted(set(indices))
    results: dict[int, np.ndarray] = {}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    prev: int | None = None
    for target in indices:
        if prev is None or (target - prev) > seek_gap:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            ret, frame = cap.read()
            if ret:
                results[target] = frame
        else:
            fn = prev + 1
            while fn <= target:
                ret, frame = cap.read()
                if not ret:
                    break
                if fn == target:
                    results[target] = frame
                fn += 1
        prev = target
    cap.release()
    return results


def _gt_ids_in_crop(
    frame_data: pd.DataFrame,
    frame_idx: int,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    id_list: list[Any],
) -> list[tuple[int, float, float, int]]:
    """Return (tag_id, cx, cy, ass_type) for every tag centred in the crop."""
    row = frame_data.loc[frame_idx]
    out = []
    for tag_id in id_list:
        cx = row.get((tag_id, "center_x"), np.nan)
        cy = row.get((tag_id, "center_y"), np.nan)
        if pd.isna(cx) or pd.isna(cy):
            continue
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            ass_type = int(row.get((tag_id, "ass_type"), 0))
            out.append((int(tag_id), float(cx), float(cy), ass_type))
    return out


def _select_frames_by_box_count(
    counts: pd.Series,
    sample_n: int = 50,
    top_n: int = 1000,
    sample_strategy: str = "mixed",
) -> list[int]:
    """Pick frames for a YOLO-mode testset, ranked by number of YOLO boxes.

    This is the no-ground-truth analogue of :func:`_select_frames`: with no
    decoded tags there is no "visibility" to rank on, so we rank by how many
    YOLO boxes (tag candidates) each frame contains.

    Parameters
    ----------
    counts:
        Series indexed by frame number, value = number of YOLO boxes in that
        frame (frames with zero boxes are absent).
    """
    all_frames = list(counts.index)
    if not all_frames:
        return []

    if sample_strategy == "uniform":
        step = max(1, len(all_frames) // sample_n)
        return sorted(all_frames[::step][:sample_n])

    top = counts.nlargest(min(top_n, len(counts))).index.tolist()

    if sample_strategy == "top":
        if len(top) <= sample_n:
            return sorted(top)
        step = max(1, len(top) // sample_n)
        return sorted(top[::step][:sample_n])

    # "mixed": half highest-box-count, half uniform across the video
    n_top = sample_n // 2
    n_uniform = sample_n - n_top
    step = max(1, len(top) // max(n_top, 1))
    top_sample = set(top[::step][:n_top])

    remaining = [f for f in all_frames if f not in top_sample]
    if remaining:
        step2 = max(1, len(remaining) // max(n_uniform, 1))
        uniform_sample = set(remaining[::step2][:n_uniform])
    else:
        uniform_sample = set()

    return sorted(top_sample | uniform_sample)


def _build_testset_yolo(
    video_path: str | Path,
    yolo_sidecar: str | Path,
    out_dir: str | Path,
    *,
    sample_per_video: int = 50,
    top_n: int = 1000,
    pad_ratio: float = DEFAULT_PAD_RATIO,
    sample_strategy: str = "mixed",
    force: bool = False,
) -> dict[str, Any] | None:
    """Build a testset from the ``_yolo.pkl`` sidecar when no clean pickle exists.

    This is the cold-start path: YOLO already knows *where* every tag is (the
    sidecar boxes), so we can build composites/crops for preset optimisation
    even though nothing has decoded yet.  Every crop is written with an empty
    ``gt_ids`` list — there is **no ground truth** — and the manifest is stamped
    ``gt_mode="yolo"`` so :func:`yoto.tuning.optimize_preset` switches to its
    decode-yield objective.

    The YOLO boxes come straight from the sidecar (no YOLO model is loaded), so
    ``--yoloweights`` is not needed in this mode.
    """
    video_path = Path(video_path)
    yolo_sidecar = Path(yolo_sidecar)
    out_dir = Path(out_dir)
    video_key = _make_video_key(video_path)

    vid_dir = out_dir / video_key
    comp_dir = vid_dir / "composites"
    crops_dir = vid_dir / "crops"
    manifest_path = vid_dir / "manifest.json"

    if manifest_path.exists() and not force:
        print(f"  SKIP (already exists): {vid_dir}")
        return None
    if force and manifest_path.exists():
        manifest_path.unlink()

    comp_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    print("  *** YOLO MODE — no ground truth (decode yield only) ***")
    print(f"  Video key: {video_key}")
    print(f"  Sidecar:   {yolo_sidecar}")

    boxes_df: pd.DataFrame = pd.read_pickle(str(yolo_sidecar))
    if boxes_df.empty:
        print("  SKIP: YOLO sidecar has no boxes")
        return None

    # Recover the tag family from the sibling raw detect pickle if present
    # (<stem> next to <stem>_yolo); optimize --tag-family overrides it.
    tag_family = "unknown"
    base = strip_pickle_ext(str(yolo_sidecar))
    if base.endswith("_yolo"):
        base = base[: -len("_yolo")]
    raw_pkl = find_pickle(base)
    if raw_pkl is not None:
        try:
            raw_df = pd.read_pickle(raw_pkl)
            tag_family = raw_df.attrs.get("yoto_tag_family", "unknown")
        except Exception:
            pass

    counts = boxes_df.groupby(level=0).size()
    selected = _select_frames_by_box_count(
        counts, sample_n=sample_per_video, top_n=top_n, sample_strategy=sample_strategy
    )
    print(f"  Selected {len(selected)} frames (ranked by YOLO box count)")
    frames_dict = _grab_frames(video_path, selected)
    grabbed = [f for f in selected if f in frames_dict]
    print(f"  Grabbed {len(grabbed)}/{len(selected)} frames")

    box_cols = ["box_x1", "box_y1", "box_x2", "box_y2"]
    manifest: list[dict[str, Any]] = []

    for frame_idx in tqdm(grabbed, desc=f"  {video_key} [crops]", mininterval=1):
        frame = frames_dict[frame_idx]
        H, W = frame.shape[:2]
        rows = boxes_df.loc[[frame_idx]]
        boxes_np = rows[box_cols].to_numpy(dtype=float)
        bounds, offsets_xy, canvas_x_offsets, composite_size = _compute_crop_layout(
            boxes_np, H, W, pad_ratio
        )
        if not bounds:
            continue

        strip_h, strip_w = composite_size
        composite = np.zeros((strip_h, strip_w, 3), dtype=np.uint8)
        crops_bgr: list[np.ndarray] = []
        for (y1, y2, x1, x2), dst_x in zip(bounds, canvas_x_offsets):
            crop = frame[y1:y2, x1:x2]
            composite[0 : crop.shape[0], dst_x : dst_x + crop.shape[1]] = crop
            crops_bgr.append(crop)

        comp_name = f"f{frame_idx:06d}.png"
        cv2.imwrite(str(comp_dir / comp_name), composite)

        crop_entries: list[dict[str, Any]] = []
        for j, ((y1, y2, x1, x2), (ox, oy), dst_x, crop) in enumerate(
            zip(bounds, offsets_xy, canvas_x_offsets, crops_bgr)
        ):
            crop_file = f"f{frame_idx:06d}_c{j:03d}.jpg"
            crop_entries.append(
                {
                    "crop_idx": j,
                    "crop_file": crop_file,
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "crop_shape": [crop.shape[0], crop.shape[1]],
                    "canvas_x_offset": dst_x,
                    "frame_offset_xy": [int(ox), int(oy)],
                    "gt_ids": [],
                    "gt_positions_in_frame": [],
                    "gt_positions_in_crop": [],
                    "gt_positions_in_composite": [],
                    "gt_ass_types": [],
                }
            )
            cv2.imwrite(str(crops_dir / crop_file), crop)

        manifest.append(
            {
                "composite_file": comp_name,
                "frame_idx": int(frame_idx),
                "composite_shape": [composite.shape[0], composite.shape[1]],
                "n_crops": len(crops_bgr),
                "n_total_ids_in_frame": 0,
                "all_frame_gt_ids": [],
                "crops": crop_entries,
            }
        )

    manifest_doc: dict[str, Any] = {
        "video_key": video_key,
        "video_path": str(video_path),
        "pickle_path": str(yolo_sidecar),
        "gt_mode": "yolo",
        "n_ids": 0,
        "pad_ratio": pad_ratio,
        "tag_family": tag_family,
        "n_frames_total": int(counts.index.max()) + 1 if len(counts) else 0,
        "n_frames_selected": len(grabbed),
        "frames": manifest,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest_doc, f, indent=2)

    total_crops = sum(len(e["crops"]) for e in manifest)
    print(
        f"  Composites: {len(manifest)}, Crops: {total_crops} "
        f"(YOLO mode — no ground truth)"
    )
    return manifest_doc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_testset(
    video_path: str | Path,
    clean_pickle: str | Path,
    out_dir: str | Path,
    yolo_weights: str | Path,
    *,
    sample_per_video: int = 50,
    top_n: int = 1000,
    min_detection_frac: float = 0.8,
    pad_ratio: float = DEFAULT_PAD_RATIO,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    batch_size: int = DEFAULT_BATCH_SIZE,
    force: bool = False,
    sample_strategy: str = "mixed",
    gt_source: str = "clean",
) -> dict[str, Any] | None:
    """Build one testset subfolder from a single video + YOTO clean pickle.

    Uses :func:`yoto.detection._compute_crop_layout` with the same
    ``pad_ratio`` default as ``yoto detect``, so composites are
    pixel-identical to what the detection pipeline would produce on the same
    frames.

    When ``gt_source == "yolo"``, *clean_pickle* is interpreted as the path to
    the ``_yolo.pkl`` sidecar and the testset is built with **no ground truth**
    (cold-start preset optimisation).  See :func:`_build_testset_yolo`.

    Parameters
    ----------
    video_path:
        Source video file.
    clean_pickle:
        A YOTO clean pickle (output of ``yoto clean``).  Used as ground truth
        for tag positions.
    out_dir:
        Root testset directory.  Each video gets its own subfolder.
    yolo_weights:
        YOLO weights file used to locate tag regions in sampled frames.
    sample_per_video:
        Number of frames to sample per video.
    top_n:
        Pool size to sample from (frames ranked by tag visibility).
    min_detection_frac:
        Minimum fraction of known tags that must be visible for a frame to
        enter the pool.
    pad_ratio:
        Per-axis crop padding — must match the value used in ``yoto detect``.
    conf_threshold:
        YOLO confidence threshold for the testset YOLO run.
    batch_size:
        YOLO inference batch size.
    force:
        Rebuild even if the manifest already exists.

    Returns
    -------
    dict or None
        The written manifest document, or ``None`` if the video was skipped.
    """
    if gt_source == "yolo":
        return _build_testset_yolo(
            video_path,
            clean_pickle,
            out_dir,
            sample_per_video=sample_per_video,
            top_n=top_n,
            pad_ratio=pad_ratio,
            sample_strategy=sample_strategy,
            force=force,
        )

    from ultralytics import YOLO

    video_path = Path(video_path)
    clean_pickle = Path(clean_pickle)
    out_dir = Path(out_dir)
    video_key = _make_video_key(video_path)

    vid_dir = out_dir / video_key
    comp_dir = vid_dir / "composites"
    crops_dir = vid_dir / "crops"
    manifest_path = vid_dir / "manifest.json"

    if manifest_path.exists() and not force:
        print(f"  SKIP (already exists): {vid_dir}")
        return None

    if force and manifest_path.exists():
        manifest_path.unlink()

    comp_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Video key: {video_key}")
    print(f"  Pickle:    {clean_pickle}")

    # Load clean pickle
    frame_data: pd.DataFrame = pd.read_pickle(str(clean_pickle))
    frame_data = frame_data.sort_index(axis=1)
    all_ids = list(np.unique(frame_data.columns.get_level_values(0)))
    id_list = [
        i
        for i in all_ids
        if (i, "center_x") in frame_data.columns
        and frame_data[(i, "center_x")].notna().sum() >= 10
    ]
    if not id_list:
        print("  SKIP: no usable IDs in pickle")
        return None
    print(f"  IDs: {len(id_list)}")

    # Sample frames
    selected = _select_frames(
        frame_data,
        id_list,
        top_n=top_n,
        sample_n=sample_per_video,
        min_detection_frac=min_detection_frac,
        sample_strategy=sample_strategy,
    )
    print(f"  Selected {len(selected)} frames")
    frames_dict = _grab_frames(video_path, selected)
    grabbed = [f for f in selected if f in frames_dict]
    print(f"  Grabbed {len(grabbed)}/{len(selected)} frames")

    # YOLO inference
    model = YOLO(str(yolo_weights))
    frame_results: dict[int, Any] = {}
    n_batches = (len(grabbed) + batch_size - 1) // batch_size
    for start in tqdm(
        range(0, len(grabbed), batch_size),
        total=n_batches,
        desc=f"  {video_key} [YOLO]",
        mininterval=1,
    ):
        batch_idx = grabbed[start : start + batch_size]
        preds = model.predict(
            [frames_dict[f] for f in batch_idx],
            conf=conf_threshold,
            verbose=False,
        )
        for idx, pred in zip(batch_idx, preds):
            frame_results[idx] = pred

    totaltag = frame_data.loc[:, (slice(None), "center_x")].notna().sum(axis=1)
    manifest: list[dict[str, Any]] = []

    for frame_idx in tqdm(grabbed, desc=f"  {video_key} [crops]", mininterval=1):
        frame = frames_dict[frame_idx]
        H, W = frame.shape[:2]

        all_ids_this_frame = [
            int(i)
            for i in id_list
            if not pd.isna(frame_data.loc[frame_idx].get((i, "center_x"), np.nan))
        ]

        result = frame_results.get(frame_idx)
        if result is None or result.boxes is None or len(result.boxes) == 0:
            continue

        boxes_np = result.boxes.xyxy.cpu().numpy()
        bounds, offsets_xy, canvas_x_offsets, composite_size = _compute_crop_layout(
            boxes_np, H, W, pad_ratio
        )
        if not bounds:
            continue

        # BGR composite (optimizer applies its own grayscale + preprocessing)
        strip_h, strip_w = composite_size
        composite = np.zeros((strip_h, strip_w, 3), dtype=np.uint8)
        crops_bgr: list[np.ndarray] = []
        for (y1, y2, x1, x2), dst_x in zip(bounds, canvas_x_offsets):
            crop = frame[y1:y2, x1:x2]
            composite[0 : crop.shape[0], dst_x : dst_x + crop.shape[1]] = crop
            crops_bgr.append(crop)

        comp_name = f"f{frame_idx:06d}.png"
        cv2.imwrite(str(comp_dir / comp_name), composite)

        crop_entries: list[dict[str, Any]] = []
        for j, ((y1, y2, x1, x2), (ox, oy), dst_x, crop) in enumerate(
            zip(bounds, offsets_xy, canvas_x_offsets, crops_bgr)
        ):
            candidates = _gt_ids_in_crop(frame_data, frame_idx, x1, y1, x2, y2, id_list)
            crop_file = f"f{frame_idx:06d}_c{j:03d}.jpg"
            crop_entries.append(
                {
                    "crop_idx": j,
                    "crop_file": crop_file,
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "crop_shape": [crop.shape[0], crop.shape[1]],
                    "canvas_x_offset": dst_x,
                    "frame_offset_xy": [int(ox), int(oy)],
                    "gt_ids": [c[0] for c in candidates],
                    "gt_positions_in_frame": [[c[1], c[2]] for c in candidates],
                    "gt_positions_in_crop": [
                        [c[1] - ox, c[2] - oy] for c in candidates
                    ],
                    "gt_positions_in_composite": [
                        [dst_x + (c[1] - ox), c[2] - oy] for c in candidates
                    ],
                    "gt_ass_types": [c[3] for c in candidates],
                }
            )
            cv2.imwrite(str(crops_dir / crop_file), crop)

        manifest.append(
            {
                "composite_file": comp_name,
                "frame_idx": int(frame_idx),
                "composite_shape": [composite.shape[0], composite.shape[1]],
                "n_crops": len(crops_bgr),
                "n_total_ids_in_frame": int(totaltag.get(frame_idx, 0)),
                "all_frame_gt_ids": all_ids_this_frame,
                "crops": crop_entries,
            }
        )

    tag_family: str = frame_data.attrs.get("yoto_tag_family", "unknown")

    manifest_doc: dict[str, Any] = {
        "video_key": video_key,
        "video_path": str(video_path),
        "pickle_path": str(clean_pickle),
        "n_ids": len(id_list),
        "pad_ratio": pad_ratio,
        "tag_family": tag_family,
        "n_frames_total": len(frame_data),
        "n_frames_selected": len(grabbed),
        "frames": manifest,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest_doc, f, indent=2)

    total_crops = sum(len(e["crops"]) for e in manifest)
    crops_with_gt = sum(1 for e in manifest for c in e["crops"] if c["gt_ids"])
    gt_orig = sum(
        1 for e in manifest for c in e["crops"] for a in c["gt_ass_types"] if a == 1
    )
    gt_interp = sum(
        1 for e in manifest for c in e["crops"] for a in c["gt_ass_types"] if a == 2
    )
    print(
        f"  Composites: {len(manifest)}, Crops: {total_crops} "
        f"(GT: {crops_with_gt}, no-GT: {total_crops - crops_with_gt}, "
        f"orig: {gt_orig}, interp: {gt_interp})"
    )
    return manifest_doc
