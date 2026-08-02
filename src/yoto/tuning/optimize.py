from __future__ import annotations

"""Optuna-based AprilTag preset optimizer.

Three search-space tiers control how many parameters Optuna explores:

* ``minimal``  — AprilTag decoder knobs + upscale + one contrast method.
  Fast exploration, few trials needed (~100).
* ``standard`` — adds unsharp mask, tone mapping, Wiener deconvolution, and
  all contrast methods.  Good default for a 300–600-trial run.
* ``full``     — adds invert, bilateral filter, median blur, gamma, and
  adaptive thresholding.  Exhaustive; needs 800+ trials.

The output JSON is consumable directly by ``yoto detect --apriltag-preset``.
"""

import json
import os
import shutil
import sys
import threading
import time
import warnings
from functools import partial
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .preprocess import disk_blur_augment, preprocess_composite

# ---------------------------------------------------------------------------
# Live display callback
# ---------------------------------------------------------------------------


def _trial_line(t: Any, best_num: int) -> str:
    """One-line summary of a completed trial with full metric info."""
    a = t.user_attrs
    marker = " ◀ BEST" if t.number == best_num else ""
    return (
        f"  trial {t.number:>4d}: score={t.value:.4f}"
        f"  recall={a.get('individual_recall', 0):.4f}"
        f"  FP={a.get('false_positive_rate', 0):.4f}"
        f"  found={a.get('individual_found', '?')}/{a.get('total_gt_ids', '?')}"
        f"  fp_n={a.get('false_positives', '?')}"
        f"  pre={a.get('avg_preprocess_ms', 0):.0f}ms"
        f"  det={a.get('avg_detect_ms', 0):.0f}ms"
        f"  total={a.get('avg_total_ms', 0):.0f}ms"
        f"  wall={a.get('eval_wall_s', 0):.1f}s"
        f"{marker}"
    )


class _LiveDisplay:
    """Optuna callback: live-updating N-line block showing the most recent trials.

    When a new best is found, a scrolling block with full params is printed
    above the live section so it persists in terminal history.

    A threading lock prevents concurrent callbacks (n_jobs > 1) from
    interleaving their cursor-movement escape sequences.
    """

    def __init__(self, n_trials: int, n_show: int = 5) -> None:
        self._n_trials = n_trials
        self._n_show = n_show
        self._completed = 0
        self._block_rows = 0  # visual rows written last redraw (wrap-aware)
        self._best_num_seen = -1
        self._best_banner: list[str] = []
        self._start = time.perf_counter()
        self._lock = threading.Lock()

    def __call__(self, study: Any, trial: Any) -> None:
        with self._lock:
            self._completed += 1

            try:
                best_val = study.best_value
                best_trial = study.best_trial
                best_num = best_trial.number
            except ValueError:
                best_val = float("nan")
                best_trial = None
                best_num = -1

            n_pruned = sum(
                1 for t in study.trials if str(t.state) == "TrialState.PRUNED"
            )

            elapsed = time.perf_counter() - self._start
            eta_str = ""
            if self._completed > 0:
                spt = elapsed / self._completed
                remaining = (self._n_trials - self._completed) * spt
                h, m = divmod(int(remaining), 3600)
                m, s = divmod(m, 60)
                eta_str = (
                    f"  ETA {h:02d}:{m:02d}:{s:02d}" if h else f"  ETA {m:02d}:{s:02d}"
                )

            # Update the best banner when a new best is found
            if best_num != self._best_num_seen and best_trial is not None:
                self._best_num_seen = best_num
                a = best_trial.user_attrs
                self._best_banner = [
                    f"  ★ BEST  trial {best_num}  score={best_val:.4f}"
                    f"  recall={a.get('individual_recall', 0):.4f}"
                    f"  FP={a.get('false_positive_rate', 0):.4f}"
                    f"  found={a.get('individual_found', '?')}/{a.get('total_gt_ids', '?')}",
                    "    params: "
                    + "  ".join(
                        f"{k}={v}" for k, v in sorted(best_trial.params.items())
                    ),
                ]

            # Erase the entire previous block (banner + header + trials).
            # Long lines wrap, so we must move up by the number of VISUAL rows
            # actually occupied — not the logical line count. Under-counting
            # leaves the wrapped overflow in scrollback, which is what made the
            # "★ BEST" list appear to grow forever.
            if self._block_rows:
                sys.stdout.write(f"\033[{self._block_rows}F\033[J")

            # Last N completed trials in chronological order
            completed = [t for t in study.trials if t.value is not None]
            recent = completed[-self._n_show :]

            header = (
                f"  [{self._completed}/{self._n_trials}]{eta_str}"
                f"  best: {best_val:.4f} @ #{best_num}"
                f"  pruned: {n_pruned}"
            )
            lines = (
                self._best_banner
                + [header]
                + [_trial_line(t, best_num) for t in recent]
            )

            cols = max(shutil.get_terminal_size((100, 40)).columns, 1)
            sys.stdout.write("\n".join(lines) + "\n")
            sys.stdout.flush()
            # ceil(len / cols) visual rows per logical line (min 1).
            self._block_rows = sum(max(1, -(-len(x) // cols)) for x in lines)


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------


def make_storage(storage: Any) -> Any:
    """Wrap a SQLite storage URL so parallel workers wait for the lock.

    SQLite fails a query instantly on lock contention; with many parallel
    workers that surfaces as "database is locked" and kills workers.  A long
    busy timeout makes them wait instead.  WAL is intentionally not enabled —
    it is unsafe on the network mounts these testsets often live on.  Non
    SQLite / ``None`` storages are returned unchanged.
    """
    import optuna

    if isinstance(storage, str) and storage.startswith("sqlite"):
        return optuna.storages.RDBStorage(
            url=storage,
            engine_kwargs={"connect_args": {"timeout": 120}},
        )
    return storage


def ensure_study(study_name: str, storage: str) -> None:
    """Pre-create the study/schema once so concurrent workers only attach.

    Launching N workers against a not-yet-existing SQLite DB races on schema
    creation ("table studies already exists").  Creating it once up front —
    before spawning workers — removes that cold-start race; each worker then
    opens it with ``load_if_exists=True``.
    """
    import optuna

    optuna.create_study(
        study_name=study_name,
        storage=make_storage(storage),
        direction="maximize",
        load_if_exists=True,
    )


def load_manifests(
    testset_dir: str | Path,
) -> tuple[list[dict[str, Any]], set[int], list[str], str | None, bool]:
    """Load all per-video manifests under *testset_dir*.

    Returns
    -------
    entries : list[dict]
        Flat list of frame entries (each has ``_composites_dir`` injected).
    valid_ids : set[int]
        All tag IDs seen across all videos.
    video_keys : list[str]
        One key per video subfolder.
    tag_family : str or None
        Tag family recorded by ``build-testset``, or ``None`` if absent
        (old manifests built before this field was added).
    yield_mode : bool
        ``True`` when the testset carries no ground truth (built by
        ``build-testset --gt-source yolo``): every crop has an empty
        ``gt_ids``.  Optimisation then targets raw decode yield, not recall.
    """
    testset_dir = Path(testset_dir)
    paths = sorted(testset_dir.glob("*/manifest.json"))
    if not paths:
        raise FileNotFoundError(
            f"No manifest.json found under {testset_dir}/*/manifest.json\n"
            "Run 'yoto train build-testset' first."
        )
    entries: list[dict[str, Any]] = []
    valid_ids: set[int] = set()
    keys: list[str] = []
    tag_families: set[str] = set()
    any_yolo_flag = False
    total_crops = crops_with_gt = 0
    for mp in paths:
        with open(mp) as f:
            doc = json.load(f)
        comp_dir = str(mp.parent / "composites")
        keys.append(doc.get("video_key", mp.parent.name))
        if doc.get("gt_mode") == "yolo":
            any_yolo_flag = True
        if fam := doc.get("tag_family"):
            tag_families.add(fam)
        for fe in doc["frames"]:
            fe["_composites_dir"] = comp_dir
            entries.append(fe)
            valid_ids.update(fe.get("all_frame_gt_ids", []))
            for c in fe["crops"]:
                total_crops += 1
                if c["gt_ids"]:
                    crops_with_gt += 1
    detected_family: str | None = tag_families.pop() if len(tag_families) == 1 else None
    # No ground truth anywhere → yield mode (also honour the explicit flag).
    yield_mode = any_yolo_flag or (total_crops > 0 and crops_with_gt == 0)
    return entries, valid_ids, keys, detected_family, yield_mode


def _compute_id_difficulty(
    entries: list[dict[str, Any]],
) -> tuple[dict[int, float], dict[int, int]]:
    """Score each tag ID by difficulty = ``1 - (original decodes / total GT)``.

    Higher = harder (rarely decoded originally) = more valuable to test.
    """
    from collections import Counter

    id_original: Counter[int] = Counter()
    id_total: Counter[int] = Counter()
    for entry in entries:
        for crop in entry["crops"]:
            for tid, atype in zip(crop["gt_ids"], crop["gt_ass_types"]):
                id_total[tid] += 1
                if atype == 1:
                    id_original[tid] += 1
    difficulty = {
        tid: (1.0 - id_original[tid] / id_total[tid]) if id_total[tid] else 1.0
        for tid in id_total
    }
    return difficulty, dict(id_total)


def _greedy_set_cover(
    entries: list[dict[str, Any]],
    difficulty: dict[int, float],
    target: int,
    min_appearances: int,
) -> tuple[list[int], dict[int, int]]:
    """Greedy difficulty-weighted set cover, filled up to *target* composites.

    Each round picks the composite with the highest difficulty-weighted
    score.  IDs below *min_appearances* dominate the score (brand-new IDs get
    a 3x bonus), so the per-ID floor is reached first; once every ID is
    covered, already-covered IDs still contribute a small ``1/(covered+1)``
    term, so the greedy keeps adding composites — favouring the least-covered
    and hardest IDs — until *target* composites are chosen (or candidates run
    out).  So *target* is the size of the subset, not just a cap.
    """
    from collections import Counter

    composite_ids: list[set[int]] = []
    for entry in entries:
        ids: set[int] = set()
        for crop in entry["crops"]:
            ids.update(crop["gt_ids"])
        composite_ids.append(ids)

    id_covered: Counter[int] = Counter()
    selected: list[int] = []
    remaining = set(range(len(entries)))
    while len(selected) < target and remaining:
        best_idx, best_score = -1, -1.0
        for i in remaining:
            score = 0.0
            for tid in composite_ids[i]:
                weight = 1.0 + difficulty.get(tid, 0.5) * 2.0
                covered = id_covered[tid]
                if covered < min_appearances:
                    # Strong pull to reach the per-ID floor; new IDs worth most.
                    score += weight * 10.0 * (3.0 if covered == 0 else 1.0)
                else:
                    # Floor met — keep filling toward --target, favouring the
                    # least-covered (and hardest) IDs so coverage stays even.
                    score += weight / (covered + 1.0)
            if score > best_score:
                best_score, best_idx = score, i
        if best_idx < 0:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)
        for tid in composite_ids[best_idx]:
            id_covered[tid] += 1
    return selected, dict(id_covered)


def subsample_testset(
    testset_dir: str | Path,
    *,
    target: int = 50,
    min_appearances: int = 3,
    output: str | Path | None = None,
) -> Path:
    """Write a compact subset manifest that maximizes tag-ID diversity.

    We don't need every composite if most test the same easy IDs.  This
    scores each ID by difficulty (rarely-decoded IDs are harder, more
    valuable) then greedily picks composites so every ID is covered at
    least *min_appearances* times, up to *target* composites.  The written
    ``subset_manifest.json`` can be passed to :func:`optimize_preset`
    (``subset_manifest=``) for a much faster study.

    Returns the path to the written manifest.
    """
    from collections import Counter

    testset_dir = Path(testset_dir)
    entries, _, _, _, _ = load_manifests(testset_dir)
    for entry in entries:
        entry.setdefault("_video_key", Path(entry["_composites_dir"]).parent.name)

    total_crops = sum(len(e["crops"]) for e in entries)
    print(f"Full test set: {len(entries)} composites, {total_crops} crops")

    difficulty, id_total = _compute_id_difficulty(entries)
    all_ids = set(id_total)
    easy = sum(1 for d in difficulty.values() if d < 0.2)
    medium = sum(1 for d in difficulty.values() if 0.2 <= d < 0.5)
    hard = sum(1 for d in difficulty.values() if d >= 0.5)
    print(
        f"Unique IDs: {len(all_ids)}  |  difficulty: {easy} easy, "
        f"{medium} medium, {hard} hard (>50% interp)"
    )

    selected, id_covered = _greedy_set_cover(
        entries, difficulty, target, min_appearances
    )
    subset = [entries[i] for i in selected]

    subset_crops = sum(len(e["crops"]) for e in subset)
    subset_ids: set[int] = set()
    for entry in subset:
        for crop in entry["crops"]:
            subset_ids.update(crop["gt_ids"])
    covered_enough = sum(
        1 for tid in all_ids if id_covered.get(tid, 0) >= min_appearances
    )
    uncovered = all_ids - subset_ids

    print(f"\nSubset: {len(subset)} composites, {subset_crops} crops")
    print(
        f"  IDs covered: {len(subset_ids)}/{len(all_ids)} "
        f"({covered_enough} with >={min_appearances} appearances)"
    )
    if uncovered:
        print(f"  Uncovered IDs: {sorted(uncovered)}")
    if entries:
        speedup = len(entries) / max(len(subset), 1)
        print(
            f"  Reduction: {len(subset)}/{len(entries)} composites "
            f"({100 * len(subset) / len(entries):.0f}%, ~{speedup:.1f}x fewer)"
        )
    vids = Counter(entry["_video_key"] for entry in subset)
    print("  Per-video distribution:")
    for vk, cnt in sorted(vids.items()):
        print(f"    {vk}: {cnt}")
    if all_ids:
        cov = sorted(id_covered.get(tid, 0) for tid in all_ids)
        print(
            f"  ID coverage: min={cov[0]}, median={cov[len(cov) // 2]}, "
            f"max={cov[-1]}"
        )

    out_path = Path(output) if output else testset_dir / "subset_manifest.json"
    with open(out_path, "w") as f:
        json.dump(subset, f, indent=2)
    print(f"\nSaved subset manifest to: {out_path}")
    return out_path


def _assign_tag_to_crop(
    cx: float, cy: float, crops: list[dict[str, Any]], scale: float = 1.0
) -> int:
    """Map a detected tag centre (in possibly upscaled composite coords) to a crop index."""
    for c in crops:
        x_start = c["canvas_x_offset"] * scale
        x_end = x_start + c["crop_shape"][1] * scale
        y_end = c["crop_shape"][0] * scale
        if x_start <= cx < x_end and 0 <= cy < y_end:
            return int(c["crop_idx"])
    return -1


# ---------------------------------------------------------------------------
# Search-space tiers
# ---------------------------------------------------------------------------


def _suggest_minimal(trial: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Decoder params + upscale + one contrast method."""
    upscale = trial.suggest_categorical("upscale", [1.0, 1.5, 2.0, 2.5])
    upscale_interp = "lanczos"
    if upscale != 1.0:
        upscale_interp = trial.suggest_categorical(
            "upscale_interp", ["linear", "cubic", "lanczos"]
        )

    contrast_method = trial.suggest_categorical(
        "contrast_method", ["clahe", "simple", "none"]
    )
    clahe_clip = clahe_grid = contrast_factor = 0
    if contrast_method == "clahe":
        clahe_clip = trial.suggest_float("clahe_clip", 1.0, 8.0, step=0.5)
        clahe_grid = trial.suggest_categorical("clahe_grid", [4, 8, 16])
    elif contrast_method == "simple":
        contrast_factor = trial.suggest_float("contrast_factor", 1.0, 3.0, step=0.25)

    pre: dict[str, Any] = {
        "upscale": upscale,
        "upscale_interp": upscale_interp,
        "contrast_method": contrast_method,
        "clahe_clip": clahe_clip,
        "clahe_grid": clahe_grid,
        "contrast_factor": contrast_factor,
    }
    det: dict[str, Any] = {
        "decimate": trial.suggest_float("decimate", 1.0, 3.0, step=0.5),
        "blur": trial.suggest_float("blur", 0.0, 1.5, step=0.1),
        "decode_sharpening": trial.suggest_float(
            "decode_sharpening", 0.0, 15.0, step=0.5
        ),
        "max_hamming": trial.suggest_int("max_hamming", 0, 3),
        "refine_edges": trial.suggest_categorical("refine_edges", [0, 1]),
    }
    return pre, det


def _suggest_standard(trial: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """+ unsharp mask, tone mapping, Wiener, full contrast selection."""
    upscale = trial.suggest_categorical("upscale", [1.0, 1.5, 2.0, 2.5, 3.0])
    upscale_interp = "lanczos"
    if upscale != 1.0:
        upscale_interp = trial.suggest_categorical(
            "upscale_interp", ["linear", "cubic", "lanczos"]
        )

    tone = trial.suggest_categorical("tone_map", ["none", "log", "sqrt", "sigmoid"])

    use_unsharp = trial.suggest_categorical("use_unsharp", [True, False])
    ks = sigma = amount = 0
    if use_unsharp:
        ks = trial.suggest_categorical("kernel_size", [3, 5, 7, 9, 11])
        sigma = trial.suggest_float("sigma", 0.3, 4.0, step=0.1)
        amount = trial.suggest_float("amount", 0.25, 6.0, step=0.25)

    use_wiener = trial.suggest_categorical("use_wiener", [True, False])
    wiener_psf = wiener_noise = 0.0
    if use_wiener:
        wiener_psf = trial.suggest_float("wiener_psf_radius", 0.5, 8.0, step=0.5)
        wiener_noise = trial.suggest_float("wiener_noise_level", 0.0005, 0.05, log=True)

    contrast_method = trial.suggest_categorical(
        "contrast_method", ["clahe", "cv2", "simple", "none"]
    )
    clahe_clip = clahe_grid = cv2_alpha = cv2_beta = contrast_factor = 0
    if contrast_method == "clahe":
        clahe_clip = trial.suggest_float("clahe_clip", 1.0, 12.0, step=0.5)
        clahe_grid = trial.suggest_categorical("clahe_grid", [4, 8, 16, 32])
    elif contrast_method == "cv2":
        cv2_alpha = trial.suggest_float("cv2_alpha", 1.0, 4.0, step=0.25)
        cv2_beta = trial.suggest_int("cv2_beta", -160, 32, step=16)
    elif contrast_method == "simple":
        contrast_factor = trial.suggest_float("contrast_factor", 1.0, 3.5, step=0.25)

    pre: dict[str, Any] = {
        "upscale": upscale,
        "upscale_interp": upscale_interp,
        "tone_map": tone,
        "use_unsharp": use_unsharp,
        "kernel_size": ks,
        "sigma": sigma,
        "amount": amount,
        "use_wiener": use_wiener,
        "wiener_psf_radius": wiener_psf,
        "wiener_noise_level": wiener_noise,
        "contrast_method": contrast_method,
        "clahe_clip": clahe_clip,
        "clahe_grid": clahe_grid,
        "cv2_alpha": cv2_alpha,
        "cv2_beta": cv2_beta,
        "contrast_factor": contrast_factor,
    }
    det: dict[str, Any] = {
        "decimate": trial.suggest_float("decimate", 1.0, 3.0, step=0.5),
        "blur": trial.suggest_float("blur", 0.0, 1.5, step=0.1),
        "decode_sharpening": trial.suggest_float(
            "decode_sharpening", 0.0, 15.0, step=0.5
        ),
        "max_hamming": trial.suggest_int("max_hamming", 0, 3),
        "refine_edges": trial.suggest_categorical("refine_edges", [0, 1]),
    }
    return pre, det


def _suggest_standard_lite(trial: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Like ``standard`` but with no tone-mapping and no Wiener deconvolution.

    Keeps upscale, unsharp, and the full contrast selection (including
    ``cv2``), so it can reproduce the ``yoto detect`` default while still
    tuning the detector and contrast.
    """
    upscale = trial.suggest_categorical("upscale", [1.0, 1.5, 2.0, 2.5, 3.0])
    upscale_interp = "lanczos"
    if upscale != 1.0:
        upscale_interp = trial.suggest_categorical(
            "upscale_interp", ["linear", "cubic", "lanczos"]
        )

    use_unsharp = trial.suggest_categorical("use_unsharp", [True, False])
    ks = sigma = amount = 0
    if use_unsharp:
        ks = trial.suggest_categorical("kernel_size", [3, 5, 7, 9, 11])
        sigma = trial.suggest_float("sigma", 0.3, 4.0, step=0.1)
        amount = trial.suggest_float("amount", 0.25, 6.0, step=0.25)

    contrast_method = trial.suggest_categorical(
        "contrast_method", ["clahe", "cv2", "simple", "none"]
    )
    clahe_clip = clahe_grid = cv2_alpha = cv2_beta = contrast_factor = 0
    if contrast_method == "clahe":
        clahe_clip = trial.suggest_float("clahe_clip", 1.0, 12.0, step=0.5)
        clahe_grid = trial.suggest_categorical("clahe_grid", [4, 8, 16, 32])
    elif contrast_method == "cv2":
        cv2_alpha = trial.suggest_float("cv2_alpha", 1.0, 4.0, step=0.25)
        cv2_beta = trial.suggest_int("cv2_beta", -160, 32, step=16)
    elif contrast_method == "simple":
        contrast_factor = trial.suggest_float("contrast_factor", 1.0, 3.5, step=0.25)

    pre: dict[str, Any] = {
        "upscale": upscale,
        "upscale_interp": upscale_interp,
        "tone_map": "none",
        "use_unsharp": use_unsharp,
        "kernel_size": ks,
        "sigma": sigma,
        "amount": amount,
        "use_wiener": False,
        "wiener_psf_radius": 0.0,
        "wiener_noise_level": 0.0,
        "contrast_method": contrast_method,
        "clahe_clip": clahe_clip,
        "clahe_grid": clahe_grid,
        "cv2_alpha": cv2_alpha,
        "cv2_beta": cv2_beta,
        "contrast_factor": contrast_factor,
    }
    det: dict[str, Any] = {
        "decimate": trial.suggest_float("decimate", 1.0, 3.0, step=0.5),
        "blur": trial.suggest_float("blur", 0.0, 1.5, step=0.1),
        "decode_sharpening": trial.suggest_float(
            "decode_sharpening", 0.0, 15.0, step=0.5
        ),
        "max_hamming": trial.suggest_int("max_hamming", 0, 3),
        "refine_edges": trial.suggest_categorical("refine_edges", [0, 1]),
    }
    return pre, det


def _suggest_full(trial: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """+ invert, bilateral filter, median blur, gamma, adaptive threshold."""
    invert = trial.suggest_categorical("invert", [True, False])

    upscale = trial.suggest_categorical("upscale", [1.0, 1.5, 2.0, 2.5, 3.0, 4.0])
    upscale_interp = "lanczos"
    if upscale != 1.0:
        upscale_interp = trial.suggest_categorical(
            "upscale_interp", ["linear", "cubic", "lanczos"]
        )

    use_median = trial.suggest_categorical("use_median_blur", [True, False])
    median_k = trial.suggest_categorical("median_ksize", [3, 5]) if use_median else 3

    use_bilateral = trial.suggest_categorical("use_bilateral", [True, False])
    bd = bsc = bss = 5
    if use_bilateral:
        bd = trial.suggest_categorical("bilateral_d", [5, 7, 9])
        bsc = trial.suggest_float("bilateral_sigma_color", 20, 120, step=10)
        bss = trial.suggest_float("bilateral_sigma_space", 20, 120, step=10)

    use_wiener = trial.suggest_categorical("use_wiener", [True, False])
    wiener_psf = wiener_noise = 0.0
    if use_wiener:
        wiener_psf = trial.suggest_float("wiener_psf_radius", 0.5, 8.0, step=0.5)
        wiener_noise = trial.suggest_float("wiener_noise_level", 0.0005, 0.05, log=True)

    tone = trial.suggest_categorical("tone_map", ["none", "log", "sqrt", "sigmoid"])

    use_unsharp = trial.suggest_categorical("use_unsharp", [True, False])
    ks = sigma = amount = 0
    if use_unsharp:
        ks = trial.suggest_categorical("kernel_size", [3, 5, 7, 9, 11])
        sigma = trial.suggest_float("sigma", 0.3, 4.0, step=0.1)
        amount = trial.suggest_float("amount", 0.25, 6.0, step=0.25)

    contrast_method = trial.suggest_categorical(
        "contrast_method", ["clahe", "cv2", "simple", "adaptive", "none"]
    )
    clahe_clip = clahe_grid = cv2_alpha = cv2_beta = contrast_factor = 0
    adapt_block = adapt_C = 0
    adapt_gaussian = True
    if contrast_method == "clahe":
        clahe_clip = trial.suggest_float("clahe_clip", 1.0, 12.0, step=0.5)
        clahe_grid = trial.suggest_categorical("clahe_grid", [4, 8, 16, 32])
    elif contrast_method == "cv2":
        cv2_alpha = trial.suggest_float("cv2_alpha", 1.0, 4.0, step=0.25)
        cv2_beta = trial.suggest_int("cv2_beta", -160, 32, step=16)
    elif contrast_method == "simple":
        contrast_factor = trial.suggest_float("contrast_factor", 1.0, 3.5, step=0.25)
    elif contrast_method == "adaptive":
        adapt_block = trial.suggest_categorical("adapt_block", [11, 15, 21, 31, 51])
        adapt_C = trial.suggest_int("adapt_C", -10, 20)
        adapt_gaussian = trial.suggest_categorical("adapt_gaussian", [True, False])

    use_gamma = trial.suggest_categorical("use_gamma", [True, False])
    gamma_val = trial.suggest_float("gamma", 0.4, 2.5, step=0.1) if use_gamma else 1.0

    pre: dict[str, Any] = {
        "invert": invert,
        "upscale": upscale,
        "upscale_interp": upscale_interp,
        "use_median_blur": use_median,
        "median_ksize": median_k,
        "use_bilateral": use_bilateral,
        "bilateral_d": bd,
        "bilateral_sigma_color": bsc,
        "bilateral_sigma_space": bss,
        "use_wiener": use_wiener,
        "wiener_psf_radius": wiener_psf,
        "wiener_noise_level": wiener_noise,
        "tone_map": tone,
        "use_unsharp": use_unsharp,
        "kernel_size": ks,
        "sigma": sigma,
        "amount": amount,
        "contrast_method": contrast_method,
        "clahe_clip": clahe_clip,
        "clahe_grid": clahe_grid,
        "cv2_alpha": cv2_alpha,
        "cv2_beta": cv2_beta,
        "contrast_factor": contrast_factor,
        "adapt_block": adapt_block,
        "adapt_C": adapt_C,
        "adapt_gaussian": adapt_gaussian,
        "use_gamma": use_gamma,
        "gamma": gamma_val,
    }
    det: dict[str, Any] = {
        "decimate": trial.suggest_float("decimate", 1.0, 3.0, step=0.5),
        "blur": trial.suggest_float("blur", 0.0, 1.5, step=0.1),
        "decode_sharpening": trial.suggest_float(
            "decode_sharpening", 0.0, 15.0, step=0.5
        ),
        "max_hamming": trial.suggest_int("max_hamming", 0, 3),
        "refine_edges": trial.suggest_categorical("refine_edges", [0, 1]),
    }
    return pre, det


def _suggest_apriltag_only(trial: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Tune ONLY the AprilTag detector params on the raw crop (no enhancement).

    Mirrors the spirit of ``yoto detect --no-yolo`` (bare detector, no image
    pre-processing) while still evaluating on the fast testset crops.  The
    enhancement stage is a pure grayscale passthrough
    (``contrast_method='none'``, unsharp off).  Both keys are recorded as
    (single-choice) trial params so an exported preset carries them through
    and reproduces the no-enhance path in ``detect``.
    """
    pre: dict[str, Any] = {
        "contrast_method": trial.suggest_categorical("contrast_method", ["none"]),
        "use_unsharp": trial.suggest_categorical("use_unsharp", [False]),
    }
    det: dict[str, Any] = {
        "decimate": trial.suggest_float("decimate", 1.0, 3.0, step=0.5),
        "blur": trial.suggest_float("blur", 0.0, 1.5, step=0.1),
        "decode_sharpening": trial.suggest_float(
            "decode_sharpening", 0.0, 15.0, step=0.5
        ),
        "max_hamming": trial.suggest_int("max_hamming", 0, 3),
        "refine_edges": trial.suggest_categorical("refine_edges", [0, 1]),
    }
    return pre, det


_SUGGESTERS = {
    "apriltag-only": _suggest_apriltag_only,
    "minimal": _suggest_minimal,
    "standard-lite": _suggest_standard_lite,
    "standard": _suggest_standard,
    "full": _suggest_full,
}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _score(
    individual_recall: float,
    false_positive_rate: float,
    total_no_gt: int,
    no_gt_hallucinations: int,
    no_gt_discoveries: int,
    avg_time_ms: float,
    speed_weight: float,
    speed_floor_recall: float,
) -> float:
    """Recall-first score. Speed only penalises once recall exceeds the floor."""
    no_gt_total = max(total_no_gt, 1)
    halluc_rate = no_gt_hallucinations / no_gt_total
    discovery_rate = no_gt_discoveries / no_gt_total
    effective_sw = speed_weight if individual_recall >= speed_floor_recall else 0.0
    return (
        individual_recall
        - 0.1 * false_positive_rate
        - 0.05 * halluc_rate
        + 0.02 * discovery_rate
        - effective_sw * (avg_time_ms / 100.0)
    )


def _yield_score(hits: int, multi: int, total: int) -> float:
    """Cold-start (no-ground-truth) objective: maximise decode yield.

    Each YOLO box should contain exactly one tag, so we reward boxes that
    decode at all and lightly penalise boxes that decode more than one distinct
    ID (spurious extra decodes).  ``hits`` counts boxes with >=1 decode,
    ``multi`` counts boxes with >1 distinct decode.
    """
    total = max(total, 1)
    return hits / total - 0.2 * multi / total


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _evaluate(
    entries: list[dict[str, Any]],
    valid_ids: set[int],
    preprocess_params: dict[str, Any],
    detector_params: dict[str, Any],
    tag_family: str,
    trial: Any = None,
    prune_eval_interval: int = 50,
    speed_weight: float = 0.0,
    speed_floor_recall: float = 0.05,
    synthetic_blur: bool = False,
    max_tag_id: int = 9999,
    silence_ids: frozenset[int] = frozenset(),
    yield_mode: bool = False,
) -> dict[str, Any]:
    import apriltag

    detector = apriltag.apriltag(
        family=tag_family,
        threads=1,
        maxhamming=int(detector_params["max_hamming"]),
        decimate=float(detector_params["decimate"]),
        blur=float(detector_params["blur"]),
        refine_edges=int(detector_params["refine_edges"]),
        decode_sharpening=float(detector_params["decode_sharpening"]),
    )

    total_with_gt = correct = all_gt_found = false_positives = 0
    total_gt_ids = individual_found = 0
    total_no_gt = no_gt_hallucinations = no_gt_discoveries = 0
    # Yield-mode (no ground truth) accumulators.
    yield_total = yield_hits = yield_multi = 0
    total_preprocess_time = total_detect_time = 0.0
    total_composites = 0
    wall_start = time.perf_counter()

    for entry in entries:
        comp_path = os.path.join(entry["_composites_dir"], entry["composite_file"])
        composite = cv2.imread(comp_path)
        if composite is None:
            continue
        total_composites += 1
        if synthetic_blur:
            composite = disk_blur_augment(composite)

        t_pre = time.perf_counter()
        enhanced = preprocess_composite(composite, preprocess_params)
        total_preprocess_time += time.perf_counter() - t_pre

        upscale = float(preprocess_params.get("upscale", 1.0))

        t_det = time.perf_counter()
        raw_tags = detector.detect(enhanced)
        total_detect_time += time.perf_counter() - t_det

        tags = [
            t
            for t in raw_tags
            if int(t["id"]) <= max_tag_id and int(t["id"]) not in silence_ids
        ]

        crop_entries = entry["crops"]
        per_crop: dict[int, set[int]] = {c["crop_idx"]: set() for c in crop_entries}

        for tag in tags:
            tid = int(tag["id"])
            cx, cy = tag["center"]
            ci = _assign_tag_to_crop(cx, cy, crop_entries, scale=upscale)
            if ci >= 0 and ci in per_crop:
                per_crop[ci].add(tid)

        if yield_mode:
            # No ground truth: score raw decode yield per YOLO box.
            for crop in crop_entries:
                det = per_crop.get(crop["crop_idx"], set())
                yield_total += 1
                if len(det) >= 1:
                    yield_hits += 1
                if len(det) > 1:
                    yield_multi += 1
            if (
                trial is not None
                and total_composites % max(prune_eval_interval, 1) == 0
                and yield_total > 0
            ):
                import optuna

                trial.report(
                    _yield_score(yield_hits, yield_multi, yield_total),
                    step=total_composites,
                )
                if trial.should_prune():
                    raise optuna.TrialPruned()
            continue

        frame_gt_ids = set(entry.get("all_frame_gt_ids", []))
        for crop in crop_entries:
            gt_ids = set(crop["gt_ids"])
            det = per_crop.get(crop["crop_idx"], set())
            if gt_ids:
                total_with_gt += 1
                total_gt_ids += len(gt_ids)
                found = gt_ids & det
                individual_found += len(found)
                if found:
                    correct += 1
                if found == gt_ids:
                    all_gt_found += 1
                false_positives += len(det - gt_ids)
            else:
                total_no_gt += 1
                for did in det:
                    if did in frame_gt_ids or did not in valid_ids:
                        no_gt_hallucinations += 1
                    else:
                        no_gt_discoveries += 1

        if (
            trial is not None
            and total_composites % max(prune_eval_interval, 1) == 0
            and total_with_gt > 0
        ):
            import optuna

            total_time = total_preprocess_time + total_detect_time
            interim = _score(
                individual_found / max(total_gt_ids, 1),
                false_positives / max(total_with_gt, 1),
                total_no_gt,
                no_gt_hallucinations,
                no_gt_discoveries,
                1000 * total_time / max(total_composites, 1),
                speed_weight,
                speed_floor_recall,
            )
            trial.report(interim, step=total_composites)
            if trial.should_prune():
                raise optuna.TrialPruned()

    if yield_mode:
        n = max(total_composites, 1)
        avg_pre = 1000 * total_preprocess_time / n
        avg_det = 1000 * total_detect_time / n
        rate = yield_hits / max(yield_total, 1)
        return {
            "yield_rate": rate,
            "yield_hits": yield_hits,
            "yield_multi": yield_multi,
            "total_yield_crops": yield_total,
            # Aliases so the recall-oriented live display stays meaningful.
            "individual_recall": rate,
            "individual_found": yield_hits,
            "total_gt_ids": yield_total,
            "total_with_gt": 0,
            "total_composites": total_composites,
            "avg_preprocess_ms": avg_pre,
            "avg_detect_ms": avg_det,
            "avg_total_ms": avg_pre + avg_det,
            "eval_wall_s": time.perf_counter() - wall_start,
        }

    if total_with_gt == 0:
        return {"detection_rate": 0.0, "total_with_gt": 0}

    total_crops = total_with_gt + total_no_gt
    n = max(total_composites, 1)
    total_wall_time = time.perf_counter() - wall_start
    avg_pre_ms = 1000 * total_preprocess_time / n
    avg_det_ms = 1000 * total_detect_time / n
    avg_total_ms = avg_pre_ms + avg_det_ms
    return {
        "detection_rate": correct / total_with_gt,
        "all_gt_rate": all_gt_found / total_with_gt,
        "individual_recall": individual_found / max(total_gt_ids, 1),
        "false_positive_rate": false_positives / total_with_gt,
        "total_with_gt": total_with_gt,
        "total_gt_ids": total_gt_ids,
        "individual_found": individual_found,
        "correct_detections": correct,
        "false_positives": false_positives,
        "total_no_gt": total_no_gt,
        "no_gt_hallucinations": no_gt_hallucinations,
        "no_gt_discoveries": no_gt_discoveries,
        "total_composites": total_composites,
        "avg_preprocess_ms": avg_pre_ms,
        "avg_detect_ms": avg_det_ms,
        "avg_total_ms": avg_total_ms,
        "avg_time_per_crop_ms": 1000
        * (total_preprocess_time + total_detect_time)
        / max(total_crops, 1),
        # Actual wall-clock time to evaluate all composites for this trial
        # (includes image loading + overhead) — the honest end-to-end cost.
        "eval_wall_s": total_wall_time,
        "avg_wall_ms": 1000 * total_wall_time / n,
    }


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------


def _objective(
    trial: Any,
    entries: list[dict[str, Any]],
    valid_ids: set[int],
    search_space: str,
    tag_family: str,
    prune_eval_interval: int,
    speed_weight: float,
    speed_floor_recall: float,
    synthetic_blur: bool,
    max_tag_id: int,
    silence_ids: frozenset[int],
    yield_mode: bool = False,
) -> float:
    pre, det = _SUGGESTERS[search_space](trial)
    metrics = _evaluate(
        entries,
        valid_ids,
        pre,
        det,
        tag_family=tag_family,
        trial=trial,
        prune_eval_interval=prune_eval_interval,
        speed_weight=speed_weight,
        speed_floor_recall=speed_floor_recall,
        synthetic_blur=synthetic_blur,
        max_tag_id=max_tag_id,
        silence_ids=silence_ids,
        yield_mode=yield_mode,
    )
    for k, v in metrics.items():
        trial.set_user_attr(k, v)
    if yield_mode:
        return _yield_score(
            metrics.get("yield_hits", 0),
            metrics.get("yield_multi", 0),
            metrics.get("total_yield_crops", 0),
        )
    return _score(
        individual_recall=metrics.get("individual_recall", 0.0),
        false_positive_rate=metrics.get("false_positive_rate", 0.0),
        total_no_gt=metrics.get("total_no_gt", 0),
        no_gt_hallucinations=metrics.get("no_gt_hallucinations", 0),
        no_gt_discoveries=metrics.get("no_gt_discoveries", 0),
        avg_time_ms=metrics.get("avg_total_ms", 0.0),
        speed_weight=speed_weight,
        speed_floor_recall=speed_floor_recall,
    )


# ---------------------------------------------------------------------------
# Optuna HTML plots
# ---------------------------------------------------------------------------


def _save_optuna_plots(study: Any, out_dir: Path) -> None:
    """Save Optuna visualisation plots as HTML files (requires plotly)."""
    try:
        import optuna.visualization as ov
    except ImportError:
        print("[plots] optuna.visualization not available (install plotly)")
        return

    import logging
    import optuna

    out_dir.mkdir(parents=True, exist_ok=True)

    # Silence the optuna visualization logger and Python warnings for this block
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    logging.getLogger("optuna").setLevel(logging.ERROR)

    plots = [
        ("optimization_history", ov.plot_optimization_history),
        ("param_importances", ov.plot_param_importances),
        ("parallel_coordinate", ov.plot_parallel_coordinate),
        ("contour", lambda s: ov.plot_contour(s)),
        ("slice", lambda s: ov.plot_slice(s)),
    ]
    saved = []
    for name, fn in plots:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fig = fn(study)
            path = out_dir / f"{name}.html"
            fig.write_html(str(path))
            saved.append(name)
        except Exception as exc:
            print(f"[plots] {name}: {exc}")

    if saved:
        print(f"Optuna plots saved to: {out_dir}  ({', '.join(saved)})")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def optimize_preset(
    testset_dir: str | Path,
    *,
    out_dir: str | Path | None = None,
    search_space: str = "standard",
    n_trials: int = 500,
    n_jobs: int = 1,
    tag_family: str = "tag36ARTag",
    study_name: str = "yoto_preset",
    storage: str | None = None,
    seed_params: str | Path | None = None,
    subset: int | None = None,
    subset_manifest: str | Path | None = None,
    pruner_startup_trials: int = 40,
    pruner_warmup_steps: int = 200,
    prune_eval_interval: int = 50,
    speed_weight: float = 0.0,
    speed_floor_recall: float = 0.05,
    synthetic_blur: bool = False,
    max_tag_id: int = 9999,
    silence_ids: frozenset[int] = frozenset(),
    viz_samples: int = 8,
    viz_dir: str | Path | None = None,
    no_viz: bool = False,
) -> Path:
    """Run an Optuna study to find the best AprilTag preprocessing preset.

    The output JSON can be passed directly to ``yoto detect --apriltag-preset``.

    Parameters
    ----------
    testset_dir:
        Directory built by :func:`yoto.tuning.build_testset`.
    out_dir:
        Where to write ``best_params_<study_name>.json``.  Defaults to
        *testset_dir*.
    search_space:
        ``"minimal"``, ``"standard-lite"``, ``"standard"`` (default), or
        ``"full"``.
    n_trials:
        Total Optuna trials.
    n_jobs:
        Parallel workers (each spawns its own AprilTag detector).
    tag_family:
        AprilTag family string passed to the decoder — must match the one used
        in ``yoto detect``.
    study_name:
        Optuna study name (also used in the output file name).
    storage:
        Optuna storage URI for persistent studies (e.g. SQLite).  ``None``
        uses in-memory storage.
    seed_params:
        JSON file whose params are enqueued as the first trial.  Accepts both
        flat dicts and Optuna-style ``{"params": {...}}`` blobs.
    subset:
        If set, only use the first *N* composite entries (for quick tests).
    subset_manifest:
        Path to a ``subset_manifest.json`` from :func:`subsample_testset`.
        When given, evaluation uses exactly those composites (takes
        precedence over *subset*).
    pruner_startup_trials:
        Trials before the MedianPruner starts pruning.
    pruner_warmup_steps:
        Evaluation steps before pruning kicks in within a trial.
    prune_eval_interval:
        How often (in composites) to report an interim score to the pruner.
    speed_weight:
        Weight applied to the speed penalty.  ``0`` (default) is recall-only;
        set > 0 once you have configs that exceed *speed_floor_recall*.
    speed_floor_recall:
        Recall threshold below which the speed penalty is suppressed entirely.
    synthetic_blur:
        Apply disk-kernel blur to each composite before preprocessing
        (brightfield-style augmentation).

    Returns
    -------
    Path
        Path to the written ``best_params_<study_name>.json``.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if search_space not in _SUGGESTERS:
        raise ValueError(
            f"search_space must be one of {list(_SUGGESTERS)}; got {search_space!r}"
        )

    testset_dir = Path(testset_dir)
    out_path_dir = Path(out_dir) if out_dir else testset_dir

    entries, valid_ids, video_keys, detected_family, yield_mode = load_manifests(
        testset_dir
    )
    if subset_manifest:
        with open(subset_manifest) as f:
            entries = json.load(f)
        print(
            f"  [info] Using subset manifest ({len(entries)} composites): "
            f"{subset_manifest}"
        )
    elif subset:
        entries = entries[:subset]

    if detected_family and detected_family != "unknown" and tag_family == "tag36ARTag":
        print(f"  [info] Using tag family from testset manifest: {detected_family}")
        tag_family = detected_family
    elif (
        detected_family
        and detected_family != tag_family
        and detected_family != "unknown"
    ):
        print(
            f"  [warn] Manifest tag_family={detected_family!r} differs from "
            f"--tag-family={tag_family!r}; using --tag-family"
        )

    total_composites = len(entries)
    total_crops = sum(len(e["crops"]) for e in entries)
    crops_with_gt = sum(1 for e in entries for c in e["crops"] if c["gt_ids"])
    print(
        f"Loaded {len(video_keys)} videos, {total_composites} composites, "
        f"{total_crops} crops"
    )
    print(f"  with GT: {crops_with_gt}  |  valid IDs: {len(valid_ids)}")
    if yield_mode:
        print(
            "\n  " + "!" * 56 + "\n"
            "  !!  YOLO MODE — no ground truth in this testset.        !!\n"
            "  !!  Optimising raw decode YIELD (fraction of YOLO      !!\n"
            "  !!  boxes that decode), NOT recall/accuracy. Scores    !!\n"
            "  !!  are a yield rate; verify the winning preset with   !!\n"
            "  !!  'yoto detect' before trusting it.                  !!\n"
            "  " + "!" * 56 + "\n"
        )
    print(
        f"  search_space={search_space}  tag_family={tag_family}  "
        f"n_trials={n_trials}  n_jobs={n_jobs}"
    )
    print(
        f"  speed_weight={speed_weight} (floor={speed_floor_recall})  "
        f"synthetic_blur={synthetic_blur}"
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sampler = optuna.samplers.TPESampler(multivariate=True, group=True, seed=42)

    study = optuna.create_study(
        study_name=study_name,
        storage=make_storage(storage),
        direction="maximize",
        load_if_exists=True,
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=pruner_startup_trials,
            n_warmup_steps=pruner_warmup_steps,
        ),
    )

    if seed_params is not None:
        with open(seed_params) as f:
            blob = json.load(f)
        seed = dict(blob.get("params", blob))
        try:
            study.enqueue_trial(seed, skip_if_exists=True)
            print(f"  Seeded with params from {seed_params}")
        except Exception as exc:
            print(f"  [warn] Could not seed: {exc}")

    live = _LiveDisplay(n_trials=n_trials, n_show=5)

    study.optimize(
        partial(
            _objective,
            entries=entries,
            valid_ids=valid_ids,
            search_space=search_space,
            tag_family=tag_family,
            prune_eval_interval=prune_eval_interval,
            speed_weight=speed_weight,
            speed_floor_recall=speed_floor_recall,
            synthetic_blur=synthetic_blur,
            max_tag_id=max_tag_id,
            silence_ids=silence_ids,
            yield_mode=yield_mode,
        ),
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=False,
        callbacks=[live],
    )

    best = study.best_trial
    completed = [t for t in study.trials if t.value is not None]

    print(f"\n{'='*60}\nOPTIMIZATION RESULTS\n{'='*60}")
    print(f"Best score: {best.value:.4f}  (trial #{best.number})")
    print("Best parameters:")
    for k, v in sorted(best.params.items()):
        print(f"  {k}: {v}")
    print("Best metrics:")
    for k, v in sorted(best.user_attrs.items()):
        vstr = f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"  {k}: {vstr}")

    print(f"\nTop 5 of {len(completed)} completed trials:")
    for i, t in enumerate(sorted(completed, key=lambda x: x.value, reverse=True)[:5]):
        print(f"  #{i+1}  {_trial_line(t, best.number)}")

    # Save best params JSON
    out_path = out_path_dir / f"best_params_{study_name}.json"
    with open(out_path, "w") as f:
        json.dump(
            {"score": best.value, "params": best.params, "metrics": best.user_attrs},
            f,
            indent=2,
        )
    print(f"\nBest params saved to: {out_path}")

    # Save all trials CSV
    try:
        csv_path = out_path_dir / f"trials_{study_name}.csv"
        trials_df = study.trials_dataframe()
        # Optuna's 'duration' is a pandas Timedelta, which serializes to CSV
        # as "0 days 00:00:10.78"; store plain float seconds instead.
        if "duration" in trials_df.columns:
            trials_df["duration"] = trials_df["duration"].dt.total_seconds()
            trials_df = trials_df.rename(columns={"duration": "duration_s"})
        trials_df.to_csv(csv_path, index=False)
        print(f"All trials saved to:  {csv_path}")
    except Exception as exc:
        print(f"[warn] Could not save trials CSV: {exc}")

    _viz_dir = Path(viz_dir) if viz_dir else out_path_dir / f"viz_{study_name}"

    # Save Optuna HTML visualisation plots
    _save_optuna_plots(study, _viz_dir)

    if not no_viz:
        from .viz import render_comparison

        param_sets: list[tuple[str, dict[str, Any]]] = []
        if seed_params is not None:
            with open(seed_params) as f:
                blob = json.load(f)
            param_sets.append(("seed", dict(blob.get("params", blob))))
        param_sets.append((f"optuna_best (score={best.value:.4f})", best.params))
        try:
            render_comparison(
                entries,
                param_sets,
                _viz_dir,
                tag_family=tag_family,
                n_samples=viz_samples,
                max_tag_id=max_tag_id,
                silence_ids=silence_ids,
            )
        except Exception as exc:
            print(f"[viz] failed: {exc}")

    return out_path
