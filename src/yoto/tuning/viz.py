"""End-of-run visualisation for AprilTag preset optimisation.

Produces one PNG per sampled composite showing:
  - The raw composite with GT positions (blue circles) and detected tag
    outlines (green = true positive, orange = false positive, red = missed GT).
  - The preprocessed grayscale used by the detector (scaled back to native res).

One block of rows is rendered per param set so it's easy to compare
seed params vs Optuna best side-by-side.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .preprocess import preprocess_composite

# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

_GREEN = (0, 200, 0)  # true positive
_ORANGE = (0, 165, 255)  # false positive
_RED = (0, 0, 220)  # missed GT
_BLUE = (220, 100, 0)  # GT position dot
_GRAY = (130, 130, 130)  # crop boundary
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)


def _banner(
    width: int, text: str, bg: tuple[int, int, int] = (40, 40, 40)
) -> np.ndarray:
    h = 22
    bar = np.full((h, width, 3), bg, dtype=np.uint8)
    cv2.putText(
        bar, text, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, _WHITE, 1, cv2.LINE_AA
    )
    return bar


def _annotate_composite(
    composite_bgr: np.ndarray,
    tags: list[dict[str, Any]],
    crop_entries: list[dict[str, Any]],
    upscale: float,
) -> np.ndarray:
    """Draw crop boundaries, GT dots, and detected tag outlines on a copy."""
    canvas = composite_bgr.copy()

    all_gt_ids: set[int] = set()
    for crop in crop_entries:
        all_gt_ids.update(crop["gt_ids"])

    # Detected ids per crop
    detected_per_crop: dict[int, set[int]] = {
        c["crop_idx"]: set() for c in crop_entries
    }
    for tag in tags:
        tid = int(tag["id"])
        cx, cy = float(tag["center"][0]) / upscale, float(tag["center"][1]) / upscale
        for c in crop_entries:
            x_start = c["canvas_x_offset"]
            x_end = x_start + c["crop_shape"][1]
            y_end = c["crop_shape"][0]
            if x_start <= cx < x_end and 0 <= cy < y_end:
                detected_per_crop[c["crop_idx"]].add(tid)
                break

    # Draw crop bounding boxes
    for c in crop_entries:
        x0 = c["canvas_x_offset"]
        x1 = x0 + c["crop_shape"][1]
        y1 = c["crop_shape"][0]
        cv2.rectangle(canvas, (x0, 0), (x1 - 1, y1 - 1), _GRAY, 1)

    # GT positions (blue dots)
    for c in crop_entries:
        for gx, gy in c.get("gt_positions_in_composite", []):
            cv2.circle(canvas, (int(gx), int(gy)), 3, _BLUE, -1)

    # Detected tag outlines
    for tag in tags:
        tid = int(tag["id"])
        corners = np.array(tag["lb-rb-rt-lt"], dtype=np.float32) / upscale
        cx_nat = float(tag["center"][0]) / upscale
        cy_nat = float(tag["center"][1]) / upscale

        is_gt = tid in all_gt_ids
        color = _GREEN if is_gt else _ORANGE
        cv2.polylines(canvas, [corners.astype(np.int32)], True, color, 1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            str(tid),
            (int(cx_nat) + 3, int(cy_nat) - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            color,
            1,
            cv2.LINE_AA,
        )

    # Mark missed GT ids
    for c in crop_entries:
        gt = set(c["gt_ids"])
        det = detected_per_crop.get(c["crop_idx"], set())
        missed = gt - det
        if missed:
            x0 = c["canvas_x_offset"]
            cv2.putText(
                canvas,
                f"miss:{sorted(missed)}",
                (x0 + 2, 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                _RED,
                1,
                cv2.LINE_AA,
            )

    return canvas


# ---------------------------------------------------------------------------
# Per-param-set panel
# ---------------------------------------------------------------------------


def _render_panel(
    composite_bgr: np.ndarray,
    params: dict[str, Any],
    crop_entries: list[dict[str, Any]],
    tag_family: str,
    label: str,
    max_tag_id: int = 9999,
    silence_ids: frozenset[int] = frozenset(),
) -> np.ndarray:
    """Two-row panel: annotated composite on top, preprocessed gray below."""
    import apriltag

    preprocess_params = {
        k: v
        for k, v in params.items()
        if k
        not in {"decimate", "blur", "decode_sharpening", "max_hamming", "refine_edges"}
    }
    detector_params = {
        k: params[k]
        for k in (
            "decimate",
            "blur",
            "decode_sharpening",
            "max_hamming",
            "refine_edges",
        )
        if k in params
    }

    enhanced = preprocess_composite(composite_bgr, preprocess_params)
    upscale = float(preprocess_params.get("upscale", 1.0))

    detector = apriltag.apriltag(
        family=tag_family,
        threads=1,
        maxhamming=int(detector_params.get("max_hamming", 1)),
        decimate=float(detector_params.get("decimate", 1.0)),
        blur=float(detector_params.get("blur", 0.0)),
        refine_edges=int(detector_params.get("refine_edges", 0)),
        decode_sharpening=float(detector_params.get("decode_sharpening", 0.25)),
    )
    tags = [
        t
        for t in detector.detect(enhanced)
        if int(t["id"]) <= max_tag_id and int(t["id"]) not in silence_ids
    ]

    # Stats
    all_gt: set[int] = set()
    for c in crop_entries:
        all_gt.update(c["gt_ids"])
    detected = {int(t["id"]) for t in tags}
    tp = detected & all_gt
    fp = detected - all_gt
    missed = all_gt - detected

    stat = f"{label}  |  TP={len(tp)}/{len(all_gt)}  FP={len(fp)}  missed={len(missed)}"

    # Row 1: annotated composite
    annotated = _annotate_composite(composite_bgr, tags, crop_entries, upscale)
    W = annotated.shape[1]

    # Row 2: preprocessed gray scaled back to native width
    gray_native = cv2.resize(
        enhanced, (W, composite_bgr.shape[0]), interpolation=cv2.INTER_AREA
    )
    gray_bgr = cv2.cvtColor(gray_native, cv2.COLOR_GRAY2BGR)

    sep = np.full((2, W, 3), 180, dtype=np.uint8)
    panel = np.vstack([_banner(W, stat), annotated, sep, gray_bgr])
    return panel


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_comparison(
    entries: list[dict[str, Any]],
    param_sets: list[tuple[str, dict[str, Any]]],
    out_dir: str | Path,
    tag_family: str,
    *,
    n_samples: int = 8,
    seed: int = 0,
    max_tag_id: int = 9999,
    silence_ids: frozenset[int] = frozenset(),
) -> None:
    """Render side-by-side comparison PNGs for a sample of composites.

    Parameters
    ----------
    entries:
        Flat list of frame entries (as returned by :func:`load_manifests`).
    param_sets:
        List of ``(label, params)`` tuples.  Each gets its own block of rows
        in the output PNG so it's easy to compare visually.
    out_dir:
        Directory where PNGs and ``summary.tsv`` are written.
    tag_family:
        AprilTag family string — must match what the optimizer used.
    n_samples:
        Number of composites to visualise.
    seed:
        RNG seed for reproducible sampling.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    with_gt = [e for e in entries if e.get("all_frame_gt_ids")]
    pool = with_gt if with_gt else entries
    sample = rng.sample(pool, min(n_samples, len(pool)))

    summary_rows: list[str] = ["composite\tlabel\tTP\tFP\tmissed\tdetected"]

    for entry in sample:
        import os

        comp_path = os.path.join(entry["_composites_dir"], entry["composite_file"])
        composite = cv2.imread(comp_path)
        if composite is None:
            continue

        crop_entries = entry["crops"]
        all_gt: set[int] = set()
        for c in crop_entries:
            all_gt.update(c["gt_ids"])

        panels: list[np.ndarray] = []
        for label, params in param_sets:
            panel = _render_panel(
                composite,
                params,
                crop_entries,
                tag_family,
                label,
                max_tag_id=max_tag_id,
                silence_ids=silence_ids,
            )
            panels.append(panel)

            # Summary stats
            preprocess_params = {
                k: v
                for k, v in params.items()
                if k
                not in {
                    "decimate",
                    "blur",
                    "decode_sharpening",
                    "max_hamming",
                    "refine_edges",
                }
            }
            import apriltag

            enhanced = preprocess_composite(composite, preprocess_params)
            det = apriltag.apriltag(
                family=tag_family,
                threads=1,
                maxhamming=int(params.get("max_hamming", 1)),
                decimate=float(params.get("decimate", 1.0)),
                blur=float(params.get("blur", 0.0)),
                refine_edges=int(params.get("refine_edges", 0)),
                decode_sharpening=float(params.get("decode_sharpening", 0.25)),
            )
            detected = {int(t["id"]) for t in det.detect(enhanced)}
            tp = detected & all_gt
            fp = detected - all_gt
            missed = all_gt - detected
            summary_rows.append(
                f"{entry['composite_file']}\t{label}\t"
                f"{len(tp)}/{len(all_gt)}\t{len(fp)}\t{len(missed)}\t"
                f"{sorted(detected)}"
            )

        W = panels[0].shape[1]
        thick_sep = np.full((6, W, 3), 80, dtype=np.uint8)
        stacked = panels[0]
        for p in panels[1:]:
            stacked = np.vstack([stacked, thick_sep, p])

        out_path = out_dir / f"viz_{entry['composite_file']}.png"
        cv2.imwrite(str(out_path), stacked)

    summary_path = out_dir / "summary.tsv"
    summary_path.write_text("\n".join(summary_rows))
    print(f"[viz] {len(sample)} images + summary.tsv → {out_dir}")
