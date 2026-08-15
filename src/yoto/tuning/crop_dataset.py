"""Backend for `yoto train build-crop-dataset`.

Turns the crops that ``build-testset`` already wrote into a standard
image-classification dataset (ImageFolder layout), for training a CNN that
recognises individual tags as a backup when AprilTag decoding fails.

``build-testset`` has already done the expensive part — every YOLO box is
cropped to disk and labelled against the clean pickle in ``manifest.json`` —
so this module decodes no video and crops nothing.  It selects, splits, and
copies.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm

from yoto.constants import (
    ASS_TYPE_INTERPOLATED,
    ASS_TYPE_ORIGINAL,
    ASS_TYPE_YOLO_INFERRED,
)

#: Parallel file operations.  The crops live on NFS, where each copy is a few
#: round-trips on ~2 KB, so overlapping the waits is what makes this fast.
DEFAULT_COPY_JOBS = 16

#: Provenance codes accepted under each ``--ass-types`` choice.
ASS_TYPE_SETS: dict[str, set[int]] = {
    "original": {ASS_TYPE_ORIGINAL},
    "all": {ASS_TYPE_ORIGINAL, ASS_TYPE_INTERPOLATED, ASS_TYPE_YOLO_INFERRED},
}

CropRecord = dict[str, Any]

MANIFEST_COLUMNS = [
    "crop_path",
    "tag_id",
    "experiment",
    "frame",
    "crop_idx",
    "ass_type",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "split",
    "src_path",
]


# --------------------------------------------------------------------------- #
# discovery
# --------------------------------------------------------------------------- #


def discover_experiments(testset_dir: str | Path) -> list[Path]:
    """Return experiment dirs that actually carry a manifest, sorted by name.

    Discovery goes through ``*/manifest.json`` rather than a directory
    listing: ``build-testset`` creates the experiment dir (and an empty
    ``crops/``) *before* writing its manifest, so a run in progress — or one
    that was interrupted — leaves dirs that look valid but hold no labels.
    """
    return sorted(
        (m.parent for m in Path(testset_dir).glob("*/manifest.json")),
        key=lambda p: p.name,
    )


def load_crop_records(exp_dir: str | Path) -> list[CropRecord]:
    """Return one labelled record per usable crop in *exp_dir*.

    A crop is usable when exactly one tag centre falls inside it.  Crops with
    no tag are unlabelled; crops with several are ambiguous.  Both are dropped.

    Raises
    ------
    json.JSONDecodeError
        If the manifest is malformed — callers skip such experiments.
    """
    exp_dir = Path(exp_dir)
    doc = json.loads((exp_dir / "manifest.json").read_text())

    records: list[CropRecord] = []
    for frame in doc.get("frames", []):
        frame_idx = frame["frame_idx"]
        for crop in frame.get("crops", []):
            gt_ids = crop.get("gt_ids", [])
            if len(gt_ids) != 1:
                continue
            bbox = crop.get("bbox_xyxy", [0, 0, 0, 0])
            records.append(
                {
                    "tag_id": int(gt_ids[0]),
                    "ass_type": int(crop["gt_ass_types"][0]),
                    "experiment": exp_dir.name,
                    "frame": int(frame_idx),
                    "crop_idx": int(crop["crop_idx"]),
                    "bbox": [int(v) for v in bbox],
                    "src_path": str(exp_dir / "crops" / crop["crop_file"]),
                }
            )
    return records


# --------------------------------------------------------------------------- #
# split
# --------------------------------------------------------------------------- #


def split_experiments(
    experiments: list[str], val_frac: float
) -> tuple[list[str], list[str]]:
    """Hold out whole experiments for validation, taking the last ones.

    Separate recording sessions make a leak-free boundary: crops of the same
    ant in nearby frames cannot straddle the split.  The ``n - 1`` clamp keeps
    at least one experiment in train, so a single-experiment dataset yields an
    empty val list and the caller falls back to a per-class tail split.
    """
    n = len(experiments)
    if n < 2:
        return list(experiments), []
    n_val = min(n - 1, max(1, round(val_frac * n)))
    return experiments[: n - n_val], experiments[n - n_val :]


def _sort_key(rec: CropRecord) -> tuple[str, int, int]:
    return (rec["experiment"], rec["frame"], rec["crop_idx"])


def _tail_split(
    records: list[CropRecord], val_frac: float
) -> tuple[list[CropRecord], list[CropRecord]]:
    """Split one class's records by taking the tail *val_frac* in time order.

    Used where an experiment-level split cannot apply: a single-experiment
    dataset, or a class that appears only in training experiments.
    """
    ordered = sorted(records, key=_sort_key)
    n_val = min(len(ordered) - 1, max(1, round(val_frac * len(ordered))))
    if n_val < 1:
        return ordered, []
    return ordered[:-n_val], ordered[-n_val:]


def _stride_cap(records: list[CropRecord], cap: int) -> list[CropRecord]:
    """Thin *records* to at most *cap*, spread across the recording.

    An even stride preserves temporal (and so lighting/pose) coverage; simple
    truncation would keep one contiguous stretch of the video.
    """
    ordered = sorted(records, key=_sort_key)
    if len(ordered) <= cap:
        return ordered
    idx = (
        [round(i * (len(ordered) - 1) / (cap - 1)) for i in range(cap)]
        if cap > 1
        else [0]
    )
    return [ordered[i] for i in sorted(set(idx))]


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #


def _counts_by_class(records: Iterable[CropRecord]) -> dict[str, int]:
    c: Counter[str] = Counter(str(r["tag_id"]) for r in records)
    return dict(sorted(c.items(), key=lambda kv: int(kv[0])))


def _provenance(records: Iterable[CropRecord]) -> dict[str, int]:
    c: Counter[str] = Counter(str(r["ass_type"]) for r in records)
    return dict(sorted(c.items()))


def build_crop_dataset(
    testset_dir: str | Path,
    out_dir: str | Path,
    *,
    val_frac: float = 0.2,
    ass_types: str = "all",
    val_ass_types: str = "original",
    min_count: int = 100,
    max_per_class: int | None = None,
    symlink: bool = False,
    force: bool = False,
    jobs: int = DEFAULT_COPY_JOBS,
) -> dict[str, Any]:
    """Build an ImageFolder crop dataset from a ``build-testset`` output tree.

    Parameters
    ----------
    testset_dir
        Directory holding ``<experiment>/manifest.json`` + ``crops/``.
    out_dir
        Dataset root; ``train/<tag_id>/`` and ``val/<tag_id>/`` are written
        under it, alongside ``manifest.csv`` and ``dataset.json``.
    val_frac
        Fraction of experiments held out for validation.
    ass_types, val_ass_types
        ``"all"`` or ``"original"`` — which provenance codes may enter each
        split.  Validation defaults to decoded tags only: scoring against
        trajectory-inferred labels partly measures agreement with the chaining
        heuristic rather than correctness.
    min_count
        Classes with fewer usable crops than this are dropped.
    max_per_class
        Optional per-split cap, applied by even stride.
    symlink
        Symlink instead of copying.  Copies are self-contained and survive a
        rebuild of the testset, which is the common case.
    force
        Replace an existing dataset directory instead of refusing.
    jobs
        Threads used to place files.  The default overlaps NFS round-trips;
        pass 1 for serial I/O.

    Returns
    -------
    dict
        The ``dataset.json`` document.
    """
    testset_dir = Path(testset_dir)
    out_dir = Path(out_dir)

    try:
        train_set = ASS_TYPE_SETS[ass_types]
        val_set = ASS_TYPE_SETS[val_ass_types]
    except KeyError as exc:
        raise ValueError(
            f"ass-types must be one of {sorted(ASS_TYPE_SETS)}, got {exc}"
        ) from None

    experiments = discover_experiments(testset_dir)
    if not experiments:
        raise ValueError(
            f"No experiments with a manifest.json under {testset_dir}. "
            "Run 'yoto train build-testset' first."
        )

    disable_tqdm = bool(os.environ.get("YOTO_NO_PROGRESS"))

    records: list[CropRecord] = []
    skipped: list[str] = []
    for exp in tqdm(
        experiments, desc="Reading manifests", unit="exp", disable=disable_tqdm
    ):
        try:
            records.extend(load_crop_records(exp))
        except (json.JSONDecodeError, KeyError, OSError):
            skipped.append(exp.name)

    kept_experiments = [e.name for e in experiments if e.name not in skipped]
    if not records:
        raise ValueError(f"No labelled crops found under {testset_dir}.")

    # Class filtering happens before splitting, so a class is judged on its
    # whole population rather than on whichever split it happened to land in.
    per_class: dict[int, list[CropRecord]] = defaultdict(list)
    for rec in records:
        per_class[rec["tag_id"]].append(rec)
    dropped = sorted(t for t, rs in per_class.items() if len(rs) < min_count)
    per_class = {t: rs for t, rs in per_class.items() if len(rs) >= min_count}
    if not per_class:
        raise ValueError(
            f"Every class has fewer than {min_count} crops; lower --min-count."
        )

    train_exps, val_exps = split_experiments(kept_experiments, val_frac)
    val_exp_set = set(val_exps)

    train: list[CropRecord] = []
    val: list[CropRecord] = []
    tail_split_classes: list[int] = []

    for tag_id, recs in sorted(per_class.items()):
        if val_exp_set:
            tr = [r for r in recs if r["experiment"] not in val_exp_set]
            va = [r for r in recs if r["experiment"] in val_exp_set]
        else:
            tr, va = [], []
        # A class confined to training experiments (or a single-experiment
        # dataset) still has to appear in val, so fall back to a tail split.
        if not va or not tr:
            tr, va = _tail_split(recs, val_frac)
            tail_split_classes.append(tag_id)

        tr = [r for r in tr if r["ass_type"] in train_set]
        va = [r for r in va if r["ass_type"] in val_set]

        if max_per_class is not None:
            tr = _stride_cap(tr, max_per_class)
            va = _stride_cap(va, max_per_class)

        train.extend(tr)
        val.extend(va)

    if out_dir.exists():
        if not force:
            raise FileExistsError(
                f"{out_dir} already exists; pass force=True (--force) to rebuild."
            )
        # A previous dataset holds tens of thousands of files.  Plain rmtree
        # unlinks them one at a time and prints nothing, which on NFS reads as
        # a hang for minutes before the copy bar ever appears.
        stale = [p for p in out_dir.rglob("*") if p.is_file() or p.is_symlink()]
        if stale:
            with tqdm(
                total=len(stale),
                desc="Removing old dataset",
                unit="file",
                miniters=100,
                disable=disable_tqdm,
            ) as bar:
                with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
                    for _ in pool.map(os.unlink, stale):
                        bar.update(1)
        shutil.rmtree(out_dir)

    todo = [("train", r) for r in train] + [("val", r) for r in val]

    # Create each class directory once.  Doing it per crop costs an extra
    # round-trip per file, which on NFS (~3 ms/stat) dominates the copy of a
    # ~2 KB jpg.
    for split, tag_id in {(s, r["tag_id"]) for s, r in todo}:
        (out_dir / split / str(tag_id)).mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    pairs: list[tuple[str, Path]] = []
    for split, rec in todo:
        name = f"{rec['experiment']}_f{rec['frame']:06d}_c{rec['crop_idx']:03d}.jpg"
        dst = out_dir / split / str(rec["tag_id"]) / name
        pairs.append((rec["src_path"], dst))
        x1, y1, x2, y2 = rec["bbox"]
        rows.append(
            {
                "crop_path": str(dst.relative_to(out_dir)),
                "tag_id": rec["tag_id"],
                "experiment": rec["experiment"],
                "frame": rec["frame"],
                "crop_idx": rec["crop_idx"],
                "ass_type": rec["ass_type"],
                "bbox_x1": x1,
                "bbox_y1": y1,
                "bbox_x2": x2,
                "bbox_y2": y2,
                "split": split,
                "src_path": rec["src_path"],
            }
        )

    def _place(pair: tuple[str, Path]) -> None:
        src, dst = pair
        if symlink:
            dst.symlink_to(Path(src).resolve())
        else:
            shutil.copyfile(src, dst)

    # These crops live on NFS: each copy is a handful of network round-trips
    # on a couple of KB, so the work is latency-bound, not bandwidth-bound.
    # Threads overlap those waits (the GIL is released during I/O) and cut the
    # wall time by roughly the worker count.
    with tqdm(
        total=len(pairs),
        desc="Linking crops" if symlink else "Copying crops",
        unit="crop",
        miniters=100,
        disable=disable_tqdm,
    ) as bar:
        if jobs > 1:
            with ThreadPoolExecutor(max_workers=jobs) as pool:
                for _ in pool.map(_place, pairs):
                    bar.update(1)
        else:
            for pair in pairs:
                _place(pair)
                bar.update(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "manifest.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    doc: dict[str, Any] = {
        "testset_dir": str(testset_dir),
        "n_classes": len({r["tag_id"] for r in train} | {r["tag_id"] for r in val}),
        "n_crops": {"train": len(train), "val": len(val)},
        "counts": {"train": _counts_by_class(train), "val": _counts_by_class(val)},
        "provenance": {"train": _provenance(train), "val": _provenance(val)},
        "train_experiments": train_exps,
        "val_experiments": val_exps,
        "skipped_experiments": skipped,
        "dropped_classes": dropped,
        "classes_tail_split": tail_split_classes,
        "params": {
            "val_frac": val_frac,
            "ass_types": ass_types,
            "val_ass_types": val_ass_types,
            "min_count": min_count,
            "max_per_class": max_per_class,
            "symlink": symlink,
            "jobs": jobs,
        },
    }
    (out_dir / "dataset.json").write_text(json.dumps(doc, indent=2))
    return doc
