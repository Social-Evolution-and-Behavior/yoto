"""Command-line interface for the YOTO package.

Provides three sub-commands:

* ``yoto detect``  — run the YOLO + AprilTag detection pipeline
* ``yoto clean``   — clean and interpolate raw tracking data
* ``yoto render``  — produce a video overlay from cleaned data

Each sub-command mirrors the corresponding public API function.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import pandas as pd

VIDEO_EXTENSIONS = frozenset({".mp4", ".avi", ".mkv", ".mov"})

TRACKING_DIR = "tracking"
TRACKING_SUBDIRS = {
    "raw_data": "raw_data",
    "clean_data": "clean_data",
    "video_output": "video_output",
    "logs": "logs",
}


def _tracking_layout(recording_dir: str) -> dict[str, str]:
    """Return the standard ``tracking/`` sub-paths for a recording folder.

    Paths are not created — that happens at write time.
    """
    base = os.path.join(os.path.abspath(recording_dir), TRACKING_DIR)
    return {k: os.path.join(base, v) for k, v in TRACKING_SUBDIRS.items()}


def _recording_dir_for_video(video_path: str) -> str:
    """Recording dir for a video = its parent directory."""
    return os.path.dirname(os.path.abspath(video_path))


def _recording_dir_for_pickle(pkl_path: str) -> str:
    """Recording dir for a pickle.

    If the pickle is inside ``tracking/raw_data/`` or ``tracking/clean_data/``
    the recording dir is two levels up.  Otherwise it is the pickle's parent.
    """
    abs_pkl = os.path.abspath(pkl_path)
    parent = os.path.dirname(abs_pkl)
    grandparent = os.path.dirname(parent)
    if (
        os.path.basename(parent) in TRACKING_SUBDIRS.values()
        and os.path.basename(grandparent) == TRACKING_DIR
    ):
        return os.path.dirname(grandparent)
    return parent


def _normalize_dataname(name: str) -> str:
    """Ensure the dataname suffix begins with an underscore so it is
    visually separated from the video stem in output filenames."""
    if not name:
        return name
    return name if name.startswith("_") else "_" + name


def _resolve_video_paths(path: str) -> list[str]:
    """Resolve *path* to a list of video file paths.

    Handles three cases:

    1. *path* is a video file → ``[path]``
    2. *path* is a directory that directly contains video files → sorted list
    3. *path* is a directory whose immediate children are sub-directories,
       each containing video files → sorted list across all sub-dirs
    """
    if os.path.isfile(path):
        return [path]

    if not os.path.isdir(path):
        return [path]  # let downstream raise a clear error

    # Look for video files directly inside the directory
    direct_videos = sorted(
        os.path.join(path, f)
        for f in os.listdir(path)
        if os.path.isfile(os.path.join(path, f))
        and os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS
    )
    if direct_videos:
        return direct_videos

    # Otherwise look one level deeper (sub-directories of recordings)
    videos: list[str] = []
    for entry in sorted(os.listdir(path)):
        subdir = os.path.join(path, entry)
        if not os.path.isdir(subdir):
            continue
        videos.extend(
            sorted(
                os.path.join(subdir, f)
                for f in os.listdir(subdir)
                if os.path.isfile(os.path.join(subdir, f))
                and os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS
            )
        )
    return videos


def _resolve_pickle_paths(path: str, data_suffix: str = "") -> list[str]:
    """Resolve *path* to a list of raw-detection pickle files.

    For a directory, prefers ``{path}/tracking/raw_data/`` when it exists;
    otherwise looks at pickles in the directory itself; otherwise descends
    one level into sub-folders (each treated as its own recording dir).
    Files ending ``_clean.pkl`` are always excluded.
    """

    def _is_raw(f: str) -> bool:
        return f.endswith(".pkl") and not f.endswith("_clean.pkl")

    def _pickles_in(d: str) -> list[str]:
        return sorted(
            os.path.join(d, f)
            for f in os.listdir(d)
            if os.path.isfile(os.path.join(d, f)) and _is_raw(f)
        )

    def _find_in_recording(d: str) -> list[str]:
        tracking_raw = _tracking_layout(d)["raw_data"]
        if os.path.isdir(tracking_raw):
            return _pickles_in(tracking_raw)
        return _pickles_in(d)

    if os.path.isfile(path):
        # If a video file was passed, find its raw-detection pickle.
        if os.path.splitext(path)[1].lower() in VIDEO_EXTENSIONS:
            pkl = _find_pickle_for_video(path, data_suffix, raw_only=True)
            if pkl is None:
                return [path]  # let downstream raise a clear error
            return [pkl]
        return [path]
    if not os.path.isdir(path):
        return [path]  # let downstream raise a clear error

    direct = _find_in_recording(path)
    if direct:
        return direct

    pkls: list[str] = []
    for entry in sorted(os.listdir(path)):
        subdir = os.path.join(path, entry)
        if not os.path.isdir(subdir):
            continue
        pkls.extend(_find_in_recording(subdir))
    return pkls


def _find_pickle_for_video(
    video_path: str,
    data_suffix: str,
    raw_only: bool = False,
) -> str | None:
    """Locate the tracking pickle for *video_path*.

    Default preference order:

    1. ``<recording>/tracking/clean_data/<stem><data_suffix>_clean.pkl``
    2. ``<recording>/tracking/raw_data/<stem><data_suffix>.pkl``
    3. legacy path next to the video: ``<recording>/<stem><data_suffix>.pkl``

    When *raw_only* is True the clean_data candidate is skipped so the
    caller gets the un-interpolated raw detections.

    Returns the first existing path, or ``None`` if none are found.
    """
    recording = _recording_dir_for_video(video_path)
    stem = os.path.splitext(os.path.basename(video_path))[0]
    layout = _tracking_layout(recording)
    candidates = [
        os.path.join(layout["raw_data"], f"{stem}{data_suffix}.pkl"),
        os.path.join(recording, f"{stem}{data_suffix}.pkl"),
    ]
    if not raw_only:
        candidates.insert(
            0,
            os.path.join(layout["clean_data"], f"{stem}{data_suffix}_clean.pkl"),
        )
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def _configure_logging(debug: bool = False) -> None:
    """Set up root logger for CLI usage.

    Parameters
    ----------
    debug : bool
        When True, set level to DEBUG; otherwise INFO.
    """
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Sub-command: detect
# ---------------------------------------------------------------------------


def _add_detect_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``detect`` sub-command."""
    p = subparsers.add_parser(
        "detect",
        help="Run YOLO + AprilTag detection on a video",
    )
    p.add_argument(
        "video",
        help="Path to a video file, a directory of videos, "
        "or a directory of recording sub-folders",
    )
    p.add_argument(
        "output_dir",
        nargs="?",
        default=None,
        help="Output directory (default: same as video)",
    )
    p.add_argument(
        "--yoloweights",
        default="detect14.engine",
        help="Path to YOLO weights (default: detect14.engine)",
    )
    p.add_argument(
        "--dataname",
        default="_apriltagDetect14",
        help="Suffix for output data file",
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help="Use the fast NVDEC pipeline (requires PyNvVideoCodec)",
    )
    p.add_argument(
        "--parallel",
        type=int,
        default=None,
        metavar="N",
        help="Process N videos concurrently in separate worker processes "
        "(no default — must be set explicitly)",
    )
    p.add_argument(
        "--video-nb",
        type=int,
        default=None,
        metavar="INDEX",
        help="Run only the video at this 0-based index in the resolved list "
        "(useful for re-running a single failed video)",
    )
    p.add_argument(
        "--apriltag-preset",
        default=None,
        metavar="NAME_OR_PATH",
        help="AprilTag preset to apply on top of the pipeline defaults. "
        "Either a built-in preset name (e.g. 'ir') or a path to a JSON "
        "file (e.g. an Optuna best_params_*.json). Defaults stay untouched "
        "when omitted.",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug profiling output",
    )
    p.set_defaults(func=_run_detect)


def _run_single_video(
    vpath: str,
    fast: bool,
    output_dir: str | None,
    yolo_weights: str,
    data_suffix: str,
    debug: bool,
    preset: str | None = None,
) -> tuple[str, str | None]:
    """Run detection on one video. Returns ``(vpath, None)`` on success,
    or ``(vpath, traceback_text)`` on failure."""
    import traceback

    if output_dir is None:
        output_dir = _tracking_layout(_recording_dir_for_video(vpath))["raw_data"]

    try:
        if fast:
            from yoto.detection import run_detection_fast

            run_detection_fast(
                video_path=vpath,
                output_path=output_dir,
                yolo_weights=yolo_weights,
                data_suffix=data_suffix,
                debug=debug,
                preset=preset,
            )
        else:
            from yoto.detection import run_detection_simple

            run_detection_simple(
                video_path=vpath,
                output_path=output_dir,
                yolo_weights=yolo_weights,
                data_suffix=data_suffix,
                preset=preset,
            )
        return (vpath, None)
    except Exception:
        return (vpath, traceback.format_exc())


def _build_worker_cmd(
    vpath: str,
    args: argparse.Namespace,
) -> list[str]:
    """Build a ``yoto detect`` command for a single video (used by GNU parallel)."""
    cmd = [sys.executable, "-m", "yoto.cli", "detect", vpath]
    if args.output_dir:
        cmd.append(args.output_dir)
    cmd.extend(["--yoloweights", args.yoloweights])
    cmd.extend(["--dataname", args.dataname])
    if args.fast:
        cmd.append("--fast")
    if args.debug:
        cmd.append("--debug")
    if getattr(args, "apriltag_preset", None):
        cmd.extend(["--apriltag-preset", args.apriltag_preset])
    return cmd


def _format_duration(seconds: float) -> str:
    """Format a duration in seconds as a short human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def _format_tqdm_line(current: int, total: int, elapsed: float) -> str:
    """Format a tqdm-like progress string:
    ``'164/18000 [00:03<07:04, 42.06it/s]'``."""

    def _hms(s: float) -> str:
        s_int = max(0, int(s))
        h, rem = divmod(s_int, 3600)
        m, sec = divmod(rem, 60)
        if h:
            return f"{h:02d}:{m:02d}:{sec:02d}"
        return f"{m:02d}:{sec:02d}"

    rate = current / elapsed if elapsed > 0 else 0.0
    remaining = (total - current) / rate if rate > 0 and total > current else 0.0
    return f"{current}/{total} " f"[{_hms(elapsed)}<{_hms(remaining)}, {rate:.2f}it/s]"


def _write_progress_txt(
    progress_path: str,
    video_paths: list[str],
    joblog_path: str,
    start_time: float,
    jobs: int,
    status_dir: str | None = None,
    final_wall_time: float | None = None,
) -> None:
    """Write a simple human-readable progress summary based on the joblog.

    Refreshed every few seconds during a parallel run so users can check
    progress with ``cat`` or ``tail -f`` without parsing a TSV.
    """
    import datetime
    import time as _time

    completed: list[tuple[int, str, float, str]] = []
    if os.path.isfile(joblog_path):
        try:
            with open(joblog_path) as f:
                for line in f.readlines()[1:]:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 7:
                        continue
                    try:
                        seq = int(parts[0])
                    except ValueError:
                        continue
                    try:
                        runtime = float(parts[3])
                    except ValueError:
                        runtime = 0.0
                    exitval = parts[6]
                    if 1 <= seq <= len(video_paths):
                        completed.append((seq, video_paths[seq - 1], runtime, exitval))
        except OSError:
            pass

    done_seqs = {s for s, *_ in completed}
    ok_count = sum(1 for _, _, _, e in completed if e == "0")
    fail_count = len(completed) - ok_count
    elapsed = (
        final_wall_time if final_wall_time is not None else _time.time() - start_time
    )
    status = "FINISHED" if final_wall_time is not None else "RUNNING"

    lines: list[str] = []
    lines.append(f"YOTO parallel run — {len(video_paths)} video(s), {jobs} worker(s)")
    lines.append(
        f"Started:  "
        f"{datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}"
    )
    lines.append(f"Updated:  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Elapsed:  {_format_duration(elapsed)}")
    lines.append(f"Status:   {status}")
    lines.append("")
    lines.append(
        f"Progress: {len(completed)}/{len(video_paths)} complete "
        f"({ok_count} ok, {fail_count} failed)"
    )

    if completed:
        lines.append("")
        lines.append("Completed (in finish order):")
        # Sort by completion time = starttime + runtime, ascending.
        for seq, vpath, runtime, exitval in sorted(completed, key=lambda x: x[0]):
            marker = "OK  " if exitval == "0" else f"FAIL{exitval:>2}"
            lines.append(
                f"  [{seq:>3} {marker}] {_format_duration(runtime):>10}  "
                f"{os.path.basename(vpath)}"
            )

    pending = [
        (i + 1, video_paths[i])
        for i in range(len(video_paths))
        if (i + 1) not in done_seqs
    ]
    if pending and final_wall_time is None:
        from yoto._progress import read_status

        active: list[tuple[int, str, str]] = []
        queued: list[tuple[int, str]] = []
        for seq, vpath in pending:
            state = read_status(status_dir, vpath) if status_dir is not None else None
            if state is not None:
                current, total, p_start, updated = state
                # Treat a stale status file (>15s since last write) as queued.
                if _time.time() - updated <= 15.0:
                    active.append(
                        (
                            seq,
                            vpath,
                            _format_tqdm_line(current, total, updated - p_start),
                        )
                    )
                    continue
            queued.append((seq, vpath))

        lines.append("")
        lines.append(f"Remaining: {len(active)} running, {len(queued)} queued")
        name_w = max((len(os.path.basename(p)) for _, p, _ in active), default=0)
        for seq, vpath, tq in active:
            lines.append(f"  [{seq:>3}] {os.path.basename(vpath):<{name_w}}  {tq}")
        for seq, vpath in queued[:10]:
            lines.append(f"  [{seq:>3}] {os.path.basename(vpath)}  (queued)")
        if len(queued) > 10:
            lines.append(f"  ... and {len(queued) - 10} more queued")

    try:
        with open(progress_path, "w") as f:
            f.write("\n".join(lines) + "\n")
    except OSError:
        pass


def _run_parallel_gnu(
    video_paths: list[str],
    worker_tmpl: list[str],
    jobs: int,
    input_root: str,
) -> tuple[list[tuple[str, str | None]], dict[str, float], float]:
    """Dispatch one subprocess per video via GNU parallel.

    ``worker_tmpl`` is the full ``yoto <sub>`` command with ``{}`` as the
    video-path placeholder.  Each worker is a fully independent OS process
    with its own CUDA context, so a crash in one worker has no effect on
    the others.  Exit codes are captured via ``--joblog`` and mapped back
    to video paths.  A human-readable ``progress.txt`` is refreshed every
    3 seconds alongside the joblog.
    """
    import datetime
    import shutil
    import subprocess
    import threading
    import time

    if shutil.which("parallel") is None:
        print(
            "Error: --parallel requires GNU parallel.\n"
            "Install on Ubuntu with: sudo apt install parallel"
        )
        sys.exit(1)

    # Everything for this run lives under a single dated folder.
    logs_dir = _tracking_layout(input_root)["logs"]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(logs_dir, f"yoto-parallel-{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    joblog = os.path.join(run_dir, "joblog.tsv")
    progress_txt = os.path.join(run_dir, "progress.txt")
    status_dir = os.path.join(run_dir, "status")
    os.makedirs(status_dir, exist_ok=True)

    # Per-worker stdout/stderr: each video's logs land under its OWN
    # recording dir (tracking/logs/yoto-parallel-<timestamp>/<seq>-<stem>).
    # GNU parallel's {//} placeholder expands to the video's dirname.
    worker_results = os.path.join(
        "{//}",
        TRACKING_DIR,
        TRACKING_SUBDIRS["logs"],
        f"yoto-parallel-{timestamp}",
        "{#}-{/.}",
    )

    parallel_cmd = [
        "parallel",
        "--jobs",
        str(jobs),
        "--joblog",
        joblog,
        "--results",
        worker_results,
        "-q",
        *worker_tmpl,
        ":::",
        *video_paths,
    ]

    print(f"Launching {jobs} GNU parallel workers " f"for {len(video_paths)} video(s).")
    print("")
    print("Check progress at any time with:")
    print(f"  cat {progress_txt}")
    print(f"  watch -n 5 cat {progress_txt}")
    print("")
    print(f"Run folder: {run_dir}/")
    print("  progress.txt — human-readable summary (refreshed every 3s)")
    print("  joblog.tsv   — GNU parallel joblog (one line per finished video)")
    print(
        "  <recording>/tracking/logs/"
        f"yoto-parallel-{timestamp}/<seq>-<stem>/stdout  "
        "— per-worker output"
    )
    print("")

    # Disable in-worker tqdm so per-video log files stay readable; workers
    # publish their progress via YOTO_STATUS_DIR status files instead.
    worker_env = os.environ.copy()
    worker_env["YOTO_NO_PROGRESS"] = "1"
    worker_env["YOTO_STATUS_DIR"] = status_dir

    wall_start = time.time()

    # Seed progress.txt before parallel starts so `cat` works immediately.
    _write_progress_txt(
        progress_txt,
        video_paths,
        joblog,
        wall_start,
        jobs,
        status_dir=status_dir,
    )

    # Background thread refreshes progress.txt every 3s while parallel runs.
    stop_event = threading.Event()

    def _watch() -> None:
        while not stop_event.wait(3.0):
            _write_progress_txt(
                progress_txt,
                video_paths,
                joblog,
                wall_start,
                jobs,
                status_dir=status_dir,
            )

    watcher = threading.Thread(target=_watch, daemon=True)
    watcher.start()

    try:
        subprocess.run(parallel_cmd, env=worker_env)
    finally:
        stop_event.set()
        watcher.join(timeout=2.0)

    wall_time = time.time() - wall_start
    _write_progress_txt(
        progress_txt,
        video_paths,
        joblog,
        wall_start,
        jobs,
        status_dir=status_dir,
        final_wall_time=wall_time,
    )

    # Parse the joblog to build success/failure map. GNU parallel joblog:
    # Seq  Host  Starttime  JobRuntime  Send  Receive  Exitval  Signal  Command
    results: list[tuple[str, str | None]] = []
    runtimes: dict[str, float] = {}
    try:
        with open(joblog) as f:
            lines = f.readlines()
        # Skip header
        for line in lines[1:]:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 7:
                continue
            seq = int(parts[0])
            try:
                runtime = float(parts[3])
            except ValueError:
                runtime = 0.0
            exitval = parts[6]
            signal = parts[7] if len(parts) > 7 else "0"
            vpath = video_paths[seq - 1]
            runtimes[vpath] = runtime
            if exitval == "0":
                results.append((vpath, None))
            else:
                err = f"exit code {exitval}"
                if signal not in ("0", ""):
                    err += f" (signal {signal})"
                stem = os.path.splitext(os.path.basename(vpath))[0]
                worker_log = os.path.join(
                    _tracking_layout(os.path.dirname(vpath))["logs"],
                    f"yoto-parallel-{timestamp}",
                    f"{seq}-{stem}",
                )
                err += f" — see log dir: {worker_log}"
                results.append((vpath, err))
    except FileNotFoundError:
        print(f"Warning: GNU parallel did not produce a joblog at {joblog}")

    # Mark any videos missing from the joblog as failures (parallel crashed)
    seen = {p for p, _ in results}
    for vpath in video_paths:
        if vpath not in seen:
            results.append((vpath, "no joblog entry — worker never ran"))

    return results, runtimes, wall_time


def _run_detect(args: argparse.Namespace) -> None:
    """Execute the detect sub-command."""
    _configure_logging(args.debug)
    args.dataname = _normalize_dataname(args.dataname)

    video_paths = _resolve_video_paths(args.video)
    if not video_paths:
        print(f"No video files found in: {args.video}")
        sys.exit(1)

    if args.video_nb is not None:
        if args.video_nb < 0 or args.video_nb >= len(video_paths):
            print(
                f"Error: --video-nb {args.video_nb} out of range "
                f"(found {len(video_paths)} video(s), valid indices "
                f"0..{len(video_paths) - 1})"
            )
            sys.exit(1)
        selected = video_paths[args.video_nb]
        print(f"Selected video [{args.video_nb}]: {selected}")
        video_paths = [selected]

    if args.parallel is not None and args.parallel < 1:
        print("Error: --parallel must be >= 1")
        sys.exit(1)

    if args.fast and args.parallel and args.parallel > 1:
        print(
            f"WARNING: --fast with --parallel {args.parallel} runs multiple "
            "NVDEC sessions on one GPU. The RTX A6000 has a limited number "
            "of NVDEC engines (2–3); oversubscription may hurt throughput "
            "and contend on VRAM. The simple pipeline parallelises more "
            "cleanly at high worker counts."
        )

    use_parallel = (
        args.parallel is not None and args.parallel > 1 and len(video_paths) > 1
    )

    if len(video_paths) > 1:
        mode = (
            f"parallel ({args.parallel} GNU parallel workers)"
            if use_parallel
            else "sequentially"
        )
        print(f"Processing {len(video_paths)} video(s) {mode}")

    results: list[tuple[str, str | None]]
    runtimes: dict[str, float] = {}
    wall_time: float = 0.0
    if use_parallel:
        worker_tmpl = [sys.executable, "-m", "yoto.cli", "detect", "{}"]
        if args.output_dir:
            worker_tmpl.append(args.output_dir)
        worker_tmpl.extend(["--yoloweights", args.yoloweights])
        worker_tmpl.extend(["--dataname", args.dataname])
        if args.fast:
            worker_tmpl.append("--fast")
        if args.debug:
            worker_tmpl.append("--debug")
        if args.apriltag_preset:
            worker_tmpl.extend(["--apriltag-preset", args.apriltag_preset])
        input_root = (
            args.video
            if os.path.isdir(args.video)
            else os.path.dirname(os.path.abspath(args.video))
        )
        results, runtimes, wall_time = _run_parallel_gnu(
            video_paths=video_paths,
            worker_tmpl=worker_tmpl,
            jobs=args.parallel,
            input_root=input_root,
        )
    else:
        results = []
        for idx, vpath in enumerate(video_paths, start=1):
            if len(video_paths) > 1:
                print(f"\n[{idx}/{len(video_paths)}] {vpath}")
            result = _run_single_video(
                vpath=vpath,
                fast=args.fast,
                output_dir=args.output_dir,
                yolo_weights=args.yoloweights,
                data_suffix=args.dataname,
                debug=args.debug,
                preset=args.apriltag_preset,
            )
            if result[1] is not None:
                logging.getLogger(__name__).error(
                    "Detection failed for %s:\n%s", vpath, result[1]
                )
            results.append(result)

    failures = [(p, e) for p, e in results if e is not None]
    successes = len(results) - len(failures)

    if len(results) > 1 or failures:
        print("")
        print("─" * 60)
        print(f"Summary: {successes} succeeded, {len(failures)} failed")
        print("─" * 60)

        if use_parallel and runtimes:
            total_cpu_time = sum(runtimes.values())
            print(f"Total wall time:     {_format_duration(wall_time)}")
            print(
                f"Per videos time: {_format_duration(total_cpu_time/len(video_paths))} "
            )
            print("")
            print("Per-video runtime (longest first):")
            errors = {p: e for p, e in results}
            for vpath, rt in sorted(runtimes.items(), key=lambda x: x[1], reverse=True):
                status = "OK  " if errors.get(vpath) is None else "FAIL"
                name = os.path.basename(vpath)
                print(f"  [{status}] {_format_duration(rt):>12}  {name}")
            print("")

        if failures:
            print("Failed videos (retry individually with --video-nb):")
            for path, err in failures:
                summary_line = err.strip().splitlines()[-1] if err else ""
                print(f"  - {path}")
                print(f"      {summary_line}")
            sys.exit(1)


# ---------------------------------------------------------------------------
# Sub-command: clean
# ---------------------------------------------------------------------------


def _add_clean_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``clean`` sub-command."""
    p = subparsers.add_parser(
        "clean",
        help="Clean and interpolate raw tracking data",
    )
    p.add_argument(
        "input_pkl",
        help="Path to a raw-detection pickle, a video file (the matching "
        "raw pickle is looked up via --dataname), a directory of pickles, "
        "or a directory of recording sub-folders. Files ending "
        "'_clean.pkl' are skipped.",
    )
    p.add_argument(
        "--dataname",
        default="_apriltagDetect14",
        help="Suffix used to locate the raw pickle when a video file is "
        "passed as input (default: _apriltagDetect14)",
    )
    p.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output pickle path (default: <input>_clean.pkl). "
        "Only valid when input_pkl is a single file.",
    )
    p.add_argument(
        "--min-detections",
        type=int,
        default=100,
        help="Minimum detections to keep a tag ID (default: 100)",
    )
    p.add_argument(
        "--interp-limit",
        type=int,
        default=5,
        help="Max gap length for interpolation (default: 5)",
    )
    p.add_argument(
        "--max-jump",
        type=float,
        default=100.0,
        help="Max pixel distance before jump deletion (default: 100)",
    )
    p.add_argument(
        "--video-nb",
        type=int,
        default=None,
        metavar="INDEX",
        help="When input is a recording directory, clean only the pickle "
        "for the video at this 0-based index in the resolved video list",
    )
    p.set_defaults(func=_run_clean)


def _clean_one_pickle(
    pkl_path: str,
    output_path: str | None,
    min_detections: int,
    interp_limit: int,
    max_jump: float,
) -> tuple[str, str | None]:
    """Clean a single pickle. Returns ``(pkl_path, None)`` on success,
    or ``(pkl_path, error_message)`` on failure."""
    import traceback

    from yoto.cleaning import clean_tracking_data

    try:
        frame_data = pd.read_pickle(pkl_path)
        cleaned, _id_list, metrics = clean_tracking_data(
            frame_data,
            min_detections=min_detections,
            interpolation_limit=interp_limit,
            max_jump_distance=max_jump,
        )
        if output_path is None:
            clean_dir = _tracking_layout(_recording_dir_for_pickle(pkl_path))[
                "clean_data"
            ]
            os.makedirs(clean_dir, exist_ok=True)
            stem = os.path.splitext(os.path.basename(pkl_path))[0]
            output_path = os.path.join(clean_dir, f"{stem}_clean.pkl")
        cleaned.to_pickle(output_path)
        print(f"Cleaned: {output_path}")
        print(
            f"  detections={metrics['total_detections']}"
            f"/{metrics['total_samples']} | "
            f"errors={metrics['original_bad_count']} "
            f"({metrics['error_pct']:.2f}%) | "
            f"filled={metrics['filled_count']}/{metrics['total_gaps']} gaps "
            f"({metrics['filled_pct_of_gaps']:.2f}% recovered)"
        )
        return (pkl_path, None)
    except Exception:
        return (pkl_path, traceback.format_exc())


def _run_clean(args: argparse.Namespace) -> None:
    """Execute the clean sub-command."""
    _configure_logging(False)
    args.dataname = _normalize_dataname(args.dataname)

    if args.video_nb is not None:
        # Resolve videos in the directory, pick one, then look up its
        # raw pickle by --dataname suffix.
        video_paths = _resolve_video_paths(args.input_pkl)
        if not video_paths:
            print(f"No video files found in: {args.input_pkl}")
            sys.exit(1)
        if args.video_nb < 0 or args.video_nb >= len(video_paths):
            print(
                f"Error: --video-nb {args.video_nb} out of range "
                f"(found {len(video_paths)} video(s), valid indices "
                f"0..{len(video_paths) - 1})"
            )
            sys.exit(1)
        selected = video_paths[args.video_nb]
        print(f"Selected video [{args.video_nb}]: {selected}")
        pkl = _find_pickle_for_video(selected, args.dataname, raw_only=True)
        if pkl is None:
            print(f"No raw pickle found for {selected} " f"(suffix {args.dataname!r})")
            sys.exit(1)
        pkl_paths = [pkl]
    else:
        pkl_paths = _resolve_pickle_paths(args.input_pkl, args.dataname)

    if not pkl_paths:
        print(f"No raw-detection pickle files found in: {args.input_pkl}")
        sys.exit(1)

    if args.output is not None and len(pkl_paths) > 1:
        print(
            "Error: --output can only be used with a single input pickle. "
            f"Found {len(pkl_paths)} pickles; outputs will be written "
            "next to each input by default."
        )
        sys.exit(1)

    if len(pkl_paths) > 1:
        print(f"Processing {len(pkl_paths)} pickle file(s)")

    results: list[tuple[str, str | None]] = []
    for idx, pkl in enumerate(pkl_paths, start=1):
        if len(pkl_paths) > 1:
            print(f"\n[{idx}/{len(pkl_paths)}] {pkl}")
        result = _clean_one_pickle(
            pkl_path=pkl,
            output_path=args.output if len(pkl_paths) == 1 else None,
            min_detections=args.min_detections,
            interp_limit=args.interp_limit,
            max_jump=args.max_jump,
        )
        if result[1] is not None:
            logging.getLogger(__name__).error(
                "Cleaning failed for %s:\n%s", pkl, result[1]
            )
        results.append(result)

    failures = [(p, e) for p, e in results if e is not None]
    successes = len(results) - len(failures)

    if len(results) > 1 or failures:
        print("")
        print("─" * 60)
        print(f"Summary: {successes} succeeded, {len(failures)} failed")
        print("─" * 60)
        if failures:
            print("Failed pickles:")
            for path, err in failures:
                summary_line = err.strip().splitlines()[-1] if err else ""
                print(f"  - {path}")
                print(f"      {summary_line}")
            sys.exit(1)


# ---------------------------------------------------------------------------
# Sub-command: render
# ---------------------------------------------------------------------------


def _add_render_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``render`` sub-command."""
    p = subparsers.add_parser(
        "render",
        help="Render tracking overlay video",
    )
    p.add_argument(
        "video_path",
        help="Path to a video file, a directory of videos, "
        "or a directory of recording sub-folders",
    )
    p.add_argument(
        "--dataname",
        default="_apriltagDetect14",
        help="Suffix used to locate the tracking pickle and name the "
        "output video (default: _apriltagDetect14)",
    )
    p.add_argument(
        "--pkl",
        default=None,
        help="Explicit path to pickle file (single-video mode only; "
        "overrides --dataname pickle lookup)",
    )
    p.add_argument(
        "--short",
        action="store_true",
        help="Process only first 2000 frames",
    )
    p.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Output resolution scale factor (default: 1.0)",
    )
    p.add_argument(
        "--no-trails",
        action="store_true",
        help="Disable per-tag motion trails in the overlay",
    )
    p.add_argument(
        "--raw",
        action="store_true",
        help="Use the raw (un-interpolated) detection pickle and skip the "
        "cleaning pass. Implies --no-trails since raw data is gap-heavy.",
    )
    p.add_argument(
        "--text-scale",
        type=float,
        default=1.0,
        metavar="FACTOR",
        help="Multiplier for label/counter text size (default: 1.0; "
        "use <1 for smaller text, e.g. 0.5)",
    )
    p.add_argument(
        "--codec",
        choices=("auto", "h264", "hevc"),
        default="auto",
        help="Video encoder family (default: auto = hevc_nvenc first, "
        "then h264_nvenc, then libx264). Use 'h264' for smooth VLC "
        "playback; 'hevc' for smaller files.",
    )
    p.add_argument(
        "--parallel",
        type=int,
        default=None,
        metavar="N",
        help="Render N videos concurrently in separate worker processes "
        "(no default — must be set explicitly)",
    )
    p.add_argument(
        "--video-nb",
        type=int,
        default=None,
        metavar="INDEX",
        help="Render only the video at this 0-based index in the resolved "
        "list (useful for re-running a single failed video)",
    )
    p.set_defaults(func=_run_render)


def _render_single_video(
    vpath: str,
    pkl_override: str | None,
    data_suffix: str,
    scale: float,
    short: bool,
    draw_trails: bool,
    text_scale_factor: float,
    raw: bool,
    codec: str,
) -> tuple[str, str | None]:
    """Render one video. Returns ``(vpath, None)`` on success, or
    ``(vpath, traceback_text)`` on failure."""
    import traceback

    import numpy as np

    from yoto.cleaning import clean_tracking_data
    from yoto.constants import COL_ASS_TYPE
    from yoto.video import render_overlay_video

    try:
        if pkl_override is not None:
            pkl_path: str | None = pkl_override
        else:
            pkl_path = _find_pickle_for_video(vpath, data_suffix, raw_only=raw)
        if pkl_path is None or not os.path.isfile(pkl_path):
            where = (
                "tracking/raw_data/"
                if raw
                else "tracking/clean_data/ or tracking/raw_data/"
            )
            return (
                vpath,
                f"No pickle found for {vpath} "
                f"(expected under {where} with suffix {data_suffix!r})",
            )

        frame_data = pd.read_pickle(pkl_path)

        # If the pickle was produced by `yoto clean`, it already contains
        # `ass_type` columns; re-running clean_tracking_data on it would
        # duplicate those columns and break .loc indexing.  Detect by
        # column content rather than filename to be robust to renames.
        already_cleaned = COL_ASS_TYPE in frame_data.columns.get_level_values(1)

        if raw:
            cleaned = frame_data
            id_list = np.unique(cleaned.columns.get_level_values(0))
            print(f"Using raw pickle (no interpolation): {os.path.basename(pkl_path)}")
        elif already_cleaned:
            cleaned = frame_data
            id_list = np.unique(cleaned.columns.get_level_values(0))
            print(f"Using pre-cleaned pickle: {os.path.basename(pkl_path)}")
        else:
            cleaned, id_list, metrics = clean_tracking_data(frame_data)
            print(
                f"Data quality ({os.path.basename(pkl_path)}): "
                f"detections={metrics['total_detections']}"
                f"/{metrics['total_samples']} | "
                f"errors={metrics['original_bad_count']} "
                f"({metrics['error_pct']:.2f}%) | "
                f"filled={metrics['filled_count']}/{metrics['total_gaps']} gaps "
                f"({metrics['filled_pct_of_gaps']:.2f}% recovered)"
            )

        # Place the output under <recording>/tracking/video_output/.
        video_out_dir = _tracking_layout(_recording_dir_for_video(vpath))[
            "video_output"
        ]
        os.makedirs(video_out_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(vpath))[0]
        raw_tag = "_raw" if raw else ""
        if scale != 1.0:
            out_name = f"{stem}{data_suffix}{raw_tag}_scaled{scale:.2f}.mp4"
        else:
            out_name = f"{stem}{data_suffix}{raw_tag}.mp4"
        output_path = os.path.join(video_out_dir, out_name)

        output = render_overlay_video(
            video_path=vpath,
            frame_data=cleaned,
            id_list=id_list,
            output_path=output_path,
            data_name=data_suffix,
            scale=scale,
            short=short,
            draw_trails=draw_trails,
            text_scale_factor=text_scale_factor,
            codec=codec,
        )
        print(f"Output: {output}")
        return (vpath, None)
    except Exception:
        return (vpath, traceback.format_exc())


def _run_render(args: argparse.Namespace) -> None:
    """Execute the render sub-command."""
    _configure_logging(False)
    args.dataname = _normalize_dataname(args.dataname)

    video_paths = _resolve_video_paths(args.video_path)
    if not video_paths:
        print(f"No video files found in: {args.video_path}")
        sys.exit(1)

    if args.video_nb is not None:
        if args.video_nb < 0 or args.video_nb >= len(video_paths):
            print(
                f"Error: --video-nb {args.video_nb} out of range "
                f"(found {len(video_paths)} video(s), valid indices "
                f"0..{len(video_paths) - 1})"
            )
            sys.exit(1)
        selected = video_paths[args.video_nb]
        print(f"Selected video [{args.video_nb}]: {selected}")
        video_paths = [selected]

    if args.pkl is not None and len(video_paths) > 1:
        print(
            "Error: --pkl can only be used with a single input video. "
            f"Found {len(video_paths)} videos; remove --pkl so each pickle "
            "is looked up per-video via --dataname."
        )
        sys.exit(1)

    if args.parallel is not None and args.parallel < 1:
        print("Error: --parallel must be >= 1")
        sys.exit(1)

    use_parallel = (
        args.parallel is not None and args.parallel > 1 and len(video_paths) > 1
    )

    if len(video_paths) > 1:
        mode = (
            f"parallel ({args.parallel} GNU parallel workers)"
            if use_parallel
            else "sequentially"
        )
        print(f"Rendering {len(video_paths)} video(s) {mode}")

    results: list[tuple[str, str | None]]
    runtimes: dict[str, float] = {}
    wall_time: float = 0.0
    if use_parallel:
        worker_tmpl = [sys.executable, "-m", "yoto.cli", "render", "{}"]
        worker_tmpl.extend(["--dataname", args.dataname])
        worker_tmpl.extend(["--scale", str(args.scale)])
        worker_tmpl.extend(["--text-scale", str(args.text_scale)])
        worker_tmpl.extend(["--codec", args.codec])
        if args.short:
            worker_tmpl.append("--short")
        if args.no_trails:
            worker_tmpl.append("--no-trails")
        if args.raw:
            worker_tmpl.append("--raw")
        input_root = (
            args.video_path
            if os.path.isdir(args.video_path)
            else os.path.dirname(os.path.abspath(args.video_path))
        )
        results, runtimes, wall_time = _run_parallel_gnu(
            video_paths=video_paths,
            worker_tmpl=worker_tmpl,
            jobs=args.parallel,
            input_root=input_root,
        )
    else:
        results = []
        for idx, vpath in enumerate(video_paths, start=1):
            if len(video_paths) > 1:
                print(f"\n[{idx}/{len(video_paths)}] {vpath}")
            result = _render_single_video(
                vpath=vpath,
                pkl_override=args.pkl if len(video_paths) == 1 else None,
                data_suffix=args.dataname,
                scale=args.scale,
                short=args.short,
                draw_trails=(not args.no_trails) and (not args.raw),
                text_scale_factor=args.text_scale,
                raw=args.raw,
                codec=args.codec,
            )
            if result[1] is not None:
                logging.getLogger(__name__).error(
                    "Rendering failed for %s:\n%s", vpath, result[1]
                )
            results.append(result)

    failures = [(p, e) for p, e in results if e is not None]
    successes = len(results) - len(failures)

    if len(results) > 1 or failures:
        print("")
        print("─" * 60)
        print(f"Summary: {successes} succeeded, {len(failures)} failed")
        print("─" * 60)

        if use_parallel and runtimes:
            total_cpu_time = sum(runtimes.values())
            print(f"Total wall time:     {_format_duration(wall_time)}")
            print(
                "Per videos time: "
                f"{_format_duration(total_cpu_time / len(video_paths))} "
            )
            print("")
            print("Per-video runtime (longest first):")
            errors = {p: e for p, e in results}
            for vpath, rt in sorted(runtimes.items(), key=lambda x: x[1], reverse=True):
                status = "OK  " if errors.get(vpath) is None else "FAIL"
                name = os.path.basename(vpath)
                print(f"  [{status}] {_format_duration(rt):>12}  {name}")
            print("")

        if failures:
            print("Failed videos (retry individually with --video-nb):")
            for path, err in failures:
                summary_line = err.strip().splitlines()[-1] if err else ""
                print(f"  - {path}")
                print(f"      {summary_line}")
            sys.exit(1)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse arguments and dispatch to the appropriate sub-command."""
    parser = argparse.ArgumentParser(
        prog="yoto",
        description=(
            "YOTO — GPU-accelerated AprilTag tracking pipeline for ant "
            "behavioral studies"
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_get_version()}",
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    _add_detect_parser(subparsers)
    _add_clean_parser(subparsers)
    _add_render_parser(subparsers)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


def _get_version() -> str:
    """Return the package version string."""
    from yoto import __version__

    return __version__


if __name__ == "__main__":
    main()
