"""Internal: per-worker progress status files for parallel runs.

Each worker writes a single-line status file (throttled to ~1Hz) giving its
current frame count, total frames, and timing info.  The parent CLI reads
these files to render tqdm-style progress lines in progress.txt.

Inactive outside of parallel runs (``YOTO_STATUS_DIR`` env var unset).
"""

from __future__ import annotations

import hashlib
import os
import time
from typing import Callable


def _status_file_path(status_dir: str, video_path: str) -> str:
    """Return the path to the status file for ``video_path``.

    Uses a short hash of the absolute video path as the filename so that
    videos with identical stems in different recording folders don't collide.
    """
    key = hashlib.md5(os.path.abspath(video_path).encode()).hexdigest()[:10]
    return os.path.join(status_dir, f"{key}.status")


def make_status_updater(
    video_path: str,
    total: int,
    min_interval: float = 1.0,
) -> Callable[[int], None]:
    """Return a callable ``update(current)`` that writes a single-line status
    file for this video, throttled to one write per ``min_interval`` seconds
    (plus one final write when ``current >= total``).

    No-op when ``YOTO_STATUS_DIR`` is not set.
    """
    status_dir = os.environ.get("YOTO_STATUS_DIR")
    if not status_dir:
        return lambda _current: None

    path = _status_file_path(status_dir, video_path)
    start = time.time()
    state = {"last_write": 0.0}

    def _update(current: int) -> None:
        now = time.time()
        if now - state["last_write"] < min_interval and current < total:
            return
        state["last_write"] = now
        try:
            with open(path, "w") as f:
                f.write(f"{current}\t{total}\t{start}\t{now}\n")
        except OSError:
            pass

    return _update


def read_status(
    status_dir: str, video_path: str
) -> tuple[int, int, float, float] | None:
    """Return ``(current, total, start, updated)`` for ``video_path``,
    or ``None`` if no status file exists or it is unparseable."""
    path = _status_file_path(status_dir, video_path)
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            line = f.readline().rstrip("\n")
    except OSError:
        return None
    parts = line.split("\t")
    if len(parts) < 4:
        return None
    try:
        return int(parts[0]), int(parts[1]), float(parts[2]), float(parts[3])
    except ValueError:
        return None
