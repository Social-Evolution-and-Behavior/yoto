"""Loader for AprilTag detection presets.

A preset is a JSON file containing image-processing and AprilTag detector
parameters. Presets are merged on top of the built-in pipeline defaults
(``_build_apriltag_params_simple`` / ``_build_apriltag_params_fast`` in
``yoto.detection``) so unspecified keys keep their defaults.

Presets shipped with the package live in ``yoto/presets/``; users can
also pass an explicit path on disk (e.g. an Optuna ``best_params_*.json``).
"""

from __future__ import annotations

import json
import os
from typing import Any

PRESETS_DIR = os.path.join(os.path.dirname(__file__), "presets")


def list_builtin_presets() -> list[str]:
    """Return the names (without ``.json``) of all packaged presets."""
    if not os.path.isdir(PRESETS_DIR):
        return []
    return sorted(
        os.path.splitext(f)[0] for f in os.listdir(PRESETS_DIR) if f.endswith(".json")
    )


def resolve_preset(name_or_path: str) -> str:
    """Resolve *name_or_path* to a JSON file path.

    Tries (in order): exact path on disk, ``<presets-dir>/<name>.json``.
    """
    if os.path.isfile(name_or_path):
        return name_or_path
    candidate = os.path.join(PRESETS_DIR, name_or_path + ".json")
    if os.path.isfile(candidate):
        return candidate
    raise FileNotFoundError(
        f"AprilTag preset {name_or_path!r} not found. "
        f"Built-in presets: {list_builtin_presets()}. "
        f"Or pass an absolute path to a JSON file."
    )


def load_preset(name_or_path: str) -> dict[str, Any]:
    """Load a preset JSON file and return its parameter dict.

    Accepts two on-disk shapes:

    * a flat dict of parameters (the format used by built-in presets), or
    * an Optuna-style dump with a top-level ``"params"`` key (also
      typically containing ``"score"`` / ``"metrics"``); only the
      ``"params"`` sub-dict is returned in that case.
    """
    path = resolve_preset(name_or_path)
    with open(path, "r") as f:
        data: dict[str, Any] = json.load(f)
    if isinstance(data.get("params"), dict):
        return dict(data["params"])
    return data


def merge_preset(
    defaults: dict[str, Any],
    preset: dict[str, Any],
) -> dict[str, Any]:
    """Return a new dict = defaults overridden by *preset* keys."""
    merged = dict(defaults)
    merged.update(preset)
    return merged
