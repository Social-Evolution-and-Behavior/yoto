from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from yoto.tuning.optimize import (  # noqa: E402
    _image_score,
    _load_image_paths,
)


def test_image_score_rewards_distinct_penalizes_duplicates():
    # Pure distinct decodes score 1:1.
    assert _image_score(10, 0) == pytest.approx(10.0)
    # Duplicate decodes shave 0.2 each off the score.
    assert _image_score(10, 5) == pytest.approx(9.0)
    # More distinct always beats fewer, duplicates held equal.
    assert _image_score(12, 2) > _image_score(8, 2)


def test_load_image_paths_single_file(tmp_path):
    img = tmp_path / "one.png"
    cv2.imwrite(str(img), np.zeros((8, 8, 3), dtype=np.uint8))
    assert _load_image_paths(img) == [img]


def test_load_image_paths_directory_sorted_and_filtered(tmp_path):
    # Two images plus a non-image file that must be ignored.
    for name in ("b.jpg", "a.png"):
        cv2.imwrite(str(tmp_path / name), np.zeros((8, 8, 3), dtype=np.uint8))
    (tmp_path / "notes.txt").write_text("ignore me")
    paths = _load_image_paths(tmp_path)
    assert [p.name for p in paths] == ["a.png", "b.jpg"]


def test_load_image_paths_empty_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _load_image_paths(tmp_path)


def test_load_image_paths_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _load_image_paths(tmp_path / "does_not_exist")
