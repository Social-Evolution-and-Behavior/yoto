"""Tests for :mod:`yoto.io`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from yoto.constants import CORNER_COLS, PICKLE_EXT
from yoto.io import load_data


def _write_recording(root, stems, dataname="_run"):
    """Write one clean pickle per stem, each with a distinct center_x.

    Shaped like a real clean pickle: two tags, and a ``corners`` metric whose
    cells hold (4, 2) arrays.
    """
    clean_dir = root / "tracking" / "clean_data"
    clean_dir.mkdir(parents=True)
    for i, stem in enumerate(stems):
        df = pd.DataFrame(
            {
                ("7", "center_x"): [float(i)] * 3,
                ("7", "corners"): [np.zeros((4, 2))] * 3,
                ("9", "center_x"): [float(i)] * 3,
                ("9", "corners"): [np.zeros((4, 2))] * 3,
            },
            index=pd.RangeIndex(3, name="frame"),
        )
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        df.to_pickle(clean_dir / f"{stem}{dataname}_clean.pkl")
    return root


def test_video_nb_selects_one_video(tmp_path):
    root = _write_recording(tmp_path, ["000000", "000001", "000002"])

    df = load_data(root, dataname="_run", video_nb=1)

    assert list(df.index.get_level_values("source").unique()) == ["000001"]
    assert df[("7", "center_x")].iloc[0] == 1.0


def test_video_nb_none_loads_every_video(tmp_path):
    root = _write_recording(tmp_path, ["000000", "000001", "000002"])

    df = load_data(root, dataname="_run")

    assert len(df.index.get_level_values("source").unique()) == 3


def test_video_nb_indexes_from_the_end(tmp_path):
    root = _write_recording(tmp_path, ["000000", "000001", "000002"])

    df = load_data(root, dataname="_run", video_nb=-1)

    assert list(df.index.get_level_values("source").unique()) == ["000002"]


def test_video_nb_is_positional_not_a_filename_number(tmp_path):
    """A video skipped at clean time shifts every later position by one.

    Documented behaviour, and the reason ``source`` stays in the index.
    """
    root = _write_recording(tmp_path, ["000000", "000002", "000003"])

    df = load_data(root, dataname="_run", video_nb=1)

    assert list(df.index.get_level_values("source").unique()) == ["000002"]


def test_video_nb_out_of_range_names_the_count(tmp_path):
    root = _write_recording(tmp_path, ["000000", "000001"])

    with pytest.raises(IndexError, match="2 video"):
        load_data(root, dataname="_run", video_nb=5)


def test_corners_are_kept_by_default(tmp_path):
    root = _write_recording(tmp_path, ["000000"])

    df = load_data(root, dataname="_run")

    assert "corners" in df.columns.get_level_values(1)


def test_corners_false_drops_only_corners(tmp_path):
    root = _write_recording(tmp_path, ["000000", "000001"])

    df = load_data(root, dataname="_run", corners=False)

    assert set(df.columns.get_level_values(1)) == {"center_x"}
    assert set(df.columns.get_level_values(0)) == {"7", "9"}
    assert len(df) == 6


def _write_new_format(root, stems, dataname="_run"):
    """Write clean pickles in the current format: flat float32, zstd."""
    clean_dir = root / "tracking" / "clean_data"
    clean_dir.mkdir(parents=True)
    for i, stem in enumerate(stems):
        data = {}
        for tag in ("7", "9"):
            data[(tag, "center_x")] = [float(i)] * 3
            for c in CORNER_COLS:
                data[(tag, c)] = [float(i)] * 3
        df = pd.DataFrame(data, index=pd.RangeIndex(3, name="frame"))
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        df.to_pickle(clean_dir / f"{stem}{dataname}_clean{PICKLE_EXT}")
    return root


def test_load_data_finds_compressed_pickles(tmp_path):
    root = _write_new_format(tmp_path, ["000000", "000001"])

    df = load_data(root, dataname="_run")

    assert len(df) == 6
    assert list(df.index.get_level_values("source").unique()) == ["000000", "000001"]


def test_source_stem_excludes_the_compound_extension(tmp_path):
    """A '.pkl.zst' suffix must not leak into the source index level."""
    root = _write_new_format(tmp_path, ["000000"])

    df = load_data(root, dataname="_run")

    assert df.index.get_level_values("source")[0] == "000000"


def test_corners_false_drops_the_flat_corner_columns(tmp_path):
    root = _write_new_format(tmp_path, ["000000"])

    df = load_data(root, dataname="_run", corners=False)

    assert set(df.columns.get_level_values(1)) == {"center_x"}


def test_compressed_and_plain_pickles_load_together(tmp_path):
    """A recording part-cleaned before compression still loads as one frame."""
    root = _write_new_format(tmp_path, ["000001"])
    clean_dir = root / "tracking" / "clean_data"
    old = pd.DataFrame({("7", "center_x"): [9.0]}, index=pd.RangeIndex(1, name="frame"))
    old.columns = pd.MultiIndex.from_tuples(old.columns)
    old.to_pickle(clean_dir / "000000_run_clean.pkl")

    df = load_data(root, dataname="_run")

    assert list(df.index.get_level_values("source").unique()) == ["000000", "000001"]


def test_corners_false_tolerates_pickles_without_corners(tmp_path):
    """Older pickles predate the metric; dropping must not raise KeyError."""
    clean_dir = tmp_path / "tracking" / "clean_data"
    clean_dir.mkdir(parents=True)
    df = pd.DataFrame(
        {("7", "center_x"): [1.0, 2.0]}, index=pd.RangeIndex(2, name="frame")
    )
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    df.to_pickle(clean_dir / "000000_run_clean.pkl")

    out = load_data(tmp_path, dataname="_run", corners=False)

    assert set(out.columns.get_level_values(1)) == {"center_x"}
