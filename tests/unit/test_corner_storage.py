"""Tests for corner storage and the pickle-extension helpers.

Corners moved from a single object-dtype ``corners`` column of (4, 2) arrays
to eight flat float32 columns, and pickles are now zstd-compressed.  Readers
must keep handling both, since recordings processed by earlier versions are
not rewritten.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from yoto.constants import COL_CORNERS, CORNER_COLS, PICKLE_EXT
from yoto.io import (
    corner_row,
    corner_tag_ids,
    find_pickle,
    has_corners,
    is_pickle,
    load_corners,
    strip_pickle_ext,
)

QUAD = np.array([[1.0, 2.0], [3.0, 2.0], [3.0, 4.0], [1.0, 4.0]])


def _long_new(quads):
    """Raw-format frame using the flat float32 columns."""
    return pd.DataFrame(
        [{"tag_id": i, **corner_row(q)} for i, q in enumerate(quads)],
        index=pd.RangeIndex(len(quads), name="frame"),
    )


def _long_old(quads):
    """Raw-format frame using the legacy object corners column."""
    return pd.DataFrame(
        [{"tag_id": i, COL_CORNERS: q} for i, q in enumerate(quads)],
        index=pd.RangeIndex(len(quads), name="frame"),
    )


def _wide(quads, metric_cols):
    """Clean-format frame with a (tag_id, metric) column MultiIndex."""
    data = {}
    for tag, q in zip(["7", "9"], quads):
        if metric_cols is COL_CORNERS:
            data[("7" if tag == "7" else "9", COL_CORNERS)] = [q, q]
        else:
            for name, v in corner_row(q).items():
                data[(tag, name)] = [v, v]
    df = pd.DataFrame(data, index=pd.RangeIndex(2, name="frame"))
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


# --------------------------------------------------------------------------
# corner_row
# --------------------------------------------------------------------------


def test_corner_row_flattens_in_column_order():
    row = corner_row(QUAD)

    assert list(row) == list(CORNER_COLS)
    assert [row[c] for c in CORNER_COLS] == [1, 2, 3, 2, 3, 4, 1, 4]


def test_corner_row_accepts_a_flat_sequence():
    assert corner_row([1, 2, 3, 2, 3, 4, 1, 4]) == corner_row(QUAD)


# --------------------------------------------------------------------------
# load_corners — both formats, both shapes
# --------------------------------------------------------------------------


def test_load_corners_reads_flat_columns():
    out = load_corners(_long_new([QUAD, QUAD * 2]))

    assert out.shape == (2, 4, 2)
    np.testing.assert_allclose(out[0], QUAD)
    np.testing.assert_allclose(out[1], QUAD * 2)


def test_load_corners_reads_the_legacy_object_column():
    out = load_corners(_long_old([QUAD, QUAD * 2]))

    assert out.shape == (2, 4, 2)
    np.testing.assert_allclose(out[0], QUAD)


def test_both_formats_yield_identical_arrays():
    """The whole point of the compatibility seam."""
    quads = [QUAD, QUAD * 2, QUAD + 7]

    np.testing.assert_array_equal(
        load_corners(_long_new(quads)), load_corners(_long_old(quads))
    )


def test_load_corners_maps_missing_detections_to_nan():
    df = _long_old([QUAD, QUAD])
    df.loc[1, COL_CORNERS] = None

    out = load_corners(df)

    np.testing.assert_allclose(out[0], QUAD)
    assert np.isnan(out[1]).all()


def test_load_corners_selects_one_tag_from_a_wide_frame():
    out = load_corners(_wide([QUAD, QUAD * 2], CORNER_COLS), "9")

    assert out.shape == (2, 4, 2)
    np.testing.assert_allclose(out[0], QUAD * 2)


def test_load_corners_stacks_every_tag_when_no_tag_given():
    out = load_corners(_wide([QUAD, QUAD * 2], CORNER_COLS))

    assert out.shape == (2, 2, 4, 2)  # (frames, tags, 4, 2)
    np.testing.assert_allclose(out[0, 0], QUAD)
    np.testing.assert_allclose(out[0, 1], QUAD * 2)


def test_load_corners_raises_when_there_are_no_corners():
    df = pd.DataFrame({"center_x": [1.0]}, index=pd.RangeIndex(1, name="frame"))

    with pytest.raises(KeyError):
        load_corners(df)


# --------------------------------------------------------------------------
# has_corners / corner_tag_ids
# --------------------------------------------------------------------------


@pytest.mark.parametrize("build", [_long_new, _long_old])
def test_has_corners_accepts_both_formats(build):
    assert has_corners(build([QUAD]))


def test_has_corners_false_without_them():
    assert not has_corners(pd.DataFrame({"center_x": [1.0]}))


def test_corner_tag_ids_preserves_column_order():
    assert corner_tag_ids(_wide([QUAD, QUAD], CORNER_COLS)) == ["7", "9"]


def test_corner_tag_ids_is_empty_for_a_long_frame():
    assert corner_tag_ids(_long_new([QUAD])) == []


# --------------------------------------------------------------------------
# pickle extension helpers
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        ("a/b_clean.pkl.zst", "a/b_clean"),
        ("a/b_clean.pkl", "a/b_clean"),
        ("a/b.mp4", "a/b.mp4"),
    ],
)
def test_strip_pickle_ext(name, expected):
    assert strip_pickle_ext(name) == expected


def test_strip_pickle_ext_removes_the_whole_compound_extension():
    """os.path.splitext would leave '.pkl' behind and mangle output names."""
    assert not strip_pickle_ext(f"x{PICKLE_EXT}").endswith(".pkl")


@pytest.mark.parametrize(
    "name,expected",
    [("x.pkl.zst", True), ("x.pkl", True), ("x.csv", False), ("x.mp4", False)],
)
def test_is_pickle(name, expected):
    assert is_pickle(name) is expected


def test_find_pickle_prefers_the_compressed_form(tmp_path):
    (tmp_path / "a.pkl").touch()
    (tmp_path / "a.pkl.zst").touch()

    assert find_pickle(str(tmp_path / "a")) == str(tmp_path / "a.pkl.zst")


def test_find_pickle_falls_back_to_plain_pkl(tmp_path):
    """Recordings cleaned before compression must keep resolving."""
    (tmp_path / "a.pkl").touch()

    assert find_pickle(str(tmp_path / "a")) == str(tmp_path / "a.pkl")


def test_find_pickle_returns_none_when_absent(tmp_path):
    assert find_pickle(str(tmp_path / "nope")) is None


# --------------------------------------------------------------------------
# round-trip through a compressed pickle
# --------------------------------------------------------------------------


def test_corners_survive_a_compressed_round_trip(tmp_path):
    df = _long_new([QUAD, QUAD * 3])
    path = tmp_path / f"x{PICKLE_EXT}"

    df.astype({c: "float32" for c in CORNER_COLS}).to_pickle(path)
    back = pd.read_pickle(path)  # codec inferred from the extension

    np.testing.assert_allclose(load_corners(back), load_corners(df))
