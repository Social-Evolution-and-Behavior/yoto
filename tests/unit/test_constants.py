"""Unit tests for yoto.constants — verifies invariants and types."""

from __future__ import annotations

import yoto.constants as C


class TestConstants:
    """Smoke tests ensuring constants have sensible types and values."""

    def test_conf_threshold_range(self) -> None:
        assert 0.0 < C.DEFAULT_CONF_THRESHOLD <= 1.0

    def test_pad_pixels_positive(self) -> None:
        assert C.DEFAULT_PAD_PIXELS > 0

    def test_max_valid_tag_id_positive(self) -> None:
        assert C.MAX_VALID_TAG_ID > 0

    def test_interpolation_limit_positive(self) -> None:
        assert C.DEFAULT_INTERPOLATION_LIMIT > 0

    def test_max_jump_distance_positive(self) -> None:
        assert C.DEFAULT_MAX_JUMP_DISTANCE > 0

    def test_unsharp_kernel_is_odd(self) -> None:
        kh, kw = C.DEFAULT_UNSHARP_KERNEL
        assert kh % 2 == 1
        assert kw % 2 == 1

    def test_ass_type_codes_distinct(self) -> None:
        codes = {C.ASS_TYPE_NONE, C.ASS_TYPE_ORIGINAL, C.ASS_TYPE_INTERPOLATED}
        assert len(codes) == 3

    def test_trail_length_greater_than_skip(self) -> None:
        assert C.DEFAULT_TRAIL_LENGTH > C.DEFAULT_TRAIL_SKIP
