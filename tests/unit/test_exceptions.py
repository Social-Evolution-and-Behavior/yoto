"""Unit tests for yoto.exceptions."""

from __future__ import annotations

import pytest

from yoto.exceptions import (
    EncoderError,
    ModelLoadError,
    VideoReadError,
    YotoError,
)


class TestExceptionHierarchy:
    """All custom exceptions should inherit from YotoError."""

    @pytest.mark.parametrize(
        "exc_class",
        [VideoReadError, ModelLoadError, EncoderError],
    )
    def test_inherits_from_yoto_error(
        self, exc_class: type[YotoError]
    ) -> None:
        assert issubclass(exc_class, YotoError)


class TestVideoReadError:
    def test_message_contains_path(self) -> None:
        exc = VideoReadError("/tmp/missing.mp4")
        assert "/tmp/missing.mp4" in str(exc)
        assert exc.path == "/tmp/missing.mp4"


class TestModelLoadError:
    def test_message_contains_weights_path(self) -> None:
        exc = ModelLoadError("best.pt", reason="file not found")
        assert "best.pt" in str(exc)
        assert "file not found" in str(exc)

    def test_no_reason(self) -> None:
        exc = ModelLoadError("best.pt")
        assert "best.pt" in str(exc)


class TestEncoderError:
    def test_default_message(self) -> None:
        exc = EncoderError()
        assert "ffmpeg" in str(exc).lower()
