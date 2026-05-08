"""Unit tests for yoto.cli."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from yoto.cli import (
    TRACKING_DIR,
    TRACKING_SUBDIRS,
    _build_worker_cmd,
    _find_pickle_for_video,
    _format_duration,
    _recording_dir_for_pickle,
    _recording_dir_for_video,
    _resolve_pickle_paths,
    _resolve_video_paths,
    _tracking_layout,
    main,
)


class TestCLI:
    """Tests for the CLI entry point."""

    def test_no_args_prints_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.argv", ["yoto"]):
                main()
        assert exc_info.value.code == 1

    def test_version_flag(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.argv", ["yoto", "--version"]):
                main()
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        from yoto import __version__

        assert __version__ in captured.out

    def test_detect_subcommand_requires_video(self) -> None:
        with pytest.raises(SystemExit):
            with patch("sys.argv", ["yoto", "detect"]):
                main()

    def test_clean_subcommand_requires_input(self) -> None:
        with pytest.raises(SystemExit):
            with patch("sys.argv", ["yoto", "clean"]):
                main()

    def test_render_subcommand_requires_video(self) -> None:
        with pytest.raises(SystemExit):
            with patch("sys.argv", ["yoto", "render"]):
                main()


class TestResolveVideoPaths:
    """Tests for _resolve_video_paths."""

    def test_single_file(self, tmp_path: object) -> None:
        p = tmp_path / "video.mp4"  # type: ignore[operator]
        p.touch()
        assert _resolve_video_paths(str(p)) == [str(p)]

    def test_directory_with_videos(self, tmp_path: object) -> None:
        d = tmp_path  # type: ignore[assignment]
        (d / "000000.mp4").touch()
        (d / "000001.mp4").touch()
        (d / "data.pkl").touch()  # not a video
        result = _resolve_video_paths(str(d))
        assert len(result) == 2
        assert all(r.endswith(".mp4") for r in result)

    def test_directory_with_subfolders(self, tmp_path: object) -> None:
        d = tmp_path  # type: ignore[assignment]
        sub1 = d / "rec1"
        sub1.mkdir()
        (sub1 / "000000.mp4").touch()
        sub2 = d / "rec2"
        sub2.mkdir()
        (sub2 / "000000.mp4").touch()
        (sub2 / "000001.avi").touch()
        result = _resolve_video_paths(str(d))
        assert len(result) == 3

    def test_nonexistent_path_passes_through(self) -> None:
        result = _resolve_video_paths("/nonexistent/video.mp4")
        assert result == ["/nonexistent/video.mp4"]

    def test_empty_directory(self, tmp_path: object) -> None:
        result = _resolve_video_paths(str(tmp_path))
        assert result == []


class TestBuildWorkerCmd:
    """Tests for _build_worker_cmd (the per-video command that GNU parallel runs)."""

    def _make_args(self, **overrides: object) -> object:
        import argparse

        defaults = {
            "output_dir": None,
            "yoloweights": "detect14.engine",
            "dataname": "_apriltagDetect14",
            "fast": False,
            "debug": False,
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_minimal(self) -> None:
        args = self._make_args()
        cmd = _build_worker_cmd("/vids/a.mp4", args)  # type: ignore[arg-type]
        assert "detect" in cmd
        assert "/vids/a.mp4" in cmd
        assert "--yoloweights" in cmd
        assert "--fast" not in cmd
        assert "--debug" not in cmd

    def test_fast_and_debug_forwarded(self) -> None:
        args = self._make_args(fast=True, debug=True)
        cmd = _build_worker_cmd("/vids/a.mp4", args)  # type: ignore[arg-type]
        assert "--fast" in cmd
        assert "--debug" in cmd

    def test_output_dir_forwarded(self) -> None:
        args = self._make_args(output_dir="/out")
        cmd = _build_worker_cmd("/vids/a.mp4", args)  # type: ignore[arg-type]
        # positional video then positional output_dir
        i_video = cmd.index("/vids/a.mp4")
        assert cmd[i_video + 1] == "/out"


class TestResolvePicklePaths:
    """Tests for _resolve_pickle_paths."""

    def test_single_file(self, tmp_path: object) -> None:
        p = tmp_path / "raw.pkl"  # type: ignore[operator]
        p.touch()
        assert _resolve_pickle_paths(str(p)) == [str(p)]

    def test_directory_filters_clean_suffix(self, tmp_path: object) -> None:
        d = tmp_path  # type: ignore[assignment]
        (d / "000000_apriltagDetect14.pkl").touch()
        (d / "000001_apriltagDetect14.pkl").touch()
        (d / "000000_apriltagDetect14_clean.pkl").touch()  # should be skipped
        (d / "notes.txt").touch()  # not a pickle
        result = _resolve_pickle_paths(str(d))
        assert len(result) == 2
        assert all(not r.endswith("_clean.pkl") for r in result)

    def test_directory_with_subfolders(self, tmp_path: object) -> None:
        d = tmp_path  # type: ignore[assignment]
        sub1 = d / "rec1"
        sub1.mkdir()
        (sub1 / "raw1.pkl").touch()
        sub2 = d / "rec2"
        sub2.mkdir()
        (sub2 / "raw2.pkl").touch()
        (sub2 / "raw2_clean.pkl").touch()  # skipped
        result = _resolve_pickle_paths(str(d))
        assert len(result) == 2

    def test_empty_directory(self, tmp_path: object) -> None:
        assert _resolve_pickle_paths(str(tmp_path)) == []

    def test_prefers_tracking_raw_data(self, tmp_path: object) -> None:
        rec = tmp_path  # type: ignore[assignment]
        # Also place a stray pkl at the recording root — should be ignored
        # because tracking/raw_data/ exists.
        (rec / "stray.pkl").touch()
        raw_dir = rec / TRACKING_DIR / "raw_data"
        raw_dir.mkdir(parents=True)
        (raw_dir / "000000_apriltagDetect14.pkl").touch()
        (raw_dir / "000001_apriltagDetect14.pkl").touch()
        result = _resolve_pickle_paths(str(rec))
        assert len(result) == 2
        assert all(TRACKING_DIR in r for r in result)


class TestTrackingLayout:
    """Tests for _tracking_layout."""

    def test_returns_four_subdirs(self, tmp_path: object) -> None:
        layout = _tracking_layout(str(tmp_path))  # type: ignore[arg-type]
        assert set(layout.keys()) == set(TRACKING_SUBDIRS.keys())
        for name, path in layout.items():
            assert path.endswith(os.path.join(TRACKING_DIR, name))

    def test_paths_not_created(self, tmp_path: object) -> None:
        layout = _tracking_layout(str(tmp_path))  # type: ignore[arg-type]
        for p in layout.values():
            assert not os.path.exists(p)


class TestRecordingDirForVideo:
    """Tests for _recording_dir_for_video."""

    def test_returns_parent(self, tmp_path: object) -> None:
        v = tmp_path / "000000.mp4"  # type: ignore[operator]
        v.touch()
        assert _recording_dir_for_video(str(v)) == os.path.abspath(str(tmp_path))


class TestRecordingDirForPickle:
    """Tests for _recording_dir_for_pickle."""

    def test_pickle_inside_tracking_raw(self, tmp_path: object) -> None:
        rec = tmp_path  # type: ignore[assignment]
        raw = rec / TRACKING_DIR / "raw_data"
        raw.mkdir(parents=True)
        pkl = raw / "000000_apriltagDetect14.pkl"
        pkl.touch()
        assert _recording_dir_for_pickle(str(pkl)) == os.path.abspath(str(rec))

    def test_pickle_inside_tracking_clean(self, tmp_path: object) -> None:
        rec = tmp_path  # type: ignore[assignment]
        clean = rec / TRACKING_DIR / "clean_data"
        clean.mkdir(parents=True)
        pkl = clean / "000000_apriltagDetect14_clean.pkl"
        pkl.touch()
        assert _recording_dir_for_pickle(str(pkl)) == os.path.abspath(str(rec))

    def test_pickle_next_to_video(self, tmp_path: object) -> None:
        pkl = tmp_path / "000000_apriltagDetect14.pkl"  # type: ignore[operator]
        pkl.touch()
        assert _recording_dir_for_pickle(str(pkl)) == os.path.abspath(
            str(tmp_path)
        )  # type: ignore[arg-type]


class TestFindPickleForVideo:
    """Tests for _find_pickle_for_video."""

    def test_prefers_clean_over_raw(self, tmp_path: object) -> None:
        rec = tmp_path  # type: ignore[assignment]
        video = rec / "000000.mp4"
        video.touch()
        raw_dir = rec / TRACKING_DIR / "raw_data"
        raw_dir.mkdir(parents=True)
        (raw_dir / "000000_apriltagDetect14.pkl").touch()
        clean_dir = rec / TRACKING_DIR / "clean_data"
        clean_dir.mkdir(parents=True)
        clean_pkl = clean_dir / "000000_apriltagDetect14_clean.pkl"
        clean_pkl.touch()
        assert _find_pickle_for_video(str(video), "_apriltagDetect14") == str(clean_pkl)

    def test_falls_back_to_raw(self, tmp_path: object) -> None:
        rec = tmp_path  # type: ignore[assignment]
        video = rec / "000000.mp4"
        video.touch()
        raw_dir = rec / TRACKING_DIR / "raw_data"
        raw_dir.mkdir(parents=True)
        raw_pkl = raw_dir / "000000_apriltagDetect14.pkl"
        raw_pkl.touch()
        assert _find_pickle_for_video(str(video), "_apriltagDetect14") == str(raw_pkl)

    def test_falls_back_to_legacy(self, tmp_path: object) -> None:
        rec = tmp_path  # type: ignore[assignment]
        video = rec / "000000.mp4"
        video.touch()
        legacy = rec / "000000_apriltagDetect14.pkl"
        legacy.touch()
        assert _find_pickle_for_video(str(video), "_apriltagDetect14") == str(legacy)

    def test_returns_none_if_nothing_found(self, tmp_path: object) -> None:
        video = tmp_path / "000000.mp4"  # type: ignore[operator]
        video.touch()
        assert _find_pickle_for_video(str(video), "_apriltagDetect14") is None


class TestFormatDuration:
    """Tests for _format_duration."""

    def test_sub_minute(self) -> None:
        assert _format_duration(3.5) == "3.5s"
        assert _format_duration(59.9) == "59.9s"

    def test_minutes(self) -> None:
        assert _format_duration(60.0) == "1m 00s"
        assert _format_duration(125.0) == "2m 05s"

    def test_hours(self) -> None:
        assert _format_duration(3600.0) == "1h 00m 00s"
        assert _format_duration(3665.0) == "1h 01m 05s"
