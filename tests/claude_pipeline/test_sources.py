"""Integrity tests for source discovery: leakage, split exports, and date keying."""

from __future__ import annotations

import io
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

from claude_pipeline import loaders
from claude_pipeline.sources import UW_ROOT, build_index

REAL_ROOT_AVAILABLE = UW_ROOT.exists()


def _touch_zip(day_dir: Path, name: str, frame: pd.DataFrame | None = None) -> Path:
    day_dir.mkdir(parents=True, exist_ok=True)
    path = day_dir / name
    if frame is None:
        path.touch()
        return path
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name.replace(".zip", ".csv"), buffer.getvalue())
    return path


def test_forward_dated_oi_exports_are_rejected(tmp_path):
    day = tmp_path / "2026-05-01"
    _touch_zip(day, "chain-oi-changes-2026-05-01.zip")
    _touch_zip(day, "chain-oi-changes-latest-2026-05-04.zip")
    _touch_zip(day, "chain-oi-changes-current-2026-05-04.zip")

    index = build_index(tmp_path)

    assert index.sessions() == ["2026-05-01"]
    assert not index.get("2026-05-04", "chain-oi-changes")
    reasons = {r.reason for r in index.rejections}
    assert reasons == {"forward_dated_latest_export", "forward_dated_current_export"}


def test_overlay_directories_are_ignored(tmp_path):
    _touch_zip(tmp_path / "2026-05-19", "hot-chains-2026-05-19.zip")
    _touch_zip(tmp_path / "2026-05-19-v3-overlay-2026-05-20-live", "hot-chains-2026-05-20.zip")
    _touch_zip(tmp_path / "2026-05-22_overlay_2026-05-26_input", "hot-chains-2026-05-26.zip")

    index = build_index(tmp_path)

    assert index.sessions() == ["2026-05-19"]


def test_files_are_keyed_by_filename_date_not_folder(tmp_path):
    day = tmp_path / "2026-07-15"
    _touch_zip(day, "stock-screener-2026-07-15.zip")
    _touch_zip(day, "stock-screener-2026-07-14.zip")

    index = build_index(tmp_path)

    assert index.sessions() == ["2026-07-14", "2026-07-15"]
    assert index.get("2026-07-14", "stock-screener")[0].path.name == "stock-screener-2026-07-14.zip"


def test_incomplete_split_export_is_rejected(tmp_path):
    day = tmp_path / "2026-04-23"
    for part in (1, 2, 3):
        _touch_zip(day, f"bot-eod-report-2026-04-23.part-0{part}-of-05.zip")

    index = build_index(tmp_path)

    assert not index.get("2026-04-23", "bot-eod-report")
    assert any(r.reason == "incomplete_split_export_3_of_5" for r in index.rejections)


def test_complete_split_export_is_kept_in_order(tmp_path):
    day = tmp_path / "2026-04-23"
    for part in range(1, 6):
        _touch_zip(day, f"bot-eod-report-2026-04-23.part-0{part}-of-05.zip")

    found = build_index(tmp_path).get("2026-04-23", "bot-eod-report")

    assert [f.part for f in found] == [1, 2, 3, 4, 5]


def test_redownload_copy_loses_to_the_canonical_export(tmp_path):
    day = tmp_path / "2026-07-15"
    _touch_zip(day, "hot-chains-2026-07-15.zip")
    _touch_zip(day, "hot-chains-2026-07-15 (2).zip")

    index = build_index(tmp_path)

    found = index.get("2026-07-15", "hot-chains")
    assert len(found) == 1
    assert found[0].path.name == "hot-chains-2026-07-15.zip"
    assert any(r.reason == "duplicate_export_for_session" for r in index.rejections)


def test_redownload_copy_is_used_when_it_is_the_only_export(tmp_path):
    _touch_zip(tmp_path / "2025-12-19", "hot-chains-2025-12-19 (2).zip")

    found = build_index(tmp_path).get("2025-12-19", "hot-chains")

    assert len(found) == 1 and found[0].is_copy


def test_export_filed_under_another_day_loses_to_its_own_folder(tmp_path):
    _touch_zip(tmp_path / "2026-07-14", "stock-screener-2026-07-14.zip")
    _touch_zip(tmp_path / "2026-07-15", "stock-screener-2026-07-14.zip")

    found = build_index(tmp_path).get("2026-07-14", "stock-screener")

    assert len(found) == 1
    assert found[0].path.parent.name == "2026-07-14"


def test_unknown_archives_are_rejected_not_guessed(tmp_path):
    day = tmp_path / "2026-08-06"
    _touch_zip(day, "all.zip")
    _touch_zip(day, "option-trades-2026-08-06.zip")

    index = build_index(tmp_path)

    assert index.sessions() == []
    assert all(r.reason == "not_a_known_family" for r in index.rejections)


def test_read_concatenates_split_parts(tmp_path):
    day = tmp_path / "2026-04-23"
    for part in (1, 2):
        _touch_zip(
            day,
            f"bot-eod-report-2026-04-23.part-0{part}-of-02.zip",
            pd.DataFrame({"underlying_symbol": [f"T{part}"], "premium": [part * 100]}),
        )

    index = build_index(tmp_path)
    frame = loaders.read(index, "2026-04-23", "bot-eod-report")

    assert list(frame.underlying_symbol) == ["T1", "T2"]
    assert loaders.count_rows(index, "2026-04-23", "bot-eod-report") == 2


def test_missing_source_raises_rather_than_returning_empty(tmp_path):
    index = build_index(tmp_path)
    with pytest.raises(loaders.MissingSourceError):
        loaders.read(index, "2026-08-06", "hot-chains")


@pytest.mark.skipif(not REAL_ROOT_AVAILABLE, reason="UW data root not present")
class TestRealTree:
    @pytest.fixture(scope="class")
    def index(self):
        return build_index()

    def test_no_forward_dated_export_survives(self, index):
        forward = [r for r in index.rejections if r.reason.startswith("forward_dated_")]
        by_reason = Counter(r.reason for r in forward)
        assert by_reason == {"forward_dated_latest_export": 20, "forward_dated_current_export": 9}
        for source in index.files.values():
            for file in source:
                assert "latest" not in file.path.name and "current" not in file.path.name

    def test_july_15_folder_contributes_two_sessions(self, index):
        for session in ("2026-07-14", "2026-07-15"):
            found = index.get(session, "stock-screener")
            assert len(found) == 1
            assert session in found[0].path.name

    def test_sessions_whose_only_export_is_a_copy_are_recovered(self, index):
        # Both exist on disk solely as "... (N).zip"; dropping them lost real sessions.
        assert index.get("2025-12-19", "hot-chains")
        assert index.get("2026-05-06", "chain-oi-changes")

    def test_every_indexed_file_matches_its_session(self, index):
        for (session, family), found in index.files.items():
            for file in found:
                assert file.session == session
                assert session in file.path.name
                assert file.family == family
