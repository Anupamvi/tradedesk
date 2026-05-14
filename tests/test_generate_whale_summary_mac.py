import os
from pathlib import Path

import pytest

import uwos.generate_whale_summary_mac as mac
from uwos.generate_whale_summary_mac import default_output_path, find_download_report


def test_find_download_report_requires_full_bot_eod_file(tmp_path, monkeypatch):
    downloads = tmp_path / "Downloads"
    downloads.mkdir()
    for idx in range(1, 4):
        (
            downloads / f"bot-eod-report-2026-04-23.part-{idx:02d}-of-03.zip"
        ).write_bytes(b"zip")

    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    with pytest.raises(FileNotFoundError, match="Split part files are not accepted"):
        find_download_report("2026-04-23")


def test_find_download_report_without_date_ignores_newer_split_parts(tmp_path, monkeypatch):
    downloads = tmp_path / "Downloads"
    downloads.mkdir()
    full_report = downloads / "bot-eod-report-2026-04-22.zip"
    full_report.write_bytes(b"zip")
    os.utime(full_report, (100, 100))

    for idx, mtime in [(1, 50), (2, 200), (3, 200)]:
        path = downloads / f"bot-eod-report-2026-04-23.part-{idx:02d}-of-03.zip"
        path.write_bytes(b"zip")
        os.utime(path, (mtime, mtime))

    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    assert find_download_report() == full_report


def test_find_download_report_falls_back_to_repo_dated_folder(tmp_path, monkeypatch):
    downloads = tmp_path / "Downloads"
    downloads.mkdir()
    repo_root = tmp_path / "tradedesk"
    dated = repo_root / "2026-05-06"
    dated.mkdir(parents=True)
    report = dated / "bot-eod-report-2026-05-06.zip"
    report.write_bytes(b"zip")
    (dated / "chain-oi-changes-2026-05-06.zip").write_bytes(b"zip")
    (dated / "dp-eod-report-2026-05-06.zip").write_bytes(b"zip")

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(mac, "REPO_ROOT", repo_root)

    assert find_download_report("2026-05-06") == report


def test_default_output_path_uses_repo_dated_folder_from_input_date(tmp_path, monkeypatch):
    repo_root = tmp_path / "tradedesk"
    monkeypatch.setattr(mac, "REPO_ROOT", repo_root)

    input_path = tmp_path / "Downloads" / "bot-eod-report-2026-05-07.zip"

    assert default_output_path(input_path) == repo_root / "2026-05-07" / "whale-2026-05-07.md"


def test_default_output_path_positional_date_wins(tmp_path, monkeypatch):
    repo_root = tmp_path / "tradedesk"
    monkeypatch.setattr(mac, "REPO_ROOT", repo_root)

    input_path = tmp_path / "Downloads" / "bot-eod-report-2026-05-06.zip"

    assert default_output_path(input_path, "2026-05-07") == repo_root / "2026-05-07" / "whale-2026-05-07.md"
