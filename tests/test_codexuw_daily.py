from __future__ import annotations

import datetime as dt

from codexuw.daily import parse_args, write_data_error_report
from codexuw.data import find_export


def test_write_data_error_report_documents_no_trade_reason(tmp_path) -> None:
    base_dir = tmp_path / "2026-05-01"
    out_dir = tmp_path / "out"
    base_dir.mkdir()

    report = write_data_error_report(
        out_dir,
        dt.date(2026, 5, 1),
        base_dir,
        FileNotFoundError("No stock-screener export found"),
    )

    text = report.read_text(encoding="utf-8")
    assert "No high-quality trades today" in text
    assert "Issue type: data problem" in text
    assert "No stock-screener export found" in text
    assert (out_dir / "codexuw_manifest_2026-05-01.json").exists()


def test_find_export_prefers_latest_or_current_snapshot(tmp_path) -> None:
    base_dir = tmp_path / "2026-05-01"
    base_dir.mkdir()
    old = base_dir / "chain-oi-changes-2026-05-01.csv"
    latest = base_dir / "chain-oi-changes-latest-2026-05-02.csv"
    old.write_text("old", encoding="utf-8")
    latest.write_text("latest", encoding="utf-8")

    assert find_export(base_dir, "chain-oi-changes-") == latest


def test_daily_defaults_to_eight_final_trades(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "codexuw.daily",
            "--base-dir",
            str(tmp_path / "2026-05-01"),
            "--out-dir",
            str(tmp_path / "out"),
        ],
    )

    args = parse_args()

    assert args.max_final_trades == 8
