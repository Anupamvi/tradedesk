from __future__ import annotations

import datetime as dt
import json

from codexuw.daily import live_planning_validation_note, parse_args, write_data_error_report
import codexuw.daily as daily
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
    manifest = json.loads((out_dir / "codexuw_manifest_2026-05-01.json").read_text(encoding="utf-8"))
    assert "# Codex Daily V2 - Daily Decision Engine - 2026-05-01" in text
    assert "| Pipeline | Codex Daily V2 |" in text
    assert manifest["pipeline_name"] == "Codex Daily V2"
    assert manifest["pipeline_version"] == "v2"
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
    assert args.portfolio_income_mode == "trading-sleeve-only"
    assert args.index_income_mode == "fallback"
    assert args.risk_mandate == "capital-preservation"


def test_stale_daily_folder_warning_says_live_schwab_is_current() -> None:
    note = live_planning_validation_note(dt.date(2026, 4, 27), dt.date(2026, 5, 5))

    assert "live-planning run using current Schwab chains against historical UW flow from 2026-04-27" in note
    assert "Use codexuw.replay / historical mode" in note


def test_historical_replay_flag_delegates_to_replay_path(monkeypatch, tmp_path) -> None:
    base_dir = tmp_path / "2026-04-27"
    out_dir = tmp_path / "out"
    base_dir.mkdir()
    calls = {}

    def fake_run_replay(root, out, start, end, max_days, **kwargs):
        calls["root"] = root
        calls["out"] = out
        calls["start"] = start
        calls["end"] = end
        calls["max_days"] = max_days
        calls["kwargs"] = kwargs
        out.mkdir()
        report = out / "codexuw_replay_report.md"
        report.write_text("report", encoding="utf-8")
        return report

    monkeypatch.setattr("codexuw.replay.run_replay", fake_run_replay)
    monkeypatch.setattr(
        "sys.argv",
        [
            "codexuw.daily",
            "--base-dir",
            str(base_dir),
            "--out-dir",
            str(out_dir),
            "--historical-replay",
            "--replay-end",
            "2026-05-01",
        ],
    )

    daily.main()

    assert calls["root"] == tmp_path
    assert calls["start"] == dt.date(2026, 4, 27)
    assert calls["end"] == dt.date(2026, 5, 1)
    assert calls["max_days"] == 0
    assert calls["kwargs"]["entry_start"] == dt.date(2026, 4, 27)
