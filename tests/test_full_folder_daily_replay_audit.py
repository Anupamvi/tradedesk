import datetime as dt

import pandas as pd

from uwos.full_folder_daily_replay_audit import (
    ACTION_BOOKS,
    APPROVED_BOOKS,
    inventory,
    metric_block,
    parse_approved_counts,
)


def _touch_family(day_dir, name):
    (day_dir / name).write_text("x", encoding="utf-8")


def test_inventory_requires_dp_and_whale_source_as_separate_daily_families(tmp_path):
    complete = tmp_path / "2026-04-23"
    bot_only_for_dp = tmp_path / "2026-04-24"
    missing_whale_source = tmp_path / "2026-04-25"
    for day_dir in (complete, bot_only_for_dp, missing_whale_source):
        day_dir.mkdir()
        _touch_family(day_dir, f"stock-screener-{day_dir.name}.csv")
        _touch_family(day_dir, f"hot-chains-{day_dir.name}.csv")
        _touch_family(day_dir, f"chain-oi-changes-{day_dir.name}.csv")

    _touch_family(complete, "dp-eod-report-2026-04-23.csv")
    _touch_family(complete, "bot-eod-report-2026-04-23.zip")
    _touch_family(bot_only_for_dp, "bot-eod-report-2026-04-24.zip")
    _touch_family(missing_whale_source, "dp-eod-report-2026-04-25.csv")

    folders, incomplete = inventory(tmp_path, dt.date(2026, 4, 23), dt.date(2026, 4, 25))

    assert [folder.name for folder in folders] == ["2026-04-23"]
    missing_by_date = {row["date"]: row["missing"] for row in incomplete}
    assert missing_by_date["2026-04-24"] == ["dp"]
    assert missing_by_date["2026-04-25"] == ["whale_source"]


def test_replay_audit_treats_scout_as_research_only_by_default():
    assert APPROVED_BOOKS == {"Core", "Tactical"}
    assert {"Medium", "Income", "Pilot", "Scout"}.issubset(ACTION_BOOKS)


def test_parse_approved_counts_accepts_live_and_historical_summary_text():
    assert parse_approved_counts("Approved trades: 2 / 20") == (2, 20)
    assert parse_approved_counts("Historical gate-pass candidates (NOT live approvals): 3 / 20") == (3, 20)
    assert parse_approved_counts("SKIP: No approved trades") == (None, None)


def test_metric_block_handles_open_trades_without_realized_pnl():
    metrics = metric_block(pd.DataFrame([{"status": "open_not_expired"}]))

    assert metrics["rows"] == 1
    assert metrics["completed"] == 0
    assert metrics["open"] == 1
    assert metrics["pnl_available"] is False
