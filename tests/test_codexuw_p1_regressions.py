from __future__ import annotations

import datetime as dt
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from codexuw.daily_v4 import _compare_v4_overlay_changes
from codexuw.data import aggregate_bot_flow, find_export_bundle
from codexuw.performance import load_recent_performance


def _write_bot_part(path: Path, premium: float) -> None:
    row = pd.DataFrame(
        [
            {
                "underlying_symbol": "TEST",
                "side": "ask",
                "option_type": "call",
                "expiry": "2026-08-21",
                "strike": 100,
                "premium": premium,
                "size": 1,
                "volume": 10,
                "open_interest": 20,
                "delta": 0.5,
                "canceled": "f",
                "report_flags": "",
                "upstream_condition_detail": "",
            }
        ]
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("bot.csv", row.to_csv(index=False))


def test_split_bot_bundle_aggregates_every_part(tmp_path: Path) -> None:
    for part, premium in enumerate((100.0, 200.0, 300.0), start=1):
        _write_bot_part(
            tmp_path / f"bot-eod-report-2026-04-27.part-{part:02d}-of-03.zip",
            premium,
        )

    bundle = find_export_bundle(tmp_path, "bot-eod-report-")
    flow = aggregate_bot_flow(tmp_path, ["TEST"], chunksize=1)

    assert [path.name for path in bundle] == [
        "bot-eod-report-2026-04-27.part-01-of-03.zip",
        "bot-eod-report-2026-04-27.part-02-of-03.zip",
        "bot-eod-report-2026-04-27.part-03-of-03.zip",
    ]
    assert flow.loc[0, "bot_total_premium"] == pytest.approx(600.0)
    assert flow.attrs["source_status"] == "bot_eod_split_bundle_loaded"


def test_split_bot_bundle_rejects_missing_part(tmp_path: Path) -> None:
    _write_bot_part(tmp_path / "bot-eod-report-2026-04-27.part-01-of-03.zip", 100.0)
    _write_bot_part(tmp_path / "bot-eod-report-2026-04-27.part-03-of-03.zip", 300.0)

    with pytest.raises(ValueError, match="missing part"):
        find_export_bundle(tmp_path, "bot-eod-report-")


def test_bot_flow_point_in_time_excludes_future_latest_export(tmp_path: Path) -> None:
    day_dir = tmp_path / "2026-05-01"
    day_dir.mkdir()
    _write_bot_part(day_dir / "bot-eod-report-2026-05-01.zip", 100.0)
    _write_bot_part(day_dir / "bot-eod-report-latest-2026-05-04.zip", 900.0)

    flow = aggregate_bot_flow(day_dir, ["TEST"], point_in_time=True)

    assert flow.loc[0, "bot_total_premium"] == pytest.approx(100.0)


def test_recent_performance_uses_requested_namespace_and_cutoff(tmp_path: Path) -> None:
    namespace = "accepted_v4_history"
    history_dir = tmp_path / f"{namespace}_run"
    history_dir.mkdir()
    pd.DataFrame(
        [
            {
                "asof": "2026-07-01",
                "exit_day": "2026-07-02",
                "ticker": "A",
                "exact_evaluated": True,
                "decision_pass": True,
                "exact_win": True,
                "pnl_1x": 100.0,
            },
            {
                "asof": "2026-07-09",
                "exit_day": "2026-07-10",
                "ticker": "B",
                "exact_evaluated": True,
                "decision_pass": True,
                "exact_win": False,
                "pnl_1x": -500.0,
            },
        ]
    ).to_csv(history_dir / "codexuw_replay_detail.csv", index=False)

    summary = load_recent_performance(
        tmp_path,
        asof=dt.date(2026, 7, 9),
        history_namespace=namespace,
    )

    assert summary["status"] == "ok"
    assert summary["window"] == 1
    assert summary["total_pnl_1x"] == pytest.approx(100.0)
    assert summary["latest_asof"] == "2026-07-01"
    assert summary["history_namespace"] == namespace


def test_overlay_change_marks_refreshed_unchanged_price() -> None:
    before = pd.DataFrame(
        [{"ticker": "TEST", "direction": "Bull Call", "expiry": "2026-08-21", "short_strike": 105, "long_strike": 100, "mid_debit": 1.25, "trade_status": "Watch"}]
    )
    after = before.copy()
    after["trade_status"] = "Scout"
    after["overlay_live_pricing_refreshed"] = True

    changes = _compare_v4_overlay_changes(before, after)

    assert changes.loc[0, "changed_live_pricing"] == "refreshed_unchanged"
