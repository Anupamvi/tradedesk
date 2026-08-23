from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from codexuw import range_gex_income_book_v2 as book


def test_strict_timing_uses_market_date_not_utc_calendar_date() -> None:
    summary = pd.DataFrame([
        {"ticker": "AAA", "date": "2026-07-09", "captured_utc": "2026-07-10T02:54:22+00:00", "uw_time": "2026-07-09T19:59:44Z", "spot": 100.0, "gamma_oi_per_1pct": 1000.0, "gamma_vol_per_1pct": 100.0, "gamma_dir_per_1pct": 10.0},
        {"ticker": "BBB", "date": "2026-07-09", "captured_utc": "2026-07-11T02:54:22+00:00", "uw_time": "2026-07-09T19:59:44Z", "spot": 100.0, "gamma_oi_per_1pct": 1000.0, "gamma_vol_per_1pct": 100.0, "gamma_dir_per_1pct": 10.0},
    ])
    strikes = pd.DataFrame([
        {"ticker": ticker, "date": "2026-07-09", "spot": 100.0, "strike": strike, "call_gamma_oi": call, "put_gamma_oi": put}
        for ticker in ["AAA", "BBB"]
        for strike, call, put in [(90.0, 10.0, -100.0), (110.0, 100.0, -10.0)]
    ])
    result = book.derive_strict_gex_features(summary, strikes).set_index("ticker")
    assert bool(result.loc["AAA", "gex_source_point_in_time"]) is True
    assert bool(result.loc["BBB", "gex_source_point_in_time"]) is False


def test_joint_condor_uses_combined_four_leg_mark() -> None:
    row = pd.Series({
        "ticker": "AAA", "entry_day": "2026-07-10", "expiry": "2026-08-21",
        "total_credit": 2.0, "max_wing_width": 5.0,
        "short_contract_put": "SP", "long_contract_put": "LP",
        "short_contract_call": "SC", "long_contract_call": "LC",
        "short_strike_live_put": 95.0, "long_strike_live_put": 90.0,
        "short_strike_live_call": 105.0, "long_strike_live_call": 110.0,
    })
    quotes = {dt.date(2026, 7, 11): {"SP": {"mid": 0.7}, "LP": {"mid": 0.2}, "SC": {"mid": 0.6}, "LC": {"mid": 0.2}}}
    result = book.simulate_joint_condor_exit(row, quote_history=quotes, close_history={}, through_date=dt.date(2026, 7, 11))
    assert result["exact_evaluated"] is True
    assert result["exit_reason"] == "profit_target"
    assert result["exit_value"] == pytest.approx(0.9)
    assert result["pnl_1x"] == pytest.approx(110.0)


def test_missing_point_in_time_gex_stays_visible_and_non_executable(tmp_path) -> None:
    paths, summary = book.write_prospective_outputs(pd.DataFrame(), out_dir=tmp_path / "run", root=tmp_path, asof=dt.date(2026, 8, 13))
    _, repeated = book.write_prospective_outputs(pd.DataFrame(), out_dir=tmp_path / "run2", root=tmp_path, asof=dt.date(2026, 8, 14))
    assert summary["status"] == "MISSING_POINT_IN_TIME_GEX"
    assert summary["execution_authorized"] is False
    assert repeated["status"] == "MISSING_POINT_IN_TIME_GEX"
    assert "range_gex_shadow_status" in paths
    assert "range_gex_collection_universe" in paths
