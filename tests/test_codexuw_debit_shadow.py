import datetime as dt
from pathlib import Path

import pandas as pd

from codexuw.debit_shadow import (
    build_shadow_rows,
    write_daily_debit_shadow,
    write_replay_debit_shadow,
)
from codexuw.replay import apply_replay_decision_selection


def _credit_row(asof: dt.date) -> dict:
    return {
        "asof": asof,
        "ticker": "CREDIT",
        "direction": "Bull Put",
        "strategy": "Bull Put Credit Spread",
        "regime": "downtrend",
        "exact_fillable": True,
        "exact_evaluated": True,
        "entry_credit_pct_width": 0.25,
        "entry_quote_width_pct": 0.10,
        "stock_price_eod": 100.0,
        "short_strike_eod": 90.0,
        "long_strike_eod": 85.0,
        "iv30d": 0.30,
        "realized_volatility_30d": 0.20,
        "iv_rank": 45.0,
        "combined_flow_bias": 0.20,
        "dte": 30,
        "pnl_1x": 50.0,
    }


def _debit_row(asof: dt.date, **overrides) -> dict:
    row = {
        "asof": asof,
        "ticker": "DEBIT",
        "direction": "Bull Call",
        "strategy": "Bull Call Debit Spread",
        "exact_fillable": True,
        "exact_evaluated": True,
        "entry_debit": 1.35,
        "entry_debit_pct_width": 0.35,
        "entry_width": 5.0,
        "reward_risk": 1.80,
        "iv30d": 0.30,
        "iv_rank": 40.0,
        "combined_flow_bias": 0.30,
        "bot_flow_source_status": "bot_eod_loaded",
        "flow_quality": "directional",
        "regime": "uptrend",
        "entry_quote_width_pct": 0.10,
        "dte": 32,
        "short_leg_eod": "ABC  260918C00100000",
        "long_leg_eod": "ABC  260918C00095000",
        "short_strike_eod": 100.0,
        "long_strike_eod": 95.0,
        "expiry": dt.date(2026, 9, 18),
        "pnl_1x": 80.0,
    }
    row.update(overrides)
    return row


def test_zero_debit_cap_final_has_no_debits_shadow_still_written(tmp_path: Path):
    asof = dt.date(2026, 4, 20)
    detail = apply_replay_decision_selection(
        pd.DataFrame([_credit_row(asof), _debit_row(asof)]),
        max_selected_per_day=1,
        max_debit_selected_per_day=0,
    )
    selected = detail[detail["decision_pass"] == True]
    assert (selected["direction"] == "Bull Call").sum() == 0
    assert int((selected["ticker"] == "DEBIT").sum()) == 0

    paths, metrics = write_replay_debit_shadow(detail=detail, out_dir=tmp_path)
    ledger = pd.read_csv(paths["ledger"])
    assert len(ledger) >= 1
    assert (ledger["ticker"] == "DEBIT").any()
    assert not bool(ledger["decision_pass"].astype(str).str.lower().eq("true").any())
    assert not bool(ledger["execution_authorized"].astype(str).str.lower().eq("true").any())
    assert int(metrics["qualified_rows"]) >= 1


def test_shadow_fail_still_written_never_execute():
    asof = dt.date(2026, 4, 20)
    fail = _debit_row(asof, regime="range", entry_debit_pct_width=0.60, reward_risk=0.80)
    rows = build_shadow_rows(pd.DataFrame([fail]), asof=asof)
    assert len(rows) == 1
    assert bool(rows.iloc[0]["shadow_qualified"]) is False
    assert "regime_not_uptrend" in str(rows.iloc[0]["fail_reasons"])
    assert not bool(rows.iloc[0]["decision_pass"])
    assert "never Execute" in str(rows.iloc[0]["why_not_execute"])


def test_daily_shadow_appends_ledger(tmp_path: Path):
    asof = dt.date(2026, 8, 25)
    out_dir = tmp_path / "codexdaily_2026-08-25"
    out_dir.mkdir()
    scored = pd.DataFrame([_debit_row(asof, ticker="TSLA")])
    ledger_path = write_daily_debit_shadow(scored=scored, asof=asof, out_dir=out_dir)
    assert ledger_path == tmp_path / "debit_shadow" / "debit_shadow_ledger.csv"
    ledger = pd.read_csv(ledger_path)
    assert list(ledger["ticker"]) == ["TSLA"]
    assert (out_dir / f"debit_shadow_{asof}.csv").exists()
    write_daily_debit_shadow(scored=scored, asof=asof, out_dir=out_dir)
    rerun = pd.read_csv(ledger_path)
    assert len(rerun) == 1
