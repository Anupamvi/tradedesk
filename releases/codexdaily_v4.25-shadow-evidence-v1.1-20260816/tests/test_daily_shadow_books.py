from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from codexuw import daily_shadow_books as shadow


def test_live_adapter_uses_natural_debit_and_actual_live_legs(monkeypatch) -> None:
    captured: dict[str, pd.DataFrame] = {}

    def prepare(frame: pd.DataFrame, technical: pd.DataFrame) -> pd.DataFrame:
        captured["frame"] = frame.copy()
        out = frame.copy()
        out["debit_pct_width"] = out["entry_debit"] / out["entry_width"]
        out["breakeven_sigma"] = 0.4
        out["iv_hv_ratio"] = 1.0
        return out

    monkeypatch.setattr(shadow.debit_model, "prepare_history", prepare)
    monkeypatch.setattr(shadow.debit_model, "candidate_guard", lambda frame: pd.Series(True, index=frame.index))
    scored = pd.DataFrame(
        [{
            "ticker": "AAA", "strategy": "Bull Call Debit Spread", "expiry": "2026-09-18",
            "quote_observation_date": "2026-08-14", "natural_debit": 1.25, "mid_debit": 1.10,
            "spread_width": 5.0, "quote_width_pct": 0.10, "dte": 35,
            "stock_price_live": 100.0, "regime_trend": "uptrend", "reward_risk": 3.0,
            "breakeven": 101.25, "breakeven_expected_move_ratio": 0.40,
            "long_leg": "AAA LIVE BUY", "short_leg": "AAA LIVE SELL", "long_leg_eod": "OLD BUY",
            "short_leg_eod": "OLD SELL", "long_strike": 100.0, "short_strike": 105.0,
            "long_strike_eod": 99.0, "short_strike_eod": 104.0, "live_status": "PASS",
            "regular_session_quote": True,
        }]
    )
    result = shadow.prepare_live_debit_candidates(scored, asof=dt.date(2026, 8, 13))
    raw = captured["frame"].iloc[0]
    assert raw["entry_debit"] == 1.25
    assert raw["long_leg_eod"] == "AAA LIVE BUY"
    assert raw["short_leg_eod"] == "AAA LIVE SELL"
    assert raw["long_strike_eod"] == 100.0
    assert result.iloc[0]["breakeven_sigma"] == 0.40
    assert bool(result.iloc[0]["execution_authorized"]) is False if "execution_authorized" in result else True


def test_debit_ledger_is_idempotent_and_preserves_resolved(tmp_path: Path) -> None:
    row = {column: "" for column in shadow.DEBIT_LEDGER_COLUMNS}
    row.update({"policy_version": shadow.DEBIT_SHADOW_POLICY_VERSION, "signal_date": "2026-08-13", "ticker": "AAA", "strategy": "Bull Call Debit Spread", "expiry": "2026-09-18", "buy_leg": "BUY", "sell_leg": "SELL", "generated_at_utc": "2026-08-14T00:00:00+00:00", "outcome_status": "PENDING"})
    incoming = pd.DataFrame([row])
    path = tmp_path / shadow.DEBIT_LEDGER_NAME
    first = shadow.update_debit_shadow_ledger(path, incoming)
    second = shadow.update_debit_shadow_ledger(path, incoming)
    second.loc[0, "outcome_status"] = "RESOLVED_WIN"
    second.loc[0, "pnl_1x"] = 100.0
    second.to_csv(path, index=False)
    final = shadow.update_debit_shadow_ledger(path, incoming)
    assert len(first) == len(final) == 1
    assert final.iloc[0]["outcome_status"] == "RESOLVED_WIN"
    assert float(final.iloc[0]["pnl_1x"]) == 100.0


def test_ledger_rows_are_never_executable() -> None:
    selected = pd.DataFrame([{
        "entry_day": pd.Timestamp("2026-08-14"), "ticker": "AAA", "sector": "Tech",
        "strategy": "Bull Call Debit Spread", "direction": "Bull Call", "expiry": "2026-09-18",
        "long_leg_eod": "BUY", "short_leg_eod": "SELL", "long_strike_eod": 100.0,
        "short_strike_eod": 105.0, "entry_debit": 1.25, "entry_width": 5.0,
        "predicted_win_probability": 0.70, "predicted_ev_payoff_correct": 50.0,
        "prior_sample_size": 200, "model_training_through": "2026-08-01", "feature_parity": "8/8",
    }])
    rows = shadow.build_debit_ledger_rows(selected, asof=dt.date(2026, 8, 13))
    assert bool(rows.iloc[0]["shadow_only"]) is True
    assert bool(rows.iloc[0]["execution_authorized"]) is False
    assert bool(rows.iloc[0]["no_order_placement"]) is True
    assert rows.iloc[0]["target_exit_value"] == 2.5
