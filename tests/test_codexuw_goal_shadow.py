from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from codexuw import goal_shadow


def _candidate(ticker: str, *, dp_bias: float, earnings: bool = False) -> dict[str, object]:
    return {
        "ticker": ticker,
        "strategy": "Bull Put Credit Spread",
        "strategy_kind": "Credit",
        "direction": "Bull Put",
        "regime_trend": "uptrend",
        "expiry": "2026-08-21",
        "dte": 30,
        "stock_price_eod": 100.0,
        "short_leg_eod": f"{ticker}260821P00090000",
        "long_leg_eod": f"{ticker}260821P00085000",
        "short_strike_eod": 90.0,
        "long_strike_eod": 85.0,
        "spread_width": 5.0,
        "credit": 1.0,
        "mid_credit": 1.1,
        "natural_credit": 1.0,
        "credit_pct_width": 0.20,
        "quote_width_pct": 0.10,
        "max_loss": 400.0,
        "iv30d": 0.25,
        "option_flow_bias": 0.10,
        "combined_flow_bias": 0.10,
        "flow_total_premium": 100_000_000.0,
        "dp_flow_bias": dp_bias,
        "dp_directional_ratio": 0.80,
        "dark_pool_source_status": "dp_eod_loaded",
        "bot_flow_source_status": "bot_eod_loaded",
        "flow_quality": "directional",
        "oi_carryover_status": "supportive",
        "live_status": "PASS",
        "earnings_before_expiry": earnings,
        "hard_rejects": "",
        "penalties": "",
    }


def test_goal_shadow_selects_one_dp_blended_candidate_and_never_executes() -> None:
    scored = pd.DataFrame(
        [
            _candidate("AAA", dp_bias=0.60),
            _candidate("BBB", dp_bias=-0.60),
        ]
    )

    shadow = goal_shadow.build_goal_shadow_candidates(
        scored,
        asof=dt.date(2026, 7, 22),
    )

    assert len(shadow) == 1
    row = shadow.iloc[0]
    assert row["ticker"] == "AAA"
    assert row["policy_id"] == goal_shadow.GOAL_SHADOW_POLICY_ID
    assert bool(row["shadow_only"]) is True
    assert bool(row["execution_eligible"]) is False
    assert bool(row["no_order_placement"]) is True
    assert row["outcome_status"] == "PENDING"
    assert row["effective_flow_bias"] == 0.10 * 0.80 + 0.60 * 0.20


def test_goal_shadow_blocks_earnings_and_missing_dp() -> None:
    missing_dp = _candidate("AAA", dp_bias=0.60)
    missing_dp["dark_pool_source_status"] = "missing_dp_eod"
    scored = pd.DataFrame(
        [
            missing_dp,
            _candidate("BBB", dp_bias=0.60, earnings=True),
        ]
    )

    shadow = goal_shadow.build_goal_shadow_candidates(
        scored,
        asof=dt.date(2026, 7, 22),
    )

    assert shadow.empty


def test_goal_shadow_ledger_is_idempotent_and_preserves_resolved_outcome(tmp_path: Path) -> None:
    shadow = goal_shadow.build_goal_shadow_candidates(
        pd.DataFrame([_candidate("AAA", dp_bias=0.60)]),
        asof=dt.date(2026, 7, 22),
    )
    ledger_path = tmp_path / goal_shadow.GOAL_SHADOW_LEDGER_NAME

    first = goal_shadow.update_goal_shadow_ledger(ledger_path, shadow)
    second = goal_shadow.update_goal_shadow_ledger(ledger_path, shadow)
    resolved = second.copy()
    resolved.loc[0, "outcome_status"] = "RESOLVED_WIN"
    resolved.loc[0, "pnl_1x"] = 50.0
    resolved.to_csv(ledger_path, index=False)
    final = goal_shadow.update_goal_shadow_ledger(ledger_path, shadow)

    assert len(first) == len(second) == len(final) == 1
    assert final.iloc[0]["outcome_status"] == "RESOLVED_WIN"
    assert final.iloc[0]["pnl_1x"] == 50.0


def test_resolve_goal_shadow_ledger_updates_future_outcome(tmp_path: Path, monkeypatch) -> None:
    shadow = goal_shadow.build_goal_shadow_candidates(
        pd.DataFrame([_candidate("AAA", dp_bias=0.60)]),
        asof=dt.date(2026, 7, 22),
    )
    ledger_path = tmp_path / goal_shadow.GOAL_SHADOW_LEDGER_NAME
    shadow.to_csv(ledger_path, index=False)
    monkeypatch.setattr(goal_shadow, "dated_folders", lambda *args, **kwargs: [])
    monkeypatch.setattr(goal_shadow, "load_close_history", lambda *args, **kwargs: {})
    monkeypatch.setattr(goal_shadow, "load_hot_history", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        goal_shadow,
        "simulate_spread_exit",
        lambda *args, **kwargs: {
            "exact_evaluated": True,
            "exit_day": dt.date(2026, 7, 24),
            "exit_reason": "profit_target",
            "exit_debit": 0.40,
            "pnl_1x": 60.0,
            "return_on_risk": 0.15,
        },
    )

    resolved = goal_shadow.resolve_goal_shadow_ledger(
        root=tmp_path,
        ledger_path=ledger_path,
        through_date=dt.date(2026, 7, 24),
    )

    assert resolved.iloc[0]["outcome_status"] == "RESOLVED_WIN"
    assert resolved.iloc[0]["exit_reason"] == "profit_target"
    assert resolved.iloc[0]["pnl_1x"] == 60.0
