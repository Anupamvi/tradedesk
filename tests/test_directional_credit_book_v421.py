from __future__ import annotations

import datetime as dt

import pandas as pd

from codexuw.daily_v4 import (
    _directional_credit_execution_ready,
    _medium_debit_sleeve_eligible,
    _post_pricing_expectancy,
)
from codexuw.pipeline_versions import (
    PIPELINE_VERSION_V4,
    PIPELINE_VERSION_V420,
    PREVIOUS_PIPELINE_VERSION_LOCKS,
)
from codexuw.replay import _entry_quote
from codexuw.walk_forward_credit_book import (
    apply_walk_forward_credit_model,
    build_directional_credit_summary,
    build_walk_forward_credit_model,
    load_walk_forward_credit_history,
)


def test_directional_credit_evidence_passes_maturity_safe_train_holdout() -> None:
    summary = build_directional_credit_summary(asof=dt.date(2026, 8, 12))

    assert summary["status"] == "PASS"
    assert summary["sample_size"] == 83
    assert summary["win_rate"] > 0.90
    assert summary["stress_profit_factor_10pct"] > 3.50
    assert summary["positive_months"] == 7
    assert summary["train_sample_size"] == 52
    assert summary["train_stress_profit_factor_10pct"] > 20.0
    assert summary["holdout_sample_size"] == 23
    assert summary["holdout_win_rate"] > 0.82
    assert summary["holdout_stress_profit_factor_10pct"] > 1.70


def test_impossible_credit_is_removed_from_model_history() -> None:
    history = load_walk_forward_credit_history()

    assert (history["entry_credit"] > 0).all()
    assert (history["entry_credit"] < history["entry_width"]).all()
    assert not (
        history["ticker"].eq("SNDK")
        & history["entry_credit"].ge(history["entry_width"])
    ).any()


def test_replay_rejects_credit_at_or_above_spread_width() -> None:
    row = pd.Series(
        {
            "direction": "Bear Call",
            "strategy_kind": "Credit",
            "short_leg_eod": "SHORT",
            "long_leg_eod": "LONG",
            "short_strike_eod": 100.0,
            "long_strike_eod": 110.0,
        }
    )
    quotes = {
        "SHORT": {"bid": 20.0, "ask": 20.0, "mid": 20.0},
        "LONG": {"bid": 5.0, "ask": 5.0, "mid": 5.0},
    }

    result = _entry_quote(row, quotes, slippage_pct=0.10)

    assert result["entry_credit"] > result["entry_width"]
    assert result["exact_fillable"] is False


def test_directional_credit_lane_can_reach_medium_book_without_supportive_oi() -> None:
    summary, _, model = build_walk_forward_credit_model(asof=dt.date(2026, 8, 12))
    row = pd.DataFrame(
        [
            {
                "ticker": "TEST",
                "sector": "Industrials",
                "direction": "Bear Call",
                "flow_quality": "directional",
                "combined_flow_bias": -0.20,
                "oi_carryover_status": "contrary",
                "credit_pct_width": 0.25,
                "quote_width_pct": 0.10,
                "expected_move_ratio": 0.50,
                "dte": 30,
                "width": 5.0,
                "natural_credit": 1.10,
                "mid_credit": 1.25,
                "expiry": "2026-09-18",
                "next_earnings_dt": None,
                "hard_rejects": "",
                "regime_trend": "range",
            }
        ]
    )

    applied = apply_walk_forward_credit_model(row, summary, model)

    assert bool(applied.loc[0, "directional_credit_qualified"])
    assert bool(applied.loc[0, "walk_forward_credit_qualified"])
    assert bool(applied.loc[0, "walk_forward_credit_policy_pass"])
    assert _directional_credit_execution_ready(applied.loc[0])
    priced = applied.loc[0].copy()
    priced["max_loss"] = 375.0
    expected_value, profit_factor, _, _ = _post_pricing_expectancy(priced)
    assert expected_value > 0
    assert profit_factor > 3.5


def test_corrected_replay_disables_legacy_debit_authority() -> None:
    assert not _medium_debit_sleeve_eligible(
        {
            "strategy": "Bull Call Debit",
            "debit_policy_tier": "High",
            "payoff_calibration_status": "PASS",
            "payoff_stress_10_profit_factor": 99.0,
        }
    )


def test_v421_retains_v420_as_rollback_target() -> None:
    assert PIPELINE_VERSION_V4.startswith("v4.21-")
    assert PIPELINE_VERSION_V420.startswith("v4.20-")
    assert PREVIOUS_PIPELINE_VERSION_LOCKS[PIPELINE_VERSION_V420]["superseded_by"] == PIPELINE_VERSION_V4
