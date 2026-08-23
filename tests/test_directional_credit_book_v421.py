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
    PIPELINE_VERSION_V421,
    PIPELINE_VERSION_V422,
    PIPELINE_VERSION_V423,
    PIPELINE_VERSION_V424,
    PIPELINE_VERSION_V425,
    PIPELINE_VERSION_V426,
    PREVIOUS_PIPELINE_VERSION_LOCKS,
)
from codexuw.replay import _entry_quote
from codexuw.walk_forward_credit_book import (
    apply_walk_forward_credit_model,
    build_directional_credit_summary,
    build_walk_forward_credit_model,
    load_walk_forward_credit_history,
)


def test_directional_credit_execution_evidence_passes_recomputed_fill_stress() -> None:
    summary = build_directional_credit_summary(asof=dt.date(2026, 8, 12))

    assert summary["status"] == "PASS"
    assert summary["fill_stress_source"] == "recomputed_from_pnl_1x_and_entry_credit"
    assert summary["reference_validation_status"] == "FAIL"
    assert summary["execution_validation_status"] == "PASS"
    assert summary["sample_size"] == 83
    assert summary["win_rate"] > 0.90
    assert 2.80 < summary["stress_profit_factor_10pct"] < 3.00
    assert round(summary["stress_total_pnl_10pct"], 2) == 3933.03
    assert summary["positive_months"] == 7
    assert summary["train_sample_size"] == 52
    assert summary["train_stress_profit_factor_10pct"] > 18.0
    assert summary["holdout_sample_size"] == 23
    assert summary["holdout_win_rate"] > 0.82
    assert 1.40 < summary["holdout_stress_profit_factor_10pct"] < 1.50
    assert summary["execution_sample_size"] == 53
    assert summary["execution_wilson_lower_bound"] > 0.82
    assert summary["execution_stress_profit_factor_10pct"] > 3.40
    assert summary["execution_holdout_sample_size"] == 15
    assert summary["execution_holdout_win_rate"] > 0.86
    assert summary["execution_holdout_stress_profit_factor_10pct"] > 1.80
    assert summary["execution_family_validation_status"] == "PASS"
    assert summary["execution_family_metrics"]["Bear Call"]["validation_status"] == "PASS"
    assert summary["execution_family_metrics"]["Bear Call"]["stress_profit_factor_10pct"] > 2.20
    assert summary["execution_family_metrics"]["Bull Put"]["validation_status"] == "PROBATIONARY"
    assert summary["execution_family_metrics"]["Bull Put"]["stress_profit_factor_10pct"] > 12.0


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


def test_directional_credit_lane_requires_supportive_or_matched_oi_and_has_no_signal_cap() -> None:
    summary, _, model = build_walk_forward_credit_model(asof=dt.date(2026, 8, 12))
    row = pd.DataFrame(
        [
            {
                "ticker": "TEST",
                "sector": "Industrials",
                "direction": "Bear Call",
                "flow_quality": "directional",
                "combined_flow_bias": -0.20,
                "oi_carryover_status": "supportive",
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
    assert applied.loc[0, "directional_credit_family_validation_status"] == "PASS"
    assert applied.loc[0, "walk_forward_credit_confidence_score"] < 82.0
    priced = applied.loc[0].copy()
    priced["max_loss"] = 375.0
    expected_value, profit_factor, _, _ = _post_pricing_expectancy(priced)
    assert expected_value > 0
    assert 2.20 < profit_factor < 2.40

    contrary = row.copy()
    contrary.loc[0, "oi_carryover_status"] = "contrary"
    blocked = apply_walk_forward_credit_model(contrary, summary, model)
    assert not bool(blocked.loc[0, "directional_credit_qualified"])
    assert not bool(blocked.loc[0, "walk_forward_credit_policy_pass"])
    assert not _directional_credit_execution_ready(blocked.loc[0])

    family_blocked = applied.loc[0].copy()
    family_blocked["directional_credit_family_validation_status"] = "FAIL"
    assert not _directional_credit_execution_ready(family_blocked)


def test_corrected_replay_disables_legacy_debit_authority() -> None:
    assert not _medium_debit_sleeve_eligible(
        {
            "strategy": "Bull Call Debit",
            "debit_policy_tier": "High",
            "payoff_calibration_status": "PASS",
            "payoff_stress_10_profit_factor": 99.0,
        }
    )


def test_v427_retains_v426_through_v420_as_rollback_targets() -> None:
    assert PIPELINE_VERSION_V4.startswith("v4.27-")
    assert PIPELINE_VERSION_V424.startswith("v4.24-")
    assert PIPELINE_VERSION_V423.startswith("v4.23-")
    assert PIPELINE_VERSION_V422.startswith("v4.22-")
    assert PIPELINE_VERSION_V420.startswith("v4.20-")
    assert PIPELINE_VERSION_V421.startswith("v4.21-")
    assert PREVIOUS_PIPELINE_VERSION_LOCKS[PIPELINE_VERSION_V420]["superseded_by"] == PIPELINE_VERSION_V421
    assert PREVIOUS_PIPELINE_VERSION_LOCKS[PIPELINE_VERSION_V421]["superseded_by"] == PIPELINE_VERSION_V422
    assert PREVIOUS_PIPELINE_VERSION_LOCKS[PIPELINE_VERSION_V422]["superseded_by"] == PIPELINE_VERSION_V423
    assert PREVIOUS_PIPELINE_VERSION_LOCKS[PIPELINE_VERSION_V423]["superseded_by"] == PIPELINE_VERSION_V424
    assert PREVIOUS_PIPELINE_VERSION_LOCKS[PIPELINE_VERSION_V424]["superseded_by"] == PIPELINE_VERSION_V425
    assert PREVIOUS_PIPELINE_VERSION_LOCKS[PIPELINE_VERSION_V425]["superseded_by"] == PIPELINE_VERSION_V426
    assert PREVIOUS_PIPELINE_VERSION_LOCKS[PIPELINE_VERSION_V426]["superseded_by"] == PIPELINE_VERSION_V4
