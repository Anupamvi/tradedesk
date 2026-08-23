import datetime as dt

import numpy as np
import pandas as pd

from codexuw.walk_forward_credit_book import (
    DEFAULT_WALK_FORWARD_CREDIT_HISTORY,
    _live_credit_guard,
    _sanitize_numeric_features,
    build_walk_forward_credit_model,
)


def test_directional_credit_book_replaces_invalidated_model_lane() -> None:
    summary, evidence, model = build_walk_forward_credit_model(
        asof=dt.date(2026, 8, 11),
        history_path=DEFAULT_WALK_FORWARD_CREDIT_HISTORY,
    )

    assert summary["status"] == "PASS"
    assert summary["strict_model_status"] == "FAIL"
    assert summary["model_tier"] == "Medium"
    assert not summary["high_confidence_available"]
    directional = summary["directional_credit_lane"]
    assert directional["status"] == "PASS"
    assert directional["fill_stress_source"] == "recomputed_from_pnl_1x_and_entry_credit"
    assert directional["reference_validation_status"] == "FAIL"
    assert directional["execution_validation_status"] == "PASS"
    assert directional["sample_size"] == 83
    assert directional["win_rate"] > 0.90
    assert 2.80 < directional["stress_profit_factor_10pct"] < 3.00
    assert directional["holdout_sample_size"] >= 20
    assert 1.40 < directional["holdout_stress_profit_factor_10pct"] < 1.50
    assert directional["execution_sample_size"] == 53
    assert directional["execution_stress_profit_factor_10pct"] > 3.40
    assert directional["execution_holdout_sample_size"] == 15
    assert directional["execution_holdout_stress_profit_factor_10pct"] > 1.80
    assert len(evidence) == summary["sample_size"]
    assert model is None


def test_live_credit_guard_requires_aligned_flow_and_validated_credit_band() -> None:
    row = {
        "direction": "Bull Put",
        "credit_pct_width": 0.27,
        "quote_width_pct": 0.05,
        "combined_flow_bias": 0.15,
        "technical_close": 800.0,
        "short_strike": 760.0,
        "dte": 38,
        "iv30d": 0.40,
        "iv_rank": 40.0,
        "regime_trend": "range",
        "oi_carryover_status": "supportive",
    }

    passed, _ = _live_credit_guard(row)
    contra, reason = _live_credit_guard({**row, "combined_flow_bias": -0.15})

    assert passed
    assert not contra
    assert reason == "aggregate_flow_not_aligned"

    long_dte, long_dte_reason = _live_credit_guard({**row, "dte": 60})
    assert not long_dte
    assert long_dte_reason == "dte_outside_validated_range"

    hard, hard_reason = _live_credit_guard({**row, "hard_rejects": "no_usable_liquidity"})
    assert not hard
    assert hard_reason == "preexisting_hard_blocker:no_usable_liquidity"

    negative_natural, negative_natural_reason = _live_credit_guard({**row, "natural_credit": -0.20})
    assert not negative_natural
    assert negative_natural_reason == "nonpositive_natural_credit"


def test_credit_model_numeric_sanitation_bounds_extreme_export_values() -> None:
    frame = pd.DataFrame(
        {
            "flow_total_premium": [1e308, -1e308, np.inf],
            "iv_hv_ratio": [1e308, -1e308, np.nan],
        }
    )

    clean = _sanitize_numeric_features(frame, ("flow_total_premium", "iv_hv_ratio"))

    assert clean.iloc[0].eq(1_000_000.0).all()
    assert clean.iloc[1].eq(-1_000_000.0).all()
    assert clean.iloc[2].isna().all()
