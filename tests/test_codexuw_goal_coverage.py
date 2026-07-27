from __future__ import annotations

import pandas as pd

from codexuw.goal_coverage import (
    CoveragePolicy,
    _ridge_predict_ror,
    performance,
    prepare_detail,
    select_policy_trades,
)


def _row(*, asof: str, ticker: str, exit_day: str, pnl: float, flow_bias: float = 0.20) -> dict[str, object]:
    return {
        "asof": asof,
        "exit_day": exit_day,
        "ticker": ticker,
        "strategy": "Bull Put Credit Spread",
        "strategy_kind": "Credit",
        "direction": "Bull Put",
        "regime": "downtrend",
        "dte": 30,
        "iv_rank": 45.0,
        "stock_price_eod": 100.0,
        "short_strike_eod": 90.0,
        "long_strike_eod": 85.0,
        "entry_width": 5.0,
        "entry_price": 1.35,
        "entry_credit": 1.35,
        "entry_credit_pct_width": 0.27,
        "entry_quote_width_pct": 0.10,
        "iv30d": 0.25,
        "realized_volatility_30d": 0.17,
        "combined_flow_bias": flow_bias,
        "option_flow_bias": flow_bias,
        "flow_quality": "directional",
        "oi_carryover_status": "supportive",
        "exact_fillable": True,
        "exact_evaluated": True,
        "replay_guard_pass": True,
        "pnl_1x": pnl,
        "return_on_risk": pnl / 365.0,
    }


def test_goal_selector_does_not_learn_outcome_before_exit_day() -> None:
    detail = prepare_detail(
        pd.DataFrame(
            [
                _row(asof="2026-02-02", ticker="EARLY", exit_day="2026-02-10", pnl=100.0),
                _row(asof="2026-02-03", ticker="NEXT", exit_day="2026-02-11", pnl=50.0),
            ]
        )
    )
    policy = CoveragePolicy(
        max_per_day=1,
        eligibility_mode="policy",
        dark_pool_weight=0.0,
        oi_mode="reject_contrary",
        history_weight=2.0,
        min_prior_sample=1,
        model_weight=2.0,
    )

    selected = select_policy_trades(detail, policy)
    next_day = selected[selected["ticker"].eq("NEXT")].iloc[0]

    assert next_day["goal_history_sample"] == 0
    assert next_day["goal_history_level"] == "none"


def test_goal_performance_counts_daily_winner_and_profit_factor() -> None:
    selected = prepare_detail(
        pd.DataFrame(
            [
                _row(asof="2026-02-02", ticker="WIN", exit_day="2026-02-04", pnl=100.0),
                _row(asof="2026-02-02", ticker="LOSS", exit_day="2026-02-04", pnl=-20.0),
                _row(asof="2026-02-03", ticker="LOSS2", exit_day="2026-02-05", pnl=-20.0),
            ]
        )
    )

    result = performance(selected, pd.to_datetime(["2026-02-02", "2026-02-03"]))

    assert result["winner_days"] == 1
    assert result["winner_day_rate"] == 0.5
    assert result["profit_factor"] == 2.5
    assert result["total_pnl"] == 60.0


def test_ridge_ranker_clips_extreme_and_nonfinite_features() -> None:
    train = pd.DataFrame(
        [
            {**_row(asof="2026-01-02", ticker="A", exit_day="2026-01-05", pnl=10.0), "_ror": 0.1, "_asof_dt": pd.Timestamp("2026-01-02"), "reward_risk": float("inf"), "flow_total_premium": 1e308},
            {**_row(asof="2026-01-03", ticker="B", exit_day="2026-01-06", pnl=-10.0), "_ror": -0.1, "_asof_dt": pd.Timestamp("2026-01-03"), "reward_risk": -1e308, "flow_total_premium": float("nan")},
        ]
    )
    test = pd.DataFrame(
        [
            {**_row(asof="2026-01-04", ticker="C", exit_day="2026-01-07", pnl=0.0), "_ror": 0.0, "_asof_dt": pd.Timestamp("2026-01-04"), "reward_risk": 1e307, "flow_total_premium": float("inf")},
        ]
    )

    prediction = _ridge_predict_ror(train, test)

    assert len(prediction) == 1
    assert pd.notna(prediction[0])
    assert -1.0 <= prediction[0] <= 3.0
