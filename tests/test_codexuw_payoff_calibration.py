from __future__ import annotations

import pandas as pd

from codexuw.daily_v4 import _payoff_model_ready
from codexuw.payoff_calibration import apply_payoff_calibration, build_default_payoff_calibration


def _history_row(day: str, *, flow_quality: str, pnl_after_stress: float) -> dict[str, object]:
    return {
        "asof": day,
        "exit_day": day,
        "exact_evaluated": True,
        "replay_guard_pass": True,
        "strategy_kind": "Credit",
        "strategy": "Bear Call Credit",
        "direction": "Bear Call",
        "regime": "range",
        "flow_quality": flow_quality,
        "entry_price": 1.0,
        "entry_width": 5.0,
        "pnl_1x": pnl_after_stress + 10.0,
    }


def test_directional_route_can_pass_when_coarse_lane_is_poisoned(tmp_path) -> None:
    rows: list[dict[str, object]] = []
    warmup_days = [
        "2026-01-05",
        "2026-01-12",
        "2026-01-20",
        "2026-01-27",
        "2026-02-03",
        "2026-02-10",
        "2026-02-17",
        "2026-02-20",
        "2026-02-24",
    ]
    warmup_pnl = [88.0, 88.0, 88.0, 88.0, 88.0, -100.0, -100.0, -100.0, -100.0]
    rows.extend(
        _history_row(day, flow_quality="directional", pnl_after_stress=pnl)
        for day, pnl in zip(warmup_days, warmup_pnl)
    )
    for day in [
        "2026-03-03",
        "2026-03-17",
        "2026-04-07",
        "2026-05-05",
        "2026-05-19",
        "2026-06-02",
        "2026-06-16",
        "2026-06-23",
    ]:
        rows.append(_history_row(day, flow_quality="directional", pnl_after_stress=100.0))
    for day in pd.date_range("2026-01-06", periods=24, freq="6D"):
        rows.append(_history_row(str(day.date()), flow_quality="unclear", pnl_after_stress=-90.0))

    history_path = tmp_path / "history.csv"
    pd.DataFrame(rows).to_csv(history_path, index=False)
    summary, groups, _ = build_default_payoff_calibration(asof="2026-07-20", history_path=history_path)

    directional = groups[
        groups["group_key"].eq("flow_cost::Credit|Bear Call|range|flow=directional|cost=18to30")
    ].iloc[0]
    base = groups[groups["group_key"].eq("base::Credit|Bear Call|range")].iloc[0]
    assert directional["payoff_calibration_status"] == "PASS"
    assert directional["walk_forward_oos_sample"] == 8
    assert directional["walk_forward_failed_windows"] == 0
    assert directional["post_activation_oos_sample"] == 3
    assert directional["post_activation_oos_profit_factor"] > 1.25
    assert base["payoff_calibration_status"] == "VETO"
    assert directional["group_key"] in summary["passed_lanes"]

    live = pd.DataFrame(
        [
            {
                "strategy_kind": "Credit",
                "direction": "Bear Call",
                "regime_trend": "range",
                "flow_quality": "directional",
                "credit_pct_width": 0.25,
            },
            {
                "strategy_kind": "Credit",
                "direction": "Bear Call",
                "regime_trend": "range",
                "flow_quality": "unclear",
                "credit_pct_width": 0.25,
            },
        ]
    )
    calibrated = apply_payoff_calibration(live, groups)
    assert calibrated.iloc[0]["payoff_calibration_status"] == "PASS"
    assert calibrated.iloc[0]["payoff_route_level"] == "flow_cost"
    assert calibrated.iloc[0]["payoff_minimum_sample_required"] == 12
    assert calibrated.iloc[1]["payoff_calibration_status"] == "VETO"


def test_infinite_profit_factor_means_no_losses_not_failed_evidence() -> None:
    assert _payoff_model_ready(
        {
            "strategy_kind": "Credit",
            "direction": "Bear Call",
            "regime_trend": "range",
            "payoff_route_key": "flow_cost::Credit|Bear Call|range|flow=directional|cost=18to30",
            "payoff_calibration_status": "PASS",
            "payoff_minimum_sample_required": 12,
            "payoff_sample_size": 17,
            "payoff_stress_10_profit_factor": 2.36,
            "payoff_walk_forward_oos_sample": 8,
            "payoff_walk_forward_oos_profit_factor": float("inf"),
            "payoff_post_activation_oos_sample": 3,
            "payoff_post_activation_oos_average_pnl": 53.15,
            "payoff_post_activation_oos_profit_factor": float("inf"),
        }
    )
