from __future__ import annotations

import pandas as pd

from codexuw.daily_v4 import _payoff_model_ready
from codexuw.payoff_calibration import (
    PROBATIONARY_PAYOFF_STATUS,
    apply_payoff_calibration,
    build_default_payoff_calibration,
)


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


def test_early_exit_is_not_evidence_until_contract_expiry(tmp_path) -> None:
    matured = _history_row("2026-06-01", flow_quality="directional", pnl_after_stress=100.0)
    matured["expiry"] = "2026-06-20"
    early_winner = _history_row("2026-06-02", flow_quality="directional", pnl_after_stress=100.0)
    early_winner["exit_day"] = "2026-06-10"
    early_winner["expiry"] = "2026-08-21"
    history_path = tmp_path / "history.csv"
    pd.DataFrame([matured, early_winner]).to_csv(history_path, index=False)

    summary, _, _ = build_default_payoff_calibration(
        asof="2026-07-20",
        history_path=history_path,
    )

    assert summary["eligible_rows"] == 1


def test_monthly_walk_forward_train_waits_for_contract_maturity(tmp_path) -> None:
    rows = []
    for day in pd.date_range("2026-01-02", periods=12, freq="2D"):
        row = _history_row(str(day.date()), flow_quality="directional", pnl_after_stress=100.0)
        row["expiry"] = "2026-04-17"
        rows.append(row)
    rows.append(
        {
            **_history_row("2026-03-03", flow_quality="directional", pnl_after_stress=100.0),
            "expiry": "2026-03-20",
        }
    )
    history_path = tmp_path / "history.csv"
    pd.DataFrame(rows).to_csv(history_path, index=False)

    _, _, walk_forward = build_default_payoff_calibration(
        asof="2026-05-01",
        history_path=history_path,
    )

    march = walk_forward[
        walk_forward["test_start"].eq("2026-03-01")
        & walk_forward["group_key"].eq(
            "flow_cost::Credit|Bear Call|range|flow=directional|cost=18to30"
        )
    ]
    assert march.empty


def test_probationary_base_route_beats_thinner_child_without_bypassing_veto() -> None:
    groups = pd.DataFrame(
        [
            {
                "group_key": "flow_cost::Credit|Bear Call|uptrend|flow=ambiguous|cost=18to30",
                "route_level": "flow_cost",
                "payoff_calibration_status": "INSUFFICIENT",
                "sample_size": 5,
                "minimum_group_sample": 12,
            },
            {
                "group_key": "flow::Credit|Bear Call|uptrend|flow=ambiguous",
                "route_level": "flow",
                "payoff_calibration_status": "INSUFFICIENT",
                "sample_size": 8,
                "minimum_group_sample": 15,
            },
            {
                "group_key": "base::Credit|Bear Call|uptrend",
                "route_level": "base",
                "payoff_calibration_status": PROBATIONARY_PAYOFF_STATUS,
                "sample_size": 29,
                "minimum_group_sample": 20,
                "stress_10_profit_factor": 1.87,
                "walk_forward_oos_sample": 10,
                "walk_forward_oos_profit_factor": 1.57,
            },
        ]
    )
    live = pd.DataFrame(
        [{"strategy_kind": "Credit", "direction": "Bear Call", "regime_trend": "uptrend", "flow_quality": "ambiguous", "credit_pct_width": 0.27}]
    )

    calibrated = apply_payoff_calibration(live, groups)

    assert calibrated.iloc[0]["payoff_calibration_status"] == PROBATIONARY_PAYOFF_STATUS
    assert calibrated.iloc[0]["payoff_route_level"] == "base"
