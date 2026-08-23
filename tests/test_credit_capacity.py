import pandas as pd

from codexuw.credit_capacity import (
    capacity_curve,
    execution_population,
    portfolio_metrics,
    prepare_history,
)


def _history() -> pd.DataFrame:
    return prepare_history(pd.DataFrame([
        {"ticker": "AAA", "sector": "Tech", "entry_day": "2026-01-02",
         "exit_day": "2026-01-05", "entry_width": 5.0, "entry_credit": 1.0,
         "pnl_1x": 100.0, "stress_pnl_10pct": 90.0},
        {"ticker": "BBB", "sector": "Finance", "entry_day": "2026-01-03",
         "exit_day": "2026-01-06", "entry_width": 4.0, "entry_credit": 1.0,
         "pnl_1x": -50.0, "stress_pnl_10pct": -60.0},
    ]))


def test_portfolio_metrics_tracks_overlap_risk_and_drawdown() -> None:
    result = portfolio_metrics(_history(), 2)
    assert result["trades"] == 2
    assert result["maximum_active_positions"] == 2
    assert result["peak_defined_risk"] == 1400.0
    assert result["base_total_pnl"] == 100.0
    assert result["realized_stress_max_drawdown"] == -120.0


def test_capacity_curve_does_not_claim_reliable_target() -> None:
    curve, summary = capacity_curve(_history(), scales=(1, 2), monthly_target=1000.0)
    assert curve["contracts_per_trade"].tolist() == [1, 2]
    assert summary["contracts_required_for_historical_average_target"] == 34
    assert summary["target_is_reliably_demonstrated"] is False


def test_execution_population_excludes_contrary_oi() -> None:
    history = _history()
    history["oi_carryover_status"] = ["supportive", "contrary"]
    assert execution_population(history)["ticker"].tolist() == ["AAA"]
