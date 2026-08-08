import pandas as pd

from codexuw.strategy_registry import (
    apply_strategy_registry_gate,
    build_strategy_registry,
    strategy_key_for_row,
)


def test_registry_covers_standard_strategy_surface_and_generated_directions() -> None:
    registry = build_strategy_registry(
        payoff_summary={"status": "NO_VALIDATED_LANES"},
        payoff_groups=pd.DataFrame(),
        confidence_summary={"family_validation": {}},
    )

    assert len(registry) >= 30
    assert {"directional", "income", "vertical", "volatility", "range", "butterfly", "time_spread", "ratio", "hedge"}.issubset(set(registry["category"]))
    assert strategy_key_for_row({"direction": "Bull Put"}) == "bull_put_credit_vertical"
    assert strategy_key_for_row({"direction": "Bear Call"}) == "bear_call_credit_vertical"
    assert strategy_key_for_row({"direction": "Bull Call"}) == "bull_call_debit_vertical"
    assert strategy_key_for_row({"direction": "Bear Put"}) == "bear_put_debit_vertical"
    assert int(registry["live_builder"].sum()) == len(registry)
    assert registry["research_support"].all()
    assert registry["historical_scope"].ne("unavailable").all()


def test_registry_requires_payoff_and_confidence_before_execution() -> None:
    groups = pd.DataFrame(
        [{"direction": "Bear Call", "payoff_calibration_status": "PASS"}]
    )
    blocked = build_strategy_registry(
        payoff_summary={"status": "PASS"},
        payoff_groups=groups,
        confidence_summary={"family_validation": {"Credit": {"status": "FAIL"}}},
    )
    allowed = build_strategy_registry(
        payoff_summary={"status": "PASS"},
        payoff_groups=groups,
        confidence_summary={"family_validation": {"Credit": {"status": "PASS"}}},
    )

    assert not blocked.set_index("strategy_key").loc["bear_call_credit_vertical", "execution_authorized"]
    assert allowed.set_index("strategy_key").loc["bear_call_credit_vertical", "execution_authorized"]


def test_registry_demotes_unauthorized_execute_row() -> None:
    registry = build_strategy_registry(
        payoff_summary={"status": "NO_VALIDATED_LANES"},
        payoff_groups=pd.DataFrame(),
        confidence_summary={"family_validation": {}},
    )
    scored = pd.DataFrame(
        [{"ticker": "AAA", "direction": "Bull Call", "trade_status": "Execute", "decision_eligible": True}]
    )

    gated = apply_strategy_registry_gate(scored, registry)

    assert gated.iloc[0]["trade_status"] == "Research"
    assert gated.iloc[0]["primary_blocker"] == "strategy_not_production:bull_call_debit_vertical"


def test_registry_marks_credit_probationary_route_one_lot_only() -> None:
    registry = build_strategy_registry(
        payoff_summary={"status": "NO_VALIDATED_LANES"},
        payoff_groups=pd.DataFrame(
            [{"direction": "Bear Call", "payoff_calibration_status": "PROBATIONARY"}]
        ),
        confidence_summary={"family_validation": {"Credit": {"status": "CONSERVATIVE"}}},
    ).set_index("strategy_key")

    row = registry.loc["bear_call_credit_vertical"]
    assert row["pipeline_status"] == "PROBATIONARY"
    assert not row["execution_authorized"]
    assert not row["probationary_execution_authorized"]


def test_registry_demotes_probationary_pilot_execute() -> None:
    registry = build_strategy_registry(
        payoff_summary={"status": "NO_VALIDATED_LANES"},
        payoff_groups=pd.DataFrame(
            [{"direction": "Bear Call", "payoff_calibration_status": "PROBATIONARY"}]
        ),
        confidence_summary={"family_validation": {"Credit": {"status": "CONSERVATIVE"}}},
    )
    rows = pd.DataFrame(
        [
            {"ticker": "PILOT", "direction": "Bear Call", "trade_status": "Execute", "trade_tier": "Execute V4 Pilot - 1 Contract", "contracts": 9, "decision_eligible": True},
            {"ticker": "DIRECT", "direction": "Bear Call", "trade_status": "Execute", "trade_tier": "Execute V4 Direct", "contracts": 1, "decision_eligible": True},
        ]
    )

    gated = apply_strategy_registry_gate(rows, registry)

    assert gated.loc[gated["ticker"].eq("PILOT"), "trade_status"].iloc[0] == "Research"
    assert gated.loc[gated["ticker"].eq("DIRECT"), "trade_status"].iloc[0] == "Research"


def test_validated_generic_family_is_prospective_not_production() -> None:
    validation = pd.DataFrame(
        [
            {
                "strategy": "iron_condor",
                "release_status": "VALIDATED",
                "scope": "all_sectors",
                "scope_value": "all",
                "clustered_pf_p05": 1.30,
                "holm_adjusted_joint_p": 0.04,
            }
        ]
    )

    registry = build_strategy_registry(
        payoff_summary={"status": "NO_VALIDATED_LANES"},
        payoff_groups=pd.DataFrame(),
        confidence_summary={"family_validation": {}},
        strategy_validation=validation,
    ).set_index("strategy_key")

    row = registry.loc["iron_condor"]
    assert row["strategy_validation_status"] == "VALIDATED"
    assert row["pipeline_status"] == "PROSPECTIVE"
    assert not row["execution_authorized"]

    gated = apply_strategy_registry_gate(
        pd.DataFrame(
            [{"ticker": "AAA", "strategy_registry_key": "iron_condor", "trade_status": "Research"}]
        ),
        registry.reset_index(),
    )
    assert gated.iloc[0]["strategy_validation_status"] == "VALIDATED"
    assert gated.iloc[0]["strategy_validation_clustered_pf_p05"] == 1.30