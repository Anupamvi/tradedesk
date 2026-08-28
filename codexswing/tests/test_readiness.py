from codexswing.research.readiness import evaluate_promotion


def _metrics():
    return {
        "closed_count": 25,
        "effective_nonoverlapping_trade_count": 12,
        "mean_net_pnl_dollars": 30,
        "bootstrap_2_5_percent_mean_net_pnl_dollars": 5,
        "profit_factor": 1.4,
        "wilson_95_lower_bound": 0.45,
        "validation_pass": True,
        "parameter_stability_pass": True,
    }


def _option():
    return {
        "modeled_expected_pnl_dollars": 25,
        "maximum_leg_spread_pct": 0.12,
        "minimum_open_interest": 500,
        "minimum_volume": 50,
        "maximum_loss_dollars": 500,
        "fresh_regular_session_quote": True,
    }


def _portfolio(conflict=False):
    return {
        "accounts": [
            {
                "balances": {"liquidationValue": 200_000, "buyingPower": 50_000},
                "positions": [],
            }
        ],
        "workingOrders": [
            {"legs": [{"underlyingSymbol": "XYZ"}]}
        ] if conflict else [],
    }


def test_all_gates_promote_to_manual_ready_but_never_authorize_order() -> None:
    result = evaluate_promotion(
        ticker="XYZ",
        discovered=True,
        backtest_metrics=_metrics(),
        option=_option(),
        portfolio=_portfolio(),
    )
    assert result["stage"] == "MANUAL_READY"
    assert result["is_executable_by_user"] is True
    assert result["broker_order_authorized"] is False
    assert result["broker_order_submitted"] is False


def test_working_order_conflict_stops_at_current_contract() -> None:
    result = evaluate_promotion(
        ticker="XYZ",
        discovered=True,
        backtest_metrics=_metrics(),
        option=_option(),
        portfolio=_portfolio(conflict=True),
    )
    assert result["stage"] == "CURRENT_CONTRACT_PASS"
    assert "working Schwab order" in " ".join(result["blockers"])


def test_positive_but_uncertain_history_can_only_be_one_contract_tactical() -> None:
    metrics = {
        "closed_count": 35,
        "effective_nonoverlapping_trade_count": 22,
        "mean_net_pnl_dollars": 14,
        "bootstrap_2_5_percent_mean_net_pnl_dollars": -5,
        "mean_maximum_risk_dollars": 250,
        "profit_factor": 1.8,
        "wilson_95_lower_bound": 0.30,
        "train_mean_net_pnl_dollars": 20,
        "train_profit_factor": 2.0,
        "validation_mean_net_pnl_dollars": 35,
        "validation_profit_factor": 3.0,
        "validation_pass": True,
        "parameter_stability_pass": True,
    }
    option = dict(_option(), maximum_loss_dollars=100)
    result = evaluate_promotion(
        ticker="XYZ",
        discovered=True,
        backtest_metrics=metrics,
        option=option,
        portfolio=_portfolio(),
    )
    assert result["stage"] == "TACTICAL_READY"
    assert result["is_manual_ready"] is False
    assert result["is_executable_by_user"] is True
    assert result["recommended_max_contracts"] == 1
    assert result["full_evidence_shortfalls"]


def test_tactical_trade_cannot_exceed_five_basis_points_of_nav() -> None:
    metrics = {
        "closed_count": 35,
        "effective_nonoverlapping_trade_count": 22,
        "mean_net_pnl_dollars": 14,
        "bootstrap_2_5_percent_mean_net_pnl_dollars": -5,
        "mean_maximum_risk_dollars": 250,
        "profit_factor": 1.8,
        "train_mean_net_pnl_dollars": 20,
        "train_profit_factor": 2.0,
        "validation_mean_net_pnl_dollars": 35,
        "validation_profit_factor": 3.0,
        "validation_pass": True,
        "parameter_stability_pass": True,
    }
    result = evaluate_promotion(
        ticker="XYZ",
        discovered=True,
        backtest_metrics=metrics,
        option=_option(),
        portfolio=_portfolio(),
    )
    assert result["stage"] == "CURRENT_CONTRACT_PASS"
    assert "0.05%" in " ".join(result["blockers"])
