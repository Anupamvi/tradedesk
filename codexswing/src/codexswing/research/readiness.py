"""Explicit promotion states for manually submitted trade candidates."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


PROMOTION_STATES = (
    "DISCOVERED",
    "BACKTEST_PASS",
    "CURRENT_CONTRACT_PASS",
    "PORTFOLIO_PASS",
    "TACTICAL_READY",
    "MANUAL_READY",
)


def _number(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def historical_gate(metrics: Optional[Mapping[str, Any]]) -> Tuple[bool, Sequence[str]]:
    if not metrics:
        return False, ("same-structure ORATS holdout evidence is unavailable",)
    reasons = []
    closed = int(_number(metrics.get("closed_count")) or 0)
    effective = int(_number(metrics.get("effective_nonoverlapping_trade_count")) or 0)
    mean_pnl = _number(metrics.get("mean_net_pnl_dollars"))
    bootstrap = _number(metrics.get("bootstrap_2_5_percent_mean_net_pnl_dollars"))
    profit_factor = _number(metrics.get("profit_factor"))
    wilson = _number(metrics.get("wilson_95_lower_bound"))
    if closed < 20:
        reasons.append("holdout has fewer than 20 closed trades")
    if effective < 8:
        reasons.append("holdout has fewer than 8 effective non-overlapping trades")
    if mean_pnl is None or mean_pnl <= 0:
        reasons.append("holdout mean P/L is not positive after costs")
    if bootstrap is None or bootstrap <= 0:
        reasons.append("cluster-bootstrap lower mean P/L is not positive")
    if profit_factor is None or profit_factor < 1.10:
        reasons.append("holdout profit factor is below 1.10")
    if wilson is None or wilson < 0.40:
        reasons.append("holdout Wilson POP lower bound is below 40%")
    if metrics.get("validation_pass") is not True:
        reasons.append("chronological validation mean P/L did not pass")
    if metrics.get("parameter_stability_pass") is not True:
        reasons.append("fixed-parameter temporal stability did not pass")
    return not reasons, tuple(reasons)


def tactical_historical_gate(
    metrics: Optional[Mapping[str, Any]],
    current_maximum_loss_dollars: Optional[float] = None,
) -> Tuple[bool, Sequence[str]]:
    """Evidence floor for a one-contract exploratory trade, never full promotion."""

    if not metrics:
        return False, ("same-structure ORATS holdout evidence is unavailable",)
    reasons = []
    closed = int(_number(metrics.get("closed_count")) or 0)
    effective = int(_number(metrics.get("effective_nonoverlapping_trade_count")) or 0)
    mean_pnl = _number(metrics.get("mean_net_pnl_dollars"))
    bootstrap = _number(metrics.get("bootstrap_2_5_percent_mean_net_pnl_dollars"))
    current_risk = _number(current_maximum_loss_dollars)
    profit_factor = _number(metrics.get("profit_factor"))
    train_mean = _number(metrics.get("train_mean_net_pnl_dollars"))
    train_pf = _number(metrics.get("train_profit_factor"))
    validation_mean = _number(metrics.get("validation_mean_net_pnl_dollars"))
    validation_pf = _number(metrics.get("validation_profit_factor"))
    if closed < 30:
        reasons.append("holdout has fewer than 30 closed trades for tactical sizing")
    if effective < 15:
        reasons.append("holdout has fewer than 15 effective non-overlapping trades")
    if mean_pnl is None or mean_pnl <= 0:
        reasons.append("holdout mean P/L is not positive after costs")
    if profit_factor is None or profit_factor < 1.20:
        reasons.append("holdout profit factor is below 1.20")
    if train_mean is None or train_mean <= 0 or train_pf is None or train_pf < 1.10:
        reasons.append("train expectancy/profit factor is not positive and stable")
    if validation_mean is None or validation_mean <= 0 or validation_pf is None or validation_pf < 1.20:
        reasons.append("validation expectancy/profit factor is not positive and stable")
    if current_risk is None or current_risk <= 0:
        reasons.append("current defined risk is unavailable for uncertainty normalization")
    elif bootstrap is None or bootstrap < -0.05 * current_risk:
        reasons.append("bootstrap lower mean loss exceeds 5% of current defined risk")
    if metrics.get("parameter_stability_pass") is not True:
        reasons.append("fixed-parameter validation/holdout stability did not pass")
    return not reasons, tuple(reasons)


def current_contract_gate(option: Optional[Mapping[str, Any]]) -> Tuple[bool, Sequence[str]]:
    if not option:
        return False, ("no current defined-risk contract passed selection",)
    reasons = []
    expected = _number(option.get("modeled_expected_pnl_dollars"))
    width = _number(option.get("maximum_leg_spread_pct"))
    open_interest = int(_number(option.get("minimum_open_interest")) or 0)
    volume = int(_number(option.get("minimum_volume")) or 0)
    max_loss = _number(option.get("maximum_loss_dollars"))
    quote_fresh = bool(option.get("fresh_regular_session_quote"))
    if expected is None or expected <= 0:
        reasons.append("current-contract modeled EV is not positive after costs")
    if width is None or width > 0.25:
        reasons.append("a leg quote is wider than 25% of midpoint")
    if open_interest < 100:
        reasons.append("a leg has open interest below 100")
    if volume < 10:
        reasons.append("a leg has volume below 10")
    if max_loss is None or max_loss <= 0:
        reasons.append("defined maximum loss is unavailable")
    if not quote_fresh:
        reasons.append("exact Schwab contract quote is not fresh regular-session evidence")
    return not reasons, tuple(reasons)


def portfolio_gate(
    portfolio: Optional[Mapping[str, Any]],
    ticker: str,
    maximum_loss_dollars: Optional[float],
) -> Tuple[bool, Sequence[str], Mapping[str, Any]]:
    if not portfolio:
        return False, ("Schwab portfolio snapshot is unavailable",), {}
    ticker = ticker.upper()
    accounts = portfolio.get("accounts")
    orders = portfolio.get("workingOrders")
    account_rows = accounts if isinstance(accounts, list) else []
    order_rows = orders if isinstance(orders, list) else []
    liquidation = 0.0
    available = 0.0
    ticker_market_value = 0.0
    option_position_conflict = False
    for account in account_rows:
        if not isinstance(account, Mapping):
            continue
        balances = account.get("balances")
        balance_values = balances if isinstance(balances, Mapping) else {}
        liquidation += max(_number(balance_values.get("liquidationValue")) or 0.0, 0.0)
        available += max(
            _number(balance_values.get("buyingPower"))
            or _number(balance_values.get("availableFunds"))
            or _number(balance_values.get("cashAvailableForTrading"))
            or 0.0,
            0.0,
        )
        for position in account.get("positions") or ():
            if not isinstance(position, Mapping):
                continue
            symbol = str(position.get("symbol") or "").upper()
            underlying = str(position.get("underlyingSymbol") or "").upper()
            if symbol == ticker or underlying == ticker:
                ticker_market_value += abs(_number(position.get("marketValue")) or 0.0)
                if str(position.get("assetType") or "").upper() == "OPTION":
                    option_position_conflict = True
    working_conflict = False
    for order in order_rows:
        if not isinstance(order, Mapping):
            continue
        for leg in order.get("legs") or ():
            if not isinstance(leg, Mapping):
                continue
            symbol = str(leg.get("symbol") or "").upper()
            underlying = str(leg.get("underlyingSymbol") or "").upper()
            if symbol == ticker or underlying == ticker or symbol.startswith(ticker + " "):
                working_conflict = True
    max_loss = float(maximum_loss_dollars or 0.0)
    concentration = ticker_market_value / liquidation if liquidation > 0 else 0.0
    risk_cap = min(2_000.0, liquidation * 0.01) if liquidation > 0 else 0.0
    reasons = []
    if liquidation <= 0:
        reasons.append("positive Schwab liquidation value is unavailable")
    if available < max_loss * 1.5:
        reasons.append("available funds are below 1.5x maximum loss")
    if risk_cap <= 0 or max_loss > risk_cap:
        reasons.append("maximum loss exceeds the smaller of $2,000 and 1% of account value")
    if concentration >= 0.10:
        reasons.append("existing ticker exposure is at least 10% of account value")
    if option_position_conflict:
        reasons.append("an existing option position uses the same underlying")
    if working_conflict:
        reasons.append("a working Schwab order uses the same underlying")
    diagnostics = {
        "liquidation_value_dollars": liquidation,
        "available_funds_dollars": available,
        "ticker_market_value_dollars": ticker_market_value,
        "ticker_concentration": concentration,
        "candidate_risk_cap_dollars": risk_cap,
        "working_order_conflict": working_conflict,
        "option_position_conflict": option_position_conflict,
    }
    return not reasons, tuple(reasons), diagnostics


def tactical_risk_gate(
    portfolio_diagnostics: Mapping[str, Any],
    maximum_loss_dollars: Optional[float],
) -> Tuple[bool, Sequence[str], float]:
    liquidation = _number(portfolio_diagnostics.get("liquidation_value_dollars")) or 0.0
    maximum_loss = float(maximum_loss_dollars or 0.0)
    risk_cap = min(500.0, liquidation * 0.0005) if liquidation > 0 else 0.0
    reasons = []
    if risk_cap <= 0:
        reasons.append("positive account value is unavailable for tactical sizing")
    if maximum_loss <= 0:
        reasons.append("defined maximum loss is unavailable")
    elif maximum_loss > risk_cap:
        reasons.append("one-contract maximum loss exceeds 0.05% NAV/$500 tactical cap")
    return not reasons, tuple(reasons), risk_cap


def evaluate_promotion(
    *,
    ticker: str,
    discovered: bool,
    backtest_metrics: Optional[Mapping[str, Any]],
    option: Optional[Mapping[str, Any]],
    portfolio: Optional[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Advance only through contiguous gates; never imply an order was sent."""

    max_loss = _number(option.get("maximum_loss_dollars")) if option else None
    history_pass, history_reasons = historical_gate(backtest_metrics)
    tactical_history_pass, tactical_history_reasons = tactical_historical_gate(
        backtest_metrics, max_loss
    )
    contract_pass, contract_reasons = current_contract_gate(option)
    portfolio_pass, portfolio_reasons, diagnostics = portfolio_gate(
        portfolio, ticker, max_loss
    )
    tactical_risk_pass, tactical_risk_reasons, tactical_risk_cap = tactical_risk_gate(
        diagnostics, max_loss
    )
    diagnostics = dict(diagnostics)
    diagnostics["tactical_risk_cap_dollars"] = tactical_risk_cap
    stage = "DISCOVERED"
    if discovered and history_pass:
        stage = "BACKTEST_PASS"
        if contract_pass:
            stage = "CURRENT_CONTRACT_PASS"
            if portfolio_pass:
                stage = "PORTFOLIO_PASS"
                stage = "MANUAL_READY"
    elif (
        discovered
        and tactical_history_pass
        and contract_pass
        and portfolio_pass
        and tactical_risk_pass
    ):
        stage = "TACTICAL_READY"
    elif discovered and contract_pass:
        stage = "CURRENT_CONTRACT_PASS"
    blockers = []
    if stage in {"MANUAL_READY", "TACTICAL_READY"}:
        pass
    elif not discovered:
        blockers.append("candidate failed discovery")
    elif not history_pass and not tactical_history_pass:
        blockers.extend(tactical_history_reasons)
    elif not contract_pass:
        blockers.extend(contract_reasons)
    elif not portfolio_pass:
        blockers.extend(portfolio_reasons)
    elif not tactical_risk_pass:
        blockers.extend(tactical_risk_reasons)
    return {
        "stage": stage,
        "is_manual_ready": stage == "MANUAL_READY",
        "is_tactical_ready": stage == "TACTICAL_READY",
        "is_executable_by_user": stage in {"MANUAL_READY", "TACTICAL_READY"},
        "evidence_tier": (
            "FULL_EVIDENCE"
            if stage == "MANUAL_READY"
            else "EXPLORATORY_TACTICAL"
            if stage == "TACTICAL_READY"
            else "NOT_ACTIONABLE"
        ),
        "recommended_max_contracts": 1 if stage == "TACTICAL_READY" else None,
        "broker_order_authorized": False,
        "broker_order_submitted": False,
        "blockers": blockers,
        "full_evidence_shortfalls": list(history_reasons) if not history_pass else [],
        "tactical_evidence_shortfalls": (
            list(tactical_history_reasons) if not tactical_history_pass else []
        ),
        "gates": {
            "discovered": discovered,
            "backtest_pass": history_pass,
            "tactical_backtest_pass": tactical_history_pass,
            "current_contract_pass": contract_pass,
            "portfolio_pass": portfolio_pass,
            "tactical_risk_pass": tactical_risk_pass,
        },
        "portfolio_diagnostics": diagnostics,
    }
