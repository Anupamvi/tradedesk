"""Trade planning, stock-versus-options choice, scoring, and fail-closed status."""

from __future__ import annotations

import math
import statistics
from datetime import date
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from corat.constants import NO_POSITIVE_EDGE, REJECTED, SETUP_ONLY, TARGET_TRADE, WATCHLIST
from corat.models import HistoricalStats, OptionStructure, SetupSignal, TechnicalSnapshot, TradePlan


def _closest_below(price: float, values: Sequence[Optional[float]]) -> Optional[float]:
    eligible = [float(value) for value in values if value is not None and 0 < float(value) < price]
    return max(eligible) if eligible else None


def _closest_above(price: float, values: Sequence[Optional[float]]) -> Optional[float]:
    eligible = [float(value) for value in values if value is not None and float(value) > price]
    return min(eligible) if eligible else None


def _structural_invalidation(snapshot: TechnicalSnapshot, setup: SetupSignal) -> Optional[float]:
    """Choose the price level named by the frozen setup thesis.

    Risk is sized to thesis invalidation, not to an arbitrarily tighter stop
    selected merely to improve the displayed reward/risk ratio.
    """

    price = snapshot.price
    avwaps = [level.value for level in snapshot.avwaps]
    earnings_avwap = next(
        (level.value for level in snapshot.avwaps if level.anchor_reason == "most recent earnings"),
        None,
    )
    if setup.direction == "BULLISH":
        if setup.name == "BREAKOUT + CONFIRMATION":
            return _closest_below(price, [snapshot.prior_high_20d])
        if setup.name == "POST-EARNINGS DRIFT":
            return _closest_below(price, [earnings_avwap])
        if setup.name in {"RELATIVE-STRENGTH LEADER", "EMERGING SECTOR ROTATION"}:
            return _closest_below(price, [snapshot.ema20])
        if setup.name == "OVERSOLD REVERSAL":
            return _closest_below(price, [snapshot.support])
        return _closest_below(price, [snapshot.support, snapshot.ema20] + avwaps)
    if setup.name == "FAILED BREAKOUT / TREND BREAKDOWN":
        return _closest_above(price, [snapshot.prior_high_20d, snapshot.resistance, snapshot.ema20])
    return _closest_above(price, [snapshot.resistance, snapshot.ema20] + avwaps)


def build_stock_plan(
    snapshot: TechnicalSnapshot,
    setup: SetupSignal,
    portfolio_nav: Optional[float],
    risk_pct: float,
) -> Optional[TradePlan]:
    if setup.direction not in {"BULLISH", "BEARISH"} or not snapshot.atr14 or snapshot.atr14 <= 0:
        return None
    price = snapshot.price
    atr = snapshot.atr14
    if setup.direction == "BULLISH":
        entry_low = max(0.01, price - 0.20 * atr)
        entry_high = price
        risk_basis_price = entry_high
        structural = _structural_invalidation(snapshot, setup) or (price - 1.5 * atr)
        stop = structural - 0.25 * atr
        if stop >= risk_basis_price:
            stop = risk_basis_price - atr
        risk = risk_basis_price - stop
        actual_resistance = snapshot.resistance if snapshot.resistance and snapshot.resistance > price else None
        target1 = actual_resistance if actual_resistance and actual_resistance - price >= risk else price + 1.5 * atr
        target2 = max(price + 2.5 * atr, target1 + 0.5 * atr)
    else:
        entry_low = price
        entry_high = price + 0.20 * atr
        risk_basis_price = entry_low
        structural = _structural_invalidation(snapshot, setup) or (price + 1.5 * atr)
        stop = structural + 0.25 * atr
        if stop <= risk_basis_price:
            stop = risk_basis_price + atr
        risk = stop - risk_basis_price
        actual_support = snapshot.support if snapshot.support and snapshot.support < price else None
        target1 = actual_support if actual_support and price - actual_support >= risk else price - 1.5 * atr
        target2 = min(price - 2.5 * atr, target1 - 0.5 * atr)
    if risk <= 0:
        return None
    reward1 = abs(target1 - risk_basis_price)
    reward2 = abs(target2 - risk_basis_price)
    risk_dollars = portfolio_nav * risk_pct if portfolio_nav and portfolio_nav > 0 else None
    units = int(risk_dollars / risk) if risk_dollars is not None else None
    if units is not None and portfolio_nav and portfolio_nav > 0:
        # Default to unlevered gross stock exposure. The user can explicitly
        # choose leverage outside CORAT during manual portfolio review.
        units = min(units, int(float(portfolio_nav) / price))
    if units is not None and units < 1:
        units = None
    maximum_loss = units * risk if units is not None else None
    return TradePlan(
        vehicle="STOCK",
        entry_low=entry_low,
        entry_high=entry_high,
        trigger=setup.trigger,
        stop=stop,
        target_1=target1,
        target_2=target2,
        holding_sessions=10,
        reward_risk_1=reward1 / risk,
        reward_risk_2=reward2 / risk,
        risk_per_share=risk,
        portfolio_risk_dollars=risk_dollars,
        units=units,
        maximum_loss=maximum_loss,
        risk_basis_price=risk_basis_price,
    )


def choose_vehicle(
    stock_plan: TradePlan,
    option: OptionStructure,
    volatility: Mapping[str, Any],
    as_of: str = "",
    require_earnings_date: bool = True,
    stock_economics: Optional[Mapping[str, Any]] = None,
    option_economics: Optional[Mapping[str, Any]] = None,
    earnings_applicable: bool = True,
) -> Tuple[str, str]:
    if not option.valid:
        return "STOCK", "No exact two-sided option structure is available; use the underlying economics."
    next_earnings = str(volatility.get("next_earnings_date") or "")
    weeks_to_next = volatility.get("weeks_to_next_earnings")
    calendar_clear_through = str(volatility.get("earnings_calendar_clear_through") or "")
    required_clear_date = ""
    if as_of:
        try:
            required_clear_date = date.fromordinal(
                date.fromisoformat(as_of).toordinal() + int(stock_plan.holding_sessions * 1.8)
            ).isoformat()
        except ValueError:
            required_clear_date = ""
    calendar_clear = bool(calendar_clear_through and required_clear_date and calendar_clear_through >= required_clear_date)
    timing_known = bool(next_earnings) or calendar_clear or (weeks_to_next is not None and float(weeks_to_next) > 0)
    if earnings_applicable and require_earnings_date and not timing_known:
        return "STOCK", "Earnings timing is incomplete in the forward calendar for the intended hold, so an ordinary options swing is not used."
    if earnings_applicable and (_earnings_crossed(as_of, next_earnings, stock_plan.holding_sessions) or (
        not next_earnings
        and weeks_to_next is not None
        and float(weeks_to_next) > 0
        and float(weeks_to_next) * 5.0 <= stock_plan.holding_sessions
    )):
        return "STOCK", "The intended option hold crosses earnings; stock is compared instead."
    stock_ev = (stock_economics or {}).get("expected_return_on_capital")
    option_ev = (option_economics or {}).get("expected_return_on_max_loss")
    option_profit = (option_economics or {}).get("expected_profit_dollars")
    stock_profit = (stock_economics or {}).get("expected_profit_per_share")
    if option_profit is not None and float(option_profit) > 0 and (
        stock_profit is None
        or float(stock_profit) <= 0
        or (option_ev is not None and stock_ev is not None and float(option_ev) > float(stock_ev))
    ):
        return "OPTIONS", "The exact option structure has the stronger modeled expected return per dollar of capital at risk after quoted friction and commissions."
    if stock_profit is not None and float(stock_profit) > 0:
        return "STOCK", "The underlying has the stronger modeled expected return per dollar of capital required."
    if option_profit is not None and float(option_profit) > 0:
        return "OPTIONS", "The exact option structure has positive modeled expected profit after quoted friction and commissions."
    return "NO TRADE", "Neither the stock plan nor the exact option structure has positive modeled expected profit."


def model_stock_economics(
    snapshot: TechnicalSnapshot,
    stock_plan: Optional[TradePlan],
    history: HistoricalStats,
    evaluation_start: int = 0,
) -> Dict[str, Any]:
    start = max(0, int(evaluation_start))
    returns = [float(value) for value in history.primary_returns[start:]]
    paths = [[float(value) for value in path] for path in history.primary_paths[start:]]
    adverse_paths = [[float(value) for value in path] for path in history.primary_adverse_paths[start:]]
    favorable_paths = [[float(value) for value in path] for path in history.primary_favorable_paths[start:]]
    base: Dict[str, Any] = {
        "status": "DATA UNAVAILABLE",
        "method": (
            "Same-ticker, same-setup empirical paths using the displayed stop, first target, and holding horizon"
            + ("; evaluated on the recent holdout slice aligned with option validation." if start else ".")
        ),
        "model_sample_size": len(returns),
        "modeled_pop": None,
        "expected_profit_per_share": None,
        "median_profit_per_share": None,
        "average_winner_per_share": None,
        "average_loser_per_share": None,
        "profit_factor": None,
        "expected_return_on_invalidation_risk": None,
        "expected_return_on_capital": None,
        "expected_position_profit": None,
        "standard_error_per_share": None,
        "expected_profit_lower_95_per_share": None,
        "expected_profit_upper_95_per_share": None,
    }
    if stock_plan is None or not returns:
        return base
    basis = float(stock_plan.risk_basis_price or snapshot.price)
    stop_return = stock_plan.risk_per_share / basis
    target_return = abs(stock_plan.target_1 - basis) / basis
    profits = []
    for index, terminal_return in enumerate(returns):
        path = paths[index] if index < len(paths) else [terminal_return]
        adverse_path = adverse_paths[index] if index < len(adverse_paths) else path
        favorable_path = favorable_paths[index] if index < len(favorable_paths) else path
        exit_return = terminal_return
        for session_index, path_return in enumerate(path):
            adverse = adverse_path[session_index] if session_index < len(adverse_path) else path_return
            favorable = favorable_path[session_index] if session_index < len(favorable_path) else path_return
            # If both levels traded during the same daily bar, the sequence is
            # unknowable from EOD data. Charge the stop first rather than grant
            # a favorable target fill with look-ahead.
            if adverse <= -stop_return:
                exit_return = -stop_return
                break
            if favorable >= target_return:
                exit_return = target_return
                break
        profits.append(exit_return * basis)
    winners = [value for value in profits if value > 0]
    losers = [value for value in profits if value < 0]
    ordered = sorted(profits)
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2.0
    expected = sum(profits) / len(profits)
    standard_error = statistics.stdev(profits) / math.sqrt(len(profits)) if len(profits) >= 2 else None
    lower_95 = expected - 1.96 * standard_error if standard_error is not None else None
    upper_95 = expected + 1.96 * standard_error if standard_error is not None else None
    gross_wins = sum(winners)
    gross_losses = abs(sum(losers))
    base.update(
        {
            "status": "AVAILABLE",
            "modeled_pop": len(winners) / float(len(profits)),
            "expected_profit_per_share": expected,
            "median_profit_per_share": median,
            "average_winner_per_share": sum(winners) / len(winners) if winners else None,
            "average_loser_per_share": sum(losers) / len(losers) if losers else None,
            "profit_factor": gross_wins / gross_losses if gross_losses > 0 else (float("inf") if gross_wins > 0 else None),
            "expected_return_on_invalidation_risk": expected / stock_plan.risk_per_share,
            "expected_return_on_capital": expected / basis,
            "expected_position_profit": expected * stock_plan.units if stock_plan.units else None,
            "standard_error_per_share": standard_error,
            "expected_profit_lower_95_per_share": lower_95,
            "expected_profit_upper_95_per_share": upper_95,
        }
    )
    return base


def _regime_alignment(regime_label: str, direction: str) -> float:
    bullish = direction == "BULLISH"
    if regime_label in {"STRONG RISK-ON TREND", "WEAK RISK-ON"}:
        return 1.0 if bullish else 0.3
    if regime_label in {"RISK-OFF", "HIGH-VOLATILITY LIQUIDATION"}:
        return 1.0 if not bullish else 0.2
    if regime_label == "ROTATION":
        return 0.8
    return 0.6


def _earnings_crossed(as_of: str, next_earnings: str, holding_sessions: int) -> bool:
    if not as_of or not next_earnings:
        return False
    try:
        calendar_days = (date.fromisoformat(next_earnings) - date.fromisoformat(as_of)).days
    except ValueError:
        return False
    return 0 <= calendar_days <= int(holding_sessions * 1.6)


def score_candidate(
    snapshot: TechnicalSnapshot,
    setup: SetupSignal,
    stock_plan: Optional[TradePlan],
    option: OptionStructure,
    vehicle: str,
    volatility: Mapping[str, Any],
    context: Mapping[str, Any],
    history: HistoricalStats,
    sector: Mapping[str, Any],
    regime_label: str,
    minimum_stock_price: float,
    minimum_average_dollar_volume: float,
    minimum_reward_risk: float,
    actionability_score: int,
    require_catalyst: bool,
    require_historical: bool,
    require_earnings_for_options: bool,
    current_price_repriced: bool,
    trade_economics: Optional[Mapping[str, Any]] = None,
    earnings_applicable: bool = True,
) -> Dict[str, Any]:
    hard_rejections = []
    blockers = []
    notes = []
    if setup.direction not in {"BULLISH", "BEARISH"}:
        hard_rejections.append("No qualifying underlying setup")
    if snapshot.price <= 0:
        hard_rejections.append("Current underlying price is unavailable")
    average_dollar_volume = snapshot.average_dollar_volume_20d or 0.0
    if average_dollar_volume <= 0:
        hard_rejections.append("Underlying liquidity data is unavailable")
    elif average_dollar_volume < minimum_average_dollar_volume:
        notes.append("Underlying dollar volume is below the configured preference; review displayed liquidity.")
    if stock_plan is None:
        hard_rejections.append("Technical invalidation and risk plan could not be defined")
    elif stock_plan.reward_risk_2 < minimum_reward_risk:
        notes.append("Displayed target asymmetry is below the configured preference; expected profit remains the decision metric.")
    next_earnings = str(volatility.get("next_earnings_date") or "")
    earnings_crossed = bool(earnings_applicable and stock_plan and _earnings_crossed(snapshot.as_of, next_earnings, stock_plan.holding_sessions))
    if vehicle == "OPTIONS":
        if not option.valid:
            hard_rejections.append("Selected option structure failed execution gates")
        weeks_to_next = volatility.get("weeks_to_next_earnings")
        calendar_clear_through = str(volatility.get("earnings_calendar_clear_through") or "")
        required_clear_date = ""
        try:
            required_clear_date = date.fromordinal(
                date.fromisoformat(snapshot.as_of).toordinal() + int(stock_plan.holding_sessions * 1.8)
            ).isoformat() if stock_plan else ""
        except ValueError:
            required_clear_date = ""
        calendar_clear = bool(calendar_clear_through and required_clear_date and calendar_clear_through >= required_clear_date)
        if earnings_applicable and require_earnings_for_options and not next_earnings and not calendar_clear and not (weeks_to_next is not None and float(weeks_to_next) > 0):
            hard_rejections.append("Earnings timing unavailable for ordinary options swing")
        if earnings_crossed:
            hard_rejections.append("Ordinary options holding period crosses earnings")
    aligned_catalysts = [
        row for row in context.get("actionable_catalysts") or []
        if str(row.get("direction") or "").upper() == setup.direction
    ]
    if require_catalyst and not aligned_catalysts:
        notes.append("No current high-credibility directional catalyst was found; the trade rests on the stated technical setup and measured history.")
    if require_historical and not history.reliable:
        notes.append("Historical sample is below the configured reliability preference; sample size is shown with POP and expected profit.")
    if not current_price_repriced:
        notes.append("Optional Schwab quote was unavailable; ORATS as-of prices are used. This does not block target selection.")
    economics = dict(trade_economics or {})
    modeled_pop = economics.get("modeled_pop")
    expected_profit = economics.get("expected_profit_dollars")
    if expected_profit is None:
        expected_profit = economics.get("expected_profit_per_share")
    if modeled_pop is None or expected_profit is None:
        blockers.append("POP and expected-profit evidence are unavailable for the selected vehicle")
    elif float(expected_profit) <= 0:
        blockers.append("Modeled expected profit is not positive")
    history_adjustment = 2.0 if history.reliable and (history.expectancy or 0) > 0 else -3.0 if history.reliable else 0.0
    technical_score = max(0.0, min(20.0, 20.0 * setup.strength + history_adjustment))
    directional_strength = float((context.get("catalyst_strength_by_direction") or {}).get(setup.direction) or 0.0)
    catalyst_score = min(15.0, 15.0 * directional_strength)
    rs20 = snapshot.return_20d or 0.0
    sector_score = 5.0
    if str(sector.get("state") or "") in {"ACCELERATING LEADER", "EMERGING LEADER"}:
        sector_score += 4.0
    if str(sector.get("state") or "") == "DETERIORATING":
        sector_score -= 3.0
    sector_score += min(6.0, max(0.0, rs20 * 50.0))
    sector_score = max(0.0, min(15.0, sector_score))
    volatility_score = 0.0
    if volatility.get("status") == "AVAILABLE":
        volatility_score += 5.0
    if volatility.get("orats_forecast_realized_20d_pct") is not None:
        volatility_score += 3.0
    if option.valid:
        volatility_score += 4.0
        if option.theoretical_edge is not None and option.theoretical_edge >= 0:
            volatility_score += 3.0
    volatility_score = min(15.0, volatility_score)
    rr = stock_plan.reward_risk_2 if stock_plan else 0.0
    rr_score = min(15.0, max(0.0, rr / 3.0 * 15.0))
    liquidity_score = 10.0 if average_dollar_volume >= minimum_average_dollar_volume * 4 else 8.0 if average_dollar_volume >= minimum_average_dollar_volume * 2 else 6.0
    if vehicle == "OPTIONS" and not option.valid:
        liquidity_score = min(liquidity_score, 5.0)
    regime_score = 5.0 * _regime_alignment(regime_label, setup.direction)
    x_flow_score = min(5.0, 2.5 * float(context.get("x_strength") or 0.0) + 2.5 * float(context.get("flow_strength") or 0.0))
    components = {
        "price_technical_structure": round(technical_score, 2),
        "catalyst": round(catalyst_score, 2),
        "relative_strength_sector": round(sector_score, 2),
        "volatility_options_edge": round(volatility_score, 2),
        "risk_reward": round(rr_score, 2),
        "liquidity_execution": round(liquidity_score, 2),
        "market_regime_alignment": round(regime_score, 2),
        "x_flow_confirmation": round(x_flow_score, 2),
    }
    score = int(round(sum(components.values())))
    _ = actionability_score  # retained for config compatibility; score ranks but never authorizes.
    if hard_rejections:
        status = REJECTED
    elif not blockers and setup.triggered:
        status = TARGET_TRADE
    elif not blockers:
        status = SETUP_ONLY
    elif setup.name == "NO QUALIFYING SETUP":
        status = WATCHLIST
    else:
        status = NO_POSITIVE_EDGE
    confidence = "LOW"
    if score >= 85 and history.reliable and (history.expectancy or 0) > 0 and aligned_catalysts:
        confidence = "HIGH"
    elif score >= 70 and (history.sample_size >= 10 or aligned_catalysts):
        confidence = "MEDIUM"
    return {
        "score": score,
        "components": components,
        "status": status,
        "confidence": confidence,
        "hard_rejections": hard_rejections,
        "blockers": blockers,
        "notes": notes,
        "earnings_crossed": earnings_crossed,
    }
