"""Exact ORATS option-chain selection and conservative structure math."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import replace
import statistics
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from corat.models import OptionLeg, OptionStructure


def _float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _int(value: Any) -> int:
    return max(0, int(_float(value, 0.0)))


def _optional_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    return _int(value)


def _spread_pct(bid: float, ask: float) -> Optional[float]:
    midpoint = (bid + ask) / 2.0
    if bid < 0 or ask <= 0 or ask < bid or midpoint <= 0:
        return None
    return (ask - bid) / midpoint


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _black_scholes_value(
    spot: float,
    strike: float,
    years: float,
    volatility: float,
    option_type: str,
    risk_free_rate: float = 0.0,
    dividend_yield: float = 0.0,
) -> float:
    """Black-Scholes change model used only for a disclosed scenario mark.

    CORAT calibrates the current mark to the observed quote midpoint, then
    applies the Black-Scholes change in price/time/volatility using ORATS rates
    and dividend yield when supplied. The result is not ORATS POP.
    """

    if years <= 0 or volatility <= 0:
        return max(0.0, spot - strike) if option_type == "CALL" else max(0.0, strike - spot)
    root_time = math.sqrt(years)
    d1 = (
        math.log(max(spot, 1e-12) / strike)
        + (risk_free_rate - dividend_yield + 0.5 * volatility * volatility) * years
    ) / (volatility * root_time)
    d2 = d1 - volatility * root_time
    discounted_spot = spot * math.exp(-dividend_yield * years)
    discounted_strike = strike * math.exp(-risk_free_rate * years)
    if option_type == "CALL":
        return discounted_spot * _normal_cdf(d1) - discounted_strike * _normal_cdf(d2)
    return discounted_strike * _normal_cdf(-d2) - discounted_spot * _normal_cdf(-d1)


def _profit_factor(values: Sequence[float]) -> Optional[float]:
    gains = sum(value for value in values if value > 0)
    losses = abs(sum(value for value in values if value < 0))
    if losses == 0:
        return float("inf") if gains > 0 else None
    return gains / losses


def model_option_economics(
    structure: OptionStructure,
    spot: float,
    direction: str,
    holding_sessions: int,
    scenario_returns: Sequence[float],
    commission_per_contract: float = 0.65,
    scenario_paths: Sequence[Sequence[float]] = (),
    stop_return: Optional[float] = None,
    target_return: Optional[float] = None,
    scenario_adverse_paths: Sequence[Sequence[float]] = (),
    scenario_favorable_paths: Sequence[Sequence[float]] = (),
    iv_shift_points: float = 0.0,
    risk_free_rate: float = 0.0,
    dividend_yield: float = 0.0,
    entry_price_override: Optional[float] = None,
) -> Dict[str, Any]:
    """Estimate trade POP and P/L from same-setup historical return scenarios.

    `scenario_returns` are direction-adjusted forward underlying returns from
    the leakage-safe analogue engine. Each return is applied to today's spot,
    the exact legs are marked after the intended holding period with ORATS
    smoothed IV held constant, and current observed spread friction plus round-
    trip commissions are deducted. This is CORAT model output, not ORATS POP.
    """

    method = (
        "Same-ticker, same-setup empirical paths using the displayed underlying stop, first target, and holding horizon; exact option legs marked "
        "at each modeled exit with the disclosed ORATS forecast IV shift, ORATS rate/dividend inputs when available, "
        "observed-midpoint calibration, current quoted entry/exit friction, and round-trip commissions."
    )
    base = {
        "status": "DATA UNAVAILABLE",
        "method": method,
        "model_sample_size": len(scenario_returns),
        "modeled_pop": None,
        "expected_profit_dollars": None,
        "median_profit_dollars": None,
        "average_winner_dollars": None,
        "average_loser_dollars": None,
        "profit_factor": None,
        "expected_return_on_max_loss": None,
        "estimated_exit_slippage": None,
        "round_trip_commission": None,
        "standard_error_dollars": None,
        "expected_profit_lower_95_dollars": None,
        "expected_profit_upper_95_dollars": None,
        "expected_return_lower_95_on_max_loss": None,
        "exit_iv_pct": None,
        "iv_shift_points": float(iv_shift_points),
        "expected_entry_used": None,
        "modeled_maximum_loss": None,
    }
    if (
        not structure.valid
        or structure.expected_entry is None
        or structure.maximum_loss is None
        or not structure.legs
        or not scenario_returns
        or spot <= 0
    ):
        return base
    raw_iv = structure.implied_volatility
    if raw_iv is None or raw_iv <= 0:
        return base
    current_iv_pct = float(raw_iv) if float(raw_iv) > 3.0 else float(raw_iv) * 100.0
    exit_iv_pct = max(0.01, current_iv_pct + float(iv_shift_points))
    volatility = current_iv_pct / 100.0
    exit_volatility = exit_iv_pct / 100.0
    exit_slippage = 0.75 * sum(max(0.0, leg.ask - leg.bid) / 2.0 for leg in structure.legs)
    commission = max(0.0, float(commission_per_contract)) * len(structure.legs) * 2.0
    entry_price = float(entry_price_override) if entry_price_override is not None else float(structure.expected_entry)
    width = abs(max(leg.strike for leg in structure.legs) - min(leg.strike for leg in structure.legs)) if len(structure.legs) > 1 else None
    if structure.debit_credit == "CREDIT" and width is not None:
        modeled_maximum_loss = max(0.01, (width - entry_price) * 100.0)
    elif structure.debit_credit == "DEBIT":
        modeled_maximum_loss = max(0.01, entry_price * 100.0)
    else:
        modeled_maximum_loss = float(structure.maximum_loss)
    profits: List[float] = []
    for index, terminal_return in enumerate(scenario_returns):
        directional_return = float(terminal_return)
        elapsed_sessions = holding_sessions
        path = scenario_paths[index] if index < len(scenario_paths) else ()
        adverse_path = scenario_adverse_paths[index] if index < len(scenario_adverse_paths) else path
        favorable_path = scenario_favorable_paths[index] if index < len(scenario_favorable_paths) else path
        for session_index, path_return in enumerate(path, start=1):
            adverse = adverse_path[session_index - 1] if session_index - 1 < len(adverse_path) else path_return
            favorable = favorable_path[session_index - 1] if session_index - 1 < len(favorable_path) else path_return
            if stop_return is not None and float(adverse) <= -float(stop_return):
                directional_return = -float(stop_return)
                elapsed_sessions = session_index
                break
            if target_return is not None and float(favorable) >= float(target_return):
                directional_return = float(target_return)
                elapsed_sessions = session_index
                break
        actual_return = float(directional_return) if direction == "BULLISH" else -float(directional_return)
        scenario_spot = max(0.01, spot * (1.0 + actual_return))
        remaining_days = max(0.0, float(structure.dte) - float(elapsed_sessions) * 365.0 / 252.0)
        remaining_years = remaining_days / 365.0
        fair_exit = 0.0
        for leg in structure.legs:
            sign = 1.0 if leg.action == "BUY" else -1.0
            current_bs = _black_scholes_value(
                spot,
                float(leg.strike),
                float(structure.dte) / 365.0,
                volatility,
                leg.option_type,
                risk_free_rate,
                dividend_yield,
            )
            exit_bs = _black_scholes_value(
                scenario_spot,
                float(leg.strike),
                remaining_years,
                exit_volatility,
                leg.option_type,
                risk_free_rate,
                dividend_yield,
            )
            # The ORATS smooth theoretical value is displayed as context but is
            # not treated as free, immediately realizable alpha. Scenario P/L
            # starts from the observed market midpoint and charges fills.
            current_anchor = (float(leg.bid) + float(leg.ask)) / 2.0
            intrinsic = max(0.0, scenario_spot - leg.strike) if leg.option_type == "CALL" else max(0.0, leg.strike - scenario_spot)
            calibrated_exit = max(intrinsic, current_anchor + (exit_bs - current_bs), 0.0)
            fair_exit += sign * calibrated_exit
        if width is not None:
            if structure.debit_credit == "CREDIT":
                fair_exit = max(-width, min(0.0, fair_exit))
            else:
                fair_exit = max(0.0, min(width, fair_exit))
        if structure.debit_credit == "CREDIT":
            close_debit = max(0.0, -fair_exit + exit_slippage)
            if width is not None:
                close_debit = min(width, close_debit)
            profits.append((entry_price - close_debit) * 100.0 - commission)
        else:
            exit_credit = max(0.0, fair_exit - exit_slippage)
            if width is not None:
                exit_credit = min(width, exit_credit)
            profits.append((exit_credit - entry_price) * 100.0 - commission)
    winners = [value for value in profits if value > 0]
    losers = [value for value in profits if value < 0]
    ordered = sorted(profits)
    midpoint = len(ordered) // 2
    median = ordered[midpoint] if len(ordered) % 2 else (ordered[midpoint - 1] + ordered[midpoint]) / 2.0
    expected = sum(profits) / len(profits)
    standard_error = statistics.stdev(profits) / math.sqrt(len(profits)) if len(profits) >= 2 else None
    lower_95 = expected - 1.96 * standard_error if standard_error is not None else None
    upper_95 = expected + 1.96 * standard_error if standard_error is not None else None
    base.update(
        {
            "status": "AVAILABLE",
            "modeled_pop": len(winners) / float(len(profits)),
            "expected_profit_dollars": expected,
            "median_profit_dollars": median,
            "average_winner_dollars": sum(winners) / len(winners) if winners else None,
            "average_loser_dollars": sum(losers) / len(losers) if losers else None,
            "profit_factor": _profit_factor(profits),
            "expected_return_on_max_loss": expected / modeled_maximum_loss,
            "estimated_exit_slippage": exit_slippage * 100.0,
            "round_trip_commission": commission,
            "standard_error_dollars": standard_error,
            "expected_profit_lower_95_dollars": lower_95,
            "expected_profit_upper_95_dollars": upper_95,
            "expected_return_lower_95_on_max_loss": lower_95 / modeled_maximum_loss if lower_95 is not None else None,
            "exit_iv_pct": exit_iv_pct,
            "iv_shift_points": float(iv_shift_points),
            "expected_entry_used": entry_price,
            "modeled_maximum_loss": modeled_maximum_loss,
        }
    )
    return base


def _leg(row: Mapping[str, Any], option_type: str, action: str) -> OptionLeg:
    call = option_type == "CALL"
    bid = _float(row.get("callBidPrice" if call else "putBidPrice"))
    ask = _float(row.get("callAskPrice" if call else "putAskPrice"))
    raw_delta = _float(row.get("delta"))
    delta = raw_delta if call else raw_delta - 1.0
    sign = 1.0 if action == "BUY" else -1.0
    return OptionLeg(
        action=action,
        option_type=option_type,
        strike=_float(row.get("strike")),
        expiration=str(row.get("expirDate") or "")[:10],
        quantity=1,
        bid=bid,
        ask=ask,
        theoretical_value=(
            _float(row.get("callValue" if call else "putValue"))
            if row.get("callValue" if call else "putValue") not in (None, "") else None
        ),
        delta=sign * delta,
        gamma=sign * _float(row.get("gamma")),
        theta=sign * _float(row.get("theta")),
        vega=sign * _float(row.get("vega")),
        open_interest=_int(row.get("callOpenInterest" if call else "putOpenInterest")),
        volume=_int(row.get("callVolume" if call else "putVolume")),
        spread_pct=_spread_pct(bid, ask),
        bid_size=_optional_int(row.get("callBidSize" if call else "putBidSize")),
        ask_size=_optional_int(row.get("callAskSize" if call else "putAskSize")),
    )


def _liquid(leg: OptionLeg, minimum_oi: int, minimum_volume: int, maximum_spread_pct: float) -> bool:
    # OI, volume, and quoted width remain visible and affect fill friction and
    # ranking rather than acting as arbitrary quotas. An explicitly empty
    # market (both displayed sides have zero size) is not executable.
    _ = (minimum_oi, minimum_volume, maximum_spread_pct)
    return bool(
        leg.bid > 0
        and leg.ask >= leg.bid
        and leg.spread_pct is not None
        and not (
            leg.bid_size is not None
            and leg.ask_size is not None
            and leg.bid_size <= 0
            and leg.ask_size <= 0
        )
    )


def _liquidity_penalty(legs: Sequence[OptionLeg]) -> float:
    """Continuous execution penalty with no OI/volume cliff."""

    penalty = 0.0
    for leg in legs:
        penalty += float(leg.spread_pct or 1.0)
        penalty += 1.0 / math.sqrt(1.0 + float(leg.open_interest + leg.volume))
        if leg.bid_size is not None and leg.ask_size is not None:
            penalty += 0.25 / math.sqrt(1.0 + float(leg.bid_size + leg.ask_size))
    return penalty


def _empty(reasons: Sequence[str]) -> OptionStructure:
    return OptionStructure(
        valid=False,
        strategy="NO LIQUID DEFINED-RISK OPTION",
        expiration="",
        dte=0,
        legs=[],
        debit_credit="",
        expected_entry=None,
        natural_entry=None,
        maximum_loss=None,
        maximum_gain=None,
        breakeven=None,
        reward_risk=None,
        delta=None,
        gamma=None,
        theta=None,
        vega=None,
        theta_holding_cost=None,
        orats_theoretical_value=None,
        theoretical_edge=None,
        implied_volatility=None,
        quote_trade_date="",
        quote_updated_at="",
        reasons=list(reasons),
    )


def choose_debit_spread(
    rows: Iterable[Mapping[str, Any]],
    direction: str,
    target_price: float,
    holding_sessions: int,
    minimum_oi: int,
    minimum_volume: int,
    maximum_spread_pct: float,
    scenario_returns: Sequence[float] = (),
    commission_per_contract: float = 0.65,
    scenario_paths: Sequence[Sequence[float]] = (),
    stop_return: Optional[float] = None,
    target_return: Optional[float] = None,
) -> OptionStructure:
    rows = [row for row in rows if 21 <= int(_float(row.get("dte"))) <= 75]
    if not rows:
        return _empty(["No ORATS strikes in the configured 21-75 DTE window."])
    desired_dte = min(60, max(28, holding_sessions * 4))
    by_expiration: Dict[Tuple[str, int], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        expiration = str(row.get("expirDate") or "")[:10]
        dte = int(_float(row.get("dte")))
        if expiration and dte:
            by_expiration[(expiration, dte)].append(row)
    if not by_expiration:
        return _empty(["ORATS chain rows lacked expiration metadata."])
    expiration, dte = min(by_expiration, key=lambda key: abs(key[1] - desired_dte))
    expiry_rows = sorted(by_expiration[(expiration, dte)], key=lambda row: _float(row.get("strike")))
    option_type = "CALL" if direction == "BULLISH" else "PUT"
    candidates: List[Tuple[float, Mapping[str, Any], Mapping[str, Any], OptionLeg, OptionLeg]] = []
    for long_row in expiry_rows:
        long_leg = _leg(long_row, option_type, "BUY")
        long_abs_delta = abs(long_leg.delta)
        if not (0.40 <= long_abs_delta <= 0.70):
            continue
        if not _liquid(long_leg, minimum_oi, minimum_volume, maximum_spread_pct):
            continue
        for short_row in expiry_rows:
            long_strike = _float(long_row.get("strike"))
            short_strike = _float(short_row.get("strike"))
            if direction == "BULLISH" and short_strike <= long_strike:
                continue
            if direction == "BEARISH" and short_strike >= long_strike:
                continue
            short_leg = _leg(short_row, option_type, "SELL")
            short_abs_delta = abs(short_leg.delta)
            if not (0.15 <= short_abs_delta <= 0.42):
                continue
            raw_short = OptionLeg(**dict(short_leg.__dict__, delta=-short_leg.delta, gamma=-short_leg.gamma, theta=-short_leg.theta, vega=-short_leg.vega))
            if not _liquid(raw_short, minimum_oi, minimum_volume, maximum_spread_pct):
                continue
            width = abs(short_strike - long_strike)
            if width <= 0:
                continue
            midpoint_long = (long_leg.bid + long_leg.ask) / 2.0
            midpoint_short = (raw_short.bid + raw_short.ask) / 2.0
            midpoint_debit = midpoint_long - midpoint_short
            natural_debit = long_leg.ask - raw_short.bid
            if midpoint_debit <= 0 or natural_debit <= 0 or natural_debit >= width:
                continue
            expected = midpoint_debit + 0.50 * (natural_debit - midpoint_debit)
            if expected <= 0 or expected >= width:
                continue
            target_distance = abs(short_strike - target_price) / max(1.0, target_price)
            quality = abs(long_abs_delta - 0.55) + abs(short_abs_delta - 0.28) + target_distance
            candidates.append((quality, long_row, short_row, long_leg, short_leg))
    if not candidates:
        return _empty([
            "No same-expiration debit spread had executable two-sided quotes and coherent debit math.",
            "Stock remains eligible if the underlying thesis independently qualifies.",
        ])
    if scenario_returns:
        modeled = []
        for candidate in candidates:
            quality, long_row_value, short_row_value, long_leg_value, short_leg_value = candidate
            structure = _build_debit_structure(
                long_row_value,
                short_row_value,
                long_leg_value,
                short_leg_value,
                direction,
                holding_sessions,
                expiration,
                dte,
            )
            economics = model_option_economics(
                structure,
                _float(long_row_value.get("stockPrice")),
                direction,
                holding_sessions,
                scenario_returns,
                commission_per_contract,
                scenario_paths,
                stop_return,
                target_return,
            )
            ev_ratio = economics.get("expected_return_on_max_loss")
            pop = economics.get("modeled_pop")
            modeled.append(
                (
                    0 if economics.get("expected_profit_dollars") is not None and float(economics["expected_profit_dollars"]) > 0 else 1,
                    -float(ev_ratio) if ev_ratio is not None else float("inf"),
                    -float(pop) if pop is not None else float("inf"),
                    quality,
                    candidate,
                )
            )
        _, _, _, _, selected = min(modeled, key=lambda item: item[:4])
        _, long_row, short_row, long_leg, short_leg = selected
    else:
        _, long_row, short_row, long_leg, short_leg = min(candidates, key=lambda item: item[0])
    return _build_debit_structure(
        long_row,
        short_row,
        long_leg,
        short_leg,
        direction,
        holding_sessions,
        expiration,
        dte,
    )


def _build_debit_structure(
    long_row: Mapping[str, Any],
    short_row: Mapping[str, Any],
    long_leg: OptionLeg,
    short_leg: OptionLeg,
    direction: str,
    holding_sessions: int,
    expiration: str,
    dte: int,
) -> OptionStructure:
    raw_short = OptionLeg(**dict(short_leg.__dict__, delta=-short_leg.delta, gamma=-short_leg.gamma, theta=-short_leg.theta, vega=-short_leg.vega))
    midpoint_debit = (long_leg.bid + long_leg.ask) / 2.0 - (raw_short.bid + raw_short.ask) / 2.0
    natural_debit = long_leg.ask - raw_short.bid
    expected = midpoint_debit + 0.50 * (natural_debit - midpoint_debit)
    width = abs(raw_short.strike - long_leg.strike)
    maximum_loss = expected * 100.0
    maximum_gain = (width - expected) * 100.0
    reward_risk = maximum_gain / maximum_loss if maximum_loss > 0 else None
    breakeven = long_leg.strike + expected if direction == "BULLISH" else long_leg.strike - expected
    theoretical = None
    if long_leg.theoretical_value is not None and raw_short.theoretical_value is not None:
        theoretical = long_leg.theoretical_value - raw_short.theoretical_value
    edge = theoretical - expected if theoretical is not None else None
    net_delta = long_leg.delta + short_leg.delta
    net_gamma = long_leg.gamma + short_leg.gamma
    net_theta = long_leg.theta + short_leg.theta
    net_vega = long_leg.vega + short_leg.vega
    smv = _float(long_row.get("smvVol")) or _float(short_row.get("smvVol"))
    if smv and abs(smv) <= 3:
        smv *= 100.0
    reasons: List[str] = ["Entry limit is modeled halfway from the quoted midpoint toward the natural debit; the natural fill is also stress-tested."]
    valid = True
    return OptionStructure(
        valid=valid,
        strategy="BULL CALL DEBIT SPREAD" if direction == "BULLISH" else "BEAR PUT DEBIT SPREAD",
        expiration=expiration,
        dte=dte,
        legs=[long_leg, short_leg],
        debit_credit="DEBIT",
        expected_entry=expected,
        natural_entry=natural_debit,
        maximum_loss=maximum_loss,
        maximum_gain=maximum_gain,
        breakeven=breakeven,
        reward_risk=reward_risk,
        delta=net_delta,
        gamma=net_gamma,
        theta=net_theta,
        vega=net_vega,
        theta_holding_cost=net_theta * holding_sessions * 100.0,
        orats_theoretical_value=theoretical,
        theoretical_edge=edge,
        implied_volatility=smv or None,
        quote_trade_date=str(long_row.get("tradeDate") or "")[:10],
        quote_updated_at=max(str(long_row.get("updatedAt") or ""), str(short_row.get("updatedAt") or "")),
        reasons=reasons,
        midpoint_entry=midpoint_debit,
        entry_fill_fraction=0.50,
    )


def _normalized_iv(row: Mapping[str, Any]) -> Optional[float]:
    value = _float(row.get("smvVol"))
    if value <= 0:
        return None
    return value * 100.0 if value <= 3.0 else value


def _build_long_structure(
    row: Mapping[str, Any],
    direction: str,
    holding_sessions: int,
) -> Optional[OptionStructure]:
    option_type = "CALL" if direction == "BULLISH" else "PUT"
    leg = _leg(row, option_type, "BUY")
    if leg.ask <= 0 or leg.ask < leg.bid:
        return None
    midpoint = (leg.bid + leg.ask) / 2.0
    expected = midpoint + 0.50 * (leg.ask - midpoint)
    if expected <= 0:
        return None
    theoretical = leg.theoretical_value
    maximum_gain = None if option_type == "CALL" else max(0.0, (leg.strike - expected) * 100.0)
    breakeven = leg.strike + expected if option_type == "CALL" else leg.strike - expected
    return OptionStructure(
        valid=True,
        strategy="LONG {}".format(option_type),
        expiration=leg.expiration,
        dte=int(_float(row.get("dte"))),
        legs=[leg],
        debit_credit="DEBIT",
        expected_entry=expected,
        natural_entry=leg.ask,
        maximum_loss=expected * 100.0,
        maximum_gain=maximum_gain,
        breakeven=breakeven,
        reward_risk=None,
        delta=leg.delta,
        gamma=leg.gamma,
        theta=leg.theta,
        vega=leg.vega,
        theta_holding_cost=leg.theta * holding_sessions * 100.0,
        orats_theoretical_value=theoretical,
        theoretical_edge=(theoretical - expected) if theoretical is not None else None,
        implied_volatility=_normalized_iv(row),
        quote_trade_date=str(row.get("tradeDate") or "")[:10],
        quote_updated_at=str(row.get("updatedAt") or ""),
        reasons=["Entry limit is modeled halfway from the quoted midpoint toward the ask; the natural fill is also stress-tested."],
        midpoint_entry=midpoint,
        entry_fill_fraction=0.50,
    )


def _build_credit_structure(
    short_row: Mapping[str, Any],
    long_row: Mapping[str, Any],
    direction: str,
    holding_sessions: int,
) -> Optional[OptionStructure]:
    option_type = "PUT" if direction == "BULLISH" else "CALL"
    short_leg = _leg(short_row, option_type, "SELL")
    long_leg = _leg(long_row, option_type, "BUY")
    midpoint_credit = (short_leg.bid + short_leg.ask) / 2.0 - (long_leg.bid + long_leg.ask) / 2.0
    natural_credit = short_leg.bid - long_leg.ask
    expected = midpoint_credit + 0.50 * (natural_credit - midpoint_credit)
    width = abs(short_leg.strike - long_leg.strike)
    if expected <= 0 or width <= 0 or expected >= width:
        return None
    maximum_loss = (width - expected) * 100.0
    maximum_gain = expected * 100.0
    theoretical = None
    if short_leg.theoretical_value is not None and long_leg.theoretical_value is not None:
        theoretical = short_leg.theoretical_value - long_leg.theoretical_value
    breakeven = short_leg.strike - expected if direction == "BULLISH" else short_leg.strike + expected
    net_delta = short_leg.delta + long_leg.delta
    net_gamma = short_leg.gamma + long_leg.gamma
    net_theta = short_leg.theta + long_leg.theta
    net_vega = short_leg.vega + long_leg.vega
    return OptionStructure(
        valid=True,
        strategy="BULL PUT CREDIT SPREAD" if direction == "BULLISH" else "BEAR CALL CREDIT SPREAD",
        expiration=short_leg.expiration,
        dte=int(_float(short_row.get("dte"))),
        legs=[short_leg, long_leg],
        debit_credit="CREDIT",
        expected_entry=expected,
        natural_entry=natural_credit,
        maximum_loss=maximum_loss,
        maximum_gain=maximum_gain,
        breakeven=breakeven,
        reward_risk=maximum_gain / maximum_loss if maximum_loss > 0 else None,
        delta=net_delta,
        gamma=net_gamma,
        theta=net_theta,
        vega=net_vega,
        theta_holding_cost=net_theta * holding_sessions * 100.0,
        orats_theoretical_value=theoretical,
        theoretical_edge=(expected - theoretical) if theoretical is not None else None,
        implied_volatility=_normalized_iv(short_row),
        quote_trade_date=str(short_row.get("tradeDate") or "")[:10],
        quote_updated_at=max(str(short_row.get("updatedAt") or ""), str(long_row.get("updatedAt") or "")),
        reasons=["Entry limit is modeled halfway from the quoted midpoint toward the natural credit; the natural fill is also stress-tested."],
        midpoint_entry=midpoint_credit,
        entry_fill_fraction=0.50,
    )


def choose_option_structure(
    rows: Iterable[Mapping[str, Any]],
    direction: str,
    target_price: float,
    holding_sessions: int,
    minimum_oi: int,
    minimum_volume: int,
    maximum_spread_pct: float,
    scenario_returns: Sequence[float] = (),
    commission_per_contract: float = 0.65,
    scenario_paths: Sequence[Sequence[float]] = (),
    stop_return: Optional[float] = None,
    target_return: Optional[float] = None,
    scenario_adverse_paths: Sequence[Sequence[float]] = (),
    scenario_favorable_paths: Sequence[Sequence[float]] = (),
    exit_iv_shift_points: float = 0.0,
    risk_free_rate: float = 0.0,
    dividend_yield: float = 0.0,
) -> OptionStructure:
    """Compare directional long options, debit spreads, and credit spreads.

    Every expiration in the configured 21-75 DTE window is evaluated. Exact
    two-sided quotes, realistic entry/exit friction, commissions, and the same
    underlying stop/target path model are applied to every structure. A
    candidate is selected on the older training analogues only. Its economics
    are then evaluated on the newer held-out analogues by
    :func:`evaluate_option_evidence`; no held-out observation participates in
    strike, expiration, or structure selection.
    """

    if direction not in {"BULLISH", "BEARISH"}:
        return _empty(["Option direction must be BULLISH or BEARISH."])
    filtered = [row for row in rows if 21 <= int(_float(row.get("dte"))) <= 75]
    if not filtered:
        return _empty(["No ORATS strikes in the configured 21-75 DTE window."])
    by_expiration: Dict[Tuple[str, int], List[Mapping[str, Any]]] = defaultdict(list)
    for row in filtered:
        expiration = str(row.get("expirDate") or "")[:10]
        dte = int(_float(row.get("dte")))
        if expiration and dte:
            by_expiration[(expiration, dte)].append(row)
    if not by_expiration:
        return _empty(["ORATS chain rows lacked expiration metadata."])

    structures: List[Tuple[float, OptionStructure]] = []
    directional_type = "CALL" if direction == "BULLISH" else "PUT"
    credit_type = "PUT" if direction == "BULLISH" else "CALL"
    desired_dte = min(60, max(28, holding_sessions * 4))
    for (expiration, dte), raw_rows in by_expiration.items():
        expiry_rows = sorted(raw_rows, key=lambda row: _float(row.get("strike")))
        dte_distance = abs(dte - desired_dte) / max(1.0, float(desired_dte))

        for row in expiry_rows:
            leg = _leg(row, directional_type, "BUY")
            absolute_delta = abs(leg.delta)
            if not (0.20 <= absolute_delta <= 0.80):
                continue
            if not _liquid(leg, minimum_oi, minimum_volume, maximum_spread_pct):
                continue
            structure = _build_long_structure(row, direction, holding_sessions)
            if structure is not None:
                quality = abs(absolute_delta - 0.55) + dte_distance + _liquidity_penalty(structure.legs)
                structures.append((quality, structure))

        for long_row in expiry_rows:
            long_leg = _leg(long_row, directional_type, "BUY")
            long_delta = abs(long_leg.delta)
            if not (0.25 <= long_delta <= 0.80):
                continue
            if not _liquid(long_leg, minimum_oi, minimum_volume, maximum_spread_pct):
                continue
            for short_row in expiry_rows:
                long_strike = _float(long_row.get("strike"))
                short_strike = _float(short_row.get("strike"))
                if direction == "BULLISH" and short_strike <= long_strike:
                    continue
                if direction == "BEARISH" and short_strike >= long_strike:
                    continue
                short_leg = _leg(short_row, directional_type, "SELL")
                short_delta = abs(short_leg.delta)
                if not (0.05 <= short_delta <= 0.60):
                    continue
                if not _liquid(short_leg, minimum_oi, minimum_volume, maximum_spread_pct):
                    continue
                midpoint = (long_leg.bid + long_leg.ask) / 2.0 - (short_leg.bid + short_leg.ask) / 2.0
                natural = long_leg.ask - short_leg.bid
                width = abs(short_strike - long_strike)
                expected = midpoint + 0.50 * (natural - midpoint)
                if expected <= 0 or width <= 0 or expected >= width:
                    continue
                structure = _build_debit_structure(
                    long_row, short_row, long_leg, short_leg, direction, holding_sessions, expiration, dte,
                )
                target_distance = abs(short_strike - target_price) / max(1.0, target_price)
                quality = abs(long_delta - 0.55) + abs(short_delta - 0.28) + target_distance + dte_distance + _liquidity_penalty(structure.legs)
                structures.append((quality, structure))

        for short_row in expiry_rows:
            short_leg = _leg(short_row, credit_type, "SELL")
            short_delta = abs(short_leg.delta)
            if not (0.10 <= short_delta <= 0.50):
                continue
            if not _liquid(short_leg, minimum_oi, minimum_volume, maximum_spread_pct):
                continue
            for long_row in expiry_rows:
                short_strike = _float(short_row.get("strike"))
                long_strike = _float(long_row.get("strike"))
                if direction == "BULLISH" and long_strike >= short_strike:
                    continue
                if direction == "BEARISH" and long_strike <= short_strike:
                    continue
                long_leg = _leg(long_row, credit_type, "BUY")
                long_delta = abs(long_leg.delta)
                if not (0.02 <= long_delta < short_delta):
                    continue
                if not _liquid(long_leg, minimum_oi, minimum_volume, maximum_spread_pct):
                    continue
                structure = _build_credit_structure(short_row, long_row, direction, holding_sessions)
                if structure is None:
                    continue
                wing_ratio = abs(short_strike - long_strike) / max(1.0, _float(short_row.get("stockPrice")))
                quality = abs(short_delta - 0.28) + abs(long_delta - 0.10) + wing_ratio + dte_distance + _liquidity_penalty(structure.legs)
                structures.append((quality, structure))

    if not structures:
        return _empty([
            "No long option, debit spread, or defined-risk credit spread had executable two-sided quotes and coherent pricing.",
            "Stock remains eligible if the underlying thesis independently qualifies.",
        ])
    if not scenario_returns:
        selected = min(structures, key=lambda item: item[0])[1]
        return replace(
            selected,
            candidate_count=len(structures),
            selection_method="No analogue sample was available; selected by expiry/delta/quote-quality fit only.",
        )

    split_index = _selection_split_index(len(scenario_returns))
    train_returns = scenario_returns[:split_index] if split_index else scenario_returns
    train_paths = scenario_paths[:split_index] if split_index else scenario_paths
    train_adverse = scenario_adverse_paths[:split_index] if split_index else scenario_adverse_paths
    train_favorable = scenario_favorable_paths[:split_index] if split_index else scenario_favorable_paths
    modeled: List[Tuple[int, float, float, float, float, OptionStructure]] = []
    spot = _float(filtered[0].get("stockPrice"))
    for quality, structure in structures:
        economics = model_option_economics(
            structure,
            spot,
            direction,
            holding_sessions,
            train_returns,
            commission_per_contract,
            train_paths,
            stop_return,
            target_return,
            train_adverse,
            train_favorable,
            exit_iv_shift_points,
            risk_free_rate,
            dividend_yield,
        )
        expected_profit = economics.get("expected_profit_dollars")
        expected_return = economics.get("expected_return_on_max_loss")
        lower_return = economics.get("expected_return_lower_95_on_max_loss")
        pop = economics.get("modeled_pop")
        conservative_return = (
            float(expected_return) - float(economics.get("standard_error_dollars") or 0.0) / float(economics.get("modeled_maximum_loss") or structure.maximum_loss or 1.0)
            if expected_return is not None
            else -float("inf")
        )
        modeled.append(
            (
                1 if expected_profit is not None and float(expected_profit) > 0 else 0,
                conservative_return,
                float(lower_return) if lower_return is not None else -float("inf"),
                float(pop) if pop is not None else -float("inf"),
                -quality,
                structure,
            )
        )
    selected = max(modeled, key=lambda item: item[:5])[5]
    held_out = max(0, len(scenario_returns) - split_index)
    return replace(
        selected,
        candidate_count=len(structures),
        selection_train_size=len(train_returns),
        selection_test_size=held_out,
        selection_method=(
            "Selected from {} exact 21-75 DTE candidates using only the oldest {} analogue paths and a one-standard-error return penalty; "
            "the newest {} paths were not used to choose expiry, strikes, or structure."
        ).format(len(structures), len(train_returns), held_out),
        reasons=list(selected.reasons) + [
            "Structure selection and held-out evaluation are separated; current-chain alternatives are not reselected after seeing held-out profit."
        ],
    )


def _selection_split_index(sample_size: int) -> int:
    if sample_size <= 1:
        return sample_size
    return max(1, min(sample_size - 1, int(math.floor(sample_size * 0.65))))


def evaluate_option_evidence(
    structure: OptionStructure,
    spot: float,
    direction: str,
    holding_sessions: int,
    scenario_returns: Sequence[float],
    commission_per_contract: float = 0.65,
    scenario_paths: Sequence[Sequence[float]] = (),
    stop_return: Optional[float] = None,
    target_return: Optional[float] = None,
    scenario_adverse_paths: Sequence[Sequence[float]] = (),
    scenario_favorable_paths: Sequence[Sequence[float]] = (),
    exit_iv_shift_points: float = 0.0,
    risk_free_rate: float = 0.0,
    dividend_yield: float = 0.0,
) -> Dict[str, Any]:
    """Evaluate the already-selected structure on untouched recent paths.

    The returned headline POP and expected profit use the holdout slice when it
    exists. Full-sample, natural-fill, and +/-2 IV-point values are diagnostics,
    not alternate structures and not inputs to strike selection.
    """

    split_index = structure.selection_train_size or _selection_split_index(len(scenario_returns))
    has_holdout = 0 < split_index < len(scenario_returns)
    evaluation_returns = scenario_returns[split_index:] if has_holdout else scenario_returns
    evaluation_paths = scenario_paths[split_index:] if has_holdout else scenario_paths
    evaluation_adverse = scenario_adverse_paths[split_index:] if has_holdout else scenario_adverse_paths
    evaluation_favorable = scenario_favorable_paths[split_index:] if has_holdout else scenario_favorable_paths

    def model(
        returns: Sequence[float],
        paths: Sequence[Sequence[float]],
        adverse: Sequence[Sequence[float]],
        favorable: Sequence[Sequence[float]],
        iv_shift: float,
        entry_override: Optional[float] = None,
    ) -> Dict[str, Any]:
        return model_option_economics(
            structure,
            spot,
            direction,
            holding_sessions,
            returns,
            commission_per_contract,
            paths,
            stop_return,
            target_return,
            adverse,
            favorable,
            iv_shift,
            risk_free_rate,
            dividend_yield,
            entry_override,
        )

    base = model(
        evaluation_returns,
        evaluation_paths,
        evaluation_adverse,
        evaluation_favorable,
        exit_iv_shift_points,
    )
    train = model(
        scenario_returns[:split_index],
        scenario_paths[:split_index],
        scenario_adverse_paths[:split_index],
        scenario_favorable_paths[:split_index],
        exit_iv_shift_points,
    ) if split_index else {}
    full = model(
        scenario_returns,
        scenario_paths,
        scenario_adverse_paths,
        scenario_favorable_paths,
        exit_iv_shift_points,
    )
    natural_entry = structure.natural_entry
    if natural_entry is not None and structure.debit_credit == "CREDIT":
        natural_entry = max(0.0, float(natural_entry))
    natural = model(
        evaluation_returns,
        evaluation_paths,
        evaluation_adverse,
        evaluation_favorable,
        exit_iv_shift_points,
        natural_entry,
    ) if natural_entry is not None else {}
    iv_down = model(
        evaluation_returns,
        evaluation_paths,
        evaluation_adverse,
        evaluation_favorable,
        exit_iv_shift_points - 2.0,
    )
    iv_up = model(
        evaluation_returns,
        evaluation_paths,
        evaluation_adverse,
        evaluation_favorable,
        exit_iv_shift_points + 2.0,
    )
    robust_values = [
        value
        for value in (
            base.get("expected_profit_dollars"),
            natural.get("expected_profit_dollars"),
            iv_down.get("expected_profit_dollars"),
            iv_up.get("expected_profit_dollars"),
        )
        if value is not None
    ]
    result = dict(base)
    result.update(
        {
            "method": ("Held-out evaluation. " if has_holdout else "No held-out slice was possible. ") + str(base.get("method") or ""),
            "evidence_role": "HELD_OUT_RECENT_PATHS" if has_holdout else "FULL_SAMPLE_NO_HOLDOUT",
            "selection_candidate_count": structure.candidate_count,
            "selection_train_size": split_index,
            "holdout_sample_size": len(evaluation_returns) if has_holdout else 0,
            "train_expected_profit_dollars": train.get("expected_profit_dollars"),
            "full_sample_expected_profit_dollars": full.get("expected_profit_dollars"),
            "natural_fill_expected_profit_dollars": natural.get("expected_profit_dollars"),
            "iv_down_2_expected_profit_dollars": iv_down.get("expected_profit_dollars"),
            "iv_up_2_expected_profit_dollars": iv_up.get("expected_profit_dollars"),
            "robust_expected_profit_dollars": min(float(value) for value in robust_values) if robust_values else None,
            "robust_positive_across_fill_iv_stress": bool(robust_values) and min(float(value) for value in robust_values) > 0,
        }
    )
    return result
