"""Current six-strategy option selection from exact Schwab chains."""

from __future__ import annotations

import math
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from codexswing.clock import NEW_YORK
from codexswing.options.execution import conservative_two_leg_limit
from codexswing.options.expected_pnl import (
    CostAssumptions,
    ForecastDistribution,
    evaluate_long_option,
    evaluate_vertical,
)
from codexswing.options.structures import OptionQuote, SpreadLeg, StructureError, VerticalSpread


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _option_tick(price: float) -> float:
    return 0.01 if price < 3.0 else 0.05


def _round_up_to_tick(price: float) -> float:
    tick = _option_tick(price)
    return round(math.ceil((price - 1e-10) / tick) * tick, 2)


def _round_down_to_tick(price: float) -> float:
    tick = _option_tick(price)
    return round(math.floor((price + 1e-10) / tick) * tick, 2)


def _epoch_iso(value: Any) -> str:
    epoch = _number(value)
    if epoch <= 0:
        return ""
    if epoch > 10_000_000_000:
        epoch /= 1000.0
    try:
        return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat().replace("+00:00", "Z")
    except (OSError, OverflowError, ValueError):
        return ""


def _quote_session_date(value: Any) -> str:
    epoch = _number(value)
    if epoch <= 0:
        return ""
    if epoch > 10_000_000_000:
        epoch /= 1000.0
    try:
        return datetime.fromtimestamp(epoch, tz=timezone.utc).astimezone(NEW_YORK).date().isoformat()
    except (OSError, OverflowError, ValueError):
        return ""


def _option_quote(
    ticker: str,
    as_of_date: str,
    expiration: str,
    spot: float,
    right: str,
    contract: Mapping[str, Any],
    residual_rate: float,
) -> OptionQuote:
    iv = _number(contract.get("volatility")) / 100.0
    if iv <= 0:
        iv = _number(contract.get("theoreticalVolatility")) / 100.0
    return OptionQuote(
        ticker=ticker,
        quote_date=date.fromisoformat(as_of_date),
        expiration=date.fromisoformat(expiration),
        strike=_number(contract.get("strikePrice")),
        right=right,
        spot=spot,
        bid=_number(contract.get("bid")),
        ask=_number(contract.get("ask")),
        implied_volatility=iv,
        open_interest=int(_number(contract.get("openInterest"))),
        volume=int(_number(contract.get("totalVolume"))),
        residual_rate=residual_rate,
        updated_at_utc=_epoch_iso(contract.get("quoteTimeInLong")),
    )


def _spread(
    ticker: str,
    as_of_date: str,
    expiration: str,
    spot: float,
    right: str,
    long_contract: Mapping[str, Any],
    short_contract: Mapping[str, Any],
    residual_rate: float,
) -> VerticalSpread:
    return VerticalSpread(
        (
            SpreadLeg(
                _option_quote(
                    ticker,
                    as_of_date,
                    expiration,
                    spot,
                    right,
                    long_contract,
                    residual_rate,
                ),
                1,
            ),
            SpreadLeg(
                _option_quote(
                    ticker,
                    as_of_date,
                    expiration,
                    spot,
                    right,
                    short_contract,
                    residual_rate,
                ),
                -1,
            ),
        )
    )


def _strategy_name(side: str, strategy: str) -> str:
    names = {
        ("LONG", "call_debit"): "BULL_CALL_DEBIT",
        ("SHORT", "put_debit"): "BEAR_PUT_DEBIT",
        ("LONG", "put_credit"): "BULL_PUT_CREDIT",
        ("SHORT", "call_credit"): "BEAR_CALL_CREDIT",
    }
    return names[(side, strategy)]


def _fair_value(
    rows: Mapping[Tuple[str, float], Mapping[str, Any]],
    expiration: str,
    right: str,
    long_strike: float,
    short_strike: float,
) -> Tuple[Optional[float], Optional[str]]:
    long_row = rows.get((expiration, long_strike))
    short_row = rows.get((expiration, short_strike))
    if long_row is None or short_row is None:
        return None, None
    key = "callValue" if right == "call" else "putValue"
    long_value = _number(long_row.get(key), float("nan"))
    short_value = _number(short_row.get(key), float("nan"))
    if not math.isfinite(long_value) or not math.isfinite(short_value):
        return None, None
    return long_value - short_value, str(long_row.get("updatedAt") or "")


def _single_fair_value(
    rows: Mapping[Tuple[str, float], Mapping[str, Any]],
    expiration: str,
    right: str,
    strike: float,
) -> Tuple[Optional[float], Optional[str]]:
    row = rows.get((expiration, strike))
    if row is None:
        return None, None
    key = "callValue" if right == "call" else "putValue"
    value = _number(row.get(key), float("nan"))
    if not math.isfinite(value):
        return None, None
    return value, str(row.get("updatedAt") or "")


def select_current_verticals(
    *,
    ticker: str,
    side: str,
    as_of_date: str,
    chain: Mapping[str, Any],
    forecast: ForecastDistribution,
    current_iv_30d_pct: float,
    forecast_implied_iv_20d_pct: float,
    orats_strike_rows: Sequence[Mapping[str, Any]] = (),
    fresh_regular_session_quote: bool,
) -> Tuple[Sequence[Mapping[str, Any]], Mapping[str, int]]:
    """Enumerate two verticals plus one long option for the current direction."""

    normalized_side = side.strip().upper()
    if normalized_side not in {"LONG", "SHORT"}:
        raise ValueError("side must be LONG or SHORT")
    spot = _number(chain.get("underlyingPrice"))
    if spot <= 0:
        return (), {"missing_underlying_price": 1}
    risk_free_rate = _number(chain.get("interestRate")) / 100.0
    if risk_free_rate == 0:
        risk_free_rate = 0.04
    fair_rows = {
        (
            str(row.get("expirDate") or row.get("expiration") or "")[:10],
            _number(row.get("strike")),
        ): row
        for row in orats_strike_rows
        if str(row.get("ticker") or "").upper() == ticker.upper()
    }
    requested = (
        (("call", "debit"), ("put", "credit"))
        if normalized_side == "LONG"
        else (("put", "debit"), ("call", "credit"))
    )
    candidates: List[Mapping[str, Any]] = []
    rejected: Dict[str, int] = {}

    for right, kind in requested:
        expiration_map = chain.get("callExpDateMap" if right == "call" else "putExpDateMap")
        if not isinstance(expiration_map, Mapping):
            rejected["missing_{}_chain".format(right)] = 1
            continue
        for expiration_key, strike_map in expiration_map.items():
            expiration = str(expiration_key).split(":", 1)[0]
            try:
                dte = (date.fromisoformat(expiration) - date.fromisoformat(as_of_date)).days
            except ValueError:
                continue
            if not 21 <= dte <= 60 or not isinstance(strike_map, Mapping):
                continue
            contracts = []
            for values in strike_map.values():
                if not isinstance(values, list) or not values or not isinstance(values[0], Mapping):
                    continue
                contract = values[0]
                if (
                    _number(contract.get("bid")) <= 0
                    or _number(contract.get("ask")) < _number(contract.get("bid"))
                    or _number(contract.get("strikePrice")) <= 0
                    or _number(contract.get("volatility"), _number(contract.get("theoreticalVolatility"))) <= 0
                ):
                    rejected["invalid_contract_quote"] = rejected.get("invalid_contract_quote", 0) + 1
                    continue
                contracts.append(contract)
            for long_contract in contracts:
                long_strike = _number(long_contract.get("strikePrice"))
                long_delta = abs(_number(long_contract.get("delta")))
                for short_contract in contracts:
                    short_strike = _number(short_contract.get("strikePrice"))
                    short_delta = abs(_number(short_contract.get("delta")))
                    if kind == "debit":
                        delta_ok = 0.43 <= long_delta <= 0.72 and 0.18 <= short_delta <= 0.45
                    else:
                        delta_ok = 0.05 <= long_delta <= 0.18 and 0.18 <= short_delta <= 0.32
                    if not delta_ok:
                        continue
                    try:
                        spread = _spread(
                            ticker.upper(),
                            as_of_date,
                            expiration,
                            spot,
                            right,
                            long_contract,
                            short_contract,
                            risk_free_rate,
                        )
                    except (StructureError, ValueError):
                        continue
                    expected_strategy = (
                        "call_debit" if (normalized_side, kind) == ("LONG", "debit")
                        else "put_credit" if (normalized_side, kind) == ("LONG", "credit")
                        else "put_debit" if kind == "debit"
                        else "call_credit"
                    )
                    if spread.strategy != expected_strategy:
                        continue
                    if spread.width < max(1.0, spot * 0.005) or spread.width > spot * 0.065:
                        continue
                    if spread.maximum_quote_spread_pct > 0.25:
                        rejected["wide_leg_quote"] = rejected.get("wide_leg_quote", 0) + 1
                        continue
                    if spread.minimum_open_interest < 100 or spread.minimum_volume < 10:
                        rejected["thin_contract"] = rejected.get("thin_contract", 0) + 1
                        continue
                    package = conservative_two_leg_limit(spread)
                    try:
                        evaluation = evaluate_vertical(
                            spread,
                            forecast,
                            reference_spot=spot,
                            risk_free_rate=risk_free_rate,
                            costs=CostAssumptions(),
                            iv_multiplier=min(
                                max(
                                    forecast_implied_iv_20d_pct / max(current_iv_30d_pct, 0.01),
                                    0.50,
                                ),
                                1.50,
                            ),
                            entry_debit_per_share=package.signed_debit_per_share,
                            entry_price_source=package.method,
                        )
                    except (StructureError, ValueError):
                        rejected["expression_model_error"] = rejected.get("expression_model_error", 0) + 1
                        continue
                    fair_signed, fair_updated = _fair_value(
                        fair_rows,
                        expiration,
                        right,
                        long_strike,
                        short_strike,
                    )
                    edge = (
                        fair_signed - package.signed_debit_per_share
                        if fair_signed is not None
                        else None
                    )
                    max_loss = abs(min(evaluation.expiry_max_loss_dollars, 0.0))
                    max_profit = max(evaluation.expiry_max_profit_dollars, 0.0)
                    legs_are_from_as_of_session = all(
                        _quote_session_date(contract.get("quoteTimeInLong")) == as_of_date
                        for contract in (long_contract, short_contract)
                    )
                    candidates.append(
                        {
                            "ticker": ticker.upper(),
                            "side": normalized_side,
                            "strategy": _strategy_name(normalized_side, spread.strategy),
                            "leg_count": 2,
                            "right": right.upper(),
                            "expiration": expiration,
                            "dte": dte,
                            "long_symbol": str(long_contract.get("symbol") or ""),
                            "long_strike": long_strike,
                            "long_bid": _number(long_contract.get("bid")),
                            "long_ask": _number(long_contract.get("ask")),
                            "long_delta": _number(long_contract.get("delta")),
                            "short_symbol": str(short_contract.get("symbol") or ""),
                            "short_strike": short_strike,
                            "short_bid": _number(short_contract.get("bid")),
                            "short_ask": _number(short_contract.get("ask")),
                            "short_delta": _number(short_contract.get("delta")),
                            "width": spread.width,
                            "entry_limit_signed_debit": package.signed_debit_per_share,
                            "entry_limit_display": (
                                "PAY {:.2f}".format(package.signed_debit_per_share)
                                if package.signed_debit_per_share > 0
                                else "RECEIVE {:.2f}".format(-package.signed_debit_per_share)
                            ),
                            "entry_fill_model": package.to_dict(),
                            "maximum_loss_dollars": max_loss,
                            "maximum_profit_dollars": max_profit,
                            "reward_to_risk": max_profit / max(max_loss, 0.01),
                            "minimum_open_interest": spread.minimum_open_interest,
                            "minimum_volume": spread.minimum_volume,
                            "maximum_leg_spread_pct": spread.maximum_quote_spread_pct,
                            "modeled_expected_pnl_dollars": evaluation.expected_pnl_after_costs,
                            "modeled_probability_positive": evaluation.probability_positive_after_costs,
                            "modeled_p05_pnl_dollars": evaluation.p05_pnl_after_costs,
                            "modeled_exit_cost_dollars": evaluation.modeled_exit_cost_dollars,
                            "modeled_iv_multiplier": min(
                                max(forecast_implied_iv_20d_pct / max(current_iv_30d_pct, 0.01), 0.50),
                                1.50,
                            ),
                            "orats_fair_signed_debit": fair_signed,
                            "orats_edge_at_limit_per_share": edge,
                            "orats_fair_value_updated_at": fair_updated,
                            "fresh_regular_session_quote": (
                                fresh_regular_session_quote and legs_are_from_as_of_session
                            ),
                            "quote_time_utc": min(
                                spread.long_leg.quote.updated_at_utc,
                                spread.short_leg.quote.updated_at_utc,
                            ),
                            "expression_status": evaluation.status,
                            "selector_score": (
                                abs(dte - 36) / 30.0
                                + 2.0 * spread.maximum_quote_spread_pct
                                + (
                                    4.0 * abs(long_delta - 0.58) + 3.0 * abs(short_delta - 0.30)
                                    if kind == "debit"
                                    else 4.0 * abs(short_delta - 0.25) + 3.0 * abs(long_delta - 0.10)
                                )
                            ),
                        }
                    )

    single_right = "call" if normalized_side == "LONG" else "put"
    expiration_map = chain.get("callExpDateMap" if single_right == "call" else "putExpDateMap")
    if not isinstance(expiration_map, Mapping):
        rejected["missing_{}_chain".format(single_right)] = 1
    else:
        for expiration_key, strike_map in expiration_map.items():
            expiration = str(expiration_key).split(":", 1)[0]
            try:
                dte = (date.fromisoformat(expiration) - date.fromisoformat(as_of_date)).days
            except ValueError:
                continue
            if not 21 <= dte <= 60 or not isinstance(strike_map, Mapping):
                continue
            for values in strike_map.values():
                if not isinstance(values, list) or not values or not isinstance(values[0], Mapping):
                    continue
                contract = values[0]
                bid = _number(contract.get("bid"))
                ask = _number(contract.get("ask"))
                strike = _number(contract.get("strikePrice"))
                delta = abs(_number(contract.get("delta")))
                if (
                    bid <= 0
                    or ask < bid
                    or strike <= 0
                    or not 0.42 <= delta <= 0.62
                    or _number(contract.get("volatility"), _number(contract.get("theoreticalVolatility"))) <= 0
                ):
                    continue
                try:
                    quote = _option_quote(
                        ticker.upper(),
                        as_of_date,
                        expiration,
                        spot,
                        single_right,
                        contract,
                        risk_free_rate,
                    )
                except (StructureError, ValueError):
                    continue
                relative_spread = quote.spread / max(quote.mid, 0.05)
                if relative_spread > 0.25:
                    rejected["wide_single_quote"] = rejected.get("wide_single_quote", 0) + 1
                    continue
                if quote.open_interest < 100 or quote.volume < 10:
                    rejected["thin_single_contract"] = rejected.get("thin_single_contract", 0) + 1
                    continue
                modeled_entry_debit = quote.bid + 0.75 * quote.spread
                entry_debit = _round_up_to_tick(modeled_entry_debit)
                starting_limit = _round_down_to_tick(quote.mid)
                iv_multiplier = min(
                    max(forecast_implied_iv_20d_pct / max(current_iv_30d_pct, 0.01), 0.50),
                    1.50,
                )
                try:
                    evaluation = evaluate_long_option(
                        quote,
                        forecast,
                        reference_spot=spot,
                        risk_free_rate=risk_free_rate,
                        costs=CostAssumptions(),
                        iv_multiplier=iv_multiplier,
                        entry_debit_per_share=entry_debit,
                        entry_price_source="75_PERCENT_SINGLE_LEG_SPREAD",
                    )
                except (StructureError, ValueError):
                    rejected["single_expression_model_error"] = rejected.get(
                        "single_expression_model_error", 0
                    ) + 1
                    continue
                fair_value, fair_updated = _single_fair_value(
                    fair_rows, expiration, single_right, strike
                )
                max_loss = abs(min(evaluation.expiry_max_loss_dollars, 0.0))
                candidates.append(
                    {
                        "ticker": ticker.upper(),
                        "side": normalized_side,
                        "strategy": "LONG_CALL" if single_right == "call" else "LONG_PUT",
                        "leg_count": 1,
                        "right": single_right.upper(),
                        "expiration": expiration,
                        "dte": dte,
                        "long_symbol": str(contract.get("symbol") or ""),
                        "long_strike": strike,
                        "long_bid": bid,
                        "long_ask": ask,
                        "long_delta": _number(contract.get("delta")),
                        "short_symbol": None,
                        "short_strike": None,
                        "short_bid": None,
                        "short_ask": None,
                        "short_delta": None,
                        "width": None,
                        "entry_limit_signed_debit": entry_debit,
                        "entry_limit_display": "PAY {:.2f} MAX; START {:.2f}".format(
                            entry_debit, starting_limit
                        ),
                        "entry_fill_model": {
                            "method": "75_PERCENT_SINGLE_LEG_SPREAD_ROUNDED_UP_TO_TICK",
                            "bid": bid,
                            "ask": ask,
                            "modeled_unrounded_debit_per_share": modeled_entry_debit,
                            "starting_limit_per_share": starting_limit,
                            "hard_maximum_limit_per_share": entry_debit,
                            "signed_debit_per_share": entry_debit,
                        },
                        "maximum_loss_dollars": max_loss,
                        "maximum_profit_dollars": evaluation.expiry_max_profit_dollars,
                        "reward_to_risk": None,
                        "minimum_open_interest": quote.open_interest,
                        "minimum_volume": quote.volume,
                        "maximum_leg_spread_pct": relative_spread,
                        "modeled_expected_pnl_dollars": evaluation.expected_pnl_after_costs,
                        "modeled_probability_positive": evaluation.probability_positive_after_costs,
                        "modeled_p05_pnl_dollars": evaluation.p05_pnl_after_costs,
                        "modeled_exit_cost_dollars": evaluation.modeled_exit_cost_dollars,
                        "modeled_iv_multiplier": iv_multiplier,
                        "orats_fair_signed_debit": fair_value,
                        "orats_edge_at_limit_per_share": (
                            fair_value - entry_debit if fair_value is not None else None
                        ),
                        "orats_fair_value_updated_at": fair_updated,
                        "fresh_regular_session_quote": (
                            fresh_regular_session_quote
                            and _quote_session_date(contract.get("quoteTimeInLong")) == as_of_date
                        ),
                        "quote_time_utc": quote.updated_at_utc,
                        "expression_status": evaluation.status,
                        "selector_score": (
                            abs(dte - 36) / 30.0
                            + 2.0 * relative_spread
                            + 4.0 * abs(delta - 0.52)
                        ),
                    }
                )

    candidates.sort(
        key=lambda item: (
            _number(item.get("selector_score"), 999.0),
            -_number(item.get("modeled_expected_pnl_dollars"))
            / max(_number(item.get("maximum_loss_dollars")), 1.0),
            -_number(item.get("modeled_probability_positive")),
            _number(item.get("maximum_leg_spread_pct")),
            str(item.get("expiration")),
        )
    )
    rejected["enumerated_candidates"] = len(candidates)
    return tuple(candidates), dict(sorted(rejected.items()))
