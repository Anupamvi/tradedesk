"""Exact-leg payoff and executable-entry economics for one option unit.

The calculator is deliberately limited to same-expiration structures, where
the expiry payoff is exact.  Term structures require a separately frozen
pathwise valuation model and therefore fail closed here instead of pretending
that a static expiry diagram is sufficient.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Optional, Sequence, Tuple

from .domain import LegAction, LegQuote, OptionLeg, OptionType, exact_quote_map
from .edge import CostBreakdown, PriceConvention


CONTRACT_MULTIPLIER = 100.0


class EconomicsError(ValueError):
    """Exact-leg economics could not be established."""


@dataclass(frozen=True)
class PayoffEnvelope:
    price_convention: PriceConvention
    executable_price: float
    signed_entry_debit: float
    maximum_profit: Optional[float]
    maximum_loss: float
    breakevens: Tuple[float, ...]
    calculation_version: str = "cultra.same-expiry-payoff.v1"


def executable_entry_debit(
    legs: Sequence[OptionLeg], quotes: Sequence[LegQuote]
) -> float:
    """Return natural executable debit per share; a credit is negative."""

    quote_map = exact_quote_map(tuple(quotes))
    symbols = {leg.occ_symbol for leg in legs}
    if not legs or set(quote_map) != symbols:
        raise EconomicsError("quotes must match every exact OCC leg and no others")
    debit = 0.0
    for leg in legs:
        quote = quote_map[leg.occ_symbol]
        if leg.action is LegAction.BUY:
            if quote.ask <= 0.0:
                raise EconomicsError("a bought leg has no executable ask")
            debit += quote.ask * leg.ratio
        else:
            if quote.bid <= 0.0:
                raise EconomicsError("a sold leg has no executable bid")
            debit -= quote.bid * leg.ratio
    if math.isclose(debit, 0.0, abs_tol=1e-12):
        raise EconomicsError("zero-cost entry is not executable evidence")
    return debit


def round_trip_costs(
    legs: Sequence[OptionLeg],
    quotes: Sequence[LegQuote],
    policy: Mapping[str, object],
    *,
    assignment_exercise: float = 0.0,
    dividends: float = 0.0,
    early_exit: float = 0.0,
) -> CostBreakdown:
    """Apply one exact, versioned cost formula to history and current tickets."""

    quote_map = exact_quote_map(tuple(quotes))
    if not legs or len(quote_map) != len(quotes) or set(quote_map) != {
        leg.occ_symbol for leg in legs
    }:
        raise EconomicsError("quotes must match every exact OCC leg and no others")
    try:
        version = str(policy["version"])
        commission = float(policy["commission_per_contract_per_side"])
        fee = float(policy["fee_per_contract_per_side"])
        fraction = float(policy["slippage_fraction_of_quoted_spread_per_side"])
        minimum = float(policy["minimum_slippage_dollars_per_contract_per_side"])
    except (KeyError, TypeError, ValueError) as exc:
        raise EconomicsError("cost policy is incomplete") from exc
    if (
        not version.strip()
        or not all(
            math.isfinite(value) and value >= 0.0
            for value in (commission, fee, fraction, minimum)
        )
        or fraction > 1.0
    ):
        raise EconomicsError("cost policy is invalid")
    contracts = sum(leg.ratio for leg in legs)
    sides = 2 * contracts
    spread_reference = math.fsum(
        quote_map[leg.occ_symbol].spread * leg.ratio * CONTRACT_MULTIPLIER
        for leg in legs
    )
    slippage = math.fsum(
        2.0
        * leg.ratio
        * max(
            minimum,
            quote_map[leg.occ_symbol].spread
            * CONTRACT_MULTIPLIER
            * fraction,
        )
        for leg in legs
    )
    return CostBreakdown(
        commissions=sides * commission,
        fees=sides * fee,
        slippage=slippage,
        assignment_exercise=assignment_exercise,
        dividends=dividends,
        early_exit=early_exit,
        model_version=version,
        spread_reference=spread_reference,
    )


def _intrinsic(leg: OptionLeg, underlying_price: float) -> float:
    if leg.option_type is OptionType.CALL:
        value = max(0.0, underlying_price - leg.strike)
    else:
        value = max(0.0, leg.strike - underlying_price)
    direction = 1.0 if leg.action is LegAction.BUY else -1.0
    return direction * leg.ratio * value * CONTRACT_MULTIPLIER


def _pnl_at(
    price: float,
    legs: Sequence[OptionLeg],
    signed_entry_debit: float,
    costs: CostBreakdown,
) -> float:
    return (
        math.fsum(_intrinsic(leg, price) for leg in legs)
        - signed_entry_debit * CONTRACT_MULTIPLIER
        - costs.total
    )


def _tail_slope(legs: Sequence[OptionLeg]) -> float:
    slope = 0.0
    for leg in legs:
        if leg.option_type is OptionType.CALL:
            direction = 1.0 if leg.action is LegAction.BUY else -1.0
            slope += direction * leg.ratio * CONTRACT_MULTIPLIER
    return slope


def _roots(
    knots: Sequence[float],
    values: Sequence[float],
    tail_slope: float,
) -> Tuple[float, ...]:
    roots = []
    for left, right, left_value, right_value in zip(
        knots[:-1], knots[1:], values[:-1], values[1:]
    ):
        if math.isclose(left_value, 0.0, abs_tol=1e-9):
            roots.append(left)
        if left_value * right_value < 0.0:
            fraction = -left_value / (right_value - left_value)
            roots.append(left + fraction * (right - left))
    if math.isclose(values[-1], 0.0, abs_tol=1e-9):
        roots.append(knots[-1])
    elif not math.isclose(tail_slope, 0.0, abs_tol=1e-12):
        tail_root = knots[-1] - values[-1] / tail_slope
        if tail_root > knots[-1]:
            roots.append(tail_root)
    normalized = []
    for value in sorted(root for root in roots if root >= 0.0):
        if not normalized or not math.isclose(value, normalized[-1], abs_tol=1e-7):
            normalized.append(value)
    return tuple(normalized)


def same_expiry_payoff_envelope(
    legs: Sequence[OptionLeg],
    quotes: Sequence[LegQuote],
    costs: CostBreakdown,
) -> PayoffEnvelope:
    """Derive finite loss, profit and breakevens from exact legs and quotes."""

    if not isinstance(costs, CostBreakdown):
        raise TypeError("costs must be CostBreakdown")
    if not legs:
        raise EconomicsError("at least one exact option leg is required")
    if len({leg.expiration for leg in legs}) != 1:
        raise EconomicsError(
            "multi-expiration structures require a frozen pathwise payoff artifact"
        )
    signed_debit = executable_entry_debit(legs, quotes)
    knots = tuple(sorted({0.0} | {float(leg.strike) for leg in legs}))
    values = tuple(_pnl_at(price, legs, signed_debit, costs) for price in knots)
    tail_slope = _tail_slope(legs)
    if tail_slope < -1e-12:
        raise EconomicsError("exact legs have undefined upside loss")
    minimum = min(values)
    maximum_loss = max(0.0, -minimum)
    if maximum_loss <= 0.0:
        raise EconomicsError("payoff does not establish a positive finite risk dollar")
    maximum_profit = None if tail_slope > 1e-12 else max(0.0, max(values))
    breakevens = _roots(knots, values, tail_slope)
    if not breakevens:
        raise EconomicsError("payoff has no nonnegative breakeven")
    convention = PriceConvention.DEBIT if signed_debit > 0.0 else PriceConvention.CREDIT
    return PayoffEnvelope(
        price_convention=convention,
        executable_price=abs(signed_debit),
        signed_entry_debit=signed_debit,
        maximum_profit=maximum_profit,
        maximum_loss=maximum_loss,
        breakevens=breakevens,
    )


__all__ = [
    "CONTRACT_MULTIPLIER",
    "EconomicsError",
    "PayoffEnvelope",
    "executable_entry_debit",
    "round_trip_costs",
    "same_expiry_payoff_envelope",
]
