"""Conservative, explicit package-price assumptions for two-leg spreads."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict

from codexswing.options.structures import VerticalSpread


@dataclass(frozen=True)
class PackagePrice:
    signed_debit_per_share: float
    natural_signed_debit_per_share: float
    opposite_natural_signed_debit_per_share: float
    midpoint_signed_debit_per_share: float
    package_width_per_share: float
    slippage_fraction_from_favorable_side: float
    tick_size: float
    method: str = "ORATS_TWO_LEG_66_PERCENT_PACKAGE_WIDTH"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def conservative_two_leg_limit(
    spread: VerticalSpread,
    *,
    slippage_fraction: float = 0.66,
    tick_size: float = 0.05,
) -> PackagePrice:
    """Model an executable limit, rather than silently assuming midpoint.

    The limit is 66% of the package bid/ask width from the favorable side,
    matching ORATS's published default for two-leg backtests. Signed debit is
    positive for debit spreads and negative for credit spreads.
    """

    if not 0.5 <= slippage_fraction <= 1.0:
        raise ValueError("slippage_fraction must be between 0.5 and 1.0")
    if tick_size <= 0:
        raise ValueError("tick_size must be positive")
    long_quote = spread.long_leg.quote
    short_quote = spread.short_leg.quote
    natural = long_quote.ask - short_quote.bid
    opposite = long_quote.bid - short_quote.ask
    package_width = natural - opposite
    if package_width < 0:
        raise ValueError("invalid package market")
    raw = opposite + slippage_fraction * package_width
    # Round toward the adverse side: a higher signed debit means paying more or
    # receiving less credit.
    signed_debit = math.ceil(raw / tick_size - 1e-9) * tick_size
    if spread.strategy.endswith("debit"):
        signed_debit = min(max(signed_debit, tick_size), spread.width - tick_size)
    else:
        signed_debit = max(min(signed_debit, -tick_size), -spread.width + tick_size)
    return PackagePrice(
        signed_debit_per_share=signed_debit,
        natural_signed_debit_per_share=natural,
        opposite_natural_signed_debit_per_share=opposite,
        midpoint_signed_debit_per_share=(natural + opposite) / 2.0,
        package_width_per_share=package_width,
        slippage_fraction_from_favorable_side=slippage_fraction,
        tick_size=tick_size,
    )
