"""Exact two-leg vertical structures from ORATS rows."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple


class StructureError(RuntimeError):
    pass


def _float(row: Mapping[str, Any], key: str, required: bool = True, default: float = 0.0) -> float:
    value = row.get(key)
    if value is None or value == "":
        if required:
            raise StructureError("ORATS row is missing {}".format(key))
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        raise StructureError("ORATS row has invalid {}".format(key)) from None


@dataclass(frozen=True)
class OptionQuote:
    ticker: str
    quote_date: date
    expiration: date
    strike: float
    right: str
    spot: float
    bid: float
    ask: float
    implied_volatility: float
    open_interest: int
    volume: int
    residual_rate: float
    updated_at_utc: str

    def __post_init__(self) -> None:
        if self.right not in {"call", "put"}:
            raise StructureError("option right must be call or put")
        if self.expiration < self.quote_date:
            raise StructureError("option is expired at quote date")
        if self.spot <= 0 or self.strike <= 0:
            raise StructureError("spot and strike must be positive")
        if self.bid < 0 or self.ask < self.bid:
            raise StructureError("invalid option bid/ask")
        if self.implied_volatility <= 0:
            raise StructureError("implied volatility must be positive")
        if self.open_interest < 0 or self.volume < 0:
            raise StructureError("option volume/open interest cannot be negative")

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @classmethod
    def from_orats(cls, row: Mapping[str, Any], right: str) -> "OptionQuote":
        normalized = right.strip().lower()
        if normalized not in {"call", "put"}:
            raise StructureError("right must be call or put")
        prefix = "call" if normalized == "call" else "put"
        quote_date_text = str(row.get("quoteDate") or row.get("tradeDate") or "")[:10]
        expiration_text = str(row.get("expirDate") or row.get("expiration") or "")[:10]
        try:
            quote_date = date.fromisoformat(quote_date_text)
            expiration = date.fromisoformat(expiration_text)
        except ValueError:
            raise StructureError("ORATS row has invalid quote/expiration date") from None
        iv = _float(row, "{}MidIv".format(prefix), required=False, default=0.0)
        if iv <= 0:
            iv = _float(row, "smvVol")
        spot_key = "stockPrice" if row.get("stockPrice") not in (None, "") else "spotPrice"
        return cls(
            ticker=str(row.get("ticker") or "").upper(),
            quote_date=quote_date,
            expiration=expiration,
            strike=_float(row, "strike"),
            right=normalized,
            spot=_float(row, spot_key),
            bid=_float(row, "{}BidPrice".format(prefix)),
            ask=_float(row, "{}AskPrice".format(prefix)),
            implied_volatility=iv,
            open_interest=int(_float(row, "{}OpenInterest".format(prefix), required=False, default=0.0)),
            volume=int(_float(row, "{}Volume".format(prefix), required=False, default=0.0)),
            residual_rate=_float(row, "residualRate", required=False, default=0.0),
            updated_at_utc=str(row.get("updatedAt") or ""),
        )


@dataclass(frozen=True)
class SpreadLeg:
    quote: OptionQuote
    quantity: int

    def __post_init__(self) -> None:
        if self.quantity not in {-1, 1}:
            raise StructureError("vertical leg quantity must be +1 or -1")


@dataclass(frozen=True)
class VerticalSpread:
    legs: Tuple[SpreadLeg, SpreadLeg]

    def __post_init__(self) -> None:
        if len(self.legs) != 2:
            raise StructureError("vertical spread requires exactly two legs")
        first, second = self.legs
        if first.quantity + second.quantity != 0:
            raise StructureError("vertical spread requires one long and one short leg")
        left = first.quote
        right = second.quote
        if (left.ticker, left.quote_date, left.expiration, left.right) != (
            right.ticker,
            right.quote_date,
            right.expiration,
            right.right,
        ):
            raise StructureError("vertical legs must share ticker/date/expiration/right")
        if not math.isclose(left.spot, right.spot, rel_tol=0.002, abs_tol=0.01):
            raise StructureError("vertical legs disagree on underlying spot")
        if math.isclose(left.strike, right.strike):
            raise StructureError("vertical strikes must differ")
        entry = self.entry_debit_per_share
        if self.strategy.endswith("debit") and not (0 < entry < self.width):
            raise StructureError("debit spread natural entry must be between zero and width")
        if self.strategy.endswith("credit") and not (0 < -entry < self.width):
            raise StructureError("credit spread natural credit must be between zero and width")

    @property
    def long_leg(self) -> SpreadLeg:
        return next(leg for leg in self.legs if leg.quantity == 1)

    @property
    def short_leg(self) -> SpreadLeg:
        return next(leg for leg in self.legs if leg.quantity == -1)

    @property
    def ticker(self) -> str:
        return self.long_leg.quote.ticker

    @property
    def right(self) -> str:
        return self.long_leg.quote.right

    @property
    def spot(self) -> float:
        return self.long_leg.quote.spot

    @property
    def width(self) -> float:
        return abs(self.long_leg.quote.strike - self.short_leg.quote.strike)

    @property
    def strategy(self) -> str:
        long_strike = self.long_leg.quote.strike
        short_strike = self.short_leg.quote.strike
        if self.right == "call":
            return "call_debit" if long_strike < short_strike else "call_credit"
        return "put_debit" if long_strike > short_strike else "put_credit"

    @property
    def entry_debit_per_share(self) -> float:
        total = 0.0
        for leg in self.legs:
            total += leg.quote.ask if leg.quantity == 1 else -leg.quote.bid
        return total

    @property
    def maximum_quote_spread_pct(self) -> float:
        values = []
        for leg in self.legs:
            denominator = max(leg.quote.mid, 0.01)
            values.append(leg.quote.spread / denominator)
        return max(values)

    @property
    def minimum_open_interest(self) -> int:
        return min(leg.quote.open_interest for leg in self.legs)

    @property
    def minimum_volume(self) -> int:
        return min(leg.quote.volume for leg in self.legs)

    def payoff_per_share(self, terminal_spot: float) -> float:
        payoff = 0.0
        for leg in self.legs:
            if leg.quote.right == "call":
                intrinsic = max(terminal_spot - leg.quote.strike, 0.0)
            else:
                intrinsic = max(leg.quote.strike - terminal_spot, 0.0)
            payoff += leg.quantity * intrinsic
        return payoff

    def expiry_pnl_bounds_dollars(
        self,
        contracts: int = 1,
        entry_debit_per_share: Optional[float] = None,
    ) -> Tuple[float, float]:
        entry = (
            self.entry_debit_per_share
            if entry_debit_per_share is None
            else float(entry_debit_per_share)
        )
        if self.strategy.endswith("debit") and not (0 < entry < self.width):
            raise StructureError("debit entry must be between zero and width")
        if self.strategy.endswith("credit") and not (0 < -entry < self.width):
            raise StructureError("credit entry must be between zero and width")
        strikes = [leg.quote.strike for leg in self.legs]
        points = [0.0, min(strikes), max(strikes), max(self.spot * 4.0, max(strikes) * 2.0)]
        pnls = [
            (self.payoff_per_share(spot) - entry) * 100.0 * contracts
            for spot in points
        ]
        return min(pnls), max(pnls)


def vertical_from_orats_rows(
    rows: Iterable[Mapping[str, Any]],
    right: str,
    long_strike: float,
    short_strike: float,
    expiration: str,
) -> VerticalSpread:
    matching = [
        row
        for row in rows
        if str(row.get("expirDate") or row.get("expiration") or "")[:10] == expiration
    ]

    def find(strike: float) -> Mapping[str, Any]:
        choices = [row for row in matching if math.isclose(_float(row, "strike"), strike, abs_tol=1e-8)]
        if len(choices) != 1:
            raise StructureError("expected exactly one ORATS row for strike {}".format(strike))
        return choices[0]

    return VerticalSpread(
        legs=(
            SpreadLeg(OptionQuote.from_orats(find(long_strike), right), 1),
            SpreadLeg(OptionQuote.from_orats(find(short_strike), right), -1),
        )
    )
