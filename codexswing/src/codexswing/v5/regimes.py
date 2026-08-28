"""Point-in-time ORATS core-regime vectors and label-free analog selection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any, Iterable, Mapping, Sequence, Tuple


REGIME_VECTOR_FIELDS = (
    "iv_percentile_1y",
    "iv_vs_realized_forecast_20d",
    "implied_vol_forecast_wedge_20d",
    "term_slope",
    "contango",
    "stock_change_1w",
    "stock_change_1m",
)


def _number(row: Mapping[str, Any], names: Sequence[str], label: str) -> float:
    value = None
    for name in names:
        if row.get(name) not in (None, ""):
            value = row[name]
            break
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError("missing or invalid {}".format(label)) from None
    if not math.isfinite(result):
        raise ValueError("non-finite {}".format(label))
    return result


def _ratio(value: float) -> float:
    """Normalize volatility fields supplied as decimal or percentage points."""

    return value / 100.0 if abs(value) > 3.0 else value


def _percent_points(value: float) -> float:
    """ORATS price-change and curve fields are expressed in percentage points."""

    return value / 100.0


def _percentile(value: float) -> float:
    normalized = value / 100.0 if abs(value) > 1.0 else value
    return min(1.0, max(0.0, normalized))


def _clip(value: float, lower: float = -3.0, upper: float = 3.0) -> float:
    return min(upper, max(lower, value))


@dataclass(frozen=True)
class CoreRegimeObservation:
    ticker: str
    trade_date: str
    side: str
    iv_30d: float
    iv_percentile_1y: float
    realized_forecast_20d: float
    implied_vol_forecast_20d: float
    term_slope: float
    contango: float
    stock_change_1w: float
    stock_change_1m: float

    def __post_init__(self) -> None:
        try:
            date.fromisoformat(self.trade_date)
        except ValueError:
            raise ValueError("trade_date must be YYYY-MM-DD") from None
        if not self.ticker or self.ticker != self.ticker.upper():
            raise ValueError("ticker must be uppercase")
        if self.side not in {"LONG", "SHORT"}:
            raise ValueError("side must be LONG or SHORT")
        values = (
            self.iv_30d,
            self.iv_percentile_1y,
            self.realized_forecast_20d,
            self.implied_vol_forecast_20d,
            self.term_slope,
            self.contango,
            self.stock_change_1w,
            self.stock_change_1m,
        )
        if any(not math.isfinite(item) for item in values):
            raise ValueError("regime observations must be finite")

    @classmethod
    def from_orats(
        cls, row: Mapping[str, Any], side: str
    ) -> "CoreRegimeObservation":
        ticker = str(row.get("ticker") or row.get("symbol") or "").strip().upper()
        trade_date = str(row.get("tradeDate") or row.get("date") or "")[:10]
        return cls(
            ticker=ticker,
            trade_date=trade_date,
            side=side.strip().upper(),
            iv_30d=_ratio(_number(row, ("iv30d", "iv30", "iv"), "iv_30d")),
            iv_percentile_1y=_percentile(
                _number(
                    row,
                    ("ivPctile1y", "ivPercentile1y", "ivRank1y", "ivRank"),
                    "iv_percentile_1y",
                )
            ),
            realized_forecast_20d=_ratio(
                _number(row, ("orFcst20d",), "realized_forecast_20d")
            ),
            implied_vol_forecast_20d=_ratio(
                _number(row, ("orIvFcst20d",), "implied_vol_forecast_20d")
            ),
            term_slope=_percent_points(
                _number(row, ("slope", "termSlope", "slopeInf"), "term_slope")
            ),
            # ORATS emits contango as a decimal ratio (for example -0.019),
            # unlike slope and stock changes, which are percentage points.
            contango=_number(row, ("contango", "contango1m2m"), "contango"),
            stock_change_1w=_percent_points(
                _number(row, ("stkPxChng1wk", "stockChange1w"), "stock_change_1w")
            ),
            stock_change_1m=_percent_points(
                _number(row, ("stkPxChng1m", "stockChange1m"), "stock_change_1m")
            ),
        )

    def vector(self) -> Tuple[float, ...]:
        iv_scale = max(abs(self.iv_30d), 0.05)
        return (
            self.iv_percentile_1y,
            _clip((self.iv_30d - self.realized_forecast_20d) / iv_scale),
            _clip((self.implied_vol_forecast_20d - self.iv_30d) / iv_scale),
            _clip(self.term_slope / 0.25),
            _clip(self.contango / 0.25),
            _clip(self.stock_change_1w / 0.20),
            _clip(self.stock_change_1m / 0.40),
        )


@dataclass(frozen=True)
class AnalogMatch:
    observation: CoreRegimeObservation
    distance: float


def regime_distance(left: CoreRegimeObservation, right: CoreRegimeObservation) -> float:
    a = left.vector()
    b = right.vector()
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / len(a))


def select_nearest_analogs(
    current: CoreRegimeObservation,
    history: Iterable[CoreRegimeObservation],
    max_neighbors: int = 250,
) -> Tuple[AnalogMatch, ...]:
    """Return prior same-ticker, same-side regimes without inspecting outcomes."""

    if max_neighbors <= 0:
        raise ValueError("max_neighbors must be positive")
    eligible = (
        item
        for item in history
        if item.ticker == current.ticker
        and item.side == current.side
        and item.trade_date < current.trade_date
    )
    matches = [AnalogMatch(item, regime_distance(current, item)) for item in eligible]
    matches.sort(key=lambda item: (item.distance, item.observation.trade_date))
    return tuple(matches[:max_neighbors])
