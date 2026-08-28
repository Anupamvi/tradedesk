"""Strict ORATS end-of-day implied-volatility features."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from codexswing.clock import parse_timestamp


class VolatilityFeatureError(ValueError):
    pass


def _number(value: Any, label: str, required: bool = True) -> Optional[float]:
    if value is None or isinstance(value, bool):
        if required:
            raise VolatilityFeatureError("missing {}".format(label))
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        if required:
            raise VolatilityFeatureError("invalid {}".format(label)) from None
        return None
    if not math.isfinite(result):
        if required:
            raise VolatilityFeatureError("non-finite {}".format(label))
        return None
    return result


@dataclass(frozen=True)
class IVRankObservation:
    ticker: str
    trade_date: str
    updated_at_utc: str
    iv_percent: float
    iv_rank_1y: Optional[float]
    iv_percentile_1y: Optional[float]

    def __post_init__(self) -> None:
        try:
            date.fromisoformat(self.trade_date)
        except ValueError:
            raise VolatilityFeatureError("trade_date must be YYYY-MM-DD") from None
        parse_timestamp(self.updated_at_utc)
        if not self.ticker or self.ticker != self.ticker.upper():
            raise VolatilityFeatureError("ticker must be uppercase")
        if not math.isfinite(self.iv_percent) or self.iv_percent <= 0 or self.iv_percent > 1000:
            raise VolatilityFeatureError("iv must be a positive finite percentage")

    @classmethod
    def from_orats(cls, row: Mapping[str, Any]) -> "IVRankObservation":
        ticker = str(row.get("ticker") or row.get("symbol") or "").strip().upper()
        trade_date = str(row.get("tradeDate") or row.get("date") or "")[:10]
        updated_at = str(row.get("updatedAt") or "").strip()
        return cls(
            ticker=ticker,
            trade_date=trade_date,
            updated_at_utc=updated_at,
            iv_percent=float(_number(row.get("iv"), "iv")),
            iv_rank_1y=_number(row.get("ivRank1y"), "ivRank1y", required=False),
            iv_percentile_1y=_number(row.get("ivPct1y"), "ivPct1y", required=False),
        )


def parse_orats_ivrank_rows(
    rows: Iterable[Mapping[str, Any]],
    tickers: Optional[Iterable[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Mapping[str, Sequence[IVRankObservation]]:
    allowed = {ticker.strip().upper() for ticker in tickers or ()}
    by_ticker: Dict[str, list] = {}
    seen = set()
    for row in rows:
        raw_date = str(row.get("tradeDate") or row.get("date") or "")[:10]
        raw_ticker = str(row.get("ticker") or row.get("symbol") or "").strip().upper()
        if start_date and raw_date < start_date:
            continue
        if end_date and raw_date > end_date:
            continue
        if allowed and raw_ticker not in allowed:
            continue
        observation = IVRankObservation.from_orats(row)
        key = (observation.ticker, observation.trade_date)
        if key in seen:
            raise VolatilityFeatureError(
                "duplicate ORATS IV row for {} {}".format(observation.ticker, observation.trade_date)
            )
        seen.add(key)
        by_ticker.setdefault(observation.ticker, []).append(observation)
    return {
        ticker: tuple(sorted(values, key=lambda item: item.trade_date))
        for ticker, values in sorted(by_ticker.items())
    }
