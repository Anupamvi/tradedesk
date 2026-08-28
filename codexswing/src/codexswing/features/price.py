"""Adjusted price history derived only from ORATS historical dailies."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
TICKER_RE = re.compile(r"^[A-Z0-9.^/-]{1,32}$")


class PriceHistoryError(RuntimeError):
    pass


def _number(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise PriceHistoryError("invalid ORATS {}".format(label)) from None
    if not math.isfinite(result):
        raise PriceHistoryError("non-finite ORATS {}".format(label))
    return result


@dataclass(frozen=True)
class PriceObservation:
    session_date: str
    ticker: str
    close: float
    high: float
    low: float
    volume: float
    avg30_volume: Optional[float] = None
    market_cap: Optional[float] = None
    issue_type: Optional[str] = None
    sector: Optional[str] = None
    open: Optional[float] = None
    source: str = "ORATS_HIST_DAILIES_ADJUSTED"
    ohlc_envelope_normalized: bool = False


@dataclass(frozen=True)
class PriceHistoryResult:
    observations: Mapping[str, Sequence[PriceObservation]]
    requested_tickers: Sequence[str]
    start_date: Optional[str]
    end_date: Optional[str]
    source: str = "ORATS_HIST_DAILIES_ADJUSTED"

    @property
    def date_count(self) -> int:
        return len(
            {
                item.session_date
                for values in self.observations.values()
                for item in values
            }
        )


def _normalize_tickers(tickers: Iterable[str]) -> Tuple[str, ...]:
    result: List[str] = []
    seen = set()
    for raw in tickers:
        ticker = raw.strip().upper()
        if not TICKER_RE.fullmatch(ticker):
            raise ValueError("invalid ticker: {}".format(raw))
        if ticker not in seen:
            result.append(ticker)
            seen.add(ticker)
    if not result:
        raise ValueError("at least one ticker is required")
    return tuple(result)


def parse_orats_price_history(
    rows: Iterable[Mapping[str, Any]],
    tickers: Iterable[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> PriceHistoryResult:
    """Parse split-adjusted OHLCV rows from ``hist/dailies``.

    Adjusted fields are canonical. Unadjusted fields are deliberately excluded
    because they can manufacture false trend signals around splits.
    """

    normalized = _normalize_tickers(tickers)
    requested = set(normalized)
    for label, value in (("start_date", start_date), ("end_date", end_date)):
        if value and not DATE_RE.fullmatch(value):
            raise ValueError("{} must be YYYY-MM-DD".format(label))
    if start_date and end_date and start_date > end_date:
        raise ValueError("start_date must not be after end_date")

    observations: Dict[str, List[PriceObservation]] = {
        ticker: [] for ticker in normalized
    }
    seen = set()
    for row in rows:
        ticker = str(row.get("ticker") or row.get("symbol") or "").strip().upper()
        session_date = str(row.get("tradeDate") or row.get("date") or "")[:10]
        if ticker not in requested:
            continue
        if not DATE_RE.fullmatch(session_date):
            raise PriceHistoryError("ORATS daily row has invalid tradeDate")
        if start_date and session_date < start_date:
            continue
        if end_date and session_date > end_date:
            continue
        key = (ticker, session_date)
        if key in seen:
            raise PriceHistoryError(
                "duplicate ORATS daily bar for {} {}".format(ticker, session_date)
            )
        seen.add(key)
        close = _number(row.get("clsPx"), "clsPx")
        open_price = _number(row.get("open"), "open")
        high = _number(row.get("hiPx"), "hiPx")
        low = _number(row.get("loPx"), "loPx")
        # Very old split-adjusted penny prices can be rounded to one or two
        # decimals, making adjusted low > adjusted close. Reconstruct the
        # intraday adjusted bar from the unadjusted bar and adjusted-close
        # factor when those fields are complete.
        try:
            unadjusted_close = _number(row.get("unadjClsPx"), "unadjClsPx")
            factor = close / unadjusted_close
            unadjusted_open = _number(row.get("unadjOpen"), "unadjOpen")
            unadjusted_high = _number(row.get("unadjHiPx"), "unadjHiPx")
            unadjusted_low = _number(row.get("unadjLoPx"), "unadjLoPx")
            if min(unadjusted_close, unadjusted_open, unadjusted_high, unadjusted_low) > 0:
                open_price = unadjusted_open * factor
                high = unadjusted_high * factor
                low = unadjusted_low * factor
        except PriceHistoryError:
            pass
        volume = _number(row.get("stockVolume"), "stockVolume")
        if min(open_price, high, low, close) <= 0 or volume < 0:
            raise PriceHistoryError("ORATS daily row has invalid adjusted OHLCV")
        envelope_normalized = False
        if high < max(open_price, close, low) - 1e-9 or low > min(open_price, close, high) + 1e-9:
            inconsistency = max(
                max(open_price, close) - high,
                low - min(open_price, close),
                0.0,
            ) / close
            if inconsistency > 0.02:
                raise PriceHistoryError("ORATS daily row has inconsistent adjusted OHLC beyond 2%")
            high = max(high, open_price, close)
            low = min(low, open_price, close)
            envelope_normalized = True
        observations[ticker].append(
            PriceObservation(
                session_date=session_date,
                ticker=ticker,
                open=open_price,
                close=close,
                high=high,
                low=low,
                volume=volume,
                ohlc_envelope_normalized=envelope_normalized,
            )
        )

    available = {
        ticker: tuple(sorted(values, key=lambda item: item.session_date))
        for ticker, values in observations.items()
        if values
    }
    if not available:
        raise PriceHistoryError(
            "none of the requested tickers have ORATS adjusted daily history"
        )
    return PriceHistoryResult(
        observations=available,
        requested_tickers=normalized,
        start_date=start_date,
        end_date=end_date,
    )
