"""Outcome-only ORATS daily bars and exact next-session stock labels."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


class LabelDataError(ValueError):
    pass


def _number(value: Any, label: str) -> float:
    if value is None or isinstance(value, bool):
        raise LabelDataError("missing {}".format(label))
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise LabelDataError("invalid {}".format(label)) from None
    if not math.isfinite(result):
        raise LabelDataError("non-finite {}".format(label))
    return result


@dataclass(frozen=True)
class DailyBar:
    ticker: str
    trade_date: str
    open: float
    high: float
    low: float
    close: float
    unadjusted_close: Optional[float] = None

    def __post_init__(self) -> None:
        try:
            date.fromisoformat(self.trade_date)
        except ValueError:
            raise LabelDataError("trade_date must be YYYY-MM-DD") from None
        if not self.ticker or self.ticker != self.ticker.upper():
            raise LabelDataError("ticker must be uppercase")
        values = (self.open, self.high, self.low, self.close)
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise LabelDataError("daily OHLC values must be finite and positive")
        if self.high < max(self.open, self.close, self.low):
            raise LabelDataError("daily high is inconsistent")
        if self.low > min(self.open, self.close, self.high):
            raise LabelDataError("daily low is inconsistent")
        if self.unadjusted_close is not None and (
            not math.isfinite(self.unadjusted_close) or self.unadjusted_close <= 0
        ):
            raise LabelDataError("unadjusted close must be finite and positive")

    @classmethod
    def from_orats(cls, row: Mapping[str, Any]) -> "DailyBar":
        ticker = str(row.get("ticker") or row.get("symbol") or "").strip().upper()
        trade_date = str(row.get("tradeDate") or row.get("date") or "")[:10]
        adjusted_close = _number(row.get("clsPx"), "clsPx")
        adjusted_open = _number(row.get("open"), "open")
        adjusted_high = _number(row.get("hiPx"), "hiPx")
        adjusted_low = _number(row.get("loPx"), "loPx")
        try:
            unadjusted_close = _number(row.get("unadjClsPx"), "unadjClsPx")
            factor = adjusted_close / unadjusted_close
            unadjusted_open = _number(row.get("unadjOpen"), "unadjOpen")
            unadjusted_high = _number(row.get("unadjHiPx"), "unadjHiPx")
            unadjusted_low = _number(row.get("unadjLoPx"), "unadjLoPx")
            if min(unadjusted_close, unadjusted_open, unadjusted_high, unadjusted_low) > 0:
                adjusted_open = unadjusted_open * factor
                adjusted_high = unadjusted_high * factor
                adjusted_low = unadjusted_low * factor
        except LabelDataError:
            pass
        return cls(
            ticker=ticker,
            trade_date=trade_date,
            open=adjusted_open,
            high=max(adjusted_high, adjusted_open, adjusted_close),
            low=min(adjusted_low, adjusted_open, adjusted_close),
            close=adjusted_close,
            unadjusted_close=(
                _number(row.get("unadjClsPx"), "unadjClsPx")
                if row.get("unadjClsPx") not in (None, "")
                else None
            ),
        )


def parse_orats_daily_rows(
    rows: Iterable[Mapping[str, Any]],
    tickers: Optional[Iterable[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Mapping[str, Sequence[DailyBar]]:
    allowed = {ticker.strip().upper() for ticker in tickers or ()}
    by_ticker: Dict[str, List[DailyBar]] = {}
    seen = set()
    for row in rows:
        raw_date = str(row.get("tradeDate") or row.get("date") or "")[:10]
        if start_date and raw_date < start_date:
            continue
        if end_date and raw_date > end_date:
            continue
        raw_ticker = str(row.get("ticker") or row.get("symbol") or "").strip().upper()
        if allowed and raw_ticker not in allowed:
            continue
        bar = DailyBar.from_orats(row)
        key = (bar.ticker, bar.trade_date)
        if key in seen:
            raise LabelDataError("duplicate ORATS daily bar for {} {}".format(*key))
        seen.add(key)
        by_ticker.setdefault(bar.ticker, []).append(bar)
    return {
        ticker: tuple(sorted(values, key=lambda item: item.trade_date))
        for ticker, values in sorted(by_ticker.items())
    }


@dataclass(frozen=True)
class StockOutcome:
    sample_id: str
    ticker: str
    side: str
    decision_date: str
    entry_date: str
    label_end_date: str
    horizon_sessions: int
    entry_price: float
    exit_price: float
    gross_return: float
    round_trip_cost_bps: float
    net_return: float
    maximum_favorable_excursion: float
    maximum_adverse_excursion: float
    label_source: str = "ORATS_HIST_DAILIES_ADJUSTED_OUTCOME_ONLY"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def exact_next_open_outcome(
    sample_id: str,
    ticker: str,
    side: str,
    decision_date: str,
    bars: Sequence[DailyBar],
    horizon_sessions: int,
    round_trip_cost_bps: float = 10.0,
) -> StockOutcome:
    """Enter at the next observed adjusted open and exit at horizon close."""

    normalized_side = side.strip().upper()
    if normalized_side not in {"LONG", "SHORT"}:
        raise LabelDataError("side must be LONG or SHORT")
    if horizon_sessions <= 0:
        raise LabelDataError("horizon_sessions must be positive")
    if round_trip_cost_bps < 0:
        raise LabelDataError("round_trip_cost_bps cannot be negative")
    ordered = tuple(sorted(bars, key=lambda item: item.trade_date))
    if any(item.ticker != ticker.upper() for item in ordered):
        raise LabelDataError("bars contain a different ticker")
    dates = [item.trade_date for item in ordered]
    if len(dates) != len(set(dates)):
        raise LabelDataError("bars contain duplicate dates")
    try:
        decision_index = dates.index(decision_date)
    except ValueError:
        raise LabelDataError("decision date has no ORATS adjusted daily bar") from None
    entry_index = decision_index + 1
    exit_index = entry_index + horizon_sessions - 1
    if exit_index >= len(ordered):
        raise LabelDataError("outcome horizon is not yet complete")
    entry = ordered[entry_index]
    exit_bar = ordered[exit_index]
    window = ordered[entry_index : exit_index + 1]
    direction = 1.0 if normalized_side == "LONG" else -1.0
    gross = direction * (exit_bar.close / entry.open - 1.0)
    net = gross - round_trip_cost_bps / 10_000.0
    if normalized_side == "LONG":
        mfe = max(item.high / entry.open - 1.0 for item in window)
        mae = min(item.low / entry.open - 1.0 for item in window)
    else:
        mfe = max((entry.open - item.low) / entry.open for item in window)
        mae = min((entry.open - item.high) / entry.open for item in window)
    return StockOutcome(
        sample_id=sample_id,
        ticker=ticker.upper(),
        side=normalized_side,
        decision_date=decision_date,
        entry_date=entry.trade_date,
        label_end_date=exit_bar.trade_date,
        horizon_sessions=horizon_sessions,
        entry_price=entry.open,
        exit_price=exit_bar.close,
        gross_return=gross,
        round_trip_cost_bps=round_trip_cost_bps,
        net_return=net,
        maximum_favorable_excursion=mfe,
        maximum_adverse_excursion=mae,
    )
