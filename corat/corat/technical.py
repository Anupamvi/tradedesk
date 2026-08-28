"""Price, volume, AVWAP, and structure calculations.

No future bar is consulted by any feature function. Callers pass the exact
point-in-time slice they want evaluated.
"""

from __future__ import annotations

import math
import statistics
from datetime import date
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from corat.models import AvwapLevel, Bar, TechnicalSnapshot


def number(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def bars_from_dailies(rows: Iterable[Mapping[str, Any]]) -> Dict[str, List[Bar]]:
    grouped: Dict[str, Dict[str, Bar]] = {}
    for row in rows:
        ticker = str(row.get("ticker") or "").strip().upper()
        day = str(row.get("tradeDate") or "")[:10]
        close = number(row.get("clsPx"))
        high = number(row.get("hiPx"))
        low = number(row.get("loPx"))
        open_ = number(row.get("open"))
        volume = number(row.get("stockVolume"))
        if not ticker or len(day) != 10 or close is None or close <= 0:
            continue
        high = high if high is not None and high > 0 else close
        low = low if low is not None and low > 0 else close
        open_ = open_ if open_ is not None and open_ > 0 else close
        if high < low:
            continue
        grouped.setdefault(ticker, {})[day] = Bar(
            date=day,
            open=open_,
            high=high,
            low=low,
            close=close,
            volume=max(0.0, volume or 0.0),
            complete=True,
            updated_at=str(row.get("updatedAt") or ""),
            source="ORATS hist/dailies",
        )
    return {
        ticker: [by_day[key] for key in sorted(by_day)]
        for ticker, by_day in grouped.items()
    }


def append_core_spot(bars: Sequence[Bar], core: Mapping[str, Any], as_of: str) -> List[Bar]:
    result = [bar for bar in bars if bar.date <= as_of]
    trade_date = str(core.get("tradeDate") or "")[:10]
    if trade_date != as_of or (result and result[-1].date >= as_of):
        return result
    price = number(core.get("pxCls")) or number(core.get("pxAtmIv"))
    if price is None or price <= 0:
        return result
    prior = number(core.get("priorCls")) or (result[-1].close if result else price)
    result.append(
        Bar(
            date=as_of,
            open=prior,
            high=max(price, prior),
            low=min(price, prior),
            close=price,
            volume=max(0.0, number(core.get("stkVolu")) or 0.0),
            complete=False,
            updated_at=str(core.get("updatedAt") or ""),
            source="ORATS current core partial OHLC",
        )
    )
    return result


def sma(values: Sequence[float], period: int) -> Optional[float]:
    if len(values) < period or period <= 0:
        return None
    return sum(values[-period:]) / float(period)


def ema_series(values: Sequence[float], period: int) -> List[float]:
    if not values:
        return []
    alpha = 2.0 / float(period + 1)
    output = [float(values[0])]
    for value in values[1:]:
        output.append(alpha * float(value) + (1.0 - alpha) * output[-1])
    return output


def atr14(bars: Sequence[Bar]) -> Optional[float]:
    if len(bars) < 15:
        return None
    true_ranges: List[float] = []
    for prior, current in zip(bars[:-1], bars[1:]):
        true_ranges.append(
            max(
                current.high - current.low,
                abs(current.high - prior.close),
                abs(current.low - prior.close),
            )
        )
    if len(true_ranges) < 14:
        return None
    value = sum(true_ranges[:14]) / 14.0
    for current in true_ranges[14:]:
        value = ((value * 13.0) + current) / 14.0
    return value


def rsi14(values: Sequence[float]) -> Optional[float]:
    if len(values) < 15:
        return None
    changes = [right - left for left, right in zip(values[:-1], values[1:])]
    gains = [max(change, 0.0) for change in changes]
    losses = [max(-change, 0.0) for change in changes]
    avg_gain = sum(gains[:14]) / 14.0
    avg_loss = sum(losses[:14]) / 14.0
    for gain, loss in zip(gains[14:], losses[14:]):
        avg_gain = (avg_gain * 13.0 + gain) / 14.0
        avg_loss = (avg_loss * 13.0 + loss) / 14.0
    if avg_loss == 0:
        return 100.0
    strength = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + strength)


def return_n(values: Sequence[float], sessions: int) -> Optional[float]:
    if len(values) <= sessions or values[-sessions - 1] <= 0:
        return None
    return values[-1] / values[-sessions - 1] - 1.0


def _avwap_series(bars: Sequence[Bar], start_index: int) -> List[float]:
    weighted = 0.0
    volume = 0.0
    output: List[float] = []
    for bar in bars[start_index:]:
        bar_volume = max(0.0, bar.volume)
        if bar_volume <= 0:
            output.append(output[-1] if output else bar.close)
            continue
        typical = (bar.high + bar.low + bar.close) / 3.0
        weighted += typical * bar_volume
        volume += bar_volume
        output.append(weighted / volume)
    return output


def _anchor_candidates(bars: Sequence[Bar], last_earnings_date: str = "") -> List[Tuple[int, str]]:
    if not bars:
        return []
    anchors: List[Tuple[int, str]] = []
    current_year = bars[-1].date[:4]
    year_index = next((i for i, bar in enumerate(bars) if bar.date.startswith(current_year)), 0)
    anchors.append((year_index, "beginning of year"))
    if last_earnings_date:
        earnings_index = next((i for i, bar in enumerate(bars) if bar.date >= last_earnings_date), None)
        if earnings_index is not None:
            anchors.append((earnings_index, "most recent earnings"))
    window_start = max(0, len(bars) - 60)
    recent = list(enumerate(bars[window_start:], start=window_start))
    if recent:
        anchors.append((min(recent, key=lambda pair: pair[1].low)[0], "major 60-session swing low"))
        anchors.append((max(recent, key=lambda pair: pair[1].volume)[0], "highest-volume day in 60 sessions"))
    gap_candidates: List[Tuple[float, int]] = []
    for index in range(max(1, window_start), len(bars)):
        prior = bars[index - 1].close
        if prior > 0:
            gap_candidates.append((abs(bars[index].open / prior - 1.0), index))
    if gap_candidates:
        gap, index = max(gap_candidates)
        if gap >= 0.02:
            anchors.append((index, "largest material gap in 60 sessions"))
    breakout_index = None
    for index in range(max(20, window_start), len(bars)):
        prior_high = max(bar.high for bar in bars[index - 20 : index])
        if bars[index].close > prior_high and bars[index].volume > 0:
            breakout_index = index
    if breakout_index is not None:
        anchors.append((breakout_index, "most recent 20-session breakout"))
    unique: List[Tuple[int, str]] = []
    seen = set()
    for index, reason in anchors:
        if index not in seen:
            seen.add(index)
            unique.append((index, reason))
    return unique[:5]


def avwap_levels(bars: Sequence[Bar], last_earnings_date: str = "") -> List[AvwapLevel]:
    levels: List[AvwapLevel] = []
    for index, reason in _anchor_candidates(bars, last_earnings_date=last_earnings_date):
        series = _avwap_series(bars, index)
        if not series:
            continue
        slope = None
        if len(series) >= 6 and series[-6] != 0:
            slope = series[-1] / series[-6] - 1.0
        levels.append(AvwapLevel(bars[index].date, reason, series[-1], slope))
    return levels


def _pivots(bars: Sequence[Bar]) -> Tuple[List[float], List[float]]:
    lows: List[float] = []
    highs: List[float] = []
    start = max(2, len(bars) - 120)
    for index in range(start, len(bars) - 2):
        window = bars[index - 2 : index + 3]
        if bars[index].low <= min(item.low for item in window):
            lows.append(bars[index].low)
        if bars[index].high >= max(item.high for item in window):
            highs.append(bars[index].high)
    return lows, highs


def technical_snapshot(
    ticker: str,
    bars: Sequence[Bar],
    as_of: str,
    last_earnings_date: str = "",
) -> Optional[TechnicalSnapshot]:
    eligible = [bar for bar in bars if bar.date <= as_of]
    if len(eligible) < 21:
        return None
    closes = [bar.close for bar in eligible]
    ema20_values = ema_series(closes, 20)
    ema20 = ema20_values[-1] if ema20_values else None
    average50 = sma(closes, 50)
    average200 = sma(closes, 200)
    atr = atr14(eligible)
    volumes = [bar.volume for bar in eligible[-20:] if bar.volume > 0]
    prior_volumes = [bar.volume for bar in eligible[-21:-1] if bar.volume > 0]
    relative_volume = None
    if prior_volumes and eligible[-1].volume > 0:
        relative_volume = eligible[-1].volume / (sum(prior_volumes) / len(prior_volumes))
    average_dollar_volume = None
    liquid_bars = [bar for bar in eligible[-20:] if bar.volume > 0]
    if liquid_bars:
        average_dollar_volume = sum(bar.close * bar.volume for bar in liquid_bars) / len(liquid_bars)
    prior20 = eligible[-21:-1]
    prior_high = max(bar.high for bar in prior20) if prior20 else None
    prior_low = min(bar.low for bar in prior20) if prior20 else None
    avwaps = avwap_levels(eligible, last_earnings_date=last_earnings_date)
    pivot_lows, pivot_highs = _pivots(eligible)
    price = eligible[-1].close
    below = [value for value in pivot_lows if value < price]
    below.extend(value for value in (ema20, average50) if value is not None and value < price)
    below.extend(level.value for level in avwaps if level.value < price)
    above = [value for value in pivot_highs if value > price]
    if prior_high is not None and prior_high > price:
        above.append(prior_high)
    support = max(below) if below else prior_low
    resistance = min(above) if above else None
    extension = None
    if atr and ema20 is not None and atr > 0:
        extension = (price - ema20) / atr
    return TechnicalSnapshot(
        ticker=ticker,
        as_of=as_of,
        price=price,
        price_date=eligible[-1].date,
        price_complete=eligible[-1].complete,
        ema20=ema20,
        sma50=average50,
        sma200=average200,
        atr14=atr,
        rsi14=rsi14(closes),
        return_5d=return_n(closes, 5),
        return_20d=return_n(closes, 20),
        return_60d=return_n(closes, 60),
        relative_volume_20d=relative_volume,
        average_dollar_volume_20d=average_dollar_volume,
        prior_high_20d=prior_high,
        prior_low_20d=prior_low,
        support=support,
        resistance=resistance,
        extension_from_ema_atr=extension,
        avwaps=avwaps,
        price_source=eligible[-1].source,
        price_updated_at=eligible[-1].updated_at,
    )


def aligned_return(snapshot: Optional[TechnicalSnapshot], sessions: int) -> Optional[float]:
    if snapshot is None:
        return None
    return {5: snapshot.return_5d, 20: snapshot.return_20d, 60: snapshot.return_60d}.get(sessions)
