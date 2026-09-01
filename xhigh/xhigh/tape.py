"""Schwab daily bars: EMA, ATR, chase, downtrend."""

from __future__ import annotations

from typing import Dict, List, Optional

from xhigh.num import to_float


def ema(values: List[float], span: int) -> Optional[float]:
    if not values or span <= 0 or len(values) < span:
        return None
    k = 2.0 / (span + 1)
    current = sum(values[:span]) / float(span)
    for price in values[span:]:
        current = price * k + current * (1 - k)
    return current


def atr(bars: List[dict], n: int = 14) -> Optional[float]:
    if len(bars) < n + 1:
        return None
    trs = []
    for i in range(1, len(bars)):
        high = to_float(bars[i].get("high"))
        low = to_float(bars[i].get("low"))
        prev = to_float(bars[i - 1].get("close"))
        if high is None or low is None or prev is None:
            continue
        trs.append(max(high - low, abs(high - prev), abs(low - prev)))
    if len(trs) < n:
        return None
    return sum(trs[-n:]) / float(n)


def snapshot(bars: List[dict], atr_n: int = 14) -> Dict[str, object]:
    closes = [to_float(b.get("close")) for b in bars if to_float(b.get("close")) is not None]
    last = closes[-1] if closes else None
    e20 = ema(closes, 20)
    e50 = ema(closes, 50)
    a = atr(bars, atr_n)
    ext = None
    if last is not None and e20 is not None and a and a > 0:
        ext = (last - e20) / a
    trend_up = bool(last is not None and e20 is not None and e50 is not None and last > e20 > e50)
    trend_down = bool(last is not None and e20 is not None and e50 is not None and last < e20 < e50)
    return {
        "close": last,
        "ema20": e20,
        "ema50": e50,
        "atr": a,
        "extension_atr": ext,
        "trend_up": trend_up,
        "trend_down": trend_down,
    }


def chase(snap: dict, chase_atr: float = 2.5) -> bool:
    ext = to_float(snap.get("extension_atr"))
    return bool(ext is not None and ext > chase_atr)
