"""Daily tape helpers and weekly resample. Leakage-safe: bars through asof only."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence

from trendbook.config import ATR_N, CHASE_ATR
from trendbook.num import pct_change, to_float


def bars_through(bars: Sequence[dict], asof: str) -> List[dict]:
    cut = str(asof or "")[:10]
    return [b for b in bars if str(b.get("date") or "")[:10] <= cut]


def closes(bars: Sequence[dict]) -> List[float]:
    out = []
    for bar in bars:
        c = to_float(bar.get("close"))
        if c is not None:
            out.append(c)
    return out


def volume_expand(weekly: Sequence[dict], n: int = 10, multiple: float = 1.2) -> Optional[bool]:
    """True when the last week's volume is >= multiple times the prior n-week mean."""
    if n <= 0 or len(weekly) < n + 1:
        return None
    last = to_float(weekly[-1].get("volume"))
    prior = [to_float(w.get("volume")) for w in weekly[-(n + 1) : -1]]
    prior = [v for v in prior if v is not None and v > 0]
    if last is None or last <= 0 or len(prior) < max(6, n - 2):
        return None
    mean = sum(prior) / float(len(prior))
    if mean <= 0:
        return None
    return last >= multiple * mean


def sma(values: Sequence[float], n: int) -> Optional[float]:
    if n <= 0 or len(values) < n:
        return None
    window = values[-n:]
    return sum(window) / float(n)


def ema(values: Sequence[float], n: int) -> Optional[float]:
    if n <= 0 or len(values) < n:
        return None
    seed = sum(values[:n]) / float(n)
    k = 2.0 / (n + 1.0)
    current = seed
    for value in values[n:]:
        current = value * k + current * (1.0 - k)
    return current


def ret(values: Sequence[float], n: int) -> Optional[float]:
    if len(values) < n + 1:
        return None
    return pct_change(values[-1], values[-1 - n])


def true_range(bar: dict, prev_close: float) -> float:
    high = float(bar["high"])
    low = float(bar["low"])
    return max(high - low, abs(high - prev_close), abs(low - prev_close))


def atr_wilder(bars: Sequence[dict], n: int = ATR_N) -> Optional[float]:
    if len(bars) < n + 1:
        return None
    trs = []
    for i in range(1, len(bars)):
        trs.append(true_range(bars[i], float(bars[i - 1]["close"])))
    if len(trs) < n:
        return None
    atr = sum(trs[:n]) / float(n)
    for tr in trs[n:]:
        atr = (atr * (n - 1) + tr) / float(n)
    return atr


def rolling_extreme(bars: Sequence[dict], n: int, field: str, want_max: bool) -> Optional[float]:
    window = bars[-n:] if len(bars) >= n else bars
    vals = [to_float(b.get(field)) for b in window]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return max(vals) if want_max else min(vals)


def to_weekly(daily: Sequence[dict], asof: str) -> List[dict]:
    """Resample daily OHLCV to weeks ending Friday. Only bars through asof."""
    rows = bars_through(daily, asof)
    buckets: Dict[str, List[dict]] = {}
    for bar in rows:
        day = str(bar.get("date") or "")[:10]
        try:
            d = datetime.strptime(day, "%Y-%m-%d")
        except ValueError:
            continue
        days_to_fri = (4 - d.weekday()) % 7
        week_end = (d + timedelta(days=days_to_fri)).date().isoformat()
        buckets.setdefault(week_end, []).append(bar)
    out = []
    for week_end in sorted(buckets):
        group = buckets[week_end]
        highs = [to_float(b.get("high")) for b in group]
        lows = [to_float(b.get("low")) for b in group]
        vols = [to_float(b.get("volume")) or 0.0 for b in group]
        highs = [h for h in highs if h is not None]
        lows = [lo for lo in lows if lo is not None]
        o = to_float(group[0].get("open"))
        c = to_float(group[-1].get("close"))
        if o is None or c is None or not highs or not lows:
            continue
        out.append(
            {
                "date": str(group[-1].get("date") or "")[:10],
                "week_end": week_end,
                "open": o,
                "high": max(highs),
                "low": min(lows),
                "close": c,
                "volume": sum(vols),
            }
        )
    return out


def week_ends(weekly: Sequence[dict], asof: str) -> List[str]:
    return [str(w.get("date") or "")[:10] for w in weekly if str(w.get("date") or "")[:10] <= asof[:10]]


def daily_snapshot(bars: Sequence[dict], asof: str) -> dict:
    upto = bars_through(bars, asof)
    if not upto:
        return {"ok": False, "asof": asof, "reason": "missing_bars"}
    last = upto[-1]
    px = to_float(last.get("close"))
    if px is None:
        return {"ok": False, "asof": asof, "reason": "missing_close"}
    c = closes(upto)
    ema20 = ema(c, 20)
    sma50 = sma(c, 50)
    sma150 = sma(c, 150)
    sma200 = sma(c, 200)
    sma200_prev = sma(c[:-21], 200) if len(c) > 221 else None
    atr = atr_wilder(upto)
    ext = None
    if atr and ema20 and atr > 0:
        ext = (px - ema20) / atr
    hi20 = rolling_extreme(upto, 20, "high", True)
    lo20 = rolling_extreme(upto, 20, "low", False)
    range20 = None
    if hi20 is not None and lo20 is not None and px:
        range20 = (hi20 - lo20) / px
    hi5 = rolling_extreme(upto, 5, "high", True)
    lo5 = rolling_extreme(upto, 5, "low", False)
    range5 = None
    if hi5 is not None and lo5 is not None and px:
        range5 = (hi5 - lo5) / px
    tightness = None
    if range20 and range20 > 0 and range5 is not None:
        tightness = range5 / range20
    return {
        "ok": True,
        "asof": asof,
        "date": str(last.get("date") or "")[:10],
        "close": px,
        "ema20": ema20,
        "sma50": sma50,
        "sma150": sma150,
        "sma200": sma200,
        "sma200_prev_21": sma200_prev,
        "atr14": atr,
        "extension_atr": ext,
        "chase": ext is not None and ext > CHASE_ATR,
        "hi252": rolling_extreme(upto, 252, "high", True),
        "lo252": rolling_extreme(upto, 252, "low", False),
        "ret_63": ret(c, 63),
        "ret_126": ret(c, 126),
        "ret_189": ret(c, 189),
        "ret_252": ret(c, 252),
        "range20": range20,
        "range5": range5,
        "tightness": tightness,
        "n_bars": len(upto),
    }
