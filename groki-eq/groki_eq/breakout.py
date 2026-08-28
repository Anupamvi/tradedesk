"""20-session close high breakout and Wilder ATR(14)."""

import math
from typing import List, Optional

from groki_eq.config import (
    ACCOUNT_DOLLARS,
    ATR_N,
    HIGH_LOOKBACK,
    RISK_PCT,
    STOP_ATR_MULT,
)


def bars_through(bars: List[dict], asof: str) -> List[dict]:
    return [b for b in bars if str(b.get("date") or "")[:10] <= asof]


def prior_high(bars: List[dict], asof: str, lookback: int = HIGH_LOOKBACK) -> Optional[float]:
    upto = bars_through(bars, asof)
    if not upto or upto[-1]["date"] != asof:
        return None
    prior = upto[:-1]
    if len(prior) < lookback:
        return None
    window = prior[-lookback:]
    return max(float(b["close"]) for b in window)


def true_range(bar: dict, prev_close: float) -> float:
    high = float(bar["high"])
    low = float(bar["low"])
    return max(high - low, abs(high - prev_close), abs(low - prev_close))


def atr_wilder(bars: List[dict], asof: str, n: int = ATR_N) -> Optional[float]:
    upto = bars_through(bars, asof)
    if not upto or upto[-1]["date"] != asof:
        return None
    if len(upto) < n + 1:
        return None
    trs = []
    for i in range(1, len(upto)):
        trs.append(true_range(upto[i], float(upto[i - 1]["close"])))
    if len(trs) < n:
        return None
    atr = sum(trs[:n]) / float(n)
    for tr in trs[n:]:
        atr = (atr * (n - 1) + tr) / float(n)
    return atr


def is_breakout(close: float, high_20: float) -> bool:
    return close > high_20


def pct_above(close: float, high_20: float) -> float:
    return close / high_20 - 1.0


def stop_price(entry: float, atr: float) -> float:
    return entry - STOP_ATR_MULT * atr


def share_count(atr: float) -> int:
    risk_per_share = STOP_ATR_MULT * atr
    if risk_per_share <= 0:
        return 0
    budget = ACCOUNT_DOLLARS * RISK_PCT
    return int(math.floor(budget / risk_per_share))
