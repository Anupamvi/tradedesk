"""Price structure: EMA/SMA/ATR/AVWAP/RS/volume. RSI is supporting context only."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from groat.config import ATR_N
from groat.num import pct_change, to_float


def bars_through(bars: Sequence[dict], asof: str) -> List[dict]:
    return [b for b in bars if str(b.get("date") or "")[:10] <= asof]


def closes(bars: Sequence[dict]) -> List[float]:
    out = []
    for bar in bars:
        c = to_float(bar.get("close"))
        if c is not None:
            out.append(c)
    return out


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


def true_range(bar: dict, prev_close: float) -> float:
    high = float(bar["high"])
    low = float(bar["low"])
    return max(high - low, abs(high - prev_close), abs(low - prev_close))


def atr_wilder(bars: Sequence[dict], asof: str, n: int = ATR_N) -> Optional[float]:
    upto = bars_through(bars, asof)
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


def rsi(values: Sequence[float], n: int = 14) -> Optional[float]:
    if len(values) < n + 1:
        return None
    gains = []
    losses = []
    for i in range(1, len(values)):
        diff = values[i] - values[i - 1]
        gains.append(max(diff, 0.0))
        losses.append(max(-diff, 0.0))
    if len(gains) < n:
        return None
    avg_g = sum(gains[:n]) / float(n)
    avg_l = sum(losses[:n]) / float(n)
    for g, l in zip(gains[n:], losses[n:]):
        avg_g = (avg_g * (n - 1) + g) / float(n)
        avg_l = (avg_l * (n - 1) + l) / float(n)
    if avg_l == 0:
        return 100.0
    rs = avg_g / avg_l
    return 100.0 - (100.0 / (1.0 + rs))


def ret(values: Sequence[float], n: int) -> Optional[float]:
    if len(values) < n + 1:
        return None
    return pct_change(values[-1], values[-1 - n])


def relative_strength(stock: Sequence[float], bench: Sequence[float], n: int) -> Optional[float]:
    a = ret(stock, n)
    b = ret(bench, n)
    if a is None or b is None:
        return None
    return a - b


def typical_price(bar: dict) -> Optional[float]:
    h = to_float(bar.get("high"))
    low = to_float(bar.get("low"))
    c = to_float(bar.get("close"))
    if h is None or low is None or c is None:
        return None
    return (h + low + c) / 3.0


def avwap(bars: Sequence[dict], asof: str, anchor: str) -> Optional[float]:
    """Volume-weighted typical price from anchor date through asof. None if no volume."""
    window = [b for b in bars_through(bars, asof) if str(b.get("date") or "")[:10] >= anchor]
    if not window:
        return None
    cum_pv = 0.0
    cum_v = 0.0
    for bar in window:
        tp = typical_price(bar)
        vol = to_float(bar.get("volume"))
        if tp is None or vol is None or vol <= 0:
            continue
        cum_pv += tp * vol
        cum_v += vol
    if cum_v <= 0:
        return None
    return cum_pv / cum_v


def pivot_low_date(bars: Sequence[dict], asof: str, lookback: int = 60, wing: int = 5) -> Optional[str]:
    upto = bars_through(bars, asof)
    if len(upto) < wing * 2 + 1:
        return None
    window = upto[-lookback:] if len(upto) > lookback else upto
    best = None
    best_low = None
    for i in range(wing, len(window) - wing):
        low = to_float(window[i].get("low"))
        if low is None:
            continue
        left = [to_float(window[j].get("low")) for j in range(i - wing, i)]
        right = [to_float(window[j].get("low")) for j in range(i + 1, i + wing + 1)]
        if any(v is None or v < low for v in left + right):
            continue
        if best_low is None or low <= best_low:
            best_low = low
            best = str(window[i].get("date") or "")[:10]
    return best


def pivot_high_date(bars: Sequence[dict], asof: str, lookback: int = 60, wing: int = 5) -> Optional[str]:
    upto = bars_through(bars, asof)
    if len(upto) < wing * 2 + 1:
        return None
    window = upto[-lookback:] if len(upto) > lookback else upto
    best = None
    best_high = None
    for i in range(wing, len(window) - wing):
        high = to_float(window[i].get("high"))
        if high is None:
            continue
        left = [to_float(window[j].get("high")) for j in range(i - wing, i)]
        right = [to_float(window[j].get("high")) for j in range(i + 1, i + wing + 1)]
        if any(v is None or v > high for v in left + right):
            continue
        if best_high is None or high >= best_high:
            best_high = high
            best = str(window[i].get("date") or "")[:10]
    return best


def prior_high(bars: Sequence[dict], asof: str, lookback: int = 20, field: str = "close") -> Optional[float]:
    upto = bars_through(bars, asof)
    if not upto or upto[-1]["date"] != asof:
        return None
    prior = upto[:-1]
    if len(prior) < lookback:
        return None
    vals = [to_float(b.get(field)) for b in prior[-lookback:]]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return max(vals)


def prior_low(bars: Sequence[dict], asof: str, lookback: int = 20, field: str = "low") -> Optional[float]:
    upto = bars_through(bars, asof)
    if not upto or upto[-1]["date"] != asof:
        return None
    prior = upto[:-1]
    if len(prior) < lookback:
        return None
    vals = [to_float(b.get(field)) for b in prior[-lookback:]]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return min(vals)


def volume_avg(bars: Sequence[dict], n: int) -> Optional[float]:
    vals = [to_float(b.get("volume")) for b in bars[-n:]]
    vals = [v for v in vals if v is not None and v > 0]
    if len(vals) < max(3, n // 2):
        return None
    return sum(vals) / float(len(vals))


def year_start(asof: str) -> str:
    return asof[:4] + "-01-01"


def snapshot(bars: Sequence[dict], asof: str, bench_bars: Optional[Sequence[dict]] = None) -> Dict[str, object]:
    upto = bars_through(bars, asof)
    empty = {
        "asof": asof,
        "ok": False,
        "reason": "missing_bars",
        "date": upto[-1]["date"] if upto else "",
        "stale": True,
    }
    if not upto:
        return empty
    last = upto[-1]
    stale = last["date"] != asof
    px = to_float(last.get("close"))
    if px is None:
        return empty
    c = closes(upto)
    ema20 = ema(c, 20)
    ema20_prev = ema(c[:-1], 20) if len(c) > 21 else None
    sma50 = sma(c, 50)
    sma200 = sma(c, 200)
    atr = atr_wilder(upto, last["date"])
    rvol = None
    v20 = volume_avg(upto, 20)
    v5 = volume_avg(upto, 5)
    today_v = to_float(last.get("volume"))
    if v20 and today_v:
        rvol = today_v / v20
    swing_low = pivot_low_date(upto, last["date"])
    swing_high = pivot_high_date(upto, last["date"])
    ystart = year_start(asof)
    avwap_year = avwap(upto, last["date"], ystart)
    avwap_low = avwap(upto, last["date"], swing_low) if swing_low else None
    avwap_high = avwap(upto, last["date"], swing_high) if swing_high else None
    hi20 = prior_high(upto, last["date"], 20, "high")
    lo20 = prior_low(upto, last["date"], 20, "low")
    hi20_close = prior_high(upto, last["date"], 20, "close")
    extension = None
    if atr and ema20:
        extension = (px - ema20) / atr
    ema_rising = None
    if ema20 is not None and ema20_prev is not None:
        ema_rising = ema20 > ema20_prev
    bench_c = closes(bars_through(bench_bars, last["date"])) if bench_bars is not None else []
    out = {
        "asof": asof,
        "date": last["date"],
        "ok": True,
        "stale": stale,
        "reason": "stale_price" if stale else "",
        "close": px,
        "open": to_float(last.get("open")),
        "high": to_float(last.get("high")),
        "low": to_float(last.get("low")),
        "volume": today_v,
        "ema20": ema20,
        "sma50": sma50,
        "sma200": sma200,
        "ema20_rising": ema_rising,
        "atr14": atr,
        "rsi14": rsi(c, 14),
        "ret_1": ret(c, 1),
        "ret_2": ret(c, 2),
        "ret_5": ret(c, 5),
        "ret_20": ret(c, 20),
        "ret_60": ret(c, 60),
        "rs_5": relative_strength(c, bench_c, 5) if bench_c else None,
        "rs_20": relative_strength(c, bench_c, 20) if bench_c else None,
        "rs_60": relative_strength(c, bench_c, 60) if bench_c else None,
        "rvol": rvol,
        "vol_5": v5,
        "vol_20": v20,
        "hi20": hi20,
        "lo20": lo20,
        "hi20_close": hi20_close,
        "extension_atr": extension,
        "swing_low_date": swing_low,
        "swing_high_date": swing_high,
        "avwap_year": avwap_year,
        "avwap_swing_low": avwap_low,
        "avwap_swing_high": avwap_high,
        "above_ema20": px > ema20 if ema20 is not None else None,
        "above_sma50": px > sma50 if sma50 is not None else None,
        "above_sma200": px > sma200 if sma200 is not None else None,
        "trend": _trend(px, ema20, sma50, sma200, ema_rising),
    }
    return out


def _trend(px, ema20, sma50, sma200, ema_rising) -> str:
    if px is None or ema20 is None or sma50 is None:
        return "unknown"
    if sma200 is not None and px > ema20 > sma50 > sma200 and ema_rising:
        return "strong_up"
    if px > sma50 and (sma200 is None or px > sma200):
        return "up"
    if sma200 is not None and px < ema20 < sma50 < sma200 and ema_rising is False:
        return "strong_down"
    if px < sma50 and (sma200 is None or px < sma200):
        return "down"
    return "range"
