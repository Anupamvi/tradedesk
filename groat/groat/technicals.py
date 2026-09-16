"""Price structure: EMA/SMA/ATR/AVWAP/RS/volume. RSI is supporting context only."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from groat.config import ATR_N, INCOMPLETE_RVOL
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


def session_bar_incomplete(upto: Sequence[dict], asof: str) -> bool:
    """True when asof's bar is a live stub vs prior 20d volume. Not a completed session."""
    if not upto:
        return False
    last = upto[-1]
    if str(last.get("date") or "")[:10] != asof or len(upto) < 2:
        return False
    v20 = volume_avg(upto[:-1], 20)
    if not v20:
        return False
    today_v = to_float(last.get("volume"))
    if today_v is None:
        return True
    return today_v / v20 < INCOMPLETE_RVOL


def snapshot(bars: Sequence[dict], asof: str, bench_bars: Optional[Sequence[dict]] = None) -> Dict[str, object]:
    upto = bars_through(bars, asof)
    empty = {
        "asof": asof,
        "ok": False,
        "reason": "missing_bars",
        "date": upto[-1]["date"] if upto else "",
        "stale": True,
        "session_incomplete": False,
    }
    if not upto:
        return empty
    last = upto[-1]
    incomplete = session_bar_incomplete(upto, asof)
    struct = list(upto[:-1]) if incomplete else list(upto)
    if not struct:
        return empty
    out = _snapshot_from_bars(struct, struct[-1]["date"], bench_bars)
    if not out.get("ok"):
        return out
    out["asof"] = asof
    out["session_incomplete"] = incomplete
    out["structure_date"] = struct[-1]["date"]
    out["structure_close"] = out.get("close")
    if not incomplete:
        out["stale"] = last["date"] != asof
        out["reason"] = "stale_price" if out["stale"] else ""
        return out
    # Live last for fills/click/chase. Setups keep structure close/ret/hi20/RS.
    live_px = to_float(last.get("close"))
    out["stale"] = False
    out["reason"] = ""
    out["live_last"] = live_px
    out["live_open"] = to_float(last.get("open"))
    out["live_high"] = to_float(last.get("high"))
    out["live_low"] = to_float(last.get("low"))
    out["live_ret_1"] = pct_change(live_px, out.get("structure_close"))
    today_v = to_float(last.get("volume"))
    out["volume"] = today_v
    v20 = out.get("vol_20")
    out["rvol"] = (today_v / v20) if (v20 and today_v) else None
    atr = to_float(out.get("atr14"))
    ema20 = to_float(out.get("ema20"))
    sma50 = to_float(out.get("sma50"))
    sma200 = to_float(out.get("sma200"))
    if live_px is not None and atr and ema20:
        out["extension_atr"] = (live_px - ema20) / atr
    if live_px is not None:
        out["above_ema20"] = live_px > ema20 if ema20 is not None else None
        out["above_sma50"] = live_px > sma50 if sma50 is not None else None
        out["above_sma200"] = live_px > sma200 if sma200 is not None else None
        out["trend"] = _trend(live_px, ema20, sma50, sma200, out.get("ema20_rising"))
    return out


def _snapshot_from_bars(
    upto: Sequence[dict],
    asof: str,
    bench_bars: Optional[Sequence[dict]] = None,
) -> Dict[str, object]:
    empty = {
        "asof": asof,
        "ok": False,
        "reason": "missing_bars",
        "date": upto[-1]["date"] if upto else "",
        "stale": True,
        "session_incomplete": False,
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
    return {
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
        "session_incomplete": False,
        "structure_date": last["date"],
        "structure_close": px,
    }


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
