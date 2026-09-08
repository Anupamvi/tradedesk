"""Weinstein weekly stage from the 30-week moving average."""

from __future__ import annotations

from typing import Optional, Sequence

from trendbook.bars import sma
from trendbook.config import STAGE_WEEKS
from trendbook.num import to_float


def classify_stage(weekly: Sequence[dict]) -> dict:
    if len(weekly) < STAGE_WEEKS:
        return {
            "stage": None,
            "ok": False,
            "reason": "need_30_weeks",
            "ma30": None,
            "slope": None,
            "hh_hl": False,
            "ma10": sma([float(w["close"]) for w in weekly], 10) if len(weekly) >= 10 else None,
        }
    closes = [to_float(w.get("close")) for w in weekly]
    closes = [c for c in closes if c is not None]
    ma30 = sma(closes, STAGE_WEEKS)
    lookback = 5 if len(closes) >= STAGE_WEEKS + 4 else 1
    ma_prev = sma(closes[:-lookback], STAGE_WEEKS)
    px = closes[-1]
    slope = None
    if ma30 is not None and ma_prev is not None:
        slope = ma30 - ma_prev
    rising = slope is not None and slope > 0
    falling = slope is not None and slope < 0
    if ma30 is None:
        stage = None
        reason = "ma30_unavailable"
    elif px > ma30 and rising:
        stage = 2
        reason = "above_rising_30w"
    elif px < ma30 and falling:
        stage = 4
        reason = "below_falling_30w"
    elif px > ma30:
        stage = 3
        reason = "above_flat_or_falling_30w"
    else:
        stage = 1
        reason = "below_flat_or_rising_30w"
    hh_hl = _higher_highs_lows(weekly)
    return {
        "stage": stage,
        "ok": stage is not None,
        "reason": reason,
        "ma30": ma30,
        "ma10": sma(closes, 10),
        "slope": slope,
        "rising": rising,
        "hh_hl": hh_hl,
        "close": px,
        "weeks": len(weekly),
    }


def _higher_highs_lows(weekly: Sequence[dict]) -> bool:
    if len(weekly) < 20:
        return False
    recent = weekly[-10:]
    prior = weekly[-20:-10]
    rh = [to_float(w.get("high")) for w in recent]
    ph = [to_float(w.get("high")) for w in prior]
    rl = [to_float(w.get("low")) for w in recent]
    pl = [to_float(w.get("low")) for w in prior]
    rh = [v for v in rh if v is not None]
    ph = [v for v in ph if v is not None]
    rl = [v for v in rl if v is not None]
    pl = [v for v in pl if v is not None]
    if not rh or not ph or not rl or not pl:
        return False
    return max(rh) > max(ph) and min(rl) > min(pl)


def near_level(px: Optional[float], level: Optional[float], atr: Optional[float], frac: float = 0.75) -> bool:
    if px is None or level is None or atr is None or atr <= 0:
        return False
    return abs(px - level) <= frac * atr
