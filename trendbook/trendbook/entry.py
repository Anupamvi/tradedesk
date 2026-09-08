"""Entry state for names already ON_BOARD: ready / extended / broken / held."""

from __future__ import annotations

from trendbook.config import CHASE_ATR
from trendbook.num import to_float
from trendbook.stage import near_level


def classify_entry(daily: dict, weekly_stage: dict) -> dict:
    """20 EMA alone is a pause, not a buy. Pullback = 10-week, 50-day, or a 30-week dip."""
    px = to_float(daily.get("close"))
    ema20 = to_float(daily.get("ema20"))
    sma50 = to_float(daily.get("sma50"))
    atr = to_float(daily.get("atr14"))
    ext = to_float(daily.get("extension_atr"))
    ma10 = to_float(weekly_stage.get("ma10"))
    ma30 = to_float(weekly_stage.get("ma30"))
    stage = weekly_stage.get("stage")
    range20 = to_float(daily.get("range20"))
    tightness = to_float(daily.get("tightness"))

    if stage != 2:
        return {"state": "broken", "reason": "not_stage_2"}
    if px is not None and ma30 is not None and px < ma30:
        return {"state": "broken", "reason": "lost_30w"}
    at_30_hold = near_level(px, ma30, atr)
    if (
        px is not None
        and ema20 is not None
        and px < ema20
        and ma10 is not None
        and px < ma10
        and not at_30_hold
    ):
        return {"state": "broken", "reason": "lost_20ema_and_10w"}
    if ext is not None and ext > CHASE_ATR:
        return {"state": "extended", "reason": "chase_%.1f_atr" % ext}

    at_10w = near_level(px, ma10, atr)
    at_50 = near_level(px, sma50, atr)
    # 30-week dip only if price has come back (at or below 20 EMA). Fresh Stage 2
    # is near the 30-week by definition — that is a breakout, not a pullback.
    at_30 = near_level(px, ma30, atr) and px is not None and ema20 is not None and px <= ema20
    at_20 = near_level(px, ema20, atr)
    pullback = at_10w or at_50 or at_30
    rest = at_20 and not pullback
    tight = (range20 is not None and range20 < 0.08) or (tightness is not None and tightness < 0.45)
    if pullback and tight:
        return {"state": "ready", "reason": "pullback_and_tight"}
    if at_30 and not (at_10w or at_50):
        return {"state": "ready", "reason": "pullback_30w"}
    if pullback:
        return {"state": "ready", "reason": "pullback"}
    if rest:
        return {"state": "held", "reason": "rest_20ema"}
    if tight:
        return {"state": "held", "reason": "tightness"}
    return {"state": "held", "reason": "in_trend_wait"}
