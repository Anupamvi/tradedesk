"""Setups A–G. Indicators are context, not triggers. AVWAP alone is not a trade."""

from __future__ import annotations

from typing import Dict, List, Optional

from groat.config import CHASE_ATR
from groat.num import to_float
from groat.technicals import avwap


SETUP_NAMES = {
    "A": "Trend Pullback",
    "B": "Breakout + Confirmation",
    "C": "Post-Earnings Drift",
    "D": "Relative-Strength Leader",
    "E": "Emerging Sector Rotation",
    "F": "Oversold Reversal",
    "G": "Failed Breakout / Trend Breakdown",
    "H": "FIRE spike/dip",
}

SETUP_PRIORITY = ("C", "E", "B", "A", "D", "G", "F", "H")

SETUP_GUIDE = {
    "A": "Trend pullback — strong uptrend, dip into 20 EMA / AVWAP / 50 DMA with quieter volume.",
    "B": "Breakout — close above a 20-session high after compression; do not chase a 2.5+ ATR spike.",
    "C": "Post-earnings drift — the print already happened. Price holds the earnings AVWAP instead of filling the gap. You are trading the follow-through, not the announcement.",
    "D": "Relative-strength leader — beating SPY (and usually its group) with accumulation, even if the group is not the hottest.",
    "E": "Emerging sector rotation — capital is moving into the industry (accelerating/emerging vs SPY). You want the strongest name in that group.",
    "F": "Oversold reversal — washout plus reclaim/volume; RSI alone is not enough.",
    "G": "Failed breakout / breakdown — bearish: lost 20 EMA, failed highs, weak RS.",
    "H": "FIRE spike/dip — 1–2 day shock with ≥1.5× volume. X only confirms or vetoes.",
}


def _near(px, level, atr, frac=0.75) -> bool:
    if px is None or level is None or atr is None or atr <= 0:
        return False
    return abs(px - level) <= frac * atr


def classify_setups(
    snap: dict,
    group_row: Optional[dict] = None,
    earnings: Optional[dict] = None,
    bars: Optional[list] = None,
) -> Dict[str, object]:
    hits = []
    notes = []
    px = to_float(snap.get("close"))
    atr = to_float(snap.get("atr14"))
    ema20 = to_float(snap.get("ema20"))
    sma50 = to_float(snap.get("sma50"))
    sma200 = to_float(snap.get("sma200"))
    av_low = to_float(snap.get("avwap_swing_low"))
    av_year = to_float(snap.get("avwap_year"))
    ext = to_float(snap.get("extension_atr"))
    trend = str(snap.get("trend") or "")
    rs20 = to_float(snap.get("rs_20"))
    rvol = to_float(snap.get("rvol"))
    hi20c = to_float(snap.get("hi20_close"))
    lo20 = to_float(snap.get("lo20"))
    v5 = to_float(snap.get("vol_5"))
    v20 = to_float(snap.get("vol_20"))
    group_status = str((group_row or {}).get("status") or "")
    earn = earnings or {}

    av_earn = None
    last_ern = earn.get("last")
    if last_ern and bars:
        av_earn = avwap(bars, snap.get("date") or snap.get("asof"), last_ern)

    # C — post-earnings
    days = earn.get("days")
    last_days = None
    if last_ern and snap.get("date"):
        try:
            from datetime import datetime

            last_days = (datetime.strptime(snap["date"][:10], "%Y-%m-%d") - datetime.strptime(last_ern, "%Y-%m-%d")).days
        except (TypeError, ValueError):
            last_days = None
    if last_days is not None and 1 <= last_days <= 15 and px and av_earn and px >= av_earn:
        hits.append("C")
        notes.append("holding earnings AVWAP %s sessions after lastErn" % last_days)
    elif last_ern is None and not earn.get("usable") and str(earn.get("source") or "") != "exempt":
        notes.append("earnings date DATA UNAVAILABLE")

    # E — emerging group
    if group_status in ("accelerating", "emerging") and rs20 is not None and rs20 > 0 and trend in ("up", "strong_up"):
        hits.append("E")
        notes.append("group %s with positive 20d RS" % group_status)

    # B — breakout
    if hi20c is not None and px is not None and px > hi20c:
        if ext is not None and ext > CHASE_ATR:
            notes.append("breakout but extended %.1f ATR above 20 EMA — do not chase" % ext)
        else:
            hits.append("B")
            notes.append("close above prior 20-session close high")

    # A — trend pullback
    if trend in ("up", "strong_up") and px and ema20:
        pulled = _near(to_float(snap.get("low")), ema20, atr, 0.9) or _near(px, ema20, atr, 0.6)
        pulled = pulled or (av_low is not None and _near(px, av_low, atr, 0.7))
        pulled = pulled or (sma50 is not None and ema20 > sma50 and px < ema20 and px >= sma50)
        vol_in = v5 is not None and v20 is not None and v5 < v20
        if pulled and (ext is None or ext < 1.8):
            hits.append("A")
            notes.append("trend pullback into 20 EMA / AVWAP / 50 DMA")
            if vol_in:
                notes.append("volume contracted on the pullback")

    # D — RS leader
    if rs20 is not None and rs20 >= 0.04 and snap.get("above_sma50") and trend in ("up", "strong_up", "range"):
        hits.append("D")
        notes.append("20d RS vs SPY %+0.1f%% with accumulation structure" % (rs20 * 100.0))

    # F — oversold reversal (selective)
    rsi = to_float(snap.get("rsi14"))
    ret20 = to_float(snap.get("ret_20"))
    bullish_bar = False
    if px and snap.get("open") and snap.get("low") and snap.get("high"):
        rng = float(snap["high"]) - float(snap["low"])
        if rng > 0 and px >= float(snap["low"]) + 0.7 * rng and px > float(snap["open"]):
            bullish_bar = True
    reclaim = ema20 is not None and px is not None and px > ema20 and snap.get("low") is not None and float(snap["low"]) <= ema20
    washed = (ret20 is not None and ret20 <= -0.08) or (lo20 is not None and px and _near(px, lo20, atr, 0.8))
    if washed and (bullish_bar or reclaim) and (rvol is None or rvol >= 1.0):
        if rsi is not None and rsi < 35:
            notes.append("RSI %.0f is context only, not the trigger" % rsi)
        hits.append("F")
        notes.append("oversold reversal evidence (range close / 20 EMA reclaim), not RSI-only")

    # G — failed breakout / breakdown
    if trend in ("down", "strong_down") and snap.get("above_ema20") is False:
        hits.append("G")
        notes.append("below declining/lost 20 EMA in a downtrend")
    elif hi20c and px and px < hi20c and snap.get("above_ema20") is False and (rs20 is not None and rs20 < 0):
        if "G" not in hits:
            hits.append("G")
            notes.append("failed to hold the 20-day high with negative RS")

    # H — FIRE spike / dip (volume + 1–2 day shock). X is confirm, not the trigger.
    fire = _fire(snap, rvol, ext, trend)
    if fire.get("kind"):
        hits.append("H")
        notes.append(fire.get("note") or "FIRE tape")

    primary = None
    for code in SETUP_PRIORITY:
        if code in hits:
            primary = code
            break
    direction = "neutral"
    if primary in ("A", "B", "C", "D", "E", "F"):
        direction = "bullish"
    elif primary == "G":
        direction = "bearish"
    if ext is not None and ext > CHASE_ATR and direction == "bullish" and primary == "B":
        direction = "neutral"
        notes.append("bullish breakout parked: chase filter")
    if fire.get("kind") == "spike" and fire.get("chase"):
        notes.append("FIRE spike is extended — do not chase; FIRE watch only")
    if fire.get("kind") == "dip" and trend in ("up", "strong_up"):
        direction = "bullish"
    elif fire.get("kind") == "dip" and trend in ("down", "strong_down"):
        direction = "bearish"
    elif fire.get("kind") == "spike" and not fire.get("chase"):
        direction = "bullish"

    return {
        "setups": hits,
        "primary": primary,
        "primary_name": SETUP_NAMES.get(primary or "", ""),
        "direction": direction,
        "notes": notes,
        "avwap_earnings": av_earn,
        "chase": bool(ext is not None and ext > CHASE_ATR),
        "fire": fire,
        "lane": "FIRE" if fire.get("kind") and not fire.get("chase") else "SWING",
    }


def _fire(snap: dict, rvol, ext, trend: str) -> dict:
    ret1 = to_float(snap.get("ret_1"))
    ret2 = to_float(snap.get("ret_2"))
    rvol = to_float(rvol)
    kind = None
    note = ""
    if rvol is not None and rvol >= 1.5:
        if ret1 is not None and ret1 >= 0.03:
            kind = "spike"
            note = "FIRE spike: 1d %+0.1f%% on %.1fx volume" % (ret1 * 100.0, rvol)
        elif ret2 is not None and ret2 >= 0.05:
            kind = "spike"
            note = "FIRE spike: 2d %+0.1f%% on %.1fx volume" % (ret2 * 100.0, rvol)
        elif ret1 is not None and ret1 <= -0.03:
            kind = "dip"
            note = "FIRE dip: 1d %+0.1f%% on %.1fx volume" % (ret1 * 100.0, rvol)
        elif ret2 is not None and ret2 <= -0.05:
            kind = "dip"
            note = "FIRE dip: 2d %+0.1f%% on %.1fx volume" % (ret2 * 100.0, rvol)
    chase = bool(kind == "spike" and ext is not None and ext > CHASE_ATR)
    return {
        "kind": kind,
        "ret_1": ret1,
        "ret_2": ret2,
        "rvol": rvol,
        "chase": chase,
        "note": note,
    }
