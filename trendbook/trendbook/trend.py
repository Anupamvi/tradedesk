"""Tape-seeded campaign age, hysteresis, Stage-2 trades, A/B/C grade.

A Stage 2 book buys the transition and the first early dip — not an 8-week
wait for a mean-reversion. Spike/breakout is a valid young-campaign entry.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from trendbook.bars import to_weekly, volume_expand
from trendbook.config import STAGE_WEEKS
from trendbook.num import pct_change, to_float
from trendbook.rs import mansfield
from trendbook.stage import classify_stage

# LATE is "old and actually dying," not "slow for two months."
LATE_MIN_WEEKS = 26
LATE_OFF_HIGH = 0.35
# Kept for grade "still expanding" (13w return above this is fine).
LATE_13W_MAX = 0.10
BREAK_MAX_WEEKS = 4
PULLBACK_MAX_WEEKS = 13
PULLBACK_30W_MAX_WEEKS = 26
PULLBACK_MIN_OFF = 0.04
ADD_MIN_WEEKS = 0
ADD_MIN_RS = 0.0
ADD_MIN_OFF = PULLBACK_MIN_OFF
GAP_MAX = 2
NEAR_HIGH_GRADE = 0.15
FWD_WEEKS = (4, 8, 13)
PULLBACK_WHY = ("pullback", "pullback_and_tight", "pullback_30w")

EMPTY_RUN = {
    "on_board": False,
    "trend_start": None,
    "weeks_in_trend": 0,
    "px_at_start": None,
    "pct_from_start": None,
    "ret_13w": None,
    "late": False,
    "had_trend": False,
    "last_trend_start": None,
    "last_trend_end": None,
    "campaign_high": None,
    "off_high": None,
    "gaps": 0,
    "vol_expand": None,
    "break_vol_expand": None,
}

TIGHT_RESIDUAL = 0.08
EARNINGS_BLOCK_DAYS = 10


def beating_spy(mans: Optional[float], rs_126: Optional[float] = None) -> bool:
    if mans is not None:
        return mans > 0
    return rs_126 is not None and rs_126 > 0


def week_flags(stock: Sequence[dict], spy: Sequence[dict], asof: str) -> List[dict]:
    weekly = to_weekly(stock, asof)
    spy_w = to_weekly(spy, asof)
    rows = []
    for i, bar in enumerate(weekly):
        prefix = weekly[: i + 1]
        st = classify_stage(prefix) if len(prefix) >= STAGE_WEEKS else {"stage": None}
        px = to_float(bar.get("close"))
        mans = mansfield(prefix, spy_w)
        rising = None
        if len(prefix) >= 5:
            prev_mans = mansfield(prefix[:-4], spy_w)
            if mans is not None and prev_mans is not None:
                rising = mans > prev_mans
        on = st.get("stage") == 2 and beating_spy(mans)
        rows.append(
            {
                "asof": str(bar.get("date") or "")[:10],
                "close": px,
                "high": to_float(bar.get("high")) or px,
                "on": on,
                "stage": st.get("stage"),
                "ma10": st.get("ma10"),
                "ma30": st.get("ma30"),
                "hh_hl": st.get("hh_hl"),
                "mansfield": mans,
                "mansfield_rising": rising,
                "vol_expand": volume_expand(prefix),
                "open": to_float(bar.get("open")) or px,
                "low": to_float(bar.get("low")) or px,
            }
        )
    return rows


def campaign_start_i(flags: Sequence[dict], end_i: int) -> int:
    """Walk back from end_i, allowing up to GAP_MAX consecutive off weeks."""
    start = end_i
    gap = 0
    i = end_i
    while i >= 0:
        if flags[i].get("on"):
            gap = 0
            start = i
        else:
            gap += 1
            if gap > GAP_MAX:
                break
        i -= 1
    return start


def _last_on_index(flags: Sequence[dict]) -> Optional[int]:
    for i in range(len(flags) - 1, -1, -1):
        if flags[i].get("on"):
            return i
    return None


def _span_stats(flags: Sequence[dict], start_i: int, end_i: int) -> dict:
    start_bar = flags[start_i]
    end_bar = flags[end_i]
    px0 = to_float(start_bar.get("close"))
    px1 = to_float(end_bar.get("close"))
    highs = []
    gaps = 0
    for row in flags[start_i : end_i + 1]:
        hi = to_float(row.get("high")) or to_float(row.get("close"))
        if hi is not None:
            highs.append(hi)
        if not row.get("on"):
            gaps += 1
    campaign_high = max(highs) if highs else px1
    off_high = None
    if campaign_high and campaign_high > 0 and px1 is not None:
        off_high = 1.0 - (px1 / campaign_high)
    weeks = end_i - start_i + 1
    ret_13w = None
    if end_i >= 13:
        ret_13w = pct_change(flags[end_i].get("close"), flags[end_i - 13].get("close"))
    late = False
    if weeks >= LATE_MIN_WEEKS:
        if ret_13w is not None and ret_13w < 0:
            late = True
        if off_high is not None and off_high > LATE_OFF_HIGH:
            late = True
    return {
        "trend_start": start_bar.get("asof"),
        "weeks_in_trend": weeks,
        "px_at_start": px0,
        "pct_from_start": pct_change(px1, px0),
        "ret_13w": ret_13w,
        "late": late,
        "campaign_high": campaign_high,
        "off_high": off_high,
        "gaps": gaps,
        "last_trend_start": start_bar.get("asof"),
        "last_trend_end": end_bar.get("asof"),
    }


def last_ended_run(flags: Sequence[dict]) -> Optional[dict]:
    if not flags or flags[-1].get("on"):
        return None
    end_i = _last_on_index(flags)
    if end_i is None:
        return None
    start_i = campaign_start_i(flags, end_i)
    stats = _span_stats(flags, start_i, end_i)
    return {
        "last_trend_start": stats["last_trend_start"],
        "last_trend_end": stats["last_trend_end"],
        "campaign_high": stats["campaign_high"],
    }


def current_run_from_flags(flags: Sequence[dict]) -> dict:
    out = dict(EMPTY_RUN)
    out["had_trend"] = any(f.get("on") for f in flags)
    ended = last_ended_run(flags)
    if ended:
        out.update(ended)
    if not flags or not flags[-1].get("on"):
        return out
    end_i = len(flags) - 1
    start_i = campaign_start_i(flags, end_i)
    stats = _span_stats(flags, start_i, end_i)
    out.update(stats)
    out["on_board"] = True
    out["had_trend"] = True
    out["vol_expand"] = flags[-1].get("vol_expand")
    out["break_vol_expand"] = flags[start_i].get("vol_expand")
    return out


def current_run(
    stock: Sequence[dict],
    spy: Sequence[dict],
    asof: str,
    flags: Optional[Sequence[dict]] = None,
) -> dict:
    """Current Stage-2 campaign from the weekly tape. 1–2 week Stage-1 gaps do not reset age."""
    rows = list(flags) if flags is not None else week_flags(stock, spy, asof)
    return current_run_from_flags(rows)


def classify_regime(spy_stage, tlt_stage=None, uup_stage=None) -> dict:
    """Index + rates/dollar. off = no new ADD. tight = half size / higher residual."""
    if spy_stage != 2:
        risk = "off"
    elif uup_stage == 2 and tlt_stage == 4:
        risk = "tight"
    else:
        risk = "on"
    return {
        "risk": risk,
        "spy_stage": spy_stage,
        "tlt_stage": tlt_stage,
        "uup_stage": uup_stage,
    }


def earnings_near(core: Optional[dict]) -> bool:
    if not core:
        return False
    wks = to_float(core.get("wks_next_ern"))
    if wks is None:
        return False
    return wks * 7.0 <= float(EARNINGS_BLOCK_DAYS)


def invalidation_levels(weekly_stage: Optional[dict] = None) -> dict:
    ma30 = to_float((weekly_stage or {}).get("ma30"))
    return {
        "rule": "weekly close < 30-week MA, or Mansfield < 0 for two weeks",
        "stop": ma30,
        "ma30": ma30,
    }


def breakout_quality(
    weeks: int,
    mansfield_spy: Optional[float] = None,
    mansfield_rising: Optional[bool] = None,
    hh_hl: Optional[bool] = None,
    rs_63: Optional[float] = None,
    residual_63: Optional[float] = None,
    vol_expand: Optional[bool] = None,
    rs_sector_126: Optional[float] = None,
) -> bool:
    """Young Stage 2 with idiosyncratic RS. Week-1 needs volume expansion on the break."""
    if weeks < 1 or weeks > BREAK_MAX_WEEKS:
        return False
    if weeks == 1 and vol_expand is False:
        return False
    sec = to_float(rs_sector_126)
    if sec is not None and sec < 0:
        return False
    mans_v = to_float(mansfield_spy)
    edge = to_float(residual_63)
    if edge is None:
        edge = to_float(rs_63)
    if edge is not None and edge < 0:
        return False
    edge = edge or 0.0
    hh = bool(hh_hl)
    rising = bool(mansfield_rising)
    if rising:
        if hh or edge >= 0.05 or (mans_v is not None and mans_v >= 0.03):
            return True
        if weeks >= 2:
            return True
    if hh and (edge >= 0.05 or (mans_v is not None and mans_v >= 0.05)):
        return True
    if weeks >= 2 and (edge >= 0.08 or (mans_v is not None and mans_v >= 0.05)):
        return True
    return False


def _raw_setup(
    entry_state: Optional[str],
    why: str,
    weeks: int,
    off_high: Optional[float],
    residual_63: Optional[float],
    mansfield_spy: Optional[float],
    mansfield_rising: Optional[bool],
    hh_hl: Optional[bool],
    rs_63: Optional[float],
    vol_expand: Optional[bool],
    rs_sector_126: Optional[float],
) -> tuple:
    dip = to_float(off_high)
    resid = to_float(residual_63)
    pullback = entry_state == "ready" and why in PULLBACK_WHY and dip is not None and dip >= PULLBACK_MIN_OFF
    if pullback and resid is not None and resid < 0:
        pullback = False
    if pullback:
        if weeks <= PULLBACK_MAX_WEEKS:
            return "ADD", why
        if weeks <= PULLBACK_30W_MAX_WEEKS and why == "pullback_30w":
            return "ADD", why
    quality = breakout_quality(
        weeks,
        mansfield_spy,
        mansfield_rising,
        hh_hl,
        rs_63,
        residual_63=residual_63,
        vol_expand=vol_expand,
        rs_sector_126=rs_sector_126,
    )
    broken = entry_state == "broken"
    if weeks <= BREAK_MAX_WEEKS and quality and not broken:
        return "ADD", "breakout"
    if weeks <= BREAK_MAX_WEEKS:
        return "NEW", None
    return "HOLD", None


def decide(
    on_board: bool,
    late: bool,
    entry_state: Optional[str],
    weeks: int = 0,
    entry_reason: Optional[str] = None,
    off_high: Optional[float] = None,
    rs_126: Optional[float] = None,
    rs_63: Optional[float] = None,
    mansfield_spy: Optional[float] = None,
    mansfield_rising: Optional[bool] = None,
    hh_hl: Optional[bool] = None,
    residual_63: Optional[float] = None,
    vol_expand: Optional[bool] = None,
    break_vol_expand: Optional[bool] = None,
    already_bought: bool = False,
    regime_risk: str = "on",
    rs_sector_126: Optional[float] = None,
    earnings_near_flag: bool = False,
) -> dict:
    """One Stage 2 ticket per campaign. Breakout uses the *break week's* volume."""
    why = str(entry_reason or "")
    weeks = int(weeks or 0)
    hold = {"action": "HOLD", "setup": None, "size": None}
    if not on_board:
        return {"action": "OUT", "setup": None, "size": None}
    if late:
        return {"action": "LATE", "setup": None, "size": None}
    vol = break_vol_expand if break_vol_expand is not None else vol_expand
    action, setup = _raw_setup(
        entry_state,
        why,
        weeks,
        off_high,
        residual_63,
        mansfield_spy,
        mansfield_rising,
        hh_hl,
        rs_63,
        vol,
        rs_sector_126,
    )
    if action != "ADD":
        return {"action": action, "setup": setup, "size": None}
    if already_bought:
        return {"action": "HOLD", "setup": "entry_window_used", "size": None}
    if regime_risk == "off":
        return {"action": "HOLD", "setup": "regime_off", "size": None}
    if earnings_near_flag:
        return {"action": "HOLD", "setup": "earnings", "size": None}
    resid = to_float(residual_63)
    if regime_risk == "tight" and resid is not None and resid < 0:
        return {"action": "HOLD", "setup": "regime_tight", "size": None}
    size = "half" if regime_risk == "tight" else "full"
    return {"action": "ADD", "setup": setup, "size": size}


def assign_action(
    on_board: bool,
    late: bool,
    entry_state: Optional[str],
    weeks: int = 0,
    entry_reason: Optional[str] = None,
    off_high: Optional[float] = None,
    rs_126: Optional[float] = None,
    rs_63: Optional[float] = None,
    mansfield_spy: Optional[float] = None,
    mansfield_rising: Optional[bool] = None,
    hh_hl: Optional[bool] = None,
    residual_63: Optional[float] = None,
    vol_expand: Optional[bool] = None,
    break_vol_expand: Optional[bool] = None,
    already_bought: bool = False,
    regime_risk: str = "on",
    rs_sector_126: Optional[float] = None,
    earnings_near_flag: bool = False,
) -> str:
    """ADD = Stage 2 breakout (spike allowed) or early pullback. NEW = weak tag. Not a 20 EMA pause."""
    return decide(
        on_board,
        late,
        entry_state,
        weeks=weeks,
        entry_reason=entry_reason,
        off_high=off_high,
        rs_126=rs_126,
        rs_63=rs_63,
        mansfield_spy=mansfield_spy,
        mansfield_rising=mansfield_rising,
        hh_hl=hh_hl,
        residual_63=residual_63,
        vol_expand=vol_expand,
        break_vol_expand=break_vol_expand,
        already_bought=already_bought,
        regime_risk=regime_risk,
        rs_sector_126=rs_sector_126,
        earnings_near_flag=earnings_near_flag,
    )["action"]


def week_decide(flags: Sequence[dict], i: int, stock: Sequence[dict], spy: Sequence[dict], **extra) -> dict:
    """Leakage-safe setup at flags[i]. Used for first-ADD-per-campaign."""
    from trendbook.bars import bars_through, closes, daily_snapshot, ret
    from trendbook.entry import classify_entry
    from trendbook.rs import residual_return

    prefix = list(flags[: i + 1])
    run = current_run_from_flags(prefix)
    flag = flags[i]
    day = str(flag.get("asof") or "")[:10]
    daily = daily_snapshot(stock, day)
    st = {
        "stage": flag.get("stage"),
        "ma10": flag.get("ma10"),
        "ma30": flag.get("ma30"),
        "hh_hl": flag.get("hh_hl"),
    }
    on_board = bool(run.get("on_board") and daily.get("ok"))
    entry = classify_entry(daily, st) if on_board else {"state": "off", "reason": "not_on_board"}
    stock_c = closes(bars_through(stock, day))
    spy_c = closes(bars_through(spy, day))
    a, b = ret(stock_c, 126), ret(spy_c, 126)
    r3s, r3b = ret(stock_c, 63), ret(spy_c, 63)
    return decide(
        on_board,
        bool(run.get("late")),
        entry.get("state"),
        weeks=int(run.get("weeks_in_trend") or 0),
        entry_reason=entry.get("reason"),
        off_high=run.get("off_high"),
        rs_126=(a - b) if a is not None and b is not None else None,
        rs_63=(r3s - r3b) if r3s is not None and r3b is not None else None,
        mansfield_spy=flag.get("mansfield"),
        mansfield_rising=flag.get("mansfield_rising"),
        hh_hl=flag.get("hh_hl"),
        residual_63=residual_return(stock_c, spy_c, 63),
        vol_expand=flag.get("vol_expand"),
        break_vol_expand=run.get("break_vol_expand"),
        already_bought=bool(extra.get("already_bought")),
        regime_risk=str(extra.get("regime_risk") or "on"),
        earnings_near_flag=bool(extra.get("earnings_near_flag")),
    )


def first_add_dates(flags: Sequence[dict], stock: Sequence[dict], spy: Sequence[dict]) -> dict:
    """Map campaign start date -> first ADD asof. One ticket per campaign."""
    found: Dict[str, str] = {}
    for i, flag in enumerate(flags):
        run = current_run_from_flags(flags[: i + 1])
        start = run.get("trend_start")
        if not start or start in found:
            continue
        info = week_decide(flags, i, stock, spy)
        if info.get("action") == "ADD":
            found[str(start)] = str(flag.get("asof") or "")[:10]
    return found


def grade(
    action: str,
    weeks: int,
    ret_13w: Optional[float],
    off_high: Optional[float],
    entry_reason: Optional[str],
    beating: bool,
    hh_hl: Optional[bool] = None,
) -> Optional[str]:
    """Structure grade, not P(win). A/B/C from campaign evidence only."""
    if action == "OUT":
        return None
    if action == "NEW":
        return "C"
    if action == "LATE":
        return "C"
    expanding = ret_13w is not None and ret_13w > LATE_13W_MAX
    near_high = off_high is not None and off_high <= NEAR_HIGH_GRADE
    why = str(entry_reason or "")
    if action == "ADD" and why == "breakout" and beating and (hh_hl or weeks >= 2):
        return "B"
    usable_dip = off_high is not None and PULLBACK_MIN_OFF <= off_high <= NEAR_HIGH_GRADE
    if action == "ADD" and weeks >= 8 and beating and expanding and usable_dip:
        return "A"
    if action == "ADD" and beating:
        return "B"
    if action != "ADD" and weeks >= 13 and beating and expanding and near_high:
        return "A"
    if weeks >= 4 and beating:
        return "B"
    return "C"


def first_tag(flags: Sequence[dict]) -> Optional[dict]:
    for row in flags:
        if row.get("on"):
            return row
    return None


def close_n_weeks_later(flags: Sequence[dict], start_idx: int, n: int) -> Optional[float]:
    j = start_idx + n
    if j >= len(flags):
        return None
    return to_float(flags[j].get("close"))


def replay_row(ticker: str, flags: Sequence[dict], run: dict, stock: Sequence[dict]) -> dict:
    tagged = first_tag(flags)
    row = {
        "ticker": ticker,
        "tagged": bool(tagged),
        "first_tag": (tagged or {}).get("asof"),
        "first_px": (tagged or {}).get("close"),
        "now_px": to_float((stock[-1] or {}).get("close")) if stock else None,
        "weeks_in_trend": run.get("weeks_in_trend") or 0,
        "trend_start": run.get("trend_start"),
        "pct_from_start": run.get("pct_from_start"),
        "on_board_now": run.get("on_board"),
        "campaign_high": run.get("campaign_high"),
        "off_high": run.get("off_high"),
    }
    if tagged:
        idx = next(i for i, f in enumerate(flags) if f.get("asof") == tagged.get("asof"))
        px0 = tagged.get("close")
        for n in FWD_WEEKS:
            later = close_n_weeks_later(flags, idx, n)
            row["fwd_%sw" % n] = pct_change(later, px0)
    else:
        for n in FWD_WEEKS:
            row["fwd_%sw" % n] = None
    return row


def universe_replay(bars_map: Dict[str, list], asof: str, skip: Optional[Sequence[str]] = None) -> List[dict]:
    spy = bars_map.get("SPY") or []
    banned = set(skip or [])
    out = []
    for ticker, stock in sorted(bars_map.items()):
        if ticker == "SPY" or ticker in banned:
            continue
        flags = week_flags(stock, spy, asof)
        run = current_run_from_flags(flags)
        out.append(replay_row(ticker, flags, run, stock))
    return out
