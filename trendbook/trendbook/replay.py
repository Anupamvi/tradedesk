"""Leakage-safe weekly walk on cached daily bars. No future bars, no UW, no X."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from trendbook.bars import daily_snapshot, to_weekly
from trendbook.entry import classify_entry
from trendbook.rs import rs_bundle
from trendbook.stage import classify_stage
from trendbook.trend import decide, current_run


def week_asofs(daily: Sequence[dict], asof: str) -> List[str]:
    weekly = to_weekly(daily, asof)
    return [str(w.get("date") or "")[:10] for w in weekly]


def classify_at(
    ticker: str,
    stock: Sequence[dict],
    spy: Sequence[dict],
    asof: str,
    sector: Optional[Sequence[dict]] = None,
) -> dict:
    daily = daily_snapshot(stock, asof)
    weekly = to_weekly(stock, asof)
    st = classify_stage(weekly)
    run = current_run(stock, spy, asof)
    bundle = rs_bundle(ticker, stock, spy, sector, asof)
    on_board = bool(daily.get("ok") and run.get("on_board"))
    entry = classify_entry(daily, st) if on_board else {"state": "off", "reason": "not_on_board"}
    info = decide(
        on_board,
        bool(run.get("late")),
        entry.get("state"),
        weeks=int(run.get("weeks_in_trend") or 0),
        entry_reason=entry.get("reason"),
        off_high=run.get("off_high"),
        rs_126=bundle.get("rs_126"),
        rs_63=bundle.get("rs_63"),
        mansfield_spy=bundle.get("mansfield_spy"),
        mansfield_rising=bundle.get("mansfield_spy_rising"),
        hh_hl=st.get("hh_hl"),
        residual_63=bundle.get("residual_63"),
        vol_expand=run.get("vol_expand"),
        break_vol_expand=run.get("break_vol_expand"),
        rs_sector_126=bundle.get("rs_sector_126"),
    )
    action = info["action"]
    return {
        "asof": asof,
        "ticker": ticker,
        "action": action,
        "status": action,
        "on_board": on_board,
        "stage": st.get("stage"),
        "entry": entry.get("state"),
        "entry_reason": info.get("setup") or entry.get("reason"),
        "setup": info.get("setup"),
        "mansfield_spy": bundle.get("mansfield_spy"),
        "rs_126": bundle.get("rs_126"),
        "close": daily.get("close"),
        "weeks": st.get("weeks"),
        "weeks_in_trend": run.get("weeks_in_trend"),
        "trend_start": run.get("trend_start"),
        "hh_hl": st.get("hh_hl"),
        "late": bool(run.get("late")),
    }


def walk(
    ticker: str,
    stock: Sequence[dict],
    spy: Sequence[dict],
    asof: str,
    sector: Optional[Sequence[dict]] = None,
) -> Dict[str, object]:
    dates = week_asofs(stock, asof)
    rows = []
    for day in dates:
        row = classify_at(ticker, stock, spy, day, sector)
        rows.append(row)
    on_flags = [bool(r.get("on_board")) for r in rows]
    longest = 0
    cur = 0
    first_on = None
    last_on = None
    for r, flag in zip(rows, on_flags):
        if flag:
            cur += 1
            longest = max(longest, cur)
            if first_on is None:
                first_on = r["asof"]
            last_on = r["asof"]
        else:
            cur = 0
    return {
        "ticker": ticker,
        "asof": asof,
        "weeks": len(rows),
        "on_board_weeks": sum(on_flags),
        "longest_run": longest,
        "first_on": first_on,
        "last_on": last_on,
        "rows": rows,
    }


def load_json_bars(path) -> List[dict]:
    import json
    from pathlib import Path

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("data") if isinstance(payload, dict) else payload
    return [b for b in (rows or []) if isinstance(b, dict) and b.get("date")]
