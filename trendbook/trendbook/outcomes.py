"""Campaign outcomes for ADD only. Not a 20-day swing timer.

Fill = next week's open when present.
Hold 8 weeks or until invalidation (two consecutive off weeks).
CAPTURE = 8-week (or exit) return > 0. FAIL = return <= 0 or stopped.
OPEN = fewer than 8 weeks and not stopped.
HOLD / NEW / LATE are not trades. A wick is not a win.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

from trendbook.bars import bars_through, closes, daily_snapshot, ret, to_weekly
from trendbook.rs import residual_return
from trendbook.config import CODE_DIR, INDEX_TICKERS, MACRO_TICKERS, SECTOR_ETFS
from trendbook.entry import classify_entry
from trendbook.num import fmt, fmt_pct, pct_change, to_float
from trendbook.stage import classify_stage
from trendbook.trend import classify_regime, current_run_from_flags, decide, grade, week_flags

HORIZON_WEEKS = 8


def ledger_path(path: Optional[Path] = None) -> Path:
    return Path(path) if path is not None else CODE_DIR / "var" / "outcomes" / "ledger.json"


def load_ledger(path: Optional[Path] = None) -> List[dict]:
    target = ledger_path(path)
    if not target.is_file():
        return []
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    return []


def save_ledger(rows: Sequence[dict], path: Optional[Path] = None) -> None:
    target = ledger_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(list(rows), indent=2) + "\n", encoding="utf-8")


def score_add(flags: Sequence[dict], signal_i: int, entry_px: Optional[float] = None) -> dict:
    """Hold to 8 weeks or two consecutive off weeks. Fill at next week's open."""
    empty = {
        "result": "OPEN",
        "reason": "need_8_weeks",
        "exit_date": None,
        "exit_px": None,
        "ret_8w": None,
        "mae": None,
        "fill_px": None,
    }
    if signal_i < 0 or signal_i >= len(flags):
        return dict(empty)
    future = list(flags[signal_i + 1 :])
    if not future:
        return dict(empty)
    px0 = to_float(future[0].get("open"))
    if px0 is None:
        px0 = entry_px if entry_px is not None else to_float(flags[signal_i].get("close"))
    off_streak = 0
    mae = 0.0
    last = None
    for week in future[:HORIZON_WEEKS]:
        last = week
        close = to_float(week.get("close"))
        lo = to_float(week.get("low")) or close
        if px0 and lo and px0 > 0:
            mae = min(mae, lo / px0 - 1.0)
        if not week.get("on"):
            off_streak += 1
            if off_streak >= 2:
                ret = pct_change(close, px0)
                return {
                    "result": "FAIL" if (ret is None or ret <= 0) else "CAPTURE",
                    "reason": "invalidated",
                    "exit_date": week.get("asof"),
                    "exit_px": close,
                    "ret_8w": ret,
                    "mae": mae,
                    "fill_px": px0,
                }
        else:
            off_streak = 0
    if len(future) < HORIZON_WEEKS:
        out = dict(empty)
        out["mae"] = mae
        out["fill_px"] = px0
        return out
    week8 = last or future[HORIZON_WEEKS - 1]
    ret = pct_change(week8.get("close"), px0)
    return {
        "result": "CAPTURE" if ret is not None and ret > 0 else "FAIL",
        "reason": "horizon_up" if ret is not None and ret > 0 else "horizon_down",
        "exit_date": week8.get("asof"),
        "exit_px": to_float(week8.get("close")),
        "ret_8w": ret,
        "mae": mae,
        "fill_px": px0,
    }


def _board_skip() -> Set[str]:
    return set(INDEX_TICKERS) | set(SECTOR_ETFS) | set(MACRO_TICKERS)


def regime_by_week(spy: Sequence[dict], tlt: Optional[Sequence[dict]], uup: Optional[Sequence[dict]], asof: str) -> dict:
    spy_w = to_weekly(spy, asof)
    tlt_w = to_weekly(tlt or [], asof)
    uup_w = to_weekly(uup or [], asof)
    out = {}
    for i, bar in enumerate(spy_w):
        day = str(bar.get("date") or "")[:10]
        tlt_prefix = [w for w in tlt_w if str(w.get("date") or "")[:10] <= day]
        uup_prefix = [w for w in uup_w if str(w.get("date") or "")[:10] <= day]
        out[day] = classify_regime(
            classify_stage(spy_w[: i + 1]).get("stage"),
            classify_stage(tlt_prefix).get("stage") if tlt_prefix else None,
            classify_stage(uup_prefix).get("stage") if uup_prefix else None,
        )["risk"]
    return out


def _risk_on_day(series: dict, day: str) -> str:
    if not series:
        return "on"
    if day in series:
        return series[day]
    last = "on"
    for key in sorted(series):
        if key <= day:
            last = series[key]
        else:
            break
    return last


def add_signals(
    ticker: str,
    stock: Sequence[dict],
    spy: Sequence[dict],
    asof: str,
    flags: Optional[Sequence[dict]] = None,
    regime_weeks: Optional[dict] = None,
) -> List[dict]:
    """First ADD in each campaign, scored on later weeks only."""
    flags = list(flags) if flags is not None else week_flags(stock, spy, asof)
    seen_campaigns = set()
    out = []
    for i, flag in enumerate(flags):
        day = str(flag.get("asof") or "")[:10]
        if not day or day > asof[:10]:
            continue
        run = current_run_from_flags(flags[: i + 1])
        daily = daily_snapshot(stock, day)
        st = {
            "stage": flag.get("stage"),
            "ma10": flag.get("ma10"),
            "ma30": flag.get("ma30"),
            "hh_hl": flag.get("hh_hl"),
        }
        entry = classify_entry(daily, st) if run.get("on_board") else {"state": "off", "reason": "not_on_board"}
        stock_c = closes(bars_through(stock, day))
        spy_c = closes(bars_through(spy, day))
        a, b = ret(stock_c, 126), ret(spy_c, 126)
        rs_126 = (a - b) if a is not None and b is not None else None
        r3s, r3b = ret(stock_c, 63), ret(spy_c, 63)
        rs_63 = (r3s - r3b) if r3s is not None and r3b is not None else None
        resid_63 = residual_return(stock_c, spy_c, 63)
        info = decide(
            bool(run.get("on_board") and daily.get("ok")),
            bool(run.get("late")),
            entry.get("state"),
            weeks=int(run.get("weeks_in_trend") or 0),
            entry_reason=entry.get("reason"),
            off_high=run.get("off_high"),
            rs_126=rs_126,
            rs_63=rs_63,
            mansfield_spy=flag.get("mansfield"),
            mansfield_rising=flag.get("mansfield_rising"),
            hh_hl=flag.get("hh_hl"),
            residual_63=resid_63,
            vol_expand=flag.get("vol_expand"),
            break_vol_expand=run.get("break_vol_expand"),
            regime_risk=_risk_on_day(regime_weeks or {}, day),
        )
        action = info["action"]
        if action != "ADD":
            continue
        campaign = run.get("trend_start") or day
        if campaign in seen_campaigns:
            continue
        seen_campaigns.add(campaign)
        scored = score_add(flags, i, daily.get("close"))
        why = info.get("setup") or entry.get("reason")
        g = grade(
            action,
            int(run.get("weeks_in_trend") or 0),
            run.get("ret_13w"),
            run.get("off_high"),
            why,
            True,
            hh_hl=flag.get("hh_hl"),
        )
        row = {
            "ticker": str(ticker).upper(),
            "asof": day,
            "action": "ADD",
            "entry_px": daily.get("close"),
            "trend_start": campaign,
            "weeks_in_trend": run.get("weeks_in_trend") or 0,
            "grade": g,
            "entry_reason": why,
        }
        row.update(scored)
        out.append(row)
    return out


def rebuild(
    bars_map: Dict[str, list],
    asof: str,
    skip: Optional[Sequence[str]] = None,
    path: Optional[Path] = None,
    flags_map: Optional[Dict[str, list]] = None,
) -> List[dict]:
    spy = bars_map.get("SPY") or []
    banned = set(skip or []) | _board_skip()
    regime_weeks = regime_by_week(spy, bars_map.get("TLT"), bars_map.get("UUP"), asof)
    ledger = []
    for ticker, stock in sorted(bars_map.items()):
        if ticker == "SPY" or ticker in banned:
            continue
        flags = (flags_map or {}).get(ticker)
        ledger.extend(add_signals(ticker, stock, spy, asof, flags=flags, regime_weeks=regime_weeks))
    save_ledger(ledger, path)
    return ledger


def record(rows: Sequence[dict], asof: str, path: Optional[Path] = None) -> List[dict]:
    """Today's ADD rows that backfill has not already stored (OPEN until 8 weeks)."""
    ledger = load_ledger(path)
    seen = {(str(r.get("ticker")), str(r.get("trend_start"))) for r in ledger}
    for row in rows:
        if str(row.get("action") or "") != "ADD":
            continue
        ticker = str(row.get("ticker") or "").upper()
        campaign = str(row.get("trend_start") or asof)
        key = (ticker, campaign)
        if key in seen:
            continue
        seen.add(key)
        ledger.append(
            {
                "ticker": ticker,
                "asof": asof,
                "action": "ADD",
                "entry_px": row.get("close"),
                "trend_start": campaign,
                "weeks_in_trend": row.get("weeks_in_trend"),
                "grade": row.get("grade"),
                "entry_reason": row.get("entry_reason"),
                "result": "OPEN",
                "reason": "need_8_weeks",
            }
        )
    save_ledger(ledger, path)
    return ledger


def refresh(bars_map: Dict[str, list], asof: str, path: Optional[Path] = None, flags_map: Optional[Dict[str, list]] = None) -> List[dict]:
    """Re-score OPEN rows. Prefer rebuild() on a full scan."""
    spy = bars_map.get("SPY") or []
    ledger = load_ledger(path)
    for row in ledger:
        if row.get("result") in ("CAPTURE", "FAIL"):
            continue
        ticker = str(row.get("ticker") or "").upper()
        stock = bars_map.get(ticker) or []
        if not stock or not spy:
            continue
        flags = (flags_map or {}).get(ticker) or week_flags(stock, spy, asof)
        sig = str(row.get("asof") or "")[:10]
        idx = next((i for i, f in enumerate(flags) if f.get("asof") == sig), None)
        if idx is None:
            continue
        row.update(score_add(flags, idx, row.get("entry_px")))
        row["scored_asof"] = asof
    save_ledger(ledger, path)
    return ledger


def render(ledger: Sequence[dict], asof: str) -> str:
    open_rows = [r for r in ledger if r.get("result") == "OPEN"]
    capture = [r for r in ledger if r.get("result") == "CAPTURE"]
    fail = [r for r in ledger if r.get("result") == "FAIL"]
    scored = capture + fail
    lines = [
        "# Outcomes %s" % asof,
        "",
        "ADD only. Not a 20-day timer. HOLD / NEW / LATE are not trades.",
        "Fill = next week open. Hold 8 weeks or two consecutive off weeks. CAPTURE = return > 0. FAIL = return <= 0 or stopped. Wicks do not count.",
        "",
        "CAPTURE %s · FAIL %s · OPEN %s"
        % (len(capture), len(fail), len(open_rows)),
        "",
    ]
    rets = [to_float(r.get("ret_8w")) for r in scored]
    rets = [x for x in rets if x is not None]
    if rets:
        wins = [x for x in rets if x > 0]
        losses = [x for x in rets if x <= 0]
        gp = sum(wins)
        gl = abs(sum(losses))
        pf = (gp / gl) if gl else None
        hit = sum(1 for x in rets if x > 0) / float(len(rets))
        exp = sum(rets) / float(len(rets))
        lines.append(
            "Hit %.0f%% · expectancy %s · PF %s (equal $1, no costs)."
            % (
                hit * 100.0,
                fmt_pct(exp, 1),
                ("%.2f" % pf) if pf is not None else "n/a",
            )
        )
        lines.append("")
    if scored:
        lines.append("By grade:")
        for g in ("A", "B", "C"):
            bucket = [r for r in scored if r.get("grade") == g]
            if not bucket:
                continue
            wins = sum(1 for r in bucket if r.get("result") == "CAPTURE")
            lines.append("- %s: %s/%s CAPTURE" % (g, wins, len(bucket)))
        lines.append("")
        lines.extend(
            [
                "| date | ticker | grade | weeks | entry | 8w | result | why |",
                "|---|---|---|---:|---:|---:|---|---|",
            ]
        )
        for r in scored[-60:]:
            lines.append(
                "| %s | %s | %s | %s | %s | %s | %s | %s |"
                % (
                    r.get("asof"),
                    r.get("ticker"),
                    r.get("grade") or "—",
                    r.get("weeks_in_trend") if r.get("weeks_in_trend") is not None else "—",
                    fmt(r.get("entry_px")),
                    fmt_pct(r.get("ret_8w"), 0),
                    r.get("result"),
                    r.get("reason") or "",
                )
            )
        lines.append("")
    if open_rows:
        lines.extend(["## Open — waiting on 8 weeks of tape", "", "| date | ticker | grade | entry | weeks |", "|---|---|---|---:|---:|"])
        for r in open_rows[-40:]:
            lines.append(
                "| %s | %s | %s | %s | %s |"
                % (
                    r.get("asof"),
                    r.get("ticker"),
                    r.get("grade") or "—",
                    fmt(r.get("entry_px")),
                    r.get("weeks_in_trend") if r.get("weeks_in_trend") is not None else "—",
                )
            )
        lines.append("")
    if not scored and not open_rows:
        lines.append("No ADD signals on tape yet.")
        lines.append("")
    return "\n".join(lines)
