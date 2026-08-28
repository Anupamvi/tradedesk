"""Post-selection analogs: stock setup walk + thin options replay.

Runs after the daily TRADE shortlist. Same ticker + same setup, not a universe
backtest. Does not change gates. Small n. X-HOT is not in this file.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence

from groat.config import (
    EARNINGS_HOLD_DAYS,
    EVIDENCE_MAX_ANALOGS,
    EVIDENCE_MAX_EARNINGS_HTTP,
    EVIDENCE_MAX_STRIKE_HTTP,
    HOLD_SESSIONS,
    STRIKE_DTE,
    ticker_etf,
    ticker_group,
)
from groat.earnings import cadence_next, parse_any_date, parse_ern_history
from groat.num import fmt, to_float
from groat.orats import fetch_hist_earnings, fetch_strikes
from groat.rotation import classify_group
from groat.setups import SETUP_NAMES, classify_setups
from groat.structure import credit_spread, debit_spread, long_option, stock_plan
from groat.technicals import snapshot


def _days(later: str, earlier: str) -> Optional[int]:
    try:
        return (datetime.strptime(later[:10], "%Y-%m-%d") - datetime.strptime(earlier[:10], "%Y-%m-%d")).days
    except ValueError:
        return None


def earn_asof(ticker: str, day: str, hist_rows: Optional[list], core: Optional[dict]) -> dict:
    """Point-in-time last/next earnings. Do not use today's wksNextErn."""
    dates = []
    seen = set()
    for raw in hist_rows or []:
        stamp = parse_any_date((raw or {}).get("earnDate") or (raw or {}).get("date"))
        if stamp and stamp not in seen:
            dates.append(stamp)
            seen.add(stamp)
    for stamp in parse_ern_history(core):
        if stamp not in seen:
            dates.append(stamp)
            seen.add(stamp)
    dates.sort()
    past = [d for d in dates if d <= day]
    last = past[-1] if past else None
    nxt = cadence_next(past, day)
    days = _days(nxt, day) if nxt else None
    return {
        "ticker": ticker,
        "date": nxt,
        "last": last,
        "source": "orats.hist/earnings cadence" if nxt else "DATA UNAVAILABLE",
        "usable": nxt is not None,
        "days": days,
        "overlaps_hold": bool(days is not None and 0 <= days <= EARNINGS_HOLD_DAYS),
        "history": past[-8:],
    }


def _group_at(ticker: str, day: str, bars_map: dict, spy_bars: list) -> dict:
    etf = ticker_etf(ticker)
    snap = snapshot(bars_map.get(etf) or [], day, bench_bars=spy_bars)
    rs20 = to_float(snap.get("rs_20"))
    rs60 = to_float(snap.get("rs_60"))
    accel = None
    if rs20 is not None and rs60 is not None:
        accel = rs20 - rs60
    status = classify_group(rs20, rs60, accel, str(snap.get("trend") or ""))
    return {
        "etf": etf,
        "group": ticker_group(ticker),
        "status": status,
        "ok": bool(snap.get("ok")),
        "rs_20": rs20,
        "rs_60": rs60,
        "accel": accel,
        "trend": snap.get("trend"),
    }


def walk_stock_plan(bars: Sequence[dict], start: str, plan: dict, hold: int = HOLD_SESSIONS) -> dict:
    entry = to_float(plan.get("entry"))
    stop = to_float(plan.get("stop"))
    target = to_float(plan.get("target"))
    side = str(plan.get("side") or "long")
    if entry is None or stop is None or target is None:
        return {"result": "incomplete", "r": None, "exit_date": None, "exit_px": None, "hold": 0}
    risk = abs(entry - stop)
    if risk <= 0:
        return {"result": "incomplete", "r": None, "exit_date": None, "exit_px": None, "hold": 0}
    idx = None
    for i, bar in enumerate(bars):
        if str(bar.get("date") or "")[:10] == start:
            idx = i
            break
    if idx is None:
        return {"result": "incomplete", "r": None, "exit_date": None, "exit_px": None, "hold": 0}
    future = list(bars[idx + 1 : idx + 1 + hold])
    if not future:
        return {"result": "incomplete", "r": None, "exit_date": None, "exit_px": None, "hold": 0}
    for j, bar in enumerate(future):
        hi = to_float(bar.get("high"))
        lo = to_float(bar.get("low"))
        day = str(bar.get("date") or "")[:10]
        if hi is None or lo is None:
            continue
        if side == "short":
            stop_hit = hi >= stop
            tgt_hit = lo <= target
            if stop_hit:
                return {"result": "loss", "r": -1.0, "exit_date": day, "exit_px": stop, "hold": j + 1}
            if tgt_hit:
                return {
                    "result": "win",
                    "r": abs(entry - target) / risk,
                    "exit_date": day,
                    "exit_px": target,
                    "hold": j + 1,
                }
        else:
            stop_hit = lo <= stop
            tgt_hit = hi >= target
            if stop_hit:
                return {"result": "loss", "r": -1.0, "exit_date": day, "exit_px": stop, "hold": j + 1}
            if tgt_hit:
                return {
                    "result": "win",
                    "r": abs(target - entry) / risk,
                    "exit_date": day,
                    "exit_px": target,
                    "hold": j + 1,
                }
    last = future[-1]
    close = to_float(last.get("close"))
    day = str(last.get("date") or "")[:10]
    if close is None:
        return {"result": "incomplete", "r": None, "exit_date": day, "exit_px": None, "hold": len(future)}
    if side == "short":
        r_mult = (entry - close) / risk
    else:
        r_mult = (close - entry) / risk
    return {"result": "time", "r": r_mult, "exit_date": day, "exit_px": close, "hold": len(future)}


def price_like(instrument: str, strikes: Sequence[dict], earn: dict) -> Optional[dict]:
    inst = str(instrument or "")
    if inst == "debit_call_spread":
        return debit_spread(strikes, "bullish", earn)
    if inst == "debit_put_spread":
        return debit_spread(strikes, "bearish", earn)
    if inst == "long_call":
        return long_option(strikes, "bullish", earn)
    if inst == "long_put":
        return long_option(strikes, "bearish", earn)
    if inst == "put_credit_spread":
        return credit_spread(strikes, "bullish", earn, True)
    if inst == "call_credit_spread":
        return credit_spread(strikes, "bearish", earn, True)
    return None


def options_proxy(picked: dict, entry_px: float, exit_px: float, hold_days: int, direction: str) -> dict:
    debit = to_float(picked.get("target_debit"))
    credit = to_float(picked.get("target_credit"))
    delta = to_float(picked.get("delta")) or 0.0
    theta = to_float(picked.get("theta")) or 0.0
    width = to_float(picked.get("width"))
    be = to_float(picked.get("breakeven"))
    if direction == "bearish":
        move = entry_px - exit_px
        be_hit = be is not None and exit_px <= be
    else:
        move = exit_px - entry_px
        be_hit = be is not None and exit_px >= be
    raw = delta * move + theta * float(hold_days)
    if debit is not None and debit > 0:
        cap = (width - debit) if width and width > debit else debit * 2.0
        pnl = max(-debit, min(cap, raw))
        return {"pnl": pnl, "pnl_per_risk": pnl / debit, "be_hit": be_hit, "priced": True}
    if credit is not None and credit > 0:
        max_loss = (width - credit) if width and width > credit else credit * 4.0
        pnl = max(-max_loss, min(credit, raw))
        return {"pnl": pnl, "pnl_per_risk": pnl / max_loss if max_loss else None, "be_hit": be_hit, "priced": True}
    return {"pnl": None, "pnl_per_risk": None, "be_hit": None, "priced": False}


def _sessions_between(bars: Sequence[dict], start: str, end: str) -> int:
    n = 0
    for bar in bars:
        day = str(bar.get("date") or "")[:10]
        if start < day <= end:
            n += 1
    return n


def _overlaps(day: str, taken: List[dict], bars: Sequence[dict], gap: int = HOLD_SESSIONS) -> bool:
    for row in taken:
        start = row.get("date") or ""
        end = row.get("exit_date") or start
        if start <= day <= end:
            return True
        later, earlier = (start, day) if start > day else (day, start)
        if _sessions_between(bars, earlier, later) <= gap:
            return True
    return False


def find_stock_analogs(
    ticker: str,
    bars: Sequence[dict],
    asof: str,
    primary: str,
    direction: str,
    bars_map: dict,
    spy_bars: list,
    hist_rows: Optional[list],
    core: Optional[dict],
    limit: int = EVIDENCE_MAX_ANALOGS,
) -> Dict[str, Any]:
    days = [str(b.get("date") or "")[:10] for b in bars if str(b.get("date") or "")[:10] < asof]
    days = [d for d in days if len(d) == 10]
    chase_n = 0
    scanned = 0
    hits = []
    for day in reversed(days):
        if len(hits) >= limit:
            break
        if _overlaps(day, hits, bars):
            continue
        snap = snapshot(bars, day, bench_bars=spy_bars)
        scanned += 1
        if not snap.get("ok") or snap.get("stale"):
            continue
        if to_float(snap.get("atr14")) is None:
            continue
        group_row = _group_at(ticker, day, bars_map, spy_bars)
        earn = earn_asof(ticker, day, hist_rows, core)
        setup = classify_setups(snap, group_row=group_row, earnings=earn, bars=list(bars))
        if setup.get("primary") != primary:
            continue
        if setup.get("direction") != direction:
            continue
        plan = stock_plan({**snap, "primary": primary, "chase": setup.get("chase")}, direction)
        if not plan.get("ok"):
            if plan.get("reason") == "chase_filter":
                chase_n += 1
            continue
        outcome = walk_stock_plan(bars, day, plan)
        if outcome.get("result") == "incomplete":
            continue
        if outcome.get("result") == "time" and int(outcome.get("hold") or 0) < HOLD_SESSIONS:
            continue
        hits.append(
            {
                "date": day,
                "primary": primary,
                "direction": direction,
                "entry": plan.get("entry"),
                "stop": plan.get("stop"),
                "target": plan.get("target"),
                "result": outcome.get("result"),
                "r": outcome.get("r"),
                "exit_date": outcome.get("exit_date"),
                "exit_px": outcome.get("exit_px"),
                "hold": outcome.get("hold"),
            }
        )
    return {"hits": hits, "chase_skipped": chase_n, "scanned": scanned}


def _summarize_stock(hits: List[dict]) -> dict:
    scored = [h for h in hits if h.get("r") is not None]
    if not scored:
        return {"n": 0, "wins": 0, "losses": 0, "time": 0, "win_rate": None, "avg_r": None, "expectancy_r": None}
    wins = sum(1 for h in scored if h.get("result") == "win")
    losses = sum(1 for h in scored if h.get("result") == "loss")
    timed = sum(1 for h in scored if h.get("result") == "time")
    rs = [float(h["r"]) for h in scored]
    avg = sum(rs) / float(len(rs))
    return {
        "n": len(scored),
        "wins": wins,
        "losses": losses,
        "time": timed,
        "win_rate": wins / float(len(scored)),
        "avg_r": avg,
        "expectancy_r": avg,
    }


def _summarize_opt(rows: List[dict]) -> dict:
    priced = [r for r in rows if r.get("priced")]
    if not priced:
        return {"n": 0, "avg_pnl_per_risk": None, "be_rate": None, "mean_pop": None}
    pnls = [to_float(r.get("pnl_per_risk")) for r in priced]
    pnls = [p for p in pnls if p is not None]
    be = [1 if r.get("be_hit") else 0 for r in priced if r.get("be_hit") is not None]
    pops = [to_float(r.get("naive_pop")) for r in priced]
    pops = [p for p in pops if p is not None]
    return {
        "n": len(priced),
        "avg_pnl_per_risk": (sum(pnls) / float(len(pnls))) if pnls else None,
        "be_rate": (sum(be) / float(len(be))) if be else None,
        "mean_pop": (sum(pops) / float(len(pops))) if pops else None,
    }


def _fetch_analog_strikes(
    ticker: str,
    day: str,
    token: str,
    today: str,
    getter,
    max_requests,
    allow_http: bool,
    budget: List[int],
) -> List[dict]:
    use_http = bool(allow_http and token and budget[0] < EVIDENCE_MAX_STRIKE_HTTP)
    pack = fetch_strikes(
        day,
        [ticker],
        token or "",
        today,
        getter=getter,
        max_requests=(max_requests if use_http else 0),
        dte=STRIKE_DTE,
    )
    budget[0] += int(pack.get("http") or 0)
    return (pack.get("rows") or {}).get(ticker) or []


def attach_evidence(
    asof: str,
    trades: Sequence[dict],
    picks: dict,
    bars_map: dict,
    hist_e: Optional[dict] = None,
    cores: Optional[dict] = None,
    token: str = "",
    today: Optional[str] = None,
    getter=None,
    max_requests: Optional[int] = None,
    allow_orats_http: bool = False,
) -> dict:
    hist_e = dict(hist_e or {})
    cores = cores or {}
    today = today or asof
    spy_bars = bars_map.get("SPY") or []
    strike_budget = [0]
    earn_http = 0
    rows = []
    ordered = []
    seen_q = set()
    best_opt = (picks or {}).get("best_options") or {}
    best_opt_ticker = str(best_opt.get("ticker") or "").upper()
    if best_opt_ticker:
        for trade in trades:
            if str(trade.get("ticker") or "").upper() == best_opt_ticker:
                ordered.append(trade)
                seen_q.add(best_opt_ticker)
                break
    for trade in trades:
        ticker = str(trade.get("ticker") or "").upper()
        if not ticker or ticker in seen_q:
            continue
        if trade.get("choice") == "OPTIONS":
            ordered.append(trade)
            seen_q.add(ticker)
    for trade in trades:
        ticker = str(trade.get("ticker") or "").upper()
        if not ticker or ticker in seen_q:
            continue
        ordered.append(trade)
        seen_q.add(ticker)
    seen = set()
    for trade in ordered:
        ticker = str(trade.get("ticker") or "").upper()
        if ticker in seen:
            continue
        seen.add(ticker)
        primary = str(trade.get("primary") or "")
        direction = str(trade.get("direction") or "")
        bars = bars_map.get(ticker) or []
        if not primary or direction not in ("bullish", "bearish") or not bars:
            row = {
                "ticker": ticker,
                "primary": primary,
                "choice": trade.get("choice"),
                "stock": {"n": 0},
                "options": {"n": 0},
                "note": "DATA UNAVAILABLE",
            }
            trade["evidence"] = row
            rows.append(row)
            continue
        if ticker not in hist_e and allow_orats_http and token and earn_http < EVIDENCE_MAX_EARNINGS_HTTP:
            pack = fetch_hist_earnings(ticker, token, getter=getter, max_requests=max_requests)
            hist_e[ticker] = pack.get("rows") or []
            earn_http += int(pack.get("http") or 0)
        analog = find_stock_analogs(
            ticker,
            bars,
            asof,
            primary,
            direction,
            bars_map,
            spy_bars,
            hist_e.get(ticker),
            cores.get(ticker),
        )
        hits = analog.get("hits") or []
        stock_sum = _summarize_stock(hits)
        opt_rows = []
        want_opt = trade.get("choice") == "OPTIONS"
        instrument = str(((trade.get("picked") or {}) if isinstance(trade.get("picked"), dict) else {}).get("instrument") or "")
        if want_opt and instrument:
            for hit in hits:
                day = hit["date"]
                strikes = _fetch_analog_strikes(
                    ticker,
                    day,
                    token,
                    today,
                    getter,
                    max_requests,
                    allow_orats_http,
                    strike_budget,
                )
                if not strikes:
                    opt_rows.append({"date": day, "priced": False, "reason": "chain DATA UNAVAILABLE"})
                    continue
                earn = earn_asof(ticker, day, hist_e.get(ticker), cores.get(ticker))
                priced = price_like(instrument, strikes, earn)
                if not priced or not priced.get("ok"):
                    opt_rows.append(
                        {
                            "date": day,
                            "priced": False,
                            "reason": (priced or {}).get("reason") or "no analog structure",
                        }
                    )
                    continue
                entry_px = to_float(hit.get("entry"))
                exit_px = to_float(hit.get("exit_px"))
                if entry_px is None or exit_px is None:
                    opt_rows.append({"date": day, "priced": False, "reason": "missing exit px"})
                    continue
                proxy = options_proxy(priced, entry_px, exit_px, int(hit.get("hold") or 0), direction)
                proxy.update(
                    {
                        "date": day,
                        "naive_pop": priced.get("naive_pop"),
                        "instrument": priced.get("instrument"),
                        "target_debit": priced.get("target_debit"),
                        "target_credit": priced.get("target_credit"),
                    }
                )
                opt_rows.append(proxy)
        opt_sum = _summarize_opt(opt_rows)
        weak = int(stock_sum.get("n") or 0) >= 5 and (stock_sum.get("avg_r") is not None) and stock_sum["avg_r"] < 0
        row = {
            "ticker": ticker,
            "primary": primary,
            "primary_name": SETUP_NAMES.get(primary, primary),
            "direction": direction,
            "choice": trade.get("choice"),
            "instrument": instrument,
            "stock": stock_sum,
            "options": opt_sum,
            "chase_skipped": analog.get("chase_skipped") or 0,
            "hits": hits,
            "opt_rows": opt_rows,
            "weak": weak,
            "note": "",
        }
        if stock_sum["n"] == 0:
            row["note"] = "no same-setup analog in cached tape"
        trade["evidence"] = {
            "ticker": ticker,
            "stock": stock_sum,
            "options": opt_sum,
            "weak": weak,
            "note": row["note"],
        }
        rows.append(row)
    if picks is not None:
        weak_names = [r["ticker"] for r in rows if r.get("weak")]
        if weak_names:
            picks["evidence_line"] = "Analog caution (stock setup avg R < 0, n≥5): " + ", ".join(weak_names) + "."
        else:
            picks["evidence_line"] = ""
    return {
        "asof": asof,
        "rows": rows,
        "strike_http": strike_budget[0],
        "earnings_http": earn_http,
        "http": strike_budget[0] + earn_http,
        "note": (
            "Same ticker + same setup on cached tape. Options use hist/strikes when cached or under the daily cap. "
            "Option P&L is delta+theta clamped to defined risk, not a live exit mark. "
            "Not a system win rate. X-HOT is not tested. Does not change today's gates."
        ),
    }


def render_evidence(payload: dict) -> List[str]:
    rows = payload.get("rows") or []
    lines = [
        "## Evidence — same setup, this ticker",
        "",
        str(payload.get("note") or ""),
        "",
    ]
    if not rows:
        lines.append("No TRADE rows to analog. Valid.")
        lines.append("")
        return lines
    lines.append("| ticker | setup | stock n | W/L/time | avg R | options n | opt P&L / risk | BE vs naive POP |")
    lines.append("|---|---|---:|---|---:|---:|---:|---|")
    for row in rows:
        stock = row.get("stock") or {}
        opt = row.get("options") or {}
        n = int(stock.get("n") or 0)
        if n:
            wlt = "%s/%s/%s" % (stock.get("wins") or 0, stock.get("losses") or 0, stock.get("time") or 0)
            avg = fmt(stock.get("avg_r"), 2)
        else:
            wlt = "—"
            avg = "—"
        on = int(opt.get("n") or 0)
        if on:
            pnl = fmt(opt.get("avg_pnl_per_risk"), 2)
            be = opt.get("be_rate")
            pop = opt.get("mean_pop")
            vs = "%s vs %s" % (
                ("%.0f%%" % (be * 100)) if be is not None else "n/a",
                ("%.0f%%" % (pop * 100)) if pop is not None else "n/a",
            )
        else:
            pnl = "—"
            vs = "—"
        flag = " ⚠" if row.get("weak") else ""
        lines.append(
            "| **%s**%s | %s %s | %s | %s | %s | %s | %s | %s |"
            % (
                row.get("ticker"),
                flag,
                row.get("primary") or "",
                row.get("primary_name") or "",
                n if n else "0",
                wlt,
                avg,
                on if on else "0",
                pnl,
                vs,
            )
        )
    lines.append("")
    lines.append("| ticker | analog dates (entry → exit / result / R) |")
    lines.append("|---|---|")
    for row in rows:
        bits = []
        for hit in (row.get("hits") or [])[:6]:
            r_txt = fmt(hit.get("r"), 2) if hit.get("r") is not None else "n/a"
            bits.append("%s→%s %s %sR" % (hit.get("date"), hit.get("exit_date") or "", hit.get("result"), r_txt))
        lines.append("| **%s** | %s |" % (row.get("ticker"), "; ".join(bits) if bits else (row.get("note") or "none")))
    lines.append("")
    return lines


def render_evidence_file(asof: str, payload: dict) -> str:
    lines = [
        "# Groat evidence %s" % asof,
        "",
    ]
    lines.extend(render_evidence(payload))
    lines.extend(
        [
            "HTTP this run: hist/strikes %s · hist/earnings %s."
            % (payload.get("strike_http") or 0, payload.get("earnings_http") or 0),
            "",
            "Empty analogs are valid. Do not invent missing chains.",
            "",
        ]
    )
    return "\n".join(lines)
