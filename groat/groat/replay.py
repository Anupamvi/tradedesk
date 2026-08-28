"""Universe replay: every independent setup print on cached tape.

Stock walk is always on. Options use hist/strikes cache, then a capped HTTP
budget. Not an LLM. Not X-HOT. Does not change daily gates.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from groat.config import (
    EARNINGS_EXEMPT,
    HOLD_SESSIONS,
    INDEX_TICKERS,
    MACRO_TICKERS,
    SECTOR_ETFS,
    load_universe,
)
from groat.gates import trade_park_reason
from groat.evidence import (
    _overlaps,
    earn_asof,
    _group_at,
    options_proxy,
    price_like,
    walk_stock_plan,
)
from groat.num import fmt, to_float
from groat.orats import _read_json, cores_cache, earnings_cache, fetch_strikes, load_usage, rows_of
from groat.prices import load_cached_bars
from groat.setups import SETUP_NAMES, classify_setups
from groat.structure import stock_plan
from groat.technicals import snapshot

REPLAY_MAX_PER_TICKER = 40
REPLAY_MAX_STRIKE_HTTP = 40


def _skip_name(ticker: str) -> bool:
    up = str(ticker).upper()
    if up in INDEX_TICKERS or up in MACRO_TICKERS or up in SECTOR_ETFS:
        return True
    if up in EARNINGS_EXEMPT:
        return True
    return False


def _hist_rows(ticker: str) -> list:
    payload = _read_json(earnings_cache(ticker))
    return rows_of(payload) if payload is not None else []


def _core_row(asof: str, ticker: str, cores: dict) -> Optional[dict]:
    row = cores.get(ticker)
    if isinstance(row, dict):
        return row
    return None


def _pick_option(strikes: list, direction: str, earn: dict) -> Optional[dict]:
    if direction == "bullish":
        order = ("debit_call_spread", "long_call")
    else:
        order = ("debit_put_spread", "long_put")
    for inst in order:
        cand = price_like(inst, strikes, earn)
        if cand and cand.get("ok"):
            return cand
    return None


def _fetch_strikes(
    ticker: str,
    day: str,
    token: str,
    today: str,
    getter,
    max_requests,
    budget: List[int],
    max_http: int,
) -> List[dict]:
    use_http = bool(token and budget[0] < max_http)
    pack = fetch_strikes(
        day,
        [ticker],
        token or "",
        today,
        getter=getter,
        max_requests=(max_requests if use_http else 0),
        dte="21,30,45,60",
    )
    budget[0] += int(pack.get("http") or 0)
    return (pack.get("rows") or {}).get(ticker) or []


def _summarize(hits: List[dict]) -> dict:
    stock = [h for h in hits if h.get("r") is not None]
    wins = sum(1 for h in stock if h.get("result") == "win")
    losses = sum(1 for h in stock if h.get("result") == "loss")
    timed = sum(1 for h in stock if h.get("result") == "time")
    rs = [float(h["r"]) for h in stock]
    opt = [h for h in hits if h.get("opt_priced")]
    pnls = [to_float(h.get("opt_pnl_per_risk")) for h in opt]
    pnls = [p for p in pnls if p is not None]
    be = [1 if h.get("opt_be_hit") else 0 for h in opt if h.get("opt_be_hit") is not None]
    pops = [to_float(h.get("opt_pop")) for h in opt]
    pops = [p for p in pops if p is not None]
    return {
        "n": len(stock),
        "wins": wins,
        "losses": losses,
        "time": timed,
        "win_rate": (wins / float(len(stock))) if stock else None,
        "avg_r": (sum(rs) / float(len(rs))) if rs else None,
        "opt_n": len(opt),
        "opt_avg_pnl_per_risk": (sum(pnls) / float(len(pnls))) if pnls else None,
        "opt_be_rate": (sum(be) / float(len(be))) if be else None,
        "opt_mean_pop": (sum(pops) / float(len(pops))) if pops else None,
        "opt_unpriced": sum(1 for h in hits if h.get("opt_tried") and not h.get("opt_priced")),
    }


def walk_ticker(
    ticker: str,
    bars: list,
    asof: str,
    bars_map: dict,
    spy_bars: list,
    hist_rows: list,
    core: Optional[dict],
    limit: int = REPLAY_MAX_PER_TICKER,
) -> List[dict]:
    days = [str(b.get("date") or "")[:10] for b in bars if str(b.get("date") or "")[:10] < asof]
    days = [d for d in days if len(d) == 10]
    hits = []
    for day in reversed(days):
        if len(hits) >= limit:
            break
        if _overlaps(day, hits, bars):
            continue
        snap = snapshot(bars, day, bench_bars=spy_bars)
        if not snap.get("ok") or snap.get("stale"):
            continue
        if to_float(snap.get("atr14")) is None:
            continue
        group_row = _group_at(ticker, day, bars_map, spy_bars)
        earn = earn_asof(ticker, day, hist_rows, core)
        setup = classify_setups(snap, group_row=group_row, earnings=earn, bars=list(bars))
        primary = setup.get("primary")
        direction = setup.get("direction")
        if not primary or direction not in ("bullish", "bearish"):
            continue
        if trade_park_reason(primary, snap, setup):
            continue
        plan = stock_plan({**snap, "primary": primary, "chase": setup.get("chase")}, direction)
        if not plan.get("ok"):
            continue
        outcome = walk_stock_plan(bars, day, plan)
        if outcome.get("result") == "incomplete":
            continue
        if outcome.get("result") == "time" and int(outcome.get("hold") or 0) < HOLD_SESSIONS:
            continue
        hit = {
            "ticker": ticker,
            "date": day,
            "primary": primary,
            "direction": direction,
            "result": outcome.get("result"),
            "r": outcome.get("r"),
            "exit_date": outcome.get("exit_date"),
            "exit_px": outcome.get("exit_px"),
            "hold": outcome.get("hold"),
            "entry": plan.get("entry"),
            "opt_tried": False,
            "opt_priced": False,
        }
        hits.append(hit)
    return hits


def fill_option_slice(
    hits: List[dict],
    bars_map: dict,
    cores: dict,
    token: str,
    today: str,
    getter,
    max_requests,
    max_http: int,
) -> dict:
    budget = [0]
    priced_before = sum(1 for h in hits if h.get("opt_priced"))
    if max_http <= 0:
        return {"http": 0, "priced_new": 0, "priced_total": priced_before}
    for hit in hits:
        if budget[0] >= max_http:
            break
        if hit.get("opt_priced"):
            continue
        ticker = str(hit.get("ticker") or "")
        day = str(hit.get("date") or "")
        entry_px = to_float(hit.get("entry"))
        exit_px = to_float(hit.get("exit_px"))
        if not ticker or not day or entry_px is None or exit_px is None:
            continue
        hit["opt_tried"] = True
        strikes = _fetch_strikes(
            ticker,
            day,
            token,
            today,
            getter,
            max_requests,
            budget,
            max_http,
        )
        earn = earn_asof(ticker, day, _hist_rows(ticker), cores.get(ticker))
        priced = _pick_option(strikes, str(hit.get("direction") or ""), earn) if strikes else None
        if priced:
            proxy = options_proxy(priced, entry_px, exit_px, int(hit.get("hold") or 0), str(hit.get("direction") or ""))
            hit["opt_priced"] = bool(proxy.get("priced"))
            hit["opt_pnl_per_risk"] = proxy.get("pnl_per_risk")
            hit["opt_be_hit"] = proxy.get("be_hit")
            hit["opt_pop"] = priced.get("naive_pop")
            hit["opt_instrument"] = priced.get("instrument")
    priced_after = sum(1 for h in hits if h.get("opt_priced"))
    return {"http": budget[0], "priced_new": priced_after - priced_before, "priced_total": priced_after}


def run_replay(
    asof: str,
    token: str = "",
    today: Optional[str] = None,
    getter=None,
    max_requests: Optional[int] = None,
    max_strike_http: int = REPLAY_MAX_STRIKE_HTTP,
    option_slices: int = 0,
    universe: Optional[List[str]] = None,
    bars_by_ticker: Optional[dict] = None,
) -> dict:
    today = today or asof
    names = list(universe or load_universe())
    extra = list(INDEX_TICKERS) + list(MACRO_TICKERS) + list(SECTOR_ETFS)
    bars_map = {}
    if bars_by_ticker is not None:
        bars_map = {str(k).upper(): list(v or []) for k, v in bars_by_ticker.items()}
    else:
        wanted = []
        seen = set()
        for name in names + extra:
            up = str(name).upper()
            if up in seen:
                continue
            seen.add(up)
            wanted.append(up)
        for name in wanted:
            bars_map[name] = load_cached_bars(name)
    spy_bars = bars_map.get("SPY") or []
    cores_payload = _read_json(cores_cache(asof)) or {}
    cores = {}
    for row in rows_of(cores_payload):
        t = str(row.get("ticker") or "").upper()
        if t:
            cores[t] = row
    all_hits = []
    tape_from = None
    tape_to = None
    for ticker in names:
        up = str(ticker).upper()
        if _skip_name(up):
            continue
        bars = bars_map.get(up) or []
        if len(bars) < 40:
            continue
        if bars:
            d0 = str(bars[0].get("date") or "")[:10]
            d1 = str(bars[-1].get("date") or "")[:10]
            if not tape_from or d0 < tape_from:
                tape_from = d0
            if not tape_to or d1 > tape_to:
                tape_to = d1
        hits = walk_ticker(
            up,
            bars,
            asof,
            bars_map,
            spy_bars,
            _hist_rows(up),
            _core_row(asof, up, cores),
        )
        all_hits.extend(hits)
    slices = []
    http_total = 0
    allow_http = bars_by_ticker is None and max_strike_http > 0 and option_slices > 0
    if allow_http:
        for i in range(int(option_slices)):
            one = fill_option_slice(
                all_hits,
                bars_map,
                cores,
                token,
                today,
                getter,
                max_requests,
                max_strike_http,
            )
            one["slice"] = i + 1
            slices.append(one)
            http_total += int(one.get("http") or 0)
    by_setup = {code: [] for code in SETUP_NAMES}
    for hit in all_hits:
        by_setup.setdefault(hit.get("primary") or "?", []).append(hit)
    setups = []
    for code in SETUP_NAMES:
        row = _summarize(by_setup.get(code) or [])
        row["primary"] = code
        row["primary_name"] = SETUP_NAMES[code]
        setups.append(row)
    overall = _summarize(all_hits)
    usage = load_usage()
    return {
        "asof": asof,
        "mode": "replay",
        "tape_from": tape_from,
        "tape_to": tape_to,
        "n_hits": len(all_hits),
        "setups": setups,
        "overall": overall,
        "hits": all_hits,
        "option_slices": slices,
        "strike_http": http_total,
        "orats_http": http_total,
        "orats_requests_used": usage.get("used") or 0,
        "orats_requests_left": usage.get("left") or 0,
        "note": (
            "Surviving setups only (park B/C/G; park post-rip E). "
            "Cached daily tape, independent entries (15-session gap). "
            "Options: same dates, debit then long, conservative entry structure, "
            "delta+theta P&L clamped to defined risk — not a live option mark. "
            "X-HOT not tested."
        ),
    }


def render_replay(payload: dict) -> str:
    overall = payload.get("overall") or {}
    lines = [
        "# Groat replay %s" % (payload.get("asof") or ""),
        "",
        "Tape **%s → %s**. Independent setup prints, stock 2:1 plan + options on the same dates."
        % (payload.get("tape_from") or "DATA UNAVAILABLE", payload.get("tape_to") or "DATA UNAVAILABLE"),
        "",
        str(payload.get("note") or ""),
        "",
        "Hits **%s**. Strike HTTP **%s**. Account ledger used **%s** / left **%s**."
        % (
            payload.get("n_hits") or 0,
            payload.get("strike_http") or 0,
            payload.get("orats_requests_used") or 0,
            payload.get("orats_requests_left") or 0,
        ),
        "",
    ]
    if payload.get("option_slices"):
        lines.append("| slice | HTTP | priced new | priced total |")
        lines.append("|---:|---:|---:|---:|")
        for one in payload.get("option_slices") or []:
            lines.append(
                "| %s | %s | %s | %s |"
                % (one.get("slice"), one.get("http") or 0, one.get("priced_new") or 0, one.get("priced_total") or 0)
            )
        lines.append("")
    lines.extend(
        [
            "| setup | stock n | W/L/time | win%% | avg R | options n | opt P&L/risk | BE vs naive POP | unpriced |",
            "|---|---:|---|---:|---:|---:|---:|---|---:|",
        ]
    )
    for row in payload.get("setups") or []:
        n = int(row.get("n") or 0)
        on = int(row.get("opt_n") or 0)
        wr = row.get("win_rate")
        be = row.get("opt_be_rate")
        pop = row.get("opt_mean_pop")
        vs = "—"
        if on and (be is not None or pop is not None):
            vs = "%s vs %s" % (
                ("%.0f%%" % (be * 100)) if be is not None else "n/a",
                ("%.0f%%" % (pop * 100)) if pop is not None else "n/a",
            )
        lines.append(
            "| **%s** %s | %s | %s/%s/%s | %s | %s | %s | %s | %s | %s |"
            % (
                row.get("primary"),
                row.get("primary_name") or "",
                n,
                row.get("wins") or 0,
                row.get("losses") or 0,
                row.get("time") or 0,
                ("%.0f%%" % (wr * 100)) if wr is not None else "—",
                fmt(row.get("avg_r"), 2) if n else "—",
                on,
                fmt(row.get("opt_avg_pnl_per_risk"), 2) if on else "—",
                vs,
                row.get("opt_unpriced") or 0,
            )
        )
    n = int(overall.get("n") or 0)
    on = int(overall.get("opt_n") or 0)
    wr = overall.get("win_rate")
    lines.append(
        "| **ALL** | %s | %s/%s/%s | %s | %s | %s | %s | — | %s |"
        % (
            n,
            overall.get("wins") or 0,
            overall.get("losses") or 0,
            overall.get("time") or 0,
            ("%.0f%%" % (wr * 100)) if wr is not None else "—",
            fmt(overall.get("avg_r"), 2) if n else "—",
            on,
            fmt(overall.get("opt_avg_pnl_per_risk"), 2) if on else "—",
            overall.get("opt_unpriced") or 0,
        )
    )
    lines.extend(["", "Keep/kill: avg R is expectancy per independent stock print. Options column is only where a 21–75 DTE debit/long cleared liquidity+earnings. Empty options n is DATA UNAVAILABLE, not a zero edge.", ""])
    return "\n".join(lines)
