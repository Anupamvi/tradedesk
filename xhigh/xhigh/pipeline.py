"""Discovery only. Movers → last → ORATS → chain → all catalog structures. No cap. No harvest."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from xhigh import earnings as earnings_mod
from xhigh import orats as orats_mod
from xhigh import report
from xhigh import schwab
from xhigh import tape
from xhigh.config import OUT_DIR
from xhigh.dates import add_days, fmt_expiry
from xhigh.envload import ORATS_TOKEN_MISSING, load_orats_token
from xhigh.gates import load_gates
from xhigh.geometry import catalog_for_name, spot_from_quote, ticket_legal
from xhigh.num import fmt, to_float
from xhigh.orats import parse_core
from xhigh.schwab import flatten_chain, use_live_schwab
from xhigh.score import cheap_vol, confidence, ev_proxy, pop_delta


def _junk_symbol(name: str) -> bool:
    u = str(name or "").upper()
    if not u or len(u) > 5:
        return True
    if u.endswith(("W", "U", "R", "WS", "WT")):
        return True
    return False


def layer0_quote_ok(q: dict, gates: dict, ticker: str = "") -> bool:
    if ticker and _junk_symbol(ticker):
        return False
    g = gates.get("quote") or {}
    last = spot_from_quote(q)
    bid = to_float(q.get("bid"))
    ask = to_float(q.get("ask"))
    if last is None or last < float(g.get("min_last") or 15):
        return False
    if bid is None or bid <= float(g.get("min_bid") or 0):
        return False
    if ask is None:
        return False
    spread = ask - bid
    cap = max(float(g.get("max_spread_abs") or 0.10), float(g.get("max_spread_frac") or 0.01) * last)
    if spread > cap:
        return False
    asset = str(q.get("asset") or "").upper()
    if asset in ("WARRANT", "MUTUAL_FUND", "BOND"):
        return False
    return True


def orats_quality_ok(core: dict, gates: dict) -> bool:
    g = gates.get("orats") or {}
    mcap = to_float(core.get("mkt_cap"))
    vol = to_float(core.get("avg_opt_vol_20d"))
    if mcap is None or mcap < float(g.get("min_mkt_cap") or 2000):
        return False
    if vol is None or vol < float(g.get("min_avg_opt_vol_20d") or 200):
        return False
    tk = core.get("tk_over")
    if tk not in (None, "", 0, "0", False):
        try:
            if float(tk) != 0:
                return False
        except (TypeError, ValueError):
            if str(tk).lower() in ("true", "yes", "y"):
                return False
    return True


def _strategy_line(idea: dict) -> str:
    s = idea.get("structure")
    if s == "csp":
        return "SELL %s P" % idea.get("strike")
    if s == "put_credit":
        return "SELL %s P / BUY %s P" % (idea.get("short_strike"), idea.get("long_strike"))
    if s == "call_debit":
        return "BUY %s C / SELL %s C" % (idea.get("long_strike"), idea.get("short_strike"))
    if s == "call_credit":
        return "SELL %s C / BUY %s C" % (idea.get("short_strike"), idea.get("long_strike"))
    if s == "put_debit":
        return "BUY %s P / SELL %s P" % (idea.get("long_strike"), idea.get("short_strike"))
    if s == "iron_condor":
        return "SELL %s/%s P / SELL %s/%s C" % (
            idea.get("put_short"),
            idea.get("put_long"),
            idea.get("call_short"),
            idea.get("call_long"),
        )
    return str(s or "")


def _legal(idea: dict, last: float, gates: dict) -> bool:
    s = idea.get("structure")
    if s == "csp":
        return ticket_legal("csp", last, idea.get("strike"), None, gates)
    if s == "call_debit":
        return ticket_legal("call_debit", last, idea.get("long_strike"), idea.get("short_strike"), gates)
    if s == "put_debit":
        return ticket_legal("put_debit", last, idea.get("long_strike"), idea.get("short_strike"), gates)
    if s == "put_credit":
        return ticket_legal(
            "put_credit", last, gates=gates, short_put=idea.get("short_strike"), long_put=idea.get("long_strike")
        )
    if s == "call_credit":
        return ticket_legal(
            "call_credit", last, gates=gates, short_call=idea.get("short_strike"), long_call=idea.get("long_strike")
        )
    if s == "iron_condor":
        return ticket_legal(
            "iron_condor",
            last,
            gates=gates,
            short_put=idea.get("put_short"),
            long_put=idea.get("put_long"),
            short_call=idea.get("call_short"),
            long_call=idea.get("call_long"),
        )
    return False


def _earn_s(earn: dict) -> str:
    day = earn.get("date")
    if not day:
        return "DATA UNAVAILABLE"
    src = str(earn.get("source") or "")
    if "wksNextErn" in src or "cadence" in src:
        return "est %s" % day
    return str(day)


def _ticket(ticker: str, last: float, earn: dict, idea: dict, gates: dict) -> Optional[dict]:
    if idea.get("structure") in ("csp", "put_credit", "call_credit", "iron_condor"):
        target_s = "credit %s" % fmt(idea.get("credit"))
    elif idea.get("structure") in ("call_debit", "put_debit"):
        target_s = "debit %s" % fmt(idea.get("debit"))
    else:
        target_s = fmt(last)
    pop = pop_delta(idea)
    ev = ev_proxy(idea, pop)
    row = {
        "desk": "xhigh",
        "ticker": ticker,
        "last": round(last, 2),
        "structure": idea.get("structure"),
        "strategy": _strategy_line(idea),
        "expiry": idea.get("expiry") or "",
        "expiry_s": fmt_expiry(idea.get("expiry") or "") if idea.get("expiry") else "",
        "dte": idea.get("dte"),
        "target_s": target_s,
        "otm_s": idea.get("otm_s") or "",
        "pop_delta": None if pop is None else round(pop, 3),
        "pop_s": "DATA UNAVAILABLE" if pop is None else "%.0f%%" % (pop * 100),
        "ev_proxy": None if ev is None else round(ev, 2),
        "invalidation": idea.get("invalidation"),
        "earnings_date": earn.get("date"),
        "earnings_source": earn.get("source"),
        "earn_s": _earn_s(earn),
        "x_tag": "DATA UNAVAILABLE",
        "spot": last,
    }
    row.update({k: v for k, v in idea.items() if k not in row})
    if isinstance(row.get("invalidation"), float):
        row["invalidation"] = round(row["invalidation"], 2)
    row["conf"] = confidence(row, earn, gates)
    from xhigh.rec import decorate

    return decorate(row, gates)


def _spy_tape(date: str, live: bool) -> dict:
    quotes = schwab.quotes_many(["SPY", "VIX", "$VIX.X"], date) if live else {}
    spy_q = quotes.get("SPY") or {}
    spy_last = spot_from_quote(spy_q)
    bars = schwab.price_history_bars("SPY", date, use_cache=False) if live else []
    ret = None
    closes = [to_float(b.get("close")) for b in bars if to_float(b.get("close")) is not None]
    if len(closes) >= 6 and closes[-1] and closes[-6]:
        ret = closes[-1] / closes[-6] - 1.0
    vix = spot_from_quote(quotes.get("VIX") or quotes.get("$VIX.X") or {})
    return {
        "spy_last": spy_last,
        "spy_5d": None if ret is None else round(ret * 100, 2),
        "vix_last": vix,
    }


def build_full(
    date: str,
    out_dir: Optional[Path] = None,
    live_schwab: bool = False,
    no_schwab: bool = False,
    max_orats_http: int = 15,
    orats_token_file: Optional[str] = None,
    tickers: Optional[List[str]] = None,
) -> Dict[str, Any]:
    schwab.reset_http()
    orats_mod.reset_process_http()
    gates = load_gates()
    live = use_live_schwab(date, live_flag=live_schwab, no_schwab=no_schwab)
    macro = _spy_tape(date, live) if live else {}
    if tickers:
        seed = [str(t).upper() for t in tickers if t]
    else:
        movers = schwab.movers_symbols() if live else []
        seed = []
        seen = set()
        for name in movers:
            if name in seen or _junk_symbol(name):
                continue
            seen.add(name)
            seed.append(name)
    quotes = schwab.quotes_many(seed, date) if live and seed else {}

    shortlist = []
    skips: List[dict] = []
    for name in seed:
        q = quotes.get(name)
        if not q or not layer0_quote_ok(q, gates, name):
            skips.append({"ticker": name, "reason": "quote"})
            continue
        shortlist.append(name)

    token = load_orats_token(token_file=orats_token_file)
    cores_raw = {}
    if token and shortlist:
        cores_raw = orats_mod.fetch_cores(shortlist, token, max_requests=max_orats_http)
    cores = {k: parse_core(v) for k, v in cores_raw.items()}

    quality = []
    for name in shortlist:
        core = cores.get(name) or {}
        if name not in cores or not orats_quality_ok(core, gates):
            skips.append({"ticker": name, "reason": "orats"})
            continue
        quality.append(name)

    from_d = add_days(date, int(gates.get("dte_min") or 25) - 2) or date
    to_d = add_days(date, int(gates.get("dte_max") or 45) + 2) or date

    new_rows: List[dict] = []
    chain_http = 0
    for name in quality:
        q = quotes.get(name) or {}
        last = spot_from_quote(q)
        if last is None:
            skips.append({"ticker": name, "reason": "no_last"})
            continue
        core = cores.get(name) or {}
        nasdaq = earnings_mod.nasdaq_next(name, date) if live else None
        earn = earnings_mod.resolve(name, date, core, nasdaq)
        bars = schwab.price_history_bars(name, date, use_cache=False) if live else []
        snap = tape.snapshot(bars, int((gates.get("swing") or {}).get("atr_n") or 14)) if bars else {}
        if tape.chase(snap, float((gates.get("swing") or {}).get("chase_atr") or 2.5)):
            skips.append({"ticker": name, "reason": "chase"})
            continue
        payload = schwab.option_chain(name, from_d, to_d) if live else None
        if payload is not None:
            chain_http += 1
        legs = flatten_chain(payload)
        puts = [x for x in legs if x.get("side") == "put"]
        calls = [x for x in legs if x.get("side") == "call"]
        ideas = catalog_for_name(
            puts, calls, last, date, earn, gates, snap, cheap_vol(core, gates)
        )
        if not ideas:
            skips.append({"ticker": name, "reason": "no_geometry"})
            continue
        for idea in ideas:
            if not _legal(idea, last, gates):
                skips.append({"ticker": name, "reason": "illegal_%s" % idea.get("structure")})
                continue
            row = _ticket(name, last, earn, idea, gates)
            if row:
                new_rows.append(row)

    ranked = sorted(
        new_rows,
        key=lambda r: (
            1 if r.get("action") == "CLICK" else 0,
            float(r.get("ev_proxy") or -1e18),
            int(r.get("conf") or 0),
        ),
        reverse=True,
    )
    trades = [r for r in ranked if r.get("action") == "CLICK"]
    skip = [r for r in ranked if r.get("action") == "SKIP"]
    watch = [r for r in ranked if r.get("action") == "WATCH"]

    dest = Path(out_dir) if out_dir else OUT_DIR
    dest = dest / date
    dest.mkdir(parents=True, exist_ok=True)
    artifacts = report.write_run(
        dest,
        date=date,
        tickets=trades,
        skip=skip,
        watch=watch,
        x_queue=sorted({r["ticker"] for r in trades}),
        gates=gates,
        skips=skips,
        manifest={
            "date": date,
            "live_schwab": live,
            "orats_http": orats_mod.process_http(),
            "schwab_http": schwab.http_count(),
            "movers": seed,
            "shortlist": shortlist,
            "quality": quality,
            "orats_token": bool(token),
            "chain_http": chain_http,
            "macro": macro,
        },
        macro=macro,
    )
    if not token:
        artifacts["orats_error"] = ORATS_TOKEN_MISSING
    return {
        "mode": "analyze" if tickers else "full",
        "date": date,
        "out_dir": str(dest),
        "n_trade": len(trades),
        "n_skip": len(skip),
        "n_watch": len(watch),
        "n_shortlist": len(shortlist),
        "orats_http": orats_mod.process_http(),
        "schwab_http": schwab.http_count(),
        "orats_token": bool(token),
        "files": artifacts,
    }
