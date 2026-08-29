"""Select funnel: Schwab shortlist → ORATS cores → quality → ORATS strikes → allocate."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from wheelo.config import (
    BOOK_PATH,
    CONTRACT_MULTIPLIER,
    CORE_FIELDS,
    ORATS_STRIKE_DTE,
    OUT_DIR,
    load_json_config,
    load_scan_universe,
)
from wheelo.daily import evaluate_book, load_book
from wheelo.dates import today_et
from wheelo.num import to_float
from wheelo.orats import (
    cap_tickers,
    fetch_cores,
    fetch_strikes,
    load_usage,
    parse_core,
    process_http,
    reset_process_http,
)
from wheelo.report import (
    BOARD_COLUMNS,
    day_dir,
    render_board,
    render_daily,
    render_report,
    write_csv,
    write_json,
    write_text,
)
from wheelo.confidence import ticket_confidence
from wheelo.scoring import (
    allocate_capital,
    apply_sentiment,
    assign_tier,
    compute_composite,
    earnings_days,
    score_premium,
    score_quality,
    stage0_reason,
)
from wheelo.schwab import option_chain, option_quotes, price_history_bars, quotes_many, use_live_schwab
from wheelo.xhot import load_hot
from wheelo.yfinance_overlay import fetch_yfinance


def _cfg(cfg: Optional[dict] = None) -> dict:
    return cfg if cfg is not None else load_json_config()


def _max_affordable_spot(capital: float, cfg: dict) -> float:
    a = cfg.get("allocation") or {}
    max_name = capital * float(a.get("max_single_name_pct") or 0.25)
    return max_name / float(CONTRACT_MULTIPLIER)


def _priority_names(cfg: dict) -> List[str]:
    raw = (cfg.get("universe") or {}).get("priority") or []
    out = []
    seen = set()
    for item in raw:
        name = str(item or "").upper()
        if name and name not in seen:
            seen.add(name)
            out.append(name)
    return out


def _price_shortlist(
    universe: Sequence[str],
    quotes: Dict[str, dict],
    cfg: dict,
    hot: Dict[str, dict],
    cap: int,
    capital: float,
) -> List[str]:
    u = cfg.get("universe") or {}
    lo = float(u.get("min_price") or 15)
    configured_hi = float(u.get("max_price") or 0)
    hi = configured_hi if configured_hi > 0 else None
    ranked = []
    for name in universe:
        q = quotes.get(name)
        last = to_float(q.get("last")) if q else None
        if last is None or last < lo:
            continue
        if hi is not None and last > hi:
            continue
        vol = to_float(q.get("volume")) or 0.0
        ranked.append((vol, name))
    ranked.sort(reverse=True)
    names = [n for _, n in ranked]
    allowed = set(names)
    for ticker in hot:
        if ticker not in allowed and ticker in universe:
            q = quotes.get(ticker)
            last = to_float(q.get("last")) if q else None
            if last is not None and last >= lo and (hi is None or last <= hi):
                names.insert(0, ticker)
                allowed.add(ticker)
    priority = [t for t in _priority_names(cfg) if t in allowed]
    rest = [n for n in names if n not in set(priority)]
    return cap_tickers(priority + rest, cap)


def _asof_shortlist(universe: Sequence[str], hot: Dict[str, dict], cap: int, cfg: Optional[dict] = None) -> List[str]:
    names = list(universe)
    allowed = set(names)
    for ticker in hot:
        if ticker not in allowed:
            names.insert(0, ticker)
            allowed.add(ticker)
    priority = [t for t in _priority_names(cfg or {}) if t in allowed]
    rest = [n for n in names if n not in set(priority)]
    return cap_tickers(priority + rest, cap)


def _contracts(strike: float, capital: float, cfg: dict) -> int:
    """One-lot ticket. User sizes cash; do not zero a name because it is expensive."""
    if strike is None or strike <= 0:
        return 0
    return 1


def _strike_rank(ticker: str, core: dict, quality_composite: float, asof: str, priority: Sequence[str]) -> float:
    score = float(quality_composite or 0)
    ivr = to_float(core.get("iv_pctile_1y")) or 0.0
    iv_hv = to_float(core.get("iv_hv")) or 0.0
    if ivr >= 50:
        score += 10
    if iv_hv >= 1.10:
        score += 10
    if ticker in set(priority):
        score += 15
    earn = earnings_days(core, asof)
    if earn is not None and earn <= 7:
        score -= 50
    chg1y = to_float(core.get("chg_1y"))
    if chg1y is not None and chg1y <= -40:
        score -= 20
    return score


def _candidate_row(ticker, core, quality, premium, sentiment, cfg, capital) -> dict:
    qw = float((cfg.get("scoring") or {}).get("quality_weight") or 0.7)
    pw = float((cfg.get("scoring") or {}).get("premium_weight") or 0.3)
    composite = compute_composite(quality.composite, premium.composite, sentiment.total, qw, pw)
    tier = assign_tier(composite, cfg)
    contracts = _contracts(premium.csp_strike, capital, cfg)
    return {
        "ticker": ticker,
        "spot": core.get("px"),
        "sector": core.get("sector") or "",
        "quality": {
            "composite": quality.composite,
            "yfinance_note": quality.yfinance_note,
            "disqualified": quality.disqualified,
        },
        "premium": {
            "csp_strike": premium.csp_strike,
            "csp_bid": premium.csp_bid,
            "csp_ask": premium.csp_ask,
            "csp_premium": premium.csp_premium,
            "csp_yield_ann": premium.csp_yield_ann,
            "cc_strike": premium.cc_strike,
            "cc_bid": premium.cc_bid,
            "cc_premium": premium.cc_premium,
            "iv_rank": premium.iv_rank,
            "spread_pct": premium.spread_pct,
            "dte": premium.dte,
            "expiry": premium.expiry,
            "composite": premium.composite,
        },
        "sentiment": sentiment.total,
        "notes": list(sentiment.notes),
        "x_status": sentiment.x_status,
        "composite": composite,
        "conf": None,
        "conf_label": "NO_TRADE",
        "conf_drivers": [],
        "tier": tier,
        "contracts": contracts,
        "capital_required": premium.csp_strike * CONTRACT_MULTIPLIER * contracts if contracts else 0.0,
        "allocated": False,
        "action": "SELL_CSP",
    }


def _reprice_allocated(cand: dict, chain: Optional[dict], cfg: dict) -> dict:
    if not chain or not isinstance(chain, dict):
        return cand
    snap = float((cfg.get("schwab") or {}).get("max_strike_snap_pct") or 0.03)
    target = to_float((cand.get("premium") or {}).get("csp_strike"))
    if target is None:
        return cand
    put_map = chain.get("putExpDateMap") or {}
    best = None
    best_dte = None
    want = int((cfg.get("management") or {}).get("dte_target") or 30)
    for key, strikes in put_map.items():
        parts = str(key).split(":")
        dte = int(parts[-1]) if len(parts) == 2 else 0
        if best_dte is None or abs(dte - want) < abs(best_dte - want):
            best_dte = dte
            best = (key, strikes)
    if not best:
        return cand
    _, strikes = best
    picked = None
    picked_k = None
    for strike_s, contracts in (strikes or {}).items():
        k = to_float(strike_s)
        if k is None:
            continue
        if abs(k - target) / target <= snap:
            if picked_k is None or abs(k - target) < abs(picked_k - target):
                picked_k = k
                picked = (contracts or [{}])[0]
    if not picked:
        return cand
    bid = to_float(picked.get("bid"))
    ask = to_float(picked.get("ask"))
    prem = cand.setdefault("premium", {})
    if bid is not None:
        prem["csp_bid"] = bid
        prem["csp_premium"] = bid
        prem["live_validated"] = True
    if ask is not None:
        prem["csp_ask"] = ask
    if picked_k is not None:
        prem["csp_strike"] = picked_k
    if best_dte:
        prem["dte"] = best_dte
    cand["live_validated"] = True
    return cand


def build_select(
    asof: str,
    token: str,
    capital: float,
    today: Optional[str] = None,
    live: bool = False,
    getter=None,
    max_requests: Optional[int] = None,
    cfg: Optional[dict] = None,
    universe: Optional[Sequence[str]] = None,
    quotes: Optional[Dict[str, dict]] = None,
    quotes_fn: Optional[Callable] = None,
    history_fn: Optional[Callable] = None,
    chain_fn: Optional[Callable] = None,
    yfinance_fn: Optional[Callable] = None,
    use_yfinance: bool = False,
    cores_by_ticker: Optional[Dict[str, dict]] = None,
    strikes_by_ticker: Optional[Dict[str, list]] = None,
) -> Dict[str, Any]:
    cfg = _cfg(cfg)
    today = today or today_et()
    names = list(universe if universe is not None else load_scan_universe(cfg))
    hot = load_hot(asof)
    orats_cfg = cfg.get("orats") or {}
    max_core = int(orats_cfg.get("max_core_tickers") or 40)
    max_strike = int(orats_cfg.get("max_strike_tickers") or 20)
    if max_requests is None:
        max_requests = int(orats_cfg.get("max_http_per_run") or 15)
    dte = str(orats_cfg.get("strike_dte") or ORATS_STRIKE_DTE)

    if quotes is None:
        if live:
            fn = quotes_fn or quotes_many
            quotes = fn(names, asof) or {}
        else:
            quotes = {}

    if live and quotes:
        short_a = _price_shortlist(names, quotes, cfg, hot, max_core, capital)
    else:
        short_a = _asof_shortlist(names, hot, max_core, cfg=cfg)

    rejections = []
    orats_http = 0
    cache_hits = 0
    orats_refresh = asof == today
    cores = dict(cores_by_ticker or {})
    if cores_by_ticker is None and short_a and token:
        pack = fetch_cores(
            asof,
            short_a,
            token,
            today,
            getter=getter,
            max_requests=max_requests,
            max_tickers=max_core,
            fields=CORE_FIELDS,
            refresh=orats_refresh,
        )
        cores = pack.get("rows") or {}
        orats_http += int(pack.get("http") or 0)
        if pack.get("cache"):
            cache_hits += 1
        if pack.get("error") == "orats_budget":
            return _empty(asof, capital, live, orats_http, short_a, [], [], "orats_budget", cache_hits)

    parsed = {t: parse_core(cores.get(t)) for t in short_a}
    short_b = []
    for ticker in short_a:
        core = parsed.get(ticker) or parse_core({})
        if quotes.get(ticker) and to_float(quotes[ticker].get("last")) is not None:
            core = dict(core)
            core["px"] = to_float(quotes[ticker].get("last"))
            parsed[ticker] = core
        why = stage0_reason(core, quotes.get(ticker), asof, cfg)
        if why:
            rejections.append({"ticker": ticker, "stage": "0", "reason": why})
            continue
        short_b.append(ticker)

    if history_fn is not None:
        hist_fn = history_fn
    elif live:
        hist_fn = lambda t, d: price_history_bars(t, d, use_cache=False)
    else:
        hist_fn = None
    yf_fn = yfinance_fn or (fetch_yfinance if use_yfinance else None)
    quality_map = {}
    for ticker in short_b:
        bars = hist_fn(ticker, asof) if hist_fn else []
        yf = yf_fn(ticker) if yf_fn else {"ok": False, "error": "yfinance_skipped"}
        qs = score_quality(parsed[ticker], quotes.get(ticker), bars, yf, cfg, asof)
        if qs.disqualified:
            rejections.append({"ticker": ticker, "stage": "1", "reason": qs.disqualify_reason})
            continue
        quality_map[ticker] = qs

    priority = _priority_names(cfg)
    pin = [t for t in priority if t in quality_map]
    rest = sorted(
        [t for t in quality_map if t not in set(pin)],
        key=lambda t: _strike_rank(t, parsed.get(t) or {}, quality_map[t].composite, asof, priority),
        reverse=True,
    )
    short_c = cap_tickers(pin + rest, max_strike)

    strikes = dict(strikes_by_ticker or {})
    if strikes_by_ticker is None and short_c and token:
        pack_s = fetch_strikes(
            asof,
            short_c,
            token,
            today,
            getter=getter,
            max_requests=max_requests,
            max_tickers=max_strike,
            dte=dte,
            refresh=orats_refresh,
        )
        strikes = pack_s.get("rows") or {}
        orats_http += int(pack_s.get("http") or 0)
        if pack_s.get("cache"):
            cache_hits += 1
        if pack_s.get("error") == "orats_budget" and not strikes:
            return _empty(asof, capital, live, orats_http, short_a, short_b, short_c, "orats_budget", cache_hits)

    candidates = []
    for ticker in short_c:
        prem = score_premium(strikes.get(ticker) or [], parsed[ticker], cfg)
        if prem.rejected:
            rejections.append({"ticker": ticker, "stage": "2", "reason": prem.reject_reason})
            continue
        sent = apply_sentiment(parsed[ticker], hot.get(ticker), asof, cfg)
        row = _candidate_row(ticker, parsed[ticker], quality_map[ticker], prem, sent, cfg, capital)
        pack = ticket_confidence(
            parsed[ticker],
            row["premium"],
            asof,
            x_status=row.get("x_status") or "",
            live_validated=False,
        )
        row["conf"] = pack.get("conf")
        row["conf_label"] = pack.get("label")
        row["conf_drivers"] = pack.get("drivers") or []
        row["conf_hard"] = pack.get("hard") or []
        row["credit_pct"] = pack.get("credit_pct")
        row["otm_pct"] = pack.get("otm_pct")
        if row["tier"] == "excluded":
            rejections.append({"ticker": ticker, "stage": "3", "reason": "below_watchlist"})
            continue
        if pack.get("label") == "NO_TRADE":
            rejections.append({"ticker": ticker, "stage": "3", "reason": ",".join(pack.get("hard") or ["low_conf"])})
        candidates.append(row)

    allocated = allocate_capital(candidates, capital, cfg)

    if live:
        ch_fn = chain_fn or option_chain
        for cand in allocated:
            if cand.get("conf_label") not in ("TRADE", "WATCH"):
                continue
            expiry = (cand.get("premium") or {}).get("expiry") or asof
            try:
                start = datetime.strptime(asof[:10], "%Y-%m-%d")
                end = datetime.strptime(str(expiry)[:10], "%Y-%m-%d") + timedelta(days=3)
            except ValueError:
                continue
            chain = ch_fn(cand["ticker"], start.date().isoformat(), end.date().isoformat())
            _reprice_allocated(cand, chain, cfg)
            core = parsed.get(cand["ticker"]) or {}
            pack = ticket_confidence(
                core,
                cand.get("premium") or {},
                asof,
                x_status=cand.get("x_status") or "",
                live_validated=True,
            )
            cand["conf"] = pack.get("conf")
            cand["conf_label"] = pack.get("label")
            cand["conf_drivers"] = pack.get("drivers") or []
            cand["conf_hard"] = pack.get("hard") or []
            cand["credit_pct"] = pack.get("credit_pct")
            cand["otm_pct"] = pack.get("otm_pct")
        allocated = allocate_capital(allocated, capital, cfg)

    usage = load_usage()
    planned = 0
    if short_a:
        planned += (len(short_a) + 9) // 10
    if short_c:
        planned += (len(short_c) + 9) // 10
    manifest = {
        "date": asof,
        "orats_http": orats_http,
        "orats_planned": planned,
        "orats_used": usage.get("used") or 0,
        "orats_left": usage.get("left") or 0,
        "shortlist_a": len(short_a),
        "shortlist_b": len(short_b),
        "shortlist_c": len(short_c),
        "schwab": bool(live),
        "cache_hits": cache_hits,
        "error": "",
        "process_http": process_http(),
    }
    x_queue = []
    for cand in allocated:
        if cand.get("x_status") == "DATA UNAVAILABLE":
            x_queue.append({"ticker": cand["ticker"], "need": "x_sentiment"})
    return {
        "asof": asof,
        "candidates": allocated,
        "rejections": rejections,
        "manifest": manifest,
        "x_queue": x_queue,
        "shortlist_a": short_a,
        "shortlist_b": short_b,
        "shortlist_c": short_c,
        "capital": capital,
        "cores": parsed,
    }


def _empty(asof, capital, live, http_n, a, b, c, err, cache_hits):
    usage = load_usage()
    return {
        "asof": asof,
        "candidates": [],
        "rejections": [],
        "manifest": {
            "date": asof,
            "orats_http": http_n,
            "orats_planned": 0,
            "orats_used": usage.get("used") or 0,
            "orats_left": usage.get("left") or 0,
            "shortlist_a": len(a),
            "shortlist_b": len(b),
            "shortlist_c": len(c),
            "schwab": bool(live),
            "cache_hits": cache_hits,
            "error": err,
            "process_http": process_http(),
        },
        "x_queue": [],
        "shortlist_a": a,
        "shortlist_b": b,
        "shortlist_c": c,
        "capital": capital,
        "cores": {},
    }


def write_select_artifacts(built: dict, out_dir: Path) -> Path:
    asof = built["asof"]
    day = day_dir(out_dir, asof)
    cands = built.get("candidates") or []
    man = built.get("manifest") or {}
    capital = float(built.get("capital") or 35000)
    write_text(day / "board.md", render_board(asof, cands, capital, man))
    write_text(day / "report.md", render_report(asof, cands, built.get("rejections") or [], man, capital))
    write_json(day / "candidates.json", cands)
    write_json(day / "manifest.json", man)
    write_json(day / "x_queue.json", built.get("x_queue") or [])
    write_csv(day / "rejections.csv", ["ticker", "stage", "reason"], built.get("rejections") or [])
    board_rows = []
    for cand in cands:
        prem = cand.get("premium") or {}
        board_rows.append(
            {
                "ticker": cand.get("ticker"),
                "tier": cand.get("tier"),
                "allocated": cand.get("allocated"),
                "spot": cand.get("spot"),
                "csp_strike": prem.get("csp_strike"),
                "csp_bid": prem.get("csp_bid"),
                "expiry": prem.get("expiry"),
                "dte": prem.get("dte"),
                "csp_yield_ann": prem.get("csp_yield_ann"),
                "quality": (cand.get("quality") or {}).get("composite"),
                "premium": prem.get("composite"),
                "composite": cand.get("composite"),
                "conf": cand.get("conf"),
                "conf_label": cand.get("conf_label"),
                "credit_pct": cand.get("credit_pct"),
                "otm_pct": cand.get("otm_pct"),
                "capital": cand.get("capital_required"),
                "contracts": cand.get("contracts"),
                "x_status": cand.get("x_status"),
            }
        )
    write_csv(day / "board.csv", BOARD_COLUMNS, board_rows)
    return day


def build_daily(
    asof: str,
    token: str,
    today: Optional[str] = None,
    live: bool = False,
    getter=None,
    max_requests: Optional[int] = None,
    cfg: Optional[dict] = None,
    book_path: Path = BOOK_PATH,
    marks: Optional[Dict[str, dict]] = None,
    spots: Optional[Dict[str, float]] = None,
    cores_by_ticker: Optional[Dict[str, dict]] = None,
) -> Dict[str, Any]:
    cfg = _cfg(cfg)
    today = today or today_et()
    book = load_book(book_path)
    positions = book.get("positions") or []
    tickers = [str(p.get("ticker") or "").upper() for p in positions if p.get("ticker")]
    local_spots = dict(spots or {})
    local_marks = dict(marks or {})
    cores = dict(cores_by_ticker or {})
    orats_http = 0
    orats_refresh = asof == today
    if live and positions and marks is None:
        symbols = [p.get("option_symbol") for p in positions if p.get("option_symbol")]
        if symbols:
            local_marks.update(option_quotes(symbols))
        if tickers:
            q = quotes_many(tickers, asof)
            for name, row in q.items():
                last = to_float(row.get("last"))
                if last is not None:
                    local_spots[name] = last
    if (not live or not local_marks) and tickers and token and cores_by_ticker is None:
        pack = fetch_cores(
            asof,
            tickers,
            token,
            today,
            getter=getter,
            max_requests=max_requests if max_requests is not None else 2,
            max_tickers=len(tickers),
            refresh=orats_refresh,
        )
        cores = {k: parse_core(v) for k, v in (pack.get("rows") or {}).items()}
        orats_http += int(pack.get("http") or 0)
        if not local_marks:
            pack_s = fetch_strikes(
                asof,
                tickers,
                token,
                today,
                getter=getter,
                max_requests=max_requests if max_requests is not None else 4,
                max_tickers=len(tickers),
                refresh=orats_refresh,
            )
            orats_http += int(pack_s.get("http") or 0)
            for name, rows in (pack_s.get("rows") or {}).items():
                for pos in positions:
                    if str(pos.get("ticker") or "").upper() != name:
                        continue
                    strike = to_float(pos.get("strike"))
                    for row in rows or []:
                        if to_float(row.get("strike")) == strike:
                            local_marks[name] = {
                                "bid": to_float(row.get("putBidPrice")) or to_float(row.get("callBidPrice")),
                                "ask": to_float(row.get("putAskPrice")) or to_float(row.get("callAskPrice")),
                            }
                            break
        for name, core in cores.items():
            if name not in local_spots and to_float(core.get("px")) is not None:
                local_spots[name] = to_float(core.get("px"))
    actions = evaluate_book(asof, cfg, local_marks, local_spots, cores, book=book)
    return {
        "asof": asof,
        "actions": actions,
        "positions": positions,
        "orats_http": orats_http,
        "schwab": bool(live),
    }


def write_daily_artifacts(built: dict, out_dir: Path) -> Path:
    asof = built["asof"]
    day = day_dir(out_dir, asof)
    write_text(day / "daily.md", render_daily(asof, built.get("actions") or [], built.get("positions") or []))
    write_json(day / "daily.json", {"actions": built.get("actions"), "positions": built.get("positions")})
    return day


def run_pipeline(
    cmd: str,
    asof: str,
    token: str,
    capital: float,
    out_dir: Path = OUT_DIR,
    live_schwab: bool = False,
    no_schwab: bool = False,
    getter=None,
    max_orats_requests: Optional[int] = None,
    ticker: Optional[str] = None,
    use_yfinance: bool = False,
    universe: Optional[Sequence[str]] = None,
    quotes: Optional[Dict[str, dict]] = None,
    cores_by_ticker: Optional[Dict[str, dict]] = None,
    strikes_by_ticker: Optional[Dict[str, list]] = None,
    book_path: Path = BOOK_PATH,
    today: Optional[str] = None,
    quotes_fn=None,
    history_fn=None,
    chain_fn=None,
    yfinance_fn=None,
    marks=None,
    spots=None,
) -> Dict[str, Any]:
    reset_process_http()
    today = today or today_et()
    live = use_live_schwab(asof, live_flag=live_schwab, no_schwab=no_schwab, today=today)
    cfg = load_json_config()
    uni = list(universe) if universe is not None else load_scan_universe(cfg)
    if cmd == "analyze" and ticker:
        uni = [str(ticker).upper()]
    info = {"mode": cmd, "asof": asof, "out_dir": "", "error": ""}
    if cmd in ("select", "full", "analyze"):
        built = build_select(
            asof,
            token,
            capital,
            today=today,
            live=live,
            getter=getter,
            max_requests=max_orats_requests,
            cfg=cfg,
            universe=uni,
            quotes=quotes,
            quotes_fn=quotes_fn,
            history_fn=history_fn,
            chain_fn=chain_fn,
            yfinance_fn=yfinance_fn,
            use_yfinance=use_yfinance,
            cores_by_ticker=cores_by_ticker,
            strikes_by_ticker=strikes_by_ticker,
        )
        day = write_select_artifacts(built, out_dir)
        info.update(built.get("manifest") or {})
        info["out_dir"] = str(day)
        info["candidates"] = built.get("candidates")
        info["trade_count"] = sum(1 for c in (built.get("candidates") or []) if c.get("allocated"))
        info["error"] = (built.get("manifest") or {}).get("error") or ""
        if cmd != "full":
            return info
    if cmd in ("daily", "review", "full"):
        daily = build_daily(
            asof,
            token,
            today=today,
            live=live,
            getter=getter,
            max_requests=max_orats_requests,
            cfg=cfg,
            book_path=book_path,
            marks=marks,
            spots=spots,
            cores_by_ticker=cores_by_ticker,
        )
        day = write_daily_artifacts(daily, out_dir)
        info["out_dir"] = str(day)
        info["actions"] = daily.get("actions")
        info["daily_orats_http"] = daily.get("orats_http") or 0
        info["orats_http"] = int(info.get("orats_http") or 0) + int(daily.get("orats_http") or 0)
        if cmd == "review":
            info["positions"] = daily.get("positions") or []
    return info
