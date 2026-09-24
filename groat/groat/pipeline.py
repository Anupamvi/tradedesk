"""Groat scan: regime → universe → underlying thesis → structure. Empty board is valid."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence

from groat.calendar import earnings_info, events_between
from groat.config import (
    EARNINGS_HOLD_DAYS,
    INDEX_TICKERS,
    MACRO_TICKERS,
    MAX_FINAL,
    SLEEVE,
    STRIKE_DTE,
    TRADE_SCORE_MIN,
    VIX_SYMBOL,
    WATCH_SCORE_MIN,
    load_universe,
    ticker_etf,
    ticker_group,
)
from groat.dates import today_et
from groat.gates import (
    apply_already_held_park,
    apply_analog_0win_park,
    apply_below_ema_park,
    apply_group_trade_cap,
    apply_same_group_book_park,
    stamp_fill_guard,
    trade_park_reason,
)
from groat.earnings import web_resolve
from groat.num import to_float
from groat.evidence import attach_evidence
from groat.picks import desk_picks
from groat.orats import fetch_cores, fetch_hist_earnings, fetch_strikes, load_usage, parse_core
from groat.prices import ensure_bars
from groat.regime import classify as classify_regime
from groat.rotation import group_status_map, name_group_row, rank_groups
from groat.confidence import options_confidence
from groat.pd import attach_trade_pd, sort_by_pd
from groat.setups import classify_setups
from groat.book import book_index, open_group_sets, same_ticket, schwab_held_index
from groat.chainfill import overlay_strikes
from groat.structure import choose
from groat.technicals import snapshot
from groat.thesis import build_thesis
from groat.xhot import classify_xhot, load_hot
from groat.xintel import load_xintel


def _end_hold(asof: str, days: int = EARNINGS_HOLD_DAYS) -> str:
    try:
        return (datetime.strptime(asof[:10], "%Y-%m-%d") + timedelta(days=days)).date().isoformat()
    except ValueError:
        return asof


def score_row(row: dict, regime: str, group_status: str) -> float:
    """Score from replay expectancy, not letter order.

    D/E can clear TRADE_SCORE_MIN=52 on a clean tape. Mature A cannot
    (replay A ≈ 0R). FIRE/H is a confirm, not a reason to TRADE.
    """
    s = 0.0
    primary = row.get("primary") or ""
    s += {"E": 18, "D": 18, "A": 10, "C": 10, "F": 8, "B": 8, "H": 6, "G": 6}.get(primary, 0)
    fire = row.get("fire") or {}
    if fire.get("kind") and not fire.get("chase") and primary != "H":
        s += 4
    rs20 = row.get("rs_20")
    if rs20 is not None:
        if rs20 > 0.05:
            s += 10
        elif rs20 > 0:
            s += 5
        elif rs20 < -0.05:
            s -= 8
    # D's edge is beating SPY, not sitting in a hot group.
    # Neutral used to score 49 and only energy E could TRADE — that is the original bug.
    if primary == "D":
        if group_status in ("accelerating", "emerging"):
            s += 6
        elif group_status == "deteriorating":
            s -= 2
        else:
            s += 4
    else:
        if group_status in ("accelerating", "emerging"):
            s += 8
        elif group_status == "mature":
            s += 2
        elif group_status == "deteriorating":
            s -= 10
    if row.get("above_sma200"):
        s += 4
    if row.get("above_sma50"):
        s += 3
    av = row.get("avwap_swing_low") or row.get("avwap_year")
    if av is not None and row.get("close") is not None and row["close"] > av:
        s += 4
    ext = row.get("extension_atr")
    if ext is not None and ext > 2.5:
        s -= 12
    picked = row.get("picked") or {}
    rr = picked.get("rr") if isinstance(picked, dict) else None
    inst = str((picked or {}).get("instrument") or row.get("choice") or "")
    if rr is not None:
        defined = "spread" in inst
        if defined:
            # Defined-risk debit/credit is the click ticket. Do not haircut vs stock 2:1.
            if rr >= 1.2:
                s += 10
            elif rr < 1:
                s -= 10
        elif inst.startswith("long_") or row.get("choice") == "OPTIONS":
            if rr >= 1.5:
                s += 10
            elif rr >= 1.2:
                s += 6
            elif rr < 1:
                s -= 10
        elif rr >= 2:
            s += 10
        elif rr >= 1.5:
            s += 6
        elif rr < 1:
            s -= 10
    if regime == "risk_off" and row.get("direction") == "bullish" and primary in ("A", "B"):
        s -= 12
    if regime == "high_vol_liquidation":
        s -= 20
    if regime == "strong_risk_on" and row.get("direction") == "bearish":
        s -= 8
    if regime == "range_chop" and primary in ("B",):
        s -= 8
    if row.get("choice") == "NO TRADE":
        s -= 15
    if row.get("stale"):
        s -= 20
    return s


def _maybe_fallback_stock(row: dict, chosen: dict, regime: str, group_status: str) -> bool:
    """STOCK fallback only when the options ticket is a naked long, not a vertical.

    A priced debit/credit spread is the working ticket. Do not swap it for shares
    because the spread RR is 1.2–1.5 vs stock 2:1.
    """
    stock = chosen.get("stock") if isinstance(chosen, dict) else None
    if row.get("choice") != "OPTIONS" or not isinstance(stock, dict) or not stock.get("ok"):
        return False
    picked = row.get("picked") if isinstance(row.get("picked"), dict) else {}
    inst = str(picked.get("instrument") or "")
    if "spread" in inst:
        return False
    if (to_float(row.get("score")) or 0) >= TRADE_SCORE_MIN:
        return False
    probe = dict(row)
    probe["choice"] = "STOCK"
    probe["picked"] = stock
    probe_score = score_row(probe, regime, group_status)
    if probe_score < TRADE_SCORE_MIN:
        return False
    row["choice"] = "STOCK"
    row["picked"] = stock
    row["target_debit"] = None
    row["target_credit"] = None
    row["premium_side"] = stock.get("premium_side")
    row["opt_conf"] = None
    row["opt_conf_label"] = "n/a"
    row["opt_conf_note"] = "stock or no trade"
    row["opt_conf_drivers"] = []
    row["naive_pop"] = None
    row["naive_pop_note"] = None
    why = list(row.get("choice_why") or [])
    why.append("options ticket scores below TRADE; stock still clears")
    row["choice_why"] = why
    stamp_fill_guard(row)
    row["score"] = probe_score
    return True


def load_bars(
    tickers: Sequence[str],
    token: str,
    asof: str,
    live: bool,
    getter=None,
    max_requests=None,
    bars_by_ticker: Optional[Dict[str, list]] = None,
) -> Dict[str, dict]:
    out = {}
    if bars_by_ticker is not None:
        for name in tickers:
            bars = bars_by_ticker.get(name) or []
            out[name] = {"bars": bars, "tape": "injected", "http": 0, "error": "" if bars else "missing_bars"}
        return out
    for name in tickers:
        out[name] = ensure_bars(
            name,
            token,
            getter=getter,
            max_requests=max_requests,
            asof=asof,
            live=live,
            refresh=True,
        )
    return out


def _prelim_key(item):
    name, setup, snap = item
    hits = len(setup.get("setups") or [])
    rs = snap.get("rs_20")
    rs_n = rs if rs is not None else -9
    return (1 if setup.get("primary") else 0, hits, rs_n)


# Setups that can still become TRADE after replay parks. Price these before X-HOT fillers.
TRADE_CHAIN_SETUPS = ("A", "D", "E", "F")


def select_option_names(prelim, hot_map=None, cap: int = 40) -> list:
    """FIRE, then TRADE-eligible theses, then X-HOT, then remaining ranked names."""
    ordered = []
    seen = set()

    def add(name):
        up = str(name or "").upper()
        if not up or up in seen:
            return
        seen.add(up)
        ordered.append(up)

    for name, setup, snap in prelim or []:
        if (setup.get("fire") or {}).get("kind"):
            add(name)
    ranked = sorted(list(prelim or []), key=_prelim_key, reverse=True)
    for name, setup, snap in ranked:
        if setup.get("primary") in TRADE_CHAIN_SETUPS and setup.get("direction") in ("bullish", "bearish"):
            add(name)
    for ticker in hot_map or {}:
        add(ticker)
    for name, setup, snap in ranked:
        if setup.get("primary") and setup.get("direction") in ("bullish", "bearish"):
            add(name)
    return ordered[:cap]


def build_candidate(
    asof: str,
    ticker: str,
    snap: dict,
    core_row: Optional[dict],
    group_row: dict,
    regime: dict,
    strikes: Optional[list],
    bars: list,
    earn: Optional[dict] = None,
    hist_rows: Optional[list] = None,
    chain_status: Optional[str] = None,
) -> dict:
    vol = parse_core(core_row)
    earn = earn or earnings_info(ticker, core_row, asof, hist_rows=hist_rows)
    setup = classify_setups(snap, group_row=group_row, earnings=earn, bars=bars)
    direction = setup.get("direction") or "neutral"
    choose_snap = snap
    if snap.get("session_incomplete") and snap.get("live_last") is not None:
        choose_snap = dict(snap)
        choose_snap["close"] = snap.get("live_last")
    chosen = choose(choose_snap, direction, vol, strikes or [], earn, setup=setup, chain_status=chain_status)
    macros = events_between(asof, _end_hold(asof))
    xinfo = load_xintel(asof, ticker)
    picked = chosen.get("picked") or {}
    conf = (
        options_confidence(picked, vol, earn, snap, setup=setup, x_tag=xinfo.get("tag"))
        if chosen.get("choice") == "OPTIONS"
        else {"conf": None, "label": "n/a", "note": "stock or no trade", "drivers": []}
    )
    row = {
        "asof_date": asof,
        "ticker": ticker,
        "group": ticker_group(ticker),
        "etf": ticker_etf(ticker),
        "group_status": group_row.get("status") or "DATA UNAVAILABLE",
        "sleeve": SLEEVE,
        "close": snap.get("live_last")
        if snap.get("session_incomplete") and snap.get("live_last") is not None
        else snap.get("close"),
        "structure_close": snap.get("structure_close") or snap.get("close"),
        "session_incomplete": bool(snap.get("session_incomplete")),
        "ema20": snap.get("ema20"),
        "sma50": snap.get("sma50"),
        "sma200": snap.get("sma200"),
        "atr14": snap.get("atr14"),
        "trend": snap.get("trend"),
        "rs_5": snap.get("rs_5"),
        "rs_20": snap.get("rs_20"),
        "rs_60": snap.get("rs_60"),
        "rvol": snap.get("rvol"),
        "rsi14": snap.get("rsi14"),
        "ret_1": snap.get("live_ret_1")
        if snap.get("session_incomplete") and snap.get("live_ret_1") is not None
        else snap.get("ret_1"),
        "ret_2": snap.get("ret_2"),
        "extension_atr": snap.get("extension_atr"),
        "avwap_year": snap.get("avwap_year"),
        "avwap_swing_low": snap.get("avwap_swing_low"),
        "avwap_swing_high": snap.get("avwap_swing_high"),
        "avwap_earnings": setup.get("avwap_earnings"),
        "above_ema20": snap.get("above_ema20"),
        "above_sma50": snap.get("above_sma50"),
        "above_sma200": snap.get("above_sma200"),
        "stale": snap.get("stale"),
        "tape_ok": snap.get("ok"),
        "primary": setup.get("primary"),
        "primary_name": setup.get("primary_name"),
        "setups": setup.get("setups") or [],
        "setup_notes": setup.get("notes") or [],
        "direction": direction,
        "choice": chosen.get("choice"),
        "picked": picked,
        "stock": chosen.get("stock"),
        "options": chosen.get("options"),
        "reviews": chosen.get("reviews") or [],
        "target_debit": picked.get("target_debit") if isinstance(picked, dict) else None,
        "target_credit": picked.get("target_credit") if isinstance(picked, dict) else None,
        "premium_side": picked.get("premium_side") if isinstance(picked, dict) else None,
        "opt_conf": conf.get("conf"),
        "opt_conf_label": conf.get("label"),
        "opt_conf_note": conf.get("note"),
        "opt_conf_drivers": conf.get("drivers") or [],
        "naive_pop": picked.get("naive_pop") if isinstance(picked, dict) else None,
        "naive_pop_note": picked.get("naive_pop_note") if isinstance(picked, dict) else None,
        "choice_why": list(chosen.get("why") or []) if not isinstance(chosen.get("why"), str) else [chosen.get("why")],
        "fire": setup.get("fire") or {},
        "lane": setup.get("lane") or "SWING",
        "xhot": {},
        "iv30": vol.get("iv30"),
        "hv20": vol.get("hv20"),
        "vrp": vol.get("vrp"),
        "iv_pctile": vol.get("iv_pctile_1y"),
        "forecast_20d": vol.get("forecast_20d"),
        "iv_vs_forecast": vol.get("iv_vs_forecast"),
        "slope": vol.get("slope"),
        "orats_raw": vol.get("raw"),
        "earnings": earn,
        "macros": macros,
        "regime": regime.get("regime"),
        "x": xinfo.get("tag") or "DATA UNAVAILABLE",
        "x_notes": xinfo.get("notes") or "",
        "news": "DATA UNAVAILABLE",
        "filings": "DATA UNAVAILABLE",
    }
    stamp_fill_guard(row)
    row["thesis"] = build_thesis(row)
    regime_label = str(regime.get("regime") or "")
    group_status = str(group_row.get("status") or "")
    row["score"] = score_row(row, regime_label, group_status)
    if _maybe_fallback_stock(row, chosen, regime_label, group_status):
        row["thesis"] = build_thesis(row)
    if row["stale"] or not snap.get("ok"):
        action = "IGNORE"
        reasons = [snap.get("reason") or "missing_bars"]
    elif row["choice"] == "NO TRADE" or row["score"] < WATCH_SCORE_MIN:
        action = "IGNORE"
        reasons = []
        if row["score"] < WATCH_SCORE_MIN:
            reasons.append("score_below_watch")
        if row["choice"] == "NO TRADE":
            reasons.extend(list(row["choice_why"] or []))
        elif row.get("choice_why"):
            reasons.extend(list(row["choice_why"]))
        if not reasons:
            reasons = ["score_below_watch"]
        if not row.get("primary"):
            reasons.append("no_setup")
    elif (
        row["choice"] == "OPTIONS"
        and "spread" in str((row.get("picked") or {}).get("instrument") or "")
        and row["score"] >= TRADE_SCORE_MIN
    ):
        park = trade_park_reason(row.get("primary"), snap, setup)
        if park:
            action = "WATCH"
            reasons = [park]
        else:
            action = "TRADE"
            reasons = []
    else:
        action = "WATCH"
        reasons = []
        if row["choice"] == "STOCK":
            reasons.append("spread_required")
        elif row["score"] < TRADE_SCORE_MIN:
            reasons.append("below_trade_score")
    row["action"] = action
    row["reasons"] = reasons
    apply_below_ema_park(row)
    return row


def _rank_actionable(candidates: Sequence[dict]) -> tuple:
    ranked = sorted(
        [c for c in candidates if c.get("action") in ("TRADE", "WATCH")],
        key=lambda r: (r.get("action") == "TRADE", r.get("score") or 0, r.get("rs_20") or -9),
        reverse=True,
    )
    trades = [c for c in ranked if c.get("action") == "TRADE"][:MAX_FINAL]
    watch = [c for c in ranked if c.get("action") == "WATCH"][:MAX_FINAL]
    board = (trades + watch)[:MAX_FINAL]
    return ranked, trades, watch, board


def _attach_holdings(row: dict, held: dict, schwab_held: dict, open_groups=None, open_tickers=None) -> dict:
    name = str(row.get("ticker") or "").upper()
    book_pos = held.get(name) or {}
    schwab_pos = schwab_held.get(name) or {}
    row["in_book"] = bool(book_pos.get("in_book"))
    row["held_schwab"] = bool(schwab_pos.get("held_schwab"))
    row["held"] = bool(row["in_book"] or row["held_schwab"])
    row["schwab_legs"] = list(schwab_pos.get("legs") or [])
    notes = []
    picked = row.get("picked") if isinstance(row.get("picked"), dict) else {}
    row["same_ticket"] = bool(row["in_book"] and same_ticket(book_pos, picked))
    if row["in_book"]:
        open_line = str(book_pos.get("structure") or "")
        if book_pos.get("entry") is not None:
            open_line += " @ %s" % book_pos.get("entry")
        if book_pos.get("expiry"):
            open_line += " exp %s" % book_pos.get("expiry")
        if row["same_ticket"]:
            notes.append("IN BOOK — this is the open ticket (%s). Do not add." % open_line.strip(" —"))
        else:
            notes.append(
                "IN BOOK open: %s. Board structure is different — visibility only, not a roll/add."
                % (open_line or "see book.json")
            )
    if row["held_schwab"]:
        legs = schwab_pos.get("legs") or []
        bits = []
        for leg in legs[:4]:
            if leg.get("right"):
                bits.append(
                    "%s %s %s"
                    % (str(leg.get("right") or "").upper(), leg.get("expiry") or "", leg.get("strike") or "")
                )
            elif leg.get("symbol"):
                bits.append(str(leg.get("symbol")))
        notes.append("Schwab holds: %s" % (", ".join(bits) if bits else "this underlying"))
    row["held_note"] = "; ".join(notes)
    if row["held_note"]:
        thesis = dict(row.get("thesis") or {})
        paras = list(thesis.get("paragraphs") or [])
        extra = row["held_note"] + " Shown for visibility — do not add a second lot unless you have a scale plan."
        if extra not in paras:
            paras.append(extra)
        thesis["paragraphs"] = paras
        row["thesis"] = thesis
    apply_already_held_park(row)
    apply_same_group_book_park(row, open_groups, open_tickers)
    return row


def _board_from_candidates(candidates: Sequence[dict], live: bool = False, evidence_line: str = "") -> dict:
    ranked, trades, watch, board = _rank_actionable(candidates)
    for row in trades:
        attach_trade_pd(row, eod=not live)
    trades = sort_by_pd(
        trades,
        tie=lambda r: (-(r.get("score") or 0), -(to_float(r.get("rs_20")) or -9)),
    )
    group_parked = apply_group_trade_cap(trades)
    extra = str(evidence_line or "").strip()
    if group_parked:
        ranked, trades, watch, board = _rank_actionable(candidates)
        for row in trades:
            attach_trade_pd(row, eod=not live)
        trades = sort_by_pd(
            trades,
            tie=lambda r: (-(r.get("score") or 0), -(to_float(r.get("rs_20")) or -9)),
        )
        cap_line = "Group cap parked to WATCH: " + ", ".join(group_parked) + "."
        extra = (" ".join(x for x in (extra, cap_line) if x)).strip()
    picks = desk_picks(trades)
    if isinstance(picks, dict):
        picks["trade_names"] = [r.get("ticker") for r in trades if r.get("ticker")]
        picks["evidence_line"] = extra
    board = (list(trades) + list(watch))[:MAX_FINAL]
    return {
        "ranked": ranked,
        "trades": trades,
        "watch": watch,
        "board": board,
        "picks": picks,
        "group_parked": group_parked,
    }


def apply_xintel_row(asof: str, row: dict) -> dict:
    ticker = str(row.get("ticker") or "").upper()
    if not ticker:
        return row
    xinfo = load_xintel(asof, ticker)
    tag = str(xinfo.get("tag") or "DATA UNAVAILABLE")
    if tag in ("Quiet", "Informed", "Crowded"):
        row["x"] = tag
        row["x_notes"] = xinfo.get("notes") or row.get("x_notes") or ""
    elif row.get("x") in (None, "", "DATA UNAVAILABLE"):
        row["x"] = "DATA UNAVAILABLE"
    if row.get("choice") == "OPTIONS" and isinstance(row.get("picked"), dict):
        vol = {"vrp": row.get("vrp"), "iv30": row.get("iv30"), "hv20": row.get("hv20")}
        conf = options_confidence(
            row.get("picked"),
            vol,
            row.get("earnings") or {},
            row,
            setup={"primary": row.get("primary"), "direction": row.get("direction")},
            x_tag=row.get("x"),
        )
        row["opt_conf"] = conf.get("conf")
        row["opt_conf_label"] = conf.get("label")
        row["opt_conf_note"] = conf.get("note")
        row["opt_conf_drivers"] = conf.get("drivers") or []
    return row


def overlay_xintel(asof: str, candidates: Sequence[dict], live: bool = False, hot_map=None) -> Dict[str, Any]:
    """Re-tag TRADE/WATCH from var/xintel and re-render. No ORATS/Schwab refresh."""
    rows = list(candidates or [])
    hot_map = hot_map if hot_map is not None else load_hot(asof)
    for row in rows:
        apply_xintel_row(asof, row)
    analog = [
        str(r.get("ticker") or "")
        for r in rows
        if r.get("action") == "WATCH" and any("analog" in str(x) for x in (r.get("reasons") or []))
    ]
    evidence_line = ("Analog veto parked to WATCH: " + ", ".join(x for x in analog if x) + ".") if analog else ""
    fire_rows = [
        c
        for c in rows
        if (c.get("fire") or {}).get("kind")
        and not (c.get("fire") or {}).get("chase")
        and c.get("choice") in ("STOCK", "OPTIONS")
    ]
    fire_rows.sort(
        key=lambda r: abs(to_float(r.get("ret_1")) or 0) * (to_float(r.get("rvol")) or 1.0),
        reverse=True,
    )
    fire_rows = fire_rows[:5]
    xhot_rows = []
    for row in rows:
        hot = hot_map.get(str(row.get("ticker") or "").upper())
        if not hot:
            continue
        info = classify_xhot(hot, row)
        row["xhot"] = info
        if info.get("tag") and (row.get("x") in (None, "", "DATA UNAVAILABLE")):
            row["x"] = info.get("tag")
            row["x_notes"] = info.get("narrative") or row.get("x_notes")
        xhot_rows.append(row)
    move_rank = {"dipped": 3, "will_rise": 2, "will_dip": 2, "noise": 0}
    xhot_rows.sort(
        key=lambda r: (
            1 if (r.get("xhot") or {}).get("playable") else 0,
            move_rank.get((r.get("xhot") or {}).get("move") or "", 0),
            abs(to_float(r.get("ret_1")) or 0),
        ),
        reverse=True,
    )
    xhot_rows = xhot_rows[:10]
    finished = _board_from_candidates(rows, live=live, evidence_line=evidence_line)
    return {
        "asof": asof,
        "candidates": rows,
        "trades": finished["trades"],
        "watch": finished["watch"],
        "board": finished["board"],
        "picks": finished["picks"],
        "fire": fire_rows,
        "fire_count": len(fire_rows),
        "xhot": xhot_rows,
        "xhot_count": len(xhot_rows),
        "trade_count": len(finished["trades"]),
        "watch_count": len(finished["watch"]),
        "orats_http": 0,
        "overlay": True,
    }


def _wanted_tickers(universe: Sequence[str]) -> List[str]:
    extra = list(INDEX_TICKERS) + list(MACRO_TICKERS)
    seen = set()
    out = []
    for name in list(universe) + extra:
        up = str(name).upper()
        if up in seen:
            continue
        seen.add(up)
        out.append(up)
    return out


def build_full(
    asof: str,
    token: str,
    today: Optional[str] = None,
    live: bool = False,
    getter=None,
    max_requests: Optional[int] = None,
    universe: Optional[Sequence[str]] = None,
    bars_by_ticker: Optional[Dict[str, list]] = None,
    cores_by_ticker: Optional[Dict[str, dict]] = None,
    strikes_by_ticker: Optional[Dict[str, list]] = None,
    vix_bars: Optional[list] = None,
    use_web: Optional[bool] = None,
) -> Dict[str, Any]:
    today = today or today_et()
    names = list(universe or load_universe())
    hot_map = load_hot(asof)
    for ticker in hot_map:
        if ticker not in names:
            names.append(ticker)
    wanted = _wanted_tickers(names)
    tapes = load_bars(
        wanted,
        token,
        asof,
        live,
        getter=getter,
        max_requests=max_requests,
        bars_by_ticker=bars_by_ticker,
    )
    bars_map = {k: (v.get("bars") or []) for k, v in tapes.items()}
    if vix_bars is not None:
        bars_map[VIX_SYMBOL] = vix_bars
    elif bars_by_ticker is None:
        vix_pack = {"bars": []}
        for vix_sym in (VIX_SYMBOL, "VIX", "$VIX"):
            vix_pack = ensure_bars(vix_sym, token, getter=getter, max_requests=max_requests, asof=asof, live=live)
            if vix_pack.get("bars"):
                break
        tapes[VIX_SYMBOL] = vix_pack
        bars_map[VIX_SYMBOL] = vix_pack.get("bars") or []

    spy_bars = bars_map.get("SPY") or []
    snaps = {}
    for name in wanted:
        snaps[name] = snapshot(bars_map.get(name) or [], asof, bench_bars=spy_bars)
    universe_snaps = [snaps[n] for n in names if n in snaps]
    regime = classify_regime(asof, bars_map, vix_bars=bars_map.get(VIX_SYMBOL), universe_snaps=universe_snaps)
    groups = rank_groups(asof, bars_map, spy_bars)
    gmap = group_status_map(groups)

    orats_error = ""
    cores = dict(cores_by_ticker or {})
    orats_http = 0
    if cores_by_ticker is None:
        pack = fetch_cores(
            asof, names, token, today, getter=getter, max_requests=max_requests, refresh=True
        )
        cores = pack.get("rows") or {}
        orats_error = pack.get("error") or ""
        orats_http = int(pack.get("http") or 0)
        if not pack.get("ok") and not cores:
            orats_error = orats_error or "DATA UNAVAILABLE"

    prelim = []
    for name in names:
        snap = snaps.get(name) or {}
        setup = classify_setups(
            snap,
            group_row=name_group_row(name, groups),
            earnings=earnings_info(name, cores.get(name), asof),
            bars=bars_map.get(name) or [],
        )
        prelim.append((name, setup, snap))

    prelim.sort(key=_prelim_key, reverse=True)
    option_names = select_option_names(prelim, hot_map, cap=40)

    strikes = dict(strikes_by_ticker or {})
    chain_errors: List[dict] = []
    if strikes_by_ticker is None and option_names and token:
        pack_s = fetch_strikes(
            asof,
            option_names,
            token,
            today,
            getter=getter,
            max_requests=max_requests,
            dte=STRIKE_DTE,
            refresh=True,
        )
        strikes = pack_s.get("rows") or {}
        orats_http += int(pack_s.get("http") or 0)
        if pack_s.get("error") and not orats_error:
            orats_error = pack_s.get("error")
        if live or bars_by_ticker is None:
            strikes = overlay_strikes(asof, option_names, strikes, errors=chain_errors)

    if use_web is None:
        use_web = bars_by_ticker is None
    hist_e = {}
    web_e = {}
    if option_names and token and bars_by_ticker is None:
        for name in option_names:
            pack_e = fetch_hist_earnings(name, token, getter=getter, max_requests=max_requests)
            hist_e[name] = pack_e.get("rows") or []
            orats_http += int(pack_e.get("http") or 0)
    if use_web:
        for name in option_names:
            web_e[name] = web_resolve(name, asof, use_web=True)

    held = book_index()
    open_groups, open_tickers = open_group_sets()
    schwab_held = {}
    schwab_pos_error = ""
    if live or bars_by_ticker is None:
        try:
            from groat.schwab import positions_all

            schwab_held = schwab_held_index(positions_all())
        except Exception as exc:
            schwab_held = {}
            schwab_pos_error = str(exc)[:160]

    earn_map = {}
    for name in names:
        earn_map[name] = earnings_info(
            name,
            cores.get(name),
            asof,
            hist_rows=hist_e.get(name),
            use_web=False,
            web_payload=web_e.get(name),
        )

    candidates = []
    rejections = []
    for name in names:
        snap = snaps.get(name) or {"ok": False, "reason": "missing_bars", "stale": True}
        if name in option_names:
            chain_status = "ok" if (strikes.get(name) or []) else "empty"
        else:
            chain_status = "not_requested"
        row = build_candidate(
            asof,
            name,
            snap,
            cores.get(name),
            name_group_row(name, groups),
            regime,
            strikes.get(name) or [],
            bars_map.get(name) or [],
            earn=earn_map.get(name),
            hist_rows=hist_e.get(name),
            chain_status=chain_status,
        )
        row["chain_status"] = chain_status
        _attach_holdings(row, held, schwab_held, open_groups, open_tickers)
        candidates.append(row)
        if row["action"] == "IGNORE":
            rejections.append(
                {
                    "asof_date": asof,
                    "ticker": name,
                    "reasons": " ".join(row.get("reasons") or []) or "ignored",
                    "stage": "screen",
                }
            )

    need_chain = [
        str(c.get("ticker") or "").upper()
        for c in candidates
        if c.get("action") == "TRADE"
        and c.get("chain_status") == "not_requested"
        and c.get("direction") in ("bullish", "bearish")
        and c.get("ticker")
    ]
    if need_chain and strikes_by_ticker is None and token and bars_by_ticker is None:
        pack_s2 = fetch_strikes(
            asof,
            need_chain,
            token,
            today,
            getter=getter,
            max_requests=max_requests,
            dte=STRIKE_DTE,
            refresh=False,
        )
        extra_strikes = pack_s2.get("rows") or {}
        orats_http += int(pack_s2.get("http") or 0)
        if live or bars_by_ticker is None:
            extra_strikes = overlay_strikes(asof, need_chain, extra_strikes, errors=chain_errors)
        strikes.update(extra_strikes)
        for name in need_chain:
            if name not in option_names:
                option_names.append(name)
            chain_status = "ok" if (strikes.get(name) or []) else "empty"
            rebuilt = build_candidate(
                asof,
                name,
                snaps.get(name) or {"ok": False, "reason": "missing_bars", "stale": True},
                cores.get(name),
                name_group_row(name, groups),
                regime,
                strikes.get(name) or [],
                bars_map.get(name) or [],
                earn=earn_map.get(name),
                hist_rows=hist_e.get(name),
                chain_status=chain_status,
            )
            rebuilt["chain_status"] = chain_status
            _attach_holdings(rebuilt, held, schwab_held, open_groups, open_tickers)
            for i, old in enumerate(candidates):
                if str(old.get("ticker") or "").upper() == name:
                    candidates[i] = rebuilt
                    break

    fire_rows = [
        c
        for c in candidates
        if (c.get("fire") or {}).get("kind")
        and not (c.get("fire") or {}).get("chase")
        and c.get("choice") in ("STOCK", "OPTIONS")
    ]
    fire_rows.sort(
        key=lambda r: abs(to_float(r.get("ret_1")) or 0) * (to_float(r.get("rvol")) or 1.0),
        reverse=True,
    )
    fire_rows = fire_rows[:5]

    xhot_rows = []
    for row in candidates:
        hot = hot_map.get(str(row.get("ticker") or "").upper())
        if not hot:
            continue
        info = classify_xhot(hot, row)
        row["xhot"] = info
        if info.get("tag") and (row.get("x") in (None, "", "DATA UNAVAILABLE")):
            row["x"] = info.get("tag")
            row["x_notes"] = info.get("narrative") or row.get("x_notes")
        xhot_rows.append(row)
    move_rank = {"dipped": 3, "will_rise": 2, "will_dip": 2, "noise": 0}
    xhot_rows.sort(
        key=lambda r: (
            1 if (r.get("xhot") or {}).get("playable") else 0,
            move_rank.get((r.get("xhot") or {}).get("move") or "", 0),
            abs(to_float(r.get("ret_1")) or 0),
        ),
        reverse=True,
    )
    xhot_rows = xhot_rows[:10]

    ranked, trades, watch, board = _rank_actionable(candidates)
    picks = desk_picks(trades)
    evidence = attach_evidence(
        asof,
        trades,
        picks,
        bars_map,
        hist_e=hist_e,
        cores=cores,
        token=token,
        today=today,
        getter=getter,
        max_requests=max_requests,
        allow_orats_http=bars_by_ticker is None,
    )
    orats_http += int(evidence.get("http") or 0)
    analog_parked = []
    for row in list(trades):
        if apply_analog_0win_park(row):
            ticker = str(row.get("ticker") or "")
            if ticker:
                analog_parked.append(ticker)
    evidence_line = str(picks.get("evidence_line") or "").strip()
    if analog_parked:
        veto_line = "Analog veto parked to WATCH: " + ", ".join(analog_parked) + "."
        evidence_line = (" ".join(x for x in (evidence_line, veto_line) if x)).strip()
    finished = _board_from_candidates(candidates, live=live, evidence_line=evidence_line)
    trades = finished["trades"]
    watch = finished["watch"]
    board = finished["board"]
    picks = finished["picks"]
    usage = load_usage()
    tape_summary = {k: (tapes[k].get("tape") if k in tapes else "") for k in list(INDEX_TICKERS)}
    rvols = [to_float(s.get("rvol")) for s in snaps.values()]
    rvols = [v for v in rvols if v is not None]
    median_rvol = sorted(rvols)[len(rvols) // 2] if rvols else None
    session_incomplete = bool(live and median_rvol is not None and median_rvol < 0.45)
    tape_errors = [
        {"ticker": k, "error": v.get("error")}
        for k, v in tapes.items()
        if v.get("error")
    ]
    if schwab_pos_error:
        tape_errors.append({"ticker": "SCHWAB_POSITIONS", "error": schwab_pos_error})
    return {
        "asof": asof,
        "sleeve": SLEEVE,
        "regime": regime,
        "groups": groups,
        "group_status": gmap,
        "candidates": candidates,
        "board": board,
        "trades": trades,
        "watch": watch,
        "fire": fire_rows,
        "fire_count": len(fire_rows),
        "xhot": xhot_rows,
        "xhot_count": len(xhot_rows),
        "picks": picks,
        "evidence": evidence,
        "rejections": rejections,
        "trade_count": len(trades),
        "watch_count": len(watch),
        "orats_ok": 1 if cores else 0,
        "orats_http": orats_http,
        "orats_rows": len(cores),
        "orats_error": orats_error,
        "orats_requests_used": usage.get("used") or 0,
        "orats_requests_left": usage.get("left") or 0,
        "tapes": tape_summary,
        "option_names": option_names,
        "chain_empty": [n for n in option_names if not (strikes.get(n) or [])],
        "chain_not_requested": [
            str(c.get("ticker"))
            for c in candidates
            if c.get("chain_status") == "not_requested" and c.get("primary") and c.get("direction") in ("bullish", "bearish")
        ],
        "snaps": snaps,
        "cores_n": len(cores),
        "schwab_chain_errors": chain_errors,
        "tape_errors": tape_errors,
        "session_incomplete": session_incomplete,
        "median_rvol": median_rvol,
        "schwab_pos_error": schwab_pos_error,
    }


def build_delta(asof: str, previous: Optional[dict], current: dict) -> Dict[str, Any]:
    prev_rows = {}
    if previous:
        for row in (previous.get("candidates") or previous.get("board") or []):
            prev_rows[str(row.get("ticker") or "").upper()] = row
    changes = []
    for row in current.get("candidates") or []:
        name = str(row.get("ticker") or "").upper()
        old = prev_rows.get(name)
        if not old:
            if row.get("action") in ("TRADE", "WATCH"):
                changes.append({"ticker": name, "kind": "new", "detail": row.get("primary_name") or row.get("action")})
            continue
        notes = []
        if old.get("action") != row.get("action"):
            notes.append("action %s → %s" % (old.get("action"), row.get("action")))
        if old.get("primary") != row.get("primary"):
            notes.append("setup %s → %s" % (old.get("primary"), row.get("primary")))
        if old.get("choice") != row.get("choice"):
            notes.append("instrument %s → %s" % (old.get("choice"), row.get("choice")))
        old_av = old.get("close")
        new_av = row.get("close")
        av_old = old.get("avwap_swing_low")
        av_new = row.get("avwap_swing_low")
        if av_old and av_new and old_av and new_av:
            old_side = old_av >= av_old
            new_side = new_av >= av_new
            if old_side != new_side:
                notes.append("AVWAP side changed")
        if (old.get("vrp") is not None) and (row.get("vrp") is not None) and abs(old["vrp"] - row["vrp"]) >= 2:
            notes.append("VRP %.1f → %.1f" % (old["vrp"], row["vrp"]))
        if notes:
            changes.append({"ticker": name, "kind": "changed", "detail": "; ".join(notes), "action": row.get("action")})
    removed = []
    for name, old in prev_rows.items():
        if old.get("action") in ("TRADE", "WATCH"):
            now = None
            for row in current.get("candidates") or []:
                if str(row.get("ticker") or "").upper() == name:
                    now = row
                    break
            if now is None or now.get("action") == "IGNORE":
                removed.append({"ticker": name, "kind": "removed", "detail": "no longer actionable"})
    return {
        "asof": asof,
        "changes": changes,
        "removed": removed,
        "new_trades": [c for c in current.get("trades") or []],
        "invalidated": removed,
    }


def build_analyze(asof: str, ticker: str, built: dict) -> dict:
    name = str(ticker).upper()
    for row in built.get("candidates") or []:
        if str(row.get("ticker") or "").upper() == name:
            return row
    return {
        "ticker": name,
        "asof_date": asof,
        "action": "IGNORE",
        "choice": "NO TRADE",
        "reasons": ["not_in_universe_or_missing"],
        "choice_why": ["DATA UNAVAILABLE"],
    }
