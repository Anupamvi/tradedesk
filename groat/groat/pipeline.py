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
from groat.gates import trade_park_reason
from groat.earnings import web_resolve
from groat.num import to_float
from groat.evidence import attach_evidence
from groat.picks import desk_picks
from groat.orats import fetch_cores, fetch_hist_earnings, fetch_strikes, load_usage, parse_core
from groat.prices import ensure_bars
from groat.regime import classify as classify_regime
from groat.rotation import group_status_map, name_group_row, rank_groups
from groat.confidence import options_confidence
from groat.setups import classify_setups
from groat.book import book_index, same_ticket, schwab_held_index
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
    s = 0.0
    primary = row.get("primary") or ""
    s += {"C": 18, "E": 16, "H": 15, "B": 14, "A": 14, "D": 12, "G": 10, "F": 6}.get(primary, 0)
    if (row.get("fire") or {}).get("kind") and not (row.get("fire") or {}).get("chase"):
        s += 6
    rs20 = row.get("rs_20")
    if rs20 is not None:
        if rs20 > 0.05:
            s += 10
        elif rs20 > 0:
            s += 5
        elif rs20 < -0.05:
            s -= 8
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
        defined = "spread" in inst or inst.startswith("long_") or row.get("choice") == "OPTIONS"
        if defined:
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
) -> dict:
    vol = parse_core(core_row)
    earn = earn or earnings_info(ticker, core_row, asof, hist_rows=hist_rows)
    setup = classify_setups(snap, group_row=group_row, earnings=earn, bars=bars)
    direction = setup.get("direction") or "neutral"
    chosen = choose(snap, direction, vol, strikes or [], earn, setup=setup)
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
        "close": snap.get("close"),
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
        "ret_1": snap.get("ret_1"),
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
    row["thesis"] = build_thesis(row)
    row["score"] = score_row(row, str(regime.get("regime") or ""), str(group_row.get("status") or ""))
    if row["stale"] or not snap.get("ok"):
        action = "IGNORE"
        reasons = [snap.get("reason") or "missing_bars"]
    elif row["choice"] == "NO TRADE" or row["score"] < WATCH_SCORE_MIN:
        action = "IGNORE"
        reasons = list(row["choice_why"] or ["score_below_watch"])
        if not row.get("primary"):
            reasons.append("no_setup")
    elif row["choice"] in ("STOCK", "OPTIONS") and row["score"] >= TRADE_SCORE_MIN:
        park = trade_park_reason(row.get("primary"), snap, setup)
        if park:
            action = "WATCH"
            reasons = [park]
        else:
            action = "TRADE"
            reasons = []
    else:
        action = "WATCH"
        reasons = ["below_trade_score"] if row["score"] < TRADE_SCORE_MIN else []
    row["action"] = action
    row["reasons"] = reasons
    return row


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

    def prelim_key(item):
        name, setup, snap = item
        hits = len(setup.get("setups") or [])
        rs = snap.get("rs_20")
        rs_n = rs if rs is not None else -9
        return (1 if setup.get("primary") else 0, hits, rs_n)

    prelim.sort(key=prelim_key, reverse=True)
    option_names = [
        n for n, setup, snap in prelim if setup.get("primary") and setup.get("direction") in ("bullish", "bearish")
    ]
    for name, setup, snap in prelim:
        if (setup.get("fire") or {}).get("kind") and name not in option_names:
            option_names.append(name)
    for ticker in hot_map:
        if ticker not in option_names:
            option_names.append(ticker)
    option_names = option_names[:40]

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
    schwab_held = {}
    if live or bars_by_ticker is None:
        try:
            from groat.schwab import positions_all

            schwab_held = schwab_held_index(positions_all())
        except Exception:
            schwab_held = {}

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
        row = build_candidate(
            asof,
            name,
            snap,
            cores.get(name),
            name_group_row(name, groups),
            regime,
            strikes.get(name),
            bars_map.get(name) or [],
            earn=earn_map.get(name),
            hist_rows=hist_e.get(name),
        )
        book_pos = held.get(name) or {}
        schwab_pos = schwab_held.get(name) or {}
        row["in_book"] = bool(book_pos.get("in_book"))
        row["held_schwab"] = bool(schwab_pos.get("held_schwab"))
        row["held"] = bool(row["in_book"] or row["held_schwab"])
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
            paras.append(row["held_note"] + " Shown for visibility — do not add a second lot unless you have a scale plan.")
            thesis["paragraphs"] = paras
            row["thesis"] = thesis
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

    ranked = sorted(
        [c for c in candidates if c.get("action") in ("TRADE", "WATCH")],
        key=lambda r: (r.get("action") == "TRADE", r.get("score") or 0, r.get("rs_20") or -9),
        reverse=True,
    )
    trades = [c for c in ranked if c.get("action") == "TRADE"][:MAX_FINAL]
    watch = [c for c in ranked if c.get("action") == "WATCH"][:MAX_FINAL]
    # keep only top 5-10 combined, trades first
    board = (trades + watch)[:MAX_FINAL]
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
    usage = load_usage()
    tape_summary = {k: (tapes[k].get("tape") if k in tapes else "") for k in list(INDEX_TICKERS)}
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
        "snaps": snaps,
        "cores_n": len(cores),
        "schwab_chain_errors": chain_errors,
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
