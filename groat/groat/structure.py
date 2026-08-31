"""Underlying first, then stock vs options. Conservative fills. No invented quotes."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

from groat.config import (
    ACCOUNT_DOLLARS,
    CHASE_ATR,
    CONTRACT_MULTIPLIER,
    CREDIT_PCT_MIN,
    DTE_CREDIT_PREF,
    DTE_LONG_PREF,
    DTE_MAX,
    DTE_MIN,
    HOLD_SESSIONS,
    MIN_OI,
    MIN_OI_SHORT,
    RISK_PCT,
    RR_MIN,
    RR_PREFER,
    STOP_ATR_MULT,
    quote_width_cap,
)
from groat.num import to_float


def parse_strike(row: dict) -> Optional[dict]:
    strike = to_float(row.get("strike"))
    dte = to_float(row.get("dte"))
    expiry = str(row.get("expirDate") or "")[:10]
    if strike is None or not expiry:
        return None
    return {
        "strike": strike,
        "dte": dte,
        "expiry": expiry,
        "spot": to_float(row.get("stockPrice")) or to_float(row.get("spotPrice")),
        "delta": to_float(row.get("delta")),
        "gamma": to_float(row.get("gamma")),
        "theta": to_float(row.get("theta")),
        "vega": to_float(row.get("vega")),
        "call_bid": to_float(row.get("callBidPrice")),
        "call_ask": to_float(row.get("callAskPrice")),
        "call_oi": to_float(row.get("callOpenInterest")),
        "call_vol": to_float(row.get("callVolume")),
        "put_bid": to_float(row.get("putBidPrice")),
        "put_ask": to_float(row.get("putAskPrice")),
        "put_oi": to_float(row.get("putOpenInterest")),
        "put_vol": to_float(row.get("putVolume")),
        "smv": to_float(row.get("smvVol")),
    }


def quote_ok(bid, ask, oi, min_oi=MIN_OI) -> bool:
    if bid is None or ask is None or ask <= 0 or bid <= 0:
        return False
    if ask < bid:
        return False
    mid = (bid + ask) / 2.0
    width = ask - bid
    cap = quote_width_cap(mid)
    if cap is None or round(width, 2) > round(cap, 2):
        return False
    if oi is None or oi < min_oi:
        return False
    return True


def _mid(bid, ask) -> Optional[float]:
    if bid is None or ask is None:
        return None
    return (bid + ask) / 2.0


def _clamp01(value) -> Optional[float]:
    number = to_float(value)
    if number is None:
        return None
    if number < 0:
        return 0.0
    if number > 1:
        return 1.0
    return number


def naive_pop_long(delta) -> tuple:
    pop = _clamp01(abs(delta) if delta is not None else None)
    if pop is None:
        return None, "DATA UNAVAILABLE"
    return pop, "naive P(finish ITM) ≈ |delta|. Profit needs a larger move than ITM. Not a backtested win rate."


def naive_pop_debit_vertical(long_leg: dict, short_leg: dict, debit: float, side: str) -> tuple:
    d_l = to_float(long_leg.get("delta"))
    d_s = to_float(short_leg.get("delta"))
    k_l = to_float(long_leg.get("strike"))
    k_s = to_float(short_leg.get("strike"))
    debit = to_float(debit)
    if None in (d_l, d_s, k_l, k_s, debit) or k_l == k_s:
        return None, "DATA UNAVAILABLE"
    if side == "call":
        breakeven = k_l + debit
        t = (breakeven - k_l) / (k_s - k_l)
        pop = d_l + (d_s - d_l) * t
        note = "naive P(spot > breakeven %.2f) from ORATS call deltas. Not a backtested win rate." % breakeven
    else:
        breakeven = k_l - debit
        t = (breakeven - k_s) / (k_l - k_s)
        call_be = d_s + (d_l - d_s) * t
        pop = 1.0 - call_be
        note = "naive P(spot < breakeven %.2f) from ORATS call deltas. Not a backtested win rate." % breakeven
    pop = _clamp01(pop)
    return pop, note


def naive_pop_credit(short_leg: dict, side: str) -> tuple:
    d_s = to_float(short_leg.get("delta"))
    if d_s is None:
        return None, "DATA UNAVAILABLE"
    if side == "put":
        pop = _clamp01(d_s)
        note = "naive P(finish above short put) ≈ call delta. Not a backtested win rate."
    else:
        pop = _clamp01(1.0 - d_s)
        note = "naive P(finish below short call) ≈ 1 − call delta. Not a backtested win rate."
    return pop, note


def stock_plan(snap: dict, direction: str) -> Dict[str, object]:
    px = to_float(snap.get("close"))
    atr = to_float(snap.get("atr14"))
    if px is None or atr is None or atr <= 0 or direction not in ("bullish", "bearish"):
        return {"ok": False, "reason": "missing_price_or_atr", "instrument": "stock"}
    if snap.get("chase") or (to_float(snap.get("extension_atr")) or 0) > CHASE_ATR:
        if direction == "bullish" and str(snap.get("primary") or "") == "B":
            return {"ok": False, "reason": "chase_filter", "instrument": "stock"}
    if direction == "bullish":
        ema_stop = to_float(snap.get("ema20"))
        stop = px - STOP_ATR_MULT * atr
        if ema_stop is not None:
            stop = min(stop, ema_stop - 0.35 * atr)
        hi = to_float(snap.get("hi20"))
        risk = abs(px - stop)
        target = px + RR_PREFER * risk
        if hi is not None and hi > target:
            target = hi
    else:
        stop = px + STOP_ATR_MULT * atr
        lo = to_float(snap.get("lo20"))
        risk = abs(stop - px)
        target = px - RR_PREFER * risk
        if lo is not None and lo < target:
            target = lo
    risk = abs(px - stop)
    reward = abs(target - px)
    if risk <= 0:
        return {"ok": False, "reason": "zero_risk", "instrument": "stock"}
    rr = reward / risk
    shares = int(math.floor((ACCOUNT_DOLLARS * RISK_PCT) / risk))
    if shares < 1:
        return {"ok": False, "reason": "size_zero", "instrument": "stock", "rr": rr}
    return {
        "ok": rr >= RR_MIN,
        "reason": "" if rr >= RR_MIN else "rr_below_min",
        "instrument": "stock",
        "side": "long" if direction == "bullish" else "short",
        "entry": px,
        "stop": stop,
        "target": target,
        "rr": rr,
        "shares": shares,
        "risk_dollars": shares * risk,
        "notional": shares * px,
        "hold_sessions": HOLD_SESSIONS,
        "invalidation": "close beyond stop %.2f" % stop,
    }


def _by_expiry(rows: Sequence[dict]) -> Dict[str, list]:
    out = {}
    for raw in rows or []:
        parsed = parse_strike(raw)
        if not parsed:
            continue
        dte = parsed.get("dte")
        if dte is None or not (DTE_MIN <= dte <= DTE_MAX):
            continue
        out.setdefault(parsed["expiry"], []).append(parsed)
    return out


def _closest(rows: List[dict], target_delta: float, side: str) -> Optional[dict]:
    best = None
    best_gap = None
    for row in rows:
        delta = to_float(row.get("delta"))
        if delta is None:
            continue
        use = delta if side == "call" else delta - 1.0
        gap = abs(abs(use) - target_delta)
        bid = row.get("call_bid") if side == "call" else row.get("put_bid")
        ask = row.get("call_ask") if side == "call" else row.get("put_ask")
        oi = row.get("call_oi") if side == "call" else row.get("put_oi")
        if not quote_ok(bid, ask, oi, MIN_OI):
            continue
        if best is None or gap < best_gap:
            best = row
            best_gap = gap
    return best


def long_option(rows: Sequence[dict], direction: str, earnings: dict) -> Optional[dict]:
    chain = _by_expiry(rows)
    if not chain:
        return None
    side = "call" if direction == "bullish" else "put"
    ranked = []
    for expiry, legs in chain.items():
        if _earnings_blocks(expiry, earnings):
            continue
        dte = to_float(legs[0].get("dte"))
        pref_lo, pref_hi = DTE_LONG_PREF
        pref = 0 if dte is not None and pref_lo <= dte <= pref_hi else 1
        pick = _closest(legs, 0.55, side)
        if not pick:
            continue
        ask = pick["call_ask"] if side == "call" else pick["put_ask"]
        bid = pick["call_bid"] if side == "call" else pick["put_bid"]
        debit = ask
        theta = to_float(pick.get("theta")) or 0.0
        hold_theta = abs(theta) * HOLD_SESSIONS
        if debit and hold_theta > 0.40 * debit:
            continue
        delta = to_float(pick.get("delta"))
        if side == "put" and delta is not None:
            delta = delta - 1.0
        if delta is not None and abs(delta) < 0.35:
            continue
        contracts = int(math.floor((ACCOUNT_DOLLARS * RISK_PCT) / (debit * CONTRACT_MULTIPLIER))) if debit else 0
        if contracts < 1:
            continue
        target_mult = 2.0
        rr = (target_mult * debit - debit) / debit if debit else 0
        ranked.append(
            (
                pref,
                -abs((delta or 0) - (0.55 if side == "call" else -0.55)),
                {
                    "ok": True,
                    "instrument": "long_%s" % side,
                    "expiry": expiry,
                    "dte": dte,
                    "strike": pick["strike"],
                    "debit": debit,
                    "bid": bid,
                    "ask": ask,
                    "mid": _mid(bid, ask),
                    "delta": delta,
                    "gamma": pick.get("gamma"),
                    "theta": pick.get("theta"),
                    "vega": pick.get("vega"),
                    "oi": pick.get("call_oi") if side == "call" else pick.get("put_oi"),
                    "volume": pick.get("call_vol") if side == "call" else pick.get("put_vol"),
                    "contracts": contracts,
                    "max_loss": contracts * debit * CONTRACT_MULTIPLIER,
                    "breakeven": (pick["spot"] or 0) + debit if side == "call" else (pick["spot"] or 0) - debit,
                    "rr": rr,
                    "target_debit": debit,
                    "target_credit": None,
                    "premium": debit,
                    "premium_side": "debit",
                    "naive_pop": naive_pop_long(delta)[0],
                    "naive_pop_note": naive_pop_long(delta)[1],
                    "fill_assumption": "ask (never mid)",
                    "legs": "BUY %s %s %s %s @ %.2f ask"
                    % (contracts, expiry, pick["strike"], side[0].upper(), debit),
                    "reason": "cheap-to-fair vol directional with 21-75 DTE",
                },
            )
        )
    if not ranked:
        return None
    ranked.sort(key=lambda x: (x[0], x[1]))
    return ranked[0][2]


def _vertical_width_ok(gap) -> bool:
    gap = to_float(gap)
    if gap is None:
        return False
    return gap in (2.5, 5.0, 10.0) or 4.0 <= gap <= 11.0


def debit_spread(rows: Sequence[dict], direction: str, earnings: dict) -> Optional[dict]:
    chain = _by_expiry(rows)
    side = "call" if direction == "bullish" else "put"
    ranked = []
    for expiry, legs in chain.items():
        if _earnings_blocks(expiry, earnings):
            continue
        dte = to_float(legs[0].get("dte"))
        pref_lo, pref_hi = DTE_LONG_PREF
        pref = 0 if dte is not None and pref_lo <= dte <= pref_hi else 1
        for long_leg in legs:
            ld = to_float(long_leg.get("delta"))
            if ld is None:
                continue
            use = ld if side == "call" else ld - 1.0
            if not (0.30 <= abs(use) <= 0.60):
                continue
            long_ask = long_leg["call_ask"] if side == "call" else long_leg["put_ask"]
            long_bid = long_leg["call_bid"] if side == "call" else long_leg["put_bid"]
            long_oi = long_leg["call_oi"] if side == "call" else long_leg["put_oi"]
            if not quote_ok(long_bid, long_ask, long_oi, MIN_OI):
                continue
            for short_leg in legs:
                gap = abs(short_leg["strike"] - long_leg["strike"])
                if not _vertical_width_ok(gap):
                    continue
                if side == "call" and short_leg["strike"] <= long_leg["strike"]:
                    continue
                if side == "put" and short_leg["strike"] >= long_leg["strike"]:
                    continue
                short_bid = short_leg["call_bid"] if side == "call" else short_leg["put_bid"]
                short_ask = short_leg["call_ask"] if side == "call" else short_leg["put_ask"]
                short_oi = short_leg["call_oi"] if side == "call" else short_leg["put_oi"]
                if not quote_ok(short_bid, short_ask, short_oi, MIN_OI_SHORT):
                    continue
                if long_ask is None or short_bid is None:
                    continue
                debit = long_ask - short_bid
                width = gap
                if debit <= 0 or width <= 0 or debit >= 0.70 * width:
                    continue
                max_gain = width - debit
                rr = max_gain / debit if debit else 0
                contracts = int(math.floor((ACCOUNT_DOLLARS * RISK_PCT) / (debit * CONTRACT_MULTIPLIER)))
                if contracts < 1 or rr < RR_MIN:
                    continue
                pop, pop_note = naive_pop_debit_vertical(long_leg, short_leg, debit, side)
                if side == "call":
                    breakeven = long_leg["strike"] + debit
                else:
                    breakeven = long_leg["strike"] - debit
                cand = {
                    "ok": True,
                    "instrument": "debit_%s_spread" % side,
                    "expiry": expiry,
                    "dte": dte,
                    "long_strike": long_leg["strike"],
                    "short_strike": short_leg["strike"],
                    "debit": debit,
                    "width": width,
                    "rr": rr,
                    "delta": (long_leg.get("delta") or 0) - (short_leg.get("delta") or 0)
                    if side == "call"
                    else ((long_leg.get("delta") or 1) - 1) - ((short_leg.get("delta") or 1) - 1),
                    "theta": (to_float(long_leg.get("theta")) or 0) - (to_float(short_leg.get("theta")) or 0),
                    "vega": (to_float(long_leg.get("vega")) or 0) - (to_float(short_leg.get("vega")) or 0),
                    "gamma": (to_float(long_leg.get("gamma")) or 0) - (to_float(short_leg.get("gamma")) or 0),
                    "contracts": contracts,
                    "max_loss": contracts * debit * CONTRACT_MULTIPLIER,
                    "max_gain": contracts * max_gain * CONTRACT_MULTIPLIER,
                    "target_debit": debit,
                    "target_credit": None,
                    "premium": debit,
                    "premium_side": "debit",
                    "long_delta": long_leg.get("delta"),
                    "short_delta": short_leg.get("delta"),
                    "breakeven": breakeven,
                    "naive_pop": pop,
                    "naive_pop_note": pop_note,
                    "fill_assumption": "long ask minus short bid (never mid)",
                    "legs": "BUY %s %s / SELL %s %s %s"
                    % (long_leg["strike"], side, short_leg["strike"], side, expiry),
                    "reason": "defined-risk directional; better than naked long when IV is not cheap",
                }
                long_d = abs(use)
                ranked.append(
                    (
                        pref,
                        0 if rr >= RR_PREFER else 1,
                        abs(long_d - 0.45),
                        -rr,
                        cand,
                    )
                )
    if not ranked:
        return None
    ranked.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    return ranked[0][4]


def credit_spread(rows: Sequence[dict], direction: str, earnings: dict, iv_rich: bool) -> Optional[dict]:
    chain = _by_expiry(rows)
    # bullish → put credit; bearish → call credit
    side = "put" if direction == "bullish" else "call"
    best = None
    for expiry, legs in chain.items():
        if _earnings_blocks(expiry, earnings):
            continue
        dte = to_float(legs[0].get("dte"))
        pref_lo, pref_hi = DTE_CREDIT_PREF
        if dte is None or not (pref_lo <= dte <= pref_hi or DTE_MIN <= dte <= DTE_MAX):
            continue
        short = _closest(legs, 0.22, side)
        if not short:
            continue
        width_choices = []
        for other in legs:
            gap = abs(other["strike"] - short["strike"])
            if _vertical_width_ok(gap):
                width_choices.append(other)
        for long_leg in width_choices:
            if side == "put" and long_leg["strike"] >= short["strike"]:
                continue
            if side == "call" and long_leg["strike"] <= short["strike"]:
                continue
            short_bid = short["put_bid"] if side == "put" else short["call_bid"]
            long_ask = long_leg["put_ask"] if side == "put" else long_leg["call_ask"]
            short_ask = short["put_ask"] if side == "put" else short["call_ask"]
            short_oi = short["put_oi"] if side == "put" else short["call_oi"]
            long_oi = long_leg["put_oi"] if side == "put" else long_leg["call_oi"]
            if not quote_ok(short_bid, short_ask, short_oi, MIN_OI_SHORT):
                continue
            if long_ask is None:
                continue
            credit = short_bid - long_ask
            width = abs(short["strike"] - long_leg["strike"])
            if credit is None or credit <= 0 or width <= 0:
                continue
            if credit / width < CREDIT_PCT_MIN:
                continue
            if long_oi is None or long_oi < MIN_OI:
                continue
            max_loss = width - credit
            rr = credit / max_loss if max_loss else 0
            contracts = int(math.floor((ACCOUNT_DOLLARS * RISK_PCT) / (max_loss * CONTRACT_MULTIPLIER)))
            if contracts < 1:
                continue
            pop, pop_note = naive_pop_credit(short, side)
            cand = {
                "ok": True,
                "instrument": "%s_credit_spread" % side,
                "expiry": expiry,
                "dte": dte,
                "short_strike": short["strike"],
                "long_strike": long_leg["strike"],
                "credit": credit,
                "width": width,
                "credit_pct": credit / width,
                "rr": rr,
                "contracts": contracts,
                "max_loss": contracts * max_loss * CONTRACT_MULTIPLIER,
                "max_gain": contracts * credit * CONTRACT_MULTIPLIER,
                "target_debit": None,
                "target_credit": credit,
                "premium": credit,
                "premium_side": "credit",
                "naive_pop": pop,
                "naive_pop_note": pop_note,
                "delta": (short.get("delta") - 1.0) if side == "put" else short.get("delta"),
                "theta": short.get("theta"),
                "vega": short.get("vega"),
                "gamma": short.get("gamma"),
                "fill_assumption": "short bid minus long ask (never mid)",
                "legs": "SELL %s %s / BUY %s %s %s" % (short["strike"], side, long_leg["strike"], side, expiry),
                "reason": "IV rich vs realized; defined-risk short premium. R/R is credit/width, not 2:1 directional.",
            }
            key = cand["credit_pct"]
            if best is None or key > best.get("credit_pct", 0):
                best = cand
    if best is None:
        return None
    if not iv_rich:
        out = dict(best)
        out["ok"] = False
        out["reason"] = "IV not rich — do not sell premium"
        return out
    return best


def _earnings_blocks(expiry: str, earnings: dict) -> bool:
    if not earnings:
        return True
    if earnings.get("source") == "exempt":
        return False
    if not earnings.get("usable"):
        return True
    earn = earnings.get("date")
    if not earn:
        return True
    if earnings.get("overlaps_hold"):
        return True
    return earn <= expiry


def _review(strategy: str, status: str, reason: str, cand: Optional[dict] = None) -> dict:
    row = {
        "strategy": strategy,
        "status": status,
        "reason": reason,
        "instrument": (cand or {}).get("instrument") if cand else strategy,
        "target_debit": (cand or {}).get("target_debit") if cand else None,
        "target_credit": (cand or {}).get("target_credit") if cand else None,
        "expiry": (cand or {}).get("expiry") if cand else None,
        "legs": (cand or {}).get("legs") if cand else None,
        "rr": (cand or {}).get("rr") if cand else None,
    }
    return row


def choose(
    snap: dict,
    direction: str,
    vol: dict,
    strikes: Sequence[dict],
    earnings: dict,
    setup: Optional[dict] = None,
) -> Dict[str, object]:
    setup = setup or {}
    reviews = []
    if direction not in ("bullish", "bearish"):
        return {
            "choice": "NO TRADE",
            "why": ["no directional underlying thesis"],
            "stock": None,
            "options": None,
            "reviews": [_review("all", "REJECT", "no directional underlying thesis")],
        }
    if setup.get("chase") and setup.get("primary") == "B":
        return {
            "choice": "NO TRADE",
            "why": ["extended breakout — do not chase"],
            "stock": None,
            "options": None,
            "reviews": [_review("all", "REJECT", "extended breakout — do not chase")],
        }
    stock = stock_plan({**snap, "primary": setup.get("primary"), "chase": setup.get("chase")}, direction)
    if stock.get("ok"):
        reviews.append(_review("stock", "PASS", stock.get("reason") or "stock plan clears R/R", stock))
    else:
        reviews.append(_review("stock", "REJECT", stock.get("reason") or "stock plan failed", stock))

    iv = to_float(vol.get("iv30"))
    hv = to_float(vol.get("hv20"))
    fcst = to_float(vol.get("forecast_20d"))
    vrp = to_float(vol.get("vrp"))
    vol_missing = iv is None or hv is None
    iv_rich = (vrp is not None and vrp >= 4.0) or (iv is not None and fcst is not None and iv > fcst + 2)
    iv_cheap = (vrp is not None and vrp <= -2.0) or (iv is not None and fcst is not None and iv + 1 < fcst)

    options_block = None
    if not earnings.get("usable") and earnings.get("source") != "exempt":
        options_block = "earnings DATA UNAVAILABLE — ordinary options rejected"
    elif earnings.get("overlaps_hold") and earnings.get("source") != "exempt":
        options_block = "earnings inside intended hold — ordinary options rejected (not an EVENT TRADE)"
    elif not strikes:
        options_block = "option chain DATA UNAVAILABLE"

    builders = (
        ("long_call", "bullish", lambda: long_option(strikes, "bullish", earnings)),
        ("long_put", "bearish", lambda: long_option(strikes, "bearish", earnings)),
        ("debit_call_spread", "bullish", lambda: debit_spread(strikes, "bullish", earnings)),
        ("debit_put_spread", "bearish", lambda: debit_spread(strikes, "bearish", earnings)),
        ("put_credit_spread", "bullish", lambda: credit_spread(strikes, "bullish", earnings, iv_rich)),
        ("call_credit_spread", "bearish", lambda: credit_spread(strikes, "bearish", earnings, iv_rich)),
    )
    priced = {}
    for name, aligned, fn in builders:
        if options_block:
            reviews.append(_review(name, "REJECT", options_block))
            continue
        cand = fn()
        priced[name] = cand
        if aligned != direction:
            reviews.append(
                _review(
                    name,
                    "REJECT",
                    "against %s underlying thesis" % direction,
                    cand if cand and cand.get("ok") else None,
                )
            )
            continue
        if not cand:
            reviews.append(_review(name, "REJECT", "no liquid structure in 21-75 DTE"))
            continue
        if not cand.get("ok"):
            reviews.append(_review(name, "REJECT", cand.get("reason") or "failed gates", cand))
            continue
        reviews.append(_review(name, "PASS", cand.get("reason") or "clears gates", cand))

    long = priced.get("long_call") if direction == "bullish" else priced.get("long_put")
    debit = priced.get("debit_call_spread") if direction == "bullish" else priced.get("debit_put_spread")
    credit = priced.get("put_credit_spread") if direction == "bullish" else priced.get("call_credit_spread")
    if long and not long.get("ok"):
        long = None
    if debit and not debit.get("ok"):
        debit = None
    if credit and not credit.get("ok"):
        credit = None

    best_opt = None
    for cand in (debit, long, credit):
        if not cand or not cand.get("ok"):
            continue
        if best_opt is None:
            best_opt = cand
            continue
        if cand.get("instrument", "").endswith("credit_spread") and (stock.get("ok") and (stock.get("rr") or 0) >= RR_PREFER):
            continue
        if (cand.get("rr") or 0) > (best_opt.get("rr") or 0):
            best_opt = cand

    why = []
    if vol_missing:
        why.append("ORATS IV/HV DATA UNAVAILABLE — do not invent; options de-prioritized")
    if iv_cheap:
        why.append("IV below realized/forecast — long premium allowed if liquid")
    if iv_rich:
        why.append("IV rich vs realized — prefer stock or defined-risk short premium")
    if options_block:
        why.append(options_block)

    choice = "NO TRADE"
    picked = None
    if best_opt and best_opt.get("ok") and (not stock.get("ok") or _options_better(best_opt, stock, iv_cheap, iv_rich)):
        choice = "OPTIONS"
        picked = best_opt
        why.append(
            "shortlisted %s after reviewing stock, long call/put, debit spreads, and credit spreads"
            % (best_opt.get("instrument") or "options")
        )
    elif stock.get("ok"):
        choice = "STOCK"
        picked = stock
        why.append("stock won the shortlist versus priced option structures")
    else:
        why.append(stock.get("reason") or "neither stock nor options cleared gates")

    return {
        "choice": choice,
        "why": why,
        "picked": picked,
        "stock": stock,
        "options": best_opt,
        "long": long,
        "debit": debit,
        "credit": credit,
        "reviews": reviews,
        "iv_rich": iv_rich,
        "iv_cheap": iv_cheap,
        "vol_missing": vol_missing,
        "options_block": options_block,
    }


def _options_better(opt: dict, stock: dict, iv_cheap: bool, iv_rich: bool) -> bool:
    opt_rr = to_float(opt.get("rr")) or 0
    stk_rr = to_float(stock.get("rr")) or 0
    inst = str(opt.get("instrument") or "")
    if inst.endswith("credit_spread"):
        return iv_rich and stk_rr < RR_PREFER
    if "long_" in inst:
        return iv_cheap and opt_rr >= RR_MIN
    if "debit_" in inst:
        if iv_cheap and opt_rr >= RR_MIN:
            return True
        return opt_rr >= stk_rr or (opt.get("max_loss") or 0) < (stock.get("risk_dollars") or 0) * 0.8
    return opt_rr > stk_rr
