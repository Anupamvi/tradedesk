"""Hard strike bands vs live last. Fail closed. Gates only — no looser fallbacks."""

from __future__ import annotations

from typing import List, Optional

from xhigh.config import CONTRACT_MULTIPLIER
from xhigh.dates import days_between
from xhigh.earnings import options_allowed
from xhigh.num import to_float
from xhigh.tape import chase as tape_chase

SHORT_PREMIUM = ("csp", "put_credit", "call_credit", "iron_condor")


def spot_from_quote(q: Optional[dict]) -> Optional[float]:
    if not q:
        return None
    last = to_float(q.get("last"))
    if last is not None and last > 0:
        return last
    close = to_float(q.get("close"))
    if close is not None and close > 0 and not q.get("last_is_bid_or_mark"):
        return close
    return None


def _dte(row: dict, asof: str) -> Optional[int]:
    dte = to_float(row.get("dte"))
    if dte is not None:
        return int(dte)
    expiry = str(row.get("expiry") or "")[:10]
    gap = days_between(expiry, asof)
    return gap


def _need(block: dict, key: str) -> Optional[float]:
    if key not in block or block.get(key) is None:
        return None
    return to_float(block.get(key))


def _in_band(value: Optional[float], lo: Optional[float], hi: Optional[float]) -> bool:
    if value is None or lo is None or hi is None:
        return False
    return lo - 1e-9 <= value <= hi + 1e-9


def otm_below(strike: float, last: float) -> Optional[float]:
    if last is None or last <= 0 or strike is None:
        return None
    return (last - strike) / last


def otm_above(strike: float, last: float) -> Optional[float]:
    if last is None or last <= 0 or strike is None:
        return None
    return (strike - last) / last


def width_ok(left: float, right: float, last: float, max_frac: Optional[float]) -> bool:
    if last is None or last <= 0 or max_frac is None:
        return False
    return abs(left - right) / last <= max_frac + 1e-9


def ticket_legal(
    structure: str,
    last: float,
    strike: Optional[float] = None,
    short_strike: Optional[float] = None,
    gates: Optional[dict] = None,
    long_put: Optional[float] = None,
    long_call: Optional[float] = None,
    short_put: Optional[float] = None,
    short_call: Optional[float] = None,
) -> bool:
    if not gates or last is None or last <= 0:
        return False
    width_cap = _need(gates, "max_width_frac")
    if structure == "csp":
        g = gates.get("csp") or {}
        otm = otm_below(strike, last)
        return _in_band(otm, _need(g, "otm_min"), _need(g, "otm_max"))
    if structure == "call_debit":
        g = gates.get("call_debit") or {}
        if short_strike is None or strike is None:
            return False
        lo = otm_above(strike, last)
        so = otm_above(short_strike, last)
        if not _in_band(lo, _need(g, "long_otm_min"), _need(g, "long_otm_max")):
            return False
        if not _in_band(so, _need(g, "short_otm_min"), _need(g, "short_otm_max")):
            return False
        if short_strike <= strike:
            return False
        return width_ok(strike, short_strike, last, width_cap)
    if structure == "put_credit":
        g = gates.get("put_credit") or {}
        sp = short_put if short_put is not None else strike
        lp = long_put
        if sp is None or lp is None or lp >= sp:
            return False
        otm = otm_below(sp, last)
        if not _in_band(otm, _need(g, "short_otm_min"), _need(g, "short_otm_max")):
            return False
        return width_ok(sp, lp, last, width_cap)
    if structure == "call_credit":
        g = gates.get("call_credit") or {}
        sc = short_call if short_call is not None else strike
        lc = long_call if long_call is not None else short_strike
        if sc is None or lc is None or lc <= sc:
            return False
        otm = otm_above(sc, last)
        if not _in_band(otm, _need(g, "short_otm_min"), _need(g, "short_otm_max")):
            return False
        return width_ok(sc, lc, last, width_cap)
    if structure == "put_debit":
        g = gates.get("put_debit") or {}
        lp = strike
        sp = short_strike
        if lp is None or sp is None or sp >= lp:
            return False
        lo = otm_above(lp, last)
        if not _in_band(lo, _need(g, "long_otm_min"), _need(g, "long_otm_max")):
            return False
        return width_ok(lp, sp, last, width_cap)
    if structure == "iron_condor":
        return ticket_legal("put_credit", last, gates=gates, short_put=short_put, long_put=long_put) and ticket_legal(
            "call_credit", last, gates=gates, short_call=short_call, long_call=long_call
        )
    if structure == "stock":
        return True
    return False


def _session_ok(row: dict, asof: str, earn: dict, gates: dict) -> Optional[int]:
    dte_min = int(gates.get("dte_min") or 25)
    dte_max = int(gates.get("dte_max") or 45)
    buf = int(gates.get("earnings_buffer_days") or 3)
    dte = _dte(row, asof)
    expiry = str(row.get("expiry") or "")[:10]
    if dte is None or dte < dte_min or dte > dte_max:
        return None
    if not options_allowed(earn, expiry, buf):
        return None
    return dte


def _spread_ok(bid: Optional[float], ask: Optional[float], gates: dict) -> bool:
    g = gates.get("swing") or {}
    if bid is None or ask is None or bid <= 0:
        return True if ask is None else False
    mid = (ask + bid) / 2.0
    cap = max(float(g.get("quote_width_abs") or 0.25), float(g.get("quote_width_frac") or 0.10) * mid)
    return (ask - bid) <= cap


def pick_csp(puts: List[dict], last: float, asof: str, earn: dict, gates: dict) -> Optional[dict]:
    g = gates.get("csp") or {}
    otm_min = _need(g, "otm_min")
    otm_max = _need(g, "otm_max")
    ideal = to_float(g.get("ideal_otm")) or 0.10
    min_cr = to_float(g.get("min_credit_frac")) or 0.004
    max_sp = to_float(g.get("max_spread_frac_of_credit")) or 0.35
    if last is None or last <= 0 or not earn.get("usable") or otm_min is None or otm_max is None:
        return None
    best = None
    best_key = -1.0
    for row in puts:
        dte = _session_ok(row, asof, earn, gates)
        strike = to_float(row.get("strike"))
        bid = to_float(row.get("bid"))
        ask = to_float(row.get("ask"))
        expiry = str(row.get("expiry") or "")[:10]
        if dte is None or strike is None or bid is None or bid <= 0:
            continue
        otm = otm_below(strike, last)
        if not _in_band(otm, otm_min, otm_max):
            continue
        if bid < min_cr * strike:
            continue
        if ask is not None and ask - bid > max_sp * bid:
            continue
        if not ticket_legal("csp", last, strike, None, gates):
            continue
        band = 1.0 - min(abs((otm or 0) - ideal) / max(ideal, 1e-6), 1.0)
        if band > best_key:
            best_key = band
            best = {
                "structure": "csp",
                "strike": strike,
                "credit": bid,
                "limit": bid,
                "width": strike,
                "collateral": strike * CONTRACT_MULTIPLIER,
                "expiry": expiry,
                "dte": dte,
                "otm": otm,
                "otm_s": "%.1f%% OTM" % ((otm or 0) * 100),
                "spot": last,
                "delta": to_float(row.get("delta")),
                "short_delta": to_float(row.get("delta")),
                "invalidation": "assignment at %s" % strike,
            }
    return best


def pick_put_credit(puts: List[dict], last: float, asof: str, earn: dict, gates: dict) -> Optional[dict]:
    g = gates.get("put_credit") or {}
    width_cap = _need(gates, "max_width_frac")
    lo = _need(g, "short_otm_min")
    hi = _need(g, "short_otm_max")
    min_cr = to_float(g.get("min_credit")) or 0.20
    if last is None or last <= 0 or not earn.get("usable"):
        return None
    best = None
    best_cr = -1.0
    shorts = []
    longs = []
    for row in puts:
        dte = _session_ok(row, asof, earn, gates)
        strike = to_float(row.get("strike"))
        if dte is None or strike is None:
            continue
        otm = otm_below(strike, last)
        if _in_band(otm, lo, hi):
            shorts.append(row)
        longs.append(row)
    for short in shorts:
        s_strike = to_float(short.get("strike"))
        s_bid = to_float(short.get("bid"))
        expiry = str(short.get("expiry") or "")[:10]
        dte = _dte(short, asof)
        if s_strike is None or s_bid is None or s_bid <= 0:
            continue
        for long in longs:
            if str(long.get("expiry") or "")[:10] != expiry:
                continue
            l_strike = to_float(long.get("strike"))
            l_ask = to_float(long.get("ask"))
            if l_strike is None or l_ask is None or l_ask <= 0 or l_strike >= s_strike:
                continue
            if not width_ok(s_strike, l_strike, last, width_cap):
                continue
            if not ticket_legal("put_credit", last, gates=gates, short_put=s_strike, long_put=l_strike):
                continue
            net = s_bid - l_ask
            if net < min_cr:
                continue
            if net > best_cr:
                best_cr = net
                width = s_strike - l_strike
                best = {
                    "structure": "put_credit",
                    "strike": s_strike,
                    "short_strike": s_strike,
                    "long_strike": l_strike,
                    "credit": net,
                    "limit": net,
                    "width": width,
                    "expiry": expiry,
                    "dte": dte,
                    "otm": otm_below(s_strike, last),
                    "otm_s": "short %.1f%% OTM" % ((otm_below(s_strike, last) or 0) * 100),
                    "spot": last,
                    "short_delta": to_float(short.get("delta")),
                    "long_delta": to_float(long.get("delta")),
                    "invalidation": "short put %s" % s_strike,
                }
    return best


def pick_call_credit(calls: List[dict], last: float, asof: str, earn: dict, gates: dict) -> Optional[dict]:
    g = gates.get("call_credit") or {}
    width_cap = _need(gates, "max_width_frac")
    lo = _need(g, "short_otm_min")
    hi = _need(g, "short_otm_max")
    min_cr = to_float(g.get("min_credit")) or 0.20
    if last is None or last <= 0 or not earn.get("usable"):
        return None
    best = None
    best_cr = -1.0
    shorts = []
    longs = []
    for row in calls:
        dte = _session_ok(row, asof, earn, gates)
        strike = to_float(row.get("strike"))
        if dte is None or strike is None:
            continue
        if _in_band(otm_above(strike, last), lo, hi):
            shorts.append(row)
        longs.append(row)
    for short in shorts:
        s_strike = to_float(short.get("strike"))
        s_bid = to_float(short.get("bid"))
        expiry = str(short.get("expiry") or "")[:10]
        dte = _dte(short, asof)
        if s_strike is None or s_bid is None or s_bid <= 0:
            continue
        for long in longs:
            if str(long.get("expiry") or "")[:10] != expiry:
                continue
            l_strike = to_float(long.get("strike"))
            l_ask = to_float(long.get("ask"))
            if l_strike is None or l_ask is None or l_ask <= 0 or l_strike <= s_strike:
                continue
            if not width_ok(s_strike, l_strike, last, width_cap):
                continue
            if not ticket_legal("call_credit", last, gates=gates, short_call=s_strike, long_call=l_strike):
                continue
            net = s_bid - l_ask
            if net < min_cr:
                continue
            if net > best_cr:
                best_cr = net
                best = {
                    "structure": "call_credit",
                    "strike": s_strike,
                    "short_strike": s_strike,
                    "long_strike": l_strike,
                    "credit": net,
                    "limit": net,
                    "width": l_strike - s_strike,
                    "expiry": expiry,
                    "dte": dte,
                    "otm": otm_above(s_strike, last),
                    "otm_s": "short %.1f%% OTM" % ((otm_above(s_strike, last) or 0) * 100),
                    "spot": last,
                    "short_delta": to_float(short.get("delta")),
                    "long_delta": to_float(long.get("delta")),
                    "invalidation": "short call %s" % s_strike,
                }
    return best


def pick_call_debit(calls: List[dict], last: float, asof: str, earn: dict, gates: dict) -> Optional[dict]:
    g = gates.get("call_debit") or {}
    width_cap = _need(gates, "max_width_frac")
    if last is None or last <= 0 or not earn.get("usable"):
        return None
    long_lo = last * (1 + (_need(g, "long_otm_min") or 0))
    long_hi = last * (1 + (_need(g, "long_otm_max") or 0))
    short_lo = last * (1 + (_need(g, "short_otm_min") or 0))
    short_hi = last * (1 + (_need(g, "short_otm_max") or 0))
    min_debit = to_float(g.get("min_debit")) or 0.40
    min_rr = to_float(g.get("min_rr")) or 1.2
    max_rr = to_float(g.get("max_rr")) or 4.0
    longs = []
    shorts = []
    for row in calls:
        dte = _session_ok(row, asof, earn, gates)
        strike = to_float(row.get("strike"))
        if dte is None or strike is None:
            continue
        if long_lo <= strike <= long_hi:
            longs.append(row)
        if short_lo <= strike <= short_hi:
            shorts.append(row)
    best = None
    best_key = -1.0
    for long in longs:
        l_strike = to_float(long.get("strike"))
        l_ask = to_float(long.get("ask"))
        l_bid = to_float(long.get("bid"))
        expiry = str(long.get("expiry") or "")[:10]
        dte = _dte(long, asof)
        if l_strike is None or l_ask is None or l_ask <= 0:
            continue
        if not _spread_ok(l_bid, l_ask, gates):
            continue
        for short in shorts:
            if str(short.get("expiry") or "")[:10] != expiry:
                continue
            s_strike = to_float(short.get("strike"))
            s_bid = to_float(short.get("bid"))
            if s_strike is None or s_bid is None or s_strike <= l_strike:
                continue
            if not ticket_legal("call_debit", last, l_strike, s_strike, gates):
                continue
            debit = l_ask - s_bid
            width = s_strike - l_strike
            if debit < min_debit or width <= 0 or not width_ok(l_strike, s_strike, last, width_cap):
                continue
            max_gain = width - debit
            if max_gain <= 0:
                continue
            rr = max_gain / debit
            if rr < min_rr or rr > max_rr:
                continue
            key = 1.0 / (1.0 + abs(rr - 2.0))
            if key > best_key:
                best_key = key
                lo = otm_above(l_strike, last) or 0
                so = otm_above(s_strike, last) or 0
                best = {
                    "structure": "call_debit",
                    "long_strike": l_strike,
                    "short_strike": s_strike,
                    "debit": debit,
                    "limit": debit,
                    "width": width,
                    "expiry": expiry,
                    "dte": dte,
                    "rr": rr,
                    "max_gain": max_gain,
                    "spot": last,
                    "otm_s": "long %+.1f%% / short %+.1f%%" % (lo * 100, so * 100),
                    "long_delta": to_float(long.get("delta")),
                    "short_delta": to_float(short.get("delta")),
                }
    return best


def pick_put_debit(puts: List[dict], last: float, asof: str, earn: dict, gates: dict) -> Optional[dict]:
    g = gates.get("put_debit") or {}
    width_cap = _need(gates, "max_width_frac")
    if last is None or last <= 0 or not earn.get("usable"):
        return None
    long_lo = last * (1 + (_need(g, "long_otm_min") or 0))
    long_hi = last * (1 + (_need(g, "long_otm_max") or 0))
    min_debit = to_float(g.get("min_debit")) or 0.40
    min_rr = to_float(g.get("min_rr")) or 1.2
    max_rr = to_float(g.get("max_rr")) or 4.0
    longs = []
    shorts = []
    for row in puts:
        dte = _session_ok(row, asof, earn, gates)
        strike = to_float(row.get("strike"))
        if dte is None or strike is None:
            continue
        if long_lo <= strike <= long_hi:
            longs.append(row)
        shorts.append(row)
    best = None
    best_key = -1.0
    for long in longs:
        l_strike = to_float(long.get("strike"))
        l_ask = to_float(long.get("ask"))
        expiry = str(long.get("expiry") or "")[:10]
        dte = _dte(long, asof)
        if l_strike is None or l_ask is None or l_ask <= 0:
            continue
        for short in shorts:
            if str(short.get("expiry") or "")[:10] != expiry:
                continue
            s_strike = to_float(short.get("strike"))
            s_bid = to_float(short.get("bid"))
            if s_strike is None or s_bid is None or s_strike >= l_strike:
                continue
            if not ticket_legal("put_debit", last, l_strike, s_strike, gates):
                continue
            debit = l_ask - s_bid
            width = l_strike - s_strike
            if debit < min_debit or width <= 0 or not width_ok(l_strike, s_strike, last, width_cap):
                continue
            max_gain = width - debit
            if max_gain <= 0:
                continue
            rr = max_gain / debit
            if rr < min_rr or rr > max_rr:
                continue
            key = 1.0 / (1.0 + abs(rr - 2.0))
            if key > best_key:
                best_key = key
                best = {
                    "structure": "put_debit",
                    "long_strike": l_strike,
                    "short_strike": s_strike,
                    "debit": debit,
                    "limit": debit,
                    "width": width,
                    "expiry": expiry,
                    "dte": dte,
                    "rr": rr,
                    "max_gain": max_gain,
                    "spot": last,
                    "otm_s": "long %s / short %s" % (l_strike, s_strike),
                    "long_delta": to_float(long.get("delta")),
                    "short_delta": to_float(short.get("delta")),
                }
    return best


def pick_iron_condor(
    puts: List[dict],
    calls: List[dict],
    last: float,
    asof: str,
    earn: dict,
    gates: dict,
) -> Optional[dict]:
    if last is None or last <= 0 or not earn.get("usable"):
        return None
    expiries = sorted({str(r.get("expiry") or "")[:10] for r in puts + calls if r.get("expiry")})
    best = None
    best_cr = -1.0
    for expiry in expiries:
        pc = pick_put_credit([p for p in puts if str(p.get("expiry") or "")[:10] == expiry], last, asof, earn, gates)
        cc = pick_call_credit([c for c in calls if str(c.get("expiry") or "")[:10] == expiry], last, asof, earn, gates)
        if not pc or not cc:
            continue
        if not ticket_legal(
            "iron_condor",
            last,
            gates=gates,
            short_put=pc.get("short_strike"),
            long_put=pc.get("long_strike"),
            short_call=cc.get("short_strike"),
            long_call=cc.get("long_strike"),
        ):
            continue
        net = (pc.get("credit") or 0) + (cc.get("credit") or 0)
        width = max(pc.get("width") or 0, cc.get("width") or 0)
        if net > best_cr:
            best_cr = net
            best = {
                "structure": "iron_condor",
                "put_short": pc.get("short_strike"),
                "put_long": pc.get("long_strike"),
                "call_short": cc.get("short_strike"),
                "call_long": cc.get("long_strike"),
                "short_put": pc.get("short_strike"),
                "long_put": pc.get("long_strike"),
                "short_call": cc.get("short_strike"),
                "long_call": cc.get("long_strike"),
                "credit": net,
                "limit": net,
                "width": width,
                "expiry": expiry,
                "dte": pc.get("dte") or cc.get("dte"),
                "spot": last,
                "otm_s": "P %.1f%% / C %.1f%%"
                % (((pc.get("otm") or 0) * 100), ((cc.get("otm") or 0) * 100)),
                "put_short_delta": pc.get("short_delta"),
                "call_short_delta": cc.get("short_delta"),
                "invalidation": "short put %s / short call %s" % (pc.get("short_strike"), cc.get("short_strike")),
            }
    return best


def stock_idea(snap: dict, last: float, gates: dict) -> Optional[dict]:
    g = gates.get("swing") or {}
    if not snap.get("trend_up"):
        return None
    if tape_chase(snap, float(g.get("chase_atr") or 2.5)):
        return None
    ema20 = to_float(snap.get("ema20"))
    atr = to_float(snap.get("atr"))
    if last is None or last <= 0 or ema20 is None:
        return None
    stop = ema20 - (atr or 0) * float(g.get("stop_atr") or 1.0)
    if last <= stop:
        return None
    if (last - stop) / last > float(g.get("max_stop_frac") or 0.08):
        return None
    risk = last - stop
    return {
        "structure": "stock",
        "spot": last,
        "stop": stop,
        "target": last + 2 * risk,
        "invalidation": stop,
    }


def catalog_for_name(
    puts: List[dict],
    calls: List[dict],
    last: float,
    asof: str,
    earn: dict,
    gates: dict,
    snap: dict,
    cheap: bool,
) -> List[dict]:
    chase_atr = float((gates.get("swing") or {}).get("chase_atr") or 2.5)
    if tape_chase(snap, chase_atr):
        return []
    wanted = []
    if not snap.get("trend_down"):
        wanted.extend(["csp", "put_credit"])
    if snap.get("trend_up"):
        wanted.append("call_debit")
    else:
        wanted.append("call_credit")
    if snap.get("trend_down"):
        wanted.append("put_debit")
    wanted.append("iron_condor")
    rows = []
    for name in wanted:
        if cheap and name in SHORT_PREMIUM:
            continue
        idea = None
        if name == "csp":
            idea = pick_csp(puts, last, asof, earn, gates)
        elif name == "put_credit":
            idea = pick_put_credit(puts, last, asof, earn, gates)
        elif name == "call_debit":
            idea = pick_call_debit(calls, last, asof, earn, gates)
        elif name == "call_credit":
            idea = pick_call_credit(calls, last, asof, earn, gates)
        elif name == "put_debit":
            idea = pick_put_debit(puts, last, asof, earn, gates)
        elif name == "iron_condor":
            idea = pick_iron_condor(puts, calls, last, asof, earn, gates)
        if idea:
            rows.append(idea)
    return rows
