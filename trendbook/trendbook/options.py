"""Optional call-debit overlay on ADD names. Stock is the default expression."""

from __future__ import annotations

from typing import Optional, Sequence

from trendbook.config import OPTION_DTE_MAX, OPTION_DTE_MIN
from trendbook.dates import add_days
from trendbook.num import to_float
from trendbook.orats import cheap_iv
from trendbook.schwab import flatten_chain, option_chain


def earnings_blocks(expiry: str, core: Optional[dict]) -> bool:
    if not core:
        return False
    wks = to_float(core.get("wks_next_ern"))
    if wks is None:
        return True
    days = wks * 7.0
    dte = None
    # expiry vs asof is handled by DTE on the contract
    return days <= (OPTION_DTE_MAX + 5)


def pick_call_debit(legs: Sequence[dict], core: Optional[dict] = None) -> Optional[dict]:
    calls = [r for r in legs if r.get("side") == "call"]
    by_exp = {}
    for row in calls:
        exp = str(row.get("expiry") or "")[:10]
        dte = to_float(row.get("dte"))
        if not exp or dte is None:
            continue
        if dte < OPTION_DTE_MIN or dte > OPTION_DTE_MAX:
            continue
        if earnings_blocks(exp, core) and core and to_float(core.get("wks_next_ern")) is not None:
            if to_float(core.get("wks_next_ern")) * 7 <= dte:
                continue
        by_exp.setdefault(exp, []).append(row)
    best = None
    for exp, rows in by_exp.items():
        longs = []
        for row in rows:
            delta = to_float(row.get("delta"))
            ask = to_float(row.get("ask"))
            strike = to_float(row.get("strike"))
            if delta is None or ask is None or ask <= 0 or strike is None:
                continue
            if not (0.40 <= abs(delta) <= 0.60):
                continue
            longs.append(row)
        for long_leg in longs:
            long_ask = to_float(long_leg.get("ask"))
            long_strike = to_float(long_leg.get("strike"))
            long_delta = to_float(long_leg.get("delta"))
            shorts = []
            for row in rows:
                strike = to_float(row.get("strike"))
                bid = to_float(row.get("bid"))
                delta = to_float(row.get("delta"))
                if strike is None or bid is None or bid <= 0 or delta is None:
                    continue
                if strike <= long_strike:
                    continue
                if not (0.15 <= abs(delta) <= 0.35):
                    continue
                shorts.append(row)
            if not shorts:
                continue
            shorts.sort(key=lambda r: abs(abs(to_float(r.get("delta")) or 0) - 0.25))
            short_leg = shorts[0]
            debit = long_ask - to_float(short_leg.get("bid"))
            width = to_float(short_leg.get("strike")) - long_strike
            if debit is None or debit <= 0 or width is None or width <= 0:
                continue
            max_gain = width - debit
            score = (max_gain / debit if debit else 0, -debit)
            cand = {
                "ok": True,
                "structure": "call_debit",
                "expiry": exp,
                "dte": to_float(long_leg.get("dte")),
                "long_strike": long_strike,
                "short_strike": to_float(short_leg.get("strike")),
                "long_ask": long_ask,
                "short_bid": to_float(short_leg.get("bid")),
                "debit": debit,
                "width": width,
                "max_gain": max_gain,
                "long_delta": long_delta,
                "source": "schwab",
            }
            if best is None or score > best[0]:
                best = (score, cand)
    return best[1] if best else None


def schwab_call_debit(ticker: str, asof: str, core: Optional[dict]) -> Optional[dict]:
    start = add_days(asof, OPTION_DTE_MIN) or asof
    end = add_days(asof, OPTION_DTE_MAX) or asof
    payload = option_chain(ticker, start, end)
    legs = flatten_chain(payload)
    if not legs:
        return None
    return pick_call_debit(legs, core)


def orats_call_debit(rows: Sequence[dict], last: Optional[float], core: Optional[dict]) -> Optional[dict]:
    legs = []
    for row in rows:
        strike = to_float(row.get("strike"))
        dte = to_float(row.get("dte"))
        exp = str(row.get("expirDate") or row.get("expiry") or "")[:10]
        call_ask = to_float(row.get("callAsk") or row.get("call_ask"))
        call_bid = to_float(row.get("callBid") or row.get("call_bid"))
        delta = to_float(row.get("delta") or row.get("callDelta"))
        if strike is None:
            continue
        legs.append(
            {
                "side": "call",
                "expiry": exp,
                "strike": strike,
                "bid": call_bid,
                "ask": call_ask,
                "delta": delta,
                "dte": dte,
            }
        )
    picked = pick_call_debit(legs, core)
    if picked:
        picked["source"] = "orats"
    return picked


def expression_for(click: bool, core: Optional[dict], schwab_ticket: Optional[dict], orats_ticket: Optional[dict]) -> dict:
    if not click:
        return {"kind": "none", "reason": "not_click"}
    ticket = schwab_ticket or orats_ticket
    if ticket and cheap_iv(core):
        out = dict(ticket)
        out["kind"] = "call_debit"
        out["reason"] = "cheap_iv"
        return out
    if ticket and not cheap_iv(core):
        return {"kind": "stock", "reason": "iv_not_cheap", "alt": ticket}
    return {"kind": "stock", "reason": "stock_default"}
