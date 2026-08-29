"""Wheelo ticket confidence 0-85. Structure + research completeness. Not P(win)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from wheelo.dates import days_until, parse_any_date, usable_date
from wheelo.num import to_float


def _otm_pct(spot: Optional[float], strike: Optional[float]) -> Optional[float]:
    if spot is None or strike is None or spot <= 0:
        return None
    return (spot - strike) / spot


def _earn_days(core: dict, asof: str) -> tuple:
    """Return (days, status) status = known | soon | unknown.

    Delayed ORATS nextErn / daysToNextErn are often placeholders (0000-00-00 / 0).
    wksNextErn is the usable delayed calendar (~7d per week).
    """
    nxt = usable_date(core.get("next_ern")) or parse_any_date(core.get("next_ern"))
    if nxt:
        d = days_until(asof, nxt)
        if d is not None:
            if d <= 7:
                return d, "soon"
            return d, "known"
    wks = to_float(core.get("wks_next_ern"))
    if wks is not None and 0 < wks <= 26:
        d = max(1, int(round(wks * 7)))
        return d, "soon" if d <= 7 else "known"
    days = core.get("days_to_ern")
    if days is not None:
        try:
            d = int(days)
        except (TypeError, ValueError):
            d = None
        if d is not None and d > 0:
            return d, "soon" if d <= 7 else "known"
    return None, "unknown"


def ticket_confidence(
    core: dict,
    premium: dict,
    asof: str,
    x_status: str = "",
    live_validated: bool = False,
) -> Dict[str, Any]:
    """Score a CSP. label is NO_TRADE | WATCH | TRADE. conf is not a win probability."""
    drivers: List[str] = []
    hard: List[str] = []
    conf = 40

    spot = to_float(core.get("px"))
    strike = to_float(premium.get("csp_strike"))
    bid = to_float(premium.get("csp_bid") or premium.get("csp_premium"))
    dte = to_float(premium.get("dte"))
    spread = to_float(premium.get("spread_pct"))
    ivr = to_float(core.get("iv_pctile_1y") or premium.get("iv_rank"))
    iv_hv = to_float(core.get("iv_hv"))
    iv30 = to_float(core.get("iv30"))

    if not bid or bid <= 0 or not strike:
        hard.append("no_bid")
    credit_pct = (bid / strike) if bid and strike else None
    if credit_pct is not None and credit_pct < 0.015:
        hard.append("credit_too_small")
        drivers.append("credit %.2f%% of strike" % (100.0 * credit_pct))
    elif credit_pct is not None and credit_pct >= 0.025:
        conf += 12
        drivers.append("credit %.2f%% of strike" % (100.0 * credit_pct))
    elif credit_pct is not None:
        conf += 6
        drivers.append("credit %.2f%% of strike" % (100.0 * credit_pct))

    otm = _otm_pct(spot, strike)
    if otm is not None and otm < 0.02:
        hard.append("atm")
        drivers.append("ATM")
    elif otm is not None and 0.05 <= otm <= 0.12:
        conf += 8
        drivers.append("OTM %.0f%%" % (100.0 * otm))
    elif otm is not None and 0.02 <= otm < 0.05:
        conf += 4
        drivers.append("OTM %.0f%%" % (100.0 * otm))
    elif otm is not None and otm > 0.15:
        conf -= 6
        drivers.append("too far OTM %.0f%%" % (100.0 * otm))

    if ivr is not None and ivr >= 60:
        conf += 10
        drivers.append("IVR %.0f" % ivr)
    elif ivr is not None and ivr >= 40:
        conf += 5
        drivers.append("IVR %.0f" % ivr)
    elif ivr is not None and ivr < 20:
        conf -= 8
        drivers.append("IVR cheap %.0f" % ivr)

    if iv_hv is not None and iv_hv >= 1.15:
        conf += 10
        drivers.append("IV/HV %.2f rich" % iv_hv)
    elif iv_hv is not None and iv_hv >= 1.0:
        conf += 4
        drivers.append("IV/HV %.2f" % iv_hv)
    elif iv_hv is not None and iv_hv < 0.90:
        conf -= 10
        drivers.append("IV/HV %.2f cheap" % iv_hv)
        if ivr is None or ivr < 50:
            hard.append("cheap_vol")

    if spread is not None and spread <= 0.08:
        conf += 6
        drivers.append("spread %.0f%%" % (100.0 * spread))
    elif spread is not None and spread > 0.15:
        conf -= 8
        drivers.append("wide spread %.0f%%" % (100.0 * spread))

    if dte is not None and 21 <= dte <= 45:
        conf += 4
        drivers.append("%sd DTE" % int(dte))

    earn_d, earn_st = _earn_days(core, asof)
    if earn_st == "soon":
        hard.append("earnings")
        drivers.append("earnings in %sd" % (earn_d if earn_d is not None else "?"))
        conf -= 25
    elif earn_st == "unknown":
        hard.append("earnings_unknown")
        drivers.append("earnings DATA UNAVAILABLE")
        conf -= 20
    elif earn_d is not None and dte is not None and earn_d <= int(dte) + 3:
        # Print sits inside (or within 3d of) expiry. Week-granularity dates need the buffer.
        hard.append("earnings_in_dte")
        drivers.append("earnings in %sd crosses %sd DTE" % (earn_d, int(dte)))
        conf -= 25
    elif earn_d is not None and earn_d >= 21:
        conf += 10
        drivers.append("earnings %sd out" % earn_d)
    elif earn_d is not None:
        drivers.append("earnings %sd out" % earn_d)

    tag = str(x_status or "").lower()
    if "crowd" in tag:
        conf -= 8
        drivers.append("X crowded")
    elif tag == "informed":
        conf += 4
        drivers.append("X informed")
    elif tag in ("data unavailable", "", "unknown"):
        drivers.append("X DATA UNAVAILABLE")

    if live_validated:
        conf += 3
        drivers.append("Schwab live mark")

    if iv30 is None:
        drivers.append("iv30 DATA UNAVAILABLE")
        conf -= 8

    conf = max(0, min(85, int(round(conf))))
    if earn_st == "unknown":
        conf = min(conf, 45)
        drivers.append("cap 45: no earnings date")

    if hard:
        label = "NO_TRADE"
        conf = min(conf, 40)
    elif conf >= 63:
        label = "TRADE"
    elif conf >= 48:
        label = "WATCH"
    else:
        label = "NO_TRADE"

    return {
        "conf": conf,
        "label": label,
        "drivers": drivers,
        "hard": hard,
        "credit_pct": credit_pct,
        "otm_pct": otm,
        "earn_days": earn_d,
        "earn_status": earn_st,
        "note": "conf is structure/research quality 0-85, not P(win)",
    }
