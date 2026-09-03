"""Options structure confidence 0–100. Not a win probability. Never invent POP."""

from __future__ import annotations

from groat.num import to_float


def options_confidence(picked, vol, earnings, snap, setup=None, x_tag=None) -> dict:
    if not picked or not picked.get("ok"):
        return {"conf": None, "label": "n/a", "note": "no options structure", "drivers": []}
    conf = 40
    drivers = []
    debit = to_float(picked.get("target_debit") or picked.get("debit"))
    credit = to_float(picked.get("target_credit") or picked.get("credit"))
    oi = to_float(picked.get("oi"))
    dte = to_float(picked.get("dte"))
    rr = to_float(picked.get("rr"))
    vrp = to_float((vol or {}).get("vrp"))
    inst = str(picked.get("instrument") or "")

    if debit is not None or credit is not None:
        conf += 10
        drivers.append("quoted conservative fill")
    if oi is not None and oi >= 100:
        conf += 8
        drivers.append("OI>=100")
    elif oi is not None and oi >= 50:
        conf += 4
        drivers.append("OI>=50")
    else:
        conf -= 6
        drivers.append("thin OI")

    days = (earnings or {}).get("days")
    if (earnings or {}).get("source") == "exempt":
        conf += 6
        drivers.append("ETF earnings-exempt")
    elif (earnings or {}).get("overlaps_hold"):
        conf -= 20
        drivers.append("earnings inside hold")
    elif days is not None and days >= 45:
        conf += 10
        drivers.append("earnings >=45d")
    elif days is not None and days >= 22:
        conf += 6
        drivers.append("earnings after hold")
    elif not (earnings or {}).get("usable"):
        conf -= 15
        drivers.append("earnings DATA UNAVAILABLE")

    if "credit" in inst:
        if vrp is not None and vrp >= 4:
            conf += 10
            drivers.append("IV rich supports credit")
        elif vrp is not None and vrp < 2:
            conf -= 10
            drivers.append("IV not rich for credit")
    else:
        if vrp is not None and vrp <= -2:
            conf += 10
            drivers.append("IV cheap supports debit")
        elif vrp is not None and vrp >= 8:
            conf -= 8
            drivers.append("buying rich IV")

    if rr is not None and rr >= 1.5:
        conf += 6
        drivers.append("defined R/R>=1.5")
    if dte is not None and 21 <= dte <= 75:
        conf += 4
        drivers.append("%sd DTE" % int(dte))
    if (setup or {}).get("primary") in ("A", "C", "E", "H"):
        conf += 6
        drivers.append("primary setup %s" % setup.get("primary"))
    if (snap or {}).get("stale"):
        conf -= 12
        drivers.append("stale price")

    close = to_float((snap or {}).get("close"))
    long_k = to_float(picked.get("long_strike") or picked.get("strike"))
    if close and close > 0 and long_k is not None:
        if "put" in inst and "credit" not in inst:
            otm = 1.0 - (long_k / close)
        else:
            otm = (long_k / close) - 1.0
        if "put" in inst and "credit" in inst:
            otm = 1.0 - (to_float(picked.get("short_strike")) or long_k) / close
        if otm is not None and otm > 0.08:
            conf -= 16
            drivers.append("long strike >8% OTM")
        elif otm is not None and otm > 0.05:
            conf -= 10
            drivers.append("long strike >5% OTM")
    net_d = abs(to_float(picked.get("delta")) or 0)
    if 0 < net_d < 0.10:
        conf -= 16
        drivers.append("net delta <0.10")
    elif 0 < net_d < 0.16:
        conf -= 8
        drivers.append("net delta <0.16")

    tag = str(x_tag or "").lower()
    ret1 = to_float((snap or {}).get("ret_1"))
    direction = str((setup or {}).get("direction") or "")
    if tag == "informed":
        if direction == "bullish" and ret1 is not None and ret1 <= -0.03:
            drivers.append("X Informed, no bonus (red day)")
        else:
            conf += 5
            drivers.append("X Informed")
    elif tag == "crowded":
        conf -= 10
        drivers.append("X Crowded")
    elif tag in ("quiet",):
        drivers.append("X Quiet")
    else:
        drivers.append("X DATA UNAVAILABLE")

    conf = max(15, min(85, int(round(conf))))
    if conf >= 70:
        label = "high"
    elif conf >= 55:
        label = "medium"
    else:
        label = "low"
    return {
        "conf": conf,
        "label": label,
        "note": "structure confidence, not P(win)",
        "drivers": drivers,
    }
