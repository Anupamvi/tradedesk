"""Delta-implied POP, research conf 0-85, defined-risk EV proxy. Not P(win)."""

from __future__ import annotations

from typing import Optional

from xhigh.config import CONTRACT_MULTIPLIER
from xhigh.num import to_float


WHEEL = ("csp",)
DEFINED_CREDIT = ("put_credit", "call_credit", "iron_condor")
CREDIT = WHEEL + DEFINED_CREDIT
DEBIT = ("call_debit", "put_debit")


def csp_annualized(idea: dict) -> Optional[float]:
    credit = to_float(idea.get("credit"))
    strike = to_float(idea.get("strike"))
    dte = to_float(idea.get("dte"))
    if credit is None or strike is None or strike <= 0 or dte is None or dte <= 0:
        return None
    return (credit / strike) * (365.0 / dte)


def credit_over_width(idea: dict) -> Optional[float]:
    credit = to_float(idea.get("credit"))
    width = to_float(idea.get("width"))
    if credit is None or width is None or width <= 0:
        return None
    return credit / width


def credit_to_risk(idea: dict) -> Optional[float]:
    credit = to_float(idea.get("credit"))
    width = to_float(idea.get("width"))
    if credit is None or width is None or width <= credit:
        return None
    return credit / (width - credit)


def rr_line(idea: dict) -> str:
    s = idea.get("structure")
    if s in DEBIT:
        debit = to_float(idea.get("debit"))
        gain = to_float(idea.get("max_gain"))
        if debit is None or debit <= 0 or gain is None:
            return "DATA UNAVAILABLE"
        return "%.1f : 1" % (gain / debit)
    cr = credit_to_risk(idea)
    if cr is None:
        return "DATA UNAVAILABLE"
    if cr <= 0:
        return "DATA UNAVAILABLE"
    return "1 : %.1f" % (1.0 / cr)


def short_abs_delta(idea: dict) -> Optional[float]:
    d = to_float(idea.get("delta") or idea.get("short_delta") or idea.get("put_short_delta"))
    if d is None:
        return None
    return abs(d)


def pop_delta(idea: dict) -> Optional[float]:
    structure = idea.get("structure")
    if structure == "csp":
        d = to_float(idea.get("delta") or idea.get("short_delta"))
        if d is None:
            return None
        return max(0.0, min(1.0, 1.0 - abs(d)))
    if structure in ("put_credit", "call_credit"):
        d = to_float(idea.get("short_delta"))
        if d is None:
            return None
        return max(0.0, min(1.0, 1.0 - abs(d)))
    if structure == "iron_condor":
        pd = to_float(idea.get("put_short_delta"))
        cd = to_float(idea.get("call_short_delta"))
        if pd is None or cd is None:
            return None
        return max(0.0, min(1.0, 1.0 - abs(pd) - abs(cd)))
    if structure in DEBIT:
        ld = to_float(idea.get("long_delta"))
        if ld is None:
            return None
        return max(0.0, min(1.0, abs(ld)))
    return None


def max_loss_per_share(idea: dict) -> Optional[float]:
    structure = idea.get("structure")
    credit = to_float(idea.get("credit"))
    debit = to_float(idea.get("debit"))
    if structure == "csp":
        return None
    if structure in ("put_credit", "call_credit", "iron_condor"):
        width = to_float(idea.get("width"))
        if width is None or credit is None:
            return None
        return max(0.0, width - credit)
    if structure in DEBIT:
        return debit
    return None


def ev_proxy(idea: dict, pop: Optional[float]) -> Optional[float]:
    if pop is None:
        return None
    structure = idea.get("structure")
    if structure == "csp":
        ann = csp_annualized(idea)
        if ann is None:
            return None
        return round(ann * 100.0, 2)
    ml = max_loss_per_share(idea)
    if ml is None:
        return None
    if structure in DEFINED_CREDIT:
        credit = to_float(idea.get("credit"))
        if credit is None:
            return None
        return CONTRACT_MULTIPLIER * (credit * pop - ml * (1.0 - pop))
    if structure in DEBIT:
        debit = to_float(idea.get("debit"))
        max_gain = to_float(idea.get("max_gain"))
        if debit is None or max_gain is None:
            return None
        return CONTRACT_MULTIPLIER * (max_gain * pop - debit * (1.0 - pop))
    return None


def confidence(idea: dict, earn: dict, gates: dict) -> int:
    cap = int((gates.get("score") or {}).get("conf_max") or 85)
    conf = 50
    src = str(earn.get("source") or "")
    if "nasdaq" in src:
        conf += 15
    elif "wksNextErn" in src:
        conf -= 5
    elif "cadence" in src:
        conf -= 8
    elif not earn.get("usable"):
        conf -= 20
    if pop_delta(idea) is not None:
        conf += 10
    if idea.get("structure") in CREDIT and to_float(idea.get("credit")):
        conf += 5
    if idea.get("structure") in DEBIT and to_float(idea.get("debit")):
        conf += 5
    tag = str(idea.get("x_tag") or "")
    if tag == "Crowded":
        conf -= 15
    elif tag == "Informed":
        conf += 5
    elif tag == "Quiet":
        conf += 2
    extra = to_float(idea.get("conf_delta"))
    if extra is not None:
        conf += int(extra)
    return max(0, min(cap, int(conf)))


def cheap_vol(core: dict, gates: dict) -> bool:
    g = gates.get("orats") or {}
    iv_hv = to_float(core.get("iv_hv"))
    ivp = to_float(core.get("iv_pctile_1y"))
    if iv_hv is not None and iv_hv > 5:
        return False
    cheap_hv = float(g.get("cheap_iv_hv") or 0.90)
    cheap_pct = float(g.get("cheap_iv_pctile") or 40)
    return bool(iv_hv is not None and ivp is not None and iv_hv < cheap_hv and ivp < cheap_pct)
