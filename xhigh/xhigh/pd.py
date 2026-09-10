"""PD ranking for CLICK rows. Does not change conf, gates, or CLICK membership."""

from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence

from xhigh.config import CONTRACT_MULTIPLIER
from xhigh.num import fmt, to_float
from xhigh.score import DEFINED_CREDIT, DEBIT

ACCOUNT_DOLLARS = 50000.0
RISK_PCT = 0.01
R_CAP = 3.0
QUOTE_MAX_AGE_SEC = 120.0
SPREAD_TIGHT = 0.08
SPREAD_WIDE = 0.15
PD_NOTE = "PD sort only — conf unchanged."


def risk_budget() -> float:
    return RISK_PCT * ACCOUNT_DOLLARS


def quote_age_seconds(quote_time_ms=None, quote_date=None, explicit=None, now=None) -> Optional[float]:
    age = to_float(explicit)
    if age is not None:
        return age
    now = time.time() if now is None else now
    ms = to_float(quote_time_ms)
    if ms is not None and ms > 0:
        stamp = ms / 1000.0 if ms > 1e11 else ms
        return max(0.0, now - stamp)
    raw = str(quote_date or "").strip()
    if not raw:
        return None
    for spec in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%d %H:%M ET", "%Y-%m-%d %H:%M UTC"):
        try:
            dt = datetime.strptime(raw[: len(spec) + 8], spec)
            if spec.endswith("Z") or "UTC" in spec:
                dt = dt.replace(tzinfo=timezone.utc)
            return max(0.0, now - dt.timestamp())
        except ValueError:
            continue
    return None


def liquidity_factor(size, n: int, spread_frac) -> Optional[float]:
    if n < 1:
        return 0.0
    sp = to_float(spread_frac)
    if sp is None:
        return None
    if sp > SPREAD_WIDE:
        return 0.0
    sz = to_float(size)
    if sz is None:
        sz = float(n)
    if sz >= n and sp <= SPREAD_TIGHT:
        return 1.0
    if sz >= 1:
        return 0.5
    return 0.0


def compute_pd(
    *,
    max_loss: Optional[float],
    planned_reward: Optional[float],
    planned_risk: Optional[float],
    liquidity_lots: Optional[float],
    quote_age_sec: Optional[float],
    spread_frac: Optional[float],
    size: Optional[float] = None,
) -> Dict[str, Any]:
    out = {
        "pd": None,
        "N": None,
        "R_cons": None,
        "L": None,
        "reason": "DATA UNAVAILABLE",
        "risk_budget": risk_budget(),
    }
    ml = to_float(max_loss)
    reward = to_float(planned_reward)
    risk = to_float(planned_risk)
    liq = to_float(liquidity_lots)
    age = to_float(quote_age_sec)
    if ml is None or ml <= 0 or reward is None or risk is None or risk <= 0 or liq is None or age is None:
        return out
    r_cons = min(reward / risk, R_CAP)
    out["R_cons"] = r_cons
    n = int(math.floor(min(risk_budget() / ml, liq)))
    if n < 0:
        n = 0
    out["N"] = n
    l = liquidity_factor(size if size is not None else n, n, spread_frac)
    out["L"] = l
    if age > QUOTE_MAX_AGE_SEC:
        return out
    if n < 1:
        out["reason"] = "N=0"
        return out
    if l is None:
        return out
    out["pd"] = n * r_cons * l
    out["reason"] = ""
    return out


def sort_by_pd(rows: Sequence[dict], *, tie: Optional[Callable] = None) -> List[dict]:
    def key(row):
        pd = to_float(row.get("pd"))
        extra = tie(row) if tie else ()
        if not isinstance(extra, tuple):
            extra = (extra,)
        return (0 if pd is not None else 1, -(pd or 0.0)) + extra

    return sorted(list(rows), key=key)


def stamp_pd(row: dict, pack: dict) -> dict:
    row["pd"] = pack.get("pd")
    row["pd_n"] = pack.get("N")
    row["r_cons"] = pack.get("R_cons")
    row["pd_l"] = pack.get("L")
    row["pd_reason"] = pack.get("reason") or ""
    return row


def pd_cells(row: dict) -> Dict[str, str]:
    pd = row.get("pd")
    if pd is None:
        pd_s = row.get("pd_reason") or "DATA UNAVAILABLE"
    else:
        pd_s = fmt(pd, 2)
    n = row.get("pd_n")
    r = row.get("r_cons")
    l = row.get("pd_l")
    return {
        "pd_s": pd_s,
        "n_s": "—" if n is None else str(int(n)),
        "r_cons_s": "—" if r is None else fmt(r, 2),
        "l_s": "—" if l is None else fmt(l, 1),
    }


def economics(row: dict) -> tuple:
    s = row.get("structure")
    mult = float(CONTRACT_MULTIPLIER)
    if s in DEBIT:
        debit = to_float(row.get("debit"))
        gain = to_float(row.get("max_gain"))
        if debit is None:
            return None, None, None
        return debit * mult, (gain * mult if gain is not None else None), debit * mult
    if s in DEFINED_CREDIT:
        credit = to_float(row.get("credit"))
        width = to_float(row.get("width"))
        if credit is None or width is None:
            return None, None, None
        ml = max(0.0, width - credit) * mult
        return ml, credit * mult, ml
    if s == "csp":
        strike = to_float(row.get("strike"))
        credit = to_float(row.get("credit"))
        if strike is None or credit is None:
            return None, None, None
        ml = max(0.0, strike - credit) * mult
        return ml, credit * mult, ml
    if s == "stock":
        last = to_float(row.get("last") or row.get("spot"))
        stop = to_float(row.get("stop") or row.get("invalidation"))
        target = to_float(row.get("target"))
        if last is None or stop is None:
            return None, None, None
        risk = abs(last - stop)
        reward = abs(target - last) if target is not None else None
        return risk, reward, risk
    return None, None, None


def attach_trade_pd(row: dict, now=None) -> dict:
    if not isinstance(row, dict) or row.get("action") != "CLICK":
        return row
    ml, reward, risk = economics(row)
    if row.get("max_loss_1lot") is not None:
        ml = to_float(row.get("max_loss_1lot"))
    if row.get("planned_reward") is not None:
        reward = to_float(row.get("planned_reward"))
    if row.get("planned_risk") is not None:
        risk = to_float(row.get("planned_risk"))
    age = quote_age_seconds(
        quote_time_ms=row.get("quote_time_ms"),
        quote_date=row.get("quote_date") or row.get("quote_asof"),
        explicit=row.get("quote_age_sec"),
        now=now,
    )
    pack = compute_pd(
        max_loss=ml,
        planned_reward=reward,
        planned_risk=risk,
        liquidity_lots=row.get("liquidity_lots") if row.get("liquidity_lots") is not None else row.get("oi"),
        quote_age_sec=age,
        spread_frac=row.get("spread_frac"),
        size=row.get("pd_size") if row.get("pd_size") is not None else row.get("contracts"),
    )
    stamp_pd(row, pack)
    row.update(pd_cells(row))
    return row
