"""PD ranking for Expert/TRADE rows. Does not change Conf, gates, or sleeves."""

from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence

ACCOUNT_DOLLARS = 50000.0
RISK_PCT = 0.01
R_CAP = 3.0
QUOTE_MAX_AGE_SEC = 120.0
SPREAD_TIGHT = 0.08
SPREAD_WIDE = 0.15
PD_NOTE = "PD sort only — conf unchanged."
CONTRACT_MULTIPLIER = 100.0


def to_float(value) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def fmt(value, digits=2) -> str:
    number = to_float(value)
    if number is None:
        return "DATA UNAVAILABLE"
    return ("%." + str(digits) + "f") % number


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


def _leg_spread(leg: Optional[dict]) -> Optional[float]:
    if not isinstance(leg, dict):
        return None
    bid = to_float(leg.get("bid"))
    ask = to_float(leg.get("ask"))
    if bid is None or ask is None or ask <= 0:
        return None
    mid = (bid + ask) / 2.0
    if mid <= 0:
        return None
    return (ask - bid) / mid


def _leg_liq(leg: Optional[dict]) -> List[float]:
    out = []
    if not isinstance(leg, dict):
        return out
    for key in ("bidSize", "askSize", "bid_size", "ask_size", "oi", "openInterest"):
        v = to_float(leg.get(key))
        if v is not None:
            out.append(v)
    return out


def attach_structure_pd(row: dict, now=None) -> dict:
    """Stamp PD on a geometry-pass structure. Caller decides TRADE vs review."""
    if not isinstance(row, dict):
        return row
    pricing = row.get("pricing") if isinstance(row.get("pricing"), dict) else {}
    kind = str(pricing.get("kind") or row.get("kind") or "")
    net = to_float(pricing.get("net") or row.get("net"))
    width = to_float(pricing.get("width") or row.get("width"))
    ml = to_float(pricing.get("max_loss_1lot") or row.get("max_loss_1lot") or row.get("max_loss"))
    reward = to_float(row.get("planned_reward") or pricing.get("max_profit_1lot"))
    if ml is None and net is not None and width is not None:
        if kind == "credit":
            ml = max(0.0, width - net) * CONTRACT_MULTIPLIER
        elif kind == "debit":
            ml = net * CONTRACT_MULTIPLIER
    if reward is None and net is not None and width is not None:
        if kind == "credit":
            reward = net * CONTRACT_MULTIPLIER
        elif kind == "debit":
            reward = max(0.0, width - net) * CONTRACT_MULTIPLIER
    short = row.get("short_leg") if isinstance(row.get("short_leg"), dict) else (
        row.get("short") if isinstance(row.get("short"), dict) else None
    )
    long = row.get("long_leg") if isinstance(row.get("long_leg"), dict) else (
        row.get("long") if isinstance(row.get("long"), dict) else None
    )
    fracs = [f for f in (_leg_spread(short), _leg_spread(long), to_float(row.get("spread_frac"))) if f is not None]
    lots = []
    lots.extend(_leg_liq(short))
    lots.extend(_leg_liq(long))
    if row.get("liquidity_lots") is not None:
        lots.append(float(row["liquidity_lots"]))
    age = quote_age_seconds(
        quote_time_ms=row.get("quote_time_ms"),
        quote_date=row.get("quote_date") or row.get("quote_time"),
        explicit=row.get("quote_age_sec"),
        now=now,
    )
    pack = compute_pd(
        max_loss=ml,
        planned_reward=reward,
        planned_risk=to_float(row.get("planned_risk")) if row.get("planned_risk") is not None else ml,
        liquidity_lots=min(lots) if lots else None,
        quote_age_sec=age,
        spread_frac=max(fracs) if fracs else None,
        size=row.get("pd_size") if row.get("pd_size") is not None else row.get("rec_lots") or row.get("contracts"),
    )
    stamp_pd(row, pack)
    row.update(pd_cells(row))
    return row
