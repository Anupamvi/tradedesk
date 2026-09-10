"""PD ranking for TRADE rows. Does not change conf, gates, or TRADE membership."""

from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence

from groat.num import fmt, to_float

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


def _picked(row: dict) -> dict:
    p = row.get("picked")
    return p if isinstance(p, dict) else {}


def attach_trade_pd(row: dict, now=None) -> dict:
    """Stamp PD on a row that is already TRADE. No-op for other actions."""
    if not isinstance(row, dict) or row.get("action") != "TRADE":
        return row
    picked = _picked(row)
    src = picked if picked else row
    age = quote_age_seconds(
        quote_time_ms=src.get("quote_time_ms") or row.get("quote_time_ms"),
        quote_date=src.get("fill_asof") or src.get("quote_asof") or row.get("quote_asof"),
        explicit=row.get("quote_age_sec") or src.get("quote_age_sec"),
        now=now,
    )
    max_loss = to_float(src.get("max_loss_1lot") or row.get("max_loss_1lot"))
    reward = to_float(src.get("planned_reward") or row.get("planned_reward"))
    risk = to_float(src.get("planned_risk") or row.get("planned_risk"))
    if max_loss is None and row.get("choice") == "STOCK":
        entry = to_float(src.get("entry"))
        stop = to_float(src.get("stop"))
        target = to_float(src.get("target"))
        if entry is not None and stop is not None:
            max_loss = abs(entry - stop)
            risk = max_loss if risk is None else risk
        if reward is None and entry is not None and target is not None:
            reward = abs(target - entry)
    pack = compute_pd(
        max_loss=max_loss,
        planned_reward=reward,
        planned_risk=risk if risk is not None else max_loss,
        liquidity_lots=src.get("liquidity_lots") if src.get("liquidity_lots") is not None else row.get("liquidity_lots"),
        quote_age_sec=age,
        spread_frac=src.get("spread_frac") if src.get("spread_frac") is not None else row.get("spread_frac"),
        size=src.get("pd_size") if src.get("pd_size") is not None else (src.get("contracts") or src.get("shares") or row.get("contracts")),
    )
    stamp_pd(row, pack)
    row.update(pd_cells(row))
    return row
