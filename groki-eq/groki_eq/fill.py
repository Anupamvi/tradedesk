"""Equity replay fills. Stop / time-stop only. No profit target."""

import math
from typing import Dict, List, Optional

from groki_eq.config import MONTHLY_PROFIT_TARGET, TIME_STOP_SESSIONS


def pnl_dollars(entry: float, exit_px: float, shares: int) -> float:
    return (exit_px - entry) * float(shares)


def max_drawdown(pnls: List[float]) -> float:
    equity = 0.0
    peak = 0.0
    dd = 0.0
    for value in pnls:
        equity += value
        if equity > peak:
            peak = equity
        dd = min(dd, equity - peak)
    return dd


def summarize(pnls: List[float]) -> Dict[str, float]:
    n = len(pnls)
    if n == 0:
        return {"n": 0, "win": 0, "win_rate": 0.0, "pf": 0.0, "pnl": 0.0, "ev": 0.0, "maxdd": 0.0}
    wins = [x for x in pnls if x > 0]
    losses = [x for x in pnls if x < 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    if gross_loss == 0:
        pf = float("inf") if gross_win > 0 else 0.0
    else:
        pf = gross_win / gross_loss
    total = sum(pnls)
    return {
        "n": n,
        "win": len(wins),
        "win_rate": len(wins) / float(n),
        "pf": pf,
        "pnl": total,
        "ev": total / float(n),
        "maxdd": max_drawdown(pnls),
    }


def tenk_contracts(ev_per_trade: float, trades_per_month: float) -> Optional[int]:
    if ev_per_trade <= 0 or trades_per_month <= 0:
        return None
    monthly = ev_per_trade * trades_per_month
    if monthly <= 0:
        return None
    return int(math.ceil(MONTHLY_PROFIT_TARGET / monthly))


def fmt_pf(pf: float) -> str:
    if pf == float("inf"):
        return "inf"
    return "%.2f" % pf


def fmt_metrics(row: dict) -> str:
    pf = row.get("pf")
    if row.get("pf_inf"):
        pf_s = "inf"
    else:
        pf_s = fmt_pf(float(pf or 0.0))
    return "n=%d win=%d pf=%s $%.0f maxDD=$%.0f" % (
        int(row.get("n") or 0),
        int(row.get("win") or 0),
        pf_s,
        float(row.get("pnl") or 0.0),
        float(row.get("maxdd") or 0.0),
    )


def stop_fill(bar: dict, stop: float) -> Optional[float]:
    open_ = bar.get("open")
    low = bar.get("low")
    if open_ is not None and float(open_) < stop:
        return float(open_)
    if low is not None and float(low) <= stop:
        return float(stop)
    return None


def time_stop_hit(sessions_held: int) -> bool:
    return sessions_held >= TIME_STOP_SESSIONS
