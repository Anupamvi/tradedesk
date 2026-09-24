"""Relative strength vs SPY, vs sector ETF, IBD mix, Mansfield, universe percentile."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from trendbook.bars import bars_through, closes, ret, sma, to_weekly
from trendbook.num import pct_change, to_float


def relative_return(stock: Sequence[float], bench: Sequence[float], n: int) -> Optional[float]:
    a = ret(stock, n)
    b = ret(bench, n)
    if a is None or b is None:
        return None
    return a - b


def _paired_returns(stock: Sequence[float], bench: Sequence[float], n: int):
    m = min(len(stock), len(bench), n + 1)
    if m < 40:
        return None
    s = stock[-m:]
    b = bench[-m:]
    rs, rb = [], []
    for i in range(1, m):
        if s[i - 1] and b[i - 1] and s[i - 1] != 0 and b[i - 1] != 0:
            rs.append(s[i] / s[i - 1] - 1.0)
            rb.append(b[i] / b[i - 1] - 1.0)
    if len(rs) < 30:
        return None
    return rs, rb


def ols_beta(stock: Sequence[float], bench: Sequence[float], n: int = 126) -> Optional[float]:
    paired = _paired_returns(stock, bench, n)
    if not paired:
        return None
    rs, rb = paired
    mb = sum(rb) / float(len(rb))
    ms = sum(rs) / float(len(rs))
    varb = sum((x - mb) ** 2 for x in rb)
    if varb <= 1e-18:
        return None
    cov = sum((x - ms) * (y - mb) for x, y in zip(rs, rb))
    return cov / varb


def residual_return(stock: Sequence[float], bench: Sequence[float], n: int) -> Optional[float]:
    """Window return minus beta * benchmark return. Falls back to arithmetic excess."""
    a = ret(stock, n)
    b = ret(bench, n)
    if a is None or b is None:
        return None
    beta = ols_beta(stock, bench, n)
    if beta is None:
        return a - b
    return a - beta * b


def ibd_score(stock: Sequence[float]) -> Optional[float]:
    r3 = ret(stock, 63)
    r6 = ret(stock, 126)
    r9 = ret(stock, 189)
    r12 = ret(stock, 252)
    parts = [(0.40, r3), (0.20, r6), (0.20, r9), (0.20, r12)]
    if any(v is None for _, v in parts):
        # Fall back to whatever windows exist, renormalized.
        usable = [(w, v) for w, v in parts if v is not None]
        if not usable:
            return None
        total_w = sum(w for w, _ in usable)
        return sum(w * v for w, v in usable) / total_w
    return 0.40 * r3 + 0.20 * r6 + 0.20 * r9 + 0.20 * r12


def mansfield(stock_weekly: Sequence[dict], bench_weekly: Sequence[dict], n: int = 52) -> Optional[float]:
    """(stock/bench) / SMA_n(stock/bench) - 1. Uses min(n, available) down to 30."""
    by_date = {}
    for row in bench_weekly:
        day = str(row.get("date") or "")[:10]
        px = to_float(row.get("close"))
        if day and px:
            by_date[day] = px
    ratios = []
    for row in stock_weekly:
        day = str(row.get("date") or "")[:10]
        px = to_float(row.get("close"))
        bench = by_date.get(day)
        if px is None or bench is None or bench == 0:
            continue
        ratios.append(px / bench)
    window = n if len(ratios) >= n else (30 if len(ratios) >= 30 else None)
    if window is None:
        return None
    mean = sma(ratios, window)
    if mean is None or mean == 0:
        return None
    return ratios[-1] / mean - 1.0


def mansfield_rising(stock_weekly: Sequence[dict], bench_weekly: Sequence[dict], n: int = 52) -> Optional[bool]:
    if len(stock_weekly) < 5:
        return None
    now = mansfield(stock_weekly, bench_weekly, n)
    prev = mansfield(stock_weekly[:-4], bench_weekly, n)
    if now is None or prev is None:
        return None
    return now > prev


def percentile_rank(values: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    pairs = [(k, v) for k, v in values.items() if v is not None]
    if not pairs:
        return {k: None for k in values}
    pairs.sort(key=lambda kv: kv[1])
    n = len(pairs)
    out = {k: None for k in values}
    for i, (k, _) in enumerate(pairs):
        out[k] = 100.0 * i / max(n - 1, 1) if n > 1 else 100.0
    return out


def rs_bundle(
    ticker: str,
    daily: Sequence[dict],
    spy_daily: Sequence[dict],
    sector_daily: Optional[Sequence[dict]],
    asof: str,
) -> dict:
    stock = closes(bars_through(daily, asof))
    spy = closes(bars_through(spy_daily, asof))
    sector = closes(bars_through(sector_daily or [], asof)) if sector_daily else []
    weekly = to_weekly(daily, asof)
    spy_w = to_weekly(spy_daily, asof)
    sector_w = to_weekly(sector_daily or [], asof) if sector_daily else []
    return {
        "ticker": ticker,
        "rs_63": relative_return(stock, spy, 63),
        "rs_126": relative_return(stock, spy, 126),
        "rs_252": relative_return(stock, spy, 252),
        "rs_sector_126": relative_return(stock, sector, 126) if sector else None,
        "beta_126": ols_beta(stock, spy, 126),
        "residual_63": residual_return(stock, spy, 63),
        "residual_126": residual_return(stock, spy, 126),
        "residual_sector_126": residual_return(stock, sector, 126) if sector else None,
        "ibd": ibd_score(stock),
        "mansfield_spy": mansfield(weekly, spy_w),
        "mansfield_spy_rising": mansfield_rising(weekly, spy_w),
        "mansfield_sector": mansfield(weekly, sector_w) if sector_w else None,
        "ret_126": ret(stock, 126),
        "spy_ret_126": ret(spy, 126),
    }


def rs_ok(bundle: dict, pctile: Optional[float] = None, universe_n: int = 0, min_pctile: float = 70.0) -> bool:
    """Beating SPY. Percentile is sort order only — never a kill switch."""
    mans = to_float(bundle.get("mansfield_spy"))
    if mans is not None:
        return mans > 0
    rs126 = to_float(bundle.get("rs_126"))
    return rs126 is not None and rs126 > 0
