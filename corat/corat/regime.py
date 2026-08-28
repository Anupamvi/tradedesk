"""Market regime and sector-rotation classification."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

from corat.constants import DATA_UNAVAILABLE
from corat.models import TechnicalSnapshot


def _above(value: float, level: Optional[float]) -> Optional[bool]:
    return None if level is None else value > level


def classify_market(
    snapshots: Mapping[str, TechnicalSnapshot],
    candidate_snapshots: Sequence[TechnicalSnapshot],
) -> Dict[str, Any]:
    spy = snapshots.get("SPY")
    qqq = snapshots.get("QQQ")
    iwm = snapshots.get("IWM")
    if spy is None or qqq is None:
        return {
            "label": DATA_UNAVAILABLE,
            "strategy_bias": "NO TRADE until benchmark history is available",
            "breadth_above_20": None,
            "breadth_above_50": None,
            "reasoning": ["SPY or QQQ structure unavailable"],
            "proxies": {},
        }
    eligible20 = [item for item in candidate_snapshots if item.ema20 is not None]
    eligible50 = [item for item in candidate_snapshots if item.sma50 is not None]
    breadth_sample_size = min(len(eligible20), len(eligible50))
    breadth_reliable = breadth_sample_size >= 20
    breadth20 = (
        sum(1 for item in eligible20 if item.price > float(item.ema20)) / float(len(eligible20))
        if eligible20 and breadth_reliable else None
    )
    breadth50 = (
        sum(1 for item in eligible50 if item.price > float(item.sma50)) / float(len(eligible50))
        if eligible50 and breadth_reliable else None
    )
    spy_above20 = bool(spy.ema20 is not None and spy.price > spy.ema20)
    spy_above50 = bool(spy.sma50 is not None and spy.price > spy.sma50)
    spy_above200 = bool(spy.sma200 is not None and spy.price > spy.sma200)
    qqq_above20 = bool(qqq.ema20 is not None and qqq.price > qqq.ema20)
    qqq_above50 = bool(qqq.sma50 is not None and qqq.price > qqq.sma50)
    label = "RANGE / CHOP"
    bias = "Favor confirmed triggers, smaller size, and defined risk; avoid extension."
    if (
        spy_above20 and spy_above50 and qqq_above20 and qqq_above50
        and (spy.return_20d or 0) > 0.02 and (breadth50 or 0) >= 0.60
    ):
        label = "STRONG RISK-ON TREND"
        bias = "Favor leader pullbacks, confirmed breakouts, and post-event drift."
    elif spy_above50 and qqq_above50 and (breadth50 is None or breadth50 >= 0.45):
        label = "WEAK RISK-ON"
        bias = "Favor selective relative-strength leaders; demand clean invalidation." + (
            " Breadth sample is insufficient for a stronger regime label." if breadth50 is None else ""
        )
    elif spy_above200 and qqq_above50 and (breadth20 or 0) < 0.50:
        label = "ROTATION"
        bias = "Concentrate on emerging sector leadership and avoid weakening groups."
    elif (not spy_above50) and (not qqq_above50) and (breadth50 or 1) < 0.40:
        label = "RISK-OFF"
        bias = "Prefer cash, failed-breakout shorts, or tightly defined bearish structures."
    if (
        (spy.return_5d or 0) < -0.06
        and (qqq.return_5d or 0) < -0.07
        and (breadth20 or 1) < 0.25
    ):
        label = "HIGH-VOLATILITY LIQUIDATION"
        bias = "Do not catch falling knives; wait for stabilization and source-backed reversals."
    elif (
        (spy.return_5d or 0) > 0.04
        and not spy_above50
        and (breadth20 or 0) > 0.55
    ):
        label = "POST-CORRECTION RECOVERY / SHORT-COVERING"
        bias = "Prefer reclaim-and-retest setups; do not assume the primary trend has recovered."
    proxies: Dict[str, Any] = {}
    for ticker, label_name in (("VIX", "volatility"), ("TLT", "long-duration Treasury proxy"), ("UUP", "DXY proxy"), ("HYG", "credit-risk proxy"), ("GLD", "gold proxy")):
        item = snapshots.get(ticker)
        proxies[ticker] = {
            "label": label_name,
            "price": item.price if item else None,
            "return_5d": item.return_5d if item else None,
            "return_20d": item.return_20d if item else None,
            "status": "AVAILABLE" if item else DATA_UNAVAILABLE,
        }
    reasoning = [
        "SPY above 20/50/200: {}/{}/{}".format(spy_above20, spy_above50, spy_above200),
        "QQQ above 20/50: {}/{}".format(qqq_above20, qqq_above50),
        "Breadth above 20/50: {}/{}".format(
            "{:.1%}".format(breadth20) if breadth20 is not None else DATA_UNAVAILABLE,
            "{:.1%}".format(breadth50) if breadth50 is not None else DATA_UNAVAILABLE,
        ),
        "Breadth sample: {} securities; reliable threshold met: {}".format(breadth_sample_size, breadth_reliable),
        "IWM 20-session return: {}".format(
            "{:.1%}".format(iwm.return_20d) if iwm and iwm.return_20d is not None else DATA_UNAVAILABLE
        ),
    ]
    return {
        "label": label,
        "strategy_bias": bias,
        "breadth_above_20": breadth20,
        "breadth_above_50": breadth50,
        "breadth_sample_size": breadth_sample_size,
        "breadth_reliable": breadth_reliable,
        "reasoning": reasoning,
        "proxies": proxies,
    }


def rank_sectors(
    snapshots: Mapping[str, TechnicalSnapshot],
    sector_tickers: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    spy = snapshots.get("SPY")
    rows = []
    for ticker in sector_tickers:
        item = snapshots.get(ticker)
        if item is None:
            continue
        rs5 = (item.return_5d or 0.0) - ((spy.return_5d or 0.0) if spy else 0.0)
        rs20 = (item.return_20d or 0.0) - ((spy.return_20d or 0.0) if spy else 0.0)
        rs60 = (item.return_60d or 0.0) - ((spy.return_60d or 0.0) if spy else 0.0)
        score = 100.0 * (0.30 * rs5 + 0.40 * rs20 + 0.20 * rs60)
        if item.ema20 is not None and item.price > item.ema20:
            score += 1.0
        if item.sma50 is not None and item.price > item.sma50:
            score += 1.0
        if (item.relative_volume_20d or 0.0) > 1.1:
            score += 0.5
        state = "NEUTRAL"
        if rs20 > 0.03 and rs5 > 0 and (item.extension_from_ema_atr or 0) > 1.75:
            state = "MATURE / EXTENDED LEADER"
        elif rs20 > 0.015 and rs5 > 0.005 and item.price > (item.ema20 or item.price + 1):
            state = "ACCELERATING LEADER"
        elif rs5 > 0.01 and rs20 > -0.01 and rs5 > rs20 / 4.0:
            state = "EMERGING LEADER"
        elif rs20 < -0.02 or (item.sma50 is not None and item.price < item.sma50):
            state = "DETERIORATING"
        rows.append(
            {
                "ticker": ticker,
                "score": score,
                "state": state,
                "return_5d": item.return_5d,
                "return_20d": item.return_20d,
                "return_60d": item.return_60d,
                "rs_5d": rs5,
                "rs_20d": rs20,
                "rs_60d": rs60,
                "relative_volume": item.relative_volume_20d,
            }
        )
    rows.sort(key=lambda row: float(row["score"]), reverse=True)
    return {row["ticker"]: dict(row, rank=index + 1) for index, row in enumerate(rows)}
