"""Fixed price/move baseline used as the first ablation benchmark."""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass
from typing import Any, Dict, Sequence

from codexswing.features.price import PriceObservation


class BaselineDataError(RuntimeError):
    pass


@dataclass(frozen=True)
class PriceMoveBaseline:
    ticker: str
    as_of_date: str
    observation_count: int
    close: float
    return_1d: float
    return_5d: float
    return_10d: float
    return_20d: float
    realized_vol_20d_annualized: float
    expected_abs_move_5d_pct: float
    range_10d_pct: float
    volume_ratio_20d: float
    average_dollar_volume_20d: float
    trend_score_raw: float
    status: str = "RESEARCH_BASELINE_UNVALIDATED"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _simple_return(values: Sequence[float], horizon: int) -> float:
    return values[-1] / values[-1 - horizon] - 1.0


def compute_price_move_baseline(observations: Sequence[PriceObservation]) -> PriceMoveBaseline:
    ordered = sorted(observations, key=lambda item: item.session_date)
    if len(ordered) < 21:
        raise BaselineDataError("price baseline requires at least 21 observations")
    if len({item.ticker for item in ordered}) != 1:
        raise BaselineDataError("price baseline accepts one ticker at a time")
    if len({item.session_date for item in ordered}) != len(ordered):
        raise BaselineDataError("price baseline contains duplicate dates")
    closes = [item.close for item in ordered]
    log_returns = [math.log(closes[index] / closes[index - 1]) for index in range(1, len(closes))]
    recent_returns = log_returns[-20:]
    if any(abs(value) > math.log(1.5) for value in recent_returns):
        raise BaselineDataError("possible split/corporate-action jump requires adjustment review")
    daily_sigma = statistics.stdev(recent_returns)
    realized_vol = daily_sigma * math.sqrt(252.0)
    expected_abs_move_5d = daily_sigma * math.sqrt(5.0) * math.sqrt(2.0 / math.pi)
    recent_ten = ordered[-10:]
    range_10d = (max(item.high for item in recent_ten) - min(item.low for item in recent_ten)) / closes[-1]
    prior_volumes = [item.volume for item in ordered[-21:-1]]
    average_volume = statistics.fmean(prior_volumes)
    volume_ratio = ordered[-1].volume / average_volume if average_volume > 0 else 0.0
    average_dollar_volume = statistics.fmean(item.close * item.volume for item in ordered[-20:])
    return_5d = _simple_return(closes, 5)
    return_10d = _simple_return(closes, 10)
    return_20d = _simple_return(closes, 20)
    trend_score = 0.20 * return_5d + 0.30 * return_10d + 0.50 * return_20d
    return PriceMoveBaseline(
        ticker=ordered[-1].ticker,
        as_of_date=ordered[-1].session_date,
        observation_count=len(ordered),
        close=closes[-1],
        return_1d=_simple_return(closes, 1),
        return_5d=return_5d,
        return_10d=return_10d,
        return_20d=return_20d,
        realized_vol_20d_annualized=realized_vol,
        expected_abs_move_5d_pct=expected_abs_move_5d,
        range_10d_pct=range_10d,
        volume_ratio_20d=volume_ratio,
        average_dollar_volume_20d=average_dollar_volume,
        trend_score_raw=trend_score,
    )

