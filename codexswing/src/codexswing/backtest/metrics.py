"""Conservative replay metrics with decision-date cluster bootstrap."""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence


class ReturnLike(Protocol):
    sample_id: str
    ticker: str
    decision_date: str
    net_return: float


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("quantile requires values")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1))
    return ordered[index]


def cluster_bootstrap_lower_mean(
    values_by_date: Mapping[str, Sequence[float]],
    repetitions: int = 2000,
    seed: int = 1729,
    alpha: float = 0.05,
) -> float:
    if not values_by_date:
        raise ValueError("bootstrap requires at least one decision-date cluster")
    if repetitions < 100:
        raise ValueError("bootstrap repetitions must be at least 100")
    if not 0 < alpha < 0.5:
        raise ValueError("alpha must be between zero and one-half")
    cluster_means = [statistics.fmean(values) for _, values in sorted(values_by_date.items())]
    generator = random.Random(seed)
    estimates = []
    for _ in range(repetitions):
        draw = [generator.choice(cluster_means) for _ in cluster_means]
        estimates.append(statistics.fmean(draw))
    return _quantile(estimates, alpha)


@dataclass(frozen=True)
class ReplayMetrics:
    observation_count: int
    unique_decision_dates: int
    unique_tickers: int
    winning_observations: int
    win_rate: float
    mean_net_return: float
    median_net_return: float
    standard_deviation: float
    net_return_sum: float
    profit_factor: Optional[float]
    profit_factor_is_infinite: bool
    maximum_drawdown: float
    tail_p05_return: float
    bootstrap_p05_mean_return: float
    positive_decision_date_fraction: float
    maximum_ticker_observation_share: float
    maximum_date_observation_share: float
    bootstrap_repetitions: int
    bootstrap_seed: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_replay_metrics(
    outcomes: Sequence[ReturnLike],
    bootstrap_repetitions: int = 2000,
    bootstrap_seed: int = 1729,
) -> ReplayMetrics:
    materialized = tuple(outcomes)
    if not materialized:
        raise ValueError("metrics require at least one outcome")
    if len({item.sample_id for item in materialized}) != len(materialized):
        raise ValueError("metrics input has duplicate sample_id values")
    returns = [float(item.net_return) for item in materialized]
    if any(not math.isfinite(value) or value <= -1.0 for value in returns):
        raise ValueError("net returns must be finite and greater than -100%")
    by_date: Dict[str, list] = {}
    ticker_counts: Dict[str, int] = {}
    for item, value in zip(materialized, returns):
        by_date.setdefault(item.decision_date, []).append(value)
        ticker_counts[item.ticker] = ticker_counts.get(item.ticker, 0) + 1
    date_returns = [statistics.fmean(values) for _, values in sorted(by_date.items())]
    equity = 1.0
    peak = 1.0
    maximum_drawdown = 0.0
    for value in date_returns:
        equity *= 1.0 + value
        peak = max(peak, equity)
        maximum_drawdown = min(maximum_drawdown, equity / peak - 1.0)
    gains = sum(value for value in returns if value > 0)
    losses = -sum(value for value in returns if value < 0)
    profit_factor_is_infinite = losses == 0 and gains > 0
    profit_factor = None if losses == 0 else gains / losses
    positive_dates = sum(1 for value in date_returns if value > 0)
    count = len(materialized)
    return ReplayMetrics(
        observation_count=count,
        unique_decision_dates=len(by_date),
        unique_tickers=len(ticker_counts),
        winning_observations=sum(1 for value in returns if value > 0),
        win_rate=sum(1 for value in returns if value > 0) / count,
        mean_net_return=statistics.fmean(returns),
        median_net_return=statistics.median(returns),
        standard_deviation=statistics.stdev(returns) if count > 1 else 0.0,
        net_return_sum=sum(returns),
        profit_factor=profit_factor,
        profit_factor_is_infinite=profit_factor_is_infinite,
        maximum_drawdown=maximum_drawdown,
        tail_p05_return=_quantile(returns, 0.05),
        bootstrap_p05_mean_return=cluster_bootstrap_lower_mean(
            by_date,
            repetitions=bootstrap_repetitions,
            seed=bootstrap_seed,
        ),
        positive_decision_date_fraction=positive_dates / len(date_returns),
        maximum_ticker_observation_share=max(ticker_counts.values()) / count,
        maximum_date_observation_share=max(len(values) for values in by_date.values()) / count,
        bootstrap_repetitions=bootstrap_repetitions,
        bootstrap_seed=bootstrap_seed,
    )
