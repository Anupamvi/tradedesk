"""Deterministic statistical primitives used by Cultra evidence gates."""

from dataclasses import dataclass
import math
import random
from typing import Dict, Hashable, Iterable, Mapping, Sequence, Tuple


def _finite_probability(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError("%s must be a finite probability" % name)
    return value


def arithmetic_mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("at least one value is required")
    converted = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in converted):
        raise ValueError("values must be finite")
    return math.fsum(converted) / len(converted)


def empirical_quantile(values: Sequence[float], probability: float) -> float:
    """Return a deterministic linearly interpolated empirical quantile."""

    probability = _finite_probability(probability, "probability")
    if not values:
        raise ValueError("at least one value is required")
    ordered = sorted(float(value) for value in values)
    if not all(math.isfinite(value) for value in ordered):
        raise ValueError("values must be finite")
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return ordered[lower_index]
    weight = position - lower_index
    return ordered[lower_index] * (1.0 - weight) + ordered[upper_index] * weight


@dataclass(frozen=True)
class BootstrapInterval:
    point: float
    lower: float
    upper: float
    confidence: float
    iterations: int
    cluster_count: int
    seed: int


@dataclass(frozen=True)
class MultiwayBootstrapInterval:
    point: float
    lower: float
    upper: float
    confidence: float
    iterations: int
    first_cluster_count: int
    second_cluster_count: int
    joint_cluster_count: int
    seed: int


def two_way_clustered_bootstrap_mean_ci(
    values: Sequence[float],
    first_clusters: Sequence[Hashable],
    second_clusters: Sequence[Hashable],
    confidence: float = 0.95,
    iterations: int = 5000,
    seed: int = 0,
) -> MultiwayBootstrapInterval:
    """Bootstrap the mean under simultaneous ticker and date dependence.

    Each iteration independently resamples both cluster dimensions. An
    observation receives the product of its two sampled multiplicities. This
    prevents either repeated ticker exposure or a common market date from
    being treated as independent evidence.
    """

    if not (
        len(values) == len(first_clusters) == len(second_clusters) and values
    ):
        raise ValueError("values and both cluster dimensions must have equal non-zero length")
    confidence = float(confidence)
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0:
        raise ValueError("iterations must be a positive integer")
    converted = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in converted):
        raise ValueError("values must be finite")
    if any(value is None for value in first_clusters) or any(
        value is None for value in second_clusters
    ):
        raise ValueError("cluster identifiers cannot be None")
    first_keys = tuple(
        sorted(set(first_clusters), key=lambda item: (type(item).__name__, repr(item)))
    )
    second_keys = tuple(
        sorted(set(second_clusters), key=lambda item: (type(item).__name__, repr(item)))
    )
    joint_count = len(set(zip(first_clusters, second_clusters)))
    rng = random.Random(seed)
    draws = []
    for _ in range(iterations):
        first_weights = {key: 0 for key in first_keys}
        second_weights = {key: 0 for key in second_keys}
        for _unused in first_keys:
            first_weights[first_keys[rng.randrange(len(first_keys))]] += 1
        for _unused in second_keys:
            second_weights[second_keys[rng.randrange(len(second_keys))]] += 1
        numerator = 0.0
        denominator = 0
        for value, first, second in zip(
            converted, first_clusters, second_clusters
        ):
            weight = first_weights[first] * second_weights[second]
            if weight:
                numerator += value * weight
                denominator += weight
        # Empty intersections are possible with sparse panels. Redraw rather
        # than inventing a value or silently reverting to IID sampling.
        if denominator == 0:
            continue
        draws.append(numerator / denominator)
    if len(draws) < max(100, iterations // 2):
        raise ValueError("two-way bootstrap panel is too sparse for stable inference")
    alpha = (1.0 - confidence) / 2.0
    return MultiwayBootstrapInterval(
        point=arithmetic_mean(converted),
        lower=empirical_quantile(draws, alpha),
        upper=empirical_quantile(draws, 1.0 - alpha),
        confidence=confidence,
        iterations=len(draws),
        first_cluster_count=len(first_keys),
        second_cluster_count=len(second_keys),
        joint_cluster_count=joint_count,
        seed=seed,
    )


def two_way_clustered_positive_mean_p_value(
    values: Sequence[float],
    first_clusters: Sequence[Hashable],
    second_clusters: Sequence[Hashable],
    iterations: int = 5000,
    seed: int = 0,
) -> float:
    """One-sided, null-centered two-way cluster-bootstrap p-value.

    The null distribution is formed after centering the observed panel at
    zero, while independently resampling both cluster dimensions.  The plus-one
    correction prevents a simulated p-value of exactly zero.
    """

    if not (
        len(values) == len(first_clusters) == len(second_clusters) and values
    ):
        raise ValueError(
            "values and both cluster dimensions must have equal non-zero length"
        )
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0:
        raise ValueError("iterations must be a positive integer")
    converted = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in converted):
        raise ValueError("values must be finite")
    if any(value is None for value in first_clusters) or any(
        value is None for value in second_clusters
    ):
        raise ValueError("cluster identifiers cannot be None")
    observed = arithmetic_mean(converted)
    if observed <= 0.0:
        return 1.0
    centered = tuple(value - observed for value in converted)
    first_keys = tuple(
        sorted(set(first_clusters), key=lambda item: (type(item).__name__, repr(item)))
    )
    second_keys = tuple(
        sorted(set(second_clusters), key=lambda item: (type(item).__name__, repr(item)))
    )
    rng = random.Random(seed)
    extreme = 0
    usable = 0
    for _ in range(iterations):
        first_weights = {key: 0 for key in first_keys}
        second_weights = {key: 0 for key in second_keys}
        for _unused in first_keys:
            first_weights[first_keys[rng.randrange(len(first_keys))]] += 1
        for _unused in second_keys:
            second_weights[second_keys[rng.randrange(len(second_keys))]] += 1
        numerator = 0.0
        denominator = 0
        for value, first, second in zip(centered, first_clusters, second_clusters):
            weight = first_weights[first] * second_weights[second]
            if weight:
                numerator += value * weight
                denominator += weight
        if denominator == 0:
            continue
        usable += 1
        if numerator / denominator >= observed:
            extreme += 1
    if usable < max(100, iterations // 2):
        raise ValueError("two-way bootstrap panel is too sparse for stable inference")
    return (extreme + 1.0) / (usable + 1.0)


def clustered_bootstrap_mean_ci(
    values: Sequence[float],
    clusters: Sequence[Hashable],
    confidence: float = 0.95,
    iterations: int = 5000,
    seed: int = 0,
) -> BootstrapInterval:
    """Cluster-resample observations and return a percentile mean interval.

    Clusters, rather than individual trades, are sampled with replacement.
    Sampling and ordering are fully deterministic for a given seed.
    """

    confidence = float(confidence)
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0:
        raise ValueError("iterations must be a positive integer")
    if len(values) != len(clusters) or not values:
        raise ValueError("values and clusters must have equal non-zero length")

    grouped = {}  # type: Dict[Hashable, Tuple[float, ...]]
    mutable = {}  # type: Dict[Hashable, list]
    for value, cluster in zip(values, clusters):
        converted = float(value)
        if not math.isfinite(converted):
            raise ValueError("values must be finite")
        if cluster is None:
            raise ValueError("cluster identifiers cannot be None")
        mutable.setdefault(cluster, []).append(converted)
    for key, items in mutable.items():
        grouped[key] = tuple(items)

    # repr provides deterministic ordering even when callers use sets/dicts.
    cluster_keys = tuple(sorted(grouped, key=lambda item: (type(item).__name__, repr(item))))
    rng = random.Random(seed)
    draws = []
    for _ in range(iterations):
        sampled_values = []
        for _cluster_index in range(len(cluster_keys)):
            sampled_key = cluster_keys[rng.randrange(len(cluster_keys))]
            sampled_values.extend(grouped[sampled_key])
        draws.append(arithmetic_mean(sampled_values))

    alpha = (1.0 - confidence) / 2.0
    return BootstrapInterval(
        point=arithmetic_mean(tuple(float(value) for value in values)),
        lower=empirical_quantile(draws, alpha),
        upper=empirical_quantile(draws, 1.0 - alpha),
        confidence=confidence,
        iterations=iterations,
        cluster_count=len(cluster_keys),
        seed=seed,
    )


def holm_adjust(p_values: Sequence[float]) -> Tuple[float, ...]:
    """Return Holm step-down adjusted p-values in original order."""

    if not p_values:
        return ()
    checked = tuple(_finite_probability(value, "p_value") for value in p_values)
    ranked = sorted(enumerate(checked), key=lambda pair: (pair[1], pair[0]))
    adjusted = [0.0] * len(checked)
    running = 0.0
    total = len(checked)
    for rank, (original_index, p_value) in enumerate(ranked):
        candidate = min(1.0, (total - rank) * p_value)
        running = max(running, candidate)
        adjusted[original_index] = running
    return tuple(adjusted)


def holm_adjust_mapping(p_values: Mapping[str, float]) -> Dict[str, float]:
    keys = tuple(sorted(p_values))
    adjusted = holm_adjust(tuple(p_values[key] for key in keys))
    return {key: value for key, value in zip(keys, adjusted)}


@dataclass(frozen=True)
class ContributionConcentration:
    max_fraction: float
    max_cluster: Hashable
    positive_profit: float
    by_cluster: Mapping[Hashable, float]


def contribution_concentration(
    net_pnls: Sequence[float], clusters: Sequence[Hashable]
) -> ContributionConcentration:
    """Measure the largest cluster's share of all positive profit.

    Losses do not cancel a winner's concentration.  This conservative
    denominator prevents one profitable ticker or calendar period from being
    hidden by unrelated losses.
    """

    if len(net_pnls) != len(clusters) or not net_pnls:
        raise ValueError("net_pnls and clusters must have equal non-zero length")
    totals = {}  # type: Dict[Hashable, float]
    for pnl, cluster in zip(net_pnls, clusters):
        pnl = float(pnl)
        if not math.isfinite(pnl):
            raise ValueError("net_pnls must be finite")
        if cluster is None:
            raise ValueError("cluster identifiers cannot be None")
        totals[cluster] = totals.get(cluster, 0.0) + max(0.0, pnl)
    positive_profit = math.fsum(totals.values())
    if positive_profit <= 0.0:
        raise ValueError("contribution concentration requires positive profit")
    max_cluster = min(
        totals,
        key=lambda key: (-totals[key], type(key).__name__, repr(key)),
    )
    return ContributionConcentration(
        max_fraction=totals[max_cluster] / positive_profit,
        max_cluster=max_cluster,
        positive_profit=positive_profit,
        by_cluster=dict(totals),
    )
