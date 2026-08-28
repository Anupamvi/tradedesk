"""Cluster-aware inference and family-wise multiple-testing correction."""

from __future__ import annotations

import hashlib
import math
import random
import statistics
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def holm_bonferroni(p_values: Mapping[str, float]) -> Mapping[str, float]:
    """Return monotone Holm-adjusted p-values keyed by hypothesis ID."""

    if not p_values:
        raise ValueError("at least one p-value is required")
    for hypothesis_id, value in p_values.items():
        if not hypothesis_id or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("invalid p-value for {}".format(hypothesis_id))
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    count = len(ordered)
    running = 0.0
    adjusted: Dict[str, float] = {}
    for rank, (hypothesis_id, value) in enumerate(ordered):
        candidate = min(1.0, (count - rank) * value)
        running = max(running, candidate)
        adjusted[hypothesis_id] = running
    return adjusted


def cluster_bootstrap_one_sided_pvalue(
    outcomes: Sequence[float],
    cluster_ids: Sequence[str],
    iterations: int = 4_000,
    seed_text: str = "codexswing-v0.5",
) -> float:
    """Test H0 mean <= 0 using a centered, cluster-resampled distribution."""

    if len(outcomes) != len(cluster_ids) or not outcomes:
        raise ValueError("outcomes and cluster_ids must be non-empty and aligned")
    if iterations < 100:
        raise ValueError("at least 100 bootstrap iterations are required")
    if any(not math.isfinite(float(value)) for value in outcomes):
        raise ValueError("outcomes must be finite")
    clusters: Dict[str, List[float]] = {}
    for outcome, cluster_id in zip(outcomes, cluster_ids):
        if not cluster_id:
            raise ValueError("cluster IDs cannot be empty")
        clusters.setdefault(cluster_id, []).append(float(outcome))
    if len(clusters) < 2:
        raise ValueError("at least two independent clusters are required")
    cluster_means = [statistics.fmean(clusters[key]) for key in sorted(clusters)]
    observed = statistics.fmean(cluster_means)
    if observed <= 0:
        return 1.0
    generator = random.Random(
        int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:16], 16)
    )
    extreme = 0
    for _ in range(iterations):
        sampled_mean = statistics.fmean(
            generator.choice(cluster_means) for _cluster_index in cluster_means
        )
        centered_mean = sampled_mean - observed
        if centered_mean >= observed:
            extreme += 1
    return (extreme + 1.0) / (iterations + 1.0)


@dataclass(frozen=True)
class HypothesisEvidence:
    hypothesis_id: str
    raw_p_value: float
    train_mean_pnl: float
    validation_mean_pnl: float
    holdout_mean_pnl: float
    holdout_cluster_count: int

    def __post_init__(self) -> None:
        if not self.hypothesis_id:
            raise ValueError("hypothesis ID is required")
        values = (
            self.raw_p_value,
            self.train_mean_pnl,
            self.validation_mean_pnl,
            self.holdout_mean_pnl,
        )
        if any(not math.isfinite(item) for item in values):
            raise ValueError("hypothesis evidence values must be finite")
        if not 0 <= self.raw_p_value <= 1:
            raise ValueError("raw p-value must be between zero and one")
        if self.holdout_cluster_count < 0:
            raise ValueError("holdout cluster count cannot be negative")


@dataclass(frozen=True)
class HypothesisDecision:
    hypothesis_id: str
    raw_p_value: float
    adjusted_p_value: float
    status: str
    reasons: Tuple[str, ...]


@dataclass(frozen=True)
class FamilyEvaluation:
    method: str
    family_size: int
    alpha: float
    decisions: Tuple[HypothesisDecision, ...]

    @property
    def promotion_eligible_count(self) -> int:
        return sum(item.status == "PROMOTION_ELIGIBLE" for item in self.decisions)


def evaluate_hypothesis_family(
    evidence: Sequence[HypothesisEvidence],
    alpha: float = 0.05,
    minimum_holdout_clusters: int = 15,
    expected_hypothesis_ids: Optional[Iterable[str]] = None,
) -> FamilyEvaluation:
    """Count the full declared family and reject uncorrected winner selection."""

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    if minimum_holdout_clusters <= 0:
        raise ValueError("minimum holdout clusters must be positive")
    by_id = {item.hypothesis_id: item for item in evidence}
    if len(by_id) != len(evidence) or not by_id:
        raise ValueError("hypothesis evidence must be non-empty and unique")
    if expected_hypothesis_ids is not None:
        expected = set(expected_hypothesis_ids)
        if set(by_id) != expected:
            missing = sorted(expected - set(by_id))
            extra = sorted(set(by_id) - expected)
            raise ValueError(
                "incomplete hypothesis family; missing={} extra={}".format(missing, extra)
            )
    adjusted = holm_bonferroni(
        {key: item.raw_p_value for key, item in by_id.items()}
    )
    decisions = []
    for hypothesis_id in sorted(by_id):
        item = by_id[hypothesis_id]
        reasons = []
        if item.train_mean_pnl <= 0:
            reasons.append("TRAIN_EXPECTANCY_NOT_POSITIVE")
        if item.validation_mean_pnl <= 0:
            reasons.append("VALIDATION_EXPECTANCY_NOT_POSITIVE")
        if item.holdout_mean_pnl <= 0:
            reasons.append("HOLDOUT_EXPECTANCY_NOT_POSITIVE")
        if item.holdout_cluster_count < minimum_holdout_clusters:
            reasons.append("INSUFFICIENT_INDEPENDENT_HOLDOUT_CLUSTERS")
        if adjusted[hypothesis_id] > alpha:
            reasons.append("HOLM_ADJUSTED_P_VALUE_ABOVE_ALPHA")
        decisions.append(
            HypothesisDecision(
                hypothesis_id=hypothesis_id,
                raw_p_value=item.raw_p_value,
                adjusted_p_value=adjusted[hypothesis_id],
                status=("PROMOTION_ELIGIBLE" if not reasons else "EXPLORATORY_NOT_PROMOTED"),
                reasons=tuple(reasons),
            )
        )
    return FamilyEvaluation(
        method="CLUSTER_BOOTSTRAP_PLUS_HOLM_BONFERRONI",
        family_size=len(decisions),
        alpha=alpha,
        decisions=tuple(decisions),
    )
