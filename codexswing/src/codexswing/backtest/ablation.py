"""Paired, frozen-source ablation comparisons."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Sequence

from codexswing.backtest.metrics import (
    ReplayMetrics,
    ReturnLike,
    cluster_bootstrap_lower_mean,
    compute_replay_metrics,
)


@dataclass(frozen=True)
class AblationStep:
    name: str
    metrics: ReplayMetrics
    incremental_mean_return: float
    paired_bootstrap_p05_incremental_return: float
    incremental_status: str

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["metrics"] = self.metrics.to_dict()
        return payload


def compare_ablation_variants(
    variants: Mapping[str, Sequence[ReturnLike]],
    ordered_names: Sequence[str],
    bootstrap_repetitions: int = 2000,
    bootstrap_seed: int = 1729,
) -> Sequence[AblationStep]:
    if not ordered_names:
        raise ValueError("ablation order cannot be empty")
    if set(ordered_names) != set(variants):
        raise ValueError("ablation names must exactly match supplied variants")
    baseline_by_id = {item.sample_id: item for item in variants[ordered_names[0]]}
    if not baseline_by_id:
        raise ValueError("ablation requires observations")
    if len(baseline_by_id) != len(variants[ordered_names[0]]):
        raise ValueError("ablation baseline has duplicate sample ids")
    steps = []
    prior_by_id = baseline_by_id
    for index, name in enumerate(ordered_names):
        current = tuple(variants[name])
        current_by_id = {item.sample_id: item for item in current}
        if len(current_by_id) != len(current) or set(current_by_id) != set(baseline_by_id):
            raise ValueError("all ablation variants must use the exact same sample ids")
        for sample_id, item in current_by_id.items():
            baseline = baseline_by_id[sample_id]
            if item.decision_date != baseline.decision_date or item.ticker != baseline.ticker:
                raise ValueError("ablation sample identity changed across variants")
        metrics = compute_replay_metrics(
            current,
            bootstrap_repetitions=bootstrap_repetitions,
            bootstrap_seed=bootstrap_seed,
        )
        if index == 0:
            incremental_mean = 0.0
            incremental_lower = 0.0
            status = "FROZEN_BASELINE"
        else:
            paired_by_date: Dict[str, list] = {}
            differences = []
            for sample_id, item in current_by_id.items():
                difference = item.net_return - prior_by_id[sample_id].net_return
                differences.append(difference)
                paired_by_date.setdefault(item.decision_date, []).append(difference)
            incremental_mean = sum(differences) / len(differences)
            incremental_lower = cluster_bootstrap_lower_mean(
                paired_by_date,
                repetitions=bootstrap_repetitions,
                seed=bootstrap_seed,
            )
            status = (
                "INCREMENTAL_EDGE_CANDIDATE_NOT_PROMOTED"
                if incremental_mean > 0 and incremental_lower > 0
                else "NO_INCREMENTAL_EDGE"
            )
        steps.append(
            AblationStep(
                name=name,
                metrics=metrics,
                incremental_mean_return=incremental_mean,
                paired_bootstrap_p05_incremental_return=incremental_lower,
                incremental_status=status,
            )
        )
        prior_by_id = current_by_id
    return tuple(steps)
