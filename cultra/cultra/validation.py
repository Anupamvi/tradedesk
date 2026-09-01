"""Leakage controls and strict evidence-state promotion gates."""

from dataclasses import dataclass, replace
from datetime import date
import math
from typing import Dict, Optional, Sequence, Tuple

from .hypotheses import (
    FROZEN_HYPOTHESIS_COUNT,
    HYPOTHESIS_REGISTRY_HASH,
    HYPOTHESIS_REGISTRY_VERSION,
)
from .domain import EvidenceState, FamilyEvidence, HistoricalObservation
from .statistics import (
    BootstrapInterval,
    ContributionConcentration,
    clustered_bootstrap_mean_ci,
    contribution_concentration,
    holm_adjust,
    holm_adjust_mapping,
)


@dataclass(frozen=True)
class ChronologicalSplit:
    training: Tuple[HistoricalObservation, ...]
    validation: Tuple[HistoricalObservation, ...]
    holdout: Tuple[HistoricalObservation, ...]
    embargoed: Tuple[HistoricalObservation, ...]
    embargo_sessions: int
    validation_fraction: float
    holdout_fraction: float

    def __post_init__(self) -> None:
        partitions = (self.training, self.validation, self.holdout, self.embargoed)
        identifiers = [item.observation_id for part in partitions for item in part]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("an observation appears in multiple split partitions")
        if not self.training or not self.validation or not self.holdout:
            raise ValueError("training, validation, and holdout must all be non-empty")
        if max(item.session_date for item in self.training) >= min(
            item.session_date for item in self.validation
        ):
            raise ValueError("training must strictly precede validation")
        if max(item.session_date for item in self.validation) >= min(
            item.session_date for item in self.holdout
        ):
            raise ValueError("validation must strictly precede holdout")


@dataclass(frozen=True)
class WalkForwardFold:
    """One expanding-window development fold with a full session embargo."""

    fold_index: int
    training: Tuple[HistoricalObservation, ...]
    validation: Tuple[HistoricalObservation, ...]
    embargoed: Tuple[HistoricalObservation, ...]
    embargo_sessions: int

    def __post_init__(self) -> None:
        if self.fold_index < 0:
            raise ValueError("fold_index cannot be negative")
        if not self.training or not self.validation:
            raise ValueError("walk-forward training and validation cannot be empty")
        train_dates = {item.session_date for item in self.training}
        validation_dates = {item.session_date for item in self.validation}
        embargo_dates = {item.session_date for item in self.embargoed}
        if train_dates & validation_dates or train_dates & embargo_dates or validation_dates & embargo_dates:
            raise ValueError("walk-forward fold partitions overlap")
        if len(embargo_dates) != self.embargo_sessions:
            raise ValueError("walk-forward fold does not contain the full embargo")
        if max(train_dates) >= min(embargo_dates) or max(embargo_dates) >= min(validation_dates):
            raise ValueError("walk-forward fold is not strictly chronological")


@dataclass(frozen=True)
class WalkForwardPlan:
    """Development folds plus one untouched final holdout."""

    folds: Tuple[WalkForwardFold, ...]
    final_holdout: Tuple[HistoricalObservation, ...]
    final_holdout_embargoed: Tuple[HistoricalObservation, ...]
    embargo_sessions: int
    holdout_fraction: float

    def __post_init__(self) -> None:
        if not self.folds or not self.final_holdout:
            raise ValueError("walk-forward folds and final holdout are required")
        holdout_dates = {item.session_date for item in self.final_holdout}
        embargo_dates = {item.session_date for item in self.final_holdout_embargoed}
        if holdout_dates & embargo_dates:
            raise ValueError("final holdout overlaps its embargo")
        if len(embargo_dates) != self.embargo_sessions:
            raise ValueError("final holdout does not contain the full embargo")
        if max(embargo_dates) >= min(holdout_dates):
            raise ValueError("final holdout embargo is not chronological")
        for fold in self.folds:
            if {item.session_date for item in fold.validation} & holdout_dates:
                raise ValueError("development validation overlaps final holdout")


def _observations_by_session(
    observations: Sequence[HistoricalObservation],
) -> Tuple[Tuple[date, ...], Dict[date, list]]:
    if not observations:
        raise ValueError("observations cannot be empty")
    identifiers = tuple(item.observation_id for item in observations)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("observation_id values must be unique")
    by_session = {}  # type: Dict[date, list]
    for observation in observations:
        by_session.setdefault(observation.session_date, []).append(observation)
    return tuple(sorted(by_session)), by_session


def _collect_sessions(
    selected_sessions: Sequence[date], by_session: Dict[date, list]
) -> Tuple[HistoricalObservation, ...]:
    return tuple(
        sorted(
            (
                observation
                for session in selected_sessions
                for observation in by_session[session]
            ),
            key=lambda observation: (
                observation.session_date,
                observation.observation_id,
            ),
        )
    )


def walk_forward_development_splits(
    observations: Sequence[HistoricalObservation],
    *,
    min_training_sessions: int = 120,
    validation_sessions: int = 20,
    embargo_sessions: int = 60,
    step_sessions: Optional[int] = None,
) -> Tuple[WalkForwardFold, ...]:
    """Build deterministic expanding-window folds without a holdout.

    This lower-level function is used when the caller has already removed and
    sealed its final holdout.  Validation windows never overlap and every fold
    has exactly ``embargo_sessions`` distinct sessions between training and
    validation.
    """

    for name, value in (
        ("min_training_sessions", min_training_sessions),
        ("validation_sessions", validation_sessions),
        ("embargo_sessions", embargo_sessions),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("%s must be a positive integer" % name)
    step = validation_sessions if step_sessions is None else step_sessions
    if isinstance(step, bool) or not isinstance(step, int) or step < validation_sessions:
        raise ValueError("step_sessions must be at least validation_sessions")
    sessions, by_session = _observations_by_session(observations)
    first_validation = min_training_sessions + embargo_sessions
    if first_validation + validation_sessions > len(sessions):
        raise ValueError("insufficient sessions for one walk-forward fold")
    folds = []
    validation_start = first_validation
    fold_index = 0
    while validation_start + validation_sessions <= len(sessions):
        training_end = validation_start - embargo_sessions
        fold = WalkForwardFold(
            fold_index=fold_index,
            training=_collect_sessions(sessions[:training_end], by_session),
            embargoed=_collect_sessions(
                sessions[training_end:validation_start], by_session
            ),
            validation=_collect_sessions(
                sessions[
                    validation_start : validation_start + validation_sessions
                ],
                by_session,
            ),
            embargo_sessions=embargo_sessions,
        )
        folds.append(fold)
        fold_index += 1
        validation_start += step
    return tuple(folds)


def walk_forward_splits(
    observations: Sequence[HistoricalObservation],
    *,
    min_training_sessions: int = 120,
    validation_sessions: int = 20,
    holdout_fraction: float = 0.20,
    embargo_sessions: int = 60,
    step_sessions: Optional[int] = None,
) -> WalkForwardPlan:
    """Build expanding development folds and seal the final 20% holdout."""

    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be between 0 and 1")
    sessions, by_session = _observations_by_session(observations)
    holdout_count = max(1, int(math.ceil(len(sessions) * holdout_fraction)))
    holdout_start = len(sessions) - holdout_count
    development_end = holdout_start - embargo_sessions
    if development_end <= 0:
        raise ValueError("insufficient sessions before final holdout embargo")
    development = _collect_sessions(sessions[:development_end], by_session)
    folds = walk_forward_development_splits(
        development,
        min_training_sessions=min_training_sessions,
        validation_sessions=validation_sessions,
        embargo_sessions=embargo_sessions,
        step_sessions=step_sessions,
    )
    return WalkForwardPlan(
        folds=folds,
        final_holdout=_collect_sessions(sessions[holdout_start:], by_session),
        final_holdout_embargoed=_collect_sessions(
            sessions[development_end:holdout_start], by_session
        ),
        embargo_sessions=embargo_sessions,
        holdout_fraction=holdout_fraction,
    )


def chronological_split(
    observations: Sequence[HistoricalObservation],
    validation_fraction: float = 0.20,
    holdout_fraction: float = 0.20,
    embargo_sessions: int = 60,
) -> ChronologicalSplit:
    """Create one development/validation/final-holdout split by session.

    All observations from the same session remain together.  A full embargo is
    removed immediately before validation and immediately before the untouched
    final holdout.
    """

    if not observations:
        raise ValueError("observations cannot be empty")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be between 0 and 1")
    if validation_fraction + holdout_fraction >= 1.0:
        raise ValueError("validation and holdout fractions must leave training data")
    if isinstance(embargo_sessions, bool) or not isinstance(embargo_sessions, int):
        raise TypeError("embargo_sessions must be an integer")
    if embargo_sessions < 0:
        raise ValueError("embargo_sessions cannot be negative")
    ids = tuple(item.observation_id for item in observations)
    if len(ids) != len(set(ids)):
        raise ValueError("observation_id values must be unique")

    by_session = {}  # type: Dict[date, list]
    for observation in observations:
        by_session.setdefault(observation.session_date, []).append(observation)
    sessions = tuple(sorted(by_session))
    session_count = len(sessions)
    holdout_count = max(1, int(math.ceil(session_count * holdout_fraction)))
    validation_count = max(1, int(math.ceil(session_count * validation_fraction)))

    holdout_start = session_count - holdout_count
    validation_end = holdout_start - embargo_sessions
    validation_start = validation_end - validation_count
    training_end = validation_start - embargo_sessions
    if training_end <= 0 or validation_start < 0 or validation_end <= validation_start:
        raise ValueError("insufficient distinct sessions for requested splits and embargoes")

    training_dates = set(sessions[:training_end])
    validation_dates = set(sessions[validation_start:validation_end])
    holdout_dates = set(sessions[holdout_start:])
    embargo_dates = set(sessions[training_end:validation_start]) | set(
        sessions[validation_end:holdout_start]
    )

    def collect(selected_dates: set) -> Tuple[HistoricalObservation, ...]:
        return tuple(
            sorted(
                (
                    item
                    for selected_date in selected_dates
                    for item in by_session[selected_date]
                ),
                key=lambda item: (item.session_date, item.observation_id),
            )
        )

    return ChronologicalSplit(
        training=collect(training_dates),
        validation=collect(validation_dates),
        holdout=collect(holdout_dates),
        embargoed=collect(embargo_dates),
        embargo_sessions=embargo_sessions,
        validation_fraction=validation_fraction,
        holdout_fraction=holdout_fraction,
    )


_DEFAULT_NEXT_STATE = {
    EvidenceState.UNPROVEN: EvidenceState.RESEARCH_PASS,
    EvidenceState.RESEARCH_PASS: EvidenceState.VALIDATION_PASS,
    EvidenceState.VALIDATION_PASS: EvidenceState.HOLDOUT_PASS,
    EvidenceState.HOLDOUT_PASS: EvidenceState.MANUAL_TICKET_ENABLED,
    EvidenceState.SHADOW_PASS: EvidenceState.MANUAL_TICKET_ENABLED,
}

_ALLOWED_TRANSITIONS = {
    EvidenceState.UNPROVEN: {EvidenceState.RESEARCH_PASS},
    EvidenceState.RESEARCH_PASS: {EvidenceState.VALIDATION_PASS},
    EvidenceState.VALIDATION_PASS: {EvidenceState.HOLDOUT_PASS},
    EvidenceState.HOLDOUT_PASS: {
        EvidenceState.MANUAL_TICKET_ENABLED,
        EvidenceState.SHADOW_PASS,
    },
    EvidenceState.SHADOW_PASS: {EvidenceState.MANUAL_TICKET_ENABLED},
}


def assert_transition(current: EvidenceState, target: EvidenceState) -> None:
    allowed = _ALLOWED_TRANSITIONS.get(current)
    if allowed is None:
        raise ValueError("%s is terminal" % current.value)
    if target not in allowed:
        raise ValueError(
            "invalid evidence transition %s -> %s; allowed %s"
            % (
                current.value,
                target.value,
                ", ".join(sorted(item.value for item in allowed)),
            )
        )


@dataclass(frozen=True)
class PromotionPolicy:
    min_holdout_trades: int = 100
    min_holdout_clusters: int = 40
    max_holm_adjusted_p_value: float = 0.05
    max_contribution_fraction: float = 0.20
    min_shadow_calendar_days: int = 90
    min_shadow_trades: int = 30
    max_pop_ece: float = 0.05
    min_pop_sample_size: int = 100
    min_probability_positive_events: int = 20
    min_probability_negative_events: int = 20
    min_holdout_resolution_rate: float = 0.95

    def __post_init__(self) -> None:
        for field_name in (
            "min_holdout_trades",
            "min_holdout_clusters",
            "min_shadow_calendar_days",
            "min_shadow_trades",
            "min_pop_sample_size",
            "min_probability_positive_events",
            "min_probability_negative_events",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("%s must be a positive integer" % field_name)
        for field_name in (
            "max_holm_adjusted_p_value",
            "max_contribution_fraction",
            "max_pop_ece",
            "min_holdout_resolution_rate",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("%s must be a probability" % field_name)


@dataclass(frozen=True)
class PromotionDecision:
    current_state: EvidenceState
    target_state: EvidenceState
    passed: bool
    reasons: Tuple[str, ...]


def _reasons_for_target(
    evidence: FamilyEvidence, target: EvidenceState, policy: PromotionPolicy
) -> Tuple[str, ...]:
    reasons = []
    if target in (
        EvidenceState.RESEARCH_PASS,
        EvidenceState.VALIDATION_PASS,
        EvidenceState.HOLDOUT_PASS,
        EvidenceState.SHADOW_PASS,
        EvidenceState.MANUAL_TICKET_ENABLED,
    ):
        if evidence.training.expectancy <= 0.0:
            reasons.append("training expectancy is not positive")
    if target in (
        EvidenceState.VALIDATION_PASS,
        EvidenceState.HOLDOUT_PASS,
        EvidenceState.SHADOW_PASS,
        EvidenceState.MANUAL_TICKET_ENABLED,
    ):
        if evidence.validation.expectancy <= 0.0:
            reasons.append("validation expectancy is not positive")
    if target in (
        EvidenceState.HOLDOUT_PASS,
        EvidenceState.SHADOW_PASS,
        EvidenceState.MANUAL_TICKET_ENABLED,
    ):
        if evidence.holdout.expectancy <= 0.0:
            reasons.append("holdout expectancy is not positive")
        if evidence.holdout.lower_confidence_bound <= 0.0:
            reasons.append("holdout 95% lower confidence bound is not positive")
        if evidence.holdout.resolved_trades < policy.min_holdout_trades:
            reasons.append("holdout has fewer than %d resolved trades" % policy.min_holdout_trades)
        if evidence.holdout.independent_clusters < policy.min_holdout_clusters:
            reasons.append(
                "holdout has fewer than %d independent clusters"
                % policy.min_holdout_clusters
            )
        if evidence.holm_adjusted_p_value > policy.max_holm_adjusted_p_value:
            reasons.append("Holm-adjusted significance gate failed")
        if evidence.holm_family_size != FROZEN_HYPOTHESIS_COUNT:
            reasons.append(
                "Holm correction does not cover all %d frozen hypotheses"
                % FROZEN_HYPOTHESIS_COUNT
            )
        if evidence.holm_catalog_version != HYPOTHESIS_REGISTRY_VERSION:
            reasons.append(
                "Holm correction registry version does not match the frozen hypotheses"
            )
        if evidence.max_contribution_fraction > policy.max_contribution_fraction:
            reasons.append("profit contribution concentration gate failed")
        if set(evidence.contribution_dimensions) != {"calendar_period", "ticker"}:
            reasons.append("contribution concentration must cover ticker and calendar period")
        if not math.isclose(
            evidence.holdout.confidence_level, 0.95, rel_tol=0.0, abs_tol=1e-12
        ):
            reasons.append("holdout lower confidence bound must use 95% confidence")
        if not evidence.holdout_consumed_once:
            reasons.append("holdout was not consumed exactly once")
        if evidence.hypothesis_registry_hash != HYPOTHESIS_REGISTRY_HASH:
            reasons.append("family evidence is not bound to the frozen hypothesis registry")
        if evidence.timing_policy_version == "UNSPECIFIED" or not evidence.next_session_entry:
            reasons.append("historical entry timing is not proven next-session executable")
        if (
            evidence.universe_policy_version == "UNSPECIFIED"
            or not evidence.point_in_time_membership
        ):
            reasons.append("historical universe is not point-in-time membership safe")
        if not evidence.two_way_clustered:
            reasons.append("holdout confidence bound is not two-way ticker/date clustered")
        if evidence.holdout_resolved_candidates != evidence.holdout.resolved_trades:
            reasons.append("holdout resolution ledger does not match resolved evidence")
        total_candidates = (
            evidence.holdout_resolved_candidates
            + evidence.holdout_unresolved_candidates
        )
        if total_candidates <= 0:
            reasons.append("holdout candidate resolution ledger is empty")
        elif (
            evidence.holdout_resolved_candidates / float(total_candidates)
            < policy.min_holdout_resolution_rate
        ):
            reasons.append(
                "holdout resolution rate is below %.0f percent"
                % (100.0 * policy.min_holdout_resolution_rate)
            )
        if evidence.unresolved_worst_case_expectancy <= 0.0:
            reasons.append("holdout expectancy is not positive under unresolved attrition stress")
        expected_targets = {"POP_NET", "P_TARGET", "P_STOP", "P_MAX_LOSS"}
        event_counts = {
            str(name): (int(positive), int(negative))
            for name, positive, negative in evidence.probability_event_counts
        }
        if set(event_counts) != expected_targets:
            reasons.append("probability event counts are incomplete")
        else:
            for name in sorted(expected_targets):
                positive, negative = event_counts[name]
                if positive < policy.min_probability_positive_events:
                    reasons.append("%s has too few positive outcomes" % name)
                if negative < policy.min_probability_negative_events:
                    reasons.append("%s has too few negative outcomes" % name)
        receipt = evidence.holdout_registry_receipt.lower().removeprefix("sha256:")
        if len(receipt) != 64 or any(char not in "0123456789abcdef" for char in receipt):
            reasons.append("holdout registry receipt is missing or invalid")
        if evidence.model_frozen_at is None:
            reasons.append("model freeze timestamp is missing")
        elif evidence.holdout_evaluated_at is None:
            reasons.append("holdout evaluation timestamp is missing")
        elif evidence.model_frozen_at >= evidence.holdout_evaluated_at:
            reasons.append(
                "model was not frozen before the untouched holdout was opened"
            )
    if target is EvidenceState.SHADOW_PASS:
        if evidence.shadow is None:
            reasons.append("prospective shadow evidence is missing")
        else:
            if evidence.shadow.expectancy <= 0.0:
                reasons.append("shadow expectancy is not positive")
            if evidence.shadow.lower_confidence_bound <= 0.0:
                reasons.append("shadow 90% lower confidence bound is not positive")
            if not math.isclose(
                evidence.shadow.confidence_level,
                0.90,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                reasons.append("shadow lower confidence bound must use 90% confidence")
            if evidence.shadow.resolved_trades < policy.min_shadow_trades:
                reasons.append("shadow has fewer than %d resolved trades" % policy.min_shadow_trades)
        if evidence.shadow_calendar_days < policy.min_shadow_calendar_days:
            reasons.append(
                "shadow period has fewer than %d calendar days"
                % policy.min_shadow_calendar_days
            )
    if target in (EvidenceState.SHADOW_PASS, EvidenceState.MANUAL_TICKET_ENABLED):
        if evidence.pop_ece > policy.max_pop_ece:
            reasons.append("POP expected calibration error exceeds tolerance")
        if evidence.pop_brier_score >= evidence.base_rate_brier_score:
            reasons.append("POP Brier score does not beat the unconditional base rate")
    return tuple(reasons)


def evaluate_promotion(
    evidence: FamilyEvidence,
    target_state: Optional[EvidenceState] = None,
    policy: Optional[PromotionPolicy] = None,
) -> PromotionDecision:
    """Evaluate exactly the next state; skipped transitions are never allowed."""

    policy = policy or PromotionPolicy()
    expected = _DEFAULT_NEXT_STATE.get(evidence.state)
    if expected is None:
        raise ValueError("%s is terminal" % evidence.state.value)
    target = target_state or expected
    assert_transition(evidence.state, target)
    reasons = _reasons_for_target(evidence, target, policy)
    return PromotionDecision(
        current_state=evidence.state,
        target_state=target,
        passed=not reasons,
        reasons=reasons,
    )


def promote_evidence(
    evidence: FamilyEvidence,
    target_state: Optional[EvidenceState] = None,
    policy: Optional[PromotionPolicy] = None,
) -> FamilyEvidence:
    decision = evaluate_promotion(evidence, target_state=target_state, policy=policy)
    if not decision.passed:
        raise ValueError("promotion rejected: " + "; ".join(decision.reasons))
    return replace(evidence, state=decision.target_state)


def validate_shadow_pass(
    evidence: FamilyEvidence, policy: Optional[PromotionPolicy] = None
) -> Tuple[str, ...]:
    """Return cumulative SHADOW_PASS gate failures without changing state."""

    return _reasons_for_target(
        evidence, EvidenceState.SHADOW_PASS, policy or PromotionPolicy()
    )


def validate_holdout_pass(
    evidence: FamilyEvidence, policy: Optional[PromotionPolicy] = None
) -> Tuple[str, ...]:
    """Return cumulative untouched-holdout failures without changing state."""

    return _reasons_for_target(
        evidence, EvidenceState.HOLDOUT_PASS, policy or PromotionPolicy()
    )


__all__ = [
    "BootstrapInterval",
    "ChronologicalSplit",
    "ContributionConcentration",
    "PromotionDecision",
    "PromotionPolicy",
    "WalkForwardFold",
    "WalkForwardPlan",
    "assert_transition",
    "chronological_split",
    "clustered_bootstrap_mean_ci",
    "contribution_concentration",
    "evaluate_promotion",
    "holm_adjust",
    "holm_adjust_mapping",
    "promote_evidence",
    "validate_holdout_pass",
    "validate_shadow_pass",
    "walk_forward_development_splits",
    "walk_forward_splits",
]
