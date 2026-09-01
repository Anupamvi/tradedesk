"""Offline assembly of one current V2 candidate into a manual ticket.

This module is deliberately transport-free.  The caller must supply a verified
frozen model artifact, holdout evidence, saved ORATS analytical provenance, and
fresh Schwab quotes.  It binds those inputs, derives exact one-unit economics,
and delegates the final fail-closed decision to ``build_manual_ticket``.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

from .cache import SnapshotManifest
from .domain import (
    EntryExitPolicy,
    FamilyEvidence,
    LegQuote,
    OptionLeg,
    ProbabilityBundle,
    ProbabilityEstimate,
    Scenario,
    ScenarioOutcome,
    UnderlyingQuote,
)
from .economics import round_trip_costs, same_expiry_payoff_envelope
from .edge import CostBreakdown, PriceConvention, compute_edge
from .hypotheses import FROZEN_HYPOTHESIS_REGISTRY
from .modeling_v2 import (
    EXIT_CATEGORIES,
    MODEL_VERSION,
    ModelingV2Error,
    load_frozen_models_v2,
    score_current_candidate_v2,
)
from .protocol import load_historical_campaign_protocol
from .structures import get_structure_template
from .tickets import (
    CurrentModelCalculation,
    EventEvidence,
    ManualTicket,
    PathwisePayoffArtifact,
    TicketCandidate,
    TicketFieldProfile,
    build_manual_ticket,
)


class CurrentV2Error(ValueError):
    """Current inputs cannot support a reproducible V2 manual ticket."""


_OUTCOME = {
    "TARGET": ScenarioOutcome.TARGET,
    "TIME_PROFIT": ScenarioOutcome.TIME_PROFIT,
    "STOP": ScenarioOutcome.STOP,
    "MAX_LOSS": ScenarioOutcome.MAX_LOSS,
    "TIME_LOSS": ScenarioOutcome.TIME_LOSS,
}


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _model_for_hypothesis(
    models: Mapping[str, Any], hypothesis_id: str, evidence: FamilyEvidence
) -> Mapping[str, Any]:
    matches = tuple(
        item
        for item in models.get("hypotheses", ())
        if isinstance(item, Mapping) and item.get("hypothesis_id") == hypothesis_id
    )
    if len(matches) != 1:
        raise CurrentV2Error("current hypothesis is not uniquely frozen")
    model = matches[0]
    identity = "sha256:" + hashlib.sha256(_canonical(model)).hexdigest()
    if identity.lower().removeprefix(
        "sha256:"
    ) != evidence.pop_model_artifact_id.lower().removeprefix("sha256:"):
        raise CurrentV2Error("current model artifact does not match holdout evidence")
    if str(models.get("model_version")) != evidence.model_version:
        raise CurrentV2Error("current model version does not match holdout evidence")
    if evidence.model_version != MODEL_VERSION:
        raise CurrentV2Error("holdout evidence is not a Cultra V2 model")
    if evidence.strategy_family != hypothesis_id:
        raise CurrentV2Error("holdout evidence does not match the hypothesis")
    return model


def _costs(
    legs: Sequence[OptionLeg],
    quotes: Sequence[LegQuote],
    *,
    assignment_exercise: float,
    dividends: float,
    early_exit: float,
) -> CostBreakdown:
    protocol = load_historical_campaign_protocol()
    policy = protocol["cost_policy"]
    return round_trip_costs(
        legs,
        quotes,
        policy,
        assignment_exercise=assignment_exercise,
        dividends=dividends,
        early_exit=early_exit,
    )


def _payoff(
    legs: Tuple[OptionLeg, ...],
    quotes: Tuple[LegQuote, ...],
    costs: CostBreakdown,
    pathwise: Optional[PathwisePayoffArtifact],
) -> Tuple[Any, float, Optional[float], Tuple[float, ...], PriceConvention, float]:
    if len({item.expiration for item in legs}) == 1:
        if pathwise is not None:
            raise CurrentV2Error(
                "same-expiration structure cannot substitute pathwise economics"
            )
        resolved = same_expiry_payoff_envelope(legs, quotes, costs)
        return (
            resolved,
            resolved.maximum_loss,
            resolved.maximum_profit,
            resolved.breakevens,
            resolved.price_convention,
            resolved.executable_price,
        )
    if pathwise is None:
        raise CurrentV2Error(
            "multi-expiration structure requires a complete pathwise payoff artifact"
        )
    return (
        pathwise,
        pathwise.maximum_loss,
        pathwise.maximum_profit,
        pathwise.breakevens,
        pathwise.price_convention,
        pathwise.executable_price,
    )


def _scenario_pnl(
    *,
    category: str,
    return_on_risk: float,
    maximum_loss: float,
    maximum_profit: Optional[float],
) -> float:
    value = max(-maximum_loss, float(return_on_risk) * maximum_loss)
    if maximum_profit is not None:
        value = min(value, maximum_profit)
    if category == "MAX_LOSS":
        return -maximum_loss
    return value


def _expected_shortfall(
    scenarios: Sequence[Scenario], costs: CostBreakdown, tail_probability: float = 0.05
) -> float:
    remaining = tail_probability
    weighted = 0.0
    for item in sorted(scenarios, key=lambda value: value.net_pnl - costs.total):
        if remaining <= 1e-15:
            break
        consumed = min(remaining, item.probability)
        weighted += consumed * (item.net_pnl - costs.total)
        remaining -= consumed
    if remaining > 1e-12:
        raise CurrentV2Error("scenario distribution cannot fill expected-shortfall tail")
    return max(0.0, -weighted / tail_probability)


def _probabilities(
    score: Mapping[str, Any], *, hypothesis_id: str, artifact_id: str
) -> ProbabilityBundle:
    values = score["probabilities"]
    estimates = {}
    bin_index = min(9, int(float(values["POP_NET"]["point"]) * 10.0))
    bucket_id = "%s|PROJECTED_POP_NET_BIN_%d" % (hypothesis_id, bin_index)
    for target in ("POP_NET", "P_TARGET", "P_STOP", "P_MAX_LOSS"):
        item = values[target]
        interval = item.get("interval")
        period = item.get("calibration_period")
        if not isinstance(interval, Mapping) or not isinstance(period, Mapping):
            raise CurrentV2Error("applicable calibrated probability bucket is unavailable")
        estimates[target] = ProbabilityEstimate(
            point=float(item["point"]),
            lower=float(interval["lower"]),
            upper=float(interval["upper"]),
            sample_size=int(item["applicable_calibration_bin_sample_size"]),
            model_version=str(item["model_version"]),
            calibration_start=date.fromisoformat(str(period["start"])),
            calibration_end=date.fromisoformat(str(period["end"])),
            confidence_level=float(interval["confidence"]),
            interval_method=str(interval["method"]),
            bucket_id=bucket_id,
            artifact_id=artifact_id,
            target_name=target,
        )
    return ProbabilityBundle(
        pop_net=estimates["POP_NET"],
        p_target=estimates["P_TARGET"],
        p_stop=estimates["P_STOP"],
        p_max_loss=estimates["P_MAX_LOSS"],
    )


def build_current_manual_ticket_v2(
    *,
    model_artifact_path: Path,
    hypothesis_id: str,
    features: Mapping[str, float],
    evidence: FamilyEvidence,
    candidate_id: str,
    symbol: str,
    thesis: str,
    signal: str,
    legs: Sequence[OptionLeg],
    leg_quotes: Sequence[LegQuote],
    underlying_quote: UnderlyingQuote,
    orats_snapshot_id: str,
    provider_trade_date: date,
    analytical_fields: Sequence[str],
    snapshot_manifest: SnapshotManifest,
    field_profile: TicketFieldProfile,
    event_evidence: EventEvidence,
    invalidation: str,
    now: datetime,
    pathwise_payoff: Optional[PathwisePayoffArtifact] = None,
    assignment_exercise_cost: float = 0.0,
    dividend_cost: float = 0.0,
    early_exit_cost: float = 0.0,
    max_quote_age: timedelta = timedelta(minutes=5),
) -> ManualTicket:
    """Build a fully gated ticket from supplied, already acquired current inputs.

    No network operation exists in this path.  A negative model result or any
    missing/tampered provenance raises instead of producing a watchlist ticket.
    """

    models = load_frozen_models_v2(Path(model_artifact_path))
    model = _model_for_hypothesis(models, hypothesis_id, evidence)
    definition = next(
        (
            item
            for item in FROZEN_HYPOTHESIS_REGISTRY
            if item.hypothesis_id == hypothesis_id
        ),
        None,
    )
    if definition is None:
        raise CurrentV2Error("hypothesis is outside the frozen registry")
    if model.get("state") != "VALIDATION_PASS":
        raise CurrentV2Error("hypothesis did not pass frozen development validation")
    template = get_structure_template(definition.strategy_id)
    exact_legs = tuple(legs)
    exact_quotes = tuple(leg_quotes)
    costs = _costs(
        exact_legs,
        exact_quotes,
        assignment_exercise=assignment_exercise_cost,
        dividends=dividend_cost,
        early_exit=early_exit_cost,
    )
    if costs.model_version != evidence.cost_model_version:
        raise CurrentV2Error("current cost policy does not match holdout evidence")
    (
        _payoff_artifact,
        maximum_loss,
        maximum_profit,
        breakevens,
        price_convention,
        executable_price,
    ) = _payoff(exact_legs, exact_quotes, costs, pathwise_payoff)
    try:
        score = score_current_candidate_v2(
            models,
            hypothesis_id=hypothesis_id,
            features=features,
            finite_maximum_loss=maximum_loss,
        )
    except ModelingV2Error as exc:
        raise CurrentV2Error(str(exc)) from exc
    if not bool(score["model_candidate_eligible"]):
        raise CurrentV2Error("current candidate does not have positive conservative V2 edge")

    point = []
    conservative = []
    probabilities_by_category = score["joint_exit_probabilities"]
    for category in EXIT_CATEGORIES:
        probability = float(probabilities_by_category[category])
        point_after_cost = _scenario_pnl(
            category=category,
            return_on_risk=float(score["scenario_net_returns_on_risk"][category]),
            maximum_loss=maximum_loss,
            maximum_profit=maximum_profit,
        )
        conservative_after_cost = _scenario_pnl(
            category=category,
            return_on_risk=float(
                score["conservative_scenario_net_returns_on_risk"][category]
            ),
            maximum_loss=maximum_loss,
            maximum_profit=maximum_profit,
        )
        conservative_after_cost = min(conservative_after_cost, point_after_cost)
        if probability > 1e-12:
            if category in {"TARGET", "TIME_PROFIT"} and point_after_cost <= 0.0:
                raise CurrentV2Error(
                    "historical scenario profile contradicts the profit outcome class"
                )
            if category in {"STOP", "MAX_LOSS", "TIME_LOSS"} and point_after_cost >= 0.0:
                raise CurrentV2Error(
                    "historical scenario profile contradicts the loss outcome class"
                )
        point.append(
            Scenario(
                category.lower(),
                probability,
                point_after_cost + costs.total,
                _OUTCOME[category],
            )
        )
        conservative.append(
            Scenario(
                category.lower(),
                probability,
                conservative_after_cost + costs.total,
                _OUTCOME[category],
            )
        )

    target_pnl = template.target_fraction_of_risk * maximum_loss
    if maximum_profit is not None:
        target_pnl = min(target_pnl, maximum_profit)
    if target_pnl <= 0.0:
        raise CurrentV2Error("exact structure has no positive target region")
    stop_pnl = -min(template.stop_fraction_of_risk * maximum_loss, maximum_loss)
    shortfall = _expected_shortfall(point, costs)
    if shortfall <= 0.0:
        raise CurrentV2Error("current scenario distribution has no measurable loss tail")
    point_net_ev = math.fsum(
        item.probability * (item.net_pnl - costs.total) for item in point
    )
    if price_convention is PriceConvention.DEBIT:
        model_fair_price = max(0.0, executable_price + point_net_ev / 100.0)
    else:
        model_fair_price = max(0.0, executable_price - point_net_ev / 100.0)
    edge = compute_edge(
        point,
        conservative,
        maximum_loss=maximum_loss,
        costs=costs,
        model_fair_price=model_fair_price,
        executable_limit_price=executable_price,
        price_convention=price_convention,
        maximum_profit=maximum_profit,
        breakevens=breakevens,
        target_pnl=target_pnl,
        stop_pnl=stop_pnl,
        expected_shortfall=shortfall,
        adverse_gap_stress_loss=maximum_loss,
    )
    if not edge.is_positive:
        raise CurrentV2Error(
            "current exact-leg point and conservative net EV are not both positive"
        )
    probability_bundle = _probabilities(
        score,
        hypothesis_id=hypothesis_id,
        artifact_id=evidence.pop_model_artifact_id,
    )
    policy = EntryExitPolicy(
        entry_condition=(
            "All exact legs fill together at or better than the stated Schwab natural limit"
        ),
        profit_target="Close at +%.2f dollars net per unit" % target_pnl,
        stop_condition="Close at %.2f dollars net per unit; stop wins same-day ambiguity"
        % stop_pnl,
        time_exit="Close after %d market sessions" % definition.holding_sessions,
        invalidation=invalidation,
        assignment_handling=(
            "Close before exercise; if assignment or early exercise occurs, stop and perform manual review"
        ),
        next_review=event_evidence.holding_window_start,
        policy_version=definition.exit_policy,
        time_exit_sessions=definition.holding_sessions,
    )
    model_calculation = CurrentModelCalculation(
        calculation_version="CULTRA_CURRENT_MODEL_CALCULATION_V2",
        hypothesis_id=hypothesis_id,
        model_version=str(score["model_version"]),
        model_artifact_id=evidence.pop_model_artifact_id,
        features=tuple(
            sorted((str(name), float(value)) for name, value in features.items())
        ),
        selection_point_return_on_max_loss=float(
            score["selection_model_point_return_on_maximum_loss"]
        ),
        selection_conservative_return_on_max_loss=float(
            score["selection_model_conservative_return_on_maximum_loss"]
        ),
        scenario_point_return_on_max_loss=float(
            score["point_expected_return_on_maximum_loss"]
        ),
        scenario_conservative_return_on_max_loss=float(
            score["conservative_expected_return_on_maximum_loss"]
        ),
        probability_projection_l1_distance=float(
            score["probability_projection_l1_distance"]
        ),
        joint_exit_probabilities=tuple(
            sorted(
                (str(name), float(value))
                for name, value in score["joint_exit_probabilities"].items()
            )
        ),
        scenario_net_returns_on_risk=tuple(
            sorted(
                (str(name), float(value))
                for name, value in score["scenario_net_returns_on_risk"].items()
            )
        ),
        conservative_scenario_net_returns_on_risk=tuple(
            sorted(
                (str(name), float(value))
                for name, value in score[
                    "conservative_scenario_net_returns_on_risk"
                ].items()
            )
        ),
    )
    candidate = TicketCandidate(
        candidate_id=candidate_id,
        symbol=symbol.strip().upper(),
        thesis=thesis,
        signal=signal,
        strategy_id=definition.strategy_id,
        hypothesis_id=hypothesis_id,
        evidence=evidence,
        legs=exact_legs,
        leg_quotes=exact_quotes,
        underlying_quote=underlying_quote,
        orats_snapshot_id=orats_snapshot_id,
        provider_trade_date=provider_trade_date,
        analytical_fields=tuple(analytical_fields),
        probabilities=probability_bundle,
        edge=edge,
        policy=policy,
        event_evidence=event_evidence,
        model_calculation=model_calculation,
        snapshot_manifest=snapshot_manifest,
        field_profile=field_profile,
        pathwise_payoff=pathwise_payoff,
    )
    return build_manual_ticket(candidate, now, max_quote_age=max_quote_age)


__all__ = ["CurrentV2Error", "build_current_manual_ticket_v2"]
