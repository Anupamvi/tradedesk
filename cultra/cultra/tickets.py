"""Fail-closed construction of manual-review option tickets."""

from dataclasses import dataclass, field, fields, is_dataclass
from datetime import date, datetime, timedelta
from enum import Enum
import hashlib
import json
import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .cache import SnapshotManifest
from .catalog import CATALOG_VERSION, StrategyDefinition, get_strategy
from .domain import (
    EntryExitPolicy,
    EvidenceState,
    FamilyEvidence,
    LegAction,
    LegQuote,
    OptionLeg,
    OptionType,
    ProbabilityBundle,
    ScenarioOutcome,
    UnderlyingQuote,
    exact_quote_map,
    parse_occ_symbol,
)
from .economics import EconomicsError, PayoffEnvelope, same_expiry_payoff_envelope
from .edge import EdgeEstimate, PriceConvention
from .hypotheses import FROZEN_HYPOTHESIS_REGISTRY
from .validation import PromotionPolicy, validate_holdout_pass, validate_shadow_pass


QUANTITY_USER_DETERMINED = "USER DETERMINED"


def _canonical_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _hash_text(value: str) -> bool:
    normalized = value[7:] if value.startswith("sha256:") else value
    return len(normalized) == 64 and all(
        char in "0123456789abcdef" for char in normalized
    )


@dataclass(frozen=True)
class EventEvidence:
    """Pipeline-resolved earnings/dividend clearance for the holding window."""

    asset_type: str
    source: str
    source_timestamp: datetime
    holding_window_start: date
    holding_window_end: date
    market_sessions: Tuple[date, ...]
    earnings_date: Optional[date]
    dividend_dates: Tuple[date, ...]
    status: str
    artifact_id: str

    def __post_init__(self) -> None:
        asset_type = str(self.asset_type).upper().strip()
        if asset_type not in {"STOCK", "ETF"}:
            raise ValueError("event evidence asset_type must be STOCK or ETF")
        object.__setattr__(self, "asset_type", asset_type)
        if not self.source.strip():
            raise ValueError("event evidence source is required")
        if self.source_timestamp.tzinfo is None or self.source_timestamp.utcoffset() is None:
            raise ValueError("event evidence timestamp must be timezone-aware")
        if self.holding_window_end < self.holding_window_start:
            raise ValueError("event holding window is reversed")
        sessions = tuple(self.market_sessions)
        if not sessions or sessions != tuple(sorted(set(sessions))):
            raise ValueError("event market sessions must be non-empty, sorted and unique")
        if (
            sessions[0] != self.holding_window_start
            or sessions[-1] != self.holding_window_end
        ):
            raise ValueError("event holding window must match its market-session calendar")
        dividends = tuple(sorted(set(self.dividend_dates)))
        if dividends != self.dividend_dates:
            raise ValueError("dividend dates must be sorted and unique")
        if asset_type == "STOCK" and self.earnings_date is None:
            raise ValueError("stock event evidence requires the next earnings date")
        expected_status = "CLEAR"
        if self.earnings_date is not None and (
            self.holding_window_start <= self.earnings_date <= self.holding_window_end
        ):
            expected_status = "BLOCKED_EARNINGS_IN_HOLDING_WINDOW"
        if self.status != expected_status:
            raise ValueError("event evidence status does not match the frozen dates")
        if not _hash_text(self.artifact_id):
            raise ValueError("event evidence must bind a hashed artifact")


@dataclass(frozen=True)
class TicketFieldProfile:
    """Entitlement-verified field profile bound to a validated snapshot."""

    profile_id: str
    schema_version: str
    fields: Tuple[str, ...]
    concept_mapping: Tuple[Tuple[str, str], ...]
    status: str = "ENTITLEMENT_VERIFIED"

    def __post_init__(self) -> None:
        if not self.profile_id.strip() or not self.schema_version.strip():
            raise ValueError("field profile identity is required")
        if tuple(sorted(set(self.fields))) != self.fields or not self.fields:
            raise ValueError("field profile fields must be non-empty, sorted and unique")
        concepts = tuple(sorted(set(self.concept_mapping)))
        if concepts != self.concept_mapping or not concepts:
            raise ValueError("field profile concept mapping must be sorted and unique")
        if any(not concept or not field for concept, field in concepts):
            raise ValueError("field profile concept mapping is incomplete")
        if self.status != "ENTITLEMENT_VERIFIED":
            raise ValueError("ticket field profile must be entitlement verified")

    @property
    def profile_hash(self) -> str:
        return _canonical_hash(
            {
                "profile_id": self.profile_id,
                "schema_version": self.schema_version,
                "fields": self.fields,
                "concept_mapping": self.concept_mapping,
                "status": self.status,
            }
        )


@dataclass(frozen=True)
class PathwisePayoffArtifact:
    """Frozen multi-expiration risk proof; static expiry payoff is insufficient."""

    calculation_version: str
    exact_occ_symbols: Tuple[str, ...]
    executable_price: float
    price_convention: PriceConvention
    maximum_profit: Optional[float]
    maximum_loss: float
    breakevens: Tuple[float, ...]
    adverse_gap_stress_loss: float
    scenario_count: int
    includes_assignment: bool
    includes_early_exercise: bool
    includes_dividends: bool
    includes_volatility_shock: bool
    includes_liquidity_collapse: bool
    includes_gap: bool
    includes_partial_fill: bool

    def __post_init__(self) -> None:
        if not self.calculation_version.strip():
            raise ValueError("pathwise calculation version is required")
        if tuple(sorted(set(self.exact_occ_symbols))) != self.exact_occ_symbols:
            raise ValueError("pathwise OCC symbols must be sorted and unique")
        if not self.exact_occ_symbols:
            raise ValueError("pathwise payoff requires exact OCC symbols")
        for value, name in (
            (self.executable_price, "executable_price"),
            (self.maximum_loss, "maximum_loss"),
            (self.adverse_gap_stress_loss, "adverse_gap_stress_loss"),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError("pathwise %s must be finite and positive" % name)
        if self.maximum_profit is not None and (
            not math.isfinite(float(self.maximum_profit))
            or float(self.maximum_profit) < 0.0
        ):
            raise ValueError("pathwise maximum_profit is invalid")
        if not self.breakevens:
            raise ValueError("pathwise payoff requires breakevens")
        if self.scenario_count < 100:
            raise ValueError("pathwise payoff requires at least 100 stress scenarios")
        coverage = (
            self.includes_assignment,
            self.includes_early_exercise,
            self.includes_dividends,
            self.includes_volatility_shock,
            self.includes_liquidity_collapse,
            self.includes_gap,
            self.includes_partial_fill,
        )
        if not all(coverage):
            raise ValueError("pathwise payoff stress coverage is incomplete")

    @property
    def artifact_id(self) -> str:
        return _canonical_hash(_jsonable(self))


@dataclass(frozen=True)
class CurrentModelCalculation:
    """Immutable current feature vector and frozen V2 score output."""

    calculation_version: str
    hypothesis_id: str
    model_version: str
    model_artifact_id: str
    features: Tuple[Tuple[str, float], ...]
    selection_point_return_on_max_loss: float
    selection_conservative_return_on_max_loss: float
    scenario_point_return_on_max_loss: float
    scenario_conservative_return_on_max_loss: float
    probability_projection_l1_distance: float
    joint_exit_probabilities: Tuple[Tuple[str, float], ...]
    scenario_net_returns_on_risk: Tuple[Tuple[str, float], ...]
    conservative_scenario_net_returns_on_risk: Tuple[Tuple[str, float], ...]
    calculation_id: str = field(init=False)

    def __post_init__(self) -> None:
        for name in ("calculation_version", "hypothesis_id", "model_version"):
            if not str(getattr(self, name)).strip():
                raise ValueError("%s is required" % name)
        if not _hash_text(self.model_artifact_id):
            raise ValueError("current model calculation requires a hashed artifact")
        expected_categories = {
            "TARGET",
            "TIME_PROFIT",
            "STOP",
            "MAX_LOSS",
            "TIME_LOSS",
        }

        def normalized_pairs(
            values: Tuple[Tuple[str, float], ...], label: str, names: Optional[set] = None
        ) -> Tuple[Tuple[str, float], ...]:
            converted = tuple((str(name), float(value)) for name, value in values)
            if not converted or converted != tuple(sorted(converted)):
                raise ValueError("%s must be non-empty, sorted and unique" % label)
            if len({name for name, _value in converted}) != len(converted):
                raise ValueError("%s names must be unique" % label)
            if names is not None and {name for name, _value in converted} != names:
                raise ValueError("%s does not cover every exit category" % label)
            if not all(math.isfinite(value) for _name, value in converted):
                raise ValueError("%s contains a non-finite value" % label)
            return converted

        features = normalized_pairs(self.features, "current model features")
        point_probabilities = normalized_pairs(
            self.joint_exit_probabilities,
            "joint exit probabilities",
            expected_categories,
        )
        point_returns = normalized_pairs(
            self.scenario_net_returns_on_risk,
            "scenario returns",
            expected_categories,
        )
        conservative_returns = normalized_pairs(
            self.conservative_scenario_net_returns_on_risk,
            "conservative scenario returns",
            expected_categories,
        )
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "joint_exit_probabilities", point_probabilities)
        object.__setattr__(self, "scenario_net_returns_on_risk", point_returns)
        object.__setattr__(
            self,
            "conservative_scenario_net_returns_on_risk",
            conservative_returns,
        )
        probabilities = dict(point_probabilities)
        if any(value < 0.0 or value > 1.0 for value in probabilities.values()) or not math.isclose(
            math.fsum(probabilities.values()), 1.0, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError("joint exit probabilities must form one distribution")
        point_by_category = dict(point_returns)
        conservative_by_category = dict(conservative_returns)
        if any(
            conservative_by_category[name] > point_by_category[name] + 1e-12
            for name in expected_categories
        ):
            raise ValueError("conservative category return cannot exceed point return")
        numeric_fields = (
            "selection_point_return_on_max_loss",
            "selection_conservative_return_on_max_loss",
            "scenario_point_return_on_max_loss",
            "scenario_conservative_return_on_max_loss",
            "probability_projection_l1_distance",
        )
        for name in numeric_fields:
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError("%s must be finite" % name)
            object.__setattr__(self, name, value)
        if self.probability_projection_l1_distance < 0.0:
            raise ValueError("probability projection distance cannot be negative")
        if (
            self.selection_conservative_return_on_max_loss
            > self.selection_point_return_on_max_loss + 1e-12
            or self.scenario_conservative_return_on_max_loss
            > self.scenario_point_return_on_max_loss + 1e-12
        ):
            raise ValueError("conservative model return cannot exceed point return")
        reproduced_point = math.fsum(
            probabilities[name] * point_by_category[name]
            for name in expected_categories
        )
        reproduced_conservative = math.fsum(
            probabilities[name] * conservative_by_category[name]
            for name in expected_categories
        )
        if not math.isclose(
            self.scenario_point_return_on_max_loss,
            reproduced_point,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ) or not math.isclose(
            self.scenario_conservative_return_on_max_loss,
            reproduced_conservative,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("current scenario model return is not reproducible")
        object.__setattr__(
            self,
            "calculation_id",
            _canonical_hash(
                {
                    "calculation_version": self.calculation_version,
                    "hypothesis_id": self.hypothesis_id,
                    "model_version": self.model_version,
                    "model_artifact_id": self.model_artifact_id,
                    "features": self.features,
                    "selection_point_return_on_max_loss": self.selection_point_return_on_max_loss,
                    "selection_conservative_return_on_max_loss": self.selection_conservative_return_on_max_loss,
                    "scenario_point_return_on_max_loss": self.scenario_point_return_on_max_loss,
                    "scenario_conservative_return_on_max_loss": self.scenario_conservative_return_on_max_loss,
                    "probability_projection_l1_distance": self.probability_projection_l1_distance,
                    "joint_exit_probabilities": self.joint_exit_probabilities,
                    "scenario_net_returns_on_risk": self.scenario_net_returns_on_risk,
                    "conservative_scenario_net_returns_on_risk": self.conservative_scenario_net_returns_on_risk,
                }
            ),
        )


@dataclass(frozen=True)
class TicketCandidate:
    candidate_id: str
    symbol: str
    thesis: str
    signal: str
    strategy_id: str
    hypothesis_id: str
    evidence: FamilyEvidence
    legs: Tuple[OptionLeg, ...]
    leg_quotes: Tuple[LegQuote, ...]
    underlying_quote: UnderlyingQuote
    orats_snapshot_id: str
    provider_trade_date: date
    analytical_fields: Tuple[str, ...]
    probabilities: ProbabilityBundle
    edge: EdgeEstimate
    policy: EntryExitPolicy
    event_evidence: EventEvidence
    model_calculation: Optional[CurrentModelCalculation] = None
    quote_source: str = "SCHWAB"
    snapshot_manifest: Optional[SnapshotManifest] = None
    field_profile: Optional[TicketFieldProfile] = None
    pathwise_payoff: Optional[PathwisePayoffArtifact] = None


@dataclass(frozen=True)
class ManualTicket:
    candidate_id: str
    symbol: str
    thesis: str
    signal: str
    strategy_id: str
    hypothesis_id: str
    evidence: FamilyEvidence
    evidence_state: EvidenceState
    legs: Tuple[OptionLeg, ...]
    leg_quotes: Tuple[LegQuote, ...]
    underlying_quote: UnderlyingQuote
    orats_snapshot_id: str
    provider_trade_date: date
    analytical_fields: Tuple[str, ...]
    snapshot_manifest: SnapshotManifest
    field_profile: TicketFieldProfile
    payoff_evidence: Any
    probabilities: ProbabilityBundle
    edge: EdgeEstimate
    policy: EntryExitPolicy
    event_evidence: EventEvidence
    model_calculation: CurrentModelCalculation
    quote_source: str
    quantity: str
    ranking_score: float
    created_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        payload = _jsonable(self)
        for quote, quote_payload in zip(self.leg_quotes, payload["leg_quotes"]):
            quote_payload["spread"] = quote.spread
        payload["underlying_quote"]["spread"] = (
            self.underlying_quote.ask - self.underlying_quote.bid
        )
        return payload


class TicketRejection(ValueError):
    def __init__(self, reasons: Sequence[str]):
        self.reasons = tuple(reasons)
        super().__init__("manual ticket rejected: " + "; ".join(self.reasons))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _leg_shape(leg: OptionLeg) -> Tuple[LegAction, OptionType, date, float, int]:
    return leg.action, leg.option_type, leg.expiration, leg.strike, leg.ratio


def _same_expiration(legs: Sequence[OptionLeg]) -> bool:
    return len({leg.expiration for leg in legs}) == 1


def _validate_two_leg_directional(
    strategy_id: str, legs: Sequence[OptionLeg], reasons: list
) -> None:
    buys = [leg for leg in legs if leg.action is LegAction.BUY]
    sells = [leg for leg in legs if leg.action is LegAction.SELL]
    if len(buys) != 1 or len(sells) != 1:
        reasons.append("strategy requires exactly one buy leg and one sell leg")
        return
    buy, sell = buys[0], sells[0]
    expected_type = OptionType.CALL if "CALL" in strategy_id else OptionType.PUT
    if buy.option_type is not expected_type or sell.option_type is not expected_type:
        reasons.append("leg option types do not match strategy")
    if "VERTICAL" in strategy_id and buy.expiration != sell.expiration:
        reasons.append("vertical legs must share one expiration")
    if strategy_id == "CALL_DEBIT_VERTICAL" and not buy.strike < sell.strike:
        reasons.append("call debit vertical requires bought strike below sold strike")
    elif strategy_id == "PUT_DEBIT_VERTICAL" and not buy.strike > sell.strike:
        reasons.append("put debit vertical requires bought strike above sold strike")
    elif strategy_id == "CALL_CREDIT_VERTICAL" and not sell.strike < buy.strike:
        reasons.append("call credit vertical requires sold strike below bought strike")
    elif strategy_id == "PUT_CREDIT_VERTICAL" and not sell.strike > buy.strike:
        reasons.append("put credit vertical requires sold strike above bought strike")
    elif strategy_id.endswith("DIAGONAL"):
        if not buy.expiration > sell.expiration:
            reasons.append("diagonal requires bought expiration after sold expiration")
        if buy.strike == sell.strike:
            reasons.append("diagonal requires different strikes; use calendar for equal strikes")
    elif strategy_id.endswith("CALENDAR"):
        if not buy.expiration > sell.expiration:
            reasons.append("calendar requires bought expiration after sold expiration")
        if buy.strike != sell.strike:
            reasons.append("calendar legs must share one strike")


def _validate_call_put_pair(
    strategy_id: str, legs: Sequence[OptionLeg], reasons: list
) -> None:
    if not _same_expiration(legs):
        reasons.append("paired call and put legs must share one expiration")
    calls = [leg for leg in legs if leg.option_type is OptionType.CALL]
    puts = [leg for leg in legs if leg.option_type is OptionType.PUT]
    if len(calls) != 1 or len(puts) != 1:
        reasons.append("strategy requires one call and one put")
        return
    if any(leg.action is not LegAction.BUY for leg in legs):
        reasons.append("long volatility strategy requires bought legs")
    if strategy_id == "LONG_STRADDLE" and calls[0].strike != puts[0].strike:
        reasons.append("long straddle legs must share one strike")
    if strategy_id == "LONG_STRANGLE" and not puts[0].strike < calls[0].strike:
        reasons.append("long strangle requires put strike below call strike")


def _validate_iron_shape(
    strategy_id: str, legs: Sequence[OptionLeg], reasons: list
) -> None:
    if not _same_expiration(legs):
        reasons.append("iron structure legs must share one expiration")
    calls = sorted(
        (leg for leg in legs if leg.option_type is OptionType.CALL),
        key=lambda leg: leg.strike,
    )
    puts = sorted(
        (leg for leg in legs if leg.option_type is OptionType.PUT),
        key=lambda leg: leg.strike,
    )
    if len(calls) != 2 or len(puts) != 2:
        reasons.append("iron structure requires two calls and two puts")
        return
    if not (
        puts[0].action is LegAction.BUY
        and puts[1].action is LegAction.SELL
        and calls[0].action is LegAction.SELL
        and calls[1].action is LegAction.BUY
    ):
        reasons.append("iron structure must buy outer wings and sell inner legs")
    fly = strategy_id in ("IRON_FLY", "WING_CAPPED_SHORT_STRADDLE")
    if fly:
        if puts[1].strike != calls[0].strike:
            reasons.append("iron-fly short call and put must share one strike")
    elif not puts[1].strike < calls[0].strike:
        reasons.append("iron-condor short put strike must be below short call strike")


def _validate_fly_shape(
    strategy_id: str, legs: Sequence[OptionLeg], reasons: list
) -> None:
    expected_type = OptionType.CALL if "CALL" in strategy_id else OptionType.PUT
    if any(leg.option_type is not expected_type for leg in legs):
        reasons.append("butterfly legs do not match strategy option type")
    if not _same_expiration(legs):
        reasons.append("butterfly legs must share one expiration")
    ordered = sorted(legs, key=lambda leg: leg.strike)
    if len({leg.strike for leg in ordered}) != 3:
        reasons.append("butterfly requires three distinct strikes")
        return
    if [leg.action for leg in ordered] != [LegAction.BUY, LegAction.SELL, LegAction.BUY]:
        reasons.append("butterfly must buy outer legs and sell the middle leg")
    if [leg.ratio for leg in ordered] != [1, 2, 1]:
        reasons.append("butterfly requires a 1:2:1 ratio")
    low_width = ordered[1].strike - ordered[0].strike
    high_width = ordered[2].strike - ordered[1].strike
    broken = "BROKEN_WING" in strategy_id or "WING_CAPPED" in strategy_id
    if "BUTTERFLY" in strategy_id and not broken and not math.isclose(low_width, high_width):
        reasons.append("standard butterfly wing widths must be equal")
    if "BROKEN_WING" in strategy_id and math.isclose(low_width, high_width):
        reasons.append("broken-wing butterfly widths must differ")


def _validate_backspread(
    strategy_id: str, legs: Sequence[OptionLeg], reasons: list
) -> None:
    expected_type = OptionType.CALL if "CALL" in strategy_id else OptionType.PUT
    if any(leg.option_type is not expected_type for leg in legs):
        reasons.append("backspread legs do not match strategy option type")
    if not _same_expiration(legs):
        reasons.append("backspread legs must share one expiration")
    buys = [leg for leg in legs if leg.action is LegAction.BUY]
    sells = [leg for leg in legs if leg.action is LegAction.SELL]
    if len(buys) != 1 or len(sells) != 1 or buys[0].ratio != 2 or sells[0].ratio != 1:
        reasons.append("backspread requires buy-two/sell-one legs")
        return
    if expected_type is OptionType.CALL and not sells[0].strike < buys[0].strike:
        reasons.append("call backspread requires bought calls above sold call")
    if expected_type is OptionType.PUT and not buys[0].strike < sells[0].strike:
        reasons.append("put backspread requires bought puts below sold put")


def _validate_geometry(
    definition: StrategyDefinition, legs: Sequence[OptionLeg]
) -> Tuple[str, ...]:
    reasons = []
    strategy_id = definition.strategy_id
    if len(legs) != definition.leg_count:
        return ("strategy requires %d distinct legs" % definition.leg_count,)
    if len({leg.occ_symbol for leg in legs}) != len(legs):
        reasons.append("exact OCC leg symbols must be unique")
    if strategy_id in ("LONG_CALL", "LONG_PUT"):
        expected_type = OptionType.CALL if strategy_id == "LONG_CALL" else OptionType.PUT
        if legs[0].action is not LegAction.BUY or legs[0].option_type is not expected_type:
            reasons.append("long option strategy requires one bought matching option")
    elif strategy_id.endswith("VERTICAL") or strategy_id.endswith("DIAGONAL") or strategy_id.endswith("CALENDAR"):
        _validate_two_leg_directional(strategy_id, legs, reasons)
    elif strategy_id in ("LONG_STRADDLE", "LONG_STRANGLE"):
        _validate_call_put_pair(strategy_id, legs, reasons)
    elif strategy_id in (
        "IRON_FLY",
        "IRON_CONDOR",
        "WING_CAPPED_SHORT_STRADDLE",
        "WING_CAPPED_SHORT_STRANGLE",
    ):
        _validate_iron_shape(strategy_id, legs, reasons)
    elif "BUTTERFLY" in strategy_id or strategy_id in (
        "WING_CAPPED_CALL_RATIO",
        "WING_CAPPED_PUT_RATIO",
    ):
        _validate_fly_shape(strategy_id, legs, reasons)
    elif strategy_id in ("CALL_BACKSPREAD", "PUT_BACKSPREAD"):
        _validate_backspread(strategy_id, legs, reasons)
    return tuple(reasons)


def _natural_price(candidate: TicketCandidate, reasons: list) -> Optional[float]:
    quote_map = exact_quote_map(candidate.leg_quotes)
    signed_debit = 0.0
    for leg in candidate.legs:
        quote = quote_map[leg.occ_symbol]
        if leg.action is LegAction.BUY:
            if quote.ask <= 0.0:
                reasons.append("buy leg %s has no executable ask" % leg.occ_symbol)
            signed_debit += quote.ask * leg.ratio
        else:
            if quote.bid <= 0.0:
                reasons.append("sell leg %s has no executable bid" % leg.occ_symbol)
            signed_debit -= quote.bid * leg.ratio
    if candidate.edge.price_convention is PriceConvention.DEBIT:
        if signed_debit <= 0.0:
            reasons.append("quoted legs do not form a positive natural debit")
            return None
        return signed_debit
    natural_credit = -signed_debit
    if natural_credit <= 0.0:
        reasons.append("quoted legs do not form a positive natural credit")
        return None
    return natural_credit


def _scenario_probability_reconciliation(
    candidate: TicketCandidate, reasons: list
) -> None:
    scenarios = candidate.edge.point_scenarios
    if not scenarios:
        return
    if any(item.outcome is ScenarioOutcome.UNCLASSIFIED for item in scenarios):
        reasons.append("every EV scenario requires a frozen exit-outcome class")
        return
    after_cost = tuple(
        (item, item.net_pnl - candidate.edge.costs.total) for item in scenarios
    )
    implied = {
        "POP_net": math.fsum(item.probability for item, pnl in after_cost if pnl > 0.0),
        "P_target": math.fsum(
            item.probability
            for item, _pnl in after_cost
            if item.outcome is ScenarioOutcome.TARGET
        ),
        "P_stop": math.fsum(
            item.probability
            for item, _pnl in after_cost
            if item.outcome in (ScenarioOutcome.STOP, ScenarioOutcome.MAX_LOSS)
        ),
        "P_max_loss": math.fsum(
            item.probability
            for item, _pnl in after_cost
            if item.outcome is ScenarioOutcome.MAX_LOSS
        ),
    }
    supplied = {
        "POP_net": candidate.probabilities.pop_net.point,
        "P_target": candidate.probabilities.p_target.point,
        "P_stop": candidate.probabilities.p_stop.point,
        "P_max_loss": candidate.probabilities.p_max_loss.point,
    }
    tolerance = max(0.05, float(candidate.evidence.pop_ece))
    for name in ("POP_net", "P_target", "P_stop", "P_max_loss"):
        if abs(implied[name] - supplied[name]) > tolerance + 1e-12:
            reasons.append(
                "%s does not reconcile to the saved point scenario distribution" % name
            )


def _expected_shortfall_loss(edge: EdgeEstimate, tail_probability: float = 0.05) -> float:
    """Calculate the positive dollar loss in the worst probability tail."""

    remaining = tail_probability
    weighted = 0.0
    for scenario in sorted(
        edge.point_scenarios, key=lambda item: item.net_pnl - edge.costs.total
    ):
        if remaining <= 1e-15:
            break
        consumed = min(remaining, scenario.probability)
        weighted += consumed * (scenario.net_pnl - edge.costs.total)
        remaining -= consumed
    if remaining > 1e-12:
        raise ValueError("scenario distribution cannot fill expected-shortfall tail")
    average = weighted / tail_probability
    return max(0.0, -average)


def _resolved_payoff(
    candidate: TicketCandidate, reasons: list
) -> Optional[Any]:
    if len({leg.expiration for leg in candidate.legs}) == 1:
        try:
            payoff = same_expiry_payoff_envelope(
                candidate.legs, candidate.leg_quotes, candidate.edge.costs
            )
        except (EconomicsError, ValueError) as exc:
            reasons.append(str(exc))
            return None
        if candidate.pathwise_payoff is not None:
            reasons.append("same-expiration ticket cannot substitute a pathwise payoff")
        return payoff
    payoff = candidate.pathwise_payoff
    if payoff is None:
        reasons.append(
            "multi-expiration ticket requires a frozen complete pathwise payoff artifact"
        )
        return None
    if payoff.exact_occ_symbols != tuple(
        sorted(leg.occ_symbol for leg in candidate.legs)
    ):
        reasons.append("pathwise payoff exact OCC legs do not match the ticket")
    return payoff


def _compare_payoff_to_edge(
    payoff: Any, edge: EdgeEstimate, reasons: list
) -> None:
    if payoff.price_convention is not edge.price_convention:
        reasons.append("derived payoff price convention does not match edge")
    if not math.isclose(
        payoff.executable_price,
        edge.executable_limit_price,
        rel_tol=0.0,
        abs_tol=0.011,
    ):
        reasons.append("derived executable price does not match edge")
    if not math.isclose(
        payoff.maximum_loss, edge.maximum_loss, rel_tol=0.0, abs_tol=0.011
    ):
        reasons.append("maximum loss does not reconcile to exact-leg payoff")
    if payoff.maximum_profit is None:
        if edge.maximum_profit is not None:
            reasons.append("maximum profit should be unbounded for the exact-leg payoff")
    elif edge.maximum_profit is None or not math.isclose(
        payoff.maximum_profit, edge.maximum_profit, rel_tol=0.0, abs_tol=0.011
    ):
        reasons.append("maximum profit does not reconcile to exact-leg payoff")
    if len(payoff.breakevens) != len(edge.breakevens) or any(
        not math.isclose(left, right, rel_tol=0.0, abs_tol=0.011)
        for left, right in zip(payoff.breakevens, edge.breakevens)
    ):
        reasons.append("breakevens do not reconcile to exact-leg payoff")


def build_manual_ticket(
    candidate: TicketCandidate,
    now: datetime,
    max_quote_age: timedelta = timedelta(minutes=5),
    max_event_age: timedelta = timedelta(days=7),
    promotion_policy: Optional[PromotionPolicy] = None,
) -> ManualTicket:
    """Build one ticket only when every mandatory datum passes closed gates."""

    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    if max_quote_age <= timedelta(0):
        raise ValueError("max_quote_age must be positive")
    if max_event_age <= timedelta(0):
        raise ValueError("max_event_age must be positive")
    policy = promotion_policy or PromotionPolicy()
    reasons = []
    for name in ("candidate_id", "symbol", "thesis", "signal", "orats_snapshot_id"):
        value = getattr(candidate, name)
        if not value or not value.strip():
            reasons.append("%s is required" % name)
    if candidate.quote_source != "SCHWAB":
        reasons.append("quote_source must be SCHWAB")
    if candidate.underlying_quote.symbol != candidate.symbol:
        reasons.append("underlying quote symbol does not match candidate")
    if candidate.provider_trade_date > now.date():
        reasons.append("provider trade date cannot be in the future")
    if not candidate.analytical_fields or any(
        not field or not field.strip() for field in candidate.analytical_fields
    ):
        reasons.append("analytical_fields must be complete")
    if len(candidate.analytical_fields) != len(set(candidate.analytical_fields)):
        reasons.append("analytical_fields must not contain duplicates")
    normalized_symbol = candidate.symbol.strip().upper()
    for leg in candidate.legs:
        try:
            root, _expiration, _option_type, _strike = parse_occ_symbol(
                leg.occ_symbol
            )
        except ValueError:
            continue
        if root != normalized_symbol:
            reasons.append("exact OCC leg root does not match candidate symbol")

    snapshot = candidate.snapshot_manifest
    profile = candidate.field_profile
    if snapshot is None:
        reasons.append("validated ORATS snapshot manifest is required")
    else:
        if candidate.orats_snapshot_id != snapshot.snapshot_id:
            reasons.append("ORATS snapshot id does not resolve to its manifest")
        if tuple(snapshot.provider_trade_dates) != (
            candidate.provider_trade_date.isoformat(),
        ):
            reasons.append("provider trade date does not match the snapshot manifest")
        if snapshot.row_count <= 0:
            reasons.append("ORATS snapshot manifest contains no validated rows")
    if profile is None:
        reasons.append("entitlement-verified field profile is required")
    elif snapshot is not None:
        if profile.profile_id != snapshot.field_profile:
            reasons.append("field profile does not match the ORATS snapshot")
        if profile.schema_version != snapshot.schema_version:
            reasons.append("field-profile schema does not match the ORATS snapshot")
        if not set(candidate.analytical_fields).issubset(set(profile.fields)):
            reasons.append("analytical fields are not covered by the frozen field profile")

    try:
        definition = get_strategy(candidate.strategy_id)
    except KeyError as error:
        definition = None
        reasons.append(str(error))
    if definition is not None:
        if not definition.ticket_eligible_structure:
            reasons.append("research-only undefined-risk structure cannot become a ticket")
        reasons.extend(_validate_geometry(definition, candidate.legs))
    hypothesis_matches = tuple(
        item
        for item in FROZEN_HYPOTHESIS_REGISTRY
        if item.hypothesis_id == candidate.hypothesis_id
    )
    if len(hypothesis_matches) != 1:
        reasons.append("ticket hypothesis is not in the frozen registry")
    else:
        hypothesis = hypothesis_matches[0]
        if hypothesis.strategy_id != candidate.strategy_id:
            reasons.append("ticket hypothesis does not match strategy")
        if hypothesis.holding_sessions != candidate.policy.time_exit_sessions:
            reasons.append("ticket hypothesis does not match holding horizon")
        if hypothesis.exit_policy != candidate.policy.policy_version:
            reasons.append("ticket hypothesis does not match frozen exit policy")
    if candidate.evidence.strategy_family != candidate.hypothesis_id:
        reasons.append("family evidence does not match frozen hypothesis")
    if candidate.evidence.frozen_catalog_version != CATALOG_VERSION:
        reasons.append("family evidence does not use the frozen catalog version")
    if candidate.evidence.state not in {
        EvidenceState.HOLDOUT_PASS,
        EvidenceState.SHADOW_PASS,
    }:
        reasons.append("family evidence state must be HOLDOUT_PASS or SHADOW_PASS")
    if candidate.evidence.evidence_expires_at is None:
        reasons.append("family evidence expiry is missing")
    elif now > candidate.evidence.evidence_expires_at:
        reasons.append("family evidence has expired")
    if candidate.policy.policy_version != candidate.evidence.frozen_exit_policy:
        reasons.append("ticket exit policy does not match frozen family evidence")
    if candidate.evidence.state is EvidenceState.SHADOW_PASS:
        reasons.extend(validate_shadow_pass(candidate.evidence, policy))
    else:
        reasons.extend(validate_holdout_pass(candidate.evidence, policy))
    if candidate.evidence.pop_ece > policy.max_pop_ece:
        reasons.append("POP expected calibration error exceeds tolerance")
    if candidate.evidence.pop_brier_score >= candidate.evidence.base_rate_brier_score:
        reasons.append("POP Brier score does not beat the unconditional base rate")

    probability_estimates = (
        candidate.probabilities.pop_net,
        candidate.probabilities.p_target,
        candidate.probabilities.p_stop,
        candidate.probabilities.p_max_loss,
    )
    expected_targets = ("POP_NET", "P_TARGET", "P_STOP", "P_MAX_LOSS")
    for estimate, expected_target in zip(probability_estimates, expected_targets):
        if estimate.sample_size < policy.min_pop_sample_size:
            reasons.append(
                "POP bucket has fewer than %d observations" % policy.min_pop_sample_size
            )
        if estimate.model_version != candidate.evidence.model_version:
            reasons.append("probability model version does not match family evidence")
        if estimate.artifact_id.lower().removeprefix(
            "sha256:"
        ) != candidate.evidence.pop_model_artifact_id.lower().removeprefix("sha256:"):
            reasons.append("probability artifact does not match family evidence")
        if estimate.calibration_end > now.date():
            reasons.append("probability calibration period cannot end in the future")
        if not math.isclose(
            estimate.confidence_level, 0.95, rel_tol=0.0, abs_tol=1e-12
        ):
            reasons.append("ticket probability intervals must be 95 percent")
        if estimate.interval_method == "UNSPECIFIED":
            reasons.append("probability interval method provenance is required")
        if estimate.bucket_id == "UNSPECIFIED":
            reasons.append("probability strategy/regime bucket identity is required")
        if not _hash_text(estimate.artifact_id):
            reasons.append("probability estimate must bind a hashed model artifact")
        if estimate.target_name != expected_target:
            reasons.append("probability artifact target does not match %s" % expected_target)

    calculation = candidate.model_calculation
    if calculation is None:
        reasons.append("saved current model feature/score calculation is required")
    else:
        if calculation.hypothesis_id != candidate.hypothesis_id:
            reasons.append("current model calculation does not match the hypothesis")
        if calculation.model_version != candidate.evidence.model_version:
            reasons.append("current model calculation version does not match evidence")
        if calculation.model_artifact_id.lower().removeprefix(
            "sha256:"
        ) != candidate.evidence.pop_model_artifact_id.lower().removeprefix("sha256:"):
            reasons.append("current model calculation artifact does not match evidence")
        category_probabilities = dict(calculation.joint_exit_probabilities)
        calculated_metrics = {
            "POP_NET": category_probabilities.get("TARGET", 0.0)
            + category_probabilities.get("TIME_PROFIT", 0.0),
            "P_TARGET": category_probabilities.get("TARGET", 0.0),
            "P_STOP": category_probabilities.get("STOP", 0.0)
            + category_probabilities.get("MAX_LOSS", 0.0),
            "P_MAX_LOSS": category_probabilities.get("MAX_LOSS", 0.0),
        }
        for estimate, expected_target in zip(probability_estimates, expected_targets):
            if not math.isclose(
                estimate.point,
                calculated_metrics[expected_target],
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                reasons.append(
                    "ticket probability does not reproduce from current model calculation: %s"
                    % expected_target
                )

    _scenario_probability_reconciliation(candidate, reasons)

    if not candidate.edge.is_positive:
        reasons.append("point and conservative net EV must both be positive")
    if not math.isfinite(candidate.edge.maximum_loss) or candidate.edge.maximum_loss <= 0.0:
        reasons.append("maximum loss must be finite and positive")
    if not candidate.edge.point_scenarios or not candidate.edge.conservative_scenarios:
        reasons.append("saved point and conservative scenarios are required to reproduce EV")
    if candidate.edge.conservative_gross_expected_value is None:
        reasons.append("saved conservative gross EV is required to reproduce net edge")
    if candidate.edge.conservative_net_expected_value > candidate.edge.net_expected_value:
        reasons.append("conservative net EV cannot exceed point net EV")
    if not candidate.edge.breakevens:
        reasons.append("ticket edge requires at least one breakeven")
    if candidate.edge.target_pnl <= 0.0:
        reasons.append("target P/L must be positive")
    if candidate.edge.stop_pnl >= 0.0 or candidate.edge.stop_pnl < -candidate.edge.maximum_loss:
        reasons.append("stop P/L must be negative and within defined maximum loss")
    if candidate.edge.expected_shortfall <= 0.0:
        reasons.append("expected shortfall must be positive")
    if (
        candidate.edge.adverse_gap_stress_loss <= 0.0
        or candidate.edge.adverse_gap_stress_loss > candidate.edge.maximum_loss + 0.011
    ):
        reasons.append("adverse-gap stress loss must be positive and within maximum loss")
    if candidate.edge.adverse_gap_stress_loss + 0.011 < candidate.edge.expected_shortfall:
        reasons.append("adverse-gap stress loss cannot be below expected shortfall")
    if candidate.edge.point_scenarios:
        reproduced_shortfall = _expected_shortfall_loss(candidate.edge)
        if not math.isclose(
            candidate.edge.expected_shortfall,
            reproduced_shortfall,
            rel_tol=0.0,
            abs_tol=0.011,
        ):
            reasons.append("expected shortfall does not reconcile to saved scenarios")

    try:
        quote_map = exact_quote_map(candidate.leg_quotes)
    except ValueError as error:
        quote_map = {}
        reasons.append(str(error))
    leg_symbols = {leg.occ_symbol for leg in candidate.legs}
    if set(quote_map) != leg_symbols:
        reasons.append("quotes must match every exact OCC leg and no others")
    timestamps = [candidate.underlying_quote.timestamp] + [
        quote.timestamp for quote in candidate.leg_quotes
    ]
    for timestamp in timestamps:
        age = now - timestamp
        if age > max_quote_age:
            reasons.append("Schwab quote is stale")
        if age < -timedelta(seconds=30):
            reasons.append("Schwab quote timestamp is in the future")
    if quote_map and set(quote_map) == leg_symbols:
        natural_price = _natural_price(candidate, reasons)
        if natural_price is not None and not math.isclose(
            candidate.edge.executable_limit_price,
            natural_price,
            rel_tol=0.0,
            abs_tol=0.011,
        ):
            reasons.append("executable limit does not match conservative Schwab natural price")

        total_spread_dollars = math.fsum(
            quote_map[leg.occ_symbol].spread * leg.ratio * 100.0
            for leg in candidate.legs
        )
        if candidate.edge.costs.model_version == "UNSPECIFIED":
            reasons.append("versioned commission/fee/slippage model is required")
        elif (
            candidate.edge.costs.model_version
            != candidate.evidence.cost_model_version
        ):
            reasons.append("ticket cost model does not match family evidence")
        if total_spread_dollars > 0.0:
            if candidate.edge.costs.slippage <= 0.0:
                reasons.append("slippage must be positive when executable spreads are nonzero")
            if not math.isclose(
                candidate.edge.costs.spread_reference,
                total_spread_dollars,
                rel_tol=0.0,
                abs_tol=0.011,
            ):
                reasons.append("slippage spread reference does not match exact leg quotes")

        payoff = _resolved_payoff(candidate, reasons)
        if payoff is not None:
            _compare_payoff_to_edge(payoff, candidate.edge, reasons)
            if isinstance(payoff, PathwisePayoffArtifact) and not math.isclose(
                payoff.adverse_gap_stress_loss,
                candidate.edge.adverse_gap_stress_loss,
                rel_tol=0.0,
                abs_tol=0.011,
            ):
                reasons.append("pathwise gap stress does not match ticket edge")
    else:
        payoff = None

    if candidate.policy.next_review < now.date():
        reasons.append("next review date cannot be in the past")
    event = candidate.event_evidence
    if event.holding_window_start < now.date():
        reasons.append("event holding window starts before ticket creation")
    if len(event.market_sessions) != candidate.policy.time_exit_sessions:
        reasons.append("event holding window does not match the frozen session exit")
    if event.source_timestamp > now + timedelta(seconds=30):
        reasons.append("event evidence timestamp is in the future")
    elif now - event.source_timestamp > max_event_age:
        reasons.append("event evidence is stale")
    if event.status != "CLEAR":
        reasons.append("earnings falls inside the frozen holding window")
    if reasons:
        # Stable de-duplication keeps rejection artifacts reproducible.
        raise TicketRejection(tuple(dict.fromkeys(reasons)))

    return ManualTicket(
        candidate_id=candidate.candidate_id,
        symbol=candidate.symbol,
        thesis=candidate.thesis,
        signal=candidate.signal,
        strategy_id=candidate.strategy_id,
        hypothesis_id=candidate.hypothesis_id,
        evidence=candidate.evidence,
        evidence_state=EvidenceState.MANUAL_TICKET_ENABLED,
        legs=candidate.legs,
        leg_quotes=candidate.leg_quotes,
        underlying_quote=candidate.underlying_quote,
        orats_snapshot_id=candidate.orats_snapshot_id,
        provider_trade_date=candidate.provider_trade_date,
        analytical_fields=candidate.analytical_fields,
        snapshot_manifest=candidate.snapshot_manifest,
        field_profile=candidate.field_profile,
        payoff_evidence=payoff,
        probabilities=candidate.probabilities,
        edge=candidate.edge,
        policy=candidate.policy,
        event_evidence=candidate.event_evidence,
        model_calculation=candidate.model_calculation,
        quote_source=candidate.quote_source,
        quantity=QUANTITY_USER_DETERMINED,
        ranking_score=candidate.edge.ranking_score,
        created_at=now,
    )


def revalidate_manual_ticket(
    ticket: ManualTicket,
    now: datetime,
    max_quote_age: timedelta = timedelta(minutes=5),
    promotion_policy: Optional[PromotionPolicy] = None,
) -> ManualTicket:
    """Re-run every construction gate before a ticket enters a run artifact."""

    if not isinstance(ticket, ManualTicket):
        raise TicketRejection(("ticket must be a Cultra ManualTicket",))
    pathwise = (
        ticket.payoff_evidence
        if isinstance(ticket.payoff_evidence, PathwisePayoffArtifact)
        else None
    )
    rebuilt = build_manual_ticket(
        TicketCandidate(
            candidate_id=ticket.candidate_id,
            symbol=ticket.symbol,
            thesis=ticket.thesis,
            signal=ticket.signal,
            strategy_id=ticket.strategy_id,
            hypothesis_id=ticket.hypothesis_id,
            evidence=ticket.evidence,
            legs=ticket.legs,
            leg_quotes=ticket.leg_quotes,
            underlying_quote=ticket.underlying_quote,
            orats_snapshot_id=ticket.orats_snapshot_id,
            provider_trade_date=ticket.provider_trade_date,
            analytical_fields=ticket.analytical_fields,
            probabilities=ticket.probabilities,
            edge=ticket.edge,
            policy=ticket.policy,
            event_evidence=ticket.event_evidence,
            model_calculation=ticket.model_calculation,
            quote_source=ticket.quote_source,
            snapshot_manifest=ticket.snapshot_manifest,
            field_profile=ticket.field_profile,
            pathwise_payoff=pathwise,
        ),
        now,
        max_quote_age=max_quote_age,
        promotion_policy=promotion_policy,
    )
    if ticket.evidence_state is not EvidenceState.MANUAL_TICKET_ENABLED:
        raise TicketRejection(("ticket evidence_state is not enabled",))
    if ticket.quantity != QUANTITY_USER_DETERMINED:
        raise TicketRejection(("ticket quantity policy is invalid",))
    if not math.isclose(
        ticket.ranking_score,
        rebuilt.ranking_score,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise TicketRejection(("ticket ranking score is not reproducible",))
    return ticket


__all__ = [
    "ManualTicket",
    "CurrentModelCalculation",
    "EventEvidence",
    "PathwisePayoffArtifact",
    "QUANTITY_USER_DETERMINED",
    "TicketCandidate",
    "TicketFieldProfile",
    "TicketRejection",
    "build_manual_ticket",
    "revalidate_manual_ticket",
]
