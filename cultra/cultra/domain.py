"""Core value objects for Cultra's clean-room options research pipeline.

The types in this module deliberately contain no broker, account, portfolio, or
network concepts.  They describe one normalized option structure and the
evidence needed to reason about it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
import math
import re
from typing import Optional, Tuple


_OCC_RE = re.compile(r"^([A-Z0-9.]{1,6})\s*(\d{6})([CP])(\d{8})$")


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("%s must be finite" % name)
    return value


def _probability(value: float, name: str) -> float:
    value = _finite(value, name)
    if value < 0.0 or value > 1.0:
        raise ValueError("%s must be between 0 and 1" % name)
    return value


def _aware_timestamp(value: datetime, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("%s must be timezone-aware" % name)
    return value


def parse_occ_symbol(value: str) -> Tuple[str, date, "OptionType", float]:
    """Parse the canonical OCC identity used to cross-check every exact leg."""

    normalized = str(value).strip().upper()
    match = _OCC_RE.fullmatch(normalized)
    if match is None:
        raise ValueError("occ_symbol must be a canonical OCC option symbol")
    root, compact_date, raw_type, raw_strike = match.groups()
    try:
        expiration = datetime.strptime(compact_date, "%y%m%d").date()
    except ValueError as exc:
        raise ValueError("occ_symbol contains an invalid expiration") from exc
    option_type = OptionType.CALL if raw_type == "C" else OptionType.PUT
    strike = int(raw_strike) / 1000.0
    if strike <= 0.0:
        raise ValueError("occ_symbol contains a non-positive strike")
    return root, expiration, option_type, strike


class OptionType(str, Enum):
    CALL = "CALL"
    PUT = "PUT"


class LegAction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class ScenarioOutcome(str, Enum):
    """Mutually exclusive exit outcome used to reconcile POP with EV."""

    TARGET = "TARGET"
    STOP = "STOP"
    TIME_PROFIT = "TIME_PROFIT"
    TIME_LOSS = "TIME_LOSS"
    MAX_LOSS = "MAX_LOSS"
    UNCLASSIFIED = "UNCLASSIFIED"


class EvidenceState(str, Enum):
    UNPROVEN = "UNPROVEN"
    RESEARCH_PASS = "RESEARCH_PASS"
    VALIDATION_PASS = "VALIDATION_PASS"
    HOLDOUT_PASS = "HOLDOUT_PASS"
    SHADOW_PASS = "SHADOW_PASS"
    MANUAL_TICKET_ENABLED = "MANUAL_TICKET_ENABLED"


class CandidateDisposition(str, Enum):
    ELIGIBLE = "ELIGIBLE"
    WATCHLIST = "WATCHLIST"
    REJECTED = "REJECTED"
    DATA_UNAVAILABLE = "DATA_UNAVAILABLE"
    NOT_FULLY_EVALUATED_BUDGET = "NOT_FULLY_EVALUATED_BUDGET"


@dataclass(frozen=True)
class OptionLeg:
    """An exact OCC option leg in a normalized one-unit structure."""

    occ_symbol: str
    action: LegAction
    option_type: OptionType
    expiration: date
    strike: float
    ratio: int = 1

    def __post_init__(self) -> None:
        normalized_symbol = str(self.occ_symbol).strip().upper()
        _root, occ_expiration, occ_type, occ_strike = parse_occ_symbol(
            normalized_symbol
        )
        object.__setattr__(self, "occ_symbol", normalized_symbol)
        if self.expiration is None:
            raise ValueError("expiration is required")
        object.__setattr__(self, "strike", _finite(self.strike, "strike"))
        if self.strike <= 0.0:
            raise ValueError("strike must be positive")
        if isinstance(self.ratio, bool) or not isinstance(self.ratio, int):
            raise TypeError("ratio must be an integer")
        if self.ratio <= 0:
            raise ValueError("ratio must be positive")
        if occ_expiration != self.expiration:
            raise ValueError("occ_symbol expiration does not match the leg")
        if occ_type is not self.option_type:
            raise ValueError("occ_symbol option type does not match the leg")
        if not math.isclose(occ_strike, self.strike, rel_tol=0.0, abs_tol=0.0005):
            raise ValueError("occ_symbol strike does not match the leg")


@dataclass(frozen=True)
class LegQuote:
    """Executable bid/ask evidence for one exact OCC symbol."""

    occ_symbol: str
    bid: float
    ask: float
    timestamp: datetime

    def __post_init__(self) -> None:
        normalized_symbol = str(self.occ_symbol).strip().upper()
        parse_occ_symbol(normalized_symbol)
        object.__setattr__(self, "occ_symbol", normalized_symbol)
        object.__setattr__(self, "bid", _finite(self.bid, "bid"))
        object.__setattr__(self, "ask", _finite(self.ask, "ask"))
        _aware_timestamp(self.timestamp, "timestamp")
        if self.bid < 0.0 or self.ask < 0.0:
            raise ValueError("bid and ask cannot be negative")
        if self.ask < self.bid:
            raise ValueError("ask cannot be below bid")

    @property
    def midpoint(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid


@dataclass(frozen=True)
class UnderlyingQuote:
    symbol: str
    bid: float
    ask: float
    timestamp: datetime

    def __post_init__(self) -> None:
        if not self.symbol or not self.symbol.strip():
            raise ValueError("symbol is required")
        object.__setattr__(self, "bid", _finite(self.bid, "bid"))
        object.__setattr__(self, "ask", _finite(self.ask, "ask"))
        _aware_timestamp(self.timestamp, "timestamp")
        if self.bid < 0.0 or self.ask < 0.0:
            raise ValueError("bid and ask cannot be negative")
        if self.ask < self.bid:
            raise ValueError("ask cannot be below bid")

    @property
    def midpoint(self) -> float:
        return (self.bid + self.ask) / 2.0


@dataclass(frozen=True)
class Scenario:
    """One mutually exclusive net-P/L outcome before common modeled costs."""

    label: str
    probability: float
    net_pnl: float
    outcome: ScenarioOutcome = ScenarioOutcome.UNCLASSIFIED

    def __post_init__(self) -> None:
        if not self.label or not self.label.strip():
            raise ValueError("scenario label is required")
        object.__setattr__(
            self, "probability", _probability(self.probability, "probability")
        )
        object.__setattr__(self, "net_pnl", _finite(self.net_pnl, "net_pnl"))
        if not isinstance(self.outcome, ScenarioOutcome):
            raise TypeError("scenario outcome must be ScenarioOutcome")


@dataclass(frozen=True)
class ProbabilityEstimate:
    """A calibrated probability together with reproducible provenance."""

    point: float
    lower: float
    upper: float
    sample_size: int
    model_version: str
    calibration_start: date
    calibration_end: date
    confidence_level: float = 0.95
    interval_method: str = "UNSPECIFIED"
    bucket_id: str = "UNSPECIFIED"
    artifact_id: str = "UNSPECIFIED"
    target_name: str = "UNSPECIFIED"

    def __post_init__(self) -> None:
        object.__setattr__(self, "point", _probability(self.point, "point"))
        object.__setattr__(self, "lower", _probability(self.lower, "lower"))
        object.__setattr__(self, "upper", _probability(self.upper, "upper"))
        if not self.lower <= self.point <= self.upper:
            raise ValueError("probability interval must contain point")
        if isinstance(self.sample_size, bool) or not isinstance(self.sample_size, int):
            raise TypeError("sample_size must be an integer")
        if self.sample_size <= 0:
            raise ValueError("sample_size must be positive")
        if not self.model_version or not self.model_version.strip():
            raise ValueError("model_version is required")
        if self.calibration_start > self.calibration_end:
            raise ValueError("calibration_start cannot follow calibration_end")
        object.__setattr__(
            self,
            "confidence_level",
            _probability(self.confidence_level, "confidence_level"),
        )
        if self.confidence_level <= 0.0 or self.confidence_level >= 1.0:
            raise ValueError("confidence_level must be strictly between 0 and 1")
        for field_name in (
            "interval_method",
            "bucket_id",
            "artifact_id",
            "target_name",
        ):
            value = str(getattr(self, field_name)).strip()
            if not value:
                raise ValueError("%s is required" % field_name)


@dataclass(frozen=True)
class ProbabilityBundle:
    """Required ticket probabilities; delta is intentionally absent."""

    pop_net: ProbabilityEstimate
    p_target: ProbabilityEstimate
    p_stop: ProbabilityEstimate
    p_max_loss: ProbabilityEstimate

    def __post_init__(self) -> None:
        versions = {
            self.pop_net.model_version,
            self.p_target.model_version,
            self.p_stop.model_version,
            self.p_max_loss.model_version,
        }
        if len(versions) != 1:
            raise ValueError("all probability estimates must use one model version")
        estimates = (self.pop_net, self.p_target, self.p_stop, self.p_max_loss)
        if len({item.sample_size for item in estimates}) != 1:
            raise ValueError("all probability estimates must use one resolved bucket")
        if len({item.calibration_start for item in estimates}) != 1 or len(
            {item.calibration_end for item in estimates}
        ) != 1:
            raise ValueError("all probability estimates must use one calibration period")
        if self.p_max_loss.point > self.p_stop.point + 1e-12:
            raise ValueError("P_max_loss cannot exceed P_stop")


@dataclass(frozen=True)
class PeriodEvidence:
    """Net expectancy evidence for one chronological evaluation period."""

    name: str
    expectancy: float
    lower_confidence_bound: float
    resolved_trades: int
    independent_clusters: int
    start: date
    end: date
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("period name is required")
        object.__setattr__(self, "expectancy", _finite(self.expectancy, "expectancy"))
        object.__setattr__(
            self,
            "lower_confidence_bound",
            _finite(self.lower_confidence_bound, "lower_confidence_bound"),
        )
        for field_name in ("resolved_trades", "independent_clusters"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError("%s must be an integer" % field_name)
            if value < 0:
                raise ValueError("%s cannot be negative" % field_name)
        if self.start > self.end:
            raise ValueError("period start cannot follow end")
        object.__setattr__(
            self,
            "confidence_level",
            _probability(self.confidence_level, "confidence_level"),
        )
        if self.confidence_level <= 0.0 or self.confidence_level >= 1.0:
            raise ValueError("confidence_level must be strictly between 0 and 1")


@dataclass(frozen=True)
class FamilyEvidence:
    """Complete evidence record for one frozen strategy family."""

    strategy_family: str
    state: EvidenceState
    training: PeriodEvidence
    validation: PeriodEvidence
    holdout: PeriodEvidence
    shadow: Optional[PeriodEvidence]
    holm_adjusted_p_value: float
    holm_family_size: int
    holm_catalog_version: str
    max_contribution_fraction: float
    contribution_dimensions: Tuple[str, ...]
    pop_ece: float
    pop_brier_score: float
    base_rate_brier_score: float
    cost_model_version: str
    model_version: str
    pop_model_artifact_id: str
    frozen_catalog_version: str
    frozen_exit_policy: str
    holdout_consumed_once: bool
    shadow_calendar_days: int = 0
    hypothesis_registry_hash: str = "UNSPECIFIED"
    timing_policy_version: str = "UNSPECIFIED"
    universe_policy_version: str = "UNSPECIFIED"
    model_frozen_at: Optional[datetime] = None
    holdout_evaluated_at: Optional[datetime] = None
    evidence_expires_at: Optional[datetime] = None
    holdout_resolved_candidates: int = 0
    holdout_unresolved_candidates: int = 0
    unresolved_worst_case_expectancy: float = 0.0
    probability_event_counts: Tuple[Tuple[str, int, int], ...] = ()
    two_way_clustered: bool = False
    point_in_time_membership: bool = False
    next_session_entry: bool = False
    holdout_registry_receipt: str = "UNSPECIFIED"

    def __post_init__(self) -> None:
        for field_name in (
            "strategy_family",
            "cost_model_version",
            "model_version",
            "pop_model_artifact_id",
            "frozen_catalog_version",
            "frozen_exit_policy",
            "holm_catalog_version",
        ):
            value = getattr(self, field_name)
            if not value or not value.strip():
                raise ValueError("%s is required" % field_name)
        artifact = self.pop_model_artifact_id.lower().removeprefix("sha256:")
        if len(artifact) != 64 or any(
            character not in "0123456789abcdef" for character in artifact
        ):
            raise ValueError("pop_model_artifact_id must be a SHA-256 identity")
        object.__setattr__(
            self,
            "holm_adjusted_p_value",
            _probability(self.holm_adjusted_p_value, "holm_adjusted_p_value"),
        )
        object.__setattr__(
            self,
            "max_contribution_fraction",
            _probability(self.max_contribution_fraction, "max_contribution_fraction"),
        )
        if isinstance(self.holm_family_size, bool) or not isinstance(
            self.holm_family_size, int
        ):
            raise TypeError("holm_family_size must be an integer")
        if self.holm_family_size <= 0:
            raise ValueError("holm_family_size must be positive")
        dimensions = tuple(sorted(set(self.contribution_dimensions)))
        if dimensions != tuple(sorted(self.contribution_dimensions)):
            raise ValueError("contribution_dimensions must be unique and sorted")
        if not dimensions:
            raise ValueError("contribution_dimensions are required")
        object.__setattr__(self, "pop_ece", _probability(self.pop_ece, "pop_ece"))
        object.__setattr__(
            self, "pop_brier_score", _probability(self.pop_brier_score, "pop_brier_score")
        )
        object.__setattr__(
            self,
            "base_rate_brier_score",
            _probability(self.base_rate_brier_score, "base_rate_brier_score"),
        )
        if not isinstance(self.holdout_consumed_once, bool):
            raise TypeError("holdout_consumed_once must be bool")
        if isinstance(self.shadow_calendar_days, bool) or not isinstance(
            self.shadow_calendar_days, int
        ):
            raise TypeError("shadow_calendar_days must be an integer")
        if self.shadow_calendar_days < 0:
            raise ValueError("shadow_calendar_days cannot be negative")
        for field_name in (
            "hypothesis_registry_hash",
            "timing_policy_version",
            "universe_policy_version",
            "holdout_registry_receipt",
        ):
            if not str(getattr(self, field_name)).strip():
                raise ValueError("%s is required" % field_name)
        for field_name in (
            "holdout_resolved_candidates",
            "holdout_unresolved_candidates",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("%s must be a nonnegative integer" % field_name)
        object.__setattr__(
            self,
            "unresolved_worst_case_expectancy",
            _finite(
                self.unresolved_worst_case_expectancy,
                "unresolved_worst_case_expectancy",
            ),
        )
        supplied_times = (
            self.model_frozen_at,
            self.holdout_evaluated_at,
            self.evidence_expires_at,
        )
        if any(value is None for value in supplied_times) and not all(
            value is None for value in supplied_times
        ):
            raise ValueError(
                "model freeze, holdout evaluation, and expiry timestamps must be supplied together"
            )
        if self.model_frozen_at is not None:
            _aware_timestamp(self.model_frozen_at, "model_frozen_at")
            _aware_timestamp(self.holdout_evaluated_at, "holdout_evaluated_at")
            _aware_timestamp(self.evidence_expires_at, "evidence_expires_at")
            if self.holdout_evaluated_at <= self.model_frozen_at:
                raise ValueError("holdout evaluation must follow model freeze")
            if self.evidence_expires_at <= self.model_frozen_at:
                raise ValueError("evidence expiry must follow model freeze")
        targets = []
        for item in self.probability_event_counts:
            if len(item) != 3:
                raise ValueError("probability event counts must be target/positive/negative")
            target, positive, negative = item
            if target in targets or not str(target).strip():
                raise ValueError("probability event targets must be unique and named")
            targets.append(target)
            for value in (positive, negative):
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError("probability event counts must be nonnegative integers")
        for field_name in (
            "two_way_clustered",
            "point_in_time_membership",
            "next_session_entry",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError("%s must be bool" % field_name)


@dataclass(frozen=True)
class HistoricalObservation:
    """Minimal leakage-safe observation used by chronological splitting."""

    observation_id: str
    session_date: date
    cluster_id: str
    net_pnl: float

    def __post_init__(self) -> None:
        if not self.observation_id or not self.observation_id.strip():
            raise ValueError("observation_id is required")
        if not self.cluster_id or not self.cluster_id.strip():
            raise ValueError("cluster_id is required")
        object.__setattr__(self, "net_pnl", _finite(self.net_pnl, "net_pnl"))


@dataclass(frozen=True)
class EntryExitPolicy:
    entry_condition: str
    profit_target: str
    stop_condition: str
    time_exit: str
    invalidation: str
    assignment_handling: str
    next_review: date
    policy_version: str = "UNSPECIFIED"
    time_exit_sessions: int = 0

    def __post_init__(self) -> None:
        for field_name in (
            "entry_condition",
            "profit_target",
            "stop_condition",
            "time_exit",
            "invalidation",
            "assignment_handling",
        ):
            value = getattr(self, field_name)
            if not value or not value.strip():
                raise ValueError("%s is required" % field_name)
        if not self.policy_version or not self.policy_version.strip():
            raise ValueError("policy_version is required")
        if isinstance(self.time_exit_sessions, bool) or not isinstance(
            self.time_exit_sessions, int
        ):
            raise TypeError("time_exit_sessions must be an integer")
        if not 20 <= self.time_exit_sessions <= 60:
            raise ValueError("time_exit_sessions must be between 20 and 60")


def exact_quote_map(quotes: Tuple[LegQuote, ...]) -> dict:
    """Return a duplicate-free exact OCC-symbol map."""

    result = {}
    for quote in quotes:
        if quote.occ_symbol in result:
            raise ValueError("duplicate quote for %s" % quote.occ_symbol)
        result[quote.occ_symbol] = quote
    return result
