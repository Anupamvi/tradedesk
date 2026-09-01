"""Leakage-safe exact-leg historical evidence contracts for Cultra Stage 1.

This module models what a historical record *must contain*.  It intentionally
does not reconstruct chains, select substitute contracts, fetch data, or infer
missing costs.  Invalid/incomplete records can be represented so the validator
can produce durable rejection reasons.
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import Enum
import math
from typing import Optional, Sequence, Tuple

from .domain import LegAction, OptionLeg, parse_occ_symbol


class ObservationOrigin(str, Enum):
    CONTEMPORANEOUS = "CONTEMPORANEOUS"
    RECONSTRUCTED = "RECONSTRUCTED"
    SUBSTITUTED = "SUBSTITUTED"


class AmbiguityResolution(str, Enum):
    STOP_FIRST = "STOP_FIRST"
    TARGET_FIRST = "TARGET_FIRST"
    WORST_CASE = "WORST_CASE"


class ExitReason(str, Enum):
    TARGET = "TARGET"
    STOP = "STOP"
    TIME = "TIME"
    INVALIDATION = "INVALIDATION"
    ASSIGNMENT_EXERCISE = "ASSIGNMENT_EXERCISE"


@dataclass(frozen=True)
class OptionGreeks:
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    implied_volatility: float


@dataclass(frozen=True)
class HistoricalLegSnapshot:
    """Observed quote and Greeks for one exact OCC symbol at one decision."""

    occ_symbol: str
    bid: Optional[float]
    ask: Optional[float]
    greeks: Optional[OptionGreeks]
    observed_at: datetime
    available_at: datetime
    origin: ObservationOrigin
    source_snapshot_id: str
    contract_multiplier: float = 100.0


@dataclass(frozen=True)
class HistoricalFeature:
    name: str
    value: Optional[float]
    observed_at: datetime
    available_at: datetime
    origin: ObservationOrigin
    source_snapshot_id: str


@dataclass(frozen=True)
class CorporateActionReview:
    reviewed: bool
    reviewed_at: Optional[datetime]
    source: str
    relevant_action: bool
    adjustment_details: Optional[str]
    exact_contracts_verified: bool


@dataclass(frozen=True)
class FrozenExitPolicy:
    policy_id: str
    version: str
    time_exit_sessions: int
    profit_target_return: float
    stop_loss_return: float
    ambiguity_resolution: Optional[AmbiguityResolution]
    frozen_at: Optional[datetime]
    is_frozen: bool


@dataclass(frozen=True)
class HistoricalCostInputs:
    commissions: Optional[float]
    fees: Optional[float]
    entry_slippage: Optional[float]
    exit_slippage: Optional[float]
    assignment_exercise: Optional[float]
    dividend_effect: Optional[float]
    early_exit: Optional[float]

    @property
    def is_complete(self) -> bool:
        return all(
            value is not None
            for value in (
                self.commissions,
                self.fees,
                self.entry_slippage,
                self.exit_slippage,
                self.assignment_exercise,
                self.dividend_effect,
                self.early_exit,
            )
        )

    @property
    def total(self) -> float:
        if not self.is_complete:
            raise ValueError("historical cost inputs are incomplete")
        return math.fsum(
            float(value)
            for value in (
                self.commissions,
                self.fees,
                self.entry_slippage,
                self.exit_slippage,
                self.assignment_exercise,
                self.dividend_effect,
                self.early_exit,
            )
        )


@dataclass(frozen=True)
class HistoricalExitPath:
    reason: ExitReason
    target_hit_session: Optional[date]
    stop_hit_session: Optional[date]
    ambiguity_resolution_applied: Optional[AmbiguityResolution]

    @property
    def is_ambiguous(self) -> bool:
        return (
            self.target_hit_session is not None
            and self.stop_hit_session is not None
            and self.target_hit_session == self.stop_hit_session
        )


@dataclass(frozen=True)
class HistoricalTradeRecord:
    record_id: str
    symbol: str
    strategy_id: str
    signal_timestamp: datetime
    entry_decision_timestamp: datetime
    exit_decision_timestamp: datetime
    expected_legs: Tuple[OptionLeg, ...]
    entry_snapshots: Tuple[HistoricalLegSnapshot, ...]
    exit_snapshots: Tuple[HistoricalLegSnapshot, ...]
    features: Tuple[HistoricalFeature, ...]
    corporate_action_review: CorporateActionReview
    exit_policy: FrozenExitPolicy
    costs: HistoricalCostInputs
    exit_path: HistoricalExitPath
    holding_sessions: int
    gross_pnl: Optional[float]
    net_pnl: Optional[float]
    market_sessions: Tuple[date, ...]


class HistoricalValidationError(ValueError):
    def __init__(self, reasons: Sequence[str]):
        self.reasons = tuple(reasons)
        super().__init__("historical record rejected: " + "; ".join(self.reasons))


def _is_aware(value: object) -> bool:
    return (
        isinstance(value, datetime)
        and value.tzinfo is not None
        and value.utcoffset() is not None
    )


def _finite(value: object) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _snapshot_errors(
    snapshots: Sequence[HistoricalLegSnapshot],
    expected_symbols: set,
    decision_timestamp: datetime,
    label: str,
    max_quote_age: timedelta,
) -> Tuple[str, ...]:
    reasons = []
    symbols = [snapshot.occ_symbol for snapshot in snapshots]
    if len(symbols) != len(set(symbols)):
        reasons.append("%s snapshots contain duplicate OCC symbols" % label)
    if set(symbols) != expected_symbols:
        reasons.append("%s snapshots do not match every exact expected OCC contract" % label)
    for snapshot in snapshots:
        prefix = "%s snapshot %s" % (label, snapshot.occ_symbol or "<missing>")
        if snapshot.origin is not ObservationOrigin.CONTEMPORANEOUS:
            reasons.append("%s is reconstructed or substituted" % prefix)
        if not snapshot.source_snapshot_id or not snapshot.source_snapshot_id.strip():
            reasons.append("%s has no source snapshot id" % prefix)
        try:
            parse_occ_symbol(snapshot.occ_symbol)
        except ValueError:
            reasons.append("%s is not a canonical OCC contract" % prefix)
        if not _finite(snapshot.contract_multiplier) or float(
            snapshot.contract_multiplier
        ) <= 0.0:
            reasons.append("%s has an invalid contract multiplier" % prefix)
        if not _finite(snapshot.bid) or not _finite(snapshot.ask):
            reasons.append("%s is missing a finite bid/ask" % prefix)
        elif float(snapshot.bid) < 0.0 or float(snapshot.ask) < float(snapshot.bid):
            reasons.append("%s has an invalid bid/ask" % prefix)
        if snapshot.greeks is None:
            reasons.append("%s is missing contemporaneous Greeks" % prefix)
        else:
            greek_values = (
                snapshot.greeks.delta,
                snapshot.greeks.gamma,
                snapshot.greeks.theta,
                snapshot.greeks.vega,
                snapshot.greeks.rho,
                snapshot.greeks.implied_volatility,
            )
            if not all(_finite(value) for value in greek_values):
                reasons.append("%s has incomplete or non-finite Greeks" % prefix)
            elif snapshot.greeks.implied_volatility < 0.0:
                reasons.append("%s has negative implied volatility" % prefix)
        if not _is_aware(snapshot.observed_at) or not _is_aware(snapshot.available_at):
            reasons.append("%s timestamps must be timezone-aware" % prefix)
        else:
            if snapshot.available_at < snapshot.observed_at:
                reasons.append("%s became available before it was observed" % prefix)
            if _is_aware(decision_timestamp):
                if snapshot.available_at > decision_timestamp:
                    reasons.append("%s was unavailable at the decision timestamp" % prefix)
                elif decision_timestamp - snapshot.observed_at > max_quote_age:
                    reasons.append("%s is not contemporaneous with the decision" % prefix)
    return tuple(reasons)


def _historical_gross_pnl(record: HistoricalTradeRecord) -> Optional[float]:
    """Reproduce executable-side gross P/L from exact entry and exit quotes."""

    entry = {snapshot.occ_symbol: snapshot for snapshot in record.entry_snapshots}
    exit_ = {snapshot.occ_symbol: snapshot for snapshot in record.exit_snapshots}
    expected = {leg.occ_symbol for leg in record.expected_legs}
    if set(entry) != expected or set(exit_) != expected:
        return None
    values = []
    for leg in record.expected_legs:
        entry_snapshot = entry[leg.occ_symbol]
        exit_snapshot = exit_[leg.occ_symbol]
        required = (
            entry_snapshot.bid,
            entry_snapshot.ask,
            exit_snapshot.bid,
            exit_snapshot.ask,
            entry_snapshot.contract_multiplier,
            exit_snapshot.contract_multiplier,
        )
        if not all(_finite(value) for value in required):
            return None
        if not math.isclose(
            float(entry_snapshot.contract_multiplier),
            float(exit_snapshot.contract_multiplier),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            return None
        multiplier = float(entry_snapshot.contract_multiplier)
        if leg.action is LegAction.BUY:
            per_share = float(exit_snapshot.bid) - float(entry_snapshot.ask)
        else:
            per_share = float(entry_snapshot.bid) - float(exit_snapshot.ask)
        values.append(per_share * leg.ratio * multiplier)
    return math.fsum(values)


def historical_validation_errors(
    record: HistoricalTradeRecord,
    max_quote_age: timedelta = timedelta(minutes=15),
) -> Tuple[str, ...]:
    """Return stable fail-closed rejection reasons without mutating the record."""

    if max_quote_age <= timedelta(0):
        raise ValueError("max_quote_age must be positive")
    reasons = []
    for name in ("record_id", "symbol", "strategy_id"):
        value = getattr(record, name)
        if not value or not value.strip():
            reasons.append("%s is required" % name)
    for name in (
        "signal_timestamp",
        "entry_decision_timestamp",
        "exit_decision_timestamp",
    ):
        if not _is_aware(getattr(record, name)):
            reasons.append("%s must be timezone-aware" % name)
    if all(
        _is_aware(getattr(record, name))
        for name in (
            "signal_timestamp",
            "entry_decision_timestamp",
            "exit_decision_timestamp",
        )
    ):
        if record.signal_timestamp > record.entry_decision_timestamp:
            reasons.append("signal timestamp is after the entry decision")
        if record.signal_timestamp.date() >= record.entry_decision_timestamp.date():
            reasons.append("entry decision is not on the next market session after the signal")
        if record.entry_decision_timestamp >= record.exit_decision_timestamp:
            reasons.append("entry decision must precede exit decision")

    sessions = tuple(record.market_sessions)
    if not sessions or sessions != tuple(sorted(set(sessions))):
        reasons.append("frozen market-session calendar is missing, duplicated, or unsorted")
    else:
        signal_day = record.signal_timestamp.date() if _is_aware(record.signal_timestamp) else None
        entry_day = record.entry_decision_timestamp.date() if _is_aware(record.entry_decision_timestamp) else None
        exit_day = record.exit_decision_timestamp.date() if _is_aware(record.exit_decision_timestamp) else None
        try:
            signal_index = sessions.index(signal_day)
            entry_index = sessions.index(entry_day)
            exit_index = sessions.index(exit_day)
        except ValueError:
            reasons.append("signal, entry, and exit must resolve to the frozen session calendar")
        else:
            if entry_index != signal_index + 1:
                reasons.append("entry decision is not exactly T+1 in the frozen session calendar")
            if exit_index - entry_index != record.holding_sessions:
                reasons.append("holding_sessions does not reconcile to the frozen session calendar")

    expected_symbols = [leg.occ_symbol for leg in record.expected_legs]
    if not expected_symbols:
        reasons.append("at least one exact expected option leg is required")
    if len(expected_symbols) != len(set(expected_symbols)):
        reasons.append("expected exact OCC leg symbols must be unique")
    normalized_record_symbol = str(record.symbol).strip().upper()
    for leg in record.expected_legs:
        try:
            root, _expiration, _option_type, _strike = parse_occ_symbol(
                leg.occ_symbol
            )
        except ValueError:
            continue
        if root != normalized_record_symbol:
            reasons.append(
                "expected OCC contract root does not match the historical symbol"
            )
    expected_set = set(expected_symbols)
    if _is_aware(record.entry_decision_timestamp):
        reasons.extend(
            _snapshot_errors(
                record.entry_snapshots,
                expected_set,
                record.entry_decision_timestamp,
                "entry",
                max_quote_age,
            )
        )
    if _is_aware(record.exit_decision_timestamp):
        reasons.extend(
            _snapshot_errors(
                record.exit_snapshots,
                expected_set,
                record.exit_decision_timestamp,
                "exit",
                max_quote_age,
            )
        )

    feature_names = []
    for feature in record.features:
        feature_names.append(feature.name)
        prefix = "feature %s" % (feature.name or "<missing>")
        if not feature.name or not feature.name.strip():
            reasons.append("historical feature name is required")
        if feature.origin is not ObservationOrigin.CONTEMPORANEOUS:
            reasons.append("%s is reconstructed or substituted" % prefix)
        if not feature.source_snapshot_id or not feature.source_snapshot_id.strip():
            reasons.append("%s has no source snapshot id" % prefix)
        if not _finite(feature.value):
            reasons.append("%s has no finite value" % prefix)
        if not _is_aware(feature.observed_at) or not _is_aware(feature.available_at):
            reasons.append("%s timestamps must be timezone-aware" % prefix)
        else:
            if feature.available_at < feature.observed_at:
                reasons.append("%s became available before it was observed" % prefix)
            if _is_aware(record.signal_timestamp) and feature.available_at > record.signal_timestamp:
                reasons.append("%s leaks data unavailable at the signal timestamp" % prefix)
    if len(feature_names) != len(set(feature_names)):
        reasons.append("historical feature names must be unique")
    if not record.features:
        reasons.append("at least one contemporaneous signal feature is required")

    review = record.corporate_action_review
    if not review.reviewed:
        reasons.append("corporate actions were not reviewed")
    if not review.source or not review.source.strip():
        reasons.append("corporate-action review source is required")
    if not review.exact_contracts_verified:
        reasons.append("corporate-action review did not verify exact OCC contracts")
    if review.reviewed_at is None or not _is_aware(review.reviewed_at):
        reasons.append("corporate-action review timestamp is required and timezone-aware")
    elif _is_aware(record.exit_decision_timestamp) and review.reviewed_at < record.exit_decision_timestamp:
        reasons.append("corporate-action review was completed before the full trade period")
    if review.relevant_action and (
        not review.adjustment_details or not review.adjustment_details.strip()
    ):
        reasons.append("relevant corporate action lacks adjustment details")

    policy = record.exit_policy
    if not policy.is_frozen:
        reasons.append("exit and ambiguity policy is not frozen")
    if not policy.policy_id or not policy.policy_id.strip() or not policy.version or not policy.version.strip():
        reasons.append("frozen exit policy id and version are required")
    if isinstance(policy.time_exit_sessions, bool) or not isinstance(
        policy.time_exit_sessions, int
    ) or not 20 <= policy.time_exit_sessions <= 60:
        reasons.append("frozen time exit must be between 20 and 60 sessions")
    if not _finite(policy.profit_target_return) or float(policy.profit_target_return) <= 0.0:
        reasons.append("frozen profit target must be finite and positive")
    if not _finite(policy.stop_loss_return) or float(policy.stop_loss_return) <= 0.0:
        reasons.append("frozen stop loss must be finite and positive")
    if policy.ambiguity_resolution is None:
        reasons.append("stop/target ambiguity ordering is not frozen")
    if policy.frozen_at is None or not _is_aware(policy.frozen_at):
        reasons.append("exit policy freeze timestamp is required and timezone-aware")
    elif _is_aware(record.signal_timestamp) and policy.frozen_at > record.signal_timestamp:
        reasons.append("exit policy was frozen after the signal timestamp")

    if isinstance(record.holding_sessions, bool) or not isinstance(record.holding_sessions, int):
        reasons.append("holding_sessions must be an integer")
    elif record.holding_sessions <= 0:
        reasons.append("holding_sessions must be positive")
    elif (
        isinstance(policy.time_exit_sessions, int)
        and not isinstance(policy.time_exit_sessions, bool)
        and record.holding_sessions > policy.time_exit_sessions
    ):
        reasons.append("record exceeds the frozen time-exit session count")
    if (
        record.exit_path.reason is ExitReason.TIME
        and isinstance(policy.time_exit_sessions, int)
        and record.holding_sessions != policy.time_exit_sessions
    ):
        reasons.append("time exit did not occur at the frozen session count")

    if record.exit_path.is_ambiguous:
        applied = record.exit_path.ambiguity_resolution_applied
        if not policy.is_frozen or policy.ambiguity_resolution is None:
            reasons.append("ambiguous stop/target path has no frozen ordering")
        elif applied is not policy.ambiguity_resolution:
            reasons.append("ambiguous stop/target ordering does not match frozen policy")
        expected_reason = (
            ExitReason.TARGET
            if applied is AmbiguityResolution.TARGET_FIRST
            else ExitReason.STOP
        )
        if applied is not None and record.exit_path.reason is not expected_reason:
            reasons.append("exit reason contradicts the applied ambiguity ordering")
    elif record.exit_path.ambiguity_resolution_applied is not None:
        reasons.append("ambiguity resolution was applied to an unambiguous path")

    cost_values = (
        record.costs.commissions,
        record.costs.fees,
        record.costs.entry_slippage,
        record.costs.exit_slippage,
        record.costs.assignment_exercise,
        record.costs.dividend_effect,
        record.costs.early_exit,
    )
    if not record.costs.is_complete:
        reasons.append("historical cost inputs are incomplete")
    elif not all(_finite(value) and float(value) >= 0.0 for value in cost_values):
        reasons.append("historical cost inputs must be finite and nonnegative")
    if not _finite(record.gross_pnl) or not _finite(record.net_pnl):
        reasons.append("gross and net P/L must be finite")
    elif record.costs.is_complete and all(
        _finite(value) and float(value) >= 0.0 for value in cost_values
    ):
        expected_net = float(record.gross_pnl) - record.costs.total
        if not math.isclose(float(record.net_pnl), expected_net, rel_tol=1e-12, abs_tol=1e-9):
            reasons.append("net P/L does not reconcile to gross P/L and complete costs")
    reproduced_gross = _historical_gross_pnl(record)
    if reproduced_gross is None:
        if expected_set and set(snapshot.occ_symbol for snapshot in record.entry_snapshots) == expected_set and set(
            snapshot.occ_symbol for snapshot in record.exit_snapshots
        ) == expected_set:
            reasons.append("historical gross P/L cannot be reproduced from exact quotes")
    elif _finite(record.gross_pnl) and not math.isclose(
        float(record.gross_pnl), reproduced_gross, rel_tol=1e-12, abs_tol=1e-9
    ):
        reasons.append(
            "gross P/L does not reconcile to executable entry/exit quote sides"
        )

    if record.costs.is_complete:
        entry_has_spread = any(
            _finite(snapshot.bid)
            and _finite(snapshot.ask)
            and float(snapshot.ask) > float(snapshot.bid)
            for snapshot in record.entry_snapshots
        )
        exit_has_spread = any(
            _finite(snapshot.bid)
            and _finite(snapshot.ask)
            and float(snapshot.ask) > float(snapshot.bid)
            for snapshot in record.exit_snapshots
        )
        if entry_has_spread and float(record.costs.entry_slippage or 0.0) <= 0.0:
            reasons.append("entry slippage must be positive when quoted spreads are nonzero")
        if exit_has_spread and float(record.costs.exit_slippage or 0.0) <= 0.0:
            reasons.append("exit slippage must be positive when quoted spreads are nonzero")

    return tuple(dict.fromkeys(reasons))


def validate_historical_trade(
    record: HistoricalTradeRecord,
    max_quote_age: timedelta = timedelta(minutes=15),
) -> HistoricalTradeRecord:
    """Return the record if complete; otherwise raise with stable reasons."""

    reasons = historical_validation_errors(record, max_quote_age=max_quote_age)
    if reasons:
        raise HistoricalValidationError(reasons)
    return record


__all__ = [
    "AmbiguityResolution",
    "CorporateActionReview",
    "ExitReason",
    "FrozenExitPolicy",
    "HistoricalCostInputs",
    "HistoricalExitPath",
    "HistoricalFeature",
    "HistoricalLegSnapshot",
    "HistoricalTradeRecord",
    "HistoricalValidationError",
    "ObservationOrigin",
    "OptionGreeks",
    "historical_validation_errors",
    "validate_historical_trade",
]
