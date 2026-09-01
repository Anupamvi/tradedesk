"""Pure offline ORATS request optimization for Cultra.

The canonical historical design fetches each rotating research cohort's full
chain once per market session.  Those same rows support every frozen strategy,
horizon, selected leg, and exit path.  It therefore has no request-per-signal,
request-per-strike, or request-per-hypothesis multiplier.  Large historical
work is split into independently authorized runs capped at 90 attempts each;
the 90-attempt ceiling is not misrepresented as a whole-campaign estimate.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .requesting import (
    Endpoint,
    PlanningError,
    RequestPlan,
    RunType,
    deterministic_batches,
    make_planned_request,
    normalize_entities,
)


# Historical work may span several independently authorized, immutable slices.
# Ninety is a per-slice ceiling, never a claim that the complete campaign fits
# in ninety calls.
DEFAULT_HISTORICAL_SLICE_CAP = 90
DEFAULT_HISTORICAL_CAMPAIGN_CAP = DEFAULT_HISTORICAL_SLICE_CAP
ABSOLUTE_HISTORICAL_CAMPAIGN_CAP = 99

# The documented endpoint accepts up to ten tickers, but Cultra's saved
# account evidence is materially different for full-history Core payloads:
# one two-ticker request succeeded while the same ten-ticker request failed
# three times with 502 responses.  Historical chain and split requests have
# succeeded at ten tickers.  Freeze the empirically supported geometries here
# instead of treating a provider maximum as a reliable payload size.
HISTORICAL_CORE_TICKER_BATCH_SIZE = 2
HISTORICAL_CHAIN_TICKER_BATCH_SIZE = 10
HISTORICAL_SPLIT_TICKER_BATCH_SIZE = 10

HISTORICAL_CORE_FIELDS = (
    "atmFcstIvM1",
    "atmIvM1",
    "confidence",
    "contango",
    "dtExM1",
    "iv30d",
    "iv60d",
    "orFcst20d",
    "orHv20d",
    "orHv60d",
    "priorCls",
    "pxAtmIv",
    "slope",
    "ticker",
    "tradeDate",
    "updatedAt",
)

HISTORICAL_STRIKE_FIELDS = (
    "callAskPrice",
    "callBidIv",
    "callBidPrice",
    "callMidIv",
    "callOpenInterest",
    "callVolume",
    "delta",
    "dte",
    "expirDate",
    "gamma",
    "putAskPrice",
    "putBidIv",
    "putBidPrice",
    "putMidIv",
    "putOpenInterest",
    "putVolume",
    "rho",
    "smvVol",
    "stockPrice",
    "strike",
    "theta",
    "ticker",
    "tradeDate",
    "updatedAt",
    "vega",
)

HISTORICAL_SPLIT_FIELDS = ("divisor", "splitDate", "ticker")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _suffix(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()[:12]


def _valid_date(value: str, label: str) -> str:
    try:
        return date.fromisoformat(str(value)).isoformat()
    except ValueError as exc:
        raise PlanningError("%s must use YYYY-MM-DD" % label) from exc


def _campaign_cap(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PlanningError("historical campaign cap must be an integer")
    if not 1 <= value <= ABSOLUTE_HISTORICAL_CAMPAIGN_CAP:
        raise PlanningError("historical campaign cap must be between 1 and 99")
    return value


@dataclass(frozen=True, order=True)
class SignalDateKey:
    """One locally prequalified underlying/date needing an entry chain."""

    trade_date: str
    ticker: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "trade_date", _valid_date(self.trade_date, "trade date"))
        object.__setattr__(
            self, "ticker", normalize_entities((self.ticker,), "ticker")[0]
        )


@dataclass(frozen=True, order=True)
class ExactStrikeHistoryKey:
    """One ORATS exact-strike series; a row contains both call and put sides."""

    ticker: str
    expiration: str
    strike: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "ticker", normalize_entities((self.ticker,), "ticker")[0]
        )
        object.__setattr__(
            self, "expiration", _valid_date(self.expiration, "expiration")
        )
        try:
            strike = Decimal(str(self.strike))
        except (InvalidOperation, ValueError) as exc:
            raise PlanningError("exact-history strike is invalid") from exc
        if not strike.is_finite() or strike <= 0:
            raise PlanningError("exact-history strike is invalid")
        canonical = format(strike.normalize(), "f")
        object.__setattr__(self, "strike", canonical)


@dataclass(frozen=True)
class RotatingCohortPolicy:
    """Cold-cache request envelope for point-in-time sampled validation.

    The cohort is not a named or permanent ticker list. A later offline freeze
    must select each block from a point-in-time liquid-optionable universe using
    only information available at the block boundary. New entries are censored
    far enough before the next rotation to resolve the maximum holding path
    inside the same block. That removes transition-overlap requests without
    dropping or reconstructing any selected trade path.
    """

    eligible_symbols: int
    historical_sessions: int = 450
    cohort_size: int = 10
    cohort_block_sessions: int = 120
    maximum_holding_sessions: int = 60
    core_ticker_batch_size: int = HISTORICAL_CORE_TICKER_BATCH_SIZE
    ticker_batch_size: int = HISTORICAL_CHAIN_TICKER_BATCH_SIZE
    split_ticker_batch_size: int = HISTORICAL_SPLIT_TICKER_BATCH_SIZE
    slice_cap: int = DEFAULT_HISTORICAL_SLICE_CAP
    transition_policy: str = "CENSOR_ENTRIES_BEFORE_COHORT_ROTATION"

    def __post_init__(self) -> None:
        for name in (
            "eligible_symbols",
            "historical_sessions",
            "cohort_size",
            "cohort_block_sessions",
            "maximum_holding_sessions",
            "core_ticker_batch_size",
            "ticker_batch_size",
            "split_ticker_batch_size",
            "slice_cap",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise PlanningError("%s must be a positive integer" % name)
        if self.cohort_size > self.eligible_symbols:
            raise PlanningError("cohort_size cannot exceed the eligible universe")
        if self.cohort_size > self.ticker_batch_size:
            raise PlanningError("one frozen cohort must fit one historical ticker batch")
        if self.core_ticker_batch_size > 10:
            raise PlanningError("historical Core ticker batch cannot exceed ten")
        if self.split_ticker_batch_size > 10:
            raise PlanningError("historical split ticker batch cannot exceed ten")
        if self.maximum_holding_sessions > self.cohort_block_sessions:
            raise PlanningError(
                "maximum holding window cannot exceed the cohort block length"
            )
        blocks = int(
            math.ceil(self.historical_sessions / float(self.cohort_block_sessions))
        )
        if self.eligible_symbols < blocks * self.cohort_size:
            raise PlanningError(
                "eligible universe cannot supply disjoint rotating cohorts"
            )
        if self.slice_cap > DEFAULT_HISTORICAL_SLICE_CAP:
            raise PlanningError("historical slice cap cannot exceed 90")
        if self.transition_policy != "CENSOR_ENTRIES_BEFORE_COHORT_ROTATION":
            raise PlanningError("historical transition policy is not frozen")


def rotating_cohort_campaign_forecast(
    policy: RotatingCohortPolicy,
    *,
    cached_core_calls: int = 0,
    cached_chain_calls: int = 0,
    cached_corporate_action_calls: int = 0,
) -> Mapping[str, Any]:
    """Return a complete structural estimate before any ticker is selected.

    Full historical chains are fetched once per session for the active cohort.
    Signals too close to a cohort boundary are ineligible, so every T+1 entry
    and maximum holding path is contained in the already fetched block. This
    supports all frozen structures and horizons from the same rows and avoids a
    request per strategy, candidate, option leg, or transition overlap.
    """

    cached_values = (
        cached_core_calls,
        cached_chain_calls,
        cached_corporate_action_calls,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in cached_values
    ):
        raise PlanningError("cached request counts must be nonnegative integers")
    blocks = int(
        math.ceil(policy.historical_sessions / float(policy.cohort_block_sessions))
    )
    transitions = max(0, blocks - 1)
    unique_sampled_symbols = min(
        policy.eligible_symbols, blocks * policy.cohort_size
    )
    # Cohort selection is owned by the frozen external point-in-time universe,
    # not by ORATS. Historical Core is therefore needed only for the sampled
    # symbols after selection, never for every current constituent.
    core_calls = int(
        math.ceil(unique_sampled_symbols / float(policy.core_ticker_batch_size))
    )
    base_chain_calls = policy.historical_sessions * int(
        math.ceil(policy.cohort_size / float(policy.ticker_batch_size))
    )
    overlap_chain_calls = 0
    chain_calls = base_chain_calls
    continuous_entry_extension_calls = (
        transitions
        * (policy.maximum_holding_sessions + 1)
        * int(math.ceil(policy.cohort_size / float(policy.ticker_batch_size)))
    )
    full_blocks, final_block = divmod(
        policy.historical_sessions, policy.cohort_block_sessions
    )
    block_lengths = [policy.cohort_block_sessions] * full_blocks
    if final_block:
        block_lengths.append(final_block)
    eligible_signal_sessions = sum(
        max(0, length - policy.maximum_holding_sessions - 1)
        for length in block_lengths
    )
    corporate_action_calls = int(
        math.ceil(unique_sampled_symbols / float(policy.split_ticker_batch_size))
    )
    remaining = {
        "historical_core": max(0, core_calls - cached_core_calls),
        "historical_chains": max(0, chain_calls - cached_chain_calls),
        "corporate_actions": max(
            0, corporate_action_calls - cached_corporate_action_calls
        ),
    }
    expected_total = sum(remaining.values())
    slice_count = (
        int(math.ceil(expected_total / float(policy.slice_cap)))
        if expected_total
        else 0
    )
    exact_slice_attempts = tuple(
        min(policy.slice_cap, expected_total - offset)
        for offset in range(0, expected_total, policy.slice_cap)
    )
    generic_cap_sum = slice_count * policy.slice_cap
    return {
        "schema": "cultra.historical-rotating-cohort-forecast.v1",
        "status": "OFFLINE_ESTIMATE_REQUIRES_COHORT_AND_EVENT_SOURCE_FREEZE",
        "network_attempted": False,
        "execution_authorized": False,
        "design": "POINT_IN_TIME_ROTATING_COHORT_WITH_MAX_HOLD_OVERLAP",
        "eligible_symbols": policy.eligible_symbols,
        "historical_sessions": policy.historical_sessions,
        "cohort_size_per_block": policy.cohort_size,
        "cohort_block_sessions": policy.cohort_block_sessions,
        "maximum_holding_sessions": policy.maximum_holding_sessions,
        "historical_core_ticker_batch_size": policy.core_ticker_batch_size,
        "historical_chain_ticker_batch_size": policy.ticker_batch_size,
        "historical_split_ticker_batch_size": policy.split_ticker_batch_size,
        "transition_policy": policy.transition_policy,
        "blocks": blocks,
        "transitions": transitions,
        "unique_sampled_symbols_upper_bound": unique_sampled_symbols,
        "requests": {
            "historical_core": core_calls,
            "base_daily_chain_batches": base_chain_calls,
            "transition_overlap_chain_batches": overlap_chain_calls,
            "historical_chain_total": chain_calls,
            "optional_continuous_entry_extension_chain_batches": (
                continuous_entry_extension_calls
            ),
            "split_history": corporate_action_calls,
            "cold_cache_total": core_calls + chain_calls + corporate_action_calls,
            "cache_adjusted_remaining": remaining,
            "cache_adjusted_total": expected_total,
        },
        "slicing": {
            "per_slice_cap": policy.slice_cap,
            "slice_count": slice_count,
            "exact_slice_attempts": list(exact_slice_attempts),
            "initial_campaign_max_actual_attempts": expected_total,
            "sum_of_generic_slice_caps": generic_cap_sum,
            "unused_cap_capacity_not_authorized": generic_cap_sum - expected_total,
            "attempt_100_within_any_slice_possible": False,
        },
        "research_capacity": {
            "maximum_horizon_eligible_signal_sessions": eligible_signal_sessions,
            "maximum_horizon_ticker_date_candidates_before_signals": (
                eligible_signal_sessions * policy.cohort_size
            ),
            "continuous_entry_extension_total_if_separately_frozen": (
                expected_total + continuous_entry_extension_calls
            ),
        },
        "scope_limits": [
            "estimate assumes one complete point-in-time cohort per block",
            "cohort selection requires a frozen external point-in-time eligibility and liquidity source",
            "entries are censored before every rotation and the final campaign boundary so no selected path is truncated",
            "estimate does not include entitlement discovery",
            "earnings and dividend evidence must come from a separately frozen point-in-time source and is not hidden in this ORATS estimate",
            "failed or incomplete families may require a newly frozen extension campaign",
            "daily production requests are a separate budget",
        ],
    }


def _verified_cohort_manifest(
    cohort_manifest: Mapping[str, Any],
    sessions: Sequence[str],
) -> Tuple[Tuple[str, ...], ...]:
    """Validate a frozen cohort manifest without selecting or changing names."""

    if cohort_manifest.get("schema") != "cultra.rotating-historical-cohorts.v1":
        raise PlanningError("rotating cohort manifest schema is unsupported")
    if cohort_manifest.get("selection_policy") != "POINT_IN_TIME_STRATIFIED_DETERMINISTIC_SAMPLE":
        raise PlanningError("rotating cohorts are not point-in-time frozen")
    supplied_hash = str(cohort_manifest.get("freeze_hash", ""))
    payload = dict(cohort_manifest)
    payload.pop("freeze_hash", None)
    expected_hash = hashlib.sha256(_canonical_json(payload)).hexdigest()
    if supplied_hash != expected_hash:
        raise PlanningError("rotating cohort freeze hash does not reconcile")
    ordered_sessions = tuple(_valid_date(item, "historical session") for item in sessions)
    if ordered_sessions != tuple(sorted(set(ordered_sessions))):
        raise PlanningError("historical sessions must be sorted and unique")
    if int(cohort_manifest.get("session_count", 0)) != len(ordered_sessions):
        raise PlanningError("cohort manifest session count does not reconcile")
    if (
        cohort_manifest.get("session_start") != ordered_sessions[0]
        or cohort_manifest.get("session_end") != ordered_sessions[-1]
    ):
        raise PlanningError("cohort manifest session boundary does not reconcile")
    cohort_size = int(cohort_manifest.get("cohort_size", 0))
    block_sessions = int(cohort_manifest.get("block_sessions", 0))
    maximum_holding = int(cohort_manifest.get("maximum_holding_sessions", 0))
    if cohort_size <= 0 or block_sessions <= 0 or maximum_holding <= 0:
        raise PlanningError("cohort manifest policy is incomplete")
    if cohort_size > 10 or maximum_holding > block_sessions:
        raise PlanningError("cohort manifest exceeds the frozen request geometry")
    if cohort_manifest.get("transition_policy") != (
        "CENSOR_ENTRIES_BEFORE_COHORT_ROTATION"
    ):
        raise PlanningError("cohort transition policy is not frozen")
    if int(cohort_manifest.get("minimum_point_in_time_universe", 0)) < 100:
        raise PlanningError("cohort manifest does not prove broad point-in-time coverage")
    stock_fraction = float(cohort_manifest.get("minimum_stock_fraction", -1.0))
    if not 0.8 <= stock_fraction <= 1.0:
        raise PlanningError("cohort manifest does not prove stock-relevant coverage")
    raw_blocks = cohort_manifest.get("blocks")
    if not isinstance(raw_blocks, list) or not raw_blocks:
        raise PlanningError("cohort manifest has no frozen blocks")
    expected_blocks = int(math.ceil(len(ordered_sessions) / float(block_sessions)))
    if len(raw_blocks) != expected_blocks:
        raise PlanningError("cohort block count does not reconcile")
    blocks = []
    used = set()
    for block_index, raw in enumerate(raw_blocks):
        if not isinstance(raw, Mapping) or int(raw.get("block_index", -1)) != block_index:
            raise PlanningError("cohort block order is invalid")
        offset = block_index * block_sessions
        block = ordered_sessions[offset : offset + block_sessions]
        if raw.get("block_start") != block[0] or raw.get("block_end") != block[-1]:
            raise PlanningError("cohort block dates do not match the session calendar")
        if raw.get("future_membership_used_for_selection") is not False:
            raise PlanningError("cohort selection does not prove future-membership isolation")
        eligible_signal_count = max(0, len(block) - maximum_holding - 1)
        expected_last_signal = (
            block[eligible_signal_count - 1] if eligible_signal_count else None
        )
        if raw.get("eligible_signal_session_count") != eligible_signal_count:
            raise PlanningError("cohort signal-censor count does not reconcile")
        if raw.get("last_eligible_signal_date") != expected_last_signal:
            raise PlanningError("cohort signal cutoff does not reconcile")
        if raw.get("required_coverage_through") != block[-1]:
            raise PlanningError("cohort path coverage is not contained in its block")
        tickers = normalize_entities(tuple(raw.get("tickers", ())), "ticker")
        if len(tickers) != cohort_size:
            raise PlanningError("cohort ticker count does not match the frozen policy")
        if used.intersection(tickers):
            raise PlanningError("cohorts must be disjoint for the frozen request estimate")
        used.update(tickers)
        blocks.append(tickers)
    return tuple(blocks)


def build_rotating_cohort_requests(
    *,
    eligible_symbols: Sequence[str],
    sessions: Sequence[str],
    cohort_manifest: Mapping[str, Any],
    through_date: str,
) -> Tuple[Any, ...]:
    """Build the complete cold-cache request sequence for a frozen campaign.

    This function is tokenless and performs no cache or network access.  Exact
    request identities are produced only after the point-in-time cohorts have
    been frozen.  A separate event-data prerequisite must cover earnings,
    dividends, delistings, and contract adjustments before validation.
    """

    through = _valid_date(through_date, "history through date")
    ordered_sessions = tuple(_valid_date(item, "historical session") for item in sessions)
    if not ordered_sessions or ordered_sessions[-1] != through:
        raise PlanningError("history through date must equal the final frozen session")
    symbols = normalize_entities(eligible_symbols, "ticker")
    if len(symbols) < 100:
        raise PlanningError("historical campaign requires a broad eligible universe")
    cohorts = _verified_cohort_manifest(cohort_manifest, ordered_sessions)
    sampled = {ticker for cohort in cohorts for ticker in cohort}
    if not sampled.issubset(set(symbols)):
        raise PlanningError("frozen cohort contains a symbol outside the eligible universe")
    block_sessions = int(cohort_manifest["block_sessions"])
    maximum_holding = int(cohort_manifest["maximum_holding_sessions"])
    requests = []
    for batch in deterministic_batches(
        tuple(sampled), HISTORICAL_CORE_TICKER_BATCH_SIZE, kind="ticker"
    ):
        requests.append(
            make_planned_request(
                logical_request_id="hist-core-%s" % _suffix((batch, through)),
                endpoint=Endpoint.HIST_CORES,
                run_type=RunType.HISTORICAL_BACKFILL,
                entities=batch,
                fields=HISTORICAL_CORE_FIELDS,
                field_profile="HIST_CORE_SIGNAL_V3",
                purpose="full-history signal features for externally frozen cohorts",
                expected_vintage=through,
                expected_rows=20_000,
                expected_bytes=8_000_000,
                retry_limit=0,
            )
        )
    for session_index, trade_date in enumerate(ordered_sessions):
        block_index = min(session_index // block_sessions, len(cohorts) - 1)
        cohort = cohorts[block_index]
        requests.append(
            make_planned_request(
                logical_request_id="hist-chain-%s-%s"
                % (trade_date.replace("-", ""), _suffix(cohort)),
                endpoint=Endpoint.HIST_STRIKES,
                run_type=RunType.HISTORICAL_BACKFILL,
                entities=cohort,
                fields=HISTORICAL_STRIKE_FIELDS,
                field_profile="HIST_ROTATING_COHORT_CHAIN_V2",
                purpose="complete contemporaneous chain for all frozen hypotheses",
                expected_vintage=trade_date,
                expected_rows=100_000,
                expected_bytes=25_000_000,
                retry_limit=0,
                # Selected contracts remain at least 20 DTE through the frozen
                # path. Delta, however, can legitimately reach an endpoint
                # after a large move; 0..1 coverage prevents outcome-dependent
                # disappearance of winners or losers.
                params={"tradeDate": trade_date, "dte": "20,180", "delta": "0,1"},
            )
        )
    for batch in deterministic_batches(
        tuple(sampled), HISTORICAL_SPLIT_TICKER_BATCH_SIZE, kind="ticker"
    ):
        requests.append(
            make_planned_request(
                logical_request_id="hist-splits-%s" % _suffix((batch, through)),
                endpoint=Endpoint.HIST_SPLITS,
                run_type=RunType.HISTORICAL_BACKFILL,
                entities=batch,
                fields=HISTORICAL_SPLIT_FIELDS,
                field_profile="HIST_SPLITS_V2",
                purpose="corporate-action review for sampled exact contracts",
                expected_vintage=through,
                expected_rows=2_000,
                expected_bytes=1_000_000,
                retry_limit=0,
            )
        )
    if len({item.logical_request_id for item in requests}) != len(requests):
        raise PlanningError("historical campaign contains duplicate request identities")
    return tuple(requests)


def build_rotating_cohort_slices(
    *,
    campaign_id: str,
    eligible_symbols: Sequence[str],
    sessions: Sequence[str],
    cohort_manifest: Mapping[str, Any],
    through_date: str,
    slice_cap: int = DEFAULT_HISTORICAL_SLICE_CAP,
) -> Tuple[RequestPlan, ...]:
    """Partition one frozen historical request set into <=90-attempt plans."""

    cap = _campaign_cap(slice_cap)
    if cap > DEFAULT_HISTORICAL_SLICE_CAP:
        raise PlanningError("historical slice cap cannot exceed 90")
    requests = build_rotating_cohort_requests(
        eligible_symbols=eligible_symbols,
        sessions=sessions,
        cohort_manifest=cohort_manifest,
        through_date=through_date,
    )
    slices = []
    for index in range(0, len(requests), cap):
        values = requests[index : index + cap]
        suffix = index // cap
        slice_campaign = "%s-slice-%02d" % (campaign_id, suffix)
        slices.append(
            RequestPlan(
                run_id=slice_campaign,
                run_type=RunType.HISTORICAL_BACKFILL,
                requests=values,
                target=min(60, len(values)),
                hard_cap=cap,
                retry_reserve=0,
                campaign_id=slice_campaign,
                campaign_hard_cap=cap,
            )
        )
    return tuple(slices)


def daily_request_budget(
    *,
    core_symbols: int,
    summary_symbols: int = 0,
    monies_symbols: int = 0,
    exact_contracts: int = 0,
) -> Mapping[str, int]:
    """Return the documented cold-cache daily call count; retries are zero."""

    values = (core_symbols, summary_symbols, monies_symbols, exact_contracts)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        raise PlanningError("daily request-budget inputs must be nonnegative integers")
    calls = {
        "core": int(math.ceil(core_symbols / 10.0)),
        "summary": int(math.ceil(summary_symbols / 10.0)),
        "monies_implied": int(math.ceil(monies_symbols / 10.0)),
        "monies_forecast": int(math.ceil(monies_symbols / 10.0)),
        "exact_options": int(math.ceil(exact_contracts / 100.0)),
    }
    total = sum(calls.values())
    return dict(
        calls,
        logical_requests=total,
        automatic_retries=0,
        worst_charged_attempts=total,
        daily_logical_cap=60,
        admissible=total <= 60,
        requests_over_cap=max(0, total - 60),
        same_vintage_warm_attempts=0,
    )


def historical_campaign_forecast(
    *,
    symbols: Sequence[str],
    signal_dates: Optional[Iterable[SignalDateKey]] = None,
    exact_strikes: Optional[Iterable[ExactStrikeHistoryKey]] = None,
    cached_feature_batches: Iterable[Tuple[str, ...]] = (),
    cached_signal_groups: Iterable[Tuple[str, Tuple[str, ...]]] = (),
    cached_exact_strikes: Iterable[ExactStrikeHistoryKey] = (),
    campaign_cap: int = DEFAULT_HISTORICAL_CAMPAIGN_CAP,
) -> Mapping[str, Any]:
    """Reject the superseded signal/strike N+1 acquisition design."""

    del (
        symbols,
        signal_dates,
        exact_strikes,
        cached_feature_batches,
        cached_signal_groups,
        cached_exact_strikes,
        campaign_cap,
    )
    raise PlanningError(
        "signal-date plus exact-strike history planning is disabled; "
        "use the rotating-cohort full-chain campaign"
    )


def build_bulk_history_feature_plan(
    *,
    run_id: str,
    campaign_id: str,
    symbols: Sequence[str],
    through_date: str,
    cached_batches: Iterable[Tuple[str, ...]] = (),
    campaign_cap: int = DEFAULT_HISTORICAL_CAMPAIGN_CAP,
) -> RequestPlan:
    """Reject the superseded standalone broad-Core acquisition phase."""

    del run_id, campaign_id, symbols, through_date, cached_batches, campaign_cap
    raise PlanningError(
        "standalone broad historical Core planning is disabled; "
        "use the rotating-cohort frozen campaign"
    )


def build_signal_entry_plan(
    *,
    run_id: str,
    campaign_id: str,
    signal_dates: Iterable[SignalDateKey],
    cached_groups: Iterable[Tuple[str, Tuple[str, ...]]] = (),
    campaign_cap: int = DEFAULT_HISTORICAL_CAMPAIGN_CAP,
) -> RequestPlan:
    """Reject the superseded request-per-signal entry-chain phase."""

    del run_id, campaign_id, signal_dates, cached_groups, campaign_cap
    raise PlanningError(
        "request-per-signal history is disabled; use rotating cohort slices"
    )


def build_exact_strike_history_plan(
    *,
    run_id: str,
    campaign_id: str,
    exact_strikes: Iterable[ExactStrikeHistoryKey],
    through_date: str,
    cached_strikes: Iterable[ExactStrikeHistoryKey] = (),
    campaign_cap: int = DEFAULT_HISTORICAL_CAMPAIGN_CAP,
) -> RequestPlan:
    """Reject the superseded request-per-strike path acquisition phase."""

    del run_id, campaign_id, exact_strikes, through_date, cached_strikes, campaign_cap
    raise PlanningError(
        "request-per-exact-strike history is disabled; use rotating cohort slices"
    )


__all__ = [
    "ABSOLUTE_HISTORICAL_CAMPAIGN_CAP",
    "DEFAULT_HISTORICAL_CAMPAIGN_CAP",
    "DEFAULT_HISTORICAL_SLICE_CAP",
    "ExactStrikeHistoryKey",
    "HISTORICAL_CORE_FIELDS",
    "HISTORICAL_SPLIT_FIELDS",
    "HISTORICAL_STRIKE_FIELDS",
    "SignalDateKey",
    "RotatingCohortPolicy",
    "build_bulk_history_feature_plan",
    "build_exact_strike_history_plan",
    "build_rotating_cohort_requests",
    "build_rotating_cohort_slices",
    "build_signal_entry_plan",
    "daily_request_budget",
    "historical_campaign_forecast",
    "rotating_cohort_campaign_forecast",
]
