"""Strict loader and receipt builder for the Cultra V2 research protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .hypotheses import (
    FROZEN_HYPOTHESIS_COUNT,
    FROZEN_HYPOTHESIS_REGISTRY,
    HYPOTHESIS_REGISTRY_HASH,
    HYPOTHESIS_REGISTRY_VERSION,
)
from .request_optimization import RotatingCohortPolicy, rotating_cohort_campaign_forecast
from .structures import STRUCTURE_TEMPLATE_REGISTRY_HASH, STRUCTURE_TEMPLATE_VERSION


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_CAMPAIGN_CONFIG = PROJECT_ROOT / "configs" / "historical_campaign.v2.json"


class ProtocolError(ValueError):
    """The pre-holdout research protocol is incomplete or has drifted."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _load_json(path: Path) -> Mapping[str, Any]:
    supplied = Path(path).expanduser().resolve()
    if supplied != HISTORICAL_CAMPAIGN_CONFIG.resolve():
        raise ProtocolError("historical research must use the canonical V2 protocol")
    try:
        value = json.loads(supplied.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("historical V2 protocol is unreadable") from exc
    if not isinstance(value, Mapping):
        raise ProtocolError("historical V2 protocol must be an object")
    return value


def load_historical_campaign_protocol(
    path: Path = HISTORICAL_CAMPAIGN_CONFIG,
) -> Mapping[str, Any]:
    value = _load_json(path)
    if value.get("schema") != "cultra.historical-campaign.v2":
        raise ProtocolError("historical protocol schema is unsupported")
    if value.get("version") != "CULTRA_HISTORICAL_CAMPAIGN_V2":
        raise ProtocolError("historical protocol version is not frozen")
    scope = value.get("scope")
    acquisition = value.get("acquisition")
    registry = value.get("hypothesis_registry")
    timing = value.get("timing_policy")
    sessions = value.get("session_calendar_policy")
    learning = value.get("learning_policy")
    split = value.get("split_policy")
    calibration = value.get("calibration_policy")
    promotion = value.get("promotion_policy")
    events = value.get("event_policy")
    costs = value.get("cost_policy")
    prerequisites = value.get("historical_prerequisite_policy")
    universe = value.get("universe_policy")
    if not all(
        isinstance(item, Mapping)
        for item in (
            scope,
            acquisition,
            registry,
            learning,
            split,
            calibration,
            timing,
            sessions,
            costs,
            promotion,
            events,
            prerequisites,
            universe,
        )
    ):
        raise ProtocolError("historical protocol sections are incomplete")
    if scope.get("named_universe") is not None:
        raise ProtocolError("historical protocol cannot contain a named ticker list")
    if scope.get("coverage") != (
        "US_LISTED_SECURITY_UNDERLYINGS_WITH_MIN_1000_DAILY_CBOE_OPTIONS_"
        "VOLUME_ACROSS_2_CBOE_VENUES"
    ):
        raise ProtocolError("historical sampling-frame coverage drifted")
    if scope.get("daily_production_universe_cap") is not None or scope.get("ticket_output_cap") is not None:
        raise ProtocolError("historical protocol cannot impose production or ticket caps")
    if registry.get("version") != HYPOTHESIS_REGISTRY_VERSION:
        raise ProtocolError("hypothesis registry version drifted")
    if registry.get("registry_hash") != HYPOTHESIS_REGISTRY_HASH:
        raise ProtocolError("hypothesis registry hash drifted")
    if int(registry.get("hypothesis_count", 0)) != FROZEN_HYPOTHESIS_COUNT:
        raise ProtocolError("hypothesis registry count drifted")
    if registry.get("structure_template_version") != STRUCTURE_TEMPLATE_VERSION:
        raise ProtocolError("structure template version drifted")
    if registry.get("structure_template_registry_hash") != STRUCTURE_TEMPLATE_REGISTRY_HASH:
        raise ProtocolError("structure template registry hash drifted")
    if learning.get("candidate_generation") != "EVERY_GEOMETRICALLY_EXECUTABLE_TICKER_DATE_STRUCTURE":
        raise ProtocolError("historical candidate generation is not frozen")
    if learning.get("outcome_dependent_prescreen_allowed") is not False:
        raise ProtocolError("historical learning permits outcome-dependent prescreening")
    if learning.get("hyperparameter_selection") != "CHRONOLOGICAL_VALIDATION_ONLY":
        raise ProtocolError("historical hyperparameter selection is not leakage safe")
    if (
        learning.get("probability_model") != "LOGISTIC_L2"
        or learning.get("return_model") != "RIDGE_RETURN_ON_RISK"
        or tuple(float(item) for item in learning.get("l2_grid", ()))
        != (0.1, 1.0, 10.0)
        or int(learning.get("maximum_newton_iterations", 0)) <= 0
        or float(learning.get("convergence_tolerance", 0.0)) <= 0.0
        or learning.get("standardization") != "TRAINING_WINDOW_ONLY"
        or learning.get("raw_model_scores_publishable_as_pop") is not False
    ):
        raise ProtocolError("historical learning model policy drifted")
    feature_profiles = learning.get("feature_profiles")
    if not isinstance(feature_profiles, Mapping) or set(feature_profiles) != {
        item.signal_profile for item in FROZEN_HYPOTHESIS_REGISTRY
    }:
        raise ProtocolError("historical feature profiles do not cover the frozen hypotheses")
    policy = RotatingCohortPolicy(
        eligible_symbols=max(
            int(acquisition["minimum_point_in_time_universe"]),
            int(acquisition["cohort_size"])
            * ((int(acquisition["historical_sessions"]) + int(acquisition["cohort_block_sessions"]) - 1)
               // int(acquisition["cohort_block_sessions"])),
        ),
        historical_sessions=int(acquisition["historical_sessions"]),
        cohort_size=int(acquisition["cohort_size"]),
        cohort_block_sessions=int(acquisition["cohort_block_sessions"]),
        maximum_holding_sessions=int(acquisition["maximum_holding_sessions"]),
        core_ticker_batch_size=int(
            acquisition["historical_core_ticker_batch_size"]
        ),
        ticker_batch_size=int(
            acquisition["historical_chain_ticker_batch_size"]
        ),
        split_ticker_batch_size=int(
            acquisition["historical_split_ticker_batch_size"]
        ),
        slice_cap=int(acquisition["slice_hard_cap"]),
        transition_policy=str(acquisition["transition_policy"]),
    )
    forecast = rotating_cohort_campaign_forecast(policy)
    expected = {
        "expected_cold_attempts": forecast["requests"]["cold_cache_total"],
        "expected_slices": forecast["slicing"]["slice_count"],
        "exact_slice_attempts": forecast["slicing"]["exact_slice_attempts"],
        "initial_campaign_max_actual_attempts": forecast["slicing"][
            "initial_campaign_max_actual_attempts"
        ],
        "sum_of_generic_slice_caps": forecast["slicing"][
            "sum_of_generic_slice_caps"
        ],
        "unused_cap_capacity_not_authorized": forecast["slicing"][
            "unused_cap_capacity_not_authorized"
        ],
        "optional_continuous_entry_extension_attempts": forecast["requests"][
            "optional_continuous_entry_extension_chain_batches"
        ],
    }
    for name, expected_value in expected.items():
        supplied_value = acquisition.get(name)
        if isinstance(expected_value, list):
            matches = supplied_value == expected_value
        else:
            matches = int(supplied_value if supplied_value is not None else -1) == int(expected_value)
        if not matches:
            raise ProtocolError("historical request estimate drifted: %s" % name)
    if timing.get("version") != "SIGNAL_CLOSE_T_ENTRY_T_PLUS_1_V1":
        raise ProtocolError("historical entry timing is not frozen")
    if (
        timing.get("entry_timestamp")
        != "SESSION_T_PLUS_1_CLOSE_EXECUTABLE_QUOTE"
        or timing.get("path_starts") != "FIRST_SESSION_AFTER_ENTRY"
    ):
        raise ProtocolError("historical EOD entry/exit timing is ambiguous")
    if (
        tuple(int(item) for item in timing.get("holding_horizons_sessions", ()))
        != (20, 40, 60)
        or timing.get("ambiguous_daily_target_stop_order") != "STOP_FIRST"
    ):
        raise ProtocolError("historical holding or exit ordering policy drifted")
    if (
        sessions.get("manifest_schema") != "cultra.market-session-calendar.v1"
        or int(sessions.get("required_session_count", 0))
        != int(acquisition["historical_sessions"])
        or sessions.get("market_timezone") != "America/New_York"
        or sessions.get("timezone_aware_close_timestamp_required") is not True
        or sessions.get("entry_quote_observation") != "SESSION_T_PLUS_1_CLOSE"
        or sessions.get("independent_source_required") is not True
        or sessions.get("orats_source_allowed") is not False
    ):
        raise ProtocolError("historical session-calendar policy is incomplete")
    if (
        universe.get("manifest_schema") != "cultra.point-in-time-universe.v1"
        or universe.get("selection_observation") != "EXACT_BLOCK_SELECTION_DATE"
        or universe.get("unknown_or_outcome_fields_rejected") is not True
        or universe.get("future_membership_used_for_selection") is not False
        or universe.get("cohorts_disjoint") is not True
        or universe.get("unresolved_members_preserved_but_not_sampled") is not True
        or universe.get("stock_floor_enforced_during_selection") is not True
        or universe.get("raw_source_hash_binding_required") is not True
        or universe.get("orats_source_allowed") is not False
        or int(universe.get("exact_selection_snapshot_count", 0)) != 4
    ):
        raise ProtocolError("historical point-in-time universe policy is incomplete")
    if (
        split.get("walk_forward") != "EXPANDING"
        or int(split.get("minimum_training_sessions", 0)) != 120
        or int(split.get("validation_sessions", 0)) != 59
        or float(split.get("final_holdout_fraction", 0.0)) != 0.20
        or int(split.get("embargo_sessions_at_every_boundary", 0)) != 60
        or split.get("holdout_use") != "ONCE"
        or tuple(split.get("cluster_dimensions", ())) != ("ticker", "signal_date")
    ):
        raise ProtocolError("historical split policy drifted")
    expected_windows = (
        {"block_index": 0, "role": "RESEARCH", "signal_sessions": 59},
        {"block_index": 1, "role": "TUNING", "signal_sessions": 59},
        {"block_index": 2, "role": "VALIDATION", "signal_sessions": 59},
    )
    if (
        tuple(split.get("development_signal_windows", ())) != expected_windows
        or split.get("holdout_signal_window")
        != {"block_index": 3, "signal_sessions": 29}
    ):
        raise ProtocolError(
            "historical model windows do not align with cohort entry censorship"
        )
    if (
        tuple(calibration.get("targets", ()))
        != ("POP_NET", "P_TARGET", "P_STOP", "P_MAX_LOSS")
        or tuple(calibration.get("candidate_methods", ()))
        != ("LOGISTIC", "ISOTONIC")
        or calibration.get("selection_metric") != "VALIDATION_BRIER"
        or calibration.get("selection_frozen_before_holdout") is not True
        or float(calibration.get("maximum_ece", -1.0)) != 0.05
        or calibration.get("must_beat_unconditional_brier") is not True
        or int(calibration.get("minimum_positive_events_per_target", 0)) != 20
        or int(calibration.get("minimum_negative_events_per_target", 0)) != 20
    ):
        raise ProtocolError("historical probability calibration policy drifted")
    if (
        int(promotion.get("holm_family_size", 0)) != FROZEN_HYPOTHESIS_COUNT
        or tuple(promotion.get("positive_net_expectancy_required_in", ()))
        != ("TRAINING", "VALIDATION", "HOLDOUT")
        or int(promotion.get("minimum_holdout_resolved_trades", 0)) != 100
        or int(promotion.get("minimum_holdout_ticker_date_clusters", 0)) != 40
        or float(promotion.get("minimum_resolution_rate", 0.0)) != 0.95
        or promotion.get("unresolved_worst_case_expectancy_must_be_positive") is not True
        or float(promotion.get("holdout_confidence", 0.0)) != 0.95
        or promotion.get("two_way_cluster_bootstrap_required") is not True
        or float(promotion.get("maximum_holm_adjusted_p_value", -1.0)) != 0.05
        or float(promotion.get("maximum_single_ticker_or_period_profit_fraction", -1.0)) != 0.20
        or int(promotion.get("calendar_concentration_period_sessions", 0)) != 5
        or promotion.get("manual_ticket_minimum_state") != "HOLDOUT_PASS"
        or int(promotion.get("maximum_evidence_age_calendar_days", 0)) != 180
    ):
        raise ProtocolError("promotion policy drifted")
    if (
        events.get("independent_source_manifest_required_before_slice_1") is not True
        or events.get("point_in_time_earnings_required") is not True
        or events.get("point_in_time_dividends_required") is not True
        or events.get("delistings_and_contract_adjustments_required") is not True
        or int(events.get("orats_event_requests_in_base_estimate", -1)) != 0
        or events.get("raw_source_hash_binding_required") is not True
        or events.get("coverage_attestation_required") is not True
        or events.get("sampled_stock_earnings_required_inside_cohort_block") is not True
    ):
        raise ProtocolError("point-in-time event evidence is not a campaign prerequisite")
    if (
        prerequisites.get("freeze_schema")
        != "cultra.historical-prerequisite-freeze.v1"
        or prerequisites.get("raw_source_hash_binding_required") is not True
        or prerequisites.get("orats_source_allowed") is not False
        or prerequisites.get("network_during_preparation_allowed") is not False
    ):
        raise ProtocolError("historical prerequisite source binding is incomplete")
    if (
        costs.get("entry_execution") != "BUY_ASK_SELL_BID"
        or costs.get("exit_execution") != "SELL_BID_BUY_ASK"
        or float(costs.get("commission_per_contract_per_side", -1.0)) < 0.0
        or float(costs.get("fee_per_contract_per_side", -1.0)) < 0.0
        or not 0.0
        <= float(costs.get("slippage_fraction_of_quoted_spread_per_side", -1.0))
        <= 1.0
        or float(costs.get("minimum_slippage_dollars_per_contract_per_side", -1.0))
        < 0.0
        or costs.get("spread_dependent_slippage_required") is not True
        or costs.get("assignment_exercise_dividend_and_partial_fill_stress_required") is not True
    ):
        raise ProtocolError("historical cost policy is incomplete")
    return value


def historical_protocol_hash() -> str:
    return hashlib.sha256(_canonical(load_historical_campaign_protocol())).hexdigest()


def build_campaign_freeze_receipt(
    *,
    cohort_manifest: Mapping[str, Any],
    session_calendar_sha256: str,
    event_manifest_sha256: str,
    prerequisite_freeze_sha256: str,
) -> Mapping[str, Any]:
    """Bind all externally frozen inputs before any historical slice executes."""

    for name, value in (
        ("cohort freeze", cohort_manifest.get("freeze_hash")),
        ("session calendar", session_calendar_sha256),
        ("event manifest", event_manifest_sha256),
        ("historical prerequisite freeze", prerequisite_freeze_sha256),
    ):
        digest = str(value).lower().removeprefix("sha256:")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ProtocolError("%s hash is invalid" % name)
    if cohort_manifest.get("schema") != "cultra.rotating-historical-cohorts.v1":
        raise ProtocolError("cohort manifest schema is unsupported")
    payload = {
        "schema": "cultra.historical-campaign-freeze-receipt.v2",
        "protocol_hash": historical_protocol_hash(),
        "hypothesis_registry_hash": HYPOTHESIS_REGISTRY_HASH,
        "cohort_freeze_hash": str(cohort_manifest["freeze_hash"]),
        "universe_fingerprint": str(cohort_manifest.get("universe_fingerprint", "")),
        "session_calendar_sha256": session_calendar_sha256,
        "event_manifest_sha256": event_manifest_sha256,
        "prerequisite_freeze_sha256": prerequisite_freeze_sha256,
    }
    return dict(payload, receipt_hash=hashlib.sha256(_canonical(payload)).hexdigest())


__all__ = [
    "HISTORICAL_CAMPAIGN_CONFIG",
    "ProtocolError",
    "build_campaign_freeze_receipt",
    "historical_protocol_hash",
    "load_historical_campaign_protocol",
]
