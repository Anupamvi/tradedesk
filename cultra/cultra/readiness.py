"""Offline, fail-closed production-readiness assessment for Cultra."""

from __future__ import annotations

import math
import sqlite3
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .catalog import FROZEN_STRATEGY_CATALOG
from .domain import EvidenceState, FamilyEvidence
from .validation import validate_holdout_pass


class ReadinessError(RuntimeError):
    """Saved inputs cannot support an auditable readiness decision."""


def _check(
    check_id: str,
    passed: bool,
    evidence: Mapping[str, Any],
    required_fix: str,
) -> Mapping[str, Any]:
    return {
        "check_id": check_id,
        "status": "PASS" if passed else "BLOCKED",
        "evidence": dict(evidence),
        "required_fix": None if passed else required_fix,
    }


def _historical_database_coverage(database: Path) -> Mapping[str, Any]:
    resolved = Path(database).expanduser().resolve()
    if not resolved.is_file():
        raise ReadinessError("historical database is unavailable")
    try:
        connection = sqlite3.connect("file:%s?mode=ro" % resolved, uri=True)
        rows = tuple(
            connection.execute(
                """
                SELECT ticker, COUNT(DISTINCT trade_date), MIN(trade_date),
                       MAX(trade_date), COUNT(*)
                  FROM chains
                 GROUP BY ticker
                 ORDER BY ticker
                """
            )
        )
        sessions = int(connection.execute("SELECT COUNT(*) FROM sessions").fetchone()[0])
        chain_rows = int(connection.execute("SELECT COUNT(*) FROM chains").fetchone()[0])
    except sqlite3.Error as exc:
        raise ReadinessError("historical database coverage cannot be read") from exc
    finally:
        try:
            connection.close()
        except UnboundLocalError:
            pass
    return {
        "sessions": sessions,
        "chain_rows": chain_rows,
        "tickers": [str(item[0]) for item in rows],
        "ticker_coverage": {
            str(item[0]): {
                "sessions": int(item[1]),
                "first_date": str(item[2]),
                "last_date": str(item[3]),
                "rows": int(item[4]),
            }
            for item in rows
        },
    }


def assess_production_readiness(
    *,
    screen: Mapping[str, Any],
    history: Mapping[str, Any],
    orats: Mapping[str, Any],
    chains: Mapping[str, Any],
    selection: Optional[Mapping[str, Any]],
    config: Mapping[str, Any],
    models: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    confirmed_events: Mapping[str, Mapping[str, Any]],
    database: Path,
    family_evidence: Sequence[FamilyEvidence] = (),
) -> Mapping[str, Any]:
    """Return exact blockers; never infer readiness from a report label."""

    coverage = _historical_database_coverage(database)
    source_symbols = {str(item["ticker"]) for item in screen.get("quotes", ())}
    admitted = {str(item["ticker"]) for item in screen.get("admitted", ())}
    budget_unresolved = {
        str(item["ticker"]) for item in screen.get("budget_unresolved", ())
    }
    locally_eligible = admitted.union(budget_unresolved)
    history_symbols = {str(item["ticker"]) for item in history.get("rows", ())}
    orats_symbols = {str(item["ticker"]) for item in orats.get("rows", ())}
    chain_symbols = {str(item["ticker"]) for item in chains.get("chains", ())}
    selected_symbols = (
        set()
        if selection is None
        else {str(item) for item in selection.get("selected_symbols", ())}
    )
    candidate_symbols = {str(item["ticker"]) for item in candidates}
    candidate_families = {str(item["strategy_family"]) for item in candidates}
    historical_symbols = set(coverage["tickers"])
    modeled_families = set(str(item) for item in config.get("families", {}))
    catalog_families = {item.strategy_id for item in FROZEN_STRATEGY_CATALOG}
    promotable_catalog_families = {
        item.strategy_id
        for item in FROZEN_STRATEGY_CATALOG
        if item.ticket_eligible_structure
    }
    evidence_by_family = {
        item.strategy_family: item for item in family_evidence
    }
    if len(evidence_by_family) != len(tuple(family_evidence)):
        raise ReadinessError("duplicate family evidence")

    checks = []
    checks.append(
        _check(
            "BROAD_SOURCE_UNIVERSE",
            len(source_symbols) >= 100
            and screen.get("universe", {}).get("coverage")
            == "US_LISTED_LIQUID_OPTIONABLE_STOCKS_AND_ETFS",
            {
                "source_symbols": len(source_symbols),
                "coverage": screen.get("universe", {}).get("coverage"),
            },
            "supply an explicit point-in-time liquid-optionable U.S. stocks-and-ETFs universe",
        )
    )
    checks.append(
        _check(
            "NO_ARBITRARY_CURRENT_UNIVERSE_CAP",
            admitted.isdisjoint(budget_unresolved)
            and locally_eligible == admitted.union(budget_unresolved),
            {
                "locally_eligible_symbols": len(locally_eligible),
                "admitted_symbols": len(admitted),
                "budget_suppressed_symbols": len(budget_unresolved),
            },
            "reconcile every locally eligible name as evaluated or NOT_FULLY_EVALUATED_BUDGET",
        )
    )
    checks.append(
        _check(
            "CURRENT_ANALYTICS_COVERAGE",
            candidate_symbols.issubset(orats_symbols.intersection(history_symbols)),
            {
                "locally_eligible": len(locally_eligible),
                "history_resolved": len(history_symbols),
                "orats_resolved": len(orats_symbols),
                "candidate_missing_history": sorted(candidate_symbols - history_symbols),
                "candidate_missing_orats": sorted(candidate_symbols - orats_symbols),
            },
            "move any candidate lacking mandatory history or analytics to DATA_UNAVAILABLE",
        )
    )
    checks.append(
        _check(
            "REPRODUCIBLE_CHAIN_SELECTION",
            selection is not None and candidate_symbols.issubset(selected_symbols),
            {
                "selection_manifest_available": selection is not None,
                "selected_symbols": len(selected_symbols),
                "candidate_symbols": len(candidate_symbols),
            },
            "supply a fingerprinted selection manifest covering every evaluated candidate",
        )
    )
    checks.append(
        _check(
            "EXACT_CHAIN_COVERAGE",
            candidate_symbols.issubset(chain_symbols),
            {
                "selected_symbols": len(selected_symbols),
                "saved_exact_chains": len(chain_symbols),
                "candidate_missing_exact_chains": sorted(candidate_symbols - chain_symbols),
                "chain_errors": int(chains.get("error_count", 0)),
            },
            "move candidates without complete read-only Schwab chains to DATA_UNAVAILABLE",
        )
    )
    evidence_domain_safe = all(
        item.point_in_time_membership and item.next_session_entry
        for item in family_evidence
    )
    checks.append(
        _check(
            "HISTORICAL_DOMAIN_MATCH",
            bool(family_evidence) and evidence_domain_safe,
            {
                "historical_tickers": sorted(historical_symbols),
                "candidate_tickers": sorted(candidate_symbols),
                "evidence_families": sorted(evidence_by_family),
                "point_in_time_and_timing_safe": evidence_domain_safe,
            },
            "bind each family to point-in-time broad-domain evidence and next-session entry timing",
        )
    )
    membership_safe_families = sorted(
        item.strategy_family
        for item in family_evidence
        if item.point_in_time_membership
    )
    checks.append(
        _check(
            "POINT_IN_TIME_HISTORICAL_MEMBERSHIP",
            bool(family_evidence)
            and len(membership_safe_families) == len(tuple(family_evidence)),
            {"verified_families": membership_safe_families},
            "freeze point-in-time universe membership so historical evidence is not survivorship-biased",
        )
    )
    checks.append(
        _check(
            "FROZEN_CATALOG_IMPLEMENTATION",
            modeled_families.issubset(catalog_families),
            {
                "catalog_family_count": len(catalog_families),
                "modeled_family_count": len(modeled_families),
                "missing_families": sorted(catalog_families - modeled_families),
                "promotable_catalog_family_count": len(promotable_catalog_families),
            },
            "remove any modeled family outside the frozen catalog; unimplemented families remain UNPROVEN and do not block another family",
        )
    )

    family_calibration = {}
    family_edge = {}
    for family in sorted(candidate_families):
        result = models.get(family, {})
        metrics = result.get("metrics") or {}
        pop = (metrics.get("probabilities") or {}).get("POP_NET") or {}
        family_calibration[family] = {
            "status": pop.get("status"),
            "selected_method": pop.get("selected_method"),
            "oof_observations": pop.get("oof_observations"),
            "brier": pop.get("oof_brier"),
            "base_brier": pop.get("base_rate_brier"),
            "ece": pop.get("expected_calibration_error"),
            "gate_pass": bool(metrics.get("pop_gate_pass")),
            "gate_reasons": list(metrics.get("pop_gate_reasons", ())),
        }
        return_model = metrics.get("return_model") or {}
        family_edge[family] = {
            "gate_pass": bool(metrics.get("ev_gate_pass")),
            "gate_reasons": list(metrics.get("ev_gate_reasons", ())),
            "oof_mse": return_model.get("oof_mse"),
            "base_mse": return_model.get("base_mean_mse"),
            "selected_95_lower_return_on_risk": return_model.get(
                "selected_oof_95_lower_return_on_risk"
            ),
        }
    checks.append(
        _check(
            "CALIBRATED_POP_GATE",
            bool(family_calibration)
            and all(item["gate_pass"] for item in family_calibration.values()),
            {"families": family_calibration},
            "the nested chronological calibration is implemented; reject or redesign each failing frozen signal using new development evidence, then require calibrated POP to beat base Brier with ECE at or below 0.05",
        )
    )
    checks.append(
        _check(
            "CONSERVATIVE_EDGE_GATE",
            bool(family_edge) and all(item["gate_pass"] for item in family_edge.values()),
            {"families": family_edge},
            "retain only families whose out-of-fold return model beats baseline and whose clustered lower-bound selected return is positive",
        )
    )
    holdout_failures = {
        family: list(validate_holdout_pass(evidence))
        for family, evidence in sorted(evidence_by_family.items())
        if evidence.state in {EvidenceState.HOLDOUT_PASS, EvidenceState.SHADOW_PASS}
    }
    enabled_evidence = {
        family: reasons
        for family, reasons in holdout_failures.items()
        if not reasons
    }
    required_evidence_families = candidate_families or set(evidence_by_family)
    holdout_ready = bool(enabled_evidence) and required_evidence_families.issubset(
        set(enabled_evidence)
    )
    checks.append(
        _check(
            "NEW_UNTOUCHED_HOLDOUT",
            holdout_ready,
            {
                "enabled_families": sorted(enabled_evidence),
                "required_families": sorted(required_evidence_families),
                "family_gate_failures": holdout_failures,
            },
            "supply registry-backed FamilyEvidence; a config status string cannot prove holdout passage",
        )
    )
    trade_dates = {
        str(item.get("tradeDate")) for item in orats.get("rows", ()) if item.get("tradeDate")
    }
    event_window_days = int(
        float(
            config.get("manual_research_action_policy", {}).get(
                "earnings_confirmation_window_weeks", 0
            )
        )
        * 7.0
    )
    provider_date = None
    if len(trade_dates) == 1:
        try:
            provider_date = date.fromisoformat(next(iter(trade_dates)))
        except ValueError:
            provider_date = None
    missing_events = sorted(candidate_symbols - set(confirmed_events))
    blocking_events = []
    cleared_events = []
    for ticker in sorted(candidate_symbols.intersection(confirmed_events)):
        try:
            event_date = date.fromisoformat(str(confirmed_events[ticker]["date"]))
        except (KeyError, TypeError, ValueError):
            blocking_events.append(ticker)
            continue
        if (
            provider_date is None
            or event_window_days <= 0
            or event_date <= provider_date + timedelta(days=event_window_days)
        ):
            blocking_events.append(ticker)
        else:
            cleared_events.append(ticker)
    checks.append(
        _check(
            "EXACT_EVENT_COVERAGE",
            bool(candidate_symbols) and not missing_events and not blocking_events,
            {
                "candidate_symbols": len(candidate_symbols),
                "provider_trade_date": (
                    None if provider_date is None else provider_date.isoformat()
                ),
                "holding_window_days": event_window_days,
                "confirmed_event_records": len(candidate_symbols) - len(missing_events),
                "missing_event_records": missing_events,
                "event_records_blocking_entry": blocking_events,
                "event_records_clearing_holding_window": cleared_events,
            },
            "resolve an authoritative next earnings date for every candidate and require it to fall after the holding window",
        )
    )
    quote_refresh = chains.get("decision_quote_refresh")
    checks.append(
        _check(
            "FRESH_EXECUTABLE_SCHWAB_QUOTES",
            isinstance(quote_refresh, Mapping)
            and quote_refresh.get("complete") is True
            and quote_refresh.get("source") == "SCHWAB"
            and quote_refresh.get("purpose") == "MARKET_OPEN_DECISION"
            and quote_refresh.get("broker_order_surface") is False
            and candidate_symbols.issubset(
                set(quote_refresh.get("requested_symbols", ()))
            )
            and candidate_symbols.issubset(
                set(quote_refresh.get("resolved_symbols", ()))
            ),
            {"decision_quote_refresh": quote_refresh},
            "at decision time, refresh the underlying and every exact leg through Schwab and rebuild executable economics automatically",
        )
    )
    finite_costed = True
    malformed = []
    for candidate in candidates:
        try:
            economics = candidate["economics"]
            maximum_loss = float(economics["maximum_loss"])
            if (
                not math.isfinite(maximum_loss)
                or maximum_loss <= 0.0
                or float(economics["commissions_and_fees"]) < 0.0
                or float(economics["modeled_round_trip_slippage"]) < 0.0
            ):
                raise ValueError
        except (KeyError, TypeError, ValueError):
            finite_costed = False
            malformed.append(str(candidate.get("candidate_id")))
    checks.append(
        _check(
            "FINITE_COSTED_ONE_UNIT_ECONOMICS",
            bool(candidates) and finite_costed,
            {"candidate_count": len(candidates), "malformed_candidates": malformed},
            "rebuild every exact structure with finite maximum loss and complete commissions, fees, and slippage",
        )
    )

    blockers = [item for item in checks if item["status"] == "BLOCKED"]
    return {
        "schema": "cultra.production-readiness.v1",
        "status": "READY" if not blockers else "BLOCKED",
        "profit_confidence": "UNPROVEN" if blockers else "HISTORICALLY_VALIDATED",
        "historically_validated_action_enabled": not blockers,
        "manual_ticket_enabled": not blockers,
        "shadow_policy": (
            "PROSPECTIVE_SHADOW_IS_CONTINUOUS_MONITORING_NOT_A_90_DAY_WAIT_GATE; "
            "a new broad-domain untouched holdout is still mandatory"
        ),
        "check_count": len(checks),
        "blocker_count": len(blockers),
        "checks": checks,
        "historical_database": coverage,
    }


def render_readiness_markdown(readiness: Mapping[str, Any]) -> str:
    lines = [
        "## Production readiness",
        "",
        "- Status: **%s**" % readiness["status"],
        "- Blocking checks: **%d of %d**"
        % (int(readiness["blocker_count"]), int(readiness["check_count"])),
        "- Historically validated action enabled: **%s**"
        % ("YES" if readiness["historically_validated_action_enabled"] else "NO"),
        "- Manual research ticket enabled: **%s**"
        % ("YES" if readiness["manual_ticket_enabled"] else "NO"),
        "- 90-day wait: **not a gate**. A new untouched historical holdout remains mandatory.",
        "",
        "| Check | Status | Exact blocker/fix |",
        "|---|---:|---|",
    ]
    for item in readiness["checks"]:
        lines.append(
            "| `%s` | %s | %s |"
            % (
                item["check_id"],
                "✅ PASS" if item["status"] == "PASS" else "⛔ BLOCKED",
                "—" if item["required_fix"] is None else str(item["required_fix"]),
            )
        )
    return "\n".join(lines)


__all__ = [
    "ReadinessError",
    "assess_production_readiness",
    "render_readiness_markdown",
]
