"""Command-line entry point for Cultra's offline-safe surface."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, Optional, Sequence

from .artifacts import verify_manifest
from .backfill import execute_chain_backfill
from .campaign import (
    build_historical_campaign_freeze,
    load_historical_campaign_freeze,
    save_historical_campaign_freeze,
)
from .campaign_completion import (
    save_historical_campaign_completion,
    verify_historical_campaign_completion,
)
from .cohorts import (
    freeze_rotating_cohorts,
    load_point_in_time_universe,
    save_rotating_cohorts,
)
from .pipeline import (
    CultraPipeline,
    PipelineRunConfig,
    reference_request_budget,
    run_doctor,
)
from .current import verify_current_research
from .offline_audit import build_offline_audit, verify_offline_audit
from .eod import (
    CORE_SCREEN_FIELDS,
    build_core_plan,
    execute_eod_plan,
    save_chain_finalists,
)
from .historical_v2 import ingest_historical_v2_campaign
from .outcomes_v2 import generate_historical_v2_outcomes
from .modeling_v2 import freeze_historical_v2_models
from .holdout_v2 import consume_historical_v2_holdout
from .patterns import (
    DEFAULT_CHAINS as DEFAULT_PATTERN_CHAINS,
    DEFAULT_HISTORY as DEFAULT_PATTERN_HISTORY,
    DEFAULT_ORATS as DEFAULT_PATTERN_ORATS,
    DEFAULT_SCREEN as DEFAULT_PATTERN_SCREEN,
    verify_pattern_run,
)
from .prerequisites import (
    load_historical_prerequisites,
    prepare_historical_prerequisites,
    save_historical_prerequisites,
)
from .public_history_sources import (
    save_public_history_source_audit,
    verify_public_history_source_audit,
)
from .public_classification import (
    save_public_classification_audit,
    verify_public_classification_audit,
)
from .public_event_audit import (
    save_public_event_audit,
    verify_public_event_audit,
)
from .requesting import build_reference_eod_plan
from .request_optimization import (
    RotatingCohortPolicy,
    rotating_cohort_campaign_forecast,
)
from .research import (
    DEFAULT_CHAIN_DB,
    verify_historical_validation,
)
from .security import bootstrap_orats_env
from .sessions import load_historical_session_calendar
from .universe import (
    rebuild_broad_screen_offline,
    fetch_broad_quote_snapshot,
    fetch_finalist_chains,
    fetch_history_snapshot,
)


def _date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD") from exc


def _json_print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


def _doctor(args: argparse.Namespace) -> int:
    report = run_doctor(Path(args.project_root))
    if args.json:
        _json_print(report.to_dict())
    else:
        for check in report.checks:
            print("%-5s %-26s %s" % (check.status.value, check.name, check.detail))
        print(
            "ENGINEERING RESULT %s; PRODUCTION READY NO; PROFIT CONFIDENCE UNPROVEN"
            % ("PASS" if report.ok else "FAIL")
        )
    return 0 if report.ok else 1


def _plan(args: argparse.Namespace) -> int:
    if not (args.core or args.summary or args.monies or args.option):
        _json_print(reference_request_budget())
        return 0
    plan = build_reference_eod_plan(
        run_id=args.run_id,
        core_symbols=args.core,
        summary_symbols=args.summary,
        monies_symbols=args.monies,
        option_symbols=args.option,
        expected_vintage=args.expected_vintage,
        retry_reserve=0,
    )
    _json_print(plan.to_dict())
    return 0


def _estimate_history(args: argparse.Namespace) -> int:
    forecast = rotating_cohort_campaign_forecast(
        RotatingCohortPolicy(
            eligible_symbols=args.eligible_symbols,
            historical_sessions=args.sessions,
            cohort_size=args.cohort_size,
            cohort_block_sessions=args.block_sessions,
            maximum_holding_sessions=args.maximum_holding_sessions,
            slice_cap=args.slice_cap,
        ),
        cached_core_calls=args.cached_core_calls,
        cached_chain_calls=args.cached_chain_calls,
        cached_corporate_action_calls=args.cached_corporate_action_calls,
    )
    _json_print(forecast)
    return 0


def _run(args: argparse.Namespace) -> int:
    result = CultraPipeline().run(
        PipelineRunConfig(
            as_of=args.as_of,
            output_root=Path(args.output_root),
            run_id=args.run_id,
            overall_status="UNPROVEN",
            execute_orats=False,
        )
    )
    _json_print(
        {
            "run_id": result.run_id,
            "run_dir": str(result.run_dir),
            "board": str(result.board_path),
            "overall_status": result.manifest.overall_status,
            "manual_ticket_count": result.ticket_count,
            "network_attempted": False,
        }
    )
    return 0


def _verify(args: argparse.Namespace) -> int:
    errors = verify_manifest(Path(args.run_dir))
    _json_print({"ok": not errors, "errors": list(errors)})
    return 0 if not errors else 1


def _secrets_bootstrap(args: argparse.Namespace) -> int:
    source = Path(args.source)
    destination = bootstrap_orats_env(source)
    _json_print(
        {
            "ok": True,
            "destination": str(destination.resolve()),
            "copied_key": "ORATS_TOKEN",
            "mode": "0600",
            "network_attempted": False,
        }
    )
    return 0


def _backfill_chains(args: argparse.Namespace) -> int:
    freeze_path = Path(args.campaign_freeze)
    campaign = load_historical_campaign_freeze(freeze_path)
    if args.slice_index < 0 or args.slice_index >= len(campaign.slices):
        raise ValueError(
            "slice_index %d is outside the %d planned slices"
            % (args.slice_index, len(campaign.slices))
        )
    plan = campaign.slices[args.slice_index]
    if args.run_id is not None and args.run_id != plan.run_id:
        raise ValueError("run_id is derived from the frozen campaign and slice")
    if not args.execute:
        _json_print(
            {
                "network_attempted": False,
                "campaign_id": campaign.campaign_id,
                "campaign_freeze_hash": campaign.payload["freeze_hash"],
                "slice_index": args.slice_index,
                "slice_count": len(campaign.slices),
                "first_trade_date": plan.requests[0].expected_vintage,
                "last_trade_date": plan.requests[-1].expected_vintage,
                "plan": plan.to_dict(),
            }
        )
        return 0
    result = execute_chain_backfill(
        plan,
        output_root=Path(args.output_root),
        workers=args.workers,
        campaign_freeze_path=freeze_path,
        slice_index=args.slice_index,
    )
    _json_print(
        {
            "run_id": result.run_id,
            "run_dir": str(result.run_dir),
            "plan_hash": result.plan_hash,
            "completed_dates": len(result.completed_dates),
            "failed_dates": list(result.failed_dates),
            "cache_hits": result.cache_hits,
            "charged_attempts": result.charged_attempts,
        }
    )
    return 0 if not result.failed_dates else 1


def _freeze_history_campaign(args: argparse.Namespace) -> int:
    campaign = build_historical_campaign_freeze(
        campaign_id=args.campaign_id,
        prerequisite_freeze_path=Path(args.prerequisite_freeze),
    )
    manifest = save_historical_campaign_freeze(Path(args.output_dir), campaign)
    _json_print(
        {
            "campaign_id": campaign.campaign_id,
            "campaign_freeze": str(manifest),
            "campaign_freeze_hash": campaign.payload["freeze_hash"],
            "expected_attempts": campaign.payload["request_campaign"][
                "expected_attempts"
            ],
            "slice_attempts": campaign.payload["request_campaign"][
                "slice_attempts"
            ],
            "network_attempted": False,
            "execution_authorized": False,
        }
    )
    return 0


def _prepare_history_inputs(args: argparse.Namespace) -> int:
    prepared = prepare_historical_prerequisites(
        input_set_id=args.input_set_id,
        universe_source_path=Path(args.universe_source),
        session_source_path=Path(args.session_source),
        event_source_path=Path(args.event_source),
    )
    freeze_path = save_historical_prerequisites(Path(args.output_dir), prepared)
    frozen = load_historical_prerequisites(freeze_path)
    _json_print(
        {
            "input_set_id": frozen.input_set_id,
            "prerequisite_freeze": str(freeze_path),
            "prerequisite_freeze_hash": frozen.payload["freeze_hash"],
            "selection_dates": list(frozen.payload["selection_dates"]),
            "sampled_symbols": list(frozen.payload["sampled_symbols"]),
            "sampled_symbol_count": frozen.payload["sampled_symbol_count"],
            "network_attempted": False,
            "orats_source_used": False,
        }
    )
    return 0


def _audit_public_history_sources(args: argparse.Namespace) -> int:
    result = save_public_history_source_audit(
        source_root=Path(args.source_root),
        output_root=Path(args.output_root),
        run_id=args.run_id,
    )
    errors = verify_public_history_source_audit(result.run_dir)
    _json_print(
        {
            "run_id": args.run_id,
            "run_dir": str(result.run_dir),
            "board": str(result.board_path),
            "audit": str(result.audit_path),
            "status": result.status,
            "profit_confidence": "UNPROVEN",
            "orats_attempts": 0,
            "schwab_attempts": 0,
            "paid_data_attempts": 0,
            "audit_network_attempted": False,
            "verified": not errors,
            "verification_errors": list(errors),
        }
    )
    return 0 if not errors else 1


def _verify_public_history_sources(args: argparse.Namespace) -> int:
    errors = verify_public_history_source_audit(Path(args.run_dir))
    _json_print({"ok": not errors, "errors": list(errors)})
    return 0 if not errors else 1


def _classify_public_history_universe(args: argparse.Namespace) -> int:
    result = save_public_classification_audit(
        public_source_audit_dir=Path(args.public_source_audit),
        sec_submission_root=Path(args.sec_submission_root),
        output_root=Path(args.output_root),
        run_id=args.run_id,
    )
    errors = verify_public_classification_audit(result.run_dir)
    _json_print(
        {
            "run_id": args.run_id,
            "run_dir": str(result.run_dir),
            "board": str(result.board_path),
            "audit": str(result.audit_path),
            "universe_source": str(result.universe_source_path),
            "status": result.status,
            "profit_confidence": "UNPROVEN",
            "orats_attempts": 0,
            "schwab_attempts": 0,
            "paid_data_attempts": 0,
            "verified": not errors,
            "verification_errors": list(errors),
        }
    )
    return 0 if not errors else 1


def _verify_public_classification(args: argparse.Namespace) -> int:
    errors = verify_public_classification_audit(Path(args.run_dir))
    _json_print({"ok": not errors, "errors": list(errors)})
    return 0 if not errors else 1


def _audit_public_events(args: argparse.Namespace) -> int:
    result = save_public_event_audit(
        classification_run_dir=Path(args.classification_run),
        event_source_root=Path(args.event_source_root),
        output_root=Path(args.output_root),
        run_id=args.run_id,
    )
    errors = verify_public_event_audit(result.run_dir)
    audit = json.loads(result.audit_path.read_text(encoding="utf-8"))
    _json_print(
        {
            "run_id": args.run_id,
            "run_dir": str(result.run_dir),
            "board": str(result.board_path),
            "audit": str(result.audit_path),
            "status": result.status,
            "profit_confidence": "UNPROVEN",
            "historical_campaign_authorized": False,
            "blocking_event_cells": audit["blocking_cell_count"],
            "orats_attempts": 0,
            "schwab_attempts": 0,
            "paid_data_attempts": 0,
            "audit_network_attempted": False,
            "verified": not errors,
            "verification_errors": list(errors),
        }
    )
    return 0 if not errors else 1


def _verify_public_events(args: argparse.Namespace) -> int:
    errors = verify_public_event_audit(Path(args.run_dir))
    _json_print({"ok": not errors, "errors": list(errors)})
    return 0 if not errors else 1


def _verify_history_campaign(args: argparse.Namespace) -> int:
    completion = verify_historical_campaign_completion(
        campaign_freeze_path=Path(args.campaign_freeze),
        runs_root=Path(args.runs_root),
    )
    artifact = save_historical_campaign_completion(
        Path(args.output_dir), completion
    )
    _json_print(
        {
            "campaign_id": completion["campaign_id"],
            "complete": completion["complete"],
            "completed_requests": completion["completed_requests"],
            "charged_attempts": completion["charged_attempts"],
            "cache_hits": completion["cache_hits"],
            "campaign_completion": str(artifact),
            "network_attempted": False,
        }
    )
    return 0


def _freeze_cohorts(args: argparse.Namespace) -> int:
    universe = load_point_in_time_universe(Path(args.universe))
    calendar = load_historical_session_calendar(Path(args.sessions_file))
    sessions = tuple(item.session_date for item in calendar.sessions)
    manifest = freeze_rotating_cohorts(
        universe,
        sessions,
        cohort_size=args.cohort_size,
        block_sessions=args.block_sessions,
        maximum_holding_sessions=args.maximum_holding_sessions,
        minimum_point_in_time_universe=args.minimum_point_in_time_universe,
        minimum_stock_fraction=args.minimum_stock_fraction,
    )
    output = save_rotating_cohorts(Path(args.output), manifest)
    _json_print(
        {
            "output": str(output),
            "freeze_hash": manifest["freeze_hash"],
            "blocks": len(manifest["blocks"]),
            "sampled_symbols": sum(len(item["tickers"]) for item in manifest["blocks"]),
            "network_attempted": False,
        }
    )
    return 0


def _ingest_history(args: argparse.Namespace) -> int:
    del args
    raise ValueError(
        "legacy V1 ingestion is disabled; use ingest-history-v2 with a verified "
        "complete campaign receipt"
    )


def _ingest_history_v2(args: argparse.Namespace) -> int:
    result = ingest_historical_v2_campaign(
        campaign_completion_path=Path(args.campaign_completion),
        database_path=Path(args.database),
    )
    _json_print(result)
    return 0


def _build_history_outcomes_v2(args: argparse.Namespace) -> int:
    result = generate_historical_v2_outcomes(
        normalized_database=Path(args.normalized_database),
        output_database=Path(args.output_database),
    )
    _json_print(result)
    return 0


def _freeze_history_models_v2(args: argparse.Namespace) -> int:
    result = freeze_historical_v2_models(
        outcome_database=Path(args.outcome_database),
        artifact_path=Path(args.artifact),
        evidence_registry_path=Path(args.evidence_registry),
    )
    _json_print(result)
    return 0


def _consume_history_holdout_v2(args: argparse.Namespace) -> int:
    result = consume_historical_v2_holdout(
        model_artifact_path=Path(args.model_artifact),
        evidence_registry_path=Path(args.evidence_registry),
        output_path=Path(args.output),
    )
    _json_print(result)
    return 0


def _validate_history(args: argparse.Namespace) -> int:
    del args
    raise ValueError(
        "legacy V1 validation is disabled because its exposed evidence cannot "
        "establish V2 holdout passage; use the ingest-history-v2, "
        "build-history-outcomes-v2, freeze-history-models-v2, and "
        "consume-history-holdout-v2 sequence"
    )


def _verify_history(args: argparse.Namespace) -> int:
    errors = verify_historical_validation(Path(args.run_dir))
    _json_print({"ok": not errors, "errors": list(errors)})
    return 0 if not errors else 1


def _research_orders(args: argparse.Namespace) -> int:
    del args
    raise ValueError(
        "legacy ten-ETF research-orders is disabled because its overlap-based "
        "confidence was invalid; preserved artifacts are verification-only"
    )


def _verify_current(args: argparse.Namespace) -> int:
    errors = verify_current_research(Path(args.run_dir))
    _json_print({"ok": not errors, "errors": list(errors)})
    return 0 if not errors else 1


def _broad_screen(args: argparse.Namespace) -> int:
    if not args.execute:
        raise ValueError("broad-screen requires --execute for Schwab market-data access")
    result = fetch_broad_quote_snapshot(
        universe_path=Path(args.universe),
        as_of=args.as_of,
        output_path=Path(args.output),
        orats_capacity=args.orats_capacity,
    )
    _json_print({"output": str(Path(args.output).resolve()), "counts": result["counts"]})
    return 0


def _rebuild_broad_screen_offline(args: argparse.Namespace) -> int:
    result = rebuild_broad_screen_offline(
        source_path=Path(args.source), output_path=Path(args.output)
    )
    _json_print(
        {
            "output": str(Path(args.output).resolve()),
            "counts": result["counts"],
            "network_attempted": result["offline_rebuild"]["network_attempted"],
        }
    )
    return 0


def _broad_history(args: argparse.Namespace) -> int:
    if not args.execute:
        raise ValueError("broad-history requires --execute for Schwab market-data access")
    result = fetch_history_snapshot(
        screen_path=Path(args.screen),
        output_path=Path(args.output),
        as_of=args.as_of,
        workers=args.workers,
    )
    _json_print(
        {
            "output": str(Path(args.output).resolve()),
            "requested": result["requested"],
            "resolved": result["resolved"],
            "errors": len(result["errors"]),
        }
    )
    return 0


def _eod_core(args: argparse.Namespace) -> int:
    screen = json.loads(Path(args.screen).read_text(encoding="utf-8"))
    plan = build_core_plan(
        run_id=args.run_id,
        symbols=screen["orats_admitted_symbols"],
        expected_vintage=args.expected_vintage,
        fields=CORE_SCREEN_FIELDS,
    )
    if not args.execute:
        _json_print({"network_attempted": False, "plan": plan.to_dict()})
        return 0
    result = execute_eod_plan(
        plan, output_root=Path(args.output_root), workers=args.workers
    )
    _json_print(result["counts"])
    return 0


def _select_finalists(args: argparse.Namespace) -> int:
    result = save_chain_finalists(
        history_screen=Path(args.history),
        orats_enrichment=Path(args.orats),
        output_path=Path(args.output),
        capacity=args.capacity,
    )
    _json_print(
        {
            "output": str(Path(args.output).resolve()),
            "selected_symbols": result["selected_symbols"],
            "budget_unresolved": len(result["budget_unresolved"]),
        }
    )
    return 0


def _broad_chains(args: argparse.Namespace) -> int:
    if not args.execute:
        raise ValueError("broad-chains requires --execute for Schwab market-data access")
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    result = fetch_finalist_chains(
        symbols=selection["selected_symbols"],
        output_path=Path(args.output),
        from_date=args.from_date,
        to_date=args.to_date,
        workers=args.workers,
    )
    _json_print(
        {
            "output": str(Path(args.output).resolve()),
            "resolved": result["resolved_count"],
            "errors": result["error_count"],
        }
    )
    return 0


def _refresh_decision_chains(args: argparse.Namespace) -> int:
    if not args.execute:
        raise ValueError(
            "refresh-decision-chains requires --execute for Schwab market-data access"
        )
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    result = fetch_finalist_chains(
        symbols=selection["selected_symbols"],
        output_path=Path(args.output),
        from_date=args.from_date,
        to_date=args.to_date,
        workers=args.workers,
        decision_refresh=True,
    )
    _json_print(
        {
            "output": str(Path(args.output).resolve()),
            "resolved": result["resolved_count"],
            "errors": result["error_count"],
            "decision_quote_refresh": result["decision_quote_refresh"],
        }
    )
    return 0


def _build_opportunities(args: argparse.Namespace) -> int:
    del args
    raise ValueError(
        "legacy V1 opportunity construction is disabled; it cannot create a "
        "V2 evidence-bound manual ticket"
    )


def _verify_opportunities(args: argparse.Namespace) -> int:
    errors = verify_pattern_run(Path(args.run_dir))
    _json_print({"ok": not errors, "errors": list(errors)})
    return 0 if not errors else 1


def _rebuild_patterns(args: argparse.Namespace) -> int:
    del args
    raise ValueError(
        "legacy V1 pattern rebuilding is disabled; preserved V6/V7 artifacts "
        "are not profitability or production evidence"
    )


def _verify_patterns(args: argparse.Namespace) -> int:
    errors = verify_pattern_run(Path(args.run_dir))
    _json_print({"ok": not errors, "errors": list(errors)})
    return 0 if not errors else 1


def _offline_audit(args: argparse.Namespace) -> int:
    result = build_offline_audit(
        run_id=args.run_id,
        output_root=Path(args.output_root),
    )
    _json_print(
        {
            "run_id": result["run_id"],
            "run_dir": result["run_dir"],
            "board": result["board"],
            "charged_attempts": result["orats_usage"]["charged_attempts"],
            "recommended_additional_orats_requests_now": result["decision"][
                "recommended_additional_orats_requests_now"
            ],
            "historical_campaign_expected_attempts": result["decision"][
                "historical_campaign_expected_attempts"
            ],
            "entire_campaign_under_100": result["decision"][
                "entire_campaign_under_100"
            ],
            "each_authorized_slice_under_100": result["decision"][
                "each_authorized_slice_under_100"
            ],
            "request_optimization_status": result["decision"]["status"],
            "production_status": result["production_readiness"]["status"],
            "production_blockers": result["production_readiness"]["blocker_count"],
            "network_attempted": False,
        }
    )
    return 0


def _verify_offline_audit(args: argparse.Namespace) -> int:
    errors = verify_offline_audit(Path(args.run_dir))
    _json_print({"ok": not errors, "errors": list(errors)})
    return 0 if not errors else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cultra",
        description="Clean-room, evidence-gated options research",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser(
        "doctor",
        help="run zero-request architecture, permission, and isolation checks",
    )
    doctor.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parents[1]),
    )
    doctor.add_argument("--json", action="store_true")
    doctor.set_defaults(handler=_doctor)

    plan = subparsers.add_parser(
        "plan",
        help="print the reference budget or an immutable request plan; never fetch",
    )
    plan.add_argument("--run-id", default="cultra-plan")
    plan.add_argument("--expected-vintage", default=date.today().isoformat())
    plan.add_argument("--core", nargs="*", default=[])
    plan.add_argument("--summary", nargs="*", default=[])
    plan.add_argument("--monies", nargs="*", default=[])
    plan.add_argument("--option", nargs="*", default=[])
    plan.set_defaults(handler=_plan)

    estimate_history = subparsers.add_parser(
        "estimate-history",
        help="estimate a complete rotating-cohort historical campaign; never fetch",
    )
    estimate_history.add_argument("--eligible-symbols", type=int, required=True)
    estimate_history.add_argument("--sessions", type=int, default=450)
    estimate_history.add_argument("--cohort-size", type=int, default=10)
    estimate_history.add_argument("--block-sessions", type=int, default=120)
    estimate_history.add_argument("--maximum-holding-sessions", type=int, default=60)
    estimate_history.add_argument("--slice-cap", type=int, default=90)
    estimate_history.add_argument("--cached-core-calls", type=int, default=0)
    estimate_history.add_argument("--cached-chain-calls", type=int, default=0)
    estimate_history.add_argument(
        "--cached-corporate-action-calls", type=int, default=0
    )
    estimate_history.set_defaults(handler=_estimate_history)

    freeze_cohorts = subparsers.add_parser(
        "freeze-cohorts",
        help="freeze leakage-safe rotating cohorts from a Cultra-owned point-in-time universe; never fetch",
    )
    freeze_cohorts.add_argument("--universe", required=True)
    freeze_cohorts.add_argument("--sessions-file", required=True)
    freeze_cohorts.add_argument("--output", required=True)
    freeze_cohorts.add_argument("--cohort-size", type=int, default=10)
    freeze_cohorts.add_argument("--block-sessions", type=int, default=120)
    freeze_cohorts.add_argument("--maximum-holding-sessions", type=int, default=60)
    freeze_cohorts.add_argument("--minimum-point-in-time-universe", type=int, default=100)
    freeze_cohorts.add_argument("--minimum-stock-fraction", type=float, default=0.80)
    freeze_cohorts.set_defaults(handler=_freeze_cohorts)

    prepare_inputs = subparsers.add_parser(
        "prepare-history-inputs",
        help="validate raw independent sources and freeze all historical prerequisites; never fetch",
    )
    prepare_inputs.add_argument("--input-set-id", required=True)
    prepare_inputs.add_argument("--universe-source", required=True)
    prepare_inputs.add_argument("--session-source", required=True)
    prepare_inputs.add_argument("--event-source", required=True)
    prepare_inputs.add_argument("--output-dir", required=True)
    prepare_inputs.set_defaults(handler=_prepare_history_inputs)

    audit_public_sources = subparsers.add_parser(
        "audit-public-history-sources",
        help="validate preserved Cboe/NYSE/OCC/SEC/Nasdaq prerequisite evidence; never fetch",
    )
    audit_public_sources.add_argument("--source-root", required=True)
    audit_public_sources.add_argument("--run-id", required=True)
    audit_public_sources.add_argument(
        "--output-root",
        default=str(Path(__file__).resolve().parents[1] / "out"),
    )
    audit_public_sources.set_defaults(handler=_audit_public_history_sources)

    verify_public_sources = subparsers.add_parser(
        "verify-public-history-sources",
        help="reproduce a saved public-source audit from its preserved raw bytes",
    )
    verify_public_sources.add_argument("run_dir")
    verify_public_sources.set_defaults(handler=_verify_public_history_sources)

    classify_public_universe = subparsers.add_parser(
        "classify-public-history-universe",
        help="freeze point-in-time classifications and rotating cohorts from preserved public evidence; never fetch",
    )
    classify_public_universe.add_argument("--public-source-audit", required=True)
    classify_public_universe.add_argument("--sec-submission-root", required=True)
    classify_public_universe.add_argument("--run-id", required=True)
    classify_public_universe.add_argument(
        "--output-root",
        default=str(Path(__file__).resolve().parents[1] / "out"),
    )
    classify_public_universe.set_defaults(handler=_classify_public_history_universe)

    verify_public_classification = subparsers.add_parser(
        "verify-public-history-classification",
        help="reproduce a saved public point-in-time classification audit",
    )
    verify_public_classification.add_argument("run_dir")
    verify_public_classification.set_defaults(handler=_verify_public_classification)

    audit_public_events = subparsers.add_parser(
        "audit-public-history-events",
        help="audit cohort-scoped public earnings, dividends, transitions, and contract adjustments; never fetch",
    )
    audit_public_events.add_argument("--classification-run", required=True)
    audit_public_events.add_argument("--event-source-root", required=True)
    audit_public_events.add_argument("--run-id", required=True)
    audit_public_events.add_argument(
        "--output-root",
        default=str(Path(__file__).resolve().parents[1] / "out"),
    )
    audit_public_events.set_defaults(handler=_audit_public_events)

    verify_public_events = subparsers.add_parser(
        "verify-public-history-events",
        help="reproduce a saved public event audit from preserved bytes",
    )
    verify_public_events.add_argument("run_dir")
    verify_public_events.set_defaults(handler=_verify_public_events)

    freeze_campaign = subparsers.add_parser(
        "freeze-history-campaign",
        help="freeze every V2 historical input and all 474 request IDs; never fetch",
    )
    freeze_campaign.add_argument("--campaign-id", required=True)
    freeze_campaign.add_argument("--prerequisite-freeze", required=True)
    freeze_campaign.add_argument("--output-dir", required=True)
    freeze_campaign.set_defaults(handler=_freeze_history_campaign)

    verify_campaign = subparsers.add_parser(
        "verify-history-campaign",
        help="reconcile all frozen slices and cached snapshots; never fetch",
    )
    verify_campaign.add_argument("--campaign-freeze", required=True)
    verify_campaign.add_argument(
        "--runs-root",
        default=str(Path(__file__).resolve().parents[1] / "out"),
    )
    verify_campaign.add_argument("--output-dir", required=True)
    verify_campaign.set_defaults(handler=_verify_history_campaign)

    run = subparsers.add_parser(
        "run",
        help="produce an initial UNPROVEN, zero-request, zero-ticket artifact set",
    )
    run.add_argument("--as-of", required=True, type=_date)
    run.add_argument(
        "--output-root",
        default=str(Path(__file__).resolve().parents[1] / "out"),
    )
    run.add_argument("--run-id")
    run.set_defaults(handler=_run)

    verify = subparsers.add_parser("verify", help="verify a saved run manifest")
    verify.add_argument("run_dir")
    verify.set_defaults(handler=_verify)

    bootstrap = subparsers.add_parser(
        "secrets-bootstrap",
        help="copy only ORATS_TOKEN from an explicitly supplied env file",
    )
    bootstrap.add_argument("--source", required=True)
    bootstrap.set_defaults(handler=_secrets_bootstrap)

    backfill = subparsers.add_parser(
        "backfill-chains",
        help="inspect or explicitly execute one immutable campaign-freeze slice",
    )
    backfill.add_argument("--campaign-freeze", required=True)
    backfill.add_argument("--slice-index", type=int, required=True)
    backfill.add_argument("--workers", type=int, default=3)
    backfill.add_argument("--run-id")
    backfill.add_argument(
        "--output-root",
        default=str(Path(__file__).resolve().parents[1] / "out"),
    )
    backfill.add_argument(
        "--execute",
        action="store_true",
        help="make the frozen historical requests; omission prints the plan only",
    )
    backfill.set_defaults(handler=_backfill_chains)

    ingest = subparsers.add_parser(
        "ingest-history",
        help="disabled V1 ingestion surface",
    )
    ingest.add_argument("--database", default=str(DEFAULT_CHAIN_DB))
    ingest.set_defaults(handler=_ingest_history)

    ingest_v2 = subparsers.add_parser(
        "ingest-history-v2",
        help="normalize one completely verified V2 campaign; never fetch",
    )
    ingest_v2.add_argument("--campaign-completion", required=True)
    ingest_v2.add_argument("--database", required=True)
    ingest_v2.set_defaults(handler=_ingest_history_v2)

    outcomes_v2 = subparsers.add_parser(
        "build-history-outcomes-v2",
        help="generate the complete exact-leg candidate/outcome ledger; never fetch",
    )
    outcomes_v2.add_argument("--normalized-database", required=True)
    outcomes_v2.add_argument("--output-database", required=True)
    outcomes_v2.set_defaults(handler=_build_history_outcomes_v2)

    models_v2 = subparsers.add_parser(
        "freeze-history-models-v2",
        help="fit chronological OOF models and freeze calibration without reading holdout outcomes",
    )
    models_v2.add_argument("--outcome-database", required=True)
    models_v2.add_argument("--artifact", required=True)
    models_v2.add_argument("--evidence-registry", required=True)
    models_v2.set_defaults(handler=_freeze_history_models_v2)

    holdout_v2 = subparsers.add_parser(
        "consume-history-holdout-v2",
        help="open the final holdout once, apply all gates and Holm correction, and commit atomically",
    )
    holdout_v2.add_argument("--model-artifact", required=True)
    holdout_v2.add_argument("--evidence-registry", required=True)
    holdout_v2.add_argument("--output", required=True)
    holdout_v2.set_defaults(handler=_consume_history_holdout_v2)

    validate_history = subparsers.add_parser(
        "validate-history",
        help="disabled V1 command; exposed holdout cannot be reused",
    )
    validate_history.add_argument("--database", default=str(DEFAULT_CHAIN_DB))
    validate_history.add_argument(
        "--output-root",
        default=str(Path(__file__).resolve().parents[1] / "out"),
    )
    validate_history.add_argument(
        "--run-id", default="cultra-historical-validation-v1-1"
    )
    validate_history.set_defaults(handler=_validate_history)

    verify_history = subparsers.add_parser(
        "verify-history", help="verify a historical-validation artifact set"
    )
    verify_history.add_argument("run_dir")
    verify_history.set_defaults(handler=_verify_history)

    research_orders = subparsers.add_parser(
        "research-orders",
        help="disabled V1 ten-ETF surface; preserved artifacts are verification-only",
    )
    research_orders.add_argument("--as-of", required=True, type=_date)
    research_orders.add_argument("--run-id")
    research_orders.add_argument("--database", default=str(DEFAULT_CHAIN_DB))
    research_orders.add_argument(
        "--output-root",
        default=str(Path(__file__).resolve().parents[1] / "out"),
    )
    research_orders.set_defaults(handler=_research_orders)

    verify_current = subparsers.add_parser(
        "verify-current",
        help="verify a preserved legacy research artifact only",
    )
    verify_current.add_argument("run_dir")
    verify_current.set_defaults(handler=_verify_current)

    broad_screen = subparsers.add_parser(
        "broad-screen",
        help="quote an explicit point-in-time stocks-and-ETFs universe from Schwab",
    )
    broad_screen.add_argument("--universe", required=True)
    broad_screen.add_argument("--as-of", required=True, type=_date)
    broad_screen.add_argument("--output", required=True)
    broad_screen.add_argument(
        "--orats-capacity",
        type=int,
        default=None,
        help="optional explicit diagnostic cap; production default preserves all eligible names",
    )
    broad_screen.add_argument("--execute", action="store_true")
    broad_screen.set_defaults(handler=_broad_screen)

    offline_screen = subparsers.add_parser(
        "rebuild-screen-offline",
        help="remove a legacy capacity cut from a saved broad-screen snapshot",
    )
    offline_screen.add_argument("--source", required=True)
    offline_screen.add_argument("--output", required=True)
    offline_screen.set_defaults(handler=_rebuild_broad_screen_offline)

    broad_history = subparsers.add_parser(
        "broad-history", help="fetch the locally screened symbols' Schwab history"
    )
    broad_history.add_argument("--screen", required=True)
    broad_history.add_argument("--output", required=True)
    broad_history.add_argument("--as-of", required=True, type=_date)
    broad_history.add_argument("--workers", type=int, default=4)
    broad_history.add_argument("--execute", action="store_true")
    broad_history.set_defaults(handler=_broad_history)

    eod_core = subparsers.add_parser(
        "eod-core", help="plan or execute the full frozen ORATS Core profile"
    )
    eod_core.add_argument("--screen", required=True)
    eod_core.add_argument("--run-id", required=True)
    eod_core.add_argument("--expected-vintage", required=True)
    eod_core.add_argument("--workers", type=int, default=3)
    eod_core.add_argument("--output-root", default=str(Path(__file__).resolve().parents[1] / "out"))
    eod_core.add_argument("--execute", action="store_true")
    eod_core.set_defaults(handler=_eod_core)

    finalists = subparsers.add_parser(
        "select-finalists", help="freeze exact-chain coverage without a default top-N cap"
    )
    finalists.add_argument("--history", required=True)
    finalists.add_argument("--orats", required=True)
    finalists.add_argument("--output", required=True)
    finalists.add_argument(
        "--capacity",
        type=int,
        default=None,
        help="optional explicit diagnostic cap; omission selects every resolved symbol",
    )
    finalists.set_defaults(handler=_select_finalists)

    chains = subparsers.add_parser(
        "broad-chains", help="fetch exact Schwab chains for frozen finalists"
    )
    chains.add_argument("--selection", required=True)
    chains.add_argument("--output", required=True)
    chains.add_argument("--from-date", required=True, type=_date)
    chains.add_argument("--to-date", required=True, type=_date)
    chains.add_argument("--workers", type=int, default=4)
    chains.add_argument("--execute", action="store_true")
    chains.set_defaults(handler=_broad_chains)

    decision_chains = subparsers.add_parser(
        "refresh-decision-chains",
        help="refresh selected exact chains from Schwab for automatic decision-time repricing",
    )
    decision_chains.add_argument("--selection", required=True)
    decision_chains.add_argument("--output", required=True)
    decision_chains.add_argument("--from-date", required=True, type=_date)
    decision_chains.add_argument("--to-date", required=True, type=_date)
    decision_chains.add_argument("--workers", type=int, default=4)
    decision_chains.add_argument("--execute", action="store_true")
    decision_chains.set_defaults(handler=_refresh_decision_chains)

    opportunities = subparsers.add_parser(
        "build-opportunities",
        help="disabled V1 opportunity surface; cannot create V2 evidence-bound tickets",
    )
    opportunities.add_argument("--screen", required=True)
    opportunities.add_argument("--history", required=True)
    opportunities.add_argument("--orats", required=True)
    opportunities.add_argument("--chains", required=True)
    opportunities.add_argument("--selection")
    opportunities.add_argument("--run-id", required=True)
    opportunities.add_argument("--as-of", type=_date)
    opportunities.add_argument("--database", default=str(DEFAULT_CHAIN_DB))
    opportunities.add_argument("--orats-ledger", action="append", default=[])
    opportunities.set_defaults(handler=_build_opportunities)

    verify_opportunities = subparsers.add_parser(
        "verify-opportunities", help="verify the Cultra V6 candidate/action audit"
    )
    verify_opportunities.add_argument("run_dir")
    verify_opportunities.set_defaults(handler=_verify_opportunities)

    patterns = subparsers.add_parser(
        "rebuild-patterns",
        help="disabled V1/V6 pattern surface; preserved artifacts are verification-only",
    )
    patterns.add_argument("--as-of", required=True, type=_date)
    patterns.add_argument("--run-id", required=True)
    patterns.add_argument("--screen", default=str(DEFAULT_PATTERN_SCREEN))
    patterns.add_argument("--history", default=str(DEFAULT_PATTERN_HISTORY))
    patterns.add_argument("--orats", default=str(DEFAULT_PATTERN_ORATS))
    patterns.add_argument("--chains", default=str(DEFAULT_PATTERN_CHAINS))
    patterns.add_argument(
        "--selection",
        default=None,
        help="saved Cultra chain-selection manifest; required for production-readiness evidence",
    )
    patterns.add_argument("--database", default=str(DEFAULT_CHAIN_DB))
    patterns.add_argument("--output-root", default=str(Path(__file__).resolve().parents[1] / "out"))
    patterns.set_defaults(handler=_rebuild_patterns)

    verify_patterns = subparsers.add_parser(
        "verify-patterns", help="verify a Cultra V6 candidate/action artifact set"
    )
    verify_patterns.add_argument("run_dir")
    verify_patterns.set_defaults(handler=_verify_patterns)

    offline_audit = subparsers.add_parser(
        "offline-audit",
        help="reconcile usage, historical coverage, production gaps, and sub-100 feasibility with zero network",
    )
    offline_audit.add_argument("--run-id", required=True)
    offline_audit.add_argument(
        "--output-root",
        default=str(Path(__file__).resolve().parents[1] / "out"),
    )
    offline_audit.set_defaults(handler=_offline_audit)

    verify_audit = subparsers.add_parser(
        "verify-offline-audit",
        help="reconcile a V7 offline audit, its inputs, and artifact hashes",
    )
    verify_audit.add_argument("run_dir")
    verify_audit.set_defaults(handler=_verify_offline_audit)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (OSError, RuntimeError, ValueError) as exc:
        print("cultra: %s" % exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
