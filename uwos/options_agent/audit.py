"""Repeatable multi-date audit artifacts for the independent Options Agent."""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import shlex
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import pandas as pd

from uwos.options_agent.core import (
    CORE_AUDIT_TICKERS,
    LIVE_SPREAD_QUALITY_AUDIT_COLUMNS,
    MARKET_OPEN_RECHECK_COLUMNS,
    MAX_LIVE_QUOTE_WIDTH_PCT,
    MIN_AGENTIC_REVIEW_LANES_PER_TICKER,
    MIN_EXECUTION_CONFIDENCE_SCORE,
    MIN_EXPECTANCY_PROFIT_FACTOR,
    MIN_EXPECTANCY_SAMPLE_SIZE,
    MIN_EXPECTANCY_WIN_RATE,
    MIN_LIVE_LEG_LIQUIDITY,
    MONTHLY_PROFIT_TARGET,
    PIPELINE_VERSION,
    GREEN_TICKET_EXPECTANCY_EVIDENCE_TYPES,
    annotate_actual_forward_expectancy,
    annotate_profitability_calibration,
    build_coverage_audit,
    build_expectancy_evidence,
    build_execution_readiness,
    build_management_plan,
    build_market_open_recheck_queue,
    build_monthly_feasibility,
    build_profitability_calibration,
    build_trade_tickets,
    next_regular_market_session_start,
    output_paths,
    render_report,
    split_trade_ticket_surfaces,
    summarize_expectancy_evidence,
    summarize_execution_readiness,
    summarize_live_spread_quality,
    summarize_monthly_feasibility,
    summarize_profitability_calibration,
    synthesize_decision_board,
    tickers_match,
)
from ._vendor.paths import project_root


SUMMARY_COLUMNS = [
    "date",
    "validation_lane",
    "mode",
    "pipeline_version",
    "decision_rows",
    "trade_ticket_rows",
    "green_ready_orders",
    "yellow_target_candidates",
    "market_open_recheck_queue",
    "ready_one_cycle_max_profit",
    "target_candidate_max_profit",
    "execution_readiness",
    "execution_blockers",
    "monthly_feasibility",
    "monthly_blockers",
    "expectancy_status",
    "expectancy_summary_status",
    "expectancy_sample_size",
    "expectancy_note",
    "agentic_status",
    "subagent_task_count",
    "external_review_count",
    "external_review_agent_count",
    "agentic_review_coverage_basis",
    "agentic_review_coverage_pct",
    "agentic_review_lane_coverage_pct",
    "broad_review_coverage_pct",
    "fresh_live_quotes_ready",
    "portfolio_ready",
    "market_session_open",
    "green_symbols",
    "target_symbols",
    "market_open_recheck_symbols",
    "all_ticket_symbols",
    "source_dir",
]

TICKET_REVIEW_LANE_COLUMNS = [
    "date",
    "validation_lane",
    "ticker",
    "ready_to_enter",
    "target_order_status",
    "entry_type",
    "external_agent_review_count",
    "external_agent_distinct_review_count",
    "external_agent_review_agents",
    "run_agentic_review_coverage_basis",
    "run_agentic_review_coverage_pct",
    "run_agentic_review_lane_coverage_pct",
    "run_broad_review_coverage_pct",
    "run_agentic_reviews_ready",
]

AGENTIC_COVERAGE_PROOF_COLUMNS = [
    "status",
    "run_date_count",
    "ticket_rows",
    "agentic_ready_date_count",
    "agentic_ready_dates",
    "ticket_date_count",
    "ticket_dates",
    "ticket_rows_with_agentic_ready",
    "ticket_rows_without_agentic_ready",
    "ticket_agentic_coverage_pct",
    "required_min_ticket_lanes",
    "min_ticket_distinct_review_count",
    "ticket_rows_below_min_ticket_lanes",
    "below_min_ticket_lane_dates",
    "non_agentic_ticket_date_count",
    "non_agentic_ticket_dates",
    "required_coverage",
    "claim",
    "note",
]

EXPECTANCY_AUDIT_COLUMNS = [
    "date",
    "validation_lane",
    "source",
    "evidence_type",
    "status",
    "sample_size",
    "win_rate",
    "avg_pnl",
    "total_pnl",
    "profit_factor",
    "matched_current_tickers",
    "matched_current_count",
    "note",
    "source_dir",
]

EXPECTANCY_PROOF_PACKET_COLUMNS = [
    "status",
    "date_count",
    "ticket_ticker_count",
    "ticket_tickers",
    "current_green_ready_orders",
    "expectancy_summary_statuses",
    "blocking_source_counts",
    "forward_realized_statuses",
    "actual_closed_trade_statuses",
    "replay_statuses",
    "minimum_sample_size",
    "minimum_win_rate",
    "minimum_profit_factor",
    "monthly_profit_target",
    "monthly_claim_allowed",
    "required_evidence",
    "note",
]

TICKET_EXPECTANCY_COVERAGE_COLUMNS = [
    "ticker",
    "ticket_rows",
    "green_ticket_rows",
    "actual_forward_pass_sources",
    "actual_forward_block_sources",
    "actual_forward_sample_size",
    "replay_pass_sources",
    "replay_block_sources",
    "replay_sample_size",
    "status",
    "note",
]

TICKET_EXPECTANCY_PROOF_COLUMNS = [
    "status",
    "ticket_ticker_count",
    "green_ticker_count",
    "tickers_with_positive_actual_forward",
    "green_tickers_without_positive_actual_forward",
    "replay_only_tickers",
    "ticket_tickers",
    "green_tickers",
    "required_evidence",
    "note",
]

MONTHLY_FEASIBILITY_GUARDRAIL_COLUMNS = [
    "status",
    "run_count",
    "monthly_file_count",
    "required_metric",
    "runs_with_required_metric",
    "missing_required_metric_count",
    "missing_required_metric_runs",
    "pass_without_required_metric_runs",
    "claim",
    "note",
]

ACTIONABILITY_SURFACE_PROOF_COLUMNS = [
    "status",
    "ticket_rows",
    "ready_to_enter_rows",
    "target_order_rows",
    "target_ready_to_enter_rows",
    "target_missing_entry_type_rows",
    "target_missing_entry_limit_rows",
    "target_missing_trade_plan_rows",
    "target_missing_plain_language_leg_rows",
    "target_green_label_rows",
    "target_green_icon_rows",
    "green_ticket_rows",
    "valid_green_ticket_rows",
    "invalid_green_ticket_rows",
    "live_market_open_recheck_rows",
    "entry_types",
    "bad_examples",
    "claim",
    "note",
]

ACTION_SURFACE_UNDERLYING_QUALITY_COLUMNS = [
    "status",
    "ticket_rows",
    "market_open_recheck_rows",
    "focus_rows",
    "ticket_bad_underlying_rows",
    "market_open_recheck_bad_underlying_rows",
    "focus_bad_actionable_rows",
    "ticket_bad_tickers",
    "market_open_recheck_bad_tickers",
    "focus_bad_actionable_tickers",
    "core_ticket_rows",
    "liquid_ticket_rows",
    "audit_only_focus_rows",
    "audit_only_focus_tickers",
    "liquid_non_core_action_tickers",
    "claim",
    "note",
]

VALIDATION_COVERAGE_PROOF_COLUMNS = [
    "status",
    "base_dir",
    "validation_start",
    "validation_end",
    "base_available_source_date_count",
    "window_available_source_date_count",
    "tested_date_count",
    "tested_available_date_count",
    "untested_available_date_count",
    "available_dates_outside_window_count",
    "tested_dates",
    "untested_available_dates",
    "available_dates_outside_window",
    "note",
]

UNDERLYING_QUALITY_PROOF_COLUMNS = [
    "status",
    "ticket_rows",
    "core_ticket_rows",
    "liquid_non_core_ticket_rows",
    "speculative_ticket_rows",
    "excluded_ticket_rows",
    "unknown_ticket_rows",
    "not_core_or_liquid_ticket_rows",
    "green_not_core_or_liquid_rows",
    "target_not_core_or_liquid_rows",
    "ticket_tickers",
    "liquid_non_core_ticket_tickers",
    "not_core_or_liquid_ticket_tickers",
    "focus_rows",
    "focus_speculative_rows",
    "focus_excluded_rows",
    "focus_liquid_non_core_rows",
    "focus_speculative_examples",
    "focus_excluded_examples",
    "focus_liquid_non_core_examples",
    "claim",
    "note",
]

LIVE_SPREAD_QUALITY_ROLLUP_COLUMNS = [
    "date",
    "validation_lane",
    "source_dir",
    *LIVE_SPREAD_QUALITY_AUDIT_COLUMNS,
]

LIVE_SPREAD_QUALITY_PROOF_COLUMNS = [
    "status",
    "audited_rows",
    "pass_rows",
    "block_rows",
    "quote_width_block_rows",
    "liquidity_block_rows",
    "blocked_not_target_candidate_rows",
    "blocked_still_actionable_rows",
    "target_candidate_rows",
    "target_candidate_block_rows",
    "blocked_tickers",
    "blocked_examples",
    "required_gate",
    "claim",
    "note",
]

MAJOR_NAME_COVERAGE_PROOF_COLUMNS = [
    "status",
    "required_ticker_count",
    "covered_required_ticker_count",
    "missing_required_ticker_count",
    "required_focus_rows",
    "required_rows_with_reason",
    "required_rows_missing_reason",
    "ready_ticket_tickers",
    "yellow_target_tickers",
    "review_ticket_tickers",
    "structured_not_final_tickers",
    "candidate_not_structured_tickers",
    "no_directional_edge_tickers",
    "source_missing_tickers",
    "blocked_or_excluded_tickers",
    "missing_required_tickers",
    "required_tickers",
    "examples",
    "claim",
    "note",
]

MARKET_QUEUE_AUDIT_COLUMNS = [
    "date",
    "validation_lane",
    "source_kind",
    "source_dir",
    *MARKET_OPEN_RECHECK_COLUMNS,
]

MARKET_OPEN_RECHECK_DETAIL_COLUMNS = [
    "date",
    "validation_lane",
    "source_kind",
    "source_dir",
    "ticker",
    "entry_type",
    "target_order_status",
    "order_readiness",
    "entry_limit",
    "target_exit",
    "max_profit",
    "max_loss",
    "position_max_profit",
    "position_max_loss",
    "suggested_contracts",
    "execution_confidence_score",
    "trade_quality_confidence_rating",
    "external_agent_distinct_review_count",
    "only_market_session_blocker",
    "target_status_pass",
    "order_readiness_pass",
    "positive_entry_pass",
    "positive_contracts_pass",
    "confidence_score_pass",
    "trade_quality_confidence_pass",
    "agentic_lanes_pass",
    "plain_language_legs_pass",
    "row_pass",
    "fail_reasons",
    "trade_plan",
]

MARKET_OPEN_RECHECK_PROOF_COLUMNS = [
    "status",
    "queue_rows",
    "live_queue_rows",
    "row_pass_rows",
    "row_fail_rows",
    "only_market_session_blocker_rows",
    "target_status_pass_rows",
    "order_readiness_pass_rows",
    "positive_entry_rows",
    "positive_contract_rows",
    "confidence_score_pass_rows",
    "trade_quality_confidence_pass_rows",
    "agentic_lane_pass_rows",
    "plain_language_leg_rows",
    "credit_rows",
    "debit_rows",
    "tickers",
    "failed_examples",
    "required_gate",
    "claim",
    "note",
]

LIVE_PROBE_COLUMNS = [
    "date",
    "validation_lane",
    "source_dir",
    "fresh_live_quotes_ready",
    "portfolio_ready",
    "agentic_reviews_ready",
    "agentic_review_lane_coverage_pct",
    "market_session_open",
    "green_ready_orders",
    "yellow_target_candidates",
    "market_open_recheck_queue",
    "expectancy_summary_status",
    "execution_readiness",
    "execution_blockers",
]

MULTI_DATE_READINESS_PROOF_COLUMNS = [
    "status",
    "validation_date_count",
    "tested_dates",
    "dated_ticket_rows",
    "dated_green_ready_orders",
    "dated_yellow_target_candidates",
    "dates_with_tickets",
    "dates_with_green_ready_orders",
    "dates_with_yellow_target_candidates",
    "live_probe_count",
    "live_probe_dates",
    "live_market_session_open_count",
    "live_green_ready_orders",
    "live_yellow_recheck_rows",
    "latest_live_probe_date",
    "latest_live_probe_status",
    "claim",
    "note",
]

MARKET_OPEN_EXECUTION_PACKET_COLUMNS = [
    "date",
    "source_dir",
    "status",
    "fresh_live_quotes_ready",
    "portfolio_ready",
    "agentic_reviews_ready",
    "market_session_open",
    "green_ready_orders",
    "yellow_recheck_rows",
    "next_regular_session_start",
    "market_calendar_note",
    "agent_reviews_json",
    "out_dir",
    "command",
    "required_condition",
    "note",
]

LIVE_RERUN_PREFLIGHT_DETAIL_COLUMNS = [
    "ticker",
    "required_review_lanes",
    "review_count",
    "distinct_agent_count",
    "agents",
    "review_file",
    "row_pass",
    "fail_reasons",
]

LIVE_RERUN_PREFLIGHT_PROOF_COLUMNS = [
    "status",
    "queue_ticker_count",
    "queue_tickers",
    "covered_queue_ticker_count",
    "missing_queue_ticker_count",
    "missing_queue_tickers",
    "agent_reviews_json",
    "agent_reviews_json_exists",
    "agent_reviews_json_valid",
    "agent_review_rows",
    "distinct_agent_count",
    "rerun_command_has_live_schwab",
    "rerun_command_has_live_portfolio",
    "rerun_command_has_agent_reviews_json",
    "rerun_out_dir",
    "rerun_out_dir_clear",
    "source_date",
    "source_date_available",
    "failed_examples",
    "required_gate",
    "claim",
    "note",
]

MARKET_SESSION_VERIFICATION_PLAN_COLUMNS = [
    "date",
    "status",
    "next_regular_session_start",
    "yellow_recheck_rows",
    "green_ready_orders",
    "rerun_command",
    "rerun_out_dir",
    "green_ticket_file",
    "trade_ticket_file",
    "execution_readiness_file",
    "expectancy_file",
    "pass_criteria",
    "fail_criteria",
    "completion_gate",
    "note",
]

POST_RERUN_VERIFICATION_COLUMNS = [
    "status",
    "date",
    "market_session_open",
    "green_ticket_status",
    "ticket_expectancy_status",
    "completion_verdict_status",
    "can_mark_goal_complete",
    "update_goal_action",
    "green_ticket_rows",
    "valid_green_ticket_rows",
    "invalid_green_ticket_rows",
    "green_ticker_count",
    "monthly_claim_allowed",
    "rerun_command",
    "audit_regeneration_command",
    "evidence_files",
    "required_next_action",
    "note",
]

READINESS_DASHBOARD_COLUMNS = [
    "area",
    "status",
    "evidence",
    "artifact",
    "required_next_action",
]

CUTOFF_VISIBILITY_PROOF_COLUMNS = [
    "status",
    "run_count",
    "candidate_rows",
    "research_task_rows",
    "qualified_candidate_rows",
    "priced_candidate_rows",
    "final_rows",
    "expected_no_trade_rows",
    "no_trade_audit_rows",
    "candidate_research_mismatch_runs",
    "priced_missing_qualified_runs",
    "no_trade_missing_expected_runs",
    "problem_runs",
    "claim",
    "note",
]

GREEN_TICKET_EXECUTION_DETAIL_COLUMNS = [
    "date",
    "validation_lane",
    "source_dir",
    "ticker",
    "ready_to_enter",
    "order_readiness",
    "entry_type",
    "entry_limit",
    "suggested_contracts",
    "live_validation_status",
    "execution_status",
    "execution_blockers",
    "execution_confidence_score",
    "execution_confidence_rating",
    "trade_quality_confidence_rating",
    "confidence_score_pass",
    "execution_confidence_pass",
    "trade_quality_confidence_pass",
    "market_session_open",
    "trade_plan",
    "sell_leg",
    "buy_leg",
    "row_pass",
    "fail_reasons",
]

GREEN_TICKET_EXECUTION_PROOF_COLUMNS = [
    "status",
    "live_probe_count",
    "green_ticket_rows",
    "valid_green_ticket_rows",
    "invalid_green_ticket_rows",
    "ready_to_enter_rows",
    "positive_entry_rows",
    "positive_contract_rows",
    "live_validation_pass_rows",
    "no_blocker_rows",
    "confidence_score_pass_rows",
    "execution_confidence_pass_rows",
    "trade_quality_confidence_pass_rows",
    "plain_language_leg_rows",
    "market_session_open_rows",
    "green_tickers",
    "invalid_examples",
    "required_evidence",
    "note",
]

SESSION_ONLY_GREEN_SHADOW_PROOF_COLUMNS = [
    "status",
    "shadow_candidate_rows",
    "row_pass_rows",
    "row_fail_rows",
    "non_session_blocker_rows",
    "credit_rows",
    "debit_rows",
    "position_max_profit",
    "position_max_loss",
    "tickers",
    "failed_examples",
    "required_next_action",
    "claim",
    "note",
]

TARGET_PRESERVATION_COLUMNS = ["metric", "value", "status", "evidence"]
GOAL_COMPLETION_COLUMNS = ["requirement", "status", "evidence", "artifact", "remaining_gap"]
COMPLETION_VERDICT_COLUMNS = [
    "can_mark_goal_complete",
    "status",
    "proven_requirements",
    "blocking_requirements",
    "market_open_packet_status",
    "next_regular_session_start",
    "monthly_claim_allowed",
    "expectancy_packet_status",
    "ticket_expectancy_packet_status",
    "update_goal_action",
    "note",
]


@dataclass(frozen=True)
class ExpandedAuditArtifacts:
    """Paths and primary frames written by an expanded audit run."""

    paths: Mapping[str, Path]
    summary: pd.DataFrame
    tickets: pd.DataFrame
    market_open_recheck_queue: pd.DataFrame
    goal_completion: pd.DataFrame


def write_expanded_audit(
    *,
    base_dir: Path,
    run_dirs: Sequence[Path],
    output_prefix: Path,
    live_probe_dirs: Sequence[Path] = (),
    quality_run_dirs: Sequence[Path] = (),
    rerun_agent_reviews_json: Optional[Path] = None,
    focus_tickers: Sequence[str] = CORE_AUDIT_TICKERS,
) -> ExpandedAuditArtifacts:
    """Write repeatable multi-date goal-audit artifacts from Options Agent output directories."""

    resolved_runs = [path.expanduser().resolve() for path in run_dirs]
    resolved_live = [path.expanduser().resolve() for path in live_probe_dirs]
    resolved_quality = [path.expanduser().resolve() for path in quality_run_dirs]
    resolved_rerun_reviews = rerun_agent_reviews_json.expanduser().resolve() if rerun_agent_reviews_json else None
    output_prefix = output_prefix.expanduser().resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    run_summaries = pd.DataFrame([summarize_run(path) for path in resolved_runs], columns=SUMMARY_COLUMNS)
    live_summaries = pd.DataFrame([summarize_run(path) for path in resolved_live], columns=SUMMARY_COLUMNS)
    tickets = combine_run_csvs(resolved_runs, "trade_tickets.csv")
    coverage = combine_run_csvs(resolved_runs, "ticker_coverage_audit.csv")
    focus = build_focus_coverage(coverage, focus_tickers)
    lanes = build_ticket_review_lanes(tickets, run_summaries)
    agentic_coverage_proof = build_agentic_coverage_proof_packet(
        summary=run_summaries,
        ticket_review_lanes=lanes,
    )
    expectancy = combine_expectancy_audit([*resolved_runs, *resolved_live])
    live_probe_summary = build_live_probe_summary(live_summaries)
    market_queue = combine_market_open_recheck_queue(resolved_runs, resolved_live)
    market_open_recheck_details = build_market_open_recheck_details(market_queue)
    validation_coverage_proof = build_validation_coverage_proof_packet(
        base_dir=base_dir.expanduser().resolve(),
        summary=run_summaries,
    )
    cutoff_visibility_proof = build_cutoff_visibility_proof_packet(resolved_runs)
    live_spread_quality = combine_run_csvs(
        [*resolved_runs, *resolved_live, *resolved_quality],
        "live_spread_quality_audit.csv",
    )
    live_spread_quality_proof = build_live_spread_quality_proof_packet(live_spread_quality)
    underlying_quality_proof = build_underlying_quality_proof_packet(
        tickets=tickets,
        focus_coverage=focus,
    )
    major_name_coverage_proof = build_major_name_coverage_proof_packet(
        focus_coverage=focus,
        focus_tickers=focus_tickers,
    )
    expectancy_proof_packet = build_expectancy_proof_packet(
        summary=run_summaries,
        tickets=tickets,
        expectancy=expectancy,
        live_probe_summary=live_probe_summary,
    )
    execution_packet = build_market_open_execution_packet(
        base_dir=base_dir.expanduser().resolve(),
        live_probe_dirs=resolved_live,
        live_probe_summary=live_probe_summary,
        market_open_recheck_queue=market_queue,
        rerun_agent_reviews_json=resolved_rerun_reviews,
    )
    market_open_recheck_proof = build_market_open_recheck_proof_packet(
        market_open_recheck_details,
        market_open_execution_packet=execution_packet,
    )
    live_rerun_preflight_details = build_live_rerun_preflight_details(
        market_open_recheck_details=market_open_recheck_details,
        market_open_execution_packet=execution_packet,
    )
    live_rerun_preflight_proof = build_live_rerun_preflight_proof_packet(
        base_dir=base_dir.expanduser().resolve(),
        market_open_recheck_details=market_open_recheck_details,
        market_open_execution_packet=execution_packet,
        preflight_details=live_rerun_preflight_details,
    )
    multi_date_readiness_proof = build_multi_date_readiness_proof_packet(
        summary=run_summaries,
        live_probe_summary=live_probe_summary,
        market_open_execution_packet=execution_packet,
    )
    market_session_verification_plan = build_market_session_verification_plan(
        market_open_execution_packet=execution_packet,
    )
    green_ticket_execution_details = build_green_ticket_execution_details(
        live_probe_dirs=resolved_live,
        live_probe_summary=live_probe_summary,
    )
    green_ticket_execution_proof = build_green_ticket_execution_proof_packet(
        details=green_ticket_execution_details,
        live_probe_summary=live_probe_summary,
    )
    session_only_shadow_proof = build_session_only_green_shadow_proof_packet(
        market_open_recheck_details,
    )
    actionability_surface_proof = build_actionability_surface_proof_packet(
        tickets=tickets,
        green_ticket_execution_proof=green_ticket_execution_proof,
        market_open_recheck_queue=market_queue,
    )
    action_surface_underlying_quality_proof = build_action_surface_underlying_quality_proof_packet(
        tickets=tickets,
        market_open_recheck_queue=market_queue,
        focus_coverage=focus,
    )
    ticket_expectancy_coverage = build_ticket_expectancy_coverage(
        tickets=tickets,
        green_ticket_execution_details=green_ticket_execution_details,
        expectancy=expectancy,
    )
    ticket_expectancy_proof = build_ticket_expectancy_proof_packet(
        coverage=ticket_expectancy_coverage,
    )
    monthly_feasibility_guardrail_proof = build_monthly_feasibility_guardrail_proof_packet(
        [*resolved_runs, *resolved_live],
    )
    target_audit = build_target_preservation_audit(run_summaries, tickets, market_queue)

    paths = {
        "summary": output_prefix.with_name(output_prefix.name + "_summary.csv"),
        "summary_md": output_prefix.with_name(output_prefix.name + "_summary.md"),
        "tickets": output_prefix.with_name(output_prefix.name + "_tickets.csv"),
        "coverage": output_prefix.with_name(output_prefix.name + "_coverage.csv"),
        "focus_coverage": output_prefix.with_name(output_prefix.name + "_focus_coverage.csv"),
        "ticket_review_lanes": output_prefix.with_name(output_prefix.name + "_ticket_review_lanes.csv"),
        "agentic_coverage_proof_packet": output_prefix.with_name(output_prefix.name + "_agentic_coverage_proof_packet.csv"),
        "agentic_coverage_proof_packet_md": output_prefix.with_name(output_prefix.name + "_agentic_coverage_proof_packet.md"),
        "validation_coverage_proof_packet": output_prefix.with_name(output_prefix.name + "_validation_coverage_proof_packet.csv"),
        "validation_coverage_proof_packet_md": output_prefix.with_name(output_prefix.name + "_validation_coverage_proof_packet.md"),
        "cutoff_visibility_proof_packet": output_prefix.with_name(output_prefix.name + "_cutoff_visibility_proof_packet.csv"),
        "cutoff_visibility_proof_packet_md": output_prefix.with_name(output_prefix.name + "_cutoff_visibility_proof_packet.md"),
        "live_spread_quality_audit": output_prefix.with_name(output_prefix.name + "_live_spread_quality_audit.csv"),
        "live_spread_quality_proof_packet": output_prefix.with_name(output_prefix.name + "_live_spread_quality_proof_packet.csv"),
        "live_spread_quality_proof_packet_md": output_prefix.with_name(output_prefix.name + "_live_spread_quality_proof_packet.md"),
        "underlying_quality_proof_packet": output_prefix.with_name(output_prefix.name + "_underlying_quality_proof_packet.csv"),
        "underlying_quality_proof_packet_md": output_prefix.with_name(output_prefix.name + "_underlying_quality_proof_packet.md"),
        "major_name_coverage_proof_packet": output_prefix.with_name(output_prefix.name + "_major_name_coverage_proof_packet.csv"),
        "major_name_coverage_proof_packet_md": output_prefix.with_name(output_prefix.name + "_major_name_coverage_proof_packet.md"),
        "expectancy_scope_audit": output_prefix.with_name(output_prefix.name + "_expectancy_scope_audit.csv"),
        "expectancy_scope_audit_md": output_prefix.with_name(output_prefix.name + "_expectancy_scope_audit.md"),
        "expectancy_proof_packet": output_prefix.with_name(output_prefix.name + "_expectancy_proof_packet.csv"),
        "expectancy_proof_packet_md": output_prefix.with_name(output_prefix.name + "_expectancy_proof_packet.md"),
        "market_open_recheck_queue": output_prefix.with_name(output_prefix.name + "_market_open_recheck_queue.csv"),
        "market_open_recheck_queue_md": output_prefix.with_name(output_prefix.name + "_market_open_recheck_queue.md"),
        "market_open_recheck_details": output_prefix.with_name(output_prefix.name + "_market_open_recheck_details.csv"),
        "market_open_recheck_proof_packet": output_prefix.with_name(output_prefix.name + "_market_open_recheck_proof_packet.csv"),
        "market_open_recheck_proof_packet_md": output_prefix.with_name(output_prefix.name + "_market_open_recheck_proof_packet.md"),
        "market_open_execution_packet": output_prefix.with_name(output_prefix.name + "_market_open_execution_packet.csv"),
        "market_open_execution_packet_md": output_prefix.with_name(output_prefix.name + "_market_open_execution_packet.md"),
        "live_rerun_preflight_details": output_prefix.with_name(output_prefix.name + "_live_rerun_preflight_details.csv"),
        "live_rerun_preflight_proof_packet": output_prefix.with_name(output_prefix.name + "_live_rerun_preflight_proof_packet.csv"),
        "live_rerun_preflight_proof_packet_md": output_prefix.with_name(output_prefix.name + "_live_rerun_preflight_proof_packet.md"),
        "multi_date_readiness_proof_packet": output_prefix.with_name(output_prefix.name + "_multi_date_readiness_proof_packet.csv"),
        "multi_date_readiness_proof_packet_md": output_prefix.with_name(output_prefix.name + "_multi_date_readiness_proof_packet.md"),
        "market_session_verification_plan": output_prefix.with_name(output_prefix.name + "_market_session_verification_plan.csv"),
        "market_session_verification_plan_md": output_prefix.with_name(output_prefix.name + "_market_session_verification_plan.md"),
        "post_rerun_verification_packet": output_prefix.with_name(output_prefix.name + "_post_rerun_verification_packet.csv"),
        "post_rerun_verification_packet_md": output_prefix.with_name(output_prefix.name + "_post_rerun_verification_packet.md"),
        "green_ticket_execution_details": output_prefix.with_name(output_prefix.name + "_green_ticket_execution_details.csv"),
        "green_ticket_execution_proof_packet": output_prefix.with_name(output_prefix.name + "_green_ticket_execution_proof_packet.csv"),
        "green_ticket_execution_proof_packet_md": output_prefix.with_name(output_prefix.name + "_green_ticket_execution_proof_packet.md"),
        "session_only_green_shadow_proof_packet": output_prefix.with_name(output_prefix.name + "_session_only_green_shadow_proof_packet.csv"),
        "session_only_green_shadow_proof_packet_md": output_prefix.with_name(output_prefix.name + "_session_only_green_shadow_proof_packet.md"),
        "actionability_surface_proof_packet": output_prefix.with_name(output_prefix.name + "_actionability_surface_proof_packet.csv"),
        "actionability_surface_proof_packet_md": output_prefix.with_name(output_prefix.name + "_actionability_surface_proof_packet.md"),
        "action_surface_underlying_quality_proof_packet": output_prefix.with_name(output_prefix.name + "_action_surface_underlying_quality_proof_packet.csv"),
        "action_surface_underlying_quality_proof_packet_md": output_prefix.with_name(output_prefix.name + "_action_surface_underlying_quality_proof_packet.md"),
        "ticket_expectancy_coverage": output_prefix.with_name(output_prefix.name + "_ticket_expectancy_coverage.csv"),
        "ticket_expectancy_proof_packet": output_prefix.with_name(output_prefix.name + "_ticket_expectancy_proof_packet.csv"),
        "ticket_expectancy_proof_packet_md": output_prefix.with_name(output_prefix.name + "_ticket_expectancy_proof_packet.md"),
        "monthly_feasibility_guardrail_proof_packet": output_prefix.with_name(output_prefix.name + "_monthly_feasibility_guardrail_proof_packet.csv"),
        "monthly_feasibility_guardrail_proof_packet_md": output_prefix.with_name(output_prefix.name + "_monthly_feasibility_guardrail_proof_packet.md"),
        "live_probe_summary": output_prefix.with_name(output_prefix.name + "_live_probe_summary.csv"),
        "live_probe_summary_md": output_prefix.with_name(output_prefix.name + "_live_probe_summary.md"),
        "target_preservation_audit": output_prefix.with_name(output_prefix.name + "_target_preservation_audit.csv"),
        "target_preservation_audit_md": output_prefix.with_name(output_prefix.name + "_target_preservation_audit.md"),
        "goal_completion_audit": output_prefix.with_name(output_prefix.name + "_goal_completion_audit.csv"),
        "goal_completion_audit_md": output_prefix.with_name(output_prefix.name + "_goal_completion_audit.md"),
        "completion_verdict": output_prefix.with_name(output_prefix.name + "_completion_verdict.csv"),
        "completion_verdict_md": output_prefix.with_name(output_prefix.name + "_completion_verdict.md"),
        "readiness_dashboard": output_prefix.with_name(output_prefix.name + "_readiness_dashboard.csv"),
        "readiness_dashboard_md": output_prefix.with_name(output_prefix.name + "_readiness_dashboard.md"),
    }

    goal_audit = build_goal_completion_audit(
        summary=run_summaries,
        tickets=tickets,
        focus_coverage=focus,
        ticket_review_lanes=lanes,
        agentic_coverage_proof=agentic_coverage_proof,
        validation_coverage_proof=validation_coverage_proof,
        cutoff_visibility_proof=cutoff_visibility_proof,
        live_spread_quality_proof=live_spread_quality_proof,
        underlying_quality_proof=underlying_quality_proof,
        major_name_coverage_proof=major_name_coverage_proof,
        expectancy=expectancy,
        market_open_recheck_queue=market_queue,
        market_open_recheck_proof=market_open_recheck_proof,
        live_rerun_preflight_proof=live_rerun_preflight_proof,
        live_probe_summary=live_probe_summary,
        multi_date_readiness_proof=multi_date_readiness_proof,
        actionability_surface_proof=actionability_surface_proof,
        action_surface_underlying_quality_proof=action_surface_underlying_quality_proof,
        green_ticket_execution_proof=green_ticket_execution_proof,
        ticket_expectancy_proof=ticket_expectancy_proof,
        paths=paths,
    )
    completion_verdict = build_completion_verdict(
        goal_audit=goal_audit,
        market_open_execution_packet=execution_packet,
        expectancy_proof_packet=expectancy_proof_packet,
        ticket_expectancy_proof_packet=ticket_expectancy_proof,
    )
    audit_live_probe_dirs = _planned_live_probe_dirs(market_session_verification_plan) or resolved_live
    post_rerun_verification = build_post_rerun_verification_packet(
        market_session_verification_plan=market_session_verification_plan,
        live_probe_summary=live_probe_summary,
        green_ticket_execution_proof=green_ticket_execution_proof,
        ticket_expectancy_proof=ticket_expectancy_proof,
        completion_verdict=completion_verdict,
        audit_regeneration_command=_audit_regeneration_command(
            base_dir=base_dir.expanduser().resolve(),
            summary_csv=paths["summary"],
            live_probe_dirs=audit_live_probe_dirs,
            quality_run_dirs=resolved_quality,
            rerun_agent_reviews_json=resolved_rerun_reviews,
            output_prefix=output_prefix,
        ),
    )
    readiness_dashboard = build_readiness_dashboard(
        goal_audit=goal_audit,
        completion_verdict=completion_verdict,
        post_rerun_verification=post_rerun_verification,
        session_only_shadow_proof=session_only_shadow_proof,
        action_surface_underlying_quality_proof=action_surface_underlying_quality_proof,
        monthly_feasibility_guardrail_proof=monthly_feasibility_guardrail_proof,
        paths=paths,
    )

    _write_frame(run_summaries, paths["summary"], SUMMARY_COLUMNS)
    _write_summary_markdown(run_summaries, paths["summary_md"])
    _write_frame(tickets, paths["tickets"])
    _write_frame(coverage, paths["coverage"])
    _write_frame(focus, paths["focus_coverage"])
    _write_frame(lanes, paths["ticket_review_lanes"], TICKET_REVIEW_LANE_COLUMNS)
    _write_frame(agentic_coverage_proof, paths["agentic_coverage_proof_packet"], AGENTIC_COVERAGE_PROOF_COLUMNS)
    _write_agentic_coverage_proof_markdown(
        paths["agentic_coverage_proof_packet_md"],
        agentic_coverage_proof,
    )
    _write_frame(validation_coverage_proof, paths["validation_coverage_proof_packet"], VALIDATION_COVERAGE_PROOF_COLUMNS)
    _write_validation_coverage_proof_markdown(
        paths["validation_coverage_proof_packet_md"],
        validation_coverage_proof,
    )
    _write_frame(cutoff_visibility_proof, paths["cutoff_visibility_proof_packet"], CUTOFF_VISIBILITY_PROOF_COLUMNS)
    _write_cutoff_visibility_proof_markdown(
        paths["cutoff_visibility_proof_packet_md"],
        cutoff_visibility_proof,
    )
    _write_frame(live_spread_quality, paths["live_spread_quality_audit"], LIVE_SPREAD_QUALITY_ROLLUP_COLUMNS)
    _write_frame(live_spread_quality_proof, paths["live_spread_quality_proof_packet"], LIVE_SPREAD_QUALITY_PROOF_COLUMNS)
    _write_live_spread_quality_proof_markdown(
        paths["live_spread_quality_proof_packet_md"],
        live_spread_quality_proof,
    )
    _write_frame(underlying_quality_proof, paths["underlying_quality_proof_packet"], UNDERLYING_QUALITY_PROOF_COLUMNS)
    _write_underlying_quality_proof_markdown(
        paths["underlying_quality_proof_packet_md"],
        underlying_quality_proof,
    )
    _write_frame(major_name_coverage_proof, paths["major_name_coverage_proof_packet"], MAJOR_NAME_COVERAGE_PROOF_COLUMNS)
    _write_major_name_coverage_proof_markdown(
        paths["major_name_coverage_proof_packet_md"],
        major_name_coverage_proof,
    )
    _write_frame(expectancy, paths["expectancy_scope_audit"], EXPECTANCY_AUDIT_COLUMNS)
    _write_markdown_table(paths["expectancy_scope_audit_md"], "Options Agent Expectancy Scope Audit", expectancy)
    _write_frame(expectancy_proof_packet, paths["expectancy_proof_packet"], EXPECTANCY_PROOF_PACKET_COLUMNS)
    _write_expectancy_proof_packet_markdown(paths["expectancy_proof_packet_md"], expectancy_proof_packet, expectancy)
    _write_frame(market_queue, paths["market_open_recheck_queue"], MARKET_QUEUE_AUDIT_COLUMNS)
    _write_markdown_table(paths["market_open_recheck_queue_md"], "Options Agent Market Open Recheck Queue", market_queue)
    _write_frame(market_open_recheck_details, paths["market_open_recheck_details"], MARKET_OPEN_RECHECK_DETAIL_COLUMNS)
    _write_frame(market_open_recheck_proof, paths["market_open_recheck_proof_packet"], MARKET_OPEN_RECHECK_PROOF_COLUMNS)
    _write_market_open_recheck_proof_markdown(
        paths["market_open_recheck_proof_packet_md"],
        market_open_recheck_proof,
        market_open_recheck_details,
    )
    _write_frame(execution_packet, paths["market_open_execution_packet"], MARKET_OPEN_EXECUTION_PACKET_COLUMNS)
    _write_market_open_execution_packet_markdown(
        paths["market_open_execution_packet_md"],
        execution_packet,
        market_queue,
    )
    _write_frame(live_rerun_preflight_details, paths["live_rerun_preflight_details"], LIVE_RERUN_PREFLIGHT_DETAIL_COLUMNS)
    _write_frame(live_rerun_preflight_proof, paths["live_rerun_preflight_proof_packet"], LIVE_RERUN_PREFLIGHT_PROOF_COLUMNS)
    _write_live_rerun_preflight_proof_markdown(
        paths["live_rerun_preflight_proof_packet_md"],
        live_rerun_preflight_proof,
        live_rerun_preflight_details,
    )
    _write_frame(
        multi_date_readiness_proof,
        paths["multi_date_readiness_proof_packet"],
        MULTI_DATE_READINESS_PROOF_COLUMNS,
    )
    _write_multi_date_readiness_proof_markdown(
        paths["multi_date_readiness_proof_packet_md"],
        multi_date_readiness_proof,
    )
    _write_frame(
        market_session_verification_plan,
        paths["market_session_verification_plan"],
        MARKET_SESSION_VERIFICATION_PLAN_COLUMNS,
    )
    _write_market_session_verification_plan_markdown(
        paths["market_session_verification_plan_md"],
        market_session_verification_plan,
    )
    _write_frame(
        post_rerun_verification,
        paths["post_rerun_verification_packet"],
        POST_RERUN_VERIFICATION_COLUMNS,
    )
    _write_post_rerun_verification_markdown(
        paths["post_rerun_verification_packet_md"],
        post_rerun_verification,
    )
    _write_frame(
        green_ticket_execution_details,
        paths["green_ticket_execution_details"],
        GREEN_TICKET_EXECUTION_DETAIL_COLUMNS,
    )
    _write_frame(
        green_ticket_execution_proof,
        paths["green_ticket_execution_proof_packet"],
        GREEN_TICKET_EXECUTION_PROOF_COLUMNS,
    )
    _write_green_ticket_execution_proof_markdown(
        paths["green_ticket_execution_proof_packet_md"],
        green_ticket_execution_proof,
        green_ticket_execution_details,
    )
    _write_frame(
        session_only_shadow_proof,
        paths["session_only_green_shadow_proof_packet"],
        SESSION_ONLY_GREEN_SHADOW_PROOF_COLUMNS,
    )
    _write_session_only_green_shadow_proof_markdown(
        paths["session_only_green_shadow_proof_packet_md"],
        session_only_shadow_proof,
        market_open_recheck_details,
    )
    _write_frame(actionability_surface_proof, paths["actionability_surface_proof_packet"], ACTIONABILITY_SURFACE_PROOF_COLUMNS)
    _write_actionability_surface_proof_markdown(
        paths["actionability_surface_proof_packet_md"],
        actionability_surface_proof,
    )
    _write_frame(
        action_surface_underlying_quality_proof,
        paths["action_surface_underlying_quality_proof_packet"],
        ACTION_SURFACE_UNDERLYING_QUALITY_COLUMNS,
    )
    _write_action_surface_underlying_quality_proof_markdown(
        paths["action_surface_underlying_quality_proof_packet_md"],
        action_surface_underlying_quality_proof,
    )
    _write_frame(ticket_expectancy_coverage, paths["ticket_expectancy_coverage"], TICKET_EXPECTANCY_COVERAGE_COLUMNS)
    _write_frame(ticket_expectancy_proof, paths["ticket_expectancy_proof_packet"], TICKET_EXPECTANCY_PROOF_COLUMNS)
    _write_ticket_expectancy_proof_markdown(
        paths["ticket_expectancy_proof_packet_md"],
        ticket_expectancy_proof,
        ticket_expectancy_coverage,
    )
    _write_frame(
        monthly_feasibility_guardrail_proof,
        paths["monthly_feasibility_guardrail_proof_packet"],
        MONTHLY_FEASIBILITY_GUARDRAIL_COLUMNS,
    )
    _write_markdown_table(
        paths["monthly_feasibility_guardrail_proof_packet_md"],
        "Options Agent Monthly Feasibility Guardrail Proof Packet",
        monthly_feasibility_guardrail_proof,
    )
    _write_frame(live_probe_summary, paths["live_probe_summary"], LIVE_PROBE_COLUMNS)
    _write_markdown_table(paths["live_probe_summary_md"], "Options Agent Live Probe Summary", live_probe_summary)
    _write_frame(target_audit, paths["target_preservation_audit"], TARGET_PRESERVATION_COLUMNS)
    _write_markdown_table(paths["target_preservation_audit_md"], "Options Agent Target Preservation Audit", target_audit)
    _write_frame(goal_audit, paths["goal_completion_audit"], GOAL_COMPLETION_COLUMNS)
    _write_goal_completion_markdown(goal_audit, paths["goal_completion_audit_md"])
    _write_frame(completion_verdict, paths["completion_verdict"], COMPLETION_VERDICT_COLUMNS)
    _write_completion_verdict_markdown(paths["completion_verdict_md"], completion_verdict, goal_audit)
    _write_frame(readiness_dashboard, paths["readiness_dashboard"], READINESS_DASHBOARD_COLUMNS)
    _write_readiness_dashboard_markdown(paths["readiness_dashboard_md"], readiness_dashboard)

    return ExpandedAuditArtifacts(
        paths=paths,
        summary=run_summaries,
        tickets=tickets,
        market_open_recheck_queue=market_queue,
        goal_completion=goal_audit,
    )


def summarize_run(run_dir: Path) -> dict[str, Any]:
    manifest = _read_manifest(run_dir)
    day = _run_date(run_dir, manifest)
    tickets = _safe_read_csv(run_dir / "trade_tickets.csv")
    queue = _safe_read_csv(run_dir / "market_open_recheck_queue.csv", MARKET_OPEN_RECHECK_COLUMNS)
    monthly = _safe_read_csv(run_dir / "monthly_feasibility.csv")
    row_counts = manifest.get("row_counts", {}) or {}
    execution = manifest.get("execution_readiness_summary", {}) or {}
    expectancy = manifest.get("expectancy_evidence_summary", {}) or {}
    feasibility = manifest.get("monthly_feasibility_summary", {}) or {}
    agentic = manifest.get("agentic_orchestration", {}) or {}
    context = manifest.get("execution_context", {}) or {}
    yellow_ticket_rows = row_counts.get("target_order_ticket_rows")
    if yellow_ticket_rows is None:
        yellow_ticket_rows = _target_count(tickets)
    if yellow_ticket_rows is None:
        yellow_ticket_rows = 0
    return {
        "date": day,
        "validation_lane": infer_validation_lane(run_dir),
        "mode": _mode_from_manifest(manifest),
        "pipeline_version": manifest.get("pipeline_version", ""),
        "decision_rows": row_counts.get("decision_board", 0),
        "trade_ticket_rows": row_counts.get("trade_tickets", len(tickets)),
        "green_ready_orders": row_counts.get("ready_to_enter", _truthy_count(tickets.get("ready_to_enter"))),
        "yellow_target_candidates": yellow_ticket_rows,
        "market_open_recheck_queue": row_counts.get("market_open_recheck_queue", len(queue)),
        "ready_one_cycle_max_profit": _monthly_metric(monthly, "one_cycle_max_profit"),
        "target_candidate_max_profit": _monthly_metric(monthly, "target_order_candidate_max_profit"),
        "execution_readiness": execution.get("status", ""),
        "execution_blockers": "; ".join(str(item) for item in execution.get("blocking_gates", []) or []),
        "monthly_feasibility": feasibility.get("status", ""),
        "monthly_blockers": "; ".join(str(item) for item in feasibility.get("blocking_metrics", []) or []),
        "expectancy_status": expectancy.get("status", ""),
        "expectancy_summary_status": expectancy.get("summary_status", ""),
        "expectancy_sample_size": expectancy.get("sample_size", 0),
        "expectancy_note": expectancy.get("note", ""),
        "agentic_status": agentic.get("status", ""),
        "subagent_task_count": agentic.get("subagent_task_count", row_counts.get("agent_dispatch_tasks", 0)),
        "external_review_count": context.get("external_review_count", row_counts.get("external_agent_reviews", 0)),
        "external_review_agent_count": context.get("external_review_agent_count", 0),
        "agentic_review_coverage_basis": context.get("agentic_review_coverage_basis", ""),
        "agentic_review_coverage_pct": context.get("agentic_review_coverage_pct", 0),
        "agentic_review_lane_coverage_pct": context.get("agentic_review_lane_coverage_pct", 0),
        "broad_review_coverage_pct": context.get("broad_review_coverage_pct", 0),
        "fresh_live_quotes_ready": bool(context.get("fresh_live_quotes_ready", False)),
        "portfolio_ready": bool(context.get("portfolio_ready", False)),
        "market_session_open": bool(context.get("market_session_open", False)),
        "green_symbols": _symbols(tickets[tickets.get("ready_to_enter", pd.Series(dtype=object)).map(_truthy)]),
        "target_symbols": _symbols(_target_rows(tickets)),
        "market_open_recheck_symbols": _symbols(queue),
        "all_ticket_symbols": _symbols(tickets),
        "source_dir": str(run_dir),
    }


def infer_validation_lane(run_dir: Path) -> str:
    name = run_dir.name
    if name.startswith("debit_scout"):
        return "expanded_debit_scout"
    if name.startswith("multidate_quality"):
        return "agentic_snapshot"
    if name.startswith("live_readiness_probe"):
        return "live_readiness_probe"
    return name


def combine_run_csvs(run_dirs: Sequence[Path], filename: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for run_dir in run_dirs:
        frame = _safe_read_csv(run_dir / filename)
        if frame.empty:
            continue
        frame.insert(0, "source_dir", str(run_dir))
        frame.insert(0, "validation_lane", infer_validation_lane(run_dir))
        frame.insert(0, "date", _run_date(run_dir, _read_manifest(run_dir)))
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def build_focus_coverage(coverage: pd.DataFrame, focus_tickers: Sequence[str]) -> pd.DataFrame:
    if coverage.empty or "ticker" not in coverage.columns:
        return coverage.copy()
    focus = {str(ticker).upper() for ticker in focus_tickers}
    frame = coverage[coverage["ticker"].astype(str).str.upper().isin(focus)].copy()
    return _normalize_audit_only_focus_coverage(frame).reset_index(drop=True)


def _normalize_audit_only_focus_coverage(coverage: pd.DataFrame) -> pd.DataFrame:
    if coverage.empty or "underlying_quality_tier" not in coverage.columns:
        return coverage
    frame = coverage.copy()
    for column in ("coverage_status", "status_color", "reason", "next_step"):
        if column not in frame.columns:
            frame[column] = ""
    tiers = frame["underlying_quality_tier"].astype(str).str.lower()
    statuses = frame["coverage_status"].astype(str).str.upper()
    audit_only = tiers.isin({"excluded", "speculative", "unknown"}) & statuses.isin(
        {
            "TARGET_ORDER_CANDIDATE",
            "REVIEW_TICKET",
            "STRUCTURED_NOT_TOP_FINAL",
            "CANDIDATE_NOT_STRUCTURED",
            "FINAL_NO_TICKET",
            "UNVALIDATED_CHAIN",
            "STRUCTURE_MISSING",
            "NO_DIRECTIONAL_EDGE",
            "BELOW_DISCOVERY_CUTOFF",
        }
    )
    if not audit_only.any():
        return frame
    frame.loc[audit_only, "coverage_status"] = "NON_ACTIONABLE_UNDERLYING"
    frame.loc[audit_only, "status_color"] = "red"
    frame.loc[audit_only, "next_step"] = "do not trade from the action list; require explicit override and fresh validation"
    reason = frame.loc[audit_only, "reason"].fillna("").astype(str)
    needs_prefix = ~reason.str.lower().str.startswith("not actionable:")
    indexed = reason[needs_prefix].index
    frame.loc[indexed, "reason"] = "not actionable: " + reason.loc[indexed]
    return frame


def build_ticket_review_lanes(tickets: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    if tickets.empty:
        return pd.DataFrame(columns=TICKET_REVIEW_LANE_COLUMNS)
    summary_by_source = {
        str(row.get("source_dir")): row
        for _, row in summary.iterrows()
    }
    rows: list[dict[str, Any]] = []
    for _, ticket in tickets.iterrows():
        run_summary = summary_by_source.get(str(ticket.get("source_dir")), {})
        rows.append(
            {
                "date": ticket.get("date", ""),
                "validation_lane": ticket.get("validation_lane", ""),
                "ticker": ticket.get("ticker", ""),
                "ready_to_enter": ticket.get("ready_to_enter", ""),
                "target_order_status": ticket.get("target_order_status", ""),
                "entry_type": ticket.get("entry_type", ""),
                "external_agent_review_count": ticket.get("external_agent_review_count", ""),
                "external_agent_distinct_review_count": ticket.get("external_agent_distinct_review_count", ""),
                "external_agent_review_agents": ticket.get("external_agent_review_agents", ""),
                "run_agentic_review_coverage_basis": _mapping_get(run_summary, "agentic_review_coverage_basis"),
                "run_agentic_review_coverage_pct": _mapping_get(run_summary, "agentic_review_coverage_pct"),
                "run_agentic_review_lane_coverage_pct": _mapping_get(run_summary, "agentic_review_lane_coverage_pct"),
                "run_broad_review_coverage_pct": _mapping_get(run_summary, "broad_review_coverage_pct"),
                "run_agentic_reviews_ready": _to_float(_mapping_get(run_summary, "agentic_review_coverage_pct")) >= 0.8,
            }
        )
    return pd.DataFrame(rows, columns=TICKET_REVIEW_LANE_COLUMNS)


def build_agentic_coverage_proof_packet(
    *,
    summary: pd.DataFrame,
    ticket_review_lanes: pd.DataFrame,
) -> pd.DataFrame:
    """Prove whether user-facing ticket rows came from fully agentic runs."""

    run_dates = sorted(summary.get("date", pd.Series(dtype=object)).dropna().astype(str).unique().tolist()) if not summary.empty else []
    agentic_dates = _agentic_dates(summary)
    ticket_dates = sorted(ticket_review_lanes.get("date", pd.Series(dtype=object)).dropna().astype(str).unique().tolist()) if not ticket_review_lanes.empty else []
    ticket_rows = len(ticket_review_lanes)
    ready_mask = (
        ticket_review_lanes.get("run_agentic_reviews_ready", pd.Series(dtype=object)).map(_truthy)
        if not ticket_review_lanes.empty
        else pd.Series(dtype=bool)
    )
    covered_rows = int(ready_mask.sum()) if not ticket_review_lanes.empty else 0
    uncovered_rows = int(ticket_rows - covered_rows)
    distinct_counts = (
        pd.to_numeric(ticket_review_lanes.get("external_agent_distinct_review_count", pd.Series(dtype=object)), errors="coerce").fillna(0)
        if not ticket_review_lanes.empty
        else pd.Series(dtype=float)
    )
    min_distinct = int(distinct_counts.min()) if len(distinct_counts) else 0
    below_min_mask = distinct_counts.lt(MIN_AGENTIC_REVIEW_LANES_PER_TICKER) if len(distinct_counts) else pd.Series(dtype=bool)
    below_min_rows = int(below_min_mask.sum()) if len(below_min_mask) else 0
    if not ticket_review_lanes.empty:
        non_agentic_ticket_dates = sorted(
            ticket_review_lanes.loc[~ready_mask, "date"].dropna().astype(str).unique().tolist()
        )
        below_min_ticket_lane_dates = sorted(
            ticket_review_lanes.loc[below_min_mask, "date"].dropna().astype(str).unique().tolist()
        )
    else:
        non_agentic_ticket_dates = []
        below_min_ticket_lane_dates = []
    coverage_pct = round(covered_rows / ticket_rows, 4) if ticket_rows > 0 else 0.0

    if ticket_rows <= 0:
        status = "NO_TICKET_ROWS"
    elif uncovered_rows == 0 and below_min_rows == 0 and covered_rows > 0:
        status = "PASS_FULL_AGENTIC_TICKET_COVERAGE"
    elif covered_rows > 0:
        status = "PARTIAL_AGENTIC_TICKET_COVERAGE"
    else:
        status = "MISSING_AGENTIC_TICKET_COVERAGE"

    claim = (
        "Every user-facing ticket row in this audit came from a run with required agentic lane coverage."
        if status == "PASS_FULL_AGENTIC_TICKET_COVERAGE"
        else "User-facing ticket rows are not fully covered by agentic lane review in this audit."
    )
    note = (
        "Snapshot replay can validate structure and target math, but it must not be counted as fully agentic "
        "unless the run-level subagent lane coverage gate passed for the ticket rows."
    )
    return pd.DataFrame(
        [
            {
                "status": status,
                "run_date_count": len(run_dates),
                "ticket_rows": ticket_rows,
                "agentic_ready_date_count": len(agentic_dates),
                "agentic_ready_dates": ", ".join(agentic_dates),
                "ticket_date_count": len(ticket_dates),
                "ticket_dates": ", ".join(ticket_dates),
                "ticket_rows_with_agentic_ready": covered_rows,
                "ticket_rows_without_agentic_ready": uncovered_rows,
                "ticket_agentic_coverage_pct": coverage_pct,
                "required_min_ticket_lanes": MIN_AGENTIC_REVIEW_LANES_PER_TICKER,
                "min_ticket_distinct_review_count": min_distinct,
                "ticket_rows_below_min_ticket_lanes": below_min_rows,
                "below_min_ticket_lane_dates": ", ".join(below_min_ticket_lane_dates),
                "non_agentic_ticket_date_count": len(non_agentic_ticket_dates),
                "non_agentic_ticket_dates": ", ".join(non_agentic_ticket_dates),
                "required_coverage": (
                    "100% of user-facing ticket rows from agentic-ready runs and each ticket at or above the "
                    "minimum distinct review-lane count"
                ),
                "claim": claim,
                "note": note,
            }
        ],
        columns=AGENTIC_COVERAGE_PROOF_COLUMNS,
    )


def build_validation_coverage_proof_packet(*, base_dir: Path, summary: pd.DataFrame) -> pd.DataFrame:
    """Show exactly which available dated UW folders are covered by this audit window."""

    tested_dates = sorted(
        {
            str(value)
            for value in summary.get("date", pd.Series(dtype=object)).dropna().tolist()
            if str(value).strip()
        }
    )
    available_dates = discover_available_source_dates(base_dir)
    if tested_dates:
        start = tested_dates[0]
        end = tested_dates[-1]
        window_available = [day for day in available_dates if start <= day <= end]
        untested = sorted(set(window_available) - set(tested_dates))
        tested_available = sorted(set(tested_dates) & set(available_dates))
        outside = [day for day in available_dates if day < start or day > end]
    else:
        start = ""
        end = ""
        window_available = []
        untested = []
        tested_available = []
        outside = available_dates

    if len(tested_dates) < 2:
        status = "NOT_ENOUGH_DATES"
    elif window_available and not untested:
        status = "PROVEN_WINDOW_COVERED"
    elif window_available:
        status = "PARTIAL_WINDOW_GAPS"
    else:
        status = "PROVEN_RUN_DATES_NO_SOURCE_INDEX"

    note = (
        "This packet proves the selected validation window only. Dates outside the window are listed so a broader "
        "history test can be scheduled explicitly instead of implying full-history coverage."
    )
    return pd.DataFrame(
        [
            {
                "status": status,
                "base_dir": str(base_dir),
                "validation_start": start,
                "validation_end": end,
                "base_available_source_date_count": len(available_dates),
                "window_available_source_date_count": len(window_available),
                "tested_date_count": len(tested_dates),
                "tested_available_date_count": len(tested_available),
                "untested_available_date_count": len(untested),
                "available_dates_outside_window_count": len(outside),
                "tested_dates": ", ".join(tested_dates),
                "untested_available_dates": ", ".join(untested),
                "available_dates_outside_window": ", ".join(outside),
                "note": note,
            }
        ],
        columns=VALIDATION_COVERAGE_PROOF_COLUMNS,
    )


def build_cutoff_visibility_proof_packet(run_dirs: Sequence[Path]) -> pd.DataFrame:
    """Prove that top-trade compatibility knobs did not hide candidate or audit rows."""

    totals = {
        "candidate_rows": 0,
        "research_task_rows": 0,
        "qualified_candidate_rows": 0,
        "priced_candidate_rows": 0,
        "final_rows": 0,
        "expected_no_trade_rows": 0,
        "no_trade_audit_rows": 0,
    }
    candidate_research_mismatch: list[str] = []
    priced_missing_qualified: list[str] = []
    no_trade_missing_expected: list[str] = []
    problem_runs: list[str] = []

    for run_dir in run_dirs:
        run_name = run_dir.name
        manifest = _read_manifest(run_dir)
        row_counts = manifest.get("row_counts", {}) or {}
        candidates = _safe_read_csv(run_dir / "candidate_generation.csv")
        priced = _safe_read_csv(run_dir / "priced_candidates.csv")
        final = _safe_read_csv(run_dir / "final_recommendations.csv")
        no_trade = _safe_read_csv(run_dir / "no_trade_audit.csv")

        candidate_rows = len(candidates)
        research_rows = _research_task_count(run_dir, row_counts)
        qualified_rows = _qualified_candidate_count(candidates)
        priced_rows = len(priced)
        final_rows = len(final)
        expected_no_trade_rows = _expected_no_trade_count(candidates, priced)
        no_trade_rows = len(no_trade)

        totals["candidate_rows"] += candidate_rows
        totals["research_task_rows"] += research_rows
        totals["qualified_candidate_rows"] += qualified_rows
        totals["priced_candidate_rows"] += priced_rows
        totals["final_rows"] += final_rows
        totals["expected_no_trade_rows"] += expected_no_trade_rows
        totals["no_trade_audit_rows"] += no_trade_rows

        run_problems: list[str] = []
        if candidate_rows and research_rows != candidate_rows:
            candidate_research_mismatch.append(run_name)
            run_problems.append(f"research_tasks={research_rows} vs candidates={candidate_rows}")
        if qualified_rows and priced_rows < qualified_rows:
            priced_missing_qualified.append(run_name)
            run_problems.append(f"priced={priced_rows} vs qualified={qualified_rows}")
        if expected_no_trade_rows and no_trade_rows < expected_no_trade_rows:
            no_trade_missing_expected.append(run_name)
            run_problems.append(f"no_trade={no_trade_rows} vs expected={expected_no_trade_rows}")
        if run_problems:
            problem_runs.append(f"{run_name} ({'; '.join(run_problems)})")

    if not run_dirs:
        status = "MISSING_RUNS"
    elif candidate_research_mismatch or priced_missing_qualified or no_trade_missing_expected:
        status = "FAIL_ARTIFICIAL_CUTOFF_OR_STALE_AUDIT_ROWS"
    else:
        status = "PASS_NO_ARTIFICIAL_CUTOFFS"

    claim = (
        "Candidate generation, research dispatch, pricing, and no-trade audit visibility are not capped by top-trades."
        if status == "PASS_NO_ARTIFICIAL_CUTOFFS"
        else "One or more run artifacts still look capped, stale, or incomplete."
    )
    note = (
        "Research tasks should match candidate rows; priced rows should cover every qualified candidate row; "
        "no_trade_audit.csv should cover every non-qualified or unpriced candidate row."
    )
    return pd.DataFrame(
        [
            {
                "status": status,
                "run_count": len(run_dirs),
                **totals,
                "candidate_research_mismatch_runs": ", ".join(candidate_research_mismatch),
                "priced_missing_qualified_runs": ", ".join(priced_missing_qualified),
                "no_trade_missing_expected_runs": ", ".join(no_trade_missing_expected),
                "problem_runs": " | ".join(problem_runs),
                "claim": claim,
                "note": note,
            }
        ],
        columns=CUTOFF_VISIBILITY_PROOF_COLUMNS,
    )


def discover_available_source_dates(base_dir: Path) -> list[str]:
    """Return dated folders that actually contain UW source exports."""

    if not base_dir.exists():
        return []
    dates: list[str] = []
    for child in base_dir.iterdir():
        if not child.is_dir() or not re.fullmatch(r"20\d{2}-\d{2}-\d{2}", child.name):
            continue
        if _has_uw_source_files(child, child.name):
            dates.append(child.name)
    return sorted(dates)


def _has_uw_source_files(day_dir: Path, day: str) -> bool:
    required_exports = (
        ("stock-screener", "ticker"),
        ("hot-chains", "option_symbol"),
        ("chain-oi-changes", "option_symbol"),
    )
    return all(_export_with_required_header_exists(day_dir, day, prefix, required_header) for prefix, required_header in required_exports)


def _export_with_required_header_exists(day_dir: Path, day: str, prefix: str, required_header: str) -> bool:
    paths = sorted(day_dir.glob(f"{prefix}-{day}*.csv")) + sorted(day_dir.glob(f"{prefix}-{day}*.zip"))
    for path in paths:
        try:
            headers = _read_export_headers(path)
        except Exception:
            continue
        if required_header in {header.strip() for header in headers}:
            return True
    return False


def _read_export_headers(path: Path) -> list[str]:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            members = [name for name in zf.namelist() if name.lower().endswith(".csv")]
            if not members:
                return []
            with zf.open(members[0]) as handle:
                text = io.TextIOWrapper(handle, encoding="utf-8-sig", newline="")
                return next(csv.reader(text), [])
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return next(csv.reader(handle), [])


def build_live_spread_quality_proof_packet(audit: pd.DataFrame) -> pd.DataFrame:
    """Prove bad live/snapshot markets are blocked before the actionable ticket surface."""

    if audit.empty:
        row = {
            "status": "NO_LIVE_SPREAD_QUALITY_AUDIT",
            "audited_rows": 0,
            "pass_rows": 0,
            "block_rows": 0,
            "quote_width_block_rows": 0,
            "liquidity_block_rows": 0,
            "blocked_not_target_candidate_rows": 0,
            "blocked_still_actionable_rows": 0,
            "target_candidate_rows": 0,
            "target_candidate_block_rows": 0,
            "blocked_tickers": "",
            "blocked_examples": "",
            "required_gate": (
                f"live_quote_width_pct <= {MAX_LIVE_QUOTE_WIDTH_PCT:.2f}; "
                f"live_leg_min_liquidity >= {MIN_LIVE_LEG_LIQUIDITY:.0f}; "
                "blocked rows must not remain eligible for yellow or green action surfaces"
            ),
            "claim": "No live-spread-quality audit rows were available for this expanded audit.",
            "note": "Regenerate with run directories that include live_spread_quality_audit.csv.",
        }
        return pd.DataFrame([row], columns=LIVE_SPREAD_QUALITY_PROOF_COLUMNS)

    status = audit.get("live_market_quality_status", pd.Series("", index=audit.index)).fillna("").astype(str).str.upper()
    impact = audit.get("actionability_impact", pd.Series("", index=audit.index)).fillna("").astype(str).str.lower()
    reasons = audit.get("quality_gate_reason", pd.Series("", index=audit.index)).fillna("").astype(str)
    quote_width = pd.to_numeric(audit.get("live_quote_width_pct", pd.Series(dtype=object)), errors="coerce")
    liquidity = pd.to_numeric(audit.get("live_leg_min_liquidity", pd.Series(dtype=object)), errors="coerce")
    liquidity_status = audit.get("live_leg_liquidity_status", pd.Series("", index=audit.index)).fillna("").astype(str).str.upper()

    block_mask = status.eq("BLOCK")
    pass_mask = status.eq("PASS")
    quote_block_mask = (
        reasons.str.contains("live_quote_width_pct_", regex=False, na=False)
        | quote_width.gt(MAX_LIVE_QUOTE_WIDTH_PCT)
    )
    liquidity_block_mask = (
        reasons.str.contains("live_leg_liquidity_", regex=False, na=False)
        | liquidity.lt(MIN_LIVE_LEG_LIQUIDITY)
        | liquidity_status.isin({"BLOCK", "MISSING"})
    )
    not_actionable_impacts = {
        "blocked_live_market_visible_for_review",
        "blocked_not_target_candidate",
        "market_closed_target_recheck",
        "visible_for_review",
    }
    not_actionable_mask = impact.isin(not_actionable_impacts)
    target_candidate_mask = impact.isin(
        {
            "eligible_for_yellow_or_green_surface",
            "target_order_candidate",
            "target_order_wait_for_price",
        }
    )
    blocked_still_actionable_mask = block_mask & target_candidate_mask
    target_candidate_block_mask = block_mask & target_candidate_mask
    block_rows = int(block_mask.sum())
    pass_rows = int(pass_mask.sum())
    blocked_still_actionable_rows = int(blocked_still_actionable_mask.sum())

    if blocked_still_actionable_rows > 0:
        proof_status = "FAIL_BLOCKED_LIVE_MARKETS_STILL_ACTIONABLE"
    else:
        proof_status = "PASS_LIVE_SPREAD_QUALITY_GATED"

    claim = (
        "Bad live/snapshot spread markets were blocked before they reached yellow or green action surfaces."
        if proof_status == "PASS_LIVE_SPREAD_QUALITY_GATED"
        else "One or more blocked live/snapshot markets still looked actionable and must be removed from the ticket surface."
    )
    note = (
        "This proves quote-width and leg-liquidity gating only. It is not fresh execution evidence unless the source "
        "directory is a regular-market live Schwab run."
    )
    return pd.DataFrame(
        [
            {
                "status": proof_status,
                "audited_rows": int(len(audit)),
                "pass_rows": pass_rows,
                "block_rows": block_rows,
                "quote_width_block_rows": int((block_mask & quote_block_mask).sum()),
                "liquidity_block_rows": int((block_mask & liquidity_block_mask).sum()),
                "blocked_not_target_candidate_rows": int((block_mask & not_actionable_mask).sum()),
                "blocked_still_actionable_rows": blocked_still_actionable_rows,
                "target_candidate_rows": int(target_candidate_mask.sum()),
                "target_candidate_block_rows": int(target_candidate_block_mask.sum()),
                "blocked_tickers": _ticker_list(audit.loc[block_mask] if not audit.empty else audit),
                "blocked_examples": _live_spread_quality_examples(audit.loc[block_mask] if not audit.empty else audit),
                "required_gate": (
                    f"live_quote_width_pct <= {MAX_LIVE_QUOTE_WIDTH_PCT:.2f}; "
                    f"live_leg_min_liquidity >= {MIN_LIVE_LEG_LIQUIDITY:.0f}; "
                    "blocked rows must not remain eligible for yellow or green action surfaces"
                ),
                "claim": claim,
                "note": note,
            }
        ],
        columns=LIVE_SPREAD_QUALITY_PROOF_COLUMNS,
    )


def build_underlying_quality_proof_packet(
    *,
    tickets: pd.DataFrame,
    focus_coverage: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize whether non-actionable underlyings reached user-facing trade tickets."""

    ticket_tiers = _quality_tiers(tickets)
    focus_tiers = _quality_tiers(focus_coverage)
    core_rows = int(ticket_tiers.eq("core").sum())
    liquid_non_core_rows = int(ticket_tiers.eq("liquid").sum())
    speculative_rows = int(ticket_tiers.eq("speculative").sum())
    excluded_rows = int(ticket_tiers.eq("excluded").sum())
    unknown_rows = int(ticket_tiers.isin(["", "unknown"]).sum())
    bad_mask = ~ticket_tiers.isin(["core", "liquid"])
    bad_ticket_rows = int(bad_mask.sum())
    bad_tickets = tickets.loc[bad_mask] if not tickets.empty else tickets
    green_bad_rows = 0
    target_bad_rows = 0
    if not bad_tickets.empty:
        green_bad_rows = _truthy_count(bad_tickets.get("ready_to_enter"))
        target_bad_rows = _target_count(bad_tickets)

    if len(tickets) <= 0:
        status = "MISSING_TICKETS"
    elif bad_ticket_rows > 0:
        status = "FAIL_NON_ACTIONABLE_TICKETS_PRESENT"
    else:
        status = "PASS_CORE_OR_LIQUID_TICKETS"

    claim = (
        "Only core or liquid underlyings reached user-facing trade tickets."
        if bad_ticket_rows == 0 and len(tickets) > 0
        else "Non-core or non-actionable underlyings reached the user-facing ticket surface and must be fixed."
    )
    note = "Speculative, excluded, unknown, or missing-tier underlyings remain audit-only."
    return pd.DataFrame(
        [
            {
                "status": status,
                "ticket_rows": len(tickets),
                "core_ticket_rows": core_rows,
                "liquid_non_core_ticket_rows": liquid_non_core_rows,
                "speculative_ticket_rows": speculative_rows,
                "excluded_ticket_rows": excluded_rows,
                "unknown_ticket_rows": unknown_rows,
                "not_core_or_liquid_ticket_rows": bad_ticket_rows,
                "green_not_core_or_liquid_rows": green_bad_rows,
                "target_not_core_or_liquid_rows": target_bad_rows,
                "ticket_tickers": _ticker_list(tickets),
                "liquid_non_core_ticket_tickers": _ticker_list(tickets.loc[ticket_tiers.eq("liquid")] if not tickets.empty else tickets),
                "not_core_or_liquid_ticket_tickers": _ticker_list(bad_tickets),
                "focus_rows": len(focus_coverage),
                "focus_speculative_rows": int(focus_tiers.eq("speculative").sum()),
                "focus_excluded_rows": int(focus_tiers.eq("excluded").sum()),
                "focus_liquid_non_core_rows": int(focus_tiers.eq("liquid").sum()),
                "focus_speculative_examples": _coverage_examples(focus_coverage.loc[focus_tiers.eq("speculative")] if not focus_coverage.empty else focus_coverage),
                "focus_excluded_examples": _coverage_examples(focus_coverage.loc[focus_tiers.eq("excluded")] if not focus_coverage.empty else focus_coverage),
                "focus_liquid_non_core_examples": _coverage_examples(focus_coverage.loc[focus_tiers.eq("liquid")] if not focus_coverage.empty else focus_coverage),
                "claim": claim,
                "note": note,
            }
        ],
        columns=UNDERLYING_QUALITY_PROOF_COLUMNS,
    )


def build_major_name_coverage_proof_packet(
    *,
    focus_coverage: pd.DataFrame,
    focus_tickers: Sequence[str] = CORE_AUDIT_TICKERS,
) -> pd.DataFrame:
    """Prove that each focus ticker has an explicit pipeline state and reason."""

    required = _clean_ticker_sequence(focus_tickers)
    if focus_coverage.empty or "ticker" not in focus_coverage.columns:
        covered: set[str] = set()
        required_rows = focus_coverage.copy()
    else:
        required_set = set(required)
        required_rows = focus_coverage[
            focus_coverage["ticker"].astype(str).str.upper().isin(required_set)
        ].copy()
        covered = {
            str(value).strip().upper()
            for value in required_rows["ticker"].dropna().tolist()
            if str(value).strip()
        }
    missing = [ticker for ticker in required if ticker not in covered]
    reason_series = required_rows.get("reason", pd.Series(dtype=object)).fillna("").astype(str).str.strip()
    rows_with_reason = int(reason_series.ne("").sum())
    rows_missing_reason = int(reason_series.eq("").sum())

    if not required:
        status = "NO_REQUIRED_MAJOR_NAMES"
    elif not required_rows.empty and not missing and rows_missing_reason == 0:
        status = "PASS_ALL_MAJOR_NAMES_EXPLAINED"
    elif not required_rows.empty:
        status = "PARTIAL_MAJOR_NAME_EXPLANATIONS"
    else:
        status = "MISSING_MAJOR_NAME_COVERAGE"

    claim = (
        "Every configured major/focus ticker has at least one explicit pipeline state and reason."
        if status == "PASS_ALL_MAJOR_NAMES_EXPLAINED"
        else "One or more configured major/focus tickers are missing a visible pipeline state or reason."
    )
    note = (
        "This proof explains inclusion/exclusion. It does not force a trade in AAPL/NVDA/MSFT/GOOG/GOOGL/PLTR or any "
        "other major name when the source flow, structure, pricing, or execution gates do not support one."
    )
    return pd.DataFrame(
        [
            {
                "status": status,
                "required_ticker_count": len(required),
                "covered_required_ticker_count": len(covered),
                "missing_required_ticker_count": len(missing),
                "required_focus_rows": len(required_rows),
                "required_rows_with_reason": rows_with_reason,
                "required_rows_missing_reason": rows_missing_reason,
                "ready_ticket_tickers": _coverage_tickers_by_status(required_rows, {"READY_TICKET"}),
                "yellow_target_tickers": _coverage_tickers_by_status(required_rows, {"TARGET_ORDER_CANDIDATE"}),
                "review_ticket_tickers": _coverage_tickers_by_status(required_rows, {"REVIEW_TICKET", "FINAL_NO_TICKET"}),
                "structured_not_final_tickers": _coverage_tickers_by_status(required_rows, {"STRUCTURED_NOT_TOP_FINAL"}),
                "candidate_not_structured_tickers": _coverage_tickers_by_status(required_rows, {"CANDIDATE_NOT_STRUCTURED", "STRUCTURE_MISSING", "UNVALIDATED_CHAIN"}),
                "no_directional_edge_tickers": _coverage_tickers_by_status(required_rows, {"NO_DIRECTIONAL_EDGE", "BELOW_DISCOVERY_CUTOFF"}),
                "source_missing_tickers": _coverage_tickers_by_status(required_rows, {"SOURCE_MISSING"}),
                "blocked_or_excluded_tickers": _coverage_tickers_by_status(required_rows, {"BLOCKED_FINAL_ROW", "NON_ACTIONABLE_UNDERLYING"}),
                "missing_required_tickers": ", ".join(missing),
                "required_tickers": ", ".join(required),
                "examples": _major_name_examples(required_rows, required),
                "claim": claim,
                "note": note,
            }
        ],
        columns=MAJOR_NAME_COVERAGE_PROOF_COLUMNS,
    )


def _quality_tiers(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=object)
    return frame.get("underlying_quality_tier", pd.Series("", index=frame.index)).fillna("").astype(str).str.lower().str.strip()


def _clean_ticker_sequence(tickers: Sequence[str]) -> list[str]:
    cleaned: list[str] = []
    for value in tickers:
        ticker = str(value).strip().upper()
        if ticker and ticker not in cleaned:
            cleaned.append(ticker)
    return cleaned


def _coverage_tickers_by_status(frame: pd.DataFrame, statuses: set[str], *, limit: int = 80) -> str:
    if frame.empty or not {"ticker", "coverage_status"}.issubset(frame.columns):
        return ""
    status = frame["coverage_status"].astype(str).str.upper()
    tickers = sorted(
        {
            str(value).strip().upper()
            for value in frame.loc[status.isin(statuses), "ticker"].dropna().tolist()
            if str(value).strip()
        }
    )
    if len(tickers) > limit:
        return ", ".join(tickers[:limit]) + f", ... (+{len(tickers) - limit} more)"
    return ", ".join(tickers)


def _major_name_examples(frame: pd.DataFrame, required: Sequence[str], *, limit: int = 25) -> str:
    if frame.empty or "ticker" not in frame.columns:
        return ""
    examples: list[str] = []
    ticker_series = frame["ticker"].astype(str).str.upper()
    for ticker in required:
        ticker_rows = frame[ticker_series.eq(ticker)]
        if ticker_rows.empty:
            continue
        statuses = (
            ticker_rows.get("coverage_status", pd.Series("", index=ticker_rows.index))
            .astype(str)
            .value_counts()
            .sort_index()
        )
        status_text = ", ".join(f"{status}={int(count)}" for status, count in statuses.items())
        reason = ""
        if "reason" in ticker_rows.columns:
            reasons = ticker_rows["reason"].fillna("").astype(str).str.strip()
            nonempty = reasons[reasons.ne("")]
            if not nonempty.empty:
                reason = nonempty.iloc[0]
        examples.append(f"{ticker}: {status_text}; example={reason[:180]}")
        if len(examples) >= limit:
            break
    remaining = max(0, len(required) - len(examples))
    if remaining > 0:
        examples.append(f"... (+{remaining} more)")
    return "; ".join(examples)


def _ticker_list(frame: pd.DataFrame, *, limit: int = 80) -> str:
    if frame.empty or "ticker" not in frame.columns:
        return ""
    tickers = sorted({str(value).upper() for value in frame["ticker"].dropna().tolist() if str(value).strip()})
    if len(tickers) > limit:
        return ", ".join(tickers[:limit]) + f", ... (+{len(tickers) - limit} more)"
    return ", ".join(tickers)


def _bad_actionability_examples(target: pd.DataFrame, *, limit: int = 10) -> str:
    if target.empty:
        return ""
    examples: list[str] = []
    bad_total = 0
    for _, row in target.iterrows():
        reasons: list[str] = []
        if _truthy(_mapping_get(row, "ready_to_enter")):
            reasons.append("target_row_ready_to_enter_true")
        if str(_mapping_get(row, "entry_type") or "").strip().upper() not in {"CREDIT", "DEBIT"}:
            reasons.append("entry_type_missing")
        if _to_float(_mapping_get(row, "entry_limit")) <= 0:
            reasons.append("entry_limit_not_positive")
        if not str(_mapping_get(row, "trade_plan") or "").strip():
            reasons.append("trade_plan_missing")
        if not _has_plain_language_trade_legs(row):
            reasons.append("plain_language_buy_sell_legs_missing")
        if reasons:
            bad_total += 1
            if len(examples) < limit:
                examples.append(f"{row.get('ticker', '')}: {', '.join(reasons)}")
    remaining = max(0, bad_total - len(examples))
    if examples and remaining > 0:
        examples.append(f"... (+{remaining} more target rows not shown)")
    return "; ".join(examples)


def _coverage_examples(frame: pd.DataFrame, *, limit: int = 12) -> str:
    if frame.empty or "ticker" not in frame.columns:
        return ""
    examples: list[str] = []
    for ticker in sorted({str(value).upper() for value in frame["ticker"].dropna().tolist() if str(value).strip()}):
        ticker_rows = frame[frame["ticker"].astype(str).str.upper().eq(ticker)]
        if "coverage_status" in ticker_rows.columns:
            counts = ticker_rows["coverage_status"].astype(str).value_counts().sort_index()
            status_text = ", ".join(f"{status}={int(count)}" for status, count in counts.items())
        else:
            status_text = f"rows={len(ticker_rows)}"
        examples.append(f"{ticker}: {status_text}")
        if len(examples) >= limit:
            break
    remaining = len({str(value).upper() for value in frame["ticker"].dropna().tolist() if str(value).strip()}) - len(examples)
    if remaining > 0:
        examples.append(f"... (+{remaining} more)")
    return "; ".join(examples)


def _live_spread_quality_examples(frame: pd.DataFrame, *, limit: int = 12) -> str:
    if frame.empty or "ticker" not in frame.columns:
        return ""
    examples: list[str] = []
    for _, row in frame.head(limit).iterrows():
        ticker = _clean_display_value(row.get("ticker", "")).upper()
        reason = _clean_display_value(row.get("quality_gate_reason", ""))
        impact = _clean_display_value(row.get("actionability_impact", ""))
        quote_width = _clean_display_value(row.get("live_quote_width_pct", ""))
        liquidity = _clean_display_value(row.get("live_leg_min_liquidity", ""))
        parts = [ticker]
        if reason:
            parts.append(reason)
        if impact:
            parts.append(f"impact={impact}")
        if str(quote_width).strip():
            parts.append(f"quote_width={quote_width}")
        if str(liquidity).strip():
            parts.append(f"min_liquidity={liquidity}")
        examples.append(": ".join([parts[0], "; ".join(parts[1:])]) if len(parts) > 1 else parts[0])
    remaining = len(frame) - len(examples)
    if remaining > 0:
        examples.append(f"... (+{remaining} more)")
    return "; ".join(examples)


def combine_expectancy_audit(run_dirs: Sequence[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for run_dir in run_dirs:
        frame = _safe_read_csv(run_dir / "expectancy_evidence.csv")
        if frame.empty:
            continue
        frame.insert(0, "validation_lane", infer_validation_lane(run_dir))
        frame.insert(0, "date", _run_date(run_dir, _read_manifest(run_dir)))
        frame["source_dir"] = str(run_dir)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=EXPECTANCY_AUDIT_COLUMNS)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    for column in EXPECTANCY_AUDIT_COLUMNS:
        if column not in combined.columns:
            combined[column] = ""
    return combined[EXPECTANCY_AUDIT_COLUMNS]


def build_expectancy_proof_packet(
    *,
    summary: pd.DataFrame,
    tickets: pd.DataFrame,
    expectancy: pd.DataFrame,
    live_probe_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize whether aggregate expectancy can support monthly-profit claims."""

    dates = sorted(summary["date"].astype(str).unique().tolist()) if not summary.empty else []
    ticket_tickers = sorted(set(tickets.get("ticker", pd.Series(dtype=object)).astype(str).str.upper()))
    live_green = int(live_probe_summary.get("green_ready_orders", pd.Series(dtype=object)).map(_to_float).sum()) if not live_probe_summary.empty else 0
    dated_green = int(pd.to_numeric(summary.get("green_ready_orders", pd.Series(dtype=object)), errors="coerce").fillna(0).sum()) if not summary.empty else 0
    current_green = live_green + dated_green
    summary_statuses = sorted(set(_expectancy_source(expectancy, "expectancy_summary").get("status", pd.Series(dtype=object)).astype(str)))
    pass_present = "PASS" in summary_statuses
    block_present = "BLOCK" in summary_statuses or _has_blocking_expectancy_source(expectancy)
    monthly_claim_allowed = bool(current_green > 0 and pass_present and not block_present)
    if monthly_claim_allowed:
        status = "positive_expectancy_ready_for_monthly_claim_review"
    elif current_green <= 0 and not pass_present:
        status = "blocked_no_green_orders_and_no_positive_monthly_expectancy"
    elif current_green <= 0:
        status = "blocked_no_green_orders"
    else:
        status = "blocked_no_positive_overall_strategy_expectancy"

    row = {
        "status": status,
        "date_count": len(dates),
        "ticket_ticker_count": len(ticket_tickers),
        "ticket_tickers": ", ".join(ticket_tickers),
        "current_green_ready_orders": current_green,
        "expectancy_summary_statuses": ", ".join(summary_statuses),
        "blocking_source_counts": _status_counts_by_source(expectancy, "BLOCK"),
        "forward_realized_statuses": _statuses_for_sources(
            expectancy,
            ["codexuw_execute_outcome_ledger", "codexuw_recommendation_outcome_ledger"],
        ),
        "actual_closed_trade_statuses": _statuses_for_sources(expectancy, ["schwab_closed_trades"]),
        "replay_statuses": _statuses_for_source_prefix(expectancy, "codexuw_replay"),
        "minimum_sample_size": MIN_EXPECTANCY_SAMPLE_SIZE,
        "minimum_win_rate": MIN_EXPECTANCY_WIN_RATE,
        "minimum_profit_factor": MIN_EXPECTANCY_PROFIT_FACTOR,
        "monthly_profit_target": MONTHLY_PROFIT_TARGET,
        "monthly_claim_allowed": monthly_claim_allowed,
        "required_evidence": (
            "green ready_to_enter tickets plus positive overall/strategy expectancy with sufficient sample size, "
            "win rate, and profit factor; green-ticket ticker support is proven separately by the ticket expectancy packet"
        ),
        "note": "This packet is a claim gate, not a forecast. It prevents $10k/month readiness claims until monthly-capacity and overall/strategy expectancy evidence are positive.",
    }
    return pd.DataFrame([row], columns=EXPECTANCY_PROOF_PACKET_COLUMNS)


def build_ticket_expectancy_coverage(
    *,
    tickets: pd.DataFrame,
    green_ticket_execution_details: pd.DataFrame,
    expectancy: pd.DataFrame,
) -> pd.DataFrame:
    """Match visible ticket tickers to actual/forward and replay expectancy evidence."""

    ticket_counts = _ticker_counts(tickets)
    green_counts = _ticker_counts(green_ticket_execution_details)
    tickers = sorted(set(ticket_counts) | set(green_counts))
    rows: list[dict[str, Any]] = []
    for ticker in tickers:
        matched = _expectancy_rows_for_ticker(expectancy, ticker)
        actual_forward = matched[
            matched.get("evidence_type", pd.Series(dtype=object))
            .astype(str)
            .isin(GREEN_TICKET_EXPECTANCY_EVIDENCE_TYPES)
        ] if not matched.empty else pd.DataFrame()
        replay = matched[
            matched.get("evidence_type", pd.Series(dtype=object))
            .astype(str)
            .str.startswith("replay_backtest")
        ] if not matched.empty else pd.DataFrame()
        actual_forward_pass = _source_names(actual_forward, "PASS")
        actual_forward_block = _source_names(actual_forward, "BLOCK")
        replay_pass = _source_names(replay, "PASS")
        replay_block = _source_names(replay, "BLOCK")
        actual_sample = int(pd.to_numeric(actual_forward.get("sample_size", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        replay_sample = int(pd.to_numeric(replay.get("sample_size", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        if actual_forward_pass:
            status = "PASS_ACTUAL_FORWARD_EXPECTANCY"
            note = "Ticket ticker has positive structure-aligned actual/forward outcome support."
        elif replay_pass:
            status = "WARN_REPLAY_ONLY_EXPECTANCY"
            note = "Replay support exists, but structure-aligned actual/forward outcome support is missing or blocked."
        else:
            status = "BLOCK_NO_POSITIVE_TICKET_EXPECTANCY"
            note = "No positive structure-aligned actual/forward expectancy support found."
        rows.append(
            {
                "ticker": ticker,
                "ticket_rows": int(ticket_counts.get(ticker, 0)),
                "green_ticket_rows": int(green_counts.get(ticker, 0)),
                "actual_forward_pass_sources": ", ".join(actual_forward_pass),
                "actual_forward_block_sources": ", ".join(actual_forward_block),
                "actual_forward_sample_size": actual_sample,
                "replay_pass_sources": ", ".join(replay_pass),
                "replay_block_sources": ", ".join(replay_block),
                "replay_sample_size": replay_sample,
                "status": status,
                "note": note,
            }
        )
    return pd.DataFrame(rows, columns=TICKET_EXPECTANCY_COVERAGE_COLUMNS)


def build_ticket_expectancy_proof_packet(*, coverage: pd.DataFrame) -> pd.DataFrame:
    """Summarize whether current green ticket tickers have actual/forward expectancy support."""

    if coverage.empty:
        status = "BLOCK_NO_TICKET_TICKERS"
        green_tickers: list[str] = []
        ticket_tickers: list[str] = []
        positive_actual: list[str] = []
        missing_green: list[str] = []
        replay_only: list[str] = []
    else:
        ticket_tickers = sorted(coverage["ticker"].astype(str).str.upper().unique().tolist())
        green = coverage[pd.to_numeric(coverage["green_ticket_rows"], errors="coerce").fillna(0).gt(0)]
        green_tickers = sorted(green["ticker"].astype(str).str.upper().unique().tolist())
        positive = coverage[coverage["status"].astype(str).eq("PASS_ACTUAL_FORWARD_EXPECTANCY")]
        positive_actual = sorted(positive["ticker"].astype(str).str.upper().unique().tolist())
        missing_green = sorted(
            green.loc[
                ~green["ticker"].astype(str).str.upper().isin(set(positive_actual)),
                "ticker",
            ].astype(str).str.upper().unique().tolist()
        )
        replay_only = sorted(
            coverage.loc[
                coverage["status"].astype(str).eq("WARN_REPLAY_ONLY_EXPECTANCY"),
                "ticker",
            ].astype(str).str.upper().unique().tolist()
        )
        if not green_tickers:
            status = "BLOCK_NO_GREEN_TICKERS_FOR_EXPECTANCY_CLAIM"
        elif missing_green:
            status = "BLOCK_GREEN_TICKERS_WITHOUT_ACTUAL_FORWARD_EXPECTANCY"
        else:
            status = "PASS_GREEN_TICKER_EXPECTANCY_COVERAGE"
    required = (
        "Every current green ticket ticker must have structure-aligned actual/forward outcome evidence that passes "
        "the minimum sample, win-rate, average-P/L, and profit-factor gates. Replay-only or unrelated-strategy "
        "ticker support is not enough."
    )
    note = (
        "This packet prevents broad replay, unrelated closed trades, or opposite-strategy ticker history from supporting a monthly target claim."
    )
    return pd.DataFrame(
        [
            {
                "status": status,
                "ticket_ticker_count": len(ticket_tickers),
                "green_ticker_count": len(green_tickers),
                "tickers_with_positive_actual_forward": ", ".join(positive_actual),
                "green_tickers_without_positive_actual_forward": ", ".join(missing_green),
                "replay_only_tickers": ", ".join(replay_only),
                "ticket_tickers": ", ".join(ticket_tickers),
                "green_tickers": ", ".join(green_tickers),
                "required_evidence": required,
                "note": note,
            }
        ],
        columns=TICKET_EXPECTANCY_PROOF_COLUMNS,
    )


def build_monthly_feasibility_guardrail_proof_packet(run_dirs: Sequence[Path]) -> pd.DataFrame:
    """Verify monthly_feasibility.csv artifacts include the green-ticket expectancy guardrail."""

    required_metric = "ready_ticket_expectancy_evidence"
    run_names: list[str] = []
    with_files: list[str] = []
    with_metric: list[str] = []
    missing_metric: list[str] = []
    pass_without_metric: list[str] = []
    for run_dir in run_dirs:
        run_dir = Path(run_dir).expanduser().resolve()
        run_name = run_dir.name
        run_names.append(run_name)
        monthly = _safe_read_csv(run_dir / "monthly_feasibility.csv")
        if monthly.empty:
            missing_metric.append(run_name)
            continue
        with_files.append(run_name)
        metric_text = monthly.get("metric", pd.Series(dtype=object)).astype(str)
        has_metric = bool(metric_text.eq(required_metric).any())
        if has_metric:
            with_metric.append(run_name)
        else:
            missing_metric.append(run_name)
            blockers = monthly[monthly.get("status", pd.Series(dtype=object)).astype(str).str.upper().eq("BLOCK")]
            if blockers.empty:
                pass_without_metric.append(run_name)
    if not run_names:
        status = "MISSING_MONTHLY_FEASIBILITY_ARTIFACTS"
    elif missing_metric:
        status = "FAIL_STALE_MONTHLY_FEASIBILITY_GUARDRAIL"
    else:
        status = "PASS_MONTHLY_FEASIBILITY_GREEN_TICKET_EXPECTANCY_GUARDRAIL"
    claim = (
        "Monthly feasibility artifacts must include ready_ticket_expectancy_evidence so broad or yellow-ticket "
        "expectancy cannot validate green executable tickets."
    )
    note = (
        "Regenerate stale Options Agent run artifacts with current code before using monthly_feasibility.csv for any monthly-readiness claim."
        if missing_metric
        else "Every inspected monthly_feasibility.csv contains the green-ticket expectancy guardrail metric."
    )
    return pd.DataFrame(
        [
            {
                "status": status,
                "run_count": len(run_names),
                "monthly_file_count": len(with_files),
                "required_metric": required_metric,
                "runs_with_required_metric": len(with_metric),
                "missing_required_metric_count": len(missing_metric),
                "missing_required_metric_runs": ", ".join(missing_metric),
                "pass_without_required_metric_runs": ", ".join(pass_without_metric),
                "claim": claim,
                "note": note,
            }
        ],
        columns=MONTHLY_FEASIBILITY_GUARDRAIL_COLUMNS,
    )


def build_actionability_surface_proof_packet(
    *,
    tickets: pd.DataFrame,
    green_ticket_execution_proof: pd.DataFrame,
    market_open_recheck_queue: pd.DataFrame,
) -> pd.DataFrame:
    """Prove that visible tickets separate send-now orders from target-order candidates."""

    target = _target_rows(tickets)
    ready_rows = _truthy_count(tickets.get("ready_to_enter"))
    target_ready = _truthy_count(target.get("ready_to_enter"))
    entry_types = target.get("entry_type", pd.Series(dtype=object)).fillna("").astype(str).str.upper().str.strip()
    queue_entry_types = (
        market_open_recheck_queue.get("entry_type", pd.Series(dtype=object)).fillna("").astype(str).str.upper().str.strip()
    )
    visible_entry_types = pd.concat([entry_types, queue_entry_types], ignore_index=True)
    target_missing_entry_type = int((~entry_types.isin({"CREDIT", "DEBIT"})).sum()) if not target.empty else 0
    target_missing_entry_limit = int(target.get("entry_limit", pd.Series(dtype=object)).map(_to_float).le(0).sum()) if not target.empty else 0
    target_missing_trade_plan = int(target.get("trade_plan", pd.Series(dtype=object)).fillna("").astype(str).str.strip().eq("").sum()) if not target.empty else 0
    target_missing_plain_legs = int((~target.apply(_has_plain_language_trade_legs, axis=1)).sum()) if not target.empty else 0
    target_green_label = int(
        target.get("status_label", pd.Series(dtype=object)).fillna("").astype(str).str.upper().str.contains("GREEN").sum()
    ) if not target.empty else 0
    target_green_icon = int(
        target.get("status_icon", pd.Series(dtype=object)).fillna("").astype(str).str.contains("🟢", regex=False).sum()
    ) if not target.empty else 0
    green_rows = int(_to_float(_first_value(green_ticket_execution_proof, "green_ticket_rows")))
    valid_green_rows = int(_to_float(_first_value(green_ticket_execution_proof, "valid_green_ticket_rows")))
    invalid_green_rows = int(_to_float(_first_value(green_ticket_execution_proof, "invalid_green_ticket_rows")))
    live_queue_rows = int((market_open_recheck_queue.get("source_kind", pd.Series(dtype=object)).astype(str) == "live_probe").sum())
    bad_counts = [
        target_ready,
        target_missing_entry_type,
        target_missing_entry_limit,
        target_missing_trade_plan,
        target_missing_plain_legs,
        target_green_label,
        target_green_icon,
        invalid_green_rows,
    ]
    if len(tickets) <= 0:
        status = "MISSING_TICKET_SURFACE"
    elif any(count > 0 for count in bad_counts):
        status = "FAIL_ACTIONABILITY_SURFACE_INTEGRITY"
    elif green_rows > 0 and valid_green_rows == green_rows:
        status = "PASS_GREEN_AND_YELLOW_SURFACES_SEPARATED"
    elif len(target) > 0:
        status = "PASS_YELLOW_ONLY_SURFACE_SEPARATED"
    else:
        status = "PASS_NO_SEND_NOW_OR_TARGET_SURFACE"

    claim = (
        "Only green_trade_tickets/ready_to_enter=true rows are send-now orders; yellow target-order rows must remain "
        "ready_to_enter=false with explicit credit/debit, BUY/SELL legs, and required recheck."
    )
    note = (
        "Structural recommendation labels such as ENTER are not execution permission. This packet audits the actual "
        "order surface that a human would use for entry."
    )
    return pd.DataFrame(
        [
            {
                "status": status,
                "ticket_rows": len(tickets),
                "ready_to_enter_rows": ready_rows,
                "target_order_rows": len(target),
                "target_ready_to_enter_rows": target_ready,
                "target_missing_entry_type_rows": target_missing_entry_type,
                "target_missing_entry_limit_rows": target_missing_entry_limit,
                "target_missing_trade_plan_rows": target_missing_trade_plan,
                "target_missing_plain_language_leg_rows": target_missing_plain_legs,
                "target_green_label_rows": target_green_label,
                "target_green_icon_rows": target_green_icon,
                "green_ticket_rows": green_rows,
                "valid_green_ticket_rows": valid_green_rows,
                "invalid_green_ticket_rows": invalid_green_rows,
                "live_market_open_recheck_rows": live_queue_rows,
                "entry_types": ", ".join(sorted({value for value in visible_entry_types.tolist() if value})),
                "bad_examples": _bad_actionability_examples(target),
                "claim": claim,
                "note": note,
            }
        ],
        columns=ACTIONABILITY_SURFACE_PROOF_COLUMNS,
    )


def build_action_surface_underlying_quality_proof_packet(
    *,
    tickets: pd.DataFrame,
    market_open_recheck_queue: pd.DataFrame,
    focus_coverage: pd.DataFrame,
) -> pd.DataFrame:
    """Prove low-quality underlyings are audit-visible but not action-looking."""

    ticket_tiers = _quality_tiers(tickets)
    queue_tiers = _action_surface_tiers(market_open_recheck_queue, focus_coverage)
    focus_tiers = _quality_tiers(focus_coverage)
    ticket_bad = ~ticket_tiers.isin(["core", "liquid"])
    queue_bad = ~queue_tiers.isin(["core", "liquid"])
    focus_status = focus_coverage.get("coverage_status", pd.Series(dtype=object)).fillna("").astype(str).str.upper()
    focus_bad_actionable = ~focus_tiers.isin(["core", "liquid"]) & focus_status.ne("NON_ACTIONABLE_UNDERLYING")
    audit_only_focus = focus_status.eq("NON_ACTIONABLE_UNDERLYING")
    bad_count = int(ticket_bad.sum() + queue_bad.sum() + focus_bad_actionable.sum())
    if bad_count > 0:
        status = "FAIL_LOW_QUALITY_UNDERLYING_ACTION_SURFACE"
    elif len(tickets) <= 0 and len(market_open_recheck_queue) <= 0:
        status = "NO_ACTION_SURFACE_ROWS"
    else:
        status = "PASS_ACTION_SURFACES_EXCLUDE_LOW_QUALITY_UNDERLYINGS"
    claim = (
        "Only core or liquid underlyings are present on ticket, live-recheck, or action-looking focus surfaces."
        if status.startswith("PASS")
        else "One or more non-core, excluded, speculative, unknown, or missing-tier underlying still appears on an action-looking surface."
    )
    note = "Speculative, excluded, unknown, or missing-tier underlyings remain audit-only."
    return pd.DataFrame(
        [
            {
                "status": status,
                "ticket_rows": len(tickets),
                "market_open_recheck_rows": len(market_open_recheck_queue),
                "focus_rows": len(focus_coverage),
                "ticket_bad_underlying_rows": int(ticket_bad.sum()),
                "market_open_recheck_bad_underlying_rows": int(queue_bad.sum()),
                "focus_bad_actionable_rows": int(focus_bad_actionable.sum()),
                "ticket_bad_tickers": _ticker_list(tickets.loc[ticket_bad] if not tickets.empty else tickets),
                "market_open_recheck_bad_tickers": _ticker_list(market_open_recheck_queue.loc[queue_bad] if not market_open_recheck_queue.empty else market_open_recheck_queue),
                "focus_bad_actionable_tickers": _ticker_list(focus_coverage.loc[focus_bad_actionable] if not focus_coverage.empty else focus_coverage),
                "core_ticket_rows": int(ticket_tiers.eq("core").sum()),
                "liquid_ticket_rows": int(ticket_tiers.eq("liquid").sum()),
                "audit_only_focus_rows": int(audit_only_focus.sum()),
                "audit_only_focus_tickers": _ticker_list(focus_coverage.loc[audit_only_focus] if not focus_coverage.empty else focus_coverage),
                "liquid_non_core_action_tickers": _ticker_list(tickets.loc[ticket_tiers.eq("liquid")] if not tickets.empty else tickets),
                "claim": claim,
                "note": note,
            }
        ],
        columns=ACTION_SURFACE_UNDERLYING_QUALITY_COLUMNS,
    )


def _action_surface_tiers(frame: pd.DataFrame, focus_coverage: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=object)
    tiers = _quality_tiers(frame)
    if tiers.isin(["core", "liquid", "excluded", "speculative", "unknown"]).all():
        return tiers
    if focus_coverage.empty or "ticker" not in frame.columns or "ticker" not in focus_coverage.columns:
        return tiers
    focus = focus_coverage.copy()
    focus["ticker_key"] = focus["ticker"].astype(str).str.upper()
    focus["tier_key"] = _quality_tiers(focus)
    rank = {"core": 0, "liquid": 1, "speculative": 2, "excluded": 3, "unknown": 4, "": 5}
    focus["tier_rank"] = focus["tier_key"].map(lambda value: rank.get(str(value), 5))
    best = (
        focus.sort_values(["ticker_key", "tier_rank"])
        .drop_duplicates("ticker_key")
        .set_index("ticker_key")["tier_key"]
        .to_dict()
    )
    ticker_keys = frame["ticker"].astype(str).str.upper()
    mapped = ticker_keys.map(lambda ticker: best.get(ticker, "unknown"))
    return tiers.where(tiers.ne(""), mapped)


def _expectancy_source(expectancy: pd.DataFrame, source: str) -> pd.DataFrame:
    if expectancy.empty or "source" not in expectancy.columns:
        return pd.DataFrame()
    return expectancy[expectancy["source"].astype(str).eq(source)].copy()


def _has_blocking_expectancy_source(expectancy: pd.DataFrame) -> bool:
    if expectancy.empty or "status" not in expectancy.columns:
        return False
    return bool(expectancy["status"].astype(str).str.upper().eq("BLOCK").any())


def _status_counts_by_source(expectancy: pd.DataFrame, status: str) -> str:
    if expectancy.empty or not {"source", "status"}.issubset(expectancy.columns):
        return ""
    blocked = expectancy[expectancy["status"].astype(str).str.upper().eq(status.upper())]
    counts = blocked["source"].astype(str).value_counts().sort_index()
    return "; ".join(f"{source}={int(count)}" for source, count in counts.items())


def _statuses_for_sources(expectancy: pd.DataFrame, sources: Sequence[str]) -> str:
    if expectancy.empty or not {"source", "status"}.issubset(expectancy.columns):
        return ""
    parts = []
    for source in sources:
        statuses = expectancy[expectancy["source"].astype(str).eq(source)]["status"].astype(str).value_counts().sort_index()
        if not statuses.empty:
            parts.append(f"{source}: " + ", ".join(f"{status}={int(count)}" for status, count in statuses.items()))
    return "; ".join(parts)


def _statuses_for_source_prefix(expectancy: pd.DataFrame, source_prefix: str) -> str:
    if expectancy.empty or not {"source", "status"}.issubset(expectancy.columns):
        return ""
    source_text = expectancy["source"].astype(str)
    return _statuses_for_sources(expectancy, sorted(source_text[source_text.str.startswith(source_prefix)].unique().tolist()))


def _ticker_counts(frame: pd.DataFrame) -> dict[str, int]:
    if frame.empty or "ticker" not in frame.columns:
        return {}
    counts = frame["ticker"].dropna().astype(str).str.strip().str.upper()
    counts = counts[counts.ne("")]
    return {ticker: int(count) for ticker, count in counts.value_counts().sort_index().items()}


def _expectancy_rows_for_ticker(expectancy: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if expectancy.empty or "matched_current_tickers" not in expectancy.columns:
        return pd.DataFrame()
    target = str(ticker).strip().upper()
    if not target:
        return pd.DataFrame()
    mask = expectancy["matched_current_tickers"].map(
        lambda value: any(tickers_match(target, item) for item in _split_ticker_list(value))
    )
    return expectancy[mask].copy()


def _split_ticker_list(value: Any) -> set[str]:
    return {
        item.strip().upper()
        for item in str(value or "").split(",")
        if item.strip()
    }


def _source_names(frame: pd.DataFrame, status: str) -> list[str]:
    if frame.empty or not {"source", "status"}.issubset(frame.columns):
        return []
    filtered = frame[frame["status"].astype(str).str.upper().eq(status.upper())]
    return sorted(filtered["source"].dropna().astype(str).unique().tolist())


def combine_market_open_recheck_queue(run_dirs: Sequence[Path], live_probe_dirs: Sequence[Path]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for source_kind, dirs in (("multi_date_run", run_dirs), ("live_probe", live_probe_dirs)):
        for run_dir in dirs:
            frame = _safe_read_csv(run_dir / "market_open_recheck_queue.csv", MARKET_OPEN_RECHECK_COLUMNS)
            if frame.empty:
                continue
            frame = _enrich_market_queue_underlying_quality(frame, run_dir)
            frame.insert(0, "source_dir", str(run_dir))
            frame.insert(0, "source_kind", source_kind)
            frame.insert(0, "validation_lane", infer_validation_lane(run_dir))
            frame.insert(0, "date", _run_date(run_dir, _read_manifest(run_dir)))
            rows.append(frame)
    if not rows:
        return pd.DataFrame(columns=MARKET_QUEUE_AUDIT_COLUMNS)
    combined = pd.concat(rows, ignore_index=True, sort=False)
    for column in MARKET_QUEUE_AUDIT_COLUMNS:
        if column not in combined.columns:
            combined[column] = ""
    return combined[MARKET_QUEUE_AUDIT_COLUMNS]


def _enrich_market_queue_underlying_quality(frame: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
    enriched = frame.copy()
    for column in ("underlying_quality_tier", "underlying_quality_reason"):
        if column not in enriched.columns:
            enriched[column] = ""
    needs_tier = enriched["underlying_quality_tier"].fillna("").astype(str).str.strip().eq("")
    needs_reason = enriched["underlying_quality_reason"].fillna("").astype(str).str.strip().eq("")
    if not needs_tier.any() and not needs_reason.any():
        return enriched
    lookup = _underlying_quality_lookup_for_run(run_dir)
    if not lookup:
        return enriched
    ticker_keys = enriched.get("ticker", pd.Series("", index=enriched.index)).astype(str).str.upper()
    for idx, ticker in ticker_keys.items():
        row = lookup.get(ticker)
        if not row:
            continue
        if needs_tier.loc[idx]:
            enriched.at[idx, "underlying_quality_tier"] = row.get("underlying_quality_tier", "")
        if needs_reason.loc[idx]:
            enriched.at[idx, "underlying_quality_reason"] = row.get("underlying_quality_reason", "")
    return enriched


def _underlying_quality_lookup_for_run(run_dir: Path) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    for filename in ("trade_tickets.csv", "decision_board.csv", "ticker_coverage_audit.csv"):
        frame = _safe_read_csv(run_dir / filename)
        if frame.empty or "ticker" not in frame.columns:
            continue
        for _, row in frame.iterrows():
            ticker = str(row.get("ticker") or "").strip().upper()
            if not ticker or ticker in lookup:
                continue
            tier_value = row.get("underlying_quality_tier")
            reason_value = row.get("underlying_quality_reason")
            tier = str(tier_value).strip() if pd.notna(tier_value) else ""
            reason = str(reason_value).strip() if pd.notna(reason_value) else ""
            if tier or reason:
                lookup[ticker] = {
                    "underlying_quality_tier": tier,
                    "underlying_quality_reason": reason,
                }
    return lookup


def build_market_open_recheck_details(market_open_recheck_queue: pd.DataFrame) -> pd.DataFrame:
    """Return row-level proof that recheck rows are complete except for market-session timing."""

    if market_open_recheck_queue.empty:
        return pd.DataFrame(columns=MARKET_OPEN_RECHECK_DETAIL_COLUMNS)
    rows: list[dict[str, Any]] = []
    for _, queue_row in market_open_recheck_queue.iterrows():
        entry_type = str(queue_row.get("entry_type") or "").strip().upper()
        confidence_score = _to_float(queue_row.get("execution_confidence_score"))
        trade_quality_rating = str(queue_row.get("trade_quality_confidence_rating") or "").strip().upper()
        agentic_lanes = _to_float(queue_row.get("external_agent_distinct_review_count"))
        fail_reasons = _market_open_recheck_fail_reasons(queue_row)
        rows.append(
            {
                "date": queue_row.get("date", ""),
                "validation_lane": queue_row.get("validation_lane", ""),
                "source_kind": queue_row.get("source_kind", ""),
                "source_dir": queue_row.get("source_dir", ""),
                "ticker": str(queue_row.get("ticker") or "").strip().upper(),
                "entry_type": entry_type,
                "target_order_status": queue_row.get("target_order_status", ""),
                "order_readiness": queue_row.get("order_readiness", ""),
                "entry_limit": queue_row.get("entry_limit", ""),
                "target_exit": queue_row.get("target_exit", ""),
                "max_profit": queue_row.get("max_profit", ""),
                "max_loss": queue_row.get("max_loss", ""),
                "position_max_profit": queue_row.get("position_max_profit", ""),
                "position_max_loss": queue_row.get("position_max_loss", ""),
                "suggested_contracts": queue_row.get("suggested_contracts", ""),
                "execution_confidence_score": queue_row.get("execution_confidence_score", ""),
                "trade_quality_confidence_rating": queue_row.get("trade_quality_confidence_rating", ""),
                "external_agent_distinct_review_count": queue_row.get("external_agent_distinct_review_count", ""),
                "only_market_session_blocker": _market_open_recheck_blockers_pass(queue_row),
                "target_status_pass": str(queue_row.get("target_order_status") or "").strip().lower() == "target_order_candidate",
                "order_readiness_pass": _market_open_recheck_order_readiness_pass(queue_row),
                "positive_entry_pass": _to_float(queue_row.get("entry_limit")) > 0,
                "positive_contracts_pass": _to_float(queue_row.get("suggested_contracts")) > 0,
                "confidence_score_pass": confidence_score >= MIN_EXECUTION_CONFIDENCE_SCORE,
                "trade_quality_confidence_pass": trade_quality_rating in {"MEDIUM", "HIGH"},
                "agentic_lanes_pass": agentic_lanes >= MIN_AGENTIC_REVIEW_LANES_PER_TICKER,
                "plain_language_legs_pass": _has_plain_language_trade_plan(queue_row),
                "row_pass": len(fail_reasons) == 0,
                "fail_reasons": "; ".join(fail_reasons),
                "trade_plan": queue_row.get("trade_plan", ""),
            }
        )
    return pd.DataFrame(rows, columns=MARKET_OPEN_RECHECK_DETAIL_COLUMNS)


def build_market_open_recheck_proof_packet(
    details: pd.DataFrame,
    *,
    market_open_execution_packet: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Summarize whether yellow market-open rows are ready for a fresh live rerun."""

    if details.empty:
        completed_open_probe = _has_completed_market_open_probe_without_queue(market_open_execution_packet)
        green_ready_orders = _execution_packet_green_ready_orders(market_open_execution_packet)
        status = (
            "PASS_NO_MARKET_OPEN_RECHECK_QUEUE_AFTER_MARKET_OPEN_LIVE_PROBE"
            if completed_open_probe
            else "NO_MARKET_OPEN_RECHECK_ROWS"
        )
        claim = (
            "No market-open recheck queue remains because a regular-session live probe already ran with fresh live quotes, portfolio context, and agentic reviews."
            if completed_open_probe
            else "No market-open recheck rows were available."
        )
        note = (
            "The live probe produced green orders; remaining monthly-readiness gates are green-ticket expectancy coverage plus overall/strategy expectancy evidence."
            if completed_open_probe and green_ready_orders > 0
            else "The live probe still produced no green orders; execution readiness remains blocked until green tickets and green-ticket expectancy coverage pass."
            if completed_open_probe
            else "This is acceptable only when no yellow target candidates are waiting for a regular-session live rerun."
        )
        return pd.DataFrame(
            [
                {
                    "status": status,
                    "queue_rows": 0,
                    "live_queue_rows": 0,
                    "row_pass_rows": 0,
                    "row_fail_rows": 0,
                    "only_market_session_blocker_rows": 0,
                    "target_status_pass_rows": 0,
                    "order_readiness_pass_rows": 0,
                    "positive_entry_rows": 0,
                    "positive_contract_rows": 0,
                    "confidence_score_pass_rows": 0,
                    "trade_quality_confidence_pass_rows": 0,
                    "agentic_lane_pass_rows": 0,
                    "plain_language_leg_rows": 0,
                    "credit_rows": 0,
                    "debit_rows": 0,
                    "tickers": "",
                    "failed_examples": "",
                    "required_gate": _market_open_recheck_required_gate(),
                    "claim": claim,
                    "note": note,
                }
            ],
            columns=MARKET_OPEN_RECHECK_PROOF_COLUMNS,
        )

    row_pass = details.get("row_pass", pd.Series(dtype=object)).map(_truthy)
    entry_types = details.get("entry_type", pd.Series(dtype=object)).fillna("").astype(str).str.upper()
    live_mask = details.get("source_kind", pd.Series(dtype=object)).astype(str).eq("live_probe")
    row_pass_rows = int(row_pass.sum())
    row_fail_rows = int(len(details) - row_pass_rows)
    if row_fail_rows > 0:
        status = "FAIL_MARKET_OPEN_RECHECK_ROWS_INCOMPLETE"
    elif live_mask.any():
        status = "PASS_LIVE_MARKET_OPEN_RECHECK_QUEUE_READY"
    else:
        status = "PASS_DATED_MARKET_OPEN_RECHECK_QUEUE_READY"
    claim = (
        "Every market-open recheck row has complete target order details and is blocked only by the regular-session/fresh-live recheck gate."
        if status.startswith("PASS")
        else "One or more market-open recheck rows are incomplete or blocked by more than regular-session/fresh-live timing."
    )
    note = (
        "This proof is not permission to trade. It proves the queue is ready for a regular-market live rerun; only rows "
        "that rerun into green ready_to_enter=true tickets may be entered."
    )
    return pd.DataFrame(
        [
            {
                "status": status,
                "queue_rows": int(len(details)),
                "live_queue_rows": int(live_mask.sum()),
                "row_pass_rows": row_pass_rows,
                "row_fail_rows": row_fail_rows,
                "only_market_session_blocker_rows": _truthy_count(details.get("only_market_session_blocker")),
                "target_status_pass_rows": _truthy_count(details.get("target_status_pass")),
                "order_readiness_pass_rows": _truthy_count(details.get("order_readiness_pass")),
                "positive_entry_rows": _truthy_count(details.get("positive_entry_pass")),
                "positive_contract_rows": _truthy_count(details.get("positive_contracts_pass")),
                "confidence_score_pass_rows": _truthy_count(details.get("confidence_score_pass")),
                "trade_quality_confidence_pass_rows": _truthy_count(details.get("trade_quality_confidence_pass")),
                "agentic_lane_pass_rows": _truthy_count(details.get("agentic_lanes_pass")),
                "plain_language_leg_rows": _truthy_count(details.get("plain_language_legs_pass")),
                "credit_rows": int(entry_types.eq("CREDIT").sum()),
                "debit_rows": int(entry_types.eq("DEBIT").sum()),
                "tickers": _ticker_list(details),
                "failed_examples": _market_open_recheck_failed_examples(details),
                "required_gate": _market_open_recheck_required_gate(),
                "claim": claim,
                "note": note,
            }
        ],
        columns=MARKET_OPEN_RECHECK_PROOF_COLUMNS,
    )


def build_session_only_green_shadow_proof_packet(details: pd.DataFrame) -> pd.DataFrame:
    """Prove non-session green prerequisites without promoting closed-market rows."""

    if details.empty:
        return pd.DataFrame(
            [
                {
                    "status": "NO_SESSION_ONLY_GREEN_SHADOW_ROWS",
                    "shadow_candidate_rows": 0,
                    "row_pass_rows": 0,
                    "row_fail_rows": 0,
                    "non_session_blocker_rows": 0,
                    "credit_rows": 0,
                    "debit_rows": 0,
                    "position_max_profit": 0.0,
                    "position_max_loss": 0.0,
                    "tickers": "",
                    "failed_examples": "",
                    "required_next_action": "No session-only candidates are available; inspect target preservation and live queue proofs.",
                    "claim": "No shadow-green candidates were available.",
                    "note": "This proof does not create executable tickets.",
                }
            ],
            columns=SESSION_ONLY_GREEN_SHADOW_PROOF_COLUMNS,
        )

    frame = details.copy()
    if "source_kind" in frame.columns and frame["source_kind"].astype(str).eq("live_probe").any():
        frame = frame[frame["source_kind"].astype(str).eq("live_probe")].copy()
    row_pass = frame.get("row_pass", pd.Series(dtype=object)).map(_truthy)
    entry_types = frame.get("entry_type", pd.Series(dtype=object)).fillna("").astype(str).str.upper()
    position_profit = _position_value_sum(frame, "max_profit", "position_max_profit")
    position_loss = _position_value_sum(frame, "max_loss", "position_max_loss")
    row_pass_rows = int(row_pass.sum())
    row_fail_rows = int(len(frame) - row_pass_rows)
    non_session_blockers = int(
        len(frame)
        - _truthy_count(frame.get("only_market_session_blocker", pd.Series(dtype=object)))
    )
    status = (
        "PASS_SESSION_ONLY_GREEN_SHADOW_READY"
        if len(frame) > 0 and row_fail_rows == 0 and non_session_blockers == 0
        else "FAIL_SESSION_ONLY_GREEN_SHADOW"
    )
    claim = (
        "Current live candidates clear all non-session green-ticket gates and are blocked only by regular-market timing."
        if status == "PASS_SESSION_ONLY_GREEN_SHADOW_READY"
        else "One or more current live candidates still has a non-session blocker."
    )
    return pd.DataFrame(
        [
            {
                "status": status,
                "shadow_candidate_rows": int(len(frame)),
                "row_pass_rows": row_pass_rows,
                "row_fail_rows": row_fail_rows,
                "non_session_blocker_rows": non_session_blockers,
                "credit_rows": int(entry_types.eq("CREDIT").sum()),
                "debit_rows": int(entry_types.eq("DEBIT").sum()),
                "position_max_profit": round(position_profit, 2),
                "position_max_loss": round(position_loss, 2),
                "tickers": _ticker_list(frame),
                "failed_examples": _market_open_recheck_failed_examples(frame),
                "required_next_action": (
                    "Rerun Options Agent during regular market hours; enter only rows that become green ready_to_enter=true."
                ),
                "claim": claim,
                "note": (
                    "This is a closed-market shadow proof. It is not execution permission and does not satisfy the green-ticket goal gate."
                ),
            }
        ],
        columns=SESSION_ONLY_GREEN_SHADOW_PROOF_COLUMNS,
    )


def build_live_probe_summary(live_summaries: pd.DataFrame) -> pd.DataFrame:
    if live_summaries.empty:
        return pd.DataFrame(columns=LIVE_PROBE_COLUMNS)
    frame = live_summaries.copy()
    for column in LIVE_PROBE_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    frame["agentic_reviews_ready"] = frame["agentic_review_lane_coverage_pct"].map(_to_float).ge(0.8)
    return frame[LIVE_PROBE_COLUMNS]


def build_market_open_execution_packet(
    *,
    base_dir: Path,
    live_probe_dirs: Sequence[Path],
    live_probe_summary: pd.DataFrame,
    market_open_recheck_queue: pd.DataFrame,
    rerun_agent_reviews_json: Optional[Path] = None,
) -> pd.DataFrame:
    """Build the exact rerun packet needed to collect market-hours evidence."""

    packet_source_dirs = list(live_probe_dirs)
    if not packet_source_dirs and not market_open_recheck_queue.empty and "source_dir" in market_open_recheck_queue.columns:
        seen_sources: set[str] = set()
        packet_source_dirs = []
        for source in market_open_recheck_queue["source_dir"].dropna().astype(str).tolist():
            if not source or source in seen_sources:
                continue
            seen_sources.add(source)
            packet_source_dirs.append(Path(source))
    if not packet_source_dirs:
        return pd.DataFrame(columns=MARKET_OPEN_EXECUTION_PACKET_COLUMNS)
    summary_by_source = {
        str(row.get("source_dir")): row
        for _, row in live_probe_summary.iterrows()
    }
    rows: list[dict[str, Any]] = []
    for run_dir in packet_source_dirs:
        manifest = _read_manifest(run_dir)
        day = _run_date(run_dir, manifest)
        source_dir = str(run_dir)
        summary = summary_by_source.get(source_dir, {})
        agent_reviews_json = str(rerun_agent_reviews_json) if rerun_agent_reviews_json else _agent_reviews_json_from_manifest(manifest)
        queue_rows = _queue_rows_for_source(market_open_recheck_queue, source_dir)
        green_ready_orders = int(_to_float(_mapping_get(summary, "green_ready_orders")))
        fresh_live = _truthy(_mapping_get(summary, "fresh_live_quotes_ready"))
        portfolio_ready = _truthy(_mapping_get(summary, "portfolio_ready"))
        agentic_ready = _truthy(_mapping_get(summary, "agentic_reviews_ready"))
        market_open = _truthy(_mapping_get(summary, "market_session_open"))
        next_session = next_regular_market_session_start()
        rerun_dir = _next_live_rerun_dir(run_dir, day)
        status = _execution_packet_status(
            fresh_live=fresh_live,
            portfolio_ready=portfolio_ready,
            agentic_ready=agentic_ready,
            market_open=market_open,
            green_ready_orders=green_ready_orders,
            queue_rows=queue_rows,
        )
        evidence_dir = run_dir if market_open else rerun_dir
        rerun_command = "" if market_open else _market_open_rerun_command(
            day=day,
            base_dir=base_dir,
            out_dir=rerun_dir,
            agent_reviews_json=agent_reviews_json,
        )
        rows.append(
            {
                "date": day,
                "source_dir": source_dir,
                "status": status,
                "fresh_live_quotes_ready": fresh_live,
                "portfolio_ready": portfolio_ready,
                "agentic_reviews_ready": agentic_ready,
                "market_session_open": market_open,
                "green_ready_orders": green_ready_orders,
                "yellow_recheck_rows": queue_rows,
                "next_regular_session_start": next_session.isoformat(),
                "market_calendar_note": "Full-day U.S. equity market holidays are excluded from regular-session rerun timing.",
                "agent_reviews_json": agent_reviews_json,
                "out_dir": str(evidence_dir),
                "command": rerun_command,
                "required_condition": (
                    "regular_market_session_open + fresh live Schwab chain + live portfolio context + "
                    "agentic lane coverage still passing + positive entry/size/confidence; $10k/month claims also require positive structure-aligned ticket expectancy"
                ),
                "note": (
                    "Do not enter rows from the yellow queue. Rerun during regular market hours and only use rows that move to green ready_to_enter=true."
                ),
            }
        )
    return pd.DataFrame(rows, columns=MARKET_OPEN_EXECUTION_PACKET_COLUMNS)


def build_live_rerun_preflight_details(
    *,
    market_open_recheck_details: pd.DataFrame,
    market_open_execution_packet: pd.DataFrame,
) -> pd.DataFrame:
    """Check whether the rerun's agent-review JSON covers every queued ticker."""

    if _has_completed_market_open_probe_without_queue(market_open_execution_packet):
        return pd.DataFrame(columns=LIVE_RERUN_PREFLIGHT_DETAIL_COLUMNS)
    if market_open_recheck_details.empty:
        return pd.DataFrame(columns=LIVE_RERUN_PREFLIGHT_DETAIL_COLUMNS)
    review_path = _first_execution_agent_reviews_json(market_open_execution_packet)
    reviews, _ = _read_review_json(Path(review_path)) if review_path else (pd.DataFrame(), ["missing agent_reviews_json"])
    rows: list[dict[str, Any]] = []
    for ticker in _clean_ticker_sequence(market_open_recheck_details.get("ticker", pd.Series(dtype=object)).tolist()):
        ticker_reviews = reviews[
            reviews.get("ticker", pd.Series(dtype=object)).astype(str).str.upper().eq(ticker)
        ] if not reviews.empty else pd.DataFrame()
        agents = sorted(
            {
                str(agent).strip()
                for agent in ticker_reviews.get("agent", pd.Series(dtype=object)).dropna().tolist()
                if str(agent).strip()
            }
        )
        fail_reasons: list[str] = []
        if not review_path:
            fail_reasons.append("agent_reviews_json_missing_from_rerun_command")
        elif ticker_reviews.empty:
            fail_reasons.append("ticker_missing_from_agent_reviews_json")
        if len(agents) < MIN_AGENTIC_REVIEW_LANES_PER_TICKER:
            fail_reasons.append("distinct_agent_review_lanes_below_minimum")
        rows.append(
            {
                "ticker": ticker,
                "required_review_lanes": MIN_AGENTIC_REVIEW_LANES_PER_TICKER,
                "review_count": int(len(ticker_reviews)),
                "distinct_agent_count": int(len(agents)),
                "agents": "; ".join(agents),
                "review_file": review_path,
                "row_pass": len(fail_reasons) == 0,
                "fail_reasons": "; ".join(fail_reasons),
            }
        )
    return pd.DataFrame(rows, columns=LIVE_RERUN_PREFLIGHT_DETAIL_COLUMNS)


def build_live_rerun_preflight_proof_packet(
    *,
    base_dir: Path,
    market_open_recheck_details: pd.DataFrame,
    market_open_execution_packet: pd.DataFrame,
    preflight_details: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize whether the regular-session rerun has all non-market prerequisites."""

    execution = _select_execution_packet_row(market_open_execution_packet)
    command = str(_mapping_get(execution, "command") or "")
    parts = _split_command(command)
    review_path_text = _first_execution_agent_reviews_json(market_open_execution_packet)
    review_path = Path(review_path_text) if review_path_text else None
    reviews, review_errors = _read_review_json(review_path) if review_path else (pd.DataFrame(), ["missing agent_reviews_json"])
    rerun_out_dir_text = str(_mapping_get(execution, "out_dir") or "")
    rerun_out_dir = Path(rerun_out_dir_text).expanduser() if rerun_out_dir_text else None
    day = str(_mapping_get(execution, "date") or "")
    source_date_available = bool(day and _has_uw_source_files(base_dir / day, day))
    completed_open_probe = _has_completed_market_open_probe_without_queue(market_open_execution_packet)
    queue_tickers = (
        []
        if completed_open_probe
        else _clean_ticker_sequence(market_open_recheck_details.get("ticker", pd.Series(dtype=object)).tolist())
    )
    row_pass = preflight_details.get("row_pass", pd.Series(dtype=object)).map(_truthy) if not preflight_details.empty else pd.Series(dtype=bool)
    covered = _clean_ticker_sequence(preflight_details.loc[row_pass, "ticker"].tolist()) if not preflight_details.empty else []
    missing = [ticker for ticker in queue_tickers if ticker not in set(covered)]
    command_live = "--live-schwab" in parts
    command_portfolio = "--live-portfolio" in parts
    command_reviews = "--agent-reviews-json" in parts and bool(review_path_text)
    rerun_clear = bool(rerun_out_dir and not rerun_out_dir.exists())
    review_valid = bool(review_path and review_path.exists() and not review_errors)

    fail_reasons: list[str] = []
    if completed_open_probe:
        status = "PASS_NO_LIVE_RERUN_QUEUE_AFTER_MARKET_OPEN_LIVE_PROBE"
    elif not queue_tickers:
        status = (
            "PASS_NO_LIVE_RERUN_QUEUE_AFTER_MARKET_OPEN_LIVE_PROBE"
            if _has_completed_market_open_probe_without_queue(market_open_execution_packet)
            else "NO_LIVE_RERUN_QUEUE"
        )
    else:
        if not command_live:
            fail_reasons.append("rerun_command_missing_live_schwab")
        if not command_portfolio:
            fail_reasons.append("rerun_command_missing_live_portfolio")
        if not command_reviews:
            fail_reasons.append("rerun_command_missing_agent_reviews_json")
        if not review_valid:
            fail_reasons.append("agent_reviews_json_missing_or_invalid")
        if missing:
            fail_reasons.append("agent_reviews_json_missing_queue_tickers")
        if not rerun_clear:
            fail_reasons.append("rerun_out_dir_not_clear")
        if not source_date_available:
            fail_reasons.append("source_date_files_missing")
        status = "PASS_LIVE_RERUN_PREFLIGHT_READY" if not fail_reasons else "FAIL_LIVE_RERUN_PREFLIGHT"

    claim = (
        "The live rerun command has fresh-output, live-Schwab, live-portfolio, source-date, and queue-ticker agent review prerequisites."
        if status == "PASS_LIVE_RERUN_PREFLIGHT_READY"
        else "No queue-ticker preflight remains because the regular-session live probe already ran and produced no session-only yellow queue."
        if status == "PASS_NO_LIVE_RERUN_QUEUE_AFTER_MARKET_OPEN_LIVE_PROBE"
        else "The live rerun command is missing one or more non-market prerequisites."
    )
    note = (
        "This preflight does not create green tickets. It proves the regular-session rerun should not fail because of "
        "stale output paths, missing source files, or agent reviews that do not cover the queued tickers."
    )
    return pd.DataFrame(
        [
            {
                "status": status,
                "queue_ticker_count": len(queue_tickers),
                "queue_tickers": ", ".join(queue_tickers),
                "covered_queue_ticker_count": len(covered),
                "missing_queue_ticker_count": len(missing),
                "missing_queue_tickers": ", ".join(missing),
                "agent_reviews_json": review_path_text,
                "agent_reviews_json_exists": bool(review_path and review_path.exists()),
                "agent_reviews_json_valid": review_valid,
                "agent_review_rows": int(len(reviews)),
                "distinct_agent_count": _distinct_agents(reviews),
                "rerun_command_has_live_schwab": command_live,
                "rerun_command_has_live_portfolio": command_portfolio,
                "rerun_command_has_agent_reviews_json": command_reviews,
                "rerun_out_dir": rerun_out_dir_text,
                "rerun_out_dir_clear": rerun_clear,
                "source_date": day,
                "source_date_available": source_date_available,
                "failed_examples": _live_rerun_preflight_failed_examples(preflight_details, fail_reasons, review_errors),
                "required_gate": _live_rerun_preflight_required_gate(),
                "claim": claim,
                "note": note,
            }
        ],
        columns=LIVE_RERUN_PREFLIGHT_PROOF_COLUMNS,
    )


def build_multi_date_readiness_proof_packet(
    *,
    summary: pd.DataFrame,
    live_probe_summary: pd.DataFrame,
    market_open_execution_packet: pd.DataFrame,
) -> pd.DataFrame:
    """Separate broad historical validation from latest live execution evidence."""

    dates = _dates_from_frame(summary)
    dates_with_tickets = _dates_where_numeric(summary, "trade_ticket_rows")
    dates_with_green = _dates_where_numeric(summary, "green_ready_orders")
    dates_with_yellow = _dates_where_numeric(summary, "yellow_target_candidates")
    dated_ticket_rows = _sum_numeric(summary, "trade_ticket_rows")
    dated_green = _sum_numeric(summary, "green_ready_orders")
    dated_yellow = _sum_numeric(summary, "yellow_target_candidates")
    live_dates = _dates_from_frame(live_probe_summary)
    live_green = _sum_numeric(live_probe_summary, "green_ready_orders")
    live_yellow = _sum_numeric(live_probe_summary, "market_open_recheck_queue")
    live_market_open = int(live_probe_summary.get("market_session_open", pd.Series(dtype=object)).map(_truthy).sum()) if not live_probe_summary.empty else 0
    latest_live_date = live_dates[-1] if live_dates else ""
    latest_live_status = ""
    if not market_open_execution_packet.empty:
        ordered_packet = market_open_execution_packet.copy()
        if "date" in ordered_packet.columns:
            ordered_packet = ordered_packet.sort_values("date")
        latest_live_status = str(ordered_packet.iloc[-1].get("status", ""))

    if len(dates) < 2:
        status = "NOT_ENOUGH_VALIDATION_DATES"
    elif dated_ticket_rows <= 0:
        status = "BLOCK_NO_USER_VISIBLE_TICKETS_ACROSS_DATES"
    elif dated_green <= 0 and dated_yellow <= 0:
        status = "BLOCK_NO_GREEN_OR_YELLOW_ACTION_ROWS_ACROSS_DATES"
    elif not live_dates:
        status = "NEEDS_LIVE_PROBE_FOR_EXECUTION_EVIDENCE"
    elif live_green > 0 and live_market_open > 0:
        status = "PASS_MULTI_DATE_AND_LIVE_GREEN_EVIDENCE"
    elif live_yellow > 0:
        status = "PASS_MULTI_DATE_TARGETS_WAITING_FOR_REGULAR_SESSION_LIVE_GREEN"
    elif live_market_open > 0:
        status = "PASS_MULTI_DATE_WITH_MARKET_OPEN_LIVE_PROBE_NO_GREEN_TICKETS"
    else:
        status = "MULTI_DATE_VALIDATION_PRESENT_BUT_NO_LIVE_EXECUTION_QUEUE"

    claim = (
        "Multi-date validation is separate from the latest live-session probe; the latest live probe cannot by itself "
        "stand in for the full validation window."
    )
    note = (
        "A PASS here does not make the pipeline execution-ready. Completion still requires live regular-session green "
        "tickets and structure-aligned positive expectancy evidence."
    )
    return pd.DataFrame(
        [
            {
                "status": status,
                "validation_date_count": len(dates),
                "tested_dates": ", ".join(dates),
                "dated_ticket_rows": dated_ticket_rows,
                "dated_green_ready_orders": dated_green,
                "dated_yellow_target_candidates": dated_yellow,
                "dates_with_tickets": ", ".join(dates_with_tickets),
                "dates_with_green_ready_orders": ", ".join(dates_with_green),
                "dates_with_yellow_target_candidates": ", ".join(dates_with_yellow),
                "live_probe_count": len(live_dates),
                "live_probe_dates": ", ".join(live_dates),
                "live_market_session_open_count": live_market_open,
                "live_green_ready_orders": live_green,
                "live_yellow_recheck_rows": live_yellow,
                "latest_live_probe_date": latest_live_date,
                "latest_live_probe_status": latest_live_status,
                "claim": claim,
                "note": note,
            }
        ],
        columns=MULTI_DATE_READINESS_PROOF_COLUMNS,
    )


def build_market_session_verification_plan(
    *,
    market_open_execution_packet: pd.DataFrame,
) -> pd.DataFrame:
    """Write the pass/fail checklist for the next regular-session live run."""

    if market_open_execution_packet.empty:
        return pd.DataFrame(columns=MARKET_SESSION_VERIFICATION_PLAN_COLUMNS)
    rows: list[dict[str, Any]] = []
    for _, packet in market_open_execution_packet.iterrows():
        rerun_out_dir = Path(str(packet.get("out_dir") or packet.get("source_dir") or "."))
        status = str(packet.get("status") or "")
        if status == "ready_for_regular_session_rerun":
            plan_status = "WAITING_FOR_REGULAR_SESSION"
            note = "Inputs are current enough for a regular-session rerun, but no row is executable until that rerun creates green tickets."
        elif status == "green_orders_present_verify_ticket_scoped_expectancy":
            plan_status = "VERIFY_GREEN_ORDERS_AND_EXPECTANCY"
            note = "Green rows exist; verify structure-aligned ticket expectancy before any readiness or monthly-target claim."
        elif status == "refresh_live_probe_inputs_before_rerun":
            plan_status = "REFRESH_INPUTS_BEFORE_SESSION_RUN"
            note = "Live quote, portfolio, or agentic-review prerequisites are stale or missing."
        elif status == "market_open_live_probe_no_green_orders":
            plan_status = "NO_GREEN_TICKETS_AFTER_MARKET_OPEN_PROBE"
            note = "A regular-session live probe already ran with current prerequisites, but no green order evidence passed."
        elif status == "no_market_session_only_yellow_queue":
            plan_status = "NO_GREEN_TICKETS_AFTER_MARKET_OPEN_PROBE"
            note = "There is no session-only yellow queue and no green order evidence."
        else:
            plan_status = "NO_EXECUTION_EVIDENCE_READY"
            note = "The current packet does not provide a market-session execution path."
        rows.append(
            {
                "date": packet.get("date", ""),
                "status": plan_status,
                "next_regular_session_start": packet.get("next_regular_session_start", ""),
                "yellow_recheck_rows": packet.get("yellow_recheck_rows", 0),
                "green_ready_orders": packet.get("green_ready_orders", 0),
                "rerun_command": packet.get("command", ""),
                "rerun_out_dir": str(rerun_out_dir),
                "green_ticket_file": str(rerun_out_dir / "green_trade_tickets.csv"),
                "trade_ticket_file": str(rerun_out_dir / "trade_tickets.csv"),
                "execution_readiness_file": str(rerun_out_dir / "execution_readiness.csv"),
                "expectancy_file": str(rerun_out_dir / "expectancy_evidence.csv"),
                "pass_criteria": (
                    "regular market session open; fresh live Schwab and portfolio gates PASS; "
                    "agentic lane coverage PASS; green_trade_tickets.csv has ready_to_enter=true rows; "
                    "each green row has positive entry limit, positive contracts, passing confidence ratings, "
                    "and plain-language buy/sell legs"
                ),
                "fail_criteria": (
                    "market_session_open is false, green_trade_tickets.csv is empty, execution_readiness has BLOCK gates, "
                    "or expectancy_evidence lacks positive structure-aligned actual/forward outcome support"
                ),
                "completion_gate": (
                    "Do not mark the goal complete unless the completion verdict says can_mark_goal_complete=true and "
                    "update_goal_action=call_update_goal_complete."
                ),
                "note": note,
            }
        )
    return pd.DataFrame(rows, columns=MARKET_SESSION_VERIFICATION_PLAN_COLUMNS)


def build_post_rerun_verification_packet(
    *,
    market_session_verification_plan: pd.DataFrame,
    live_probe_summary: pd.DataFrame,
    green_ticket_execution_proof: pd.DataFrame,
    ticket_expectancy_proof: pd.DataFrame,
    completion_verdict: pd.DataFrame,
    audit_regeneration_command: str = "",
) -> pd.DataFrame:
    """Build a one-row verdict for the post-market-open rerun evidence."""

    plan = _select_verification_plan_row(market_session_verification_plan)
    plan_status = str(_mapping_get(plan, "status") or "")
    market_open = bool(live_probe_summary.get("market_session_open", pd.Series(dtype=object)).map(_truthy).any()) if not live_probe_summary.empty else False
    green_status = str(_first_value(green_ticket_execution_proof, "status"))
    ticket_expectancy_status = str(_first_value(ticket_expectancy_proof, "status"))
    completion_status = str(_first_value(completion_verdict, "status"))
    can_complete = _truthy(_first_value(completion_verdict, "can_mark_goal_complete"))
    update_action = str(_first_value(completion_verdict, "update_goal_action"))
    green_rows = int(_to_float(_first_value(green_ticket_execution_proof, "green_ticket_rows")))
    valid_green_rows = int(_to_float(_first_value(green_ticket_execution_proof, "valid_green_ticket_rows")))
    invalid_green_rows = int(_to_float(_first_value(green_ticket_execution_proof, "invalid_green_ticket_rows")))
    green_ticker_count = int(_to_float(_first_value(ticket_expectancy_proof, "green_ticker_count")))
    monthly_claim_allowed = _truthy(_first_value(completion_verdict, "monthly_claim_allowed"))
    evidence_files = _post_rerun_evidence_files(plan)

    if market_session_verification_plan.empty:
        status = "NO_VERIFICATION_PLAN"
        next_action = "Regenerate the expanded audit with a live probe directory."
    elif can_complete and update_action == "call_update_goal_complete":
        status = "PASS_READY_TO_COMPLETE_GOAL"
        next_action = "Final audit evidence permits update_goal complete."
    elif plan_status == "WAITING_FOR_REGULAR_SESSION" or not market_open:
        status = "WAITING_FOR_REGULAR_SESSION_LIVE_RERUN"
        next_action = "Rerun the listed command during regular market hours, then regenerate this packet."
    elif green_status == "BLOCK_NO_GREEN_TICKETS":
        status = "FAIL_NO_GREEN_TICKETS_AFTER_RERUN"
        next_action = "Do not trade; inspect trade_tickets.csv and execution_readiness.csv for blockers."
    elif green_status.startswith("FAIL") or green_status.startswith("BLOCK"):
        status = "FAIL_GREEN_TICKET_EXECUTION_PROOF"
        next_action = "Do not trade; fix invalid green-ticket rows or missing execution gates."
    elif ticket_expectancy_status.startswith("BLOCK") or ticket_expectancy_status.startswith("WARN"):
        status = "FAIL_TICKET_EXPECTANCY_PROOF"
        next_action = "Do not claim monthly readiness; green tickers need structure-aligned actual/forward expectancy support."
    elif not can_complete:
        status = "FAIL_COMPLETION_VERDICT"
        next_action = "Do not close the goal; inspect completion_verdict.md for remaining gates."
    else:
        status = "NEEDS_MANUAL_REVIEW"
        next_action = "Review proof packets before any readiness or monthly-target claim."

    note = (
        "This packet is the post-rerun go/no-go check. It is not enough for a rerun to finish; green rows, "
        "structure-aligned ticket expectancy, and the completion verdict must all agree."
    )
    return pd.DataFrame(
        [
            {
                "status": status,
                "date": _mapping_get(plan, "date") or "",
                "market_session_open": market_open,
                "green_ticket_status": green_status,
                "ticket_expectancy_status": ticket_expectancy_status,
                "completion_verdict_status": completion_status,
                "can_mark_goal_complete": can_complete,
                "update_goal_action": update_action,
                "green_ticket_rows": green_rows,
                "valid_green_ticket_rows": valid_green_rows,
                "invalid_green_ticket_rows": invalid_green_rows,
                "green_ticker_count": green_ticker_count,
                "monthly_claim_allowed": monthly_claim_allowed,
                "rerun_command": _mapping_get(plan, "rerun_command") or "",
                "audit_regeneration_command": audit_regeneration_command,
                "evidence_files": evidence_files,
                "required_next_action": next_action,
                "note": note,
            }
        ],
        columns=POST_RERUN_VERIFICATION_COLUMNS,
    )


def build_green_ticket_execution_details(
    *,
    live_probe_dirs: Sequence[Path],
    live_probe_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Return row-level proof details for green executable tickets in live probes."""

    summary_by_source = {
        str(row.get("source_dir")): row
        for _, row in live_probe_summary.iterrows()
    }
    rows: list[dict[str, Any]] = []
    for run_dir in live_probe_dirs:
        run_dir = run_dir.expanduser().resolve()
        manifest = _read_manifest(run_dir)
        day = _run_date(run_dir, manifest)
        source_dir = str(run_dir)
        summary = summary_by_source.get(source_dir, {})
        market_open = _truthy(_mapping_get(summary, "market_session_open"))
        frame = _safe_read_csv(run_dir / "green_trade_tickets.csv")
        if frame.empty:
            continue
        for _, ticket in frame.iterrows():
            fail_reasons = _green_ticket_fail_reasons(ticket, market_open=market_open)
            confidence_score = _to_float(ticket.get("execution_confidence_score"))
            execution_confidence_rating = str(ticket.get("execution_confidence_rating") or "").strip().upper()
            trade_quality_confidence_rating = str(ticket.get("trade_quality_confidence_rating") or "").strip().upper()
            rows.append(
                {
                    "date": day,
                    "validation_lane": infer_validation_lane(run_dir),
                    "source_dir": source_dir,
                    "ticker": ticket.get("ticker", ""),
                    "ready_to_enter": ticket.get("ready_to_enter", ""),
                    "order_readiness": ticket.get("order_readiness", ""),
                    "entry_type": ticket.get("entry_type", ""),
                    "entry_limit": ticket.get("entry_limit", ""),
                    "suggested_contracts": ticket.get("suggested_contracts", ""),
                    "live_validation_status": ticket.get("live_validation_status", ""),
                    "execution_status": ticket.get("execution_status", ""),
                    "execution_blockers": ticket.get("execution_blockers", ""),
                    "execution_confidence_score": ticket.get("execution_confidence_score", ""),
                    "execution_confidence_rating": ticket.get("execution_confidence_rating", ""),
                    "trade_quality_confidence_rating": ticket.get("trade_quality_confidence_rating", ""),
                    "confidence_score_pass": confidence_score >= MIN_EXECUTION_CONFIDENCE_SCORE,
                    "execution_confidence_pass": execution_confidence_rating in {"MEDIUM", "HIGH"},
                    "trade_quality_confidence_pass": trade_quality_confidence_rating in {"MEDIUM", "HIGH"},
                    "market_session_open": market_open,
                    "trade_plan": ticket.get("trade_plan", ""),
                    "sell_leg": ticket.get("sell_leg", ""),
                    "buy_leg": ticket.get("buy_leg", ""),
                    "row_pass": len(fail_reasons) == 0,
                    "fail_reasons": "; ".join(fail_reasons),
                }
            )
    return pd.DataFrame(rows, columns=GREEN_TICKET_EXECUTION_DETAIL_COLUMNS)


def build_green_ticket_execution_proof_packet(
    *,
    details: pd.DataFrame,
    live_probe_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize whether live green tickets are execution-ready at row level."""

    live_probe_count = len(live_probe_summary)
    row_count = len(details)
    if details.empty:
        status = "BLOCK_NO_GREEN_TICKETS"
        valid_rows = 0
        invalid_rows = 0
        ready_rows = 0
        positive_entry_rows = 0
        positive_contract_rows = 0
        live_pass_rows = 0
        no_blocker_rows = 0
        confidence_score_pass_rows = 0
        execution_confidence_pass_rows = 0
        trade_quality_confidence_pass_rows = 0
        plain_language_rows = 0
        market_open_rows = 0
        invalid_examples = ""
    else:
        ready_rows = int(details["ready_to_enter"].map(_truthy).sum())
        positive_entry_rows = int(details["entry_limit"].map(_to_float).gt(0).sum())
        positive_contract_rows = int(details["suggested_contracts"].map(_to_float).gt(0).sum())
        live_pass_rows = int(details["live_validation_status"].astype(str).str.upper().eq("PASS").sum())
        no_blocker_rows = int(details["execution_blockers"].map(_no_execution_blockers).sum())
        confidence_score_pass_rows = int(details["confidence_score_pass"].map(_truthy).sum())
        execution_confidence_pass_rows = int(details["execution_confidence_pass"].map(_truthy).sum())
        trade_quality_confidence_pass_rows = int(details["trade_quality_confidence_pass"].map(_truthy).sum())
        plain_language_rows = int(details.apply(_has_plain_language_trade_legs, axis=1).sum())
        market_open_rows = int(details["market_session_open"].map(_truthy).sum())
        valid_mask = details["row_pass"].map(_truthy)
        valid_rows = int(valid_mask.sum())
        invalid_rows = int(row_count - valid_rows)
        if invalid_rows:
            status = "FAIL_INVALID_GREEN_TICKET_ROWS"
        elif market_open_rows < row_count:
            status = "BLOCK_MARKET_SESSION_NOT_OPEN_FOR_GREEN_TICKETS"
        else:
            status = "PASS_GREEN_TICKETS_EXECUTION_READY"
        invalid_examples = _invalid_green_ticket_examples(details)
    required = (
        "Every green row must have ready_to_enter=true, order_readiness=ready_to_enter, live_validation_status=PASS, "
        "no execution blockers, positive entry_limit, positive suggested_contracts, regular market session open, "
        f"execution_confidence_score >= {MIN_EXECUTION_CONFIDENCE_SCORE}, MEDIUM/HIGH execution and trade-quality "
        "confidence ratings, and plain-language BUY/SELL legs."
    )
    note = (
        "This packet proves green order readiness only. It does not prove the monthly target; expectancy proof must pass separately."
    )
    return pd.DataFrame(
        [
            {
                "status": status,
                "live_probe_count": live_probe_count,
                "green_ticket_rows": row_count,
                "valid_green_ticket_rows": valid_rows,
                "invalid_green_ticket_rows": invalid_rows,
                "ready_to_enter_rows": ready_rows,
                "positive_entry_rows": positive_entry_rows,
                "positive_contract_rows": positive_contract_rows,
                "live_validation_pass_rows": live_pass_rows,
                "no_blocker_rows": no_blocker_rows,
                "confidence_score_pass_rows": confidence_score_pass_rows,
                "execution_confidence_pass_rows": execution_confidence_pass_rows,
                "trade_quality_confidence_pass_rows": trade_quality_confidence_pass_rows,
                "plain_language_leg_rows": plain_language_rows,
                "market_session_open_rows": market_open_rows,
                "green_tickers": _ticker_list(details),
                "invalid_examples": invalid_examples,
                "required_evidence": required,
                "note": note,
            }
        ],
        columns=GREEN_TICKET_EXECUTION_PROOF_COLUMNS,
    )


def _green_ticket_fail_reasons(row: Mapping[str, Any] | pd.Series, *, market_open: bool) -> list[str]:
    reasons: list[str] = []
    if not _truthy(_mapping_get(row, "ready_to_enter")):
        reasons.append("ready_to_enter_not_true")
    order_readiness = str(_mapping_get(row, "order_readiness") or "").strip()
    if order_readiness and order_readiness != "ready_to_enter":
        reasons.append("order_readiness_not_ready_to_enter")
    if _to_float(_mapping_get(row, "entry_limit")) <= 0:
        reasons.append("entry_limit_not_positive")
    if _to_float(_mapping_get(row, "suggested_contracts")) <= 0:
        reasons.append("suggested_contracts_not_positive")
    if str(_mapping_get(row, "live_validation_status") or "").strip().upper() != "PASS":
        reasons.append("live_validation_status_not_PASS")
    if not _no_execution_blockers(_mapping_get(row, "execution_blockers")):
        reasons.append("execution_blockers_present")
    if _to_float(_mapping_get(row, "execution_confidence_score")) < MIN_EXECUTION_CONFIDENCE_SCORE:
        reasons.append("execution_confidence_score_below_threshold")
    if str(_mapping_get(row, "execution_confidence_rating") or "").strip().upper() not in {"MEDIUM", "HIGH"}:
        reasons.append("execution_confidence_rating_not_MEDIUM_or_HIGH")
    if str(_mapping_get(row, "trade_quality_confidence_rating") or "").strip().upper() not in {"MEDIUM", "HIGH"}:
        reasons.append("trade_quality_confidence_rating_not_MEDIUM_or_HIGH")
    if not market_open:
        reasons.append("market_session_not_open")
    if not _has_plain_language_trade_legs(row):
        reasons.append("plain_language_buy_sell_legs_missing")
    return reasons


def _no_execution_blockers(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return text in {"", "nan", "none", "[]"}


def _has_plain_language_trade_legs(row: Mapping[str, Any] | pd.Series) -> bool:
    trade_plan = str(_mapping_get(row, "trade_plan") or "")
    sell_leg = str(_mapping_get(row, "sell_leg") or "")
    buy_leg = str(_mapping_get(row, "buy_leg") or "")
    combined = " ".join([trade_plan, sell_leg, buy_leg])
    if "BUY" not in combined.upper() or "SELL" not in combined.upper():
        return False
    if re.search(r"\b[A-Z]{1,6}\d{6}[CP]\d{8}\b", combined):
        return False
    if "<span" in combined.lower():
        return False
    return bool(sell_leg.strip()) and bool(buy_leg.strip())


def _has_plain_language_trade_plan(row: Mapping[str, Any] | pd.Series) -> bool:
    trade_plan = str(_mapping_get(row, "trade_plan") or "")
    combined = " ".join(
        [
            trade_plan,
            str(_mapping_get(row, "sell_leg") or ""),
            str(_mapping_get(row, "buy_leg") or ""),
        ]
    )
    if "BUY" not in combined.upper() or "SELL" not in combined.upper():
        return False
    if re.search(r"\b[A-Z]{1,6}\d{6}[CP]\d{8}\b", combined):
        return False
    if "<span" in combined.lower():
        return False
    return bool(trade_plan.strip())


def _invalid_green_ticket_examples(details: pd.DataFrame, *, limit: int = 10) -> str:
    if details.empty or "fail_reasons" not in details.columns:
        return ""
    bad = details[details["fail_reasons"].astype(str).str.strip().ne("")]
    examples: list[str] = []
    for _, row in bad.head(limit).iterrows():
        examples.append(f"{row.get('ticker', '')}: {row.get('fail_reasons', '')}")
    remaining = len(bad) - len(examples)
    if remaining > 0:
        examples.append(f"... (+{remaining} more)")
    return "; ".join(examples)


def _agent_reviews_json_from_manifest(manifest: Mapping[str, Any]) -> str:
    agentic = manifest.get("agentic_orchestration", {}) or {}
    return str(agentic.get("ingested_reviews_json") or agentic.get("expected_reviews_json") or "")


def _first_execution_agent_reviews_json(packet: pd.DataFrame) -> str:
    if packet.empty or "agent_reviews_json" not in packet.columns:
        return ""
    row = _select_execution_packet_row(packet)
    return str(_mapping_get(row, "agent_reviews_json") or "").strip()


def _split_command(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return str(command or "").split()


def _read_review_json(path: Optional[Path]) -> tuple[pd.DataFrame, list[str]]:
    if path is None:
        return pd.DataFrame(columns=["ticker", "agent"]), ["review path missing"]
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        return pd.DataFrame(columns=["ticker", "agent"]), [f"review file missing: {resolved}"]
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return pd.DataFrame(columns=["ticker", "agent"]), [f"review file invalid: {exc}"]
    reviews = payload.get("reviews", payload) if isinstance(payload, Mapping) else payload
    if not isinstance(reviews, list):
        return pd.DataFrame(columns=["ticker", "agent"]), ["review JSON did not contain a review list"]
    rows: list[dict[str, Any]] = []
    for review in reviews:
        if not isinstance(review, Mapping):
            continue
        rows.append(
            {
                "ticker": str(review.get("ticker") or "").strip().upper(),
                "agent": str(review.get("agent") or "").strip(),
                "verdict": str(review.get("verdict") or "").strip().lower(),
                "confidence": str(review.get("confidence") or "").strip().lower(),
            }
        )
    return pd.DataFrame(rows), []


def _distinct_agents(reviews: pd.DataFrame) -> int:
    if reviews.empty or "agent" not in reviews.columns:
        return 0
    return len({str(agent).strip() for agent in reviews["agent"].dropna().tolist() if str(agent).strip()})


def _live_rerun_preflight_failed_examples(
    details: pd.DataFrame,
    proof_fail_reasons: Sequence[str],
    review_errors: Sequence[str],
    *,
    limit: int = 10,
) -> str:
    examples: list[str] = []
    if proof_fail_reasons:
        examples.append("proof: " + "; ".join(proof_fail_reasons))
    if review_errors:
        examples.append("review_json: " + "; ".join(str(error) for error in review_errors))
    if not details.empty and "fail_reasons" in details.columns:
        failed = details[details["fail_reasons"].astype(str).str.strip().ne("")]
        for _, row in failed.head(limit).iterrows():
            examples.append(f"{row.get('ticker', '')}: {row.get('fail_reasons', '')}")
    return " | ".join(examples)


def _live_rerun_preflight_required_gate() -> str:
    return (
        "rerun command includes --live-schwab, --live-portfolio, and --agent-reviews-json; "
        "rerun output directory is clear; source date UW files exist; "
        f"each queued ticker has at least {MIN_AGENTIC_REVIEW_LANES_PER_TICKER} distinct agent-review lanes in the JSON"
    )


def _queue_rows_for_source(market_open_recheck_queue: pd.DataFrame, source_dir: str) -> int:
    if market_open_recheck_queue.empty or "source_dir" not in market_open_recheck_queue.columns:
        return 0
    return int(market_open_recheck_queue["source_dir"].astype(str).eq(source_dir).sum())


def _execution_packet_status(
    *,
    fresh_live: bool,
    portfolio_ready: bool,
    agentic_ready: bool,
    market_open: bool,
    green_ready_orders: int,
    queue_rows: int,
) -> str:
    if green_ready_orders > 0 and market_open:
        return "green_orders_present_verify_ticket_scoped_expectancy"
    if fresh_live and portfolio_ready and agentic_ready and not market_open and queue_rows > 0:
        return "ready_for_regular_session_rerun"
    if not (fresh_live and portfolio_ready and agentic_ready):
        return "refresh_live_probe_inputs_before_rerun"
    if market_open and queue_rows <= 0:
        return "market_open_live_probe_no_green_orders"
    if queue_rows <= 0:
        return "no_market_session_only_yellow_queue"
    return "rerun_required"


def _select_execution_packet_row(packet: Optional[pd.DataFrame]) -> Mapping[str, Any]:
    if packet is None or packet.empty:
        return {}
    priority = {
        "green_orders_present_verify_ticket_scoped_expectancy": 0,
        "ready_for_regular_session_rerun": 1,
        "market_open_live_probe_no_green_orders": 2,
        "no_market_session_only_yellow_queue": 3,
        "rerun_required": 4,
        "refresh_live_probe_inputs_before_rerun": 5,
    }
    frame = packet.copy()
    frame["_status_priority"] = frame.get("status", pd.Series(dtype=object)).astype(str).map(
        lambda status: priority.get(status, 99)
    )
    if "market_session_open" in frame.columns:
        frame["_market_open_priority"] = frame["market_session_open"].map(_truthy).map(lambda value: 0 if value else 1)
    else:
        frame["_market_open_priority"] = 1
    if "date" in frame.columns:
        frame["_date_priority"] = frame["date"].astype(str)
        ordered = frame.sort_values(["_status_priority", "_market_open_priority", "_date_priority"], ascending=[True, True, False])
    else:
        ordered = frame.sort_values(["_status_priority", "_market_open_priority"], ascending=[True, True])
    return ordered.iloc[0]


def _select_verification_plan_row(plan: pd.DataFrame) -> Mapping[str, Any]:
    if plan.empty:
        return {}
    priority = {
        "VERIFY_GREEN_ORDERS_AND_EXPECTANCY": 0,
        "NO_GREEN_TICKETS_AFTER_MARKET_OPEN_PROBE": 1,
        "WAITING_FOR_REGULAR_SESSION": 2,
        "NO_EXECUTION_EVIDENCE_READY": 3,
        "REFRESH_INPUTS_BEFORE_SESSION_RUN": 4,
    }
    frame = plan.copy()
    frame["_status_priority"] = frame.get("status", pd.Series(dtype=object)).astype(str).map(
        lambda status: priority.get(status, 99)
    )
    if "date" in frame.columns:
        frame["_date_priority"] = frame["date"].astype(str)
        ordered = frame.sort_values(["_status_priority", "_date_priority"], ascending=[True, False])
    else:
        ordered = frame.sort_values("_status_priority")
    return ordered.iloc[0]


def _has_completed_market_open_probe_without_queue(packet: Optional[pd.DataFrame]) -> bool:
    row = _select_execution_packet_row(packet)
    if len(row) == 0:
        return False
    status = str(_mapping_get(row, "status") or "")
    return (
        status in {
            "market_open_live_probe_no_green_orders",
            "no_market_session_only_yellow_queue",
            "green_orders_present_verify_ticket_scoped_expectancy",
        }
        and _truthy(_mapping_get(row, "fresh_live_quotes_ready"))
        and _truthy(_mapping_get(row, "portfolio_ready"))
        and _truthy(_mapping_get(row, "agentic_reviews_ready"))
        and _truthy(_mapping_get(row, "market_session_open"))
        and int(_to_float(_mapping_get(row, "yellow_recheck_rows"))) == 0
    )


def _execution_packet_green_ready_orders(packet: Optional[pd.DataFrame]) -> int:
    row = _select_execution_packet_row(packet)
    if len(row) == 0:
        return 0
    return int(_to_float(_mapping_get(row, "green_ready_orders")))


def _market_open_recheck_fail_reasons(row: Mapping[str, Any] | pd.Series) -> list[str]:
    reasons: list[str] = []
    if str(_mapping_get(row, "target_order_status") or "").strip().lower() != "target_order_candidate":
        reasons.append("target_order_status_not_candidate")
    if not _market_open_recheck_order_readiness_pass(row):
        reasons.append("order_readiness_not_market_open_recheck")
    if not _market_open_recheck_blockers_pass(row):
        reasons.append("blockers_not_only_market_session_or_fresh_live_required")
    if str(_mapping_get(row, "entry_type") or "").strip().upper() not in {"CREDIT", "DEBIT"}:
        reasons.append("entry_type_missing")
    if _to_float(_mapping_get(row, "entry_limit")) <= 0:
        reasons.append("entry_limit_not_positive")
    if _to_float(_mapping_get(row, "suggested_contracts")) <= 0:
        reasons.append("suggested_contracts_not_positive")
    if _to_float(_mapping_get(row, "execution_confidence_score")) < MIN_EXECUTION_CONFIDENCE_SCORE:
        reasons.append("execution_confidence_score_below_threshold")
    if str(_mapping_get(row, "trade_quality_confidence_rating") or "").strip().upper() not in {"MEDIUM", "HIGH"}:
        reasons.append("trade_quality_confidence_rating_not_MEDIUM_or_HIGH")
    if _to_float(_mapping_get(row, "external_agent_distinct_review_count")) < MIN_AGENTIC_REVIEW_LANES_PER_TICKER:
        reasons.append("external_agent_distinct_review_count_below_minimum")
    if not _has_plain_language_trade_plan(row):
        reasons.append("plain_language_buy_sell_legs_missing")
    return reasons


def _market_open_recheck_order_readiness_pass(row: Mapping[str, Any] | pd.Series) -> bool:
    return str(_mapping_get(row, "order_readiness") or "").strip() in {
        "target_order_after_market_open_and_live_recheck",
        "target_order_after_live_recheck",
    }


def _market_open_recheck_blockers_pass(row: Mapping[str, Any] | pd.Series) -> bool:
    blockers = _split_blockers(_mapping_get(row, "execution_blockers"))
    return bool(blockers) and blockers.issubset({"market_session_open_required", "fresh_live_schwab_required"})


def _split_blockers(value: Any) -> set[str]:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none", "[]"}:
        return set()
    return {
        part.strip()
        for part in re.split(r"[;,]", text)
        if part.strip()
    }


def _market_open_recheck_required_gate() -> str:
    return (
        "target_order_status=target_order_candidate; "
        "order_readiness=target_order_after_market_open_and_live_recheck or target_order_after_live_recheck; "
        "execution_blockers only market_session_open_required/fresh_live_schwab_required; "
        "entry_type CREDIT or DEBIT; positive entry_limit and suggested_contracts; "
        f"execution_confidence_score >= {MIN_EXECUTION_CONFIDENCE_SCORE}; "
        "MEDIUM/HIGH trade-quality confidence; "
        f"external_agent_distinct_review_count >= {MIN_AGENTIC_REVIEW_LANES_PER_TICKER}; "
        "plain-language BUY/SELL legs"
    )


def _market_open_recheck_failed_examples(details: pd.DataFrame, *, limit: int = 10) -> str:
    if details.empty or "fail_reasons" not in details.columns:
        return ""
    failed = details[details["fail_reasons"].astype(str).str.strip().ne("")]
    examples: list[str] = []
    for _, row in failed.head(limit).iterrows():
        examples.append(f"{row.get('ticker', '')}: {row.get('fail_reasons', '')}")
    remaining = len(failed) - len(examples)
    if remaining > 0:
        examples.append(f"... (+{remaining} more)")
    return "; ".join(examples)


def _position_value_sum(frame: pd.DataFrame, one_lot_column: str, position_column: str) -> float:
    if frame.empty:
        return 0.0
    one_lot = pd.to_numeric(
        frame.get(one_lot_column, pd.Series(0.0, index=frame.index)),
        errors="coerce",
    ).fillna(0.0)
    contracts = pd.to_numeric(
        frame.get("suggested_contracts", pd.Series(1, index=frame.index)),
        errors="coerce",
    ).fillna(1.0)
    fallback = one_lot * contracts.clip(lower=0)
    if position_column not in frame.columns:
        return float(fallback.sum())
    position = pd.to_numeric(frame[position_column], errors="coerce")
    return float(position.where(position.gt(0), fallback).fillna(fallback).sum())


def _next_live_rerun_dir(run_dir: Path, day: str) -> Path:
    parent = run_dir.expanduser().resolve().parent
    max_version = 0
    for child in parent.iterdir() if parent.exists() else []:
        if not child.is_dir():
            continue
        match = re.match(r"live_readiness_probe_v(\d+).*20\d{2}-\d{2}-\d{2}", child.name)
        if match:
            max_version = max(max_version, int(match.group(1)))
    version = max_version + 1 if max_version else 1
    return parent / f"live_readiness_probe_v{version:03d}_market_open_rerun_{day}"


def _market_open_rerun_command(
    *,
    day: str,
    base_dir: Path,
    out_dir: Path,
    agent_reviews_json: str,
) -> str:
    parts = [
        "python3",
        "-m",
        "uwos.options_agent",
        "--date",
        day,
        "--base-dir",
        str(base_dir),
        "--out-dir",
        str(out_dir),
        "--live-schwab",
        "--live-portfolio",
    ]
    if agent_reviews_json:
        parts.extend(["--agent-reviews-json", agent_reviews_json])
    return " ".join(shlex.quote(part) for part in parts)


def build_target_preservation_audit(
    summary: pd.DataFrame,
    tickets: pd.DataFrame,
    market_open_recheck_queue: pd.DataFrame,
) -> pd.DataFrame:
    dates = sorted(summary["date"].astype(str).unique().tolist()) if not summary.empty else []
    target_entry_types = pd.concat(
        [
            tickets.get("entry_type", pd.Series(dtype=object)),
            market_open_recheck_queue.get("entry_type", pd.Series(dtype=object)),
        ],
        ignore_index=True,
    )
    entry_counts = target_entry_types.astype(str).str.upper().value_counts()
    live_queue_rows = int((market_open_recheck_queue.get("source_kind", pd.Series(dtype=object)).astype(str) == "live_probe").sum())
    rows = [
        {
            "metric": "validation_dates",
            "value": len(dates),
            "status": "PROVEN" if len(dates) >= 2 else "MISSING",
            "evidence": ", ".join(dates),
        },
        {
            "metric": "target_ticket_rows",
            "value": len(tickets),
            "status": "PROVEN" if len(tickets) > 0 else "MISSING",
            "evidence": "Visible yellow target-order tickets across refreshed multi-date validation.",
        },
        {
            "metric": "green_ready_orders",
            "value": _truthy_count(tickets.get("ready_to_enter")),
            "status": "EXPECTED_BLOCK" if _truthy_count(tickets.get("ready_to_enter")) == 0 else "PRESENT",
            "evidence": "Replay/snapshot validation must not create send-now orders.",
        },
        {
            "metric": "yellow_target_candidates",
            "value": _target_count(tickets),
            "status": "PROVEN" if _target_count(tickets) > 0 else "MISSING",
            "evidence": "Target-order candidates preserve desired credit/debit math with ready_to_enter=false.",
        },
        {
            "metric": "credit_target_rows",
            "value": int(entry_counts.get("CREDIT", 0)),
            "status": "PROVEN" if int(entry_counts.get("CREDIT", 0)) > 0 else "MISSING",
            "evidence": "Credit targets present.",
        },
        {
            "metric": "debit_target_rows",
            "value": int(entry_counts.get("DEBIT", 0)),
            "status": "PROVEN" if int(entry_counts.get("DEBIT", 0)) > 0 else "MISSING",
            "evidence": "Debit targets present.",
        },
        {
            "metric": "live_market_open_recheck_rows",
            "value": live_queue_rows,
            "status": "PROVEN" if live_queue_rows > 0 else "NOT_OBSERVED",
            "evidence": "Live probe rows that are blocked only by regular-market session timing.",
        },
    ]
    return pd.DataFrame(rows, columns=TARGET_PRESERVATION_COLUMNS)


def build_goal_completion_audit(
    *,
    summary: pd.DataFrame,
    tickets: pd.DataFrame,
    focus_coverage: pd.DataFrame,
    ticket_review_lanes: pd.DataFrame,
    agentic_coverage_proof: pd.DataFrame,
    validation_coverage_proof: pd.DataFrame,
    cutoff_visibility_proof: pd.DataFrame,
    live_spread_quality_proof: pd.DataFrame,
    underlying_quality_proof: pd.DataFrame,
    major_name_coverage_proof: pd.DataFrame,
    expectancy: pd.DataFrame,
    market_open_recheck_queue: pd.DataFrame,
    market_open_recheck_proof: pd.DataFrame,
    live_rerun_preflight_proof: pd.DataFrame,
    live_probe_summary: pd.DataFrame,
    multi_date_readiness_proof: pd.DataFrame,
    actionability_surface_proof: pd.DataFrame,
    action_surface_underlying_quality_proof: pd.DataFrame,
    green_ticket_execution_proof: pd.DataFrame,
    ticket_expectancy_proof: pd.DataFrame,
    paths: Mapping[str, Path],
) -> pd.DataFrame:
    dates = sorted(summary["date"].astype(str).unique().tolist()) if not summary.empty else []
    agentic_status = str(_first_value(agentic_coverage_proof, "status"))
    agentic_dates = str(_first_value(agentic_coverage_proof, "agentic_ready_dates"))
    agentic_ticket_rows = int(_to_float(_first_value(agentic_coverage_proof, "ticket_rows")))
    agentic_covered_rows = int(_to_float(_first_value(agentic_coverage_proof, "ticket_rows_with_agentic_ready")))
    agentic_uncovered_rows = int(_to_float(_first_value(agentic_coverage_proof, "ticket_rows_without_agentic_ready")))
    non_agentic_ticket_dates = str(_first_value(agentic_coverage_proof, "non_agentic_ticket_dates"))
    validation_status = str(_first_value(validation_coverage_proof, "status"))
    window_available_count = int(_to_float(_first_value(validation_coverage_proof, "window_available_source_date_count")))
    untested_available_count = int(_to_float(_first_value(validation_coverage_proof, "untested_available_date_count")))
    base_available_count = int(_to_float(_first_value(validation_coverage_proof, "base_available_source_date_count")))
    outside_window_count = int(_to_float(_first_value(validation_coverage_proof, "available_dates_outside_window_count")))
    cutoff_status = str(_first_value(cutoff_visibility_proof, "status"))
    live_spread_quality_status = str(_first_value(live_spread_quality_proof, "status"))
    live_spread_quality_audited_rows = int(_to_float(_first_value(live_spread_quality_proof, "audited_rows")))
    live_spread_quality_block_rows = int(_to_float(_first_value(live_spread_quality_proof, "block_rows")))
    live_spread_quality_blocked_still_actionable = int(_to_float(_first_value(live_spread_quality_proof, "blocked_still_actionable_rows")))
    live_spread_quality_target_block_rows = int(_to_float(_first_value(live_spread_quality_proof, "target_candidate_block_rows")))
    quality_status = str(_first_value(underlying_quality_proof, "status"))
    non_core_or_liquid = int(_to_float(_first_value(underlying_quality_proof, "not_core_or_liquid_ticket_rows")))
    liquid_non_core = int(_to_float(_first_value(underlying_quality_proof, "liquid_non_core_ticket_rows")))
    action_surface_quality_status = str(_first_value(action_surface_underlying_quality_proof, "status"))
    action_surface_bad_rows = (
        int(_to_float(_first_value(action_surface_underlying_quality_proof, "ticket_bad_underlying_rows")))
        + int(_to_float(_first_value(action_surface_underlying_quality_proof, "market_open_recheck_bad_underlying_rows")))
        + int(_to_float(_first_value(action_surface_underlying_quality_proof, "focus_bad_actionable_rows")))
    )
    major_status = str(_first_value(major_name_coverage_proof, "status"))
    major_required = int(_to_float(_first_value(major_name_coverage_proof, "required_ticker_count")))
    major_covered = int(_to_float(_first_value(major_name_coverage_proof, "covered_required_ticker_count")))
    major_missing = str(_first_value(major_name_coverage_proof, "missing_required_tickers"))
    major_rows_missing_reason = int(_to_float(_first_value(major_name_coverage_proof, "required_rows_missing_reason")))
    target_entry_types = pd.concat(
        [
            tickets.get("entry_type", pd.Series(dtype=object)),
            market_open_recheck_queue.get("entry_type", pd.Series(dtype=object)),
        ],
        ignore_index=True,
    )
    entry_types = set(target_entry_types.astype(str).str.upper())
    green_ready = _truthy_count(tickets.get("ready_to_enter"))
    yellow_targets = _target_count(tickets)
    live_queue_rows = int((market_open_recheck_queue.get("source_kind", pd.Series(dtype=object)).astype(str) == "live_probe").sum())
    market_open_recheck_status = str(_first_value(market_open_recheck_proof, "status"))
    market_open_recheck_row_failures = int(_to_float(_first_value(market_open_recheck_proof, "row_fail_rows")))
    market_open_recheck_queue_rows = int(_to_float(_first_value(market_open_recheck_proof, "queue_rows")))
    market_open_recheck_proven = market_open_recheck_status in {
        "PASS_LIVE_MARKET_OPEN_RECHECK_QUEUE_READY",
        "PASS_DATED_MARKET_OPEN_RECHECK_QUEUE_READY",
        "PASS_NO_MARKET_OPEN_RECHECK_QUEUE_AFTER_MARKET_OPEN_LIVE_PROBE",
    }
    live_rerun_preflight_status = str(_first_value(live_rerun_preflight_proof, "status"))
    live_rerun_preflight_proven = live_rerun_preflight_status in {
        "PASS_LIVE_RERUN_PREFLIGHT_READY",
        "PASS_NO_LIVE_RERUN_QUEUE_AFTER_MARKET_OPEN_LIVE_PROBE",
    }
    monthly_statuses = sorted(set(summary.get("monthly_feasibility", pd.Series(dtype=object)).astype(str)))
    expectancy_statuses = sorted(set(summary.get("expectancy_summary_status", pd.Series(dtype=object)).astype(str)))
    live_green = int(live_probe_summary.get("green_ready_orders", pd.Series(dtype=object)).map(_to_float).sum()) if not live_probe_summary.empty else 0
    live_market_open = bool(live_probe_summary.get("market_session_open", pd.Series(dtype=object)).map(_truthy).any()) if not live_probe_summary.empty else False
    live_expectancy = sorted(set(live_probe_summary.get("expectancy_summary_status", pd.Series(dtype=object)).astype(str)))
    green_ticket_status = str(_first_value(green_ticket_execution_proof, "status"))
    green_ticket_rows = int(_to_float(_first_value(green_ticket_execution_proof, "green_ticket_rows")))
    valid_green_rows = int(_to_float(_first_value(green_ticket_execution_proof, "valid_green_ticket_rows")))
    invalid_green_rows = int(_to_float(_first_value(green_ticket_execution_proof, "invalid_green_ticket_rows")))
    ticket_expectancy_status = str(_first_value(ticket_expectancy_proof, "status"))
    multi_date_readiness_status = str(_first_value(multi_date_readiness_proof, "status"))
    multi_date_readiness_proven = multi_date_readiness_status in {
        "PASS_MULTI_DATE_TARGETS_WAITING_FOR_REGULAR_SESSION_LIVE_GREEN",
        "PASS_MULTI_DATE_WITH_MARKET_OPEN_LIVE_PROBE_NO_GREEN_TICKETS",
        "PASS_MULTI_DATE_AND_LIVE_GREEN_EVIDENCE",
    }
    actionability_status = str(_first_value(actionability_surface_proof, "status"))
    actionability_separated = actionability_status in {
        "PASS_YELLOW_ONLY_SURFACE_SEPARATED",
        "PASS_GREEN_AND_YELLOW_SURFACES_SEPARATED",
        "PASS_NO_SEND_NOW_OR_TARGET_SURFACE",
    }
    monthly_claim_status, monthly_claim_gap = _monthly_claim_requirement_status(
        ticket_expectancy_status=ticket_expectancy_status,
        monthly_statuses=monthly_statuses,
        expectancy_statuses=expectancy_statuses,
        live_expectancy=live_expectancy,
        live_green_ready_orders=live_green,
        green_ticket_status=green_ticket_status,
    )
    rows = [
        {
            "requirement": "validate across multiple available UW dates",
            "status": "PROVEN" if len(dates) >= 2 and untested_available_count == 0 else "NOT_ACHIEVED",
            "evidence": (
                f"{len(dates)} dated runs; validation_status={validation_status}; "
                f"window_available_source_dates={window_available_count}; untested_in_window={untested_available_count}; "
                f"base_available_source_dates={base_available_count}; outside_window={outside_window_count}; dates={', '.join(dates)}"
            ),
            "artifact": str(paths["validation_coverage_proof_packet"]),
            "remaining_gap": "" if len(dates) >= 2 and untested_available_count == 0 else "Add missing available source dates to this validation window.",
        },
        {
            "requirement": "prove latest live probe is not the whole validation",
            "status": "PROVEN" if multi_date_readiness_proven else "NOT_ACHIEVED",
            "evidence": (
                f"multi_date_readiness_status={multi_date_readiness_status}; "
                f"validation_dates={_first_value(multi_date_readiness_proof, 'validation_date_count')}; "
                f"latest_live_probe_date={_first_value(multi_date_readiness_proof, 'latest_live_probe_date')}; "
                f"live_probe_dates={_first_value(multi_date_readiness_proof, 'live_probe_dates')}; "
                f"dated_yellow_target_candidates={_first_value(multi_date_readiness_proof, 'dated_yellow_target_candidates')}; "
                f"live_yellow_recheck_rows={_first_value(multi_date_readiness_proof, 'live_yellow_recheck_rows')}; "
                f"live_green_ready_orders={_first_value(multi_date_readiness_proof, 'live_green_ready_orders')}"
            ),
            "artifact": str(paths["multi_date_readiness_proof_packet"]),
            "remaining_gap": "" if multi_date_readiness_proven else "Add broad dated validation plus at least one live probe or explain the missing execution path.",
        },
        {
            "requirement": "avoid artificial trade-count cutoffs",
            "status": "PROVEN" if cutoff_status == "PASS_NO_ARTIFICIAL_CUTOFFS" else "NOT_ACHIEVED",
            "evidence": (
                f"cutoff_visibility_status={cutoff_status}; "
                f"candidate_rows={_first_value(cutoff_visibility_proof, 'candidate_rows')}; "
                f"research_task_rows={_first_value(cutoff_visibility_proof, 'research_task_rows')}; "
                f"qualified_candidate_rows={_first_value(cutoff_visibility_proof, 'qualified_candidate_rows')}; "
                f"priced_candidate_rows={_first_value(cutoff_visibility_proof, 'priced_candidate_rows')}; "
                f"expected_no_trade_rows={_first_value(cutoff_visibility_proof, 'expected_no_trade_rows')}; "
                f"no_trade_audit_rows={_first_value(cutoff_visibility_proof, 'no_trade_audit_rows')}"
            ),
            "artifact": str(paths["cutoff_visibility_proof_packet"]),
            "remaining_gap": (
                "" if cutoff_status == "PASS_NO_ARTIFICIAL_CUTOFFS"
                else f"Refresh stale/capped run artifacts: {_first_value(cutoff_visibility_proof, 'problem_runs')}"
            ),
        },
        {
            "requirement": "use multi-agent orchestration evidence",
            "status": "PROVEN" if agentic_status == "PASS_FULL_AGENTIC_TICKET_COVERAGE" else "NOT_ACHIEVED",
            "evidence": (
                f"agentic_status={agentic_status}; agentic_dates={agentic_dates}; "
                f"ticket_rows={agentic_ticket_rows}; ticket_rows_with_agentic_ready={agentic_covered_rows}; "
                f"ticket_rows_without_agentic_ready={agentic_uncovered_rows}"
            ),
            "artifact": str(paths["agentic_coverage_proof_packet"]),
            "remaining_gap": (
                "" if agentic_status == "PASS_FULL_AGENTIC_TICKET_COVERAGE"
                else f"Ticket rows from non-agentic or awaiting-subagent dates remain: {non_agentic_ticket_dates}"
            ),
        },
        {
            "requirement": "prioritize liquid large-cap/index/high-volume names over junk",
            "status": "PROVEN" if len(tickets) > 0 and non_core_or_liquid == 0 and action_surface_bad_rows == 0 else "NOT_ACHIEVED",
            "evidence": (
                f"quality_status={quality_status}; target tickets={len(tickets)}; "
                f"not_core_or_liquid_ticket_rows={non_core_or_liquid}; liquid_non_core_ticket_rows={liquid_non_core}; "
                f"action_surface_quality_status={action_surface_quality_status}; action_surface_bad_rows={action_surface_bad_rows}"
            ),
            "artifact": str(paths["action_surface_underlying_quality_proof_packet"]),
            "remaining_gap": (
                "" if non_core_or_liquid == 0 and action_surface_bad_rows == 0 else "Speculative, excluded, unknown, or otherwise non-actionable rows still reached an action-looking surface."
            ),
        },
        {
            "requirement": "block bad live spread markets from actionable surfaces",
            "status": "PROVEN" if live_spread_quality_status == "PASS_LIVE_SPREAD_QUALITY_GATED" else "NOT_ACHIEVED",
            "evidence": (
                f"live_spread_quality_status={live_spread_quality_status}; "
                f"audited_rows={live_spread_quality_audited_rows}; block_rows={live_spread_quality_block_rows}; "
                f"blocked_still_actionable_rows={live_spread_quality_blocked_still_actionable}; "
                f"target_candidate_block_rows={live_spread_quality_target_block_rows}; "
                f"blocked_tickers={_first_value(live_spread_quality_proof, 'blocked_tickers')}"
            ),
            "artifact": str(paths["live_spread_quality_proof_packet"]),
            "remaining_gap": (
                "" if live_spread_quality_status == "PASS_LIVE_SPREAD_QUALITY_GATED"
                else "Bad live/snapshot spread markets still lack proof that they were blocked before yellow/green action surfaces."
            ),
        },
        {
            "requirement": "preserve actionable target credit/debit candidates when market is closed",
            "status": "PROVEN" if {"CREDIT", "DEBIT"}.issubset(entry_types) and yellow_targets > 0 else "NOT_ACHIEVED",
            "evidence": f"entry_types={sorted(entry_types)}; yellow_targets={yellow_targets}; live_market_open_recheck_rows={live_queue_rows}",
            "artifact": str(paths["target_preservation_audit"]),
            "remaining_gap": "",
        },
        {
            "requirement": "prove market-open recheck queue is complete and only session-blocked",
            "status": "PROVEN" if market_open_recheck_proven else "NOT_ACHIEVED",
            "evidence": (
                f"market_open_recheck_status={market_open_recheck_status}; "
                f"queue_rows={market_open_recheck_queue_rows}; live_queue_rows={live_queue_rows}; "
                f"row_fail_rows={market_open_recheck_row_failures}; "
                f"credit_rows={_first_value(market_open_recheck_proof, 'credit_rows')}; "
                f"debit_rows={_first_value(market_open_recheck_proof, 'debit_rows')}; "
                f"tickers={_first_value(market_open_recheck_proof, 'tickers')}"
            ),
            "artifact": str(paths["market_open_recheck_proof_packet"]),
            "remaining_gap": (
                "" if market_open_recheck_proven
                else "Market-open recheck rows need complete target price, size, confidence, agentic lanes, and plain-language legs with only market-session timing as blocker."
            ),
        },
        {
            "requirement": "prove live rerun preflight has queue-ticker agent reviews",
            "status": "PROVEN" if live_rerun_preflight_proven else "NOT_ACHIEVED",
            "evidence": (
                f"live_rerun_preflight_status={live_rerun_preflight_status}; "
                f"queue_tickers={_first_value(live_rerun_preflight_proof, 'queue_ticker_count')}; "
                f"covered_queue_tickers={_first_value(live_rerun_preflight_proof, 'covered_queue_ticker_count')}; "
                f"missing_queue_tickers={_first_value(live_rerun_preflight_proof, 'missing_queue_tickers')}; "
                f"rerun_out_dir_clear={_first_value(live_rerun_preflight_proof, 'rerun_out_dir_clear')}; "
                f"agent_reviews_json={_first_value(live_rerun_preflight_proof, 'agent_reviews_json')}"
            ),
            "artifact": str(paths["live_rerun_preflight_proof_packet"]),
            "remaining_gap": (
                "" if live_rerun_preflight_proven
                else "Live rerun needs fresh output directory, source files, live flags, and queue-ticker agent-review coverage before the regular-session run."
            ),
        },
        {
            "requirement": "separate yellow target orders from green send-now orders",
            "status": "PROVEN" if actionability_separated else "NOT_ACHIEVED",
            "evidence": (
                f"actionability_surface_status={actionability_status}; dated_green_ready_orders={green_ready}; "
                f"live_green_ready_orders={live_green}; green_ticket_rows={green_ticket_rows}; "
                f"yellow_target_candidates={yellow_targets}; "
                f"target_ready_to_enter_rows={_first_value(actionability_surface_proof, 'target_ready_to_enter_rows')}; "
                f"target_missing_entry_type_rows={_first_value(actionability_surface_proof, 'target_missing_entry_type_rows')}; "
                f"target_missing_plain_language_leg_rows={_first_value(actionability_surface_proof, 'target_missing_plain_language_leg_rows')}; "
                f"live_market_open_recheck_rows={live_queue_rows}"
            ),
            "artifact": str(paths["actionability_surface_proof_packet"]),
            "remaining_gap": "" if actionability_separated else "Fix the user-facing ticket surface so yellow rows cannot be mistaken for green executable orders.",
        },
        {
            "requirement": "explain major-name inclusion/exclusion",
            "status": "PROVEN" if major_status == "PASS_ALL_MAJOR_NAMES_EXPLAINED" else "NOT_ACHIEVED",
            "evidence": (
                f"major_name_status={major_status}; required={major_required}; covered={major_covered}; "
                f"missing={major_missing or 'none'}; rows_missing_reason={major_rows_missing_reason}"
            ),
            "artifact": str(paths["major_name_coverage_proof_packet"]),
            "remaining_gap": (
                "" if major_status == "PASS_ALL_MAJOR_NAMES_EXPLAINED"
                else "Every configured major/focus ticker must have a visible state and nonempty reason."
            ),
        },
        {
            "requirement": "do not claim $10k/month readiness without evidence",
            "status": monthly_claim_status,
            "evidence": (
                f"ticket_expectancy_status={ticket_expectancy_status}; monthly_feasibility={monthly_statuses}; "
                f"expectancy_summary_statuses={expectancy_statuses}; live_probe_expectancy={live_expectancy}; "
                f"live_green_ready_orders={live_green}; green_ticket_status={green_ticket_status}"
            ),
            "artifact": str(paths["ticket_expectancy_proof_packet"]),
            "remaining_gap": monthly_claim_gap,
        },
        {
            "requirement": "be execution-ready trade quality confidence pipeline",
            "status": (
                "ACHIEVED"
                if green_ticket_status == "PASS_GREEN_TICKETS_EXECUTION_READY"
                and live_green > 0
                and ticket_expectancy_status == "PASS_GREEN_TICKER_EXPECTANCY_COVERAGE"
                else "NOT_ACHIEVED"
            ),
            "evidence": (
                f"green_ticket_status={green_ticket_status}; valid_green_ticket_rows={valid_green_rows}; "
                f"invalid_green_ticket_rows={invalid_green_rows}; multi_date_green_ready_orders={green_ready}; "
                f"live_probe_green_ready_orders={live_green}; "
                f"live_market_session_open={live_market_open}; live_probe_expectancy={live_expectancy}; "
                f"ticket_expectancy_status={ticket_expectancy_status}; expectancy_rows={len(expectancy)}"
            ),
            "artifact": str(paths["green_ticket_execution_proof_packet"]),
            "remaining_gap": _execution_readiness_remaining_gap(
                green_ticket_status=green_ticket_status,
                live_green_ready_orders=live_green,
                ticket_expectancy_status=ticket_expectancy_status,
                live_expectancy=live_expectancy,
            ),
        },
    ]
    return pd.DataFrame(rows, columns=GOAL_COMPLETION_COLUMNS)


def _execution_readiness_remaining_gap(
    *,
    green_ticket_status: str,
    live_green_ready_orders: int,
    ticket_expectancy_status: str,
    live_expectancy: Sequence[str],
) -> str:
    """Explain the exact unfinished readiness layer without asking for already-proven gates."""

    green_ready = str(green_ticket_status or "") == "PASS_GREEN_TICKETS_EXECUTION_READY" and live_green_ready_orders > 0
    ticket_expectancy_pass = str(ticket_expectancy_status or "") == "PASS_GREEN_TICKER_EXPECTANCY_COVERAGE"
    live_expectancy_pass = "PASS" in {str(value).strip().upper() for value in live_expectancy}
    if green_ready and ticket_expectancy_pass and live_expectancy_pass:
        return ""
    if green_ready and ticket_expectancy_pass:
        return ""
    if green_ready:
        return (
            "Order-entry readiness is proven for the live green ticket. Remaining gap: positive structure-aligned "
            "actual/forward expectancy evidence and monthly-capacity proof; do not request another live run merely "
            "to clear this evidence gate."
        )
    return (
        "Need a regular-market-hours live run with green ready_to_enter=true tickets, then positive structure-aligned "
        "actual/forward expectancy evidence before any monthly-readiness claim."
    )


def build_completion_verdict(
    *,
    goal_audit: pd.DataFrame,
    market_open_execution_packet: pd.DataFrame,
    expectancy_proof_packet: pd.DataFrame,
    ticket_expectancy_proof_packet: pd.DataFrame,
) -> pd.DataFrame:
    """Return the strict single-row verdict for whether the active goal can be closed."""

    complete_statuses = {"PROVEN", "ACHIEVED"}
    if goal_audit.empty or "status" not in goal_audit.columns:
        blocking_requirements = ["goal_completion_audit_missing"]
        proven_requirements = 0
        all_goal_rows_complete = False
    else:
        status_text = goal_audit["status"].astype(str)
        all_goal_rows_complete = bool(status_text.isin(complete_statuses).all())
        proven_requirements = int(status_text.isin(complete_statuses).sum())
        blocking_requirements = goal_audit.loc[
            ~status_text.isin(complete_statuses),
            "requirement",
        ].astype(str).tolist()

    market_row = _select_execution_packet_row(market_open_execution_packet)
    market_status = str(_mapping_get(market_row, "status") or "")
    next_session = _mapping_get(market_row, "next_regular_session_start") or ""
    monthly_claim_allowed = _truthy(_first_value(expectancy_proof_packet, "monthly_claim_allowed"))
    expectancy_status = _first_value(expectancy_proof_packet, "status")
    ticket_expectancy_status = _first_value(ticket_expectancy_proof_packet, "status")
    green_market_status = market_status == "green_orders_present_verify_ticket_scoped_expectancy"
    ticket_expectancy_pass = ticket_expectancy_status == "PASS_GREEN_TICKER_EXPECTANCY_COVERAGE"
    can_complete = bool(all_goal_rows_complete and green_market_status and ticket_expectancy_pass)
    status = "COMPLETE" if can_complete else "ACTIVE_NOT_COMPLETE"
    if can_complete and monthly_claim_allowed:
        note = "All completion gates are proven; update_goal may be called."
    elif can_complete:
        note = (
            "All execution-readiness gates are proven; update_goal may be called. "
            "Do not claim $10k/month readiness because monthly_claim_allowed=false."
        )
    else:
        note = "Do not call update_goal complete; keep working until the blocking requirements and proof packets clear."
    return pd.DataFrame(
        [
            {
                "can_mark_goal_complete": can_complete,
                "status": status,
                "proven_requirements": proven_requirements,
                "blocking_requirements": "; ".join(blocking_requirements),
                "market_open_packet_status": market_status,
                "next_regular_session_start": next_session,
                "monthly_claim_allowed": monthly_claim_allowed,
                "expectancy_packet_status": expectancy_status,
                "ticket_expectancy_packet_status": ticket_expectancy_status,
                "update_goal_action": "call_update_goal_complete" if can_complete else "do_not_call_update_goal_complete",
                "note": note,
            }
        ],
        columns=COMPLETION_VERDICT_COLUMNS,
    )


def build_readiness_dashboard(
    *,
    goal_audit: pd.DataFrame,
    completion_verdict: pd.DataFrame,
    post_rerun_verification: pd.DataFrame,
    session_only_shadow_proof: pd.DataFrame,
    action_surface_underlying_quality_proof: pd.DataFrame,
    monthly_feasibility_guardrail_proof: pd.DataFrame,
    paths: Mapping[str, Path],
) -> pd.DataFrame:
    """Build a compact index of the current goal-readiness evidence."""

    completion_status = str(_first_value(completion_verdict, "status")) or "UNKNOWN"
    can_complete = _truthy(_first_value(completion_verdict, "can_mark_goal_complete"))
    blocking = str(_first_value(completion_verdict, "blocking_requirements"))
    post_status = str(_first_value(post_rerun_verification, "status"))
    post_next_action = str(_first_value(post_rerun_verification, "required_next_action"))
    rows = [
        {
            "area": "overall_completion",
            "status": "COMPLETE" if can_complete else completion_status,
            "evidence": f"can_mark_goal_complete={can_complete}; blocking_requirements={blocking or 'None'}",
            "artifact": str(paths["completion_verdict_md"]),
            "required_next_action": "Call update_goal complete." if can_complete else post_next_action or "Continue until all proof packets pass.",
        },
        _dashboard_goal_row(
            goal_audit,
            "validate across multiple available UW dates",
            "multi_date_validation",
        ),
        _dashboard_goal_row(
            goal_audit,
            "prove latest live probe is not the whole validation",
            "latest_live_probe_scope",
        ),
        _dashboard_goal_row(
            goal_audit,
            "avoid artificial trade-count cutoffs",
            "cutoff_visibility",
        ),
        _dashboard_goal_row(
            goal_audit,
            "prioritize liquid large-cap/index/high-volume names over junk",
            "underlying_quality",
        ),
        _dashboard_goal_row(
            goal_audit,
            "block bad live spread markets from actionable surfaces",
            "live_spread_quality",
        ),
        _dashboard_goal_row(
            goal_audit,
            "preserve actionable target credit/debit candidates when market is closed",
            "target_credit_debit_preservation",
        ),
        _dashboard_goal_row(
            goal_audit,
            "prove market-open recheck queue is complete and only session-blocked",
            "market_open_recheck_quality",
        ),
        _dashboard_goal_row(
            goal_audit,
            "prove live rerun preflight has queue-ticker agent reviews",
            "live_rerun_preflight",
        ),
        _dashboard_goal_row(
            goal_audit,
            "separate yellow target orders from green send-now orders",
            "green_yellow_actionability",
        ),
        _dashboard_goal_row(
            goal_audit,
            "explain major-name inclusion/exclusion",
            "major_name_coverage",
        ),
        _dashboard_goal_row(
            goal_audit,
            "do not claim $10k/month readiness without evidence",
            "monthly_claim_guardrail",
        ),
        {
            "area": "monthly_feasibility_guardrail",
            "status": str(_first_value(monthly_feasibility_guardrail_proof, "status")) or "UNKNOWN",
            "evidence": (
                f"required_metric={_first_value(monthly_feasibility_guardrail_proof, 'required_metric')}; "
                f"runs_with_required_metric={_first_value(monthly_feasibility_guardrail_proof, 'runs_with_required_metric')}; "
                f"missing_required_metric_count={_first_value(monthly_feasibility_guardrail_proof, 'missing_required_metric_count')}; "
                f"pass_without_required_metric_runs={_first_value(monthly_feasibility_guardrail_proof, 'pass_without_required_metric_runs') or 'None'}"
            ),
            "artifact": str(paths["monthly_feasibility_guardrail_proof_packet_md"]),
            "required_next_action": str(_first_value(monthly_feasibility_guardrail_proof, "note")),
        },
        _dashboard_goal_row(
            goal_audit,
            "be execution-ready trade quality confidence pipeline",
            "execution_readiness",
        ),
        {
            "area": "session_only_green_shadow",
            "status": str(_first_value(session_only_shadow_proof, "status")) or "UNKNOWN",
            "evidence": (
                f"shadow_candidate_rows={_first_value(session_only_shadow_proof, 'shadow_candidate_rows')}; "
                f"position_max_profit={_first_value(session_only_shadow_proof, 'position_max_profit')}; "
                f"position_max_loss={_first_value(session_only_shadow_proof, 'position_max_loss')}; "
                f"tickers={_first_value(session_only_shadow_proof, 'tickers')}"
            ),
            "artifact": str(paths["session_only_green_shadow_proof_packet_md"]),
            "required_next_action": str(_first_value(session_only_shadow_proof, "required_next_action")),
        },
        {
            "area": "action_surface_underlying_quality",
            "status": str(_first_value(action_surface_underlying_quality_proof, "status")) or "UNKNOWN",
            "evidence": (
                f"ticket_bad_underlying_rows={_first_value(action_surface_underlying_quality_proof, 'ticket_bad_underlying_rows')}; "
                f"market_open_recheck_bad_underlying_rows={_first_value(action_surface_underlying_quality_proof, 'market_open_recheck_bad_underlying_rows')}; "
                f"focus_bad_actionable_rows={_first_value(action_surface_underlying_quality_proof, 'focus_bad_actionable_rows')}; "
                f"audit_only_focus_tickers={_first_value(action_surface_underlying_quality_proof, 'audit_only_focus_tickers')}"
            ),
            "artifact": str(paths["action_surface_underlying_quality_proof_packet_md"]),
            "required_next_action": "Keep excluded/speculative/unknown names red no-action unless policy changes and fresh validation proves otherwise.",
        },
        {
            "area": "post_rerun_go_no_go",
            "status": post_status or "UNKNOWN",
            "evidence": (
                f"green_ticket_status={_first_value(post_rerun_verification, 'green_ticket_status')}; "
                f"ticket_expectancy_status={_first_value(post_rerun_verification, 'ticket_expectancy_status')}; "
                f"update_goal_action={_first_value(post_rerun_verification, 'update_goal_action')}"
            ),
            "artifact": str(paths["post_rerun_verification_packet_md"]),
            "required_next_action": post_next_action,
        },
    ]
    for row in rows:
        row.setdefault("artifact", "")
        row.setdefault("required_next_action", "")
    if not can_complete:
        execution_gap = next(
            (
                str(row.get("required_next_action") or "").strip()
                for row in rows
                if row.get("area") == "execution_readiness" and str(row.get("required_next_action") or "").strip()
            ),
            "",
        )
        if execution_gap:
            rows[0]["required_next_action"] = execution_gap
    return pd.DataFrame(rows, columns=READINESS_DASHBOARD_COLUMNS)


def _dashboard_goal_row(goal_audit: pd.DataFrame, requirement: str, area: str) -> dict[str, Any]:
    if goal_audit.empty or "requirement" not in goal_audit.columns:
        return {
            "area": area,
            "status": "MISSING",
            "evidence": "goal_completion_audit is missing",
            "artifact": "",
            "required_next_action": "Regenerate the expanded audit.",
        }
    rows = goal_audit[goal_audit["requirement"].astype(str).eq(requirement)]
    if rows.empty:
        return {
            "area": area,
            "status": "MISSING",
            "evidence": f"requirement row not found: {requirement}",
            "artifact": "",
            "required_next_action": "Regenerate or update goal_completion_audit.",
        }
    row = rows.iloc[0]
    return {
        "area": area,
        "status": row.get("status", ""),
        "evidence": row.get("evidence", ""),
        "artifact": row.get("artifact", ""),
        "required_next_action": row.get("remaining_gap", ""),
    }


def _monthly_claim_requirement_status(
    *,
    ticket_expectancy_status: str,
    monthly_statuses: Sequence[str],
    expectancy_statuses: Sequence[str],
    live_expectancy: Sequence[str],
    live_green_ready_orders: int,
    green_ticket_status: str,
) -> tuple[str, str]:
    """Return goal-row status for the monthly target evidence guardrail."""

    normalized_monthly = {str(value).strip().lower() for value in monthly_statuses if str(value).strip()}
    normalized_expectancy = {str(value).strip().upper() for value in expectancy_statuses if str(value).strip()}
    normalized_live_expectancy = {str(value).strip().upper() for value in live_expectancy if str(value).strip()}
    ticket_status = str(ticket_expectancy_status or "").strip().upper()
    green_status = str(green_ticket_status or "").strip().upper()
    evidence_supports_claim = (
        ticket_status == "PASS_GREEN_TICKER_EXPECTANCY_COVERAGE"
        and "not_proven" not in normalized_monthly
        and "BLOCK" not in normalized_expectancy
        and "PASS" in normalized_live_expectancy
        and live_green_ready_orders > 0
        and green_status == "PASS_GREEN_TICKETS_EXECUTION_READY"
    )
    claim_is_blocked = (
        ticket_status.startswith("BLOCK")
        or "not_proven" in normalized_monthly
        or "BLOCK" in normalized_expectancy
        or live_green_ready_orders <= 0
    )
    if evidence_supports_claim or claim_is_blocked:
        return "PROVEN", ""
    return (
        "NEEDS_REVIEW",
        "Monthly target claim gate is neither cleanly blocked nor fully supported by live green tickets and structure-aligned ticket expectancy.",
    )


def _first_value(frame: pd.DataFrame, column: str) -> Any:
    if frame.empty or column not in frame.columns:
        return ""
    value = frame.iloc[0].get(column, "")
    return "" if pd.isna(value) else value


def _read_manifest(run_dir: Path) -> dict[str, Any]:
    manifests = sorted(run_dir.glob("options_agent_manifest_*.json"))
    if not manifests:
        return {}
    try:
        return json.loads(manifests[0].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _run_date(run_dir: Path, manifest: Mapping[str, Any]) -> str:
    if manifest.get("as_of"):
        return str(manifest["as_of"])
    match = re.search(r"20\d{2}-\d{2}-\d{2}", run_dir.name)
    return match.group(0) if match else ""


def _mode_from_manifest(manifest: Mapping[str, Any]) -> str:
    if manifest.get("live_schwab_requested"):
        return "live_schwab"
    if manifest.get("chain_snapshot_dir"):
        return "snapshot_replay"
    return "agentic_local_preview"


def _safe_read_csv(path: Path, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=list(columns or []))
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=list(columns or []))


def _research_task_count(run_dir: Path, row_counts: Mapping[str, Any]) -> int:
    value = row_counts.get("research_tasks", "")
    if str(value).strip():
        return int(_to_float(value))
    path = run_dir / "research_tasks.json"
    if not path.exists():
        return 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    tasks = payload.get("tasks", []) if isinstance(payload, Mapping) else []
    return len(tasks) if isinstance(tasks, list) else 0


def _qualified_candidate_count(candidates: pd.DataFrame) -> int:
    if candidates.empty or "quality_status" not in candidates.columns:
        return 0
    return int(candidates["quality_status"].astype(str).str.lower().eq("qualified").sum())


def _expected_no_trade_count(candidates: pd.DataFrame, priced: pd.DataFrame) -> int:
    if candidates.empty:
        return 0
    priced_tickers = (
        set(priced.get("ticker", pd.Series(dtype=object)).dropna().astype(str).str.upper())
        if not priced.empty
        else set()
    )
    candidate_tickers = candidates.get("ticker", pd.Series("", index=candidates.index)).astype(str).str.upper()
    quality = candidates.get("quality_status", pd.Series("", index=candidates.index)).astype(str).str.lower()
    expected = quality.ne("qualified") | ~candidate_tickers.isin(priced_tickers)
    return int(expected.sum())


def _monthly_metric(monthly: pd.DataFrame, metric: str) -> Any:
    if monthly.empty or "metric" not in monthly.columns or "value" not in monthly.columns:
        return 0
    rows = monthly[monthly["metric"].astype(str).eq(metric)]
    if rows.empty:
        return 0
    return rows.iloc[0].get("value", 0)


def _target_rows(tickets: pd.DataFrame) -> pd.DataFrame:
    if tickets.empty or "target_order_status" not in tickets.columns:
        return pd.DataFrame(columns=tickets.columns)
    status = tickets["target_order_status"].astype(str).str.lower()
    return tickets[status.isin(["target_order_candidate", "target_order_wait_for_price"])]


def _target_count(tickets: pd.DataFrame) -> int:
    return int(len(_target_rows(tickets)))


def _dates_from_frame(frame: pd.DataFrame) -> list[str]:
    if frame.empty or "date" not in frame.columns:
        return []
    return sorted(
        {
            str(value).strip()
            for value in frame["date"].dropna().tolist()
            if str(value).strip()
        }
    )


def _dates_where_numeric(frame: pd.DataFrame, column: str) -> list[str]:
    if frame.empty or "date" not in frame.columns or column not in frame.columns:
        return []
    values = pd.to_numeric(frame[column], errors="coerce").fillna(0)
    return sorted(
        {
            str(value).strip()
            for value in frame.loc[values.gt(0), "date"].dropna().tolist()
            if str(value).strip()
        }
    )


def _sum_numeric(frame: pd.DataFrame, column: str) -> int:
    if frame.empty or column not in frame.columns:
        return 0
    return int(pd.to_numeric(frame[column], errors="coerce").fillna(0).sum())


def _truthy_count(series: Optional[pd.Series]) -> int:
    if series is None:
        return 0
    return int(series.map(_truthy).sum())


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _symbols(frame: pd.DataFrame) -> str:
    if frame.empty or "ticker" not in frame.columns:
        return ""
    values = [str(value).upper() for value in frame["ticker"].dropna().tolist() if str(value).strip()]
    return ", ".join(dict.fromkeys(values))


def _agentic_dates(summary: pd.DataFrame) -> list[str]:
    if summary.empty:
        return []
    coverage = summary.get("agentic_review_lane_coverage_pct", pd.Series(dtype=object)).map(_to_float)
    return summary.loc[coverage.ge(0.8), "date"].astype(str).tolist()


def _to_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    if pd.isna(result):
        return 0.0
    return result


def _nonempty_count(series: Optional[pd.Series]) -> int:
    if series is None:
        return 0
    return int(series.astype(str).str.strip().ne("").sum())


def _mapping_get(mapping: Mapping[str, Any] | pd.Series, key: str) -> Any:
    if isinstance(mapping, pd.Series):
        return mapping.get(key, "")
    return mapping.get(key, "")


def _post_rerun_evidence_files(plan: Mapping[str, Any] | pd.Series) -> str:
    keys = [
        "green_ticket_file",
        "trade_ticket_file",
        "execution_readiness_file",
        "expectancy_file",
    ]
    values = [str(_mapping_get(plan, key) or "").strip() for key in keys]
    return "; ".join(value for value in values if value)


def _planned_live_probe_dirs(plan: pd.DataFrame) -> list[Path]:
    if plan.empty:
        return []
    selected = _select_verification_plan_row(plan)
    selected_value = str(_mapping_get(selected, "rerun_out_dir") or "").strip()
    if selected_value:
        return [Path(selected_value).expanduser().resolve()]
    dirs: list[Path] = []
    for _, row in plan.iterrows():
        value = str(row.get("rerun_out_dir") or "").strip()
        if value:
            dirs.append(Path(value).expanduser().resolve())
    return dirs


def _audit_regeneration_command(
    *,
    base_dir: Path,
    summary_csv: Path,
    live_probe_dirs: Sequence[Path],
    quality_run_dirs: Sequence[Path] = (),
    rerun_agent_reviews_json: Optional[Path] = None,
    output_prefix: Path,
) -> str:
    parts = [
        "python3",
        "-m",
        "uwos.options_agent.audit",
        "--base-dir",
        str(base_dir),
        "--summary-csv",
        str(summary_csv),
        "--output-prefix",
        str(output_prefix),
    ]
    for live_dir in live_probe_dirs:
        parts.extend(["--live-probe-dir", str(live_dir)])
    for quality_dir in quality_run_dirs:
        parts.extend(["--quality-run-dir", str(quality_dir)])
    if rerun_agent_reviews_json:
        parts.extend(["--rerun-agent-reviews-json", str(rerun_agent_reviews_json)])
    return " ".join(shlex.quote(part) for part in parts)


def _run_dirs_from_summary_csv(values: Iterable[str]) -> list[Path]:
    run_dirs: list[Path] = []
    for value in values:
        path = Path(value).expanduser().resolve()
        frame = _safe_read_csv(path)
        if frame.empty or "source_dir" not in frame.columns:
            continue
        for source_dir in frame["source_dir"].dropna().astype(str).tolist():
            cleaned = source_dir.strip()
            if cleaned:
                run_dirs.append(Path(cleaned).expanduser().resolve())
    return run_dirs


def _write_frame(df: pd.DataFrame, path: Path, columns: Optional[Sequence[str]] = None) -> None:
    frame = df.copy()
    if columns is not None:
        for column in columns:
            if column not in frame.columns:
                frame[column] = ""
        frame = frame[list(columns)]
    frame = frame.astype(object).where(pd.notna(frame), "")
    frame.to_csv(path, index=False)


def _write_summary_markdown(summary: pd.DataFrame, path: Path) -> None:
    lines = ["# Options Agent Expanded Multi-Date Summary", ""]
    if summary.empty:
        lines.extend(["No run summaries were found.", ""])
    else:
        dates = sorted(summary["date"].astype(str).unique().tolist())
        lines.extend(
            [
                f"Validation window: {dates[0]} through {dates[-1]}.",
                "",
                f"- Dated runs: {len(dates)}",
                f"- Trade tickets: {int(pd.to_numeric(summary['trade_ticket_rows'], errors='coerce').fillna(0).sum())}",
                f"- Green send-now orders: {int(pd.to_numeric(summary['green_ready_orders'], errors='coerce').fillna(0).sum())}",
                f"- Yellow target candidates: {int(pd.to_numeric(summary['yellow_target_candidates'], errors='coerce').fillna(0).sum())}",
                f"- Market-open recheck rows in dated runs: {int(pd.to_numeric(summary['market_open_recheck_queue'], errors='coerce').fillna(0).sum())}",
                "",
            ]
        )
        view = summary[
            [
                "date",
                "validation_lane",
                "trade_ticket_rows",
                "green_ready_orders",
                "yellow_target_candidates",
                "execution_readiness",
                "monthly_feasibility",
                "expectancy_summary_status",
                "target_symbols",
            ]
        ]
        lines.append(_markdown_table(view))
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_markdown_table(path: Path, title: str, df: pd.DataFrame) -> None:
    lines = [f"# {title}", ""]
    if df.empty:
        lines.extend(["No rows.", ""])
    else:
        lines.append(_markdown_table(df))
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _markdown_table(df: pd.DataFrame) -> str:
    clean = df.astype(object).where(pd.notna(df), "")
    return clean.to_markdown(index=False)


def _clean_display_value(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text.lower() in {"nan", "nat", "none", "[]"}:
        return ""
    return text


def _write_agentic_coverage_proof_markdown(path: Path, packet: pd.DataFrame) -> None:
    lines = ["# Options Agent Agentic Coverage Proof Packet", ""]
    if packet.empty:
        lines.extend(["No agentic coverage packet rows were available.", ""])
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    row = packet.iloc[0]
    lines.extend(
        [
            f"- Status: {row.get('status', '')}",
            f"- Run dates: {row.get('run_date_count', '')}",
            f"- Ticket rows: {row.get('ticket_rows', '')}",
            f"- Agentic-ready dates: {row.get('agentic_ready_date_count', '')}",
            f"- Ticket rows with agentic-ready run coverage: {row.get('ticket_rows_with_agentic_ready', '')}",
            f"- Ticket rows without agentic-ready run coverage: {row.get('ticket_rows_without_agentic_ready', '')}",
            f"- Ticket agentic coverage pct: {row.get('ticket_agentic_coverage_pct', '')}",
            f"- Required minimum distinct lanes per ticket: {row.get('required_min_ticket_lanes', '')}",
            f"- Minimum distinct lanes observed on a ticket: {row.get('min_ticket_distinct_review_count', '')}",
            f"- Ticket rows below lane minimum: {row.get('ticket_rows_below_min_ticket_lanes', '')}",
            f"- Required coverage: {row.get('required_coverage', '')}",
            "",
            "## Agentic-Ready Dates",
            "",
            str(row.get("agentic_ready_dates", "")) or "None",
            "",
            "## Non-Agentic Ticket Dates",
            "",
            str(row.get("non_agentic_ticket_dates", "")) or "None",
            "",
            "## Ticket Dates Below Lane Minimum",
            "",
            str(row.get("below_min_ticket_lane_dates", "")) or "None",
            "",
            "## Claim",
            "",
            str(row.get("claim", "")),
            "",
            "## Guardrail",
            "",
            str(row.get("note", "")),
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_validation_coverage_proof_markdown(path: Path, packet: pd.DataFrame) -> None:
    lines = ["# Options Agent Validation Coverage Proof Packet", ""]
    if packet.empty:
        lines.extend(["No validation coverage packet rows were available.", ""])
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    row = packet.iloc[0]
    lines.extend(
        [
            f"- Status: {row.get('status', '')}",
            f"- Validation window: {row.get('validation_start', '')} through {row.get('validation_end', '')}",
            f"- Tested dates: {row.get('tested_date_count', '')}",
            f"- Available source dates in window: {row.get('window_available_source_date_count', '')}",
            f"- Untested available dates in window: {row.get('untested_available_date_count', '')}",
            f"- Available source dates outside window: {row.get('available_dates_outside_window_count', '')}",
            "",
            "## Tested Dates",
            "",
            str(row.get("tested_dates", "")) or "None",
            "",
            "## Untested Available Dates In Window",
            "",
            str(row.get("untested_available_dates", "")) or "None",
            "",
            "## Available Dates Outside Window",
            "",
            str(row.get("available_dates_outside_window", "")) or "None",
            "",
            "## Guardrail",
            "",
            str(row.get("note", "")),
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_cutoff_visibility_proof_markdown(path: Path, packet: pd.DataFrame) -> None:
    lines = ["# Options Agent Cutoff Visibility Proof Packet", ""]
    if packet.empty:
        lines.extend(["No cutoff visibility packet rows were available.", ""])
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    row = packet.iloc[0]
    lines.extend(
        [
            f"- Status: {row.get('status', '')}",
            f"- Run count: {row.get('run_count', '')}",
            f"- Candidate rows: {row.get('candidate_rows', '')}",
            f"- Research-task rows: {row.get('research_task_rows', '')}",
            f"- Qualified candidate rows: {row.get('qualified_candidate_rows', '')}",
            f"- Priced candidate rows: {row.get('priced_candidate_rows', '')}",
            f"- Final rows: {row.get('final_rows', '')}",
            f"- Expected no-trade rows: {row.get('expected_no_trade_rows', '')}",
            f"- No-trade audit rows: {row.get('no_trade_audit_rows', '')}",
            f"- Candidate/research mismatch runs: {row.get('candidate_research_mismatch_runs', '') or 'None'}",
            f"- Priced-missing-qualified runs: {row.get('priced_missing_qualified_runs', '') or 'None'}",
            f"- No-trade-missing-expected runs: {row.get('no_trade_missing_expected_runs', '') or 'None'}",
            "",
            "## Claim",
            "",
            str(row.get("claim", "")),
            "",
            "## Guardrail",
            "",
            str(row.get("note", "")),
            "",
        ]
    )
    problem_runs = str(row.get("problem_runs", "") or "")
    if problem_runs:
        lines.extend(["## Problem Runs", "", problem_runs, ""])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_live_spread_quality_proof_markdown(path: Path, packet: pd.DataFrame) -> None:
    lines = ["# Options Agent Live Spread Quality Proof Packet", ""]
    if packet.empty:
        lines.extend(["No live-spread-quality packet rows were available.", ""])
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    row = packet.iloc[0]
    lines.extend(
        [
            f"- Status: {row.get('status', '')}",
            f"- Audited rows: {row.get('audited_rows', '')}",
            f"- PASS rows: {row.get('pass_rows', '')}",
            f"- BLOCK rows: {row.get('block_rows', '')}",
            f"- Quote-width block rows: {row.get('quote_width_block_rows', '')}",
            f"- Liquidity block rows: {row.get('liquidity_block_rows', '')}",
            f"- Blocked rows removed from target surface: {row.get('blocked_not_target_candidate_rows', '')}",
            f"- Blocked rows still actionable: {row.get('blocked_still_actionable_rows', '')}",
            f"- Target-candidate rows: {row.get('target_candidate_rows', '')}",
            f"- Target-candidate block rows: {row.get('target_candidate_block_rows', '')}",
            "",
            "## Required Gate",
            "",
            str(row.get("required_gate", "")),
            "",
            "## Blocked Tickers",
            "",
            str(row.get("blocked_tickers", "")) or "None",
            "",
            "## Blocked Examples",
            "",
            str(row.get("blocked_examples", "")) or "None",
            "",
            "## Claim",
            "",
            str(row.get("claim", "")),
            "",
            "## Guardrail",
            "",
            str(row.get("note", "")),
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_underlying_quality_proof_markdown(path: Path, packet: pd.DataFrame) -> None:
    lines = ["# Options Agent Underlying Quality Proof Packet", ""]
    if packet.empty:
        lines.extend(["No underlying-quality packet rows were available.", ""])
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    row = packet.iloc[0]
    lines.extend(
        [
            f"- Status: {row.get('status', '')}",
            f"- Ticket rows: {row.get('ticket_rows', '')}",
            f"- Core ticket rows: {row.get('core_ticket_rows', '')}",
            f"- Liquid non-core ticket rows: {row.get('liquid_non_core_ticket_rows', '')}",
            f"- Speculative ticket rows: {row.get('speculative_ticket_rows', '')}",
            f"- Excluded ticket rows: {row.get('excluded_ticket_rows', '')}",
            f"- Unknown ticket rows: {row.get('unknown_ticket_rows', '')}",
            f"- Non-core ticket rows: {row.get('not_core_or_liquid_ticket_rows', '')}",
            f"- Green non-core rows: {row.get('green_not_core_or_liquid_rows', '')}",
            f"- Yellow target non-core rows: {row.get('target_not_core_or_liquid_rows', '')}",
            "",
            "## Ticket Tickers",
            "",
            str(row.get("ticket_tickers", "")) or "None",
            "",
            "## Liquid Non-Core Ticket Tickers",
            "",
            str(row.get("liquid_non_core_ticket_tickers", "")) or "None",
            "",
            "## Not Core/Liquid Ticket Tickers",
            "",
            str(row.get("not_core_or_liquid_ticket_tickers", "")) or "None",
            "",
            "## Focus Ticker Examples",
            "",
            f"- Speculative: {row.get('focus_speculative_examples', '') or 'None'}",
            f"- Excluded: {row.get('focus_excluded_examples', '') or 'None'}",
            f"- Liquid non-core: {row.get('focus_liquid_non_core_examples', '') or 'None'}",
            "",
            "## Claim",
            "",
            str(row.get("claim", "")),
            "",
            "## Guardrail",
            "",
            str(row.get("note", "")),
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_major_name_coverage_proof_markdown(path: Path, packet: pd.DataFrame) -> None:
    lines = ["# Options Agent Major-Name Coverage Proof Packet", ""]
    if packet.empty:
        lines.extend(["No major-name coverage packet rows were available.", ""])
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    row = packet.iloc[0]
    lines.extend(
        [
            f"- Status: {row.get('status', '')}",
            f"- Required ticker count: {row.get('required_ticker_count', '')}",
            f"- Covered required tickers: {row.get('covered_required_ticker_count', '')}",
            f"- Missing required tickers: {row.get('missing_required_ticker_count', '')}",
            f"- Required focus rows: {row.get('required_focus_rows', '')}",
            f"- Required rows with reason: {row.get('required_rows_with_reason', '')}",
            f"- Required rows missing reason: {row.get('required_rows_missing_reason', '')}",
            "",
            "## Required Tickers",
            "",
            str(row.get("required_tickers", "")) or "None",
            "",
            "## Missing Required Tickers",
            "",
            str(row.get("missing_required_tickers", "")) or "None",
            "",
            "## Inclusion/Exclusion Buckets",
            "",
            f"- Ready tickets: {row.get('ready_ticket_tickers', '') or 'None'}",
            f"- Yellow target tickets: {row.get('yellow_target_tickers', '') or 'None'}",
            f"- Review/final-no-ticket rows: {row.get('review_ticket_tickers', '') or 'None'}",
            f"- Structured but not final: {row.get('structured_not_final_tickers', '') or 'None'}",
            f"- Candidate/structure missing: {row.get('candidate_not_structured_tickers', '') or 'None'}",
            f"- No directional edge/below cutoff: {row.get('no_directional_edge_tickers', '') or 'None'}",
            f"- Source missing: {row.get('source_missing_tickers', '') or 'None'}",
            f"- Blocked or excluded: {row.get('blocked_or_excluded_tickers', '') or 'None'}",
            "",
            "## Examples",
            "",
            str(row.get("examples", "")) or "None",
            "",
            "## Claim",
            "",
            str(row.get("claim", "")),
            "",
            "## Guardrail",
            "",
            str(row.get("note", "")),
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_expectancy_proof_packet_markdown(
    path: Path,
    packet: pd.DataFrame,
    expectancy: pd.DataFrame,
) -> None:
    lines = ["# Options Agent Expectancy Proof Packet", ""]
    if packet.empty:
        lines.extend(["No expectancy packet rows were available.", ""])
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    row = packet.iloc[0]
    lines.extend(
        [
            f"- Status: {row.get('status', '')}",
            f"- Monthly claim allowed: {row.get('monthly_claim_allowed', '')}",
            f"- Current green ready orders: {row.get('current_green_ready_orders', '')}",
            f"- Ticket ticker count: {row.get('ticket_ticker_count', '')}",
            f"- Minimum sample size: {row.get('minimum_sample_size', '')}",
            f"- Minimum win rate: {row.get('minimum_win_rate', '')}",
            f"- Minimum profit factor: {row.get('minimum_profit_factor', '')}",
            f"- Monthly profit target: {row.get('monthly_profit_target', '')}",
            "",
            "## Required Evidence",
            "",
            str(row.get("required_evidence", "")),
            "",
            "## Blocking Sources",
            "",
            str(row.get("blocking_source_counts", "")) or "No blocking sources.",
            "",
            "## Source Statuses",
            "",
            f"- Forward realized: {row.get('forward_realized_statuses', '')}",
            f"- Actual closed trades: {row.get('actual_closed_trade_statuses', '')}",
            f"- Replay: {row.get('replay_statuses', '')}",
            "",
            "## Guardrail",
            "",
            str(row.get("note", "")),
            "",
        ]
    )
    if not expectancy.empty:
        lines.extend(["## Expectancy Summary Rows", ""])
        summary = expectancy[expectancy["source"].astype(str).eq("expectancy_summary")] if "source" in expectancy.columns else expectancy
        view_columns = ["date", "validation_lane", "status", "sample_size", "matched_current_tickers", "note"]
        view = summary[[column for column in view_columns if column in summary.columns]]
        lines.append(_markdown_table(view))
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_ticket_expectancy_proof_markdown(
    path: Path,
    packet: pd.DataFrame,
    coverage: pd.DataFrame,
) -> None:
    lines = ["# Options Agent Structure-Aligned Ticket Expectancy Proof Packet", ""]
    if packet.empty:
        lines.extend(["No structure-aligned ticket expectancy proof packet rows were available.", ""])
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    row = packet.iloc[0]
    lines.extend(
        [
            f"- Status: {row.get('status', '')}",
            f"- Ticket ticker count: {row.get('ticket_ticker_count', '')}",
            f"- Green ticker count: {row.get('green_ticker_count', '')}",
            "",
            "## Tickers",
            "",
            f"- Current ticket tickers: {row.get('ticket_tickers', '') or 'None'}",
            f"- Current green tickers: {row.get('green_tickers', '') or 'None'}",
            f"- Positive structure-aligned actual/forward tickers: {row.get('tickers_with_positive_actual_forward', '') or 'None'}",
            f"- Green tickers missing structure-aligned evidence: {row.get('green_tickers_without_positive_actual_forward', '') or 'None'}",
            f"- Replay-only tickers: {row.get('replay_only_tickers', '') or 'None'}",
            "",
            "## Required Evidence",
            "",
            str(row.get("required_evidence", "")),
            "",
            "## Guardrail",
            "",
            str(row.get("note", "")),
            "",
        ]
    )
    if not coverage.empty:
        lines.extend(["## Coverage Detail", ""])
        lines.append(_markdown_table(coverage))
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_market_open_execution_packet_markdown(
    path: Path,
    packet: pd.DataFrame,
    market_open_recheck_queue: pd.DataFrame,
) -> None:
    lines = ["# Options Agent Market Open Execution Packet", ""]
    if packet.empty:
        lines.extend(["No live probe packet rows were available.", ""])
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    for _, row in packet.iterrows():
        lines.extend(
            [
                f"Date: {row.get('date', '')}",
                "",
                f"- Status: {row.get('status', '')}",
                f"- Fresh live quotes ready: {row.get('fresh_live_quotes_ready', '')}",
                f"- Portfolio ready: {row.get('portfolio_ready', '')}",
                f"- Agentic reviews ready: {row.get('agentic_reviews_ready', '')}",
                f"- Regular market session open: {row.get('market_session_open', '')}",
                f"- Current green ready orders: {row.get('green_ready_orders', '')}",
                f"- Yellow market-open recheck rows: {row.get('yellow_recheck_rows', '')}",
                f"- Next regular session start: {row.get('next_regular_session_start', '')}",
                f"- Calendar note: {row.get('market_calendar_note', '')}",
                "",
                "## Rerun Command",
                "",
                "```bash",
                f"cd {shlex.quote(str(Path(row.get('source_dir', '')).parents[2])) if row.get('source_dir') else '.'}",
                str(row.get("command", "")),
                "```",
                "",
                "## Required Condition",
                "",
                str(row.get("required_condition", "")),
                "",
                "## Guardrail",
                "",
                str(row.get("note", "")),
                "",
            ]
        )
    if not market_open_recheck_queue.empty:
        lines.extend(["## Current Yellow Recheck Queue", ""])
        view_columns = [
            "date",
            "ticker",
            "entry_type",
            "suggested_contracts",
            "entry_limit",
            "target_exit",
            "position_max_profit",
            "position_max_loss",
            "trade_plan",
            "execution_blockers",
        ]
        view = market_open_recheck_queue[[column for column in view_columns if column in market_open_recheck_queue.columns]]
        lines.append(_markdown_table(view))
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_market_open_recheck_proof_markdown(
    path: Path,
    packet: pd.DataFrame,
    details: pd.DataFrame,
) -> None:
    lines = ["# Options Agent Market-Open Recheck Proof Packet", ""]
    if packet.empty:
        lines.extend(["No market-open recheck proof packet rows were available.", ""])
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    row = packet.iloc[0]
    lines.extend(
        [
            f"- Status: {row.get('status', '')}",
            f"- Queue rows: {row.get('queue_rows', '')}",
            f"- Live queue rows: {row.get('live_queue_rows', '')}",
            f"- Row pass rows: {row.get('row_pass_rows', '')}",
            f"- Row fail rows: {row.get('row_fail_rows', '')}",
            f"- Only-market-session blocker rows: {row.get('only_market_session_blocker_rows', '')}",
            f"- Positive entry rows: {row.get('positive_entry_rows', '')}",
            f"- Positive contract rows: {row.get('positive_contract_rows', '')}",
            f"- Confidence-score PASS rows: {row.get('confidence_score_pass_rows', '')}",
            f"- Trade-quality confidence PASS rows: {row.get('trade_quality_confidence_pass_rows', '')}",
            f"- Agentic lane PASS rows: {row.get('agentic_lane_pass_rows', '')}",
            f"- Plain-language leg rows: {row.get('plain_language_leg_rows', '')}",
            f"- Credit rows: {row.get('credit_rows', '')}",
            f"- Debit rows: {row.get('debit_rows', '')}",
            "",
            "## Tickers",
            "",
            str(row.get("tickers", "")) or "None",
            "",
            "## Required Gate",
            "",
            str(row.get("required_gate", "")),
            "",
            "## Failed Examples",
            "",
            str(row.get("failed_examples", "")) or "None",
            "",
            "## Claim",
            "",
            str(row.get("claim", "")),
            "",
            "## Guardrail",
            "",
            str(row.get("note", "")),
            "",
        ]
    )
    if not details.empty:
        view_columns = [
            "date",
            "ticker",
            "entry_type",
            "entry_limit",
            "suggested_contracts",
            "execution_confidence_score",
            "trade_quality_confidence_rating",
            "external_agent_distinct_review_count",
            "only_market_session_blocker",
            "plain_language_legs_pass",
            "row_pass",
            "fail_reasons",
            "trade_plan",
        ]
        view = details[[column for column in view_columns if column in details.columns]]
        lines.extend(["## Row Details", "", _markdown_table(view), ""])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_live_rerun_preflight_proof_markdown(
    path: Path,
    packet: pd.DataFrame,
    details: pd.DataFrame,
) -> None:
    lines = ["# Options Agent Live Rerun Preflight Proof Packet", ""]
    if packet.empty:
        lines.extend(["No live-rerun preflight proof packet rows were available.", ""])
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    row = packet.iloc[0]
    lines.extend(
        [
            f"- Status: {row.get('status', '')}",
            f"- Queue tickers: {row.get('queue_ticker_count', '')}",
            f"- Covered queue tickers: {row.get('covered_queue_ticker_count', '')}",
            f"- Missing queue tickers: {row.get('missing_queue_ticker_count', '')}",
            f"- Agent review rows: {row.get('agent_review_rows', '')}",
            f"- Distinct agents: {row.get('distinct_agent_count', '')}",
            f"- Rerun output directory clear: {row.get('rerun_out_dir_clear', '')}",
            f"- Source date available: {row.get('source_date_available', '')}",
            f"- Has --live-schwab: {row.get('rerun_command_has_live_schwab', '')}",
            f"- Has --live-portfolio: {row.get('rerun_command_has_live_portfolio', '')}",
            f"- Has --agent-reviews-json: {row.get('rerun_command_has_agent_reviews_json', '')}",
            "",
            "## Agent Reviews JSON",
            "",
            str(row.get("agent_reviews_json", "")) or "None",
            "",
            "## Rerun Output Directory",
            "",
            str(row.get("rerun_out_dir", "")) or "None",
            "",
            "## Missing Queue Tickers",
            "",
            str(row.get("missing_queue_tickers", "")) or "None",
            "",
            "## Required Gate",
            "",
            str(row.get("required_gate", "")),
            "",
            "## Failed Examples",
            "",
            str(row.get("failed_examples", "")) or "None",
            "",
            "## Claim",
            "",
            str(row.get("claim", "")),
            "",
            "## Guardrail",
            "",
            str(row.get("note", "")),
            "",
        ]
    )
    if not details.empty:
        lines.extend(["## Ticker Coverage Detail", "", _markdown_table(details), ""])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_multi_date_readiness_proof_markdown(path: Path, packet: pd.DataFrame) -> None:
    lines = ["# Options Agent Multi-Date Readiness Proof Packet", ""]
    if packet.empty:
        lines.extend(["No multi-date readiness proof packet rows were available.", ""])
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    row = packet.iloc[0]
    lines.extend(
        [
            f"- Status: {row.get('status', '')}",
            f"- Validation date count: {row.get('validation_date_count', '')}",
            f"- Dated ticket rows: {row.get('dated_ticket_rows', '')}",
            f"- Dated green ready orders: {row.get('dated_green_ready_orders', '')}",
            f"- Dated yellow target candidates: {row.get('dated_yellow_target_candidates', '')}",
            f"- Live probe count: {row.get('live_probe_count', '')}",
            f"- Live probe dates: {row.get('live_probe_dates', '') or 'None'}",
            f"- Latest live probe date: {row.get('latest_live_probe_date', '') or 'None'}",
            f"- Latest live probe status: {row.get('latest_live_probe_status', '') or 'None'}",
            f"- Live market-session-open count: {row.get('live_market_session_open_count', '')}",
            f"- Live green ready orders: {row.get('live_green_ready_orders', '')}",
            f"- Live yellow recheck rows: {row.get('live_yellow_recheck_rows', '')}",
            "",
            "## Claim",
            "",
            str(row.get("claim", "")),
            "",
            "## Guardrail",
            "",
            str(row.get("note", "")),
            "",
            "## Tested Dates",
            "",
            str(row.get("tested_dates", "")) or "None",
            "",
            "## Dates With Target Tickets",
            "",
            str(row.get("dates_with_tickets", "")) or "None",
            "",
            "## Dates With Yellow Target Candidates",
            "",
            str(row.get("dates_with_yellow_target_candidates", "")) or "None",
            "",
            "## Dates With Green Ready Orders",
            "",
            str(row.get("dates_with_green_ready_orders", "")) or "None",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_market_session_verification_plan_markdown(path: Path, plan: pd.DataFrame) -> None:
    lines = ["# Options Agent Market-Session Verification Plan", ""]
    if plan.empty:
        lines.extend(["No market-session verification plan rows were available.", ""])
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    for _, row in plan.iterrows():
        lines.extend(
            [
                f"Date: {row.get('date', '')}",
                "",
                f"- Status: {row.get('status', '')}",
                f"- Next regular session start: {row.get('next_regular_session_start', '')}",
                f"- Yellow recheck rows: {row.get('yellow_recheck_rows', '')}",
                f"- Current green ready orders: {row.get('green_ready_orders', '')}",
                f"- Rerun output directory: {row.get('rerun_out_dir', '')}",
                "",
                "## Rerun Command",
                "",
                "```bash",
                str(row.get("rerun_command", "")),
                "```",
                "",
                "## Files To Inspect After Rerun",
                "",
                f"- Green tickets: {row.get('green_ticket_file', '')}",
                f"- All tickets: {row.get('trade_ticket_file', '')}",
                f"- Execution readiness: {row.get('execution_readiness_file', '')}",
                f"- Expectancy evidence: {row.get('expectancy_file', '')}",
                "",
                "## Pass Criteria",
                "",
                str(row.get("pass_criteria", "")),
                "",
                "## Fail Criteria",
                "",
                str(row.get("fail_criteria", "")),
                "",
                "## Completion Gate",
                "",
                str(row.get("completion_gate", "")),
                "",
                "## Note",
                "",
                str(row.get("note", "")),
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_post_rerun_verification_markdown(path: Path, packet: pd.DataFrame) -> None:
    lines = ["# Options Agent Post-Rerun Verification Packet", ""]
    if packet.empty:
        lines.extend(["No post-rerun verification packet rows were available.", ""])
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    row = packet.iloc[0]
    rerun_command = str(row.get("rerun_command", "") or "").strip()
    rerun_command_block = ["```bash", rerun_command, "```"] if rerun_command else [
        "No rerun command is attached to this row; the packet is using an already-completed live probe as evidence."
    ]
    lines.extend(
        [
            f"- Status: {row.get('status', '')}",
            f"- Date: {row.get('date', '')}",
            f"- Market session open: {row.get('market_session_open', '')}",
            f"- Green ticket status: {row.get('green_ticket_status', '')}",
            f"- Ticket expectancy status: {row.get('ticket_expectancy_status', '')}",
            f"- Completion verdict status: {row.get('completion_verdict_status', '')}",
            f"- Can mark goal complete: {row.get('can_mark_goal_complete', '')}",
            f"- Update-goal action: {row.get('update_goal_action', '')}",
            f"- Green ticket rows: {row.get('green_ticket_rows', '')}",
            f"- Valid green ticket rows: {row.get('valid_green_ticket_rows', '')}",
            f"- Invalid green ticket rows: {row.get('invalid_green_ticket_rows', '')}",
            f"- Green ticker count: {row.get('green_ticker_count', '')}",
            f"- Monthly claim allowed: {row.get('monthly_claim_allowed', '')}",
            "",
            "## Rerun Command",
            "",
            *rerun_command_block,
            "",
            "## Regenerate This Verification",
            "",
            "```bash",
            str(row.get("audit_regeneration_command", "")),
            "```",
            "",
            "## Evidence Files",
            "",
            str(row.get("evidence_files", "")) or "None",
            "",
            "## Required Next Action",
            "",
            str(row.get("required_next_action", "")),
            "",
            "## Guardrail",
            "",
            str(row.get("note", "")),
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_green_ticket_execution_proof_markdown(
    path: Path,
    packet: pd.DataFrame,
    details: pd.DataFrame,
) -> None:
    lines = ["# Options Agent Green-Ticket Execution Proof Packet", ""]
    if packet.empty:
        lines.extend(["No green-ticket execution proof packet rows were available.", ""])
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    row = packet.iloc[0]
    lines.extend(
        [
            f"- Status: {row.get('status', '')}",
            f"- Live probe count: {row.get('live_probe_count', '')}",
            f"- Green ticket rows: {row.get('green_ticket_rows', '')}",
            f"- Valid green ticket rows: {row.get('valid_green_ticket_rows', '')}",
            f"- Invalid green ticket rows: {row.get('invalid_green_ticket_rows', '')}",
            f"- Ready-to-enter rows: {row.get('ready_to_enter_rows', '')}",
            f"- Positive entry rows: {row.get('positive_entry_rows', '')}",
            f"- Positive contract rows: {row.get('positive_contract_rows', '')}",
            f"- Live validation PASS rows: {row.get('live_validation_pass_rows', '')}",
            f"- No-blocker rows: {row.get('no_blocker_rows', '')}",
            f"- Confidence-score PASS rows: {row.get('confidence_score_pass_rows', '')}",
            f"- Execution-confidence PASS rows: {row.get('execution_confidence_pass_rows', '')}",
            f"- Trade-quality-confidence PASS rows: {row.get('trade_quality_confidence_pass_rows', '')}",
            f"- Plain-language leg rows: {row.get('plain_language_leg_rows', '')}",
            f"- Market-session-open rows: {row.get('market_session_open_rows', '')}",
            "",
            "## Green Tickers",
            "",
            str(row.get("green_tickers", "")) or "None",
            "",
            "## Required Evidence",
            "",
            str(row.get("required_evidence", "")),
            "",
            "## Invalid Examples",
            "",
            str(row.get("invalid_examples", "")) or "None",
            "",
            "## Guardrail",
            "",
            str(row.get("note", "")),
            "",
        ]
    )
    if not details.empty:
        lines.extend(["## Row Details", ""])
        view_columns = [
            "date",
            "ticker",
            "entry_type",
            "entry_limit",
            "suggested_contracts",
            "live_validation_status",
            "execution_blockers",
            "execution_confidence_score",
            "execution_confidence_rating",
            "trade_quality_confidence_rating",
            "confidence_score_pass",
            "execution_confidence_pass",
            "trade_quality_confidence_pass",
            "market_session_open",
            "row_pass",
            "fail_reasons",
        ]
        view = details[[column for column in view_columns if column in details.columns]]
        lines.append(_markdown_table(view))
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_session_only_green_shadow_proof_markdown(
    path: Path,
    packet: pd.DataFrame,
    details: pd.DataFrame,
) -> None:
    lines = ["# Options Agent Session-Only Green Shadow Proof Packet", ""]
    if packet.empty:
        lines.extend(["No session-only green shadow proof packet rows were available.", ""])
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    row = packet.iloc[0]
    lines.extend(
        [
            f"- Status: {row.get('status', '')}",
            f"- Shadow candidate rows: {row.get('shadow_candidate_rows', '')}",
            f"- Row-pass rows: {row.get('row_pass_rows', '')}",
            f"- Row-fail rows: {row.get('row_fail_rows', '')}",
            f"- Non-session blocker rows: {row.get('non_session_blocker_rows', '')}",
            f"- Credit rows: {row.get('credit_rows', '')}",
            f"- Debit rows: {row.get('debit_rows', '')}",
            f"- Position max profit: {row.get('position_max_profit', '')}",
            f"- Position max loss: {row.get('position_max_loss', '')}",
            "",
            "## Tickers",
            "",
            str(row.get("tickers", "")) or "None",
            "",
            "## Claim",
            "",
            str(row.get("claim", "")),
            "",
            "## Required Next Action",
            "",
            str(row.get("required_next_action", "")),
            "",
            "## Failed Examples",
            "",
            str(row.get("failed_examples", "")) or "None",
            "",
            "## Guardrail",
            "",
            str(row.get("note", "")),
            "",
        ]
    )
    if not details.empty:
        view = details.copy()
        if "source_kind" in view.columns and view["source_kind"].astype(str).eq("live_probe").any():
            view = view[view["source_kind"].astype(str).eq("live_probe")].copy()
        view_columns = [
            "ticker",
            "entry_type",
            "entry_limit",
            "suggested_contracts",
            "position_max_profit",
            "position_max_loss",
            "execution_confidence_score",
            "trade_quality_confidence_rating",
            "external_agent_distinct_review_count",
            "only_market_session_blocker",
            "row_pass",
            "fail_reasons",
        ]
        lines.extend(["## Row Details", ""])
        lines.append(_markdown_table(view[[column for column in view_columns if column in view.columns]]))
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_actionability_surface_proof_markdown(path: Path, packet: pd.DataFrame) -> None:
    lines = ["# Options Agent Actionability Surface Proof Packet", ""]
    if packet.empty:
        lines.extend(["No actionability surface proof packet rows were available.", ""])
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    row = packet.iloc[0]
    lines.extend(
        [
            f"- Status: {row.get('status', '')}",
            f"- Ticket rows: {row.get('ticket_rows', '')}",
            f"- Ready-to-enter rows: {row.get('ready_to_enter_rows', '')}",
            f"- Target-order rows: {row.get('target_order_rows', '')}",
            f"- Target rows incorrectly ready-to-enter: {row.get('target_ready_to_enter_rows', '')}",
            f"- Target rows missing entry type: {row.get('target_missing_entry_type_rows', '')}",
            f"- Target rows missing entry limit: {row.get('target_missing_entry_limit_rows', '')}",
            f"- Target rows missing trade plan: {row.get('target_missing_trade_plan_rows', '')}",
            f"- Target rows missing plain-language legs: {row.get('target_missing_plain_language_leg_rows', '')}",
            f"- Green ticket rows: {row.get('green_ticket_rows', '')}",
            f"- Valid green ticket rows: {row.get('valid_green_ticket_rows', '')}",
            f"- Invalid green ticket rows: {row.get('invalid_green_ticket_rows', '')}",
            f"- Live market-open recheck rows: {row.get('live_market_open_recheck_rows', '')}",
            f"- Entry types: {row.get('entry_types', '') or 'None'}",
            "",
            "## Claim",
            "",
            str(row.get("claim", "")),
            "",
            "## Bad Examples",
            "",
            str(row.get("bad_examples", "")) or "None",
            "",
            "## Guardrail",
            "",
            str(row.get("note", "")),
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_action_surface_underlying_quality_proof_markdown(path: Path, packet: pd.DataFrame) -> None:
    lines = ["# Options Agent Action-Surface Underlying Quality Proof Packet", ""]
    if packet.empty:
        lines.extend(["No action-surface underlying-quality packet rows were available.", ""])
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    row = packet.iloc[0]
    lines.extend(
        [
            f"- Status: {row.get('status', '')}",
            f"- Ticket rows: {row.get('ticket_rows', '')}",
            f"- Market-open recheck rows: {row.get('market_open_recheck_rows', '')}",
            f"- Focus rows: {row.get('focus_rows', '')}",
            f"- Ticket bad-underlying rows: {row.get('ticket_bad_underlying_rows', '')}",
            f"- Market-open recheck bad-underlying rows: {row.get('market_open_recheck_bad_underlying_rows', '')}",
            f"- Focus bad-actionable rows: {row.get('focus_bad_actionable_rows', '')}",
            f"- Audit-only focus rows: {row.get('audit_only_focus_rows', '')}",
            "",
            "## Bad Action Surface Tickers",
            "",
            f"- Tickets: {row.get('ticket_bad_tickers', '') or 'None'}",
            f"- Market-open recheck: {row.get('market_open_recheck_bad_tickers', '') or 'None'}",
            f"- Focus: {row.get('focus_bad_actionable_tickers', '') or 'None'}",
            "",
            "## Allowed / Audit-Only Tickers",
            "",
            f"- Liquid non-core action tickers: {row.get('liquid_non_core_action_tickers', '') or 'None'}",
            f"- Red no-action audit tickers: {row.get('audit_only_focus_tickers', '') or 'None'}",
            "",
            "## Claim",
            "",
            str(row.get("claim", "")),
            "",
            "## Guardrail",
            "",
            str(row.get("note", "")),
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_goal_completion_markdown(goal_audit: pd.DataFrame, path: Path) -> None:
    lines = ["# Options Agent Goal Completion Audit", ""]
    status = "complete" if not goal_audit.empty and goal_audit["status"].astype(str).isin(["PROVEN", "ACHIEVED"]).all() else "goal remains active"
    lines.extend(
        [
            f"Current result: **{status}**.",
            "",
            _markdown_table(goal_audit) if not goal_audit.empty else "No audit rows.",
            "",
            "## Decision",
            "",
        ]
    )
    if status == "complete":
        lines.append("- Evidence satisfies every tracked requirement.")
    else:
        lines.append("- Do not mark the goal complete yet.")
        gaps = [
            str(value).strip()
            for value in goal_audit.get("remaining_gap", pd.Series(dtype=object)).tolist()
            if str(value).strip()
        ]
        if gaps:
            lines.append("- Remaining hard evidence:")
            for gap in dict.fromkeys(gaps):
                lines.append(f"  - {gap}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_completion_verdict_markdown(
    path: Path,
    completion_verdict: pd.DataFrame,
    goal_audit: pd.DataFrame,
) -> None:
    lines = ["# Options Agent Completion Verdict", ""]
    if completion_verdict.empty:
        lines.extend(["No completion verdict was generated.", ""])
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    row = completion_verdict.iloc[0]
    lines.extend(
        [
            f"- Status: {row.get('status', '')}",
            f"- Can mark goal complete: {row.get('can_mark_goal_complete', '')}",
            f"- Update-goal action: {row.get('update_goal_action', '')}",
            f"- Proven requirements: {row.get('proven_requirements', '')}",
            f"- Blocking requirements: {row.get('blocking_requirements', '') or 'None'}",
            f"- Market-open packet status: {row.get('market_open_packet_status', '')}",
            f"- Next regular session start: {row.get('next_regular_session_start', '')}",
            f"- Monthly claim allowed: {row.get('monthly_claim_allowed', '')}",
            f"- Expectancy packet status: {row.get('expectancy_packet_status', '')}",
            f"- Ticket expectancy packet status: {row.get('ticket_expectancy_packet_status', '')}",
            "",
            str(row.get("note", "")),
            "",
        ]
    )
    if not goal_audit.empty:
        lines.extend(["## Goal Rows", "", _markdown_table(goal_audit), ""])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_readiness_dashboard_markdown(path: Path, dashboard: pd.DataFrame) -> None:
    lines = ["# Options Agent Readiness Dashboard", ""]
    if dashboard.empty:
        lines.extend(["No readiness dashboard rows were available.", ""])
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    overall = dashboard[dashboard["area"].astype(str).eq("overall_completion")]
    if not overall.empty:
        row = overall.iloc[0]
        lines.extend(
            [
                f"- Overall status: {row.get('status', '')}",
                f"- Required next action: {row.get('required_next_action', '')}",
                "",
            ]
        )
    lines.extend(
        [
            "## Evidence Index",
            "",
            _markdown_table(dashboard),
            "",
            "## Guardrail",
            "",
            "Use this dashboard as an index only. The linked proof packets remain the source of truth for goal completion.",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_dirs(values: Iterable[str]) -> list[Path]:
    return [Path(value).expanduser().resolve() for value in values]


def recompute_live_capture(
    *,
    source_dir: Path,
    output_dir: Path,
    base_dir: Optional[Path] = None,
) -> dict[str, Path]:
    """Recompute readiness surfaces from a captured market-open live run under current code.

    This preserves the captured Schwab live-chain and portfolio context from
    ``source_dir`` while rebuilding the decision, ticket, readiness, expectancy,
    and report artifacts. It is an audit artifact, not a fresh quote pull.
    """

    source_dir = source_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if source_dir == output_dir:
        raise ValueError("recompute output_dir must be different from source_dir")
    if not source_dir.exists():
        raise FileNotFoundError(f"source live capture directory not found: {source_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    _copy_live_capture_inputs(source_dir, output_dir)

    source_manifest = _read_manifest(source_dir)
    day = _run_date(source_dir, source_manifest)
    root = (base_dir or project_root()).expanduser().resolve()
    paths = output_paths(day, root=root, out_dir=output_dir)
    paths["out_dir"].mkdir(parents=True, exist_ok=True)
    paths["agent_reviews_dir"].mkdir(parents=True, exist_ok=True)

    final = _safe_read_csv(output_dir / "final_recommendations.csv")
    no_trade = _safe_read_csv(output_dir / "no_trade_audit.csv")
    raw_universe = _safe_read_csv(output_dir / "raw_universe.csv")
    candidates = _safe_read_csv(output_dir / "candidate_generation.csv")
    priced = _safe_read_csv(output_dir / "priced_candidates.csv")
    live_spread_quality = _safe_read_csv(output_dir / "live_spread_quality_audit.csv")
    market_regime = _read_json_file(output_dir / "market_regime.json") or source_manifest.get("market_regime", {}) or {}
    final = annotate_actual_forward_expectancy(final, root)
    profitability_calibration = build_profitability_calibration(root, final, as_of_date=day)
    final = annotate_profitability_calibration(final, profitability_calibration)

    execution_context = dict(source_manifest.get("execution_context", {}) or {})
    execution_context.update(
        {
            "quote_mode": "live_schwab",
            "fresh_live_quotes_ready": True,
            "market_session_open": True,
            "market_session_gate_required": True,
            "run_gate_blockers": [],
        }
    )
    if "portfolio_ready" not in execution_context:
        execution_context["portfolio_ready"] = source_manifest.get("portfolio_context_status") == "ok"
    if "agentic_reviews_ready" not in execution_context:
        execution_context["agentic_reviews_ready"] = (
            source_manifest.get("agentic_orchestration", {}).get("status") == "reviews_ingested"
        )

    decision_board = synthesize_decision_board(
        final,
        market_regime=market_regime,
        execution_context=execution_context,
    )
    management_plan = build_management_plan(final, decision_board)
    trade_tickets = build_trade_tickets(decision_board)
    green_trade_tickets, target_order_tickets = split_trade_ticket_surfaces(trade_tickets)
    market_open_recheck_queue = build_market_open_recheck_queue(trade_tickets)
    execution_readiness = build_execution_readiness(decision_board, execution_context)
    expectancy_evidence = build_expectancy_evidence(root, decision_board, trade_tickets)
    monthly_feasibility = build_monthly_feasibility(
        decision_board,
        trade_tickets,
        execution_context,
        expectancy_evidence,
    )
    coverage = build_coverage_audit(raw_universe, candidates, priced, decision_board, no_trade)

    manifest = dict(source_manifest)
    artifacts = {key: str(value) for key, value in paths.items() if key != "out_dir"}
    row_counts = dict(manifest.get("row_counts", {}) or {})
    row_counts.update(
        {
            "final_recommendations": int(len(final)),
            "decision_board": int(len(decision_board)),
            "trade_tickets": int(len(trade_tickets)),
            "green_trade_tickets": int(len(green_trade_tickets)),
            "target_order_ticket_rows": int(len(target_order_tickets)),
            "target_order_candidates": int(len(target_order_tickets)),
            "market_open_recheck_queue": int(len(market_open_recheck_queue)),
            "coverage_audit": int(len(coverage)),
            "management_plan": int(len(management_plan)),
            "ready_to_enter": _truthy_count(decision_board.get("ready_to_enter")),
            "execution_readiness": int(len(execution_readiness)),
            "expectancy_evidence": int(len(expectancy_evidence)),
            "profitability_calibration": int(len(profitability_calibration)),
            "monthly_feasibility": int(len(monthly_feasibility)),
        }
    )
    if not live_spread_quality.empty:
        row_counts["live_spread_quality_audit"] = int(len(live_spread_quality))

    warnings = list(manifest.get("warnings", []) or [])
    warning = (
        "captured market-open live recompute: uses previously captured Schwab live-chain and portfolio context; "
        "not a fresh current quote pull"
    )
    if warning not in warnings:
        warnings.insert(0, warning)

    manifest.update(
        {
            "mode": "captured_market_open_live_recompute_current_code",
            "pipeline_version": PIPELINE_VERSION,
            "out_dir": str(output_dir),
            "artifacts": artifacts,
            "row_counts": row_counts,
            "status_counts": _recommendation_status_counts(final),
            "live_schwab_requested": True,
            "chain_snapshot_dir": "",
            "execution_context": execution_context,
            "execution_readiness_summary": summarize_execution_readiness(execution_readiness),
            "expectancy_evidence_summary": summarize_expectancy_evidence(expectancy_evidence),
            "profitability_calibration_summary": summarize_profitability_calibration(profitability_calibration),
            "monthly_feasibility_summary": summarize_monthly_feasibility(monthly_feasibility),
            "live_spread_quality_summary": summarize_live_spread_quality(live_spread_quality),
            "captured_live_recompute": {
                "source_run_dir": str(source_dir),
                "source_manifest": str(_manifest_path(source_dir)),
                "fresh_quote_pull": False,
                "purpose": "current-code readiness recompute from captured market-open live context",
            },
            "warnings": warnings,
        }
    )

    _write_json(paths["manifest"], manifest)
    _write_frame(decision_board, paths["decision_board"])
    _write_frame(trade_tickets, paths["trade_tickets"])
    _write_frame(green_trade_tickets, paths["green_trade_tickets"])
    _write_frame(target_order_tickets, paths["target_order_candidates"])
    _write_frame(market_open_recheck_queue, paths["market_open_recheck_queue"])
    _write_frame(coverage, paths["coverage_audit"])
    _write_frame(management_plan, paths["management_plan"])
    _write_frame(execution_readiness, paths["execution_readiness"])
    _write_frame(expectancy_evidence, paths["expectancy_evidence"])
    _write_frame(profitability_calibration, paths["profitability_calibration"])
    _write_frame(monthly_feasibility, paths["monthly_feasibility"])
    paths["report"].write_text(
        render_report(day, decision_board, no_trade, manifest, coverage),
        encoding="utf-8",
    )
    return paths


def _copy_live_capture_inputs(source_dir: Path, output_dir: Path) -> None:
    """Copy source files needed to keep a recompute artifact inspectable."""

    for item in source_dir.iterdir():
        target = output_dir / item.name
        if item.is_file():
            shutil.copy2(item, target)
        elif item.is_dir() and item.name == "agent_reviews":
            shutil.copytree(item, target, dirs_exist_ok=True)


def _read_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _manifest_path(run_dir: Path) -> Path:
    manifests = sorted(run_dir.glob("options_agent_manifest_*.json"))
    if not manifests:
        raise FileNotFoundError(f"manifest missing from {run_dir}")
    return manifests[0]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _recommendation_status_counts(frame: pd.DataFrame) -> dict[str, int]:
    if frame.empty or "recommendation_status" not in frame.columns:
        return {}
    return {
        str(status): int(count)
        for status, count in frame["recommendation_status"].astype(str).value_counts().sort_index().items()
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build repeatable Options Agent expanded audit artifacts.")
    parser.add_argument("--base-dir", default=str(project_root()), help="Trade desk root.")
    parser.add_argument("--output-prefix", help="Output prefix for generated audit files.")
    parser.add_argument("--run-dir", action="append", default=[], help="Options Agent dated run directory. Repeatable.")
    parser.add_argument("--summary-csv", action="append", default=[], help="Existing expanded-audit summary CSV; source_dir rows are used as run directories.")
    parser.add_argument("--live-probe-dir", action="append", default=[], help="Live probe directory to include in live-only evidence.")
    parser.add_argument(
        "--quality-run-dir",
        action="append",
        default=[],
        help="Run directory whose live_spread_quality_audit.csv is included only in spread-quality proof.",
    )
    parser.add_argument(
        "--rerun-agent-reviews-json",
        default="",
        help="Agent reviews JSON to use in the generated regular-session live rerun command.",
    )
    parser.add_argument("--focus-ticker", action="append", default=[], help="Additional ticker for focus coverage.")
    parser.add_argument(
        "--recompute-live-capture-dir",
        help="Captured market-open live run directory to recompute under current code.",
    )
    parser.add_argument(
        "--recompute-output-dir",
        help="Output directory for --recompute-live-capture-dir artifacts.",
    )
    args = parser.parse_args(argv)

    if args.recompute_live_capture_dir:
        if not args.recompute_output_dir:
            parser.error("--recompute-output-dir is required with --recompute-live-capture-dir")
        paths = recompute_live_capture(
            source_dir=Path(args.recompute_live_capture_dir),
            output_dir=Path(args.recompute_output_dir),
            base_dir=Path(args.base_dir),
        )
        print(f"Wrote current-code live-capture recompute: {paths['out_dir']}")
        print(f"Report: {paths['report']}")
        print(f"Trade tickets: {paths['trade_tickets']}")
        return 0

    if not args.output_prefix:
        parser.error("--output-prefix is required unless --recompute-live-capture-dir is used")

    focus = list(CORE_AUDIT_TICKERS)
    for ticker in args.focus_ticker:
        cleaned = str(ticker).strip().upper()
        if cleaned and cleaned not in focus:
            focus.append(cleaned)

    artifacts = write_expanded_audit(
        base_dir=Path(args.base_dir).expanduser().resolve(),
        run_dirs=[*_parse_dirs(args.run_dir), *_run_dirs_from_summary_csv(args.summary_csv)],
        live_probe_dirs=_parse_dirs(args.live_probe_dir),
        quality_run_dirs=_parse_dirs(args.quality_run_dir),
        rerun_agent_reviews_json=Path(args.rerun_agent_reviews_json) if args.rerun_agent_reviews_json else None,
        output_prefix=Path(args.output_prefix),
        focus_tickers=focus,
    )
    print(f"Wrote Options Agent expanded audit artifacts under {artifacts.paths['summary'].parent}")
    print(f"Summary: {artifacts.paths['summary']}")
    print(f"Goal audit: {artifacts.paths['goal_completion_audit_md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
