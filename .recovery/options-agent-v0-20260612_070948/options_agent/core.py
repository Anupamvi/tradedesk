"""Independent Options Agent orchestration and policy helpers.

This module is intentionally small at first. It establishes the namespace,
artifact contract, agent roster, and portfolio-risk visibility policy before
the live data integrations are wired in.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import re
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo

import pandas as pd

from uwos.lessonengine.core import (
    apply_synthesis_actions,
    build_application_audit,
    lesson_manifest_metadata,
    load_active_lesson_pack,
    write_lesson_snapshots,
)
from uwos.paths import project_root

PIPELINE_NAME = "Options Agent"
PIPELINE_VERSION = "options_agent.0.22"
DEFAULT_OUTPUT_NAMESPACE = "options_agent"
DEFAULT_TOP_TRADES = 20
DEFAULT_DISCOVERY_LIMIT = 120
DEFAULT_RISK_BUDGET_PCT = 0.005
MAX_SUGGESTED_CONTRACTS = 5
MIN_GREEN_POSITION_MAX_PROFIT = 750.0
MIN_GREEN_POSITION_MAX_PROFIT_PORTFOLIO_PCT = 0.001
MIN_TRADE_CREDIT = 0.25
MIN_CREDIT_WIDTH_RATIO = 0.18
MIN_SEND_NOW_CREDIT = 0.50
MIN_SEND_NOW_CREDIT_WIDTH_RATIO = 0.30
MIN_SEND_NOW_DEBIT_REWARD_RISK_RATIO = 1.50
MIN_SEND_NOW_DEBIT_FLOW_BIAS_WITHOUT_EXPECTANCY = 0.15
MAX_SHORT_DTE_SEND_NOW_DEBIT_BREAKEVEN_MOVE_PCT = 0.04
MAX_MEDIUM_DTE_SEND_NOW_DEBIT_BREAKEVEN_MOVE_PCT = 0.08
MAX_LONG_DTE_SEND_NOW_DEBIT_BREAKEVEN_MOVE_PCT = 0.10
MIN_SHORT_PUT_DISTANCE_PCT = 0.03
MIN_SHORT_PUT_DELTA = 0.08
MAX_SHORT_PUT_DELTA = 0.35
MAX_SHORT_PUT_ACCOUNT_RISK_PCT = 0.02
MAX_SHORT_PUT_CASH_USAGE_PCT = 0.75
FAMILY_LEVEL_STRATEGY_EXPECTANCY_FALLBACKS = {"short_put"}
MIN_SIGNAL_PREMIUM = 1_000_000.0
MIN_DIRECTIONAL_BIAS = 0.10
MAX_ONE_LOT_LOSS = 750.0
MIN_LIVE_DTE = 7
MAX_LIVE_DTE = 60
MIN_EXECUTION_CONFIDENCE_SCORE = 70.0
MIN_AGENTIC_REVIEW_COVERAGE = 0.80
MIN_AGENTIC_REVIEW_LANES_PER_TICKER = 4
MONTHLY_PROFIT_TARGET = 10_000.0
MAX_LIVE_QUOTE_WIDTH_PCT = 0.40
MIN_LIVE_LEG_LIQUIDITY = 100.0
MARKET_TIME_ZONE = ZoneInfo("America/Los_Angeles")
REGULAR_MARKET_OPEN = dt.time(6, 30)
REGULAR_MARKET_CLOSE = dt.time(13, 0)
MARKET_HOLIDAY_LOOKAHEAD_DAYS = 10
MIN_EXPECTANCY_SAMPLE_SIZE = 30
MIN_EXPECTANCY_WIN_RATE = 0.55
MIN_EXPECTANCY_PROFIT_FACTOR = 1.20
MIN_TICKER_EXPECTANCY_SAMPLE_SIZE = 3
MIN_TICKER_EXPECTANCY_WIN_RATE = 0.50
MIN_TICKER_EXPECTANCY_PROFIT_FACTOR = 1.10
MIN_GOAL_PROFITABILITY_CONFIDENCE_RATING = 7.0
MIN_GOAL_ORDER_ENTRY_CONFIDENCE_RATING = 7.0
MIN_LIQUID_MARKET_CAP = 20_000_000_000.0
MIN_CORE_MARKET_CAP = 75_000_000_000.0
MIN_LIQUID_AVG_VOLUME = 5_000_000.0
MIN_LIQUID_OPTION_OI = 250_000.0
MACRO_TAPE_DIRECTIONAL_MOVE_PCT = 0.0075
MACRO_TAPE_CANDIDATE_SCORE_FLOOR = 50.0
ACTIONABLE_ETF_ALLOWLIST = (
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    "TLT",
    "GLD",
    "SLV",
    "SMH",
    "XLK",
    "XLE",
    "XLF",
    "XLV",
    "XLY",
    "XLI",
    "XLU",
    "XLP",
    "XLB",
    "XLRE",
    "XLC",
)
UNDERLYING_QUALITY_SCORE_ADJUSTMENT = {
    "core": 12.0,
    "liquid": 6.0,
    "speculative": -25.0,
    "excluded": -45.0,
}
UNDERLYING_QUALITY_SORT_RANK = {
    "core": 0,
    "liquid": 1,
    "speculative": 2,
    "excluded": 3,
    "unknown": 4,
    "": 4,
}
DEFAULT_COVERAGE_WATCHLIST = (
    "AAPL",
    "NVDA",
    "MSFT",
    "GOOG",
    "GOOGL",
    "PLTR",
    "AMZN",
    "META",
    "TSLA",
    "AMD",
    "QQQ",
    "SPY",
    "HOOD",
    "WMT",
    "URA",
    "DVN",
    "OKLO",
)
CORE_MEGA_CAP_TICKERS = (
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
    "GOOG",
    "GOOGL",
    "TSLA",
    "AMD",
    "AVGO",
)
CORE_INDEX_ETF_TICKERS = ("SPY", "QQQ", "IWM", "DIA")
CORE_AUDIT_TICKERS = tuple(dict.fromkeys((*CORE_MEGA_CAP_TICKERS, *CORE_INDEX_ETF_TICKERS, *DEFAULT_COVERAGE_WATCHLIST)))
TICKER_CANONICAL_GROUPS = {
    "GOOG": "GOOG_GOOGL",
    "GOOGL": "GOOG_GOOGL",
    "BRK.B": "BRKB",
    "BRK-B": "BRKB",
    "BRKB": "BRKB",
}
TICKER_SCOPED_ACTUAL_FORWARD_TYPES = {
    "actual_closed_trades_by_ticker",
    "forward_realized_outcomes_by_ticker",
}
STRATEGY_SCOPED_ACTUAL_FORWARD_TYPES = {
    "actual_closed_trades_by_ticker_strategy",
    "forward_realized_outcomes_by_ticker_strategy",
}
GREEN_TICKET_EXPECTANCY_EVIDENCE_TYPES = STRATEGY_SCOPED_ACTUAL_FORWARD_TYPES
EXPECTANCY_EVIDENCE_COLUMNS = [
    "source",
    "source_path",
    "evidence_type",
    "status",
    "sample_size",
    "win_rate",
    "avg_pnl",
    "total_pnl",
    "profit_factor",
    "max_drawdown",
    "matched_current_tickers",
    "matched_current_count",
    "open_or_unrealized_count",
    "note",
]
STRATEGY_OUTCOME_ATLAS_COLUMNS = [
    "scope",
    "ticker",
    "strategy_family",
    "status",
    "sample_size",
    "win_rate",
    "avg_pnl",
    "total_pnl",
    "profit_factor",
    "max_drawdown",
    "source_tickers",
    "current_ticket_count",
    "current_green_count",
    "suggested_action",
    "source_path",
    "note",
]
STRATEGY_ROUTING_AUDIT_COLUMNS = [
    "candidate_rank",
    "ticker",
    "bias",
    "underlying_quality_tier",
    "macro_tape_direction",
    "strategy",
    "strategy_family",
    "route_action",
    "route_status",
    "route_reason",
    "evidence_status",
    "evidence_sample_size",
    "evidence_avg_pnl",
    "evidence_profit_factor",
    "selected_structure",
    "selected_trade_plan",
    "entry_type",
    "entry_limit",
    "quality_gate_reason",
]
POSITION_PROFIT_MATERIALITY_BLOCKER = "position_profit_below_materiality_floor"
NEGATIVE_STRATEGY_EXPECTANCY_BLOCKER = "negative_strategy_expectancy_for_green"
POSITIVE_STRATEGY_EXPECTANCY_BLOCKER = "positive_strategy_expectancy_required_for_green"
PROFITABILITY_CALIBRATION_BLOCKER = "profitability_calibration_required_for_green"
PROFITABILITY_CALIBRATION_ACTUAL_NEGATIVE_BLOCKER = "profitability_calibration_actual_bucket_negative"
PROFITABILITY_CALIBRATION_COLUMNS = [
    "scope",
    "ticker",
    "strategy_route",
    "strategy_family",
    "entry_type",
    "direction_bucket",
    "regime",
    "dte_bucket",
    "iv_rank_bucket",
    "economics_bucket",
    "liquidity_bucket",
    "status",
    "sample_size",
    "win_rate",
    "avg_pnl",
    "total_pnl",
    "profit_factor",
    "max_drawdown",
    "actual_support_status",
    "actual_support_scope",
    "actual_support_sample_size",
    "actual_support_sample_gap",
    "actual_support_avg_pnl",
    "actual_support_profit_factor",
    "replay_bucket_status",
    "replay_bucket_sample_size",
    "replay_bucket_sample_gap",
    "replay_bucket_avg_pnl",
    "replay_bucket_profit_factor",
    "diagnostic_replay_status",
    "diagnostic_replay_sample_size",
    "diagnostic_replay_avg_pnl",
    "diagnostic_replay_profit_factor",
    "diagnostic_replay_relaxed_dimensions",
    "matched_current_tickers",
    "current_ticket_count",
    "current_green_count",
    "suggested_action",
    "source_path",
    "note",
]
PROFITABILITY_GAP_PLAN_COLUMNS = [
    "gap_rank",
    "exact_bucket_key",
    "strategy_route",
    "strategy_family",
    "entry_type",
    "direction_bucket",
    "regime",
    "dte_bucket",
    "iv_rank_bucket",
    "economics_bucket",
    "liquidity_bucket",
    "current_ticket_count",
    "current_tickers",
    "status",
    "actual_support_status",
    "actual_support_scope",
    "actual_support_sample_size",
    "actual_support_sample_gap",
    "actual_support_avg_pnl",
    "actual_support_profit_factor",
    "replay_bucket_status",
    "replay_bucket_sample_size",
    "replay_bucket_sample_gap",
    "replay_bucket_avg_pnl",
    "replay_bucket_profit_factor",
    "diagnostic_replay_status",
    "diagnostic_replay_sample_size",
    "diagnostic_replay_relaxed_dimensions",
    "primary_gap",
    "next_evidence_needed",
    "suggested_action",
    "source_path",
    "note",
]
ROUTE_OPPORTUNITY_GAP_COLUMNS = [
    "strategy_route",
    "strategy_family",
    "current_ticket_count",
    "current_green_count",
    "calibration_pass_rows",
    "calibration_warn_rows",
    "calibration_block_rows",
    "actual_status",
    "actual_sample_size",
    "actual_win_rate",
    "actual_avg_pnl",
    "actual_profit_factor",
    "replay_status",
    "replay_sample_size",
    "replay_win_rate",
    "replay_avg_pnl",
    "replay_profit_factor",
    "route_status",
    "development_gap",
    "suggested_action",
    "source_path",
    "note",
]
MARKET_OPEN_RECHECK_COLUMNS = [
    "recommendation_rank",
    "ticker",
    "status_icon",
    "status_label",
    "entry_type",
    "order_readiness",
    "target_order_status",
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
    "underlying_quality_tier",
    "underlying_quality_reason",
    "trade_plan",
    "required_recheck",
    "recheck_action",
    "execution_blockers",
]
LIVE_SPREAD_QUALITY_AUDIT_COLUMNS = [
    "recommendation_rank",
    "ticker",
    "live_market_quality_status",
    "actionability_impact",
    "recommendation_status",
    "live_validation_status",
    "structure",
    "entry_type",
    "entry_limit",
    "target_entry",
    "spot_live",
    "short_strike",
    "long_strike",
    "spread_width",
    "live_quote_width_pct",
    "live_leg_min_liquidity",
    "live_leg_liquidity_status",
    "quality_gate_reason",
    "trade_plan",
]
EXECUTION_FILL_QUALITY_COLUMNS = [
    "recommendation_rank",
    "ticker",
    "action_surface",
    "fill_quality_status",
    "entry_type",
    "entry_limit",
    "target_entry",
    "price_improvement_vs_target",
    "slippage_vs_target",
    "live_quote_width_pct",
    "live_leg_min_liquidity",
    "live_validation_status",
    "target_order_status",
    "ready_to_enter",
    "reason",
    "trade_plan",
]
EXTERNAL_REVIEW_COLUMNS = [
    "candidate_id",
    "ticker",
    "agent",
    "agent_type",
    "review_stage",
    "verdict",
    "confidence",
    "note",
    "objective_blocker",
    "blocker_type",
    "portfolio_risk_only",
    "evidence",
    "source_artifact",
    "as_of",
]
AGENT_REVIEW_COLUMNS = [
    "candidate_id",
    "ticker",
    "agent",
    "agent_type",
    "review_stage",
    "verdict",
    "confidence",
    "objective_blocker",
    "blocker_type",
    "portfolio_risk_only",
    "note",
    "evidence",
    "source_artifact",
    "as_of",
]

PORTFOLIO_REJECT_TERMS = (
    "portfolio",
    "existing_exposure",
    "existing exposure",
    "sector_crowding",
    "sector crowding",
    "portfolio_concentration",
    "portfolio concentration",
    "portfolio_correlation",
    "portfolio correlation",
    "correlated exposure",
)


class RecommendationStatus(str, Enum):
    """Recommendation states emitted by Options Agent."""

    ENTER = "ENTER"
    ENTER_WITH_PORTFOLIO_RISK = "ENTER_WITH_PORTFOLIO_RISK"
    WAIT_FOR_PRICE = "WAIT_FOR_PRICE"
    REVIEW = "REVIEW"
    AVOID = "AVOID"


@dataclass(frozen=True)
class AgentSpec:
    """Static description of an Options Agent role."""

    name: str
    role: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]


def agent_roster() -> list[dict[str, Any]]:
    """Return the initial multi-agent roster as serializable dictionaries."""

    specs = [
        AgentSpec(
            name="data",
            role="Verify dated UW sources and normalize raw market data.",
            inputs=("dated UW folder", "Schwab positions", "live quotes"),
            outputs=("source_inventory.json", "raw_universe.csv"),
        ),
        AgentSpec(
            name="flow_oi",
            role="Score flow, OI, repeat activity, and directional pressure.",
            inputs=("raw_universe.csv", "multi-day UW history"),
            outputs=("candidate_generation.csv", "agent_reviews/flow_oi.md"),
        ),
        AgentSpec(
            name="market_regime",
            role="Classify broad market conditions and structure preference.",
            inputs=("index quotes", "volatility proxy", "macro calendar"),
            outputs=("agent_reviews/market_regime.md",),
        ),
        AgentSpec(
            name="catalyst",
            role="Check earnings, news, analyst, filing, and macro event risk.",
            inputs=("candidate_generation.csv", "news/calendar context"),
            outputs=("catalyst_evidence.csv", "catalyst_reviews.csv", "agent_reviews/catalyst.md"),
        ),
        AgentSpec(
            name="research_dispatch",
            role="Package top setups for external or subagent review and normalize returned reviews.",
            inputs=("candidate_generation.csv", "market_regime.json", "catalyst_evidence.csv", "catalyst_reviews.csv"),
            outputs=("research_tasks.json", "external_agent_reviews.csv"),
        ),
        AgentSpec(
            name="structure",
            role="Construct executable option tickets and compute payoff math.",
            inputs=("candidate_generation.csv", "external_agent_reviews.csv", "Schwab option chains"),
            outputs=("priced_candidates.csv", "live_spread_quality_audit.csv", "structure_attempts.csv", "agent_reviews/structure.md"),
        ),
        AgentSpec(
            name="portfolio_risk",
            role="Annotate exposure, concentration, and buying-power risks.",
            inputs=("priced_candidates.csv", "Schwab positions"),
            outputs=("risk_audit.csv", "agent_reviews/portfolio_risk.md"),
        ),
        AgentSpec(
            name="sizing",
            role="Calculate suggested contract count and account-risk annotations without suppressing trades.",
            inputs=("final_recommendations.csv", "options_agent_portfolio_context.json"),
            outputs=("sizing_audit.csv", "agent_reviews/sizing.md"),
        ),
        AgentSpec(
            name="management",
            role="Write entry, target-exit, invalidation, and review-trigger plans for visible recommendations.",
            inputs=("final_recommendations.csv", "decision_board.csv", "sizing_audit.csv"),
            outputs=("management_plan.csv", "agent_reviews/management.md"),
        ),
        AgentSpec(
            name="skeptic",
            role="Try to invalidate each trade and preserve near-miss reasons.",
            inputs=("priced_candidates.csv", "risk_audit.csv", "sizing_audit.csv", "external_agent_reviews.csv", "agent reviews"),
            outputs=("no_trade_audit.csv", "agent_reviews/skeptic.md"),
        ),
        AgentSpec(
            name="synthesis",
            role="Rank visible recommendations and write the final board.",
            inputs=("all agent outputs", "external_agent_reviews.csv"),
            outputs=("final_recommendations.csv", "options_agent_report.md"),
        ),
    ]
    return [asdict(spec) for spec in specs]


def parse_as_of(value: str | dt.date) -> dt.date:
    """Parse an as-of date."""

    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value))


def default_output_dir(root: Optional[Path], as_of: str | dt.date) -> Path:
    """Return the default Options Agent output directory."""

    resolved_root = root or project_root()
    day = parse_as_of(as_of).isoformat()
    return resolved_root / "out" / DEFAULT_OUTPUT_NAMESPACE / day


def output_paths(
    as_of: str | dt.date,
    root: Optional[Path] = None,
    out_dir: Optional[Path] = None,
) -> dict[str, Path]:
    """Build the standard artifact paths for an Options Agent run."""

    day = parse_as_of(as_of).isoformat()
    resolved_out = out_dir or default_output_dir(root, day)
    reviews = resolved_out / "agent_reviews"
    return {
        "out_dir": resolved_out,
        "agent_reviews_dir": reviews,
        "manifest": resolved_out / f"options_agent_manifest_{day}.json",
        "report": resolved_out / f"options_agent_report_{day}.md",
        "source_inventory": resolved_out / "source_inventory.json",
        "raw_universe": resolved_out / "raw_universe.csv",
        "market_price_regime": resolved_out / "market_price_regime.json",
        "market_regime": resolved_out / "market_regime.json",
        "candidate_generation": resolved_out / "candidate_generation.csv",
        "catalyst_evidence": resolved_out / "catalyst_evidence.csv",
        "catalyst_reviews": resolved_out / "catalyst_reviews.csv",
        "research_tasks": resolved_out / "research_tasks.json",
        "agent_dispatch_plan": resolved_out / "agent_dispatch_plan.json",
        "agentic_reviews": resolved_out / "agentic_reviews.json",
        "external_agent_reviews": resolved_out / "external_agent_reviews.csv",
        "agent_review_board": resolved_out / "agent_review_board.csv",
        "structure_attempts": resolved_out / "structure_attempts.csv",
        "strategy_routing_audit": resolved_out / "strategy_routing_audit.csv",
        "priced_candidates": resolved_out / "priced_candidates.csv",
        "live_spread_quality_audit": resolved_out / "live_spread_quality_audit.csv",
        "execution_fill_quality": resolved_out / "execution_fill_quality.csv",
        "live_chain_validation": resolved_out / "live_chain_validation.csv",
        "final_recommendations": resolved_out / "final_recommendations.csv",
        "decision_board": resolved_out / "decision_board.csv",
        "trade_tickets": resolved_out / "trade_tickets.csv",
        "green_trade_tickets": resolved_out / "green_trade_tickets.csv",
        "target_order_candidates": resolved_out / "target_order_candidates.csv",
        "market_open_recheck_queue": resolved_out / "market_open_recheck_queue.csv",
        "coverage_audit": resolved_out / "ticker_coverage_audit.csv",
        "no_trade_audit": resolved_out / "no_trade_audit.csv",
        "risk_audit": resolved_out / "risk_audit.csv",
        "sizing_audit": resolved_out / "sizing_audit.csv",
        "management_plan": resolved_out / "management_plan.csv",
        "execution_readiness": resolved_out / "execution_readiness.csv",
        "expectancy_evidence": resolved_out / "expectancy_evidence.csv",
        "strategy_outcome_atlas": resolved_out / "strategy_outcome_atlas.csv",
        "profitability_calibration": resolved_out / "profitability_calibration.csv",
        "profitability_gap_plan": resolved_out / "profitability_gap_plan.csv",
        "route_opportunity_gap": resolved_out / "route_opportunity_gap.csv",
        "monthly_feasibility": resolved_out / "monthly_feasibility.csv",
        "confidence_audit": resolved_out / "confidence_audit.csv",
        "confidence_audit_json": resolved_out / "confidence_audit.json",
        "lessons_snapshot_md": resolved_out / "lessons_snapshot.md",
        "lessons_snapshot_json": resolved_out / "lessons_snapshot.json",
        "lessons_application_audit": resolved_out / "lessons_application_audit.csv",
        "portfolio_context": resolved_out / "options_agent_portfolio_context.json",
        "agent_orchestration": resolved_out / "agent_orchestration.json",
    }


def resolve_date_dir(base_dir: Path, as_of: str | dt.date) -> Path:
    """Resolve either a trade-desk root or an already dated source directory."""

    day = parse_as_of(as_of).isoformat()
    base = Path(base_dir).expanduser().resolve()
    if base.name == day:
        return base
    return base / day


def apply_portfolio_risk_annotations(
    candidates: Iterable[Mapping[str, Any]],
    portfolio_context: Optional[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Annotate portfolio risk without hiding otherwise qualified trades."""

    portfolio = dict(portfolio_context or {})
    annotated: list[dict[str, Any]] = []
    for candidate in candidates:
        row = dict(candidate)
        ticker = _candidate_ticker(row)
        notes = _portfolio_notes_for_ticker(ticker, portfolio)
        removed_rejects, remaining_rejects = _split_portfolio_rejects(row.get("hard_rejects"))
        if removed_rejects and _is_otherwise_qualified(row, remaining_rejects):
            notes.append("portfolio-only blocker converted to annotation: " + "; ".join(removed_rejects))
            row["hard_rejects"] = "; ".join(remaining_rejects)
        elif remaining_rejects:
            row["hard_rejects"] = "; ".join(remaining_rejects)

        existing_note = str(row.get("portfolio_risk_note") or row.get("portfolio_note") or "").strip()
        if existing_note:
            notes.insert(0, existing_note)

        row["visible_in_final_board"] = True
        row["portfolio_risk_policy"] = "ANNOTATE_ONLY"
        row["portfolio_risk_flag"] = bool(notes)
        row["portfolio_risk_note"] = "; ".join(_dedupe_notes(notes))

        if remaining_rejects:
            row["recommendation_status"] = RecommendationStatus.AVOID.value
            row["status_reason"] = _append_reason(
                row.get("status_reason"),
                "objective hard blocker: " + "; ".join(remaining_rejects),
            )
        elif notes and _is_otherwise_qualified(row, remaining_rejects):
            current_status = str(row.get("recommendation_status") or "").strip().upper()
            if current_status in {"", RecommendationStatus.ENTER.value, "EXECUTE"}:
                row["recommendation_status"] = RecommendationStatus.ENTER_WITH_PORTFOLIO_RISK.value
            else:
                row.setdefault("recommendation_status", current_status)
        else:
            row.setdefault("recommendation_status", _default_status(row, remaining_rejects))

        annotated.append(row)
    return annotated


def build_manifest(
    as_of: str | dt.date,
    root: Optional[Path] = None,
    out_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Create a manifest skeleton for a design/smoke run."""

    day = parse_as_of(as_of).isoformat()
    resolved_root = root or project_root()
    paths = output_paths(day, root=resolved_root, out_dir=out_dir)
    return {
        "pipeline_name": PIPELINE_NAME,
        "pipeline_version": PIPELINE_VERSION,
        "as_of": day,
        "source_root": str(resolved_root),
        "out_dir": str(paths["out_dir"]),
        "agents": agent_roster(),
        "artifacts": {key: str(value) for key, value in paths.items() if key != "out_dir"},
        "visibility_invariant": (
            "portfolio annotations stay in portfolio_risk_note and risk_audit.csv; "
            "they must not hide an otherwise qualified trade"
        ),
        "status_counts": {},
        "warnings": ["live data integrations are not wired in this smoke slice"],
    }


def run_pipeline(
    as_of: str | dt.date,
    root: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    *,
    top_trades: int = DEFAULT_TOP_TRADES,
    max_bot_rows: Optional[int] = None,
    portfolio_context: Optional[Mapping[str, Any]] = None,
    portfolio_json: Optional[Path] = None,
    live_portfolio: bool = False,
    live_schwab: bool = False,
    chain_snapshot_dir: Optional[Path] = None,
    chain_strike_count: int = 80,
    agent_reviews_json: Optional[Path] = None,
    dispatch_only: bool = False,
    lesson_pack_version: Optional[str] = None,
    lesson_pack_path: Optional[Path] = None,
) -> dict[str, Path]:
    """Run the first independent Options Agent EOD research path."""

    day = parse_as_of(as_of).isoformat()
    resolved_root = root or project_root()
    date_dir = resolve_date_dir(resolved_root, day)
    paths = output_paths(day, root=resolved_root, out_dir=out_dir)
    paths["out_dir"].mkdir(parents=True, exist_ok=True)
    paths["agent_reviews_dir"].mkdir(parents=True, exist_ok=True)
    if lesson_pack_path is not None:
        from uwos.lessonengine.core import load_lesson_pack

        lesson_pack = load_lesson_pack(Path(lesson_pack_path).expanduser().resolve())
    else:
        lesson_pack = load_active_lesson_pack(resolved_root, version=lesson_pack_version)
    lesson_metadata = lesson_manifest_metadata(lesson_pack, paths)

    inventory = build_source_inventory(date_dir, day)
    raw_universe, source_notes = build_raw_universe(
        date_dir,
        day,
        discovery_limit=None,
        max_bot_rows=max_bot_rows,
    )
    market_spots, market_spot_meta, market_spot_notes = collect_market_price_spots(
        day,
        paths["out_dir"],
        live_schwab=live_schwab,
        chain_snapshot_dir=chain_snapshot_dir,
        strike_count=chain_strike_count,
    )
    source_notes = source_notes + market_spot_notes
    market_price_regime = build_market_price_regime(
        raw_universe,
        day,
        live_spots=market_spots,
        live_source_meta=market_spot_meta,
        live_schwab_requested=live_schwab,
        chain_snapshot_dir=chain_snapshot_dir,
    )
    raw_universe = annotate_macro_tape_candidates(raw_universe, market_price_regime)
    market_regime = build_market_regime(raw_universe, market_price_regime=market_price_regime)
    candidates = generate_candidates(
        raw_universe,
        limit=None,
        focus_tickers=CORE_AUDIT_TICKERS,
        market_price_regime=market_price_regime,
    )
    catalyst_evidence = build_catalyst_evidence(date_dir, day, candidates)
    catalyst_reviews = build_catalyst_reviews(date_dir, day, candidates, catalyst_evidence=catalyst_evidence)
    research_tasks = build_research_tasks(
        candidates,
        market_regime,
        catalyst_reviews,
        top_trades=top_trades,
        lesson_pack=lesson_pack,
    )
    agent_dispatch_plan = build_agent_dispatch_plan(research_tasks, day, paths, lesson_pack=lesson_pack)
    if dispatch_only:
        coverage_audit = build_coverage_audit(
            raw_universe,
            candidates,
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )
        manifest = build_manifest(day, root=resolved_root, out_dir=paths["out_dir"])
        dispatch_execution_context = build_execution_context(
            live_schwab=False,
            chain_snapshot_dir=None,
            portfolio_context=unavailable_portfolio_context("dispatch-only pass"),
            research_task_count=len(research_tasks.get("tasks", [])),
            external_review_count=0,
            agent_reviews_json=None,
        )
        dispatch_decision = synthesize_decision_board(pd.DataFrame(), market_regime=market_regime)
        dispatch_tickets = build_trade_tickets(dispatch_decision)
        dispatch_readiness = build_execution_readiness(dispatch_decision, dispatch_execution_context)
        dispatch_expectancy = pd.DataFrame(columns=EXPECTANCY_EVIDENCE_COLUMNS)
        dispatch_monthly = build_monthly_feasibility(
            dispatch_decision,
            dispatch_tickets,
            dispatch_execution_context,
            dispatch_expectancy,
        )
        dispatch_confidence_audit = build_confidence_audit(
            dispatch_decision,
            dispatch_tickets,
            dispatch_readiness,
            dispatch_expectancy,
            dispatch_monthly,
            dispatch_execution_context,
            profitability_calibration=pd.DataFrame(columns=PROFITABILITY_CALIBRATION_COLUMNS),
        )
        manifest.update(
            {
                "mode": "agentic_dispatch_pass",
                "source_dir": str(date_dir),
                "source_inventory": inventory,
                "status_counts": {},
                "row_counts": {
                    "raw_universe": int(len(raw_universe)),
                    "candidate_generation": int(len(candidates)),
                    "catalyst_evidence": int(len(catalyst_evidence)),
                    "catalyst_reviews": int(len(catalyst_reviews)),
                    "research_tasks": len(research_tasks.get("tasks", [])),
                    "agent_dispatch_tasks": len(agent_dispatch_plan.get("subagent_tasks", [])),
                    "external_agent_reviews": 0,
                    "agent_review_board": 0,
                    "structure_attempts": 0,
                    "strategy_routing_audit": 0,
                    "priced_candidates": 0,
                    "live_spread_quality_audit": 0,
                    "execution_fill_quality": 0,
                    "live_chain_validation": 0,
                    "final_recommendations": 0,
                    "decision_board": 0,
                    "trade_tickets": 0,
                    "green_trade_tickets": 0,
                    "target_order_ticket_rows": 0,
                    "market_open_recheck_queue": 0,
                    "coverage_audit": int(len(coverage_audit)),
                    "no_trade_audit": 0,
                    "risk_audit": 0,
                    "sizing_audit": 0,
                    "management_plan": 0,
                    "ready_to_enter": 0,
                    "target_order_candidates": 0,
                    "execution_readiness": 0,
                    "expectancy_evidence": 0,
                    "strategy_outcome_atlas": 0,
                    "profitability_calibration": 0,
                    "profitability_gap_plan": 0,
                    "route_opportunity_gap": 0,
                    "monthly_feasibility": 0,
                    "confidence_audit": int(len(dispatch_confidence_audit)),
                },
                "portfolio_context_status": "not_loaded_dispatch_only",
                "market_price_regime": market_price_regime,
                "market_regime": market_regime,
                "confidence_audit_summary": summarize_confidence_audit(dispatch_confidence_audit),
                "route_opportunity_gap_summary": summarize_route_opportunity_gap(
                    pd.DataFrame(columns=ROUTE_OPPORTUNITY_GAP_COLUMNS)
                ),
                "profitability_gap_plan_summary": summarize_profitability_gap_plan(
                    pd.DataFrame(columns=PROFITABILITY_GAP_PLAN_COLUMNS)
                ),
                "agentic_orchestration": {
                    "status": "dispatch_ready",
                    "dispatch_plan": str(paths["agent_dispatch_plan"]),
                    "expected_reviews_json": str(paths["agentic_reviews"]),
                    "ingested_reviews_json": "",
                    "subagent_task_count": len(agent_dispatch_plan.get("subagent_tasks", [])),
                    "ingested_review_count": 0,
                    "runner": "Codex options-agent skill with multi_agent_v1",
                },
                "agent_review_summary": summarize_agent_reviews(pd.DataFrame(columns=AGENT_REVIEW_COLUMNS)),
                "lessonengine": lesson_metadata,
                **lesson_metadata,
                "warnings": source_notes
                + [
                    "dispatch-only pass complete; spawn subagents, write agentic_reviews.json, then rerun synthesis"
                ],
            }
        )
        _write_dispatch_only_artifacts(
            paths,
            manifest,
            inventory,
            market_price_regime,
            market_regime,
            research_tasks,
            agent_dispatch_plan,
            raw_universe,
            candidates,
            catalyst_evidence,
            catalyst_reviews,
            coverage_audit,
            source_notes,
            day,
        )
        write_lesson_snapshots(lesson_pack, paths)
        _write_frame(
            build_application_audit(pd.DataFrame(), pd.DataFrame(), lesson_pack),
            paths["lessons_application_audit"],
        )
        return paths
    external_agent_reviews, review_notes = load_external_agent_reviews(agent_reviews_json)
    priced, strategy_routing_audit = price_candidates_with_routing_audit(date_dir, day, candidates, root=resolved_root)
    priced = apply_catalyst_reviews(priced, catalyst_reviews)
    dated_priced = priced.copy()
    live_validation = empty_live_validation_frame()
    validation_notes: list[str] = []
    live_market_session_open = (
        is_regular_market_session_open()
        if live_schwab and chain_snapshot_dir is None
        else None
    )
    if live_schwab or chain_snapshot_dir is not None:
        priced, live_validation, validation_notes = validate_priced_candidates_live(
            priced,
            day,
            paths["out_dir"],
            chain_snapshot_dir=chain_snapshot_dir,
            strike_count=chain_strike_count,
            allow_live_fallback=bool(live_schwab),
            market_session_open=live_market_session_open,
        )
    structure_attempts = build_structure_attempts(dated_priced, priced, live_validation)
    pre_portfolio_agent_reviews = build_internal_agent_reviews(candidates, market_regime, catalyst_reviews, priced, as_of=day)
    actionable_agent_reviews = combine_agent_reviews(pre_portfolio_agent_reviews, external_agent_reviews, as_of=day)
    priced = apply_agent_reviews(priced, actionable_agent_reviews)
    resolved_portfolio, portfolio_notes = resolve_portfolio_context(
        paths["out_dir"],
        portfolio_context=portfolio_context,
        portfolio_json=portfolio_json,
        live_portfolio=live_portfolio,
    )
    annotated = apply_portfolio_risk_annotations(priced.to_dict("records"), resolved_portfolio)
    sized = apply_position_sizing(annotated, resolved_portfolio, market_regime)
    final = pd.DataFrame(sized)
    final = annotate_actual_forward_expectancy(final, resolved_root)
    if not final.empty:
        final = final.copy()
        run_regime = _regime_bucket(market_regime.get("regime") if isinstance(market_regime, Mapping) else "")
        if "regime" not in final.columns:
            final["regime"] = run_regime
        else:
            existing_regime = final["regime"].astype(str).str.strip()
            final["regime"] = existing_regime.where(existing_regime.ne(""), run_regime).map(_regime_bucket)
    resolved_out_root = resolved_root if resolved_root.name == "out" else resolved_root / "out"
    shared_actual_frame = _actual_calibration_frame(resolved_root, resolved_out_root) if not final.empty else pd.DataFrame()
    shared_replay_bundle = (
        _profitability_replay_frame(resolved_out_root, as_of=parse_as_of(day))
        if not final.empty
        else (pd.DataFrame(), "", "")
    )
    profitability_calibration = build_profitability_calibration(
        resolved_root,
        final,
        as_of_date=day,
        actual_frame=shared_actual_frame,
        replay_bundle=shared_replay_bundle,
    )
    final = annotate_profitability_calibration(final, profitability_calibration)
    live_spread_quality_audit = build_live_spread_quality_audit(final)
    execution_context = build_execution_context(
        live_schwab=live_schwab,
        chain_snapshot_dir=chain_snapshot_dir,
        portfolio_context=resolved_portfolio,
        research_task_count=len(research_tasks.get("tasks", [])),
        external_review_count=len(external_agent_reviews),
        external_review_agent_count=_distinct_external_review_agent_count(external_agent_reviews),
        agent_dispatch_task_count=len(agent_dispatch_plan.get("subagent_tasks", [])),
        agent_reviews_json=agent_reviews_json,
        market_session_open=live_market_session_open,
    )

    portfolio_agent_reviews = build_portfolio_agent_reviews(final, as_of=day)
    agent_review_board = combine_agent_reviews(
        pre_portfolio_agent_reviews,
        portfolio_agent_reviews,
        external_agent_reviews,
        as_of=day,
    )
    final = apply_synthesis_ranking(
        final,
        agent_review_board,
        top_trades=top_trades,
        lesson_pack=lesson_pack,
        execution_context=execution_context,
    )
    no_trade = build_no_trade_audit(candidates, priced, top_trades=top_trades, raw_universe=raw_universe)
    risk_audit = build_risk_audit(final)
    sizing_audit = build_sizing_audit(final)
    decision_board = synthesize_decision_board(final, market_regime=market_regime, execution_context=execution_context)
    lessons_application_audit = build_application_audit(final, decision_board, lesson_pack)
    management_plan = build_management_plan(final, decision_board)
    trade_tickets = build_trade_tickets(decision_board)
    green_trade_tickets, target_order_tickets = split_trade_ticket_surfaces(trade_tickets)
    execution_fill_quality = build_execution_fill_quality_audit(final, trade_tickets)
    market_open_recheck_queue = build_market_open_recheck_queue(trade_tickets)
    execution_readiness = build_execution_readiness(decision_board, execution_context)
    expectancy_evidence = build_expectancy_evidence(resolved_root, decision_board, trade_tickets)
    strategy_outcome_atlas = build_strategy_outcome_atlas(resolved_root, decision_board, trade_tickets)
    route_opportunity_gap = build_route_opportunity_gap(
        resolved_root,
        decision_board,
        trade_tickets,
        profitability_calibration,
        as_of_date=day,
        actual_frame=shared_actual_frame,
        replay_bundle=shared_replay_bundle,
    )
    profitability_gap_plan = build_profitability_gap_plan(profitability_calibration)
    monthly_feasibility = build_monthly_feasibility(decision_board, trade_tickets, execution_context, expectancy_evidence)
    confidence_audit = build_confidence_audit(
        decision_board,
        trade_tickets,
        execution_readiness,
        expectancy_evidence,
        monthly_feasibility,
        execution_context,
        profitability_calibration=profitability_calibration,
        execution_fill_quality=execution_fill_quality,
    )
    coverage_audit = build_coverage_audit(raw_universe, candidates, priced, decision_board, no_trade)
    manifest = build_manifest(day, root=resolved_root, out_dir=paths["out_dir"])
    live_readiness_notes = []
    if not live_schwab and chain_snapshot_dir is None:
        live_readiness_notes.append(
            "live Schwab validation was not requested; green ready_to_enter trades cannot be produced"
        )
    manifest.update(
        {
            "mode": "agentic_synthesis_pass" if agent_reviews_json else "agentic_local_preview",
            "source_dir": str(date_dir),
            "source_inventory": inventory,
            "status_counts": _status_counts(final),
            "row_counts": {
                "raw_universe": int(len(raw_universe)),
                "candidate_generation": int(len(candidates)),
                "catalyst_evidence": int(len(catalyst_evidence)),
                "catalyst_reviews": int(len(catalyst_reviews)),
                "research_tasks": len(research_tasks.get("tasks", [])),
                "agent_dispatch_tasks": len(agent_dispatch_plan.get("subagent_tasks", [])),
                "external_agent_reviews": int(len(external_agent_reviews)),
                "agent_review_board": int(len(agent_review_board)),
                "structure_attempts": int(len(structure_attempts)),
                "strategy_routing_audit": int(len(strategy_routing_audit)),
                "priced_candidates": int(len(priced)),
                "live_spread_quality_audit": int(len(live_spread_quality_audit)),
                "execution_fill_quality": int(len(execution_fill_quality)),
                "live_chain_validation": int(len(live_validation)),
                "final_recommendations": int(len(final)),
                "decision_board": int(len(decision_board)),
                "trade_tickets": int(len(trade_tickets)),
                "green_trade_tickets": int(len(green_trade_tickets)),
                "target_order_ticket_rows": int(len(target_order_tickets)),
                "market_open_recheck_queue": int(len(market_open_recheck_queue)),
                "coverage_audit": int(len(coverage_audit)),
                "no_trade_audit": int(len(no_trade)),
                "risk_audit": int(len(risk_audit)),
                "sizing_audit": int(len(sizing_audit)),
                "management_plan": int(len(management_plan)),
                "ready_to_enter": _ready_to_enter_count(decision_board),
                "target_order_candidates": _target_order_candidate_count(decision_board),
                "execution_readiness": int(len(execution_readiness)),
                "expectancy_evidence": int(len(expectancy_evidence)),
                "strategy_outcome_atlas": int(len(strategy_outcome_atlas)),
                "profitability_calibration": int(len(profitability_calibration)),
                "profitability_gap_plan": int(len(profitability_gap_plan)),
                "route_opportunity_gap": int(len(route_opportunity_gap)),
                "monthly_feasibility": int(len(monthly_feasibility)),
                "confidence_audit": int(len(confidence_audit)),
                "lessons_application_audit": int(len(lessons_application_audit)),
            },
            "live_schwab_requested": bool(live_schwab),
            "chain_snapshot_dir": str(Path(chain_snapshot_dir).expanduser().resolve()) if chain_snapshot_dir else "",
            "portfolio_context_status": resolved_portfolio.get("status", "unknown"),
            "market_price_regime": market_price_regime,
            "market_regime": market_regime,
            "execution_readiness_summary": summarize_execution_readiness(execution_readiness),
            "expectancy_evidence_summary": summarize_expectancy_evidence(expectancy_evidence),
            "strategy_outcome_atlas_summary": summarize_strategy_outcome_atlas(strategy_outcome_atlas),
            "profitability_calibration_summary": summarize_profitability_calibration(profitability_calibration),
            "profitability_gap_plan_summary": summarize_profitability_gap_plan(profitability_gap_plan),
            "route_opportunity_gap_summary": summarize_route_opportunity_gap(route_opportunity_gap),
            "monthly_feasibility_summary": summarize_monthly_feasibility(monthly_feasibility),
            "confidence_audit_summary": summarize_confidence_audit(confidence_audit),
            "calibrated_order_entry_blocker_summary": summarize_calibrated_order_entry_blockers(decision_board),
            "live_spread_quality_summary": summarize_live_spread_quality(live_spread_quality_audit),
            "execution_fill_quality_summary": summarize_execution_fill_quality(execution_fill_quality),
            "execution_context": execution_context,
            "agentic_orchestration": {
                "status": "reviews_ingested" if not external_agent_reviews.empty else "awaiting_subagents",
                "dispatch_plan": str(paths["agent_dispatch_plan"]),
                "expected_reviews_json": str(paths["agentic_reviews"]),
                "ingested_reviews_json": str(Path(agent_reviews_json).expanduser().resolve()) if agent_reviews_json else "",
                "subagent_task_count": len(agent_dispatch_plan.get("subagent_tasks", [])),
                "ingested_review_count": int(len(external_agent_reviews)),
                "runner": "Codex options-agent skill with multi_agent_v1",
            },
            "agent_review_summary": summarize_agent_reviews(agent_review_board),
            "lessonengine": lesson_metadata,
            **lesson_metadata,
            "warnings": source_notes
            + validation_notes
            + review_notes
            + portfolio_notes
            + live_readiness_notes
            + ([] if agent_reviews_json else ["agentic subagent reviews not ingested yet; run the skill two-pass dispatch"])
            + (
                []
                if not agent_reviews_json
                or execution_context.get("agentic_review_coverage_pct", 0) >= MIN_AGENTIC_REVIEW_COVERAGE
                else ["agentic review lane coverage is below the execution threshold; target tickets remain visible but green orders are blocked"]
            )
            + (
                []
                if not agent_reviews_json
                or execution_context.get("broad_review_coverage_pct", 0) >= MIN_AGENTIC_REVIEW_COVERAGE
                else ["broad research-task coverage is low; execution readiness is based on subagent lane coverage plus per-ticket lane coverage"]
            )
            + [
                "fresh quote validation is required before manual order entry"
            ],
        }
    )

    _write_json(paths["manifest"], manifest)
    _write_json(paths["agent_orchestration"], build_agent_orchestration(manifest))
    _write_json(paths["source_inventory"], inventory)
    _write_json(paths["market_price_regime"], market_price_regime)
    _write_json(paths["market_regime"], market_regime)
    _write_json(paths["research_tasks"], research_tasks)
    _write_json(paths["agent_dispatch_plan"], agent_dispatch_plan)
    if agent_reviews_json is None:
        _write_json(paths["agentic_reviews"], {"reviews": []})
    else:
        _write_json(paths["agentic_reviews"], {"reviews": external_agent_reviews.to_dict("records")})
    _write_frame(raw_universe, paths["raw_universe"])
    _write_frame(candidates, paths["candidate_generation"])
    _write_frame(catalyst_evidence, paths["catalyst_evidence"])
    _write_frame(catalyst_reviews, paths["catalyst_reviews"])
    _write_frame(external_agent_reviews, paths["external_agent_reviews"])
    _write_frame(agent_review_board, paths["agent_review_board"])
    _write_frame(structure_attempts, paths["structure_attempts"])
    _write_frame(strategy_routing_audit, paths["strategy_routing_audit"])
    _write_frame(priced, paths["priced_candidates"])
    _write_frame(live_spread_quality_audit, paths["live_spread_quality_audit"])
    _write_frame(execution_fill_quality, paths["execution_fill_quality"])
    _write_frame(live_validation, paths["live_chain_validation"])
    _write_frame(final, paths["final_recommendations"])
    _write_frame(decision_board, paths["decision_board"])
    _write_frame(trade_tickets, paths["trade_tickets"])
    _write_frame(green_trade_tickets, paths["green_trade_tickets"])
    _write_frame(target_order_tickets, paths["target_order_candidates"])
    _write_frame(market_open_recheck_queue, paths["market_open_recheck_queue"])
    _write_frame(coverage_audit, paths["coverage_audit"])
    _write_frame(no_trade, paths["no_trade_audit"])
    _write_frame(risk_audit, paths["risk_audit"])
    _write_frame(sizing_audit, paths["sizing_audit"])
    _write_frame(management_plan, paths["management_plan"])
    _write_frame(execution_readiness, paths["execution_readiness"])
    _write_frame(expectancy_evidence, paths["expectancy_evidence"])
    _write_frame(strategy_outcome_atlas, paths["strategy_outcome_atlas"])
    _write_frame(profitability_calibration, paths["profitability_calibration"])
    _write_frame(profitability_gap_plan, paths["profitability_gap_plan"])
    _write_frame(route_opportunity_gap, paths["route_opportunity_gap"])
    _write_frame(monthly_feasibility, paths["monthly_feasibility"])
    _write_frame(confidence_audit, paths["confidence_audit"])
    _write_json(paths["confidence_audit_json"], summarize_confidence_audit(confidence_audit))
    write_lesson_snapshots(lesson_pack, paths)
    _write_frame(lessons_application_audit, paths["lessons_application_audit"])
    _write_json(paths["portfolio_context"], resolved_portfolio)
    write_agent_reviews(paths["agent_reviews_dir"], manifest, source_notes, agent_review_board)
    paths["report"].write_text(render_report(day, decision_board, no_trade, manifest, coverage_audit), encoding="utf-8")
    return paths


def _write_dispatch_only_artifacts(
    paths: Mapping[str, Path],
    manifest: Mapping[str, Any],
    inventory: Mapping[str, Any],
    market_price_regime: Mapping[str, Any],
    market_regime: Mapping[str, Any],
    research_tasks: Mapping[str, Any],
    agent_dispatch_plan: Mapping[str, Any],
    raw_universe: pd.DataFrame,
    candidates: pd.DataFrame,
    catalyst_evidence: pd.DataFrame,
    catalyst_reviews: pd.DataFrame,
    coverage_audit: pd.DataFrame,
    source_notes: Sequence[str],
    day: str,
) -> None:
    """Write first-pass agentic dispatch artifacts and empty synthesis shells."""

    _write_json(paths["manifest"], manifest)
    _write_json(paths["agent_orchestration"], build_agent_orchestration(manifest))
    _write_json(paths["source_inventory"], inventory)
    _write_json(paths["market_price_regime"], market_price_regime)
    _write_json(paths["market_regime"], market_regime)
    _write_json(paths["research_tasks"], research_tasks)
    _write_json(paths["agent_dispatch_plan"], agent_dispatch_plan)
    _write_json(paths["agentic_reviews"], {"reviews": []})
    _write_frame(raw_universe, paths["raw_universe"])
    _write_frame(candidates, paths["candidate_generation"])
    _write_frame(catalyst_evidence, paths["catalyst_evidence"])
    _write_frame(catalyst_reviews, paths["catalyst_reviews"])
    _write_frame(pd.DataFrame(columns=EXTERNAL_REVIEW_COLUMNS), paths["external_agent_reviews"])
    _write_frame(pd.DataFrame(columns=AGENT_REVIEW_COLUMNS), paths["agent_review_board"])
    _write_frame(build_structure_attempts(pd.DataFrame(), pd.DataFrame(), empty_live_validation_frame()), paths["structure_attempts"])
    _write_frame(pd.DataFrame(columns=STRATEGY_ROUTING_AUDIT_COLUMNS), paths["strategy_routing_audit"])
    _write_frame(pd.DataFrame(columns=["ticker", "structure", "entry_limit", "max_profit", "max_loss"]), paths["priced_candidates"])
    _write_frame(pd.DataFrame(columns=LIVE_SPREAD_QUALITY_AUDIT_COLUMNS), paths["live_spread_quality_audit"])
    _write_frame(pd.DataFrame(columns=EXECUTION_FILL_QUALITY_COLUMNS), paths["execution_fill_quality"])
    _write_frame(empty_live_validation_frame(), paths["live_chain_validation"])
    empty_decision = synthesize_decision_board(pd.DataFrame(), market_regime=market_regime)
    _write_frame(empty_decision, paths["decision_board"])
    empty_tickets = build_trade_tickets(empty_decision)
    _write_frame(empty_tickets, paths["trade_tickets"])
    _write_frame(empty_tickets, paths["green_trade_tickets"])
    _write_frame(empty_tickets, paths["target_order_candidates"])
    _write_frame(build_market_open_recheck_queue(empty_tickets), paths["market_open_recheck_queue"])
    _write_frame(coverage_audit, paths["coverage_audit"])
    _write_frame(pd.DataFrame(columns=["ticker", "recommendation_status", "visible_in_final_board"]), paths["final_recommendations"])
    _write_frame(pd.DataFrame(columns=["ticker", "reason", "hard_blocker"]), paths["no_trade_audit"])
    _write_frame(build_risk_audit(pd.DataFrame()), paths["risk_audit"])
    _write_frame(build_sizing_audit(pd.DataFrame()), paths["sizing_audit"])
    _write_frame(build_management_plan(pd.DataFrame(), empty_decision), paths["management_plan"])
    empty_execution_context = build_execution_context(
        live_schwab=False,
        chain_snapshot_dir=None,
        portfolio_context=unavailable_portfolio_context("dispatch-only pass"),
        research_task_count=len(research_tasks.get("tasks", [])),
        external_review_count=0,
        agent_reviews_json=None,
    )
    empty_execution_readiness = build_execution_readiness(empty_decision, empty_execution_context)
    _write_frame(empty_execution_readiness, paths["execution_readiness"])
    empty_expectancy = pd.DataFrame(columns=EXPECTANCY_EVIDENCE_COLUMNS)
    _write_frame(empty_expectancy, paths["expectancy_evidence"])
    _write_frame(pd.DataFrame(columns=STRATEGY_OUTCOME_ATLAS_COLUMNS), paths["strategy_outcome_atlas"])
    empty_calibration = pd.DataFrame(columns=PROFITABILITY_CALIBRATION_COLUMNS)
    _write_frame(empty_calibration, paths["profitability_calibration"])
    _write_frame(pd.DataFrame(columns=PROFITABILITY_GAP_PLAN_COLUMNS), paths["profitability_gap_plan"])
    _write_frame(pd.DataFrame(columns=ROUTE_OPPORTUNITY_GAP_COLUMNS), paths["route_opportunity_gap"])
    empty_monthly = build_monthly_feasibility(empty_decision, pd.DataFrame(), empty_execution_context, empty_expectancy)
    _write_frame(empty_monthly, paths["monthly_feasibility"])
    empty_confidence = build_confidence_audit(
        empty_decision,
        empty_tickets,
        empty_execution_readiness,
        empty_expectancy,
        empty_monthly,
        empty_execution_context,
        profitability_calibration=empty_calibration,
    )
    _write_frame(empty_confidence, paths["confidence_audit"])
    _write_json(paths["confidence_audit_json"], summarize_confidence_audit(empty_confidence))
    _write_json(paths["portfolio_context"], unavailable_portfolio_context("dispatch-only pass"))
    write_agent_reviews(paths["agent_reviews_dir"], manifest, source_notes, pd.DataFrame(columns=AGENT_REVIEW_COLUMNS))
    paths["report"].write_text(render_report(day, empty_decision, pd.DataFrame(), manifest, coverage_audit), encoding="utf-8")


def build_source_inventory(date_dir: Path, as_of: str | dt.date) -> dict[str, Any]:
    """Inventory dated UW inputs without relying on any prior pipeline output."""

    from codexuw import data as uw_data

    day = parse_as_of(as_of).isoformat()
    sources: dict[str, dict[str, Any]] = {}
    for label, prefix in {
        "stock_screener": "stock-screener-",
        "hot_chains": "hot-chains-",
        "chain_oi": "chain-oi-changes-",
        "bot_eod": "bot-eod-report-",
    }.items():
        try:
            path = uw_data.find_export(date_dir, prefix)
            sources[label] = {"status": "present", "path": str(path), "size_bytes": path.stat().st_size}
        except Exception as exc:
            sources[label] = {"status": "missing", "error": str(exc)}
    return {"as_of": day, "source_dir": str(date_dir), "sources": sources}


def build_raw_universe(
    date_dir: Path,
    as_of: str | dt.date,
    *,
    discovery_limit: Optional[int] = None,
    max_bot_rows: Optional[int] = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Build a ticker-level raw universe directly from dated UW source files."""

    from codexuw import data as uw_data

    day = parse_as_of(as_of)
    notes: list[str] = []
    screener = uw_data.load_stock_screener(date_dir)
    hot = uw_data.load_hot_chains(date_dir, day)
    chain_oi = uw_data.load_chain_oi(date_dir, day)

    screen_cols = [
        col
        for col in [
            "ticker",
            "close",
            "prev_close",
            "high",
            "low",
            "week_52_high",
            "week_52_low",
            "sector",
            "call_volume",
            "put_volume",
            "call_premium",
            "put_premium",
            "bullish_premium",
            "bearish_premium",
            "flow_total_premium",
            "flow_bias",
            "iv_rank",
            "iv30d",
            "next_earnings_dt",
            "marketcap",
            "issue_type",
            "is_index",
            "full_name",
            "total_volume",
            "avg30_volume",
            "total_open_interest",
        ]
        if col in screener.columns
    ]
    universe = screener[screen_cols].copy()
    universe = universe.rename(columns={"flow_bias": "screen_flow_bias", "flow_total_premium": "screen_total_premium"})

    hot_agg = aggregate_hot_chains(hot)
    oi_agg = aggregate_chain_oi(chain_oi)
    universe = universe.merge(hot_agg, on="ticker", how="outer").merge(oi_agg, on="ticker", how="outer")
    universe["ticker"] = universe["ticker"].astype(str).str.upper().str.strip()

    seed = universe.copy()
    seed["seed_premium"] = seed[["screen_total_premium", "hot_total_premium"]].fillna(0).max(axis=1)
    tickers = seed.sort_values("seed_premium", ascending=False)["ticker"].dropna().tolist()
    if discovery_limit is not None:
        tickers = tickers[: int(discovery_limit)]
    try:
        bot = uw_data.aggregate_bot_flow(date_dir, tickers, max_rows=max_bot_rows)
        universe = universe.merge(bot, on="ticker", how="left")
    except Exception as exc:
        notes.append(f"bot_eod aggregation unavailable: {exc}")

    for col in ("screen_flow_bias", "hot_flow_bias", "bot_flow_bias"):
        if col not in universe.columns:
            universe[col] = math.nan
    for col in ("screen_total_premium", "hot_total_premium", "bot_total_premium", "positive_oi_change"):
        if col not in universe.columns:
            universe[col] = 0.0

    universe["combined_flow_bias"] = universe.apply(_combined_flow_bias, axis=1)
    universe["bias"] = universe["combined_flow_bias"].map(_bias_label)
    universe["flow_bias_label"] = universe["bias"]
    universe["signal_premium"] = universe[["screen_total_premium", "hot_total_premium", "bot_total_premium"]].fillna(0).max(axis=1)
    quality_rows = universe.apply(
        lambda row: pd.Series(_underlying_quality(row), index=["underlying_quality_tier", "underlying_quality_reason"]),
        axis=1,
    )
    universe = pd.concat([universe, quality_rows], axis=1)
    universe["underlying_quality_rank"] = universe["underlying_quality_tier"].map(_underlying_quality_sort_rank)
    universe["score"] = universe.apply(_score_universe_row, axis=1)
    universe["quality_status"] = universe.apply(_quality_status, axis=1)
    universe["flow_reason"] = universe.apply(_flow_reason, axis=1)
    universe = universe.sort_values(
        ["underlying_quality_rank", "score", "signal_premium"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    return universe, notes


def collect_market_price_spots(
    as_of: str | dt.date,
    out_dir: Path,
    *,
    live_schwab: bool = False,
    chain_snapshot_dir: Optional[Path] = None,
    strike_count: int = 80,
) -> tuple[dict[str, float], dict[str, Any], list[str]]:
    """Fetch early broad-market spot prices when a live/snapshot source is requested."""

    if not live_schwab and chain_snapshot_dir is None:
        return {}, {"source_mode": "uw_eod_only", "source": "UW stock screener close/prev_close"}, []
    try:
        from codexuw.schwab_live import SchwabChainValidator, chain_spot
    except Exception as exc:
        return {}, {"source_mode": "uw_eod_only", "source": "UW stock screener close/prev_close"}, [
            f"early market-price Schwab import unavailable: {exc}"
        ]

    day = parse_as_of(as_of)
    validator = SchwabChainValidator(
        out_dir / "market_price_regime_live",
        strike_count=int(strike_count),
        snapshot_dir=chain_snapshot_dir,
        allow_live_fallback=bool(live_schwab),
    )
    query_from = day + dt.timedelta(days=MIN_LIVE_DTE)
    query_to = day + dt.timedelta(days=MAX_LIVE_DTE)
    spots: dict[str, float] = {}
    notes: list[str] = []
    for ticker in CORE_AUDIT_TICKERS:
        chain = validator.get_chain(ticker, query_from, query_to)
        if not chain:
            continue
        spot = chain_spot(chain)
        if math.isfinite(spot) and spot > 0:
            spots[ticker] = float(spot)
    try:
        validator.save()
    except Exception as exc:
        notes.append(f"early market-price chain snapshot save failed: {exc}")
    if not spots:
        notes.append("early market-price live/snapshot spots unavailable; market_price_regime fell back to UW EOD close/prev_close")
    source_mode = "schwab_live" if live_schwab and chain_snapshot_dir is None else "schwab_snapshot"
    return (
        spots,
        {
            "source_mode": source_mode if spots else "uw_eod_only",
            "source": "Schwab option chain underlying quote" if spots else "UW stock screener close/prev_close",
            "requested_tickers": list(CORE_AUDIT_TICKERS),
            "spot_tickers": sorted(spots),
            "chain_sources": validator.sources,
            "chain_errors": validator.errors,
        },
        notes,
    )


def build_market_price_regime(
    raw_universe: pd.DataFrame,
    as_of: str | dt.date,
    *,
    live_spots: Optional[Mapping[str, float]] = None,
    live_source_meta: Optional[Mapping[str, Any]] = None,
    live_schwab_requested: bool = False,
    chain_snapshot_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Classify broad market tape from price moves, separate from option-flow regime."""

    day = parse_as_of(as_of).isoformat()
    live_spots = {str(k).upper(): float(v) for k, v in dict(live_spots or {}).items() if _as_float(v) is not None}
    meta = dict(live_source_meta or {})
    if raw_universe is None or raw_universe.empty:
        return {
            "status": "unavailable",
            "as_of": day,
            "source_mode": meta.get("source_mode", "uw_eod_only"),
            "reason": "raw universe empty",
            "tape_direction": "unavailable",
            "regime": "unknown",
            "readings": [],
        }
    raw_by_ticker = _frame_by_ticker(raw_universe)
    readings: list[dict[str, Any]] = []
    for ticker in CORE_AUDIT_TICKERS:
        row = raw_by_ticker.get(ticker, {})
        live_spot = _as_float(live_spots.get(ticker))
        close = _as_float(row.get("close"))
        prev_close = _as_float(row.get("prev_close"))
        spot = live_spot if live_spot is not None and live_spot > 0 else close
        source = "schwab_live_or_snapshot" if live_spot is not None and live_spot > 0 else "uw_eod_close"
        move = (spot - prev_close) / prev_close if spot is not None and prev_close is not None and prev_close > 0 else None
        if row or spot is not None:
            readings.append(
                {
                    "ticker": ticker,
                    "source": source,
                    "spot": spot,
                    "close": close,
                    "prev_close": prev_close,
                    "move_pct": round(move, 6) if move is not None else None,
                    "underlying_quality_tier": row.get("underlying_quality_tier", ""),
                    "score": row.get("score", ""),
                }
            )
    index_moves = [
        _as_float(item.get("move_pct"))
        for item in readings
        if item.get("ticker") in CORE_INDEX_ETF_TICKERS and _as_float(item.get("move_pct")) is not None
    ]
    avg_index_move = sum(index_moves) / len(index_moves) if index_moves else None
    if avg_index_move is None:
        tape_direction = "unavailable"
        regime = "unknown"
        sizing = "normal"
        note = "price tape unavailable; using option-flow regime only"
    elif avg_index_move <= -MACRO_TAPE_DIRECTIONAL_MOVE_PCT:
        tape_direction = "bearish"
        regime = "risk_off"
        sizing = "reduced"
        note = "index price tape is bearish"
    elif avg_index_move >= MACRO_TAPE_DIRECTIONAL_MOVE_PCT:
        tape_direction = "bullish"
        regime = "risk_on"
        sizing = "normal"
        note = "index price tape is bullish"
    else:
        tape_direction = "mixed"
        regime = "mixed"
        sizing = "normal"
        note = "index price tape is mixed"
    source_mode = meta.get("source_mode") or ("schwab_live" if live_spots else "uw_eod_only")
    if live_schwab_requested and not live_spots:
        source_mode = "uw_eod_only"
        note = _append_reason(note, "live Schwab market-price snapshot unavailable; using dated UW EOD tape")
    return {
        "status": "ok" if readings else "unavailable",
        "as_of": day,
        "source_mode": source_mode,
        "source": meta.get("source", "UW stock screener close/prev_close"),
        "chain_snapshot_dir": str(Path(chain_snapshot_dir).expanduser().resolve()) if chain_snapshot_dir else "",
        "tape_direction": tape_direction,
        "regime": regime,
        "sizing_stance": sizing,
        "avg_index_move_pct": round(avg_index_move, 6) if avg_index_move is not None else None,
        "directional_threshold_pct": MACRO_TAPE_DIRECTIONAL_MOVE_PCT,
        "note": note,
        "readings": readings,
        "readings_by_ticker": {str(item["ticker"]): item for item in readings},
    }


def annotate_macro_tape_candidates(raw_universe: pd.DataFrame, market_price_regime: Mapping[str, Any]) -> pd.DataFrame:
    """Promote neutral-flow core names into an auditable macro-tape candidate lane."""

    if raw_universe is None or raw_universe.empty:
        return raw_universe.copy() if raw_universe is not None else pd.DataFrame()
    out = raw_universe.copy()
    readings = market_price_regime.get("readings_by_ticker", {}) if isinstance(market_price_regime, Mapping) else {}
    direction = _as_text(market_price_regime.get("tape_direction") if isinstance(market_price_regime, Mapping) else "").lower()
    out["core_universe_member"] = out["ticker"].astype(str).str.upper().map(_is_core_audit_ticker)
    out["price_move_pct"] = out["ticker"].astype(str).str.upper().map(
        lambda ticker: _as_float((readings.get(ticker, {}) or {}).get("move_pct"))
    )
    out["price_tape_source"] = out["ticker"].astype(str).str.upper().map(
        lambda ticker: _as_text((readings.get(ticker, {}) or {}).get("source"))
    )
    out["macro_tape_direction"] = ""
    out["macro_tape_candidate"] = False
    out["candidate_source"] = out.get("candidate_source", pd.Series("flow_oi", index=out.index)).replace("", "flow_oi")
    out["macro_tape_reason"] = ""
    if direction not in {"bearish", "bullish"}:
        return out
    score = pd.to_numeric(out.get("score", pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0)
    tier = out.get("underlying_quality_tier", pd.Series("", index=out.index)).astype(str).str.lower()
    bias = out.get("bias", pd.Series("", index=out.index)).astype(str).str.lower()
    ticker = out.get("ticker", pd.Series("", index=out.index)).astype(str).str.upper()
    marketcap = pd.to_numeric(out.get("marketcap", pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0)
    core_or_mega = ticker.map(_is_core_audit_ticker) | marketcap.ge(MIN_CORE_MARKET_CAP)
    evaluated = tier.eq("core") & core_or_mega
    out.loc[evaluated, "macro_tape_direction"] = direction
    out.loc[evaluated, "macro_tape_reason"] = "core/mega-cap reviewed against " + direction + " price tape"
    promote = bias.eq("neutral") & tier.eq("core") & core_or_mega & score.ge(MACRO_TAPE_CANDIDATE_SCORE_FLOOR)
    if not promote.any():
        return out
    out.loc[promote, "macro_tape_direction"] = direction
    out.loc[promote, "macro_tape_candidate"] = True
    out.loc[promote, "candidate_source"] = "macro_tape_candidate"
    out.loc[promote, "quality_status"] = "qualified"
    out.loc[promote, "bias"] = direction
    out.loc[promote, "macro_tape_reason"] = (
        "neutral UW flow rescued by "
        + direction
        + " price tape; original flow label preserved in flow_bias_label"
    )
    out.loc[promote, "flow_reason"] = out.loc[promote].apply(
        lambda row: _append_reason(row.get("flow_reason"), row.get("macro_tape_reason")),
        axis=1,
    )
    return out


def _is_core_audit_ticker(ticker: Any) -> bool:
    return _as_text(ticker).upper() in set(CORE_AUDIT_TICKERS)


def aggregate_hot_chains(hot: pd.DataFrame) -> pd.DataFrame:
    """Aggregate option-chain activity by ticker."""

    if hot.empty:
        return pd.DataFrame(columns=["ticker"])
    df = hot.copy()
    volume = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0)
    premium = pd.to_numeric(df.get("premium", 0), errors="coerce").fillna(0)
    ask_ratio = pd.to_numeric(df.get("ask_side_volume", 0), errors="coerce").fillna(0) / volume.where(volume > 0)
    bid_ratio = pd.to_numeric(df.get("bid_side_volume", 0), errors="coerce").fillna(0) / volume.where(volume > 0)
    is_call = df.get("right", "").astype(str).str.upper().eq("C")
    is_put = df.get("right", "").astype(str).str.upper().eq("P")
    df["hot_call_premium"] = premium.where(is_call, 0.0)
    df["hot_put_premium"] = premium.where(is_put, 0.0)
    df["hot_bull_premium"] = premium.where(is_call, 0.0).mul(ask_ratio.fillna(0)) + premium.where(is_put, 0.0).mul(
        bid_ratio.fillna(0)
    )
    df["hot_bear_premium"] = premium.where(is_put, 0.0).mul(ask_ratio.fillna(0)) + premium.where(is_call, 0.0).mul(
        bid_ratio.fillna(0)
    )
    df["hot_call_volume"] = volume.where(is_call, 0.0)
    df["hot_put_volume"] = volume.where(is_put, 0.0)
    top = df.sort_values("premium", ascending=False).drop_duplicates("ticker")[
        ["ticker", "option_symbol", "right", "expiry_dt", "strike", "bid", "ask", "dte"]
    ].rename(
        columns={
            "option_symbol": "hot_top_option",
            "right": "hot_top_right",
            "expiry_dt": "hot_top_expiry",
            "strike": "hot_top_strike",
            "bid": "hot_top_bid",
            "ask": "hot_top_ask",
            "dte": "hot_top_dte",
        }
    )
    agg = df.groupby("ticker", as_index=False).agg(
        hot_total_premium=("premium", "sum"),
        hot_volume=("volume", "sum"),
        hot_open_interest=("open_interest", "sum"),
        hot_call_premium=("hot_call_premium", "sum"),
        hot_put_premium=("hot_put_premium", "sum"),
        hot_bull_premium=("hot_bull_premium", "sum"),
        hot_bear_premium=("hot_bear_premium", "sum"),
        hot_call_volume=("hot_call_volume", "sum"),
        hot_put_volume=("hot_put_volume", "sum"),
        hot_avg_dte=("dte", "mean"),
    )
    denom = agg["hot_total_premium"].where(agg["hot_total_premium"].abs() > 0)
    agg["hot_flow_bias"] = (agg["hot_bull_premium"] - agg["hot_bear_premium"]) / denom
    return agg.merge(top, on="ticker", how="left")


def aggregate_chain_oi(chain_oi: pd.DataFrame) -> pd.DataFrame:
    """Aggregate chain OI changes by ticker."""

    if chain_oi.empty:
        return pd.DataFrame(columns=["ticker"])
    df = chain_oi.copy()
    df["ticker"] = df.apply(_clean_chain_oi_ticker, axis=1)
    df = df[df["ticker"].astype(bool)].copy()
    df["positive_oi_change"] = pd.to_numeric(df.get("oi_diff_plain", 0), errors="coerce").fillna(0).clip(lower=0)
    return df.groupby("ticker", as_index=False).agg(
        oi_diff_plain=("oi_diff_plain", "sum"),
        positive_oi_change=("positive_oi_change", "sum"),
        chain_oi_volume=("volume", "sum"),
        curr_oi=("curr_oi", "sum"),
        oi_avg_dte=("dte", "mean"),
    )


def _clean_chain_oi_ticker(row: Mapping[str, Any]) -> str:
    ticker = row.get("ticker")
    if isinstance(ticker, str) and ticker.strip():
        return ticker.strip().upper()
    underlying = row.get("underlying_symbol")
    if isinstance(underlying, str) and underlying.strip():
        return underlying.strip().upper()
    return ""


def generate_candidates(
    raw_universe: pd.DataFrame,
    *,
    limit: Optional[int] = None,
    focus_tickers: Sequence[str] = CORE_AUDIT_TICKERS,
    market_price_regime: Optional[Mapping[str, Any]] = None,
) -> pd.DataFrame:
    """Create the ranked candidate generation table."""

    if raw_universe.empty:
        return raw_universe.copy()
    _ = market_price_regime
    macro_mask = raw_universe.get("macro_tape_candidate", pd.Series(False, index=raw_universe.index)).map(_truthy)
    macro_direction = raw_universe.get("macro_tape_direction", pd.Series("", index=raw_universe.index)).astype(str).str.lower()
    keep = raw_universe[raw_universe["bias"].ne("neutral") | macro_mask].copy()
    if "underlying_quality_rank" not in keep.columns:
        keep["underlying_quality_rank"] = keep.get("underlying_quality_tier", pd.Series("", index=keep.index)).map(
            _underlying_quality_sort_rank
        )
    keep = keep.sort_values(["underlying_quality_rank", "score", "signal_premium"], ascending=[True, False, False])
    if limit is not None:
        keep = keep.head(int(limit))
    if focus_tickers:
        focus_set = {str(ticker).strip().upper() for ticker in focus_tickers if str(ticker).strip()}
        focus = raw_universe[
            raw_universe["ticker"].astype(str).str.upper().isin(focus_set)
            & (raw_universe["bias"].ne("neutral") | macro_mask | macro_direction.isin(["bearish", "bullish"]))
            & raw_universe["quality_status"].eq("qualified")
        ].copy()
        if not focus.empty:
            keep = pd.concat([keep, focus], ignore_index=True).drop_duplicates("ticker", keep="first")
            if "underlying_quality_rank" not in keep.columns:
                keep["underlying_quality_rank"] = keep.get("underlying_quality_tier", pd.Series("", index=keep.index)).map(
                    _underlying_quality_sort_rank
                )
            keep = keep.sort_values(["underlying_quality_rank", "score", "signal_premium"], ascending=[True, False, False])
    keep["candidate_rank"] = range(1, len(keep) + 1)
    keep["recommendation_status"] = keep["quality_status"].map(lambda v: "REVIEW" if v == "qualified" else "AVOID")
    keep["status_reason"] = keep.apply(
        lambda row: _candidate_status_reason(row),
        axis=1,
    )
    return keep.reset_index(drop=True)


def _candidate_status_reason(row: Mapping[str, Any]) -> str:
    if row.get("quality_status") != "qualified":
        return _as_text(row.get("flow_reason"))
    if _truthy(row.get("macro_tape_candidate")):
        return _append_reason(
            "qualified macro-tape research candidate; requires fresh Schwab quote and subagent review",
            row.get("macro_tape_reason"),
        )
    return "qualified EOD research candidate; requires fresh Schwab quote"


def build_market_regime(raw_universe: pd.DataFrame, market_price_regime: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """Build a simple market-regime review from index rows in the raw universe."""

    if raw_universe.empty:
        return {"status": "unavailable", "reason": "raw universe empty", "sizing_stance": "review"}
    index_rows = raw_universe[raw_universe["ticker"].isin(["SPY", "QQQ", "IWM", "DIA"])] if "ticker" in raw_universe.columns else pd.DataFrame()
    readings = []
    for _, row in index_rows.iterrows():
        readings.append(
            {
                "ticker": row.get("ticker", ""),
                "bias": row.get("bias", ""),
                "combined_flow_bias": _as_float(row.get("combined_flow_bias")),
                "signal_premium": _as_float(row.get("signal_premium")) or 0.0,
                "score": _as_float(row.get("score")) or 0.0,
            }
        )
    if not readings:
        return {"status": "unavailable", "reason": "SPY/QQQ/IWM rows not found", "sizing_stance": "normal"}

    avg_bias = sum(float(item["combined_flow_bias"] or 0.0) for item in readings) / len(readings)
    price_regime = dict(market_price_regime or {})
    price_direction = _as_text(price_regime.get("tape_direction")).lower()
    if price_direction == "bullish":
        regime = "risk_on"
        sizing = "normal"
        note = "index price tape leans bullish"
    elif price_direction == "bearish":
        regime = "risk_off"
        sizing = "reduced"
        note = "index price tape leans bearish; prefer smaller size and faster validation"
    elif avg_bias >= 0.08:
        regime = "risk_on"
        sizing = "normal"
        note = "index option flow leans bullish"
    elif avg_bias <= -0.08:
        regime = "risk_off"
        sizing = "reduced"
        note = "index option flow leans bearish; prefer smaller size and faster validation"
    else:
        regime = "mixed"
        sizing = "normal"
        note = "index option flow is mixed"
    return {
        "status": "ok",
        "regime": regime,
        "sizing_stance": sizing,
        "avg_index_flow_bias": round(avg_bias, 4),
        "price_tape_direction": price_direction or "unavailable",
        "avg_index_price_move_pct": price_regime.get("avg_index_move_pct"),
        "note": note,
        "readings": readings,
    }


def build_catalyst_evidence(date_dir: Path, as_of: str | dt.date, candidates: pd.DataFrame) -> pd.DataFrame:
    """Collect auditable earnings and local news evidence for catalyst reviews."""

    columns = [
        "ticker",
        "evidence_type",
        "evidence_status",
        "source",
        "headline",
        "sentiment",
        "red_flag_terms",
        "support_terms",
        "objective_blocker",
        "days_to_earnings",
        "note",
    ]
    if candidates.empty:
        return pd.DataFrame(columns=columns)

    day = parse_as_of(as_of)
    rows: list[dict[str, Any]] = []
    for _, candidate in candidates.iterrows():
        ticker = str(candidate.get("ticker") or "").strip().upper()
        if not ticker:
            continue
        found = False
        earnings = _row_date(candidate.get("next_earnings_dt"))
        days_to_earnings = (earnings - day).days if earnings else None
        if days_to_earnings is not None:
            found = True
            if 0 <= days_to_earnings <= 7:
                status = "event_risk"
            elif 8 <= days_to_earnings <= 21:
                status = "watch_event"
            else:
                status = "clear"
            rows.append(
                {
                    "ticker": ticker,
                    "evidence_type": "earnings",
                    "evidence_status": status,
                    "source": "stock_screener.next_earnings_dt",
                    "headline": "",
                    "sentiment": "event" if status != "clear" else "neutral",
                    "red_flag_terms": "",
                    "support_terms": "",
                    "objective_blocker": False,
                    "days_to_earnings": days_to_earnings,
                    "note": f"earnings in {days_to_earnings} days",
                }
            )

        news_paths = _local_news_paths(date_dir, ticker, day)
        for news_path in news_paths:
            found = True
            evidence = _news_evidence_row(ticker, news_path)
            evidence["days_to_earnings"] = days_to_earnings if days_to_earnings is not None else ""
            rows.append(evidence)

        if not found:
            rows.append(
                {
                    "ticker": ticker,
                    "evidence_type": "summary",
                    "evidence_status": "clear",
                    "source": "",
                    "headline": "",
                    "sentiment": "neutral",
                    "red_flag_terms": "",
                    "support_terms": "",
                    "objective_blocker": False,
                    "days_to_earnings": "",
                    "note": "no local catalyst file or near-term earnings flag",
                }
            )
    return pd.DataFrame(rows, columns=columns)


def build_catalyst_reviews(
    date_dir: Path,
    as_of: str | dt.date,
    candidates: pd.DataFrame,
    *,
    catalyst_evidence: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Review local earnings/news catalyst context for generated candidates."""

    if candidates.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "catalyst_status",
                "catalyst_note",
                "news_source",
                "days_to_earnings",
                "news_sentiment",
                "red_flag_terms",
                "support_terms",
                "objective_blocker",
            ]
        )
    day = parse_as_of(as_of)
    evidence = catalyst_evidence if catalyst_evidence is not None else build_catalyst_evidence(date_dir, day, candidates)
    evidence_by_ticker = {
        ticker: group.copy()
        for ticker, group in evidence.groupby(evidence["ticker"].astype(str).str.upper())
    } if not evidence.empty and "ticker" in evidence.columns else {}
    rows: list[dict[str, Any]] = []
    for _, candidate in candidates.iterrows():
        ticker = str(candidate.get("ticker") or "").strip().upper()
        ticker_evidence = evidence_by_ticker.get(ticker, pd.DataFrame())
        status, notes, source, days_to_earnings, sentiment, red_terms, support_terms, objective = _summarize_catalyst_evidence(
            ticker_evidence
        )
        if not notes:
            notes.append("no local catalyst file or near-term earnings flag")
        rows.append(
            {
                "ticker": ticker,
                "catalyst_status": status,
                "catalyst_note": "; ".join(notes),
                "news_source": source,
                "days_to_earnings": days_to_earnings if days_to_earnings is not None else "",
                "news_sentiment": sentiment,
                "red_flag_terms": red_terms,
                "support_terms": support_terms,
                "objective_blocker": objective,
            }
        )
    return pd.DataFrame(rows)


def apply_catalyst_reviews(priced: pd.DataFrame, catalyst_reviews: pd.DataFrame) -> pd.DataFrame:
    """Attach catalyst review fields without suppressing otherwise visible rows."""

    if priced.empty or catalyst_reviews.empty:
        return priced.copy()
    catalyst_reviews = _dedupe_ticker_rows(catalyst_reviews)
    out = priced.merge(
        catalyst_reviews[
            [
                "ticker",
                "catalyst_status",
                "catalyst_note",
                "days_to_earnings",
                "news_sentiment",
                "red_flag_terms",
                "support_terms",
                "objective_blocker",
            ]
        ],
        on="ticker",
        how="left",
    )
    event_mask = out["catalyst_status"].astype(str).isin(["event_risk", "news_red_flag"])
    if event_mask.any():
        out.loc[event_mask, "recommendation_status"] = RecommendationStatus.REVIEW.value
        out.loc[event_mask, "status_reason"] = out.loc[event_mask].apply(
            lambda row: _append_reason(row.get("status_reason"), f"catalyst review: {row.get('catalyst_note')}"),
            axis=1,
        )
    objective_mask = (
        out["objective_blocker"].map(_truthy) if "objective_blocker" in out.columns else pd.Series(False, index=out.index)
    )
    if objective_mask.any():
        out.loc[objective_mask, "recommendation_status"] = RecommendationStatus.AVOID.value
        out.loc[objective_mask, "hard_rejects"] = out.loc[objective_mask].apply(
            lambda row: _append_reason(row.get("hard_rejects"), "catalyst_objective_blocker"),
            axis=1,
        )
        out.loc[objective_mask, "status_reason"] = out.loc[objective_mask].apply(
            lambda row: _append_reason(row.get("status_reason"), f"catalyst objective blocker: {row.get('catalyst_note')}"),
            axis=1,
        )
    return out


def build_research_tasks(
    candidates: pd.DataFrame,
    market_regime: Mapping[str, Any],
    catalyst_reviews: pd.DataFrame,
    *,
    top_trades: int,
    lesson_pack: Optional[Any] = None,
) -> dict[str, Any]:
    """Create explicit research tasks for external/subagent reviewers."""

    # Kept for API compatibility; research dispatch must cover the full candidate set.
    _ = top_trades
    lesson_digest = str(getattr(lesson_pack, "digest", ""))
    lesson_context = str(getattr(lesson_pack, "markdown", "") or "")
    lesson_ids = [
        str(lesson.get("id") or "")
        for lesson in getattr(lesson_pack, "lessons", [])
        if str(lesson.get("status") or "") == "active"
    ]
    catalyst_by_ticker = _frame_by_ticker(catalyst_reviews)
    tasks: list[dict[str, Any]] = []
    for _, row in candidates.iterrows():
        ticker = str(row.get("ticker") or "").strip().upper()
        catalyst = catalyst_by_ticker.get(ticker, {})
        candidate_id = f"{ticker}:{row.get('bias', '')}:{int(_as_float(row.get('score')) or 0)}"
        tasks.append(
            {
                "candidate_id": candidate_id,
                "ticker": ticker,
                "bias": row.get("bias", ""),
                "flow_bias_label": row.get("flow_bias_label", row.get("bias", "")),
                "candidate_source": row.get("candidate_source", "flow_oi"),
                "macro_tape_candidate": bool(_truthy(row.get("macro_tape_candidate"))),
                "macro_tape_direction": row.get("macro_tape_direction", ""),
                "macro_tape_reason": row.get("macro_tape_reason", ""),
                "price_move_pct": row.get("price_move_pct", ""),
                "price_tape_source": row.get("price_tape_source", ""),
                "score": _as_float(row.get("score")) or 0.0,
                "flow_reason": row.get("flow_reason", ""),
                "market_regime": market_regime.get("regime", "unknown"),
                "catalyst_status": catalyst.get("catalyst_status", ""),
                "catalyst_note": catalyst.get("catalyst_note", ""),
                "lesson_pack_digest": lesson_digest,
                "must_apply_lesson_ids": lesson_ids,
                "reviewer_assignments": [
                    "catalyst_news",
                    "macro_regime",
                    "structure_builder",
                    "skeptic",
                    "portfolio_management",
                ],
                "research_questions": [
                    "Does current news/catalyst context support or invalidate the setup?",
                    "Are there earnings, regulatory, macro, analyst, or company-specific risks before expiry?",
                    "Is the proposed option structure aligned with the thesis and market regime?",
                    "If this is a good setup but portfolio exposure is crowded, call out the risk without hiding it.",
                ],
                "expected_review_schema": {
                    "candidate_id": candidate_id,
                    "ticker": ticker,
                    "agent": "catalyst_news|macro_regime|structure_builder|skeptic|portfolio_management",
                    "agent_type": "subagent",
                    "review_stage": "subagent_review",
                    "verdict": "supportive|caution|avoid",
                    "confidence": "low|medium|high",
                    "note": "short evidence-backed review",
                    "objective_blocker": False,
                    "portfolio_risk_only": False,
                    "evidence": "optional source or observation",
                },
            }
        )
    return {
        "schema_version": "options_agent.dispatch_tasks.v1",
        "dispatch_model": "codex_subagents",
        "review_output_path": "agentic_reviews.json",
        "lesson_pack_digest": lesson_digest,
        "lesson_context": lesson_context,
        "must_apply_lesson_ids": lesson_ids,
        "review_schema": {
            "candidate_id": "stable candidate id from research task",
            "ticker": "ticker",
            "agent": "catalyst_news|macro_regime|structure_builder|skeptic|portfolio_management",
            "agent_type": "subagent",
            "review_stage": "subagent_review",
            "verdict": "supportive|caution|avoid",
            "confidence": "low|medium|high",
            "note": "short evidence-backed review",
            "objective_blocker": False,
            "portfolio_risk_only": False,
            "evidence": "optional source or observation",
        },
        "instructions": (
            "External or subagent reviews may flag objective blockers, but portfolio risk must be annotated "
            "rather than used to hide an otherwise good trade."
        ),
        "tasks": tasks,
    }


def build_agent_dispatch_plan(
    research_tasks: Mapping[str, Any],
    as_of: str | dt.date,
    paths: Mapping[str, Path],
    lesson_pack: Optional[Any] = None,
) -> dict[str, Any]:
    """Create the Codex subagent dispatch contract for the agentic second pass."""

    day = parse_as_of(as_of).isoformat()
    tasks = [task for task in research_tasks.get("tasks", []) if isinstance(task, Mapping)]
    tickers = [str(task.get("ticker") or "").strip().upper() for task in tasks if str(task.get("ticker") or "").strip()]
    active_lessons = [
        lesson
        for lesson in getattr(lesson_pack, "lessons", [])
        if isinstance(lesson, Mapping) and str(lesson.get("status") or "") == "active"
    ]
    lesson_digest = str(getattr(lesson_pack, "digest", ""))
    lesson_context = str(getattr(lesson_pack, "markdown", "") or "")
    lesson_ids = [str(lesson.get("id") or "") for lesson in active_lessons]
    common_context = {
        "as_of": day,
        "portfolio_policy": "Portfolio risk is annotate-only. Do not hide or suppress an otherwise good setup for portfolio risk.",
        "lesson_pack_version": str(getattr(lesson_pack, "version", "none")),
        "lesson_pack_digest": lesson_digest,
        "lesson_context": lesson_context,
        "must_apply_lesson_ids": lesson_ids,
        "review_output_schema": {
            "ticker": "string ticker from the task list",
            "agent": "catalyst_news|macro_regime|structure_builder|skeptic|portfolio_management",
            "verdict": "supportive|caution|avoid",
            "confidence": "low|medium|high",
            "note": "concise evidence-backed reason",
            "objective_blocker": False,
            "evidence": "optional source or observation",
        },
        "input_artifacts": {
            "research_tasks": str(paths["research_tasks"]),
            "candidate_generation": str(paths["candidate_generation"]),
            "catalyst_evidence": str(paths["catalyst_evidence"]),
            "catalyst_reviews": str(paths["catalyst_reviews"]),
            "priced_candidates": str(paths["priced_candidates"]),
            "structure_attempts": str(paths["structure_attempts"]),
            "strategy_routing_audit": str(paths["strategy_routing_audit"]),
            "decision_board": str(paths["decision_board"]),
            "management_plan": str(paths["management_plan"]),
        },
    }
    lanes = [
        {
            "task_id": f"{day}:catalyst_news",
            "agent": "catalyst_news",
            "role": "Research current news, events, earnings, filings, analyst actions, and macro catalysts for the candidate list.",
            "focus": [
                "Confirm whether catalyst context supports, cautions, or objectively invalidates each setup.",
                "Use objective_blocker=true only for non-portfolio facts that should block entry.",
            ],
        },
        {
            "task_id": f"{day}:macro_regime",
            "agent": "macro_regime",
            "role": "Review market regime, sector exposure, macro tape, event calendar, and whether each setup direction fits the backdrop.",
            "focus": [
                "Call out regime or sector conflicts as caution unless they objectively invalidate the setup.",
                "Do not treat portfolio concentration as a setup-quality blocker.",
            ],
        },
        {
            "task_id": f"{day}:structure_builder",
            "agent": "structure_builder",
            "role": "Propose or validate executable option structures, payoff math, entry limits, expiries, and live-validation needs.",
            "focus": [
                "Prefer defined-risk structures with explicit max profit, max loss, target exit, and invalidation.",
                "Call out missing legs, wide markets, and absent live validation.",
            ],
        },
        {
            "task_id": f"{day}:skeptic",
            "agent": "skeptic",
            "role": "Independently try to disprove each setup and identify objective blockers, stale tickers, deal risk, bad liquidity, or thesis breaks.",
            "focus": [
                "Use objective_blocker=true only for non-portfolio facts that should block entry.",
                "Do not mark portfolio crowding as an objective blocker.",
            ],
        },
        {
            "task_id": f"{day}:portfolio_management",
            "agent": "portfolio_management",
            "role": "Review portfolio fit, sizing, target exits, invalidation, and management plan quality.",
            "focus": [
                "Flag concentration, correlation, and size risks as portfolio-risk annotations.",
                "Portfolio-risk-only avoid/caution must keep objective_blocker=false.",
            ],
        },
    ]
    subagent_tasks = []
    for lane in lanes:
        subagent_tasks.append(
            {
                **lane,
                "tickers": tickers,
                "input_task_count": len(tasks),
                "lesson_pack_digest": lesson_digest,
                "lesson_context": lesson_context,
                "must_apply_lesson_ids": lesson_ids,
                "prompt": _agent_dispatch_prompt(lane, common_context, tasks),
            }
        )
    return {
        "schema_version": "options_agent.agent_dispatch.v1",
        "as_of": day,
        "dispatch_status": "ready_for_codex_subagents",
        "dispatch_tool": "multi_agent_v1.spawn_agent",
        "expected_reviews_json": str(paths["agentic_reviews"]),
        "common_context": common_context,
        "subagent_tasks": subagent_tasks,
        "merge_instruction": (
            "Collect each subagent's JSON reviews into agentic_reviews.json as {'reviews': [...]} "
            "and rerun uwos.options_agent with --agent-reviews-json pointing at that file."
        ),
    }


def _agent_dispatch_prompt(lane: Mapping[str, Any], common_context: Mapping[str, Any], tasks: Sequence[Mapping[str, Any]]) -> str:
    tickers = ", ".join(str(task.get("ticker") or "").strip().upper() for task in tasks)
    lesson_context = str(common_context.get("lesson_context") or "").strip()
    lesson_block = (
        f"Active lesson pack: {common_context.get('lesson_pack_version')} "
        f"({common_context.get('lesson_pack_digest')}).\n"
        f"Lessons you must apply:\n{lesson_context}\n"
        if lesson_context
        else f"Active lesson pack: {common_context.get('lesson_pack_version')} ({common_context.get('lesson_pack_digest')}).\n"
    )
    return (
        f"You are the Options Agent {lane.get('agent')} subagent for {common_context.get('as_of')}.\n"
        f"Role: {lane.get('role')}\n"
        f"Tickers: {tickers}\n"
        f"{lesson_block}"
        "Read the input artifacts listed below from the local workspace, then return only JSON with a top-level "
        "`reviews` list. Each review must follow this schema: "
        f"{json.dumps(common_context.get('review_output_schema', {}), sort_keys=True)}\n"
        "Portfolio risk is annotate-only: never set objective_blocker=true solely because of concentration, "
        "correlation, buying power, or existing exposure. Use objective_blocker=true only for non-portfolio facts "
        "that should block the setup.\n"
        f"Input artifacts: {json.dumps(common_context.get('input_artifacts', {}), sort_keys=True)}\n"
        f"Focus: {' '.join(str(item) for item in lane.get('focus', []))}"
    )


def load_external_agent_reviews(path: Optional[Path]) -> tuple[pd.DataFrame, list[str]]:
    """Load optional external/subagent reviews from JSON."""

    if path is None:
        return pd.DataFrame(columns=EXTERNAL_REVIEW_COLUMNS), []
    resolved_path = Path(path).expanduser().resolve()
    default_agent_type = "subagent" if resolved_path.name in {"agentic_reviews.json", "subagent_reviews.json"} else "external"
    try:
        payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return pd.DataFrame(columns=EXTERNAL_REVIEW_COLUMNS), [f"external agent reviews unavailable: {exc}"]

    reviews = payload.get("reviews", payload) if isinstance(payload, dict) else payload
    if not isinstance(reviews, list):
        return pd.DataFrame(columns=EXTERNAL_REVIEW_COLUMNS), ["external agent reviews JSON did not contain a review list"]
    rows = []
    for review in reviews:
        if not isinstance(review, Mapping):
            continue
        ticker = str(review.get("ticker") or "").strip().upper()
        agent = str(review.get("agent") or "external").strip()
        rows.append(
            {
                "candidate_id": str(review.get("candidate_id") or f"{ticker}:{agent}:subagent_review").strip(),
                "ticker": ticker,
                "agent": agent,
                "agent_type": str(review.get("agent_type") or default_agent_type).strip(),
                "review_stage": str(review.get("review_stage") or "subagent_review").strip(),
                "verdict": str(review.get("verdict") or "").strip().lower(),
                "confidence": str(review.get("confidence") or "").strip().lower(),
                "note": _sanitize_visible_review_note(review.get("note")),
                "objective_blocker": _truthy(review.get("objective_blocker")),
                "blocker_type": str(review.get("blocker_type") or "").strip(),
                "portfolio_risk_only": _truthy(review.get("portfolio_risk_only")) if "portfolio_risk_only" in review else "",
                "evidence": str(review.get("evidence") or "").strip(),
                "source_artifact": str(review.get("source_artifact") or "agentic_reviews.json").strip(),
                "as_of": str(review.get("as_of") or "").strip(),
            }
        )
    return pd.DataFrame(rows, columns=EXTERNAL_REVIEW_COLUMNS), []


def _sanitize_visible_review_note(value: Any) -> str:
    """Keep account/portfolio routing language out of user-facing trade notes."""

    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(
        r"(?i)\bexisting\s+[^.;]*?\s+exposure\s+is\s+portfolio\s+annotation\s+only,\s*but\s*",
        "",
        text,
    )
    text = re.sub(r"(?i)\bportfolio\s+note\s+is\s+annotation\s+only\b", "", text)
    text = re.sub(r"(?i)\bportfolio\s+annotation\s+only,\s*but\s*", "", text)
    text = re.sub(r"(?i)\bportfolio\s+annotation\s+only\b", "", text)
    text = re.sub(r"(?i)\bportfolio\s+risk\s+(?:noted|annotated)\b", "", text)
    text = re.sub(r"(?i)\bvisibility\s+policy\b:?\s*", "", text)
    text = re.sub(r"(?i)\btrade\s+remains\s+visible\b", "", text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    text = re.sub(r"^[,.;:\s]+", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    if text.lower().startswith("but "):
        text = text[4:]
    return text.strip(" ;")


def build_internal_agent_reviews(
    candidates: pd.DataFrame,
    market_regime: Mapping[str, Any],
    catalyst_reviews: pd.DataFrame,
    priced: pd.DataFrame,
    *,
    as_of: str,
) -> pd.DataFrame:
    """Run deterministic built-in review agents and return a normalized board."""

    if candidates.empty and priced.empty:
        return pd.DataFrame(columns=AGENT_REVIEW_COLUMNS)

    candidate_by_ticker = _frame_by_ticker(candidates)
    catalyst_by_ticker = _frame_by_ticker(catalyst_reviews)
    priced_by_ticker = _frame_by_ticker(priced)
    ordered_tickers = _ordered_review_tickers(priced, candidates)
    rows: list[dict[str, Any]] = []
    regime = str(market_regime.get("regime") or "unknown")
    regime_note = str(market_regime.get("note") or market_regime.get("reason") or "market regime unavailable")

    for ticker in ordered_tickers:
        candidate = candidate_by_ticker.get(ticker, {})
        priced_row = priced_by_ticker.get(ticker, {})
        catalyst = catalyst_by_ticker.get(ticker, {})
        bias = str(candidate.get("bias") or priced_row.get("bias") or "").strip().lower()

        regime_verdict = "supportive"
        regime_confidence = "medium" if market_regime.get("status") == "ok" else "low"
        if (regime == "risk_off" and bias == "bullish") or (regime == "risk_on" and bias == "bearish"):
            regime_verdict = "caution"
        rows.append(
            _agent_review_row(
                ticker,
                "market_regime",
                "market_regime",
                regime_verdict,
                regime_confidence,
                f"{regime}: {regime_note}",
                source_artifact="market_regime.json",
                as_of=as_of,
            )
        )

        catalyst_status = str(catalyst.get("catalyst_status") or "").strip()
        catalyst_note = str(catalyst.get("catalyst_note") or "no catalyst review note").strip()
        catalyst_verdict = "supportive"
        catalyst_confidence = "medium"
        if catalyst_status in {"event_risk", "news_red_flag", "watch_event", "news_mixed", "local_news_available"}:
            catalyst_verdict = "caution"
            catalyst_confidence = "high" if catalyst_status in {"event_risk", "news_red_flag"} else "medium"
        rows.append(
            _agent_review_row(
                ticker,
                "catalyst",
                "catalyst",
                catalyst_verdict,
                catalyst_confidence,
                catalyst_note,
                evidence=catalyst_status,
                source_artifact="catalyst_reviews.csv",
                as_of=as_of,
            )
        )

        if priced_row:
            status = str(priced_row.get("recommendation_status") or "").strip().upper()
            hard_rejects = str(priced_row.get("hard_rejects") or "").strip()
            ticket = str(priced_row.get("full_ticket") or "").strip()
            status_reason = str(priced_row.get("status_reason") or "").strip()
            if hard_rejects:
                structure_verdict = "avoid"
                structure_note = f"objective structure blocker: {hard_rejects}"
                objective_blocker = True
                blocker_type = "execution"
                confidence = "high"
            elif not ticket:
                structure_verdict = "caution"
                structure_note = status_reason or "no executable ticket yet"
                objective_blocker = False
                blocker_type = ""
                confidence = "high"
            elif status == RecommendationStatus.WAIT_FOR_PRICE.value:
                structure_verdict = "caution"
                structure_note = status_reason or "entry price not ready"
                objective_blocker = False
                blocker_type = ""
                confidence = "medium"
            else:
                structure_verdict = "supportive"
                structure_note = status_reason or "structure has executable pricing"
                objective_blocker = False
                blocker_type = ""
                confidence = "medium"
            rows.append(
                _agent_review_row(
                    ticker,
                    "structure",
                    "structure",
                    structure_verdict,
                    confidence,
                    structure_note,
                    objective_blocker=objective_blocker,
                    blocker_type=blocker_type,
                    evidence=str(priced_row.get("full_ticket") or priced_row.get("invalidation") or ""),
                    source_artifact="priced_candidates.csv",
                    as_of=as_of,
                )
            )

            skeptic_verdict = "supportive"
            skeptic_note = "no objective blocker after structure review"
            skeptic_objective = False
            skeptic_blocker = ""
            if hard_rejects:
                skeptic_verdict = "avoid"
                skeptic_note = f"objective blocker remains: {hard_rejects}"
                skeptic_objective = True
                skeptic_blocker = "thesis_or_execution"
            elif status in {RecommendationStatus.REVIEW.value, RecommendationStatus.WAIT_FOR_PRICE.value}:
                skeptic_verdict = "caution"
                skeptic_note = status_reason or "requires human confirmation before entry"
            rows.append(
                _agent_review_row(
                    ticker,
                    "skeptic",
                    "skeptic",
                    skeptic_verdict,
                    "medium",
                    skeptic_note,
                    objective_blocker=skeptic_objective,
                    blocker_type=skeptic_blocker,
                    source_artifact="priced_candidates.csv",
                    as_of=as_of,
                )
            )
    return pd.DataFrame(rows, columns=AGENT_REVIEW_COLUMNS)


def build_portfolio_agent_reviews(final: pd.DataFrame, *, as_of: str) -> pd.DataFrame:
    """Emit portfolio-risk agent reviews after portfolio annotations are applied."""

    if final.empty:
        return pd.DataFrame(columns=AGENT_REVIEW_COLUMNS)
    rows: list[dict[str, Any]] = []
    for _, row in final.iterrows():
        ticker = str(row.get("ticker") or "").strip().upper()
        risk_note = str(row.get("portfolio_risk_note") or "").strip()
        if risk_note:
            rows.append(
                _agent_review_row(
                    ticker,
                    "portfolio_risk",
                    "portfolio",
                    "caution",
                    "high",
                    risk_note,
                    portfolio_risk_only=True,
                    blocker_type="portfolio",
                    source_artifact="risk_audit.csv",
                    as_of=as_of,
                )
            )
    return pd.DataFrame(rows, columns=AGENT_REVIEW_COLUMNS)


def combine_agent_reviews(*frames: pd.DataFrame, as_of: str) -> pd.DataFrame:
    """Merge built-in and optional external reviews into the canonical board."""

    rows: list[dict[str, Any]] = []
    for frame in frames:
        if frame is None or frame.empty:
            continue
        for _, review in frame.iterrows():
            rows.append(_normalize_agent_review(review.to_dict(), as_of=as_of))
    if not rows:
        return pd.DataFrame(columns=AGENT_REVIEW_COLUMNS)
    return pd.DataFrame(rows, columns=AGENT_REVIEW_COLUMNS)


def summarize_agent_reviews(agent_review_board: pd.DataFrame) -> dict[str, Any]:
    """Summarize review-board coverage for the manifest."""

    if agent_review_board.empty:
        return {
            "total": 0,
            "by_agent_type": {},
            "by_verdict": {},
            "objective_blockers": 0,
            "portfolio_risk_only": 0,
            "external_reviews_present": False,
        }
    return {
        "total": int(len(agent_review_board)),
        "by_agent_type": _value_counts(agent_review_board, "agent_type"),
        "by_verdict": _value_counts(agent_review_board, "verdict"),
        "objective_blockers": int(agent_review_board["objective_blocker"].astype(bool).sum()),
        "portfolio_risk_only": int(agent_review_board["portfolio_risk_only"].astype(bool).sum()),
        "external_reviews_present": bool(agent_review_board["agent_type"].astype(str).isin(["external", "subagent"]).any()),
    }


def apply_synthesis_ranking(
    final: pd.DataFrame,
    agent_review_board: pd.DataFrame,
    *,
    top_trades: int,
    lesson_pack: Optional[Any] = None,
    execution_context: Optional[Mapping[str, Any]] = None,
) -> pd.DataFrame:
    """Rank final rows using setup quality, execution readiness, and agent evidence."""

    # Kept for API compatibility; ranking must not trim the final board.
    _ = top_trades
    if final.empty:
        return final.copy()
    review_summary = _review_summary_by_ticker(agent_review_board)
    rows: list[dict[str, Any]] = []
    for _, row in final.iterrows():
        out = row.to_dict()
        ticker = str(out.get("ticker") or "").strip().upper()
        summary = review_summary.get(ticker, {})
        score, reason = _synthesis_score(out, summary, execution_context=execution_context)
        lesson_delta, lesson_reasons, lesson_rows = apply_synthesis_actions(
            out,
            getattr(lesson_pack, "lessons", []) if lesson_pack is not None else [],
        )
        if lesson_delta:
            score = round(score + lesson_delta, 2)
        if lesson_reasons:
            reason = "; ".join([reason, *lesson_reasons])
        out["synthesis_score"] = score
        out["synthesis_reason"] = reason
        out["lesson_score_delta"] = lesson_delta
        out["lesson_ids_applied"] = "; ".join(
            str(item.get("lesson_id") or "") for item in lesson_rows if str(item.get("lesson_id") or "")
        )
        out["lesson_application_rows"] = json.dumps(lesson_rows, sort_keys=True)
        out["agent_support_count"] = int(summary.get("supportive", 0))
        out["agent_caution_count"] = int(summary.get("caution", 0))
        out["agent_objective_blocker_count"] = int(summary.get("objective_blockers", 0))
        out["agent_portfolio_risk_only_count"] = int(summary.get("portfolio_risk_only", 0))
        rows.append(out)
    ranked = pd.DataFrame(rows)
    ranked = ranked.sort_values(["synthesis_score", "score", "signal_premium"], ascending=[False, False, False])
    ranked["recommendation_rank"] = range(1, len(ranked) + 1)
    return ranked.reset_index(drop=True)


def apply_agent_reviews(priced: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    """Apply normalized agent reviews before portfolio annotation."""

    if priced.empty:
        return _ensure_external_review_columns(priced.copy())
    if reviews.empty:
        return _ensure_external_review_columns(priced.copy())
    grouped: dict[str, list[dict[str, Any]]] = {}
    for _, review in reviews.iterrows():
        ticker = str(review.get("ticker") or "").strip().upper()
        if ticker:
            grouped.setdefault(ticker, []).append(review.to_dict())

    rows: list[dict[str, Any]] = []
    for _, row in priced.iterrows():
        out = row.to_dict()
        ticker = str(out.get("ticker") or "").strip().upper()
        ticker_reviews = grouped.get(ticker, [])
        if not ticker_reviews:
            rows.append(out)
            continue
        notes = [
            f"{review.get('agent', 'external')}={review.get('verdict', '')}: {review.get('note', '')}".strip()
            for review in ticker_reviews
            if str(review.get("note") or "").strip() or str(review.get("verdict") or "").strip()
        ]
        external_reviews = [
            review
            for review in ticker_reviews
            if str(review.get("agent_type") or "").strip().lower() in {"external", "subagent"}
        ]
        external_notes = [
            f"{review.get('agent', 'external')}={review.get('verdict', '')}: {review.get('note', '')}".strip()
            for review in external_reviews
            if str(review.get("note") or "").strip() or str(review.get("verdict") or "").strip()
            if not _is_portfolio_management_process_note(review)
            and not _is_portfolio_risk_review(review)
        ]
        distinct_external_agents = sorted(
            {
                str(review.get("agent") or "").strip()
                for review in external_reviews
                if str(review.get("agent") or "").strip()
            }
        )
        out["external_agent_review_count"] = len(external_reviews)
        out["external_agent_distinct_review_count"] = len(distinct_external_agents)
        out["external_agent_review_agents"] = "; ".join(distinct_external_agents)
        out["external_agent_review_note"] = "; ".join(_dedupe_notes(external_notes))
        portfolio_review_notes = [
            str(review.get("note") or "external portfolio risk").strip()
            for review in ticker_reviews
            if not _truthy(review.get("objective_blocker"))
            and _is_portfolio_risk_review(review)
            and str(review.get("verdict") or "").strip().lower() == "avoid"
        ]
        objective_blockers = [
            str(review.get("note") or review.get("verdict") or "external objective blocker").strip()
            for review in ticker_reviews
            if _truthy(review.get("objective_blocker"))
        ]
        cautions = [
            str(review.get("note") or "external caution").strip()
            for review in ticker_reviews
            if str(review.get("verdict") or "").strip().lower() == "caution"
            and not _truthy(review.get("objective_blocker"))
            and not _is_portfolio_risk_review(review)
            and not _is_portfolio_management_process_note(review)
        ]
        unsupported_avoids = [
            str(review.get("note") or "avoid verdict without objective blocker").strip()
            for review in ticker_reviews
            if str(review.get("verdict") or "").strip().lower() == "avoid"
            and not _truthy(review.get("objective_blocker"))
            and not _is_portfolio_risk_review(review)
            and not _is_portfolio_management_process_note(review)
        ]
        built_in_cautions = [
            str(review.get("note") or "built-in agent caution").strip()
            for review in ticker_reviews
            if str(review.get("verdict") or "").strip().lower() == "caution"
            and not _truthy(review.get("objective_blocker"))
            and not _is_portfolio_risk_review(review)
            and str(review.get("agent_type") or "").strip().lower() not in {"external", "subagent"}
        ]
        review_blocking_builtin_cautions = [
            str(review.get("note") or "built-in review gate").strip()
            for review in ticker_reviews
            if _is_review_blocking_builtin_caution(review)
        ]
        built_in_avoids = [
            str(review.get("note") or "built-in avoid verdict").strip()
            for review in ticker_reviews
            if str(review.get("verdict") or "").strip().lower() == "avoid"
            and not _truthy(review.get("objective_blocker"))
            and not _is_portfolio_risk_review(review)
            and str(review.get("agent_type") or "").strip().lower() not in {"external", "subagent"}
        ]
        if portfolio_review_notes:
            out["portfolio_risk_note"] = _append_reason(
                out.get("portfolio_risk_note"),
                "external portfolio risk review: " + "; ".join(_dedupe_notes(portfolio_review_notes)),
            )
        if objective_blockers:
            out["recommendation_status"] = RecommendationStatus.AVOID.value
            out["hard_rejects"] = _append_reason(out.get("hard_rejects"), "external_agent_objective_blocker")
            out["status_reason"] = _append_reason(out.get("status_reason"), "external agent blocker: " + "; ".join(objective_blockers))
        elif review_blocking_builtin_cautions and str(out.get("recommendation_status") or "").upper() in {
            RecommendationStatus.ENTER.value,
            RecommendationStatus.ENTER_WITH_PORTFOLIO_RISK.value,
        }:
            out["recommendation_status"] = RecommendationStatus.REVIEW.value
            caution_text = "; ".join(_dedupe_notes(review_blocking_builtin_cautions))
            out["status_reason"] = _append_reason(out.get("status_reason"), "built-in review gate: " + caution_text)
        elif built_in_cautions or built_in_avoids:
            caution_text = "; ".join(_dedupe_notes(built_in_cautions + [f"avoid without objective blocker: {note}" for note in built_in_avoids]))
            out["status_reason"] = _append_reason(out.get("status_reason"), "built-in agent caution: " + caution_text)
        elif cautions or unsupported_avoids:
            caution_text = "; ".join(_dedupe_notes(cautions + [f"avoid without objective blocker: {note}" for note in unsupported_avoids]))
            out["status_reason"] = _append_reason(out.get("status_reason"), "external agent caution: " + caution_text)
        rows.append(out)
    return _ensure_external_review_columns(pd.DataFrame(rows))


def _ensure_external_review_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column, default in (
        ("external_agent_review_count", 0),
        ("external_agent_distinct_review_count", 0),
        ("external_agent_review_agents", ""),
        ("external_agent_review_note", ""),
    ):
        if column not in out.columns:
            out[column] = default
        else:
            out[column] = out[column].fillna(default)
    return out


def price_candidates(
    date_dir: Path,
    as_of: str | dt.date,
    candidates: pd.DataFrame,
    *,
    limit: Optional[int] = None,
    root: Optional[Path] = None,
) -> pd.DataFrame:
    """Construct first-pass spread tickets from dated UW hot-chain quotes."""

    priced, _ = price_candidates_with_routing_audit(
        date_dir,
        as_of,
        candidates,
        limit=limit,
        root=root,
    )
    return priced


def price_candidates_with_routing_audit(
    date_dir: Path,
    as_of: str | dt.date,
    candidates: pd.DataFrame,
    *,
    limit: Optional[int] = None,
    root: Optional[Path] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct first-pass tickets and the strategy-router audit."""

    from codexuw import data as uw_data

    if candidates.empty:
        return pd.DataFrame(), pd.DataFrame(columns=STRATEGY_ROUTING_AUDIT_COLUMNS)
    hot = uw_data.load_hot_chains(date_dir, parse_as_of(as_of))
    rows = []
    audit_rows: list[dict[str, Any]] = []
    qualified = candidates[candidates["quality_status"].eq("qualified")]
    if limit is not None:
        qualified = qualified.head(int(limit))
    strategy_metrics = _closed_trade_strategy_family_metrics(root)
    positive_strategy_families = _positive_closed_trade_strategy_families(root, metrics_by_family=strategy_metrics)
    for _, candidate in qualified.iterrows():
        candidate_rows: list[dict[str, Any]] = []
        candidate_errors: list[dict[str, Any]] = []
        routes = _candidate_strategy_routes(candidate, positive_strategy_families, strategy_metrics)
        for route in routes:
            constructed = _construct_strategy_route(candidate, hot, route)
            constructed = _attach_strategy_route_metadata(constructed, route, strategy_metrics)
            constructed_ok = bool(_as_text(constructed.get("trade_plan"))) and ((_as_float(constructed.get("entry_limit")) or 0.0) > 0)
            if constructed_ok:
                candidate_rows.append(constructed)
            else:
                candidate_errors.append(constructed)
            audit_rows.append(_strategy_route_audit_row(candidate, constructed, route, strategy_metrics, constructed_ok))
        if candidate_rows:
            rows.extend(candidate_rows)
        elif candidate_errors:
            rows.append(candidate_errors[0])
    return pd.DataFrame(rows), pd.DataFrame(audit_rows, columns=STRATEGY_ROUTING_AUDIT_COLUMNS)


def _closed_trade_strategy_family_metrics(root: Optional[Path]) -> dict[str, dict[str, Any]]:
    if root is None:
        return {}
    resolved_root = Path(root).expanduser().resolve()
    out_root = resolved_root if resolved_root.name == "out" else resolved_root / "out"
    path = _closed_trades_evidence_path(resolved_root, out_root)
    if not _safe_non_v4_path(path) or not path.exists():
        return {}
    frame, error = _read_closed_trades_frame(path)
    if error or frame.empty or not {"strategy", "realized_pnl"}.issubset(frame.columns):
        return {}
    working = frame.copy()
    working["ticker"] = working.get("ticker", pd.Series("", index=working.index)).astype(str).str.upper()
    working["canonical_ticker"] = working["ticker"].map(canonical_ticker_key)
    working["strategy_family"] = working["strategy"].map(_normal_strategy_family)
    working["realized_pnl"] = pd.to_numeric(working["realized_pnl"], errors="coerce")
    return _actual_forward_metrics_by_strategy_family(
        working[working["realized_pnl"].notna() & working["strategy_family"].astype(str).str.strip().ne("")]
    )


def _positive_closed_trade_strategy_families(
    root: Optional[Path],
    *,
    metrics_by_family: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> set[str]:
    if metrics_by_family is None:
        metrics_by_family = _closed_trade_strategy_family_metrics(root)
    if not metrics_by_family:
        return set()
    positive: set[str] = set()
    for family, metrics in metrics_by_family.items():
        family_text = _as_text(family)
        if not family_text:
            continue
        if _as_text(metrics.get("status")).upper() == "PASS":
            positive.add(family_text)
    return positive


def _candidate_strategy_routes(
    candidate: Mapping[str, Any],
    positive_strategy_families: set[str],
    metrics_by_family: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    bias = _as_text(candidate.get("bias")).lower()
    if bias not in {"bullish", "bearish"}:
        return []
    tier = _as_text(candidate.get("underlying_quality_tier")).lower()
    if not tier:
        tier, _ = _underlying_quality(candidate)
    core_or_liquid = tier in {"core", "liquid"}
    if not core_or_liquid:
        return [_strategy_route("bull_put_credit" if bias == "bullish" else "bear_call_credit", "audit_only_low_quality_underlying")]

    routes: list[dict[str, Any]] = []
    if bias == "bullish":
        if _candidate_prefers_short_put(candidate, positive_strategy_families):
            routes.append(_strategy_route("short_put", "positive_short_put_family_evidence"))
        routes.extend(
            [
                _strategy_route("bull_call_debit", "bullish_core_defined_risk_upside_route"),
                _strategy_route("bull_put_credit", "bullish_core_credit_route"),
            ]
        )
    else:
        routes.extend(
            [
                _strategy_route("bear_put_debit", "bearish_core_defined_risk_downside_route"),
                _strategy_route("bear_call_credit", "bearish_core_credit_route"),
            ]
        )

    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for route in routes:
        strategy = route["strategy"]
        if strategy in seen:
            continue
        seen.add(strategy)
        family_metrics = metrics_by_family.get(route["strategy_family"], {})
        status = _as_text(family_metrics.get("status")).upper()
        if status == "BLOCK":
            route["route_action"] = "construct_research_only_negative_family_evidence"
        elif status == "PASS" and route["strategy_family"] in FAMILY_LEVEL_STRATEGY_EXPECTANCY_FALLBACKS:
            route["route_action"] = "construct_allowed_positive_family_route"
        elif status == "PASS":
            route["route_action"] = "construct_research_only_family_positive_ticker_proof_required"
        else:
            route["route_action"] = "construct_research_only_expectancy_missing"
        out.append(route)
    return out


def _strategy_route(strategy: str, reason: str) -> dict[str, Any]:
    routes = {
        "short_put": {
            "bias": "bullish",
            "strategy_family": "short_put",
            "entry_type": "CREDIT",
            "direction": "Short Put",
            "structure": "cash secured put",
        },
        "bull_put_credit": {
            "bias": "bullish",
            "strategy_family": "vertical_spread",
            "entry_type": "CREDIT",
            "direction": "Bull Put",
            "structure": "bull put spread",
        },
        "bull_call_debit": {
            "bias": "bullish",
            "strategy_family": "vertical_spread",
            "entry_type": "DEBIT",
            "direction": "Bull Call",
            "structure": "bull call debit spread",
        },
        "bear_call_credit": {
            "bias": "bearish",
            "strategy_family": "vertical_spread",
            "entry_type": "CREDIT",
            "direction": "Bear Call",
            "structure": "bear call spread",
        },
        "bear_put_debit": {
            "bias": "bearish",
            "strategy_family": "vertical_spread",
            "entry_type": "DEBIT",
            "direction": "Bear Put",
            "structure": "bear put debit spread",
        },
    }
    route = dict(routes[strategy])
    route["strategy"] = strategy
    route["route_reason"] = reason
    route["route_action"] = "construct_research_only_expectancy_missing"
    return route


def _construct_strategy_route(candidate: Mapping[str, Any], hot: pd.DataFrame, route: Mapping[str, Any]) -> dict[str, Any]:
    strategy = _as_text(route.get("strategy"))
    if strategy == "short_put":
        return construct_short_put(candidate, hot)
    if _as_text(route.get("entry_type")).upper() == "DEBIT":
        return construct_debit_spread(candidate, hot, direction=_as_text(route.get("direction")))
    return construct_credit_spread(candidate, hot)


def _attach_strategy_route_metadata(
    row: Mapping[str, Any],
    route: Mapping[str, Any],
    metrics_by_family: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    out = dict(row)
    family = _as_text(route.get("strategy_family"))
    metrics = metrics_by_family.get(family, {})
    out["strategy"] = _as_text(route.get("strategy"))
    out["strategy_family"] = family
    out["strategy_route"] = _as_text(route.get("strategy"))
    out["route_action"] = _as_text(route.get("route_action"))
    out["route_reason"] = _as_text(route.get("route_reason"))
    out["route_evidence_status"] = _as_text(metrics.get("status")) or "MISSING"
    out["route_evidence_sample_size"] = metrics.get("sample_size", "")
    out["route_evidence_avg_pnl"] = metrics.get("avg_pnl", "")
    out["route_evidence_profit_factor"] = metrics.get("profit_factor", "")
    return out


def _strategy_route_audit_row(
    candidate: Mapping[str, Any],
    row: Mapping[str, Any],
    route: Mapping[str, Any],
    metrics_by_family: Mapping[str, Mapping[str, Any]],
    constructed_ok: bool,
) -> dict[str, Any]:
    family = _as_text(route.get("strategy_family"))
    metrics = metrics_by_family.get(family, {})
    return {
        "candidate_rank": candidate.get("candidate_rank", ""),
        "ticker": _as_text(candidate.get("ticker")).upper(),
        "bias": candidate.get("bias", ""),
        "underlying_quality_tier": candidate.get("underlying_quality_tier", ""),
        "macro_tape_direction": candidate.get("macro_tape_direction", ""),
        "strategy": route.get("strategy", ""),
        "strategy_family": family,
        "route_action": route.get("route_action", ""),
        "route_status": "constructed" if constructed_ok else "construction_failed",
        "route_reason": route.get("route_reason", ""),
        "evidence_status": metrics.get("status", "MISSING"),
        "evidence_sample_size": metrics.get("sample_size", ""),
        "evidence_avg_pnl": metrics.get("avg_pnl", ""),
        "evidence_profit_factor": metrics.get("profit_factor", ""),
        "selected_structure": row.get("structure", route.get("structure", "")),
        "selected_trade_plan": row.get("trade_plan", ""),
        "entry_type": route.get("entry_type", ""),
        "entry_limit": row.get("entry_limit", ""),
        "quality_gate_reason": row.get("quality_gate_reason", row.get("status_reason", "")),
    }


def _candidate_prefers_short_put(candidate: Mapping[str, Any], positive_strategy_families: set[str]) -> bool:
    if "short_put" not in positive_strategy_families:
        return False
    if _as_text(candidate.get("bias")).lower() != "bullish":
        return False
    tier = _as_text(candidate.get("underlying_quality_tier")).lower()
    if not tier:
        tier, _ = _underlying_quality(candidate)
    if tier != "core":
        return False
    if not _truthy(candidate.get("macro_tape_candidate")) and (_as_float(candidate.get("combined_flow_bias")) or 0.0) < MIN_DIRECTIONAL_BIAS:
        return False
    return True


def validate_priced_candidates_live(
    priced: pd.DataFrame,
    as_of: str | dt.date,
    out_dir: Path,
    *,
    chain_snapshot_dir: Optional[Path] = None,
    strike_count: int = 80,
    allow_live_fallback: bool = True,
    market_session_open: Optional[bool] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Replace dated pricing with live/snapshot Schwab chain alternatives when available."""

    if priced.empty:
        return priced.copy(), empty_live_validation_frame(), []

    from codexuw.schwab_live import (
        SchwabChainValidator,
        chain_spot,
        chain_to_contracts,
        find_credit_spread_alternatives,
        find_debit_spread_alternatives,
    )

    asof_date = parse_as_of(as_of)
    validator = SchwabChainValidator(
        out_dir,
        strike_count=int(strike_count),
        snapshot_dir=chain_snapshot_dir,
        allow_live_fallback=allow_live_fallback,
    )
    market_closed_live_recheck = bool(
        chain_snapshot_dir is None
        and allow_live_fallback
        and market_session_open is False
    )
    unavailable_source_label = "Schwab snapshot chain" if chain_snapshot_dir is not None and not allow_live_fallback else "live Schwab chain"
    rows: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    notes: list[str] = []
    for _, row in priced.iterrows():
        current = row.to_dict()
        ticker = str(current.get("ticker") or "").strip().upper()
        if not ticker:
            rows.append(current)
            continue

        preferred_expiry = _row_date(current.get("expiry")) or _row_date(current.get("anchor_expiry"))
        preferred_query_expiry = preferred_expiry if _live_expiry_in_range(asof_date, preferred_expiry) else None
        query_from = preferred_query_expiry or (asof_date + dt.timedelta(days=MIN_LIVE_DTE))
        query_to = preferred_query_expiry or (asof_date + dt.timedelta(days=MAX_LIVE_DTE))
        chain = validator.get_chain(ticker, query_from, query_to)
        if not chain:
            message = validator.errors.get(ticker, "no live chain returned")
            if market_closed_live_recheck and _has_complete_target_math(current):
                updated = _preserve_market_closed_target_recheck(current, message)
                rows.append(updated)
                audit.append(_live_audit_row(updated, "TARGET_QUOTE_REFRESH", updated["live_validation_note"], validator.sources.get(ticker, "")))
                continue
            current["live_validation_status"] = "CHAIN_UNAVAILABLE"
            current["live_validation_note"] = message
            current["recommendation_status"] = _preserve_non_entry_status(current)
            current["status_reason"] = _append_reason(
                current.get("status_reason"),
                f"{unavailable_source_label} unavailable; keep as visible review row",
            )
            rows.append(current)
            audit.append(_live_audit_row(current, "CHAIN_UNAVAILABLE", message, validator.sources.get(ticker, "")))
            continue

        contracts = chain_to_contracts(chain)
        spot = chain_spot(chain)
        if contracts.empty or not math.isfinite(spot):
            message = "chain returned no contracts or no usable underlying price"
            if market_closed_live_recheck and _has_complete_target_math(current):
                updated = _preserve_market_closed_target_recheck(current, message)
                rows.append(updated)
                audit.append(_live_audit_row(updated, "TARGET_QUOTE_REFRESH", updated["live_validation_note"], validator.sources.get(ticker, "")))
                continue
            current["live_validation_status"] = "NO_USABLE_CHAIN"
            current["live_validation_note"] = message
            current["recommendation_status"] = _preserve_non_entry_status(current)
            current["status_reason"] = _append_reason(current.get("status_reason"), message)
            rows.append(current)
            audit.append(_live_audit_row(current, "NO_USABLE_CHAIN", message, validator.sources.get(ticker, "")))
            continue

        if _strategy_family_from_ticket_row(current) == "short_put":
            expiry = _select_live_expiry(contracts, asof_date, preferred_expiry, "Bull Put")
            if expiry is None:
                message = "no suitable Schwab-chain expiry for short-put construction"
                if market_closed_live_recheck and _has_complete_target_math(current):
                    updated = _preserve_market_closed_target_recheck(current, message)
                    rows.append(updated)
                    audit.append(_live_audit_row(updated, "TARGET_QUOTE_REFRESH", updated["live_validation_note"], validator.sources.get(ticker, "")))
                    continue
                current["live_validation_status"] = "NO_LIVE_EXPIRY"
                current["live_validation_note"] = message
                current["recommendation_status"] = _preserve_non_entry_status(current)
                current["status_reason"] = _append_reason(current.get("status_reason"), message)
                rows.append(current)
                audit.append(_live_audit_row(current, "NO_LIVE_EXPIRY", message, validator.sources.get(ticker, "")))
                continue
            alternatives = _find_short_put_alternatives(
                contracts,
                expiry=expiry,
                spot=spot,
                expected_move_pct=_as_float(current.get("iv30d")),
                max_alternatives=8,
            )
            best = _select_live_short_put_alternative(current, alternatives)
            if best.get("live_status") != "PASS":
                message = str(best.get("live_blocker") or best.get("live_status") or "live chain did not produce a valid short put")
                if market_closed_live_recheck and _has_complete_target_math(current):
                    updated = _preserve_market_closed_target_recheck(current, message)
                    rows.append(updated)
                    audit.append(_live_audit_row(updated, "TARGET_QUOTE_REFRESH", updated["live_validation_note"], validator.sources.get(ticker, "")))
                    continue
                current["live_validation_status"] = str(best.get("live_status") or "NO_LIVE_SHORT_PUT")
                current["live_validation_note"] = message
                current["recommendation_status"] = _preserve_non_entry_status(current)
                current["status_reason"] = _append_reason(current.get("status_reason"), message)
                rows.append(current)
                audit.append(_live_audit_row(current, current["live_validation_status"], message, validator.sources.get(ticker, "")))
                continue
            chain_source = validator.sources.get(ticker, "")
            updated = _apply_live_short_put(
                current,
                best,
                expiry=expiry,
                spot=spot,
                asof_date=asof_date,
                chain_source=chain_source,
            )
            updated["live_validation_status"] = "PASS"
            updated["live_validation_note"] = str(updated.get("status_reason") or best.get("construction_reason") or "chain validated")
            updated["live_chain_source"] = chain_source
            rows.append(updated)
            audit.append(_live_audit_row(updated, "PASS", updated["live_validation_note"], chain_source))
            continue

        if _route_prefers_debit_spread(current):
            debit_direction = _debit_direction(current)
            debit_expiry = _select_live_expiry(contracts, asof_date, preferred_expiry, debit_direction)
            if debit_expiry is None:
                message = "no suitable Schwab-chain expiry for debit-spread construction"
                if market_closed_live_recheck and _has_complete_target_math(current):
                    updated = _preserve_market_closed_target_recheck(current, message)
                    rows.append(updated)
                    audit.append(_live_audit_row(updated, "TARGET_QUOTE_REFRESH", updated["live_validation_note"], validator.sources.get(ticker, "")))
                    continue
                current["live_validation_status"] = "NO_LIVE_EXPIRY"
                current["live_validation_note"] = message
                current["recommendation_status"] = _preserve_non_entry_status(current)
                current["status_reason"] = _append_reason(current.get("status_reason"), message)
                rows.append(current)
                audit.append(_live_audit_row(current, "NO_LIVE_EXPIRY", message, validator.sources.get(ticker, "")))
                continue
            debit_alternatives = find_debit_spread_alternatives(
                contracts,
                direction=debit_direction,
                expiry=debit_expiry,
                spot=spot,
                preferred_width=_preferred_width(current),
                anchor_strike=_as_float(current.get("anchor_strike")),
                expected_move_pct=_as_float(current.get("iv30d")),
                max_alternatives=8,
            )
            debit_best = (
                _select_live_alternative(current, debit_alternatives, entry_type="DEBIT")
                if debit_alternatives
                else {"live_status": "no_live_alternative", "live_blocker": "no debit alternative returned"}
            )
            if debit_best.get("live_status") != "PASS":
                message = str(debit_best.get("live_blocker") or debit_best.get("live_status") or "live chain did not produce a valid debit spread")
                if market_closed_live_recheck and _has_complete_target_math(current):
                    updated = _preserve_market_closed_target_recheck(current, message)
                    rows.append(updated)
                    audit.append(_live_audit_row(updated, "TARGET_QUOTE_REFRESH", updated["live_validation_note"], validator.sources.get(ticker, "")))
                    continue
                current["live_validation_status"] = str(debit_best.get("live_status") or "NO_LIVE_DEBIT_SPREAD")
                current["live_validation_note"] = message
                current["recommendation_status"] = _preserve_non_entry_status(current)
                current["status_reason"] = _append_reason(current.get("status_reason"), message)
                rows.append(current)
                audit.append(_live_audit_row(current, current["live_validation_status"], message, validator.sources.get(ticker, "")))
                continue
            chain_source = validator.sources.get(ticker, "")
            updated = _apply_live_debit_spread(
                current,
                debit_best,
                direction=debit_direction,
                expiry=debit_expiry,
                spot=spot,
                asof_date=asof_date,
                chain_source=chain_source,
            )
            updated["live_validation_status"] = "PASS"
            updated["live_validation_note"] = str(
                updated.get("status_reason") or debit_best.get("construction_reason") or "chain validated"
            )
            updated["live_chain_source"] = chain_source
            rows.append(updated)
            audit.append(_live_audit_row(updated, "PASS", updated["live_validation_note"], chain_source))
            continue

        direction = _credit_direction(current)
        expiry = _select_live_expiry(contracts, asof_date, preferred_expiry, direction)
        if expiry is None:
            debit_direction = _debit_direction(current)
            debit_expiry = _select_live_expiry(contracts, asof_date, preferred_expiry, debit_direction)
            if debit_expiry is not None:
                debit_alternatives = find_debit_spread_alternatives(
                    contracts,
                    direction=debit_direction,
                    expiry=debit_expiry,
                    spot=spot,
                    preferred_width=_preferred_width(current),
                    anchor_strike=_as_float(current.get("anchor_strike")),
                    expected_move_pct=_as_float(current.get("iv30d")),
                    max_alternatives=8,
                )
                debit_best = (
                    _select_live_alternative(current, debit_alternatives, entry_type="DEBIT")
                    if debit_alternatives
                    else {"live_status": "no_live_alternative", "live_blocker": "no debit alternative returned"}
                )
                if debit_best.get("live_status") == "PASS":
                    chain_source = validator.sources.get(ticker, "")
                    updated = _apply_live_debit_spread(
                        current,
                        debit_best,
                        direction=debit_direction,
                        expiry=debit_expiry,
                        spot=spot,
                        asof_date=asof_date,
                        chain_source=chain_source,
                    )
                    updated["live_validation_status"] = "PASS"
                    updated["live_validation_note"] = str(
                        updated.get("status_reason") or debit_best.get("construction_reason") or "chain validated"
                    )
                    updated["live_chain_source"] = chain_source
                    rows.append(updated)
                    audit.append(_live_audit_row(updated, "PASS", updated["live_validation_note"], chain_source))
                    continue
            message = "no suitable Schwab-chain expiry for credit-spread construction"
            if market_closed_live_recheck and _has_complete_target_math(current):
                updated = _preserve_market_closed_target_recheck(current, message)
                rows.append(updated)
                audit.append(_live_audit_row(updated, "TARGET_QUOTE_REFRESH", updated["live_validation_note"], validator.sources.get(ticker, "")))
                continue
            current["live_validation_status"] = "NO_LIVE_EXPIRY"
            current["live_validation_note"] = message
            current["recommendation_status"] = _preserve_non_entry_status(current)
            current["status_reason"] = _append_reason(current.get("status_reason"), message)
            rows.append(current)
            audit.append(_live_audit_row(current, "NO_LIVE_EXPIRY", message, validator.sources.get(ticker, "")))
            continue

        alternatives = find_credit_spread_alternatives(
            contracts,
            direction=direction,
            expiry=expiry,
            spot=spot,
            preferred_width=_preferred_width(current),
            anchor_strike=_as_float(current.get("anchor_strike")),
            expected_move_pct=_as_float(current.get("iv30d")),
            max_alternatives=8,
        )
        best = (
            _select_live_alternative(current, alternatives, entry_type="CREDIT")
            if alternatives
            else {"live_status": "no_live_alternative", "live_blocker": "no alternative returned"}
        )
        if best.get("live_status") != "PASS":
            debit_direction = _debit_direction(current)
            debit_expiry = _select_live_expiry(contracts, asof_date, preferred_expiry, debit_direction)
            if debit_expiry is not None:
                debit_alternatives = find_debit_spread_alternatives(
                    contracts,
                    direction=debit_direction,
                    expiry=debit_expiry,
                    spot=spot,
                    preferred_width=_preferred_width(current),
                    anchor_strike=_as_float(current.get("anchor_strike")),
                    expected_move_pct=_as_float(current.get("iv30d")),
                    max_alternatives=8,
                )
                debit_best = (
                    _select_live_alternative(current, debit_alternatives, entry_type="DEBIT")
                    if debit_alternatives
                    else {"live_status": "no_live_alternative", "live_blocker": "no debit alternative returned"}
                )
                if debit_best.get("live_status") == "PASS":
                    chain_source = validator.sources.get(ticker, "")
                    updated = _apply_live_debit_spread(
                        current,
                        debit_best,
                        direction=debit_direction,
                        expiry=debit_expiry,
                        spot=spot,
                        asof_date=asof_date,
                        chain_source=chain_source,
                    )
                    updated["live_validation_status"] = "PASS"
                    updated["live_validation_note"] = str(
                        updated.get("status_reason") or debit_best.get("construction_reason") or "chain validated"
                    )
                    updated["live_chain_source"] = chain_source
                    rows.append(updated)
                    audit.append(_live_audit_row(updated, "PASS", updated["live_validation_note"], chain_source))
                    continue
            message = str(best.get("live_blocker") or best.get("live_status") or "live chain did not produce a valid spread")
            if market_closed_live_recheck and _has_complete_target_math(current):
                updated = _preserve_market_closed_target_recheck(current, message)
                rows.append(updated)
                audit.append(_live_audit_row(updated, "TARGET_QUOTE_REFRESH", updated["live_validation_note"], validator.sources.get(ticker, "")))
                continue
            current["live_validation_status"] = str(best.get("live_status") or "NO_LIVE_SPREAD")
            current["live_validation_note"] = message
            current["recommendation_status"] = _preserve_non_entry_status(current)
            current["status_reason"] = _append_reason(current.get("status_reason"), message)
            rows.append(current)
            audit.append(_live_audit_row(current, current["live_validation_status"], message, validator.sources.get(ticker, "")))
            continue

        chain_source = validator.sources.get(ticker, "")
        updated = _apply_live_credit_spread(
            current,
            best,
            direction=direction,
            expiry=expiry,
            spot=spot,
            asof_date=asof_date,
            chain_source=chain_source,
        )
        if str(updated.get("recommendation_status") or "").upper() == RecommendationStatus.AVOID.value:
            debit_direction = _debit_direction(current)
            debit_expiry = _select_live_expiry(contracts, asof_date, preferred_expiry, debit_direction)
            if debit_expiry is not None:
                debit_alternatives = find_debit_spread_alternatives(
                    contracts,
                    direction=debit_direction,
                    expiry=debit_expiry,
                    spot=spot,
                    preferred_width=_preferred_width(current),
                    anchor_strike=_as_float(current.get("anchor_strike")),
                    expected_move_pct=_as_float(current.get("iv30d")),
                    max_alternatives=8,
                )
                debit_best = (
                    _select_live_alternative(current, debit_alternatives, entry_type="DEBIT")
                    if debit_alternatives
                    else {"live_status": "no_live_alternative", "live_blocker": "no debit alternative returned"}
                )
                if debit_best.get("live_status") == "PASS":
                    updated = _apply_live_debit_spread(
                        current,
                        debit_best,
                        direction=debit_direction,
                        expiry=debit_expiry,
                        spot=spot,
                        asof_date=asof_date,
                        chain_source=chain_source,
                    )
        updated["live_validation_status"] = "PASS"
        updated["live_validation_note"] = str(updated.get("status_reason") or best.get("construction_reason") or "chain validated")
        updated["live_chain_source"] = chain_source
        rows.append(updated)
        audit.append(_live_audit_row(updated, "PASS", updated["live_validation_note"], chain_source))

    try:
        validator.save()
    except Exception as exc:
        notes.append(f"live chain snapshot save failed: {exc}")

    audit_frame = pd.DataFrame(audit)
    if audit_frame.empty:
        audit_frame = empty_live_validation_frame()
    return pd.DataFrame(rows), audit_frame, notes


def empty_live_validation_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "ticker",
            "live_validation_status",
            "recommendation_status",
            "chain_source",
            "expiry",
            "entry_limit",
            "target_entry",
            "trade_plan",
            "sell_leg",
            "buy_leg",
            "short_leg",
            "long_leg",
            "note",
        ]
    )


def build_structure_attempts(
    dated_priced: pd.DataFrame,
    final_priced: pd.DataFrame,
    live_validation: pd.DataFrame,
) -> pd.DataFrame:
    """Create an audit trail for dated and live structure construction attempts."""

    columns = [
        "ticker",
        "attempt_stage",
        "attempt_status",
        "structure",
        "full_ticket",
        "trade_plan",
        "expiry",
        "entry_limit",
        "target_entry",
        "max_profit",
        "max_loss",
        "sell_leg",
        "buy_leg",
        "short_leg",
        "long_leg",
        "source",
        "note",
    ]
    rows: list[dict[str, Any]] = []
    if not dated_priced.empty:
        for _, row in dated_priced.iterrows():
            status = _structure_attempt_status(row, stage="dated")
            rows.append(
                {
                    "ticker": row.get("ticker", ""),
                    "attempt_stage": "dated_hot_chain",
                    "attempt_status": status,
                    "structure": row.get("structure", ""),
                    "full_ticket": row.get("full_ticket", ""),
                    "trade_plan": row.get("trade_plan", row.get("full_ticket", "")),
                    "expiry": row.get("expiry", ""),
                    "entry_limit": row.get("entry_limit", ""),
                    "target_entry": "",
                    "max_profit": row.get("max_profit", ""),
                    "max_loss": row.get("max_loss", ""),
                    "sell_leg": row.get("sell_leg", row.get("short_leg", "")),
                    "buy_leg": row.get("buy_leg", row.get("long_leg", "")),
                    "short_leg": row.get("short_leg", ""),
                    "long_leg": row.get("long_leg", ""),
                    "source": "dated UW hot-chain",
                    "note": row.get("status_reason", ""),
                }
            )
    if not live_validation.empty:
        final_by_ticker = _frame_by_ticker(final_priced)
        for _, row in live_validation.iterrows():
            ticker = str(row.get("ticker") or "").strip().upper()
            priced = final_by_ticker.get(ticker, {})
            rows.append(
                {
                    "ticker": ticker,
                    "attempt_stage": "live_schwab_chain",
                    "attempt_status": str(row.get("live_validation_status") or "").strip() or "UNKNOWN",
                    "structure": priced.get("structure", ""),
                    "full_ticket": priced.get("full_ticket", ""),
                    "trade_plan": priced.get("trade_plan", priced.get("full_ticket", "")),
                    "expiry": row.get("expiry", ""),
                    "entry_limit": row.get("entry_limit", ""),
                    "target_entry": row.get("target_entry", ""),
                    "max_profit": priced.get("max_profit", ""),
                    "max_loss": priced.get("max_loss", ""),
                    "sell_leg": row.get("sell_leg", priced.get("sell_leg", "")),
                    "buy_leg": row.get("buy_leg", priced.get("buy_leg", "")),
                    "short_leg": row.get("short_leg", ""),
                    "long_leg": row.get("long_leg", ""),
                    "source": row.get("chain_source", ""),
                    "note": row.get("note", ""),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _structure_attempt_status(row: Mapping[str, Any], *, stage: str) -> str:
    ticket = str(row.get("full_ticket") or "").strip()
    entry = _as_float(row.get("entry_limit")) or 0.0
    max_profit = _as_float(row.get("max_profit")) or 0.0
    max_loss = _as_float(row.get("max_loss")) or 0.0
    if ticket and entry > 0 and max_profit > 0 and max_loss > 0:
        return str(row.get("recommendation_status") or "CONSTRUCTED").strip() or "CONSTRUCTED"
    if ticket:
        return "INCOMPLETE_PAYOFF"
    if stage == "dated":
        return "CONSTRUCTION_GAP"
    return "NO_STRUCTURE"


def unavailable_portfolio_context(error: str) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "error": str(error),
        "total_value": 0.0,
        "cash": 0.0,
        "position_count": 0,
        "option_underlyings": [],
        "short_option_underlyings": [],
        "equity_exposure": {},
        "option_market_value": {},
        "large_equity_exposure": {},
        "risk_actions": [],
        "portfolio_income_actions": [],
    }


def resolve_portfolio_context(
    out_dir: Path,
    *,
    portfolio_context: Optional[Mapping[str, Any]] = None,
    portfolio_json: Optional[Path] = None,
    live_portfolio: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    """Load or fetch portfolio context with Options Agent-owned artifacts."""

    if portfolio_context is not None:
        return dict(portfolio_context), []

    if portfolio_json is not None:
        try:
            path = Path(portfolio_json).expanduser().resolve()
            return json.loads(path.read_text(encoding="utf-8")), []
        except Exception as exc:
            return unavailable_portfolio_context(f"portfolio json unavailable: {exc}"), [f"portfolio json unavailable: {exc}"]

    if not live_portfolio:
        return unavailable_portfolio_context("portfolio context not requested"), []

    try:
        from codexuw.portfolio import summarize_positions
        from uwos.schwab_auth import SchwabAuthConfig, SchwabLiveDataService

        service = SchwabLiveDataService(SchwabAuthConfig.from_env(load_dotenv_file=True), interactive_login=False)
        payload = service.get_account_positions()
        positions = pd.DataFrame(payload.get("positions", []) or [])
        positions.to_csv(out_dir / "options_agent_open_positions_from_schwab.csv", index=False)
        return summarize_positions(payload), []
    except Exception as exc:
        message = f"live portfolio context unavailable: {exc}"
        return unavailable_portfolio_context(message), [message]


def construct_credit_spread(candidate: Mapping[str, Any], hot: pd.DataFrame) -> dict[str, Any]:
    """Build a conservative one-lot credit spread attempt for a candidate."""

    ticker = str(candidate.get("ticker") or "").upper()
    bias = str(candidate.get("bias") or "").lower()
    close = _as_float(candidate.get("close"))
    right = "P" if bias == "bullish" else "C"
    structure = "bull put spread" if right == "P" else "bear call spread"
    chain = hot[hot["ticker"].astype(str).str.upper().eq(ticker) & hot["right"].astype(str).str.upper().eq(right)].copy()
    if close is not None:
        if right == "P":
            chain = chain[chain["strike"] < close]
        else:
            chain = chain[chain["strike"] > close]
    if chain.empty:
        return priced_error_row(candidate, structure, "missing suitable short leg in dated UW hot-chain source")

    preferred = chain[
        (pd.to_numeric(chain["dte"], errors="coerce") >= MIN_LIVE_DTE)
        & (pd.to_numeric(chain["dte"], errors="coerce") <= MAX_LIVE_DTE)
    ]
    if preferred.empty:
        return priced_error_row(candidate, structure, "no dated UW hot-chain expiry in 7-60 DTE window")
    short = preferred.sort_values(["premium", "volume"], ascending=[False, False]).iloc[0]
    same_expiry = chain[chain["expiry_dt"].eq(short["expiry_dt"])].copy()
    if right == "P":
        longs = same_expiry[same_expiry["strike"] < short["strike"]].sort_values("strike", ascending=False)
    else:
        longs = same_expiry[same_expiry["strike"] > short["strike"]].sort_values("strike", ascending=True)
    if longs.empty:
        return priced_error_row(candidate, structure, "missing protective long leg in dated UW hot-chain source")

    long = longs.iloc[0]
    short_bid = _as_float(short.get("bid")) or 0.0
    short_ask = _as_float(short.get("ask")) or 0.0
    long_bid = _as_float(long.get("bid")) or 0.0
    long_ask = _as_float(long.get("ask")) or 0.0
    short_mid = (short_bid + short_ask) / 2 if short_bid or short_ask else 0.0
    long_mid = (long_bid + long_ask) / 2 if long_bid or long_ask else 0.0
    conservative_credit = short_bid - long_ask
    mid_credit = short_mid - long_mid
    entry_credit = conservative_credit if conservative_credit > 0 else mid_credit
    width = abs(float(short["strike"]) - float(long["strike"]))

    if entry_credit <= 0 or width <= 0:
        return priced_error_row(candidate, structure, "spread credit could not be estimated from dated bid/ask")

    max_profit = round(entry_credit * 100, 2)
    max_loss = round(max((width - entry_credit) * 100, 0), 2)
    credit_width_ratio = round(entry_credit / width, 4) if width > 0 else 0.0
    breakeven = float(short["strike"]) - entry_credit if right == "P" else float(short["strike"]) + entry_credit
    short_symbol = _as_text(short.get("option_symbol"))
    long_symbol = _as_text(long.get("option_symbol"))
    sell_leg = _format_option_leg(
        short_symbol,
        "SELL",
        ticker=ticker,
        expiry=short["expiry_dt"],
        strike=short["strike"],
        right=right,
    )
    buy_leg = _format_option_leg(
        long_symbol,
        "BUY",
        ticker=ticker,
        expiry=long["expiry_dt"],
        strike=long["strike"],
        right=right,
    )
    trade_plan = _format_trade_plan(sell_leg, buy_leg, entry_credit, entry_type="CREDIT")
    bid_ask_width = max((short_ask - short_bid) + (long_ask - long_bid), 0.0)
    liquidity_note = "dated UW EOD quote; refresh Schwab chain before entry"
    status = RecommendationStatus.REVIEW.value
    hard_rejects = ""
    if bid_ask_width > max(entry_credit, 0.01) * 1.5:
        status = RecommendationStatus.WAIT_FOR_PRICE.value
        liquidity_note = "dated bid/ask width is wide; keep visible and re-price with fresh Schwab chain"
    quality_rejects = _trade_quality_rejects(
        entry_credit=entry_credit,
        credit_width_ratio=credit_width_ratio,
        max_loss=max_loss,
        signal_premium=_as_float(candidate.get("signal_premium")) or 0.0,
        combined_flow_bias=_as_float(candidate.get("combined_flow_bias")) or 0.0,
        macro_tape_candidate=_truthy(candidate.get("macro_tape_candidate")),
    )
    if quality_rejects:
        status = RecommendationStatus.AVOID.value
        hard_rejects = "; ".join(quality_rejects)
        liquidity_note = _append_reason(liquidity_note, "setup quality gate reject: " + hard_rejects)
    underlying_tier, underlying_reason = _underlying_quality(candidate)

    return {
        "ticker": ticker,
        "full_name": candidate.get("full_name", ""),
        "issue_type": candidate.get("issue_type", ""),
        "marketcap": candidate.get("marketcap", ""),
        "avg30_volume": candidate.get("avg30_volume", ""),
        "total_volume": candidate.get("total_volume", ""),
        "total_open_interest": candidate.get("total_open_interest", ""),
        "underlying_quality_tier": underlying_tier,
        "underlying_quality_reason": underlying_reason,
        "bias": candidate.get("bias"),
        "flow_bias_label": candidate.get("flow_bias_label", candidate.get("bias")),
        "candidate_source": candidate.get("candidate_source", "flow_oi"),
        "macro_tape_candidate": candidate.get("macro_tape_candidate", False),
        "macro_tape_direction": candidate.get("macro_tape_direction", ""),
        "macro_tape_reason": candidate.get("macro_tape_reason", ""),
        "price_move_pct": candidate.get("price_move_pct", ""),
        "price_tape_source": candidate.get("price_tape_source", ""),
        "structure": structure,
        "full_ticket": trade_plan,
        "trade_plan": trade_plan,
        "expiry": str(short["expiry_dt"]),
        "dte": int(float(short.get("dte") or 0)),
        "sell_leg": sell_leg,
        "buy_leg": buy_leg,
        "short_leg": sell_leg,
        "long_leg": buy_leg,
        "anchor_expiry": str(candidate.get("hot_top_expiry") or short["expiry_dt"]),
        "anchor_strike": float(candidate.get("hot_top_strike") or short["strike"]),
        "entry_limit": round(entry_credit, 2),
        "mid": round(mid_credit, 2),
        "bid": round(conservative_credit, 2),
        "ask": "",
        "max_profit": max_profit,
        "max_loss": max_loss,
        "credit_width_ratio": credit_width_ratio,
        "trade_quality_status": "rejected" if quality_rejects else "reviewable",
        "quality_gate_reason": "; ".join(quality_rejects),
        "remaining_upside": max_profit,
        "breakeven": round(breakeven, 2),
        "target_exit": round(entry_credit * 0.35, 2),
        "invalidation": "fresh quote fails, thesis breaks, or underlying violates breakeven/flow support",
        "score": round(float(candidate.get("score") or 0), 2),
        "signal_premium": round(float(candidate.get("signal_premium") or 0), 2),
        "combined_flow_bias": candidate.get("combined_flow_bias", ""),
        "quality_status": candidate.get("quality_status"),
        "recommendation_status": status,
        "status_reason": liquidity_note,
        "hard_rejects": hard_rejects,
        "portfolio_risk_flag": False,
        "portfolio_risk_note": "",
        "visible_in_final_board": True,
        "flow_reason": candidate.get("flow_reason"),
    }


def construct_debit_spread(candidate: Mapping[str, Any], hot: pd.DataFrame, *, direction: str) -> dict[str, Any]:
    """Build a conservative one-lot debit spread attempt from dated UW hot-chain quotes."""

    ticker = str(candidate.get("ticker") or "").upper()
    direction = _as_text(direction)
    if direction not in {"Bull Call", "Bear Put"}:
        return priced_error_row(candidate, "debit spread", f"unsupported debit direction {direction or 'missing'}")
    close = _as_float(candidate.get("close"))
    right = "C" if direction == "Bull Call" else "P"
    structure = "bull call debit spread" if direction == "Bull Call" else "bear put debit spread"
    chain = hot[hot["ticker"].astype(str).str.upper().eq(ticker) & hot["right"].astype(str).str.upper().eq(right)].copy()
    if close is not None and close > 0:
        strikes = pd.to_numeric(chain["strike"], errors="coerce")
        if direction == "Bull Call":
            chain = chain[strikes.between(close * 0.96, close * 1.08, inclusive="both")]
        else:
            chain = chain[strikes.between(close * 0.90, close * 1.04, inclusive="both")]
    if chain.empty:
        return priced_error_row(candidate, structure, "missing suitable debit-spread long leg in dated UW hot-chain source")

    preferred = chain[
        (pd.to_numeric(chain["dte"], errors="coerce") >= MIN_LIVE_DTE)
        & (pd.to_numeric(chain["dte"], errors="coerce") <= MAX_LIVE_DTE)
    ].copy()
    if preferred.empty:
        return priced_error_row(candidate, structure, "no dated UW hot-chain debit expiry in 7-60 DTE window")
    if close is not None and close > 0:
        preferred["_moneyness_distance"] = (pd.to_numeric(preferred["strike"], errors="coerce") - close).abs()
    else:
        preferred["_moneyness_distance"] = 0.0
    long = preferred.sort_values(["_moneyness_distance", "premium", "volume"], ascending=[True, False, False]).iloc[0]
    same_expiry = chain[chain["expiry_dt"].eq(long["expiry_dt"])].copy()
    if direction == "Bull Call":
        shorts = same_expiry[pd.to_numeric(same_expiry["strike"], errors="coerce") > float(long["strike"])].sort_values("strike", ascending=True)
    else:
        shorts = same_expiry[pd.to_numeric(same_expiry["strike"], errors="coerce") < float(long["strike"])].sort_values("strike", ascending=False)
    if shorts.empty:
        return priced_error_row(candidate, structure, "missing debit-spread short leg in dated UW hot-chain source")

    short = shorts.iloc[0]
    long_bid = _as_float(long.get("bid")) or 0.0
    long_ask = _as_float(long.get("ask")) or 0.0
    short_bid = _as_float(short.get("bid")) or 0.0
    short_ask = _as_float(short.get("ask")) or 0.0
    long_mid = (long_bid + long_ask) / 2 if long_bid or long_ask else 0.0
    short_mid = (short_bid + short_ask) / 2 if short_bid or short_ask else 0.0
    natural_debit = long_ask - short_bid
    mid_debit = long_mid - short_mid
    entry_debit = natural_debit if natural_debit > 0 else mid_debit
    width = abs(float(short["strike"]) - float(long["strike"]))

    if entry_debit <= 0 or width <= 0 or entry_debit >= width:
        return priced_error_row(candidate, structure, "spread debit could not be estimated from dated bid/ask")

    max_profit = round(max((width - entry_debit) * 100, 0), 2)
    max_loss = round(entry_debit * 100, 2)
    debit_width_ratio = round(entry_debit / width, 4) if width > 0 else 0.0
    breakeven = float(long["strike"]) + entry_debit if direction == "Bull Call" else float(long["strike"]) - entry_debit
    buy_leg = _format_option_leg(
        _as_text(long.get("option_symbol")),
        "BUY",
        ticker=ticker,
        expiry=long["expiry_dt"],
        strike=long["strike"],
        right=right,
    )
    sell_leg = _format_option_leg(
        _as_text(short.get("option_symbol")),
        "SELL",
        ticker=ticker,
        expiry=short["expiry_dt"],
        strike=short["strike"],
        right=right,
    )
    trade_plan = _format_trade_plan(buy_leg, sell_leg, entry_debit, entry_type="DEBIT")
    bid_ask_width = max((long_ask - long_bid) + (short_ask - short_bid), 0.0)
    liquidity_note = "dated UW EOD debit-spread quote; refresh Schwab chain before entry"
    status = RecommendationStatus.REVIEW.value
    hard_rejects = ""
    quality_rejects = _debit_trade_quality_rejects(
        entry_debit=entry_debit,
        debit_width_ratio=debit_width_ratio,
        max_profit=max_profit,
        max_loss=max_loss,
        signal_premium=_as_float(candidate.get("signal_premium")) or 0.0,
        combined_flow_bias=_as_float(candidate.get("combined_flow_bias")) or 0.0,
        macro_tape_candidate=_truthy(candidate.get("macro_tape_candidate")),
    )
    if bid_ask_width > max(entry_debit, 0.01) * 1.5:
        status = RecommendationStatus.WAIT_FOR_PRICE.value
        liquidity_note = "dated debit-spread bid/ask width is wide; keep visible and re-price with fresh Schwab chain"
    if quality_rejects:
        status = RecommendationStatus.AVOID.value
        hard_rejects = "; ".join(quality_rejects)
        liquidity_note = _append_reason(liquidity_note, "setup quality gate reject: " + hard_rejects)
    underlying_tier, underlying_reason = _underlying_quality(candidate)

    return {
        "ticker": ticker,
        "full_name": candidate.get("full_name", ""),
        "issue_type": candidate.get("issue_type", ""),
        "marketcap": candidate.get("marketcap", ""),
        "avg30_volume": candidate.get("avg30_volume", ""),
        "total_volume": candidate.get("total_volume", ""),
        "total_open_interest": candidate.get("total_open_interest", ""),
        "underlying_quality_tier": underlying_tier,
        "underlying_quality_reason": underlying_reason,
        "bias": candidate.get("bias"),
        "flow_bias_label": candidate.get("flow_bias_label", candidate.get("bias")),
        "candidate_source": candidate.get("candidate_source", "flow_oi"),
        "macro_tape_candidate": candidate.get("macro_tape_candidate", False),
        "macro_tape_direction": candidate.get("macro_tape_direction", ""),
        "macro_tape_reason": candidate.get("macro_tape_reason", ""),
        "price_move_pct": candidate.get("price_move_pct", ""),
        "price_tape_source": candidate.get("price_tape_source", ""),
        "structure": structure,
        "full_ticket": trade_plan,
        "trade_plan": trade_plan,
        "expiry": str(long["expiry_dt"]),
        "dte": int(float(long.get("dte") or 0)),
        "sell_leg": sell_leg,
        "buy_leg": buy_leg,
        "short_leg": sell_leg,
        "long_leg": buy_leg,
        "anchor_expiry": str(candidate.get("hot_top_expiry") or long["expiry_dt"]),
        "anchor_strike": float(candidate.get("hot_top_strike") or long["strike"]),
        "short_strike": float(short["strike"]),
        "long_strike": float(long["strike"]),
        "spread_width": width,
        "entry_limit": round(entry_debit, 2),
        "mid": round(mid_debit, 2),
        "bid": "",
        "ask": round(natural_debit, 2),
        "max_profit": max_profit,
        "max_loss": max_loss,
        "credit_width_ratio": 0.0,
        "debit_width_ratio": debit_width_ratio,
        "trade_quality_status": "rejected" if quality_rejects else "reviewable",
        "quality_gate_reason": "; ".join(quality_rejects),
        "remaining_upside": max_profit,
        "breakeven": round(breakeven, 2),
        "target_exit": round(min(width * 0.80, entry_debit * 1.80), 2),
        "target_entry": round(width * 0.45, 2),
        "invalidation": "fresh quote fails, thesis breaks, or underlying fails to progress toward breakeven",
        "score": round(float(candidate.get("score") or 0), 2),
        "signal_premium": round(float(candidate.get("signal_premium") or 0), 2),
        "combined_flow_bias": candidate.get("combined_flow_bias", ""),
        "quality_status": candidate.get("quality_status"),
        "recommendation_status": status,
        "status_reason": liquidity_note,
        "hard_rejects": hard_rejects,
        "portfolio_risk_flag": False,
        "portfolio_risk_note": "",
        "visible_in_final_board": True,
        "flow_reason": candidate.get("flow_reason"),
    }


def construct_short_put(candidate: Mapping[str, Any], hot: pd.DataFrame) -> dict[str, Any]:
    """Build a one-lot cash-secured put attempt from dated UW hot-chain quotes."""

    ticker = str(candidate.get("ticker") or "").upper()
    close = _as_float(candidate.get("close"))
    chain = hot[hot["ticker"].astype(str).str.upper().eq(ticker) & hot["right"].astype(str).str.upper().eq("P")].copy()
    if close is not None:
        chain = chain[chain["strike"] < close]
        distance = (close - pd.to_numeric(chain["strike"], errors="coerce")) / close
        chain = chain[distance >= MIN_SHORT_PUT_DISTANCE_PCT]
    if chain.empty:
        return priced_error_row(candidate, "cash secured put", "missing suitable short-put leg in dated UW hot-chain source")

    preferred = chain[
        (pd.to_numeric(chain["dte"], errors="coerce") >= MIN_LIVE_DTE)
        & (pd.to_numeric(chain["dte"], errors="coerce") <= MAX_LIVE_DTE)
    ].copy()
    if preferred.empty:
        return priced_error_row(candidate, "cash secured put", "no dated UW hot-chain put expiry in 7-60 DTE window")
    short = preferred.sort_values(["premium", "volume"], ascending=[False, False]).iloc[0]
    short_bid = _as_float(short.get("bid")) or 0.0
    short_ask = _as_float(short.get("ask")) or 0.0
    short_mid = (short_bid + short_ask) / 2 if short_bid or short_ask else (_as_float(short.get("premium")) or 0.0)
    entry_credit = short_bid if short_bid > 0 else short_mid
    short_strike = float(short["strike"])
    if entry_credit <= 0 or short_strike <= 0:
        return priced_error_row(candidate, "cash secured put", "short-put credit could not be estimated from dated bid/ask")

    max_profit = round(entry_credit * 100, 2)
    max_loss = round(max((short_strike - entry_credit) * 100, 0), 2)
    breakeven = short_strike - entry_credit
    sell_leg = _format_option_leg(
        _as_text(short.get("option_symbol")),
        "SELL",
        ticker=ticker,
        expiry=short["expiry_dt"],
        strike=short_strike,
        right="P",
    )
    trade_plan = _format_trade_plan(sell_leg, "", entry_credit, entry_type="CREDIT")
    bid_ask_width = max(short_ask - short_bid, 0.0)
    liquidity_note = "dated UW EOD short-put quote; refresh Schwab chain before entry"
    status = RecommendationStatus.REVIEW.value
    hard_rejects = ""
    quality_rejects = _short_put_quality_rejects(
        entry_credit=entry_credit,
        max_loss=max_loss,
        signal_premium=_as_float(candidate.get("signal_premium")) or 0.0,
        combined_flow_bias=_as_float(candidate.get("combined_flow_bias")) or 0.0,
        macro_tape_candidate=_truthy(candidate.get("macro_tape_candidate")),
        spot=close,
        short_strike=short_strike,
    )
    if bid_ask_width > max(entry_credit, 0.01) * 1.5:
        status = RecommendationStatus.WAIT_FOR_PRICE.value
        liquidity_note = "dated short-put bid/ask width is wide; keep visible and re-price with fresh Schwab chain"
    if quality_rejects:
        status = RecommendationStatus.AVOID.value
        hard_rejects = "; ".join(quality_rejects)
        liquidity_note = _append_reason(liquidity_note, "setup quality gate reject: " + hard_rejects)
    underlying_tier, underlying_reason = _underlying_quality(candidate)

    return {
        "ticker": ticker,
        "full_name": candidate.get("full_name", ""),
        "issue_type": candidate.get("issue_type", ""),
        "marketcap": candidate.get("marketcap", ""),
        "avg30_volume": candidate.get("avg30_volume", ""),
        "total_volume": candidate.get("total_volume", ""),
        "total_open_interest": candidate.get("total_open_interest", ""),
        "underlying_quality_tier": underlying_tier,
        "underlying_quality_reason": underlying_reason,
        "bias": candidate.get("bias"),
        "flow_bias_label": candidate.get("flow_bias_label", candidate.get("bias")),
        "candidate_source": candidate.get("candidate_source", "flow_oi"),
        "macro_tape_candidate": candidate.get("macro_tape_candidate", False),
        "macro_tape_direction": candidate.get("macro_tape_direction", ""),
        "macro_tape_reason": candidate.get("macro_tape_reason", ""),
        "price_move_pct": candidate.get("price_move_pct", ""),
        "price_tape_source": candidate.get("price_tape_source", ""),
        "structure": "cash secured put",
        "full_ticket": trade_plan,
        "trade_plan": trade_plan,
        "expiry": str(short["expiry_dt"]),
        "dte": int(float(short.get("dte") or 0)),
        "sell_leg": sell_leg,
        "buy_leg": "",
        "short_leg": sell_leg,
        "long_leg": "",
        "anchor_expiry": str(candidate.get("hot_top_expiry") or short["expiry_dt"]),
        "anchor_strike": float(candidate.get("hot_top_strike") or short_strike),
        "short_strike": short_strike,
        "long_strike": "",
        "spread_width": "",
        "entry_limit": round(entry_credit, 2),
        "mid": round(short_mid, 2),
        "bid": round(short_bid, 2),
        "ask": round(short_ask, 2) if short_ask else "",
        "max_profit": max_profit,
        "max_loss": max_loss,
        "credit_width_ratio": "",
        "short_put_cash_required": max_loss,
        "trade_quality_status": "rejected" if quality_rejects else "reviewable",
        "quality_gate_reason": "; ".join(quality_rejects),
        "remaining_upside": max_profit,
        "breakeven": round(breakeven, 2),
        "target_exit": round(entry_credit * 0.35, 2),
        "target_entry": max(MIN_SEND_NOW_CREDIT, round(entry_credit, 2)),
        "invalidation": "fresh quote fails, thesis breaks, or underlying violates breakeven/flow support",
        "score": round(float(candidate.get("score") or 0), 2),
        "signal_premium": round(float(candidate.get("signal_premium") or 0), 2),
        "combined_flow_bias": candidate.get("combined_flow_bias", ""),
        "quality_status": candidate.get("quality_status"),
        "recommendation_status": status,
        "status_reason": liquidity_note,
        "hard_rejects": hard_rejects,
        "portfolio_risk_flag": False,
        "portfolio_risk_note": "",
        "visible_in_final_board": True,
        "flow_reason": candidate.get("flow_reason"),
    }


def priced_error_row(candidate: Mapping[str, Any], structure: str, reason: str) -> dict[str, Any]:
    """Return a visible priced-candidate row when construction fails."""

    underlying_tier, underlying_reason = _underlying_quality(candidate)
    return {
        "ticker": str(candidate.get("ticker") or "").upper(),
        "full_name": candidate.get("full_name", ""),
        "issue_type": candidate.get("issue_type", ""),
        "marketcap": candidate.get("marketcap", ""),
        "avg30_volume": candidate.get("avg30_volume", ""),
        "total_volume": candidate.get("total_volume", ""),
        "total_open_interest": candidate.get("total_open_interest", ""),
        "underlying_quality_tier": underlying_tier,
        "underlying_quality_reason": underlying_reason,
        "bias": candidate.get("bias"),
        "flow_bias_label": candidate.get("flow_bias_label", candidate.get("bias")),
        "candidate_source": candidate.get("candidate_source", "flow_oi"),
        "macro_tape_candidate": candidate.get("macro_tape_candidate", False),
        "macro_tape_direction": candidate.get("macro_tape_direction", ""),
        "macro_tape_reason": candidate.get("macro_tape_reason", ""),
        "price_move_pct": candidate.get("price_move_pct", ""),
        "price_tape_source": candidate.get("price_tape_source", ""),
        "structure": structure,
        "full_ticket": "",
        "trade_plan": "",
        "expiry": "",
        "dte": "",
        "sell_leg": "",
        "buy_leg": "",
        "short_leg": "",
        "long_leg": "",
        "anchor_expiry": str(candidate.get("hot_top_expiry") or ""),
        "anchor_strike": candidate.get("hot_top_strike") if candidate.get("hot_top_strike") is not None else "",
        "entry_limit": "",
        "mid": "",
        "bid": "",
        "ask": "",
        "max_profit": "",
        "max_loss": "",
        "credit_width_ratio": "",
        "trade_quality_status": "no_structure",
        "quality_gate_reason": reason,
        "remaining_upside": "",
        "breakeven": "",
        "target_exit": "",
        "invalidation": reason,
        "score": round(float(candidate.get("score") or 0), 2),
        "signal_premium": round(float(candidate.get("signal_premium") or 0), 2),
        "combined_flow_bias": candidate.get("combined_flow_bias", ""),
        "quality_status": candidate.get("quality_status"),
        "recommendation_status": RecommendationStatus.REVIEW.value,
        "status_reason": f"{reason}; live chain expansion required",
        "hard_rejects": "",
        "portfolio_risk_flag": False,
        "portfolio_risk_note": "",
        "visible_in_final_board": True,
        "flow_reason": candidate.get("flow_reason"),
    }


def build_no_trade_audit(
    candidates: pd.DataFrame,
    priced: pd.DataFrame,
    *,
    top_trades: int,
    raw_universe: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Keep every non-priced or rejected candidate visible with reasons."""

    # Kept for API compatibility; this audit surface must not be capped.
    _ = top_trades
    rows: list[dict[str, Any]] = []
    priced_tickers = set(priced.get("ticker", pd.Series(dtype=str)).astype(str).str.upper()) if not priced.empty else set()
    candidate_tickers = set(candidates.get("ticker", pd.Series(dtype=str)).astype(str).str.upper()) if not candidates.empty else set()
    for _, row in candidates.iterrows():
        ticker = str(row.get("ticker") or "").upper()
        if row.get("quality_status") != "qualified" or ticker not in priced_tickers:
            rows.append(
                {
                    "ticker": ticker,
                    "bias": row.get("bias"),
                    "score": round(float(row.get("score") or 0), 2),
                    "reason": row.get("flow_reason") or row.get("status_reason"),
                    "hard_blocker": _no_trade_hard_blocker(row),
                }
            )
    if raw_universe is not None and not raw_universe.empty and "ticker" in raw_universe.columns:
        for _, row in raw_universe.iterrows():
            ticker = str(row.get("ticker") or "").strip().upper()
            if not ticker or ticker in candidate_tickers:
                continue
            if not _truthy(row.get("core_universe_member")) and _as_text(row.get("underlying_quality_tier")).lower() != "core":
                continue
            if _as_text(row.get("macro_tape_direction")).lower() not in {"bearish", "bullish"}:
                continue
            rows.append(
                {
                    "ticker": ticker,
                    "bias": row.get("bias"),
                    "score": round(float(row.get("score") or 0), 2),
                    "reason": _append_reason(
                        row.get("flow_reason"),
                        "macro tape reviewed but did not clear candidate promotion gates",
                    ),
                    "hard_blocker": "macro_tape_rejected_before_candidate_generation",
                }
            )
    return pd.DataFrame(rows)


def _no_trade_hard_blocker(row: Mapping[str, Any]) -> str:
    if row.get("quality_status") == "qualified":
        return ""
    if _as_text(row.get("macro_tape_direction")).lower() in {"bearish", "bullish"}:
        return "macro_tape_rejected"
    return "insufficient_score_or_neutral_bias"


def build_coverage_audit(
    raw_universe: pd.DataFrame,
    candidates: pd.DataFrame,
    priced: pd.DataFrame,
    decision_board: pd.DataFrame,
    no_trade: pd.DataFrame,
    *,
    watchlist: Sequence[str] = CORE_AUDIT_TICKERS,
) -> pd.DataFrame:
    """Explain where important tickers went in the pipeline."""

    columns = [
        "ticker",
        "coverage_status",
        "status_color",
        "raw_rank",
        "candidate_rank",
        "bias",
        "score",
        "quality_status",
        "underlying_quality_tier",
        "underlying_quality_reason",
        "marketcap",
        "avg30_volume",
        "total_open_interest",
        "signal_premium",
        "candidate_source",
        "macro_tape_direction",
        "macro_tape_candidate",
        "final_action",
        "execution_status",
        "trade_plan",
        "reason",
        "next_step",
    ]
    raw_by_ticker = _frame_by_ticker(raw_universe)
    candidate_by_ticker = _frame_by_ticker(candidates)
    priced_by_ticker = _frame_by_ticker(priced)
    decision_by_ticker = _frame_by_ticker(decision_board)
    no_trade_by_ticker = _frame_by_ticker(no_trade)
    raw_ranks: dict[str, int] = {}
    if not raw_universe.empty and {"ticker", "score", "signal_premium"}.issubset(raw_universe.columns):
        raw = raw_universe.copy()
        if "underlying_quality_rank" not in raw.columns:
            raw["underlying_quality_rank"] = raw.get("underlying_quality_tier", pd.Series("", index=raw.index)).map(
                _underlying_quality_sort_rank
            )
        ranked = raw.sort_values(["underlying_quality_rank", "score", "signal_premium"], ascending=[True, False, False]).reset_index(drop=True)
        raw_ranks = {str(row.get("ticker") or "").upper(): idx + 1 for idx, row in ranked.iterrows()}

    tickers: list[str] = []
    expanded_tickers = _coverage_ticker_universe(raw_universe, candidates, priced, decision_board, no_trade, watchlist)
    for ticker in expanded_tickers:
        clean = str(ticker or "").strip().upper()
        if clean and clean not in tickers:
            tickers.append(clean)

    rows: list[dict[str, Any]] = []
    for ticker in tickers:
        raw = raw_by_ticker.get(ticker, {})
        candidate = candidate_by_ticker.get(ticker, {})
        priced_row = priced_by_ticker.get(ticker, {})
        decision = decision_by_ticker.get(ticker, {})
        no_trade_row = no_trade_by_ticker.get(ticker, {})
        source = decision or priced_row or candidate or raw
        trade_plan = _as_text(decision.get("trade_plan")) or _as_text(priced_row.get("trade_plan"))
        ready = _truthy(decision.get("ready_to_enter"))
        execution = _as_text(decision.get("execution_status"))
        final_action = _as_text(decision.get("final_action"))
        if decision:
            target_status = _as_text(decision.get("target_order_status"))
            if trade_plan and ready:
                status = "READY_TICKET"
            elif target_status == "target_order_candidate":
                status = "TARGET_ORDER_CANDIDATE"
            elif target_status == "not_actionable_underlying_quality":
                status = "NON_ACTIONABLE_UNDERLYING"
            elif target_status == "not_actionable_unvalidated_chain":
                status = "UNVALIDATED_CHAIN"
            elif execution == "blocked":
                status = "BLOCKED_FINAL_ROW"
            elif trade_plan:
                status = "REVIEW_TICKET"
            else:
                status = "FINAL_NO_TICKET"
        elif priced_row:
            if _truthy(priced_row.get("macro_tape_candidate")) and not trade_plan:
                status = "MACRO_TAPE_NO_LIVE_EDGE"
            else:
                status = "STRUCTURED_NOT_TOP_FINAL" if trade_plan else "STRUCTURE_MISSING"
        elif candidate:
            status = "MACRO_TAPE_CANDIDATE" if _truthy(candidate.get("macro_tape_candidate")) else "CANDIDATE_NOT_STRUCTURED"
        elif raw:
            if _as_text(raw.get("macro_tape_direction")).lower() in {"bearish", "bullish"}:
                status = "MACRO_TAPE_REJECTED"
            else:
                status = "NO_DIRECTIONAL_EDGE" if _as_text(raw.get("bias")) == "neutral" else "BELOW_DISCOVERY_CUTOFF"
        else:
            status = "SOURCE_MISSING"
        if _is_audit_only_underlying(source) and status in _LOW_QUALITY_COVERAGE_STATUSES:
            status = "NON_ACTIONABLE_UNDERLYING"

        reason = _coverage_reason(status, ticker, raw, candidate, priced_row, decision, no_trade_row)
        rows.append(
            {
                "ticker": ticker,
                "coverage_status": status,
                "status_color": _coverage_color(status),
                "raw_rank": raw_ranks.get(ticker, ""),
                "candidate_rank": candidate.get("candidate_rank", ""),
                "bias": source.get("bias", ""),
                "score": source.get("score", ""),
                "quality_status": source.get("quality_status", ""),
                "underlying_quality_tier": source.get("underlying_quality_tier", ""),
                "underlying_quality_reason": source.get("underlying_quality_reason", ""),
                "marketcap": source.get("marketcap", ""),
                "avg30_volume": source.get("avg30_volume", ""),
                "total_open_interest": source.get("total_open_interest", ""),
                "signal_premium": source.get("signal_premium", ""),
                "candidate_source": source.get("candidate_source", ""),
                "macro_tape_direction": source.get("macro_tape_direction", ""),
                "macro_tape_candidate": source.get("macro_tape_candidate", ""),
                "final_action": final_action,
                "execution_status": execution,
                "trade_plan": trade_plan,
                "reason": reason,
                "next_step": _coverage_next_step(status, decision),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _coverage_ticker_universe(
    raw_universe: pd.DataFrame,
    candidates: pd.DataFrame,
    priced: pd.DataFrame,
    decision_board: pd.DataFrame,
    no_trade: pd.DataFrame,
    watchlist: Sequence[str],
) -> list[str]:
    tickers: list[str] = []

    def add(values: Iterable[Any]) -> None:
        for value in values:
            ticker = _as_text(value).upper()
            if ticker and ticker not in {"NAN", "NONE", "NULL"} and ticker not in tickers:
                tickers.append(ticker)

    add(watchlist)
    for frame in (decision_board, priced, candidates.head(50) if not candidates.empty else candidates, no_trade):
        if frame is not None and not frame.empty and "ticker" in frame.columns:
            add(frame["ticker"].tolist())
    if raw_universe is not None and not raw_universe.empty and "ticker" in raw_universe.columns:
        raw = raw_universe.copy()
        if "underlying_quality_rank" not in raw.columns:
            raw["underlying_quality_rank"] = raw.get("underlying_quality_tier", pd.Series("", index=raw.index)).map(
                _underlying_quality_sort_rank
            )
        sort_cols = [col for col in ["underlying_quality_rank", "score", "signal_premium"] if col in raw.columns]
        if sort_cols:
            ascending = [True] + [False] * (len(sort_cols) - 1)
            raw = raw.sort_values(sort_cols, ascending=ascending)
        liquid = raw[
            raw.get("underlying_quality_tier", pd.Series("", index=raw.index)).astype(str).str.lower().isin(["core", "liquid"])
        ]
        add(liquid.head(50)["ticker"].tolist())
    return tickers


def _coverage_reason(
    status: str,
    ticker: str,
    raw: Mapping[str, Any],
    candidate: Mapping[str, Any],
    priced_row: Mapping[str, Any],
    decision: Mapping[str, Any],
    no_trade_row: Mapping[str, Any],
) -> str:
    if status == "NON_ACTIONABLE_UNDERLYING":
        source = decision or priced_row or candidate or raw
        tier = _as_text(source.get("underlying_quality_tier")) or "low-quality"
        reason = (
            _as_text(source.get("underlying_quality_reason"))
            or _as_text(no_trade_row.get("reason"))
            or _as_text(source.get("flow_reason"))
            or _as_text(source.get("status_reason"))
            or "underlying quality/liquidity below action threshold"
        )
        return f"not actionable: {tier} underlying; {reason}"
    if decision:
        if _as_text(decision.get("target_order_status")) == "not_actionable_underlying_quality":
            return "not actionable: " + (
                _as_text(decision.get("underlying_quality_reason")) or "underlying quality/liquidity below action threshold"
            )
        if _as_text(decision.get("target_order_status")) == "not_actionable_unvalidated_chain":
            return "not actionable until Schwab chain validates target price: " + (
                _as_text(decision.get("status_reason")) or "fresh chain validation required"
            )
        if _as_text(decision.get("quality_gate_reason")):
            return "setup quality gate reject: " + _as_text(decision.get("quality_gate_reason"))
        return (
            _as_text(decision.get("status_reason"))
            or _as_text(decision.get("synthesis_reason"))
            or _as_text(decision.get("portfolio_risk_note"))
            or "visible in final decision board"
        )
    if priced_row:
        if status == "MACRO_TAPE_NO_LIVE_EDGE":
            return _as_text(priced_row.get("status_reason")) or "macro-tape candidate could not find a complete liquid structure"
        if _as_text(priced_row.get("trade_plan")):
            return "structured, but did not rank into the final visible board after synthesis"
        return _as_text(priced_row.get("status_reason")) or _as_text(priced_row.get("invalidation")) or "structure was attempted without a ticket"
    if candidate:
        if status == "MACRO_TAPE_CANDIDATE":
            return _as_text(candidate.get("macro_tape_reason")) or "macro-tape candidate awaiting structure/live-chain validation"
        return _as_text(no_trade_row.get("reason")) or _as_text(candidate.get("status_reason")) or "candidate generated but not structured"
    if raw:
        score = _as_float(raw.get("score"))
        score_text = f"{score:.2f}" if score is not None else "unscored"
        if status == "MACRO_TAPE_REJECTED":
            return (
                _as_text(raw.get("macro_tape_reason"))
                or f"macro tape was directional, but {ticker} did not clear candidate promotion gates; score {score_text}"
            )
        if status == "NO_DIRECTIONAL_EDGE":
            return f"neutral flow bias in UW sources; score {score_text}; no directional spread should be forced"
        return f"raw universe only; score {score_text}; {(_as_text(raw.get('flow_reason')) or 'below candidate generation priority')}"
    return f"{ticker} was not present in the loaded UW ticker universe"


def _coverage_color(status: str) -> str:
    if status == "READY_TICKET":
        return "green"
    if status in {
        "TARGET_ORDER_CANDIDATE",
        "REVIEW_TICKET",
        "STRUCTURED_NOT_TOP_FINAL",
        "CANDIDATE_NOT_STRUCTURED",
        "FINAL_NO_TICKET",
        "UNVALIDATED_CHAIN",
        "MACRO_TAPE_CANDIDATE",
        "MACRO_TAPE_NO_LIVE_EDGE",
    }:
        return "yellow"
    if status in {"BLOCKED_FINAL_ROW", "NON_ACTIONABLE_UNDERLYING", "STRUCTURE_MISSING", "SOURCE_MISSING"}:
        return "red"
    return "gray"


_LOW_QUALITY_COVERAGE_STATUSES = {
    "TARGET_ORDER_CANDIDATE",
    "REVIEW_TICKET",
    "STRUCTURED_NOT_TOP_FINAL",
    "CANDIDATE_NOT_STRUCTURED",
    "FINAL_NO_TICKET",
    "UNVALIDATED_CHAIN",
    "STRUCTURE_MISSING",
    "NO_DIRECTIONAL_EDGE",
    "BELOW_DISCOVERY_CUTOFF",
    "MACRO_TAPE_CANDIDATE",
    "MACRO_TAPE_NO_LIVE_EDGE",
    "MACRO_TAPE_REJECTED",
}


def _is_audit_only_underlying(row: Mapping[str, Any]) -> bool:
    tier = _as_text(row.get("underlying_quality_tier")).lower()
    return tier not in {"core"}


def _coverage_next_step(status: str, row: Optional[Mapping[str, Any]] = None) -> str:
    if status == "READY_TICKET":
        return "verify live quote and place manually if thesis still holds"
    if status == "TARGET_ORDER_CANDIDATE":
        return _ticket_next_step(row or {})
    if status == "NON_ACTIONABLE_UNDERLYING":
        return "do not trade from the action list; require explicit override and fresh validation"
    if status == "UNVALIDATED_CHAIN":
        return "rebuild with live Schwab chain before considering an order"
    if status == "REVIEW_TICKET":
        row_data = {} if row is None else row
        blockers = _blocker_set(row_data.get("execution_blockers"))
        if "trade_quality_review_required" in blockers or _as_text(row_data.get("target_order_status")) == "review_only_low_trade_quality":
            return "reprice in Schwab and resolve trade-quality review"
        if "portfolio_context_required" in blockers:
            return "refresh portfolio context before manual entry"
        if _as_text(row_data.get("portfolio_risk_note")) or _truthy(row_data.get("requires_portfolio_ack")):
            return "reprice in Schwab and resolve catalyst/quality review"
        return "reprice in Schwab and resolve catalyst/quality review"
    if status == "STRUCTURED_NOT_TOP_FINAL":
        return "inspect priced_candidates.csv; promote only if manual thesis beats top board"
    if status == "STRUCTURE_MISSING":
        return "use live Schwab chain to find a liquid expiry/width"
    if status == "BLOCKED_FINAL_ROW":
        return "do not trade unless the objective blocker is cleared in a fresh run"
    if status == "CANDIDATE_NOT_STRUCTURED":
        return "run structure expansion or live-chain construction"
    if status == "MACRO_TAPE_CANDIDATE":
        return "run subagent review and live-chain structure validation"
    if status == "MACRO_TAPE_NO_LIVE_EDGE":
        return "use live Schwab chain to find a liquid macro-aligned spread"
    if status == "MACRO_TAPE_REJECTED":
        return "watch only unless macro tape strengthens or score/liquidity improves"
    if status == "NO_DIRECTIONAL_EDGE":
        return "wait for directional flow or use a neutral strategy outside this credit-spread path"
    if status == "BELOW_DISCOVERY_CUTOFF":
        return "watch only unless fresh flow improves score or manual focus override is desired"
    return "check UW source coverage"


def _trade_quality_rejects(
    *,
    entry_credit: float,
    credit_width_ratio: float,
    max_loss: float,
    signal_premium: float,
    combined_flow_bias: float,
    macro_tape_candidate: bool = False,
) -> list[str]:
    rejects: list[str] = []
    if entry_credit < MIN_TRADE_CREDIT:
        rejects.append(f"entry_credit_below_{MIN_TRADE_CREDIT:.2f}")
    if credit_width_ratio < MIN_CREDIT_WIDTH_RATIO:
        rejects.append(f"credit_width_ratio_below_{int(MIN_CREDIT_WIDTH_RATIO * 100)}pct")
    if signal_premium < MIN_SIGNAL_PREMIUM:
        rejects.append(f"signal_premium_below_{int(MIN_SIGNAL_PREMIUM):d}")
    if not macro_tape_candidate and abs(combined_flow_bias) < MIN_DIRECTIONAL_BIAS:
        rejects.append(f"directional_bias_below_{MIN_DIRECTIONAL_BIAS:.2f}")
    if max_loss > MAX_ONE_LOT_LOSS:
        rejects.append(f"one_lot_max_loss_above_{int(MAX_ONE_LOT_LOSS):d}")
    return rejects


def _short_put_quality_rejects(
    *,
    entry_credit: float,
    max_loss: float,
    signal_premium: float,
    combined_flow_bias: float,
    macro_tape_candidate: bool = False,
    spot: Optional[float] = None,
    short_strike: Optional[float] = None,
) -> list[str]:
    rejects: list[str] = []
    if entry_credit < MIN_TRADE_CREDIT:
        rejects.append(f"entry_credit_below_{MIN_TRADE_CREDIT:.2f}")
    if max_loss <= 0:
        rejects.append("short_put_cash_risk_unavailable")
    if signal_premium < MIN_SIGNAL_PREMIUM:
        rejects.append(f"signal_premium_below_{int(MIN_SIGNAL_PREMIUM):d}")
    if not macro_tape_candidate and combined_flow_bias < MIN_DIRECTIONAL_BIAS:
        rejects.append(f"directional_bias_below_{MIN_DIRECTIONAL_BIAS:.2f}")
    if spot is not None and short_strike is not None and spot > 0:
        distance = (spot - short_strike) / spot
        if distance < MIN_SHORT_PUT_DISTANCE_PCT:
            rejects.append(f"short_put_distance_below_{int(MIN_SHORT_PUT_DISTANCE_PCT * 100)}pct")
    return rejects


def _debit_trade_quality_rejects(
    *,
    entry_debit: float,
    debit_width_ratio: float,
    max_profit: float,
    max_loss: float,
    signal_premium: float,
    combined_flow_bias: float,
    macro_tape_candidate: bool = False,
) -> list[str]:
    rejects: list[str] = []
    if entry_debit <= 0:
        rejects.append("entry_debit_required")
    if debit_width_ratio > 0.65:
        rejects.append("debit_width_ratio_above_65pct")
    if max_profit <= 0:
        rejects.append("positive_max_profit_required")
    if signal_premium < MIN_SIGNAL_PREMIUM:
        rejects.append(f"signal_premium_below_{int(MIN_SIGNAL_PREMIUM):d}")
    if not macro_tape_candidate and abs(combined_flow_bias) < MIN_DIRECTIONAL_BIAS:
        rejects.append(f"directional_bias_below_{MIN_DIRECTIONAL_BIAS:.2f}")
    if max_loss > MAX_ONE_LOT_LOSS:
        rejects.append(f"one_lot_max_loss_above_{int(MAX_ONE_LOT_LOSS):d}")
    return rejects


def _live_spread_quality_rejects(live: Mapping[str, Any]) -> list[str]:
    rejects: list[str] = []
    quote_width = _as_float(live.get("quote_width_pct"))
    if quote_width is not None and quote_width > MAX_LIVE_QUOTE_WIDTH_PCT:
        rejects.append(f"live_quote_width_pct_above_{int(MAX_LIVE_QUOTE_WIDTH_PCT * 100)}pct")

    leg_liquidity = []
    for prefix in ("short", "long"):
        oi = _as_float(live.get(f"{prefix}_oi"))
        volume = _as_float(live.get(f"{prefix}_volume"))
        if oi is None and volume is None:
            continue
        leg_liquidity.append((oi or 0.0) + (volume or 0.0))
    if leg_liquidity:
        min_liquidity = min(leg_liquidity)
        if min_liquidity < MIN_LIVE_LEG_LIQUIDITY:
            rejects.append(f"live_leg_liquidity_below_{int(MIN_LIVE_LEG_LIQUIDITY)}")
    else:
        rejects.append("live_leg_liquidity_missing")
    return rejects


def _live_alternative_quality_rejects(
    row: Mapping[str, Any],
    live: Mapping[str, Any],
    *,
    entry_type: str,
) -> list[str]:
    """Evaluate a Schwab-chain alternative before mutating the candidate row."""

    if str(live.get("live_status") or "").upper() != "PASS":
        return [str(live.get("live_status") or "no_live_alternative")]
    width = _as_float(live.get("spread_width")) or 0.0
    signal_premium = _as_float(row.get("signal_premium")) or 0.0
    combined_flow_bias = _as_float(row.get("combined_flow_bias")) or 0.0
    macro_tape_candidate = _truthy(row.get("macro_tape_candidate"))
    if entry_type == "CREDIT":
        credit = _as_float(live.get("credit")) or 0.0
        rejects = _trade_quality_rejects(
            entry_credit=credit,
            credit_width_ratio=credit / width if width > 0 else 0.0,
            max_loss=max((width - credit) * 100.0, 0.0),
            signal_premium=signal_premium,
            combined_flow_bias=combined_flow_bias,
            macro_tape_candidate=macro_tape_candidate,
        )
    else:
        debit = _as_float(live.get("debit")) or 0.0
        rejects = _debit_trade_quality_rejects(
            entry_debit=debit,
            debit_width_ratio=debit / width if width > 0 else 0.0,
            max_profit=max((width - debit) * 100.0, 0.0),
            max_loss=debit * 100.0,
            signal_premium=signal_premium,
            combined_flow_bias=combined_flow_bias,
            macro_tape_candidate=macro_tape_candidate,
        )
    rejects.extend(_live_spread_quality_rejects(live))
    return rejects


def _live_alternative_target_met(live: Mapping[str, Any], *, entry_type: str) -> bool:
    width = _as_float(live.get("spread_width")) or 0.0
    target = _as_float(live.get("target_entry"))
    if entry_type == "CREDIT":
        credit = _as_float(live.get("credit")) or 0.0
        target = target if target is not None else round(width * MIN_CREDIT_WIDTH_RATIO, 2)
        return credit >= target
    debit = _as_float(live.get("debit")) or 0.0
    target = target if target is not None else round(width * 0.45, 2)
    return debit <= target


def _live_alternative_send_now_economics(live: Mapping[str, Any], *, entry_type: str) -> bool:
    width = _as_float(live.get("spread_width")) or 0.0
    if width <= 0:
        return False
    if entry_type == "CREDIT":
        credit = _as_float(live.get("credit")) or 0.0
        return credit >= MIN_SEND_NOW_CREDIT and (credit / width) >= MIN_SEND_NOW_CREDIT_WIDTH_RATIO
    debit = _as_float(live.get("debit")) or 0.0
    max_profit = width - debit
    reward_risk = max_profit / debit if debit > 0 else 0.0
    return reward_risk >= MIN_SEND_NOW_DEBIT_REWARD_RISK_RATIO


def _select_live_alternative(
    row: Mapping[str, Any],
    alternatives: Sequence[Mapping[str, Any]],
    *,
    entry_type: str,
) -> Mapping[str, Any]:
    """Prefer actionable Schwab alternatives over merely flow-anchored ones."""

    valid = [dict(alt) for alt in alternatives if isinstance(alt, Mapping)]
    if not valid:
        return {"live_status": "no_live_alternative", "live_blocker": "no alternative returned"}

    entry_type = entry_type.upper()

    def score(alt: Mapping[str, Any]) -> tuple[float, ...]:
        rejects = _live_alternative_quality_rejects(row, alt, entry_type=entry_type)
        quality_pass = 1.0 if not rejects else 0.0
        send_now = 1.0 if quality_pass and _live_alternative_send_now_economics(alt, entry_type=entry_type) else 0.0
        target_met = 1.0 if quality_pass and _live_alternative_target_met(alt, entry_type=entry_type) else 0.0
        reject_penalty = -float(len(rejects))
        liq = _as_float(alt.get("liq_score")) or 0.0
        quote_width = _as_float(alt.get("quote_width_pct"))
        quote_score = -(quote_width if quote_width is not None else 9.0)
        if entry_type == "CREDIT":
            width = _as_float(alt.get("spread_width")) or 0.0
            entry_quality = ((_as_float(alt.get("credit")) or 0.0) / width) if width > 0 else 0.0
        else:
            entry_quality = _as_float(alt.get("reward_risk")) or 0.0
        rank = _as_float(alt.get("_rank")) or 0.0
        return (quality_pass, send_now, target_met, reject_penalty, entry_quality, min(liq, 10_000.0), quote_score, rank)

    return max(valid, key=score)


def _underlying_quality(row: Mapping[str, Any]) -> tuple[str, str]:
    ticker = _as_text(row.get("ticker")).upper()
    issue_type = _as_text(row.get("issue_type")).upper()
    market_cap = _as_float(row.get("marketcap")) or 0.0
    avg_volume = _as_float(row.get("avg30_volume")) or _as_float(row.get("total_volume")) or 0.0
    option_oi = max(
        _as_float(row.get("total_open_interest")) or 0.0,
        _as_float(row.get("hot_open_interest")) or 0.0,
        _as_float(row.get("curr_oi")) or 0.0,
        _as_float(row.get("chain_oi_volume")) or 0.0,
    )
    if not issue_type and market_cap <= 0 and avg_volume <= 0 and option_oi <= 0:
        return "unknown", "underlying quality fields missing from source"
    is_index_or_etf = issue_type == "ETF" or _truthy(row.get("is_index")) or ticker in ACTIONABLE_ETF_ALLOWLIST
    if is_index_or_etf and ticker not in ACTIONABLE_ETF_ALLOWLIST:
        return "excluded", f"non-core ETF; not in actionable ETF allowlist ({ticker})"
    if is_index_or_etf and ticker in ACTIONABLE_ETF_ALLOWLIST:
        if avg_volume >= MIN_LIQUID_AVG_VOLUME and option_oi >= MIN_LIQUID_OPTION_OI:
            return "core", "core liquid ETF with sufficient stock volume and option open interest"
        return "speculative", "ETF is allowlisted but liquidity is below actionable thresholds"
    if issue_type and issue_type != "COMMON STOCK":
        return "excluded", f"unsupported issue_type={issue_type}"
    if market_cap >= MIN_CORE_MARKET_CAP and avg_volume >= MIN_LIQUID_AVG_VOLUME and option_oi >= MIN_LIQUID_OPTION_OI:
        return "core", "large-cap liquid common stock with sufficient option open interest"
    if market_cap >= MIN_LIQUID_MARKET_CAP and avg_volume >= MIN_LIQUID_AVG_VOLUME and option_oi >= MIN_LIQUID_OPTION_OI:
        return "liquid", "liquid common stock with sufficient market cap, stock volume, and option open interest"
    reasons: list[str] = []
    if market_cap < MIN_LIQUID_MARKET_CAP:
        reasons.append(f"marketcap_below_{int(MIN_LIQUID_MARKET_CAP)}")
    if avg_volume < MIN_LIQUID_AVG_VOLUME:
        reasons.append(f"avg_volume_below_{int(MIN_LIQUID_AVG_VOLUME)}")
    if option_oi < MIN_LIQUID_OPTION_OI:
        reasons.append(f"option_oi_below_{int(MIN_LIQUID_OPTION_OI)}")
    return "speculative", "; ".join(reasons) if reasons else "liquidity profile incomplete"


def _underlying_quality_sort_rank(value: Any) -> int:
    return int(UNDERLYING_QUALITY_SORT_RANK.get(_as_text(value).lower(), UNDERLYING_QUALITY_SORT_RANK["unknown"]))


def build_live_spread_quality_audit(final: pd.DataFrame) -> pd.DataFrame:
    """Expose live/snapshot spread-market quality gates as a row-level audit."""

    columns = LIVE_SPREAD_QUALITY_AUDIT_COLUMNS
    if final is None or final.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for _, row in final.iterrows():
        live_status = _as_text(row.get("live_validation_status")).upper()
        if live_status not in {
            "PASS",
            "WAIT_FOR_PRICE",
            "NO_LIVE_SPREAD",
            "NO_REALISTIC_SPREAD",
            "TARGET_QUOTE_REFRESH",
            "MARKET_CLOSED_RECHECK",
        } and not _as_text(row.get("spot_live")):
            continue
        quality_reason = _as_text(row.get("quality_gate_reason"))
        live_quality_reasons = [
            reason
            for reason in _blocker_set(quality_reason)
            if reason.startswith("live_quote_width_pct_") or reason.startswith("live_leg_liquidity_")
        ]
        min_liquidity = _live_leg_min_liquidity_from_row(row)
        liquidity_status = "PASS"
        if min_liquidity is None:
            liquidity_status = "MISSING"
        elif min_liquidity < MIN_LIVE_LEG_LIQUIDITY:
            liquidity_status = "BLOCK"
        quote_width = _as_float(row.get("live_quote_width_pct"))
        quote_width_status = quote_width is not None and quote_width > MAX_LIVE_QUOTE_WIDTH_PCT
        if live_status in {"TARGET_QUOTE_REFRESH", "MARKET_CLOSED_RECHECK"} and not live_quality_reasons and not quote_width_status:
            market_quality_status = "DEFERRED_QUOTE_REFRESH"
        else:
            market_quality_status = "BLOCK" if live_quality_reasons or quote_width_status or liquidity_status in {"BLOCK", "MISSING"} else "PASS"
        recommendation_status = _as_text(row.get("recommendation_status")).upper()
        target_status = "blocked_not_target_candidate" if recommendation_status == RecommendationStatus.AVOID.value else "visible_for_review"
        if market_quality_status == "DEFERRED_QUOTE_REFRESH":
            target_status = "target_order_price_validation"
        if market_quality_status == "PASS" and recommendation_status in {RecommendationStatus.ENTER.value, RecommendationStatus.WAIT_FOR_PRICE.value}:
            target_status = "eligible_for_yellow_or_green_surface"
        rows.append(
            {
                "recommendation_rank": row.get("recommendation_rank", ""),
                "ticker": _as_text(row.get("ticker")).upper(),
                "live_market_quality_status": market_quality_status,
                "actionability_impact": target_status,
                "recommendation_status": recommendation_status,
                "live_validation_status": live_status,
                "structure": row.get("structure", ""),
                "entry_type": _entry_type_from_ticket(row.get("trade_plan", row.get("full_ticket", ""))),
                "entry_limit": row.get("entry_limit", ""),
                "target_entry": row.get("target_entry", ""),
                "spot_live": row.get("spot_live", ""),
                "short_strike": row.get("short_strike", ""),
                "long_strike": row.get("long_strike", ""),
                "spread_width": row.get("spread_width", ""),
                "live_quote_width_pct": row.get("live_quote_width_pct", ""),
                "live_leg_min_liquidity": "" if min_liquidity is None else round(min_liquidity, 2),
                "live_leg_liquidity_status": liquidity_status,
                "quality_gate_reason": quality_reason,
                "trade_plan": row.get("trade_plan", row.get("full_ticket", "")),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def summarize_live_spread_quality(audit: pd.DataFrame) -> dict[str, Any]:
    if audit is None or audit.empty:
        return {
            "status": "not_evaluated",
            "total_rows": 0,
            "pass_rows": 0,
            "block_rows": 0,
            "quote_width_block_rows": 0,
            "liquidity_block_rows": 0,
        }
    status = audit.get("live_market_quality_status", pd.Series(dtype=object)).astype(str).str.upper()
    reasons = audit.get("quality_gate_reason", pd.Series("", index=audit.index)).astype(str)
    quote_blocks = reasons.str.contains("live_quote_width_pct_", regex=False, na=False)
    liquidity_blocks = reasons.str.contains("live_leg_liquidity_", regex=False, na=False)
    block_rows = int(status.eq("BLOCK").sum())
    deferred_rows = int(status.eq("DEFERRED_MARKET_CLOSED").sum())
    return {
        "status": "blocked_bad_live_markets" if block_rows else "deferred_market_closed" if deferred_rows else "pass",
        "total_rows": int(len(audit)),
        "pass_rows": int(status.eq("PASS").sum()),
        "block_rows": block_rows,
        "deferred_market_closed_rows": deferred_rows,
        "quote_width_block_rows": int(quote_blocks.sum()),
        "liquidity_block_rows": int(liquidity_blocks.sum()),
    }


def build_execution_fill_quality_audit(final: pd.DataFrame, trade_tickets: pd.DataFrame) -> pd.DataFrame:
    """Audit whether validated entries are at or better than the target fill."""

    columns = EXECUTION_FILL_QUALITY_COLUMNS
    if final is None or final.empty:
        return pd.DataFrame(columns=columns)
    ticket_lookup = _trade_ticket_surface_lookup(trade_tickets)
    rows: list[dict[str, Any]] = []
    for _, row in final.iterrows():
        trade_plan = _as_text(row.get("trade_plan") or row.get("full_ticket"))
        entry_limit = _as_float(row.get("entry_limit"))
        if not trade_plan or entry_limit is None or entry_limit <= 0:
            continue
        ticker = _as_text(row.get("ticker")).upper()
        ticket_meta = ticket_lookup.get(_ticket_surface_key(ticker, trade_plan), {})
        entry_type = _entry_type_from_ticket(trade_plan) or _as_text(row.get("entry_type")).upper()
        target_entry = _as_float(row.get("target_entry"))
        quote_width = _as_float(row.get("live_quote_width_pct"))
        min_liquidity = _live_leg_min_liquidity_from_row(row)
        live_status = _as_text(row.get("live_validation_status")).upper()
        status, reason = _execution_fill_quality_verdict(
            entry_type=entry_type,
            entry_limit=entry_limit,
            target_entry=target_entry,
            live_validation_status=live_status,
            live_quote_width_pct=quote_width,
            live_leg_min_liquidity=min_liquidity,
        )
        improvement = _fill_quality_price_improvement(entry_type, entry_limit, target_entry)
        slippage = max(0.0, -improvement) if improvement is not None else ""
        rows.append(
            {
                "recommendation_rank": row.get("recommendation_rank", ticket_meta.get("recommendation_rank", "")),
                "ticker": ticker,
                "action_surface": ticket_meta.get("action_surface", "not_on_ticket_surface"),
                "fill_quality_status": status,
                "entry_type": entry_type,
                "entry_limit": _round_or_blank(entry_limit, 2),
                "target_entry": _round_or_blank(target_entry, 2),
                "price_improvement_vs_target": _round_or_blank(improvement, 2),
                "slippage_vs_target": _round_or_blank(slippage, 2),
                "live_quote_width_pct": _round_or_blank(quote_width, 4),
                "live_leg_min_liquidity": "" if min_liquidity is None else _round_or_blank(min_liquidity, 2),
                "live_validation_status": live_status,
                "target_order_status": ticket_meta.get("target_order_status", ""),
                "ready_to_enter": bool(ticket_meta.get("ready_to_enter", False)),
                "reason": reason,
                "trade_plan": trade_plan,
            }
        )
    if not rows:
        return pd.DataFrame(columns=columns)
    out = pd.DataFrame(rows, columns=columns)
    surface_rank = {"green_send_now": 0, "yellow_target": 1, "not_on_ticket_surface": 2, "ticket_review": 3}
    status_rank = {"BLOCK": 0, "WARN": 1, "PASS": 2}
    out["__surface_rank"] = out["action_surface"].map(lambda value: surface_rank.get(_as_text(value), 9))
    out["__status_rank"] = out["fill_quality_status"].map(lambda value: status_rank.get(_as_text(value).upper(), 9))
    out = out.sort_values(
        ["__surface_rank", "__status_rank", "recommendation_rank", "ticker"],
        ascending=[True, True, True, True],
        kind="mergesort",
    )
    return out[columns].reset_index(drop=True)


def summarize_execution_fill_quality(audit: pd.DataFrame) -> dict[str, Any]:
    if audit is None or audit.empty:
        return {
            "status": "not_evaluated",
            "total_rows": 0,
            "pass_rows": 0,
            "warn_rows": 0,
            "block_rows": 0,
            "green_block_rows": 0,
            "yellow_block_rows": 0,
        }
    status = audit.get("fill_quality_status", pd.Series("", index=audit.index)).astype(str).str.upper()
    surface = audit.get("action_surface", pd.Series("", index=audit.index)).astype(str)
    block = status.eq("BLOCK")
    green_block_rows = int((block & surface.eq("green_send_now")).sum())
    yellow_block_rows = int((block & surface.eq("yellow_target")).sum())
    return {
        "status": "blocked_green_fill_quality" if green_block_rows else "review" if block.any() else "pass",
        "total_rows": int(len(audit)),
        "pass_rows": int(status.eq("PASS").sum()),
        "warn_rows": int(status.eq("WARN").sum()),
        "block_rows": int(block.sum()),
        "green_block_rows": green_block_rows,
        "yellow_block_rows": yellow_block_rows,
    }


def _trade_ticket_surface_lookup(trade_tickets: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    if trade_tickets is None or trade_tickets.empty:
        return lookup
    for _, row in trade_tickets.iterrows():
        ticker = _as_text(row.get("ticker")).upper()
        trade_plan = _as_text(row.get("trade_plan") or row.get("full_ticket"))
        if not ticker or not trade_plan:
            continue
        ready = _truthy(row.get("ready_to_enter"))
        target_status = _as_text(row.get("target_order_status")).lower()
        action_surface = (
            "green_send_now"
            if ready
            else "yellow_target"
            if target_status in {"target_order_candidate", "target_order_wait_for_price"}
            else "ticket_review"
        )
        lookup[_ticket_surface_key(ticker, trade_plan)] = {
            "action_surface": action_surface,
            "target_order_status": row.get("target_order_status", ""),
            "ready_to_enter": ready,
            "recommendation_rank": row.get("recommendation_rank", ""),
        }
    return lookup


def _ticket_surface_key(ticker: Any, trade_plan: Any) -> tuple[str, str]:
    return (_as_text(ticker).upper(), re.sub(r"\s+", " ", _as_text(trade_plan)).upper())


def _execution_fill_quality_verdict(
    *,
    entry_type: str,
    entry_limit: float,
    target_entry: Optional[float],
    live_validation_status: str,
    live_quote_width_pct: Optional[float],
    live_leg_min_liquidity: Optional[float],
) -> tuple[str, str]:
    blockers: list[str] = []
    warnings: list[str] = []
    entry_type = _as_text(entry_type).upper()
    live_status = _as_text(live_validation_status).upper()
    if live_status in {"TARGET_QUOTE_REFRESH", "MARKET_CLOSED_RECHECK"}:
        warnings.append("fresh_quote_refresh_required_before_order_entry")
    elif live_status != "PASS":
        blockers.append("live_validation_not_pass")
    if entry_type not in {"CREDIT", "DEBIT"}:
        blockers.append("entry_type_unknown")
    if target_entry is None or target_entry <= 0:
        warnings.append("target_entry_missing")
    elif entry_type == "CREDIT" and entry_limit + 1e-9 < target_entry:
        blockers.append("credit_below_target")
    elif entry_type == "DEBIT" and entry_limit - 1e-9 > target_entry:
        blockers.append("debit_above_target")
    if live_quote_width_pct is not None and live_quote_width_pct > MAX_LIVE_QUOTE_WIDTH_PCT:
        blockers.append("live_quote_width_too_wide")
    if live_leg_min_liquidity is None:
        warnings.append("live_leg_liquidity_missing")
    elif live_leg_min_liquidity < MIN_LIVE_LEG_LIQUIDITY:
        blockers.append("live_leg_liquidity_too_low")
    if blockers:
        return "BLOCK", "; ".join(_dedupe_notes(blockers + warnings))
    if warnings:
        return "WARN", "; ".join(_dedupe_notes(warnings))
    return "PASS", "entry is at or better than target with acceptable live quote quality"


def _fill_quality_price_improvement(
    entry_type: str,
    entry_limit: Optional[float],
    target_entry: Optional[float],
) -> Optional[float]:
    if entry_limit is None or target_entry is None:
        return None
    if _as_text(entry_type).upper() == "CREDIT":
        return float(entry_limit) - float(target_entry)
    if _as_text(entry_type).upper() == "DEBIT":
        return float(target_entry) - float(entry_limit)
    return None


def _live_leg_min_liquidity_from_row(row: Mapping[str, Any] | pd.Series) -> Optional[float]:
    values: list[float] = []
    for prefix in ("short", "long"):
        oi = _as_float(row.get(f"live_{prefix}_oi"))
        volume = _as_float(row.get(f"live_{prefix}_volume"))
        if oi is None and volume is None:
            continue
        values.append((oi or 0.0) + (volume or 0.0))
    if values:
        return min(values)
    summary = _as_text(row.get("live_liquidity_summary"))
    matches = [float(value) for value in re.findall(r"(?:short|long) oi\\+vol ([0-9]+(?:\\.[0-9]+)?)", summary)]
    if matches:
        return min(matches)
    return None


def build_risk_audit(final: pd.DataFrame) -> pd.DataFrame:
    """Create the portfolio risk audit table."""

    if final.empty:
        return pd.DataFrame(columns=["ticker", "risk_type", "risk_note", "visibility_action"])
    rows = []
    for _, row in final.iterrows():
        note = str(row.get("portfolio_risk_note") or "")
        if note:
            rows.append(
                {
                    "ticker": row.get("ticker"),
                    "risk_type": "portfolio",
                    "risk_note": note,
                    "visibility_action": "annotated_not_hidden",
                }
            )
    return pd.DataFrame(rows, columns=["ticker", "risk_type", "risk_note", "visibility_action"])


def apply_position_sizing(
    candidates: Iterable[Mapping[str, Any]],
    portfolio_context: Optional[Mapping[str, Any]],
    market_regime: Optional[Mapping[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Annotate suggested contract sizing without suppressing visible trades."""

    portfolio = dict(portfolio_context or {})
    regime = dict(market_regime or {})
    total_value = _as_float(portfolio.get("total_value")) or _as_float(portfolio.get("net_liquidation")) or 0.0
    cash = _as_float(portfolio.get("cash")) or _as_float(portfolio.get("available_funds")) or 0.0
    sizing_stance = str(regime.get("sizing_stance") or "normal").strip().lower()
    base_budget = total_value * DEFAULT_RISK_BUDGET_PCT if total_value > 0 else 0.0
    if cash > 0:
        base_budget = min(base_budget, cash * 0.05) if base_budget > 0 else cash * 0.05
    if sizing_stance == "reduced" and base_budget > 0:
        base_budget *= 0.5

    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        row = dict(candidate)
        status = str(row.get("recommendation_status") or "").strip().upper()
        live_status = str(row.get("live_validation_status") or "").strip().upper()
        max_loss = _as_float(row.get("max_loss"))
        entry_ready_status = status in {RecommendationStatus.ENTER.value, RecommendationStatus.ENTER_WITH_PORTFOLIO_RISK.value}
        portfolio_flag = bool(row.get("portfolio_risk_flag"))
        risk_budget = base_budget

        suggested_contracts = 0
        sizing_risk_flag = False
        sizing_note = ""
        max_position_loss = 0.0
        account_risk_pct = 0.0

        if status == RecommendationStatus.AVOID.value or str(row.get("hard_rejects") or "").strip():
            sizing_note = "objective blocker present; sizing shown as zero until blocker is resolved"
        elif max_loss is None or max_loss <= 0:
            sizing_note = "max loss unavailable; structure must be rebuilt before sizing"
        elif risk_budget <= 0:
            sizing_note = "portfolio value unavailable; show one-lot planning size only"
            suggested_contracts = 1
            max_position_loss = round(max_loss, 2)
            sizing_risk_flag = True
        else:
            budget_contracts = int(risk_budget // max_loss)
            if budget_contracts >= 1:
                suggested_contracts = max(1, min(MAX_SUGGESTED_CONTRACTS, budget_contracts))
                sizing_note = f"risk budget supports {suggested_contracts} contract(s)"
                if portfolio_flag:
                    sizing_note = _append_reason(
                        sizing_note,
                        "sizing uses the explicit risk budget",
                    )
            else:
                suggested_contracts = 1
                sizing_risk_flag = True
                sizing_note = "one-lot exceeds normal risk budget; manual acknowledgement required"
            max_position_loss = round(max_loss * suggested_contracts, 2)
            account_risk_pct = round(max_position_loss / total_value, 4) if total_value > 0 else 0.0

        if suggested_contracts:
            if not entry_ready_status:
                sizing_note = _append_reason(sizing_note, "planning size only; setup remains in review")
            elif live_status != "PASS":
                sizing_note = _append_reason(sizing_note, "planning size only; requires fresh Schwab validation before entry")

        if suggested_contracts and max_position_loss == 0.0 and max_loss is not None and max_loss > 0:
            max_position_loss = round(max_loss * suggested_contracts, 2)
            account_risk_pct = round(max_position_loss / total_value, 4) if total_value > 0 else 0.0

        row["suggested_contracts"] = int(suggested_contracts)
        row["risk_budget"] = round(risk_budget, 2) if risk_budget else 0.0
        row["portfolio_total_value"] = round(total_value, 2) if total_value else 0.0
        row["portfolio_cash"] = round(cash, 2) if cash else 0.0
        row["max_position_loss"] = max_position_loss
        row["account_risk_pct"] = account_risk_pct
        row["buying_power_effect"] = max_position_loss
        row["sizing_risk_flag"] = bool(sizing_risk_flag)
        row["sizing_note"] = sizing_note
        if sizing_risk_flag:
            row["portfolio_risk_flag"] = True
            row["portfolio_risk_note"] = _append_reason(row.get("portfolio_risk_note"), sizing_note)
        rows.append(row)
    return rows


def build_sizing_audit(final: pd.DataFrame) -> pd.DataFrame:
    """Create a sizing audit table for visible recommendation rows."""

    columns = [
        "ticker",
        "suggested_contracts",
        "risk_budget",
        "max_loss",
        "max_position_loss",
        "account_risk_pct",
        "buying_power_effect",
        "sizing_risk_flag",
        "sizing_note",
        "visibility_action",
    ]
    if final.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for _, row in final.iterrows():
        rows.append(
            {
                "ticker": row.get("ticker", ""),
                "suggested_contracts": row.get("suggested_contracts", 0),
                "risk_budget": row.get("risk_budget", 0.0),
                "max_loss": row.get("max_loss", ""),
                "max_position_loss": row.get("max_position_loss", 0.0),
                "account_risk_pct": row.get("account_risk_pct", 0.0),
                "buying_power_effect": row.get("buying_power_effect", 0.0),
                "sizing_risk_flag": row.get("sizing_risk_flag", False),
                "sizing_note": row.get("sizing_note", ""),
                "visibility_action": "annotated_not_hidden",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_management_plan(final: pd.DataFrame, decision_board: pd.DataFrame) -> pd.DataFrame:
    """Create post-synthesis entry and management instructions for visible rows."""

    columns = [
        "recommendation_rank",
        "ticker",
        "management_action",
        "entry_condition",
        "entry_limit",
        "suggested_contracts",
        "target_exit",
        "max_profit",
        "max_loss",
        "credit_width_ratio",
        "trade_quality_status",
        "quality_gate_reason",
        "max_position_loss",
        "invalidation",
        "review_triggers",
        "management_note",
        "visibility_action",
    ]
    if final.empty:
        return pd.DataFrame(columns=columns)

    decision_by_ticker = _frame_by_ticker(decision_board)
    rows: list[dict[str, Any]] = []
    for _, row in final.iterrows():
        ticker = str(row.get("ticker") or "").strip().upper()
        decision = decision_by_ticker.get(ticker, {})
        ready = bool(decision.get("ready_to_enter", False))
        status = str(row.get("recommendation_status") or decision.get("final_action") or "").strip().upper()
        execution = str(decision.get("execution_status") or "").strip()
        live_status = str(row.get("live_validation_status") or decision.get("live_validation_status") or "").strip().upper()
        entry_limit = row.get("entry_limit", "")
        target_exit = row.get("target_exit", "")
        invalidation = str(row.get("invalidation") or row.get("status_reason") or "").strip()
        status_reason = str(row.get("status_reason") or "").strip()
        sizing_note = str(row.get("sizing_note") or "").strip()

        if ready:
            action = "ENTRY_READY"
            entry_condition = f"Enter only if live quote is at or better than {entry_limit} credit and Schwab validation remains PASS."
            note = "Use trade ticket with suggested size; re-check quote immediately before sending order."
        elif status == RecommendationStatus.WAIT_FOR_PRICE.value:
            action = "WAIT_FOR_PRICE"
            entry_condition = f"Wait for credit at or above {entry_limit}; do not chase a worse fill."
            note = status_reason or "Price target not met."
        elif execution == "needs_live_validation":
            action = "REPRICE"
            entry_condition = "Fetch fresh Schwab chain and rebuild the spread before any order."
            note = status_reason or "Live validation is required."
        elif execution == "needs_fresh_live_quote":
            action = "REPRICE"
            entry_condition = "Run with live Schwab quotes, not replay snapshots, before any order."
            note = status_reason or "Fresh live quote gate is not satisfied."
        elif execution == "needs_portfolio_sizing":
            action = "RESOLVE_PORTFOLIO"
            entry_condition = "Load live or explicit portfolio context and recompute size before any order."
            note = sizing_note or "Portfolio sizing gate is not satisfied."
        elif execution == "needs_agentic_review":
            action = "REVIEW"
            entry_condition = "Ingest agentic reviews before treating this as execution-ready."
            note = status_reason or "Agentic review gate is not satisfied."
        elif execution == "needs_confidence":
            action = "REVIEW"
            entry_condition = "Do not enter until execution confidence clears the threshold."
            note = status_reason or "Execution confidence is below threshold."
        elif status == RecommendationStatus.AVOID.value:
            action = "AVOID"
            entry_condition = "Do not enter unless objective blocker is resolved and the setup is re-run."
            note = status_reason or "Objective blocker present."
        else:
            action = "REVIEW"
            entry_condition = "Do not enter from this row; complete the listed review gates first."
            note = status_reason or "Manual review required."

        review_triggers = _dedupe_notes(
            [
                "fresh quote degrades",
                "underlying violates breakeven/invalidation",
                "new catalyst headline changes thesis",
                "portfolio exposure changes",
                sizing_note,
            ]
        )
        rows.append(
            {
                "recommendation_rank": row.get("recommendation_rank", ""),
                "ticker": ticker,
                "management_action": action,
                "entry_condition": entry_condition,
                "entry_limit": entry_limit,
                "suggested_contracts": row.get("suggested_contracts", 0),
                "target_exit": target_exit,
                "max_profit": row.get("max_profit", ""),
                "max_loss": row.get("max_loss", ""),
                "credit_width_ratio": row.get("credit_width_ratio", ""),
                "trade_quality_status": row.get("trade_quality_status", ""),
                "quality_gate_reason": row.get("quality_gate_reason", ""),
                "max_position_loss": row.get("max_position_loss", ""),
                "invalidation": invalidation,
                "review_triggers": "; ".join(review_triggers),
                "management_note": note,
                "visibility_action": "visible_not_entry_ready" if not ready else "entry_ready",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def annotate_actual_forward_expectancy(final: pd.DataFrame, root: Path) -> pd.DataFrame:
    """Attach ticker and structure-aligned actual/forward outcome support before green-ticket gating."""

    columns = [
        "actual_forward_expectancy_status",
        "actual_forward_expectancy_sample_size",
        "actual_forward_expectancy_win_rate",
        "actual_forward_expectancy_avg_pnl",
        "actual_forward_expectancy_profit_factor",
        "actual_forward_expectancy_source_tickers",
        "actual_forward_expectancy_note",
        "actual_forward_strategy_expectancy_status",
        "actual_forward_strategy_expectancy_sample_size",
        "actual_forward_strategy_expectancy_win_rate",
        "actual_forward_strategy_expectancy_avg_pnl",
        "actual_forward_strategy_expectancy_profit_factor",
        "actual_forward_strategy_expectancy_source_tickers",
        "actual_forward_strategy_expectancy_family",
        "actual_forward_strategy_expectancy_scope",
        "actual_forward_strategy_expectancy_note",
    ]
    if final is None:
        return pd.DataFrame(columns=columns)
    out = final.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = ""
    if out.empty or "ticker" not in out.columns:
        return out

    root = Path(root).expanduser().resolve()
    out_root = root if root.name == "out" else root / "out"
    evidence_frame = _actual_forward_outcome_frame(root, out_root)
    metrics_by_key = _actual_forward_metrics_by_canonical_ticker(evidence_frame)
    metrics_by_strategy = _actual_forward_metrics_by_canonical_ticker_strategy(evidence_frame)
    metrics_by_family = _actual_forward_metrics_by_strategy_family(evidence_frame)
    for idx, row in out.iterrows():
        ticker = _as_text(row.get("ticker")).upper()
        key = canonical_ticker_key(ticker)
        metrics = metrics_by_key.get(key)
        family = _strategy_family_from_ticket_row(row)
        if metrics is None:
            out.at[idx, "actual_forward_expectancy_status"] = "BLOCK"
            out.at[idx, "actual_forward_expectancy_sample_size"] = 0
            out.at[idx, "actual_forward_expectancy_note"] = (
                f"No actual/forward realized outcome support found for {ticker or 'ticker'}."
            )
            family_metrics = metrics_by_family.get(family)
            if _family_level_strategy_fallback_allowed(family, family_metrics, None) or _negative_family_strategy_metrics_available(family_metrics):
                out.at[idx, "actual_forward_strategy_expectancy_status"] = family_metrics["status"]
                out.at[idx, "actual_forward_strategy_expectancy_sample_size"] = family_metrics["sample_size"]
                out.at[idx, "actual_forward_strategy_expectancy_win_rate"] = family_metrics["win_rate"]
                out.at[idx, "actual_forward_strategy_expectancy_avg_pnl"] = family_metrics["avg_pnl"]
                out.at[idx, "actual_forward_strategy_expectancy_profit_factor"] = family_metrics["profit_factor"]
                out.at[idx, "actual_forward_strategy_expectancy_source_tickers"] = family_metrics["source_tickers"]
                out.at[idx, "actual_forward_strategy_expectancy_family"] = family
                out.at[idx, "actual_forward_strategy_expectancy_scope"] = "strategy_family"
                out.at[idx, "actual_forward_strategy_expectancy_note"] = (
                    family_metrics["note"]
                    + f" No ticker-specific {ticker or 'ticker'} {family or 'strategy'} rows exist; family-level evidence is used only to promote allowed positive lanes or block known-negative strategy families."
                )
            else:
                out.at[idx, "actual_forward_strategy_expectancy_status"] = "BLOCK"
                out.at[idx, "actual_forward_strategy_expectancy_sample_size"] = 0
                out.at[idx, "actual_forward_strategy_expectancy_family"] = family
                out.at[idx, "actual_forward_strategy_expectancy_scope"] = "missing"
                out.at[idx, "actual_forward_strategy_expectancy_note"] = (
                    f"No structure-aligned actual/forward realized support found for {ticker or 'ticker'} {family or 'strategy'}."
                )
            continue
        out.at[idx, "actual_forward_expectancy_status"] = metrics["status"]
        out.at[idx, "actual_forward_expectancy_sample_size"] = metrics["sample_size"]
        out.at[idx, "actual_forward_expectancy_win_rate"] = metrics["win_rate"]
        out.at[idx, "actual_forward_expectancy_avg_pnl"] = metrics["avg_pnl"]
        out.at[idx, "actual_forward_expectancy_profit_factor"] = metrics["profit_factor"]
        out.at[idx, "actual_forward_expectancy_source_tickers"] = metrics["source_tickers"]
        out.at[idx, "actual_forward_expectancy_note"] = metrics["note"]
        strategy_metrics = metrics_by_strategy.get((key, family)) if family else None
        if not family:
            out.at[idx, "actual_forward_strategy_expectancy_status"] = "BLOCK"
            out.at[idx, "actual_forward_strategy_expectancy_sample_size"] = 0
            out.at[idx, "actual_forward_strategy_expectancy_scope"] = "missing"
            out.at[idx, "actual_forward_strategy_expectancy_note"] = (
                f"No strategy family could be inferred for {ticker or 'ticker'}."
            )
        elif strategy_metrics is None:
            family_metrics = metrics_by_family.get(family)
            if _family_level_strategy_fallback_allowed(family, family_metrics, metrics) or _negative_family_strategy_metrics_available(family_metrics):
                out.at[idx, "actual_forward_strategy_expectancy_status"] = family_metrics["status"]
                out.at[idx, "actual_forward_strategy_expectancy_sample_size"] = family_metrics["sample_size"]
                out.at[idx, "actual_forward_strategy_expectancy_win_rate"] = family_metrics["win_rate"]
                out.at[idx, "actual_forward_strategy_expectancy_avg_pnl"] = family_metrics["avg_pnl"]
                out.at[idx, "actual_forward_strategy_expectancy_profit_factor"] = family_metrics["profit_factor"]
                out.at[idx, "actual_forward_strategy_expectancy_source_tickers"] = family_metrics["source_tickers"]
                out.at[idx, "actual_forward_strategy_expectancy_family"] = family
                out.at[idx, "actual_forward_strategy_expectancy_scope"] = "strategy_family"
                out.at[idx, "actual_forward_strategy_expectancy_note"] = (
                    family_metrics["note"]
                    + f" No ticker-specific {ticker or 'ticker'} {family} rows exist; family-level evidence is used only to promote allowed positive lanes or block known-negative strategy families."
                )
            else:
                out.at[idx, "actual_forward_strategy_expectancy_status"] = "BLOCK"
                out.at[idx, "actual_forward_strategy_expectancy_sample_size"] = 0
                out.at[idx, "actual_forward_strategy_expectancy_family"] = family
                out.at[idx, "actual_forward_strategy_expectancy_scope"] = "missing"
                out.at[idx, "actual_forward_strategy_expectancy_note"] = (
                    f"No structure-aligned actual/forward realized support found for {ticker or 'ticker'} {family}."
                )
        else:
            family_metrics = metrics_by_family.get(family)
            if _as_text(strategy_metrics.get("status")).upper() != "PASS" and _negative_family_strategy_metrics_available(family_metrics):
                out.at[idx, "actual_forward_strategy_expectancy_status"] = family_metrics["status"]
                out.at[idx, "actual_forward_strategy_expectancy_sample_size"] = family_metrics["sample_size"]
                out.at[idx, "actual_forward_strategy_expectancy_win_rate"] = family_metrics["win_rate"]
                out.at[idx, "actual_forward_strategy_expectancy_avg_pnl"] = family_metrics["avg_pnl"]
                out.at[idx, "actual_forward_strategy_expectancy_profit_factor"] = family_metrics["profit_factor"]
                out.at[idx, "actual_forward_strategy_expectancy_source_tickers"] = family_metrics["source_tickers"]
                out.at[idx, "actual_forward_strategy_expectancy_family"] = family
                out.at[idx, "actual_forward_strategy_expectancy_scope"] = "strategy_family"
                out.at[idx, "actual_forward_strategy_expectancy_note"] = (
                    family_metrics["note"]
                    + f" Sparse ticker-specific {ticker or 'ticker'} {family} support did not override the known-negative strategy-family cohort."
                )
            else:
                out.at[idx, "actual_forward_strategy_expectancy_status"] = strategy_metrics["status"]
                out.at[idx, "actual_forward_strategy_expectancy_sample_size"] = strategy_metrics["sample_size"]
                out.at[idx, "actual_forward_strategy_expectancy_win_rate"] = strategy_metrics["win_rate"]
                out.at[idx, "actual_forward_strategy_expectancy_avg_pnl"] = strategy_metrics["avg_pnl"]
                out.at[idx, "actual_forward_strategy_expectancy_profit_factor"] = strategy_metrics["profit_factor"]
                out.at[idx, "actual_forward_strategy_expectancy_source_tickers"] = strategy_metrics["source_tickers"]
                out.at[idx, "actual_forward_strategy_expectancy_family"] = strategy_metrics["strategy_family"]
                out.at[idx, "actual_forward_strategy_expectancy_scope"] = "ticker_strategy"
                out.at[idx, "actual_forward_strategy_expectancy_note"] = strategy_metrics["note"]
    return out


def build_profitability_calibration(
    root: Path,
    current_rows: pd.DataFrame,
    *,
    as_of_date: str | dt.date | None = None,
    actual_frame: Optional[pd.DataFrame] = None,
    replay_bundle: Optional[tuple[pd.DataFrame, str, str]] = None,
) -> pd.DataFrame:
    """Build row-level profitability buckets from actual outcomes plus replay buckets."""

    root = Path(root).expanduser().resolve()
    out_root = root if root.name == "out" else root / "out"
    if current_rows is None or current_rows.empty:
        return pd.DataFrame(columns=PROFITABILITY_CALIBRATION_COLUMNS)

    current = current_rows.copy()
    actual = actual_frame.copy() if actual_frame is not None else _actual_calibration_frame(root, out_root)
    as_of_day = parse_as_of(as_of_date) if as_of_date is not None else None
    replay, replay_path, replay_error = (
        replay_bundle if replay_bundle is not None else _profitability_replay_frame(out_root, as_of=as_of_day)
    )
    rows: list[dict[str, Any]] = []
    for _, row in current.iterrows():
        key = _calibration_key_from_row(row)
        ticker = _as_text(row.get("ticker")).upper()
        route = key["strategy_route"]
        family = key["strategy_family"]
        actual_ticker_bucket = _actual_calibration_slice(
            actual,
            ticker=ticker,
            route=route,
            family=family,
            ticker_scoped=True,
            key=key,
            bucket_scoped=True,
        )
        actual_route_bucket = _actual_calibration_slice(
            actual,
            ticker="",
            route=route,
            family=family,
            ticker_scoped=False,
            key=key,
            bucket_scoped=True,
        )
        actual_ticker_route = _actual_calibration_slice(
            actual,
            ticker=ticker,
            route=route,
            family=family,
            ticker_scoped=True,
        )
        actual_route = _actual_calibration_slice(
            actual,
            ticker="",
            route=route,
            family=family,
            ticker_scoped=False,
        )
        actual_family = _actual_family_calibration_slice(actual, family=family, key=key)
        actual_scope, actual_metrics = _select_actual_calibration_support(
            actual_ticker_bucket,
            actual_route_bucket,
            actual_ticker_route,
            actual_route,
            actual_family,
        )
        replay_bucket = _replay_calibration_slice(replay, key)
        replay_metrics = _calibration_metrics_row(
            replay_bucket.get("pnl_1x", pd.Series(dtype=float)),
            status_func=_expectancy_status,
        )
        diagnostic_replay, diagnostic_relaxed_dimensions = _diagnostic_replay_calibration_slice(replay, key)
        diagnostic_replay_metrics = _calibration_metrics_row(
            diagnostic_replay.get("pnl_1x", pd.Series(dtype=float)),
            status_func=_expectancy_status,
        )
        actual_sample_gap = _actual_support_sample_gap(actual_scope, actual_metrics)
        replay_sample_gap = _replay_support_sample_gap(replay_metrics)
        status, action, note = _current_calibration_verdict(
            ticker=ticker,
            key=key,
            actual_scope=actual_scope,
            actual_metrics=actual_metrics,
            replay_metrics=replay_metrics,
            replay_path=replay_path,
            replay_error=replay_error,
        )
        rows.append(
            {
                "scope": "current_trade_calibration",
                "ticker": ticker,
                **key,
                "status": status,
                "sample_size": int(actual_metrics.get("sample_size") or 0)
                + int(replay_metrics.get("sample_size") or 0),
                "win_rate": replay_metrics.get("win_rate", ""),
                "avg_pnl": replay_metrics.get("avg_pnl", ""),
                "total_pnl": replay_metrics.get("total_pnl", ""),
                "profit_factor": replay_metrics.get("profit_factor", ""),
                "max_drawdown": replay_metrics.get("max_drawdown", ""),
                "actual_support_status": actual_metrics.get("status", "BLOCK"),
                "actual_support_scope": actual_scope,
                "actual_support_sample_size": actual_metrics.get("sample_size", 0),
                "actual_support_sample_gap": actual_sample_gap,
                "actual_support_avg_pnl": actual_metrics.get("avg_pnl", ""),
                "actual_support_profit_factor": actual_metrics.get("profit_factor", ""),
                "replay_bucket_status": replay_metrics.get("status", "BLOCK"),
                "replay_bucket_sample_size": replay_metrics.get("sample_size", 0),
                "replay_bucket_sample_gap": replay_sample_gap,
                "replay_bucket_avg_pnl": replay_metrics.get("avg_pnl", ""),
                "replay_bucket_profit_factor": replay_metrics.get("profit_factor", ""),
                "diagnostic_replay_status": diagnostic_replay_metrics.get("status", "BLOCK"),
                "diagnostic_replay_sample_size": diagnostic_replay_metrics.get("sample_size", 0),
                "diagnostic_replay_avg_pnl": diagnostic_replay_metrics.get("avg_pnl", ""),
                "diagnostic_replay_profit_factor": diagnostic_replay_metrics.get("profit_factor", ""),
                "diagnostic_replay_relaxed_dimensions": diagnostic_relaxed_dimensions,
                "matched_current_tickers": ticker,
                "current_ticket_count": _current_calibration_ticket_count(current, ticker, key),
                "current_green_count": 0,
                "suggested_action": action,
                "source_path": str(replay_path),
                "note": note,
            }
        )

    out = pd.DataFrame(rows, columns=PROFITABILITY_CALIBRATION_COLUMNS)
    status_rank = {"PASS": 0, "WARN": 1, "BLOCK": 2}
    out["__status_rank"] = out["status"].map(lambda value: status_rank.get(_as_text(value).upper(), 9))
    out["__sample"] = pd.to_numeric(out["sample_size"], errors="coerce").fillna(0)
    out = out.sort_values(
        ["__status_rank", "__sample", "strategy_route", "ticker"],
        ascending=[True, False, True, True],
        kind="mergesort",
    )
    return out[PROFITABILITY_CALIBRATION_COLUMNS].reset_index(drop=True)


def annotate_profitability_calibration(final: pd.DataFrame, calibration: pd.DataFrame) -> pd.DataFrame:
    """Attach row-level profitability calibration status used by green gating."""

    columns = [
        "profitability_calibration_status",
        "profitability_calibration_scope",
        "profitability_calibration_sample_size",
        "profitability_calibration_actual_status",
        "profitability_calibration_actual_sample_size",
        "profitability_calibration_actual_avg_pnl",
        "profitability_calibration_actual_profit_factor",
        "profitability_calibration_replay_status",
        "profitability_calibration_replay_sample_size",
        "profitability_calibration_replay_avg_pnl",
        "profitability_calibration_replay_profit_factor",
        "profitability_calibration_key",
        "profitability_calibration_note",
    ]
    if final is None:
        return pd.DataFrame(columns=columns)
    out = final.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = ""
    if out.empty:
        return out
    lookup = _profitability_calibration_lookup(calibration)
    for idx, row in out.iterrows():
        ticker = _as_text(row.get("ticker")).upper()
        key = _calibration_key_from_row(row)
        lookup_key = _calibration_lookup_key(ticker, key)
        match = lookup.get(lookup_key)
        if match is None:
            out.at[idx, "profitability_calibration_status"] = "BLOCK"
            out.at[idx, "profitability_calibration_scope"] = "missing"
            out.at[idx, "profitability_calibration_sample_size"] = 0
            out.at[idx, "profitability_calibration_actual_status"] = "BLOCK"
            out.at[idx, "profitability_calibration_actual_sample_size"] = 0
            out.at[idx, "profitability_calibration_actual_avg_pnl"] = ""
            out.at[idx, "profitability_calibration_actual_profit_factor"] = ""
            out.at[idx, "profitability_calibration_replay_status"] = "BLOCK"
            out.at[idx, "profitability_calibration_replay_sample_size"] = 0
            out.at[idx, "profitability_calibration_replay_avg_pnl"] = ""
            out.at[idx, "profitability_calibration_replay_profit_factor"] = ""
            out.at[idx, "profitability_calibration_key"] = _calibration_key_text(key)
            out.at[idx, "profitability_calibration_note"] = (
                f"No profitability calibration row exists for {ticker or 'ticker'} {_calibration_key_text(key)}."
            )
            continue
        out.at[idx, "profitability_calibration_status"] = match.get("status", "BLOCK")
        out.at[idx, "profitability_calibration_scope"] = match.get("actual_support_scope", "")
        out.at[idx, "profitability_calibration_sample_size"] = match.get("sample_size", 0)
        out.at[idx, "profitability_calibration_actual_status"] = match.get("actual_support_status", "")
        out.at[idx, "profitability_calibration_actual_sample_size"] = match.get("actual_support_sample_size", "")
        out.at[idx, "profitability_calibration_actual_avg_pnl"] = match.get("actual_support_avg_pnl", "")
        out.at[idx, "profitability_calibration_actual_profit_factor"] = match.get("actual_support_profit_factor", "")
        out.at[idx, "profitability_calibration_replay_status"] = match.get("replay_bucket_status", "")
        out.at[idx, "profitability_calibration_replay_sample_size"] = match.get("replay_bucket_sample_size", "")
        out.at[idx, "profitability_calibration_replay_avg_pnl"] = match.get("replay_bucket_avg_pnl", "")
        out.at[idx, "profitability_calibration_replay_profit_factor"] = match.get("replay_bucket_profit_factor", "")
        out.at[idx, "profitability_calibration_key"] = _calibration_key_text(match)
        out.at[idx, "profitability_calibration_note"] = match.get("note", "")
    return out


def summarize_profitability_calibration(calibration: pd.DataFrame) -> dict[str, Any]:
    if calibration is None or calibration.empty:
        return {
            "status": "missing",
            "current_trade_rows": 0,
            "pass_rows": 0,
            "warn_rows": 0,
            "block_rows": 0,
            "blocking_routes": [],
            "actual_support_status_counts": {},
            "actual_support_scope_counts": {},
            "replay_bucket_status_counts": {},
            "actual_family_only_rows": 0,
            "bucket_precision_rows": 0,
            "bucket_shortfall_rows": 0,
            "bucket_shortfall_routes": [],
            "bucket_blocker_examples": [],
            "missing_replay_bucket_rows": 0,
            "missing_replay_routes": [],
        }
    current = calibration[calibration["scope"].astype(str).eq("current_trade_calibration")].copy()
    status = current.get("status", pd.Series("", index=current.index)).astype(str).str.upper()
    blockers = current[status.ne("PASS")]
    actual_status = current.get("actual_support_status", pd.Series("", index=current.index)).astype(str).str.upper()
    actual_scope = current.get("actual_support_scope", pd.Series("", index=current.index)).astype(str)
    replay_status = current.get("replay_bucket_status", pd.Series("", index=current.index)).astype(str).str.upper()
    replay_sample = pd.to_numeric(current.get("replay_bucket_sample_size", pd.Series(dtype=float)), errors="coerce").fillna(0)
    missing_replay = current[replay_status.eq("BLOCK") & replay_sample.eq(0)].copy()
    bucket_scopes = {"actual_ticker_bucket", "actual_route_bucket"}
    bucket_precision_mask = actual_scope.isin(bucket_scopes)
    bucket_shortfall = current[
        status.ne("PASS")
        & (
            bucket_precision_mask
            | actual_scope.eq("actual_strategy_family")
            | actual_scope.eq("actual_route")
            | replay_status.ne("PASS")
        )
    ].copy()
    by_route: list[dict[str, Any]] = []
    if "strategy_route" in current.columns and not current.empty:
        for route, group in current.groupby(current["strategy_route"].astype(str), dropna=False):
            route_status = group.get("status", pd.Series("", index=group.index)).astype(str).str.upper()
            route_replay_status = group.get("replay_bucket_status", pd.Series("", index=group.index)).astype(str).str.upper()
            route_replay_sample = pd.to_numeric(
                group.get("replay_bucket_sample_size", pd.Series(dtype=float)),
                errors="coerce",
            ).fillna(0)
            route_actual_status = group.get("actual_support_status", pd.Series("", index=group.index)).astype(str).str.upper()
            by_route.append(
                {
                    "strategy_route": route,
                    "rows": int(len(group)),
                    "pass_rows": int(route_status.eq("PASS").sum()),
                    "warn_rows": int(route_status.eq("WARN").sum()),
                    "block_rows": int(route_status.eq("BLOCK").sum()),
                    "actual_block_rows": int(route_actual_status.eq("BLOCK").sum()),
                    "replay_block_rows": int(route_replay_status.eq("BLOCK").sum()),
                    "missing_replay_bucket_rows": int((route_replay_status.eq("BLOCK") & route_replay_sample.eq(0)).sum()),
                }
            )
        by_route = sorted(
            by_route,
            key=lambda item: (
                -int(item["block_rows"]) - int(item["warn_rows"]),
                -int(item["missing_replay_bucket_rows"]),
                str(item["strategy_route"]),
            ),
        )
    return {
        "status": "pass" if not current.empty and blockers.empty else "block",
        "current_trade_rows": int(len(current)),
        "pass_rows": int(status.eq("PASS").sum()),
        "warn_rows": int(status.eq("WARN").sum()),
        "block_rows": int(status.eq("BLOCK").sum()),
        "blocking_routes": sorted(set(blockers.get("strategy_route", pd.Series(dtype=object)).astype(str).tolist())),
        "actual_support_status_counts": {
            str(key): int(value) for key, value in actual_status.value_counts(dropna=False).to_dict().items()
        },
        "actual_support_scope_counts": {
            str(key): int(value) for key, value in actual_scope.value_counts(dropna=False).to_dict().items()
        },
        "replay_bucket_status_counts": {
            str(key): int(value) for key, value in replay_status.value_counts(dropna=False).to_dict().items()
        },
        "actual_family_only_rows": int(actual_scope.eq("actual_strategy_family").sum()),
        "bucket_precision_rows": int(bucket_precision_mask.sum()),
        "bucket_shortfall_rows": int(len(bucket_shortfall)),
        "bucket_shortfall_routes": sorted(
            set(bucket_shortfall.get("strategy_route", pd.Series(dtype=object)).astype(str).tolist())
        ),
        "bucket_blocker_examples": _calibration_bucket_blocker_examples(bucket_shortfall),
        "missing_replay_bucket_rows": int(len(missing_replay)),
        "missing_replay_routes": sorted(
            set(missing_replay.get("strategy_route", pd.Series(dtype=object)).astype(str).tolist())
        ),
        "routes": by_route,
    }


def build_profitability_gap_plan(calibration: pd.DataFrame) -> pd.DataFrame:
    """Turn calibration blockers into exact bucket evidence collection steps."""

    if calibration is None or calibration.empty:
        return pd.DataFrame(columns=PROFITABILITY_GAP_PLAN_COLUMNS)
    current = calibration[
        calibration.get("scope", pd.Series("", index=calibration.index)).astype(str).eq("current_trade_calibration")
    ].copy()
    if current.empty:
        return pd.DataFrame(columns=PROFITABILITY_GAP_PLAN_COLUMNS)

    key_columns = [
        "strategy_route",
        "strategy_family",
        "entry_type",
        "direction_bucket",
        "regime",
        "dte_bucket",
        "iv_rank_bucket",
        "economics_bucket",
        "liquidity_bucket",
    ]
    for column in key_columns + [
        "ticker",
        "status",
        "actual_support_status",
        "actual_support_scope",
        "replay_bucket_status",
        "diagnostic_replay_status",
        "diagnostic_replay_relaxed_dimensions",
        "source_path",
        "note",
    ]:
        if column not in current.columns:
            current[column] = ""
    for column in [
        "current_ticket_count",
        "actual_support_sample_size",
        "actual_support_sample_gap",
        "actual_support_avg_pnl",
        "actual_support_profit_factor",
        "replay_bucket_sample_size",
        "replay_bucket_sample_gap",
        "replay_bucket_avg_pnl",
        "replay_bucket_profit_factor",
        "diagnostic_replay_sample_size",
    ]:
        if column not in current.columns:
            current[column] = 0
        current[column] = pd.to_numeric(current[column], errors="coerce").fillna(0)

    rows: list[dict[str, Any]] = []
    grouped = current.groupby(
        [current[column].map(_as_text) for column in key_columns],
        dropna=False,
        sort=False,
    )
    for key, group in grouped:
        values = list(key) if isinstance(key, tuple) else [key]
        key_map = {column: _as_text(value) for column, value in zip(key_columns, values)}
        actual_row = _profitability_gap_worst_row(group, "actual_support_status", "actual_support_sample_gap")
        replay_row = _profitability_gap_worst_row(group, "replay_bucket_status", "replay_bucket_sample_gap")
        status = _profitability_gap_worst_status(group.get("status", pd.Series("BLOCK", index=group.index)))
        primary_gap = _profitability_gap_primary_gap(actual_row, replay_row, status)
        current_tickers = sorted(
            {
                _as_text(value).upper()
                for value in group.get("ticker", pd.Series(dtype=object)).tolist()
                if _as_text(value)
            }
        )
        current_ticket_count = int(
            pd.to_numeric(group.get("current_ticket_count", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
        )
        if current_ticket_count <= 0:
            current_ticket_count = int(len(group))
        diagnostic_replay_status = _profitability_gap_worst_status(
            group.get("diagnostic_replay_status", pd.Series("BLOCK", index=group.index))
        )
        diagnostic_sample = int(
            pd.to_numeric(group.get("diagnostic_replay_sample_size", pd.Series(dtype=float)), errors="coerce").fillna(0).max()
        )
        relaxed_dimensions = sorted(
            {
                _as_text(value)
                for value in group.get("diagnostic_replay_relaxed_dimensions", pd.Series(dtype=object)).tolist()
                if _as_text(value)
            }
        )
        source_paths = sorted(
            {
                _as_text(value)
                for value in group.get("source_path", pd.Series(dtype=object)).tolist()
                if _as_text(value)
            }
        )
        notes = [
            _as_text(value)
            for value in group.get("note", pd.Series(dtype=object)).tolist()
            if _as_text(value)
        ]
        bucket_key = _calibration_key_text(key_map)
        rows.append(
            {
                "gap_rank": 0,
                "exact_bucket_key": bucket_key,
                **key_map,
                "current_ticket_count": current_ticket_count,
                "current_tickers": ",".join(current_tickers),
                "status": status,
                "actual_support_status": _as_text(actual_row.get("actual_support_status")).upper() or "BLOCK",
                "actual_support_scope": _as_text(actual_row.get("actual_support_scope")),
                "actual_support_sample_size": int(_as_float(actual_row.get("actual_support_sample_size")) or 0),
                "actual_support_sample_gap": int(_as_float(actual_row.get("actual_support_sample_gap")) or 0),
                "actual_support_avg_pnl": _round_or_blank(actual_row.get("actual_support_avg_pnl"), 2),
                "actual_support_profit_factor": _round_or_blank(actual_row.get("actual_support_profit_factor"), 2),
                "replay_bucket_status": _as_text(replay_row.get("replay_bucket_status")).upper() or "BLOCK",
                "replay_bucket_sample_size": int(_as_float(replay_row.get("replay_bucket_sample_size")) or 0),
                "replay_bucket_sample_gap": int(_as_float(replay_row.get("replay_bucket_sample_gap")) or 0),
                "replay_bucket_avg_pnl": _round_or_blank(replay_row.get("replay_bucket_avg_pnl"), 2),
                "replay_bucket_profit_factor": _round_or_blank(replay_row.get("replay_bucket_profit_factor"), 2),
                "diagnostic_replay_status": diagnostic_replay_status,
                "diagnostic_replay_sample_size": diagnostic_sample,
                "diagnostic_replay_relaxed_dimensions": ",".join(relaxed_dimensions),
                "primary_gap": primary_gap,
                "next_evidence_needed": _profitability_gap_next_evidence(
                    key_text=bucket_key,
                    actual_row=actual_row,
                    replay_row=replay_row,
                    relaxed_dimensions=relaxed_dimensions,
                ),
                "suggested_action": _profitability_gap_suggested_action(primary_gap),
                "source_path": "; ".join(source_paths),
                "note": notes[0] if notes else "",
            }
        )

    out = pd.DataFrame(rows, columns=PROFITABILITY_GAP_PLAN_COLUMNS)
    primary_rank = {
        "actual_closed_outcomes_negative_or_weak": 0,
        "actual_closed_outcomes_sample_gap": 1,
        "actual_bucket_precision_gap": 2,
        "replay_exact_bucket_negative_or_weak": 3,
        "replay_exact_bucket_sample_gap": 4,
        "combined_calibration_gap": 5,
        "calibrated": 9,
    }
    out["__primary_rank"] = out["primary_gap"].map(lambda value: primary_rank.get(_as_text(value), 8))
    out["__status_rank"] = out["status"].map(lambda value: {"BLOCK": 0, "WARN": 1, "PASS": 9}.get(_as_text(value).upper(), 8))
    out["__actual_gap"] = pd.to_numeric(out["actual_support_sample_gap"], errors="coerce").fillna(0)
    out["__replay_gap"] = pd.to_numeric(out["replay_bucket_sample_gap"], errors="coerce").fillna(0)
    out["__tickets"] = pd.to_numeric(out["current_ticket_count"], errors="coerce").fillna(0)
    out = out.sort_values(
        ["__primary_rank", "__status_rank", "__tickets", "__actual_gap", "__replay_gap", "strategy_route"],
        ascending=[True, True, False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    out["gap_rank"] = range(1, len(out) + 1)
    return out[PROFITABILITY_GAP_PLAN_COLUMNS]


def summarize_profitability_gap_plan(gap_plan: pd.DataFrame) -> dict[str, Any]:
    if gap_plan is None or gap_plan.empty:
        return {
            "status": "missing",
            "gap_rows": 0,
            "blocking_rows": 0,
            "primary_gap_counts": {},
            "top_gaps": [],
        }
    primary = gap_plan.get("primary_gap", pd.Series("", index=gap_plan.index)).astype(str)
    blocking = gap_plan[primary.ne("calibrated")].copy()
    top_gaps: list[dict[str, Any]] = []
    for _, row in blocking.head(6).iterrows():
        top_gaps.append(
            {
                "gap_rank": int(_as_float(row.get("gap_rank")) or 0),
                "strategy_route": _as_text(row.get("strategy_route")),
                "current_tickers": _as_text(row.get("current_tickers")),
                "primary_gap": _as_text(row.get("primary_gap")),
                "actual_gap": int(_as_float(row.get("actual_support_sample_gap")) or 0),
                "replay_gap": int(_as_float(row.get("replay_bucket_sample_gap")) or 0),
                "relaxed_dimensions": _as_text(row.get("diagnostic_replay_relaxed_dimensions")),
            }
        )
    return {
        "status": "pass" if blocking.empty and not gap_plan.empty else "block",
        "gap_rows": int(len(gap_plan)),
        "blocking_rows": int(len(blocking)),
        "primary_gap_counts": {
            str(key): int(value) for key, value in primary.value_counts(dropna=False).to_dict().items()
        },
        "top_gaps": top_gaps,
    }


def _profitability_gap_plan_detail(summary: Mapping[str, Any]) -> str:
    if not summary or int(_as_float(summary.get("gap_rows")) or 0) <= 0:
        return "no exact bucket gap rows"
    counts = summary.get("primary_gap_counts") or {}
    parts = [", ".join(f"{key}:{value}" for key, value in sorted(counts.items()))] if counts else []
    examples = []
    for item in list(summary.get("top_gaps") or [])[:3]:
        route = _as_text(_mapping_get(item, "strategy_route")) or "unknown_route"
        tickers = _as_text(_mapping_get(item, "current_tickers")) or "UNKNOWN"
        primary = _as_text(_mapping_get(item, "primary_gap")) or "unknown_gap"
        actual_gap = int(_as_float(_mapping_get(item, "actual_gap")) or 0)
        replay_gap = int(_as_float(_mapping_get(item, "replay_gap")) or 0)
        relaxed = _as_text(_mapping_get(item, "relaxed_dimensions"))
        examples.append(
            f"{tickers} {route} {primary} actual_gap={actual_gap} replay_gap={replay_gap}"
            + (f" relaxed={relaxed}" if relaxed else "")
        )
    if examples:
        parts.append("top=" + "; ".join(examples))
    return "; ".join(part for part in parts if part)


def _profitability_gap_worst_row(group: pd.DataFrame, status_column: str, gap_column: str) -> pd.Series:
    working = group.copy()
    if status_column not in working.columns:
        working[status_column] = "BLOCK"
    if gap_column not in working.columns:
        working[gap_column] = 0
    status_rank = {"PASS": 0, "WARN": 1, "BLOCK": 2}
    working["__gap_status_rank"] = working[status_column].map(
        lambda value: status_rank.get(_as_text(value).upper(), 2)
    )
    working["__gap_sample_gap"] = pd.to_numeric(working[gap_column], errors="coerce").fillna(0)
    working = working.sort_values(
        ["__gap_status_rank", "__gap_sample_gap"],
        ascending=[False, False],
        kind="mergesort",
    )
    return working.iloc[0]


def _profitability_gap_worst_status(values: pd.Series) -> str:
    rank = {"PASS": 0, "WARN": 1, "BLOCK": 2}
    statuses = [_as_text(value).upper() for value in values.tolist()]
    statuses = [status if status in rank else "BLOCK" for status in statuses]
    if not statuses:
        return "BLOCK"
    return max(statuses, key=lambda status: rank.get(status, 2))


def _profitability_gap_primary_gap(
    actual_row: Mapping[str, Any],
    replay_row: Mapping[str, Any],
    status: str,
) -> str:
    actual_status = _as_text(_mapping_get(actual_row, "actual_support_status")).upper() or "BLOCK"
    actual_scope = _as_text(_mapping_get(actual_row, "actual_support_scope"))
    actual_gap = int(_as_float(_mapping_get(actual_row, "actual_support_sample_gap")) or 0)
    replay_status = _as_text(_mapping_get(replay_row, "replay_bucket_status")).upper() or "BLOCK"
    replay_gap = int(_as_float(_mapping_get(replay_row, "replay_bucket_sample_gap")) or 0)
    if _as_text(status).upper() == "PASS":
        return "calibrated"
    if actual_status != "PASS" and actual_gap > 0:
        return "actual_closed_outcomes_sample_gap"
    if actual_status != "PASS":
        return "actual_closed_outcomes_negative_or_weak"
    if actual_scope not in {"actual_ticker_bucket", "actual_route_bucket"}:
        return "actual_bucket_precision_gap"
    if replay_status != "PASS" and replay_gap > 0:
        return "replay_exact_bucket_sample_gap"
    if replay_status != "PASS":
        return "replay_exact_bucket_negative_or_weak"
    return "combined_calibration_gap"


def _profitability_gap_next_evidence(
    *,
    key_text: str,
    actual_row: Mapping[str, Any],
    replay_row: Mapping[str, Any],
    relaxed_dimensions: Sequence[str],
) -> str:
    needs: list[str] = []
    actual_status = _as_text(_mapping_get(actual_row, "actual_support_status")).upper() or "BLOCK"
    actual_scope = _as_text(_mapping_get(actual_row, "actual_support_scope"))
    actual_gap = int(_as_float(_mapping_get(actual_row, "actual_support_sample_gap")) or 0)
    actual_sample = int(_as_float(_mapping_get(actual_row, "actual_support_sample_size")) or 0)
    replay_status = _as_text(_mapping_get(replay_row, "replay_bucket_status")).upper() or "BLOCK"
    replay_gap = int(_as_float(_mapping_get(replay_row, "replay_bucket_sample_gap")) or 0)
    replay_sample = int(_as_float(_mapping_get(replay_row, "replay_bucket_sample_size")) or 0)
    if actual_status != "PASS" and actual_gap > 0:
        needs.append(
            f"Need {actual_gap} more positive closed/forward outcomes in exact {key_text}."
        )
    elif actual_status != "PASS":
        needs.append(
            f"Actual outcomes are sampled but not positive enough for {key_text} "
            f"(status={actual_status}, sample={actual_sample})."
        )
    elif actual_scope not in {"actual_ticker_bucket", "actual_route_bucket"}:
        needs.append(
            f"Actual support is only {actual_scope or 'broad'}; collect route/regime/DTE/economics bucket outcomes for {key_text}."
        )
    if replay_status != "PASS" and replay_gap > 0:
        needs.append(
            f"Need {replay_gap} more leakage-safe replay outcomes in exact {key_text}."
        )
    elif replay_status != "PASS":
        needs.append(
            f"Exact replay sample exists but expectancy is not positive for {key_text} "
            f"(status={replay_status}, sample={replay_sample})."
        )
    if relaxed_dimensions:
        needs.append(
            "Nearest replay support only appears after relaxing "
            + ",".join(relaxed_dimensions)
            + "; do not use that as green evidence."
        )
    return " ".join(needs) if needs else f"{key_text} is calibrated; live execution gates still decide order entry."


def _profitability_gap_suggested_action(primary_gap: str) -> str:
    actions = {
        "actual_closed_outcomes_sample_gap": "Backfill Schwab closed/forward outcomes for this exact bucket; keep rows yellow.",
        "actual_closed_outcomes_negative_or_weak": "Keep off green; require new positive actual outcomes or change the route bucket.",
        "actual_bucket_precision_gap": "Collect bucket-precise actual outcomes before green eligibility.",
        "replay_exact_bucket_sample_gap": "Run or import leakage-safe replay for this exact bucket; keep rows yellow.",
        "replay_exact_bucket_negative_or_weak": "Keep off green; revise route selection or require positive replay expectancy.",
        "combined_calibration_gap": "Keep research-only until actual and replay bucket proof both pass.",
        "calibrated": "Monitor realized outcomes; green eligibility still requires live validation, sizing, and review gates.",
    }
    return actions.get(_as_text(primary_gap), "Keep off green until the evidence gap is resolved.")


def _calibration_bucket_blocker_examples(calibration: pd.DataFrame, *, limit: int = 6) -> list[dict[str, Any]]:
    if calibration is None or calibration.empty:
        return []
    working = calibration.copy()
    working["__actual_sample"] = pd.to_numeric(
        working.get("actual_support_sample_size", pd.Series(dtype=float)),
        errors="coerce",
    ).fillna(0)
    working["__replay_sample"] = pd.to_numeric(
        working.get("replay_bucket_sample_size", pd.Series(dtype=float)),
        errors="coerce",
    ).fillna(0)
    working["__actual_gap"] = working.apply(
        lambda row: max(
            0,
            _actual_scope_min_sample(row.get("actual_support_scope"))
            - int(_as_float(row.get("__actual_sample")) or 0),
        ),
        axis=1,
    )
    if "actual_support_sample_gap" in working.columns:
        working["__actual_gap"] = pd.to_numeric(working["actual_support_sample_gap"], errors="coerce").fillna(
            working["__actual_gap"]
        )
    working["__replay_gap"] = working["__replay_sample"].map(
        lambda value: max(0, MIN_EXPECTANCY_SAMPLE_SIZE - int(_as_float(value) or 0))
    )
    if "replay_bucket_sample_gap" in working.columns:
        working["__replay_gap"] = pd.to_numeric(working["replay_bucket_sample_gap"], errors="coerce").fillna(
            working["__replay_gap"]
        )
    working["__priority"] = (
        working.get("replay_bucket_status", pd.Series("", index=working.index)).astype(str).str.upper().ne("PASS").astype(int)
        + working.get("actual_support_status", pd.Series("", index=working.index)).astype(str).str.upper().ne("PASS").astype(int)
    )
    working = working.sort_values(
        ["__priority", "__actual_gap", "__replay_gap", "strategy_route", "ticker"],
        ascending=[False, False, False, True, True],
        kind="mergesort",
    )
    examples: list[dict[str, Any]] = []
    for _, row in working.head(limit).iterrows():
        examples.append(
            {
                "ticker": _as_text(row.get("ticker")).upper(),
                "strategy_route": _as_text(row.get("strategy_route")),
                "direction_bucket": _as_text(row.get("direction_bucket")),
                "dte_bucket": _as_text(row.get("dte_bucket")),
                "economics_bucket": _as_text(row.get("economics_bucket")),
                "liquidity_bucket": _as_text(row.get("liquidity_bucket")),
                "actual_scope": _as_text(row.get("actual_support_scope")),
                "actual_status": _as_text(row.get("actual_support_status")).upper(),
                "actual_sample_size": int(_as_float(row.get("actual_support_sample_size")) or 0),
                "actual_sample_gap": int(row.get("__actual_gap") or 0),
                "replay_status": _as_text(row.get("replay_bucket_status")).upper(),
                "replay_sample_size": int(_as_float(row.get("replay_bucket_sample_size")) or 0),
                "replay_sample_gap": int(row.get("__replay_gap") or 0),
                "replay_relaxed_dimensions": _as_text(row.get("diagnostic_replay_relaxed_dimensions")),
            }
        )
    return examples


def _actual_scope_min_sample(scope: Any) -> int:
    return MIN_TICKER_EXPECTANCY_SAMPLE_SIZE if _as_text(scope) == "actual_ticker_bucket" else MIN_EXPECTANCY_SAMPLE_SIZE


def _actual_support_sample_gap(scope: Any, metrics: Mapping[str, Any]) -> int:
    if _as_text(metrics.get("status")).upper() == "PASS":
        return 0
    sample = int(_as_float(metrics.get("sample_size")) or 0)
    return max(0, _actual_scope_min_sample(scope) - sample)


def _replay_support_sample_gap(metrics: Mapping[str, Any]) -> int:
    if _as_text(metrics.get("status")).upper() == "PASS":
        return 0
    sample = int(_as_float(metrics.get("sample_size")) or 0)
    return max(0, MIN_EXPECTANCY_SAMPLE_SIZE - sample)


def _profitability_calibration_blocker_detail(summary: Mapping[str, Any]) -> str:
    if not summary or _as_text(summary.get("status")) == "pass":
        return ""
    parts: list[str] = []
    actual_counts = summary.get("actual_support_status_counts") or {}
    replay_counts = summary.get("replay_bucket_status_counts") or {}
    if actual_counts:
        parts.append(
            "actual_support="
            + ", ".join(f"{key}:{value}" for key, value in sorted(actual_counts.items()))
        )
    if replay_counts:
        parts.append(
            "replay_bucket="
            + ", ".join(f"{key}:{value}" for key, value in sorted(replay_counts.items()))
        )
    family_only = int(_as_float(summary.get("actual_family_only_rows")) or 0)
    if family_only:
        parts.append(f"family_only_actual_rows={family_only}")
    bucket_shortfall = int(_as_float(summary.get("bucket_shortfall_rows")) or 0)
    if bucket_shortfall:
        route_text = ",".join(str(route) for route in (summary.get("bucket_shortfall_routes") or []) if str(route))
        parts.append(
            f"bucket_shortfall_rows={bucket_shortfall}" + (f" routes={route_text}" if route_text else "")
        )
    missing_replay = int(_as_float(summary.get("missing_replay_bucket_rows")) or 0)
    missing_routes = summary.get("missing_replay_routes") or []
    if missing_replay:
        route_text = ",".join(str(route) for route in missing_routes if str(route))
        parts.append(f"missing_replay_bucket_rows={missing_replay}" + (f" routes={route_text}" if route_text else ""))
    return "; ".join(parts)


def _calibration_bucket_examples_detail(summary: Mapping[str, Any], *, limit: int = 3) -> str:
    examples = summary.get("bucket_blocker_examples") if summary else []
    if not examples:
        return ""
    parts: list[str] = []
    for item in list(examples)[:limit]:
        ticker = _as_text(_mapping_get(item, "ticker")) or "UNKNOWN"
        route = _as_text(_mapping_get(item, "strategy_route")) or "unknown_route"
        direction = _as_text(_mapping_get(item, "direction_bucket")) or "direction_unknown"
        dte = _as_text(_mapping_get(item, "dte_bucket")) or "dte_unknown"
        economics = _as_text(_mapping_get(item, "economics_bucket")) or "economics_unknown"
        actual_status = _as_text(_mapping_get(item, "actual_status")) or "UNKNOWN"
        actual_sample = int(_as_float(_mapping_get(item, "actual_sample_size")) or 0)
        actual_gap = int(_as_float(_mapping_get(item, "actual_sample_gap")) or 0)
        replay_status = _as_text(_mapping_get(item, "replay_status")) or "UNKNOWN"
        replay_sample = int(_as_float(_mapping_get(item, "replay_sample_size")) or 0)
        replay_gap = int(_as_float(_mapping_get(item, "replay_sample_gap")) or 0)
        missing_dims = _as_text(_mapping_get(item, "replay_relaxed_dimensions"))
        replay_detail = (
            f"replay={replay_status} sample={replay_sample} gap={replay_gap}"
            + (f" missing_dims={missing_dims}" if missing_dims else "")
        )
        parts.append(
            f"{ticker} {route}/{direction}/{dte}/{economics} "
            f"actual={actual_status} sample={actual_sample} gap={actual_gap} "
            f"{replay_detail}"
        )
    return "; ".join(parts)


def summarize_calibrated_order_entry_blockers(decision_board: pd.DataFrame) -> dict[str, Any]:
    if decision_board is None or decision_board.empty:
        return {
            "calibrated_rows": 0,
            "ready_rows": 0,
            "blocked_rows": 0,
            "blocker_counts": {},
            "examples": [],
        }
    if "profitability_calibration_status" not in decision_board.columns:
        return {
            "calibrated_rows": 0,
            "ready_rows": 0,
            "blocked_rows": 0,
            "blocker_counts": {},
            "examples": [],
        }
    calibrated = decision_board[
        decision_board["profitability_calibration_status"].astype(str).str.upper().eq("PASS")
    ].copy()
    if calibrated.empty:
        return {
            "calibrated_rows": 0,
            "ready_rows": 0,
            "blocked_rows": 0,
            "blocker_counts": {},
            "examples": [],
        }
    ready = calibrated.get("ready_to_enter", pd.Series(False, index=calibrated.index)).map(_truthy)
    blocked = calibrated[~ready].copy()
    blocker_counts: dict[str, int] = {}
    examples: list[dict[str, Any]] = []
    for _, row in blocked.iterrows():
        blockers = sorted(_blocker_set(row.get("execution_blockers")))
        if not blockers:
            status = _as_text(row.get("execution_status") or row.get("target_order_status"))
            blockers = [status or "not_ready_without_explicit_blocker"]
        for blocker in blockers:
            blocker_counts[blocker] = blocker_counts.get(blocker, 0) + 1
        if len(examples) < 6:
            examples.append(
                {
                    "ticker": _as_text(row.get("ticker")).upper(),
                    "strategy_route": _strategy_route_from_row(row),
                    "trade_plan": _as_text(row.get("trade_plan") or row.get("full_ticket")),
                    "entry_limit": _round_or_blank(_as_float(row.get("entry_limit")), 2),
                    "suggested_contracts": int(_as_float(row.get("suggested_contracts")) or 0),
                    "target_order_status": _as_text(row.get("target_order_status")),
                    "execution_status": _as_text(row.get("execution_status")),
                    "blockers": "; ".join(blockers),
                }
            )
    return {
        "calibrated_rows": int(len(calibrated)),
        "ready_rows": int(ready.sum()),
        "blocked_rows": int(len(blocked)),
        "blocker_counts": dict(sorted(blocker_counts.items(), key=lambda item: (-item[1], item[0]))),
        "examples": examples,
    }


def _calibrated_order_entry_blocker_detail(summary: Mapping[str, Any]) -> str:
    if not summary:
        return ""
    calibrated = int(_as_float(summary.get("calibrated_rows")) or 0)
    ready = int(_as_float(summary.get("ready_rows")) or 0)
    blocked = int(_as_float(summary.get("blocked_rows")) or 0)
    if calibrated <= 0:
        return "no calibrated rows"
    counts = summary.get("blocker_counts") or {}
    count_text = ", ".join(f"{key}:{value}" for key, value in list(counts.items())[:5]) if counts else "none"
    return f"{ready}/{calibrated} ready; {blocked} blocked; {count_text}"


def build_route_opportunity_gap(
    root: Path,
    decision_board: pd.DataFrame,
    trade_tickets: pd.DataFrame,
    profitability_calibration: pd.DataFrame,
    *,
    as_of_date: str | dt.date | None = None,
    actual_frame: Optional[pd.DataFrame] = None,
    replay_bundle: Optional[tuple[pd.DataFrame, str, str]] = None,
) -> pd.DataFrame:
    """Audit strategy routes that have evidence but are not yet green-order routes."""

    root = Path(root).expanduser().resolve()
    out_root = root if root.name == "out" else root / "out"
    as_of_day = parse_as_of(as_of_date) if as_of_date is not None else None
    actual = actual_frame.copy() if actual_frame is not None else _actual_calibration_frame(root, out_root)
    replay, replay_path, replay_error = (
        replay_bundle if replay_bundle is not None else _profitability_replay_frame(out_root, as_of=as_of_day)
    )
    current_counts = _current_ticket_count_by_route(decision_board, trade_tickets)
    green_counts = _current_green_count_by_route(trade_tickets)
    calibration_counts = _calibration_counts_by_route(profitability_calibration)

    routes: set[str] = set(current_counts) | set(green_counts) | set(calibration_counts)
    if actual is not None and not actual.empty and "strategy_route" in actual.columns:
        routes.update(
            route
            for route in actual["strategy_route"].astype(str).str.strip().tolist()
            if route and route != "unknown"
        )
    if replay is not None and not replay.empty and "strategy_route" in replay.columns:
        routes.update(
            route
            for route in replay["strategy_route"].astype(str).str.strip().tolist()
            if route and route != "unknown"
        )
    if not routes:
        return pd.DataFrame(columns=ROUTE_OPPORTUNITY_GAP_COLUMNS)

    rows: list[dict[str, Any]] = []
    for route in sorted(routes):
        actual_slice = (
            actual[actual["strategy_route"].astype(str).eq(route)].copy()
            if actual is not None and not actual.empty and "strategy_route" in actual.columns
            else pd.DataFrame()
        )
        replay_slice = (
            replay[replay["strategy_route"].astype(str).eq(route)].copy()
            if replay is not None and not replay.empty and "strategy_route" in replay.columns
            else pd.DataFrame()
        )
        actual_metrics = _calibration_metrics_row(
            actual_slice.get("realized_pnl", pd.Series(dtype=float)),
            status_func=_expectancy_status,
        )
        replay_metrics = _calibration_metrics_row(
            replay_slice.get("pnl_1x", pd.Series(dtype=float)),
            status_func=_expectancy_status,
        )
        counts = calibration_counts.get(route, {})
        current_count = int(current_counts.get(route, 0))
        green_count = int(green_counts.get(route, 0))
        route_status, gap, action, note = _route_opportunity_verdict(
            route=route,
            current_ticket_count=current_count,
            current_green_count=green_count,
            calibration_pass_rows=int(counts.get("PASS", 0)),
            calibration_warn_rows=int(counts.get("WARN", 0)),
            calibration_block_rows=int(counts.get("BLOCK", 0)),
            actual_metrics=actual_metrics,
            replay_metrics=replay_metrics,
            replay_error=replay_error,
        )
        source_parts: list[str] = []
        actual_sources = sorted(
            {
                str(value).strip()
                for value in actual_slice.get("source", pd.Series(dtype=object)).dropna().tolist()
                if str(value).strip()
            }
        )
        if actual_sources:
            source_parts.append("actual=" + ",".join(actual_sources))
        if replay_path:
            source_parts.append(f"replay={replay_path}")
        rows.append(
            {
                "strategy_route": route,
                "strategy_family": _strategy_family_from_route(route),
                "current_ticket_count": current_count,
                "current_green_count": green_count,
                "calibration_pass_rows": int(counts.get("PASS", 0)),
                "calibration_warn_rows": int(counts.get("WARN", 0)),
                "calibration_block_rows": int(counts.get("BLOCK", 0)),
                "actual_status": actual_metrics.get("status", "BLOCK"),
                "actual_sample_size": actual_metrics.get("sample_size", 0),
                "actual_win_rate": actual_metrics.get("win_rate", ""),
                "actual_avg_pnl": actual_metrics.get("avg_pnl", ""),
                "actual_profit_factor": actual_metrics.get("profit_factor", ""),
                "replay_status": replay_metrics.get("status", "BLOCK"),
                "replay_sample_size": replay_metrics.get("sample_size", 0),
                "replay_win_rate": replay_metrics.get("win_rate", ""),
                "replay_avg_pnl": replay_metrics.get("avg_pnl", ""),
                "replay_profit_factor": replay_metrics.get("profit_factor", ""),
                "route_status": route_status,
                "development_gap": gap,
                "suggested_action": action,
                "source_path": "; ".join(source_parts),
                "note": note,
            }
        )
    out = pd.DataFrame(rows, columns=ROUTE_OPPORTUNITY_GAP_COLUMNS)
    status_rank = {
        "current_route_execution_gates_remaining": 0,
        "current_rows_need_bucket_calibration": 1,
        "evidence_ready_no_current_ticket": 2,
        "near_ready_more_actual_sample_needed": 3,
        "actual_closed_trade_support_needed": 4,
        "replay_bucket_needed": 5,
        "actual_outcomes_negative_or_weak": 6,
        "not_proven": 7,
    }
    out["__rank"] = out["route_status"].map(lambda value: status_rank.get(_as_text(value), 9))
    out["__actual_sample"] = pd.to_numeric(out["actual_sample_size"], errors="coerce").fillna(0)
    out["__replay_sample"] = pd.to_numeric(out["replay_sample_size"], errors="coerce").fillna(0)
    out = out.sort_values(
        ["__rank", "current_ticket_count", "__actual_sample", "__replay_sample", "strategy_route"],
        ascending=[True, False, False, False, True],
        kind="mergesort",
    )
    return out[ROUTE_OPPORTUNITY_GAP_COLUMNS].reset_index(drop=True)


def summarize_route_opportunity_gap(route_gap: pd.DataFrame) -> dict[str, Any]:
    if route_gap is None or route_gap.empty:
        return {
            "status": "missing",
            "route_rows": 0,
            "status_counts": {},
            "candidate_expansion_routes": [],
            "near_ready_routes": [],
            "negative_or_weak_routes": [],
            "current_route_execution_gap_routes": [],
            "bucket_calibration_routes": [],
        }
    status_series = route_gap.get("route_status", pd.Series("", index=route_gap.index)).astype(str)
    return {
        "status": "review",
        "route_rows": int(len(route_gap)),
        "status_counts": {
            str(key): int(value) for key, value in status_series.value_counts(dropna=False).to_dict().items()
        },
        "candidate_expansion_routes": _routes_for_route_gap_status(route_gap, "evidence_ready_no_current_ticket"),
        "near_ready_routes": _routes_for_route_gap_status(route_gap, "near_ready_more_actual_sample_needed"),
        "negative_or_weak_routes": _routes_for_route_gap_status(route_gap, "actual_outcomes_negative_or_weak"),
        "current_route_execution_gap_routes": _routes_for_route_gap_status(
            route_gap,
            "current_route_execution_gates_remaining",
        ),
        "bucket_calibration_routes": _routes_for_route_gap_status(route_gap, "current_rows_need_bucket_calibration"),
    }


def _route_opportunity_gap_detail(summary: Mapping[str, Any]) -> str:
    if not summary or int(_as_float(summary.get("route_rows")) or 0) <= 0:
        return "no route evidence rows"
    parts: list[str] = []
    current = summary.get("current_route_execution_gap_routes") or []
    expansion = summary.get("candidate_expansion_routes") or []
    near = summary.get("near_ready_routes") or []
    weak = summary.get("negative_or_weak_routes") or []
    bucket = summary.get("bucket_calibration_routes") or []
    if current:
        parts.append("current_execution_gaps=" + ",".join(str(route) for route in current))
    if bucket:
        parts.append("bucket_calibration_needed=" + ",".join(str(route) for route in bucket))
    if expansion:
        parts.append("candidate_expansion=" + ",".join(str(route) for route in expansion))
    if near:
        parts.append("near_ready=" + ",".join(str(route) for route in near))
    if weak:
        parts.append("actual_weak=" + ",".join(str(route) for route in weak))
    return "; ".join(parts) if parts else "no evidence-backed route expansion yet"


def _routes_for_route_gap_status(route_gap: pd.DataFrame, status: str) -> list[str]:
    if route_gap is None or route_gap.empty or "route_status" not in route_gap.columns:
        return []
    scoped = route_gap[route_gap["route_status"].astype(str).eq(status)]
    return sorted({str(value) for value in scoped.get("strategy_route", pd.Series(dtype=object)).tolist() if str(value)})


def _route_opportunity_verdict(
    *,
    route: str,
    current_ticket_count: int,
    current_green_count: int,
    calibration_pass_rows: int,
    calibration_warn_rows: int,
    calibration_block_rows: int,
    actual_metrics: Mapping[str, Any],
    replay_metrics: Mapping[str, Any],
    replay_error: str,
) -> tuple[str, str, str, str]:
    actual_status = _as_text(actual_metrics.get("status")).upper() or "BLOCK"
    replay_status = _as_text(replay_metrics.get("status")).upper() or "BLOCK"
    actual_sample = int(_as_float(actual_metrics.get("sample_size")) or 0)
    replay_sample = int(_as_float(replay_metrics.get("sample_size")) or 0)
    actual_gap = max(0, MIN_EXPECTANCY_SAMPLE_SIZE - actual_sample)
    replay_gap = max(0, MIN_EXPECTANCY_SAMPLE_SIZE - replay_sample)

    if actual_status == "PASS" and replay_status == "PASS":
        if current_ticket_count > 0:
            if current_green_count <= 0 and calibration_pass_rows <= 0:
                return (
                    "current_rows_need_bucket_calibration",
                    _current_route_gap_text(
                        current_green_count=current_green_count,
                        calibration_pass_rows=calibration_pass_rows,
                        calibration_warn_rows=calibration_warn_rows,
                        calibration_block_rows=calibration_block_rows,
                    ),
                    "Keep route yellow until current route/DTE/credit-debit buckets have PASS actual and replay calibration.",
                    f"{route} route-level actual and replay evidence pass, but no current row has bucket-level calibration pass.",
                )
            return (
                "current_route_execution_gates_remaining",
                _current_route_gap_text(
                    current_green_count=current_green_count,
                    calibration_pass_rows=calibration_pass_rows,
                    calibration_warn_rows=calibration_warn_rows,
                    calibration_block_rows=calibration_block_rows,
                ),
                "Keep route eligible for green only when live validation, sizing, material profit, and review gates pass.",
                f"{route} has PASS actual and replay evidence; remaining blockers are execution-row specific.",
            )
        return (
            "evidence_ready_no_current_ticket",
            "candidate_generation_or_live_structure_coverage_missing",
            "Add candidate construction/routing coverage for this route, then require normal live and portfolio gates.",
            f"{route} has PASS route evidence but no current ticket rows.",
        )
    if actual_status == "WARN" and replay_status == "PASS" and actual_sample > 0:
        return (
            "near_ready_more_actual_sample_needed",
            f"actual_route_sample_below_{MIN_EXPECTANCY_SAMPLE_SIZE}_needs_{actual_gap}",
            "Collect more closed/forward outcomes or route-precise fills before green eligibility.",
            f"{route} replay is positive, but actual route support is still WARN with sample={actual_sample}.",
        )
    if actual_status == "BLOCK" and actual_sample >= MIN_EXPECTANCY_SAMPLE_SIZE:
        return (
            "actual_outcomes_negative_or_weak",
            "actual_route_outcomes_not_positive",
            "Do not promote this route; require new positive closed-trade evidence before green eligibility.",
            f"{route} has enough actual samples to judge, but the route evidence is not positive.",
        )
    if replay_status == "PASS" and actual_status != "PASS":
        return (
            "actual_closed_trade_support_needed",
            f"actual_route_status_{actual_status.lower()}_sample_{actual_sample}",
            "Keep as research/yellow only until route-precise actual outcomes are positive.",
            f"{route} replay bucket is positive, but actual route evidence is {actual_status}.",
        )
    if actual_status == "PASS" and replay_status != "PASS":
        replay_note = replay_error or f"replay_route_status_{replay_status.lower()}_sample_{replay_sample}"
        return (
            "replay_bucket_needed",
            f"replay_bucket_sample_below_{MIN_EXPECTANCY_SAMPLE_SIZE}_needs_{replay_gap}",
            "Run or import leakage-safe replay for this route before green eligibility.",
            f"{route} actual route evidence is positive, but replay support is missing or weak: {replay_note}.",
        )
    return (
        "not_proven",
        "actual_and_replay_route_evidence_missing_or_too_small",
        "Keep off green order-entry until both route-precise actual and leakage-safe replay evidence pass.",
        f"{route} is not proven: actual={actual_status} sample={actual_sample}; replay={replay_status} sample={replay_sample}.",
    )


def _current_route_gap_text(
    *,
    current_green_count: int,
    calibration_pass_rows: int,
    calibration_warn_rows: int,
    calibration_block_rows: int,
) -> str:
    if current_green_count > 0:
        return "green_rows_exist_monitor_realized_outcomes"
    if calibration_pass_rows > 0:
        return "calibrated_rows_blocked_by_execution_sizing_or_materiality"
    if calibration_warn_rows or calibration_block_rows:
        return "current_rows_need_route_bucket_calibration"
    return "current_rows_need_execution_gate_review"


def _calibration_counts_by_route(calibration: pd.DataFrame) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    if calibration is None or calibration.empty or "strategy_route" not in calibration.columns:
        return counts
    current = calibration[calibration.get("scope", pd.Series("", index=calibration.index)).astype(str).eq("current_trade_calibration")]
    for _, row in current.iterrows():
        route = _as_text(row.get("strategy_route")) or "unknown"
        status = _as_text(row.get("status")).upper() or "BLOCK"
        route_counts = counts.setdefault(route, {"PASS": 0, "WARN": 0, "BLOCK": 0})
        route_counts[status] = route_counts.get(status, 0) + 1
    return counts


def _current_ticket_count_by_route(decision_board: pd.DataFrame, trade_tickets: pd.DataFrame) -> dict[str, int]:
    frame = trade_tickets if trade_tickets is not None and not trade_tickets.empty else decision_board
    return _count_rows_by_route(frame)


def _current_green_count_by_route(trade_tickets: pd.DataFrame) -> dict[str, int]:
    if trade_tickets is None or trade_tickets.empty:
        return {}
    ready = trade_tickets[trade_tickets.get("ready_to_enter", pd.Series(False, index=trade_tickets.index)).map(_truthy)].copy()
    return _count_rows_by_route(ready)


def _count_rows_by_route(frame: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    if frame is None or frame.empty:
        return counts
    for _, row in frame.iterrows():
        route = _strategy_route_from_row(row)
        if route:
            counts[route] = counts.get(route, 0) + 1
    return counts


def _strategy_family_from_route(route: str) -> str:
    route_text = _as_text(route)
    if route_text in {"bull_call_debit", "bear_put_debit", "bull_put_credit", "bear_call_credit", "vertical_spread"}:
        return "vertical_spread"
    return _normal_strategy_family(route_text) or route_text


def _profitability_calibration_lookup(calibration: pd.DataFrame) -> dict[tuple[str, str, str, str, str, str, str, str], dict[str, Any]]:
    if calibration is None or calibration.empty:
        return {}
    current = calibration[calibration["scope"].astype(str).eq("current_trade_calibration")]
    lookup: dict[tuple[str, str, str, str, str, str, str, str], dict[str, Any]] = {}
    for _, row in current.iterrows():
        key = {
            "strategy_route": row.get("strategy_route", ""),
            "entry_type": row.get("entry_type", ""),
            "direction_bucket": row.get("direction_bucket", ""),
            "regime": row.get("regime", ""),
            "dte_bucket": row.get("dte_bucket", ""),
            "economics_bucket": row.get("economics_bucket", ""),
            "liquidity_bucket": row.get("liquidity_bucket", ""),
        }
        lookup[_calibration_lookup_key(_as_text(row.get("ticker")).upper(), key)] = dict(row)
    return lookup


def _calibration_lookup_key(ticker: str, key: Mapping[str, Any]) -> tuple[str, str, str, str, str, str, str, str]:
    return (
        canonical_ticker_key(ticker),
        _as_text(key.get("strategy_route")),
        _as_text(key.get("entry_type")),
        _as_text(key.get("direction_bucket")),
        _regime_bucket(key.get("regime")),
        _as_text(key.get("dte_bucket")),
        _as_text(key.get("economics_bucket")),
        _as_text(key.get("liquidity_bucket")),
    )


def _calibration_key_from_row(row: Mapping[str, Any]) -> dict[str, str]:
    ticket = _as_text(_mapping_get(row, "trade_plan") or _mapping_get(row, "full_ticket"))
    route = _strategy_route_from_row(row)
    family = _strategy_family_from_ticket_row(row) or _normal_strategy_family(_mapping_get(row, "strategy"))
    entry_type = _entry_type_from_ticket(ticket)
    if not entry_type:
        entry_type = _as_text(_mapping_get(row, "entry_type")).upper()
    return {
        "strategy_route": route or family or "unknown",
        "strategy_family": family or "unknown",
        "entry_type": entry_type or "UNKNOWN",
        "direction_bucket": _direction_bucket_from_row(row, route),
        "regime": _regime_bucket(_mapping_get(row, "regime") or _mapping_get(row, "market_regime")),
        "dte_bucket": _dte_bucket(_mapping_get(row, "dte")),
        "iv_rank_bucket": _iv_rank_bucket(_mapping_get(row, "iv_rank")),
        "economics_bucket": _economics_bucket(row, entry_type),
        "liquidity_bucket": _liquidity_bucket(row),
    }


def _calibration_key_text(row: Mapping[str, Any]) -> str:
    parts = [
        _as_text(row.get("strategy_route")),
        _as_text(row.get("entry_type")),
        _as_text(row.get("direction_bucket")),
        _regime_bucket(row.get("regime")),
        _as_text(row.get("dte_bucket")),
        _as_text(row.get("economics_bucket")),
        _as_text(row.get("liquidity_bucket")),
    ]
    return "|".join(part or "unknown" for part in parts)


def _strategy_route_from_row(row: Mapping[str, Any]) -> str:
    explicit = _as_text(_mapping_get(row, "strategy") or _mapping_get(row, "structure"))
    ticket = _as_text(_mapping_get(row, "trade_plan") or _mapping_get(row, "full_ticket"))
    ticket_route = _strategy_route_from_text(ticket)
    if ticket_route and ticket_route != "vertical_spread":
        return ticket_route
    explicit_route = _strategy_route_from_text(explicit)
    return explicit_route or ticket_route


def _strategy_route_from_text(value: Any) -> str:
    text = _as_text(value).lower().replace("-", " ").replace("_", " ")
    has_buy = "buy" in text
    has_sell = "sell" in text
    has_put = " put" in f" {text}"
    has_call = " call" in f" {text}"
    if has_buy and has_sell and "debit" in text and has_put:
        return "bear_put_debit"
    if has_buy and has_sell and "debit" in text and has_call:
        return "bull_call_debit"
    if has_buy and has_sell and "credit" in text and has_put:
        return "bull_put_credit"
    if has_buy and has_sell and "credit" in text and has_call:
        return "bear_call_credit"
    if "bull call" in text and ("debit" in text or "spread" in text):
        return "bull_call_debit"
    if "bear put" in text and ("debit" in text or "spread" in text):
        return "bear_put_debit"
    if "bull put" in text and ("credit" in text or "spread" in text):
        return "bull_put_credit"
    if "bear call" in text and ("credit" in text or "spread" in text):
        return "bear_call_credit"
    if "short put" in text or ("sell" in text and " put" in text and "buy" not in text):
        return "short_put"
    if "short call" in text or ("sell" in text and " call" in text and "buy" not in text):
        return "short_call"
    if "long call" in text or ("buy" in text and " call" in text and "sell" not in text):
        return "long_call"
    if "long put" in text or ("buy" in text and " put" in text and "sell" not in text):
        return "long_put"
    if "spread" in text or (has_buy and has_sell):
        return "vertical_spread"
    return _normal_strategy_family(text)


def _direction_bucket_from_row(row: Mapping[str, Any], route: str) -> str:
    text = _as_text(_mapping_get(row, "direction") or _mapping_get(row, "bias") or route).lower()
    if "bull" in text or "call_debit" in route or route in {"short_put", "bull_put_credit"}:
        return "bullish"
    if "bear" in text or "put_debit" in route or route == "bear_call_credit":
        return "bearish"
    return "neutral_or_unknown"


def _dte_bucket(value: Any) -> str:
    dte = _as_float(value)
    if dte is None or dte <= 0:
        return "dte_unknown"
    if dte <= 14:
        return "dte_0_14"
    if dte <= 30:
        return "dte_15_30"
    if dte <= 60:
        return "dte_31_60"
    return "dte_61_plus"


def _iv_rank_bucket(value: Any) -> str:
    iv = _as_float(value)
    if iv is None:
        return "iv_unknown"
    if iv < 30:
        return "iv_low"
    if iv < 60:
        return "iv_mid"
    return "iv_high"


def _regime_bucket(value: Any) -> str:
    text = _as_text(value).lower().strip().replace("-", "_").replace(" ", "_")
    if not text or text in {"nan", "none", "unknown", "regime_unknown"}:
        return "regime_unknown"
    if text in {"risk_on", "bullish", "up", "uptrend", "trend_up", "positive"}:
        return "risk_on"
    if text in {"risk_off", "bearish", "down", "downtrend", "trend_down", "negative"}:
        return "risk_off"
    if text in {"mixed", "neutral", "sideways", "range", "range_bound", "choppy"}:
        return "mixed"
    return text


def _economics_bucket(row: Mapping[str, Any], entry_type: str) -> str:
    entry_type = _as_text(entry_type).upper()
    if entry_type == "CREDIT":
        ratio = (
            _as_float(_mapping_get(row, "credit_width_ratio"))
            or _as_float(_mapping_get(row, "entry_credit_pct_width"))
            or _as_float(_mapping_get(row, "estimated_credit_pct_width"))
        )
        if ratio is not None:
            if ratio < 0.20:
                return "credit_width_low"
            if ratio < 0.35:
                return "credit_width_mid"
            return "credit_width_high"
        credit = _as_float(_mapping_get(row, "entry_limit") or _mapping_get(row, "entry_credit"))
        if credit is None:
            return "credit_unknown"
        if credit < MIN_SEND_NOW_CREDIT:
            return "credit_small"
        if credit < 1.50:
            return "credit_standard"
        return "credit_rich"
    if entry_type == "DEBIT":
        reward_risk = _as_float(_mapping_get(row, "reward_risk"))
        if reward_risk is None:
            max_profit = _as_float(_mapping_get(row, "max_profit")) or 0.0
            max_loss = _as_float(_mapping_get(row, "max_loss")) or 0.0
            reward_risk = max_profit / max_loss if max_loss > 0 else None
        if reward_risk is not None:
            if reward_risk < MIN_SEND_NOW_DEBIT_REWARD_RISK_RATIO:
                return "debit_reward_risk_weak"
            if reward_risk < 2.50:
                return "debit_reward_risk_mid"
            return "debit_reward_risk_high"
        debit_pct = _as_float(_mapping_get(row, "entry_debit_pct_width") or _mapping_get(row, "estimated_debit_pct_width"))
        if debit_pct is None:
            return "debit_unknown"
        if debit_pct <= 0.35:
            return "debit_width_low"
        if debit_pct <= 0.65:
            return "debit_width_mid"
        return "debit_width_high"
    return "economics_unknown"


def _liquidity_bucket(row: Mapping[str, Any]) -> str:
    liquidity = (
        _as_float(_mapping_get(row, "live_leg_min_liquidity"))
        or _as_float(_mapping_get(row, "source_contract_oi"))
        or _as_float(_mapping_get(row, "source_contract_volume"))
    )
    if liquidity is None:
        return "liquidity_unknown"
    if liquidity >= 1000:
        return "liquidity_deep"
    if liquidity >= 250:
        return "liquidity_adequate"
    return "liquidity_thin"


def _profitability_replay_frame(out_root: Path, *, as_of: Optional[dt.date] = None) -> tuple[pd.DataFrame, str, str]:
    frames: list[pd.DataFrame] = []
    source_paths: list[str] = []
    errors: list[str] = []

    codex_replay, codex_path, codex_error = _codexuw_profitability_replay_frame(out_root, as_of=as_of)
    if not codex_replay.empty:
        frames.append(codex_replay)
        source_paths.append(codex_path)
    elif codex_error:
        errors.append(codex_error)

    wheel_replay, wheel_path, wheel_error = _wheel_csp_profitability_replay_frame(out_root, as_of=as_of)
    if not wheel_replay.empty:
        frames.append(wheel_replay)
        source_paths.append(wheel_path)
    elif wheel_error:
        errors.append(wheel_error)

    pattern_replay, pattern_path, pattern_error = _pattern_validation_replay_frame(out_root, as_of=as_of)
    if not pattern_replay.empty:
        frames.append(pattern_replay)
        source_paths.append(pattern_path)
    elif pattern_error:
        errors.append(pattern_error)

    if frames:
        replay = pd.concat(frames, ignore_index=True, sort=False)
        replay = _backfill_regime_from_history(
            replay,
            out_root,
            date_columns=["signal_date", "asof", "entry_date"],
        )
        replay = _backfill_replay_liquidity_from_uw_sources(
            replay,
            out_root,
            date_columns=["signal_date", "asof", "entry_date"],
        )
        return replay, "; ".join(source_paths), ""
    source_label = codex_path or wheel_path or str(out_root / "codexuw_*backtest*/codexuw_replay_detail.csv")
    return pd.DataFrame(), source_label, "; ".join(errors) or "no replay detail source found"


def _actual_calibration_frame(root: Path, out_root: Path) -> pd.DataFrame:
    actual = _actual_forward_outcome_frame(root, out_root)
    if actual is None or actual.empty:
        return pd.DataFrame()
    out = actual.copy()
    inferred_route = out.get("strategy", pd.Series("", index=out.index)).map(_strategy_route_from_text)
    if "strategy_route" not in out.columns:
        out["strategy_route"] = inferred_route
    else:
        existing_route = out["strategy_route"].astype(str).str.strip()
        out["strategy_route"] = existing_route.where(existing_route.ne(""), inferred_route)
    out["strategy_family"] = out.get("strategy_family", pd.Series("", index=out.index)).astype(str)
    return out


def _codexuw_profitability_replay_frame(out_root: Path, *, as_of: Optional[dt.date] = None) -> tuple[pd.DataFrame, str, str]:
    replay_paths = [
        path
        for path in sorted(
            out_root.glob("codexuw_*backtest*/codexuw_replay_detail.csv"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        if _safe_non_v4_path(path)
    ]
    if not replay_paths:
        return pd.DataFrame(), str(out_root / "codexuw_*backtest*/codexuw_replay_detail.csv"), "no codexuw replay detail source found"
    path = replay_paths[0]
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        return pd.DataFrame(), str(path), f"codexuw replay source unreadable: {exc}"
    if "pnl_1x" not in df.columns:
        return pd.DataFrame(), str(path), "codexuw replay detail missing pnl_1x"
    replay = df[pd.to_numeric(df["pnl_1x"], errors="coerce").notna()].copy()
    replay = _filter_replay_by_exit_date(replay, "exit_day", as_of)
    if "exact_evaluated" in replay.columns:
        replay = replay[replay["exact_evaluated"].map(_truthy)].copy()
    if "decision_pass" in replay.columns:
        replay = replay[replay["decision_pass"].map(_truthy)].copy()
    replay["strategy_route"] = replay.get("strategy", pd.Series("", index=replay.index)).map(_strategy_route_from_text)
    replay["entry_type"] = replay.get("entry_side", pd.Series("", index=replay.index)).astype(str).str.upper()
    if "strategy_kind" in replay.columns:
        replay["entry_type"] = replay["entry_type"].mask(
            replay["entry_type"].astype(str).str.strip().eq(""),
            replay["strategy_kind"].astype(str).str.upper(),
        )
    replay["direction_bucket"] = replay.apply(lambda row: _direction_bucket_from_row(row, row.get("strategy_route")), axis=1)
    replay["regime"] = replay.get("regime", pd.Series("", index=replay.index)).map(_regime_bucket)
    replay["dte_bucket"] = replay.get("dte", pd.Series("", index=replay.index)).map(_dte_bucket)
    replay["iv_rank_bucket"] = replay.get("iv_rank", pd.Series("", index=replay.index)).map(_iv_rank_bucket)
    replay["economics_bucket"] = replay.apply(lambda item: _economics_bucket(item, item.get("entry_type")), axis=1)
    replay["liquidity_bucket"] = replay.apply(_liquidity_bucket, axis=1)
    replay["replay_source"] = "codexuw_spread_replay"
    replay["replay_source_path"] = str(path)
    return replay, str(path), ""


def _wheel_csp_profitability_replay_frame(out_root: Path, *, as_of: Optional[dt.date] = None) -> tuple[pd.DataFrame, str, str]:
    paths = [
        path
        for path in sorted(
            out_root.glob("fresh_wheel*replay*/fresh-wheel-replay-outcomes*.csv"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        if _safe_non_v4_path(path)
    ]
    if not paths:
        return (
            pd.DataFrame(),
            str(out_root / "fresh_wheel*replay*/fresh-wheel-replay-outcomes*.csv"),
            "no wheel CSP replay source found",
        )
    candidates: list[tuple[int, float, Path, pd.DataFrame]] = []
    errors: list[str] = []
    for path in paths:
        try:
            raw = pd.read_csv(path, low_memory=False)
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")
            continue
        converted = _convert_wheel_csp_replay_frame(raw, path=path, as_of=as_of)
        if not converted.empty:
            candidates.append((len(converted), path.stat().st_mtime, path, converted))
    if not candidates:
        return pd.DataFrame(), str(paths[0]), "; ".join(errors) or "no scored wheel CSP replay rows before as-of date"
    _, _, path, replay = sorted(candidates, key=lambda item: (item[0], item[1]), reverse=True)[0]
    return replay, str(path), ""


def _convert_wheel_csp_replay_frame(raw: pd.DataFrame, *, path: Path, as_of: Optional[dt.date]) -> pd.DataFrame:
    required = {"ticker", "action", "pnl_per_contract", "entry_credit", "dte"}
    if raw is None or raw.empty or not required.issubset(raw.columns):
        return pd.DataFrame()
    action = raw["action"].astype(str).str.upper()
    frame = raw[action.isin({"OPEN_CSP", "SELL_CSP"})].copy()
    if "outcome_status" in frame.columns:
        frame = frame[frame["outcome_status"].astype(str).str.lower().eq("scored")].copy()
    frame = _filter_replay_by_exit_date(frame, "exit_date", as_of)
    if frame.empty:
        return pd.DataFrame()
    frame["pnl_1x"] = pd.to_numeric(frame["pnl_per_contract"], errors="coerce")
    frame["entry_credit"] = pd.to_numeric(frame["entry_credit"], errors="coerce")
    frame["entry_limit"] = frame["entry_credit"]
    frame["dte"] = pd.to_numeric(frame["dte"], errors="coerce")
    frame = frame[frame["pnl_1x"].notna() & frame["entry_credit"].notna() & frame["dte"].notna()].copy()
    if frame.empty:
        return pd.DataFrame()
    frame["strategy"] = "Short Put"
    frame["strategy_route"] = "short_put"
    frame["strategy_family"] = "short_put"
    frame["entry_side"] = "credit"
    frame["entry_type"] = "CREDIT"
    frame["strategy_kind"] = "Credit"
    frame["direction_bucket"] = frame.apply(lambda row: _direction_bucket_from_row(row, row.get("strategy_route")), axis=1)
    frame["regime"] = frame.get("regime", pd.Series("", index=frame.index)).map(_regime_bucket)
    frame["exact_evaluated"] = True
    frame["decision_pass"] = True
    frame["iv_rank_bucket"] = "iv_unknown"
    frame["dte_bucket"] = frame["dte"].map(_dte_bucket)
    frame["economics_bucket"] = frame.apply(lambda item: _economics_bucket(item, item.get("entry_type")), axis=1)
    frame["liquidity_bucket"] = frame.apply(_liquidity_bucket, axis=1)
    frame["replay_source"] = "fresh_wheel_csp_replay"
    frame["replay_source_path"] = str(path)
    return frame


def _pattern_validation_replay_frame(out_root: Path, *, as_of: Optional[dt.date] = None) -> tuple[pd.DataFrame, str, str]:
    paths = [
        path
        for path in sorted(
            out_root.glob("options_pattern_pipeline_v1/*/validation_details.csv"),
            key=lambda item: (_pattern_validation_path_date(item) or dt.date.min, item.stat().st_mtime),
            reverse=True,
        )
        if _safe_non_v4_path(path)
    ]
    if as_of is not None:
        paths = [
            path
            for path in paths
            if (_pattern_validation_path_date(path) is None or _pattern_validation_path_date(path) <= as_of)
        ]
    if not paths:
        return (
            pd.DataFrame(),
            str(out_root / "options_pattern_pipeline_v1/*/validation_details.csv"),
            "no pattern validation replay source found",
        )
    errors: list[str] = []
    for path in paths:
        try:
            raw = pd.read_csv(path, low_memory=False)
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")
            continue
        converted = _convert_pattern_validation_replay_frame(raw, path=path, as_of=as_of)
        if not converted.empty:
            return converted, str(path), ""
    return pd.DataFrame(), str(paths[0]), "; ".join(errors) or "no scored pattern validation rows before as-of date"


def _convert_pattern_validation_replay_frame(raw: pd.DataFrame, *, path: Path, as_of: Optional[dt.date]) -> pd.DataFrame:
    required = {"sample", "status", "blocked", "strategy_type", "net_r"}
    if raw is None or raw.empty or not required.issubset(raw.columns):
        return pd.DataFrame()
    frame = raw.copy()
    frame = frame[
        frame["sample"].astype(str).str.upper().eq("VALIDATION")
        & frame["status"].astype(str).str.upper().eq("SCORED")
        & ~frame["blocked"].map(_truthy)
    ].copy()
    if frame.empty:
        return pd.DataFrame()
    exit_source = pd.Series("", index=frame.index, dtype=object)
    if "managed_exit_date" in frame.columns:
        exit_source = frame["managed_exit_date"]
    if "target_date" in frame.columns:
        has_exit = exit_source.notna() & exit_source.astype(str).str.strip().ne("") & exit_source.astype(str).str.lower().ne("nan")
        exit_source = exit_source.where(has_exit, frame["target_date"])
    frame["exit_day"] = exit_source
    frame = _filter_replay_by_exit_date(frame, "exit_day", as_of)
    frame["pnl_1x"] = pd.to_numeric(frame["net_r"], errors="coerce")
    frame = frame[frame["pnl_1x"].notna()].copy()
    if frame.empty:
        return pd.DataFrame()
    dedupe_columns = [
        column
        for column in [
            "signal_date",
            "target_date",
            "managed_exit_date",
            "ticker",
            "strategy_type",
            "lead_option_symbol",
            "horizon",
            "pattern_family",
        ]
        if column in frame.columns
    ]
    if dedupe_columns:
        frame = frame.sort_values(dedupe_columns, kind="mergesort").drop_duplicates(dedupe_columns, keep="last")
    frame["strategy"] = frame["strategy_type"].astype(str)
    frame["strategy_route"] = frame["strategy_type"].map(_strategy_route_from_text)
    frame = frame[frame["strategy_route"].astype(str).str.strip().ne("")].copy()
    if frame.empty:
        return pd.DataFrame()
    frame["strategy_family"] = frame["strategy_route"].map(_strategy_family_from_route)
    frame["entry_type"] = frame["strategy_type"].map(_pattern_validation_entry_type)
    frame["entry_side"] = frame["entry_type"].str.lower()
    frame["strategy_kind"] = frame["entry_type"].str.title()
    frame["direction_bucket"] = frame.apply(lambda row: _direction_bucket_from_row(row, row.get("strategy_route")), axis=1)
    frame["regime"] = frame.get("market_regime", pd.Series("", index=frame.index)).map(_regime_bucket)
    frame["entry_credit"] = pd.to_numeric(frame.get("entry_credit", pd.Series(dtype=float)), errors="coerce")
    frame["entry_ask"] = pd.to_numeric(frame.get("entry_ask", pd.Series(dtype=float)), errors="coerce")
    frame["entry_limit"] = frame["entry_credit"].where(frame["entry_type"].eq("CREDIT"), frame["entry_ask"])
    frame["spread_width"] = frame.apply(_pattern_validation_spread_width, axis=1)
    frame["entry_credit_pct_width"] = frame["entry_credit"] / frame["spread_width"].replace(0, pd.NA)
    frame["dte"] = frame.apply(_pattern_validation_option_dte, axis=1)
    frame["dte_bucket"] = frame["dte"].map(_dte_bucket)
    frame["iv_rank_bucket"] = "iv_unknown"
    frame["economics_bucket"] = frame.apply(lambda item: _economics_bucket(item, item.get("entry_type")), axis=1)
    frame["liquidity_bucket"] = "liquidity_unknown"
    frame["exact_evaluated"] = True
    frame["decision_pass"] = True
    frame["replay_source"] = "options_pattern_validation_replay"
    frame["replay_source_path"] = str(path)
    return frame


def _pattern_validation_entry_type(value: Any) -> str:
    text = _as_text(value).lower()
    if "credit" in text:
        return "CREDIT"
    return "DEBIT"


def _pattern_validation_option_dte(row: Mapping[str, Any]) -> Optional[int]:
    signal_day = _parse_optional_date_value(_mapping_get(row, "signal_date"))
    expiry = _pattern_validation_option_expiry(row)
    if signal_day is None or expiry is None:
        return None
    return max((expiry - signal_day).days, 0)


def _parse_optional_date_value(value: Any) -> Optional[dt.date]:
    text = _as_text(value)
    if not text or text.lower() == "nan":
        return None
    try:
        return parse_as_of(text)
    except Exception:
        parsed = pd.to_datetime(text, errors="coerce")
        if pd.isna(parsed):
            return None
        return parsed.date()


def _pattern_validation_option_expiry(row: Mapping[str, Any]) -> Optional[dt.date]:
    text = " ".join(
        _as_text(_mapping_get(row, key))
        for key in ("lead_option_symbol", "legs_json")
        if _as_text(_mapping_get(row, key))
    )
    match = re.search(r"(?<!\d)(\d{6})[CP]\d{8}(?!\d)", text.upper())
    if not match:
        return None
    value = match.group(1)
    try:
        return dt.date(2000 + int(value[:2]), int(value[2:4]), int(value[4:6]))
    except ValueError:
        return None


def _pattern_validation_spread_width(row: Mapping[str, Any]) -> Optional[float]:
    legs_text = _as_text(_mapping_get(row, "legs_json"))
    if not legs_text:
        return None
    try:
        legs = json.loads(legs_text)
    except Exception:
        return None
    if not isinstance(legs, list):
        return None
    strikes = [
        _as_float(_mapping_get(leg, "strike"))
        for leg in legs
        if isinstance(leg, Mapping) and _as_float(_mapping_get(leg, "strike")) is not None
    ]
    if len(strikes) < 2:
        return None
    return abs(max(strikes) - min(strikes))


def _pattern_validation_path_date(path: Path) -> Optional[dt.date]:
    match = re.search(r"(\d{4}-\d{2}-\d{2})", path.parent.name)
    if not match:
        return None
    try:
        return parse_as_of(match.group(1))
    except Exception:
        return None


def _filter_replay_by_exit_date(frame: pd.DataFrame, column: str, as_of: Optional[dt.date]) -> pd.DataFrame:
    if as_of is None or frame is None or frame.empty or column not in frame.columns:
        return frame
    parsed = pd.to_datetime(frame[column], errors="coerce")
    mask = parsed.notna() & (parsed < pd.Timestamp(as_of))
    return frame[mask].copy()


def _backfill_regime_from_history(
    frame: pd.DataFrame,
    out_root: Path,
    *,
    date_columns: Sequence[str],
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return frame
    history = _market_regime_history_by_date(out_root)
    if not history:
        return frame
    out = frame.copy()
    existing = out.get("regime", pd.Series("", index=out.index)).map(_regime_bucket)
    backfill = pd.Series("", index=out.index, dtype=object)
    for column in date_columns:
        if column not in out.columns:
            continue
        candidate = out[column].map(lambda value: history.get(_date_key(_parse_optional_date_value(value)), ""))
        backfill = backfill.where(backfill.astype(str).str.strip().ne(""), candidate)
    out["regime"] = existing.where(existing.ne("regime_unknown"), backfill.map(_regime_bucket)).map(_regime_bucket)
    return out


def _backfill_replay_liquidity_from_uw_sources(
    frame: pd.DataFrame,
    out_root: Path,
    *,
    date_columns: Sequence[str],
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return frame
    out = frame.copy()
    if "source_contract_oi" not in out.columns:
        out["source_contract_oi"] = math.nan
    if "source_contract_volume" not in out.columns:
        out["source_contract_volume"] = math.nan
    base_root = Path(out_root).parent
    requested: dict[int, list[tuple[str, str]]] = {}
    symbols_by_day: dict[str, set[str]] = {}
    for idx, row in out.iterrows():
        if _liquidity_bucket(row) != "liquidity_unknown":
            continue
        symbol = _option_symbol_key(
            _mapping_get(row, "option_symbol")
            or _mapping_get(row, "source_contract")
            or _mapping_get(row, "lead_option_symbol")
        )
        if not symbol:
            continue
        for column in date_columns:
            if column not in out.columns:
                continue
            day = _date_key(_parse_optional_date_value(row.get(column)))
            if not day:
                continue
            requested.setdefault(idx, []).append((day, symbol))
            symbols_by_day.setdefault(day, set()).add(symbol)

    if not requested:
        out["liquidity_bucket"] = out.apply(_liquidity_bucket, axis=1)
        return out

    cache = {
        day: _same_day_option_liquidity_lookup(base_root, day, symbols=symbols)
        for day, symbols in symbols_by_day.items()
    }
    for idx, requests in requested.items():
        for day, symbol in requests:
            match = cache.get(day, {}).get(symbol)
            if not match:
                continue
            if _as_float(out.at[idx, "source_contract_oi"]) is None:
                out.at[idx, "source_contract_oi"] = match.get("source_contract_oi", math.nan)
            if _as_float(out.at[idx, "source_contract_volume"]) is None:
                out.at[idx, "source_contract_volume"] = match.get("source_contract_volume", math.nan)
            break
    out["liquidity_bucket"] = out.apply(_liquidity_bucket, axis=1)
    return out


def _same_day_option_liquidity_lookup(
    base_root: Path,
    day: str,
    *,
    symbols: Optional[set[str]] = None,
) -> dict[str, dict[str, Any]]:
    try:
        as_of = parse_as_of(day)
    except Exception:
        return {}
    date_dir = Path(base_root) / as_of.isoformat()
    if not date_dir.exists():
        return {}
    lookup: dict[str, dict[str, Any]] = {}
    wanted = {symbol for symbol in (symbols or set()) if symbol}
    for prefix, oi_columns, volume_columns in [
        ("hot-chains-", ["open_interest"], ["volume"]),
        ("chain-oi-changes-", ["curr_oi", "last_oi"], ["volume", "curr_vol", "prev_vol"]),
    ]:
        source = _read_same_day_option_liquidity_export(
            date_dir,
            prefix=prefix,
            oi_columns=oi_columns,
            volume_columns=volume_columns,
            symbols=wanted,
        )
        if source is None or source.empty or "option_symbol" not in source.columns:
            continue
        scoped = source.copy()
        scoped["__symbol_key"] = scoped["option_symbol"].astype(str).str.replace(r"\s+", "", regex=True).str.upper()
        if wanted:
            scoped = scoped[scoped["__symbol_key"].isin(wanted)].copy()
        if scoped.empty:
            continue
        oi_parts = [
            pd.to_numeric(scoped[column], errors="coerce")
            for column in ["open_interest", "curr_oi", "last_oi"]
            if column in scoped.columns
        ]
        volume_parts = [
            pd.to_numeric(scoped[column], errors="coerce")
            for column in ["volume", "curr_vol", "prev_vol"]
            if column in scoped.columns
        ]
        if not oi_parts and not volume_parts:
            continue
        values = pd.DataFrame({"symbol": scoped["__symbol_key"]})
        values["source_contract_oi"] = pd.concat(oi_parts, axis=1).max(axis=1) if oi_parts else math.nan
        values["source_contract_volume"] = pd.concat(volume_parts, axis=1).max(axis=1) if volume_parts else math.nan
        grouped = values.groupby("symbol", as_index=False).max(numeric_only=True)
        for item in grouped.to_dict("records"):
            symbol = _as_text(item.get("symbol"))
            existing = lookup.get(symbol, {})
            lookup[symbol] = {
                "source_contract_oi": max(
                    _as_float(existing.get("source_contract_oi")) or 0.0,
                    _as_float(item.get("source_contract_oi")) or 0.0,
                ),
                "source_contract_volume": max(
                    _as_float(existing.get("source_contract_volume")) or 0.0,
                    _as_float(item.get("source_contract_volume")) or 0.0,
                ),
            }
    return lookup


def _read_same_day_option_liquidity_export(
    date_dir: Path,
    *,
    prefix: str,
    oi_columns: Sequence[str],
    volume_columns: Sequence[str],
    symbols: set[str],
) -> pd.DataFrame:
    try:
        from codexuw import data as uw_data

        path = uw_data.find_export(date_dir, prefix)
    except Exception:
        return pd.DataFrame()
    wanted_columns = {"option_symbol", *oi_columns, *volume_columns}
    try:
        frame = pd.read_csv(path, usecols=lambda column: column in wanted_columns, low_memory=False)
    except Exception:
        return pd.DataFrame()
    if frame.empty or "option_symbol" not in frame.columns:
        return pd.DataFrame()
    frame["__symbol_key"] = frame["option_symbol"].astype(str).str.replace(r"\s+", "", regex=True).str.upper()
    if symbols:
        frame = frame[frame["__symbol_key"].isin(symbols)].copy()
    return frame


def _option_symbol_key(value: Any) -> str:
    return re.sub(r"\s+", "", _as_text(value)).upper()


def _actual_calibration_slice(
    actual: pd.DataFrame,
    *,
    ticker: str,
    route: str,
    family: str,
    ticker_scoped: bool,
    key: Optional[Mapping[str, Any]] = None,
    bucket_scoped: bool = False,
) -> pd.DataFrame:
    if actual is None or actual.empty:
        return pd.DataFrame()
    scoped = actual.copy()
    if ticker_scoped:
        ticker_key = canonical_ticker_key(ticker)
        scoped = scoped[scoped["canonical_ticker"].astype(str).eq(ticker_key)].copy()
    route_series = scoped.get("strategy_route", pd.Series("", index=scoped.index)).astype(str)
    exact = scoped[route_series.eq(route)].copy()
    if exact.empty:
        return pd.DataFrame()
    if not bucket_scoped:
        return exact
    return _actual_bucket_slice(exact, key or {})


def _actual_bucket_slice(actual: pd.DataFrame, key: Mapping[str, Any]) -> pd.DataFrame:
    if actual is None or actual.empty:
        return pd.DataFrame()
    scoped = actual.copy()
    for column in ["entry_type", "direction_bucket", "regime", "dte_bucket", "economics_bucket"]:
        value = _as_text(key.get(column))
        if (
            value
            and value
            not in {"UNKNOWN", "neutral_or_unknown", "regime_unknown", "dte_unknown", "economics_unknown"}
            and column in scoped.columns
        ):
            scoped = scoped[scoped[column].astype(str).eq(value)].copy()
            if scoped.empty:
                return scoped
    return scoped


def _actual_family_calibration_slice(
    actual: pd.DataFrame,
    *,
    family: str,
    key: Optional[Mapping[str, Any]] = None,
) -> pd.DataFrame:
    if actual is None or actual.empty or not family:
        return pd.DataFrame()
    scoped = actual[actual.get("strategy_family", pd.Series("", index=actual.index)).astype(str).eq(family)].copy()
    direction = _as_text((key or {}).get("direction_bucket"))
    if direction and direction != "neutral_or_unknown" and "direction_bucket" in scoped.columns:
        narrowed = scoped[scoped["direction_bucket"].astype(str).eq(direction)].copy()
        if not narrowed.empty:
            return narrowed
        direction_values = {
            _as_text(value)
            for value in scoped["direction_bucket"].dropna().tolist()
            if _as_text(value)
        }
        if direction_values and not direction_values.issubset({"neutral_or_unknown"}):
            return narrowed
    return scoped


def _select_actual_calibration_support(
    ticker_bucket: pd.DataFrame,
    route_bucket: pd.DataFrame,
    ticker_route: pd.DataFrame,
    route: pd.DataFrame,
    family: pd.DataFrame,
) -> tuple[str, dict[str, Any]]:
    bucket_candidates = [
        ("actual_ticker_bucket", ticker_bucket, _ticker_expectancy_status),
        ("actual_route_bucket", route_bucket, _expectancy_status),
    ]
    best_bucket_scope = "missing"
    best_bucket_metrics = _calibration_metrics_row(pd.Series(dtype=float), status_func=_expectancy_status)
    for scope, frame, status_func in bucket_candidates:
        if frame is None or frame.empty:
            continue
        metrics = _calibration_metrics_row(frame.get("realized_pnl", pd.Series(dtype=float)), status_func=status_func)
        if _as_text(metrics.get("status")).upper() == "PASS":
            return scope, metrics
        if int(metrics.get("sample_size") or 0) > int(best_bucket_metrics.get("sample_size") or 0):
            best_bucket_scope = scope
            best_bucket_metrics = metrics
    if int(best_bucket_metrics.get("sample_size") or 0) > 0:
        return best_bucket_scope, best_bucket_metrics

    broad_candidates = [
        ("actual_ticker_route", ticker_route, _ticker_expectancy_status),
        ("actual_route", route, _expectancy_status),
        ("actual_strategy_family", family, _expectancy_status),
    ]
    best_scope = "missing"
    best_metrics = _calibration_metrics_row(pd.Series(dtype=float), status_func=_expectancy_status)
    for scope, frame, status_func in broad_candidates:
        if frame is None or frame.empty:
            continue
        metrics = _calibration_metrics_row(frame.get("realized_pnl", pd.Series(dtype=float)), status_func=status_func)
        if _as_text(metrics.get("status")).upper() == "PASS":
            return scope, metrics
        if int(metrics.get("sample_size") or 0) > int(best_metrics.get("sample_size") or 0):
            best_scope = scope
            best_metrics = metrics
    return best_scope, best_metrics


def _replay_calibration_slice(replay: pd.DataFrame, key: Mapping[str, Any]) -> pd.DataFrame:
    if replay is None or replay.empty:
        return pd.DataFrame()
    scoped = replay[replay["strategy_route"].astype(str).eq(_as_text(key.get("strategy_route")))].copy()
    if scoped.empty:
        return scoped
    for column in ["entry_type", "direction_bucket", "regime", "dte_bucket", "economics_bucket"]:
        value = _as_text(key.get(column))
        if (
            value
            and value
            not in {"UNKNOWN", "neutral_or_unknown", "regime_unknown", "dte_unknown", "economics_unknown"}
            and column in scoped.columns
        ):
            narrowed = scoped[scoped[column].astype(str).eq(value)].copy()
            if not narrowed.empty:
                scoped = narrowed
            else:
                return narrowed
    liquidity = _as_text(key.get("liquidity_bucket"))
    if liquidity and liquidity != "liquidity_unknown" and "liquidity_bucket" in scoped.columns:
        scoped = scoped[scoped["liquidity_bucket"].astype(str).eq(liquidity)].copy()
    return scoped


def _diagnostic_replay_calibration_slice(replay: pd.DataFrame, key: Mapping[str, Any]) -> tuple[pd.DataFrame, str]:
    """Return nearest replay evidence for gap diagnosis only; never green-gate from this."""

    if replay is None or replay.empty:
        return pd.DataFrame(), "replay_source_empty"
    route = _as_text(key.get("strategy_route"))
    if "strategy_route" not in replay.columns:
        return pd.DataFrame(), "strategy_route:source_missing"
    scoped = replay[replay["strategy_route"].astype(str).eq(route)].copy()
    if scoped.empty:
        return scoped, "strategy_route"

    relaxed: list[str] = []
    unknowns = {"UNKNOWN", "neutral_or_unknown", "regime_unknown", "dte_unknown", "economics_unknown", "liquidity_unknown"}
    for column in ["entry_type", "direction_bucket", "regime", "dte_bucket", "economics_bucket", "liquidity_bucket"]:
        value = _as_text(key.get(column))
        if not value or value in unknowns:
            continue
        if column not in scoped.columns:
            relaxed.append(f"{column}:source_missing")
            continue
        narrowed = scoped[scoped[column].astype(str).eq(value)].copy()
        if narrowed.empty:
            relaxed.append(column)
            continue
        scoped = narrowed
    return scoped, ",".join(relaxed)


def _calibration_metrics_row(pnl_values: pd.Series, *, status_func: Any) -> dict[str, Any]:
    metrics = _expectancy_metrics_row(
        "profitability_calibration",
        Path(""),
        "profitability_calibration",
        pd.to_numeric(pnl_values, errors="coerce"),
        tickers=set(),
        current_tickers=set(),
        open_or_unrealized_count=0,
        note="profitability calibration metrics",
        status_override_func=status_func,
    )
    return metrics


def _current_calibration_verdict(
    *,
    ticker: str,
    key: Mapping[str, Any],
    actual_scope: str,
    actual_metrics: Mapping[str, Any],
    replay_metrics: Mapping[str, Any],
    replay_path: Path,
    replay_error: str,
) -> tuple[str, str, str]:
    actual_status = _as_text(actual_metrics.get("status")).upper() or "BLOCK"
    replay_status = _as_text(replay_metrics.get("status")).upper() or "BLOCK"
    actual_sample = int(actual_metrics.get("sample_size") or 0)
    replay_sample = int(replay_metrics.get("sample_size") or 0)
    key_text = _calibration_key_text(key)
    bucket_precise_actual = actual_scope in {"actual_ticker_bucket", "actual_route_bucket"}
    if actual_status == "PASS" and replay_status == "PASS" and bucket_precise_actual:
        return (
            "PASS",
            "eligible_for_green_if_all_execution_gates_pass",
            f"{ticker} {key_text} has PASS actual support via {actual_scope} and PASS replay bucket support.",
        )
    blockers: list[str] = []
    if actual_status != "PASS":
        blockers.append(f"actual_support={actual_status or 'BLOCK'} sample={actual_sample} scope={actual_scope}")
    elif not bucket_precise_actual:
        blockers.append(f"actual_bucket_precision=route_or_family_only scope={actual_scope} sample={actual_sample}")
    if replay_status != "PASS":
        replay_note = replay_error or f"replay_bucket={replay_status or 'BLOCK'} sample={replay_sample}"
        blockers.append(replay_note)
    status = "WARN" if actual_sample > 0 or replay_sample > 0 else "BLOCK"
    return (
        status,
        "keep_yellow_until_actual_and_replay_bucket_calibration_pass",
        f"{ticker} {key_text} is not green-calibrated: " + "; ".join(blockers) + f". Replay source: {replay_path}.",
    )


def _current_calibration_ticket_count(current: pd.DataFrame, ticker: str, key: Mapping[str, Any]) -> int:
    if current is None or current.empty:
        return 0
    wanted = _calibration_lookup_key(ticker, key)
    count = 0
    for _, row in current.iterrows():
        if _calibration_lookup_key(_as_text(row.get("ticker")).upper(), _calibration_key_from_row(row)) == wanted:
            count += 1
    return count


def synthesize_decision_board(
    final: pd.DataFrame,
    *,
    market_regime: Mapping[str, Any],
    execution_context: Optional[Mapping[str, Any]] = None,
) -> pd.DataFrame:
    """Build the Synthesis Agent board separating quality, execution, and portfolio fit."""

    columns = [
        "recommendation_rank",
        "ticker",
        "status_icon",
        "status_label",
        "final_action",
        "setup_quality_status",
        "execution_status",
        "execution_gate_status",
        "execution_blockers",
        "execution_confidence_score",
        "execution_confidence_rating",
        "trade_quality_confidence_rating",
        "actual_forward_expectancy_status",
        "actual_forward_expectancy_sample_size",
        "actual_forward_expectancy_note",
        "actual_forward_strategy_expectancy_status",
        "actual_forward_strategy_expectancy_sample_size",
        "actual_forward_strategy_expectancy_family",
        "actual_forward_strategy_expectancy_note",
        "profitability_calibration_status",
        "profitability_calibration_scope",
        "profitability_calibration_sample_size",
        "profitability_calibration_actual_status",
        "profitability_calibration_actual_sample_size",
        "profitability_calibration_actual_avg_pnl",
        "profitability_calibration_actual_profit_factor",
        "profitability_calibration_replay_status",
        "profitability_calibration_replay_sample_size",
        "profitability_calibration_replay_avg_pnl",
        "profitability_calibration_replay_profit_factor",
        "profitability_calibration_key",
        "profitability_calibration_note",
        "external_agent_review_count",
        "external_agent_distinct_review_count",
        "external_agent_review_agents",
        "portfolio_fit_status",
        "underlying_quality_tier",
        "underlying_quality_reason",
        "target_order_status",
        "ready_to_enter",
        "requires_portfolio_ack",
        "live_validation_status",
        "bias",
        "structure",
        "full_ticket",
        "trade_plan",
        "expiry",
        "sell_leg",
        "buy_leg",
        "suggested_contracts",
        "entry_limit",
        "max_profit",
        "max_loss",
        "credit_width_ratio",
        "trade_quality_status",
        "quality_gate_reason",
        "max_position_loss",
        "account_risk_pct",
        "remaining_upside",
        "target_exit",
        "invalidation",
        "synthesis_score",
        "score",
        "regime",
        "synthesis_reason",
        "status_reason",
        "sizing_note",
        "portfolio_risk_note",
        "visible_in_final_board",
    ]
    if final.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    regime = str(market_regime.get("regime") or "unknown")
    context = _execution_context_or_default(execution_context)
    for _, row in final.iterrows():
        status = _as_text(row.get("recommendation_status")).upper()
        hard_rejects = _as_text(row.get("hard_rejects"))
        ticket = _as_text(row.get("trade_plan")) or _as_text(row.get("full_ticket"))
        entry_limit = _as_float(row.get("entry_limit"))
        live_validation_status = _as_text(row.get("live_validation_status")).upper()
        portfolio_flag = _truthy(row.get("portfolio_risk_flag"))
        underlying_tier = _as_text(row.get("underlying_quality_tier")) or "unknown"
        underlying_reason = _as_text(row.get("underlying_quality_reason"))
        suggested_contracts = int(_as_float(row.get("suggested_contracts")) or 0)
        if status in {RecommendationStatus.ENTER.value, RecommendationStatus.ENTER_WITH_PORTFOLIO_RISK.value}:
            execution_status = "ready" if live_validation_status == "PASS" else "needs_live_validation"
        elif status == RecommendationStatus.WAIT_FOR_PRICE.value:
            execution_status = "waiting_for_price"
        elif status == RecommendationStatus.AVOID.value or hard_rejects:
            execution_status = "blocked"
        else:
            execution_status = "needs_review"
        if execution_status == "ready" and suggested_contracts <= 0:
            execution_status = "needs_sizing"

        if hard_rejects:
            quality = "blocked"
        elif str(row.get("quality_status") or "").strip().lower() == "qualified":
            quality = "qualified"
        else:
            quality = "watch"

        portfolio_fit = "risk_flagged" if portfolio_flag else "clear"
        execution_blockers = _execution_blockers_for_row(row, execution_status, ticket, entry_limit, suggested_contracts, context)
        if execution_status == "ready" and execution_blockers:
            if "fresh_live_schwab_required" in execution_blockers:
                execution_status = "needs_fresh_live_quote"
            elif "regular_session_quote_refresh_required" in execution_blockers:
                execution_status = "needs_fresh_live_quote"
            elif "portfolio_context_required" in execution_blockers:
                execution_status = "needs_portfolio_sizing"
            elif any(str(blocker).startswith("send_now_") for blocker in execution_blockers):
                execution_status = "waiting_for_price"
            elif POSITION_PROFIT_MATERIALITY_BLOCKER in execution_blockers:
                execution_status = "waiting_for_price"
            elif NEGATIVE_STRATEGY_EXPECTANCY_BLOCKER in execution_blockers:
                execution_status = "needs_confidence"
            elif POSITIVE_STRATEGY_EXPECTANCY_BLOCKER in execution_blockers:
                execution_status = "needs_confidence"
            elif PROFITABILITY_CALIBRATION_BLOCKER in execution_blockers:
                execution_status = "needs_confidence"
            elif (
                "agentic_reviews_required" in execution_blockers
                or "agentic_review_coverage_below_threshold" in execution_blockers
                or "ticker_agentic_review_coverage_below_threshold" in execution_blockers
            ):
                execution_status = "needs_agentic_review"
            elif "execution_confidence_below_threshold" in execution_blockers:
                execution_status = "needs_confidence"
        confidence_score, execution_confidence, quality_confidence = _execution_confidence(row, context, execution_blockers)
        if confidence_score < MIN_EXECUTION_CONFIDENCE_SCORE and execution_status == "ready":
            execution_blockers.append("execution_confidence_below_threshold")
            execution_status = "needs_confidence"
            confidence_score, execution_confidence, quality_confidence = _execution_confidence(row, context, execution_blockers)
        execution_gate_status = "pass" if not execution_blockers and execution_status == "ready" else "blocked"
        ready = (
            execution_status == "ready"
            and execution_gate_status == "pass"
            and bool(ticket)
            and entry_limit is not None
            and entry_limit > 0
            and suggested_contracts > 0
        )
        target_order_status = _target_order_status(
            row,
            ticket=ticket,
            entry_limit=entry_limit,
            execution_status=execution_status,
            underlying_tier=underlying_tier,
            trade_quality_confidence=quality_confidence,
            execution_blockers=execution_blockers,
        )
        if ready and target_order_status != "target_order_candidate":
            ready = False
            execution_gate_status = "blocked"
            execution_blockers = _dedupe_notes([*execution_blockers, "trade_quality_review_required"])
            if execution_status == "ready":
                execution_status = "needs_review"
        if target_order_status == "not_actionable_underlying_quality":
            execution_status = "blocked"
            execution_blockers = _dedupe_notes([*execution_blockers, "not_actionable_underlying_quality"])
            execution_gate_status = "blocked"
            ready = False
            quality = "blocked"
        decision_row = {
                "recommendation_rank": row.get("recommendation_rank", ""),
                "ticker": row.get("ticker", ""),
                "final_action": status or RecommendationStatus.REVIEW.value,
                "setup_quality_status": quality,
                "execution_status": execution_status,
                "execution_gate_status": execution_gate_status,
                "execution_blockers": "; ".join(_dedupe_notes(execution_blockers)),
                "execution_confidence_score": confidence_score,
                "execution_confidence_rating": execution_confidence,
                "trade_quality_confidence_rating": quality_confidence,
                "actual_forward_expectancy_status": row.get("actual_forward_expectancy_status", ""),
                "actual_forward_expectancy_sample_size": row.get("actual_forward_expectancy_sample_size", ""),
                "actual_forward_expectancy_note": row.get("actual_forward_expectancy_note", ""),
                "actual_forward_strategy_expectancy_status": row.get("actual_forward_strategy_expectancy_status", ""),
                "actual_forward_strategy_expectancy_sample_size": row.get("actual_forward_strategy_expectancy_sample_size", ""),
                "actual_forward_strategy_expectancy_family": row.get("actual_forward_strategy_expectancy_family", ""),
                "actual_forward_strategy_expectancy_note": row.get("actual_forward_strategy_expectancy_note", ""),
                "profitability_calibration_status": row.get("profitability_calibration_status", ""),
                "profitability_calibration_scope": row.get("profitability_calibration_scope", ""),
                "profitability_calibration_sample_size": row.get("profitability_calibration_sample_size", ""),
                "profitability_calibration_actual_status": row.get("profitability_calibration_actual_status", ""),
                "profitability_calibration_actual_sample_size": row.get("profitability_calibration_actual_sample_size", ""),
                "profitability_calibration_actual_avg_pnl": row.get("profitability_calibration_actual_avg_pnl", ""),
                "profitability_calibration_actual_profit_factor": row.get("profitability_calibration_actual_profit_factor", ""),
                "profitability_calibration_replay_status": row.get("profitability_calibration_replay_status", ""),
                "profitability_calibration_replay_sample_size": row.get("profitability_calibration_replay_sample_size", ""),
                "profitability_calibration_replay_avg_pnl": row.get("profitability_calibration_replay_avg_pnl", ""),
                "profitability_calibration_replay_profit_factor": row.get("profitability_calibration_replay_profit_factor", ""),
                "profitability_calibration_key": row.get("profitability_calibration_key", ""),
                "profitability_calibration_note": row.get("profitability_calibration_note", ""),
                "external_agent_review_count": int(_as_float(row.get("external_agent_review_count")) or 0),
                "external_agent_distinct_review_count": int(_as_float(row.get("external_agent_distinct_review_count")) or 0),
                "external_agent_review_agents": row.get("external_agent_review_agents", ""),
                "portfolio_fit_status": portfolio_fit,
                "underlying_quality_tier": underlying_tier,
                "underlying_quality_reason": underlying_reason,
                "target_order_status": target_order_status,
                "ready_to_enter": ready,
                "requires_portfolio_ack": portfolio_flag and ready,
                "live_validation_status": live_validation_status,
                "bias": row.get("bias", ""),
                "structure": row.get("structure", ""),
                "full_ticket": ticket,
                "trade_plan": ticket,
                "expiry": row.get("expiry", ""),
                "sell_leg": row.get("sell_leg", row.get("short_leg", "")),
                "buy_leg": row.get("buy_leg", row.get("long_leg", "")),
                "suggested_contracts": suggested_contracts,
                "entry_limit": row.get("entry_limit", ""),
                "max_profit": row.get("max_profit", ""),
                "max_loss": row.get("max_loss", ""),
                "credit_width_ratio": row.get("credit_width_ratio", ""),
                "trade_quality_status": row.get("trade_quality_status", ""),
                "quality_gate_reason": row.get("quality_gate_reason", ""),
                "max_position_loss": row.get("max_position_loss", ""),
                "account_risk_pct": row.get("account_risk_pct", ""),
                "remaining_upside": row.get("remaining_upside", ""),
                "target_exit": row.get("target_exit", ""),
                "invalidation": row.get("invalidation", ""),
                "synthesis_score": row.get("synthesis_score", ""),
                "score": row.get("score", ""),
                "regime": regime,
                "synthesis_reason": row.get("synthesis_reason", ""),
                "status_reason": row.get("status_reason", ""),
                "sizing_note": row.get("sizing_note", ""),
                "portfolio_risk_note": row.get("portfolio_risk_note", ""),
                "visible_in_final_board": bool(row.get("visible_in_final_board", True)),
            }
        decision_row["status_icon"] = _decision_icon(decision_row)
        decision_row["status_label"] = _decision_status_label(decision_row)
        rows.append(decision_row)
    return pd.DataFrame(rows, columns=columns)


def build_trade_tickets(decision_board: pd.DataFrame) -> pd.DataFrame:
    """Emit visible trade plans, with readiness separated from desired entry."""

    columns = [
        "recommendation_rank",
        "ticker",
        "status_icon",
        "status_label",
        "action",
        "order_readiness",
        "ready_to_enter",
        "execution_status",
        "execution_confidence_score",
        "execution_confidence_rating",
        "trade_quality_confidence_rating",
        "actual_forward_expectancy_status",
        "actual_forward_expectancy_sample_size",
        "actual_forward_expectancy_note",
        "actual_forward_strategy_expectancy_status",
        "actual_forward_strategy_expectancy_sample_size",
        "actual_forward_strategy_expectancy_family",
        "actual_forward_strategy_expectancy_note",
        "profitability_calibration_status",
        "profitability_calibration_scope",
        "profitability_calibration_sample_size",
        "profitability_calibration_actual_status",
        "profitability_calibration_actual_sample_size",
        "profitability_calibration_actual_avg_pnl",
        "profitability_calibration_actual_profit_factor",
        "profitability_calibration_replay_status",
        "profitability_calibration_replay_sample_size",
        "profitability_calibration_replay_avg_pnl",
        "profitability_calibration_replay_profit_factor",
        "profitability_calibration_key",
        "profitability_calibration_note",
        "external_agent_review_count",
        "external_agent_distinct_review_count",
        "external_agent_review_agents",
        "underlying_quality_tier",
        "underlying_quality_reason",
        "target_order_status",
        "execution_blockers",
        "suggested_contracts",
        "trade_plan",
        "expiry",
        "sell_leg",
        "buy_leg",
        "entry_limit",
        "entry_type",
        "max_profit",
        "max_loss",
        "position_max_profit",
        "position_max_loss",
        "credit_width_ratio",
        "trade_quality_status",
        "quality_gate_reason",
        "max_position_loss",
        "account_risk_pct",
        "target_exit",
        "invalidation",
        "synthesis_score",
        "live_validation_status",
        "requires_portfolio_ack",
        "portfolio_risk_note",
        "sizing_note",
        "status_reason",
    ]
    if decision_board.empty:
        return pd.DataFrame(columns=columns)
    working = decision_board.copy()
    if "trade_plan" not in working.columns:
        working["trade_plan"] = working["full_ticket"] if "full_ticket" in working.columns else ""
    for column in (
        "expiry",
        "sell_leg",
        "buy_leg",
        "credit_width_ratio",
        "trade_quality_status",
        "quality_gate_reason",
        "execution_gate_status",
        "execution_confidence_score",
        "execution_confidence_rating",
        "trade_quality_confidence_rating",
        "actual_forward_expectancy_status",
        "actual_forward_expectancy_sample_size",
        "actual_forward_expectancy_note",
        "actual_forward_strategy_expectancy_status",
        "actual_forward_strategy_expectancy_sample_size",
        "actual_forward_strategy_expectancy_family",
        "actual_forward_strategy_expectancy_note",
        "external_agent_review_count",
        "external_agent_distinct_review_count",
        "external_agent_review_agents",
        "underlying_quality_tier",
        "underlying_quality_reason",
        "target_order_status",
        "execution_blockers",
    ):
        if column not in working.columns:
            working[column] = "pass" if column == "execution_gate_status" else ""
    tickets = working[
        working["trade_plan"].astype(str).str.strip().ne("")
        & working["entry_limit"].map(lambda value: (_as_float(value) or 0.0) > 0)
        & ~working["execution_status"].astype(str).str.lower().eq("blocked")
        & (
            (
                working["execution_gate_status"].map(lambda value: str(value).strip().lower() == "pass")
                & working["ready_to_enter"].map(_truthy)
            )
            | working["target_order_status"].astype(str).str.lower().isin(
                ["target_order_candidate", "target_order_wait_for_price"]
            )
        )
        & ~working["execution_blockers"].map(
            lambda value: "ticker_agentic_review_coverage_below_threshold" in _blocker_set(value)
        )
    ].copy()
    if tickets.empty:
        return pd.DataFrame(columns=columns)
    tickets["ready_to_enter"] = tickets["ready_to_enter"].astype(bool)
    tickets["order_readiness"] = tickets.apply(_ticket_order_readiness, axis=1)
    tickets["action"] = tickets.apply(_ticket_action, axis=1)
    tickets["status_icon"] = tickets.apply(_decision_icon, axis=1)
    tickets["status_label"] = tickets.apply(_decision_status_label, axis=1)
    tickets["entry_type"] = tickets["trade_plan"].map(_entry_type_from_ticket)
    tickets["position_max_profit"] = tickets.apply(lambda row: _position_amount(row, "max_profit"), axis=1)
    tickets["position_max_loss"] = tickets.apply(lambda row: _position_amount(row, "max_loss"), axis=1)
    for column in columns:
        if column not in tickets.columns:
            tickets[column] = ""
    tickets = _sort_trades_by_confidence(tickets)
    return tickets[columns].reset_index(drop=True)


def split_trade_ticket_surfaces(trade_tickets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return send-now green rows separately from target-order rows."""

    if trade_tickets.empty:
        return trade_tickets.copy(), trade_tickets.copy()
    ready = trade_tickets[trade_tickets["ready_to_enter"].map(_truthy)].copy()
    target = trade_tickets[
        trade_tickets["target_order_status"]
        .astype(str)
        .str.lower()
        .isin(["target_order_candidate", "target_order_wait_for_price"])
        & ~trade_tickets["ready_to_enter"].map(_truthy)
    ].copy()
    return (
        _sort_trades_by_confidence(ready).reset_index(drop=True),
        _sort_trades_by_confidence(target).reset_index(drop=True),
    )


def _position_amount(row: Mapping[str, Any], column: str) -> float:
    one_lot = _as_float(row.get(column)) or 0.0
    contracts = int(_as_float(row.get("suggested_contracts")) or 0)
    return round(one_lot * max(contracts, 0), 2)


def _sort_trades_by_confidence(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort executable and target-order trade surfaces by actionability, then confidence."""

    if frame is None or frame.empty:
        return frame.copy() if frame is not None else pd.DataFrame()
    original_columns = list(frame.columns)
    working = frame.copy()
    working["__ready_rank"] = working["ready_to_enter"].map(lambda value: 1 if _truthy(value) else 0) if "ready_to_enter" in working.columns else 0
    working["__materiality_rank"] = (
        working["execution_blockers"].map(
            lambda value: 0 if POSITION_PROFIT_MATERIALITY_BLOCKER in _blocker_set(value) else 1
        )
        if "execution_blockers" in working.columns
        else 1
    )
    working["__order_readiness_rank"] = _order_readiness_sort_series(working)
    working["__confidence_score"] = _numeric_sort_series(working, "execution_confidence_score", default=-1.0)
    working["__quality_confidence_rank"] = _rating_sort_series(working, "trade_quality_confidence_rating")
    working["__execution_confidence_rank"] = _rating_sort_series(working, "execution_confidence_rating")
    working["__external_lane_count"] = _numeric_sort_series(working, "external_agent_distinct_review_count", default=0.0)
    working["__synthesis_score"] = _numeric_sort_series(working, "synthesis_score", default=-1.0)
    working["__recommendation_rank"] = _numeric_sort_series(working, "recommendation_rank", default=1_000_000.0)
    working["__original_order"] = range(len(working))
    sorted_frame = working.sort_values(
        [
            "__ready_rank",
            "__confidence_score",
            "__quality_confidence_rank",
            "__execution_confidence_rank",
            "__external_lane_count",
            "__materiality_rank",
            "__synthesis_score",
            "__order_readiness_rank",
            "__recommendation_rank",
            "__original_order",
        ],
        ascending=[False, False, False, False, False, False, False, False, True, True],
        kind="mergesort",
    )
    return sorted_frame[original_columns].reset_index(drop=True)


def _numeric_sort_series(frame: pd.DataFrame, column: str, *, default: float) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return frame[column].map(lambda value: _as_float(value) if _as_float(value) is not None else default)


def _rating_sort_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0, index=frame.index, dtype=int)
    ranks = {"HIGH": 3, "MEDIUM": 2, "LOW": 1, "NOT_EXECUTION_READY": 0}
    return frame[column].map(lambda value: ranks.get(_as_text(value).upper(), 0))


def _order_readiness_sort_series(frame: pd.DataFrame) -> pd.Series:
    if "order_readiness" not in frame.columns:
        return pd.Series(0, index=frame.index, dtype=int)
    ranks = {
        "ready_to_enter": 6,
        "target_order_price_validation": 5,
        "target_order_after_quote_refresh": 5,
        "target_order_after_market_open_and_live_recheck": 5,
        "target_order_after_live_recheck": 5,
        "target_order_profit_floor": 4,
        "target_order_wait_for_price": 4,
        "target_order_after_portfolio_sizing": 3,
        "target_order_after_portfolio_and_live_recheck": 3,
        "target_order_after_agentic_review": 2,
        "target_order_after_agentic_review_and_live_recheck": 2,
        "target_order_after_profitability_calibration": 1,
        "target_order_after_expectancy_evidence": 1,
    }
    return frame["order_readiness"].map(lambda value: ranks.get(_as_text(value), 0))


def build_market_open_recheck_queue(trade_tickets: pd.DataFrame) -> pd.DataFrame:
    """Queue target orders that only need a quote refresh before order entry."""

    columns = MARKET_OPEN_RECHECK_COLUMNS
    if trade_tickets is None or trade_tickets.empty:
        return pd.DataFrame(columns=columns)
    working = trade_tickets.copy()
    has_order_readiness = "order_readiness" in working.columns
    for column in columns:
        if column not in working.columns:
            working[column] = ""
    if not has_order_readiness:
        working["order_readiness"] = working.apply(_ticket_order_readiness, axis=1)
    allowed_recheck_blockers = {
        "market_session_open_required",
        "regular_session_quote_refresh_required",
        "fresh_live_schwab_required",
    }
    eligible_readiness = {
        "target_order_price_validation",
        "target_order_after_quote_refresh",
        "target_order_after_market_open_and_live_recheck",
        "target_order_after_live_recheck",
    }
    queue = working[
        ~working["ready_to_enter"].map(_truthy)
        & working["target_order_status"].astype(str).str.lower().eq("target_order_candidate")
        & working["order_readiness"].astype(str).isin(eligible_readiness)
        & working["execution_blockers"].map(
            lambda value: bool(_blocker_set(value)) and _blocker_set(value).issubset(allowed_recheck_blockers)
        )
        & working["entry_type"].astype(str).str.upper().isin({"CREDIT", "DEBIT"})
        & working["execution_confidence_score"].map(lambda value: (_as_float(value) or 0.0) >= MIN_EXECUTION_CONFIDENCE_SCORE)
        & working["trade_quality_confidence_rating"].astype(str).str.upper().isin({"MEDIUM", "HIGH"})
        & working["external_agent_distinct_review_count"].map(
            lambda value: (_as_float(value) or 0.0) >= MIN_AGENTIC_REVIEW_LANES_PER_TICKER
        )
    ].copy()
    if queue.empty:
        return pd.DataFrame(columns=columns)
    queue["required_recheck"] = "fresh Schwab quote + portfolio context + agentic lanes still passing"
    queue["recheck_action"] = "refresh the target limit in Schwab; enter only if this row moves to green ready_to_enter=true"
    queue = _sort_trades_by_confidence(queue)
    return queue[columns].reset_index(drop=True)


def _ticket_order_readiness(row: Mapping[str, Any]) -> str:
    if _truthy(row.get("ready_to_enter")):
        return "ready_to_enter"
    target_status = str(row.get("target_order_status") or "").strip().lower()
    blockers = _blocker_set(row.get("execution_blockers"))
    if target_status in {"target_order_candidate", "target_order_wait_for_price"}:
        if _profitability_calibration_status_blocks_target(row):
            return "target_order_after_profitability_calibration"
        if blockers in ({"market_session_open_required"}, {"regular_session_quote_refresh_required"}):
            return "target_order_price_validation"
        if POSITION_PROFIT_MATERIALITY_BLOCKER in blockers:
            return "target_order_profit_floor"
        if blockers & {NEGATIVE_STRATEGY_EXPECTANCY_BLOCKER, POSITIVE_STRATEGY_EXPECTANCY_BLOCKER}:
            return "target_order_after_expectancy_evidence"
        if PROFITABILITY_CALIBRATION_BLOCKER in blockers:
            return "target_order_after_profitability_calibration"
        if blockers & {
            "agentic_reviews_required",
            "agentic_review_coverage_below_threshold",
            "ticker_agentic_review_coverage_below_threshold",
        }:
            return "target_order_after_agentic_review"
        if "portfolio_context_required" in blockers:
            return "target_order_after_portfolio_sizing"
        if target_status == "target_order_wait_for_price":
            return "target_order_wait_for_price"
        return "target_order_price_validation"
    execution = str(row.get("execution_status") or "").strip().lower()
    if execution == "blocked":
        return "not_ready_objective_blocker"
    if execution == "needs_live_validation":
        return "not_ready_live_validation_required"
    if execution == "needs_fresh_live_quote":
        return "not_ready_fresh_live_quote_required"
    if execution == "needs_portfolio_sizing":
        return "not_ready_portfolio_required"
    if execution == "needs_agentic_review":
        return "not_ready_agentic_review_required"
    if execution == "needs_confidence":
        return "not_ready_confidence_required"
    if execution == "waiting_for_price":
        return "not_ready_wait_for_price"
    if execution == "needs_sizing":
        return "not_ready_sizing_required"
    return "not_ready_review_required"


def _ticket_action(row: Mapping[str, Any]) -> str:
    if _truthy(row.get("ready_to_enter")):
        return (
            "manual_entry_with_portfolio_ack"
            if str(row.get("final_action") or "").strip().upper() == RecommendationStatus.ENTER_WITH_PORTFOLIO_RISK.value
            else "manual_entry"
        )
    readiness = _ticket_order_readiness(row)
    if readiness == "not_ready_objective_blocker":
        return "blocked_review_only"
    if readiness == "not_ready_live_validation_required":
        return "reprice_with_fresh_schwab"
    if readiness == "not_ready_fresh_live_quote_required":
        return "rerun_with_live_schwab"
    if readiness in {"target_order_price_validation", "target_order_after_quote_refresh", "target_order_after_live_recheck"}:
        return "work_target_limit"
    if readiness == "target_order_profit_floor":
        return "work_target_only_if_profit_floor_clears"
    if readiness in {"target_order_after_agentic_review", "target_order_after_agentic_review_and_live_recheck"}:
        return "complete_agentic_reviews_then_price_validation"
    if readiness in {"target_order_after_portfolio_sizing", "target_order_after_portfolio_and_live_recheck"}:
        return "load_portfolio_then_price_validation"
    if readiness == "target_order_after_market_open_and_live_recheck":
        return "work_target_limit"
    if readiness == "target_order_wait_for_price":
        return "work_target_limit_if_price_improves"
    if readiness == "not_ready_portfolio_required":
        return "load_portfolio_context"
    if readiness == "not_ready_agentic_review_required":
        return "ingest_agentic_reviews"
    if readiness == "not_ready_confidence_required":
        return "manual_confidence_review"
    if readiness == "not_ready_wait_for_price":
        return "wait_for_desired_price"
    return "review_desired_ticket"


def _blocker_set(value: Any) -> set[str]:
    return {part.strip() for part in _as_text(value).split(";") if part.strip() and part.strip().lower() != "nan"}


def _distinct_external_review_agent_count(external_agent_reviews: pd.DataFrame) -> int:
    if external_agent_reviews is None or external_agent_reviews.empty or "agent" not in external_agent_reviews.columns:
        return 0
    return len(
        {
            str(agent).strip().lower()
            for agent in external_agent_reviews["agent"].dropna().tolist()
            if str(agent).strip()
        }
    )


def _market_datetime(now: Optional[dt.datetime] = None) -> dt.datetime:
    current = now or dt.datetime.now(MARKET_TIME_ZONE)
    if current.tzinfo is None:
        return current.replace(tzinfo=MARKET_TIME_ZONE)
    return current.astimezone(MARKET_TIME_ZONE)


def is_regular_market_session_open(now: Optional[dt.datetime] = None) -> bool:
    """Return whether regular U.S. equity/options hours are currently open."""

    current = _market_datetime(now)
    if not is_regular_market_day(current.date()):
        return False
    current_time = current.time()
    return REGULAR_MARKET_OPEN <= current_time < REGULAR_MARKET_CLOSE


def is_regular_market_day(day: dt.date) -> bool:
    """Return whether U.S. equity/options markets have a regular session that day."""

    return day.weekday() < 5 and day not in us_equity_market_holidays(day.year)


def next_regular_market_session_start(now: Optional[dt.datetime] = None) -> dt.datetime:
    """Return the next regular U.S. equity/options session start in Pacific time."""

    current = _market_datetime(now)
    current_day = current.date()
    current_start = dt.datetime.combine(current_day, REGULAR_MARKET_OPEN, tzinfo=MARKET_TIME_ZONE)
    if is_regular_market_day(current_day) and current <= current_start:
        return current_start
    search_day = current_day + dt.timedelta(days=0 if current.time() < REGULAR_MARKET_OPEN else 1)
    for offset in range(MARKET_HOLIDAY_LOOKAHEAD_DAYS + 1):
        candidate_day = search_day + dt.timedelta(days=offset)
        if is_regular_market_day(candidate_day):
            return dt.datetime.combine(candidate_day, REGULAR_MARKET_OPEN, tzinfo=MARKET_TIME_ZONE)
    raise RuntimeError("could not find the next regular market session within the configured lookahead")


def us_equity_market_holidays(year: int) -> set[dt.date]:
    """NYSE/Nasdaq full-day U.S. equity market holidays, including observed dates."""

    holidays: set[dt.date] = set()
    for scoped_year in (year - 1, year, year + 1):
        holidays.update(
            {
                _observed_fixed_holiday(scoped_year, 1, 1),
                _nth_weekday(scoped_year, 1, 0, 3),
                _nth_weekday(scoped_year, 2, 0, 3),
                _easter_date(scoped_year) - dt.timedelta(days=2),
                _last_weekday(scoped_year, 5, 0),
                _observed_fixed_holiday(scoped_year, 6, 19),
                _observed_fixed_holiday(scoped_year, 7, 4),
                _nth_weekday(scoped_year, 9, 0, 1),
                _nth_weekday(scoped_year, 11, 3, 4),
                _observed_fixed_holiday(scoped_year, 12, 25),
            }
        )
    return holidays


def _observed_fixed_holiday(year: int, month: int, day: int) -> dt.date:
    holiday = dt.date(year, month, day)
    if holiday.weekday() == 5:
        return holiday - dt.timedelta(days=1)
    if holiday.weekday() == 6:
        return holiday + dt.timedelta(days=1)
    return holiday


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> dt.date:
    day = dt.date(year, month, 1)
    delta = (weekday - day.weekday()) % 7
    return day + dt.timedelta(days=delta + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> dt.date:
    if month == 12:
        day = dt.date(year + 1, 1, 1) - dt.timedelta(days=1)
    else:
        day = dt.date(year, month + 1, 1) - dt.timedelta(days=1)
    delta = (day.weekday() - weekday) % 7
    return day - dt.timedelta(days=delta)


def _easter_date(year: int) -> dt.date:
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return dt.date(year, month, day)


def build_execution_context(
    *,
    live_schwab: bool,
    chain_snapshot_dir: Optional[Path],
    portfolio_context: Mapping[str, Any],
    research_task_count: int,
    external_review_count: int,
    agent_reviews_json: Optional[Path],
    external_review_agent_count: Optional[int] = None,
    agent_dispatch_task_count: Optional[int] = None,
    market_session_open: Optional[bool] = None,
) -> dict[str, Any]:
    """Describe run-level gates required before a row can become executable."""

    portfolio = dict(portfolio_context or {})
    total_value = _as_float(portfolio.get("total_value")) or _as_float(portfolio.get("net_liquidation")) or 0.0
    portfolio_ready = str(portfolio.get("status") or "").strip().lower() == "ok" and total_value > 0
    snapshot_mode = chain_snapshot_dir is not None
    fresh_live_ready = bool(live_schwab and not snapshot_mode)
    market_session_ready = True if market_session_open is None else bool(market_session_open)
    market_session_recheck_required = bool(fresh_live_ready and not market_session_ready)
    broad_review_coverage = float(external_review_count) / float(research_task_count) if research_task_count > 0 else 0.0
    review_agent_count = int(external_review_agent_count or 0)
    dispatch_task_count = int(agent_dispatch_task_count or 0)
    lane_review_coverage = float(review_agent_count) / float(dispatch_task_count) if dispatch_task_count > 0 else broad_review_coverage
    review_coverage = lane_review_coverage if dispatch_task_count > 0 else broad_review_coverage
    agentic_reviews_present = bool(agent_reviews_json and external_review_count > 0)
    agentic_reviews_ready = bool(agentic_reviews_present and review_coverage >= MIN_AGENTIC_REVIEW_COVERAGE)
    blockers: list[str] = []
    if not fresh_live_ready:
        blockers.append("fresh_live_schwab_required")
    if not portfolio_ready:
        blockers.append("portfolio_context_required")
    if not agentic_reviews_present:
        blockers.append("agentic_reviews_required")
    elif not agentic_reviews_ready:
        blockers.append("agentic_review_coverage_below_threshold")
    return {
        "quote_mode": "live_schwab" if fresh_live_ready else "snapshot_replay" if snapshot_mode else "unvalidated",
        "fresh_live_quotes_ready": fresh_live_ready,
        "market_session_open": market_session_ready if fresh_live_ready else False,
        "market_session_gate_required": False,
        "market_session_recheck_required": market_session_recheck_required,
        "portfolio_ready": portfolio_ready,
        "portfolio_status": str(portfolio.get("status") or "unknown"),
        "portfolio_total_value": total_value,
        "agentic_reviews_present": agentic_reviews_present,
        "agentic_reviews_ready": agentic_reviews_ready,
        "external_review_count": int(external_review_count),
        "external_review_agent_count": review_agent_count,
        "research_task_count": int(research_task_count),
        "agent_dispatch_task_count": dispatch_task_count,
        "broad_review_coverage_pct": round(broad_review_coverage, 4),
        "agentic_review_lane_coverage_pct": round(lane_review_coverage, 4),
        "agentic_review_coverage_pct": round(review_coverage, 4),
        "agentic_review_coverage_basis": "subagent_lanes" if dispatch_task_count > 0 else "research_tasks",
        "external_review_coverage_pct": round(review_coverage, 4),
        "min_agentic_review_coverage_pct": MIN_AGENTIC_REVIEW_COVERAGE,
        "min_agentic_review_lanes_per_ticker": MIN_AGENTIC_REVIEW_LANES_PER_TICKER,
        "run_gate_blockers": blockers,
        "monthly_profit_target": MONTHLY_PROFIT_TARGET,
    }


def _execution_context_or_default(execution_context: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    if execution_context is not None:
        return dict(execution_context)
    return {
        "quote_mode": "unit_test_permissive",
        "fresh_live_quotes_ready": True,
        "market_session_open": True,
        "market_session_gate_required": False,
        "market_session_recheck_required": False,
        "portfolio_ready": True,
        "portfolio_status": "ok",
        "portfolio_total_value": 100_000.0,
        "agentic_reviews_present": True,
        "agentic_reviews_ready": True,
        "external_review_count": 1,
        "external_review_agent_count": 1,
        "research_task_count": 1,
        "agent_dispatch_task_count": 1,
        "broad_review_coverage_pct": 1.0,
        "agentic_review_lane_coverage_pct": 1.0,
        "agentic_review_coverage_pct": 1.0,
        "agentic_review_coverage_basis": "research_tasks",
        "external_review_coverage_pct": 1.0,
        "min_agentic_review_coverage_pct": MIN_AGENTIC_REVIEW_COVERAGE,
        "min_agentic_review_lanes_per_ticker": 0,
        "run_gate_blockers": [],
        "monthly_profit_target": MONTHLY_PROFIT_TARGET,
    }


def _green_position_profit_floor(execution_context: Mapping[str, Any]) -> float:
    portfolio_value = _as_float(execution_context.get("portfolio_total_value")) or 0.0
    portfolio_floor = portfolio_value * MIN_GREEN_POSITION_MAX_PROFIT_PORTFOLIO_PCT
    return round(max(MIN_GREEN_POSITION_MAX_PROFIT, portfolio_floor), 2)


def _position_max_profit_value(row: Mapping[str, Any], suggested_contracts: int) -> float:
    explicit = _as_float(row.get("position_max_profit"))
    if explicit is not None and explicit > 0:
        return round(explicit, 2)
    one_lot = _as_float(row.get("max_profit")) or 0.0
    return round(one_lot * max(int(suggested_contracts or 0), 0), 2)


def _materiality_exception_reason(row: Mapping[str, Any]) -> str:
    for key in (
        "materiality_exception_reason",
        "exceptional_edge_reason",
        "manual_materiality_override_reason",
    ):
        reason = _as_text(row.get(key))
        if reason:
            return reason
    return ""


def _is_position_profit_below_green_floor(
    row: Mapping[str, Any],
    *,
    suggested_contracts: int,
    execution_context: Mapping[str, Any],
) -> bool:
    if suggested_contracts <= 0 or _materiality_exception_reason(row):
        return False
    position_profit = _position_max_profit_value(row, suggested_contracts)
    return 0 < position_profit < _green_position_profit_floor(execution_context)


def _expectancy_values_are_negative(row: Mapping[str, Any], prefix: str) -> bool:
    sample = int(_as_float(row.get(f"{prefix}_sample_size")) or 0)
    if sample <= 0:
        return False
    avg_pnl = _as_float(row.get(f"{prefix}_avg_pnl"))
    profit_factor = _as_float(row.get(f"{prefix}_profit_factor"))
    win_rate = _as_float(row.get(f"{prefix}_win_rate"))
    return bool(
        (avg_pnl is not None and avg_pnl < 0)
        or (profit_factor is not None and profit_factor < 1.0)
        or (win_rate is not None and win_rate < MIN_TICKER_EXPECTANCY_WIN_RATE)
    )


def _negative_strategy_expectancy_blocks_green(row: Mapping[str, Any]) -> bool:
    status = _as_text(row.get("actual_forward_strategy_expectancy_status")).upper()
    sample = int(_as_float(row.get("actual_forward_strategy_expectancy_sample_size")) or 0)
    if status not in {"WARN", "BLOCK"} or sample < MIN_TICKER_EXPECTANCY_SAMPLE_SIZE:
        return False
    return _expectancy_values_are_negative(row, "actual_forward_strategy_expectancy")


def _current_edge_override_reason(row: Mapping[str, Any]) -> str:
    for key in (
        "current_edge_override_reason",
        "exceptional_current_edge_reason",
        "manual_current_edge_override_reason",
    ):
        reason = _as_text(row.get(key))
        if reason:
            return reason
    return ""


def _strategy_expectancy_annotation_present(row: Mapping[str, Any]) -> bool:
    return any(
        key in row and _as_text(row.get(key))
        for key in (
            "actual_forward_strategy_expectancy_status",
            "actual_forward_strategy_expectancy_note",
            "actual_forward_strategy_expectancy_family",
        )
    ) or _as_float(row.get("actual_forward_strategy_expectancy_sample_size")) is not None


def _positive_strategy_expectancy_ready_for_green(row: Mapping[str, Any]) -> bool:
    if _current_edge_override_reason(row):
        return True
    if not _strategy_expectancy_annotation_present(row):
        return False
    status = _as_text(row.get("actual_forward_strategy_expectancy_status")).upper()
    sample = int(_as_float(row.get("actual_forward_strategy_expectancy_sample_size")) or 0)
    if status != "PASS" or sample < MIN_TICKER_EXPECTANCY_SAMPLE_SIZE:
        return False
    return not _expectancy_values_are_negative(row, "actual_forward_strategy_expectancy")


def _profitability_calibration_ready_for_green(row: Mapping[str, Any]) -> bool:
    if "profitability_calibration_status" not in row:
        return True
    status = _as_text(row.get("profitability_calibration_status")).upper()
    if not status:
        return True
    return status == "PASS"


def _profitability_calibration_status_blocks_target(row: Mapping[str, Any]) -> bool:
    if "profitability_calibration_status" not in row:
        return False
    status = _as_text(row.get("profitability_calibration_status")).upper()
    return bool(status and status != "PASS")


def _profitability_calibration_actual_support_negative(row: Mapping[str, Any]) -> bool:
    status = _as_text(row.get("profitability_calibration_status")).upper()
    if status == "PASS":
        return False
    actual_status = _as_text(row.get("profitability_calibration_actual_status")).upper()
    if actual_status not in {"WARN", "BLOCK"}:
        return False
    sample = int(_as_float(row.get("profitability_calibration_actual_sample_size")) or 0)
    if sample <= 0:
        return False
    avg_pnl = _as_float(row.get("profitability_calibration_actual_avg_pnl"))
    profit_factor = _as_float(row.get("profitability_calibration_actual_profit_factor"))
    return bool(
        (avg_pnl is not None and avg_pnl < 0)
        or (profit_factor is not None and profit_factor < 1.0)
    )


def _calibration_materiality_blocks_target_surface(blockers: set[str]) -> bool:
    if not {PROFITABILITY_CALIBRATION_BLOCKER, POSITION_PROFIT_MATERIALITY_BLOCKER}.issubset(blockers):
        return False
    prerequisite_blockers = {
        "fresh_live_schwab_required",
        "regular_session_quote_refresh_required",
        "live_validation_pass_required",
        "agentic_reviews_required",
        "agentic_review_coverage_below_threshold",
        "ticker_agentic_review_coverage_below_threshold",
        "portfolio_context_required",
    }
    return not bool(blockers & prerequisite_blockers)


def _short_put_cash_risk_blockers(row: Mapping[str, Any]) -> list[str]:
    if _strategy_family_from_ticket_row(row) != "short_put":
        return []
    blockers: list[str] = []
    max_loss = _as_float(row.get("max_loss")) or 0.0
    if max_loss <= 0:
        return ["short_put_cash_risk_unavailable"]
    portfolio_cash = _as_float(row.get("portfolio_cash")) or 0.0
    if portfolio_cash > 0 and max_loss > portfolio_cash * MAX_SHORT_PUT_CASH_USAGE_PCT:
        blockers.append(f"short_put_cash_required_above_{int(MAX_SHORT_PUT_CASH_USAGE_PCT * 100)}pct_cash")
    account_risk = _as_float(row.get("account_risk_pct"))
    if account_risk is not None and account_risk > MAX_SHORT_PUT_ACCOUNT_RISK_PCT:
        blockers.append(f"short_put_account_risk_above_{MAX_SHORT_PUT_ACCOUNT_RISK_PCT:.2%}")
    return blockers


def _execution_blockers_for_row(
    row: Mapping[str, Any],
    execution_status: str,
    ticket: str,
    entry_limit: Optional[float],
    suggested_contracts: int,
    execution_context: Mapping[str, Any],
) -> list[str]:
    blockers = list(execution_context.get("run_gate_blockers") or [])
    live_status = _as_text(row.get("live_validation_status")).upper()
    status = _as_text(row.get("recommendation_status")).upper()
    if status == RecommendationStatus.AVOID.value or _as_text(row.get("hard_rejects")):
        blockers.append("objective_blocker")
    review_resolved_for_recheck = _review_resolved_for_target_recheck(
        row,
        ticket=ticket,
        entry_limit=entry_limit,
        execution_context=execution_context,
    )
    if status == RecommendationStatus.REVIEW.value and not review_resolved_for_recheck:
        blockers.append("manual_review_required")
    if status == RecommendationStatus.WAIT_FOR_PRICE.value:
        blockers.append("wait_for_price")
    live_validation_deferred = _market_closed_live_validation_deferred(row, execution_context)
    if live_status != "PASS" and not live_validation_deferred:
        blockers.append("live_validation_pass_required")
    if execution_context.get("agentic_reviews_ready"):
        configured_min_lanes = _as_float(execution_context.get("min_agentic_review_lanes_per_ticker"))
        min_lanes = int(MIN_AGENTIC_REVIEW_LANES_PER_TICKER if configured_min_lanes is None else configured_min_lanes)
        distinct_review_count = int(_as_float(row.get("external_agent_distinct_review_count")) or 0)
        if min_lanes > 0 and distinct_review_count < min_lanes:
            blockers.append("ticker_agentic_review_coverage_below_threshold")
    if not ticket:
        blockers.append("trade_plan_required")
    if entry_limit is None or entry_limit <= 0:
        blockers.append("positive_entry_limit_required")
    if suggested_contracts <= 0:
        blockers.append("positive_contract_size_required")
    if _profitability_calibration_status_blocks_target(row):
        blockers.append(PROFITABILITY_CALIBRATION_BLOCKER)
    if _profitability_calibration_actual_support_negative(row):
        blockers.append(PROFITABILITY_CALIBRATION_ACTUAL_NEGATIVE_BLOCKER)
    entry_like_statuses = {
        RecommendationStatus.ENTER.value,
        RecommendationStatus.ENTER_WITH_PORTFOLIO_RISK.value,
        RecommendationStatus.REVIEW.value,
        RecommendationStatus.WAIT_FOR_PRICE.value,
    }
    if status in entry_like_statuses and _negative_strategy_expectancy_blocks_green(row):
        blockers.append(NEGATIVE_STRATEGY_EXPECTANCY_BLOCKER)
    if status in {RecommendationStatus.ENTER.value, RecommendationStatus.ENTER_WITH_PORTFOLIO_RISK.value}:
        if _is_position_profit_below_green_floor(
            row,
            suggested_contracts=suggested_contracts,
            execution_context=execution_context,
        ):
            blockers.append(POSITION_PROFIT_MATERIALITY_BLOCKER)
        if not _positive_strategy_expectancy_ready_for_green(row):
            blockers.append(POSITIVE_STRATEGY_EXPECTANCY_BLOCKER)
        blockers.extend(_short_put_cash_risk_blockers(row))
    blockers.extend(_send_now_economics_blockers(row, ticket=ticket, entry_limit=entry_limit))
    if execution_status == "needs_sizing":
        blockers.append("sizing_required")
    return _dedupe_notes(blockers)


def _send_now_economics_blockers(
    row: Mapping[str, Any],
    *,
    ticket: str,
    entry_limit: Optional[float],
) -> list[str]:
    if _as_text(row.get("recommendation_status")).upper() not in {
        RecommendationStatus.ENTER.value,
        RecommendationStatus.ENTER_WITH_PORTFOLIO_RISK.value,
    }:
        return []
    entry_type = _entry_type_from_ticket(ticket)
    if not entry_type or entry_limit is None or entry_limit <= 0:
        return []
    blockers: list[str] = []
    if entry_type == "CREDIT":
        if _strategy_family_from_ticket_row(row) == "short_put":
            if entry_limit < MIN_SEND_NOW_CREDIT:
                blockers.append(f"send_now_credit_below_{MIN_SEND_NOW_CREDIT:.2f}")
            return blockers
        credit_width = _as_float(row.get("credit_width_ratio")) or 0.0
        if entry_limit < MIN_SEND_NOW_CREDIT:
            blockers.append(f"send_now_credit_below_{MIN_SEND_NOW_CREDIT:.2f}")
        if credit_width < MIN_SEND_NOW_CREDIT_WIDTH_RATIO:
            blockers.append(f"send_now_credit_width_below_{int(MIN_SEND_NOW_CREDIT_WIDTH_RATIO * 100)}pct")
    elif entry_type == "DEBIT":
        max_profit = _as_float(row.get("max_profit")) or 0.0
        max_loss = _as_float(row.get("max_loss")) or 0.0
        reward_risk = max_profit / max_loss if max_loss > 0 else 0.0
        if reward_risk < MIN_SEND_NOW_DEBIT_REWARD_RISK_RATIO:
            blockers.append(
                f"send_now_debit_reward_risk_below_{MIN_SEND_NOW_DEBIT_REWARD_RISK_RATIO:.1f}x"
            )
        breakeven_move = _debit_breakeven_move_pct(row, ticket)
        breakeven_limit = _send_now_debit_breakeven_move_limit_pct(row.get("dte"))
        if breakeven_move is not None and breakeven_limit is not None and breakeven_move > breakeven_limit:
            blockers.append(f"send_now_debit_breakeven_move_above_{int(breakeven_limit * 100)}pct")
        if _send_now_debit_directional_edge_too_weak(row):
            blockers.append("send_now_debit_directional_edge_below_threshold")
    return blockers


def _send_now_debit_directional_edge_too_weak(row: Mapping[str, Any]) -> bool:
    has_expectancy_columns = any(
        key in row
        for key in (
            "actual_forward_expectancy_status",
            "actual_forward_strategy_expectancy_status",
            "actual_forward_expectancy_sample_size",
            "actual_forward_strategy_expectancy_sample_size",
        )
    )
    if not has_expectancy_columns:
        return False
    expectancy_status = _as_text(row.get("actual_forward_expectancy_status")).upper()
    strategy_expectancy_status = _as_text(row.get("actual_forward_strategy_expectancy_status")).upper()
    expectancy_sample = int(_as_float(row.get("actual_forward_expectancy_sample_size")) or 0)
    strategy_expectancy_sample = int(_as_float(row.get("actual_forward_strategy_expectancy_sample_size")) or 0)
    has_outcome_support = (
        expectancy_status in {"PASS", "WARN"}
        or strategy_expectancy_status in {"PASS", "WARN"}
        or expectancy_sample > 0
        or strategy_expectancy_sample > 0
    )
    if has_outcome_support:
        return False
    flow_bias = _as_float(row.get("combined_flow_bias"))
    if flow_bias is None:
        flow_bias = _as_float(row.get("flow_bias"))
    if flow_bias is None:
        return False
    return abs(flow_bias) < MIN_SEND_NOW_DEBIT_FLOW_BIAS_WITHOUT_EXPECTANCY


def _send_now_debit_breakeven_move_limit_pct(dte_value: Any) -> Optional[float]:
    dte = _as_float(dte_value)
    if dte is None or dte <= 0:
        return None
    if dte <= 14:
        return MAX_SHORT_DTE_SEND_NOW_DEBIT_BREAKEVEN_MOVE_PCT
    if dte <= 30:
        return MAX_MEDIUM_DTE_SEND_NOW_DEBIT_BREAKEVEN_MOVE_PCT
    if dte <= MAX_LIVE_DTE:
        return MAX_LONG_DTE_SEND_NOW_DEBIT_BREAKEVEN_MOVE_PCT
    return None


def _debit_breakeven_move_pct(row: Mapping[str, Any], ticket: str) -> Optional[float]:
    spot = (
        _as_float(row.get("spot_live"))
        or _as_float(row.get("live_underlying_price"))
        or _as_float(row.get("underlying_price"))
        or _as_float(row.get("close"))
    )
    breakeven = _as_float(row.get("breakeven"))
    if spot is None or spot <= 0 or breakeven is None or breakeven <= 0:
        return None
    text = _as_text(ticket).upper()
    if " PUT" in text:
        if spot <= breakeven:
            return 0.0
        return (spot - breakeven) / spot
    if " CALL" in text:
        if spot >= breakeven:
            return 0.0
        return (breakeven - spot) / spot
    return None


def _review_resolved_for_target_recheck(
    row: Mapping[str, Any],
    *,
    ticket: str,
    entry_limit: Optional[float],
    execution_context: Mapping[str, Any],
) -> bool:
    if _as_text(row.get("hard_rejects")) or _as_text(row.get("quality_gate_reason")):
        return False
    if not _has_complete_target_math({**dict(row), "trade_plan": ticket, "entry_limit": entry_limit}):
        return False
    if not execution_context.get("agentic_reviews_ready"):
        return False
    configured_min_lanes = _as_float(execution_context.get("min_agentic_review_lanes_per_ticker"))
    min_lanes = int(MIN_AGENTIC_REVIEW_LANES_PER_TICKER if configured_min_lanes is None else configured_min_lanes)
    distinct_review_count = int(_as_float(row.get("external_agent_distinct_review_count")) or 0)
    if min_lanes > 0 and distinct_review_count < min_lanes:
        return False
    return int(_as_float(row.get("agent_objective_blocker_count")) or 0) <= 0


def _market_closed_live_validation_deferred(row: Mapping[str, Any], execution_context: Mapping[str, Any]) -> bool:
    return bool(
        _as_text(row.get("live_validation_status")).upper() in {"TARGET_QUOTE_REFRESH", "MARKET_CLOSED_RECHECK"}
        and execution_context.get("market_session_recheck_required")
        and execution_context.get("fresh_live_quotes_ready")
    )


def _target_order_status(
    row: Mapping[str, Any],
    *,
    ticket: str,
    entry_limit: Optional[float],
    execution_status: str,
    underlying_tier: str,
    trade_quality_confidence: str = "",
    execution_blockers: Sequence[str] = (),
) -> str:
    status = _as_text(row.get("recommendation_status")).upper()
    live_status = _as_text(row.get("live_validation_status")).upper()
    entry_type = _entry_type_from_ticket(ticket)
    strategy_family = _strategy_family_from_ticket_row(row)
    max_profit = _as_float(row.get("max_profit")) or 0.0
    max_loss = _as_float(row.get("max_loss")) or 0.0
    credit_width = _as_float(row.get("credit_width_ratio")) or 0.0
    low_quality_confidence = _as_text(trade_quality_confidence).upper() == "LOW"
    materiality_floor_target = POSITION_PROFIT_MATERIALITY_BLOCKER in set(execution_blockers or ())
    external_lane_count = int(_as_float(row.get("external_agent_distinct_review_count")) or 0)
    has_agentic_ticket_review = external_lane_count >= MIN_AGENTIC_REVIEW_LANES_PER_TICKER
    if status == RecommendationStatus.AVOID.value or _as_text(row.get("hard_rejects")):
        return "blocked_objective_reject"
    if underlying_tier != "core":
        return "not_actionable_underlying_quality"
    if not ticket or entry_limit is None or entry_limit <= 0:
        return "not_actionable_missing_price_or_ticket"
    if max_profit <= 0 or max_loss <= 0:
        return "not_actionable_risk_reward"
    if strategy_family == "short_put":
        if _short_put_cash_risk_blockers(row):
            return "not_actionable_cash_secured_risk"
    elif max_loss > MAX_ONE_LOT_LOSS:
        return "not_actionable_risk_reward"
    if strategy_family != "short_put" and entry_type != "DEBIT" and credit_width < MIN_CREDIT_WIDTH_RATIO:
        return "not_actionable_credit_width"
    blocker_set = set(execution_blockers or ())
    if NEGATIVE_STRATEGY_EXPECTANCY_BLOCKER in set(execution_blockers or ()):
        return "review_only_expectancy_evidence"
    if PROFITABILITY_CALIBRATION_ACTUAL_NEGATIVE_BLOCKER in blocker_set:
        return "review_only_profitability_calibration"
    if _calibration_materiality_blocks_target_surface(blocker_set):
        return "review_only_profitability_calibration"
    if live_status != "PASS":
        if _is_dated_recheck_target(row, status=status, live_status=live_status):
            if low_quality_confidence and not has_agentic_ticket_review:
                return "review_only_low_trade_quality"
            if status == RecommendationStatus.WAIT_FOR_PRICE.value:
                return "target_order_wait_for_price"
            return "target_order_candidate"
        if low_quality_confidence:
            return "review_only_low_trade_quality"
        return "not_actionable_unvalidated_chain"
    if low_quality_confidence and not materiality_floor_target:
        return "review_only_low_trade_quality"
    if status in {RecommendationStatus.ENTER.value, RecommendationStatus.ENTER_WITH_PORTFOLIO_RISK.value}:
        return "target_order_candidate"
    if status == RecommendationStatus.WAIT_FOR_PRICE.value:
        return "target_order_wait_for_price"
    if execution_status == "needs_review":
        return "review_only_not_target_order"
    return "not_actionable"


def _is_dated_recheck_target(row: Mapping[str, Any], *, status: str, live_status: str) -> bool:
    if live_status and live_status not in {"TARGET_QUOTE_REFRESH", "MARKET_CLOSED_RECHECK"}:
        return False
    if status not in {
        RecommendationStatus.ENTER.value,
        RecommendationStatus.ENTER_WITH_PORTFOLIO_RISK.value,
        RecommendationStatus.REVIEW.value,
        RecommendationStatus.WAIT_FOR_PRICE.value,
    }:
        return False
    reason = _as_text(row.get("status_reason")).lower()
    if "dated uw" not in reason and "fresh schwab chain" not in reason:
        return False
    if _as_text(row.get("quality_gate_reason")) or _as_text(row.get("hard_rejects")):
        return False
    return True


def _entry_type_from_ticket(ticket: Any) -> str:
    text = _as_text(ticket).upper()
    if " DEBIT" in text:
        return "DEBIT"
    if " CREDIT" in text:
        return "CREDIT"
    return ""


def _execution_confidence(
    row: Mapping[str, Any],
    execution_context: Mapping[str, Any],
    execution_blockers: Sequence[str],
) -> tuple[float, str, str]:
    score = 35.0
    quality_score = 35.0
    if _as_text(row.get("live_validation_status")).upper() == "PASS":
        score += 20.0
        quality_score += 10.0
    elif _market_closed_live_validation_deferred(row, execution_context):
        score += 15.0
        quality_score += 5.0
    if execution_context.get("fresh_live_quotes_ready"):
        score += 15.0
    if execution_context.get("portfolio_ready"):
        score += 12.0
    if execution_context.get("agentic_reviews_ready"):
        score += 8.0
    support = int(_as_float(row.get("agent_support_count")) or 0)
    caution = int(_as_float(row.get("agent_caution_count")) or 0)
    objective = int(_as_float(row.get("agent_objective_blocker_count")) or 0)
    score += min(support, 6)
    score -= min(caution * 4, 24)
    score -= objective * 30
    quality_score += min(support, 8)
    quality_score -= min(caution * 5, 25)
    quality_score -= objective * 35
    credit_width = _as_float(row.get("credit_width_ratio")) or 0.0
    max_profit = _as_float(row.get("max_profit")) or 0.0
    max_loss = _as_float(row.get("max_loss")) or 0.0
    entry_type = _entry_type_from_ticket(row.get("trade_plan") or row.get("full_ticket") or "")
    strategy_family = _strategy_family_from_ticket_row(row)
    if strategy_family == "short_put":
        entry_limit = _as_float(row.get("entry_limit")) or 0.0
        distance = _as_float(row.get("live_distance_pct"))
        if entry_limit >= MIN_SEND_NOW_CREDIT:
            quality_score += 8
        if distance is not None and distance >= MIN_SHORT_PUT_DISTANCE_PCT * 1.5:
            quality_score += 6
        if _short_put_cash_risk_blockers(row):
            score -= 12
            quality_score -= 25
    else:
        if credit_width >= 0.25:
            quality_score += 12
        elif credit_width >= MIN_CREDIT_WIDTH_RATIO:
            quality_score += 6
    if entry_type == "DEBIT" and max_loss > 0:
        reward_risk = max_profit / max_loss
        if reward_risk >= 2.0:
            quality_score += 12
        elif reward_risk >= MIN_SEND_NOW_DEBIT_REWARD_RISK_RATIO:
            quality_score += 8
    if 0 < max_loss <= MAX_ONE_LOT_LOSS:
        quality_score += 10
    suggested_contracts = int(_as_float(row.get("suggested_contracts")) or 0)
    if _is_position_profit_below_green_floor(
        row,
        suggested_contracts=suggested_contracts,
        execution_context=execution_context,
    ):
        score -= 18
        quality_score -= 25
    if _as_text(row.get("trade_quality_status")).lower() == "reviewable":
        quality_score += 10
    expectancy_status = _as_text(row.get("actual_forward_expectancy_status")).upper()
    expectancy_sample = int(_as_float(row.get("actual_forward_expectancy_sample_size")) or 0)
    strategy_expectancy_status = _as_text(row.get("actual_forward_strategy_expectancy_status")).upper()
    strategy_expectancy_sample = int(_as_float(row.get("actual_forward_strategy_expectancy_sample_size")) or 0)
    ticker_expectancy_negative = _expectancy_values_are_negative(row, "actual_forward_expectancy")
    strategy_expectancy_negative = _expectancy_values_are_negative(row, "actual_forward_strategy_expectancy")
    if expectancy_status == "PASS":
        score += 5 if expectancy_sample >= 5 else 3
        quality_score += 5 if expectancy_sample >= 5 else 3
    elif expectancy_status == "WARN":
        if ticker_expectancy_negative:
            score -= 6
            quality_score -= 10
        else:
            score += 1
            quality_score += 2
    if strategy_expectancy_status == "PASS":
        score += 10 if strategy_expectancy_sample >= 5 else 7
        quality_score += 15 if strategy_expectancy_sample >= 5 else 10
    elif strategy_expectancy_status == "WARN":
        if strategy_expectancy_negative:
            score -= 12
            quality_score -= 18
        else:
            score += 2
            quality_score += 4
    if _negative_strategy_expectancy_blocks_green(row):
        score -= 20
        quality_score -= 30
    if not _positive_strategy_expectancy_ready_for_green(row):
        score -= 15
        quality_score -= 20
    if _as_text(row.get("quality_gate_reason")):
        quality_score -= 30
    if execution_blockers:
        score -= min(len(execution_blockers) * 8.0, 32.0)
    score = round(max(0.0, min(100.0, score)), 2)
    quality_score = round(max(0.0, min(100.0, quality_score)), 2)
    if execution_blockers:
        execution_rating = "NOT_EXECUTION_READY"
    elif score >= 85:
        execution_rating = "HIGH"
    elif score >= MIN_EXECUTION_CONFIDENCE_SCORE:
        execution_rating = "MEDIUM"
    else:
        execution_rating = "LOW"
    if quality_score >= 80:
        quality_rating = "HIGH"
    elif quality_score >= 65:
        quality_rating = "MEDIUM"
    else:
        quality_rating = "LOW"
    return score, execution_rating, quality_rating


def build_execution_readiness(decision_board: pd.DataFrame, execution_context: Mapping[str, Any]) -> pd.DataFrame:
    """Summarize run and row-level readiness gates."""

    columns = ["gate", "status", "detail", "affected_rows"]
    ready_rows = int(decision_board["ready_to_enter"].map(_truthy).sum()) if not decision_board.empty else 0
    rows = [
        {
            "gate": "fresh_live_schwab",
            "status": "PASS" if execution_context.get("fresh_live_quotes_ready") else "BLOCK",
            "detail": f"quote_mode={execution_context.get('quote_mode')}",
            "affected_rows": len(decision_board),
        },
        {
            "gate": "quote_freshness",
            "status": "INFO",
            "detail": (
                "execution_blocker=false; "
                "target_refresh_before_order=true; "
                f"quote_mode={execution_context.get('quote_mode')}"
            ),
            "affected_rows": len(decision_board),
        },
        {
            "gate": "portfolio_sizing",
            "status": "PASS" if execution_context.get("portfolio_ready") else "BLOCK",
            "detail": f"portfolio_status={execution_context.get('portfolio_status')}; total_value={execution_context.get('portfolio_total_value')}",
            "affected_rows": len(decision_board),
        },
        {
            "gate": "agentic_reviews",
            "status": "PASS" if execution_context.get("agentic_reviews_ready") else "BLOCK",
            "detail": (
                f"external_reviews={execution_context.get('external_review_count')}; "
                f"review_agents={execution_context.get('external_review_agent_count')}; "
                f"dispatch_lanes={execution_context.get('agent_dispatch_task_count')}; "
                f"coverage_basis={execution_context.get('agentic_review_coverage_basis')}; "
                f"coverage={execution_context.get('agentic_review_coverage_pct')}; "
                f"required={execution_context.get('min_agentic_review_coverage_pct')}; "
                f"research_tasks={execution_context.get('research_task_count')}; "
                f"broad_universe_coverage={execution_context.get('broad_review_coverage_pct')}"
            ),
            "affected_rows": len(decision_board),
        },
        {
            "gate": "ready_trade_tickets",
            "status": "PASS" if ready_rows > 0 else "BLOCK",
            "detail": f"ready_to_enter_rows={ready_rows}",
            "affected_rows": ready_rows,
        },
    ]
    if not decision_board.empty and "execution_blockers" in decision_board.columns:
        blocker_counts: dict[str, int] = {}
        for text in decision_board["execution_blockers"].astype(str).tolist():
            for blocker in [part.strip() for part in text.split(";") if part.strip() and part.strip().lower() != "nan"]:
                blocker_counts[blocker] = blocker_counts.get(blocker, 0) + 1
        for blocker, count in sorted(blocker_counts.items(), key=lambda item: (-item[1], item[0])):
            rows.append({"gate": f"row_blocker:{blocker}", "status": "INFO", "detail": blocker, "affected_rows": count})
    return pd.DataFrame(rows, columns=columns)


def summarize_execution_readiness(execution_readiness: pd.DataFrame) -> dict[str, Any]:
    if execution_readiness.empty:
        return {"status": "unknown", "blocking_gates": []}
    blocking = execution_readiness[execution_readiness["status"].astype(str).str.upper().eq("BLOCK")]
    blocking_gates = blocking["gate"].astype(str).tolist()
    only_no_orders = blocking_gates == ["ready_trade_tickets"]
    status = "gates_pass_no_send_now_orders" if only_no_orders else "not_execution_ready" if blocking_gates else "execution_ready"
    return {
        "status": status,
        "blocking_gates": blocking_gates,
    }


def build_expectancy_evidence(root: Path, decision_board: pd.DataFrame, trade_tickets: pd.DataFrame) -> pd.DataFrame:
    """Build a non-v4 evidence ledger for expectancy and monthly-target claims."""

    root = Path(root).expanduser().resolve()
    out_root = root if root.name == "out" else root / "out"
    current_tickers = _current_ticket_tickers(decision_board, trade_tickets)
    current_strategy_families = _current_ticket_strategy_families(decision_board, trade_tickets)
    current_strategy_families_by_ticker = _current_ticket_strategy_families_by_ticker(decision_board, trade_tickets)
    closed_trades_path = _closed_trades_evidence_path(root, out_root)
    rows: list[dict[str, Any]] = []

    rows.append(
        _expectancy_from_outcome_csv(
            out_root / "codexuw_execute_outcome_ledger.csv",
            source="codexuw_execute_outcome_ledger",
            evidence_type="forward_realized_outcomes",
            current_tickers=current_tickers,
        )
    )
    rows.extend(
        _expectancy_by_ticker_from_outcome_csv(
            out_root / "codexuw_execute_outcome_ledger.csv",
            source="codexuw_execute_outcome_ledger_by_ticker",
            evidence_type="forward_realized_outcomes_by_ticker",
            current_tickers=current_tickers,
        )
    )
    rows.append(
        _expectancy_from_outcome_csv(
            out_root / "codexuw_recommendation_outcome_ledger.csv",
            source="codexuw_recommendation_outcome_ledger",
            evidence_type="forward_recommendation_outcomes",
            current_tickers=current_tickers,
        )
    )
    rows.extend(_expectancy_from_replay_history(out_root, current_tickers))
    rows.append(_expectancy_from_closed_trades(closed_trades_path, current_tickers))
    rows.extend(_expectancy_by_ticker_from_closed_trades(closed_trades_path, current_tickers))
    rows.extend(_expectancy_by_ticker_strategy_from_closed_trades(closed_trades_path, current_strategy_families_by_ticker))
    rows.append(_expectancy_from_closed_trades_strategy_cohort(closed_trades_path, current_strategy_families))
    rows.append(_expectancy_summary_row(rows))
    return pd.DataFrame(rows, columns=EXPECTANCY_EVIDENCE_COLUMNS)


def build_strategy_outcome_atlas(root: Path, decision_board: pd.DataFrame, trade_tickets: pd.DataFrame) -> pd.DataFrame:
    """Summarize actual closed-trade outcomes by strategy family and current ticket strategy."""

    root = Path(root).expanduser().resolve()
    out_root = root if root.name == "out" else root / "out"
    closed_path = _closed_trades_evidence_path(root, out_root)
    current_tickers = _current_ticket_tickers(decision_board, trade_tickets)
    current_families = _current_ticket_strategy_families(decision_board, trade_tickets)
    current_by_ticker = _current_ticket_strategy_families_by_ticker(decision_board, trade_tickets)
    current_green_by_family = _current_green_count_by_strategy(trade_tickets)
    current_ticket_count_by_family = _current_ticket_count_by_strategy(decision_board, trade_tickets)

    if not _safe_non_v4_path(closed_path) or not closed_path.exists():
        return pd.DataFrame(
            [
                {
                    "scope": "source",
                    "ticker": "",
                    "strategy_family": "",
                    "status": "BLOCK",
                    "sample_size": 0,
                    "win_rate": "",
                    "avg_pnl": "",
                    "total_pnl": "",
                    "profit_factor": "",
                    "max_drawdown": "",
                    "source_tickers": "",
                    "current_ticket_count": 0,
                    "current_green_count": 0,
                    "suggested_action": "collect_closed_trade_outcomes",
                    "source_path": str(closed_path),
                    "note": "Actual Schwab closed-trade ledger is missing; strategy outcome atlas cannot prove profitability.",
                }
            ],
            columns=STRATEGY_OUTCOME_ATLAS_COLUMNS,
        )
    closed, error = _read_closed_trades_frame(closed_path)
    if error or closed.empty or not {"ticker", "realized_pnl", "strategy"}.issubset(closed.columns):
        return pd.DataFrame(
            [
                {
                    "scope": "source",
                    "ticker": "",
                    "strategy_family": "",
                    "status": "BLOCK",
                    "sample_size": 0,
                    "win_rate": "",
                    "avg_pnl": "",
                    "total_pnl": "",
                    "profit_factor": "",
                    "max_drawdown": "",
                    "source_tickers": "",
                    "current_ticket_count": 0,
                    "current_green_count": 0,
                    "suggested_action": "repair_closed_trade_outcome_source",
                    "source_path": str(closed_path),
                    "note": f"Actual Schwab closed-trade ledger is unusable: {error or 'missing ticker/strategy/realized_pnl columns'}.",
                }
            ],
            columns=STRATEGY_OUTCOME_ATLAS_COLUMNS,
        )

    frame = closed.copy()
    frame["ticker"] = frame["ticker"].astype(str).str.strip().str.upper()
    frame["canonical_ticker"] = frame["ticker"].map(canonical_ticker_key)
    frame["strategy_family"] = frame["strategy"].map(_normal_strategy_family)
    frame["realized_pnl"] = pd.to_numeric(frame["realized_pnl"], errors="coerce")
    frame = frame[
        frame["realized_pnl"].notna()
        & frame["canonical_ticker"].astype(str).ne("")
        & frame["strategy_family"].astype(str).str.strip().ne("")
    ].copy()
    rows: list[dict[str, Any]] = []
    for family, group in frame.groupby("strategy_family"):
        source_tickers = _ticker_set_from_frame(group)
        rows.append(
            _strategy_outcome_row(
                scope="strategy_family",
                ticker="",
                strategy_family=str(family),
                path=closed_path,
                pnl_values=group["realized_pnl"],
                source_tickers=source_tickers,
                current_ticket_count=current_ticket_count_by_family.get(str(family), 0),
                current_green_count=current_green_by_family.get(str(family), 0),
                status_func=_expectancy_status,
                current_family=str(family) in current_families,
                note=(
                    f"Actual Schwab closed-trade cohort for {family}. "
                    "Positive broad strategy evidence can suggest what to research next; green rows require positive strategy evidence, live validation, sizing, and any configured ticker-specific gate."
                ),
            )
        )

    ticker_series = frame["canonical_ticker"].astype(str)
    for ticker in sorted(current_by_ticker):
        key = canonical_ticker_key(ticker)
        for family in sorted(current_by_ticker.get(ticker) or set()):
            scoped = frame[ticker_series.eq(key) & frame["strategy_family"].astype(str).eq(family)].copy()
            if scoped.empty:
                rows.append(
                    {
                        "scope": "current_ticker_strategy",
                        "ticker": ticker,
                        "strategy_family": family,
                        "status": "BLOCK",
                        "sample_size": 0,
                        "win_rate": "",
                        "avg_pnl": "",
                        "total_pnl": "",
                        "profit_factor": "",
                        "max_drawdown": "",
                        "source_tickers": "",
                        "current_ticket_count": _current_ticket_count_for_ticker_strategy(decision_board, trade_tickets, ticker, family),
                        "current_green_count": _current_green_count_for_ticker_strategy(trade_tickets, ticker, family),
                        "suggested_action": "keep_watch_only_until_ticker_strategy_outcomes_exist",
                        "source_path": str(closed_path),
                        "note": f"No actual closed Schwab outcomes found for current {ticker} {family} tickets.",
                    }
                )
                continue
            rows.append(
                _strategy_outcome_row(
                    scope="current_ticker_strategy",
                    ticker=ticker,
                    strategy_family=family,
                    path=closed_path,
                    pnl_values=scoped["realized_pnl"],
                    source_tickers=_ticker_set_from_frame(scoped),
                    current_ticket_count=_current_ticket_count_for_ticker_strategy(decision_board, trade_tickets, ticker, family),
                    current_green_count=_current_green_count_for_ticker_strategy(trade_tickets, ticker, family),
                    status_func=_ticker_expectancy_status,
                    current_family=True,
                    note=(
                        f"Actual Schwab closed-trade support for current {ticker} {family}. "
                        "This row is the ticket-scoped evidence gate for green promotion."
                    ),
                )
            )

    if not rows:
        return pd.DataFrame(columns=STRATEGY_OUTCOME_ATLAS_COLUMNS)
    out = pd.DataFrame(rows, columns=STRATEGY_OUTCOME_ATLAS_COLUMNS)
    scope_rank = {"current_ticker_strategy": 0, "strategy_family": 1, "source": 2}
    status_rank = {"PASS": 0, "WARN": 1, "BLOCK": 2}
    out["__scope_rank"] = out["scope"].map(lambda value: scope_rank.get(str(value), 9))
    out["__current_ticket_count"] = pd.to_numeric(out["current_ticket_count"], errors="coerce").fillna(0)
    out["__status_rank"] = out["status"].map(lambda value: status_rank.get(str(value), 9))
    out["__sample"] = pd.to_numeric(out["sample_size"], errors="coerce").fillna(0)
    out = out.sort_values(
        ["__scope_rank", "__current_ticket_count", "__status_rank", "__sample", "strategy_family", "ticker"],
        ascending=[True, False, True, False, True, True],
        kind="mergesort",
    )
    return out[STRATEGY_OUTCOME_ATLAS_COLUMNS].reset_index(drop=True)


def summarize_strategy_outcome_atlas(strategy_outcome_atlas: pd.DataFrame) -> dict[str, Any]:
    if strategy_outcome_atlas is None or strategy_outcome_atlas.empty:
        return {
            "status": "missing",
            "positive_strategy_families": [],
            "negative_current_strategy_families": [],
            "blocking_current_ticker_strategy_rows": 0,
        }
    atlas = strategy_outcome_atlas.copy()
    family_rows = atlas[atlas["scope"].astype(str).eq("strategy_family")]
    current_rows = atlas[atlas["scope"].astype(str).eq("current_ticker_strategy")]
    positive = family_rows[family_rows["status"].astype(str).str.upper().eq("PASS")]["strategy_family"].astype(str).tolist()
    current_negative = family_rows[
        family_rows["strategy_family"].astype(str).isin(_current_strategy_families_from_atlas(current_rows))
        & family_rows["status"].astype(str).str.upper().eq("BLOCK")
    ]["strategy_family"].astype(str).tolist()
    blocking_current_rows = current_rows[current_rows["status"].astype(str).str.upper().ne("PASS")]
    return {
        "status": "strategy_outcomes_positive_for_some_families"
        if positive
        else "no_positive_strategy_family_evidence",
        "positive_strategy_families": sorted(set(positive)),
        "negative_current_strategy_families": sorted(set(current_negative)),
        "blocking_current_ticker_strategy_rows": int(len(blocking_current_rows)),
        "strategy_family_rows": int(len(family_rows)),
        "current_ticker_strategy_rows": int(len(current_rows)),
    }


def _strategy_outcome_row(
    *,
    scope: str,
    ticker: str,
    strategy_family: str,
    path: Path,
    pnl_values: pd.Series,
    source_tickers: set[str],
    current_ticket_count: int,
    current_green_count: int,
    status_func: Any,
    current_family: bool,
    note: str,
) -> dict[str, Any]:
    metrics = _expectancy_metrics_row(
        "schwab_closed_trades_strategy_outcome_atlas",
        path,
        scope,
        pd.to_numeric(pnl_values, errors="coerce"),
        tickers=source_tickers,
        current_tickers={ticker} if ticker else set(),
        open_or_unrealized_count=0,
        note=note,
        status_override_func=status_func,
    )
    status = str(metrics.get("status") or "").upper()
    sample = int(metrics.get("sample_size") or 0)
    if status == "PASS":
        action = (
            "eligible_for_research_but_green_requires_ticker_strategy_proof"
            if scope == "strategy_family"
            else "eligible_for_green_expectancy_gate_if_all_execution_gates_pass"
        )
    elif status == "WARN":
        action = "collect_more_closed_outcomes_before_green_promotion"
    elif current_family and sample >= MIN_EXPECTANCY_SAMPLE_SIZE:
        action = "do_not_promote_current_strategy_family"
    else:
        action = "keep_watch_only_until_positive_outcomes_exist"
    return {
        "scope": scope,
        "ticker": ticker,
        "strategy_family": strategy_family,
        "status": status,
        "sample_size": metrics.get("sample_size", 0),
        "win_rate": metrics.get("win_rate", ""),
        "avg_pnl": metrics.get("avg_pnl", ""),
        "total_pnl": metrics.get("total_pnl", ""),
        "profit_factor": metrics.get("profit_factor", ""),
        "max_drawdown": metrics.get("max_drawdown", ""),
        "source_tickers": ", ".join(sorted(source_tickers)),
        "current_ticket_count": int(current_ticket_count),
        "current_green_count": int(current_green_count),
        "suggested_action": action,
        "source_path": str(path),
        "note": note,
    }


def _current_strategy_families_from_atlas(current_rows: pd.DataFrame) -> set[str]:
    if current_rows is None or current_rows.empty or "strategy_family" not in current_rows.columns:
        return set()
    return {str(value).strip() for value in current_rows["strategy_family"].dropna().tolist() if str(value).strip()}


def _current_ticket_count_by_strategy(decision_board: pd.DataFrame, trade_tickets: pd.DataFrame) -> dict[str, int]:
    frame = trade_tickets if trade_tickets is not None and not trade_tickets.empty else decision_board
    return _count_rows_by_strategy(frame)


def _current_green_count_by_strategy(trade_tickets: pd.DataFrame) -> dict[str, int]:
    if trade_tickets is None or trade_tickets.empty:
        return {}
    ready = trade_tickets[trade_tickets.get("ready_to_enter", pd.Series(False, index=trade_tickets.index)).map(_truthy)].copy()
    return _count_rows_by_strategy(ready)


def _count_rows_by_strategy(frame: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    if frame is None or frame.empty:
        return counts
    for _, row in frame.iterrows():
        family = _strategy_family_from_ticket_row(row)
        if family:
            counts[family] = counts.get(family, 0) + 1
    return counts


def _current_ticket_count_for_ticker_strategy(
    decision_board: pd.DataFrame,
    trade_tickets: pd.DataFrame,
    ticker: str,
    family: str,
) -> int:
    frame = trade_tickets if trade_tickets is not None and not trade_tickets.empty else decision_board
    return _count_rows_for_ticker_strategy(frame, ticker, family)


def _current_green_count_for_ticker_strategy(trade_tickets: pd.DataFrame, ticker: str, family: str) -> int:
    if trade_tickets is None or trade_tickets.empty:
        return 0
    ready = trade_tickets[trade_tickets.get("ready_to_enter", pd.Series(False, index=trade_tickets.index)).map(_truthy)].copy()
    return _count_rows_for_ticker_strategy(ready, ticker, family)


def _count_rows_for_ticker_strategy(frame: pd.DataFrame, ticker: str, family: str) -> int:
    if frame is None or frame.empty:
        return 0
    wanted_ticker = canonical_ticker_key(ticker)
    count = 0
    for _, row in frame.iterrows():
        if canonical_ticker_key(_mapping_get(row, "ticker")) == wanted_ticker and _strategy_family_from_ticket_row(row) == family:
            count += 1
    return count


def _closed_trades_evidence_path(root: Path, out_root: Path) -> Path:
    candidates = [out_root / "schwab_pull_state" / "closed_trades_acct_3326.jsonl"]
    project = project_root().expanduser().resolve()
    try:
        root.relative_to(project)
    except ValueError:
        return candidates[0]
    project_closed = project / "out" / "schwab_pull_state" / "closed_trades_acct_3326.jsonl"
    if project_closed not in candidates:
        candidates.append(project_closed)
    for candidate in candidates:
        if _safe_non_v4_path(candidate) and candidate.exists():
            return candidate
    return candidates[0]


def _raw_orders_evidence_path(root: Path, out_root: Path) -> Path:
    candidates = [out_root / "schwab_pull_state" / "raw_orders_acct_3326.jsonl"]
    project = project_root().expanduser().resolve()
    try:
        root.relative_to(project)
    except ValueError:
        return candidates[0]
    project_orders = project / "out" / "schwab_pull_state" / "raw_orders_acct_3326.jsonl"
    if project_orders not in candidates:
        candidates.append(project_orders)
    for candidate in candidates:
        if _safe_non_v4_path(candidate) and candidate.exists():
            return candidate
    return candidates[0]


def _raw_order_strategy_route_lookup(root: Path, out_root: Path) -> dict[str, str]:
    return {
        order_id: _as_text(meta.get("strategy_route"))
        for order_id, meta in _raw_order_calibration_lookup(root, out_root).items()
        if _as_text(meta.get("strategy_route"))
    }


def _raw_order_calibration_lookup(root: Path, out_root: Path) -> dict[str, dict[str, Any]]:
    path = _raw_orders_evidence_path(root, out_root)
    if not _safe_non_v4_path(path) or not path.exists():
        return {}
    lookup: dict[str, dict[str, Any]] = {}
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                continue
            order_id = _as_text(payload.get("orderId") or payload.get("order_id") or payload.get("id"))
            if not order_id:
                continue
            meta = _schwab_order_calibration_metadata(payload)
            if meta:
                lookup[order_id] = meta
    except Exception:
        return {}
    return lookup


def _entry_order_id_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [_as_text(item) for item in value if _as_text(item)]
    text = _as_text(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text.replace("'", '"'))
            if isinstance(parsed, list):
                return [_as_text(item) for item in parsed if _as_text(item)]
        except Exception:
            pass
    return [part.strip() for part in text.replace(";", ",").split(",") if part.strip()]


def _route_from_entry_order_ids(value: Any, order_routes: Mapping[str, str]) -> str:
    if not order_routes:
        return ""
    routes = [order_routes.get(order_id) for order_id in _entry_order_id_list(value) if order_routes.get(order_id)]
    routes = [route for route in routes if route]
    if not routes:
        return ""
    counts: dict[str, int] = {}
    for route in routes:
        counts[route] = counts.get(route, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _order_calibration_from_entry_order_ids(value: Any, order_meta: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if not order_meta:
        return {}
    metas = [order_meta[order_id] for order_id in _entry_order_id_list(value) if order_id in order_meta]
    metas = [meta for meta in metas if meta]
    if not metas:
        return {}
    if len(metas) == 1:
        return dict(metas[0])
    route_counts: dict[str, int] = {}
    for meta in metas:
        route = _as_text(meta.get("strategy_route"))
        if route:
            route_counts[route] = route_counts.get(route, 0) + 1
    preferred_route = sorted(route_counts.items(), key=lambda item: (-item[1], item[0]))[0][0] if route_counts else ""
    for meta in metas:
        if preferred_route and _as_text(meta.get("strategy_route")) == preferred_route:
            return dict(meta)
    return dict(metas[0])


def _schwab_order_calibration_metadata(order: Mapping[str, Any]) -> dict[str, Any]:
    route = _strategy_route_from_schwab_order(order)
    legs = [
        leg
        for leg in order.get("orderLegCollection", []) or []
        if isinstance(leg, Mapping) and _as_text(leg.get("orderLegType")).upper() == "OPTION"
    ]
    opening_legs = [
        leg
        for leg in legs
        if _as_text(leg.get("positionEffect")).upper() in {"OPENING", "OPEN"}
        or _as_text(leg.get("instruction")).upper().endswith("_TO_OPEN")
    ]
    legs = opening_legs or legs
    entry_type = _entry_type_from_schwab_order(order, route=route, legs=legs)
    entry_limit = _as_float(order.get("price"))
    spread_width = _spread_width_from_schwab_legs(legs)
    direction_bucket = _direction_bucket_from_row({}, route)
    meta: dict[str, Any] = {
        "strategy_route": route,
        "entry_type": entry_type,
        "entry_limit": entry_limit,
        "spread_width": spread_width,
    }
    if direction_bucket != "neutral_or_unknown":
        meta["direction_bucket"] = direction_bucket
    if entry_type == "CREDIT":
        meta["entry_credit"] = entry_limit
        if entry_limit is not None and spread_width and spread_width > 0:
            meta["entry_credit_pct_width"] = entry_limit / spread_width
    elif entry_type == "DEBIT":
        meta["entry_ask"] = entry_limit
        if entry_limit is not None and spread_width and spread_width > 0:
            meta["entry_debit_pct_width"] = entry_limit / spread_width
            meta["reward_risk"] = (spread_width - entry_limit) / entry_limit if entry_limit > 0 else None
    if entry_type:
        meta["economics_bucket"] = _economics_bucket(meta, entry_type)
    return {key: value for key, value in meta.items() if value not in ("", None)}


def _entry_type_from_schwab_order(
    order: Mapping[str, Any],
    *,
    route: str,
    legs: Sequence[Mapping[str, Any]],
) -> str:
    order_type = _as_text(order.get("orderType")).upper()
    if "CREDIT" in order_type:
        return "CREDIT"
    if "DEBIT" in order_type:
        return "DEBIT"
    instructions = [_as_text(leg.get("instruction")).upper() for leg in legs]
    if instructions and all(instruction.startswith("SELL") for instruction in instructions):
        return "CREDIT"
    if instructions and all(instruction.startswith("BUY") for instruction in instructions):
        return "DEBIT"
    return _entry_type_from_route(route)


def _entry_type_from_route(route: str) -> str:
    route = _as_text(route)
    if route in {"short_put", "short_call", "bull_put_credit", "bear_call_credit"}:
        return "CREDIT"
    if route in {"long_call", "long_put", "bull_call_debit", "bear_put_debit"}:
        return "DEBIT"
    return ""


def _spread_width_from_schwab_legs(legs: Sequence[Mapping[str, Any]]) -> Optional[float]:
    strikes = [_option_leg_strike(leg) for leg in legs]
    strikes = [strike for strike in strikes if strike is not None]
    if len(strikes) < 2:
        return None
    return abs(max(strikes) - min(strikes))


def _strategy_route_from_schwab_order(order: Mapping[str, Any]) -> str:
    legs_all = [
        leg
        for leg in order.get("orderLegCollection", []) or []
        if isinstance(leg, Mapping) and _as_text(leg.get("orderLegType")).upper() == "OPTION"
    ]
    if not legs_all:
        return ""
    opening_legs = [
        leg
        for leg in legs_all
        if _as_text(leg.get("positionEffect")).upper() in {"OPENING", "OPEN"}
        or _as_text(leg.get("instruction")).upper().endswith("_TO_OPEN")
    ]
    legs = opening_legs or legs_all
    if len(legs) == 1:
        return _single_option_route_from_leg(legs[0])
    if len(legs) != 2:
        return ""
    rights = {
        _as_text(_mapping_get(_mapping_get(leg, "instrument") or {}, "putCall")).upper()
        for leg in legs
    }
    order_type = _as_text(order.get("orderType")).upper()
    if rights == {"PUT"}:
        if "CREDIT" in order_type:
            return "bull_put_credit"
        if "DEBIT" in order_type:
            return "bear_put_debit"
        return _put_vertical_route_from_strikes(legs)
    if rights == {"CALL"}:
        if "CREDIT" in order_type:
            return "bear_call_credit"
        if "DEBIT" in order_type:
            return "bull_call_debit"
        return _call_vertical_route_from_strikes(legs)
    return "vertical_spread"


def _single_option_route_from_leg(leg: Mapping[str, Any]) -> str:
    instruction = _as_text(leg.get("instruction")).upper()
    instrument = _mapping_get(leg, "instrument") or {}
    right = _as_text(_mapping_get(instrument, "putCall")).upper()
    if instruction.startswith("SELL") and right == "PUT":
        return "short_put"
    if instruction.startswith("SELL") and right == "CALL":
        return "short_call"
    if instruction.startswith("BUY") and right == "PUT":
        return "long_put"
    if instruction.startswith("BUY") and right == "CALL":
        return "long_call"
    return ""


def _put_vertical_route_from_strikes(legs: Sequence[Mapping[str, Any]]) -> str:
    buy = _option_leg_strike_by_instruction(legs, "BUY")
    sell = _option_leg_strike_by_instruction(legs, "SELL")
    if buy is None or sell is None:
        return "vertical_spread"
    return "bear_put_debit" if buy > sell else "bull_put_credit"


def _call_vertical_route_from_strikes(legs: Sequence[Mapping[str, Any]]) -> str:
    buy = _option_leg_strike_by_instruction(legs, "BUY")
    sell = _option_leg_strike_by_instruction(legs, "SELL")
    if buy is None or sell is None:
        return "vertical_spread"
    return "bull_call_debit" if buy < sell else "bear_call_credit"


def _option_leg_strike_by_instruction(legs: Sequence[Mapping[str, Any]], instruction_prefix: str) -> Optional[float]:
    prefix = instruction_prefix.upper()
    for leg in legs:
        if _as_text(leg.get("instruction")).upper().startswith(prefix):
            return _option_leg_strike(leg)
    return None


def _option_leg_strike(leg: Mapping[str, Any]) -> Optional[float]:
    instrument = _mapping_get(leg, "instrument") or {}
    symbol = _as_text(_mapping_get(instrument, "symbol"))
    match = re.search(r"([CP])(\d{8})$", symbol.replace(" ", ""))
    if match:
        return float(int(match.group(2)) / 1000.0)
    description = _as_text(_mapping_get(instrument, "description"))
    match = re.search(r"\$(\d+(?:\.\d+)?)\s+(?:Put|Call)\b", description, flags=re.IGNORECASE)
    if match:
        return _as_float(match.group(1))
    return None


def canonical_ticker_key(value: Any) -> str:
    ticker = str(value or "").strip().upper()
    return TICKER_CANONICAL_GROUPS.get(ticker, ticker)


def tickers_match(left: Any, right: Any) -> bool:
    left_key = canonical_ticker_key(left)
    right_key = canonical_ticker_key(right)
    return bool(left_key and right_key and left_key == right_key)


def _canonical_ticker_set(values: Iterable[Any]) -> set[str]:
    return {canonical_ticker_key(value) for value in values if canonical_ticker_key(value)}


def _matched_current_tickers(tickers: set[str], current_tickers: set[str]) -> list[str]:
    evidence_keys = _canonical_ticker_set(tickers)
    return sorted(
        {
            current
            for current in current_tickers
            if canonical_ticker_key(current) in evidence_keys
        }
    )


def _actual_forward_outcome_frame(root: Path, out_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    order_meta = _raw_order_calibration_lookup(root, out_root)
    order_routes = {
        order_id: _as_text(meta.get("strategy_route"))
        for order_id, meta in order_meta.items()
        if _as_text(meta.get("strategy_route"))
    }
    execute_path = out_root / "codexuw_execute_outcome_ledger.csv"
    if _safe_non_v4_path(execute_path) and execute_path.exists():
        try:
            execute = pd.read_csv(execute_path)
            if {"ticker", "realized_pnl"}.issubset(execute.columns):
                columns = ["ticker", "realized_pnl"]
                if "strategy" in execute.columns:
                    columns.append("strategy")
                if "entry_order_ids" in execute.columns:
                    columns.append("entry_order_ids")
                if "opened_at" in execute.columns:
                    columns.append("opened_at")
                if "expiry" in execute.columns:
                    columns.append("expiry")
                if "regime" in execute.columns:
                    columns.append("regime")
                if "direction" in execute.columns:
                    columns.append("direction")
                frame = execute[columns].copy()
                if "strategy" not in frame.columns:
                    frame["strategy"] = ""
                if "entry_order_ids" not in frame.columns:
                    frame["entry_order_ids"] = ""
                if "opened_at" not in frame.columns:
                    frame["opened_at"] = ""
                if "expiry" not in frame.columns:
                    frame["expiry"] = ""
                if "regime" not in frame.columns:
                    frame["regime"] = ""
                if "direction" not in frame.columns:
                    frame["direction"] = ""
                frames.append(frame.assign(source="codexuw_execute_outcome_ledger"))
        except Exception:
            pass
    closed_path = _closed_trades_evidence_path(root, out_root)
    if _safe_non_v4_path(closed_path) and closed_path.exists():
        closed, error = _read_closed_trades_frame(closed_path)
        if not error and {"ticker", "realized_pnl"}.issubset(closed.columns):
            columns = ["ticker", "realized_pnl"]
            if "strategy" in closed.columns:
                columns.append("strategy")
            if "entry_order_ids" in closed.columns:
                columns.append("entry_order_ids")
            if "opened_at" in closed.columns:
                columns.append("opened_at")
            if "expiry" in closed.columns:
                columns.append("expiry")
            if "regime" in closed.columns:
                columns.append("regime")
            if "direction" in closed.columns:
                columns.append("direction")
            frame = closed[columns].copy()
            if "strategy" not in frame.columns:
                frame["strategy"] = ""
            if "entry_order_ids" not in frame.columns:
                frame["entry_order_ids"] = ""
            if "opened_at" not in frame.columns:
                frame["opened_at"] = ""
            if "expiry" not in frame.columns:
                frame["expiry"] = ""
            if "regime" not in frame.columns:
                frame["regime"] = ""
            if "direction" not in frame.columns:
                frame["direction"] = ""
            frames.append(frame.assign(source="schwab_closed_trades"))
    if not frames:
        return pd.DataFrame(columns=["ticker", "realized_pnl", "source"])
    combined = pd.concat(frames, ignore_index=True)
    combined["ticker"] = combined["ticker"].astype(str).str.strip().str.upper()
    combined["canonical_ticker"] = combined["ticker"].map(canonical_ticker_key)
    combined["__order_calibration"] = combined["entry_order_ids"].map(
        lambda value: _order_calibration_from_entry_order_ids(value, order_meta)
    )
    combined["strategy_route"] = combined.apply(
        lambda row: _as_text(_mapping_get(row.get("__order_calibration") or {}, "strategy_route"))
        or _route_from_entry_order_ids(row.get("entry_order_ids"), order_routes)
        or _strategy_route_from_text(row.get("strategy")),
        axis=1,
    )
    combined["strategy_family"] = combined["strategy_route"].map(_strategy_family_from_route).mask(
        combined["strategy_route"].astype(str).str.strip().eq(""),
        combined.get("strategy", pd.Series("", index=combined.index)).map(_normal_strategy_family),
    )
    combined["entry_type"] = combined.apply(
        lambda row: _as_text(_mapping_get(row.get("__order_calibration") or {}, "entry_type"))
        or _entry_type_from_route(row.get("strategy_route")),
        axis=1,
    )
    combined["direction_bucket"] = combined.apply(
        lambda row: (
            _as_text(_mapping_get(row.get("__order_calibration") or {}, "direction_bucket"))
            if _as_text(_mapping_get(row.get("__order_calibration") or {}, "direction_bucket"))
            and _as_text(_mapping_get(row.get("__order_calibration") or {}, "direction_bucket")) != "neutral_or_unknown"
            else _direction_bucket_from_row(row, row.get("strategy_route"))
        ),
        axis=1,
    )
    for column in ["entry_limit", "entry_credit", "entry_ask", "spread_width", "entry_credit_pct_width", "entry_debit_pct_width", "reward_risk"]:
        combined[column] = combined["__order_calibration"].map(lambda meta: _mapping_get(meta or {}, column))
        combined[column] = pd.to_numeric(combined[column], errors="coerce")
    combined["dte"] = combined.apply(_actual_outcome_dte, axis=1)
    combined["dte_bucket"] = combined["dte"].map(_dte_bucket)
    existing_regime = combined.get("regime", pd.Series("", index=combined.index)).map(_regime_bucket)
    regime_history = _market_regime_history_by_date(out_root)
    opened_regime = combined.get("opened_at", pd.Series("", index=combined.index)).map(
        lambda value: regime_history.get(_date_key(_parse_optional_date_value(value)), "")
    )
    combined["regime"] = existing_regime.where(existing_regime.ne("regime_unknown"), opened_regime.map(_regime_bucket)).map(
        _regime_bucket
    )
    combined["iv_rank_bucket"] = "iv_unknown"
    combined["economics_bucket"] = combined.apply(
        lambda row: _as_text(_mapping_get(row.get("__order_calibration") or {}, "economics_bucket"))
        or _economics_bucket(row, row.get("entry_type")),
        axis=1,
    )
    combined["liquidity_bucket"] = "liquidity_unknown"
    combined["realized_pnl"] = pd.to_numeric(combined["realized_pnl"], errors="coerce")
    combined = combined.drop(columns=["__order_calibration"], errors="ignore")
    return combined[combined["realized_pnl"].notna() & combined["canonical_ticker"].astype(str).ne("")].copy()


def _market_regime_history_by_date(out_root: Path) -> dict[str, str]:
    agent_root = Path(out_root) / "options_agent"
    if not agent_root.exists():
        return {}
    history: dict[str, str] = {}
    for path in sorted(agent_root.glob("*/market_regime.json")):
        if not _safe_non_v4_path(path):
            continue
        day = _date_from_path_text(path.parent.name)
        if day is None:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
        if not isinstance(payload, Mapping):
            continue
        regime = _regime_bucket(
            payload.get("regime")
            or payload.get("market_regime")
            or payload.get("tape_direction")
            or payload.get("direction")
        )
        if regime != "regime_unknown":
            history.setdefault(day.isoformat(), regime)
    return history


def _date_from_path_text(value: Any) -> Optional[dt.date]:
    match = re.search(r"(\d{4}-\d{2}-\d{2})", _as_text(value))
    if not match:
        return None
    try:
        return parse_as_of(match.group(1))
    except Exception:
        return None


def _date_key(value: Optional[dt.date]) -> str:
    return value.isoformat() if value is not None else ""


def _actual_outcome_dte(row: Mapping[str, Any]) -> Optional[int]:
    opened = _parse_optional_date_value(_mapping_get(row, "opened_at"))
    expiry = _parse_optional_date_value(_mapping_get(row, "expiry"))
    if opened is None or expiry is None:
        return None
    return max((expiry - opened).days, 0)


def _actual_forward_metrics_by_canonical_ticker(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if frame.empty or "canonical_ticker" not in frame.columns:
        return {}
    metrics: dict[str, dict[str, Any]] = {}
    for key, group in frame.groupby("canonical_ticker"):
        pnl = pd.to_numeric(group.get("realized_pnl", pd.Series(dtype=float)), errors="coerce").dropna()
        if pnl.empty:
            continue
        sample = int(len(pnl))
        win_rate = float((pnl > 0).mean())
        avg_pnl = float(pnl.mean())
        wins = float(pnl[pnl > 0].sum())
        losses = abs(float(pnl[pnl < 0].sum()))
        profit_factor = wins / losses if losses > 0 else (math.inf if wins > 0 else math.nan)
        status = _ticker_expectancy_status(sample, win_rate, avg_pnl, profit_factor)
        source_tickers = sorted({str(value).strip().upper() for value in group["ticker"].dropna().tolist() if str(value).strip()})
        metrics[str(key)] = {
            "status": status,
            "sample_size": sample,
            "win_rate": _round_or_blank(win_rate, 4),
            "avg_pnl": _round_or_blank(avg_pnl, 2),
            "profit_factor": _round_or_blank(profit_factor, 3),
            "source_tickers": ", ".join(source_tickers),
            "note": (
                f"Per-ticker actual/forward realized support from {', '.join(source_tickers)}: "
                f"sample={sample}, avg_pnl={_round_or_blank(avg_pnl, 2)}, "
                f"win_rate={_round_or_blank(win_rate, 4)}, profit_factor={_round_or_blank(profit_factor, 3)}."
            ),
        }
    return metrics


def _actual_forward_metrics_by_canonical_ticker_strategy(frame: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    if frame.empty or not {"canonical_ticker", "strategy_family"}.issubset(frame.columns):
        return {}
    scoped = frame[frame["strategy_family"].astype(str).str.strip().ne("")].copy()
    if scoped.empty:
        return {}
    metrics: dict[tuple[str, str], dict[str, Any]] = {}
    for (key, family), group in scoped.groupby(["canonical_ticker", "strategy_family"]):
        pnl = pd.to_numeric(group.get("realized_pnl", pd.Series(dtype=float)), errors="coerce").dropna()
        if pnl.empty:
            continue
        sample = int(len(pnl))
        win_rate = float((pnl > 0).mean())
        avg_pnl = float(pnl.mean())
        wins = float(pnl[pnl > 0].sum())
        losses = abs(float(pnl[pnl < 0].sum()))
        profit_factor = wins / losses if losses > 0 else (math.inf if wins > 0 else math.nan)
        status = _ticker_expectancy_status(sample, win_rate, avg_pnl, profit_factor)
        source_tickers = sorted({str(value).strip().upper() for value in group["ticker"].dropna().tolist() if str(value).strip()})
        family_text = str(family)
        metrics[(str(key), family_text)] = {
            "status": status,
            "sample_size": sample,
            "win_rate": _round_or_blank(win_rate, 4),
            "avg_pnl": _round_or_blank(avg_pnl, 2),
            "profit_factor": _round_or_blank(profit_factor, 3),
            "source_tickers": ", ".join(source_tickers),
            "strategy_family": family_text,
            "note": (
                f"Structure-aligned actual/forward realized support from {', '.join(source_tickers)} "
                f"for {family_text}: sample={sample}, avg_pnl={_round_or_blank(avg_pnl, 2)}, "
                f"win_rate={_round_or_blank(win_rate, 4)}, profit_factor={_round_or_blank(profit_factor, 3)}."
            ),
        }
    return metrics


def _actual_forward_metrics_by_strategy_family(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if frame.empty or "strategy_family" not in frame.columns:
        return {}
    scoped = frame[frame["strategy_family"].astype(str).str.strip().ne("")].copy()
    if scoped.empty:
        return {}
    metrics: dict[str, dict[str, Any]] = {}
    for family, group in scoped.groupby("strategy_family"):
        pnl = pd.to_numeric(group.get("realized_pnl", pd.Series(dtype=float)), errors="coerce").dropna()
        if pnl.empty:
            continue
        sample = int(len(pnl))
        win_rate = float((pnl > 0).mean())
        avg_pnl = float(pnl.mean())
        wins = float(pnl[pnl > 0].sum())
        losses = abs(float(pnl[pnl < 0].sum()))
        profit_factor = wins / losses if losses > 0 else (math.inf if wins > 0 else math.nan)
        status = _expectancy_status(sample, win_rate, avg_pnl, profit_factor)
        source_tickers = sorted({str(value).strip().upper() for value in group["ticker"].dropna().tolist() if str(value).strip()})
        family_text = str(family)
        metrics[family_text] = {
            "status": status,
            "sample_size": sample,
            "win_rate": _round_or_blank(win_rate, 4),
            "avg_pnl": _round_or_blank(avg_pnl, 2),
            "profit_factor": _round_or_blank(profit_factor, 3),
            "source_tickers": ", ".join(source_tickers),
            "strategy_family": family_text,
            "note": (
                f"Family-level actual/forward realized support from {', '.join(source_tickers)} "
                f"for {family_text}: sample={sample}, avg_pnl={_round_or_blank(avg_pnl, 2)}, "
                f"win_rate={_round_or_blank(win_rate, 4)}, profit_factor={_round_or_blank(profit_factor, 3)}."
            ),
        }
    return metrics


def _family_level_strategy_fallback_allowed(
    family: str,
    family_metrics: Optional[Mapping[str, Any]],
    ticker_metrics: Optional[Mapping[str, Any]],
) -> bool:
    if family not in FAMILY_LEVEL_STRATEGY_EXPECTANCY_FALLBACKS or not family_metrics:
        return False
    if _as_text(family_metrics.get("status")).upper() != "PASS":
        return False
    sample = int(_as_float(family_metrics.get("sample_size")) or 0)
    if sample < MIN_EXPECTANCY_SAMPLE_SIZE:
        return False
    if ticker_metrics and _metrics_are_negative(ticker_metrics, min_sample=MIN_TICKER_EXPECTANCY_SAMPLE_SIZE):
        return False
    return True


def _negative_family_strategy_metrics_available(family_metrics: Optional[Mapping[str, Any]]) -> bool:
    if not family_metrics:
        return False
    if _as_text(family_metrics.get("status")).upper() != "BLOCK":
        return False
    return _metrics_are_negative(family_metrics, min_sample=MIN_EXPECTANCY_SAMPLE_SIZE)


def _metrics_are_negative(metrics: Mapping[str, Any], *, min_sample: int) -> bool:
    sample = int(_as_float(metrics.get("sample_size")) or 0)
    if sample < min_sample:
        return False
    avg_pnl = _as_float(metrics.get("avg_pnl"))
    profit_factor = _as_float(metrics.get("profit_factor"))
    win_rate = _as_float(metrics.get("win_rate"))
    return bool(
        (avg_pnl is not None and avg_pnl < 0)
        or (profit_factor is not None and profit_factor < 1.0)
        or (win_rate is not None and win_rate < MIN_TICKER_EXPECTANCY_WIN_RATE)
    )


def _current_ticket_tickers(decision_board: pd.DataFrame, trade_tickets: pd.DataFrame) -> set[str]:
    if trade_tickets is not None and not trade_tickets.empty and "ticker" in trade_tickets.columns:
        ticket_tickers = {
            str(value).strip().upper()
            for value in trade_tickets["ticker"].dropna().tolist()
            if str(value).strip()
        }
        if ticket_tickers:
            return ticket_tickers
    if decision_board is None or decision_board.empty or "ticker" not in decision_board.columns:
        return set()
    return {
        str(value).strip().upper()
        for value in decision_board["ticker"].dropna().tolist()
        if str(value).strip()
    }


def _current_ticket_strategy_families(decision_board: pd.DataFrame, trade_tickets: pd.DataFrame) -> set[str]:
    frame = trade_tickets if trade_tickets is not None and not trade_tickets.empty else decision_board
    if frame is None or frame.empty:
        return set()
    families: set[str] = set()
    for _, row in frame.iterrows():
        family = _strategy_family_from_ticket_row(row)
        if family:
            families.add(family)
    return families


def _current_ticket_strategy_families_by_ticker(
    decision_board: pd.DataFrame,
    trade_tickets: pd.DataFrame,
) -> dict[str, set[str]]:
    frame = trade_tickets if trade_tickets is not None and not trade_tickets.empty else decision_board
    if frame is None or frame.empty or "ticker" not in frame.columns:
        return {}
    by_ticker: dict[str, set[str]] = {}
    for _, row in frame.iterrows():
        ticker = _as_text(row.get("ticker")).upper()
        family = _strategy_family_from_ticket_row(row)
        if ticker and family:
            by_ticker.setdefault(ticker, set()).add(family)
    return by_ticker


def _normal_strategy_family(value: Any) -> str:
    text = _as_text(value).lower().strip().replace("-", "_").replace(" ", "_")
    aliases = {
        "call_credit_spread": "vertical_spread",
        "put_credit_spread": "vertical_spread",
        "bear_call_spread": "vertical_spread",
        "bull_put_spread": "vertical_spread",
        "bull_call_spread": "vertical_spread",
        "bear_put_spread": "vertical_spread",
        "bull_put_credit": "vertical_spread",
        "bear_call_credit": "vertical_spread",
        "bull_call_debit": "vertical_spread",
        "bear_put_debit": "vertical_spread",
        "bull_call_debit_spread": "vertical_spread",
        "bear_put_debit_spread": "vertical_spread",
        "call_debit_spread": "vertical_spread",
        "put_debit_spread": "vertical_spread",
        "vertical": "vertical_spread",
        "verticals": "vertical_spread",
        "credit_spread": "vertical_spread",
        "debit_spread": "vertical_spread",
        "cash_secured_put": "short_put",
        "covered_call": "short_call",
    }
    return aliases.get(text, text)


def _strategy_family_from_ticket_row(row: Mapping[str, Any] | pd.Series) -> str:
    explicit = _as_text(_mapping_get(row, "strategy") or _mapping_get(row, "structure")).lower()
    trade_plan = _as_text(_mapping_get(row, "trade_plan") or _mapping_get(row, "full_ticket")).lower()
    text = " ".join(part for part in [explicit, trade_plan] if part)
    if "spread" in text or (" buy " in f" {text} " and " sell " in f" {text} "):
        return "vertical_spread"
    if "short put" in text or ("sell" in text and " put" in text):
        return "short_put"
    if "short call" in text or ("sell" in text and " call" in text):
        return "short_call"
    if "long put" in text or ("buy" in text and " put" in text):
        return "long_put"
    if "long call" in text or ("buy" in text and " call" in text):
        return "long_call"
    return ""


def _ready_to_enter_count(decision_board: pd.DataFrame) -> int:
    if decision_board.empty or "ready_to_enter" not in decision_board.columns:
        return 0
    return int(decision_board["ready_to_enter"].map(_truthy).sum())


def _target_order_candidate_count(decision_board: pd.DataFrame) -> int:
    if decision_board.empty or "target_order_status" not in decision_board.columns:
        return 0
    return int(
        decision_board["target_order_status"]
        .astype(str)
        .str.lower()
        .isin(["target_order_candidate", "target_order_wait_for_price"])
        .sum()
    )


def _expectancy_from_outcome_csv(
    path: Path,
    *,
    source: str,
    evidence_type: str,
    current_tickers: set[str],
) -> dict[str, Any]:
    if not _safe_non_v4_path(path):
        return _expectancy_missing_row(source, path, evidence_type, "v4-derived source intentionally ignored")
    if not path.exists():
        return _expectancy_missing_row(source, path, evidence_type, "source missing")
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        return _expectancy_missing_row(source, path, evidence_type, f"source unreadable: {exc}")
    current_df = _filter_frame_to_tickers(df, current_tickers)
    pnl = _numeric_frame_column(current_df, "realized_pnl")
    realized = current_df[pnl.notna()].copy()
    if realized.empty:
        return _expectancy_metrics_row(
            source,
            path,
            evidence_type,
            pd.Series(dtype=float),
            tickers=_ticker_set_from_frame(current_df),
            current_tickers=current_tickers,
            open_or_unrealized_count=len(current_df),
            note=f"{source} exists, but has no realized P/L rows for the visible current ticket set yet.",
        )
    return _expectancy_metrics_row(
        source,
        path,
        evidence_type,
        pd.to_numeric(realized["realized_pnl"], errors="coerce"),
        tickers=_ticker_set_from_frame(realized),
        current_tickers=current_tickers,
        open_or_unrealized_count=max(0, len(current_df) - len(realized)),
        note="Forward realized outcome ledger for visible current tickets.",
    )


def _expectancy_by_ticker_from_outcome_csv(
    path: Path,
    *,
    source: str,
    evidence_type: str,
    current_tickers: set[str],
) -> list[dict[str, Any]]:
    if not _safe_non_v4_path(path) or not path.exists() or not current_tickers:
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    if not {"ticker", "realized_pnl"}.issubset(df.columns):
        return []
    current_df = _filter_frame_to_tickers(df, current_tickers)
    pnl = _numeric_frame_column(current_df, "realized_pnl")
    realized = current_df[pnl.notna()].copy()
    return _expectancy_by_current_ticker_rows(
        realized,
        path=path,
        source=source,
        evidence_type=evidence_type,
        current_tickers=current_tickers,
        note_prefix="Per-ticker forward realized outcome ledger",
    )


def _expectancy_from_replay_history(out_root: Path, current_tickers: set[str]) -> list[dict[str, Any]]:
    paths = [
        path
        for path in sorted(out_root.glob("codexuw_*backtest*/codexuw_replay_detail.csv"), key=lambda item: item.stat().st_mtime, reverse=True)
        if _safe_non_v4_path(path)
    ]
    if not paths:
        missing = out_root / "codexuw_*backtest*/codexuw_replay_detail.csv"
        return [_expectancy_missing_row("codexuw_replay_history", missing, "replay_backtest", "no replay detail source found")]
    path = paths[0]
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        return [_expectancy_missing_row("codexuw_replay_history", path, "replay_backtest", f"source unreadable: {exc}")]

    pnl = _numeric_frame_column(df, "pnl_1x")
    exact = df[pnl.notna()].copy()
    if "exact_evaluated" in exact.columns:
        exact = exact[exact["exact_evaluated"].map(_truthy)].copy()
    decision_pass = exact[exact.get("decision_pass", pd.Series(False, index=exact.index)).map(_truthy)].copy()
    decision_pass_current = _filter_frame_to_tickers(decision_pass, current_tickers)
    return [
        _expectancy_metrics_row(
            "codexuw_replay_all_exact",
            path,
            "replay_backtest_all_exact",
            pd.to_numeric(exact.get("pnl_1x", pd.Series(dtype=float)), errors="coerce"),
            tickers=_ticker_set_from_frame(exact),
            current_tickers=current_tickers,
            open_or_unrealized_count=max(0, len(df) - len(exact)),
            note="Broad exact replay history. Negative broad evidence blocks blanket profit claims.",
        ),
        _expectancy_metrics_row(
            "codexuw_replay_decision_pass",
            path,
            "replay_backtest_decision_pass",
            pd.to_numeric(decision_pass_current.get("pnl_1x", pd.Series(dtype=float)), errors="coerce"),
            tickers=_ticker_set_from_frame(decision_pass_current),
            current_tickers=current_tickers,
            open_or_unrealized_count=max(0, len(exact) - len(decision_pass)),
            note="Narrow decision-pass replay slice for visible current tickets; useful evidence, not live proof by itself.",
        ),
    ]


def _expectancy_from_closed_trades(path: Path, current_tickers: set[str]) -> dict[str, Any]:
    if not _safe_non_v4_path(path):
        return _expectancy_missing_row("schwab_closed_trades", path, "actual_closed_trades", "v4-derived source intentionally ignored")
    if not path.exists():
        return _expectancy_missing_row("schwab_closed_trades", path, "actual_closed_trades", "source missing")
    df, error = _read_closed_trades_frame(path)
    if error:
        return _expectancy_missing_row("schwab_closed_trades", path, "actual_closed_trades", f"source unreadable: {error}")
    current_df = _filter_frame_to_tickers(df, current_tickers)
    if current_df.empty:
        return _expectancy_metrics_row(
            "schwab_closed_trades",
            path,
            "actual_closed_trades",
            pd.Series(dtype=float),
            tickers=set(),
            current_tickers=current_tickers,
            open_or_unrealized_count=0,
            note="Closed-trade ledger has no rows for the visible current ticket set.",
        )
    return _expectancy_metrics_row(
        "schwab_closed_trades",
        path,
        "actual_closed_trades",
        pd.to_numeric(current_df.get("realized_pnl", pd.Series(dtype=float)), errors="coerce"),
        tickers=_ticker_set_from_frame(current_df),
        current_tickers=current_tickers,
        open_or_unrealized_count=0,
        note="Actual closed Schwab trade history for visible current tickets.",
    )


def _expectancy_by_ticker_from_closed_trades(path: Path, current_tickers: set[str]) -> list[dict[str, Any]]:
    if not _safe_non_v4_path(path) or not path.exists() or not current_tickers:
        return []
    df, error = _read_closed_trades_frame(path)
    if error or df.empty or not {"ticker", "realized_pnl"}.issubset(df.columns):
        return []
    current_df = _filter_frame_to_tickers(df, current_tickers)
    return _expectancy_by_current_ticker_rows(
        current_df,
        path=path,
        source="schwab_closed_trades_by_ticker",
        evidence_type="actual_closed_trades_by_ticker",
        current_tickers=current_tickers,
        note_prefix="Per-ticker actual closed Schwab trade history",
    )


def _expectancy_by_ticker_strategy_from_closed_trades(
    path: Path,
    current_strategy_families_by_ticker: Mapping[str, set[str]],
) -> list[dict[str, Any]]:
    if not _safe_non_v4_path(path) or not path.exists() or not current_strategy_families_by_ticker:
        return []
    df, error = _read_closed_trades_frame(path)
    if error or df.empty or not {"ticker", "realized_pnl", "strategy"}.issubset(df.columns):
        return []
    rows: list[dict[str, Any]] = []
    ticker_series = df["ticker"].astype(str).str.strip().str.upper()
    strategy_series = df["strategy"].map(_normal_strategy_family)
    current_tickers = set(current_strategy_families_by_ticker)
    for current in sorted(current_tickers):
        key = canonical_ticker_key(current)
        for family in sorted(current_strategy_families_by_ticker.get(current) or set()):
            scoped = df[
                ticker_series.map(canonical_ticker_key).eq(key)
                & strategy_series.astype(str).eq(family)
            ].copy()
            if scoped.empty:
                continue
            source_tickers = _ticker_set_from_frame(scoped)
            rows.append(
                _expectancy_metrics_row(
                    "schwab_closed_trades_by_ticker_strategy",
                    path,
                    "actual_closed_trades_by_ticker_strategy",
                    pd.to_numeric(scoped.get("realized_pnl", pd.Series(dtype=float)), errors="coerce"),
                    tickers={current},
                    current_tickers=current_tickers,
                    open_or_unrealized_count=0,
                    note=(
                        f"Structure-aligned actual closed Schwab trade history for {current} {family}; "
                        f"source tickers: {', '.join(sorted(source_tickers))}."
                    ),
                    status_override_func=_ticker_expectancy_status,
                )
            )
    return rows


def _expectancy_by_current_ticker_rows(
    frame: pd.DataFrame,
    *,
    path: Path,
    source: str,
    evidence_type: str,
    current_tickers: set[str],
    note_prefix: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if frame is None or frame.empty or "ticker" not in frame.columns:
        return rows
    ticker_series = frame["ticker"].astype(str).str.strip().str.upper()
    for current in sorted(current_tickers):
        key = canonical_ticker_key(current)
        scoped = frame[ticker_series.map(canonical_ticker_key).eq(key)].copy()
        if scoped.empty:
            continue
        source_tickers = _ticker_set_from_frame(scoped)
        rows.append(
            _expectancy_metrics_row(
                source,
                path,
                evidence_type,
                pd.to_numeric(scoped.get("realized_pnl", pd.Series(dtype=float)), errors="coerce"),
                tickers={current},
                current_tickers=current_tickers,
                open_or_unrealized_count=0,
                note=f"{note_prefix} for {current}; source tickers: {', '.join(sorted(source_tickers))}.",
                status_override_func=_ticker_expectancy_status,
            )
        )
    return rows


def _expectancy_from_closed_trades_strategy_cohort(path: Path, current_strategy_families: set[str]) -> dict[str, Any]:
    source = "schwab_closed_trades_strategy_cohort"
    evidence_type = "actual_closed_trades_strategy_cohort"
    if not _safe_non_v4_path(path):
        return _expectancy_missing_row(source, path, evidence_type, "v4-derived source intentionally ignored")
    if not path.exists():
        return _expectancy_missing_row(source, path, evidence_type, "source missing")
    if not current_strategy_families:
        return _expectancy_missing_row(source, path, evidence_type, "no current ticket strategy family to compare")
    df, error = _read_closed_trades_frame(path)
    if error:
        return _expectancy_missing_row(source, path, evidence_type, f"source unreadable: {error}")
    if df.empty or "strategy" not in df.columns:
        return _expectancy_missing_row(source, path, evidence_type, "closed-trade ledger has no strategy column")
    normalized = df["strategy"].map(_normal_strategy_family)
    cohort = df[normalized.isin(current_strategy_families)].copy()
    if cohort.empty:
        return _expectancy_metrics_row(
            source,
            path,
            evidence_type,
            pd.Series(dtype=float),
            tickers=set(),
            current_tickers=set(),
            open_or_unrealized_count=0,
            note="Closed-trade ledger has no rows for current ticket strategy families: "
            + ", ".join(sorted(current_strategy_families)),
        )
    return _expectancy_metrics_row(
        source,
        path,
        evidence_type,
        pd.to_numeric(cohort.get("realized_pnl", pd.Series(dtype=float)), errors="coerce"),
        tickers=_ticker_set_from_frame(cohort),
        current_tickers=set(),
        open_or_unrealized_count=0,
        note=(
            "Actual closed Schwab history for current ticket strategy families: "
            + ", ".join(sorted(current_strategy_families))
            + ". This cohort can block broad/monthly claims when negative, but it does not by itself prove a green ticker."
        ),
    )


def _read_closed_trades_frame(path: Path) -> tuple[pd.DataFrame, str]:
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, Mapping):
                rows.append(dict(payload))
    except Exception as exc:
        return pd.DataFrame(), str(exc)
    return pd.DataFrame(rows), ""


def _expectancy_metrics_row(
    source: str,
    path: Path,
    evidence_type: str,
    pnl_values: pd.Series,
    *,
    tickers: set[str],
    current_tickers: set[str],
    open_or_unrealized_count: int,
    note: str,
    status_override_func: Any = None,
) -> dict[str, Any]:
    pnl = pd.to_numeric(pnl_values, errors="coerce").dropna().astype(float)
    sample = int(len(pnl))
    win_rate = float((pnl > 0).mean()) if sample else math.nan
    avg_pnl = float(pnl.mean()) if sample else math.nan
    total_pnl = float(pnl.sum()) if sample else math.nan
    wins = float(pnl[pnl > 0].sum()) if sample else 0.0
    losses = abs(float(pnl[pnl < 0].sum())) if sample else 0.0
    profit_factor = wins / losses if losses > 0 else (math.inf if wins > 0 else math.nan)
    max_drawdown = _series_max_drawdown(pnl) if sample else math.nan
    status_func = status_override_func or _expectancy_status
    status = status_func(sample, win_rate, avg_pnl, profit_factor)
    matched = _matched_current_tickers(tickers, current_tickers)
    return {
        "source": source,
        "source_path": str(path),
        "evidence_type": evidence_type,
        "status": status,
        "sample_size": sample,
        "win_rate": _round_or_blank(win_rate, 4),
        "avg_pnl": _round_or_blank(avg_pnl, 2),
        "total_pnl": _round_or_blank(total_pnl, 2),
        "profit_factor": _round_or_blank(profit_factor, 3),
        "max_drawdown": _round_or_blank(max_drawdown, 2),
        "matched_current_tickers": ", ".join(matched),
        "matched_current_count": len(matched),
        "open_or_unrealized_count": int(open_or_unrealized_count),
        "note": note,
    }


def _expectancy_missing_row(source: str, path: Path, evidence_type: str, note: str) -> dict[str, Any]:
    return {
        "source": source,
        "source_path": str(path),
        "evidence_type": evidence_type,
        "status": "BLOCK",
        "sample_size": 0,
        "win_rate": "",
        "avg_pnl": "",
        "total_pnl": "",
        "profit_factor": "",
        "max_drawdown": "",
        "matched_current_tickers": "",
        "matched_current_count": 0,
        "open_or_unrealized_count": 0,
        "note": note,
    }


def _expectancy_summary_row(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary_rows = [
        row
        for row in rows
        if str(row.get("evidence_type") or "")
        not in (TICKER_SCOPED_ACTUAL_FORWARD_TYPES | STRATEGY_SCOPED_ACTUAL_FORWARD_TYPES)
    ]
    actual_rows = [
        row
        for row in summary_rows
        if str(row.get("evidence_type") or "") in {"actual_closed_trades", "forward_realized_outcomes"}
    ]
    actual_claim_block_rows = [
        row
        for row in summary_rows
        if str(row.get("evidence_type") or "")
        in {"actual_closed_trades", "forward_realized_outcomes", "actual_closed_trades_strategy_cohort"}
    ]
    replay_pass_rows = [
        row
        for row in summary_rows
        if str(row.get("evidence_type") or "") == "replay_backtest_decision_pass"
        and str(row.get("status") or "").upper() == "PASS"
        and int(row.get("matched_current_count") or 0) > 0
    ]
    actual_pass = any(
        str(row.get("status") or "").upper() == "PASS" and int(row.get("matched_current_count") or 0) > 0
        for row in actual_rows
    )
    actual_negative = any(
        str(row.get("status") or "").upper() == "BLOCK" and int(row.get("sample_size") or 0) >= MIN_EXPECTANCY_SAMPLE_SIZE
        for row in actual_claim_block_rows
    )
    if actual_pass and replay_pass_rows:
        status = "PASS"
        note = "Actual closed/forward outcomes and replay decision-pass evidence are positive for current tickers."
    elif actual_negative:
        status = "BLOCK"
        note = "Actual closed-trade evidence is not positive enough; monthly-target evidence remains insufficient."
    elif replay_pass_rows:
        status = "WARN"
        note = "Replay decision-pass evidence is positive for current tickers, but live/closed Options Agent outcomes are still missing."
    else:
        status = "BLOCK"
        note = "No sufficient positive expectancy evidence is available."
    sample = sum(int(row.get("sample_size") or 0) for row in summary_rows)
    matched = sorted(
        {
            ticker.strip().upper()
            for row in rows
            for ticker in str(row.get("matched_current_tickers") or "").split(",")
            if ticker.strip()
        }
    )
    return {
        "source": "expectancy_summary",
        "source_path": "",
        "evidence_type": "summary",
        "status": status,
        "sample_size": sample,
        "win_rate": "",
        "avg_pnl": "",
        "total_pnl": "",
        "profit_factor": "",
        "max_drawdown": "",
        "matched_current_tickers": ", ".join(matched),
        "matched_current_count": len(matched),
        "open_or_unrealized_count": sum(int(row.get("open_or_unrealized_count") or 0) for row in summary_rows),
        "note": note,
    }


def summarize_expectancy_evidence(expectancy_evidence: pd.DataFrame) -> dict[str, Any]:
    if expectancy_evidence.empty:
        return {"status": "missing", "blocking_sources": [], "note": "expectancy evidence artifact is empty"}
    summary = expectancy_evidence[expectancy_evidence["source"].astype(str).eq("expectancy_summary")]
    if not summary.empty:
        row = summary.iloc[0]
        status = str(row.get("status") or "").upper()
        return {
            "status": "positive" if status == "PASS" else "mixed" if status == "WARN" else "not_proven",
            "summary_status": status,
            "blocking_sources": expectancy_evidence[
                expectancy_evidence["status"].astype(str).str.upper().eq("BLOCK")
            ]["source"].astype(str).tolist(),
            "sample_size": int(row.get("sample_size") or 0),
            "note": str(row.get("note") or ""),
        }
    blockers = expectancy_evidence[expectancy_evidence["status"].astype(str).str.upper().eq("BLOCK")]
    return {
        "status": "not_proven" if not blockers.empty else "positive",
        "blocking_sources": blockers["source"].astype(str).tolist(),
        "sample_size": int(pd.to_numeric(expectancy_evidence.get("sample_size", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
        "note": "",
    }


def _expectancy_status(sample: int, win_rate: float, avg_pnl: float, profit_factor: float) -> str:
    if sample <= 0:
        return "BLOCK"
    if sample < MIN_EXPECTANCY_SAMPLE_SIZE:
        return "WARN"
    if (
        math.isfinite(avg_pnl)
        and avg_pnl > 0
        and math.isfinite(win_rate)
        and win_rate >= MIN_EXPECTANCY_WIN_RATE
        and (math.isinf(profit_factor) or (math.isfinite(profit_factor) and profit_factor >= MIN_EXPECTANCY_PROFIT_FACTOR))
    ):
        return "PASS"
    return "BLOCK"


def _ticker_expectancy_status(sample: int, win_rate: float, avg_pnl: float, profit_factor: float) -> str:
    if sample <= 0:
        return "BLOCK"
    if sample < MIN_TICKER_EXPECTANCY_SAMPLE_SIZE:
        return "WARN"
    if (
        math.isfinite(avg_pnl)
        and avg_pnl > 0
        and math.isfinite(win_rate)
        and win_rate >= MIN_TICKER_EXPECTANCY_WIN_RATE
        and (
            math.isinf(profit_factor)
            or (math.isfinite(profit_factor) and profit_factor >= MIN_TICKER_EXPECTANCY_PROFIT_FACTOR)
        )
    ):
        return "PASS"
    return "BLOCK"


def _series_max_drawdown(values: pd.Series) -> float:
    total = 0.0
    peak = 0.0
    drawdown = 0.0
    for value in pd.to_numeric(values, errors="coerce").fillna(0.0).tolist():
        total += float(value)
        peak = max(peak, total)
        drawdown = min(drawdown, total - peak)
    return drawdown


def _round_or_blank(value: Any, digits: int) -> Any:
    try:
        number = float(value)
    except Exception:
        return ""
    if math.isnan(number):
        return ""
    if math.isinf(number):
        return "inf" if number > 0 else "-inf"
    return round(number, digits)


def _ticker_set_from_frame(frame: pd.DataFrame) -> set[str]:
    if frame is None or frame.empty or "ticker" not in frame.columns:
        return set()
    return {str(value).strip().upper() for value in frame["ticker"].dropna().tolist() if str(value).strip()}


def _filter_frame_to_tickers(frame: pd.DataFrame, tickers: set[str]) -> pd.DataFrame:
    if frame is None or frame.empty or not tickers or "ticker" not in frame.columns:
        return frame.copy() if frame is not None else pd.DataFrame()
    current_keys = _canonical_ticker_set(tickers)
    ticker_keys = frame["ticker"].astype(str).str.strip().str.upper().map(canonical_ticker_key)
    return frame[ticker_keys.isin(current_keys)].copy()


def _numeric_frame_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([math.nan] * len(frame), index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _safe_non_v4_path(path: Path) -> bool:
    lowered = str(path).lower()
    blocked = ("codexdaily" + "_v" + "4", "daily" + "_v" + "4")
    return all(token not in lowered for token in blocked)


def build_monthly_feasibility(
    decision_board: pd.DataFrame,
    trade_tickets: pd.DataFrame,
    execution_context: Mapping[str, Any],
    expectancy_evidence: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Estimate whether current executable candidates can plausibly support the monthly target."""

    columns = ["metric", "value", "status", "note"]
    tickets = trade_tickets.copy() if trade_tickets is not None else pd.DataFrame()
    ready = tickets[tickets.get("ready_to_enter", pd.Series(False, index=tickets.index)).map(_truthy)].copy()
    target_candidates = tickets[
        tickets.get("target_order_status", pd.Series("", index=tickets.index))
        .astype(str)
        .str.lower()
        .isin(["target_order_candidate", "target_order_wait_for_price"])
        & ~tickets.get("ready_to_enter", pd.Series(False, index=tickets.index)).map(_truthy)
    ].copy()
    max_profit = _position_sum(ready, "max_profit", "position_max_profit")
    max_loss = _position_sum(ready, "max_loss", "position_max_loss")
    target_max_profit = _position_sum(target_candidates, "max_profit", "position_max_profit")
    target_max_loss = _position_sum(target_candidates, "max_loss", "position_max_loss")
    target = float(execution_context.get("monthly_profit_target") or MONTHLY_PROFIT_TARGET)
    rows = [
        {"metric": "monthly_profit_target", "value": target, "status": "INFO", "note": "User target; not a guarantee."},
        {"metric": "ready_ticket_count", "value": int(len(ready)), "status": "PASS" if len(ready) else "BLOCK", "note": "Executable rows after all gates."},
        {"metric": "one_cycle_max_profit", "value": round(max_profit, 2), "status": "PASS" if max_profit > 0 else "BLOCK", "note": "Sum of max profit across current ready tickets."},
        {"metric": "one_cycle_max_loss", "value": round(max_loss, 2), "status": "INFO", "note": "Defined-risk max loss across current ready tickets."},
        {
            "metric": "target_order_candidate_count",
            "value": int(len(target_candidates)),
            "status": "INFO",
            "note": "Target-order candidates; not executable send-now capacity.",
        },
        {
            "metric": "target_order_candidate_max_profit",
            "value": round(target_max_profit, 2),
            "status": "INFO",
            "note": "Position-scaled max profit of target candidates if later revalidated live; excluded from executable monthly feasibility.",
        },
        {
            "metric": "target_order_candidate_max_loss",
            "value": round(target_max_loss, 2),
            "status": "INFO",
            "note": "Position-scaled defined-risk max loss of target candidates; excluded from executable monthly feasibility.",
        },
    ]
    cycles_needed = math.ceil(target / max_profit) if max_profit > 0 else math.inf
    feasible = math.isfinite(cycles_needed) and cycles_needed <= 4 and bool(execution_context.get("fresh_live_quotes_ready")) and bool(execution_context.get("portfolio_ready"))
    rows.append(
        {
            "metric": "cycles_needed_at_max_profit",
            "value": cycles_needed if math.isfinite(cycles_needed) else "",
            "status": "PASS" if feasible else "BLOCK",
            "note": "Requires actual fills and full max-profit capture; expectancy evidence still required.",
        }
    )
    evidence_summary = summarize_expectancy_evidence(expectancy_evidence if expectancy_evidence is not None else pd.DataFrame())
    evidence_positive = evidence_summary.get("summary_status") == "PASS"
    rows.append(
        {
            "metric": "expectancy_evidence",
            "value": evidence_summary.get("sample_size", ""),
            "status": "PASS" if evidence_positive else "BLOCK",
            "note": str(evidence_summary.get("note") or "Expectancy evidence is not sufficient."),
        }
    )
    ready_expectancy_status, ready_expectancy_value, ready_expectancy_note = _ready_ticket_expectancy_gate(
        ready,
        expectancy_evidence if expectancy_evidence is not None else pd.DataFrame(),
    )
    rows.append(
        {
            "metric": "ready_ticket_expectancy_evidence",
            "value": ready_expectancy_value,
            "status": ready_expectancy_status,
            "note": ready_expectancy_note,
        }
    )
    return pd.DataFrame(rows, columns=columns)


def _position_sum(frame: pd.DataFrame, one_lot_column: str, position_column: str) -> float:
    if frame is None or frame.empty:
        return 0.0
    if position_column in frame.columns:
        return float(pd.to_numeric(frame[position_column], errors="coerce").fillna(0.0).sum())
    one_lot = pd.to_numeric(frame.get(one_lot_column, pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    contracts = pd.to_numeric(frame.get("suggested_contracts", pd.Series(1, index=frame.index)), errors="coerce").fillna(1.0)
    return float((one_lot * contracts.clip(lower=0)).sum())


def _ready_ticket_expectancy_gate(ready: pd.DataFrame, expectancy_evidence: pd.DataFrame) -> tuple[str, int, str]:
    ready_tickers = _ticker_set_from_frame(ready)
    if not ready_tickers:
        return "BLOCK", 0, "No green ready_to_enter ticket tickers exist; monthly-target evidence remains insufficient."
    if expectancy_evidence is None or expectancy_evidence.empty:
        return "BLOCK", 0, "No expectancy evidence is available for green ready ticket tickers."
    evidence_type = expectancy_evidence.get("evidence_type", pd.Series("", index=expectancy_evidence.index)).astype(str)
    status = expectancy_evidence.get("status", pd.Series("", index=expectancy_evidence.index)).astype(str).str.upper()
    actual_forward_pass = expectancy_evidence[
        evidence_type.isin(GREEN_TICKET_EXPECTANCY_EVIDENCE_TYPES) & status.eq("PASS")
    ]
    supported = {
        ticker
        for value in actual_forward_pass.get("matched_current_tickers", pd.Series(dtype=object)).dropna().tolist()
        for ticker in _split_ticker_field(value)
    }
    supported_ready = sorted(ready_tickers & supported)
    missing = sorted(ready_tickers - supported)
    if missing:
        return (
            "BLOCK",
            len(supported_ready),
            "Green ready ticket tickers missing structure-aligned actual/forward expectancy support: " + ", ".join(missing),
        )
    return (
        "PASS",
        len(supported_ready),
        "Every green ready ticket ticker has PASS structure-aligned actual/forward expectancy support.",
    )


def _split_ticker_field(value: Any) -> set[str]:
    return {
        part.strip().upper()
        for part in str(value or "").replace(";", ",").split(",")
        if part.strip()
    }


def summarize_monthly_feasibility(monthly_feasibility: pd.DataFrame) -> dict[str, Any]:
    if monthly_feasibility.empty:
        return {"status": "unknown"}
    blockers = monthly_feasibility[monthly_feasibility["status"].astype(str).str.upper().eq("BLOCK")]
    pass_metrics = monthly_feasibility[monthly_feasibility["status"].astype(str).str.upper().eq("PASS")]
    return {
        "status": "not_proven" if not blockers.empty else "capacity_and_expectancy_positive_not_guaranteed",
        "blocking_metrics": blockers["metric"].astype(str).tolist(),
        "pass_metrics": pass_metrics["metric"].astype(str).tolist(),
        "note": (
            "Monthly-target evidence remains insufficient while any BLOCK metric remains."
            if not blockers.empty
            else "Ready-ticket capacity and expectancy evidence are positive, but the monthly target is still not guaranteed."
        ),
    }


CONFIDENCE_AUDIT_COLUMNS = [
    "metric",
    "rating",
    "threshold",
    "status",
    "sample_size",
    "evidence",
    "blockers",
    "required_next_action",
]


def build_confidence_audit(
    decision_board: pd.DataFrame,
    trade_tickets: pd.DataFrame,
    execution_readiness: pd.DataFrame,
    expectancy_evidence: pd.DataFrame,
    monthly_feasibility: pd.DataFrame,
    execution_context: Mapping[str, Any],
    profitability_calibration: Optional[pd.DataFrame] = None,
    execution_fill_quality: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Rate profitability/order-entry confidence from evidence, without relaxing gates."""

    profit_rating, profit_sample, profit_evidence, profit_blockers, profit_next = _profitability_confidence_rating(
        decision_board,
        trade_tickets,
        expectancy_evidence,
        monthly_feasibility,
        profitability_calibration,
    )
    entry_rating, entry_sample, entry_evidence, entry_blockers, entry_next = _order_entry_confidence_rating(
        decision_board,
        trade_tickets,
        execution_readiness,
        expectancy_evidence,
        execution_context,
        profitability_calibration,
        execution_fill_quality,
    )
    overall_rating = min(profit_rating, entry_rating)
    overall_blockers = _dedupe_notes([*profit_blockers, *entry_blockers])
    overall_status = (
        "PASS"
        if profit_rating >= MIN_GOAL_PROFITABILITY_CONFIDENCE_RATING
        and entry_rating >= MIN_GOAL_ORDER_ENTRY_CONFIDENCE_RATING
        else "BLOCK"
    )
    rows = [
        {
            "metric": "profitability_confidence_rating",
            "rating": profit_rating,
            "threshold": MIN_GOAL_PROFITABILITY_CONFIDENCE_RATING,
            "status": "PASS" if profit_rating >= MIN_GOAL_PROFITABILITY_CONFIDENCE_RATING else "BLOCK",
            "sample_size": profit_sample,
            "evidence": profit_evidence,
            "blockers": "; ".join(profit_blockers),
            "required_next_action": profit_next,
        },
        {
            "metric": "order_entry_confidence_rating",
            "rating": entry_rating,
            "threshold": MIN_GOAL_ORDER_ENTRY_CONFIDENCE_RATING,
            "status": "PASS" if entry_rating >= MIN_GOAL_ORDER_ENTRY_CONFIDENCE_RATING else "BLOCK",
            "sample_size": entry_sample,
            "evidence": entry_evidence,
            "blockers": "; ".join(entry_blockers),
            "required_next_action": entry_next,
        },
        {
            "metric": "goal_confidence_gate",
            "rating": overall_rating,
            "threshold": min(MIN_GOAL_PROFITABILITY_CONFIDENCE_RATING, MIN_GOAL_ORDER_ENTRY_CONFIDENCE_RATING),
            "status": overall_status,
            "sample_size": max(profit_sample, entry_sample),
            "evidence": (
                f"profitability={profit_rating}/10; order_entry={entry_rating}/10; "
                f"profitability_status={rows_status(profit_rating, MIN_GOAL_PROFITABILITY_CONFIDENCE_RATING)}; "
                f"order_entry_status={rows_status(entry_rating, MIN_GOAL_ORDER_ENTRY_CONFIDENCE_RATING)}"
            ),
            "blockers": "; ".join(overall_blockers),
            "required_next_action": (
                "Goal can be marked complete only when both ratings are >= 7/10 from evidence."
                if overall_status == "PASS"
                else "Do not loosen green gates; collect/prove positive closed/replay expectancy and produce validated green rows."
            ),
        },
    ]
    return pd.DataFrame(rows, columns=CONFIDENCE_AUDIT_COLUMNS)


def rows_status(rating: float, threshold: float) -> str:
    return "PASS" if rating >= threshold else "BLOCK"


def summarize_confidence_audit(confidence_audit: pd.DataFrame) -> dict[str, Any]:
    if confidence_audit is None or confidence_audit.empty:
        return {
            "status": "missing",
            "profitability_confidence_rating": 0.0,
            "order_entry_confidence_rating": 0.0,
            "blocking_metrics": ["confidence_audit_missing"],
        }
    by_metric = {
        str(row.get("metric") or ""): row
        for _, row in confidence_audit.iterrows()
    }
    profit = _as_float(by_metric.get("profitability_confidence_rating", {}).get("rating")) or 0.0
    entry = _as_float(by_metric.get("order_entry_confidence_rating", {}).get("rating")) or 0.0
    gate = by_metric.get("goal_confidence_gate", {})
    blockers = confidence_audit[confidence_audit["status"].astype(str).str.upper().ne("PASS")]
    return {
        "status": str(gate.get("status") or "UNKNOWN").lower(),
        "profitability_confidence_rating": profit,
        "order_entry_confidence_rating": entry,
        "goal_threshold": min(MIN_GOAL_PROFITABILITY_CONFIDENCE_RATING, MIN_GOAL_ORDER_ENTRY_CONFIDENCE_RATING),
        "blocking_metrics": blockers["metric"].astype(str).tolist(),
        "blockers": str(gate.get("blockers") or ""),
        "required_next_action": str(gate.get("required_next_action") or ""),
    }


def _profitability_confidence_rating(
    decision_board: pd.DataFrame,
    trade_tickets: pd.DataFrame,
    expectancy_evidence: pd.DataFrame,
    monthly_feasibility: pd.DataFrame,
    profitability_calibration: Optional[pd.DataFrame] = None,
) -> tuple[float, int, str, list[str], str]:
    blockers: list[str] = []
    evidence: list[str] = []
    rating = 1.0
    expectancy = expectancy_evidence.copy() if expectancy_evidence is not None else pd.DataFrame()
    summary = summarize_expectancy_evidence(expectancy)
    summary_status = str(summary.get("summary_status") or "").upper()
    summary_sample = int(_as_float(summary.get("sample_size")) or 0)
    evidence.append(f"expectancy_summary={summary_status or 'MISSING'} sample={summary_sample}")
    if summary_status == "PASS":
        rating += 2.0
    elif summary_status == "WARN":
        rating += 1.0
        blockers.append("expectancy_summary_warn_not_actual_proven")
    else:
        blockers.append("expectancy_summary_not_positive")

    actual_rows = _evidence_rows(expectancy, {"actual_closed_trades", "forward_realized_outcomes"})
    actual_positive = _positive_evidence_row(actual_rows, min_sample=MIN_EXPECTANCY_SAMPLE_SIZE)
    actual_partial = _positive_evidence_row(actual_rows, min_sample=1)
    if actual_positive:
        rating += 3.0
        evidence.append("actual_closed_or_forward_outcomes=PASS")
    elif actual_partial:
        rating += 1.5
        blockers.append("actual_closed_or_forward_outcomes_sample_too_small")
        evidence.append("actual_closed_or_forward_outcomes=partial_positive")
    else:
        blockers.append("actual_closed_or_forward_outcomes_not_positive")

    strategy_rows = _evidence_rows(expectancy, {"actual_closed_trades_strategy_cohort"})
    strategy_positive = _positive_evidence_row(
        strategy_rows[
            strategy_rows.get("status", pd.Series("", index=strategy_rows.index))
            .astype(str)
            .str.upper()
            .eq("PASS")
        ],
        min_sample=MIN_EXPECTANCY_SAMPLE_SIZE,
    )
    strategy_negative = _losing_evidence_row(strategy_rows, min_sample=MIN_EXPECTANCY_SAMPLE_SIZE)
    strategy_sampled = _sampled_evidence_row(strategy_rows, min_sample=MIN_EXPECTANCY_SAMPLE_SIZE)
    if strategy_positive:
        rating += 2.0
        evidence.append("current_strategy_cohort=PASS")
    elif strategy_negative:
        blockers.append("current_strategy_cohort_negative")
        evidence.append(_evidence_row_brief(strategy_negative, "current_strategy_cohort=negative"))
    elif strategy_sampled:
        blockers.append("current_strategy_cohort_weak_under_threshold")
        evidence.append(_evidence_row_brief(strategy_sampled, "current_strategy_cohort=weak_under_threshold"))
    else:
        blockers.append("current_strategy_cohort_not_proven")

    replay_rows = _evidence_rows(expectancy, {"replay_backtest_decision_pass"})
    replay_positive = _positive_evidence_row(replay_rows, min_sample=MIN_EXPECTANCY_SAMPLE_SIZE)
    replay_partial = _positive_evidence_row(replay_rows, min_sample=1)
    if replay_positive:
        rating += 1.5
        evidence.append("leakage_safe_replay_decision_pass=PASS")
    elif replay_partial:
        rating += 0.5
        blockers.append("replay_decision_pass_sample_too_small")
        evidence.append("leakage_safe_replay_decision_pass=partial_positive")
    else:
        blockers.append("replay_decision_pass_not_positive")

    calibration_summary = summarize_profitability_calibration(profitability_calibration) if profitability_calibration is not None else {}
    if profitability_calibration is not None:
        if calibration_summary.get("status") == "pass":
            rating += 1.0
            evidence.append("profitability_calibration=PASS")
        else:
            blockers.append("profitability_calibration_not_proven")
            if calibration_summary.get("current_trade_rows", 0):
                evidence.append(
                    "profitability_calibration="
                    + f"{calibration_summary.get('pass_rows', 0)} pass/"
                    + f"{calibration_summary.get('current_trade_rows', 0)} current rows"
                )

    ready = _ready_ticket_frame(trade_tickets)
    if ready.empty:
        blockers.append("no_green_ready_orders")
        rating = min(rating, 4.0)
    else:
        rating += 1.0
        ready_strategy_pass = ready[
            ready.get("actual_forward_strategy_expectancy_status", pd.Series("", index=ready.index))
            .astype(str)
            .str.upper()
            .eq("PASS")
        ]
        if len(ready_strategy_pass) == len(ready):
            rating += 1.0
            evidence.append("green_ticket_structure_expectancy=PASS")
        else:
            blockers.append("green_ticket_structure_expectancy_missing")
            rating = min(rating, 6.0)

    monthly = summarize_monthly_feasibility(monthly_feasibility if monthly_feasibility is not None else pd.DataFrame())
    if monthly.get("status") == "capacity_and_expectancy_positive_not_guaranteed":
        rating += 0.5
    else:
        blockers.append("monthly_feasibility_not_proven")

    if strategy_negative:
        rating = min(rating, 3.0)
    elif not strategy_positive:
        rating = min(rating, 6.0)
    if profitability_calibration is not None and calibration_summary.get("status") != "pass":
        rating = min(rating, 6.0)
    rating = round(max(0.0, min(10.0, rating)), 1)
    if rating >= MIN_GOAL_PROFITABILITY_CONFIDENCE_RATING:
        next_action = "Profitability evidence clears the goal threshold; keep green gates strict and monitor realized outcomes."
    elif strategy_negative:
        next_action = "Do not promote current strategy cohort; actual strategy P/L is negative or loss-dominated."
    elif not strategy_positive:
        next_action = "Do not promote current strategy cohort; prove positive PASS-level strategy expectancy from closed/forward outcomes."
    elif profitability_calibration is not None and calibration_summary.get("status") != "pass":
        next_action = "Prove current route/DTE/credit-debit buckets with positive actual outcomes and leakage-safe replay before increasing profitability confidence."
    elif ready.empty:
        next_action = "Produce green rows only after positive strategy expectancy, material profit, live validation, and sizing gates pass."
    else:
        next_action = "Collect more closed/forward outcomes and leakage-safe replay before increasing profitability confidence."
    return rating, summary_sample, "; ".join(_dedupe_notes(evidence)), _dedupe_notes(blockers), next_action


def _order_entry_confidence_rating(
    decision_board: pd.DataFrame,
    trade_tickets: pd.DataFrame,
    execution_readiness: pd.DataFrame,
    expectancy_evidence: pd.DataFrame,
    execution_context: Mapping[str, Any],
    profitability_calibration: Optional[pd.DataFrame] = None,
    execution_fill_quality: Optional[pd.DataFrame] = None,
) -> tuple[float, int, str, list[str], str]:
    ready = _ready_ticket_frame(trade_tickets)
    if ready.empty:
        return (
            0.0,
            0,
            "ready_to_enter_rows=0",
            ["no_green_ready_orders"],
            "No order-entry confidence is possible until a green ready_to_enter row exists.",
        )

    blockers: list[str] = []
    evidence: list[str] = [f"ready_to_enter_rows={len(ready)}"]
    rating = 2.0
    rating_cap = 10.0
    if bool(execution_context.get("fresh_live_quotes_ready")):
        rating += 1.5
        evidence.append("fresh_live_quotes_ready=true")
    else:
        blockers.append("fresh_live_quotes_not_ready")
    if bool(execution_context.get("portfolio_ready")):
        rating += 1.0
        evidence.append("portfolio_ready=true")
    else:
        blockers.append("portfolio_not_ready")
    if bool(execution_context.get("agentic_reviews_ready")):
        rating += 1.0
        evidence.append("agentic_reviews_ready=true")
    else:
        blockers.append("agentic_reviews_not_ready")

    live_pass = ready.get("live_validation_status", pd.Series("", index=ready.index)).astype(str).str.upper().eq("PASS")
    if bool(live_pass.all()):
        rating += 1.5
        evidence.append("green_live_validation=PASS")
    else:
        blockers.append("green_live_validation_not_all_pass")
        rating_cap = min(rating_cap, 4.0)

    positive_entry = ready.get("entry_limit", pd.Series(0.0, index=ready.index)).map(lambda value: (_as_float(value) or 0.0) > 0)
    positive_size = ready.get("suggested_contracts", pd.Series(0.0, index=ready.index)).map(lambda value: (_as_float(value) or 0.0) > 0)
    if bool(positive_entry.all() and positive_size.all()):
        rating += 1.0
        evidence.append("positive_entry_and_size=true")
    else:
        blockers.append("green_positive_entry_or_size_missing")
        rating_cap = min(rating_cap, 4.0)

    confidence_ok = ready.get("execution_confidence_rating", pd.Series("", index=ready.index)).astype(str).str.upper().isin({"MEDIUM", "HIGH"})
    quality_ok = ready.get("trade_quality_confidence_rating", pd.Series("", index=ready.index)).astype(str).str.upper().isin({"MEDIUM", "HIGH"})
    if bool(confidence_ok.all() and quality_ok.all()):
        rating += 1.0
        evidence.append("green_confidence_ratings=MEDIUM_OR_HIGH")
    else:
        blockers.append("green_confidence_rating_below_medium")
        rating_cap = min(rating_cap, 6.0)

    strategy_ok = ready.get("actual_forward_strategy_expectancy_status", pd.Series("", index=ready.index)).astype(str).str.upper().eq("PASS")
    if bool(strategy_ok.all()):
        rating += 1.0
        evidence.append("green_structure_expectancy=PASS")
    else:
        blockers.append("green_structure_expectancy_not_all_pass")
        rating_cap = min(rating_cap, 5.0)

    if "profitability_calibration_status" in ready.columns:
        calibration_ok = ready["profitability_calibration_status"].astype(str).str.upper().eq("PASS")
        if bool(calibration_ok.all()):
            rating += 1.0
            evidence.append("green_profitability_calibration=PASS")
        else:
            blockers.append("green_profitability_calibration_not_all_pass")
            rating_cap = min(rating_cap, 5.0)
    elif profitability_calibration is not None and not profitability_calibration.empty:
        blockers.append("green_profitability_calibration_missing_from_tickets")
        rating_cap = min(rating_cap, 5.0)

    if execution_fill_quality is not None:
        fill_ready = execution_fill_quality[
            execution_fill_quality.get("action_surface", pd.Series("", index=execution_fill_quality.index))
            .astype(str)
            .eq("green_send_now")
        ].copy()
        fill_ok = (
            not fill_ready.empty
            and len(fill_ready) == len(ready)
            and fill_ready.get("fill_quality_status", pd.Series("", index=fill_ready.index)).astype(str).str.upper().eq("PASS").all()
        )
        if bool(fill_ok):
            rating += 1.0
            evidence.append("green_execution_fill_quality=PASS")
        else:
            blockers.append("green_execution_fill_quality_not_all_pass")
            rating_cap = min(rating_cap, 6.0)

    readiness_summary = summarize_execution_readiness(execution_readiness if execution_readiness is not None else pd.DataFrame())
    if readiness_summary.get("status") == "execution_ready":
        rating += 1.0
        evidence.append("execution_readiness=PASS")
    else:
        blockers.extend(str(value) for value in readiness_summary.get("blocking_gates", []) if value)
        rating_cap = min(rating_cap, 6.0)

    expectancy_summary = summarize_expectancy_evidence(expectancy_evidence if expectancy_evidence is not None else pd.DataFrame())
    if expectancy_summary.get("summary_status") != "PASS":
        blockers.append("expectancy_summary_not_positive")
        rating_cap = min(rating_cap, 6.0)

    rating = round(max(0.0, min(10.0, rating, rating_cap)), 1)
    next_action = (
        "Order-entry evidence clears the goal threshold; keep live validation and expectancy gates mandatory."
        if rating >= MIN_GOAL_ORDER_ENTRY_CONFIDENCE_RATING
        else "Do not send orders until green rows have live validation, portfolio sizing, agentic lanes, confidence, and positive expectancy."
    )
    return rating, int(len(ready)), "; ".join(_dedupe_notes(evidence)), _dedupe_notes(blockers), next_action


def _ready_ticket_frame(trade_tickets: pd.DataFrame) -> pd.DataFrame:
    if trade_tickets is None or trade_tickets.empty:
        return pd.DataFrame()
    ready = trade_tickets.get("ready_to_enter", pd.Series(False, index=trade_tickets.index)).map(_truthy)
    return trade_tickets[ready].copy()


def _evidence_rows(expectancy_evidence: pd.DataFrame, evidence_types: set[str]) -> pd.DataFrame:
    if expectancy_evidence is None or expectancy_evidence.empty or "evidence_type" not in expectancy_evidence.columns:
        return pd.DataFrame()
    return expectancy_evidence[expectancy_evidence["evidence_type"].astype(str).isin(evidence_types)].copy()


def _positive_evidence_row(rows: pd.DataFrame, *, min_sample: int) -> Optional[dict[str, Any]]:
    if rows is None or rows.empty:
        return None
    for _, row in rows.iterrows():
        sample = int(_as_float(row.get("sample_size")) or 0)
        status = _as_text(row.get("status")).upper()
        avg_pnl = _as_float(row.get("avg_pnl"))
        profit_factor = _as_float(row.get("profit_factor"))
        if (
            sample >= min_sample
            and status in {"PASS", "WARN"}
            and avg_pnl is not None
            and avg_pnl > 0
            and (profit_factor is None or profit_factor >= 1.0)
        ):
            return row.to_dict()
    return None


def _negative_evidence_row(rows: pd.DataFrame, *, min_sample: int) -> Optional[dict[str, Any]]:
    if rows is None or rows.empty:
        return None
    for _, row in rows.iterrows():
        sample = int(_as_float(row.get("sample_size")) or 0)
        avg_pnl = _as_float(row.get("avg_pnl"))
        profit_factor = _as_float(row.get("profit_factor"))
        win_rate = _as_float(row.get("win_rate"))
        if sample >= min_sample and (
            (avg_pnl is not None and avg_pnl < 0)
            or (profit_factor is not None and profit_factor < 1.0)
            or (win_rate is not None and win_rate < MIN_EXPECTANCY_WIN_RATE)
        ):
            return row.to_dict()
    return None


def _losing_evidence_row(rows: pd.DataFrame, *, min_sample: int) -> Optional[dict[str, Any]]:
    if rows is None or rows.empty:
        return None
    for _, row in rows.iterrows():
        sample = int(_as_float(row.get("sample_size")) or 0)
        avg_pnl = _as_float(row.get("avg_pnl"))
        profit_factor = _as_float(row.get("profit_factor"))
        if sample >= min_sample and (
            (avg_pnl is not None and avg_pnl < 0)
            or (profit_factor is not None and profit_factor < 1.0)
        ):
            return row.to_dict()
    return None


def _sampled_evidence_row(rows: pd.DataFrame, *, min_sample: int) -> Optional[dict[str, Any]]:
    if rows is None or rows.empty:
        return None
    for _, row in rows.iterrows():
        sample = int(_as_float(row.get("sample_size")) or 0)
        if sample >= min_sample:
            return row.to_dict()
    return None


def _evidence_row_brief(row: Mapping[str, Any], prefix: str) -> str:
    return (
        f"{prefix} sample={row.get('sample_size', '')} win_rate={row.get('win_rate', '')} "
        f"avg_pnl={row.get('avg_pnl', '')} profit_factor={row.get('profit_factor', '')}"
    )


def write_agent_reviews(
    review_dir: Path,
    manifest: Mapping[str, Any],
    source_notes: Sequence[str],
    agent_review_board: Optional[pd.DataFrame] = None,
) -> None:
    """Write lightweight per-agent review notes for the v0 run."""

    notes = "\n".join(f"- {note}" for note in source_notes) if source_notes else "- no source warnings"
    bodies = {
        "data": f"# data\n\nSources inventoried directly from dated UW files.\n\n{notes}\n",
        "flow_oi": "# flow_oi\n\nTicker candidates ranked from screener, hot-chain, chain-OI, and bot-EOD aggregates.\n",
        "market_regime": "# market_regime\n\nRegime agent wrote `market_regime.json` from UW index flow rows.\n",
        "catalyst": "# catalyst\n\nCatalyst agent wrote `catalyst_evidence.csv` and `catalyst_reviews.csv` from UW earnings fields and local browser-text news captures.\n",
        "research_dispatch": "# research_dispatch\n\nResearch dispatch wrote `research_tasks.json` and normalized review rows into `agent_review_board.csv`.\n",
        "structure": "# structure\n\nConstructed first-pass routed option structures from dated hot-chain bid/ask quotes and wrote `strategy_routing_audit.csv` plus `structure_attempts.csv` to audit dated and live-chain construction outcomes.\n",
        "portfolio_risk": "# portfolio_risk\n\nPortfolio risk policy: annotate good trades; do not hide them.\n",
        "sizing": "# sizing\n\nSizing agent wrote `sizing_audit.csv`; sizing risk is annotated and does not hide otherwise valid trades.\n",
        "management": "# management\n\nManagement agent wrote `management_plan.csv` with entry, target-exit, invalidation, and review-trigger instructions.\n",
        "skeptic": "# skeptic\n\nObjective blockers remain blockers. Portfolio-only blockers are converted to annotations.\n",
        "synthesis": "# synthesis\n\nSynthesis agent wrote `decision_board.csv` and visible `trade_tickets.csv` plans; `ready_to_enter` separates actual order readiness from desired entry credit/debit.\n",
    }
    board = agent_review_board if agent_review_board is not None else pd.DataFrame(columns=AGENT_REVIEW_COLUMNS)
    if not board.empty:
        for agent_name, group in board.groupby("agent"):
            lines = bodies.get(str(agent_name), f"# {agent_name}\n\n")
            lines += "\n## Structured Reviews\n\n"
            for _, row in group.head(40).iterrows():
                lines += (
                    f"- {row.get('ticker', '')}: {row.get('verdict', '')}"
                    f" ({row.get('confidence', '')}) - {row.get('note', '')}\n"
                )
            bodies[str(agent_name)] = lines
    for agent in manifest["agents"]:
        name = agent["name"]
        (review_dir / f"{name}.md").write_text(bodies.get(name, f"# {name}\n\nNo review generated.\n"), encoding="utf-8")


def render_report(
    day: str,
    final: pd.DataFrame,
    no_trade: pd.DataFrame,
    manifest: Mapping[str, Any],
    coverage_audit: Optional[pd.DataFrame] = None,
) -> str:
    """Render the markdown report for the Options Agent run."""

    row_counts = manifest.get("row_counts", {}) or {}
    status_counts = manifest.get("status_counts", {}) or {}
    review_summary = manifest.get("agent_review_summary", {}) or {}
    agentic = manifest.get("agentic_orchestration", {}) or {}
    execution_context = manifest.get("execution_context", {}) or {}
    execution_summary = manifest.get("execution_readiness_summary", {}) or {}
    confidence_summary = manifest.get("confidence_audit_summary", {}) or {}
    calibrated_order_summary = manifest.get("calibrated_order_entry_blocker_summary", {}) or {}
    strategy_atlas_summary = manifest.get("strategy_outcome_atlas_summary", {}) or {}
    calibration_summary = manifest.get("profitability_calibration_summary", {}) or {}
    gap_plan_summary = manifest.get("profitability_gap_plan_summary", {}) or {}
    route_gap_summary = manifest.get("route_opportunity_gap_summary", {}) or {}
    live_quality_summary = manifest.get("live_spread_quality_summary", {}) or {}
    fill_quality_summary = manifest.get("execution_fill_quality_summary", {}) or {}
    warnings = manifest.get("warnings", []) or []
    lesson_pack_version = _as_text(manifest.get("lesson_pack_version") or manifest.get("lessonengine", {}).get("lesson_pack_version"))
    lesson_pack_digest = _as_text(manifest.get("lesson_pack_digest") or manifest.get("lessonengine", {}).get("lesson_pack_digest"))
    pipeline_version = str(manifest.get("pipeline_version") or PIPELINE_VERSION)
    live_mode = "live_schwab" if manifest.get("live_schwab_requested") else "snapshot" if manifest.get("chain_snapshot_dir") else "not_requested"
    mode_label = (
        f"independent UW + Schwab live research {pipeline_version}"
        if live_mode == "live_schwab"
        else f"independent UW research {pipeline_version}"
    )
    artifacts = manifest.get("artifacts", {}) or {}
    report_path = _as_text(artifacts.get("report"))
    trade_tickets_path = _as_text(artifacts.get("trade_tickets"))
    green_tickets_path = _as_text(artifacts.get("green_trade_tickets"))
    target_tickets_path = _as_text(artifacts.get("target_order_candidates"))
    confidence_audit_path = _as_text(artifacts.get("confidence_audit"))
    strategy_atlas_path = _as_text(artifacts.get("strategy_outcome_atlas"))
    profitability_calibration_path = _as_text(artifacts.get("profitability_calibration"))
    profitability_gap_plan_path = _as_text(artifacts.get("profitability_gap_plan"))
    execution_fill_quality_path = _as_text(artifacts.get("execution_fill_quality"))
    route_opportunity_gap_path = _as_text(artifacts.get("route_opportunity_gap"))
    green_count = int(_as_float(row_counts.get("green_trade_tickets")) or 0)
    target_count = int(_as_float(row_counts.get("target_order_ticket_rows")) or 0)
    refresh_count = int(_as_float(row_counts.get("market_open_recheck_queue")) or 0)
    blocking_gates = execution_summary.get("blocking_gates", [])
    calibration_blocker_detail = _profitability_calibration_blocker_detail(calibration_summary)
    bucket_examples_detail = _calibration_bucket_examples_detail(calibration_summary)
    gap_plan_detail = _profitability_gap_plan_detail(gap_plan_summary)
    calibrated_order_blocker_detail = _calibrated_order_entry_blocker_detail(calibrated_order_summary)
    route_gap_detail = _route_opportunity_gap_detail(route_gap_summary)
    target_rows_are_watch_only = _target_rows_require_watch_only(final)
    if green_count > 0:
        next_action = "review Send Now Orders; enter manually only after final quote check"
    elif target_count > 0 and target_rows_are_watch_only:
        next_action = "watch yellow rows; do not use as limit orders until profit and expectancy gates clear"
    elif target_count > 0:
        next_action = "use yellow target credits/debits as target-limit starting points"
    elif blocking_gates:
        next_action = "resolve blocking gates before treating any row as an order"
    else:
        next_action = "no send-now or target-order surface produced"
    lines = [
        f"# Options Agent Report - {day}",
        "",
        f"Mode: {mode_label}.",
        "",
        "Target rows show desired credits/debits. Only rows in Send Now Orders with ready_to_enter=true are executable.",
        "",
        "## Execution Snapshot",
        "",
        f"- Green send-now rows: {green_count}",
        f"- Yellow target rows: {target_count}",
        f"- Target refresh queue rows: {refresh_count}",
        f"- Next action: {next_action}",
        f"- Live quote mode: {live_mode}",
        f"- Portfolio context: {manifest.get('portfolio_context_status', 'unknown')}",
        f"- Profitability confidence: {_as_float(confidence_summary.get('profitability_confidence_rating')) or 0.0}/10",
        f"- Order-entry confidence: {_as_float(confidence_summary.get('order_entry_confidence_rating')) or 0.0}/10",
        f"- Profitability calibration: {calibration_summary.get('status', 'unknown')} "
        f"({calibration_summary.get('pass_rows', 0)} pass / {calibration_summary.get('current_trade_rows', 0)} current rows)",
        f"- Calibration blockers: {calibration_blocker_detail}" if calibration_blocker_detail else "- Calibration blockers: none",
        f"- Bucket blocker examples: {bucket_examples_detail}" if bucket_examples_detail else "- Bucket blocker examples: none",
        f"- Profitability gap plan: {gap_plan_detail}",
        f"- Calibrated order-entry blockers: {calibrated_order_blocker_detail}"
        if calibrated_order_blocker_detail
        else "- Calibrated order-entry blockers: none",
        f"- Execution fill quality: {fill_quality_summary.get('status', 'unknown')} "
        f"({fill_quality_summary.get('pass_rows', 0)} pass / {fill_quality_summary.get('block_rows', 0)} block)",
        f"- Route opportunity gaps: {route_gap_detail}",
        f"- Lesson pack: {lesson_pack_version or 'none'} `{lesson_pack_digest or 'no-digest'}`",
        f"- Report path: `{report_path}`" if report_path else "- Report path: see `options_agent_manifest_*.json`",
        "",
        "## Output Files",
        "",
        f"- Report: `{report_path}`" if report_path else "- Report: see manifest artifacts",
        f"- All visible tickets: `{trade_tickets_path}`" if trade_tickets_path else "- All visible tickets: `trade_tickets.csv`",
        f"- Green send-now tickets: `{green_tickets_path}`" if green_tickets_path else "- Green send-now tickets: `green_trade_tickets.csv`",
        f"- Yellow target candidates: `{target_tickets_path}`" if target_tickets_path else "- Yellow target candidates: `target_order_candidates.csv`",
        f"- Confidence audit: `{confidence_audit_path}`" if confidence_audit_path else "- Confidence audit: `confidence_audit.csv`",
        f"- Strategy outcome atlas: `{strategy_atlas_path}`" if strategy_atlas_path else "- Strategy outcome atlas: `strategy_outcome_atlas.csv`",
        f"- Profitability calibration: `{profitability_calibration_path}`"
        if profitability_calibration_path
        else "- Profitability calibration: `profitability_calibration.csv`",
        f"- Profitability gap plan: `{profitability_gap_plan_path}`"
        if profitability_gap_plan_path
        else "- Profitability gap plan: `profitability_gap_plan.csv`",
        f"- Execution fill quality: `{execution_fill_quality_path}`"
        if execution_fill_quality_path
        else "- Execution fill quality: `execution_fill_quality.csv`",
        f"- Route opportunity gaps: `{route_opportunity_gap_path}`"
        if route_opportunity_gap_path
        else "- Route opportunity gaps: `route_opportunity_gap.csv`",
        "",
    ]
    captured_live = manifest.get("captured_live_recompute", {}) or {}
    if captured_live:
        lines.extend(
            [
                "**Captured-live recompute:** this report rebuilds readiness from a prior Schwab live capture under current code. "
                "It is proof of gate behavior, not a fresh current quote pull; reprice in Schwab before sending anything.",
                "",
            ]
        )
    lines.extend(_render_actionable_tickets(final))
    lines.extend(_render_market_open_recheck_queue(final))
    lines.extend(
        [
            "## Run Diagnostics",
            "",
            "Diagnostics explain confidence and coverage; the order-entry surface is Send Now Orders plus `trade_tickets.csv`.",
            "",
            f"- Trade rows: {row_counts.get('green_trade_tickets', 0)} green send-now, "
            f"{row_counts.get('target_order_ticket_rows', 0)} target-order candidates",
            f"- Send-now readiness: {execution_summary.get('status', 'unknown')}; "
            f"non-green gates: {execution_summary.get('blocking_gates', [])}",
            f"- Live quote mode: {live_mode}; live validation rows: {row_counts.get('live_chain_validation', 0)}",
            (
                "- Live spread quality audit: "
                f"{live_quality_summary.get('status', 'not_evaluated')}; "
                f"{live_quality_summary.get('block_rows', 0)} blocked "
                f"({live_quality_summary.get('quote_width_block_rows', 0)} quote-width, "
                f"{live_quality_summary.get('liquidity_block_rows', 0)} liquidity)"
            ),
            (
                "- Agentic review coverage: "
                f"lane {execution_context.get('external_review_agent_count', 0)}/"
                f"{execution_context.get('agent_dispatch_task_count', 0)} "
                f"({execution_context.get('agentic_review_lane_coverage_pct', 'unknown')}); "
                f"broad rows {execution_context.get('external_review_count', row_counts.get('external_agent_reviews', 0))}/"
                f"{execution_context.get('research_task_count', row_counts.get('research_tasks', 0))} "
                f"({execution_context.get('broad_review_coverage_pct', 'unknown')})"
            ),
            f"- Structure attempt rows: {row_counts.get('structure_attempts', 0)}",
            f"- Final visible rows: {row_counts.get('final_recommendations', 0)}",
            f"- Structural status counts, not order readiness: {status_counts}",
            f"- Portfolio context: {manifest.get('portfolio_context_status', 'unknown')}",
            f"- Raw discovery: {row_counts.get('raw_universe', 0)} UW rows, "
            f"{row_counts.get('candidate_generation', 0)} generated candidates, "
            f"{row_counts.get('catalyst_evidence', 0)} catalyst rows, "
            f"{row_counts.get('agent_review_board', 0)} review rows",
            f"- Agentic dispatch tasks: {row_counts.get('agent_dispatch_tasks', 0)}; "
            f"review status: {agentic.get('status', 'unknown')}",
            f"- Agent review verdicts: {review_summary.get('by_verdict', {})}; "
            f"objective blockers: {review_summary.get('objective_blockers', 0)}",
            f"- Strategy outcome atlas: positive families {strategy_atlas_summary.get('positive_strategy_families', [])}; "
            f"negative current families {strategy_atlas_summary.get('negative_current_strategy_families', [])}; "
            f"blocking current ticker-strategy rows {strategy_atlas_summary.get('blocking_current_ticker_strategy_rows', 0)}",
            f"- Route opportunity gaps: {route_gap_detail}",
            "",
        ]
    )
    lines.extend(_render_execution_quality(final))
    lines.extend(_render_review_queue(final))
    lines.extend(_render_coverage_audit(coverage_audit))
    lines.extend(["## Decision Board Summary", ""])
    if final.empty:
        lines.append("No final recommendation rows were produced.")
    else:
        lines.append(
            "Full ranked rows are in `decision_board.csv`; rejected setup-quality rows stay audit-visible in `final_recommendations.csv`."
        )
        lines.append("")
        lines.append("| Status | Count |")
        lines.append("|---|---:|")
        if "execution_status" in final.columns:
            for status, count in final["execution_status"].astype(str).value_counts().sort_index().items():
                lines.append(f"| {status} | {count} |")
    lines.extend(["", "## Near Miss / No Trade Audit", ""])
    if no_trade.empty:
        lines.append("No near misses captured.")
    else:
        lines.append(f"Showing first 20 of {len(no_trade)} rows; full audit is in `no_trade_audit.csv`.")
        lines.append("")
        lines.append("| Ticker | Bias | Score | Reason |")
        lines.append("|---|---|---:|---|")
        for _, row in no_trade.head(20).iterrows():
            lines.append(
                f"| {row.get('ticker', '')} | {row.get('bias', '')} | {row.get('score', '')} | {str(row.get('reason', '')).replace('|', '/')} |"
            )
        if len(no_trade) > 20:
            lines.append(f"| ... |  |  | {len(no_trade) - 20} additional no-trade rows in no_trade_audit.csv |")
    if warnings:
        lines.extend(["", "## Warnings", ""])
        for warning in warnings:
            lines.append(f"- {warning}")
    return "\n".join(lines) + "\n"


def _render_monthly_evidence_diagnostics(
    feasibility_summary: Mapping[str, Any],
    expectancy_summary: Mapping[str, Any],
) -> list[str]:
    """Render monthly-target diagnostics separately from order-entry readiness."""

    feasibility_status = _as_text(feasibility_summary.get("status") or "unknown")
    expectancy_status = _as_text(expectancy_summary.get("status") or "unknown")
    expectancy_note = _as_text(expectancy_summary.get("note"))
    blocking_metrics = feasibility_summary.get("blocking_metrics", []) or []
    lines = ["## Monthly Evidence Diagnostics", ""]
    if feasibility_status == "capacity_and_expectancy_positive_not_guaranteed":
        lines.append(
            "Monthly target evidence is positive enough for review, but it is still not a guarantee. Confirm fills, exposure, and current market regime before sizing around the monthly target."
        )
    else:
        lines.append(
            "Monthly target evidence is insufficient. Green send-now rows are order-entry candidates only; do not size around a monthly target from this artifact."
        )
    lines.append("")
    lines.append(f"- Monthly feasibility: {feasibility_status}")
    lines.append(f"- Expectancy evidence: {expectancy_status}" + (f" ({expectancy_note})" if expectancy_note else ""))
    if blocking_metrics:
        lines.append(f"- Blocking metrics: {blocking_metrics}")
    lines.append("- Source of truth: `monthly_feasibility.csv` and `expectancy_evidence.csv`.")
    lines.append("")
    return lines


def _render_actionable_tickets(final: pd.DataFrame) -> list[str]:
    lines = []
    if final.empty or "trade_plan" not in final.columns:
        lines.extend(
            [
                "## Send Now Orders",
                "",
                "No green send-now orders.",
                "",
                "## Target Orders - Target Credits/Debits",
                "",
                "No target-order candidates were produced.",
                "",
            ]
        )
        return lines
    tickets = build_trade_tickets(final)
    ready, target = split_trade_ticket_surfaces(tickets)

    lines.extend(["## Send Now Orders", ""])
    if ready.empty:
        lines.extend(["No green send-now orders. Do not send an order unless a row appears here.", ""])
    else:
        lines.extend(_render_ticket_rows(ready))

    lines.extend(["## Target Orders - Target Credits/Debits", ""])
    if target.empty:
        lines.extend(["No target-order candidates were produced.", ""])
    else:
        if _target_rows_require_watch_only_from_tickets(target):
            lines.append(
                "These are watch targets, not order-entry candidates. Do not use them as limit orders until the listed profit, expectancy, and quality gates clear."
            )
        else:
            lines.append(
                "These are planning targets. Use the shown desired credit/debit as the starting limit, then refresh the Schwab quote before sending."
            )
        lines.append("")
        lines.extend(_render_ticket_rows(target))
    return lines


def _target_rows_require_watch_only(final: pd.DataFrame) -> bool:
    if final.empty or "trade_plan" not in final.columns:
        return False
    tickets = build_trade_tickets(final)
    _, target = split_trade_ticket_surfaces(tickets)
    return _target_rows_require_watch_only_from_tickets(target)


def _target_rows_require_watch_only_from_tickets(target: pd.DataFrame) -> bool:
    if target is None or target.empty or "execution_blockers" not in target.columns:
        return False
    watch_gate_rows = 0
    for _, row in target.iterrows():
        blockers = _blocker_set(row.get("execution_blockers"))
        if blockers & {
            POSITION_PROFIT_MATERIALITY_BLOCKER,
            POSITIVE_STRATEGY_EXPECTANCY_BLOCKER,
            PROFITABILITY_CALIBRATION_BLOCKER,
        } or _profitability_calibration_status_blocks_target(row):
            watch_gate_rows += 1
    return watch_gate_rows == len(target)


def _render_ticket_rows(tickets: pd.DataFrame) -> list[str]:
    lines = [
        "| Ticker | Signal | Structure | Exp | Sell Leg | Buy Leg | Qty | Target Limit | Target Exit | Max Profit | Max Loss | Confidence | Price / Risk |"
    ]
    lines.append("|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|")
    for _, row in tickets.iterrows():
        lines.append(
            "| {ticker} | {signal} | {structure} | {expiry} | {sell_leg} | {buy_leg} | {qty} | {entry} | {target} | {position_profit} | {position_loss} | {confidence} | {recheck} |".format(
                ticker=_report_cell(row.get("ticker")),
                signal=_report_cell(_decision_badge(row)),
                structure=_report_cell(_ticket_structure(row)),
                expiry=_report_cell(_ticket_expiry(row)),
                sell_leg=_report_cell(row.get("sell_leg") or _ticket_leg_from_plan(row, "SELL")),
                buy_leg=_report_cell(row.get("buy_leg") or _ticket_leg_from_plan(row, "BUY")),
                qty=row.get("suggested_contracts", ""),
                entry=_report_cell(_ticket_limit_display(row)),
                target=_display_value(row.get("target_exit")),
                position_profit=_row_position_amount(row, "max_profit", "position_max_profit"),
                position_loss=_row_position_amount(row, "max_loss", "position_max_loss"),
                confidence=_report_cell(_ticket_confidence(row)),
                recheck=_report_cell(_ticket_recheck_summary(row)),
            )
        )
    lines.append("")
    return lines


def _report_cell(value: Any) -> str:
    return _as_text(value).replace("|", "/")


def _ticket_expiry(row: Mapping[str, Any]) -> str:
    expiry = _as_text(row.get("expiry"))
    if expiry:
        return expiry
    match = re.search(r"\b20\d{2}-\d{2}-\d{2}\b", _as_text(row.get("trade_plan")))
    return match.group(0) if match else ""


def _ticket_leg_from_plan(row: Mapping[str, Any], side: str) -> str:
    side_upper = side.upper()
    for part in _as_text(row.get("trade_plan")).split("/"):
        text = part.strip()
        if text.upper().startswith(side_upper):
            return text.split("@", 1)[0].strip()
    return ""


def _ticket_structure(row: Mapping[str, Any]) -> str:
    if _strategy_family_from_ticket_row(row) == "short_put":
        return "Cash-secured put"
    text = " ".join(
        [
            _as_text(row.get("sell_leg")),
            _as_text(row.get("buy_leg")),
            _as_text(row.get("trade_plan")),
        ]
    ).upper()
    option_kind = "Call" if " CALL" in text else "Put" if " PUT" in text else "Option"
    entry_type = _as_text(row.get("entry_type")) or _entry_type_from_ticket(row.get("trade_plan"))
    if entry_type:
        return f"{option_kind} {entry_type.lower()} spread"
    return f"{option_kind} spread"


def _ticket_limit_display(row: Mapping[str, Any]) -> str:
    entry = _display_value(row.get("entry_limit"))
    entry_type = _as_text(row.get("entry_type")) or _entry_type_from_ticket(row.get("trade_plan"))
    return f"{entry} {entry_type}".strip()


def _ticket_confidence(row: Mapping[str, Any]) -> str:
    quality = _as_text(row.get("trade_quality_confidence_rating"))
    score = _display_value(row.get("execution_confidence_score"))
    if quality and score:
        return f"{quality} / {score}"
    return quality or score


def _ticket_recheck_summary(row: Mapping[str, Any]) -> str:
    if _truthy(row.get("ready_to_enter")):
        return "green ready; verify live quote before manual send"
    labels = {
        "portfolio_context_required": "portfolio refresh",
        "fresh_live_schwab_required": "fresh Schwab chain",
        "regular_session_quote_refresh_required": "fresh quote refresh",
        "live_validation_pass_required": "fresh Schwab chain",
        "manual_review_required": "manual review",
        "positive_contract_size_required": "size check",
        "positive_entry_limit_required": "price check",
        "agentic_reviews_required": "agent review coverage",
        "agentic_review_coverage_below_threshold": "agent review coverage",
        "ticker_agentic_review_coverage_below_threshold": "agent review coverage",
        "execution_confidence_below_threshold": "confidence review",
        POSITION_PROFIT_MATERIALITY_BLOCKER: "position max profit below materiality floor",
        NEGATIVE_STRATEGY_EXPECTANCY_BLOCKER: "negative realized strategy history",
        POSITIVE_STRATEGY_EXPECTANCY_BLOCKER: "positive strategy expectancy required",
        f"send_now_credit_below_{MIN_SEND_NOW_CREDIT:.2f}": "credit too small for send-now",
        "send_now_credit_below_1.00": "credit too small for send-now",
        f"send_now_credit_width_below_{int(MIN_SEND_NOW_CREDIT_WIDTH_RATIO * 100)}pct": "credit/width too weak for send-now",
        "send_now_credit_width_below_30pct": "credit/width too weak for send-now",
        "send_now_debit_reward_risk_below_1.5x": "reward/risk too weak for send-now",
        "send_now_debit_directional_edge_below_threshold": "directional edge too weak for send-now",
        "trade_quality_review_required": "trade quality review",
    }
    blocker_priority = [
        "fresh_live_schwab_required",
        "regular_session_quote_refresh_required",
        "live_validation_pass_required",
        "portfolio_context_required",
        "manual_review_required",
        "positive_contract_size_required",
        "positive_entry_limit_required",
        "agentic_reviews_required",
        "agentic_review_coverage_below_threshold",
        "ticker_agentic_review_coverage_below_threshold",
        "execution_confidence_below_threshold",
        POSITION_PROFIT_MATERIALITY_BLOCKER,
        NEGATIVE_STRATEGY_EXPECTANCY_BLOCKER,
        POSITIVE_STRATEGY_EXPECTANCY_BLOCKER,
        f"send_now_credit_below_{MIN_SEND_NOW_CREDIT:.2f}",
        "send_now_credit_below_1.00",
        f"send_now_credit_width_below_{int(MIN_SEND_NOW_CREDIT_WIDTH_RATIO * 100)}pct",
        "send_now_credit_width_below_30pct",
        "send_now_debit_reward_risk_below_1.5x",
        "send_now_debit_directional_edge_below_threshold",
        "trade_quality_review_required",
    ]
    blockers = _blocker_set(row.get("execution_blockers"))
    ordered_blockers = [blocker for blocker in blocker_priority if blocker in blockers]
    ordered_blockers.extend(sorted(blockers - set(blocker_priority)))
    parts = []
    for blocker in ordered_blockers:
        if blocker in labels:
            parts.append(labels[blocker])
        elif blocker.startswith("send_now_debit_breakeven_move_above_"):
            parts.append("breakeven move too large for send-now")
    if not parts:
        parts.append(_ticket_next_step(row))
    return "; ".join(dict.fromkeys(parts))


def _render_market_open_recheck_queue(final: pd.DataFrame) -> list[str]:
    if final.empty:
        return []
    queue = build_market_open_recheck_queue(build_trade_tickets(final))
    if queue.empty:
        return []
    lines = ["## Target Price-Validation Queue", ""]
    lines.append(
        "These target rows have complete order details and need a fresh quote refresh before any manual entry."
    )
    lines.append("")
    lines.append(
        "| Ticker | Signal | Structure | Exp | Qty | Target Limit | Max Profit | Max Loss | Required Check |"
    )
    lines.append("|---|---|---|---|---:|---:|---:|---:|---|")
    for _, row in queue.iterrows():
        lines.append(
            "| {ticker} | {signal} | {structure} | {expiry} | {qty} | {entry} | {position_profit} | {position_loss} | {required} |".format(
                ticker=_report_cell(row.get("ticker")),
                signal=_report_cell(_decision_badge(row)),
                structure=_report_cell(_ticket_structure(row)),
                expiry=_report_cell(_ticket_expiry(row)),
                qty=row.get("suggested_contracts", ""),
                entry=_report_cell(_ticket_limit_display(row)),
                position_profit=_row_position_amount(row, "max_profit", "position_max_profit"),
                position_loss=_row_position_amount(row, "max_loss", "position_max_loss"),
                required=_report_cell(row.get("required_recheck")),
            )
        )
    lines.append("")
    return lines


def _render_execution_quality(final: pd.DataFrame) -> list[str]:
    lines = ["## Execution Quality Gates", ""]
    if final.empty or "execution_confidence_rating" not in final.columns:
        lines.extend(["No execution-confidence rows were produced.", ""])
        return lines
    counts = final["execution_confidence_rating"].astype(str).value_counts().to_dict()
    quality_counts = final.get("trade_quality_confidence_rating", pd.Series(dtype=object)).astype(str).value_counts().to_dict()
    lines.append(f"- Execution confidence ratings: {counts}")
    lines.append(f"- Trade-quality confidence ratings: {quality_counts}")
    if "execution_blockers" in final.columns:
        blockers: dict[str, int] = {}
        for text in final["execution_blockers"].astype(str).tolist():
            for blocker in [part.strip() for part in text.split(";") if part.strip() and part.strip().lower() != "nan"]:
                label = _display_blocker_label(blocker)
                blockers[label] = blockers.get(label, 0) + 1
        if blockers:
            top = dict(sorted(blockers.items(), key=lambda item: (-item[1], item[0]))[:8])
            lines.append(f"- Top non-green send-now gates: {top}")
    lines.append("")
    return lines


def _display_blocker_label(blocker: Any) -> str:
    text = _as_text(blocker)
    labels = {
        "market_session_open_required": "fresh quote refresh",
        "regular_session_quote_refresh_required": "fresh quote refresh",
        "fresh_live_schwab_required": "fresh Schwab chain",
        "live_validation_pass_required": "fresh Schwab chain",
        "portfolio_context_required": "portfolio refresh",
        "agentic_reviews_required": "agent review coverage",
        "agentic_review_coverage_below_threshold": "agent review coverage",
        "ticker_agentic_review_coverage_below_threshold": "agent review coverage",
        POSITION_PROFIT_MATERIALITY_BLOCKER: "position max profit below materiality floor",
        NEGATIVE_STRATEGY_EXPECTANCY_BLOCKER: "negative realized strategy history",
        POSITIVE_STRATEGY_EXPECTANCY_BLOCKER: "positive strategy expectancy required",
        PROFITABILITY_CALIBRATION_BLOCKER: "route/economics profitability calibration required",
    }
    if text in labels:
        return labels[text]
    if text.startswith("send_now_credit_below_"):
        return "credit too small for send-now"
    if text.startswith("send_now_credit_width_below_"):
        return "credit/width too weak for send-now"
    if text.startswith("send_now_debit_breakeven_move_above_"):
        return "breakeven move too large for send-now"
    return text


def _render_review_queue(final: pd.DataFrame) -> list[str]:
    lines = ["## Focus Review Queue - Not Trades", ""]
    if final.empty or "trade_plan" not in final.columns:
        lines.extend(["No reviewable ticket candidates were constructed.", ""])
        return lines
    watchlist = set(CORE_AUDIT_TICKERS)
    target_status = final.get("target_order_status", pd.Series("", index=final.index)).astype(str).str.lower()
    live_status = final.get("live_validation_status", pd.Series("", index=final.index)).astype(str).str.upper()
    ticker_series = final.get("ticker", pd.Series("", index=final.index)).astype(str).str.upper()
    underlying_tier = final.get("underlying_quality_tier", pd.Series("", index=final.index)).astype(str).str.lower()
    review = final[
        final["trade_plan"].astype(str).str.strip().ne("")
        & final["entry_limit"].map(lambda value: (_as_float(value) or 0.0) > 0)
        & ~final["ready_to_enter"].map(_truthy)
        & ~final["execution_status"].astype(str).str.lower().eq("blocked")
        & ~target_status.isin(["target_order_candidate", "target_order_wait_for_price"])
        & live_status.eq("PASS")
        & underlying_tier.eq("core")
    ].copy()
    if review.empty:
        lines.extend(["No focus-review candidates remain after actionability and setup-quality gates.", ""])
        return lines
    lines.append("These are not orders. This section is limited to validated rows and focus tickers; tail unvalidated rows stay in CSV artifacts.")
    lines.append("")
    lines.append("| Ticker | Signal | Reason | Qty | Target Limit | Max Loss | Trade Plan |")
    lines.append("|---|---|---|---:|---:|---:|---|")
    for _, row in review.head(25).iterrows():
        reason = _as_text(row.get("quality_gate_reason")) or _as_text(row.get("status_reason")) or _ticket_next_step(row)
        lines.append(
            "| {ticker} | {icon} | {reason} | {qty} | {entry} | {position_loss} | {trade_plan} |".format(
                ticker=_report_cell(row.get("ticker")),
                icon=_report_cell(_decision_badge(row)),
                reason=_report_cell(_short_report_reason(reason, max_parts=3)),
                qty=row.get("suggested_contracts", ""),
                entry=_report_cell(_ticket_limit_display(row)),
                position_loss=_row_position_amount(row, "max_loss", "position_max_loss"),
                trade_plan=_report_cell(row.get("trade_plan")),
            )
        )
    if len(review) > 25:
        lines.append(f"| ... |  | {len(review) - 25} additional review rows in decision_board.csv |  |  |  |  |")
    lines.append("")
    return lines


def _row_position_amount(row: Mapping[str, Any], one_lot_column: str, position_column: str) -> Any:
    explicit = _as_float(row.get(position_column))
    if explicit is None and position_column == "position_max_loss":
        explicit = _as_float(row.get("max_position_loss"))
    if explicit is not None and explicit > 0:
        return round(explicit, 2)
    one_lot = _as_float(row.get(one_lot_column))
    contracts = int(_as_float(row.get("suggested_contracts")) or 0)
    if one_lot is None or contracts <= 0:
        return ""
    return round(one_lot * contracts, 2)


def _display_value(value: Any) -> str:
    numeric = _as_float(value)
    if numeric is not None:
        if numeric.is_integer():
            return str(int(numeric))
        return f"{numeric:.2f}".rstrip("0").rstrip(".")
    return _as_text(value)


def _short_report_reason(value: Any, *, max_parts: int = 3) -> str:
    text = _as_text(value)
    if not text:
        return ""
    parts: list[str] = []
    schwab_auth_needed = False
    live_refresh_seen = False
    for raw_part in text.split(";"):
        part = raw_part.strip()
        if not part:
            continue
        lower = part.lower()
        if "schwab token refresh failed" in lower or "re-auth once" in lower:
            schwab_auth_needed = True
            continue
        if lower.startswith("portfolio annotation only"):
            continue
        if lower == "trade remains visible for manual review":
            continue
        if "preserve dated target credit/debit" in lower:
            continue
        if "regular market is closed" in lower:
            cleaned = "fresh quote refresh"
        elif "dated uw eod quote" in lower:
            cleaned = "dated UW quote, fresh chain needed"
            live_refresh_seen = True
        elif "refresh schwab chain before entry" in lower and live_refresh_seen:
            continue
        elif lower.startswith("external agent caution:"):
            cleaned = part.split(":", 1)[1].strip()
        else:
            cleaned = part
        cleaned = _sanitize_visible_review_note(cleaned)
        cleaned = cleaned.replace("|", "/")
        if cleaned and cleaned not in parts:
            parts.append(cleaned)
    if schwab_auth_needed:
        parts.append("Schwab re-auth needed for fresh chain")
    return "; ".join(parts[:max_parts])


def _render_coverage_audit(coverage_audit: Optional[pd.DataFrame]) -> list[str]:
    lines = ["## Coverage Audit", ""]
    if coverage_audit is None or coverage_audit.empty:
        lines.extend(["No coverage audit rows were produced.", ""])
        return lines
    lines.append(
        "Coverage rows explain inclusion/exclusion only. They are not orders; use Send Now Orders, Target Orders, and `trade_tickets.csv` for the action surface."
    )
    lines.append("")
    lines.append("| Ticker | Signal | Bias | Score | State | Why | Next Step |")
    lines.append("|---|---|---|---:|---|---|---|")
    for _, row in coverage_audit.head(30).iterrows():
        lines.append(
            "| {ticker} | {signal} | {bias} | {score} | {state} | {reason} | {next_step} |".format(
                ticker=_report_cell(row.get("ticker")),
                signal=_report_cell(_coverage_badge(row)),
                bias=_report_cell(row.get("bias")),
                score=_display_value(row.get("score")),
                state=_report_cell(_coverage_state_label(_coverage_display_status(row))),
                reason=_report_cell(_short_report_reason(row.get("reason"), max_parts=3)),
                next_step=_report_cell(_coverage_display_next_step(row)),
            )
        )
    lines.append("")
    return lines


def _coverage_icon(color: Any) -> str:
    icons = {
        "green": "🟢",
        "yellow": "🟡",
        "red": "🔴",
        "gray": "⚪",
    }
    return icons.get(_as_text(color), "⚪")


def _coverage_badge(row: Mapping[str, Any]) -> str:
    state = _coverage_display_status(row)
    color = "red" if state == "NON_ACTIONABLE_UNDERLYING" else _as_text(row.get("status_color"))
    label = _coverage_state_label(state)
    return f"{_coverage_icon(color)} {label}".strip()


def _coverage_display_status(row: Mapping[str, Any]) -> str:
    status = _as_text(row.get("coverage_status"))
    if _is_audit_only_underlying(row) and status in _LOW_QUALITY_COVERAGE_STATUSES:
        return "NON_ACTIONABLE_UNDERLYING"
    return status


def _coverage_display_next_step(row: Mapping[str, Any]) -> str:
    status = _coverage_display_status(row)
    if status == "NON_ACTIONABLE_UNDERLYING":
        return _coverage_next_step(status)
    return _as_text(row.get("next_step"))


def _coverage_state_label(status: Any) -> str:
    state = _as_text(status)
    return {
        "READY_TICKET": "GREEN ready",
        "TARGET_ORDER_CANDIDATE": "YELLOW coverage",
        "REVIEW_TICKET": "YELLOW review",
        "UNVALIDATED_CHAIN": "YELLOW target",
        "CANDIDATE_NOT_STRUCTURED": "YELLOW candidate",
        "STRUCTURED_NOT_TOP_FINAL": "YELLOW structured",
        "STRUCTURE_MISSING": "YELLOW no-structure",
        "MACRO_TAPE_CANDIDATE": "YELLOW macro",
        "MACRO_TAPE_NO_LIVE_EDGE": "YELLOW macro no-edge",
        "MACRO_TAPE_REJECTED": "GRAY macro rejected",
        "FINAL_NO_TICKET": "YELLOW no-ticket",
        "NON_ACTIONABLE_UNDERLYING": "RED no-action",
        "BLOCKED_FINAL_ROW": "RED blocked",
        "NO_DIRECTIONAL_EDGE": "GRAY no-edge",
        "BELOW_DISCOVERY_CUTOFF": "GRAY below-cutoff",
        "SOURCE_MISSING": "RED missing",
    }.get(state, state or "UNKNOWN")


def _decision_badge(row: Mapping[str, Any]) -> str:
    if _truthy(row.get("ready_to_enter")):
        return f"{_coverage_icon('green')} GREEN ready"
    target_status = _as_text(row.get("target_order_status"))
    if target_status in {"target_order_candidate", "target_order_wait_for_price"}:
        return f"{_coverage_icon('yellow')} YELLOW target"
    if target_status == "not_actionable_underlying_quality":
        return f"{_coverage_icon('red')} RED no-action"
    execution = _as_text(row.get("execution_status"))
    if execution == "blocked":
        return f"{_coverage_icon('red')} RED blocked"
    if _as_text(row.get("trade_plan")):
        return f"{_coverage_icon('yellow')} YELLOW review"
    return f"{_coverage_icon('gray')} GRAY review"


def _decision_icon(row: Mapping[str, Any]) -> str:
    if _truthy(row.get("ready_to_enter")):
        return _coverage_icon("green")
    target_status = _as_text(row.get("target_order_status"))
    if target_status in {"target_order_candidate", "target_order_wait_for_price"}:
        return _coverage_icon("yellow")
    if target_status == "not_actionable_underlying_quality":
        return _coverage_icon("red")
    execution = _as_text(row.get("execution_status"))
    if execution == "blocked":
        return _coverage_icon("red")
    if _as_text(row.get("trade_plan")):
        return _coverage_icon("yellow")
    return _coverage_icon("gray")


def _decision_status_label(row: Mapping[str, Any]) -> str:
    if _truthy(row.get("ready_to_enter")):
        return "GREEN ready"
    target_status = _as_text(row.get("target_order_status"))
    if target_status in {"target_order_candidate", "target_order_wait_for_price"}:
        return "YELLOW target"
    if target_status == "not_actionable_underlying_quality":
        return "RED no-action"
    execution = _as_text(row.get("execution_status"))
    if execution == "blocked":
        return "RED blocked"
    if _as_text(row.get("trade_plan")):
        return "YELLOW review"
    return "GRAY review"


def _is_market_session_only_target(row: Mapping[str, Any]) -> bool:
    target_status = _as_text(row.get("target_order_status"))
    if target_status not in {"target_order_candidate", "target_order_wait_for_price"}:
        return False
    return _blocker_set(row.get("execution_blockers")) in (
        {"market_session_open_required"},
        {"regular_session_quote_refresh_required"},
    )


def _ticket_next_step(row: Mapping[str, Any]) -> str:
    readiness = _ticket_order_readiness(row)
    if readiness == "ready_to_enter":
        return "verify live quote and enter manually"
    if readiness in {"target_order_price_validation", "target_order_after_quote_refresh", "target_order_after_live_recheck"}:
        return "use the shown target limit as the starting point; adjust if the live quote moves"
    if readiness == "target_order_profit_floor":
        return "watch only unless the sized max profit clears the materiality floor"
    if readiness == "target_order_after_expectancy_evidence":
        return "watch only unless structure-aligned expectancy evidence or an explicit current-edge override passes"
    if readiness == "target_order_after_profitability_calibration":
        return "watch only unless the route, DTE, and credit/debit bucket calibration passes"
    if readiness in {"target_order_after_agentic_review", "target_order_after_agentic_review_and_live_recheck"}:
        blockers = _blocker_set(row.get("execution_blockers"))
        portfolio_step = "load portfolio context, " if "portfolio_context_required" in blockers else ""
        return f"complete subagent reviews, {portfolio_step}then use this target as the starting limit"
    if readiness in {"target_order_after_portfolio_sizing", "target_order_after_portfolio_and_live_recheck"}:
        return "refresh portfolio context, then use this target as the starting limit"
    if readiness == "target_order_after_market_open_and_live_recheck":
        return "use this target as the starting limit after a fresh quote refresh"
    if readiness == "target_order_wait_for_price":
        return "leave at target limit only if fresh quote improves"
    if readiness == "not_ready_wait_for_price":
        return "wait for desired credit/debit"
    if readiness == "not_ready_live_validation_required":
        return "refresh Schwab chain before entry"
    if readiness == "not_ready_sizing_required":
        return "set manual size before entry"
    if readiness == "not_ready_objective_blocker":
        return "do not enter unless blocker is cleared"
    return "manual review before any order"


def build_agent_orchestration(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Describe the deterministic multi-agent handoff graph for this run."""

    return {
        "pipeline_name": manifest.get("pipeline_name", PIPELINE_NAME),
        "pipeline_version": manifest.get("pipeline_version", PIPELINE_VERSION),
        "as_of": manifest.get("as_of", ""),
        "execution_model": "two-pass Codex multi-agent dispatch plus deterministic local synthesis",
        "visibility_invariant": manifest.get("visibility_invariant", ""),
        "agentic_orchestration": manifest.get("agentic_orchestration", {}),
        "agents": manifest.get("agents", []),
        "handoffs": [
            {"from": "data", "to": "flow_oi", "artifact": "raw_universe.csv"},
            {"from": "data", "to": "market_regime", "artifact": "raw_universe.csv"},
            {"from": "flow_oi", "to": "catalyst", "artifact": "candidate_generation.csv"},
            {"from": "catalyst", "to": "research_dispatch", "artifact": "catalyst_evidence.csv"},
            {"from": "flow_oi", "to": "research_dispatch", "artifact": "candidate_generation.csv"},
            {"from": "market_regime", "to": "research_dispatch", "artifact": "market_regime.json"},
            {"from": "catalyst", "to": "research_dispatch", "artifact": "catalyst_reviews.csv"},
            {"from": "research_dispatch", "to": "external_subagents", "artifact": "research_tasks.json"},
            {"from": "research_dispatch", "to": "codex_subagents", "artifact": "agent_dispatch_plan.json"},
            {"from": "codex_subagents", "to": "research_dispatch", "artifact": "agentic_reviews.json"},
            {"from": "external_subagents", "to": "research_dispatch", "artifact": "external_agent_reviews.csv"},
            {"from": "market_regime", "to": "research_dispatch", "artifact": "agent_review_board.csv"},
            {"from": "catalyst", "to": "research_dispatch", "artifact": "agent_review_board.csv"},
            {"from": "flow_oi", "to": "structure", "artifact": "candidate_generation.csv"},
            {"from": "market_regime", "to": "structure", "artifact": "market_regime.json"},
            {"from": "catalyst", "to": "structure", "artifact": "catalyst_reviews.csv"},
            {"from": "research_dispatch", "to": "structure", "artifact": "agent_review_board.csv"},
            {"from": "structure", "to": "synthesis", "artifact": "strategy_routing_audit.csv"},
            {"from": "structure", "to": "synthesis", "artifact": "structure_attempts.csv"},
            {"from": "structure", "to": "synthesis", "artifact": "live_spread_quality_audit.csv"},
            {"from": "structure", "to": "portfolio_risk", "artifact": "priced_candidates.csv"},
            {"from": "structure", "to": "research_dispatch", "artifact": "agent_review_board.csv"},
            {"from": "research_dispatch", "to": "skeptic", "artifact": "agent_review_board.csv"},
            {"from": "portfolio_risk", "to": "skeptic", "artifact": "risk_audit.csv"},
            {"from": "portfolio_risk", "to": "sizing", "artifact": "final_recommendations.csv"},
            {"from": "sizing", "to": "skeptic", "artifact": "sizing_audit.csv"},
            {"from": "sizing", "to": "management", "artifact": "sizing_audit.csv"},
            {"from": "synthesis", "to": "management", "artifact": "decision_board.csv"},
            {"from": "portfolio_risk", "to": "research_dispatch", "artifact": "agent_review_board.csv"},
            {"from": "skeptic", "to": "synthesis", "artifact": "no_trade_audit.csv"},
            {"from": "sizing", "to": "synthesis", "artifact": "sizing_audit.csv"},
            {"from": "management", "to": "synthesis", "artifact": "management_plan.csv"},
            {"from": "synthesis", "to": "execution_readiness", "artifact": "execution_readiness.csv"},
            {"from": "outcome_evidence", "to": "monthly_feasibility", "artifact": "expectancy_evidence.csv"},
            {"from": "execution_readiness", "to": "monthly_feasibility", "artifact": "monthly_feasibility.csv"},
            {"from": "research_dispatch", "to": "synthesis", "artifact": "agent_review_board.csv"},
            {"from": "portfolio_risk", "to": "synthesis", "artifact": "final_recommendations.csv"},
            {"from": "management", "to": "user", "artifact": "management_plan.csv"},
            {"from": "synthesis", "to": "user", "artifact": "decision_board.csv"},
            {"from": "synthesis", "to": "manual_entry", "artifact": "trade_tickets.csv"},
            {"from": "synthesis", "to": "manual_entry", "artifact": "green_trade_tickets.csv"},
            {"from": "synthesis", "to": "manual_entry", "artifact": "target_order_candidates.csv"},
        ],
        "row_counts": manifest.get("row_counts", {}),
        "status_counts": manifest.get("status_counts", {}),
    }


def run_design_smoke(
    as_of: str | dt.date,
    root: Optional[Path] = None,
    out_dir: Optional[Path] = None,
) -> dict[str, Path]:
    """Write a smoke-run manifest, report, and empty artifact shells."""

    day = parse_as_of(as_of).isoformat()
    paths = output_paths(day, root=root, out_dir=out_dir)
    paths["out_dir"].mkdir(parents=True, exist_ok=True)
    paths["agent_reviews_dir"].mkdir(parents=True, exist_ok=True)

    lesson_pack = load_active_lesson_pack(root or project_root())
    manifest = build_manifest(day, root=root, out_dir=paths["out_dir"])
    lesson_metadata = lesson_manifest_metadata(lesson_pack, paths)
    manifest.update({"lessonengine": lesson_metadata, **lesson_metadata})
    paths["manifest"].write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["report"].write_text(_smoke_report(day, manifest), encoding="utf-8")
    write_lesson_snapshots(lesson_pack, paths)

    _write_json(paths["source_inventory"], {"as_of": day, "sources": [], "status": "not_collected"})
    _write_json(paths["market_regime"], {"status": "not_collected"})
    _write_json(paths["agent_orchestration"], build_agent_orchestration(manifest))
    _write_json(paths["research_tasks"], {"schema_version": "options_agent.research_tasks.v1", "tasks": []})
    _write_json(
        paths["agent_dispatch_plan"],
        {
            "schema_version": "options_agent.agent_dispatch.v1",
            "as_of": day,
            "dispatch_status": "design_smoke",
            "subagent_tasks": [],
        },
    )
    _write_json(paths["agentic_reviews"], {"reviews": []})
    _write_csv(paths["raw_universe"], ("ticker", "source", "status"), [])
    _write_csv(paths["candidate_generation"], ("ticker", "bias", "score", "reason"), [])
    _write_csv(
        paths["catalyst_evidence"],
        (
            "ticker",
            "evidence_type",
            "evidence_status",
            "source",
            "headline",
            "sentiment",
            "red_flag_terms",
            "support_terms",
            "objective_blocker",
            "days_to_earnings",
            "note",
        ),
        [],
    )
    _write_csv(paths["catalyst_reviews"], ("ticker", "catalyst_status", "catalyst_note"), [])
    _write_csv(paths["external_agent_reviews"], EXTERNAL_REVIEW_COLUMNS, [])
    _write_csv(paths["agent_review_board"], AGENT_REVIEW_COLUMNS, [])
    _write_csv(
        paths["structure_attempts"],
        (
            "ticker",
            "attempt_stage",
            "attempt_status",
            "structure",
            "full_ticket",
            "trade_plan",
            "expiry",
            "entry_limit",
            "target_entry",
            "max_profit",
            "max_loss",
            "sell_leg",
            "buy_leg",
            "short_leg",
            "long_leg",
            "source",
            "note",
        ),
        [],
    )
    _write_csv(paths["strategy_routing_audit"], STRATEGY_ROUTING_AUDIT_COLUMNS, [])
    _write_csv(paths["priced_candidates"], ("ticker", "structure", "entry_limit", "max_profit", "max_loss"), [])
    _write_csv(paths["live_spread_quality_audit"], LIVE_SPREAD_QUALITY_AUDIT_COLUMNS, [])
    _write_csv(paths["execution_fill_quality"], EXECUTION_FILL_QUALITY_COLUMNS, [])
    _write_csv(paths["live_chain_validation"], ("ticker", "live_validation_status", "note"), [])
    _write_csv(
        paths["decision_board"],
        (
            "recommendation_rank",
            "ticker",
            "status_icon",
            "status_label",
            "final_action",
            "execution_status",
            "execution_gate_status",
            "execution_blockers",
            "execution_confidence_score",
            "execution_confidence_rating",
            "trade_quality_confidence_rating",
            "external_agent_review_count",
            "external_agent_distinct_review_count",
            "external_agent_review_agents",
            "underlying_quality_tier",
            "underlying_quality_reason",
            "target_order_status",
            "live_validation_status",
            "trade_plan",
            "expiry",
            "sell_leg",
            "buy_leg",
            "synthesis_score",
            "suggested_contracts",
            "credit_width_ratio",
            "trade_quality_status",
            "quality_gate_reason",
            "visible_in_final_board",
        ),
        [],
    )
    _write_csv(
        paths["green_trade_tickets"],
        (
            "recommendation_rank",
            "ticker",
            "status_icon",
            "status_label",
            "action",
            "order_readiness",
            "ready_to_enter",
            "execution_status",
            "execution_confidence_score",
            "execution_confidence_rating",
            "trade_quality_confidence_rating",
            "external_agent_review_count",
            "external_agent_distinct_review_count",
            "external_agent_review_agents",
            "underlying_quality_tier",
            "underlying_quality_reason",
            "target_order_status",
            "execution_blockers",
            "suggested_contracts",
            "trade_plan",
            "expiry",
            "sell_leg",
            "buy_leg",
            "entry_limit",
            "entry_type",
            "target_exit",
            "invalidation",
            "credit_width_ratio",
            "trade_quality_status",
            "quality_gate_reason",
            "max_position_loss",
            "account_risk_pct",
            "synthesis_score",
            "live_validation_status",
        ),
        [],
    )
    _write_csv(
        paths["target_order_candidates"],
        (
            "recommendation_rank",
            "ticker",
            "status_icon",
            "status_label",
            "action",
            "order_readiness",
            "ready_to_enter",
            "execution_status",
            "execution_confidence_score",
            "execution_confidence_rating",
            "trade_quality_confidence_rating",
            "external_agent_review_count",
            "external_agent_distinct_review_count",
            "external_agent_review_agents",
            "underlying_quality_tier",
            "underlying_quality_reason",
            "target_order_status",
            "execution_blockers",
            "suggested_contracts",
            "trade_plan",
            "expiry",
            "sell_leg",
            "buy_leg",
            "entry_limit",
            "entry_type",
            "target_exit",
            "invalidation",
            "credit_width_ratio",
            "trade_quality_status",
            "quality_gate_reason",
            "max_position_loss",
            "account_risk_pct",
            "synthesis_score",
            "live_validation_status",
        ),
        [],
    )
    _write_csv(
        paths["trade_tickets"],
        (
            "recommendation_rank",
            "ticker",
            "status_icon",
            "status_label",
            "action",
            "order_readiness",
            "ready_to_enter",
            "execution_status",
            "execution_confidence_score",
            "execution_confidence_rating",
            "trade_quality_confidence_rating",
            "external_agent_review_count",
            "external_agent_distinct_review_count",
            "external_agent_review_agents",
            "underlying_quality_tier",
            "underlying_quality_reason",
            "target_order_status",
            "execution_blockers",
            "suggested_contracts",
            "trade_plan",
            "expiry",
            "sell_leg",
            "buy_leg",
            "entry_limit",
            "entry_type",
            "target_exit",
            "invalidation",
            "credit_width_ratio",
            "trade_quality_status",
            "quality_gate_reason",
            "max_position_loss",
            "account_risk_pct",
            "synthesis_score",
            "live_validation_status",
        ),
        [],
    )
    _write_csv(
        paths["market_open_recheck_queue"],
        MARKET_OPEN_RECHECK_COLUMNS,
        [],
    )
    _write_csv(paths["execution_readiness"], ("gate", "status", "detail", "affected_rows"), [])
    _write_csv(paths["expectancy_evidence"], EXPECTANCY_EVIDENCE_COLUMNS, [])
    _write_csv(paths["profitability_calibration"], PROFITABILITY_CALIBRATION_COLUMNS, [])
    _write_csv(paths["profitability_gap_plan"], PROFITABILITY_GAP_PLAN_COLUMNS, [])
    _write_csv(paths["route_opportunity_gap"], ROUTE_OPPORTUNITY_GAP_COLUMNS, [])
    _write_csv(paths["monthly_feasibility"], ("metric", "value", "status", "note"), [])
    _write_frame(
        build_application_audit(pd.DataFrame(), pd.DataFrame(), lesson_pack),
        paths["lessons_application_audit"],
    )
    _write_csv(
        paths["coverage_audit"],
        (
            "ticker",
            "coverage_status",
            "status_color",
            "raw_rank",
            "candidate_rank",
            "bias",
            "score",
            "quality_status",
            "underlying_quality_tier",
            "underlying_quality_reason",
            "marketcap",
            "avg30_volume",
            "total_open_interest",
            "signal_premium",
            "final_action",
            "execution_status",
            "trade_plan",
            "reason",
            "next_step",
        ),
        [],
    )
    _write_csv(
        paths["final_recommendations"],
        (
            "ticker",
            "structure",
            "recommendation_status",
            "portfolio_risk_flag",
            "portfolio_risk_note",
            "suggested_contracts",
            "max_position_loss",
            "account_risk_pct",
            "sizing_note",
            "visible_in_final_board",
        ),
        [],
    )
    _write_csv(paths["no_trade_audit"], ("ticker", "reason", "hard_blocker"), [])
    _write_csv(paths["risk_audit"], ("ticker", "risk_type", "risk_note", "visibility_action"), [])
    _write_csv(
        paths["sizing_audit"],
        (
            "ticker",
            "suggested_contracts",
            "risk_budget",
            "max_loss",
            "max_position_loss",
            "account_risk_pct",
            "buying_power_effect",
            "sizing_risk_flag",
            "sizing_note",
            "visibility_action",
        ),
        [],
    )
    _write_csv(
        paths["management_plan"],
        (
            "recommendation_rank",
            "ticker",
            "management_action",
            "entry_condition",
            "entry_limit",
            "suggested_contracts",
            "target_exit",
            "max_profit",
            "max_loss",
            "max_position_loss",
            "invalidation",
            "review_triggers",
            "management_note",
            "visibility_action",
        ),
        [],
    )
    _write_json(paths["portfolio_context"], unavailable_portfolio_context("design smoke slice"))

    for agent in agent_roster():
        review_path = paths["agent_reviews_dir"] / f"{agent['name']}.md"
        review_path.write_text(
            f"# {agent['name']}\n\nStatus: not run in design smoke slice.\n",
            encoding="utf-8",
        )

    return paths


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the independent Options Agent scaffold.")
    parser.add_argument("--date", required=True, help="As-of date in YYYY-MM-DD format.")
    parser.add_argument(
        "--base-dir",
        default=str(project_root()),
        help="Trade desk root. Defaults to the detected project root.",
    )
    parser.add_argument("--out-dir", default="", help="Optional output directory override.")
    parser.add_argument(
        "--top-trades",
        type=int,
        default=DEFAULT_TOP_TRADES,
        help=(
            "Legacy compatibility knob; does not cap candidate pricing, final recommendations, "
            "or no-trade audit visibility."
        ),
    )
    parser.add_argument(
        "--max-bot-rows",
        type=int,
        default=0,
        help="Optional bot-EOD row cap for fast development runs. 0 means no cap.",
    )
    parser.add_argument(
        "--design-smoke",
        action="store_true",
        help="Write the current design/smoke artifacts without live data collection.",
    )
    parser.add_argument(
        "--live-schwab",
        action="store_true",
        help="Fetch live Schwab chains and replace dated pricing where possible.",
    )
    parser.add_argument(
        "--chain-snapshot-dir",
        default="",
        help="Optional directory of Schwab chain JSON snapshots for validation without live fetches.",
    )
    parser.add_argument(
        "--chain-strike-count",
        type=int,
        default=80,
        help="Strike count for live Schwab chain validation.",
    )
    parser.add_argument(
        "--portfolio-json",
        default="",
        help="Optional portfolio context JSON to annotate recommendations.",
    )
    parser.add_argument(
        "--live-portfolio",
        action="store_true",
        help="Fetch Schwab positions and annotate portfolio risk without suppressing rows.",
    )
    parser.add_argument(
        "--agent-reviews-json",
        default="",
        help="Optional JSON reviews from external/subagents to feed into synthesis.",
    )
    parser.add_argument(
        "--dispatch-only",
        action="store_true",
        help="Write pass-1 agent dispatch artifacts, then stop before pricing and synthesis.",
    )
    parser.add_argument(
        "--lesson-pack-version",
        default="",
        help="Optional promoted lesson-pack version for regression comparisons. Normal runs omit this and use the active pointer.",
    )
    parser.add_argument(
        "--lesson-pack-path",
        default="",
        help="Optional lesson-pack path for candidate regression comparisons. Normal runs omit this and use the active pointer.",
    )
    args = parser.parse_args(argv)

    root = Path(args.base_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else None
    chain_snapshot_dir = Path(args.chain_snapshot_dir).expanduser().resolve() if args.chain_snapshot_dir else None
    portfolio_json = Path(args.portfolio_json).expanduser().resolve() if args.portfolio_json else None
    agent_reviews_json = Path(args.agent_reviews_json).expanduser().resolve() if args.agent_reviews_json else None
    lesson_pack_path = Path(args.lesson_pack_path).expanduser().resolve() if args.lesson_pack_path else None
    if args.design_smoke:
        paths = run_design_smoke(args.date, root=root, out_dir=out_dir)
        print(f"Wrote Options Agent smoke artifacts to {paths['out_dir']}")
        return 0
    paths = run_pipeline(
        args.date,
        root=root,
        out_dir=out_dir,
        top_trades=args.top_trades,
        max_bot_rows=args.max_bot_rows or None,
        portfolio_json=portfolio_json,
        live_portfolio=args.live_portfolio,
        live_schwab=args.live_schwab,
        chain_snapshot_dir=chain_snapshot_dir,
        chain_strike_count=args.chain_strike_count,
        agent_reviews_json=agent_reviews_json,
        dispatch_only=args.dispatch_only,
        lesson_pack_version=args.lesson_pack_version or None,
        lesson_pack_path=lesson_pack_path,
    )
    if args.dispatch_only:
        print(f"Wrote Options Agent dispatch artifacts to {paths['out_dir']}")
    else:
        print(f"Wrote Options Agent recommendations to {paths['out_dir']}")
        print(f"Report markdown: {paths['report']}")
        print(f"Trade tickets: {paths['trade_tickets']}")
        print(f"Green send-now tickets: {paths['green_trade_tickets']}")
        print(f"Yellow target candidates: {paths['target_order_candidates']}")
    return 0


def _combined_flow_bias(row: Mapping[str, Any]) -> float:
    weighted: list[tuple[float, float]] = []
    for bias_col, premium_col in (
        ("bot_flow_bias", "bot_total_premium"),
        ("hot_flow_bias", "hot_total_premium"),
        ("screen_flow_bias", "screen_total_premium"),
    ):
        bias = _as_float(row.get(bias_col))
        premium = _as_float(row.get(premium_col)) or 0.0
        if bias is None or not math.isfinite(bias) or premium <= 0:
            continue
        weighted.append((bias, max(math.log10(premium + 1), 1.0)))
    if not weighted:
        return 0.0
    total_weight = sum(weight for _, weight in weighted)
    return sum(bias * weight for bias, weight in weighted) / total_weight


def _bias_label(value: Any) -> str:
    bias = _as_float(value) or 0.0
    if bias >= 0.08:
        return "bullish"
    if bias <= -0.08:
        return "bearish"
    return "neutral"


def _score_universe_row(row: Mapping[str, Any]) -> float:
    bias = abs(_as_float(row.get("combined_flow_bias")) or 0.0)
    signal_premium = _as_float(row.get("signal_premium")) or 0.0
    hot_volume = _as_float(row.get("hot_volume")) or 0.0
    hot_oi = _as_float(row.get("hot_open_interest")) or 0.0
    oi_change = _as_float(row.get("positive_oi_change")) or 0.0
    iv_rank = _as_float(row.get("iv_rank"))
    premium_score = min(30.0, math.log10(signal_premium + 1) * 3.6) if signal_premium > 0 else 0.0
    bias_score = min(30.0, bias * 55.0)
    liquidity_score = min(15.0, math.log10(hot_volume + 1) * 2.2 + math.log10(hot_oi + 1) * 0.8)
    oi_score = min(15.0, math.log10(oi_change + 1) * 2.0) if oi_change > 0 else 0.0
    iv_score = 5.0 if iv_rank is not None and iv_rank >= 45 else 0.0
    underlying_tier = _as_text(row.get("underlying_quality_tier")).lower()
    quality_adjustment = UNDERLYING_QUALITY_SCORE_ADJUSTMENT.get(underlying_tier, 0.0)
    return round(premium_score + bias_score + liquidity_score + oi_score + iv_score + quality_adjustment, 2)


def _quality_status(row: Mapping[str, Any]) -> str:
    if row.get("bias") == "neutral":
        return "watch"
    if _as_text(row.get("underlying_quality_tier")).lower() != "core":
        return "watch"
    return "qualified" if (_as_float(row.get("score")) or 0.0) >= 55.0 else "watch"


def _flow_reason(row: Mapping[str, Any]) -> str:
    bias = row.get("bias")
    score = _as_float(row.get("score")) or 0.0
    premium = _as_float(row.get("signal_premium")) or 0.0
    flow_bias = _as_float(row.get("combined_flow_bias")) or 0.0
    tier = _as_text(row.get("underlying_quality_tier")) or "unknown"
    return f"{bias} flow bias {flow_bias:.2f}; signal premium ${premium:,.0f}; {tier} underlying; score {score:.1f}"


def _status_counts(df: pd.DataFrame) -> dict[str, int]:
    if df.empty or "recommendation_status" not in df.columns:
        return {}
    return {str(key): int(value) for key, value in df["recommendation_status"].value_counts().sort_index().items()}


def _value_counts(df: pd.DataFrame, column: str) -> dict[str, int]:
    if df.empty or column not in df.columns:
        return {}
    return {str(key): int(value) for key, value in df[column].value_counts().sort_index().items()}


def _review_summary_by_ticker(agent_review_board: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if agent_review_board.empty or "ticker" not in agent_review_board.columns:
        return {}
    summary: dict[str, dict[str, Any]] = {}
    for ticker, group in agent_review_board.groupby(agent_review_board["ticker"].astype(str).str.upper()):
        verdicts = group["verdict"].astype(str).str.lower() if "verdict" in group.columns else pd.Series(dtype=str)
        objective = group["objective_blocker"].astype(bool) if "objective_blocker" in group.columns else pd.Series(dtype=bool)
        portfolio_only = group["portfolio_risk_only"].astype(bool) if "portfolio_risk_only" in group.columns else pd.Series(dtype=bool)
        agents = group["agent"].astype(str).str.lower() if "agent" in group.columns else pd.Series(dtype=str)
        process_only = group.apply(lambda review: _is_portfolio_management_process_note(review.to_dict()), axis=1)
        quality_reviews = ~portfolio_only & ~agents.eq("portfolio_risk") & ~process_only.astype(bool)
        summary[str(ticker)] = {
            "supportive": int((verdicts.eq("supportive") & quality_reviews).sum()),
            "caution": int((verdicts.eq("caution") & quality_reviews).sum()),
            "avoid": int(verdicts.eq("avoid").sum()),
            "objective_blockers": int(objective.sum()),
            "portfolio_risk_only": int(portfolio_only.sum()),
        }
    return summary


def _synthesis_score(
    row: Mapping[str, Any],
    review_summary: Mapping[str, Any],
    execution_context: Optional[Mapping[str, Any]] = None,
) -> tuple[float, str]:
    context = _execution_context_or_default(execution_context)
    base = _as_float(row.get("score")) or 0.0
    status = str(row.get("recommendation_status") or "").strip().upper()
    quality = str(row.get("quality_status") or "").strip().lower()
    hard_rejects = str(row.get("hard_rejects") or "").strip()
    live_status = str(row.get("live_validation_status") or "").strip().upper()
    ticket = str(row.get("full_ticket") or "").strip()
    entry_limit = _as_float(row.get("entry_limit"))
    underlying_tier = str(row.get("underlying_quality_tier") or "").strip().lower()
    support = int(review_summary.get("supportive", 0) or 0)
    caution = int(review_summary.get("caution", 0) or 0)
    objective = int(review_summary.get("objective_blockers", 0) or 0)
    portfolio_only = int(review_summary.get("portfolio_risk_only", 0) or 0)
    expectancy_status = str(row.get("actual_forward_expectancy_status") or "").strip().upper()
    expectancy_sample = int(_as_float(row.get("actual_forward_expectancy_sample_size")) or 0)
    strategy_expectancy_status = str(row.get("actual_forward_strategy_expectancy_status") or "").strip().upper()
    strategy_expectancy_sample = int(_as_float(row.get("actual_forward_strategy_expectancy_sample_size")) or 0)
    suggested_contracts = int(_as_float(row.get("suggested_contracts")) or 0)
    position_profit = _position_max_profit_value(row, suggested_contracts)
    materiality_floor = _green_position_profit_floor(context)
    ticker_expectancy_negative = _expectancy_values_are_negative(row, "actual_forward_expectancy")
    strategy_expectancy_negative = _expectancy_values_are_negative(row, "actual_forward_strategy_expectancy")

    adjustment = 0.0
    reasons: list[str] = [f"base flow score {base:.1f}"]
    if quality == "qualified":
        adjustment += 5.0
        reasons.append("qualified setup +5")
    else:
        adjustment -= 10.0
        reasons.append("watch-quality setup -10")

    if underlying_tier == "core":
        adjustment += 12.0
        reasons.append("core liquid underlying +12")
    elif underlying_tier == "liquid":
        adjustment += 6.0
        reasons.append("liquid underlying +6")
    elif underlying_tier == "speculative":
        adjustment -= 30.0
        reasons.append("speculative underlying -30")
    elif underlying_tier == "excluded":
        adjustment -= 60.0
        reasons.append("non-actionable underlying -60")

    if status == RecommendationStatus.ENTER.value:
        adjustment += 25.0
        reasons.append("entry status +25")
    elif status == RecommendationStatus.ENTER_WITH_PORTFOLIO_RISK.value:
        adjustment += 25.0
        reasons.append("entry status +25")
    elif status == RecommendationStatus.WAIT_FOR_PRICE.value:
        adjustment -= 5.0
        reasons.append("waiting for price -5")
    elif status == RecommendationStatus.REVIEW.value:
        adjustment -= 12.0
        reasons.append("manual review required -12")
    elif status == RecommendationStatus.AVOID.value:
        adjustment -= 80.0
        reasons.append("avoid status -80")

    if live_status == "PASS":
        adjustment += 20.0
        reasons.append("live validation PASS +20")
    elif status in {RecommendationStatus.ENTER.value, RecommendationStatus.ENTER_WITH_PORTFOLIO_RISK.value}:
        adjustment -= 35.0
        reasons.append("entry status without live validation -35")

    if ticket and entry_limit is not None and entry_limit > 0:
        adjustment += 5.0
        reasons.append("executable ticket math +5")
    elif status in {RecommendationStatus.ENTER.value, RecommendationStatus.ENTER_WITH_PORTFOLIO_RISK.value}:
        adjustment -= 30.0
        reasons.append("entry status without executable ticket -30")

    if status in {RecommendationStatus.ENTER.value, RecommendationStatus.ENTER_WITH_PORTFOLIO_RISK.value}:
        if 0 < position_profit < materiality_floor and not _materiality_exception_reason(row):
            adjustment -= 25.0
            reasons.append(
                f"position max profit ${position_profit:.0f} below ${materiality_floor:.0f} floor -25"
            )

    if expectancy_status == "PASS":
        adjustment += 4.0
        reasons.append(f"positive ticker actual/forward expectancy +4 ({expectancy_sample} samples)")
    elif expectancy_status == "WARN":
        if ticker_expectancy_negative:
            adjustment -= 8.0
            reasons.append(f"negative ticker actual/forward expectancy -8 ({expectancy_sample} samples)")
        else:
            adjustment += 1.0
            reasons.append(f"thin ticker actual/forward expectancy +1 ({expectancy_sample} samples)")
    elif expectancy_status == "BLOCK":
        adjustment -= 4.0
        reasons.append("missing/weak ticker actual-forward expectancy -4")

    if strategy_expectancy_status == "PASS":
        adjustment += 10.0
        reasons.append(f"positive structure-aligned actual/forward expectancy +10 ({strategy_expectancy_sample} samples)")
    elif strategy_expectancy_status == "WARN":
        if strategy_expectancy_negative:
            adjustment -= 16.0
            reasons.append(
                f"negative structure-aligned actual/forward expectancy -16 ({strategy_expectancy_sample} samples)"
            )
        else:
            adjustment += 2.0
            reasons.append(f"thin structure-aligned actual/forward expectancy +2 ({strategy_expectancy_sample} samples)")
    elif strategy_expectancy_status == "BLOCK":
        adjustment -= 12.0
        reasons.append("missing/weak structure-aligned actual-forward expectancy -12")
    if _negative_strategy_expectancy_blocks_green(row):
        adjustment -= 30.0
        reasons.append("negative vertical-spread realized history blocks green -30")
    if status in {RecommendationStatus.ENTER.value, RecommendationStatus.ENTER_WITH_PORTFOLIO_RISK.value} and not _positive_strategy_expectancy_ready_for_green(row):
        adjustment -= 25.0
        reasons.append("positive structure-aligned expectancy required for green -25")

    if hard_rejects:
        adjustment -= 80.0
        reasons.append("objective hard reject -80")
    if support:
        support_adj = min(float(support), 6.0)
        adjustment += support_adj
        reasons.append(f"{support} supportive agent reviews +{support_adj:.0f}")
    if caution:
        caution_adj = min(float(caution) * 3.0, 24.0)
        adjustment -= caution_adj
        reasons.append(f"{caution} caution reviews -{caution_adj:.0f}")
    if objective:
        objective_adj = float(objective) * 60.0
        adjustment -= objective_adj
        reasons.append(f"{objective} objective blocker reviews -{objective_adj:.0f}")
    if portfolio_only:
        reasons.append(f"{portfolio_only} account-context review(s) kept audit-only +0")

    return round(base + adjustment, 2), "; ".join(reasons)


def _agent_review_row(
    ticker: str,
    agent: str,
    review_stage: str,
    verdict: str,
    confidence: str,
    note: str,
    *,
    agent_type: str = "built_in",
    objective_blocker: bool = False,
    blocker_type: str = "",
    portfolio_risk_only: bool = False,
    evidence: str = "",
    source_artifact: str = "",
    as_of: str = "",
) -> dict[str, Any]:
    ticker_text = str(ticker or "").strip().upper()
    agent_text = str(agent or "external").strip() or "external"
    stage_text = str(review_stage or agent_text).strip()
    return {
        "candidate_id": f"{ticker_text}:{agent_text}:{stage_text}",
        "ticker": ticker_text,
        "agent": agent_text,
        "agent_type": str(agent_type or "built_in").strip(),
        "review_stage": stage_text,
        "verdict": str(verdict or "caution").strip().lower(),
        "confidence": str(confidence or "medium").strip().lower(),
        "objective_blocker": bool(objective_blocker),
        "blocker_type": str(blocker_type or "").strip(),
        "portfolio_risk_only": bool(portfolio_risk_only),
        "note": str(note or "").strip(),
        "evidence": str(evidence or "").strip(),
        "source_artifact": str(source_artifact or "").strip(),
        "as_of": str(as_of or "").strip(),
    }


def _normalize_agent_review(review: Mapping[str, Any], *, as_of: str) -> dict[str, Any]:
    ticker = str(review.get("ticker") or "").strip().upper()
    agent = str(review.get("agent") or "external").strip() or "external"
    agent_type = str(review.get("agent_type") or "external").strip()
    review_stage = str(review.get("review_stage") or "external_review").strip()
    portfolio_risk_value = review.get("portfolio_risk_only")
    if "portfolio_risk_only" in review and portfolio_risk_value is not None and str(portfolio_risk_value).strip():
        portfolio_only = _truthy(portfolio_risk_value)
    else:
        portfolio_only = _is_portfolio_risk_review(review)
    objective = _truthy(review.get("objective_blocker"))
    blocker_type = str(review.get("blocker_type") or "").strip()
    if portfolio_only and not blocker_type:
        blocker_type = "portfolio"
    elif objective and not blocker_type:
        blocker_type = "objective"
    return {
        "candidate_id": str(review.get("candidate_id") or f"{ticker}:{agent}:{review_stage}").strip(),
        "ticker": ticker,
        "agent": agent,
        "agent_type": agent_type,
        "review_stage": review_stage,
        "verdict": str(review.get("verdict") or "caution").strip().lower(),
        "confidence": str(review.get("confidence") or "medium").strip().lower(),
        "objective_blocker": objective,
        "blocker_type": blocker_type,
        "portfolio_risk_only": portfolio_only,
        "note": str(review.get("note") or "").strip(),
        "evidence": str(review.get("evidence") or "").strip(),
        "source_artifact": str(review.get("source_artifact") or "external_agent_reviews.csv").strip(),
        "as_of": str(review.get("as_of") or as_of).strip(),
    }


def _frame_by_ticker(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if df.empty or "ticker" not in df.columns:
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        ticker = str(row.get("ticker") or "").strip().upper()
        if ticker and ticker not in rows:
            rows[ticker] = row.to_dict()
    return rows


def _dedupe_ticker_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "ticker" not in df.columns:
        return df.copy()
    out = df.copy()
    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
    out = out[out["ticker"].ne("")]
    return out.drop_duplicates("ticker", keep="first")


def _ordered_review_tickers(primary: pd.DataFrame, fallback: pd.DataFrame) -> list[str]:
    tickers: list[str] = []
    for df in (primary, fallback):
        if df.empty or "ticker" not in df.columns:
            continue
        for value in df["ticker"].tolist():
            ticker = str(value or "").strip().upper()
            if ticker and ticker not in tickers:
                tickers.append(ticker)
    return tickers


def _row_date(value: Any) -> Optional[dt.date]:
    if value is None:
        return None
    if pd.isna(value):
        return None
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "nat", "none"}:
        return None
    try:
        return dt.date.fromisoformat(text[:10])
    except ValueError:
        return None


def _credit_direction(row: Mapping[str, Any]) -> str:
    bias = str(row.get("bias") or "").strip().lower()
    structure = " ".join(
        str(row.get(key) or "").strip().lower()
        for key in ("structure", "trade_plan", "full_ticket", "sell_leg", "buy_leg")
        if str(row.get(key) or "").strip()
    )
    if bias == "bullish":
        return "Bull Put"
    if bias == "bearish":
        return "Bear Call"
    if "put" in structure and "call" not in structure:
        return "Bull Put"
    if "call" in structure and "put" not in structure:
        return "Bear Call"
    if "bull put" in structure:
        return "Bull Put"
    if "bear call" in structure:
        return "Bear Call"
    if "call" in structure:
        return "Bear Call"
    return "Bear Call"


def _debit_direction(row: Mapping[str, Any]) -> str:
    bias = str(row.get("bias") or "").strip().lower()
    structure = str(row.get("structure") or "").strip().lower()
    if bias == "bullish":
        return "Bull Call"
    if bias == "bearish":
        return "Bear Put"
    if "call debit" in structure or "bull call" in structure:
        return "Bull Call"
    return "Bear Put"


def _route_prefers_debit_spread(row: Mapping[str, Any]) -> bool:
    strategy = _as_text(row.get("strategy_route") or row.get("strategy")).lower()
    structure = _as_text(row.get("structure")).lower()
    ticket = _as_text(row.get("trade_plan") or row.get("full_ticket")).upper()
    return bool(
        strategy in {"bull_call_debit", "bear_put_debit"}
        or "debit spread" in structure
        or " DEBIT" in ticket
    )


def _direction_right(direction: str) -> str:
    return "P" if direction in {"Bull Put", "Bear Put"} else "C"


def _select_live_expiry(
    contracts: pd.DataFrame,
    asof_date: dt.date,
    preferred_expiry: Optional[dt.date],
    direction: str,
) -> Optional[dt.date]:
    if contracts.empty:
        return None
    right = _direction_right(direction)
    expiries = sorted(
        {
            expiry
            for expiry in contracts.loc[contracts["right"].astype(str).eq(right), "expiry"].dropna().tolist()
            if isinstance(expiry, dt.date)
        }
    )
    if not expiries:
        return None
    if preferred_expiry in expiries and _live_expiry_in_range(asof_date, preferred_expiry):
        return preferred_expiry
    candidates = [expiry for expiry in expiries if _live_expiry_in_range(asof_date, expiry)]
    if candidates:
        return min(candidates, key=lambda expiry: abs((expiry - asof_date).days - 35))
    return None


def _find_short_put_alternatives(
    contracts: pd.DataFrame,
    *,
    expiry: dt.date,
    spot: float,
    expected_move_pct: Optional[float] = None,
    max_alternatives: int = 5,
) -> list[dict[str, Any]]:
    if contracts.empty or not math.isfinite(spot) or spot <= 0:
        return [{"live_status": "no_usable_chain", "live_blocker": "missing contracts or spot for short-put construction"}]
    chain = contracts[(contracts["expiry"] == expiry) & (contracts["right"].astype(str).eq("P"))].copy()
    if chain.empty:
        return [{"live_status": "missing_expiry_or_right", "live_blocker": f"no P contracts for {expiry}"}]
    for column in ["strike", "bid", "ask", "mark", "delta", "open_interest", "volume"]:
        chain[column] = pd.to_numeric(chain[column], errors="coerce")
    rows: list[dict[str, Any]] = []
    for _, put in chain.iterrows():
        strike = _as_float(put.get("strike"))
        if strike is None or strike <= 0 or strike >= spot:
            continue
        distance_pct = (spot - strike) / spot
        if distance_pct < MIN_SHORT_PUT_DISTANCE_PCT:
            continue
        delta = _as_float(put.get("delta"))
        delta_abs = abs(delta) if delta is not None else math.nan
        if math.isfinite(delta_abs) and not (MIN_SHORT_PUT_DELTA <= delta_abs <= MAX_SHORT_PUT_DELTA):
            continue
        bid = _as_float(put.get("bid")) or 0.0
        ask = _as_float(put.get("ask")) or 0.0
        mark = _as_float(put.get("mark")) or 0.0
        mid = (bid + ask) / 2 if bid or ask else mark
        credit = bid if bid > 0 else mid
        if credit <= 0:
            continue
        quote_width_pct = ((ask - bid) / mid) if mid > 0 and ask >= bid else math.nan
        liq = (_as_float(put.get("open_interest")) or 0.0) + (_as_float(put.get("volume")) or 0.0)
        expected_ratio = _short_put_expected_move_ratio(distance_pct, expected_move_pct)
        pop = 1.0 - delta_abs if math.isfinite(delta_abs) else math.nan
        rows.append(
            {
                "live_status": "PASS",
                "short_leg": put.get("symbol", ""),
                "long_leg": "",
                "short_strike": strike,
                "long_strike": "",
                "spread_width": "",
                "credit": round(credit, 2),
                "mid_credit": round(mid, 2),
                "natural_credit": round(bid, 2),
                "sell_leg_bid": bid,
                "sell_leg_ask": ask,
                "sell_leg_mid": mid,
                "buy_leg_bid": "",
                "buy_leg_ask": "",
                "buy_leg_mid": "",
                "pop_delta_proxy": pop,
                "short_delta": delta,
                "distance_pct": distance_pct,
                "expected_move_ratio": expected_ratio,
                "short_oi": _as_float(put.get("open_interest")) or 0.0,
                "short_volume": _as_float(put.get("volume")) or 0.0,
                "long_oi": "",
                "long_volume": "",
                "quote_width_pct": quote_width_pct,
                "liq_score": liq,
                "target_entry": max(MIN_SEND_NOW_CREDIT, round(credit, 2)),
                "construction_source": "short_put_cash_secured",
                "construction_reason": "best liquid OTM put from live Schwab chain with positive short-put strategy evidence",
                "liquidity_summary": (
                    f"short oi+vol {liq:.0f}; quote width "
                    f"{quote_width_pct:.1%}" if math.isfinite(quote_width_pct) else f"short oi+vol {liq:.0f}; quote width unavailable"
                ),
            }
        )
    if not rows:
        return [{"live_status": "no_realistic_short_put", "live_blocker": "no OTM put with positive credit/delta/liquidity"}]
    df = pd.DataFrame(rows)
    delta_penalty = (pd.to_numeric(df["short_delta"], errors="coerce").abs().fillna(0.22) - 0.22).abs().clip(upper=0.20)
    df["_rank"] = (
        pd.to_numeric(df["credit"], errors="coerce").fillna(0.0).clip(upper=10.0)
        + pd.to_numeric(df["distance_pct"], errors="coerce").fillna(0.0).clip(upper=0.20) * 12.0
        + pd.to_numeric(df["pop_delta_proxy"], errors="coerce").fillna(0.55)
        + (pd.to_numeric(df["liq_score"], errors="coerce").fillna(0.0).clip(upper=5000.0) / 5000.0)
        - pd.to_numeric(df["quote_width_pct"], errors="coerce").fillna(0.0).clip(upper=2.0)
        - delta_penalty * 2.0
    )
    out: list[dict[str, Any]] = []
    for _, row in df.sort_values("_rank", ascending=False).head(max_alternatives).iterrows():
        out.append(row.drop(labels=[c for c in row.index if str(c).startswith("_")], errors="ignore").to_dict())
    return out


def _short_put_expected_move_ratio(distance_pct: float, expected_move_pct: Optional[float]) -> float:
    expected = _as_float(expected_move_pct)
    distance = _as_float(distance_pct)
    if expected is None or expected <= 0 or distance is None:
        return math.nan
    return distance / expected


def _select_live_short_put_alternative(
    row: Mapping[str, Any],
    alternatives: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    valid = [dict(alt) for alt in alternatives if isinstance(alt, Mapping)]
    if not valid:
        return {"live_status": "no_live_alternative", "live_blocker": "no alternative returned"}

    def score(alt: Mapping[str, Any]) -> tuple[float, ...]:
        if str(alt.get("live_status") or "").upper() != "PASS":
            return (-10.0, 0.0, 0.0, 0.0, 0.0)
        credit = _as_float(alt.get("credit")) or 0.0
        strike = _as_float(alt.get("short_strike")) or 0.0
        max_loss = max((strike - credit) * 100.0, 0.0)
        rejects = _short_put_quality_rejects(
            entry_credit=credit,
            max_loss=max_loss,
            signal_premium=_as_float(row.get("signal_premium")) or 0.0,
            combined_flow_bias=_as_float(row.get("combined_flow_bias")) or 0.0,
            macro_tape_candidate=_truthy(row.get("macro_tape_candidate")),
            spot=_as_float(row.get("spot_live")) or _as_float(row.get("close")),
            short_strike=strike,
        )
        rejects.extend(_live_spread_quality_rejects(alt))
        quality_pass = 1.0 if not rejects else 0.0
        target_entry = _as_float(alt.get("target_entry")) or MIN_SEND_NOW_CREDIT
        target_met = 1.0 if quality_pass and credit >= target_entry else 0.0
        liq = _as_float(alt.get("liq_score")) or 0.0
        distance = _as_float(alt.get("distance_pct")) or 0.0
        quote_width = _as_float(alt.get("quote_width_pct"))
        quote_score = -(quote_width if quote_width is not None else 9.0)
        rank = _as_float(alt.get("_rank")) or 0.0
        return (quality_pass, target_met, credit, distance, min(liq, 10_000.0), quote_score, rank)

    return max(valid, key=score)


def _live_expiry_in_range(asof_date: dt.date, expiry: Optional[dt.date]) -> bool:
    if expiry is None:
        return False
    dte = (expiry - asof_date).days
    return MIN_LIVE_DTE <= dte <= MAX_LIVE_DTE


def _preferred_width(row: Mapping[str, Any]) -> Optional[float]:
    short_strike = _as_float(row.get("short_strike"))
    long_strike = _as_float(row.get("long_strike"))
    if short_strike is not None and long_strike is not None:
        width = abs(short_strike - long_strike)
        if width > 0:
            return width
    max_loss = _as_float(row.get("max_loss"))
    max_profit = _as_float(row.get("max_profit"))
    if max_loss is not None and max_profit is not None:
        width = (max_loss + max_profit) / 100.0
        if width > 0:
            return width
    return None


def _has_complete_target_math(row: Mapping[str, Any]) -> bool:
    ticket = _as_text(row.get("trade_plan")) or _as_text(row.get("full_ticket"))
    entry = _as_float(row.get("entry_limit"))
    max_profit = _as_float(row.get("max_profit"))
    max_loss = _as_float(row.get("max_loss"))
    return bool(ticket) and entry is not None and entry > 0 and max_profit is not None and max_profit > 0 and max_loss is not None and max_loss > 0


def _preserve_market_closed_target_recheck(row: Mapping[str, Any], live_message: str) -> dict[str, Any]:
    out = dict(row)
    note = "preserve dated target credit/debit for the next fresh Schwab quote refresh"
    if live_message:
        note = _append_reason(note, live_message)
    out["live_validation_status"] = "TARGET_QUOTE_REFRESH"
    out["live_validation_note"] = note
    out["status_reason"] = _append_reason(out.get("status_reason"), note)
    return out


def _preserve_non_entry_status(row: Mapping[str, Any]) -> str:
    status = str(row.get("recommendation_status") or "").strip().upper()
    if status in {RecommendationStatus.ENTER.value, RecommendationStatus.ENTER_WITH_PORTFOLIO_RISK.value}:
        return RecommendationStatus.REVIEW.value
    return status or RecommendationStatus.REVIEW.value


def _apply_live_short_put(
    row: Mapping[str, Any],
    live: Mapping[str, Any],
    *,
    expiry: dt.date,
    spot: float,
    asof_date: dt.date,
    chain_source: str = "",
) -> dict[str, Any]:
    out = dict(row)
    credit = _as_float(live.get("credit")) or 0.0
    short_strike = _as_float(live.get("short_strike")) or 0.0
    target_entry = _as_float(live.get("target_entry")) or MIN_SEND_NOW_CREDIT
    max_profit = round(credit * 100, 2)
    max_loss = round(max((short_strike - credit) * 100, 0.0), 2)
    breakeven = short_strike - credit
    status = RecommendationStatus.ENTER.value if credit >= target_entry else RecommendationStatus.WAIT_FOR_PRICE.value
    source_label = "Schwab snapshot chain" if str(chain_source).startswith("snapshot:") else "live Schwab chain"
    note = (
        f"{source_label} short put validated at {credit:.2f} credit"
        if status == RecommendationStatus.ENTER.value
        else f"{source_label} found {credit:.2f} short-put credit below target {target_entry:.2f}"
    )
    hard_rejects = ""
    quality_rejects = _short_put_quality_rejects(
        entry_credit=credit,
        max_loss=max_loss,
        signal_premium=_as_float(row.get("signal_premium")) or 0.0,
        combined_flow_bias=_as_float(row.get("combined_flow_bias")) or 0.0,
        macro_tape_candidate=_truthy(row.get("macro_tape_candidate")),
        spot=spot,
        short_strike=short_strike,
    )
    quality_rejects.extend(_live_spread_quality_rejects(live))
    if quality_rejects:
        status = RecommendationStatus.AVOID.value
        hard_rejects = "; ".join(quality_rejects)
        note = _append_reason(note, "setup quality gate reject: " + hard_rejects)
    sell_leg = _format_option_leg(
        _as_text(live.get("short_leg")),
        "SELL",
        ticker=row.get("ticker", ""),
        expiry=expiry.isoformat(),
        strike=short_strike,
        right="P",
    )
    trade_plan = _format_trade_plan(sell_leg, "", credit, entry_type="CREDIT")
    out.update(
        {
            "strategy": "short_put",
            "strategy_family": "short_put",
            "strategy_route": "short_put",
            "entry_type": "CREDIT",
            "direction": "Short Put",
            "structure": "cash secured put",
            "full_ticket": trade_plan,
            "trade_plan": trade_plan,
            "expiry": expiry.isoformat(),
            "dte": (expiry - asof_date).days,
            "sell_leg": sell_leg,
            "buy_leg": "",
            "short_leg": sell_leg,
            "long_leg": "",
            "short_strike": short_strike,
            "long_strike": "",
            "spread_width": "",
            "entry_limit": round(credit, 2),
            "mid": live.get("mid_credit", ""),
            "bid": live.get("natural_credit", ""),
            "ask": live.get("sell_leg_ask", ""),
            "max_profit": max_profit,
            "max_loss": max_loss,
            "credit_width_ratio": "",
            "short_put_cash_required": max_loss,
            "trade_quality_status": "rejected" if quality_rejects else "reviewable",
            "quality_gate_reason": "; ".join(quality_rejects),
            "remaining_upside": max_profit,
            "breakeven": round(breakeven, 2),
            "target_exit": round(credit * 0.35, 2),
            "target_entry": target_entry,
            "invalidation": f"underlying violates breakeven {breakeven:.2f}, thesis breaks, or live quote degrades",
            "recommendation_status": status,
            "status_reason": note,
            "hard_rejects": hard_rejects,
            "spot_live": round(float(spot), 2),
            "live_pop_delta_proxy": live.get("pop_delta_proxy", ""),
            "live_short_delta": live.get("short_delta", ""),
            "live_distance_pct": live.get("distance_pct", ""),
            "live_quote_width_pct": live.get("quote_width_pct", ""),
            "live_short_oi": live.get("short_oi", ""),
            "live_short_volume": live.get("short_volume", ""),
            "live_long_oi": "",
            "live_long_volume": "",
            "live_leg_min_liquidity": _live_leg_min_liquidity_from_row(
                {
                    "live_short_oi": live.get("short_oi", ""),
                    "live_short_volume": live.get("short_volume", ""),
                    "live_long_oi": "",
                    "live_long_volume": "",
                }
            ),
            "live_liquidity_summary": live.get("liquidity_summary", ""),
            "construction_source": live.get("construction_source", ""),
            "construction_reason": live.get("construction_reason", ""),
        }
    )
    return out


def _apply_live_credit_spread(
    row: Mapping[str, Any],
    live: Mapping[str, Any],
    *,
    direction: str,
    expiry: dt.date,
    spot: float,
    asof_date: dt.date,
    chain_source: str = "",
) -> dict[str, Any]:
    out = dict(row)
    credit = _as_float(live.get("credit")) or 0.0
    width = _as_float(live.get("spread_width")) or 0.0
    short_strike = _as_float(live.get("short_strike")) or 0.0
    long_strike = _as_float(live.get("long_strike")) or 0.0
    target_entry = _as_float(live.get("target_entry")) or round(width * 0.18, 2)
    max_profit = round(credit * 100, 2)
    max_loss = round(max((width - credit) * 100, 0.0), 2)
    credit_width_ratio = round(credit / width, 4) if width > 0 else 0.0
    breakeven = short_strike - credit if direction == "Bull Put" else short_strike + credit
    status = RecommendationStatus.ENTER.value if credit >= target_entry else RecommendationStatus.WAIT_FOR_PRICE.value
    source_label = "Schwab snapshot chain" if str(chain_source).startswith("snapshot:") else "live Schwab chain"
    note = (
        f"{source_label} {direction} validated at {credit:.2f} credit"
        if status == RecommendationStatus.ENTER.value
        else f"{source_label} found {credit:.2f} credit below target {target_entry:.2f}"
    )
    hard_rejects = ""
    quality_rejects = _trade_quality_rejects(
        entry_credit=credit,
        credit_width_ratio=credit_width_ratio,
        max_loss=max_loss,
        signal_premium=_as_float(row.get("signal_premium")) or 0.0,
        combined_flow_bias=_as_float(row.get("combined_flow_bias")) or 0.0,
    )
    quality_rejects.extend(_live_spread_quality_rejects(live))
    if quality_rejects:
        status = RecommendationStatus.AVOID.value
        hard_rejects = "; ".join(quality_rejects)
        note = _append_reason(note, "setup quality gate reject: " + hard_rejects)
    right = "P" if direction == "Bull Put" else "C"
    short_symbol = _as_text(live.get("short_leg"))
    long_symbol = _as_text(live.get("long_leg"))
    sell_leg = _format_option_leg(
        short_symbol,
        "SELL",
        ticker=row.get("ticker", ""),
        expiry=expiry.isoformat(),
        strike=short_strike,
        right=right,
    )
    buy_leg = _format_option_leg(
        long_symbol,
        "BUY",
        ticker=row.get("ticker", ""),
        expiry=expiry.isoformat(),
        strike=long_strike,
        right=right,
    )
    trade_plan = _format_trade_plan(sell_leg, buy_leg, credit, entry_type="CREDIT")
    route = "bull_put_credit" if direction == "Bull Put" else "bear_call_credit"
    out.update(
        {
            "strategy": route,
            "strategy_family": "vertical_spread",
            "strategy_route": route,
            "entry_type": "CREDIT",
            "direction": direction,
            "structure": "bull put spread" if direction == "Bull Put" else "bear call spread",
            "full_ticket": trade_plan,
            "trade_plan": trade_plan,
            "expiry": expiry.isoformat(),
            "dte": (expiry - asof_date).days,
            "sell_leg": sell_leg,
            "buy_leg": buy_leg,
            "short_leg": sell_leg,
            "long_leg": buy_leg,
            "short_strike": short_strike,
            "long_strike": long_strike,
            "spread_width": width,
            "entry_limit": round(credit, 2),
            "mid": live.get("mid_credit", ""),
            "bid": live.get("natural_credit", ""),
            "ask": "",
            "max_profit": max_profit,
            "max_loss": max_loss,
            "credit_width_ratio": credit_width_ratio,
            "trade_quality_status": "rejected" if quality_rejects else "reviewable",
            "quality_gate_reason": "; ".join(quality_rejects),
            "remaining_upside": max_profit,
            "breakeven": round(breakeven, 2),
            "target_exit": round(credit * 0.35, 2),
            "target_entry": target_entry,
            "invalidation": f"underlying violates breakeven {breakeven:.2f}, thesis breaks, or live quote degrades",
            "recommendation_status": status,
            "status_reason": note,
            "hard_rejects": hard_rejects,
            "spot_live": round(float(spot), 2),
            "live_pop_delta_proxy": live.get("pop_delta_proxy", ""),
            "live_short_delta": live.get("short_delta", ""),
            "live_distance_pct": live.get("distance_pct", ""),
            "live_quote_width_pct": live.get("quote_width_pct", ""),
            "live_short_oi": live.get("short_oi", ""),
            "live_short_volume": live.get("short_volume", ""),
            "live_long_oi": live.get("long_oi", ""),
            "live_long_volume": live.get("long_volume", ""),
            "live_leg_min_liquidity": _live_leg_min_liquidity_from_row(
                {
                    "live_short_oi": live.get("short_oi", ""),
                    "live_short_volume": live.get("short_volume", ""),
                    "live_long_oi": live.get("long_oi", ""),
                    "live_long_volume": live.get("long_volume", ""),
                }
            ),
            "live_liquidity_summary": live.get("liquidity_summary", ""),
            "construction_source": live.get("construction_source", ""),
            "construction_reason": live.get("construction_reason", ""),
        }
    )
    return out


def _apply_live_debit_spread(
    row: Mapping[str, Any],
    live: Mapping[str, Any],
    *,
    direction: str,
    expiry: dt.date,
    spot: float,
    asof_date: dt.date,
    chain_source: str = "",
) -> dict[str, Any]:
    out = dict(row)
    debit = _as_float(live.get("debit")) or 0.0
    width = _as_float(live.get("spread_width")) or 0.0
    short_strike = _as_float(live.get("short_strike")) or 0.0
    long_strike = _as_float(live.get("long_strike")) or 0.0
    target_entry = _as_float(live.get("target_entry")) or round(width * 0.45, 2)
    max_profit = round(max((width - debit) * 100, 0.0), 2)
    max_loss = round(debit * 100, 2)
    debit_width_ratio = round(debit / width, 4) if width > 0 else 0.0
    breakeven = long_strike + debit if direction == "Bull Call" else long_strike - debit
    status = RecommendationStatus.ENTER.value if debit <= target_entry else RecommendationStatus.WAIT_FOR_PRICE.value
    source_label = "Schwab snapshot chain" if str(chain_source).startswith("snapshot:") else "live Schwab chain"
    note = (
        f"{source_label} {direction} validated at {debit:.2f} debit"
        if status == RecommendationStatus.ENTER.value
        else f"{source_label} found {debit:.2f} debit above target {target_entry:.2f}"
    )
    hard_rejects = ""
    quality_rejects = _debit_trade_quality_rejects(
        entry_debit=debit,
        debit_width_ratio=debit_width_ratio,
        max_profit=max_profit,
        max_loss=max_loss,
        signal_premium=_as_float(row.get("signal_premium")) or 0.0,
        combined_flow_bias=_as_float(row.get("combined_flow_bias")) or 0.0,
    )
    quality_rejects.extend(_live_spread_quality_rejects(live))
    if quality_rejects:
        status = RecommendationStatus.AVOID.value
        hard_rejects = "; ".join(quality_rejects)
        note = _append_reason(note, "setup quality gate reject: " + hard_rejects)
    right = _direction_right(direction)
    short_symbol = _as_text(live.get("short_leg"))
    long_symbol = _as_text(live.get("long_leg"))
    sell_leg = _format_option_leg(
        short_symbol,
        "SELL",
        ticker=row.get("ticker", ""),
        expiry=expiry.isoformat(),
        strike=short_strike,
        right=right,
    )
    buy_leg = _format_option_leg(
        long_symbol,
        "BUY",
        ticker=row.get("ticker", ""),
        expiry=expiry.isoformat(),
        strike=long_strike,
        right=right,
    )
    trade_plan = _format_trade_plan(buy_leg, sell_leg, debit, entry_type="DEBIT")
    route = "bull_call_debit" if direction == "Bull Call" else "bear_put_debit"
    out.update(
        {
            "strategy": route,
            "strategy_family": "vertical_spread",
            "strategy_route": route,
            "entry_type": "DEBIT",
            "direction": direction,
            "structure": "bull call debit spread" if direction == "Bull Call" else "bear put debit spread",
            "full_ticket": trade_plan,
            "trade_plan": trade_plan,
            "expiry": expiry.isoformat(),
            "dte": (expiry - asof_date).days,
            "sell_leg": sell_leg,
            "buy_leg": buy_leg,
            "short_leg": sell_leg,
            "long_leg": buy_leg,
            "short_strike": short_strike,
            "long_strike": long_strike,
            "spread_width": width,
            "entry_limit": round(debit, 2),
            "mid": live.get("mid_debit", ""),
            "bid": "",
            "ask": live.get("natural_debit", ""),
            "max_profit": max_profit,
            "max_loss": max_loss,
            "credit_width_ratio": 0.0,
            "debit_width_ratio": debit_width_ratio,
            "trade_quality_status": "rejected" if quality_rejects else "reviewable",
            "quality_gate_reason": "; ".join(quality_rejects),
            "remaining_upside": max_profit,
            "breakeven": round(breakeven, 2),
            "target_exit": round(min(width * 0.80, debit * 1.80), 2) if width > 0 and debit > 0 else "",
            "target_entry": target_entry,
            "invalidation": f"underlying fails to progress toward breakeven {breakeven:.2f}, thesis breaks, or live quote degrades",
            "recommendation_status": status,
            "status_reason": note,
            "hard_rejects": hard_rejects,
            "spot_live": round(float(spot), 2),
            "live_pop_delta_proxy": live.get("pop_delta_proxy", ""),
            "live_short_delta": "",
            "live_long_delta": live.get("long_delta", ""),
            "live_distance_pct": live.get("distance_pct", ""),
            "live_quote_width_pct": live.get("quote_width_pct", ""),
            "live_short_oi": live.get("short_oi", ""),
            "live_short_volume": live.get("short_volume", ""),
            "live_long_oi": live.get("long_oi", ""),
            "live_long_volume": live.get("long_volume", ""),
            "live_leg_min_liquidity": _live_leg_min_liquidity_from_row(
                {
                    "live_short_oi": live.get("short_oi", ""),
                    "live_short_volume": live.get("short_volume", ""),
                    "live_long_oi": live.get("long_oi", ""),
                    "live_long_volume": live.get("long_volume", ""),
                }
            ),
            "live_liquidity_summary": live.get("liquidity_summary", ""),
            "construction_source": live.get("construction_source", ""),
            "construction_reason": live.get("construction_reason", ""),
        }
    )
    return out


def _live_audit_row(row: Mapping[str, Any], status: str, note: str, source: str) -> dict[str, Any]:
    return {
        "ticker": row.get("ticker", ""),
        "live_validation_status": status,
        "recommendation_status": row.get("recommendation_status", ""),
        "chain_source": source,
        "expiry": row.get("expiry", ""),
        "entry_limit": row.get("entry_limit", ""),
        "target_entry": row.get("target_entry", ""),
        "trade_plan": row.get("trade_plan", row.get("full_ticket", "")),
        "sell_leg": row.get("sell_leg", row.get("short_leg", "")),
        "buy_leg": row.get("buy_leg", row.get("long_leg", "")),
        "short_leg": row.get("short_leg", ""),
        "long_leg": row.get("long_leg", ""),
        "note": note,
    }


def _local_news_path(date_dir: Path, ticker: str, day: dt.date) -> Optional[Path]:
    if not ticker:
        return None
    path = date_dir / "browser_text" / f"browser-text-capture-news-{ticker}-{day.isoformat()}.txt"
    return path if path.exists() else None


def _local_news_paths(date_dir: Path, ticker: str, day: dt.date) -> list[Path]:
    exact = _local_news_path(date_dir, ticker, day)
    paths: list[Path] = [exact] if exact is not None else []
    news_dir = date_dir / "browser_text"
    if not news_dir.exists():
        return paths
    prefix = "browser-text-capture-news-"
    suffix = f"-{day.isoformat()}.txt"
    for path in sorted(news_dir.glob(f"{prefix}*{suffix}")):
        if path in paths:
            continue
        middle = path.name[len(prefix) : -len(suffix)]
        tickers = {part.strip().upper() for part in middle.split("-") if part.strip()}
        if ticker.upper() in tickers:
            paths.append(path)
    return paths


def _news_evidence_row(ticker: str, path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
    except Exception:
        lines = []
        text = ""
    headline = _sanitize_local_news_headline(lines[0]) if lines else ""
    if len(headline) > 180:
        headline = headline[:177] + "..."
    sentiment, status, red_terms, support_terms = _classify_news_text(text)
    if not lines:
        status = "local_news_unreadable" if text else "local_news_empty"
        sentiment = "neutral"
    note_bits = []
    if red_terms:
        note_bits.append("red flags: " + ", ".join(red_terms))
    if support_terms:
        note_bits.append("supportive: " + ", ".join(support_terms))
    if headline:
        note_bits.append("headline: " + headline)
    if not note_bits:
        note_bits.append("local news file present")
    return {
        "ticker": ticker,
        "evidence_type": "local_news",
        "evidence_status": status,
        "source": str(path),
        "headline": headline,
        "sentiment": sentiment,
        "red_flag_terms": ";".join(red_terms),
        "support_terms": ";".join(support_terms),
        "objective_blocker": False,
        "days_to_earnings": "",
        "note": "; ".join(note_bits),
    }


def _sanitize_local_news_headline(headline: str) -> str:
    text = str(headline or "").strip()
    text = re.sub(r"\s+for\s+Codex\s+Daily\s+V\d+\s+rerun", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*Codex\s+Daily\s+V\d+\s+rerun\s*", "local review", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def _classify_news_text(text: str) -> tuple[str, str, list[str], list[str]]:
    lowered = text.lower()
    red_phrases = (
        "investigation",
        "sec probe",
        "probe",
        "lawsuit",
        "fraud",
        "bankruptcy",
        "bankrupt",
        "downgrade",
        "guidance cut",
        "cuts guidance",
        "misses estimates",
        "recall",
        "delist",
        "short seller",
        "accounting issue",
        "halts",
        "warning",
    )
    support_phrases = (
        "upgrade",
        "beats estimates",
        "raises guidance",
        "guidance raise",
        "buyback",
        "contract",
        "approval",
        "partnership",
        "record revenue",
        "surge",
        "launch",
    )
    red_terms = [phrase for phrase in red_phrases if phrase in lowered]
    support_terms = [phrase for phrase in support_phrases if phrase in lowered]
    if red_terms and support_terms:
        return "mixed", "news_mixed", red_terms, support_terms
    if red_terms:
        return "negative", "news_red_flag", red_terms, support_terms
    if support_terms:
        return "positive", "news_supportive", red_terms, support_terms
    return "neutral", "local_news_available", red_terms, support_terms


def _summarize_catalyst_evidence(evidence: pd.DataFrame) -> tuple[str, list[str], str, Optional[int], str, str, str, bool]:
    if evidence.empty:
        return "clear", ["no local catalyst file or near-term earnings flag"], "", None, "neutral", "", "", False

    statuses = evidence["evidence_status"].astype(str).tolist() if "evidence_status" in evidence.columns else []
    priority = [
        "event_risk",
        "news_red_flag",
        "watch_event",
        "news_mixed",
        "news_supportive",
        "local_news_available",
        "clear",
    ]
    status = next((item for item in priority if item in statuses), "clear")
    notes = _dedupe_notes(evidence.get("note", pd.Series(dtype=str)).astype(str).tolist())[:5]
    sources = _dedupe_notes(evidence.get("source", pd.Series(dtype=str)).astype(str).tolist())
    red_terms = _dedupe_terms(evidence.get("red_flag_terms", pd.Series(dtype=str)).astype(str).tolist())
    support_terms = _dedupe_terms(evidence.get("support_terms", pd.Series(dtype=str)).astype(str).tolist())
    sentiments = {str(value) for value in evidence.get("sentiment", pd.Series(dtype=str)).astype(str).tolist() if str(value)}
    if "negative" in sentiments:
        sentiment = "negative"
    elif "mixed" in sentiments:
        sentiment = "mixed"
    elif "positive" in sentiments:
        sentiment = "positive"
    elif "event" in sentiments:
        sentiment = "event"
    else:
        sentiment = "neutral"
    days_to_earnings: Optional[int] = None
    if "days_to_earnings" in evidence.columns:
        for value in evidence["days_to_earnings"].tolist():
            parsed = _as_float(value)
            if parsed is not None:
                days_to_earnings = int(parsed)
                break
    objective = bool(evidence.get("objective_blocker", pd.Series(dtype=bool)).astype(bool).any())
    return status, notes, "; ".join(sources), days_to_earnings, sentiment, ";".join(red_terms), ";".join(support_terms), objective


def _news_hint(path: Optional[Path]) -> str:
    if path is None:
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return "local news file present but unreadable"
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "local news file present but empty"
    headline = _sanitize_local_news_headline(lines[0])
    if len(headline) > 140:
        headline = headline[:137] + "..."
    return f"local news file present: {headline}"


def _candidate_ticker(row: Mapping[str, Any]) -> str:
    for key in ("ticker", "underlying", "symbol"):
        value = str(row.get(key) or "").strip()
        if value:
            return value.split()[0].upper()
    return ""


def _portfolio_notes_for_ticker(ticker: str, portfolio: Mapping[str, Any]) -> list[str]:
    if not ticker:
        return []
    notes: list[str] = []
    option_underlyings = {str(value).upper() for value in portfolio.get("option_underlyings", []) or []}
    if ticker in option_underlyings:
        notes.append(f"existing option exposure in {ticker}; execution gate unaffected")

    total_value = _as_float(portfolio.get("total_value"))
    equity_exposure = portfolio.get("large_equity_exposure", {}) or {}
    exposure_value = _as_float(equity_exposure.get(ticker)) if isinstance(equity_exposure, Mapping) else None
    if exposure_value:
        if total_value:
            pct = exposure_value / total_value
            notes.append(f"large equity exposure in {ticker} ({pct:.1%} of book); execution gate unaffected")
        else:
            notes.append(f"large equity exposure in {ticker}; execution gate unaffected")

    sector_exposures = portfolio.get("sector_exposures", {}) or {}
    sector = str(portfolio.get("ticker_sector_map", {}).get(ticker, "")).strip() if isinstance(portfolio.get("ticker_sector_map"), Mapping) else ""
    if sector and isinstance(sector_exposures, Mapping):
        sector_pct = _as_float(sector_exposures.get(sector))
        if sector_pct is not None and sector_pct >= 0.25:
            notes.append(f"sector crowding in {sector} ({sector_pct:.1%} of book); execution gate unaffected")

    return notes


def _is_portfolio_risk_review(review: Mapping[str, Any]) -> bool:
    if _truthy(review.get("portfolio_risk_only")):
        return True
    blocker_type = str(review.get("blocker_type") or "").strip().lower().replace("-", "_")
    if blocker_type in {"portfolio", "portfolio_risk", "account_risk"}:
        return True
    agent = str(review.get("agent") or "").strip().lower().replace("-", "_")
    if agent in {"portfolio", "portfolio_risk", "portfolio_risk_agent"}:
        return True
    text = " ".join(
        [
            str(review.get("verdict") or ""),
            str(review.get("note") or ""),
        ]
    ).lower()
    phrases = (
        "existing position",
        "existing exposure",
        "existing option exposure",
        "equity exposure",
        "portfolio concentration",
        "portfolio crowding",
        "portfolio exposure",
        "sector crowding",
        "sector_crowding",
        "crowded book",
        "book exposure",
        "buying power",
        "margin strain",
        "assignment exposure",
        "account exposure",
        "correlated exposure",
        "correlated beta",
    )
    return any(phrase in text for phrase in phrases)


def _is_portfolio_management_process_note(review: Mapping[str, Any]) -> bool:
    agent = str(review.get("agent") or "").strip().lower().replace("-", "_")
    if agent != "portfolio_management":
        return False
    if _truthy(review.get("objective_blocker")) or _is_portfolio_risk_review(review):
        return False
    text = str(review.get("note") or "").strip().lower()
    process_markers = (
        "no sized spread",
        "target exit",
        "invalidation",
        "fresh portfolio context",
        "risk-off sizing",
        "planning size",
        "portfolio risk annotation",
    )
    return any(marker in text for marker in process_markers)


def _is_review_blocking_builtin_caution(review: Mapping[str, Any]) -> bool:
    if str(review.get("agent_type") or "").strip().lower() in {"external", "subagent"}:
        return False
    if str(review.get("verdict") or "").strip().lower() != "caution":
        return False
    if _truthy(review.get("objective_blocker")) or _is_portfolio_risk_review(review):
        return False
    agent = str(review.get("agent") or "").strip().lower().replace("-", "_")
    if agent != "catalyst":
        return False
    evidence = str(review.get("evidence") or "").strip().lower()
    if evidence in {"event_risk", "news_red_flag"}:
        return True
    if str(review.get("confidence") or "").strip().lower() != "high":
        return False
    text = str(review.get("note") or "").strip().lower()
    review_markers = (
        "red flag",
        "sec probe",
        "investigation",
        "fraud",
        "bankruptcy",
        "trading halt",
        "event risk",
    )
    return any(marker in text for marker in review_markers)


def _split_portfolio_rejects(value: Any) -> tuple[list[str], list[str]]:
    rejects = [item.strip() for item in str(value or "").replace("|", ";").split(";") if item.strip()]
    removed: list[str] = []
    remaining: list[str] = []
    for reject in rejects:
        lowered = reject.lower()
        if any(term in lowered for term in PORTFOLIO_REJECT_TERMS):
            removed.append(reject)
        else:
            remaining.append(reject)
    return removed, remaining


def _is_otherwise_qualified(row: Mapping[str, Any], remaining_rejects: Sequence[str]) -> bool:
    if remaining_rejects:
        return False
    quality = str(row.get("quality_status") or row.get("trade_quality") or "").strip().lower()
    status = str(row.get("recommendation_status") or row.get("trade_status") or "").strip().upper()
    if quality in {"qualified", "good", "approved", "valid", "execute"}:
        return True
    return status in {"ENTER", "EXECUTE", "BUY", "SELL_TO_OPEN"}


def _default_status(row: Mapping[str, Any], remaining_rejects: Sequence[str]) -> str:
    if remaining_rejects:
        return RecommendationStatus.AVOID.value
    if _is_otherwise_qualified(row, remaining_rejects):
        return RecommendationStatus.ENTER.value
    return RecommendationStatus.REVIEW.value


def _append_reason(existing: Any, addition: str) -> str:
    existing_text = str(existing or "").strip()
    if not existing_text:
        return addition
    if addition in existing_text:
        return existing_text
    return f"{existing_text}; {addition}"


def _dedupe_notes(notes: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    clean: list[str] = []
    for note in notes:
        text = str(note or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        clean.append(text)
    return clean


def _dedupe_terms(values: Iterable[str]) -> list[str]:
    terms: list[str] = []
    for value in values:
        for term in str(value or "").replace(",", ";").split(";"):
            clean = term.strip()
            if clean and clean not in terms:
                terms.append(clean)
    return terms


def _format_strike(value: Any) -> str:
    number = _as_float(value)
    if number is None:
        return ""
    if float(number).is_integer():
        return str(int(number))
    return f"{number:.3f}".rstrip("0").rstrip(".")


def _parse_occ_symbol(symbol: Any) -> dict[str, Any]:
    compact = re.sub(r"\s+", "", _as_text(symbol).upper())
    match = re.match(r"^(.+?)(\d{2})(\d{2})(\d{2})([CP])(\d{8})$", compact)
    if not match:
        return {}
    root, yy, month, day, right, strike_raw = match.groups()
    try:
        expiry = dt.date(2000 + int(yy), int(month), int(day)).isoformat()
    except ValueError:
        expiry = ""
    return {
        "ticker": root.strip(),
        "expiry": expiry,
        "right": right,
        "strike": int(strike_raw) / 1000.0,
    }


def _format_option_leg(
    symbol: Any,
    action: str,
    *,
    ticker: Any = "",
    expiry: Any = "",
    strike: Any = "",
    right: Any = "",
) -> str:
    parsed = _parse_occ_symbol(symbol)
    ticker_text = _as_text(ticker).upper() or _as_text(parsed.get("ticker")).upper()
    expiry_text = _as_text(expiry) or _as_text(parsed.get("expiry"))
    strike_text = _format_strike(strike) or _format_strike(parsed.get("strike"))
    right_code = (_as_text(right) or _as_text(parsed.get("right"))).upper()
    right_text = "Call" if right_code == "C" else ("Put" if right_code == "P" else "Option")
    parts = [action.upper(), "1"]
    if ticker_text:
        parts.append(ticker_text)
    if expiry_text:
        parts.append(expiry_text)
    if strike_text:
        parts.append(strike_text)
    parts.append(right_text)
    return " ".join(parts)


def _format_trade_plan(
    sell_leg: Any,
    buy_leg: Any,
    entry_limit: Any,
    *,
    entry_type: str = "CREDIT",
) -> str:
    entry = _as_float(entry_limit)
    suffix = f" @ {entry:.2f} {entry_type.upper()}" if entry is not None and entry > 0 else ""
    sell_text = _as_text(sell_leg)
    buy_text = _as_text(buy_leg)
    if sell_text and buy_text:
        return f"{sell_text} / {buy_text}{suffix}".strip()
    return f"{sell_text or buy_text}{suffix}".strip()


def _as_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _mapping_get(mapping: Mapping[str, Any] | pd.Series, key: str) -> Any:
    if isinstance(mapping, pd.Series):
        return mapping.get(key)
    if isinstance(mapping, Mapping):
        return mapping.get(key)
    return None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_frame(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _smoke_report(day: str, manifest: Mapping[str, Any]) -> str:
    agent_lines = "\n".join(f"- {agent['name']}: {agent['role']}" for agent in manifest["agents"])
    return (
        f"# Options Agent Smoke Report - {day}\n\n"
        "Status: design scaffold only. Live UW, Schwab chain, and research integrations "
        "are intentionally not wired in this first slice.\n\n"
        "## Visibility Invariant\n\n"
        f"{manifest['visibility_invariant']}.\n\n"
        "## Agent Roster\n\n"
        f"{agent_lines}\n"
    )
