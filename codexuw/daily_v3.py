from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from .catalysts import load_catalyst_context
from .confirmations import apply_confirmation_evidence, build_confirmation_evidence, write_confirmation_evidence
from .daily import infer_report_mode, latest_dated_folder, live_planning_validation_note
from .data import aggregate_bot_flow, infer_asof_date, load_chain_oi, load_hot_chains, load_stock_screener, safe_float
from .engine import (
    apply_catalyst_context,
    apply_confidence_components,
    apply_confirmation_framework,
    apply_data_quality_gate,
    apply_final_quality_guards,
    apply_high_conviction_decision_marks,
    apply_oi_carryover,
    apply_portfolio_context,
    apply_replay_edge_model,
    assign_trade_statuses,
    build_data_quality_status,
    build_entry_watchlist,
    build_intraday_change_summary,
    detect_regime,
    generate_candidates,
    live_validate_and_score,
    rejection_summary,
    select_final_trades,
    select_index_fallback_pool,
    select_ticker_pool,
)
from .fallback_income import apply_fallback_income_status, build_fallback_income_candidates
from .loss_review import apply_loss_review, load_recent_loss_review, write_loss_review
from .liquidity_shift import (
    apply_liquidity_shift_context,
    build_liquidity_shift_signals,
    expand_pool_with_top_flow,
    write_liquidity_shift_artifacts,
)
from .macro_gates import build_macro_event_gates, write_macro_event_gates
from .missed_opportunity import write_missed_opportunity_audit
from .opportunity import (
    OPPORTUNITY_COLUMNS,
    PIPELINE_NAME_V3,
    PIPELINE_VERSION_V3,
    TARGET_TICKET_COLUMNS,
    build_opportunity_board,
    build_target_ticket_board,
    classify_no_trade_audit,
    opportunity_counts,
    write_recommendation_ledger,
)
from .overlay import run_overlay
from .performance import load_live_outcome_performance, load_recent_performance
from .pipeline_versions import pipeline_version_record
from .portfolio import fetch_portfolio_context, unavailable_portfolio_context
from .provenance import build_input_provenance
from .regime import build_v3_regime_context, write_v3_regime_artifact
from .snapshots import write_reproducibility_artifacts
from .target_model import build_v3_target_model
from .validation import run_validation_harness


DEFAULT_ROOT = Path("/Users/anuppamvi/uw_root/tradedesk")


def _parse_date(value: str | None) -> dt.date | None:
    if not value:
        return None
    return dt.datetime.strptime(value, "%Y-%m-%d").date()


def _base_dir_from_args(args: argparse.Namespace) -> Path:
    if getattr(args, "base_dir", ""):
        return Path(args.base_dir).expanduser().resolve()
    date = _parse_date(getattr(args, "date", ""))
    if not date:
        raise SystemExit("--date or --base-dir is required")
    return Path(getattr(args, "root", DEFAULT_ROOT)).expanduser().resolve() / str(date)


def _default_out_dir(root: Path, asof: dt.date, mode: str, overlay_date: dt.date | None = None) -> Path:
    if mode == "overlay":
        return root / "out" / f"codexdaily_v3_{asof}_overlay_{overlay_date or asof}"
    if mode == "validation":
        return root / "out" / f"codexdaily_v3_validation_{asof}"
    return root / "out" / f"codexdaily_v3_{asof}"


def _add_common_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="Trading desk root. Used with --date.")
    parser.add_argument("--date", default="", help="Dated UW folder date, e.g. 2026-05-19.")
    parser.add_argument("--base-dir", default="", help="Dated UW folder. Overrides --date.")
    parser.add_argument("--out-dir", default="", help="Output directory. Defaults to out/codexdaily_v3_YYYY-MM-DD.")
    parser.add_argument("--max-tickers", type=int, default=0, help="Discovery cap. Default 0 scans every eligible source ticker.")
    parser.add_argument("--max-candidates", type=int, default=0, help="Candidate cap. Default 0 keeps every constructed candidate before scoring.")
    parser.add_argument(
        "--max-final-trades",
        type=int,
        default=0,
        help="Visibility cap for Execute rows. Default 0 shows all valid Execute/target rows; risk caps still control sizing.",
    )
    parser.add_argument("--risk-budget", type=float, default=15_000.0)
    parser.add_argument("--bot-max-rows", type=int, default=0)
    parser.add_argument("--offline", action="store_true", help="Test-only: skip Schwab live chain validation.")
    parser.add_argument("--skip-portfolio", action="store_true", help="Skip Schwab portfolio pull; blocks live-quality Execute in V3.")
    parser.add_argument("--skip-catalysts", action="store_true", help="Skip local browser/news catalyst checks.")
    parser.add_argument("--skip-recent-performance", action="store_true")
    parser.add_argument("--schwab-snapshot-dir", default="", help="Existing Schwab chain snapshot directory for reproducible reruns.")
    parser.add_argument("--report-mode", default="auto", choices=["auto", "pre-market", "intraday", "post-close", "historical"])
    parser.add_argument("--max-risk-per-trade", type=float, default=0.0)
    parser.add_argument("--max-risk-per-day", type=float, default=0.0)
    parser.add_argument("--max-open-risk-by-ticker", type=float, default=0.0)
    parser.add_argument("--max-correlated-sector-exposure", type=float, default=0.0)
    parser.add_argument("--max-total-open-risk", type=float, default=0.0)
    parser.add_argument("--max-contracts-per-trade", type=float, default=20.0)
    parser.add_argument("--daily-loss-limit", type=float, default=0.0)
    parser.add_argument("--weekly-loss-limit", type=float, default=0.0)
    parser.add_argument("--monthly-loss-limit", type=float, default=0.0)
    parser.add_argument("--monthly-profit-target", type=float, default=10_000.0)
    parser.add_argument("--month-to-date-realized-pnl", type=float, default=0.0)
    parser.add_argument("--open-unrealized-pnl", type=float, default=0.0)
    parser.add_argument("--max-monthly-drawdown", type=float, default=0.0)
    parser.add_argument("--minimum-expected-value-per-dollar-risk", type=float, default=0.01)
    parser.add_argument(
        "--risk-mandate",
        default="target-growth",
        choices=["capital-preservation", "balanced", "target-growth", "aggressive-intraday"],
    )
    parser.add_argument("--index-income-mode", default="primary", choices=["disabled", "fallback", "primary"])
    parser.add_argument(
        "--portfolio-income-mode",
        default="trading-sleeve-only",
        choices=["disabled", "trading-sleeve-only", "existing-core-review"],
    )
    parser.add_argument("--covered-income-allowed-tickers", default="")
    parser.add_argument("--loss-lookback-days", type=int, default=30)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw and raw[0] in {"-h", "--help"}:
        pass
    elif not raw or raw[0].startswith("-"):
        raw = ["run", *raw]
    parser = argparse.ArgumentParser(
        description=(
            "Codex Daily V3 Schwab-backed options opportunity engine. "
            "Commands: run, overlay, intraday, validate, loss-review."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Normal V3 daily run for a dated UW folder.")
    _add_common_run_args(run)

    intraday = sub.add_parser("intraday", help="Intraday V3 live scan/refresh using Schwab chains and portfolio state.")
    _add_common_run_args(intraday)

    overlay = sub.add_parser("overlay", help="Overlay a newer chain-oi file onto a prior V3 analysis.")
    overlay.add_argument("--root", default=str(DEFAULT_ROOT))
    overlay.add_argument("--date", required=True, help="Original analysis date.")
    overlay.add_argument("--prior-out-dir", default="", help="Prior V3 out dir. Defaults to out/codexdaily_v3_DATE.")
    overlay.add_argument("--overlay-file", required=True, help="chain-oi-changes-YYYY-MM-DD.csv or .zip")
    overlay.add_argument("--overlay-date", default="", help="Date of overlay file if not inferable from filename.")
    overlay.add_argument("--out-dir", default="", help="Defaults to out/codexdaily_v3_DATE_overlay_OVERLAYDATE.")

    validate = sub.add_parser("validate", help="Run V3 validation over latest N source-complete dated folders.")
    _add_common_run_args(validate)
    validate.add_argument("--as-of", default="", help="Validation as-of date. Defaults to latest dated folder.")
    validate.add_argument("--latest-n", type=int, default=5)
    validate.add_argument("--run-live", action="store_true", help="Actually run V3 for selected dates; otherwise compare existing manifests.")

    loss = sub.add_parser("loss-review", help="Post-trade/loss review for recent recommendations before adding risk.")
    loss.add_argument("--root", default=str(DEFAULT_ROOT))
    loss.add_argument("--as-of", required=True)
    loss.add_argument("--out-dir", default="")
    loss.add_argument("--loss-lookback-days", type=int, default=30)
    return parser.parse_args(raw)


def _market_data_status(run_mode: str) -> str:
    now = dt.datetime.now().time()
    if "Intraday" in run_mode and dt.datetime.now().weekday() < 5 and dt.time(6, 30) <= now <= dt.time(13, 0):
        return "regular-session check; Schwab quote timestamps still control live validity"
    return "market closed or outside regular US session; do not treat stale snapshots as live-executable"


def _schwab_status(data_quality: dict[str, Any]) -> str:
    items = data_quality.get("items") or []
    quote = next((item for item in items if item.get("check") == "Schwab quotes available"), {})
    portfolio = next((item for item in items if item.get("check") == "Schwab portfolio available"), {})
    return f"quotes={quote.get('status', 'unknown')} ({quote.get('detail', '')}); portfolio={portfolio.get('status', 'unknown')} ({portfolio.get('detail', '')})"


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "_No rows._"
    shown = df[[col for col in columns if col in df.columns]].copy()
    shown = shown.where(pd.notna(shown), "")
    return shown.to_markdown(index=False)


def _money_value(value: object) -> float:
    if isinstance(value, str):
        value = value.replace("$", "").replace(",", "").strip()
    return safe_float(value, 0.0)


def write_v3_data_error_report(out_dir: Path, asof: dt.date, base_dir: Path, error: Exception) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    reason = f"data failure: {error}"
    board = pd.DataFrame(
        [
            {
                "Lane": "Research/Avoid",
                "Status": "🔴 Blocked",
                "Ticker": "DATA",
                "Trade": "Required UW input load",
                "Expiry": "",
                "Entry limit": "blocked",
                "Live mid/natural": "",
                "Max profit": "",
                "Max loss": "",
                "Target profit": "",
                "Expected value source": "local UW files",
                "Edge sample size / win rate / avg P/L": "",
                "Required confirmation": "restore required stock-screener, hot-chains, and bot-eod exports",
                "Monitor trigger": "",
                "Why Execute, Scout, Research, or Avoid": reason,
            }
        ]
    )
    board.to_csv(out_dir / f"codexdaily_v3_opportunity_board_{asof}.csv", index=False)
    pd.DataFrame().to_csv(out_dir / f"codexdaily_v3_candidates_{asof}.csv", index=False)
    pd.DataFrame().to_csv(out_dir / f"codexdaily_v3_scored_{asof}.csv", index=False)
    pd.DataFrame([{"reason": reason, "count": 1}]).to_csv(out_dir / f"codexdaily_v3_rejections_{asof}.csv", index=False)
    no_trade = {"classification": "data failure", "exact_blocker": reason}
    target_model = build_v3_target_model(asof=asof, board=board, monthly_profit_target=10_000, risk_budget=0)
    reproducibility = write_reproducibility_artifacts(
        out_dir=out_dir,
        asof=asof,
        repo_root=DEFAULT_ROOT,
        run_config={"run_mode": "data error", "base_dir": str(base_dir), "out_dir": str(out_dir)},
        input_provenance=build_input_provenance(base_dir),
        data_quality={"status": "critical", "critical_blockers": ["uw_files_present"], "items": []},
        portfolio={"status": "not_checked"},
        regime={"status": "unavailable"},
        loss_review={"status": "not_checked"},
    )
    manifest = {
        "pipeline_name": PIPELINE_NAME_V3,
        "pipeline_version": PIPELINE_VERSION_V3,
        "version_lock": pipeline_version_record("v3"),
        "run_mode": "data error",
        "asof": str(asof),
        "base_dir": str(base_dir),
        "data_quality": {"status": "critical", "critical_blockers": ["uw_files_present"], "items": []},
        "opportunity_counts": opportunity_counts(board),
        "target_model": target_model,
        "no_trade_audit": no_trade,
        "reproducibility": reproducibility,
        "status": "blocked",
        "reason": reason,
    }
    manifest_path = out_dir / f"codexdaily_v3_manifest_{asof}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    report = out_dir / f"codexdaily_v3_report_{asof}.md"
    lines = [
        f"# {PIPELINE_NAME_V3} Report - {asof}",
        "",
        "## First Screen",
        "",
        "| Item | Value |",
        "|:--|:--|",
        f"| Pipeline | {PIPELINE_NAME_V3} |",
        f"| Version | {PIPELINE_VERSION_V3} |",
        "| Version lock | locked 2026-06-12; supersedes v3.0 |",
        "| Run mode | Data error |",
        "| Data quality | critical |",
        "| Schwab status | not checked because local UW data failed |",
        "| Portfolio status | not checked |",
        "| Market regime | unavailable |",
        "| Execute count | 0 |",
        "| Scout count | 0 |",
        "| Momentum Debit count | 0 |",
        "| Index/ETF count | 0 |",
        "| Portfolio Repair count | 0 |",
        "| Wheel/Cash count | 0 |",
        f"| Target feasibility vs $10k/month | {target_model['target_feasibility']} |",
        f"| Exact blocker if target is infeasible | {reason} |",
        "",
        "## Opportunity Board",
        "",
        _markdown_table(board, OPPORTUNITY_COLUMNS),
        "",
        "## No-Trade Audit",
        f"- Classification: {no_trade['classification']}",
        f"- Exact blocker: {no_trade['exact_blocker']}",
        "",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    manifest["report_path"] = str(report)
    return manifest


def _risk_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "max_risk_per_trade": args.max_risk_per_trade,
        "max_risk_per_day": args.max_risk_per_day,
        "max_open_risk_by_ticker": args.max_open_risk_by_ticker,
        "max_correlated_sector_exposure": args.max_correlated_sector_exposure,
        "max_total_open_risk": args.max_total_open_risk,
        "max_contracts_per_trade": args.max_contracts_per_trade,
        "minimum_expected_value_per_dollar_risk": args.minimum_expected_value_per_dollar_risk,
        "monthly_profit_target": args.monthly_profit_target,
        "daily_loss_limit": args.daily_loss_limit,
        "risk_mandate": args.risk_mandate,
        "index_income_mode": args.index_income_mode,
        "portfolio_income_mode": args.portfolio_income_mode,
        "covered_income_allowed_tickers": [
            ticker.strip().upper()
            for ticker in str(args.covered_income_allowed_tickers or "").split(",")
            if ticker.strip()
        ],
        "allow_new_trades": True,
    }


def write_v3_outputs(
    *,
    out_dir: Path,
    repo_root: Path,
    asof: dt.date,
    base_dir: Path,
    run_mode: str,
    market_data_status: str,
    input_provenance: dict[str, Any],
    data_quality: dict[str, Any],
    change_summary: dict[str, Any],
    regime: dict[str, Any],
    candidates: pd.DataFrame,
    scored: pd.DataFrame,
    final: pd.DataFrame,
    watchlist: pd.DataFrame,
    portfolio: dict[str, Any] | None,
    catalysts: pd.DataFrame | None,
    macro_gates: pd.DataFrame | None,
    confirmation_evidence: pd.DataFrame | None,
    liquidity_shift: dict[str, Any],
    v3_regime_context: dict[str, Any],
    recent_performance: dict[str, Any],
    live_outcomes: dict[str, Any],
    loss_review: dict[str, Any],
    risk_config: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(out_dir / f"codexdaily_v3_candidates_{asof}.csv", index=False)
    scored.to_csv(out_dir / f"codexdaily_v3_scored_{asof}.csv", index=False)
    final.to_csv(out_dir / f"codexdaily_v3_execute_trades_{asof}.csv", index=False)
    watchlist.to_csv(out_dir / f"codexdaily_v3_scout_watchlist_{asof}.csv", index=False)
    research = scored[scored["trade_status"].astype(str).eq("Research")].copy() if not scored.empty and "trade_status" in scored.columns else scored.iloc[0:0].copy()
    avoid = scored[scored["trade_status"].astype(str).eq("Avoid")].copy() if not scored.empty and "trade_status" in scored.columns else scored.iloc[0:0].copy()
    research.to_csv(out_dir / f"codexdaily_v3_research_candidates_{asof}.csv", index=False)
    avoid.to_csv(out_dir / f"codexdaily_v3_avoid_candidates_{asof}.csv", index=False)
    rejection_summary(scored).to_csv(out_dir / f"codexdaily_v3_rejections_{asof}.csv", index=False)
    if catalysts is not None and not catalysts.empty:
        catalysts.to_csv(out_dir / f"codexdaily_v3_catalysts_{asof}.csv", index=False)
    macro_csv, macro_json, macro_summary = write_macro_event_gates(
        out_dir,
        asof,
        macro_gates if macro_gates is not None else pd.DataFrame(),
    )
    confirmation_csv, confirmation_json, confirmation_summary = write_confirmation_evidence(
        out_dir,
        asof,
        confirmation_evidence if confirmation_evidence is not None else pd.DataFrame(),
    )
    liquidity_artifacts = write_liquidity_shift_artifacts(out_dir, asof, liquidity_shift)
    regime_json, regime_summary = write_v3_regime_artifact(out_dir, asof, v3_regime_context)

    board = build_opportunity_board(scored=scored, final=final, watchlist=watchlist, portfolio=portfolio, max_rows=0)
    target_tickets = build_target_ticket_board(board, max_rows=0)
    board_path = out_dir / f"codexdaily_v3_opportunity_board_{asof}.csv"
    target_tickets_path = out_dir / f"codexdaily_v3_swing_target_tickets_{asof}.csv"
    board.to_csv(board_path, index=False)
    target_tickets.to_csv(target_tickets_path, index=False)
    counts = opportunity_counts(board)
    target_ticket_count = int(len(target_tickets))
    target_ticket_profit = float(target_tickets["Profit target"].map(_money_value).sum()) if not target_tickets.empty else 0.0
    target_ticket_risk = float(target_tickets["Max loss"].map(_money_value).sum()) if not target_tickets.empty else 0.0
    no_trade_audit = classify_no_trade_audit(board=board, scored=scored, data_quality=data_quality, portfolio=portfolio)
    portfolio_ok = bool(portfolio and portfolio.get("status") == "ok")
    target_model = build_v3_target_model(
        asof=asof,
        board=board,
        monthly_profit_target=args.monthly_profit_target,
        month_to_date_realized_pnl=args.month_to_date_realized_pnl,
        open_unrealized_pnl=args.open_unrealized_pnl,
        account_value=safe_float((portfolio or {}).get("total_value")) if portfolio_ok else math.nan,
        available_cash=safe_float((portfolio or {}).get("cash")) if portfolio_ok else math.nan,
        risk_budget=args.risk_budget,
        max_daily_loss=args.daily_loss_limit,
        max_weekly_loss=args.weekly_loss_limit,
        max_monthly_loss=args.monthly_loss_limit,
        historical_win_rate=safe_float((recent_performance or {}).get("win_rate")),
        average_realized_win=safe_float((live_outcomes or {}).get("avg_pnl")),
        average_realized_loss=safe_float((live_outcomes or {}).get("avg_pnl")),
    )
    ledger_path, global_ledger_path = write_recommendation_ledger(out_dir, asof, board)
    try:
        ledger_frame = pd.read_csv(global_ledger_path)
    except Exception:
        ledger_frame = pd.DataFrame()
    missed_csv, missed_json, missed_summary = write_missed_opportunity_audit(out_dir, asof, ledger_frame)
    loss_json, loss_csv = write_loss_review(out_dir, asof, loss_review)
    run_config = {
        "run_mode": run_mode,
        "market_data_status": market_data_status,
        "base_dir": str(base_dir),
        "out_dir": str(out_dir),
        "risk_config": risk_config,
        "monthly_profit_target": args.monthly_profit_target,
        "max_tickers": args.max_tickers,
        "max_candidates": args.max_candidates,
        "max_final_trades": args.max_final_trades,
        "offline": args.offline,
        "schwab_snapshot_dir": args.schwab_snapshot_dir,
    }
    reproducibility = write_reproducibility_artifacts(
        out_dir=out_dir,
        asof=asof,
        repo_root=repo_root,
        run_config=run_config,
        input_provenance=input_provenance,
        data_quality=data_quality,
        portfolio=portfolio,
        regime=v3_regime_context or regime,
        loss_review=loss_review,
    )
    lane_coverage = board.groupby("Lane").size().to_dict() if not board.empty else {}
    funnel = {
        "candidate_rows": int(len(candidates)),
        "scored_rows": int(len(scored)),
        "execute_rows": int(counts["execute"]),
        "scout_rows": int(counts["scout"]),
        "momentum_debit_rows": int(counts["momentum_debit"]),
        "index_etf_rows": int(counts["index_etf"]),
        "portfolio_repair_rows": int(counts["portfolio_repair"]),
        "wheel_cash_rows": int(counts["wheel_cash"]),
        "research_rows": int(len(research)),
        "avoid_rows": int(len(avoid)),
    }
    artifacts = {
        "report": str(out_dir / f"codexdaily_v3_report_{asof}.md"),
        "manifest": str(out_dir / f"codexdaily_v3_manifest_{asof}.json"),
        "opportunity_board": str(board_path),
        "swing_target_tickets": str(target_tickets_path),
        "scored": str(out_dir / f"codexdaily_v3_scored_{asof}.csv"),
        "execute_trades": str(out_dir / f"codexdaily_v3_execute_trades_{asof}.csv"),
        "scout_watchlist": str(out_dir / f"codexdaily_v3_scout_watchlist_{asof}.csv"),
        "recommendation_ledger": str(ledger_path),
        "global_recommendation_ledger": str(global_ledger_path),
        "loss_review_json": str(loss_json),
        "loss_review_csv": str(loss_csv),
        "missed_opportunity_audit_csv": str(missed_csv),
        "missed_opportunity_audit_json": str(missed_json),
        "macro_event_gates_csv": str(macro_csv),
        "macro_event_gates_json": str(macro_json),
        "confirmation_evidence_csv": str(confirmation_csv),
        "confirmation_evidence_json": str(confirmation_json),
        "liquidity_shift_summary": liquidity_artifacts["artifacts"].get("summary_json", ""),
        "flow_velocity_csv": liquidity_artifacts["artifacts"].get("flow_velocity_csv", ""),
        "top_flow_universe_csv": liquidity_artifacts["artifacts"].get("top_flow_universe_csv", ""),
        "correlation_anomalies_csv": liquidity_artifacts["artifacts"].get("correlation_anomalies_csv", ""),
        "zero_dte_gamma_csv": liquidity_artifacts["artifacts"].get("zero_dte_gamma_csv", ""),
        "regime_context_json": str(regime_json),
        "reproducibility": reproducibility.get("reproducibility_artifact", ""),
        "schwab_snapshot_summary": reproducibility.get("schwab_snapshot_summary", ""),
    }
    manifest = {
        "pipeline_name": PIPELINE_NAME_V3,
        "pipeline_version": PIPELINE_VERSION_V3,
        "version_lock": pipeline_version_record("v3"),
        "asof": str(asof),
        "base_dir": str(base_dir),
        "out_dir": str(out_dir),
        "run_mode": run_mode,
        "market_data_status": market_data_status,
        "data_quality": data_quality,
        "schwab_status": _schwab_status(data_quality),
        "portfolio_status": (portfolio or {}).get("status", "not_checked"),
        "portfolio_position_count": (portfolio or {}).get("position_count", 0),
        "regime": regime,
        "funnel": funnel,
        "opportunity_counts": counts,
        "swing_target_ticket_count": target_ticket_count,
        "visible_signal_policy": {
            "active_execute_cap": int(args.max_final_trades) if int(args.max_final_trades or 0) > 0 else None,
            "active_board_cap": None,
            "active_target_ticket_cap": None,
            "max_final_trades_arg": int(args.max_final_trades or 0),
            "risk_caps_size_and_label_only": True,
        },
        "swing_target_ticket_profit_if_filled": round(target_ticket_profit, 2),
        "swing_target_ticket_max_loss_if_filled": round(target_ticket_risk, 2),
        "lane_coverage": lane_coverage,
        "target_model": target_model,
        "no_trade_audit": no_trade_audit,
        "recent_performance": recent_performance,
        "live_outcomes": live_outcomes,
        "loss_review": loss_review,
        "missed_opportunity_audit": missed_summary,
        "macro_event_gates": macro_summary,
        "confirmation_evidence": confirmation_summary,
        "liquidity_shift": liquidity_artifacts["summary"],
        "v3_regime_context": regime_summary,
        "intraday_change_summary": change_summary,
        "risk_config": risk_config,
        "artifacts": artifacts,
        "recommendation_ledger": str(ledger_path),
        "reproducibility": reproducibility,
    }
    manifest_path = out_dir / f"codexdaily_v3_manifest_{asof}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")

    if target_model.get("target_feasibility") == "feasible":
        target_blocker = ""
    elif no_trade_audit.get("classification") == "data failure":
        target_blocker = no_trade_audit.get("exact_blocker", "")
    else:
        target_blocker = target_model.get("explicit_infeasible_reason") or target_model.get("binding_constraint")
    regime_label = (
        f"{regime.get('trend', 'unknown')}; vol={regime.get('volatility', 'unknown')}; "
        f"flow={regime.get('flow', 'unknown')}; VIX={regime.get('vix_proxy', 'n/a')}"
    )
    target_risk_required = target_model["target_gap"].get("risk_required")
    target_risk_required_text = f"${target_risk_required:,.2f}" if target_risk_required is not None else "n/a"
    lines = [
        f"# {PIPELINE_NAME_V3} Report - {asof}",
        "",
        "## First Screen",
        "",
        "| Item | Value |",
        "|:--|:--|",
        f"| Pipeline | {PIPELINE_NAME_V3} |",
        f"| Version | {PIPELINE_VERSION_V3} |",
        "| Version lock | locked 2026-06-12; supersedes v3.0 |",
        f"| Run mode | {run_mode} |",
        f"| Data quality | {data_quality.get('status', 'unknown')} |",
        f"| Schwab status | {_schwab_status(data_quality)} |",
        f"| Portfolio status | {(portfolio or {}).get('status', 'not_checked')} |",
        f"| Market regime | {regime_label} |",
        f"| Market data freshness | {market_data_status} |",
        f"| Swing target ticket count | {target_ticket_count} |",
        f"| Execute count | {counts['execute']} |",
        f"| Scout count | {counts['scout']} |",
        f"| Momentum Debit count | {counts['momentum_debit']} |",
        f"| Index/ETF count | {counts['index_etf']} |",
        f"| Portfolio Repair count | {counts['portfolio_repair']} |",
        f"| Wheel/Cash count | {counts['wheel_cash']} |",
        f"| Liquidity-shift signals | flow velocity {liquidity_artifacts['summary'].get('flow_velocity_signals', 0)}; child accumulation {liquidity_artifacts['summary'].get('child_order_accumulation_signals', 0)}; 0DTE/index {liquidity_artifacts['summary'].get('zero_dte_index_signal_count', 0)} |",
        f"| Target feasibility vs $10k/month | {target_model['target_feasibility']} |",
        f"| Exact blocker if target is infeasible | {target_blocker} |",
        "",
        "## Swing Target Tickets For Tomorrow",
        "",
        "These are EOD swing trade targets for manual order tickets. A target miss does not hide the trade; it only tells you the limit to work and the blocker to respect.",
        "",
        _markdown_table(target_tickets, TARGET_TICKET_COLUMNS),
        "",
        "## Opportunity Board",
        "",
        _markdown_table(board, OPPORTUNITY_COLUMNS),
        "",
    ]
    coverage_rows = pd.DataFrame()
    if not scored.empty and "candidate_coverage_source" in scored.columns:
        coverage_rows = scored[
            scored["candidate_coverage_source"].astype(str).eq("per_ticker_coverage")
            & ~scored["ticker"].astype(str).isin(set(board["Ticker"].astype(str)) if not board.empty and "Ticker" in board.columns else set())
        ].copy()
    if not coverage_rows.empty:
        coverage_cols = [
            "ticker",
            "strategy",
            "direction",
            "expiry",
            "trade_status",
            "primary_blocker",
            "trade_status_reason",
            "flow_quality",
            "score",
        ]
        coverage_display = coverage_rows[[c for c in coverage_cols if c in coverage_rows.columns]].drop_duplicates()
        if "ticker" in coverage_display.columns:
            coverage_display = coverage_display.sort_values(["ticker", "strategy", "expiry"], na_position="last")
        coverage_display = coverage_display.rename(
            columns={
                "ticker": "Ticker",
                "strategy": "Strategy",
                "direction": "Direction",
                "expiry": "Expiry",
                "trade_status": "Status",
                "primary_blocker": "Primary Blocker",
                "trade_status_reason": "Reason",
                "flow_quality": "Flow Quality",
                "score": "Score",
            }
        )
        lines.extend(
            [
                "## Candidate Coverage Audit",
                "",
                "These selected-pool tickers were forced through at least one constructed setup so they cannot silently disappear before scoring. They are shown here when they do not make the opportunity board.",
                "",
                _markdown_table(coverage_display, list(coverage_display.columns)),
                "",
            ]
        )
    if counts["execute"] == 0:
        lines.extend(
            [
                "## No-Execute Alternatives",
                "",
                "The board above keeps the best Scout, Momentum Debit, Index/ETF, Portfolio Repair, and Wheel/Cash alternatives visible. None is auto-promoted to Execute.",
                "",
            ]
        )
    lines.extend(
        [
            "## No-Trade Audit",
            "",
            f"- Classification: {no_trade_audit['classification']}",
            f"- Exact blocker: {no_trade_audit['exact_blocker']}",
            "",
            "## Target Math",
            "",
            "| Metric | Value |",
            "|:--|--:|",
            f"| Required daily P/L | ${target_model['required_daily_pl']:,.2f} |",
            f"| Required weekly P/L | ${target_model['required_weekly_pl']:,.2f} |",
            f"| Current qualified opportunity expected P/L | ${target_model['current_qualified_opportunity_expected_pl']:,.2f} |",
            f"| Swing target-ticket profit if all listed targets fill | ${target_ticket_profit:,.2f} |",
            f"| Swing target-ticket max loss if all listed targets fill | ${target_ticket_risk:,.2f} |",
            f"| Expected monthly run-rate | ${target_model['expected_monthly_run_rate_from_current_qualified_opportunities']:,.2f} |",
            f"| Gap to target | ${target_model['target_gap']['dollars_remaining']:,.2f} |",
            f"| Risk required | {target_risk_required_text} |",
            f"| Max allowed risk | ${target_model['target_gap']['risk_available']:,.2f} |",
            f"| Can sizing close the gap? | {'yes' if target_model['target_feasibility'] == 'feasible' else 'no'} |",
            f"| Exact reason | {target_model['binding_constraint']} |",
            "",
            "## Liquidity Shift Signals",
            "",
            f"- Confirmation evidence: cleared {confirmation_summary.get('cleared', 0)}, manual {confirmation_summary.get('manual', 0)}, blocked {confirmation_summary.get('blocked', 0)}",
            f"- Volatility threshold regime: {(liquidity_artifacts['summary'].get('threshold_regime') or {}).get('volatility_regime', 'unknown')} - {(liquidity_artifacts['summary'].get('threshold_regime') or {}).get('why', '')}",
            f"- Flow velocity signals: {liquidity_artifacts['summary'].get('flow_velocity_signals', 0)}; child-order accumulation signals: {liquidity_artifacts['summary'].get('child_order_accumulation_signals', 0)}; 0DTE/index gamma rows: {liquidity_artifacts['summary'].get('zero_dte_index_signal_count', 0)}",
            "",
            "### Top Flow Sweep",
            "",
            _markdown_table(
                (liquidity_shift.get("top_flow_universe") if isinstance(liquidity_shift.get("top_flow_universe"), pd.DataFrame) else pd.DataFrame()).head(10),
                [
                    "rank",
                    "ticker",
                    "source",
                    "net_premium",
                    "flow_direction",
                    "max_rolling_15m_premium",
                    "volume_oi_ratio",
                    "vwap_confirmation",
                ],
            ),
            "",
            "### Correlation / 0DTE",
            "",
            _markdown_table(
                (liquidity_shift.get("correlation_anomalies") if isinstance(liquidity_shift.get("correlation_anomalies"), pd.DataFrame) else pd.DataFrame()).head(8),
                ["ticker", "benchmark", "relative_return_divergence", "flow_direction", "anomaly", "sector_leader_signal", "reason"],
            ),
            "",
            _markdown_table(
                (liquidity_shift.get("zero_dte_gamma") if isinstance(liquidity_shift.get("zero_dte_gamma"), pd.DataFrame) else pd.DataFrame()).head(8),
                ["ticker", "spot", "pinning_level", "gamma_flip_zone", "dominant_flow_direction", "setup_type", "reason"],
            ),
            "",
        ]
    )
    actionable = board[board["Status"].astype(str).str.contains("Execute|Scout", regex=True)] if not board.empty else pd.DataFrame()
    if not actionable.empty:
        lines.extend(
            [
                "## Execution Quality",
                "",
                _markdown_table(
                    actionable,
                    [
                        "Status",
                        "Ticker",
                        "Entry limit",
                        "Live mid/natural",
                        "quote_width_pct",
                        "fill_ladder",
                    ],
                ),
                "",
                "## Lifecycle Monitors",
                "",
                _markdown_table(
                    actionable,
                    [
                        "Status",
                        "Ticker",
                        "Trade",
                        "profit_take",
                        "stop_loss",
                        "roll_trigger",
                        "short_strike_threat",
                        "short_leg_delta_threshold",
                        "dte_warning",
                        "thesis_invalidation",
                        "phone_alert_text",
                    ],
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Artifacts",
            "",
            "| Artifact | Path |",
            "|:--|:--|",
        ]
    )
    for name, path in artifacts.items():
        lines.append(f"| {name} | {path} |")
    lines.append("")
    report_path = out_dir / f"codexdaily_v3_report_{asof}.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    manifest["report_path"] = str(report_path)
    return manifest


def run_v3_daily(
    *,
    base_dir: Path,
    out_dir: Path,
    args: argparse.Namespace,
    run_mode_override: str | None = None,
) -> dict[str, Any]:
    repo_root = DEFAULT_ROOT
    asof = infer_asof_date(base_dir)
    input_provenance = build_input_provenance(base_dir)
    run_mode = run_mode_override or infer_report_mode(args.report_mode, historical_replay=False)
    if args.command == "intraday":
        run_mode = "Intraday live scan"

    try:
        sc = load_stock_screener(base_dir)
        hot = load_hot_chains(base_dir, asof)
    except Exception as exc:
        return write_v3_data_error_report(out_dir, asof, base_dir, exc)
    try:
        chain_oi = load_chain_oi(base_dir, asof)
    except Exception:
        chain_oi = None

    regime = detect_regime(sc)
    latest_asof = latest_dated_folder(base_dir.parent)
    note = live_planning_validation_note(asof, latest_asof)
    if note:
        regime["validation_note"] = note
    liquidity_shift = build_liquidity_shift_signals(
        base_dir=base_dir,
        root=base_dir.parent,
        asof=asof,
        stock_screener=sc,
        hot_chains=hot,
        chain_oi=chain_oi,
        regime=regime,
        max_rows=args.bot_max_rows if args.bot_max_rows > 0 else None,
    )
    v3_regime_context = build_v3_regime_context(
        stock_screener=sc,
        base_regime=regime,
        liquidity_shift=liquidity_shift,
        asof=asof,
        run_mode=run_mode,
    )
    pool = select_ticker_pool(sc, max_tickers=args.max_tickers)
    pool = expand_pool_with_top_flow(pool, sc, liquidity_shift, max_top_flow=50)
    index_pool = select_index_fallback_pool(sc)
    bot_tickers = pool["ticker"].tolist()
    if not index_pool.empty:
        bot_tickers = sorted(set(bot_tickers + index_pool["ticker"].tolist()))
    bot_flow = aggregate_bot_flow(
        base_dir,
        bot_tickers,
        max_rows=args.bot_max_rows if args.bot_max_rows > 0 else None,
    )
    candidates = generate_candidates(pool, hot, bot_flow, asof=asof, max_candidates=args.max_candidates)
    if not index_pool.empty:
        index_candidates = generate_candidates(
            index_pool,
            hot,
            bot_flow,
            asof=asof,
            max_candidates=12,
            index_fallback=True,
        )
        if not index_candidates.empty:
            candidates = pd.concat([candidates, index_candidates], ignore_index=True) if not candidates.empty else index_candidates
    fallback_candidates = build_fallback_income_candidates(
        stock_screener=sc,
        hot_chains=hot,
        liquidity_shift=liquidity_shift,
        asof=asof,
        max_candidates=12,
    )
    if not fallback_candidates.empty:
        candidates = (
            pd.concat([candidates, fallback_candidates], ignore_index=True)
            if not candidates.empty
            else fallback_candidates
        ).drop_duplicates(
            subset=["ticker", "direction", "expiry", "short_strike_eod", "long_strike_eod"],
            keep="first",
        )

    scored = live_validate_and_score(
        candidates,
        asof=asof,
        out_dir=out_dir,
        regime=regime,
        require_live=not args.offline,
        schwab_snapshot_dir=Path(args.schwab_snapshot_dir).expanduser().resolve() if args.schwab_snapshot_dir else None,
    )
    scored = apply_oi_carryover(scored, chain_oi)
    scored = apply_replay_edge_model(scored, base_dir.parent / "out")
    if args.skip_portfolio or args.offline:
        portfolio = unavailable_portfolio_context("skipped" if args.skip_portfolio else "offline")
    else:
        try:
            portfolio = fetch_portfolio_context(
                out_dir,
                portfolio_income_mode=args.portfolio_income_mode,
                covered_income_allowed_tickers=[
                    ticker.strip().upper()
                    for ticker in str(args.covered_income_allowed_tickers or "").split(",")
                    if ticker.strip()
                ],
            )
        except Exception as exc:
            portfolio = unavailable_portfolio_context(str(exc))
    scored = apply_portfolio_context(scored, portfolio)

    if args.skip_catalysts:
        catalysts = None
    else:
        catalyst_tickers = sorted(set(scored["ticker"].dropna().astype(str).str.upper())) if not scored.empty else []
        catalysts = load_catalyst_context(base_dir, catalyst_tickers, asof=asof)
    if catalysts is not None:
        scored = apply_catalyst_context(scored, catalysts)
    macro_gates = build_macro_event_gates(base_dir=base_dir, asof=asof, stock_screener=sc, regime=regime)
    scored = apply_final_quality_guards(scored)
    scored = apply_high_conviction_decision_marks(scored, asof=asof)
    recent_performance = (
        {"status": "unavailable", "reason": "skipped"}
        if args.skip_recent_performance
        else load_recent_performance(base_dir.parent / "out")
    )
    live_outcomes = load_live_outcome_performance(base_dir.parent / "out")
    scored = apply_confirmation_framework(scored, asof=asof, regime=regime, recent_performance=recent_performance)
    scored = apply_confidence_components(scored, live_outcomes=live_outcomes)
    loss_review = load_recent_loss_review(base_dir.parent / "out", asof=asof, lookback_days=args.loss_lookback_days)
    scored = apply_loss_review(scored, loss_review)
    scored = apply_liquidity_shift_context(scored, liquidity_shift, require_intraday_vwap=args.command == "intraday")
    confirmation_evidence = build_confirmation_evidence(scored=scored, asof=asof, input_provenance=input_provenance)
    scored = apply_confirmation_evidence(scored, confirmation_evidence)
    scored = assign_trade_statuses(scored, index_income_mode=args.index_income_mode)
    scored = apply_fallback_income_status(scored)
    data_quality = build_data_quality_status(
        input_provenance=input_provenance,
        scored=scored,
        portfolio=portfolio,
        catalysts=catalysts,
        recent_performance=recent_performance,
        live_outcomes=live_outcomes,
        run_mode=run_mode,
    )
    scored = apply_data_quality_gate(scored, data_quality)
    watchlist = build_entry_watchlist(scored)
    risk_config = _risk_config(args)
    if args.daily_loss_limit > 0 and portfolio and portfolio.get("status") == "ok":
        if float(portfolio.get("day_pnl") or 0.0) <= -abs(args.daily_loss_limit):
            risk_config["allow_new_trades"] = False
    final = select_final_trades(
        scored,
        regime=regime,
        risk_budget=args.risk_budget,
        recent_performance=recent_performance,
        max_final_trades=args.max_final_trades,
        risk_config=risk_config,
    )
    change_summary = build_intraday_change_summary(
        out_dir=out_dir,
        asof=asof,
        scored=scored,
        final=final,
        watch=watchlist,
        portfolio=portfolio,
        risk_budget=args.risk_budget,
    )
    return write_v3_outputs(
        out_dir=out_dir,
        repo_root=repo_root,
        asof=asof,
        base_dir=base_dir,
        run_mode=run_mode,
        market_data_status=_market_data_status(run_mode),
        input_provenance=input_provenance,
        data_quality=data_quality,
        change_summary=change_summary,
        regime=regime,
        candidates=candidates,
        scored=scored,
        final=final,
        watchlist=watchlist,
        portfolio=portfolio,
        catalysts=catalysts,
        macro_gates=macro_gates,
        confirmation_evidence=confirmation_evidence,
        liquidity_shift=liquidity_shift,
        v3_regime_context=v3_regime_context,
        recent_performance=recent_performance,
        live_outcomes=live_outcomes,
        loss_review=loss_review,
        risk_config=risk_config,
        args=args,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    root = Path(getattr(args, "root", DEFAULT_ROOT)).expanduser().resolve()
    if args.command in {"run", "intraday"}:
        base_dir = _base_dir_from_args(args)
        asof = infer_asof_date(base_dir)
        out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else _default_out_dir(root, asof, "run")
        manifest = run_v3_daily(base_dir=base_dir, out_dir=out_dir, args=args)
        print(f"Wrote: {manifest.get('report_path')}")
        print(f"Manifest: {manifest.get('manifest_path')}")
        print(f"Execute: {(manifest.get('opportunity_counts') or {}).get('execute', 0)}")
        print(f"Scout: {(manifest.get('opportunity_counts') or {}).get('scout', 0)}")
        return
    if args.command == "overlay":
        asof = _parse_date(args.date)
        if asof is None:
            raise SystemExit("--date is required")
        overlay_file = Path(args.overlay_file).expanduser().resolve()
        from .overlay import infer_date_from_name

        overlay_date = _parse_date(args.overlay_date) or infer_date_from_name(overlay_file) or asof
        prior_out_dir = (
            Path(args.prior_out_dir).expanduser().resolve()
            if args.prior_out_dir
            else root / "out" / f"codexdaily_v3_{asof}"
        )
        out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else _default_out_dir(root, asof, "overlay", overlay_date)
        manifest = run_overlay(
            prior_out_dir=prior_out_dir,
            overlay_file=overlay_file,
            out_dir=out_dir,
            asof=asof,
            overlay_date=overlay_date,
        )
        print(f"Wrote: {manifest.get('report_path')}")
        print(f"Manifest: {manifest.get('manifest_path')}")
        return
    if args.command == "validate":
        asof = _parse_date(args.as_of) or latest_dated_folder(root)
        if asof is None:
            raise SystemExit("No dated folders found for validation")
        out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else _default_out_dir(root, asof, "validation")
        runner = None
        if args.run_live:
            def runner(folder: Path, run_out: Path) -> dict[str, Any]:
                return run_v3_daily(base_dir=folder, out_dir=run_out, args=args, run_mode_override="Validation live-planning run")

        manifest = run_validation_harness(root=root, out_dir=out_dir, asof=asof, latest_n=args.latest_n, runner=runner)
        print(f"Wrote: {manifest.get('report_path')}")
        print(f"Manifest: {manifest.get('manifest_path')}")
        return
    if args.command == "loss-review":
        asof = _parse_date(args.as_of)
        if asof is None:
            raise SystemExit("--as-of is required")
        out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else root / "out" / f"codexdaily_v3_loss_review_{asof}"
        review = load_recent_loss_review(root / "out", asof=asof, lookback_days=args.loss_lookback_days)
        json_path, csv_path = write_loss_review(out_dir, asof, review)
        print(f"Wrote: {json_path}")
        print(f"Wrote: {csv_path}")


if __name__ == "__main__":
    main()
