from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import pandas as pd

from .catalysts import load_catalyst_context
from .data import aggregate_bot_flow, infer_asof_date, load_chain_oi, load_hot_chains, load_stock_screener
from .engine import (
    apply_confirmation_framework,
    apply_catalyst_context,
    apply_confidence_components,
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
    build_target_capital_model,
    detect_regime,
    generate_candidates,
    live_validate_and_score,
    select_final_trades,
    select_index_fallback_pool,
    select_ticker_pool,
    write_outputs,
)
from .performance import load_live_outcome_performance, load_recent_performance
from .portfolio import fetch_portfolio_context, unavailable_portfolio_context
from .provenance import build_input_provenance, build_run_environment, build_schwab_snapshot_provenance


def latest_dated_folder(root: Path) -> dt.date | None:
    latest: dt.date | None = None
    for child in root.iterdir():
        if not child.is_dir():
            continue
        try:
            day = infer_asof_date(child)
        except ValueError:
            continue
        if latest is None or day > latest:
            latest = day
    return latest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean CodexUW daily options income pipeline")
    parser.add_argument("--base-dir", required=True, help="Dated UW folder, e.g. /path/to/2026-04-30")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--max-tickers", type=int, default=28)
    parser.add_argument("--max-candidates", type=int, default=50)
    parser.add_argument("--max-final-trades", type=int, default=8, help="Maximum final executable trades to select without forcing lower-quality setups")
    parser.add_argument("--risk-budget", type=float, default=3000.0)
    parser.add_argument("--bot-max-rows", type=int, default=0, help="Optional cap for bot tape rows; 0 means full file")
    parser.add_argument("--offline", action="store_true", help="Skip Schwab live chain validation")
    parser.add_argument("--skip-portfolio", action="store_true", help="Skip Schwab account-position exposure guard")
    parser.add_argument("--skip-catalysts", action="store_true", help="Skip local browser/news catalyst checks")
    parser.add_argument("--skip-recent-performance", action="store_true", help="Skip recent replay-equivalent performance feedback")
    parser.add_argument("--schwab-snapshot-dir", default="", help="Optional existing schwab_chains directory or run out-dir to replay exact chain snapshots")
    parser.add_argument("--historical-replay", action="store_true", help="Delegate this dated run to codexuw.replay historical mode")
    parser.add_argument("--replay-end", default="", help="Optional historical replay end date; defaults to latest dated folder")
    parser.add_argument(
        "--report-mode",
        default="auto",
        choices=["auto", "pre-market", "intraday", "post-close", "historical"],
        help="Label the decision report mode. auto infers from local market time unless --historical-replay is set.",
    )
    parser.add_argument("--max-risk-per-trade", type=float, default=0.0, help="Optional hard max risk per new trade.")
    parser.add_argument("--max-risk-per-day", type=float, default=0.0, help="Optional hard max total new risk for the report.")
    parser.add_argument("--max-open-risk-by-ticker", type=float, default=0.0, help="Optional hard max new open risk by ticker.")
    parser.add_argument("--max-correlated-sector-exposure", type=float, default=0.0, help="Optional hard max new correlated sector/factor risk.")
    parser.add_argument("--daily-loss-limit", type=float, default=0.0, help="If Schwab day P/L is below this loss, new trades move out of Execute.")
    parser.add_argument("--monthly-profit-target", type=float, default=10000.0, help="Target-aware v2 monthly P/L objective.")
    parser.add_argument("--month-to-date-realized-pnl", type=float, default=0.0, help="Known realized month-to-date P/L for target feasibility.")
    parser.add_argument("--max-monthly-drawdown", type=float, default=0.0, help="Optional monthly drawdown stop for target feasibility.")
    parser.add_argument("--max-total-open-risk", type=float, default=0.0, help="Optional hard cap for total new/open risk considered by v2 sizing.")
    parser.add_argument("--max-contracts-per-trade", type=float, default=0.0, help="Optional hard cap for contracts per Execute trade.")
    parser.add_argument(
        "--risk-mandate",
        default="capital-preservation",
        choices=["capital-preservation", "balanced", "target-growth", "aggressive-intraday"],
        help="Capital-allocation posture for v2 sizing. Higher mandates still require live quality, positive expectancy, sample size, and risk caps.",
    )
    parser.add_argument(
        "--index-income-mode",
        default="fallback",
        choices=["disabled", "fallback", "primary"],
        help="How SPY/QQQ/IWM index-income candidates are handled. primary lets them compete with single-name trades.",
    )
    parser.add_argument(
        "--portfolio-income-mode",
        default="trading-sleeve-only",
        choices=["disabled", "trading-sleeve-only", "existing-core-review"],
        help="Controls covered-income recommendations. trading-sleeve-only protects core investments unless tickers are explicitly allowed.",
    )
    parser.add_argument(
        "--covered-income-allowed-tickers",
        default="",
        help="Comma-separated tickers allowed for covered-income review when --portfolio-income-mode=trading-sleeve-only.",
    )
    parser.add_argument(
        "--minimum-expected-value-per-dollar-risk",
        type=float,
        default=0.0,
        help="Minimum positive expected value per dollar risk required before v2 size-up.",
    )
    return parser.parse_args()


def _parse_date(value: str) -> dt.date | None:
    if not value:
        return None
    return dt.datetime.strptime(value, "%Y-%m-%d").date()


def live_planning_validation_note(asof: dt.date, latest_asof: dt.date | None) -> str:
    if latest_asof and asof < latest_asof:
        return (
            f"This is a live-planning run using current Schwab chains against historical UW flow from {asof}. "
            f"A later UW folder {latest_asof} exists. Use codexuw.replay / historical mode for would-have-executed historical evaluation."
        )
    if latest_asof == asof:
        return "latest UW planning folder; current Schwab chains are used for executable pricing."
    return ""


def infer_report_mode(value: str, *, historical_replay: bool) -> str:
    if historical_replay or value == "historical":
        return "Historical audit/replay"
    if value == "pre-market":
        return "Pre-market planning"
    if value == "intraday":
        return "Intraday live execution"
    if value == "post-close":
        return "Post-close review"
    now = dt.datetime.now().time()
    if now < dt.time(6, 30):
        return "Pre-market planning"
    if now <= dt.time(13, 0):
        return "Intraday live execution"
    return "Post-close review"


def write_data_error_report(out_dir: Path, asof, base_dir: Path, error: Exception) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    reason = f"data_problem_missing_required_data: {error}"
    pd.DataFrame().to_csv(out_dir / f"codexuw_candidates_{asof}.csv", index=False)
    pd.DataFrame().to_csv(out_dir / f"codexuw_scored_{asof}.csv", index=False)
    pd.DataFrame().to_csv(out_dir / f"codexuw_final_trades_{asof}.csv", index=False)
    pd.DataFrame().to_csv(out_dir / f"codexuw_entry_watchlist_{asof}.csv", index=False)
    pd.DataFrame([{"reason": reason, "count": 1}]).to_csv(out_dir / f"codexuw_rejections_{asof}.csv", index=False)
    funnel = {
        "raw_screener_rows": 0,
        "ticker_pool_rows": 0,
        "candidate_rows": 0,
        "live_scored_rows": 0,
        "hard_reject_rows": 1,
        "final_trade_rows": 0,
    }
    (out_dir / f"codexuw_manifest_{asof}.json").write_text(
        json.dumps(
            {
                "asof": str(asof),
                "base_dir": str(base_dir),
                "funnel": funnel,
                "status": "no_trade_data_problem",
                "reason": reason,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    report = out_dir / f"codexuw_trade_report_{asof}.md"
    lines = [
        f"# CodexUW Daily Decision Engine - {asof}",
        "",
        "## First Screen",
        "",
        "| Item | Value |",
        "|:--|:--|",
        "| Report mode | Data error / no recommendation |",
        "| Data-quality status | 🔴 critical |",
        "| Market regime | unavailable |",
        "| Portfolio risk | not checked because required UW files failed |",
        "| Top action today | Stand down until required input files are present |",
        "| Execute Now count | 0 |",
        "| Enter Only At Price count | 0 |",
        "| Portfolio Income count | 0 |",
        "| Watch count | 0 |",
        "",
        "## Data Quality Gate",
        "",
        "| Status | Check | Detail |",
        "|:--|:--|:--|",
        f"| 🔴 missing | UW files present | {reason} |",
        "",
        "## Final Decision",
        "",
        "No high-quality trades today. No high-quality Execute trades today.",
        "",
        "## No-Trade Reason",
        "- Issue type: data problem",
        f"- Base folder: {base_dir}",
        f"- Exact reason: {reason}",
        "",
        "## Rejected Candidate Summary",
        "",
        "| reason | count |",
        "|:--|--:|",
        f"| {reason} | 1 |",
        "",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    asof = infer_asof_date(base_dir)
    input_provenance = build_input_provenance(base_dir)
    run_mode = infer_report_mode(args.report_mode, historical_replay=args.historical_replay)

    if args.historical_replay:
        from .replay import run_replay

        replay_end = _parse_date(args.replay_end) or latest_dated_folder(base_dir.parent) or asof
        report = run_replay(
            base_dir.parent,
            out_dir,
            asof,
            replay_end,
            0,
            entry_start=asof,
            entry_end=asof,
            report_date=asof,
            max_tickers=args.max_tickers,
            max_candidates=args.max_candidates,
            max_eval_candidates=args.max_candidates,
            max_selected_per_day=args.max_final_trades,
            bot_max_rows=args.bot_max_rows,
            monthly_profit_target=args.monthly_profit_target,
        )
        print("Historical audit/replay mode: delegated to codexuw.replay for would-have-executed evaluation.")
        print(f"Wrote: {report}")
        print(f"Wrote: {out_dir / f'codexuw_replay_trade_report_{asof}.md'}")
        return

    try:
        sc = load_stock_screener(base_dir)
        hot = load_hot_chains(base_dir, asof)
    except Exception as exc:
        report = write_data_error_report(out_dir, asof, base_dir, exc)
        print(f"Wrote: {report}")
        print("Final trades: 0")
        return
    chain_oi = None
    try:
        chain_oi = load_chain_oi(base_dir, asof)
    except Exception:
        chain_oi = None

    regime = detect_regime(sc)
    latest_asof = latest_dated_folder(base_dir.parent)
    validation_note = live_planning_validation_note(asof, latest_asof)
    if validation_note:
        regime["validation_note"] = validation_note
    pool = select_ticker_pool(sc, max_tickers=args.max_tickers)
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
        portfolio = unavailable_portfolio_context("skipped")
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
    scored = apply_final_quality_guards(scored)
    scored = apply_high_conviction_decision_marks(scored, asof=asof)

    recent_performance = (
        {"status": "unavailable", "reason": "skipped"}
        if args.skip_recent_performance
        else load_recent_performance(base_dir.parent / "out")
    )
    live_outcomes = load_live_outcome_performance(base_dir.parent / "out")
    scored = apply_confirmation_framework(scored, asof=asof, regime=regime, recent_performance=recent_performance)
    data_quality = build_data_quality_status(
        input_provenance=input_provenance,
        scored=scored,
        portfolio=portfolio,
        catalysts=catalysts,
        recent_performance=recent_performance,
        live_outcomes=live_outcomes,
        run_mode=run_mode,
    )
    scored = apply_confidence_components(scored, live_outcomes=live_outcomes)
    scored = assign_trade_statuses(scored, index_income_mode=args.index_income_mode)
    scored = apply_data_quality_gate(scored, data_quality)
    watchlist = build_entry_watchlist(scored)
    risk_config = {
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
    target_model = build_target_capital_model(
        asof=asof,
        monthly_profit_target=args.monthly_profit_target,
        month_to_date_realized_pnl=args.month_to_date_realized_pnl,
        max_monthly_drawdown=args.max_monthly_drawdown,
        risk_budget=args.risk_budget,
        risk_config=risk_config,
        portfolio=portfolio,
        final=final,
    )
    funnel = {
        "raw_screener_rows": int(len(sc)),
        "ticker_pool_rows": int(len(pool)),
        "candidate_rows": int(len(candidates)),
        "live_scored_rows": int(len(scored)),
        "hard_reject_rows": int(scored["hard_rejects"].fillna("").ne("").sum()) if not scored.empty and "hard_rejects" in scored.columns else 0,
        "final_trade_rows": int(len(final)),
        "watch_rows": int(len(watchlist)),
        "research_rows": int(scored["trade_status"].eq("Research").sum()) if not scored.empty and "trade_status" in scored.columns else 0,
        "avoid_rows": int(scored["trade_status"].eq("Avoid").sum()) if not scored.empty and "trade_status" in scored.columns else 0,
    }
    run_provenance = {
        "environment": build_run_environment(),
        "input_files": input_provenance,
        "schwab_snapshot": build_schwab_snapshot_provenance(out_dir),
        "schwab_snapshot_input_dir": str(Path(args.schwab_snapshot_dir).expanduser().resolve()) if args.schwab_snapshot_dir else "",
        "mode": "offline" if args.offline else run_mode,
        "risk_config": risk_config,
        "target_model": target_model,
    }
    change_summary = build_intraday_change_summary(
        out_dir=out_dir,
        asof=asof,
        scored=scored,
        final=final,
        watch=watchlist,
        portfolio=portfolio,
        risk_budget=args.risk_budget,
    )
    report = write_outputs(
        out_dir=out_dir,
        asof=asof,
        run_mode=run_mode,
        data_quality=data_quality,
        change_summary=change_summary,
        live_outcomes=live_outcomes,
        regime=regime,
        candidates=candidates,
        scored=scored,
        final=final,
        funnel=funnel,
        portfolio=portfolio,
        catalysts=catalysts,
        recent_performance=recent_performance,
        watchlist=watchlist,
        max_final_trades=args.max_final_trades,
        run_provenance=run_provenance,
        target_model=target_model,
    )
    print(f"Wrote: {report}")
    print(f"Final trades: {len(final)}")


if __name__ == "__main__":
    main()
