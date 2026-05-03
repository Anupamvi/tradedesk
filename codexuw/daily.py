from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import pandas as pd

from .catalysts import load_catalyst_context
from .data import aggregate_bot_flow, infer_asof_date, load_chain_oi, load_hot_chains, load_stock_screener
from .engine import (
    apply_catalyst_context,
    apply_final_quality_guards,
    apply_high_conviction_decision_marks,
    apply_portfolio_context,
    build_entry_watchlist,
    detect_regime,
    generate_candidates,
    live_validate_and_score,
    select_final_trades,
    select_ticker_pool,
    write_outputs,
)
from .performance import load_recent_performance
from .portfolio import fetch_portfolio_context, unavailable_portfolio_context


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
    return parser.parse_args()


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
        f"# CodexUW Daily Options Income Report - {asof}",
        "",
        "## Funnel",
        "- raw_screener_rows: 0",
        "- ticker_pool_rows: 0",
        "- candidate_rows: 0",
        "- live_scored_rows: 0",
        "- hard_reject_rows: 1",
        "- final_trade_rows: 0",
        "",
        "## Final Decision",
        "",
        "No high-quality trades today",
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

    try:
        sc = load_stock_screener(base_dir)
        hot = load_hot_chains(base_dir, asof)
    except Exception as exc:
        report = write_data_error_report(out_dir, asof, base_dir, exc)
        print(f"Wrote: {report}")
        print("Final trades: 0")
        return
    # Loaded for traceability and future OI scoring. Candidate generation is
    # intentionally not blocked if this file is sparse.
    try:
        _ = load_chain_oi(base_dir, asof)
    except Exception:
        pass

    regime = detect_regime(sc)
    latest_asof = latest_dated_folder(base_dir.parent)
    if latest_asof and asof < latest_asof:
        regime["validation_note"] = (
            f"stale historical UW folder; later UW folder {latest_asof} exists. "
            "This run still uses current Schwab chains and portfolio, so use replay for historical judgment."
        )
    elif latest_asof == asof:
        regime["validation_note"] = "latest UW planning folder; current Schwab chains are used for executable pricing."
    pool = select_ticker_pool(sc, max_tickers=args.max_tickers)
    bot_flow = aggregate_bot_flow(
        base_dir,
        pool["ticker"].tolist(),
        max_rows=args.bot_max_rows if args.bot_max_rows > 0 else None,
    )
    candidates = generate_candidates(pool, hot, bot_flow, asof=asof, max_candidates=args.max_candidates)
    scored = live_validate_and_score(
        candidates,
        asof=asof,
        out_dir=out_dir,
        regime=regime,
        require_live=not args.offline,
    )
    if args.skip_portfolio or args.offline:
        portfolio = unavailable_portfolio_context("skipped")
    else:
        try:
            portfolio = fetch_portfolio_context(out_dir)
        except Exception as exc:
            portfolio = unavailable_portfolio_context(str(exc))
    scored = apply_portfolio_context(scored, portfolio)

    if args.skip_catalysts:
        catalysts = None
    else:
        catalyst_tickers = sorted(set(scored["ticker"].dropna().astype(str).str.upper())) if not scored.empty else []
        catalysts = load_catalyst_context(base_dir, catalyst_tickers)
        scored = apply_catalyst_context(scored, catalysts)
    scored = apply_final_quality_guards(scored)
    scored = apply_high_conviction_decision_marks(scored, asof=asof)
    watchlist = build_entry_watchlist(scored)

    recent_performance = (
        {"status": "unavailable", "reason": "skipped"}
        if args.skip_recent_performance
        else load_recent_performance(base_dir.parent / "out")
    )
    final = select_final_trades(
        scored,
        regime=regime,
        risk_budget=args.risk_budget,
        recent_performance=recent_performance,
        max_final_trades=args.max_final_trades,
    )
    funnel = {
        "raw_screener_rows": int(len(sc)),
        "ticker_pool_rows": int(len(pool)),
        "candidate_rows": int(len(candidates)),
        "live_scored_rows": int(len(scored)),
        "hard_reject_rows": int(scored["hard_rejects"].fillna("").ne("").sum()) if not scored.empty and "hard_rejects" in scored.columns else 0,
        "final_trade_rows": int(len(final)),
    }
    report = write_outputs(
        out_dir=out_dir,
        asof=asof,
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
    )
    print(f"Wrote: {report}")
    print(f"Final trades: {len(final)}")


if __name__ == "__main__":
    main()
