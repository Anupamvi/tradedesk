#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from uwos import swing_trend_pipeline as swing
from uwos import trend_analysis


DEFAULT_START = dt.date(2025, 12, 1)
DEFAULT_BATCH_TOP = 3
DEFAULT_BATCH_SAMPLES = 10000
DEFAULT_BATCH_HORIZONS = "5,10,20"

BATCH_OUTCOME_COLUMNS = [
    "signal_date",
    "horizon_market_days",
    "exit_date",
    "selected_rank",
    "ticker",
    "direction",
    "strategy",
    "trade_setup",
    "swing_score",
    "edge_pct",
    "backtest_signals",
    "backtest_verdict",
    "entry_net",
    "exit_net",
    "pnl",
    "return_on_risk",
    "win",
    "outcome_verdict",
    "outcome_status",
    "outcome_reason",
]


def _parse_date(value: str) -> dt.date:
    return dt.date.fromisoformat(str(value).strip()[:10])


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a historical as-of proof loop for trend-analysis and report "
            "whether emitted trades had positive expectancy."
        )
    )
    parser.add_argument("--start", default=DEFAULT_START.isoformat(), help="Start date YYYY-MM-DD.")
    parser.add_argument("--end", default="", help="End date YYYY-MM-DD. Default: latest usable date.")
    parser.add_argument(
        "--lookback",
        type=_positive_int,
        default=30,
        help="Usable market-data-day lookback per as-of date. Default: 30.",
    )
    parser.add_argument(
        "--horizons",
        default=DEFAULT_BATCH_HORIZONS,
        help=f"Comma-separated market-day outcome horizons. Default: {DEFAULT_BATCH_HORIZONS}.",
    )
    parser.add_argument(
        "--top",
        type=_positive_int,
        default=DEFAULT_BATCH_TOP,
        help=f"Strict emitted trades per signal date. Default: {DEFAULT_BATCH_TOP}.",
    )
    parser.add_argument(
        "--samples",
        type=_positive_int,
        default=DEFAULT_BATCH_SAMPLES,
        help="Maximum eligible historical signal dates to evaluate. Default: all.",
    )
    parser.add_argument(
        "--candidate-pool",
        type=_positive_int,
        default=45,
        help="Historical raw candidate pool per signal date. Default: 45.",
    )
    parser.add_argument(
        "--root-dir",
        default="",
        help="Root directory containing YYYY-MM-DD folders. Default: repo/data root.",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Batch proof output directory. Default: <root-dir>/out/trend_analysis_batch.",
    )
    parser.add_argument(
        "--raw-cache-dir",
        default="",
        help=(
            "Directory whose walk_forward subfolder stores per-date raw candidates. "
            "Default: <root-dir>/out/trend_analysis."
        ),
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "rulebook_config_swing_trend.yaml"),
        help="Swing trend YAML config to reuse for scoring.",
    )
    parser.add_argument(
        "--cache-dir",
        default="",
        help="Optional cache directory used by setup_likelihood_backtest.",
    )
    parser.add_argument(
        "--reuse-raw",
        action="store_true",
        help="Reuse existing per-date raw candidate CSVs when present.",
    )
    parser.add_argument("--min-backtest-edge", type=float, default=0.0)
    parser.add_argument("--min-backtest-signals", type=int, default=100)
    parser.add_argument("--min-workup-signals", type=int, default=trend_analysis.DEFAULT_MIN_WORKUP_SIGNALS)
    parser.add_argument("--min-swing-score", type=float, default=trend_analysis.DEFAULT_CANDIDATE_MIN_SCORE)
    parser.add_argument("--allow-low-sample", action="store_true")
    parser.add_argument("--allow-earnings-risk", action="store_true")
    parser.add_argument("--allow-volatile-ic", action="store_true")
    parser.add_argument("--allow-flow-conflict", action="store_true")
    parser.add_argument(
        "--max-bid-ask-to-price-pct",
        type=float,
        default=trend_analysis.DEFAULT_MAX_BID_ASK_TO_PRICE_PCT,
    )
    parser.add_argument(
        "--max-bid-ask-to-width-pct",
        type=float,
        default=trend_analysis.DEFAULT_MAX_BID_ASK_TO_WIDTH_PCT,
    )
    parser.add_argument("--max-short-delta", type=float, default=trend_analysis.DEFAULT_MAX_SHORT_DELTA)
    parser.add_argument("--min-underlying-price", type=float, default=trend_analysis.DEFAULT_MIN_UNDERLYING_PRICE)
    parser.add_argument(
        "--min-debit-spread-price",
        type=float,
        default=trend_analysis.DEFAULT_MIN_DEBIT_SPREAD_PRICE,
    )
    parser.add_argument(
        "--min-whale-appearances",
        type=int,
        default=trend_analysis.DEFAULT_MIN_WHALE_APPEARANCES,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _parse_horizons(value: str) -> List[int]:
    horizons: List[int] = []
    for part in str(value or "").split(","):
        text = part.strip()
        if not text:
            continue
        parsed = int(text)
        if parsed <= 0:
            raise ValueError(f"Horizon must be positive: {text}")
        horizons.append(parsed)
    return sorted(set(horizons))


def _filter_signal_dates(df: pd.DataFrame, start: dt.date, end: dt.date) -> pd.DataFrame:
    if df.empty or "signal_date" not in df.columns:
        return df.copy()
    out = df.copy()
    dates = pd.to_datetime(out["signal_date"], errors="coerce").dt.date
    return out[dates.ge(start) & dates.le(end)].copy().reset_index(drop=True)


def _profit_factor(pnl: pd.Series) -> float:
    values = pd.to_numeric(pnl, errors="coerce").dropna()
    gains = float(values[values > 0].sum())
    losses = abs(float(values[values < 0].sum()))
    if losses <= 0:
        return math.inf if gains > 0 else math.nan
    return gains / losses


def _max_drawdown_by_signal_date(df: pd.DataFrame) -> float:
    if df.empty or "signal_date" not in df.columns or "pnl" not in df.columns:
        return math.nan
    work = df.copy()
    work["_pnl"] = pd.to_numeric(work["pnl"], errors="coerce")
    work = work[work["_pnl"].notna()].copy()
    if work.empty:
        return math.nan
    by_day = work.groupby("signal_date")["_pnl"].sum().sort_index()
    equity = by_day.cumsum()
    peak = equity.cummax()
    drawdown = equity - peak
    return float(drawdown.min())


def summarize_outcomes(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty or "pnl" not in df.columns:
        return {
            "trades": 0,
            "signal_dates": 0,
            "win_rate": math.nan,
            "avg_pnl": math.nan,
            "median_pnl": math.nan,
            "total_pnl": 0.0,
            "profit_factor": math.nan,
            "avg_return_on_risk": math.nan,
            "worst_pnl": math.nan,
            "best_pnl": math.nan,
            "max_drawdown": math.nan,
        }
    pnl = pd.to_numeric(df["pnl"], errors="coerce").dropna()
    rr = pd.to_numeric(df.get("return_on_risk", pd.Series(np.nan, index=df.index)), errors="coerce").dropna()
    return {
        "trades": int(len(pnl)),
        "signal_dates": int(df.get("signal_date", pd.Series(dtype=str)).dropna().nunique()),
        "win_rate": float(pnl.gt(0).mean()) if len(pnl) else math.nan,
        "avg_pnl": float(pnl.mean()) if len(pnl) else math.nan,
        "median_pnl": float(pnl.median()) if len(pnl) else math.nan,
        "total_pnl": float(pnl.sum()) if len(pnl) else 0.0,
        "profit_factor": _profit_factor(pnl),
        "avg_return_on_risk": float(rr.mean()) if len(rr) else math.nan,
        "worst_pnl": float(pnl.min()) if len(pnl) else math.nan,
        "best_pnl": float(pnl.max()) if len(pnl) else math.nan,
        "max_drawdown": _max_drawdown_by_signal_date(df),
    }


def _summary_verdict(summary: Dict[str, Any]) -> str:
    trades = int(summary.get("trades", 0) or 0)
    avg = trend_analysis._safe_float(summary.get("avg_pnl"))
    pf = trend_analysis._safe_float(summary.get("profit_factor"))
    win = trend_analysis._safe_float(summary.get("win_rate"))
    if trades < 20:
        return "NO_PROOF_LOW_SAMPLE"
    if math.isfinite(avg) and avg > 0 and math.isfinite(pf) and pf >= 1.20 and math.isfinite(win) and win >= 0.45:
        return "PROMOTABLE_WITH_DEFINED_RISK"
    if math.isfinite(avg) and avg > 0 and math.isfinite(pf) and pf >= 1.05:
        return "WATCHLIST_POSITIVE_BUT_WEAK"
    return "NOT_PROMOTABLE"


def _fmt_money(value: Any) -> str:
    parsed = trend_analysis._safe_float(value)
    if not math.isfinite(parsed):
        return "-"
    return f"${parsed:,.2f}"


def _fmt_num(value: Any, digits: int = 2) -> str:
    parsed = trend_analysis._safe_float(value)
    if not math.isfinite(parsed):
        return "-"
    return f"{parsed:.{digits}f}"


def _fmt_pct(value: Any) -> str:
    parsed = trend_analysis._safe_float(value)
    if not math.isfinite(parsed):
        return "-"
    return f"{parsed:.0%}"


def _render_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    return trend_analysis._render_table(headers, rows)


def _file_link(path: Path) -> str:
    return f"[{path.name}]({path})"


def _summary_table_row(label: str, summary: Dict[str, Any]) -> List[str]:
    return [
        label,
        str(int(summary.get("trades", 0) or 0)),
        str(int(summary.get("signal_dates", 0) or 0)),
        _fmt_pct(summary.get("win_rate")),
        _fmt_money(summary.get("avg_pnl")),
        _fmt_money(summary.get("total_pnl")),
        _fmt_num(summary.get("profit_factor"), 2),
        _fmt_money(summary.get("max_drawdown")),
        _summary_verdict(summary),
    ]


def _group_summary_table(df: pd.DataFrame, by: str, *, limit: int = 16) -> str:
    if df.empty or by not in df.columns:
        return "_none_"
    ranked_rows: List[Tuple[float, List[str]]] = []
    for value, group in df.groupby(by, dropna=False):
        summary = summarize_outcomes(group)
        total = trend_analysis._safe_float(summary.get("total_pnl"))
        ranked_rows.append((total if math.isfinite(total) else -math.inf, _summary_table_row(str(value), summary)))
    rows = [row for _, row in sorted(ranked_rows, key=lambda item: item[0], reverse=True)]
    return _render_table(
        ["bucket", "trades", "dates", "win", "avg", "total", "PF", "drawdown", "verdict"],
        rows[:limit],
    )


def build_report(
    *,
    start: dt.date,
    end: dt.date,
    lookback: int,
    horizons: Sequence[int],
    strict_outcomes: pd.DataFrame,
    research_outcomes: pd.DataFrame,
    research_summary: pd.DataFrame,
    horizon_summary: pd.DataFrame,
    strategy_family_audit: pd.DataFrame,
    ticker_playbook_audit: pd.DataFrame,
    rolling_ticker_playbook_audit: pd.DataFrame,
    output_paths: Dict[str, Path],
) -> str:
    strict_summary = summarize_outcomes(strict_outcomes)
    research_summary_all = summarize_outcomes(research_outcomes)
    verdict = _summary_verdict(strict_summary)

    lines: List[str] = []
    lines.append(f"# Trend Analysis Batch Proof - {start.isoformat()} to {end.isoformat()} / L{lookback}")
    lines.append("")
    lines.append("## The Question")
    lines.append(
        "Had we run `trend-analysis` every eligible day, what trades would it have recommended, what happened next, and which playbooks actually made money?"
    )
    lines.append("")
    lines.append("## Answer")
    if verdict == "PROMOTABLE_WITH_DEFINED_RISK":
        lines.append(
            "The strict emitted-trade audit is positive enough to consider defined-risk promotion, subject to live/paper confirmation and sizing caps."
        )
    elif verdict == "NO_PROOF_LOW_SAMPLE":
        lines.append(
            "Not proven yet. The strict emitted-trade sample is too small, so this is still an idea engine until more historical or live outcomes accumulate."
        )
    else:
        lines.append(
            "Not a money printer yet. The strict emitted-trade audit does not have positive enough expectancy to promote sizing."
        )
    lines.append("")
    lines.append(
        _render_table(
            ["scope", "trades", "dates", "win", "avg", "total", "PF", "drawdown", "verdict"],
            [
                _summary_table_row("strict emitted trades", strict_summary),
                _summary_table_row("all research outcomes", research_summary_all),
            ],
        )
    )
    lines.append("")
    lines.append("## Strict Emitted Trades")
    if strict_outcomes.empty:
        lines.append(
            "No strict historical trades cleared the order-ready gate with completed future option quotes in this window. That is useful: the current gate is safer, but too sparse to prove profitability."
        )
    else:
        lines.append(_group_summary_table(strict_outcomes, "horizon_market_days"))
        lines.append("")
        lines.append(_group_summary_table(strict_outcomes, "ticker"))
    lines.append("")

    lines.append("## Research Policy Expectancy")
    if research_summary.empty:
        lines.append("_no completed policy outcomes_")
    else:
        lines.append(
            _render_table(
                ["policy", "outcomes", "setups", "hit", "avg", "avg R/R", "worst", "verdict"],
                [
                    [
                        str(row.get("policy", "")),
                        str(int(row.get("outcomes", 0) or 0)),
                        str(int(row.get("unique_setups", 0) or 0)),
                        _fmt_pct(row.get("hit_rate")),
                        _fmt_money(row.get("avg_pnl")),
                        _fmt_pct(row.get("avg_return_on_risk")),
                        _fmt_money(row.get("worst_pnl")),
                        str(row.get("verdict", "")),
                    ]
                    for _, row in research_summary.iterrows()
                ],
            )
        )
    lines.append("")

    lines.append("## Horizon Expectancy")
    if horizon_summary.empty:
        lines.append("_no completed horizon outcomes_")
    else:
        lines.append(
            _render_table(
                ["policy", "horizon", "outcomes", "setups", "hit", "avg", "avg R/R", "worst", "verdict"],
                [
                    [
                        str(row.get("policy", "")),
                        str(int(row.get("horizon_market_days", 0) or 0)),
                        str(int(row.get("outcomes", 0) or 0)),
                        str(int(row.get("unique_setups", 0) or 0)),
                        _fmt_pct(row.get("hit_rate")),
                        _fmt_money(row.get("avg_pnl")),
                        _fmt_pct(row.get("avg_return_on_risk")),
                        _fmt_money(row.get("worst_pnl")),
                        str(row.get("verdict", "")),
                    ]
                    for _, row in horizon_summary.head(20).iterrows()
                ],
            )
        )
    lines.append("")

    lines.append("## Playbook Validation")
    lines.append("### Strategy Families")
    lines.extend(trend_analysis._strategy_family_report_lines(strategy_family_audit))
    lines.append("")
    lines.append("### Ticker Playbooks")
    lines.extend(trend_analysis._ticker_playbook_report_lines(ticker_playbook_audit))
    lines.append("")
    lines.append("### Rolling Ticker Playbooks")
    lines.extend(trend_analysis._rolling_ticker_playbook_report_lines(rolling_ticker_playbook_audit))
    lines.append("")

    lines.append("## Go / No-Go")
    lines.append("- Do not call this a money printer until strict emitted trades have enough completed outcomes.")
    lines.append("- Promote only playbooks with positive train/validation and rolling-forward evidence.")
    lines.append("- Keep low-sample or insufficient-forward rows at `STARTER_RISK` or lower.")
    lines.append("- Re-run this proof after new dated folders arrive; the answer should be allowed to stay `NO_PROOF`.")
    lines.append("")

    lines.append("## Files")
    for label, path in output_paths.items():
        lines.append(f"- {label}: {_file_link(path)}")
    return "\n".join(lines)


def _output_paths(out_dir: Path, start: dt.date, end: dt.date, lookback: int) -> Dict[str, Path]:
    suffix = f"{start.isoformat()}_{end.isoformat()}-L{lookback}"
    return {
        "Report": out_dir / f"trend-analysis-batch-proof-{suffix}.md",
        "Strict emitted trades CSV": out_dir / f"trend-analysis-batch-strict-trades-{suffix}.csv",
        "Research outcomes CSV": out_dir / f"trend-analysis-batch-research-outcomes-{suffix}.csv",
        "Research summary CSV": out_dir / f"trend-analysis-batch-research-summary-{suffix}.csv",
        "Horizon summary CSV": out_dir / f"trend-analysis-batch-horizon-summary-{suffix}.csv",
        "Strategy family audit CSV": out_dir / f"trend-analysis-batch-strategy-family-audit-{suffix}.csv",
        "Ticker playbook audit CSV": out_dir / f"trend-analysis-batch-ticker-playbook-audit-{suffix}.csv",
        "Rolling ticker playbook audit CSV": out_dir / f"trend-analysis-batch-rolling-ticker-playbook-audit-{suffix}.csv",
        "Metadata JSON": out_dir / f"trend-analysis-batch-metadata-{suffix}.json",
    }


def run(argv: Optional[Sequence[str]] = None) -> Dict[str, Path]:
    args = parse_args(argv)
    root = Path(args.root_dir).expanduser().resolve() if args.root_dir else trend_analysis.default_root_dir()
    end = _parse_date(args.end) if str(args.end or "").strip() else swing.discover_trading_days(root, 1, None)[-1][0]
    start = _parse_date(args.start)
    lookback = int(args.lookback)
    horizons = _parse_horizons(args.horizons)
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else (root / "out" / "trend_analysis_batch").resolve()
    )
    raw_cache_dir = (
        Path(args.raw_cache_dir).expanduser().resolve()
        if args.raw_cache_dir
        else (root / "out" / "trend_analysis").resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_cache_dir.mkdir(parents=True, exist_ok=True)
    paths = _output_paths(out_dir, start, end, lookback)

    cfg = trend_analysis._load_config(Path(args.config).expanduser().resolve())
    cfg.setdefault("schwab_validation", {})["enabled"] = False
    cfg.setdefault("backtest", {})["enabled"] = True
    cfg.setdefault("backtest", {})["min_signals"] = int(args.min_backtest_signals)
    if args.cache_dir:
        cfg.setdefault("backtest", {})["cache_dir"] = args.cache_dir

    print("Trend Analysis Batch Proof", flush=True)
    print(f"  Root: {root}", flush=True)
    print(f"  Window: {start.isoformat()} to {end.isoformat()}", flush=True)
    print(f"  Lookback: {lookback} market-data days", flush=True)
    print(f"  Horizons: {','.join(map(str, horizons))} market days", flush=True)
    print(f"  Raw cache: {raw_cache_dir}", flush=True)
    print("  Historical execution source: local UW option quote replay; Schwab live validation skipped", flush=True)

    strict_outcomes = trend_analysis.run_walk_forward_audit(
        root=root,
        out_dir=raw_cache_dir,
        cfg_template=cfg,
        as_of=end,
        lookback=lookback,
        candidate_pool=int(args.candidate_pool),
        top_n=int(args.top),
        samples=int(args.samples),
        horizons=horizons,
        cache_dir=str(args.cache_dir or ""),
        min_edge=float(args.min_backtest_edge),
        min_signals=int(args.min_backtest_signals),
        min_swing_score=float(args.min_swing_score),
        allow_low_sample=bool(args.allow_low_sample),
        allow_earnings_risk=bool(args.allow_earnings_risk),
        allow_volatile_ic=bool(args.allow_volatile_ic),
        allow_flow_conflict=bool(args.allow_flow_conflict),
        max_bid_ask_to_price_pct=float(args.max_bid_ask_to_price_pct),
        max_bid_ask_to_width_pct=float(args.max_bid_ask_to_width_pct),
        max_short_delta=float(args.max_short_delta),
        min_underlying_price=float(args.min_underlying_price),
        min_debit_spread_price=float(args.min_debit_spread_price),
        min_whale_appearances=int(args.min_whale_appearances),
        reuse_raw=bool(args.reuse_raw),
    )
    strict_outcomes = _filter_signal_dates(strict_outcomes, start, end)

    research_outcomes = trend_analysis.collect_research_confidence_outcomes(
        root=root,
        out_dir=raw_cache_dir,
        as_of=end,
        lookback=lookback,
        samples=int(args.samples),
        horizons=horizons,
        min_edge=float(args.min_backtest_edge),
        min_signals=int(args.min_backtest_signals),
        min_workup_signals=int(args.min_workup_signals),
        min_swing_score=float(args.min_swing_score),
        allow_low_sample=bool(args.allow_low_sample),
        allow_earnings_risk=bool(args.allow_earnings_risk),
        allow_volatile_ic=bool(args.allow_volatile_ic),
        allow_flow_conflict=bool(args.allow_flow_conflict),
        max_bid_ask_to_price_pct=float(args.max_bid_ask_to_price_pct),
        max_bid_ask_to_width_pct=float(args.max_bid_ask_to_width_pct),
        max_short_delta=float(args.max_short_delta),
        min_underlying_price=float(args.min_underlying_price),
        min_debit_spread_price=float(args.min_debit_spread_price),
        min_whale_appearances=int(args.min_whale_appearances),
    )
    research_outcomes = _filter_signal_dates(research_outcomes, start, end)
    research_summary = trend_analysis._research_summary_from_outcomes(research_outcomes)
    horizon_summary = trend_analysis._research_summary_by_horizon_from_outcomes(research_outcomes)
    strategy_family_audit = trend_analysis._strategy_family_audit_from_outcomes(research_outcomes)
    ticker_playbook_audit = trend_analysis._ticker_playbook_audit_from_outcomes(research_outcomes)
    rolling_ticker_playbook_audit = trend_analysis._rolling_ticker_playbook_audit_from_outcomes(
        research_outcomes,
        ticker_playbook_audit,
    )

    strict_outcomes.to_csv(paths["Strict emitted trades CSV"], index=False)
    research_outcomes.to_csv(paths["Research outcomes CSV"], index=False)
    research_summary.to_csv(paths["Research summary CSV"], index=False)
    horizon_summary.to_csv(paths["Horizon summary CSV"], index=False)
    strategy_family_audit.to_csv(paths["Strategy family audit CSV"], index=False)
    ticker_playbook_audit.to_csv(paths["Ticker playbook audit CSV"], index=False)
    rolling_ticker_playbook_audit.to_csv(paths["Rolling ticker playbook audit CSV"], index=False)

    metadata = {
        "root_dir": str(root),
        "out_dir": str(out_dir),
        "raw_cache_dir": str(raw_cache_dir),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "lookback": lookback,
        "horizons": horizons,
        "top": int(args.top),
        "samples": int(args.samples),
        "candidate_pool": int(args.candidate_pool),
        "strict_summary": summarize_outcomes(strict_outcomes),
        "research_summary": summarize_outcomes(research_outcomes),
        "verdict": _summary_verdict(summarize_outcomes(strict_outcomes)),
        "schwab_live_validation": "skipped_for_historical_proof",
    }
    paths["Metadata JSON"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    report = build_report(
        start=start,
        end=end,
        lookback=lookback,
        horizons=horizons,
        strict_outcomes=strict_outcomes,
        research_outcomes=research_outcomes,
        research_summary=research_summary,
        horizon_summary=horizon_summary,
        strategy_family_audit=strategy_family_audit,
        ticker_playbook_audit=ticker_playbook_audit,
        rolling_ticker_playbook_audit=rolling_ticker_playbook_audit,
        output_paths=paths,
    )
    paths["Report"].write_text(report, encoding="utf-8")

    for path in paths.values():
        print(f"Wrote: {path}", flush=True)
    return paths


def main(argv: Optional[Sequence[str]] = None) -> int:
    run(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
