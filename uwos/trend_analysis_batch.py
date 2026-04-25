#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from collections import Counter
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
    parser.add_argument(
        "--schwab-report-json",
        default="",
        help=(
            "Optional Schwab account JSON report. When present, batch proof uses only trades "
            "closed on or before each historical signal date as broker-history evidence."
        ),
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


def _rules_fingerprint(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    profile = {
        "config_path": str(args.config),
        "lookback": int(args.lookback),
        "horizons": _parse_horizons(args.horizons),
        "top": int(args.top),
        "samples": int(args.samples),
        "candidate_pool": int(args.candidate_pool),
        "min_backtest_edge": float(args.min_backtest_edge),
        "min_backtest_signals": int(args.min_backtest_signals),
        "min_workup_signals": int(args.min_workup_signals),
        "min_swing_score": float(args.min_swing_score),
        "allow_low_sample": bool(args.allow_low_sample),
        "allow_earnings_risk": bool(args.allow_earnings_risk),
        "allow_volatile_ic": bool(args.allow_volatile_ic),
        "allow_flow_conflict": bool(args.allow_flow_conflict),
        "max_bid_ask_to_price_pct": float(args.max_bid_ask_to_price_pct),
        "max_bid_ask_to_width_pct": float(args.max_bid_ask_to_width_pct),
        "max_short_delta": float(args.max_short_delta),
        "min_underlying_price": float(args.min_underlying_price),
        "min_debit_spread_price": float(args.min_debit_spread_price),
        "min_whale_appearances": int(args.min_whale_appearances),
    }
    payload = {"profile": profile, "config": cfg}
    raw = json.dumps(payload, sort_keys=True, default=str)
    return {
        "sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "profile": profile,
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


def summarize_group_expectancy(
    df: pd.DataFrame,
    by: str,
    *,
    bucket_name: str = "bucket",
    limit: int = 16,
) -> pd.DataFrame:
    columns = [
        bucket_name,
        "outcomes",
        "dates",
        "hit_rate",
        "avg_pnl",
        "total_pnl",
        "avg_return_on_risk",
        "profit_factor",
        "verdict",
    ]
    if df.empty or by not in df.columns:
        return pd.DataFrame(columns=columns)
    rows: List[Dict[str, Any]] = []
    for value, group in df.groupby(by, dropna=False):
        summary = summarize_outcomes(group)
        rows.append(
            {
                bucket_name: str(value),
                "outcomes": int(summary.get("trades", 0) or 0),
                "dates": int(summary.get("signal_dates", 0) or 0),
                "hit_rate": summary.get("win_rate"),
                "avg_pnl": summary.get("avg_pnl"),
                "total_pnl": summary.get("total_pnl"),
                "avg_return_on_risk": summary.get("avg_return_on_risk"),
                "profit_factor": summary.get("profit_factor"),
                "verdict": _summary_verdict(summary),
            }
        )
    out = pd.DataFrame(rows, columns=columns)
    out["_sort_total"] = pd.to_numeric(out["total_pnl"], errors="coerce").fillna(-math.inf)
    out = out.sort_values(["_sort_total", bucket_name], ascending=[False, True], kind="mergesort")
    return out.drop(columns=["_sort_total"]).head(limit).reset_index(drop=True)


def _rolling_playbook_table(df: pd.DataFrame, *, limit: int = 12) -> str:
    if df.empty:
        return "_none_"
    return _render_table(
        ["ticker", "direction", "strategy", "tests", "dates", "hit", "avg", "PF", "recent avg", "verdict"],
        [
            [
                str(row.get("ticker", "")),
                str(row.get("direction", "")),
                str(row.get("strategy", "")),
                str(trend_analysis._safe_int(row.get("forward_tests"))),
                str(trend_analysis._safe_int(row.get("forward_dates"))),
                _fmt_pct(row.get("forward_hit_rate")),
                _fmt_money(row.get("forward_avg_pnl")),
                _fmt_num(row.get("forward_profit_factor"), 2),
                _fmt_money(row.get("recent_forward_avg_pnl")),
                str(row.get("verdict", "")),
            ]
            for _, row in df.head(limit).iterrows()
        ],
    )


def _rolling_family_table(df: pd.DataFrame, *, limit: int = 12) -> str:
    if df.empty:
        return "_none_"
    return _render_table(
        ["family", "tests", "dates", "hit", "avg", "PF", "recent avg", "verdict"],
        [
            [
                str(row.get("family", "")),
                str(trend_analysis._safe_int(row.get("forward_tests"))),
                str(trend_analysis._safe_int(row.get("forward_dates"))),
                _fmt_pct(row.get("forward_hit_rate")),
                _fmt_money(row.get("forward_avg_pnl")),
                _fmt_num(row.get("forward_profit_factor"), 2),
                _fmt_money(row.get("recent_forward_avg_pnl")),
                str(row.get("verdict", "")),
            ]
            for _, row in df.head(limit).iterrows()
        ],
    )


def _playbook_example_lines(
    forward_playbook_outcomes: pd.DataFrame,
    playbooks: pd.DataFrame,
    *,
    limit_per_playbook: int = 4,
) -> List[str]:
    if forward_playbook_outcomes.empty or playbooks.empty:
        return ["_none_"]
    required = {"ticker", "direction", "strategy", "horizon_market_days", "signal_date", "trade_setup"}
    if not required.issubset(set(forward_playbook_outcomes.columns)):
        return ["_none_"]

    lines: List[str] = []
    outcomes = forward_playbook_outcomes.copy()
    outcomes["_horizon"] = pd.to_numeric(outcomes["horizon_market_days"], errors="coerce").fillna(0).astype(int)
    for _, playbook in playbooks.iterrows():
        ticker = str(playbook.get("ticker", "") or "").strip().upper()
        direction = str(playbook.get("direction", "") or "").strip().lower()
        strategy = str(playbook.get("strategy", "") or "").strip()
        horizon = trend_analysis._safe_int(playbook.get("horizon_market_days"))
        subset = outcomes[
            outcomes["ticker"].fillna("").astype(str).str.upper().eq(ticker)
            & outcomes["direction"].fillna("").astype(str).str.lower().eq(direction)
            & outcomes["strategy"].fillna("").astype(str).eq(strategy)
            & outcomes["_horizon"].eq(horizon)
        ].copy()
        if subset.empty:
            continue
        subset = subset.sort_values("signal_date", ascending=False, kind="mergesort").head(limit_per_playbook)
        lines.append(f"**{ticker} {direction} {strategy}, {horizon}d historical hold examples:**")
        for _, row in subset.iterrows():
            setup = trend_analysis._clip_text(row.get("trade_setup", ""), 140)
            parts = [
                f"{row.get('signal_date', '')}",
                setup,
                f"entry {_fmt_money(row.get('entry_net'))}",
                f"exit {_fmt_money(row.get('exit_net'))}",
                f"P&L {_fmt_money(row.get('pnl'))}",
            ]
            lines.append(f"- {'; '.join(parts)}")
    return lines or ["_none_"]


def _family_example_lines(
    research_outcomes: pd.DataFrame,
    families: pd.DataFrame,
    *,
    limit_per_family: int = 4,
) -> List[str]:
    if research_outcomes.empty or families.empty:
        return ["_none_"]
    required = {"horizon_market_days", "signal_date", "trade_setup"}
    if not required.issubset(set(research_outcomes.columns)):
        return ["_none_"]

    outcomes = research_outcomes.copy()
    if "policy" in outcomes.columns:
        outcomes = outcomes[outcomes["policy"].fillna("").astype(str).eq("entry_available_score_gate")].copy()
    if outcomes.empty:
        return ["_none_"]
    outcomes["_families"] = outcomes.apply(trend_analysis._strategy_family_labels, axis=1)
    outcomes = outcomes[outcomes["_families"].map(bool)].copy()
    if outcomes.empty:
        return ["_none_"]
    exploded = outcomes.explode("_families").rename(columns={"_families": "family"})
    exploded["_horizon"] = pd.to_numeric(exploded["horizon_market_days"], errors="coerce").fillna(0).astype(int)

    lines: List[str] = []
    for _, family_row in families.iterrows():
        family = str(family_row.get("family", "") or "").strip()
        horizon = trend_analysis._safe_int(family_row.get("horizon_market_days"))
        subset = exploded[
            exploded["family"].fillna("").astype(str).eq(family)
            & exploded["_horizon"].eq(horizon)
        ].copy()
        if subset.empty:
            continue
        subset = subset.sort_values("signal_date", ascending=False, kind="mergesort").head(limit_per_family)
        lines.append(f"**{family}, {horizon}d historical hold examples:**")
        for _, row in subset.iterrows():
            setup = trend_analysis._clip_text(row.get("trade_setup", ""), 140)
            ticker = str(row.get("ticker", "") or "").strip().upper()
            parts = [
                f"{row.get('signal_date', '')}",
                ticker,
                setup,
                f"entry {_fmt_money(row.get('entry_net'))}",
                f"exit {_fmt_money(row.get('exit_net'))}",
                f"P&L {_fmt_money(row.get('pnl'))}",
            ]
            lines.append(f"- {'; '.join(p for p in parts if p)}")
        lines.append("")
    return lines or ["_none_"]


def _split_reasons(value: Any) -> List[str]:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return []
    return [part.strip() for part in text.split(";") if part.strip() and part.strip().lower() != "nan"]


def build_gap_diagnostics(research_outcomes: pd.DataFrame) -> pd.DataFrame:
    columns = ["area", "gap", "rows", "avg_pnl", "total_pnl", "fix"]
    if research_outcomes.empty:
        return pd.DataFrame(columns=columns)

    rows: List[Dict[str, Any]] = []
    overall = summarize_outcomes(research_outcomes)
    if trend_analysis._safe_float(overall.get("avg_pnl")) < 0:
        rows.append(
            {
                "area": "broad_policy",
                "gap": "Broad entry_available_score_gate pool has negative expectancy",
                "rows": int(overall.get("trades", 0) or 0),
                "avg_pnl": overall.get("avg_pnl"),
                "total_pnl": overall.get("total_pnl"),
                "fix": "Do not use broad pattern candidates as trades; require a prior-only ticker playbook or supportive strategy family.",
            }
        )

    for policy, group in research_outcomes.groupby("policy", dropna=False):
        summary = summarize_outcomes(group)
        if trend_analysis._safe_float(summary.get("avg_pnl")) < 0:
            rows.append(
                {
                    "area": "policy",
                    "gap": f"{policy} is negative",
                    "rows": int(summary.get("trades", 0) or 0),
                    "avg_pnl": summary.get("avg_pnl"),
                    "total_pnl": summary.get("total_pnl"),
                    "fix": "Keep this policy blocked until train/validation and rolling-forward evidence turn positive.",
                }
            )

    for column, area in (
        ("base_gate_reasons", "base_gate"),
        ("quality_reject_reasons", "quality_gate"),
    ):
        if column not in research_outcomes.columns:
            continue
        stats: Dict[str, List[float]] = {}
        counts: Counter[str] = Counter()
        for _, row in research_outcomes.iterrows():
            pnl = trend_analysis._safe_float(row.get("pnl"))
            for reason in _split_reasons(row.get(column)):
                counts[reason] += 1
                stats.setdefault(reason, []).append(pnl if math.isfinite(pnl) else 0.0)
        for reason, count in counts.most_common(8):
            values = stats.get(reason, [])
            total = float(sum(values)) if values else 0.0
            avg = total / len(values) if values else math.nan
            fix = "Treat as blocker; only relax after a separate positive A/B proof."
            lower = reason.lower()
            if "backtest unknown" in lower:
                fix = "Route only through prior-only ticker playbooks; do not let UNKNOWN broad setups through."
            elif "expensive debit" in lower:
                fix = "Repair strikes so debit is <= 50% of width or skip the setup."
            elif "flow conflict" in lower:
                fix = "Block unless the ticker playbook has prior-only forward support and live flow flips back."
            elif "thin institutional" in lower:
                fix = "Keep as watchlist unless ticker-specific playbook support is strong."
            elif "price not confirming" in lower or "weak directional price" in lower:
                fix = "Use as trigger condition, not entry; require price to confirm before order-ready status."
            rows.append(
                {
                    "area": area,
                    "gap": reason,
                    "rows": int(count),
                    "avg_pnl": avg,
                    "total_pnl": total,
                    "fix": fix,
                }
            )

    if not rows:
        return pd.DataFrame(columns=columns)
    out = pd.DataFrame(rows, columns=columns)
    out["_abs_total"] = pd.to_numeric(out["total_pnl"], errors="coerce").abs().fillna(0)
    return out.sort_values(["area", "_abs_total"], ascending=[True, False]).drop(columns=["_abs_total"])


def summarize_failure_buckets(research_outcomes: pd.DataFrame, *, limit: int = 18) -> pd.DataFrame:
    columns = [
        "area",
        "bucket",
        "outcomes",
        "dates",
        "hit_rate",
        "avg_pnl",
        "total_pnl",
        "avg_return_on_risk",
        "profit_factor",
        "verdict",
    ]
    if research_outcomes.empty:
        return pd.DataFrame(columns=columns)

    rows: List[Dict[str, Any]] = []
    for column, area in (
        ("base_gate_reasons", "base_gate"),
        ("quality_reject_reasons", "quality_gate"),
        ("actionability_reject_reasons", "actionability"),
    ):
        if column not in research_outcomes.columns:
            continue
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for _, row in research_outcomes.iterrows():
            reasons = _split_reasons(row.get(column))
            if not reasons:
                continue
            for reason in reasons:
                grouped.setdefault(reason, []).append(dict(row))
        for reason, reason_rows in grouped.items():
            group = pd.DataFrame(reason_rows)
            summary = summarize_outcomes(group)
            rows.append(
                {
                    "area": area,
                    "bucket": reason,
                    "outcomes": int(summary.get("trades", 0) or 0),
                    "dates": int(summary.get("signal_dates", 0) or 0),
                    "hit_rate": summary.get("win_rate"),
                    "avg_pnl": summary.get("avg_pnl"),
                    "total_pnl": summary.get("total_pnl"),
                    "avg_return_on_risk": summary.get("avg_return_on_risk"),
                    "profit_factor": summary.get("profit_factor"),
                    "verdict": _summary_verdict(summary),
                }
            )
    if not rows:
        return pd.DataFrame(columns=columns)
    out = pd.DataFrame(rows, columns=columns)
    out["_abs_total"] = pd.to_numeric(out["total_pnl"], errors="coerce").abs().fillna(0)
    return (
        out.sort_values(["_abs_total", "area", "bucket"], ascending=[False, True, True], kind="mergesort")
        .drop(columns=["_abs_total"])
        .head(limit)
        .reset_index(drop=True)
    )


def build_report(
    *,
    start: dt.date,
    end: dt.date,
    lookback: int,
    horizons: Sequence[int],
    strict_outcomes: pd.DataFrame,
    forward_playbook_outcomes: pd.DataFrame,
    research_outcomes: pd.DataFrame,
    research_summary: pd.DataFrame,
    horizon_summary: pd.DataFrame,
    regime_summary: pd.DataFrame,
    iv_summary: pd.DataFrame,
    failure_bucket_summary: pd.DataFrame,
    strategy_family_audit: pd.DataFrame,
    rolling_strategy_family_audit: pd.DataFrame,
    ticker_playbook_audit: pd.DataFrame,
    rolling_ticker_playbook_audit: pd.DataFrame,
    gap_diagnostics: pd.DataFrame,
    rules_fingerprint: Dict[str, Any],
    schwab_actual_summary: Dict[str, Any],
    output_paths: Dict[str, Path],
) -> str:
    strict_summary = summarize_outcomes(strict_outcomes)
    forward_playbook_summary = summarize_outcomes(forward_playbook_outcomes)
    research_summary_all = summarize_outcomes(research_outcomes)
    verdict = _summary_verdict(strict_summary)
    playbook_verdict = _summary_verdict(forward_playbook_summary)

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
            "The strict order-ready audit is positive enough to consider defined-risk promotion, subject to live/paper confirmation and sizing caps."
        )
    elif int(forward_playbook_summary.get("trades", 0) or 0) > 0:
        lines.append(
            "The strict order-ready lane is still too sparse, but this is not a no-trade engine: the prior-only ticker-playbook lane found completed historical trades after the playbook was already promotable from earlier dates. Treat that lane as the live trade-generation source, with starter sizing until the sample is larger."
        )
    elif verdict == "NO_PROOF_LOW_SAMPLE":
        lines.append(
            "Not proven yet. The strict order-ready sample is too small, so this is still an idea engine until more historical or live outcomes accumulate."
        )
    else:
        lines.append(
            "Not a money printer yet. The strict order-ready audit does not have positive enough expectancy to promote sizing."
        )
    lines.append("")
    lines.append(
        _render_table(
            ["scope", "trades", "dates", "win", "avg", "total", "PF", "drawdown", "verdict"],
            [
                _summary_table_row("strict order-ready trades", strict_summary),
                _summary_table_row("prior-only playbook trade-generation lane", forward_playbook_summary),
                _summary_table_row("all research outcomes", research_summary_all),
            ],
        )
    )
    if int(forward_playbook_summary.get("trades", 0) or 0) > 0:
        lines.append("")
        lines.append(
            "- Playbook lane verdict: "
            + f"`{playbook_verdict}`; "
            + f"{int(forward_playbook_summary.get('trades', 0) or 0)} completed trade(s), "
            + f"{_fmt_pct(forward_playbook_summary.get('win_rate'))} win rate, "
            + f"avg {_fmt_money(forward_playbook_summary.get('avg_pnl'))}, "
            + f"PF {_fmt_num(forward_playbook_summary.get('profit_factor'), 2)}."
        )
    lines.append("")
    lines.append("## Frozen Rules")
    lines.append(
        "This proof should be read only against the exact rule profile and config hash below. If the rules change, re-run the proof and do not carry forward the old confidence."
    )
    lines.append("")
    lines.append(
        f"- Rule fingerprint: `{str(rules_fingerprint.get('sha256') or '')[:16]}`"
    )
    lines.append(f"- Config path: `{rules_fingerprint.get('profile', {}).get('config_path', '')}`")
    lines.append(f"- Min backtest signals: {int(rules_fingerprint.get('profile', {}).get('min_backtest_signals', 0) or 0)}")
    lines.append(f"- Min swing score: {float(rules_fingerprint.get('profile', {}).get('min_swing_score', 0.0) or 0.0):.1f}")
    if schwab_actual_summary.get("status") == "ok":
        lines.append(
            "- Schwab broker-history support: enabled with "
            + f"{int(schwab_actual_summary.get('parsed_closed_trades', 0) or 0)} parsed closed trades; "
            + "each historical signal date only sees trades closed on or before that date."
        )
    elif schwab_actual_summary.get("enabled"):
        lines.append(
            "- Schwab broker-history support: "
            + f"{schwab_actual_summary.get('status')}"
            + (f" ({schwab_actual_summary.get('issues')})" if schwab_actual_summary.get("issues") else "")
        )
    lines.append("")
    lines.append("## Actionable Repair Map")
    lines.append("- Broad trend candidates are not a trade source until the broad policy turns positive.")
    lines.append("- Use only forward-supported lanes for live candidates: prior-only ticker playbooks and rolling strategy families.")
    lines.append("- Same-day ticker/expiry variants are deduped before playbook validation, so one ticker/day cannot inflate confidence.")
    lines.append("- Rolling playbooks with recent negative decay are blocked even if older forward tests were profitable.")
    lines.append("- Live action requires an exact current option ticket from the single-date report: setup legs, expiry, clean quotes, no open-position conflict, no earnings/liquidity hard block, and starter sizing unless ticker-specific rolling support is mature.")
    lines.append("")

    eligible_playbooks = rolling_ticker_playbook_audit[
        rolling_ticker_playbook_audit.get("verdict", pd.Series(dtype=str)).fillna("").astype(str).isin(
            ["supportive", "emerging_forward"]
        )
    ].copy()
    blocked_playbooks = rolling_ticker_playbook_audit[
        rolling_ticker_playbook_audit.get("verdict", pd.Series(dtype=str)).fillna("").astype(str).isin(
            ["negative", "decaying"]
        )
    ].copy()
    lines.append("## Live-Eligible Playbooks")
    if eligible_playbooks.empty:
        lines.append("No rolling playbook is live-eligible yet. Keep current trend-analysis outputs as research/watchlist only.")
    else:
        lines.append(
            "These playbooks are allowed to feed a current trade search. They are not order tickets by themselves; a trade exists only when the current single-date report gives exact legs, expiry, price, quote validation, Schwab validation, and no open-position conflict."
        )
        lines.append("")
        lines.append(_rolling_playbook_table(eligible_playbooks))
        lines.append("")
        lines.append("Latest prior-only setup examples. These are historical structures that made the playbook eligible, not current order tickets:")
        lines.extend(_playbook_example_lines(forward_playbook_outcomes, eligible_playbooks))
    if not blocked_playbooks.empty:
        lines.append("")
        lines.append("Blocked rolling playbooks:")
        lines.append("")
        lines.append(_rolling_playbook_table(blocked_playbooks))
        lines.append("")
        lines.append("Blocked prior-only setup examples:")
        lines.extend(_playbook_example_lines(forward_playbook_outcomes, blocked_playbooks, limit_per_playbook=3))
    lines.append("")

    eligible_families = rolling_strategy_family_audit[
        rolling_strategy_family_audit.get("verdict", pd.Series(dtype=str)).fillna("").astype(str).isin(
            ["supportive", "emerging_forward"]
        )
    ].copy()
    blocked_families = rolling_strategy_family_audit[
        rolling_strategy_family_audit.get("verdict", pd.Series(dtype=str)).fillna("").astype(str).isin(
            ["negative", "decaying"]
        )
    ].copy()
    lines.append("## Live-Eligible Strategy Families")
    if eligible_families.empty:
        lines.append("No rolling strategy family is live-eligible yet. Keep family-level evidence diagnostic only.")
    else:
        lines.append(
            "These setup families are forward-supported using only earlier dates. They can feed starter-risk live tickets when the single-date report also passes tradeability and portfolio checks."
        )
        lines.append("")
        lines.append(_rolling_family_table(eligible_families))
        lines.append("")
        lines.append("Latest family examples. These are historical structures that made the family eligible, not current order tickets:")
        lines.extend(_family_example_lines(research_outcomes, eligible_families))
    if not blocked_families.empty:
        lines.append("")
        lines.append("Blocked rolling strategy families:")
        lines.append("")
        lines.append(_rolling_family_table(blocked_families))
    lines.append("")

    lines.append("## Strict Order-Ready Trades")
    if strict_outcomes.empty:
        lines.append(
            "No historical trade cleared every order-ready gate with completed future option quotes in this window. This lane is the safest possible interpretation, but by itself it is too sparse to prove profitability or to represent the whole trade-generation engine."
        )
    else:
        lines.append(_group_summary_table(strict_outcomes, "horizon_market_days"))
        lines.append("")
        lines.append(_group_summary_table(strict_outcomes, "ticker"))
    lines.append("")

    lines.append("## Prior-Only Playbook Lane")
    if forward_playbook_outcomes.empty:
        lines.append(
            "No ticker playbook produced completed forward trades after becoming promotable using prior data only. Do not use full-period playbooks as live entries yet."
        )
    else:
        lines.append(
            "These are the trades the ticker-playbook lane would have taken only after the playbook was already promotable from earlier dates. The live-eligible section above decides which of those playbooks still survives after rolling validation."
        )
        lines.append("")
        lines.append(_group_summary_table(forward_playbook_outcomes, "ticker"))
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

    lines.append("## Regime Expectancy")
    if regime_summary.empty:
        lines.append("_no completed regime buckets_")
    else:
        lines.append(
            _render_table(
                ["regime", "outcomes", "dates", "hit", "avg", "total", "avg R/R", "PF", "verdict"],
                [
                    [
                        str(row.get("market_regime", "")),
                        str(int(row.get("outcomes", 0) or 0)),
                        str(int(row.get("dates", 0) or 0)),
                        _fmt_pct(row.get("hit_rate")),
                        _fmt_money(row.get("avg_pnl")),
                        _fmt_money(row.get("total_pnl")),
                        _fmt_pct(row.get("avg_return_on_risk")),
                        _fmt_num(row.get("profit_factor"), 2),
                        str(row.get("verdict", "")),
                    ]
                    for _, row in regime_summary.iterrows()
                ],
            )
        )
    lines.append("")

    lines.append("## IV Bucket Expectancy")
    if iv_summary.empty:
        lines.append("_no completed IV buckets_")
    else:
        lines.append(
            _render_table(
                ["iv bucket", "outcomes", "dates", "hit", "avg", "total", "avg R/R", "PF", "verdict"],
                [
                    [
                        str(row.get("iv_rank_bucket", "")),
                        str(int(row.get("outcomes", 0) or 0)),
                        str(int(row.get("dates", 0) or 0)),
                        _fmt_pct(row.get("hit_rate")),
                        _fmt_money(row.get("avg_pnl")),
                        _fmt_money(row.get("total_pnl")),
                        _fmt_pct(row.get("avg_return_on_risk")),
                        _fmt_num(row.get("profit_factor"), 2),
                        str(row.get("verdict", "")),
                    ]
                    for _, row in iv_summary.iterrows()
                ],
            )
        )
    lines.append("")

    lines.append("## Failure Bucket Expectancy")
    if failure_bucket_summary.empty:
        lines.append("_no completed failure buckets_")
    else:
        lines.append(
            _render_table(
                ["area", "bucket", "outcomes", "dates", "hit", "avg", "total", "PF", "verdict"],
                [
                    [
                        str(row.get("area", "")),
                        trend_analysis._clip_text(row.get("bucket", ""), 72),
                        str(int(row.get("outcomes", 0) or 0)),
                        str(int(row.get("dates", 0) or 0)),
                        _fmt_pct(row.get("hit_rate")),
                        _fmt_money(row.get("avg_pnl")),
                        _fmt_money(row.get("total_pnl")),
                        _fmt_num(row.get("profit_factor"), 2),
                        str(row.get("verdict", "")),
                    ]
                    for _, row in failure_bucket_summary.iterrows()
                ],
            )
        )
    lines.append("")

    lines.append("## Gap Diagnostics")
    if gap_diagnostics.empty:
        lines.append("_no diagnostic gaps found_")
    else:
        lines.append(
            _render_table(
                ["area", "gap", "rows", "avg", "total", "fix"],
                [
                    [
                        str(row.get("area", "")),
                        trend_analysis._clip_text(row.get("gap", ""), 80),
                        str(int(row.get("rows", 0) or 0)),
                        _fmt_money(row.get("avg_pnl")),
                        _fmt_money(row.get("total_pnl")),
                        trend_analysis._clip_text(row.get("fix", ""), 100),
                    ]
                    for _, row in gap_diagnostics.head(18).iterrows()
                ],
            )
        )
    lines.append("")

    lines.append("## Playbook Validation")
    lines.append("### Strategy Families")
    lines.extend(trend_analysis._strategy_family_report_lines(strategy_family_audit))
    lines.append("")
    lines.append("### Rolling Strategy Families")
    lines.extend(trend_analysis._rolling_ticker_playbook_report_lines(rolling_strategy_family_audit, name_col="family"))
    lines.append("")
    lines.append("### Ticker Playbooks")
    lines.extend(trend_analysis._ticker_playbook_report_lines(ticker_playbook_audit))
    lines.append("")
    lines.append("### Rolling Ticker Playbooks")
    lines.extend(trend_analysis._rolling_ticker_playbook_report_lines(rolling_ticker_playbook_audit))
    lines.append("")

    lines.append("## Go / No-Go")
    lines.append("- Do not call this a money printer until either strict order-ready trades or prior-only playbook trades have enough completed outcomes.")
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
        "Prior-only playbook trades CSV": out_dir / f"trend-analysis-batch-prior-only-playbook-trades-{suffix}.csv",
        "Research outcomes CSV": out_dir / f"trend-analysis-batch-research-outcomes-{suffix}.csv",
        "Research summary CSV": out_dir / f"trend-analysis-batch-research-summary-{suffix}.csv",
        "Horizon summary CSV": out_dir / f"trend-analysis-batch-horizon-summary-{suffix}.csv",
        "Regime summary CSV": out_dir / f"trend-analysis-batch-regime-summary-{suffix}.csv",
        "IV summary CSV": out_dir / f"trend-analysis-batch-iv-summary-{suffix}.csv",
        "Failure bucket summary CSV": out_dir / f"trend-analysis-batch-failure-bucket-summary-{suffix}.csv",
        "Strategy family audit CSV": out_dir / f"trend-analysis-batch-strategy-family-audit-{suffix}.csv",
        "Rolling strategy family audit CSV": out_dir / f"trend-analysis-batch-rolling-strategy-family-audit-{suffix}.csv",
        "Ticker playbook audit CSV": out_dir / f"trend-analysis-batch-ticker-playbook-audit-{suffix}.csv",
        "Rolling ticker playbook audit CSV": out_dir / f"trend-analysis-batch-rolling-ticker-playbook-audit-{suffix}.csv",
        "Gap diagnostics CSV": out_dir / f"trend-analysis-batch-gap-diagnostics-{suffix}.csv",
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
    rules_fingerprint = _rules_fingerprint(cfg, args)
    schwab_report_json = (
        Path(args.schwab_report_json).expanduser().resolve()
        if str(args.schwab_report_json or "").strip()
        else trend_analysis._latest_schwab_report_json(root)
    )
    schwab_actual_trades, schwab_actual_summary = trend_analysis.load_schwab_closed_trade_history(schwab_report_json)

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
        schwab_actual_trades=schwab_actual_trades,
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
        schwab_actual_trades=schwab_actual_trades,
    )
    research_outcomes = _filter_signal_dates(research_outcomes, start, end)
    research_summary = trend_analysis._research_summary_from_outcomes(research_outcomes)
    horizon_summary = trend_analysis._research_summary_by_horizon_from_outcomes(research_outcomes)
    regime_summary = summarize_group_expectancy(
        research_outcomes,
        "market_regime",
        bucket_name="market_regime",
    )
    iv_summary = summarize_group_expectancy(
        research_outcomes,
        "iv_rank_bucket",
        bucket_name="iv_rank_bucket",
    )
    failure_bucket_summary = summarize_failure_buckets(research_outcomes)
    strategy_family_audit = trend_analysis._strategy_family_audit_from_outcomes(research_outcomes)
    rolling_strategy_family_audit = trend_analysis._rolling_strategy_family_audit_from_outcomes(
        research_outcomes,
        strategy_family_audit,
    )
    ticker_playbook_audit = trend_analysis._ticker_playbook_audit_from_outcomes(research_outcomes)
    rolling_ticker_playbook_audit = trend_analysis._rolling_ticker_playbook_audit_from_outcomes(
        research_outcomes,
        ticker_playbook_audit,
    )
    forward_playbook_outcomes = trend_analysis.rolling_ticker_playbook_forward_outcomes(
        research_outcomes,
        ticker_playbook_audit,
    )
    gap_diagnostics = build_gap_diagnostics(research_outcomes)

    strict_outcomes.to_csv(paths["Strict emitted trades CSV"], index=False)
    forward_playbook_outcomes.to_csv(paths["Prior-only playbook trades CSV"], index=False)
    research_outcomes.to_csv(paths["Research outcomes CSV"], index=False)
    research_summary.to_csv(paths["Research summary CSV"], index=False)
    horizon_summary.to_csv(paths["Horizon summary CSV"], index=False)
    regime_summary.to_csv(paths["Regime summary CSV"], index=False)
    iv_summary.to_csv(paths["IV summary CSV"], index=False)
    failure_bucket_summary.to_csv(paths["Failure bucket summary CSV"], index=False)
    strategy_family_audit.to_csv(paths["Strategy family audit CSV"], index=False)
    rolling_strategy_family_audit.to_csv(paths["Rolling strategy family audit CSV"], index=False)
    ticker_playbook_audit.to_csv(paths["Ticker playbook audit CSV"], index=False)
    rolling_ticker_playbook_audit.to_csv(paths["Rolling ticker playbook audit CSV"], index=False)
    gap_diagnostics.to_csv(paths["Gap diagnostics CSV"], index=False)

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
        "rules_fingerprint": rules_fingerprint,
        "strict_summary": summarize_outcomes(strict_outcomes),
        "prior_only_playbook_summary": summarize_outcomes(forward_playbook_outcomes),
        "research_summary": summarize_outcomes(research_outcomes),
        "regime_summary_rows": int(len(regime_summary)),
        "iv_summary_rows": int(len(iv_summary)),
        "failure_bucket_summary_rows": int(len(failure_bucket_summary)),
        "verdict": _summary_verdict(summarize_outcomes(strict_outcomes)),
        "prior_only_playbook_verdict": _summary_verdict(summarize_outcomes(forward_playbook_outcomes)),
        "schwab_live_validation": "skipped_for_historical_proof",
        "schwab_actual_summary": schwab_actual_summary,
    }
    paths["Metadata JSON"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    report = build_report(
        start=start,
        end=end,
        lookback=lookback,
        horizons=horizons,
        strict_outcomes=strict_outcomes,
        forward_playbook_outcomes=forward_playbook_outcomes,
        research_outcomes=research_outcomes,
        research_summary=research_summary,
        horizon_summary=horizon_summary,
        regime_summary=regime_summary,
        iv_summary=iv_summary,
        failure_bucket_summary=failure_bucket_summary,
        strategy_family_audit=strategy_family_audit,
        rolling_strategy_family_audit=rolling_strategy_family_audit,
        ticker_playbook_audit=ticker_playbook_audit,
        rolling_ticker_playbook_audit=rolling_ticker_playbook_audit,
        gap_diagnostics=gap_diagnostics,
        rules_fingerprint=rules_fingerprint,
        schwab_actual_summary=schwab_actual_summary,
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
