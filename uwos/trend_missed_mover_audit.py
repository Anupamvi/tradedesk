#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from uwos import swing_trend_pipeline as swing


DEFAULT_ROOT = Path("/Users/anuppamvi/uw_root/tradedesk")
DEFAULT_LOOKBACK = 30
DEFAULT_HORIZONS = "1,3,5"
DEFAULT_MOVE_THRESHOLD = 0.08
DEFAULT_MIN_PRICE = 20.0


OUTPUT_KINDS = (
    ("actionable", "trend-analysis-actionable-{date}-L{lookback}.csv"),
    ("current_setup", "trend-analysis-current-setups-{date}-L{lookback}.csv"),
    ("event_watch", "trend-analysis-event-watch-{date}-L{lookback}.csv"),
    ("trade_workup", "trend-analysis-trade-workups-{date}-L{lookback}.csv"),
    ("candidate_shortlist", "trend-analysis-candidates-{date}-L{lookback}.csv"),
    ("proven_ticket", "trend-analysis-proven-tickets-{date}-L{lookback}.csv"),
    ("pattern", "trend-analysis-patterns-{date}-L{lookback}.csv"),
    ("raw", "trend_analysis_raw_{date}-L{lookback}.csv"),
)

FINAL_VISIBLE_KINDS = {
    "actionable",
    "current_setup",
    "event_watch",
    "trade_workup",
    "candidate_shortlist",
    "proven_ticket",
    "pattern",
}


def _parse_date(value: str) -> dt.date:
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected YYYY-MM-DD, got {value!r}") from exc


def _parse_horizons(value: str) -> List[int]:
    horizons: List[int] = []
    for raw in str(value or "").split(","):
        raw = raw.strip()
        if not raw:
            continue
        parsed = int(raw)
        if parsed <= 0:
            raise argparse.ArgumentTypeError("Horizons must be positive market-day counts")
        horizons.append(parsed)
    if not horizons:
        raise argparse.ArgumentTypeError("At least one horizon is required")
    return sorted(set(horizons))


def _safe_float(value: Any) -> float:
    try:
        if value is None:
            return math.nan
        if isinstance(value, str):
            value = value.strip().replace(",", "")
            if not value:
                return math.nan
        return float(value)
    except Exception:
        return math.nan


def _fmt_pct(value: Any, digits: int = 1) -> str:
    parsed = _safe_float(value)
    if not math.isfinite(parsed):
        return "-"
    return f"{parsed:.{digits}%}"


def _fmt_num(value: Any, digits: int = 1) -> str:
    parsed = _safe_float(value)
    if not math.isfinite(parsed):
        return "-"
    return f"{parsed:.{digits}f}"


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "nat"} else text


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.DataFrame()


def _ticker_column(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        if str(col).strip().lower() in {"ticker", "symbol", "underlying", "underlying_symbol"}:
            return col
    return None


def _close_column(df: pd.DataFrame) -> Optional[str]:
    preferred = ("close", "close.1", "stock_price", "underlying_price")
    normalized = {str(col).strip().lower(): col for col in df.columns}
    for key in preferred:
        if key in normalized:
            return normalized[key]
    return None


def _load_close_by_day(
    root: Path,
    all_days: List[Tuple[dt.date, Path]],
    *,
    common_stock_only: bool = True,
) -> Dict[dt.date, pd.Series]:
    closes: Dict[dt.date, pd.Series] = {}
    for trade_date, day_dir in all_days:
        path = swing.resolve_csv_for_day(day_dir, trade_date.isoformat(), "stock-screener")
        if path is None:
            continue
        df = swing.read_csv_from_path(path)
        if df.empty:
            continue
        ticker_col = _ticker_column(df)
        close_col = _close_column(df)
        if ticker_col is None or close_col is None:
            continue
        use_cols = [ticker_col, close_col]
        issue_col = None
        for col in df.columns:
            if str(col).strip().lower() == "issue_type":
                issue_col = col
                use_cols.append(col)
                break
        is_index_col = None
        for col in df.columns:
            if str(col).strip().lower() == "is_index":
                is_index_col = col
                use_cols.append(col)
                break
        work = df[use_cols].copy()
        if common_stock_only and issue_col is not None:
            work = work[work[issue_col].fillna("").astype(str).str.lower().str.contains("common stock")]
        if common_stock_only and is_index_col is not None:
            work = work[~work[is_index_col].fillna("").astype(str).str.lower().isin({"t", "true", "1", "yes"})]
        work["_ticker"] = work[ticker_col].fillna("").astype(str).str.upper().str.strip()
        work["_close"] = pd.to_numeric(work[close_col], errors="coerce")
        work = work[work["_ticker"].ne("") & work["_close"].gt(0)].copy()
        if work.empty:
            continue
        closes[trade_date] = work.drop_duplicates("_ticker").set_index("_ticker")["_close"]
    return closes


def _output_path(out_dir: Path, kind: str, trade_date: dt.date, lookback: int) -> Path:
    for output_kind, pattern in OUTPUT_KINDS:
        if output_kind == kind:
            return out_dir / pattern.format(date=trade_date.isoformat(), lookback=lookback)
    raise KeyError(kind)


def _load_output_frames(out_dir: Path, trade_date: dt.date, lookback: int) -> Dict[str, pd.DataFrame]:
    return {
        kind: _read_csv(_output_path(out_dir, kind, trade_date, lookback))
        for kind, _ in OUTPUT_KINDS
    }


def _ticker_in_frame(frame: pd.DataFrame, ticker: str) -> bool:
    if frame.empty:
        return False
    ticker_col = _ticker_column(frame)
    if ticker_col is None:
        return False
    return frame[ticker_col].fillna("").astype(str).str.upper().str.strip().eq(ticker).any()


def _row_for_ticker(frame: pd.DataFrame, ticker: str) -> Optional[pd.Series]:
    if frame.empty:
        return None
    ticker_col = _ticker_column(frame)
    if ticker_col is None:
        return None
    rows = frame[frame[ticker_col].fillna("").astype(str).str.upper().str.strip().eq(ticker)]
    if rows.empty:
        return None
    return rows.iloc[0]


def classify_coverage(frames: Dict[str, pd.DataFrame], ticker: str) -> Tuple[str, str, Optional[pd.Series]]:
    ticker = str(ticker or "").upper().strip()
    for kind in ("actionable", "current_setup", "event_watch", "trade_workup", "candidate_shortlist", "proven_ticket", "pattern"):
        row = _row_for_ticker(frames.get(kind, pd.DataFrame()), ticker)
        if row is not None:
            return kind, kind, row
    row = _row_for_ticker(frames.get("raw", pd.DataFrame()), ticker)
    if row is not None:
        return "raw_only", "raw", row
    if frames.get("raw", pd.DataFrame()).empty:
        return "no_scan_output", "", None
    return "missed", "", None


def _rank_for_ticker(frame: pd.DataFrame, ticker: str) -> int:
    if frame.empty:
        return 0
    ticker_col = _ticker_column(frame)
    if ticker_col is None:
        return 0
    tickers = frame[ticker_col].fillna("").astype(str).str.upper().str.strip().tolist()
    try:
        return tickers.index(ticker) + 1
    except ValueError:
        return 0


def build_missed_mover_rows(
    *,
    root: Path,
    out_dir: Path,
    start: dt.date,
    end: dt.date,
    lookback: int,
    horizons: Sequence[int],
    move_threshold: float,
    min_price: float = DEFAULT_MIN_PRICE,
    max_abs_return: float = 0.0,
    common_stock_only: bool = True,
    include_missing_scan_days: bool = False,
) -> pd.DataFrame:
    latest_days = swing.discover_trading_days(root, 10000, None)
    if not latest_days:
        return pd.DataFrame()
    all_dates = [d for d, _ in latest_days]
    closes = _load_close_by_day(root, latest_days, common_stock_only=bool(common_stock_only))
    rows: List[Dict[str, Any]] = []

    for idx, signal_date in enumerate(all_dates):
        if signal_date < start or signal_date > end:
            continue
        current_close = closes.get(signal_date)
        if current_close is None or current_close.empty:
            continue
        if float(min_price) > 0:
            current_close = current_close[current_close.ge(float(min_price))]
            if current_close.empty:
                continue
        frames = _load_output_frames(out_dir, signal_date, lookback)
        raw_frame = frames.get("raw", pd.DataFrame())
        if raw_frame.empty and not include_missing_scan_days:
            continue

        for horizon in horizons:
            if idx + int(horizon) >= len(all_dates):
                continue
            future_date = all_dates[idx + int(horizon)]
            future_close = closes.get(future_date)
            if future_close is None or future_close.empty:
                continue
            common = current_close.index.intersection(future_close.index)
            if common.empty:
                continue
            returns = (future_close.loc[common] / current_close.loc[common]) - 1.0
            if float(max_abs_return) > 0:
                returns = returns[returns.abs().le(float(max_abs_return))]
            movers = returns[returns.abs().ge(float(move_threshold))].sort_values(key=lambda s: s.abs(), ascending=False)
            for ticker, ret in movers.items():
                direction = "bullish" if ret > 0 else "bearish"
                coverage, source_kind, source_row = classify_coverage(frames, str(ticker))
                raw_row = _row_for_ticker(raw_frame, str(ticker))
                row_for_detail = source_row if source_row is not None else raw_row
                pipeline_direction = _clean_text(row_for_detail.get("direction")) if row_for_detail is not None else ""
                direction_match = bool(pipeline_direction.lower() == direction) if pipeline_direction else False
                raw_rank = _rank_for_ticker(raw_frame, str(ticker))
                rows.append(
                    {
                        "signal_date": signal_date.isoformat(),
                        "future_date": future_date.isoformat(),
                        "horizon_days": int(horizon),
                        "ticker": str(ticker),
                        "move_direction": direction,
                        "entry_close": float(current_close.loc[ticker]),
                        "future_close": float(future_close.loc[ticker]),
                        "return_pct": float(ret) * 100.0,
                        "coverage": coverage,
                        "source_kind": source_kind,
                        "raw_rank": raw_rank,
                        "pipeline_direction": pipeline_direction,
                        "direction_match": direction_match,
                        "strategy": _clean_text(row_for_detail.get("strategy")) if row_for_detail is not None else "",
                        "setup": _clean_text(row_for_detail.get("live_strike_setup")) or _clean_text(row_for_detail.get("strike_setup")) if row_for_detail is not None else "",
                        "swing_score": _safe_float(row_for_detail.get("swing_score")) if row_for_detail is not None else math.nan,
                        "backtest_verdict": _clean_text(row_for_detail.get("backtest_verdict")) if row_for_detail is not None else "",
                        "edge_pct": _safe_float(row_for_detail.get("edge_pct")) if row_for_detail is not None else math.nan,
                        "event_watch_score": _safe_float(row_for_detail.get("event_watch_score")) if row_for_detail is not None else math.nan,
                        "blockers": _clean_text(row_for_detail.get("actionability_reject_reasons")) if row_for_detail is not None else "",
                    }
                )
    return pd.DataFrame(rows)


def summarize(rows: pd.DataFrame) -> Dict[str, Any]:
    if rows.empty:
        return {
            "total_big_movers": 0,
            "raw_seen": 0,
            "raw_seen_rate": math.nan,
            "final_visible": 0,
            "final_visible_rate": math.nan,
            "direction_match": 0,
            "direction_match_rate": math.nan,
            "missed": 0,
            "raw_only": 0,
            "event_watch": 0,
            "actionable": 0,
        }
    coverage = rows["coverage"].fillna("").astype(str)
    raw_seen_mask = ~coverage.isin({"missed", "no_scan_output", ""})
    final_visible_mask = coverage.isin(FINAL_VISIBLE_KINDS)
    direction_match_mask = rows.get("direction_match", pd.Series(False, index=rows.index)).fillna(False).astype(bool)
    total = len(rows)
    return {
        "total_big_movers": int(total),
        "raw_seen": int(raw_seen_mask.sum()),
        "raw_seen_rate": float(raw_seen_mask.mean()),
        "final_visible": int(final_visible_mask.sum()),
        "final_visible_rate": float(final_visible_mask.mean()),
        "direction_match": int((raw_seen_mask & direction_match_mask).sum()),
        "direction_match_rate": float((raw_seen_mask & direction_match_mask).sum() / max(1, int(raw_seen_mask.sum()))),
        "missed": int(coverage.eq("missed").sum()),
        "raw_only": int(coverage.eq("raw_only").sum()),
        "event_watch": int(coverage.eq("event_watch").sum()),
        "actionable": int(coverage.eq("actionable").sum()),
        "by_coverage": coverage.value_counts().to_dict(),
    }


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    if not rows:
        return "_none_"
    def clean(value: Any) -> str:
        return str(value).replace("\n", " ").strip()
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        out.append("| " + " | ".join(clean(v) for v in row) + " |")
    return "\n".join(out)


def build_report(
    *,
    rows: pd.DataFrame,
    summary: Dict[str, Any],
    start: dt.date,
    end: dt.date,
    lookback: int,
    horizons: Sequence[int],
    move_threshold: float,
    min_price: float,
    max_abs_return: float,
    common_stock_only: bool,
    csv_path: Path,
) -> str:
    lines: List[str] = [
        f"# Trend Missed-Mover Audit - {start.isoformat()} to {end.isoformat()} / L{lookback}",
        "",
        f"- Horizons: {', '.join(str(h) for h in horizons)} market day(s)",
        f"- Big-move threshold: >= {_fmt_pct(move_threshold)} absolute close-to-close move",
        f"- Minimum signal-date close: ${float(min_price):.2f}",
        f"- Max absolute return cap: {_fmt_pct(max_abs_return) if float(max_abs_return) > 0 else 'none'}",
        f"- Common-stock-only universe: {'yes' if common_stock_only else 'no'}",
        f"- Big mover events: {summary.get('total_big_movers', 0)}",
        f"- Seen anywhere in trend output: {summary.get('raw_seen', 0)} ({_fmt_pct(summary.get('raw_seen_rate'))})",
        f"- Visible in final report lanes: {summary.get('final_visible', 0)} ({_fmt_pct(summary.get('final_visible_rate'))})",
        f"- Direction matched when seen: {summary.get('direction_match', 0)} ({_fmt_pct(summary.get('direction_match_rate'))})",
        f"- Raw-only buried movers: {summary.get('raw_only', 0)}",
        f"- Fully missed movers: {summary.get('missed', 0)}",
        f"- Event-watch captures: {summary.get('event_watch', 0)}",
        f"- Actionable captures: {summary.get('actionable', 0)}",
        "",
        "## Coverage Mix",
    ]
    by_cov = summary.get("by_coverage", {}) if isinstance(summary.get("by_coverage"), dict) else {}
    lines.append(_table(["coverage", "events"], [[k, v] for k, v in sorted(by_cov.items())]))
    lines.append("")

    if rows.empty:
        lines.append("No big movers were available for the selected dates/horizons.")
    else:
        worst_missed = rows[rows["coverage"].isin(["missed", "no_scan_output"])].copy()
        if not worst_missed.empty:
            worst_missed["_abs"] = worst_missed["return_pct"].abs()
            worst_missed = worst_missed.sort_values("_abs", ascending=False).head(15)
        lines.append("## Worst Missed Movers")
        lines.append(
            _table(
                ["signal", "future", "h", "ticker", "move", "coverage"],
                [
                    [
                        r.signal_date,
                        r.future_date,
                        int(r.horizon_days),
                        r.ticker,
                        f"{r.return_pct:+.1f}%",
                        r.coverage,
                    ]
                    for r in worst_missed.itertuples()
                ],
            )
        )
        lines.append("")

        buried = rows[rows["coverage"].eq("raw_only")].copy()
        if not buried.empty:
            buried["_abs"] = buried["return_pct"].abs()
            buried = buried.sort_values("_abs", ascending=False).head(15)
        lines.append("## Buried Raw Movers")
        lines.append(
            _table(
                ["signal", "future", "h", "ticker", "move", "rank", "dir", "score", "strategy", "blocker"],
                [
                    [
                        r.signal_date,
                        r.future_date,
                        int(r.horizon_days),
                        r.ticker,
                        f"{r.return_pct:+.1f}%",
                        int(r.raw_rank),
                        r.pipeline_direction or "-",
                        _fmt_num(r.swing_score),
                        r.strategy or "-",
                        _clean_text(r.blockers)[:90] or "-",
                    ]
                    for r in buried.itertuples()
                ],
            )
        )
        lines.append("")

        visible = rows[rows["coverage"].isin(FINAL_VISIBLE_KINDS)].copy()
        if not visible.empty:
            visible["_abs"] = visible["return_pct"].abs()
            visible = visible.sort_values("_abs", ascending=False).head(15)
        lines.append("## Visible Captures")
        lines.append(
            _table(
                ["signal", "future", "h", "ticker", "move", "coverage", "dir", "score", "strategy"],
                [
                    [
                        r.signal_date,
                        r.future_date,
                        int(r.horizon_days),
                        r.ticker,
                        f"{r.return_pct:+.1f}%",
                        r.coverage,
                        r.pipeline_direction or "-",
                        _fmt_num(r.swing_score),
                        r.strategy or "-",
                    ]
                    for r in visible.itertuples()
                ],
            )
        )
    lines.extend(["", "## Files", f"- CSV: [{csv_path.name}]({csv_path})"])
    return "\n".join(lines) + "\n"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit trend-analysis recall against future big movers.")
    parser.add_argument("--root-dir", default=str(DEFAULT_ROOT))
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--start", type=_parse_date, required=True)
    parser.add_argument("--end", type=_parse_date, required=True)
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK)
    parser.add_argument("--horizons", default=DEFAULT_HORIZONS)
    parser.add_argument("--move-threshold", type=float, default=DEFAULT_MOVE_THRESHOLD)
    parser.add_argument("--min-price", type=float, default=DEFAULT_MIN_PRICE)
    parser.add_argument(
        "--include-etfs-and-indexes",
        action="store_true",
        help="Include non-common-stock rows from stock-screener. Default keeps common stocks only.",
    )
    parser.add_argument(
        "--max-abs-return",
        type=float,
        default=0.0,
        help="Optional cap to exclude split/merger-like close jumps. 0 disables.",
    )
    parser.add_argument(
        "--include-missing-scan-days",
        action="store_true",
        help="Count signal dates that do not have saved trend-analysis raw CSV output. Default skips them.",
    )
    return parser.parse_args(argv)


def run(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    root = Path(args.root_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else root / "out" / "trend_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    horizons = _parse_horizons(args.horizons)
    if args.start > args.end:
        raise ValueError("--start must be <= --end")
    rows = build_missed_mover_rows(
        root=root,
        out_dir=out_dir,
        start=args.start,
        end=args.end,
        lookback=int(args.lookback),
        horizons=horizons,
        move_threshold=float(args.move_threshold),
        min_price=float(args.min_price),
        max_abs_return=float(args.max_abs_return),
        common_stock_only=not bool(args.include_etfs_and_indexes),
        include_missing_scan_days=bool(args.include_missing_scan_days),
    )
    suffix = (
        f"{args.start.isoformat()}_{args.end.isoformat()}-L{int(args.lookback)}-"
        f"H{'_'.join(str(h) for h in horizons)}-M{float(args.move_threshold):.2f}-"
        f"P{float(args.min_price):.0f}"
        + (f"-C{float(args.max_abs_return):.2f}" if float(args.max_abs_return) > 0 else "")
    )
    csv_path = out_dir / f"trend-analysis-missed-movers-{suffix}.csv"
    report_path = out_dir / f"trend-analysis-missed-movers-{suffix}.md"
    metadata_path = out_dir / f"trend-analysis-missed-movers-metadata-{suffix}.json"
    rows.to_csv(csv_path, index=False)
    summary = summarize(rows)
    report = build_report(
        rows=rows,
        summary=summary,
        start=args.start,
        end=args.end,
        lookback=int(args.lookback),
        horizons=horizons,
        move_threshold=float(args.move_threshold),
        min_price=float(args.min_price),
        max_abs_return=float(args.max_abs_return),
        common_stock_only=not bool(args.include_etfs_and_indexes),
        csv_path=csv_path,
    )
    report_path.write_text(report, encoding="utf-8")
    metadata = {
        "root_dir": str(root),
        "out_dir": str(out_dir),
        "start": args.start.isoformat(),
        "end": args.end.isoformat(),
        "lookback": int(args.lookback),
        "horizons": horizons,
        "move_threshold": float(args.move_threshold),
        "min_price": float(args.min_price),
        "max_abs_return": float(args.max_abs_return),
        "common_stock_only": not bool(args.include_etfs_and_indexes),
        "include_missing_scan_days": bool(args.include_missing_scan_days),
        "summary": summary,
        "csv": str(csv_path),
        "report": str(report_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote: {report_path}")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {metadata_path}")
    print(json.dumps(summary, indent=2))
    return {"report": report_path, "csv": csv_path, "metadata": metadata_path, "summary": summary}


def main(argv: Optional[Sequence[str]] = None) -> int:
    run(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
