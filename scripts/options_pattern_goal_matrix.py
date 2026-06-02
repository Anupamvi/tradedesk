#!/usr/bin/env python3
"""Run and summarize options-pattern goal evidence across failure dates."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_BASE_DIR = Path("/Users/anuppamvi/uw_root/tradedesk")
DEFAULT_DATES = (
    "2026-05-15",
    "2026-05-18",
    "2026-05-19",
    "2026-05-20",
    "2026-05-22",
    "2026-05-26",
    "2026-05-27",
    "2026-05-28",
)
DEFAULT_REQUIRED_TICKERS = ("AMD", "MU", "NVDA", "SNDK", "IBM", "CRWD", "HOOD", "NOW")


@dataclass(frozen=True)
class DateRun:
    date: str
    out_dir: Path
    exact_suffix_output: bool
    command_ran: bool
    returncode: int | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", default=str(DEFAULT_BASE_DIR), help="Root tradedesk directory.")
    parser.add_argument("--dates", nargs="+", default=list(DEFAULT_DATES), help="As-of dates to include.")
    parser.add_argument(
        "--required-tickers",
        nargs="+",
        default=list(DEFAULT_REQUIRED_TICKERS),
        help="Ticker symbols that must surface as trade/review/avoid/coverage when high-signal.",
    )
    parser.add_argument(
        "--suffix",
        default="goal_matrix_current",
        help="Output suffix for per-date reruns: <date>_<suffix>.",
    )
    parser.add_argument(
        "--matrix-dir",
        default=None,
        help="Directory for matrix CSV/Markdown. Default: out/options_pattern_pipeline_v1/<suffix>.",
    )
    parser.add_argument(
        "--runs-root",
        default=None,
        help="Root directory for per-date pipeline reruns. Default: <base-dir>/out/options_pattern_pipeline_v1.",
    )
    parser.add_argument("--run-missing", action="store_true", help="Run pipeline for dates without an exact suffix output.")
    parser.add_argument("--force", action="store_true", help="Rerun every requested date even if output exists.")
    parser.add_argument(
        "--python",
        default=sys.executable or "python3",
        help="Python executable for pipeline subprocesses.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    base_dir = Path(args.base_dir).expanduser().resolve()
    root = (
        Path(args.runs_root).expanduser().resolve()
        if args.runs_root
        else base_dir / "out" / "options_pattern_pipeline_v1"
    )
    matrix_dir = Path(args.matrix_dir).expanduser().resolve() if args.matrix_dir else root / args.suffix
    matrix_dir.mkdir(parents=True, exist_ok=True)

    runs: list[DateRun] = []
    for date in args.dates:
        out_dir = root / f"{date}_{args.suffix}"
        exact_suffix_output = has_goal_artifacts(out_dir)
        should_run = args.force or (args.run_missing and not exact_suffix_output)
        returncode: int | None = None
        if should_run:
            returncode = run_pipeline(args.python, base_dir, date, out_dir)
            exact_suffix_output = returncode == 0 and has_goal_artifacts(out_dir)
            if returncode != 0:
                runs.append(DateRun(date, out_dir, exact_suffix_output, True, returncode))
                continue
        elif not has_goal_artifacts(out_dir):
            existing = newest_existing_goal_run(root, date)
            if existing:
                out_dir = existing
                exact_suffix_output = False
        runs.append(DateRun(date, out_dir, exact_suffix_output, should_run, returncode))

    rows = [build_matrix_row(run, args.required_tickers) for run in runs]
    write_csv(matrix_dir / "goal_acceptance_matrix.csv", rows, matrix_fieldnames(args.required_tickers))
    (matrix_dir / "goal_acceptance_matrix.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (matrix_dir / "goal_acceptance_matrix.md").write_text(
        render_matrix_markdown(rows, args.required_tickers),
        encoding="utf-8",
    )
    print(matrix_dir / "goal_acceptance_matrix.md")
    return 1 if any(str(row.get("pipeline_returncode") or "") not in ("", "0") for row in rows) else 0


def run_pipeline(python: str, base_dir: Path, date: str, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        python,
        "-m",
        "uwos.options_pattern_pipeline_v1",
        "--base-dir",
        str(base_dir),
        "--as-of",
        date,
        "--out-dir",
        str(out_dir),
    ]
    print("RUN", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=base_dir).returncode


def has_goal_artifacts(out_dir: Path) -> bool:
    return (out_dir / "goal_evidence.csv").exists() and (out_dir / "metadata.json").exists()


def newest_existing_goal_run(root: Path, date: str) -> Path | None:
    candidates = [p for p in root.glob(f"{date}*") if p.is_dir() and has_goal_artifacts(p)]
    if not candidates:
        return None
    return max(candidates, key=lambda p: (p / "goal_evidence.csv").stat().st_mtime)


def build_matrix_row(run: DateRun, required_tickers: Iterable[str]) -> dict[str, Any]:
    out_dir = run.out_dir
    row: dict[str, Any] = {
        "date": run.date,
        "run_dir": str(out_dir),
        "exact_suffix_output": "yes" if run.exact_suffix_output else "no",
        "command_ran": "yes" if run.command_ran else "no",
        "pipeline_returncode": "" if run.returncode is None else str(run.returncode),
    }
    if run.returncode not in (None, 0):
        row.update({"matrix_status": "PIPELINE_FAILED", "failed_requirements": "pipeline subprocess failed"})
        return row
    if not has_goal_artifacts(out_dir):
        row.update({"matrix_status": "MISSING_ARTIFACTS", "failed_requirements": "missing goal_evidence.csv or metadata.json"})
        return row

    metadata = read_json(out_dir / "metadata.json")
    goal_rows = list(read_csv(out_dir / "goal_evidence.csv"))
    goal_by_req = {r.get("requirement", ""): r for r in goal_rows}
    failed = [r["requirement"] for r in goal_rows if r.get("status") == "FAIL"]
    warned = [r["requirement"] for r in goal_rows if r.get("status") == "WARN"]
    if not run.exact_suffix_output:
        warned.append("not_current_suffix_run")
    missed = goal_by_req.get("missed_mover_audit_visible", {})
    known = goal_by_req.get("known_failure_ticker_surface_audit", {})
    source = goal_by_req.get("high_source_flow_not_silent", {})
    miss_bucket_counts = missed_mover_bucket_counts(out_dir / "missed_mover_audit.csv")
    candidate_generation_gaps = miss_bucket_counts.get("CANDIDATE_GENERATION_GAP", "")

    row.update(
        {
            "matrix_status": matrix_status(failed, warned),
            "goal_evidence_status": metadata.get("goal_evidence_status", ""),
            "verdict": metadata.get("verdict", ""),
            "daily_trade_decision": metadata.get("daily_trade_decision", ""),
            "auto_approved_count": (metadata.get("candidate_counts") or {}).get("auto_approved", ""),
            "trade_review_count": (metadata.get("candidate_counts") or {}).get("trade_review_candidates", ""),
            "avoid_count": (metadata.get("candidate_counts") or {}).get("avoid", ""),
            "source_coverage_count": (metadata.get("candidate_counts") or {}).get("source_ticker_coverage", ""),
            "failed_requirements": ";".join(failed),
            "warn_requirements": ";".join(warned),
            "known_ticker_status": known.get("status", ""),
            "known_ticker_evidence": known.get("evidence", ""),
            "high_source_status": source.get("status", ""),
            "missed_mover_status": missed.get("status", ""),
            "missed_mover_evidence": missed.get("evidence", ""),
            "candidate_generation_gaps": candidate_generation_gaps
            or extract_count(missed.get("evidence", ""), "candidate_generation_gaps"),
            "miss_bucket_counts": format_counts(miss_bucket_counts),
            "not_option_tradeable_missing_quote": miss_bucket_counts.get("NOT_OPTION_TRADEABLE_MISSING_QUOTE", "")
            or extract_bucket_count(missed.get("evidence", ""), "NOT_OPTION_TRADEABLE_MISSING_QUOTE"),
            "not_option_tradeable_quote_failed": miss_bucket_counts.get("NOT_OPTION_TRADEABLE_QUOTE_FAILED", "")
            or extract_bucket_count(missed.get("evidence", ""), "NOT_OPTION_TRADEABLE_QUOTE_FAILED"),
            "flagged_missed_movers": miss_bucket_counts.get("FLAGGED_BY_PATTERN_PIPELINE", "")
            or extract_bucket_count(missed.get("evidence", ""), "FLAGGED_BY_PATTERN_PIPELINE"),
            "actionable_tickers": tickers_from_csv(out_dir / "actionable_trades.csv"),
            "trade_review_tickers": tickers_from_csv(out_dir / "trade_review_candidates.csv"),
            "avoid_tickers": tickers_from_csv(out_dir / "blocked_candidates.csv"),
        }
    )
    ticker_status = required_ticker_details(out_dir, required_tickers)
    for ticker in required_tickers:
        row[f"{ticker}_status"] = ticker_status.get(ticker, {}).get("status", "")
        row[f"{ticker}_ticket"] = ticker_status.get(ticker, {}).get("ticket", "")
        row[f"{ticker}_reason"] = ticker_status.get(ticker, {}).get("reason", "")
    return row


def matrix_status(failed: list[str], warned: list[str]) -> str:
    if failed:
        return "FAIL"
    if warned:
        return "PARTIAL"
    return "PASS_DAILY_NOT_GLOBAL"


def required_ticker_details(out_dir: Path, tickers: Iterable[str]) -> dict[str, dict[str, str]]:
    result = {ticker: {"status": "", "ticket": "", "reason": ""} for ticker in tickers}
    sources = [
        ("TRADE", out_dir / "actionable_trades.csv", "ticker"),
        ("REVIEW", out_dir / "trade_review_candidates.csv", "ticker"),
        ("AVOID", out_dir / "blocked_candidates.csv", "ticker"),
        ("COVERAGE", out_dir / "source_ticker_coverage.csv", "ticker"),
    ]
    for label, path, field in sources:
        if not path.exists():
            continue
        for row in read_csv(path):
            ticker = str(row.get(field) or "").upper()
            if ticker in result and not result[ticker]["status"]:
                result[ticker] = {
                    "status": format_ticker_status(label, row),
                    "ticket": format_ticket(row),
                    "reason": format_reason(row),
                }
    return result


def format_ticker_status(label: str, row: dict[str, str]) -> str:
    status = row.get("decision_surface_status") or row.get("status") or row.get("classification") or label
    return f"{label}:{status}"


def format_ticket(row: dict[str, str]) -> str:
    parts: list[str] = []
    fields = [
        ("strategy", "strategy"),
        ("side", "buy_or_sell"),
        ("type", "call_or_put"),
        ("strikes", "strike_rates"),
        ("expiration", "expiration_date"),
        ("entry", "suggested_entry_debit_credit_range"),
        ("legs", "trade_legs"),
        ("source", "decision_artifact"),
    ]
    for label, field in fields:
        value = clean_cell(row.get(field))
        if value:
            parts.append(f"{label}={value}")
    entry_limit = clean_cell(row.get("entry_limit"))
    if entry_limit and not any(part.startswith("entry=") for part in parts):
        parts.append(f"entry={entry_limit}")
    return " | ".join(parts)


def format_reason(row: dict[str, str]) -> str:
    fields = [
        "hard_blockers",
        "block_reasons",
        "decision_block_reasons",
        "source_gap_reason",
        "edge_review_reason",
        "major_risks",
    ]
    reasons = [clean_cell(row.get(field)) for field in fields if clean_cell(row.get(field))]
    return " | ".join(reasons)


def tickers_from_csv(path: Path, limit: int = 30) -> str:
    if not path.exists():
        return ""
    tickers = []
    for row in read_csv(path):
        ticker = str(row.get("ticker") or "").upper()
        if ticker and ticker not in tickers:
            tickers.append(ticker)
        if len(tickers) >= limit:
            break
    return ",".join(tickers)


def extract_count(text: str, key: str) -> str:
    marker = f"{key}="
    if marker not in text:
        return ""
    tail = text.split(marker, 1)[1]
    return tail.split(";", 1)[0].strip()


def extract_bucket_count(text: str, bucket: str) -> str:
    marker = f"{bucket}:"
    if marker not in text:
        return ""
    tail = text.split(marker, 1)[1]
    return tail.split(",", 1)[0].split(";", 1)[0].strip()


def missed_mover_bucket_counts(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not path.exists():
        return counts
    for row in read_csv(path):
        bucket = str(row.get("miss_bucket") or "UNKNOWN").strip() or "UNKNOWN"
        counts[bucket] = counts.get(bucket, 0) + 1
    return counts


def format_counts(counts: dict[str, int]) -> str:
    return ";".join(f"{key}:{counts[key]}" for key in sorted(counts))


def clean_cell(value: str | None, limit: int = 260) -> str:
    if value is None:
        return ""
    text = " ".join(str(value).split())
    if len(text) > limit:
        return text[: limit - 3].rstrip() + "..."
    return text


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def matrix_fieldnames(required_tickers: Iterable[str]) -> list[str]:
    return [
        "date",
        "matrix_status",
        "goal_evidence_status",
        "verdict",
        "daily_trade_decision",
        "auto_approved_count",
        "trade_review_count",
        "avoid_count",
        "source_coverage_count",
        "candidate_generation_gaps",
        "miss_bucket_counts",
        "not_option_tradeable_missing_quote",
        "not_option_tradeable_quote_failed",
        "flagged_missed_movers",
        "failed_requirements",
        "warn_requirements",
        "known_ticker_status",
        "known_ticker_evidence",
        "high_source_status",
        "missed_mover_status",
        "missed_mover_evidence",
        "actionable_tickers",
        "trade_review_tickers",
        "avoid_tickers",
    ] + [field for ticker in required_tickers for field in (f"{ticker}_status", f"{ticker}_ticket", f"{ticker}_reason")] + [
        "run_dir",
        "exact_suffix_output",
        "command_ran",
        "pipeline_returncode",
    ]


def render_matrix_markdown(rows: list[dict[str, Any]], required_tickers: Iterable[str]) -> str:
    lines = [
        "# Options Pattern Goal Acceptance Matrix",
        "",
        "This matrix aggregates per-date `goal_evidence.csv` artifacts. `PARTIAL` means no hard failure, but at least one warning remains, usually missed-mover coverage or a fallback run that was not produced with the requested suffix.",
        "",
        "This is an acceptance gate, not proof that the rebuild goal is complete. The goal still requires every target date to be current, leakage-safe, and positive expectancy after costs/slippage versus baselines.",
        "",
        "| Date | Status | Exact | Decision | Auto | Review | Avoid | Gap Misses | Known Tickers | Warnings |",
        "|---|---|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        ticker_bits = []
        for ticker in required_tickers:
            status = row.get(f"{ticker}_status") or ""
            if status:
                ticker_bits.append(f"{ticker}={status.split(':', 1)[0]}")
        issue_bits = [bit for bit in (row.get("failed_requirements"), row.get("warn_requirements")) if bit]
        lines.append(
            "| {date} | {matrix_status} | {exact} | {daily_trade_decision} | {auto_approved_count} | "
            "{trade_review_count} | {avoid_count} | {candidate_generation_gaps} | {tickers} | {warns} |".format(
                date=row.get("date", ""),
                matrix_status=row.get("matrix_status", ""),
                exact=row.get("exact_suffix_output", ""),
                daily_trade_decision=row.get("daily_trade_decision", ""),
                auto_approved_count=row.get("auto_approved_count", ""),
                trade_review_count=row.get("trade_review_count", ""),
                avoid_count=row.get("avoid_count", ""),
                candidate_generation_gaps=row.get("candidate_generation_gaps", ""),
                tickers=", ".join(ticker_bits),
                warns=";".join(issue_bits),
            )
        )
    lines.extend(["", "## Detail"])
    for row in rows:
        lines.append(f"### {row.get('date')} - {row.get('matrix_status')}")
        lines.append(f"- Run dir: `{row.get('run_dir')}`")
        lines.append(f"- Exact suffix output: {row.get('exact_suffix_output')}")
        lines.append(f"- Known ticker evidence: {row.get('known_ticker_evidence') or 'n/a'}")
        lines.append(f"- Missed mover evidence: {row.get('missed_mover_evidence') or 'n/a'}")
        lines.append(f"- Miss bucket counts: {row.get('miss_bucket_counts') or 'n/a'}")
        lines.append("- Required ticker surface:")
        for ticker in required_tickers:
            status = row.get(f"{ticker}_status") or "MISSING"
            ticket = row.get(f"{ticker}_ticket") or "n/a"
            reason = row.get(f"{ticker}_reason") or "n/a"
            lines.append(f"  - {ticker}: {status}; ticket: {ticket}; reason: {reason}")
        lines.append(f"- Failed requirements: {row.get('failed_requirements') or 'none'}")
        lines.append(f"- Warning requirements: {row.get('warn_requirements') or 'none'}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
