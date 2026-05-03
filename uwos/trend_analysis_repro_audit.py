#!/usr/bin/env python3
"""Reproducibility and time-safety audit for trend-analysis.

This is the preflight gate for trusting a dated trend-analysis report. It runs
the same command twice into separate folders, compares canonical CSV outputs,
and inspects metadata for future-data leakage.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_ROOT = Path("/Users/anuppamvi/uw_root/tradedesk")

CANONICAL_CSV_NAMES = [
    "trend_analysis_raw_{suffix}.csv",
    "trend-analysis-candidates-{suffix}.csv",
    "trend-analysis-actionable-{suffix}.csv",
    "trend-analysis-tactical-probes-{suffix}.csv",
    "trend-analysis-proven-tickets-{suffix}.csv",
    "trend-analysis-current-setups-{suffix}.csv",
    "trend-analysis-event-watch-{suffix}.csv",
    "trend-analysis-max-conviction-{suffix}.csv",
    "trend-analysis-trade-workups-{suffix}.csv",
    "trend-analysis-patterns-{suffix}.csv",
    "trend-analysis-quote-replay-{suffix}.csv",
    "trend-analysis-walk-forward-{suffix}.csv",
    "trend-analysis-research-audit-{suffix}.csv",
    "trend-analysis-research-audit-by-horizon-{suffix}.csv",
    "trend-analysis-research-outcomes-{suffix}.csv",
    "trend-analysis-strategy-family-audit-{suffix}.csv",
    "trend-analysis-rolling-strategy-family-audit-{suffix}.csv",
    "trend-analysis-ticker-playbook-audit-{suffix}.csv",
    "trend-analysis-rolling-ticker-playbook-audit-{suffix}.csv",
    "trend-analysis-schwab-actual-strategy-audit-{suffix}.csv",
    "trend-analysis-schwab-actual-playbook-audit-{suffix}.csv",
    "trend-analysis-schwab-actual-shape-audit-{suffix}.csv",
]

COUNT_METADATA_BY_ARTIFACT = {
    "trend-analysis-candidates-{suffix}.csv": "candidate_shortlist",
    "trend-analysis-actionable-{suffix}.csv": "actionable",
    "trend-analysis-tactical-probes-{suffix}.csv": "tactical_probes",
    "trend-analysis-proven-tickets-{suffix}.csv": "proven_playbook_tickets",
    "trend-analysis-current-setups-{suffix}.csv": "current_setups",
    "trend-analysis-event-watch-{suffix}.csv": "event_watch",
    "trend-analysis-max-conviction-{suffix}.csv": "max_conviction",
    "trend-analysis-trade-workups-{suffix}.csv": "trade_workups",
    "trend-analysis-patterns-{suffix}.csv": "patterns",
    "trend-analysis-walk-forward-{suffix}.csv": "walk_forward_rows",
    "trend-analysis-research-audit-{suffix}.csv": "research_audit_rows",
    "trend-analysis-research-audit-by-horizon-{suffix}.csv": "research_horizon_audit_rows",
    "trend-analysis-research-outcomes-{suffix}.csv": "research_outcome_rows",
    "trend-analysis-strategy-family-audit-{suffix}.csv": "strategy_family_audit_rows",
    "trend-analysis-rolling-strategy-family-audit-{suffix}.csv": "rolling_strategy_family_audit_rows",
    "trend-analysis-ticker-playbook-audit-{suffix}.csv": "ticker_playbook_audit_rows",
    "trend-analysis-rolling-ticker-playbook-audit-{suffix}.csv": "rolling_ticker_playbook_audit_rows",
    "trend-analysis-schwab-actual-strategy-audit-{suffix}.csv": "schwab_actual_strategy_audit_rows",
    "trend-analysis-schwab-actual-playbook-audit-{suffix}.csv": "schwab_actual_playbook_audit_rows",
    "trend-analysis-schwab-actual-shape-audit-{suffix}.csv": "schwab_actual_shape_audit_rows",
}

DETERMINISTIC_METADATA_KEYS = [
    "as_of",
    "effective_signal_date",
    "lookback",
    "trading_days",
    "candidate_pool",
    "candidates",
    "candidate_shortlist",
    "proven_playbook_tickets",
    "tactical_probes",
    "current_setups",
    "event_watch",
    "max_conviction",
    "trade_workups",
    "actionable",
    "patterns",
    "backtest_enabled",
    "schwab_enabled",
    "schwab_live_reason",
    "latest_data_date",
    "quote_replay_mode",
    "quote_replay_counts",
    "quote_replay_rows",
    "walk_forward_rows",
    "research_audit_rows",
    "research_horizon_audit_rows",
    "research_outcome_rows",
    "strategy_family_audit_rows",
    "rolling_strategy_family_audit_rows",
    "ticker_playbook_audit_rows",
    "rolling_ticker_playbook_audit_rows",
    "schwab_actual_strategy_audit_rows",
    "schwab_actual_playbook_audit_rows",
    "schwab_actual_shape_audit_rows",
    "research_audit_verdicts",
    "research_horizon_audit_verdicts",
    "strategy_family_audit_verdicts",
]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run trend-analysis twice and fail on output drift or future-data leakage."
    )
    p.add_argument("as_of", help="Trend-analysis as-of date, YYYY-MM-DD.")
    p.add_argument("lookback", nargs="?", type=int, default=30, help="Usable market-data-day lookback.")
    p.add_argument("--root-dir", type=Path, default=DEFAULT_ROOT)
    p.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Audit output root. Default: <root-dir>/out/trend_analysis_repro_audit/<date>-L<lookback>.",
    )
    p.add_argument("--schwab-report-json", type=Path, default=None)
    p.add_argument("--position-json", type=Path, default=None)
    p.add_argument("--candidate-pool", type=int, default=None)
    p.add_argument("--top", type=int, default=None)
    p.add_argument("--timeout-sec", type=int, default=900)
    p.add_argument(
        "--allow-short-window",
        action="store_true",
        help="Do not fail when fewer than lookback usable market-data days are available.",
    )
    p.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Extra argument passed through to trend_analysis. Repeat for multiple args.",
    )
    return p.parse_args(argv)


def as_date(value: Any) -> Optional[dt.date]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return dt.date.fromisoformat(text[:10])
    except ValueError:
        return None


def suffix(as_of: dt.date, lookback: int) -> str:
    return f"{as_of.isoformat()}-L{int(lookback)}"


def artifact_names(as_of: dt.date, lookback: int) -> List[str]:
    s = suffix(as_of, lookback)
    return [name.format(suffix=s) for name in CANONICAL_CSV_NAMES]


def metadata_name(as_of: dt.date, lookback: int) -> str:
    return f"trend-analysis-metadata-{suffix(as_of, lookback)}.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def csv_row_count(path: Path) -> int:
    if not path.exists():
        return -1
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def compare_artifacts(run_a: Path, run_b: Path, as_of: dt.date, lookback: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name in artifact_names(as_of, lookback):
        path_a = run_a / name
        path_b = run_b / name
        exists_a = path_a.exists()
        exists_b = path_b.exists()
        sha_a = sha256_file(path_a) if exists_a else ""
        sha_b = sha256_file(path_b) if exists_b else ""
        rows.append(
            {
                "artifact": name,
                "path_a": str(path_a),
                "path_b": str(path_b),
                "exists_a": exists_a,
                "exists_b": exists_b,
                "sha256_a": sha_a,
                "sha256_b": sha_b,
                "same": bool(exists_a and exists_b and sha_a == sha_b),
                "rows_a": csv_row_count(path_a) if exists_a else -1,
                "rows_b": csv_row_count(path_b) if exists_b else -1,
            }
        )
    return rows


def load_metadata(run_dir: Path, as_of: dt.date, lookback: int) -> Dict[str, Any]:
    path = run_dir / metadata_name(as_of, lookback)
    if not path.exists():
        return {"_missing": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def deterministic_metadata_diff(meta_a: Dict[str, Any], meta_b: Dict[str, Any]) -> List[Dict[str, Any]]:
    diffs: List[Dict[str, Any]] = []
    for key in DETERMINISTIC_METADATA_KEYS:
        if meta_a.get(key) != meta_b.get(key):
            diffs.append({"key": key, "a": meta_a.get(key), "b": meta_b.get(key)})
    return diffs


def _date_from_position_json(path_text: str) -> Optional[dt.date]:
    name = Path(str(path_text or "")).name
    marker = "position_data_"
    if not name.startswith(marker) or not name.endswith(".json"):
        return None
    return as_date(name[len(marker) : -len(".json")])


def _csv_dates(path: Path, columns: Iterable[str]) -> List[Tuple[str, dt.date]]:
    if not path.exists():
        return []
    out: List[Tuple[str, dt.date]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for column in columns:
                d = as_date(row.get(column))
                if d is not None:
                    out.append((column, d))
    return out


def time_safety_failures(
    meta: Dict[str, Any],
    run_dir: Path,
    as_of: dt.date,
    lookback: int,
    *,
    allow_short_window: bool = False,
) -> List[str]:
    failures: List[str] = []
    meta_asof = as_date(meta.get("as_of"))
    if meta_asof != as_of:
        failures.append(f"metadata as_of {meta.get('as_of')} != requested {as_of.isoformat()}")

    effective = as_date(meta.get("effective_signal_date"))
    if effective is None:
        failures.append("missing effective_signal_date")
    elif effective > as_of:
        failures.append(f"effective_signal_date {effective.isoformat()} is after as_of {as_of.isoformat()}")

    trading_days = [as_date(day) for day in meta.get("trading_days", [])]
    valid_days = [day for day in trading_days if day is not None]
    if len(valid_days) != len(trading_days):
        failures.append("metadata trading_days contains unparsable dates")
    if not allow_short_window and len(valid_days) != int(lookback):
        failures.append(f"trading_days length {len(valid_days)} != requested lookback {lookback}")
    future_days = [day.isoformat() for day in valid_days if day > as_of]
    if future_days:
        failures.append(f"trading_days after as_of: {', '.join(future_days[:5])}")
    if effective is not None and valid_days and valid_days[-1] != effective:
        failures.append(
            f"last trading day {valid_days[-1].isoformat()} != effective_signal_date {effective.isoformat()}"
        )

    latest_data = as_date(meta.get("latest_data_date"))
    if latest_data is not None and latest_data > as_of and bool(meta.get("schwab_enabled")):
        failures.append(
            f"Schwab live chain enabled for historical as_of {as_of.isoformat()} with latest data {latest_data.isoformat()}"
        )

    position_summary = meta.get("open_position_summary") or {}
    position_date = _date_from_position_json(str(position_summary.get("position_json") or ""))
    if position_date is not None and position_date > as_of:
        failures.append(
            f"position snapshot {position_date.isoformat()} is after as_of {as_of.isoformat()}"
        )

    schwab_summary = meta.get("schwab_actual_summary") or {}
    if schwab_summary.get("status") == "ok":
        audit_asof = as_date(schwab_summary.get("audit_as_of"))
        if audit_asof != as_of:
            failures.append(
                f"Schwab actual audit_as_of {schwab_summary.get('audit_as_of')} != requested {as_of.isoformat()}"
            )
        parsed = int(schwab_summary.get("parsed_closed_trades", 0) or 0)
        parsed_asof = int(schwab_summary.get("parsed_closed_trades_asof", parsed) or 0)
        if parsed_asof != parsed:
            failures.append(
                f"Schwab actual parsed count {parsed} != as-of count {parsed_asof}"
            )

    quote_csv = run_dir / f"trend-analysis-quote-replay-{suffix(as_of, lookback)}.csv"
    future_quote_dates = [
        f"{column}={day.isoformat()}"
        for column, day in _csv_dates(quote_csv, ["quote_replay_signal_date", "quote_replay_exit_date"])
        if day > as_of
    ]
    if future_quote_dates:
        failures.append(f"quote replay contains future dates: {', '.join(future_quote_dates[:5])}")

    raw_csv = run_dir / f"trend_analysis_raw_{suffix(as_of, lookback)}.csv"
    if latest_data is not None and latest_data > as_of and raw_csv.exists():
        with raw_csv.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if "live_validated" in (reader.fieldnames or []):
                live_true = 0
                for row in reader:
                    value = str(row.get("live_validated") or "").strip().lower()
                    note = str(row.get("live_validation_note") or "").strip().lower()
                    local_quote_validated = note.startswith("local uw quote snapshot")
                    if value in {"true", "1", "yes", "y"} and not local_quote_validated:
                        live_true += 1
                        if live_true > 0:
                            break
                if live_true:
                    failures.append("historical run has current-Schwab live_validated=true rows")

    return failures


def count_failures(meta: Dict[str, Any], run_dir: Path, as_of: dt.date, lookback: int) -> List[str]:
    failures: List[str] = []
    for template, metadata_key in COUNT_METADATA_BY_ARTIFACT.items():
        path = run_dir / template.format(suffix=suffix(as_of, lookback))
        if not path.exists():
            failures.append(f"missing count artifact {path.name}")
            continue
        observed = csv_row_count(path)
        expected = int(meta.get(metadata_key, 0) or 0)
        if observed != expected:
            failures.append(f"{path.name} rows {observed} != metadata {metadata_key} {expected}")
    return failures


def run_trend_once(
    *,
    root: Path,
    out_dir: Path,
    as_of: dt.date,
    lookback: int,
    schwab_report_json: Optional[Path],
    position_json: Optional[Path],
    candidate_pool: Optional[int],
    top: Optional[int],
    extra_args: Sequence[str],
    timeout_sec: int,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "uwos.trend_analysis",
        as_of.isoformat(),
        str(int(lookback)),
        "--root-dir",
        str(root),
        "--out-dir",
        str(out_dir),
    ]
    if schwab_report_json is not None:
        cmd.extend(["--schwab-report-json", str(schwab_report_json)])
    if position_json is not None:
        cmd.extend(["--position-json", str(position_json)])
    if candidate_pool is not None:
        cmd.extend(["--candidate-pool", str(int(candidate_pool))])
    if top is not None:
        cmd.extend(["--top", str(int(top))])
    cmd.extend(extra_args)
    started = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=int(timeout_sec),
    )
    (out_dir / "run.log").write_text(proc.stdout, encoding="utf-8")
    return {
        "command": cmd,
        "returncode": int(proc.returncode),
        "seconds": round(time.time() - started, 2),
        "out_dir": str(out_dir),
        "tail": proc.stdout[-4000:] if proc.returncode != 0 else "",
    }


def write_report(out_root: Path, payload: Dict[str, Any]) -> None:
    lines = [
        "# Trend Analysis Repro Audit",
        "",
        f"Status: **{payload['status']}**",
        f"As of: `{payload['as_of']}`",
        f"Lookback: `{payload['lookback']}`",
        "",
        "## Run Results",
        "",
        f"- Run A: returncode `{payload['run_a']['returncode']}`, seconds `{payload['run_a']['seconds']}`",
        f"- Run B: returncode `{payload['run_b']['returncode']}`, seconds `{payload['run_b']['seconds']}`",
        "",
        "## Determinism",
        "",
    ]
    drifted = [row for row in payload["artifacts"] if not row["same"]]
    if drifted:
        for row in drifted:
            lines.append(f"- FAIL `{row['artifact']}`")
    else:
        lines.append("- PASS all canonical CSV artifacts matched")

    lines.extend(["", "## Time Safety", ""])
    failures = payload.get("time_safety_failures", [])
    if failures:
        for failure in failures:
            lines.append(f"- FAIL {failure}")
    else:
        lines.append("- PASS no future-data leakage detected in metadata/quote replay")

    lines.extend(["", "## Count Checks", ""])
    count_fail = payload.get("count_failures", [])
    if count_fail:
        for failure in count_fail:
            lines.append(f"- FAIL {failure}")
    else:
        lines.append("- PASS report metadata counts match CSV row counts")

    if payload.get("metadata_diffs"):
        lines.extend(["", "## Metadata Drift", ""])
        for row in payload["metadata_diffs"]:
            lines.append(f"- FAIL `{row['key']}` differs")

    (out_root / "TREND_ANALYSIS_REPRO_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit(args: argparse.Namespace) -> Dict[str, Any]:
    as_of = dt.date.fromisoformat(str(args.as_of))
    lookback = int(args.lookback)
    root = args.root_dir.expanduser().resolve()
    out_root = (
        args.out_root.expanduser().resolve()
        if args.out_root is not None
        else (root / "out" / "trend_analysis_repro_audit" / suffix(as_of, lookback)).resolve()
    )
    out_root.mkdir(parents=True, exist_ok=True)
    run_a_dir = out_root / "run_a"
    run_b_dir = out_root / "run_b"
    for run_dir in (run_a_dir, run_b_dir):
        if run_dir.exists():
            shutil.rmtree(run_dir)

    run_a = run_trend_once(
        root=root,
        out_dir=run_a_dir,
        as_of=as_of,
        lookback=lookback,
        schwab_report_json=args.schwab_report_json.expanduser().resolve() if args.schwab_report_json else None,
        position_json=args.position_json.expanduser().resolve() if args.position_json else None,
        candidate_pool=args.candidate_pool,
        top=args.top,
        extra_args=list(args.extra_arg or []),
        timeout_sec=int(args.timeout_sec),
    )
    run_b = run_trend_once(
        root=root,
        out_dir=run_b_dir,
        as_of=as_of,
        lookback=lookback,
        schwab_report_json=args.schwab_report_json.expanduser().resolve() if args.schwab_report_json else None,
        position_json=args.position_json.expanduser().resolve() if args.position_json else None,
        candidate_pool=args.candidate_pool,
        top=args.top,
        extra_args=list(args.extra_arg or []),
        timeout_sec=int(args.timeout_sec),
    )

    meta_a = load_metadata(run_a_dir, as_of, lookback)
    meta_b = load_metadata(run_b_dir, as_of, lookback)
    artifacts = compare_artifacts(run_a_dir, run_b_dir, as_of, lookback)
    metadata_diffs = deterministic_metadata_diff(meta_a, meta_b)
    time_fail = time_safety_failures(
        meta_a,
        run_a_dir,
        as_of,
        lookback,
        allow_short_window=bool(args.allow_short_window),
    )
    time_fail.extend(
        f"run_b: {failure}"
        for failure in time_safety_failures(
            meta_b,
            run_b_dir,
            as_of,
            lookback,
            allow_short_window=bool(args.allow_short_window),
        )
    )
    count_fail = count_failures(meta_a, run_a_dir, as_of, lookback)
    count_fail.extend(
        f"run_b: {failure}" for failure in count_failures(meta_b, run_b_dir, as_of, lookback)
    )
    failed = (
        run_a["returncode"] != 0
        or run_b["returncode"] != 0
        or any(not row["same"] for row in artifacts)
        or bool(metadata_diffs)
        or bool(time_fail)
        or bool(count_fail)
    )
    payload = {
        "status": "FAIL" if failed else "PASS",
        "root": str(root),
        "out_root": str(out_root),
        "as_of": as_of.isoformat(),
        "lookback": lookback,
        "run_a": run_a,
        "run_b": run_b,
        "artifacts": artifacts,
        "metadata_diffs": metadata_diffs,
        "time_safety_failures": time_fail,
        "count_failures": count_fail,
        "metadata_a": meta_a,
        "metadata_b": meta_b,
    }
    (out_root / "trend_analysis_repro_audit_summary.json").write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )
    write_report(out_root, payload)
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    payload = run_audit(args)
    print(
        f"{payload['status']} as_of={payload['as_of']} lookback={payload['lookback']} "
        f"out={payload['out_root']}",
        flush=True,
    )
    if payload["status"] != "PASS":
        print(f"  artifact_drift={sum(1 for row in payload['artifacts'] if not row['same'])}", flush=True)
        print(f"  metadata_diffs={len(payload['metadata_diffs'])}", flush=True)
        print(f"  time_safety_failures={len(payload['time_safety_failures'])}", flush=True)
        print(f"  count_failures={len(payload['count_failures'])}", flush=True)
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
