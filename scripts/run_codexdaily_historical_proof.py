#!/usr/bin/env python3
"""Checkpoint/resume CodexDaily V3/V4 historical proof runner.

The goal acceptance verifier needs current-code V3/V4 evidence without relying
on stale manifests under root/out.  This runner writes to a caller-provided proof
directory, so it can run safely from Codex sessions whose writable workspace is
not the tradedesk repo.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from codexuw.validation import select_systematic_date_folders


PIPELINES = ("v3", "v4")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run resumable CodexDaily V3/V4 historical proof into an external directory.")
    parser.add_argument("--root", default="/Users/anuppamvi/uw_root/tradedesk")
    parser.add_argument("--out-dir", default="/tmp/codexdaily_historical_proof")
    parser.add_argument("--from-date", default="2026-01-01")
    parser.add_argument("--to-date", default="2026-12-31")
    parser.add_argument("--as-of", default="", help="Optional upper bound. Defaults to --to-date.")
    parser.add_argument("--pipeline", choices=["v3", "v4", "both"], default="both")
    parser.add_argument("--max-dates", type=int, default=0, help="Optional smoke cap after date filtering. Default 0 means all dates.")
    parser.add_argument("--force", action="store_true", help="Rerun dates even when a manifest already exists.")
    parser.add_argument("--bot-max-rows", type=int, default=0, help="Bot-flow cap. Default 0 reads all available bot-flow rows.")
    parser.add_argument("--max-tickers", type=int, default=0, help="Ticker discovery cap. Default 0 keeps every eligible ticker.")
    parser.add_argument("--max-candidates", type=int, default=0, help="Candidate cap. Default 0 keeps every constructed candidate.")
    parser.add_argument("--risk-budget", type=float, default=3000.0)
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args(argv)


def _parse_date(value: str) -> dt.date | None:
    if not value:
        return None
    return dt.date.fromisoformat(value)


def _date_text(folder: Path) -> str:
    return folder.name[:10]


def selected_source_dates(root: Path, from_date: dt.date, to_date: dt.date, max_dates: int = 0) -> list[Path]:
    folders = select_systematic_date_folders(root, as_of=to_date, latest_n=10000)
    selected = [folder for folder in folders if from_date <= dt.date.fromisoformat(_date_text(folder)) <= to_date]
    selected = sorted(selected, key=lambda path: _date_text(path))
    return selected[:max_dates] if max_dates and max_dates > 0 else selected


def manifest_path(proof_dir: Path, pipeline: str, day: str) -> Path:
    prefix = f"codexdaily_{pipeline}"
    return proof_dir / pipeline / f"{prefix}_{day}" / f"{prefix}_manifest_{day}.json"


def report_path(proof_dir: Path, pipeline: str, day: str) -> Path:
    prefix = f"codexdaily_{pipeline}"
    return proof_dir / pipeline / f"{prefix}_{day}" / f"{prefix}_report_{day}.md"


def run_command(args: argparse.Namespace, pipeline: str, folder: Path, proof_dir: Path) -> tuple[int, list[str]]:
    day = _date_text(folder)
    out_dir = manifest_path(proof_dir, pipeline, day).parent
    module = "codexuw.daily_v3" if pipeline == "v3" else "codexuw.daily_v4"
    command = [
        args.python,
        "-m",
        module,
        "run",
        "--base-dir",
        str(folder),
        "--out-dir",
        str(out_dir),
        "--offline",
        "--skip-portfolio",
        "--skip-catalysts",
        "--skip-recent-performance",
        "--bot-max-rows",
        str(args.bot_max_rows),
        "--max-tickers",
        str(args.max_tickers),
        "--max-candidates",
        str(args.max_candidates),
        "--max-final-trades",
        "0",
        "--risk-budget",
        str(args.risk_budget),
        "--report-mode",
        "historical",
    ]
    if pipeline == "v4":
        # V4 defaults to no aggregate slate cap. Keep --risk-budget 0 so visible
        # target math does not hide candidates behind a slate budget.
        command[command.index(str(args.risk_budget))] = "0"
    result = subprocess.run(command, cwd=REPO_ROOT)
    return result.returncode, command


def validate_manifest(path: Path, pipeline: str) -> tuple[str, str]:
    if not path.exists():
        return "FAIL", "manifest_missing"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return "FAIL", f"manifest_unreadable:{exc}"
    policy = manifest.get("visible_signal_policy") or {}
    if not policy:
        return "FAIL", "visible_signal_policy_missing"
    if policy.get("active_execute_cap") not in {None, ""}:
        return "FAIL", f"active_execute_cap={policy.get('active_execute_cap')}"
    if pipeline == "v4" and policy.get("no_miss_reporting") is not True:
        return "FAIL", "v4_no_miss_reporting_missing"
    if pipeline == "v3" and policy.get("risk_caps_size_and_label_only") is not True:
        return "FAIL", "v3_risk_label_policy_missing"
    return "PASS", "current-code manifest has uncapped visibility policy"


def proof_scope(args: argparse.Namespace | None) -> tuple[str, str, dict[str, Any]]:
    if args is None:
        return "UNKNOWN", "runner args unavailable", {}
    config = {
        "max_dates": int(args.max_dates),
        "bot_max_rows": int(args.bot_max_rows),
        "max_tickers": int(args.max_tickers),
        "max_candidates": int(args.max_candidates),
        "offline": True,
        "skip_portfolio": True,
        "skip_catalysts": True,
        "skip_recent_performance": True,
    }
    caps: list[str] = []
    if int(args.max_dates) > 0:
        caps.append(f"max_dates={int(args.max_dates)}")
    if int(args.bot_max_rows) > 0:
        caps.append(f"bot_max_rows={int(args.bot_max_rows)}")
    if int(args.max_tickers) > 0:
        caps.append(f"max_tickers={int(args.max_tickers)}")
    if int(args.max_candidates) > 0:
        caps.append(f"max_candidates={int(args.max_candidates)}")
    if caps:
        return "CAPPED", ",".join(caps), config
    return "FULL", "uncapped historical discovery/candidate proof; offline mode is expected for dated EOD validation", config


def write_summary(proof_dir: Path, rows: list[dict[str, Any]], args: argparse.Namespace | None = None) -> tuple[Path, Path, Path]:
    proof_dir.mkdir(parents=True, exist_ok=True)
    csv_path = proof_dir / "codexdaily_historical_proof_summary.csv"
    json_path = proof_dir / "codexdaily_historical_proof_checkpoint.json"
    md_path = proof_dir / "codexdaily_historical_proof_report.md"
    scope_status, scope_note, scope_config = proof_scope(args)
    fields = [
        "pipeline",
        "date",
        "status",
        "returncode",
        "proof_scope_status",
        "proof_scope_note",
        "manifest_path",
        "report_path",
        "validation_note",
        "command",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    checkpoint = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "proof_scope_status": scope_status,
        "proof_scope_notes": scope_note,
        "proof_config": scope_config,
        "row_count": len(rows),
        "pass_count": sum(1 for row in rows if row.get("status") == "PASS"),
        "fail_count": sum(1 for row in rows if row.get("status") == "FAIL"),
        "rows": rows,
    }
    json_path.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
    lines = [
        "# CodexDaily Historical Proof",
        "",
        f"- rows: {checkpoint['row_count']}",
        f"- pass: {checkpoint['pass_count']}",
        f"- fail: {checkpoint['fail_count']}",
        f"- proof scope: {scope_status}",
        f"- proof scope notes: {scope_note}",
        "",
        "| Pipeline | Date | Status | Note | Manifest |",
        "|:--|:--|:--|:--|:--|",
    ]
    for row in rows:
        lines.append(
            f"| {row['pipeline']} | {row['date']} | {row['status']} | "
            f"{_md(row['validation_note'])} | {_md(row['manifest_path'])} |"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, json_path, md_path


def _md(value: Any) -> str:
    return str(value).replace("|", "\\|")[:500]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root).expanduser().resolve()
    proof_dir = Path(args.out_dir).expanduser().resolve()
    from_date = _parse_date(args.from_date) or dt.date(2026, 1, 1)
    to_date = _parse_date(args.as_of) or _parse_date(args.to_date) or dt.date(2026, 12, 31)
    pipelines = PIPELINES if args.pipeline == "both" else (args.pipeline,)
    folders = selected_source_dates(root, from_date, to_date, args.max_dates)
    rows: list[dict[str, Any]] = []
    for folder in folders:
        day = _date_text(folder)
        for pipeline in pipelines:
            mpath = manifest_path(proof_dir, pipeline, day)
            rpath = report_path(proof_dir, pipeline, day)
            command: list[str] = []
            returncode = 0
            if mpath.exists() and not args.force:
                note_status, note = validate_manifest(mpath, pipeline)
                status = note_status
            else:
                returncode, command = run_command(args, pipeline, folder, proof_dir)
                status, note = validate_manifest(mpath, pipeline)
                if returncode != 0 and status == "PASS":
                    status = "FAIL"
                    note = f"returncode={returncode}"
            rows.append(
                {
                    "pipeline": pipeline,
                    "date": day,
                    "status": status,
                    "returncode": returncode,
                    "proof_scope_status": proof_scope(args)[0],
                    "proof_scope_note": proof_scope(args)[1],
                    "manifest_path": str(mpath),
                    "report_path": str(rpath),
                    "validation_note": note,
                    "command": " ".join(command),
                }
            )
            write_summary(proof_dir, rows, args)
    csv_path, json_path, md_path = write_summary(proof_dir, rows, args)
    fail_count = sum(1 for row in rows if row.get("status") == "FAIL")
    print(md_path)
    print(f"rows={len(rows)} fail_count={fail_count}")
    print(f"csv={csv_path}")
    print(f"checkpoint={json_path}")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
