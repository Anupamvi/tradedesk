#!/usr/bin/env python3
"""Determinism audit for the daily options pipeline.

Runs each complete dated folder twice in historical replay mode and fails if any
canonical output artifact differs. This is deliberately stricter than the
full-folder replay audit: it checks reproducibility, not trade quality.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

try:
    from uwos.full_folder_daily_replay_audit import as_date, inventory, parse_approved_counts
except Exception:  # pragma: no cover - supports direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from uwos.full_folder_daily_replay_audit import as_date, inventory, parse_approved_counts


DEFAULT_ARTIFACTS = [
    "anu-expert-trade-table-{date}.md",
    "shortlist_trades_{date}_mode_a.csv",
    "setup_likelihood_{date}.csv",
    "live_trade_table_{date}_final.csv",
    "trade_decision_book_all_{date}.csv",
    "trade_decision_book_{date}.csv",
    "planned_trade_journal_{date}.csv",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run dated daily folders twice and fail if canonical outputs drift."
    )
    p.add_argument("--root", type=Path, default=Path("/Users/anuppamvi/uw_root/tradedesk"))
    p.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "rulebook_config_goal_holistic_claude.yaml",
    )
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date", default=None)
    p.add_argument(
        "--date",
        action="append",
        default=[],
        help="Specific date to audit. Can be passed more than once. Overrides start/end filtering.",
    )
    p.add_argument("--top-trades", type=int, default=20)
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact_names(date_text: str) -> list[str]:
    return [name.format(date=date_text) for name in DEFAULT_ARTIFACTS]


def compare_artifacts(run_a: Path, run_b: Path, date_text: str) -> list[dict]:
    rows: list[dict] = []
    for name in artifact_names(date_text):
        path_a = run_a / name
        path_b = run_b / name
        exists_a = path_a.exists()
        exists_b = path_b.exists()
        row = {
            "artifact": name,
            "path_a": str(path_a),
            "path_b": str(path_b),
            "exists_a": exists_a,
            "exists_b": exists_b,
            "same": False,
            "sha256_a": "",
            "sha256_b": "",
        }
        if exists_a:
            row["sha256_a"] = sha256_file(path_a)
        if exists_b:
            row["sha256_b"] = sha256_file(path_b)
        row["same"] = bool(exists_a and exists_b and row["sha256_a"] == row["sha256_b"])
        rows.append(row)
    return rows


def select_folders(root: Path, dates: Iterable[str], start: Optional[dt.date], end: Optional[dt.date]) -> tuple[list[Path], list[dict]]:
    selected_dates = {str(d).strip() for d in dates if str(d).strip()}
    if selected_dates:
        folders: list[Path] = []
        incomplete: list[dict] = []
        all_folders, all_incomplete = inventory(root, None, None)
        complete_by_date = {path.name: path for path in all_folders}
        incomplete_by_date = {row["date"]: row for row in all_incomplete}
        for date_text in sorted(selected_dates):
            if date_text in complete_by_date:
                folders.append(complete_by_date[date_text])
            else:
                incomplete.append(incomplete_by_date.get(date_text, {"date": date_text, "missing": ["folder_or_required_inputs"]}))
        return folders, incomplete
    return inventory(root, start, end)


def run_daily(folder: Path, config: Path, out_dir: Path, top_trades: int) -> dict:
    date_text = folder.name
    out_dir.mkdir(parents=True, exist_ok=True)
    output_md = out_dir / f"anu-expert-trade-table-{date_text}.md"
    cmd = [
        sys.executable,
        "-m",
        "uwos.run_mode_a_two_stage",
        "--historical-replay",
        "--no-auto-collect-uw-gex",
        "--base-dir",
        str(folder),
        "--config",
        str(config),
        "--out-dir",
        str(out_dir),
        "--top-trades",
        str(top_trades),
        "--output",
        str(output_md),
    ]
    t0 = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(Path(__file__).resolve().parents[1]),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=900,
    )
    (out_dir / "run.log").write_text(proc.stdout, encoding="utf-8")
    approved, candidates = parse_approved_counts(proc.stdout)
    result = {
        "returncode": int(proc.returncode),
        "seconds": round(time.time() - t0, 2),
        "out_dir": str(out_dir),
        "approved": approved,
        "candidates": candidates,
    }
    if proc.returncode != 0:
        result["tail"] = proc.stdout[-4000:]
    return result


def write_markdown(out_root: Path, payload: dict) -> None:
    lines = [
        "# Daily Pipeline Reproducibility Audit",
        "",
        f"Root: `{payload['root']}`",
        f"Config: `{payload['config']}`",
        "",
        "| Date | Status | Approved A/B | Drifted Artifacts |",
        "|---|---:|---:|---|",
    ]
    for row in payload["results"]:
        drifted = [a["artifact"] for a in row["artifacts"] if not a["same"]]
        approved = f"{row['run_a'].get('approved')}/{row['run_b'].get('approved')}"
        lines.append(
            f"| {row['date']} | {row['status']} | {approved} | "
            f"{', '.join(drifted) if drifted else 'none'} |"
        )
    if payload.get("incomplete"):
        lines.extend(["", "## Incomplete Folders", ""])
        for row in payload["incomplete"]:
            lines.append(f"- {row['date']}: missing {', '.join(row.get('missing', []))}")
    (out_root / "DAILY_PIPELINE_REPRO_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    start = as_date(args.start_date)
    end = as_date(args.end_date)
    folders, incomplete = select_folders(args.root, args.date, start, end)
    args.out_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "root": str(args.root.resolve()),
        "config": str(args.config.resolve()),
        "out_root": str(args.out_root.resolve()),
        "started_at": dt.datetime.now().isoformat(timespec="seconds"),
        "folders": [folder.name for folder in folders],
        "incomplete": incomplete,
        "results": [],
    }
    any_failure = False
    print(f"repro_folders={len(folders)} incomplete={len(incomplete)} out={args.out_root}", flush=True)
    for folder in folders:
        date_text = folder.name
        date_root = args.out_root / date_text
        run_a_dir = date_root / "run_a"
        run_b_dir = date_root / "run_b"
        run_a = run_daily(folder, args.config, run_a_dir, args.top_trades)
        run_b = run_daily(folder, args.config, run_b_dir, args.top_trades)
        artifacts = compare_artifacts(run_a_dir, run_b_dir, date_text)
        drifted = [row for row in artifacts if not row["same"]]
        failed = run_a["returncode"] != 0 or run_b["returncode"] != 0 or bool(drifted)
        any_failure = any_failure or failed
        result = {
            "date": date_text,
            "status": "FAIL" if failed else "PASS",
            "run_a": run_a,
            "run_b": run_b,
            "artifacts": artifacts,
        }
        payload["results"].append(result)
        (args.out_root / "repro_audit_summary.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(
            f"{date_text} {result['status']} approved={run_a.get('approved')}/{run_b.get('approved')} "
            f"drifted={len(drifted)}",
            flush=True,
        )
    payload["finished_at"] = dt.datetime.now().isoformat(timespec="seconds")
    (args.out_root / "repro_audit_summary.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    write_markdown(args.out_root, payload)
    return 1 if any_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
