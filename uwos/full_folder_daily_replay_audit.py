#!/usr/bin/env python3
"""Full-folder historical replay audit for the daily options pipeline.

This is intentionally separate from trend-analysis. It replays dated daily-input
folders through run_mode_a_two_stage.py, then backtests the emitted daily trade
reports and checks for safety exceptions in the machine-readable decision books.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

try:
    from uwos.historical_daily_report_backtest import rows_from_report, run_report_backtest, summarize
except Exception:  # pragma: no cover - allows script execution from uwos cwd with PYTHONPATH issues
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from uwos.historical_daily_report_backtest import rows_from_report, run_report_backtest, summarize

REQUIRED_FAMILIES = {
    "dp": ["dp-eod-report-"],
    "bot_eod": ["bot-eod-report-"],
    "stock": ["stock-screener-"],
    "hot": ["hot-chain-", "hot-chains-"],
    "chain": ["chain-oi-changes-"],
}
APPROVED_BOOKS = {"Core", "Tactical", "Scout"}
DATE_RE = re.compile(r"20\d\d-\d\d-\d\d")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay dated daily folders through the daily pipeline and summarize realized quality.")
    p.add_argument("--root", type=Path, default=Path("/Users/anuppamvi/uw_root/tradedesk"))
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date", default=None)
    p.add_argument("--valuation-date", default=None, help="Date used to mark open/not-expired trades. Defaults to today.")
    p.add_argument("--max-workers", type=int, default=4)
    p.add_argument("--top-trades", type=int, default=20)
    p.add_argument("--reuse-existing", action="store_true", help="Skip replay and only aggregate an existing out-root.")
    p.add_argument("--auto-collect-uw-gex", action="store_true", help="Allow browser/UW GEX collection. Default disables it for historical replay speed/reproducibility.")
    return p.parse_args()


def as_date(s: Optional[str]) -> Optional[dt.date]:
    if not s:
        return None
    return dt.date.fromisoformat(s)


def is_date_folder(path: Path) -> bool:
    return path.is_dir() and bool(DATE_RE.fullmatch(path.name))


def has_family(names: Iterable[str], prefixes: Iterable[str]) -> bool:
    return any(any(name.startswith(prefix) for prefix in prefixes) for name in names)


def inventory(root: Path, start: Optional[dt.date], end: Optional[dt.date]) -> tuple[List[Path], List[dict]]:
    folders: List[Path] = []
    incomplete: List[dict] = []
    for folder in sorted(p for p in root.iterdir() if is_date_folder(p)):
        d = dt.date.fromisoformat(folder.name)
        if start and d < start:
            continue
        if end and d > end:
            continue
        names = [p.name for p in folder.iterdir() if p.is_file()]
        missing = [family for family, prefixes in REQUIRED_FAMILIES.items() if not has_family(names, prefixes)]
        if missing:
            incomplete.append({"date": folder.name, "missing": missing})
        else:
            folders.append(folder)
    return folders, incomplete


def run_one(root: Path, config: Path, out_root: Path, top_trades: int, auto_collect_uw_gex: bool, folder: Path) -> dict:
    scan_date = folder.name
    odir = out_root / scan_date
    odir.mkdir(parents=True, exist_ok=True)
    report = odir / f"anu-expert-trade-table-{scan_date}.md"
    cmd = [
        sys.executable,
        "-m",
        "uwos.run_mode_a_two_stage",
        "--historical-replay",
        "--base-dir",
        str(folder),
        "--config",
        str(config),
        "--out-dir",
        str(odir),
        "--top-trades",
        str(top_trades),
        "--output",
        str(report),
    ]
    if not auto_collect_uw_gex:
        cmd.insert(4, "--no-auto-collect-uw-gex")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parents[1]), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=900)
    (odir / "run.log").write_text(proc.stdout)
    result = {
        "date": scan_date,
        "returncode": proc.returncode,
        "seconds": round(time.time() - t0, 2),
        "report": str(report),
        "out_dir": str(odir),
    }
    m = re.search(r"Approved trades:\s*(\d+)\s*/\s*(\d+)", proc.stdout)
    if m:
        result["approved"] = int(m.group(1))
        result["candidates"] = int(m.group(2))
    m = re.search(r"Core=(\d+), Tactical=(\d+), Scout=(\d+), Watch=(\d+)", proc.stdout)
    if m:
        result["core"] = int(m.group(1))
        result["tactical"] = int(m.group(2))
        result["scout"] = int(m.group(3))
        result["watch"] = int(m.group(4))
    if proc.returncode != 0:
        result["tail"] = proc.stdout[-4000:]
    return result


def write_manifest(out_root: Path, manifest: dict) -> None:
    (out_root / "batch_results.json").write_text(json.dumps(manifest, indent=2, default=str))


def run_batch(args: argparse.Namespace, config: Path) -> pd.DataFrame:
    start = as_date(args.start_date)
    end = as_date(args.end_date)
    folders, incomplete = inventory(args.root, start, end)
    args.out_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "started_at": dt.datetime.now().isoformat(timespec="seconds"),
        "root": str(args.root),
        "config": str(config),
        "out_root": str(args.out_root),
        "folders": [f.name for f in folders],
        "incomplete": incomplete,
        "auto_collect_uw_gex": bool(args.auto_collect_uw_gex),
        "results": [],
    }
    write_manifest(args.out_root, manifest)
    print(f"complete_folders={len(folders)} incomplete={len(incomplete)} out={args.out_root}", flush=True)
    results: List[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as ex:
        futures = {ex.submit(run_one, args.root, config, args.out_root, args.top_trades, args.auto_collect_uw_gex, f): f for f in folders}
        for idx, fut in enumerate(as_completed(futures), 1):
            result = fut.result()
            results.append(result)
            manifest["results"] = sorted(results, key=lambda x: x["date"])
            write_manifest(args.out_root, manifest)
            print(
                f"[{idx}/{len(folders)}] {result['date']} rc={result['returncode']} "
                f"approved={result.get('approved')} core={result.get('core')} tactical={result.get('tactical')} "
                f"scout={result.get('scout')} watch={result.get('watch')} sec={result['seconds']}",
                flush=True,
            )
    manifest["finished_at"] = dt.datetime.now().isoformat(timespec="seconds")
    manifest["results"] = sorted(results, key=lambda x: x["date"])
    write_manifest(args.out_root, manifest)
    return pd.DataFrame(manifest["results"])


def metric_block(df: pd.DataFrame) -> Dict[str, object]:
    if df.empty:
        return {"rows": 0, "completed": 0, "open": 0, "wins": 0, "losses": 0, "win_rate": None, "net_pnl": 0.0, "gross_profit": 0.0, "gross_loss": 0.0, "profit_factor": None}
    completed = df[df["status"].eq("completed")]
    wins = int((completed["pnl"] > 0).sum())
    losses = int((completed["pnl"] <= 0).sum())
    gross_profit = float(completed.loc[completed["pnl"] > 0, "pnl"].sum())
    gross_loss = float(-completed.loc[completed["pnl"] < 0, "pnl"].sum())
    return {
        "rows": int(len(df)),
        "completed": int(len(completed)),
        "open": int((df["status"] != "completed").sum()),
        "wins": wins,
        "losses": losses,
        "win_rate": wins / len(completed) if len(completed) else None,
        "net_pnl": float(completed["pnl"].sum()),
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": gross_profit / gross_loss if gross_loss else (math.inf if gross_profit else None),
    }


def collect_decision_books(out_root: Path) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for path in sorted(out_root.glob("20??-??-??/trade_decision_book_*.csv")):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty:
            continue
        df["scan_date"] = path.parent.name
        df["decision_book_path"] = str(path)
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def aggregate(args: argparse.Namespace, valuation_date: dt.date) -> dict:
    out_root = args.out_root
    summary_dir = out_root / "realized_backtest"
    summary_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "batch_results.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"results": []}
    day_counts = pd.DataFrame(manifest.get("results", []))
    if not day_counts.empty:
        day_counts.to_csv(out_root / "daily_trade_counts.csv", index=False)

    decision_books = collect_decision_books(out_root)
    if not decision_books.empty:
        decision_books.to_csv(out_root / "replay_decision_books_all.csv", index=False)

    reports = sorted(out_root.glob("20??-??-??/anu-expert-trade-table-*.md"))
    setups: List[dict] = []
    for report in reports:
        setups.extend(rows_from_report(report, include_watch=False))
    setup_df = pd.DataFrame(setups)
    results = run_report_backtest(setup_df, args.root, valuation_date) if not setup_df.empty else pd.DataFrame()
    if not results.empty:
        results.to_csv(summary_dir / "trade_level_results.csv", index=False)
    summary = summarize(results) if not results.empty else {}
    (summary_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    joined = results.copy()
    if not joined.empty and not decision_books.empty:
        approved = decision_books[decision_books["execution_book"].astype(str).isin(APPROVED_BOOKS)].copy()
        approved["signal_date"] = approved["scan_date"].astype(str)
        approved["entry_net_join"] = pd.to_numeric(approved.get("live_net_bid_ask", approved.get("net")), errors="coerce").round(2)
        joined["entry_net_join"] = pd.to_numeric(joined["entry_net"], errors="coerce").round(2)
        for df in (approved, joined):
            for col in ["signal_date", "ticker", "strategy", "expiry"]:
                df[col] = df[col].astype(str)
        keep = [c for c in ["signal_date", "ticker", "strategy", "expiry", "entry_net_join", "execution_book", "live_status", "is_final_live_valid", "gate_pass_effective", "hard_blockers", "quality_blockers", "approval_blockers"] if c in approved.columns]
        approved = approved[keep].drop_duplicates(subset=["signal_date", "ticker", "strategy", "expiry", "entry_net_join"], keep="first")
        joined = joined.merge(approved, on=["signal_date", "ticker", "strategy", "expiry", "entry_net_join"], how="left", suffixes=("", "_decision"))
        joined.to_csv(summary_dir / "trade_level_results_joined_decision_book.csv", index=False)

    book_col = "execution_book_decision" if "execution_book_decision" in joined.columns else "execution_book"
    book_metrics = {str(k): metric_block(v) for k, v in joined.groupby(joined[book_col].fillna("UNMATCHED"))} if book_col in joined.columns and not joined.empty else {}
    strategy_metrics = {str(k): metric_block(v) for k, v in joined.groupby(joined["strategy"].fillna("UNKNOWN"))} if "strategy" in joined.columns and not joined.empty else {}
    month_metrics = {}
    if "signal_date" in joined.columns and not joined.empty:
        by_month = joined.copy()
        by_month["signal_month"] = by_month["signal_date"].astype(str).str[:7]
        month_metrics = {str(k): metric_block(v) for k, v in by_month.groupby("signal_month")}

    invalid_approved = pd.DataFrame()
    if not decision_books.empty and "execution_book" in decision_books.columns:
        approved = decision_books[decision_books["execution_book"].astype(str).isin(APPROVED_BOOKS)].copy()
        if not approved.empty:
            live_effective = approved.get("is_final_live_valid", False).fillna(False).astype(bool) | (
                approved.get("live_status", "").astype(str).eq("fails_live_entry_gate")
                & approved.get("gate_pass_effective", False).fillna(False).astype(bool)
            )
            hard = approved.get("hard_blockers", "").fillna("").astype(str)
            contra = hard.str.contains("contract_flow_contra|stage1_contract_flow_contra|bull_call_contract_flow_not_confirmed:contra", regex=True)
            invalid_approved = approved[(~live_effective) | contra]
            invalid_approved.to_csv(out_root / "approved_rows_safety_exceptions.csv", index=False)

    blocker_counts = []
    if not decision_books.empty and "execution_book" in decision_books.columns:
        watch = decision_books[decision_books["execution_book"].astype(str).eq("Watch")]
        counter: Counter[str] = Counter()
        for field in ["hard_blockers", "quality_blockers", "approval_blockers", "stage1_blockers", "notes"]:
            if field not in watch.columns:
                continue
            for val in watch[field].fillna("").astype(str):
                for token in re.split(r"[;|,]", val):
                    token = token.strip()
                    if token and token.lower() != "nan":
                        counter[token[:180]] += 1
        blocker_counts = counter.most_common(40)

    payload = {
        "summary": summary,
        "book_metrics": book_metrics,
        "strategy_metrics": strategy_metrics,
        "month_metrics": month_metrics,
        "invalid_approved_count": int(len(invalid_approved)),
        "blocker_counts": blocker_counts,
        "files": {
            "daily_counts": str(out_root / "daily_trade_counts.csv"),
            "all_decision_books": str(out_root / "replay_decision_books_all.csv"),
            "trade_results": str(summary_dir / "trade_level_results_joined_decision_book.csv"),
            "summary_json": str(summary_dir / "summary.json"),
            "safety_exceptions": str(out_root / "approved_rows_safety_exceptions.csv"),
        },
    }
    (out_root / "full_folder_replay_audit_summary.json").write_text(json.dumps(payload, indent=2, default=str))
    write_markdown(out_root, manifest, day_counts, payload)
    return payload


def fmt_pct(value: object) -> str:
    return "n/a" if value is None or pd.isna(value) else f"{float(value):.1%}"


def fmt_pf(value: object) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    if value == math.inf:
        return "inf"
    return f"{float(value):.2f}"


def write_markdown(out_root: Path, manifest: dict, day_counts: pd.DataFrame, payload: dict) -> None:
    summary = payload.get("summary", {})
    md: List[str] = ["# Full-folder daily replay audit", ""]
    md.append("This audit replays dated daily folders through the daily pipeline only. It does not merge trend-analysis outputs into the daily pipeline.")
    md.append("")
    if not day_counts.empty:
        approved = day_counts.get("approved", pd.Series(dtype=float)).fillna(0).astype(int)
        md.append("## Replay coverage")
        md.append("")
        md.append(f"- Complete folders replayed: {len(manifest.get('folders', []))}")
        md.append(f"- Incomplete folders excluded: {len(manifest.get('incomplete', []))}")
        md.append(f"- Run failures: {int((day_counts.get('returncode', 0) != 0).sum())}")
        md.append(f"- Approved rows: {int(approved.sum())}")
        md.append(f"- Trade days: {int((approved > 0).sum())}")
        md.append(f"- Skip days: {int((approved == 0).sum())}")
        def _sum_count_col(col: str) -> int:
            if col not in day_counts.columns:
                return 0
            return int(pd.to_numeric(day_counts[col], errors="coerce").fillna(0).sum())
        md.append(f"- Core/Tactical/Scout rows: {_sum_count_col('core')}/{_sum_count_col('tactical')}/{_sum_count_col('scout')}")
        md.append("")
    if summary:
        md.append("## Realized quality")
        md.append("")
        md.append(f"- Completed/open rows: {summary.get('completed_trades')}/{summary.get('open_or_skipped_trades')}")
        md.append(f"- Win rate: {fmt_pct(summary.get('win_rate'))}")
        md.append(f"- Profit factor: {fmt_pf(summary.get('profit_factor'))}")
        md.append(f"- Net P/L: ${float(summary.get('net_pnl', 0)):,.0f}")
        md.append(f"- Good/bad/flat days: {summary.get('good_days')}/{summary.get('bad_days')}/{summary.get('flat_days')}")
        md.append("")
    for title, key in [("By execution book", "book_metrics"), ("By strategy", "strategy_metrics"), ("By month", "month_metrics")]:
        metrics = payload.get(key, {})
        if not metrics:
            continue
        md.append(f"## {title}")
        md.append("")
        for name, block in sorted(metrics.items()):
            md.append(
                f"- {name}: rows {block['rows']}, completed {block['completed']}, wins/losses {block['wins']}/{block['losses']}, "
                f"win {fmt_pct(block['win_rate'])}, PF {fmt_pf(block['profit_factor'])}, net ${float(block['net_pnl']):,.0f}"
            )
        md.append("")
    md.append("## Safety exception check")
    md.append("")
    md.append(f"- Approved rows with non-effective live gate or contra-flow hard blockers: {payload.get('invalid_approved_count', 0)}")
    md.append("")
    if payload.get("blocker_counts"):
        md.append("## Top Watch blockers")
        md.append("")
        for token, count in payload["blocker_counts"][:20]:
            md.append(f"- {count}x `{token}`")
        md.append("")
    md.append("## Output files")
    md.append("")
    for label, path in payload.get("files", {}).items():
        md.append(f"- {label}: `{path}`")
    md.append("")
    (out_root / "FULL_FOLDER_DAILY_REPLAY_AUDIT.md").write_text("\n".join(md))


def main() -> int:
    args = parse_args()
    config = args.config or (args.root / "uwos" / "rulebook_config_goal_holistic_claude.yaml")
    valuation_date = as_date(args.valuation_date) or dt.date.today()
    if not args.reuse_existing:
        run_batch(args, config)
    payload = aggregate(args, valuation_date)
    print(json.dumps({
        "summary": payload.get("summary", {}),
        "book_metrics": payload.get("book_metrics", {}),
        "invalid_approved_count": payload.get("invalid_approved_count", 0),
        "audit_md": str(args.out_root / "FULL_FOLDER_DAILY_REPLAY_AUDIT.md"),
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
