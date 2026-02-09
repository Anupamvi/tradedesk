#!/usr/bin/env python3
"""
Post-process Unusual Whales dashboard captures for a single day folder.

What it does:
- De-duplicates repeated CSV artifacts from prior reruns.
- Keeps the "best" file in each duplicate group (rows, then size, then recency).
- Builds a single markdown report with per-route signal coverage and CSV previews.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from uw_dashboard_capture import ROUTES  # same directory import when run via python scripts/...
except Exception:
    ROUTES = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize a UW capture day folder into a clean markdown report.")
    parser.add_argument(
        "--trade-date",
        default=dt.date.today().isoformat(),
        help="Day folder to process (YYYY-MM-DD). Default: today.",
    )
    parser.add_argument(
        "--base-dir",
        default=".",
        help="Root folder that contains daily YYYY-MM-DD folders.",
    )
    parser.add_argument(
        "--delete-duplicates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Delete duplicate CSV rerun files after selecting best candidate in each group.",
    )
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=8,
        help="Data rows to preview per route (header + N rows in markdown report).",
    )
    parser.add_argument(
        "--max-preview-routes",
        type=int,
        default=10,
        help="Max routes with CSV preview blocks in markdown report.",
    )
    return parser.parse_args()


def parse_trade_date(text: str) -> dt.date:
    try:
        return dt.datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError as exc:
        raise SystemExit(f"--trade-date must be YYYY-MM-DD, got: {text}") from exc


def summarize_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.2f} MB"


def estimate_csv_rows(path: Path) -> int:
    try:
        with path.open("r", newline="", encoding="utf-8", errors="replace") as fh:
            reader = csv.reader(fh)
            row_count = sum(1 for _ in reader)
        return max(0, row_count - 1)
    except Exception:
        return 0


def normalize_dup_group_key(path: Path) -> str:
    # Collapse rerun suffix "...-YYYY-MM-DD-<N>.csv" into "...-YYYY-MM-DD.csv".
    stem = path.stem
    m = re.match(r"^(.*-\d{4}-\d{2}-\d{2})(?:-\d+)?$", stem)
    if m:
        stem = m.group(1)
    else:
        # Fallback for non-date filenames.
        stem = re.sub(r"-\d+$", "", stem)
    return f"{stem}{path.suffix.lower()}"


def is_excluded_csv(path: Path) -> bool:
    name = path.name
    if name.startswith("uw_capture_manifest_"):
        return True
    if name.startswith("uw_capture_final_report_"):
        return True
    return False


def rank_file(path: Path) -> Tuple[int, int, int]:
    rows = estimate_csv_rows(path)
    stat = path.stat()
    return rows, int(stat.st_size), int(stat.st_mtime_ns)


def dedupe_csv_files(day_dir: Path, delete_duplicates: bool) -> Tuple[List[dict], List[Path]]:
    csv_files = [p for p in day_dir.glob("*.csv") if p.is_file() and not is_excluded_csv(p)]
    groups: Dict[str, List[Path]] = {}
    for p in csv_files:
        groups.setdefault(normalize_dup_group_key(p), []).append(p)

    decisions: List[dict] = []
    removed: List[Path] = []

    for group_key, files in sorted(groups.items()):
        if len(files) <= 1:
            continue
        ranked = sorted(files, key=rank_file, reverse=True)
        keep = ranked[0]
        to_remove = ranked[1:]
        decisions.append(
            {
                "group": group_key,
                "keep": keep,
                "remove": to_remove,
                "keep_rows": rank_file(keep)[0],
            }
        )
        if delete_duplicates:
            for p in to_remove:
                try:
                    p.unlink()
                    removed.append(p)
                except Exception:
                    pass
    return decisions, removed


def prefix_to_route_map() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for route_key, cfg in ROUTES.items():
        prefix = str(cfg.get("prefix", "")).strip()
        if prefix:
            mapping[prefix] = route_key
    return mapping


def detect_route_from_filename(name: str, prefixes_desc: Sequence[str], p2r: Dict[str, str]) -> str:
    for prefix in prefixes_desc:
        if name.startswith(prefix + "-"):
            return p2r[prefix]
    return "unknown"


def csv_preview(path: Path, max_data_rows: int) -> str:
    rows: List[List[str]] = []
    limit = max(1, max_data_rows + 1)  # include header
    try:
        with path.open("r", newline="", encoding="utf-8", errors="replace") as fh:
            reader = csv.reader(fh)
            for i, row in enumerate(reader):
                rows.append(row)
                if i + 1 >= limit:
                    break
    except Exception:
        return ""
    if not rows:
        return ""
    sio = io.StringIO()
    writer = csv.writer(sio)
    writer.writerows(rows)
    return sio.getvalue().strip()


def signal_label(best_rows: int, json_count: int, snapshot_count: int) -> str:
    if best_rows >= 100 or json_count >= 10:
        return "high"
    if best_rows >= 20 or json_count >= 2 or snapshot_count >= 3:
        return "medium"
    return "low"


def build_route_inventory(day_dir: Path) -> Dict[str, dict]:
    p2r = prefix_to_route_map()
    prefixes_desc = sorted(p2r.keys(), key=len, reverse=True)
    stats: Dict[str, dict] = {}

    def ensure(route: str) -> dict:
        if route not in stats:
            stats[route] = {
                "csv": [],
                "json_count": 0,
                "snapshot_count": 0,
            }
        return stats[route]

    for p in sorted(day_dir.glob("*.csv")):
        if not p.is_file() or is_excluded_csv(p):
            continue
        route = detect_route_from_filename(p.name, prefixes_desc, p2r)
        bucket = ensure(route)
        bucket["csv"].append(
            {
                "path": p,
                "rows": estimate_csv_rows(p),
                "bytes": int(p.stat().st_size),
                "mtime_ns": int(p.stat().st_mtime_ns),
            }
        )

    net_dir = day_dir / "_network_json"
    if net_dir.exists():
        for route_dir in net_dir.iterdir():
            if not route_dir.is_dir():
                continue
            bucket = ensure(route_dir.name)
            bucket["json_count"] = len([p for p in route_dir.iterdir() if p.is_file()])

    snap_dir = day_dir / "_snapshots"
    if snap_dir.exists():
        for route_dir in snap_dir.iterdir():
            if not route_dir.is_dir():
                continue
            bucket = ensure(route_dir.name)
            bucket["snapshot_count"] = len([p for p in route_dir.iterdir() if p.is_file()])

    return stats


def choose_best_csv(items: Sequence[dict]) -> Optional[dict]:
    if not items:
        return None
    return max(items, key=lambda x: (int(x["rows"]), int(x["bytes"]), int(x["mtime_ns"])))


def write_markdown_report(
    day_dir: Path,
    trade_date: dt.date,
    inventory: Dict[str, dict],
    dedupe_decisions: Sequence[dict],
    removed_files: Sequence[Path],
    preview_rows: int,
    max_preview_routes: int,
) -> Path:
    report_path = day_dir / f"uw_capture_final_report_{trade_date.isoformat()}.md"
    now = dt.datetime.now().isoformat(timespec="seconds")
    lines: List[str] = []

    lines.append(f"# UW Capture Final Report - {trade_date.isoformat()}")
    lines.append("")
    lines.append(f"- Generated: {now}")
    lines.append(f"- Folder: `{day_dir}`")
    lines.append("")

    lines.append("## Cleanup")
    lines.append(f"- Duplicate CSV groups found: {len(dedupe_decisions)}")
    lines.append(f"- Duplicate CSV files removed: {len(removed_files)}")
    if dedupe_decisions:
        for d in dedupe_decisions:
            keep_path = Path(d["keep"]).name
            removed_n = len(d["remove"])
            keep_rows = int(d["keep_rows"])
            lines.append(f"- Kept `{keep_path}` ({keep_rows} rows); removed {removed_n} duplicate file(s).")
    lines.append("")

    lines.append("## Route Coverage")
    lines.append("")
    lines.append("| Route | Best CSV rows | Best CSV | JSON files | Snapshot files | Signal |")
    lines.append("|---|---:|---|---:|---:|---|")

    ranked_routes: List[Tuple[str, int, int, int]] = []
    for route in sorted(inventory.keys()):
        bucket = inventory[route]
        best = choose_best_csv(bucket["csv"])
        best_rows = int(best["rows"]) if best else 0
        best_name = best["path"].name if best else "-"
        json_count = int(bucket["json_count"])
        snapshot_count = int(bucket["snapshot_count"])
        signal = signal_label(best_rows, json_count, snapshot_count)
        lines.append(
            f"| {route} | {best_rows} | `{best_name}` | {json_count} | {snapshot_count} | {signal} |"
        )
        ranked_routes.append((route, best_rows, json_count, snapshot_count))
    lines.append("")

    lines.append("## Best CSV Previews")
    lines.append("")
    ranked_routes.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
    shown = 0
    for route, _, _, _ in ranked_routes:
        if shown >= max_preview_routes:
            break
        best = choose_best_csv(inventory[route]["csv"])
        if not best:
            continue
        preview = csv_preview(Path(best["path"]), preview_rows)
        lines.append(f"### {route}")
        lines.append(
            f"- File: `{Path(best['path']).name}` ({int(best['rows'])} rows, {summarize_bytes(int(best['bytes']))})"
        )
        if preview:
            lines.append("")
            lines.append("```csv")
            lines.append(preview)
            lines.append("```")
        lines.append("")
        shown += 1

    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return report_path


def main() -> int:
    args = parse_args()
    trade_date = parse_trade_date(args.trade_date)
    base_dir = Path(args.base_dir).resolve()
    day_dir = base_dir / trade_date.isoformat()
    if not day_dir.exists():
        print(f"Day folder does not exist: {day_dir}")
        return 2

    decisions, removed_files = dedupe_csv_files(day_dir=day_dir, delete_duplicates=bool(args.delete_duplicates))
    inventory = build_route_inventory(day_dir=day_dir)

    report_path = write_markdown_report(
        day_dir=day_dir,
        trade_date=trade_date,
        inventory=inventory,
        dedupe_decisions=decisions,
        removed_files=removed_files,
        preview_rows=int(args.preview_rows),
        max_preview_routes=int(args.max_preview_routes),
    )

    print(f"Processed day folder: {day_dir}")
    print(f"Duplicate groups: {len(decisions)}")
    print(f"Removed duplicate CSV files: {len(removed_files)}")
    print(f"Markdown report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
