#!/usr/bin/env python3
"""
End-to-end UW -> Deep Research packet runner.

Runs, in order:
1) uw_dashboard_capture.py
2) uw_capture_finalize.py
3) build_growth_portfolio_candidates.py
4) build_deep_research_packet.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path
from typing import List


DEFAULT_ROUTES = [
    "stock-screener",
    "analysts",
    "insiders-trades",
    "institutions",
    "news-feed",
    "earnings",
    "flow-sectors",
    "market-statistics",
    "sec-filings",
    "correlations",
    "smart-money-live",
    "market-maps",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full UW capture + packet pipeline.")
    parser.add_argument("--trade-date", default=dt.date.today().isoformat(), help="Day folder (YYYY-MM-DD).")
    parser.add_argument("--base-dir", default=".", help="Root folder containing day folders.")
    parser.add_argument("--profile-dir", default="tokens/uw_playwright_profile_v2", help="Playwright profile dir.")
    parser.add_argument("--browser-channel", choices=["chromium", "chrome", "msedge"], default="msedge")
    parser.add_argument("--routes", nargs="+", default=DEFAULT_ROUTES)
    parser.add_argument("--wait-seconds", type=int, default=55)
    parser.add_argument("--scrape-scroll-cycles", type=int, default=20)
    parser.add_argument("--top-n", type=int, default=30, help="Top N for growth candidates markdown.")
    parser.add_argument("--packet-top-n", type=int, default=30, help="Top N rows included in deep research packet snapshot.")
    parser.add_argument("--stock-lookback-days", type=int, default=7, help="Fallback lookback for stock-screener.")
    parser.add_argument("--route-lookback-days", type=int, default=14, help="Fallback lookback for route CSVs in packet.")
    parser.add_argument("--artifact-lookback-days", type=int, default=14, help="Fallback lookback for packet artifacts.")
    parser.add_argument("--skip-capture", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--disable-automation-flags",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Default false; set true only if your login works with automation flags.",
    )
    return parser.parse_args()


def run_cmd(cmd: List[str], cwd: Path) -> int:
    print("")
    print("> " + " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(cwd))
    return int(proc.returncode)


def main() -> int:
    args = parse_args()
    root = Path(args.base_dir).resolve()

    py = sys.executable

    if not args.skip_capture:
        capture_cmd = [
            py,
            "scripts/uw_dashboard_capture.py",
            "--trade-date",
            str(args.trade_date),
            "--base-dir",
            str(root),
            "--routes",
            *list(args.routes),
            "--profile-dir",
            str(args.profile_dir),
            "--browser-channel",
            str(args.browser_channel),
            "--wait-seconds",
            str(int(args.wait_seconds)),
            "--scrape-scroll-cycles",
            str(int(args.scrape_scroll_cycles)),
            "--also-scrape",
        ]
        capture_cmd.append("--headless" if args.headless else "--no-headless")
        capture_cmd.append("--disable-automation-flags" if args.disable_automation_flags else "--no-disable-automation-flags")
        rc = run_cmd(capture_cmd, cwd=root)
        if rc != 0:
            return rc

    finalize_cmd = [
        py,
        "scripts/uw_capture_finalize.py",
        "--trade-date",
        str(args.trade_date),
        "--base-dir",
        str(root),
        "--delete-duplicates",
    ]
    rc = run_cmd(finalize_cmd, cwd=root)
    if rc != 0:
        return rc

    growth_cmd = [
        py,
        "scripts/build_growth_portfolio_candidates.py",
        "--trade-date",
        str(args.trade_date),
        "--base-dir",
        str(root),
        "--top-n",
        str(int(args.top_n)),
        "--portfolio-size",
        "12",
        "--max-per-sector",
        "2",
        "--min-score",
        "10",
        "--min-sources",
        "2",
        "--require-stock-screener",
        "--stock-lookback-days",
        str(int(args.stock_lookback_days)),
    ]
    rc = run_cmd(growth_cmd, cwd=root)
    if rc != 0:
        return rc

    packet_cmd = [
        py,
        "scripts/build_deep_research_packet.py",
        "--trade-date",
        str(args.trade_date),
        "--base-dir",
        str(root),
        "--stock-lookback-days",
        str(int(args.stock_lookback_days)),
        "--route-lookback-days",
        str(int(args.route_lookback_days)),
        "--artifact-lookback-days",
        str(int(args.artifact_lookback_days)),
        "--top-n",
        str(int(args.packet_top_n)),
    ]
    rc = run_cmd(packet_cmd, cwd=root)
    if rc != 0:
        return rc

    out = root / str(args.trade_date) / f"deep_research_packet_{args.trade_date}.md"
    print("")
    print(f"Done. Packet: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
