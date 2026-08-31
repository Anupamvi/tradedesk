from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

from wheelo.config import OUT_DIR
from wheelo.dates import today_et
from wheelo.envload import ORATS_TOKEN_MISSING, load_orats_token
from wheelo.orats import redact
from wheelo.pipeline import overlay_x_artifacts, run_pipeline


_YMD = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_CMDS = ("select", "daily", "full", "analyze", "review", "xhot")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="wheelo",
        description="Wheelo CSP / covered-call desk (ORATS after Schwab shortlist)",
        allow_abbrev=False,
    )
    parser.add_argument(
        "cmd",
        nargs="?",
        default=None,
        help="select | daily | full | analyze | review | xhot | YYYY-MM-DD",
    )
    parser.add_argument("ticker", nargs="?", default=None, help="ticker for analyze, or YYYY-MM-DD")
    parser.add_argument("--date", default=None, help="session date YYYY-MM-DD (default: today ET)")
    parser.add_argument("--capital", type=float, default=35000)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--live-schwab", action="store_true")
    parser.add_argument("--no-schwab", action="store_true")
    parser.add_argument("--yfinance", action="store_true")
    parser.add_argument("--orats-token-file", default=None)
    parser.add_argument("--max-orats-requests", type=int, default=15)
    args = parser.parse_args(argv)
    if args.cmd and _YMD.match(args.cmd):
        if args.date and args.date != args.cmd:
            parser.error("conflicting dates")
        args.date = args.cmd
        args.cmd = "full"
    if args.ticker and _YMD.match(args.ticker) and (args.cmd or "full") in ("select", "daily", "full", "review"):
        if args.date and args.date != args.ticker:
            parser.error("conflicting dates")
        args.date = args.ticker
        args.ticker = None
    args.cmd = args.cmd or "full"
    if args.cmd not in _CMDS:
        parser.error("cmd must be select, daily, full, analyze, review, xhot, or a YYYY-MM-DD date")
    if args.cmd == "analyze" and not args.ticker:
        parser.error("analyze requires a ticker")
    if not args.date:
        args.date = today_et()
    return args


def print_result(info: Dict[str, object]) -> None:
    print("wheelo_mode=%s" % (info.get("mode") or "full"))
    print("date=%s" % (info.get("asof") or info.get("date") or ""))
    print("orats_http=%s" % (info.get("orats_http") or 0))
    print("orats_planned=%s" % (info.get("orats_planned") or 0))
    print("shortlist_a=%s" % (info.get("shortlist_a") or 0))
    print("shortlist_b=%s" % (info.get("shortlist_b") or 0))
    print("shortlist_c=%s" % (info.get("shortlist_c") or 0))
    print("allocated=%s" % (info.get("trade_count") or 0))
    print("schwab=%s" % ("on" if info.get("schwab") else "off"))
    if info.get("error"):
        print("error=%s" % info.get("error"))
    if info.get("out_dir"):
        print("out=%s" % info.get("out_dir"))
    cands = info.get("candidates") or []
    allocated = [c for c in cands if isinstance(c, dict) and c.get("allocated")]
    if allocated:
        print("trades=%s" % ",".join(str(c.get("ticker")) for c in allocated))
    actions = info.get("actions") or []
    if actions:
        print("daily_actions=%s" % len(actions))


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.cmd == "xhot":
        day = overlay_x_artifacts(args.date, Path(args.out_dir), args.capital)
        applied = 0
        cand_path = day / "candidates.json"
        if cand_path.is_file():
            try:
                rows = json.loads(cand_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                rows = []
            applied = sum(
                1
                for row in rows
                if isinstance(row, dict) and row.get("x_status") not in (None, "", "DATA UNAVAILABLE")
            )
        print("wheelo_mode=xhot")
        print("date=%s" % args.date)
        print("orats_http=0")
        print("xhot_applied=%s" % applied)
        print("out=%s" % day)
        return 0
    token = load_orats_token(token_file=args.orats_token_file)
    if not token:
        print(ORATS_TOKEN_MISSING, file=sys.stderr)
        return 2
    try:
        info = run_pipeline(
            args.cmd,
            args.date,
            token,
            args.capital,
            out_dir=Path(args.out_dir),
            live_schwab=args.live_schwab,
            no_schwab=args.no_schwab,
            max_orats_requests=args.max_orats_requests,
            ticker=args.ticker,
            use_yfinance=args.yfinance,
        )
    except Exception as exc:
        print(redact(str(exc), token), file=sys.stderr)
        return 1
    print_result(info)
    if info.get("error") == "orats_budget":
        return 3
    return 0
