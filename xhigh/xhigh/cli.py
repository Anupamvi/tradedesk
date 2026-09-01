from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

from xhigh.config import ORATS_HTTP_DEFAULT, OUT_DIR
from xhigh.dates import today_et
from xhigh.envload import ORATS_TOKEN_MISSING, load_orats_token
from xhigh.intel import apply as apply_intel
from xhigh.pipeline import build_full
from xhigh.revalidate import apply as apply_revalidate
from xhigh.xhot import apply as apply_xhot

_YMD = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_CMDS = ("full", "xhot", "intel", "revalidate", "analyze")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="xhigh",
        description="xhigh new-setup wheel and swing scanner",
        allow_abbrev=False,
    )
    parser.add_argument("cmd", nargs="?", default=None, help="full | xhot | intel | revalidate | analyze | YYYY-MM-DD")
    parser.add_argument("name", nargs="?", default=None, help="ticker for analyze")
    parser.add_argument("--date", default=None)
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--live-schwab", action="store_true")
    parser.add_argument("--no-schwab", action="store_true")
    parser.add_argument("--orats-token-file", default=None)
    parser.add_argument("--max-orats-http", type=int, default=ORATS_HTTP_DEFAULT)
    args = parser.parse_args(argv)
    if args.cmd and _YMD.match(args.cmd):
        if args.date and args.date != args.cmd:
            parser.error("conflicting dates")
        args.date = args.cmd
        args.cmd = "full"
    args.cmd = args.cmd or "full"
    if args.cmd not in _CMDS:
        parser.error("cmd must be full, xhot, intel, revalidate, analyze, or a YYYY-MM-DD date")
    if not args.date:
        args.date = today_et()
    if args.cmd == "analyze":
        ticker = args.ticker or args.name
        if not ticker:
            parser.error("analyze needs a ticker")
        args.ticker = ticker.upper()
    return args


def print_result(info: Dict[str, object]) -> None:
    print("xhigh_mode=%s" % (info.get("mode") or "full"))
    print("date=%s" % (info.get("date") or ""))
    print("click=%s" % (info.get("n_trade") or 0))
    print("skip=%s" % (info.get("n_skip") or 0))
    print("watch=%s" % (info.get("n_watch") or 0))
    print("shortlist=%s" % (info.get("n_shortlist") or 0))
    print("orats_http=%s" % (info.get("orats_http") or 0))
    print("schwab_http=%s" % (info.get("schwab_http") or 0))
    if info.get("orats_token") is False:
        print("orats=missing")
    print("out=%s" % (info.get("out_dir") or ""))
    files = info.get("files") or {}
    if isinstance(files, dict) and files.get("board"):
        print("board=%s" % files["board"])


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.cmd == "xhot":
        info = apply_xhot(args.date, out_root=Path(args.out_dir))
        print("xhigh_mode=xhot")
        print("date=%s" % info.get("date"))
        print("board=%s" % info.get("out"))
        return 0
    if args.cmd == "intel":
        info = apply_intel(args.date, out_root=Path(args.out_dir))
        print("xhigh_mode=intel")
        print("date=%s" % info.get("date"))
        print("board=%s" % info.get("out"))
        return 0
    if args.cmd == "revalidate":
        info = apply_revalidate(args.date, out_root=Path(args.out_dir))
        print("xhigh_mode=revalidate")
        print("date=%s" % info.get("date"))
        print("kept=%s" % info.get("kept"))
        print("board=%s" % info.get("out"))
        return 0
    token = load_orats_token(token_file=args.orats_token_file)
    if not token:
        print(ORATS_TOKEN_MISSING, file=sys.stderr)
    tickers = [args.ticker] if args.cmd == "analyze" else None
    info = build_full(
        args.date,
        out_dir=Path(args.out_dir),
        live_schwab=args.live_schwab,
        no_schwab=args.no_schwab,
        max_orats_http=args.max_orats_http,
        orats_token_file=args.orats_token_file,
        tickers=tickers,
    )
    print_result(info)
    return 0
