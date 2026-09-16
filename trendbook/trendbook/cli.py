from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

from trendbook.config import ORATS_HTTP_DEFAULT, OUT_DIR
from trendbook.dates import today_et
from trendbook.envload import ORATS_TOKEN_MISSING, load_orats_token

_YMD = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_CMDS = ("full", "analyze", "replay")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="trendbook",
        description="Weekly Stage 2 / relative-strength trend book",
        allow_abbrev=False,
    )
    parser.add_argument("cmd", nargs="?", default=None, help="full | analyze | replay | YYYY-MM-DD")
    parser.add_argument("name", nargs="?", default=None, help="ticker for analyze")
    parser.add_argument("--date", default=None)
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--live-schwab", action="store_true")
    parser.add_argument("--no-cache", action="store_true", help="refetch Schwab bars; do not read disk cache")
    parser.add_argument("--no-schwab", action="store_true")
    parser.add_argument("--no-orats", action="store_true")
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
        parser.error("cmd must be full, analyze, replay, or a YYYY-MM-DD date")
    if not args.date:
        args.date = today_et()
    if args.cmd == "analyze":
        ticker = args.ticker or args.name
        if not ticker:
            parser.error("analyze needs a ticker")
        args.ticker = ticker.upper()
    return args


def print_result(info: Dict[str, object]) -> None:
    print("trendbook_mode=%s" % (info.get("mode") or "full"))
    print("date=%s" % (info.get("date") or ""))
    if info.get("mode") == "replay":
        print("tagged=%s" % (info.get("n_tagged") or 0))
        print("schwab_http=%s" % (info.get("schwab_http") or 0))
        print("out=%s" % (info.get("out_dir") or ""))
        files = info.get("files") or {}
        if isinstance(files, dict) and files.get("replay"):
            print("replay=%s" % files["replay"])
        return
    print("add=%s" % (info.get("n_add") or 0))
    print("new=%s" % (info.get("n_new") or 0))
    print("hold=%s" % (info.get("n_hold") or 0))
    print("late=%s" % (info.get("n_late") or 0))
    print("out=%s" % (info.get("n_out") or 0))
    print("on_board=%s" % (info.get("n_board") or 0))
    print("orats_http=%s" % (info.get("orats_http") or 0))
    print("schwab_http=%s" % (info.get("schwab_http") or 0))
    if info.get("orats_token") is False:
        print("orats=missing")
    print("out=%s" % (info.get("out_dir") or ""))
    files = info.get("files") or {}
    if isinstance(files, dict) and files.get("board"):
        print("board=%s" % files["board"])


def main(argv: Optional[List[str]] = None) -> int:
    from trendbook.pipeline import build_full, build_replay

    args = parse_args(argv)
    if args.cmd == "replay":
        info = build_replay(
            args.date,
            out_dir=Path(args.out_dir),
            live_schwab=args.live_schwab,
            no_schwab=args.no_schwab,
            use_cache=not args.no_cache,
        )
        print_result(info)
        return 0

    token = None if args.no_orats else load_orats_token(token_file=args.orats_token_file)
    if not token and not args.no_orats:
        print(ORATS_TOKEN_MISSING, file=sys.stderr)
    tickers = [args.ticker] if args.cmd == "analyze" else None
    if args.cmd == "analyze" and tickers:
        from trendbook.config import ticker_etf

        tickers = ["SPY", tickers[0]]
        etf = ticker_etf(args.ticker)
        if etf and etf not in tickers:
            tickers.append(etf)
    info = build_full(
        args.date,
        out_dir=Path(args.out_dir),
        live_schwab=args.live_schwab,
        no_schwab=args.no_schwab,
        no_orats=args.no_orats or not token,
        max_orats_http=args.max_orats_http,
        orats_token_file=args.orats_token_file,
        tickers=tickers,
        use_cache=not args.no_cache,
    )
    print_result(info)
    return 0
