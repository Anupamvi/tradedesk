from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

from groat import report
from groat.book import positions as load_positions
from groat.config import OUT_DIR
from groat.dates import today_et
from groat.envload import ORATS_TOKEN_MISSING, load_orats_token
from groat.gates import open_trade_verdict
from groat.num import to_float
from groat.orats import map_line, redact
from groat.pipeline import build_analyze, build_delta, build_full
from groat.schwab import live_note, use_live_schwab
from groat.xintel import missing_x_tickers


_YMD = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_CMDS = ("full", "delta", "analyze", "review", "replay")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="groat",
        description="Groat swing-trading research desk",
        allow_abbrev=False,
    )
    parser.add_argument(
        "cmd",
        nargs="?",
        default=None,
        help="full | delta | analyze | review | replay | YYYY-MM-DD (full scan that date)",
    )
    parser.add_argument("ticker", nargs="?", default=None, help="ticker for analyze, or YYYY-MM-DD")
    parser.add_argument("--date", default=None, help="session date YYYY-MM-DD (default: today ET)")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--live-schwab", action="store_true")
    parser.add_argument("--no-schwab", action="store_true")
    parser.add_argument("--orats-token-file", default=None)
    parser.add_argument("--max-orats-requests", type=int, default=None)
    parser.add_argument("--max-final", type=int, default=None)
    parser.add_argument(
        "--max-strike-http",
        type=int,
        default=40,
        help="replay: hist/strikes HTTP per option slice (0 = stock only)",
    )
    parser.add_argument(
        "--option-slices",
        type=int,
        default=0,
        help="replay: number of option hist/strikes slices (default 0 = stock only)",
    )
    args = parser.parse_args(argv)
    if args.cmd and _YMD.match(args.cmd):
        if args.date and args.date != args.cmd:
            parser.error("conflicting dates")
        args.date = args.cmd
        args.cmd = "full"
    if args.ticker and _YMD.match(args.ticker) and (args.cmd or "full") in ("full", "delta", "review", "replay"):
        if args.date and args.date != args.ticker:
            parser.error("conflicting dates")
        args.date = args.ticker
        args.ticker = None
    args.cmd = args.cmd or "full"
    if args.cmd not in _CMDS:
        parser.error("cmd must be full, delta, analyze, review, replay, or a YYYY-MM-DD date")
    if not args.date:
        args.date = today_et()
    return args


def print_result(info: Dict[str, object]) -> None:
    print("groat_mode=%s" % (info.get("mode") or "full"))
    if info.get("mode") == "replay":
        print("tape=%s..%s" % (info.get("tape_from") or "", info.get("tape_to") or ""))
        print("hits=%s" % (info.get("n_hits") or 0))
        print("orats_http=%s" % (info.get("orats_http") or 0))
        print("orats_requests_used=%s" % (info.get("orats_requests_used") or 0))
        print("orats_requests_left=%s" % (info.get("orats_requests_left") or 0))
        overall = info.get("overall") or {}
        print("stock_n=%s avg_r=%s" % (overall.get("n") or 0, overall.get("avg_r")))
        print("opt_n=%s opt_avg_pnl_per_risk=%s" % (overall.get("opt_n") or 0, overall.get("opt_avg_pnl_per_risk")))
        if info.get("option_slices"):
            print("option_slices=%s" % len(info.get("option_slices") or []))
        return
    print("regime=%s" % (info.get("regime_label") or ""))
    if info.get("session"):
        print("session=%s" % info.get("session"))
    print("orats_ok=%s" % (info.get("orats_ok") or 0))
    print("orats_http=%s" % (info.get("orats_http") or 0))
    print("orats_rows=%s" % (info.get("orats_rows") or 0))
    print("trade_count=%s" % (info.get("trade_count") or 0))
    print("watch_count=%s" % (info.get("watch_count") or 0))
    print(
        "chains_requested=%s empty=%s not_requested=%s schwab_errors=%s"
        % (
            len(info.get("option_names") or []),
            len(info.get("chain_empty") or []),
            len(info.get("chain_not_requested") or []),
            len(info.get("schwab_chain_errors") or []),
        )
    )
    trades = info.get("trade_tickers") or []
    if trades:
        print("trades=%s" % ",".join(str(t) for t in trades))
    missing_x = info.get("x_missing_on_trade")
    if missing_x is None:
        missing_x = []
    print("x_missing_on_trade=%s" % (",".join(str(t) for t in missing_x) if missing_x else "none"))
    if missing_x:
        print("x_incomplete=1 search_$TICKER_and_rerun")
    if int(info.get("trade_count") or 0) == 0:
        print("blocker=%s" % (info.get("blocker") or "empty_board"))
    ev = info.get("evidence") if isinstance(info.get("evidence"), dict) else {}
    if ev.get("rows"):
        bits = []
        for row in ev.get("rows") or []:
            stock = row.get("stock") or {}
            bits.append("%s:%s" % (row.get("ticker"), stock.get("n") or 0))
        print("evidence=%s http=%s" % (",".join(bits), ev.get("http") or 0))
    mapped = info.get("orats_map") or {}
    if mapped:
        print("orats_map %s" % map_line(mapped) if mapped.get("iv30") or mapped.get("close") else "orats_map present")


def _prev_candidates(out_dir: Path, asof: str) -> Optional[dict]:
    root = Path(out_dir)
    if not root.is_dir():
        return None
    dates = sorted(p.name for p in root.iterdir() if p.is_dir() and len(p.name) == 10 and p.name < asof)
    if not dates:
        return None
    folder = root / dates[-1]
    for rel in ("close/candidates.json", "candidates.json", "open/candidates.json"):
        path = folder / rel
        if path.is_file():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
    return None


def _review_rows(asof: str, built: dict) -> List[dict]:
    snaps = built.get("snaps") or {}
    rows = []
    for pos in load_positions():
        ticker = str(pos.get("ticker") or "").upper()
        snap = snaps.get(ticker) or {}
        judged = open_trade_verdict(pos, snap)
        rows.append(
            {
                "ticker": ticker,
                "instrument": pos.get("instrument") or pos.get("structure") or "",
                "entry": pos.get("entry"),
                "stop": judged.get("stop"),
                "target": pos.get("target"),
                "last": judged.get("last"),
                "verdict": judged.get("verdict") or "HOLD",
                "why": judged.get("why") or "",
                "side": judged.get("side") or "long",
            }
        )
    return rows


def run(
    date: str,
    out_dir: Path,
    token: str,
    cmd: str = "full",
    ticker: Optional[str] = None,
    max_orats_requests: Optional[int] = None,
    getter=None,
    live_schwab: bool = False,
    no_schwab: bool = False,
    today: Optional[str] = None,
    max_strike_http: int = 40,
    option_slices: int = 0,
    universe=None,
    bars_by_ticker=None,
    cores_by_ticker=None,
    strikes_by_ticker=None,
    vix_bars=None,
) -> Dict[str, object]:
    today = today or today_et()
    if cmd == "replay":
        from groat.replay import render_replay, run_replay

        payload = run_replay(
            date,
            token=token,
            today=today,
            getter=getter,
            max_requests=max_orats_requests,
            max_strike_http=max_strike_http,
            option_slices=option_slices,
            universe=list(universe) if universe is not None else None,
            bars_by_ticker=bars_by_ticker,
        )
        day = report.day_dir(out_dir, date)
        report.write_text(day / "replay.md", render_replay(payload))
        report.write_json(day / "replay.json", payload)
        payload["out_dir"] = str(day)
        payload["orats_ok"] = 1
        payload["orats_rows"] = 0
        payload["trade_count"] = 0
        payload["watch_count"] = 0
        payload["trade_tickers"] = []
        payload["regime_label"] = ""
        payload["orats_map"] = {}
        payload["error"] = ""
        return payload
    live = use_live_schwab(date, live_flag=live_schwab, no_schwab=no_schwab, today=today)
    session = None
    if cmd != "replay":
        from groat.dates import session_phase

        session = session_phase(date, today)
    built = build_full(
        date,
        token,
        today=today,
        live=live,
        getter=getter,
        max_requests=max_orats_requests,
        universe=universe,
        bars_by_ticker=bars_by_ticker,
        cores_by_ticker=cores_by_ticker,
        strikes_by_ticker=strikes_by_ticker,
        vix_bars=vix_bars,
        session=session,
        out_dir=out_dir,
    )
    day = report.day_dir(out_dir, date)
    report.write_scan_artifacts(day, date, built)

    if cmd == "delta":
        previous = _prev_candidates(out_dir, date)
        delta = build_delta(date, previous, built)
        report.write_text(day / "delta.md", report.render_delta(date, delta))
        report.write_json(day / "delta.json", delta)
        built["delta"] = delta
    elif cmd == "analyze":
        if not ticker:
            raise SystemExit("analyze requires a ticker")
        row = build_analyze(date, ticker, built)
        report.write_text(day / ("analyze_%s.md" % str(ticker).upper()), report.render_analyze(date, row))
        report.write_json(day / ("analyze_%s.json" % str(ticker).upper()), row)
        built["analyze"] = row
    elif cmd == "review":
        rows = _review_rows(date, built)
        note = live_note(date, live)
        extra = "\nSchwab note: %s\n" % note if note else ""
        report.write_text(day / "review.md", report.render_review(date, rows) + extra)
        report.write_json(day / "review.json", {"asof": date, "rows": rows})
        built["review"] = rows

    from groat.orats import _read_json, field_map_path

    fmap = _read_json(field_map_path()) or {}
    orats_map = fmap
    blocker = ""
    if int(built.get("trade_count") or 0) == 0:
        blocker = (built.get("regime") or {}).get("regime") or "empty_board"
        if built.get("orats_error"):
            blocker = str(built.get("orats_error"))
    manifest = {
        "date": date,
        "mode": cmd,
        "selector": "groat_swing",
        "regime": (built.get("regime") or {}).get("regime"),
        "trade_count": int(built.get("trade_count") or 0),
        "watch_count": int(built.get("watch_count") or 0),
        "trades": [r.get("ticker") for r in (built.get("trades") or [])],
        "live_schwab": live,
        "session": built.get("session") or "",
        "session_incomplete": bool(built.get("session_incomplete")),
        "orats_ok": built.get("orats_ok") or 0,
        "orats_http": built.get("orats_http") or 0,
        "orats_rows": built.get("orats_rows") or 0,
        "orats_requests_used": built.get("orats_requests_used") or 0,
        "orats_requests_left": built.get("orats_requests_left") or 0,
        "orats_error": built.get("orats_error") or "",
        "option_names": built.get("option_names") or [],
        "chain_empty": built.get("chain_empty") or [],
        "chain_not_requested": built.get("chain_not_requested") or [],
        "schwab_chain_errors": built.get("schwab_chain_errors") or [],
        "blocker": blocker,
        "x_missing_on_trade": missing_x_tickers(built.get("trades") or []),
        "tapes": built.get("tapes") or {},
        "evidence_http": ((built.get("evidence") or {}) if isinstance(built.get("evidence"), dict) else {}).get("http") or 0,
    }
    report.write_json(day / "manifest.json", manifest)
    from groat.persist import copy_session_artifacts

    copy_session_artifacts(day, str(built.get("session") or session or ""))
    built["date"] = date
    built["out_dir"] = str(day)
    built["live_schwab"] = live
    built["mode"] = cmd
    built["regime_label"] = (built.get("regime") or {}).get("regime")
    built["trade_tickers"] = [r.get("ticker") for r in (built.get("trades") or [])]
    built["x_missing_on_trade"] = missing_x_tickers(built.get("trades") or [])
    built["blocker"] = blocker
    built["orats_map"] = orats_map
    built["error"] = built.get("orats_error") or ""
    return built


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    token = load_orats_token(token_file=args.orats_token_file)
    if not token:
        print(ORATS_TOKEN_MISSING, file=sys.stderr)
        return 2
    if args.cmd == "analyze" and not args.ticker:
        print("analyze requires a ticker", file=sys.stderr)
        return 2
    try:
        result = run(
            date=args.date,
            out_dir=Path(args.out_dir),
            token=token,
            cmd=args.cmd,
            ticker=args.ticker,
            max_orats_requests=args.max_orats_requests,
            live_schwab=args.live_schwab,
            no_schwab=args.no_schwab,
            max_strike_http=args.max_strike_http,
            option_slices=args.option_slices,
        )
    except Exception as exc:
        print("orats_ok=0", file=sys.stderr)
        print(redact(str(exc), token), file=sys.stderr)
        return 1
    print_result(result)
    if result.get("x_missing_on_trade"):
        return 3
    return 0
