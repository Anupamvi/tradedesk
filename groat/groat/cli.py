from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

from groat import report
from groat.book import positions as load_positions
from groat.config import OUT_DIR
from groat.dates import today_et
from groat.envload import ORATS_TOKEN_MISSING, load_orats_token
from groat.num import to_float
from groat.orats import map_line, redact
from groat.pipeline import build_analyze, build_delta, build_full
from groat.schwab import live_note, use_live_schwab


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="groat",
        description="Groat swing-trading research desk",
        allow_abbrev=False,
    )
    parser.add_argument(
        "cmd",
        nargs="?",
        default="full",
        choices=["full", "delta", "analyze", "review"],
    )
    parser.add_argument("ticker", nargs="?", default=None, help="ticker for analyze")
    parser.add_argument("--date", required=True, help="session date YYYY-MM-DD")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--live-schwab", action="store_true")
    parser.add_argument("--no-schwab", action="store_true")
    parser.add_argument("--orats-token-file", default=None)
    parser.add_argument("--max-orats-requests", type=int, default=None)
    parser.add_argument("--max-final", type=int, default=None)
    return parser.parse_args(argv)


def print_result(info: Dict[str, object]) -> None:
    print("groat_mode=%s" % (info.get("mode") or "full"))
    print("regime=%s" % (info.get("regime_label") or ""))
    print("orats_ok=%s" % (info.get("orats_ok") or 0))
    print("orats_http=%s" % (info.get("orats_http") or 0))
    print("orats_rows=%s" % (info.get("orats_rows") or 0))
    print("trade_count=%s" % (info.get("trade_count") or 0))
    print("watch_count=%s" % (info.get("watch_count") or 0))
    trades = info.get("trade_tickers") or []
    if trades:
        print("trades=%s" % ",".join(str(t) for t in trades))
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
    path = root / dates[-1] / "candidates.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _review_rows(asof: str, built: dict) -> List[dict]:
    snaps = built.get("snaps") or {}
    rows = []
    for pos in load_positions():
        ticker = str(pos.get("ticker") or "").upper()
        snap = snaps.get(ticker) or {}
        last = to_float(snap.get("close"))
        stop = to_float(pos.get("stop"))
        target = to_float(pos.get("target"))
        side = str(pos.get("side") or pos.get("direction") or "long").lower()
        if "short" in side or "bear" in side:
            side = "short"
        else:
            side = "long"
        verdict = "HOLD"
        why = "original thesis not invalidated"
        if last is None:
            why = "last price DATA UNAVAILABLE"
        elif stop is not None and side == "long" and last <= stop:
            verdict = "EXIT"
            why = "stop / invalidation hit"
        elif stop is not None and side == "short" and last >= stop:
            verdict = "EXIT"
            why = "stop / invalidation hit"
        elif target is not None and side == "long" and last >= target:
            verdict = "TAKE PROFIT"
            why = "structure target reached"
        elif target is not None and side == "short" and last <= target:
            verdict = "TAKE PROFIT"
            why = "structure target reached"
        if verdict == "HOLD" and not pos.get("scale_plan"):
            # ADD is forbidden without a predefined scale plan
            pass
        rows.append(
            {
                "ticker": ticker,
                "instrument": pos.get("instrument") or pos.get("structure") or "",
                "entry": pos.get("entry"),
                "stop": stop,
                "target": target,
                "last": last,
                "verdict": verdict,
                "why": why,
                "side": side,
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
    universe=None,
    bars_by_ticker=None,
    cores_by_ticker=None,
    strikes_by_ticker=None,
    vix_bars=None,
) -> Dict[str, object]:
    today = today or today_et()
    live = use_live_schwab(date, live_flag=live_schwab, no_schwab=no_schwab, today=today)
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
        "orats_ok": built.get("orats_ok") or 0,
        "orats_http": built.get("orats_http") or 0,
        "orats_rows": built.get("orats_rows") or 0,
        "orats_requests_used": built.get("orats_requests_used") or 0,
        "orats_requests_left": built.get("orats_requests_left") or 0,
        "orats_error": built.get("orats_error") or "",
        "blocker": blocker,
        "tapes": built.get("tapes") or {},
        "evidence_http": ((built.get("evidence") or {}) if isinstance(built.get("evidence"), dict) else {}).get("http") or 0,
    }
    report.write_json(day / "manifest.json", manifest)
    built["date"] = date
    built["out_dir"] = str(day)
    built["live_schwab"] = live
    built["mode"] = cmd
    built["regime_label"] = (built.get("regime") or {}).get("regime")
    built["trade_tickers"] = [r.get("ticker") for r in (built.get("trades") or [])]
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
        )
    except Exception as exc:
        print("orats_ok=0", file=sys.stderr)
        print(redact(str(exc), token), file=sys.stderr)
        return 1
    print_result(result)
    if args.cmd in ("full", "delta", "analyze", "review"):
        return 0
    return 0
