import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

from groki_eq import io_out
from groki_eq.config import OUT_DIR
from groki_eq.dates import today_et
from groki_eq.envload import ORATS_TOKEN_MISSING, load_orats_token
from groki_eq.manage import render_manage_md
from groki_eq.orats import map_line, redact
from groki_eq.pipeline import build_day
from groki_eq.schwab import live_note, use_live_schwab


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="groki_eq",
        description="Index equity 20-day breakout",
        allow_abbrev=False,
    )
    parser.add_argument("--date", required=True, help="session date YYYY-MM-DD")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--live-schwab", action="store_true")
    parser.add_argument("--no-schwab", action="store_true")
    parser.add_argument("--orats-token-file", default=None)
    parser.add_argument("--max-orats-requests", type=int, default=None)
    return parser.parse_args(argv)


def print_result(info: Dict[str, object]) -> None:
    mapped = info.get("orats_map") or {}
    print("orats_ok=%s" % (info.get("orats_ok") or 0))
    print("orats_http=%s" % (info.get("orats_http") or 0))
    print("orats_rows=%s" % (info.get("orats_rows") or 0))
    if mapped.get("close"):
        print(
            "orats_map close=%s high=%s low=%s open=%s"
            % (
                mapped.get("close") or "",
                mapped.get("high") or "",
                mapped.get("low") or "",
                mapped.get("open") or "",
            )
        )
    elif mapped:
        print("orats_map %s" % map_line(mapped))
    tapes = info.get("tapes") or {}
    if tapes:
        print("tape=%s" % ",".join("%s:%s" % (k, tapes[k]) for k in tapes))
    print("selector=%s" % (info.get("selector") or "breakout_eq"))
    print("execute_count=%s" % (info.get("execute_count") or 0))
    if int(info.get("execute_count") or 0) == 1:
        print("execute=%s %s" % (info.get("execute_ticker") or "", info.get("execute_target") or ""))
    if int(info.get("execute_count") or 0) == 0:
        print("blocker=%s" % (info.get("blocker") or ""))


def run(
    date: str,
    out_dir: Path,
    token: str,
    max_orats_requests: Optional[int] = None,
    getter=None,
    live_schwab: bool = False,
    no_schwab: bool = False,
    today: Optional[str] = None,
    liquid=None,
    schwab_bars=None,
) -> Dict[str, object]:
    today = today or today_et()
    live = use_live_schwab(date, live_flag=live_schwab, no_schwab=no_schwab, today=today)
    built = build_day(
        date,
        token,
        out_dir=out_dir,
        getter=getter,
        max_requests=max_orats_requests,
        liquid=liquid,
        schwab_bars=schwab_bars,
    )
    day = io_out.day_dir(out_dir, date)
    io_out.write_csv(day / "board.csv", io_out.BOARD_COLUMNS, built.get("board") or [])
    io_out.write_board_md(
        day / "board.md",
        date=date,
        blocker=str(built.get("blocker") or ""),
        execute_count=int(built.get("execute_count") or 0),
        watch_count=int(built.get("watch_count") or 0),
        execute_rows=built.get("execute_rows") or [],
        watch_rows=built.get("watch_rows") or [],
    )
    io_out.write_csv(day / "rejections.csv", io_out.REJECTION_COLUMNS, built.get("rejections") or [])
    note = live_note(date, live)
    manage_rows = []
    if live and not note:
        from groki_eq.config import UNIVERSE
        from groki_eq.schwab import positions_universe

        for pos in positions_universe(UNIVERSE):
            manage_rows.append(
                {
                    "ticker": pos.get("ticker"),
                    "entry": "",
                    "stop": "",
                    "sessions": "",
                    "verdict": "HOLD",
                }
            )
    io_out.write_text(day / "manage_existing.md", render_manage_md(date, live, rows=manage_rows, note=note))
    evidence = [
        "# evidence",
        "",
        "selector: breakout_eq",
        "universe: SPY QQQ IWM",
        "tapes: %s" % (built.get("tapes") or {}),
        "book_open: %s" % built.get("book_open"),
        "week_new: %s" % built.get("week_new"),
        "live_schwab: %s" % ("true" if live else "false"),
        "Stop: entry − 2×ATR(14). Time stop: 15 sessions.",
        "No profit-target exit.",
        "No invented prices.",
        "No submit/cancel/replace.",
        "",
    ]
    io_out.write_text(day / "evidence.md", "\n".join(evidence))
    from groki_eq.orats import field_map_path, _read_json

    fmap = _read_json(field_map_path()) or {}
    orats_map = {
        "close": ((fmap.get("close") or {}) or {}).get("key") or "clsPx",
        "high": ((fmap.get("high") or {}) or {}).get("key") or "hiPx",
        "low": ((fmap.get("low") or {}) or {}).get("key") or "loPx",
        "open": ((fmap.get("open") or {}) or {}).get("key") or "open",
    }
    manifest = {
        "date": date,
        "selector": "breakout_eq",
        "execute_count": int(built.get("execute_count") or 0),
        "execute_ticker": built.get("execute_ticker") or "",
        "execute_target": built.get("execute_target") or "",
        "blocker": built.get("blocker") or "",
        "watch_count": int(built.get("watch_count") or 0),
        "live_schwab": live,
        "orats_ok": built.get("orats_ok") or 0,
        "orats_http": built.get("orats_http") or 0,
        "orats_rows": built.get("orats_rows") or 0,
        "orats_requests_used": built.get("orats_requests_used") or 0,
        "orats_requests_left": built.get("orats_requests_left") or 0,
        "orats_map": orats_map,
        "tapes": built.get("tapes") or {},
    }
    io_out.write_manifest(day / "manifest.json", manifest)
    built["date"] = date
    built["out_dir"] = str(day)
    built["live_schwab"] = live
    built["orats_map"] = orats_map
    built["error"] = built.get("orats_error") or ""
    return built


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    token = load_orats_token(token_file=args.orats_token_file)
    if not token:
        print(ORATS_TOKEN_MISSING, file=sys.stderr)
        return 2
    try:
        result = run(
            date=args.date,
            out_dir=Path(args.out_dir),
            token=token,
            max_orats_requests=args.max_orats_requests,
            live_schwab=args.live_schwab,
            no_schwab=args.no_schwab,
        )
    except Exception as exc:
        print("orats_ok=0", file=sys.stderr)
        print(redact(str(exc), token), file=sys.stderr)
        return 1
    print_result(result)
    if not int(result.get("orats_ok") or 0):
        err = result.get("error") or ""
        if err:
            print("orats_error=%s" % redact(str(err), token), file=sys.stderr)
        return 1
    return 0
