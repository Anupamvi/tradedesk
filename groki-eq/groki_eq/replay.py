"""Cache-first breakout_eq replay. No order placement."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from groki_eq import config
from groki_eq.book import close_open
from groki_eq.config import SLEEVE, SLEEVE_PROMOTE_PF, TRAIN_TEST_SPLIT
from groki_eq.dates import today_et
from groki_eq.envload import ORATS_TOKEN_MISSING, load_orats_token
from groki_eq.fill import fmt_metrics, pnl_dollars, stop_fill, summarize, tenk_contracts, time_stop_hit
from groki_eq.orats import load_usage, redact
from groki_eq.pipeline import build_day, load_universe
from groki_eq.prices import ensure_bars


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="groki_eq.replay",
        description="Cache-first equity breakout replay",
        allow_abbrev=False,
    )
    parser.add_argument("--start", default=config.REPLAY_DEFAULT_START)
    parser.add_argument("--end", default=None)
    parser.add_argument("--max-days", type=int, default=0, help="0 means no cap")
    parser.add_argument("--out-dir", default=str(config.OUT_DIR))
    parser.add_argument("--orats-token-file", default=None)
    parser.add_argument("--max-requests", type=int, default=None)
    return parser.parse_args(argv)


def _select_sessions(days: List[str], max_days: int) -> List[str]:
    if max_days and max_days > 0:
        return days[:max_days]
    return days


def _index_bars(bars: List[dict]) -> Dict[str, dict]:
    return {str(b.get("date") or "")[:10]: b for b in bars}


def _trade_from_row(row: dict, asof: str) -> Optional[dict]:
    try:
        entry = float(row.get("close"))
        stop = float(row.get("stop"))
        shares = int(row.get("shares") or 0)
    except (TypeError, ValueError):
        return None
    if shares < 1 or entry <= 0:
        return None
    return {
        "ticker": str(row.get("ticker") or "").upper(),
        "entry_date": asof,
        "entry": entry,
        "stop": stop,
        "shares": shares,
        "sessions_held": 0,
        "exit_date": "",
        "exit_reason": "",
        "exit_px": None,
        "pnl": None,
    }


def _close(trade: dict, day: str, reason: str, exit_px: float) -> None:
    trade["exit_date"] = day
    trade["exit_reason"] = reason
    trade["exit_px"] = exit_px
    trade["pnl"] = pnl_dollars(trade["entry"], exit_px, trade["shares"])


def _step_trade(trade: dict, day: str, bar: Optional[dict]) -> bool:
    if day <= trade["entry_date"]:
        return False
    trade["sessions_held"] = int(trade.get("sessions_held") or 0) + 1
    if bar is not None:
        hit = stop_fill(bar, float(trade["stop"]))
        if hit is not None:
            _close(trade, day, "stop", hit)
            return True
    if time_stop_hit(int(trade["sessions_held"])) and bar is not None and bar.get("close") is not None:
        _close(trade, day, "time", float(bar["close"]))
        return True
    return False


def _ordered_pnls(trades: List[dict], split: str, side: str) -> List[float]:
    picked = []
    for trade in trades:
        pnl = trade.get("pnl")
        if pnl is None:
            continue
        if side == "train" and trade["entry_date"] < split:
            picked.append(trade)
        elif side == "test" and trade["entry_date"] >= split:
            picked.append(trade)
    picked.sort(key=lambda t: (t.get("exit_date") or t["entry_date"], t["entry_date"]))
    return [float(t["pnl"]) for t in picked]


def _may_promote(test: dict) -> bool:
    n = int(test.get("n") or 0)
    if n < 30:
        return False
    if test.get("pf_inf"):
        return True
    try:
        return test.get("pf") is not None and float(test["pf"]) >= SLEEVE_PROMOTE_PF
    except (TypeError, ValueError):
        return False


def _tenk(test: dict, test_sessions: int) -> dict:
    n = int(test.get("n") or 0)
    ev = float(test.get("ev") or 0.0)
    tpm = (n / float(test_sessions) * 21.0) if test_sessions else 0.0
    contracts = tenk_contracts(ev, tpm)
    if contracts is None:
        return {"line": "infeasible (EV<=0 or no trades)", "contracts_1x": None, "ev": ev}
    return {
        "line": "1x size (test EV $%.1f/trade, ~%.1f trades/month)" % (ev, tpm),
        "contracts_1x": contracts,
        "ev": ev,
    }


def run_replay(
    out_dir: Path,
    start: str,
    end: str,
    max_days: int = 0,
    token: Optional[str] = None,
    getter=None,
    max_requests: Optional[int] = None,
) -> dict:
    token = token or load_orats_token() or ""
    names = load_universe()
    bars_by = {}
    indexes = {}
    for ticker in names:
        pack = ensure_bars(ticker, token, getter=getter, max_requests=max_requests)
        bars_by[ticker] = pack.get("bars") or []
        indexes[ticker] = _index_bars(bars_by[ticker])
    spy_days = [b["date"] for b in bars_by.get("SPY") or [] if start <= b["date"] <= end]
    sessions = _select_sessions(spy_days, max_days)
    book = {"open": [], "week_entries": {}}
    open_book: List[dict] = []
    closed: List[dict] = []
    for i, day in enumerate(sessions):
        still = []
        for trade in open_book:
            bar = indexes.get(trade["ticker"], {}).get(day)
            if _step_trade(trade, day, bar):
                close_open(book, trade["ticker"], trade["entry_date"])
                closed.append(trade)
            else:
                still.append(trade)
        open_book = still
        built = build_day(
            day,
            token,
            out_dir=None,
            getter=getter,
            max_requests=max_requests,
            liquid=names,
            book=book,
            bars_by_ticker=bars_by,
        )
        for row in built.get("execute_rows") or []:
            trade = _trade_from_row(row, day)
            if trade:
                open_book.append(trade)
        if (i + 1) % 50 == 0 or i == 0 or i + 1 == len(sessions):
            print(
                "replay_progress day=%s %d/%d open=%d closed=%d"
                % (day, i + 1, len(sessions), len(open_book), len(closed)),
                flush=True,
            )
    last = sessions[-1] if sessions else end
    for trade in open_book:
        bar = indexes.get(trade["ticker"], {}).get(last)
        px = float(bar["close"]) if bar and bar.get("close") is not None else trade["entry"]
        _close(trade, last, "end", px)
        closed.append(trade)
        close_open(book, trade["ticker"], trade["entry_date"])
    metrics = {
        "train": summarize(_ordered_pnls(closed, TRAIN_TEST_SPLIT, "train")),
        "test": summarize(_ordered_pnls(closed, TRAIN_TEST_SPLIT, "test")),
    }
    for part in ("train", "test"):
        if metrics[part].get("pf") == float("inf"):
            metrics[part]["pf"] = None
            metrics[part]["pf_inf"] = True
    test_sessions = len([d for d in sessions if d >= TRAIN_TEST_SPLIT])
    train_sessions = len(sessions) - test_sessions
    test = metrics["test"]
    pf_num = None if test.get("pf") in (None, float("inf")) else test.get("pf")
    promote = _may_promote(test)
    tenk = _tenk(test, test_sessions)
    report = {
        "start": sessions[0] if sessions else start,
        "end": last,
        "days": len(sessions),
        "split": TRAIN_TEST_SPLIT,
        "train_sessions": train_sessions,
        "test_sessions": test_sessions,
        "truncated": "",
        "sleeves": {SLEEVE: metrics},
        "tenk": {SLEEVE: tenk},
    }
    status = {
        "start": report["start"],
        "end": report["end"],
        "split": TRAIN_TEST_SPLIT,
        "sleeves": {
            SLEEVE: {
                "test_n": test["n"],
                "test_pf": pf_num,
                "test_pnl": test["pnl"],
                "test_ev": test["ev"],
                "test_maxdd": test.get("maxdd") or 0.0,
                "promote": promote,
                "tenk_line": tenk["line"],
            }
        },
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sleeve_status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "replay_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report["sleeves_status"] = status["sleeves"]
    report["sleeve_status_path"] = str(out_dir / "sleeve_status.json")
    usage = load_usage()
    report["orats_requests_used"] = usage.get("used") or 0
    report["orats_requests_left"] = usage.get("left") or 0
    return report


def print_report(report: dict) -> None:
    print("replay=ok")
    print("start=%s" % report["start"])
    print("end=%s" % report["end"])
    print("days=%s" % report["days"])
    print("split=%s" % report["split"])
    print(
        "orats_used=%s left=%s"
        % (report.get("orats_requests_used"), report.get("orats_requests_left"))
    )
    metrics = (report.get("sleeves") or {}).get(SLEEVE) or {}
    print(
        "breakout_eq train %s | test %s"
        % (fmt_metrics(metrics.get("train") or {}), fmt_metrics(metrics.get("test") or {}))
    )
    print("$10k breakout_eq: %s" % report["tenk"][SLEEVE]["line"])
    live = (report.get("sleeves_status") or {}).get(SLEEVE) or {}
    print("breakout_eq promote=%s" % live.get("promote"))
    print("sleeve_status=%s" % report.get("sleeve_status_path"))


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    token = load_orats_token(token_file=args.orats_token_file)
    end = args.end or today_et()
    if not token:
        print(ORATS_TOKEN_MISSING)
        return 2
    try:
        report = run_replay(
            Path(args.out_dir),
            args.start,
            end,
            max_days=args.max_days,
            token=token,
            max_requests=args.max_requests,
        )
    except Exception as exc:
        print(redact(str(exc), token))
        return 1
    print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
