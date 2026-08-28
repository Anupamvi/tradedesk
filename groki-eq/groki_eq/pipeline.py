"""breakout_eq daily board. 0 or 1 Execute."""

from collections import Counter
from typing import Dict, List, Optional, Sequence

from groki_eq import config
from groki_eq.book import can_open_new, load_book, record_execute, save_book, week_key
from groki_eq.breakout import atr_wilder, is_breakout, pct_above, prior_high, share_count, stop_price
from groki_eq.config import (
    ACCOUNT_DOLLARS,
    EXECUTE_CAP,
    RISK_PCT,
    SLEEVE,
    TIE_ORDER,
    UNIVERSE,
)
from groki_eq.orats import load_usage
from groki_eq.prices import ensure_bars


def load_universe(path=None) -> List[str]:
    target = path or config.UNIVERSE_PATH
    names = []
    if target.is_file():
        for raw in target.read_text(encoding="utf-8").splitlines():
            line = raw.strip().upper()
            if not line or line.startswith("#"):
                continue
            names.append(line)
    return names or list(UNIVERSE)


def pick_one(eligible: Sequence[dict]) -> Optional[dict]:
    if EXECUTE_CAP != 1:
        raise RuntimeError("EXECUTE_CAP is frozen at 1")
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda row: (
            float(row.get("pct_above") or 0.0),
            -int(TIE_ORDER.get(str(row.get("ticker") or "").upper(), 9)),
        ),
    )


def screen_ticker(asof: str, ticker: str, bars: List[dict]) -> dict:
    upto = [b for b in bars if b["date"] <= asof]
    reasons = []
    if not upto or upto[-1]["date"] != asof:
        reasons.append("missing_bars")
        return {
            "asof_date": asof,
            "ticker": ticker,
            "structure": "shares",
            "sleeve": SLEEVE,
            "close": "",
            "high_20": "",
            "pct_above": "",
            "atr14": "",
            "stop": "",
            "shares": "",
            "risk_dollars": ACCOUNT_DOLLARS * RISK_PCT,
            "notional": "",
            "pass_screen": False,
            "reasons": " ".join(reasons),
            "_eligible": False,
            "action": "IGNORE",
            "decision_pass": "false",
        }
    close = float(upto[-1]["close"])
    high_20 = prior_high(bars, asof)
    atr = atr_wilder(bars, asof)
    if high_20 is None:
        reasons.append("missing_bars")
    if atr is None:
        reasons.append("missing_atr")
    broke = high_20 is not None and is_breakout(close, high_20)
    if high_20 is not None and not broke:
        reasons.append("no_breakout")
    shares = share_count(atr) if atr is not None else 0
    stop = stop_price(close, atr) if atr is not None else None
    if atr is not None and shares < 1:
        reasons.append("size_infeasible")
    pct = pct_above(close, high_20) if high_20 else None
    row = {
        "asof_date": asof,
        "ticker": ticker,
        "structure": "shares",
        "sleeve": SLEEVE,
        "close": close,
        "high_20": high_20 if high_20 is not None else "",
        "pct_above": pct if pct is not None else "",
        "atr14": atr if atr is not None else "",
        "stop": stop if stop is not None else "",
        "shares": shares if shares else "",
        "risk_dollars": ACCOUNT_DOLLARS * RISK_PCT,
        "notional": shares * close if shares else "",
        "pass_screen": not reasons,
        "reasons": " ".join(reasons),
        "_eligible": not reasons,
        "action": "WATCH" if not reasons else "IGNORE",
        "decision_pass": "false",
    }
    return row


def _blocker(execute_count: int, rejections: List[dict], eligible_n: int) -> str:
    if execute_count:
        return ""
    tokens = Counter()
    for row in rejections:
        for token in str(row.get("reasons") or "").split():
            tokens[token] += 1
    mapping = (
        ("missing_bars", "missing_bars"),
        ("no_breakout", "no_breakout"),
        ("week_cap", "week_cap"),
        ("open_cap", "open_cap"),
        ("already_open", "already_open"),
        ("size_infeasible", "size_infeasible"),
        ("missing_atr", "missing_atr"),
    )
    for token, label in mapping:
        if tokens.get(token):
            return label
    if eligible_n == 0:
        return "no_pass"
    return "no_pass"


def _public(row: dict) -> dict:
    return {k: v for k, v in row.items() if not str(k).startswith("_")}


def build_day(
    asof: str,
    token: str,
    out_dir=None,
    getter=None,
    max_requests: Optional[int] = None,
    liquid: Optional[Sequence[str]] = None,
    book: Optional[dict] = None,
    schwab_bars: Optional[Dict[str, list]] = None,
    bars_by_ticker: Optional[Dict[str, list]] = None,
) -> Dict[str, object]:
    names = list(liquid) if liquid is not None else load_universe()
    schwab_bars = schwab_bars or {}
    tapes = {}
    http_n = 0
    built = []
    rejections = []
    bars_ok = 0
    for ticker in names:
        if bars_by_ticker is not None and ticker in bars_by_ticker:
            bars = bars_by_ticker.get(ticker) or []
            tapes[ticker] = "replay"
            pack = {"error": ""}
        else:
            pack = ensure_bars(
                ticker,
                token,
                getter=getter,
                max_requests=max_requests,
                schwab_bars=schwab_bars.get(ticker),
                asof=asof,
            )
            tapes[ticker] = pack.get("tape")
            http_n += int(pack.get("http") or 0)
            bars = pack.get("bars") or []
        if bars:
            bars_ok += 1
        row = screen_ticker(asof, ticker, bars)
        if pack.get("error") and not bars:
            row["reasons"] = (row.get("reasons") + " " + str(pack.get("error"))).strip()
            row["pass_screen"] = False
            row["_eligible"] = False
            row["action"] = "IGNORE"
        built.append(row)
    book = book if book is not None else (load_book(out_dir) if out_dir is not None else {"open": [], "week_entries": {}})
    eligible = [row for row in built if row.get("_eligible")]
    winner = pick_one(eligible)
    board = []
    for row in built:
        if winner is not None and row is winner:
            caps = can_open_new(book, asof, row["ticker"])
            if caps:
                row["action"] = "WATCH"
                row["decision_pass"] = "false"
                extra = " ".join(caps)
                row["reasons"] = (str(row.get("reasons") or "") + " " + extra).strip()
                board.append(row)
                continue
            row["action"] = "EXECUTE"
            row["decision_pass"] = "true"
            board.append(row)
            continue
        if row.get("_eligible"):
            row["action"] = "WATCH"
            row["decision_pass"] = "false"
            row["reasons"] = (str(row.get("reasons") or "") + " execute_cap").strip()
            board.append(row)
            continue
        rejections.append(
            {
                "asof_date": asof,
                "ticker": row.get("ticker"),
                "structure": "shares",
                "reasons": row.get("reasons") or "ignore",
                "stage": "screen",
            }
        )
        if row.get("action") == "IGNORE":
            continue
        board.append(row)
    execute_rows = [row for row in board if row.get("action") == "EXECUTE"]
    if execute_rows:
        record_execute(book, execute_rows[0])
        if out_dir is not None:
            save_book(out_dir, book)
    watch_rows = [row for row in board if row.get("action") == "WATCH"]
    execute_count = len(execute_rows)
    usage = load_usage()
    execute_ticker = execute_rows[0]["ticker"] if execute_rows else ""
    execute_target = ""
    if execute_rows:
        execute_target = "shares=%s stop=$%s" % (
            execute_rows[0].get("shares") or "",
            execute_rows[0].get("stop") or "",
        )
    return {
        "board": [_public(row) for row in board],
        "rejections": rejections,
        "execute_rows": [_public(row) for row in execute_rows],
        "watch_rows": [_public(row) for row in watch_rows],
        "execute_count": execute_count,
        "execute_ticker": execute_ticker,
        "execute_target": execute_target,
        "watch_count": len(watch_rows),
        "blocker": _blocker(execute_count, rejections, len(eligible)),
        "selector": "breakout_eq",
        "orats_ok": 1 if bars_ok else 0,
        "orats_http": http_n,
        "orats_rows": bars_ok,
        "orats_requests_used": usage.get("used") or 0,
        "orats_requests_left": usage.get("left") or 0,
        "tapes": tapes,
        "book_open": len(book.get("open") or []),
        "week_new": int((book.get("week_entries") or {}).get(week_key(asof)) or 0) if asof else 0,
        "orats_error": "",
    }
