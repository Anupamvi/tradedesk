import json
from datetime import date
from pathlib import Path
from typing import Dict, List

from groki_eq.config import MAX_NEW_PER_WEEK, MAX_OPEN, SLEEVE


def week_key(asof: str) -> str:
    day = date.fromisoformat(asof)
    year, week, _ = day.isocalendar()
    return "%d-W%02d" % (year, week)


def book_path(out_dir: Path) -> Path:
    return Path(out_dir) / "book.json"


def load_book(out_dir: Path) -> dict:
    path = book_path(out_dir)
    if not path.is_file():
        return {"open": [], "week_entries": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"open": [], "week_entries": {}}
    if not isinstance(payload, dict):
        return {"open": [], "week_entries": {}}
    payload.setdefault("open", [])
    payload.setdefault("week_entries", {})
    return payload


def save_book(out_dir: Path, book: dict) -> None:
    path = book_path(out_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(book, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def open_tickers(book: dict) -> List[str]:
    return [str(row.get("ticker") or "").upper() for row in (book.get("open") or []) if row.get("ticker")]


def week_new_count(book: dict, asof: str) -> int:
    try:
        return int((book.get("week_entries") or {}).get(week_key(asof)) or 0)
    except (TypeError, ValueError):
        return 0


def can_open_new(book: dict, asof: str, ticker: str) -> List[str]:
    reasons = []
    if week_new_count(book, asof) >= MAX_NEW_PER_WEEK:
        reasons.append("week_cap")
    if len(book.get("open") or []) >= MAX_OPEN:
        reasons.append("open_cap")
    if str(ticker or "").upper() in open_tickers(book):
        reasons.append("already_open")
    return reasons


def close_open(book: dict, ticker: str, entry_date: str) -> dict:
    rest = []
    for row in book.get("open") or []:
        if (
            str(row.get("ticker") or "").upper() == str(ticker or "").upper()
            and str(row.get("entry_date") or "")[:10] == str(entry_date or "")[:10]
        ):
            continue
        rest.append(row)
    book["open"] = rest
    return book


def record_execute(book: dict, row: dict) -> dict:
    asof = str(row.get("asof_date") or "")
    open_rows: List[Dict] = list(book.get("open") or [])
    open_rows.append(
        {
            "sleeve": SLEEVE,
            "ticker": row.get("ticker"),
            "entry_date": asof,
            "entry": row.get("close"),
            "stop": row.get("stop"),
            "shares": row.get("shares"),
        }
    )
    week_entries = dict(book.get("week_entries") or {})
    key = week_key(asof)
    week_entries[key] = int(week_entries.get(key) or 0) + 1
    book["open"] = open_rows
    book["week_entries"] = week_entries
    return book
