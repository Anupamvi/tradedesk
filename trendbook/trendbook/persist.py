"""Standing book: first-on-board date and weeks in Stage 2. Names stay until stage dies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from trendbook.config import CODE_DIR
from trendbook.dates import parse_ymd
from datetime import datetime


def book_dir() -> Path:
    return CODE_DIR / "var" / "book"


def _path(ticker: str) -> Path:
    return book_dir() / ("%s.json" % str(ticker).upper())


def load(ticker: str) -> Optional[dict]:
    path = _path(ticker)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def save(row: dict) -> None:
    ticker = str(row.get("ticker") or "").upper()
    if not ticker:
        return
    path = _path(ticker)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")


def _weeks_between(later: str, earlier: str) -> int:
    a = parse_ymd(later)
    b = parse_ymd(earlier)
    if not a or not b:
        return 0
    delta = datetime.strptime(a, "%Y-%m-%d") - datetime.strptime(b, "%Y-%m-%d")
    return max(0, int(round(delta.days / 7.0)))


def update(ticker: str, asof: str, on_board: bool, stage, run: Optional[dict] = None) -> dict:
    name = str(ticker).upper()
    prev = load(name) or {}
    run = run or {}
    if on_board:
        first = run.get("trend_start") or (prev.get("first_on_board") if prev.get("on_board") else None) or asof
        weeks = run.get("weeks_in_trend")
        if weeks is None:
            weeks = max(1, _weeks_between(asof, first) + 1)
        row = {
            "ticker": name,
            "on_board": True,
            "first_on_board": first,
            "weeks_in_stage": weeks,
            "px_at_start": run.get("px_at_start"),
            "last_stage": stage,
            "last_asof": asof,
            "left_on": prev.get("left_on"),
        }
        save(row)
        return row
    row = {
        "ticker": name,
        "on_board": False,
        "first_on_board": prev.get("first_on_board"),
        "weeks_in_stage": 0,
        "last_stage": stage,
        "last_asof": asof,
        "left_on": asof if prev.get("on_board") else prev.get("left_on"),
    }
    save(row)
    return row
