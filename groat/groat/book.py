from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from groat.config import BOOK_PATH


def load_book(path: Optional[Path] = None) -> Dict[str, Any]:
    target = path or BOOK_PATH
    if not target.is_file():
        return {"positions": []}
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"positions": []}
    if not isinstance(payload, dict):
        return {"positions": []}
    rows = payload.get("positions") or []
    payload["positions"] = [r for r in rows if isinstance(r, dict)]
    return payload


def save_book(book: Dict[str, Any], path: Optional[Path] = None) -> Path:
    target = path or BOOK_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(book, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def positions(path: Optional[Path] = None) -> List[dict]:
    return list(load_book(path).get("positions") or [])


def parse_occ(symbol: str) -> dict:
    raw = str(symbol or "").replace(" ", "").upper()
    if not raw:
        return {"underlying": "", "right": None, "expiry": None, "strike": None}
    for i in range(1, 7):
        rest = raw[i:]
        if len(rest) >= 15 and rest[:6].isdigit() and rest[6] in ("C", "P") and rest[7:15].isdigit():
            return {
                "underlying": raw[:i],
                "expiry": "20%s-%s-%s" % (rest[:2], rest[2:4], rest[4:6]),
                "right": "call" if rest[6] == "C" else "put",
                "strike": int(rest[7:15]) / 1000.0,
            }
    return {"underlying": raw, "right": None, "expiry": None, "strike": None}


def underlying_symbol(symbol: str) -> str:
    parsed = parse_occ(symbol)
    if parsed.get("underlying") and parsed.get("right"):
        return parsed["underlying"]
    raw = str(symbol or "").strip().upper()
    if not raw:
        return ""
    if " " in raw:
        return raw.split()[0].strip()
    if len(raw) >= 15:
        head = raw[:6].strip()
        if head.isalpha():
            return head
    return raw


def same_ticket(book_pos: dict, picked: Optional[dict]) -> bool:
    if not book_pos or not isinstance(picked, dict):
        return False
    b_exp = str(book_pos.get("expiry") or "")[:10]
    p_exp = str(picked.get("expiry") or "")[:10]
    if not b_exp or not p_exp or b_exp != p_exp:
        return False
    b_long = book_pos.get("long_strike")
    p_long = picked.get("long_strike") or picked.get("strike")
    if b_long is not None and p_long is not None and abs(float(b_long) - float(p_long)) > 0.011:
        return False
    b_short = book_pos.get("short_strike")
    p_short = picked.get("short_strike")
    if b_short is not None and p_short is not None and abs(float(b_short) - float(p_short)) > 0.011:
        return False
    return True


def open_group_sets(path: Optional[Path] = None):
    """Open book tickers and their industry groups. Skips other/index/macro."""
    from groat.config import ticker_group

    groups = set()
    tickers = set()
    for pos in positions(path):
        ticker = underlying_symbol(str(pos.get("ticker") or "")).upper()
        if not ticker:
            continue
        tickers.add(ticker)
        group = ticker_group(ticker)
        if group and group not in ("other", "index", "macro"):
            groups.add(group)
    return groups, tickers


def book_index(path: Optional[Path] = None) -> Dict[str, dict]:
    out = {}
    for pos in positions(path):
        ticker = underlying_symbol(str(pos.get("ticker") or ""))
        if not ticker:
            continue
        out[ticker] = {
            "in_book": True,
            "source": "book",
            "structure": pos.get("structure") or pos.get("instrument") or "",
            "entry": pos.get("entry") or pos.get("entry_dollars"),
            "opened": pos.get("opened"),
            "expiry": pos.get("expiry"),
            "long_strike": pos.get("long_strike"),
            "short_strike": pos.get("short_strike"),
            "instrument": pos.get("instrument"),
        }
    return out


def schwab_held_index(rows: Optional[List[dict]] = None) -> Dict[str, dict]:
    out = {}
    for pos in rows or []:
        ticker = underlying_symbol(str(pos.get("ticker") or ""))
        if not ticker:
            continue
        slot = out.setdefault(ticker, {"held_schwab": True, "source": "schwab", "legs": []})
        occ = parse_occ(str(pos.get("symbol") or pos.get("ticker") or ""))
        slot["legs"].append(
            {
                "symbol": pos.get("ticker"),
                "asset": pos.get("asset"),
                "quantity": pos.get("quantity"),
                "right": occ.get("right"),
                "expiry": occ.get("expiry"),
                "strike": occ.get("strike"),
            }
        )
    return out
