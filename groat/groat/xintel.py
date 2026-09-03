"""X.com intel is confirm/veto, not a trigger. Missing file → DATA UNAVAILABLE."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from groat.config import CODE_DIR


def xintel_dir(asof: str) -> Path:
    return CODE_DIR / "var" / "xintel" / asof


def xintel_path(asof: str, ticker: str) -> Path:
    return xintel_dir(asof) / ("%s.json" % str(ticker).upper())


def load_xintel(asof: str, ticker: str) -> Dict[str, Any]:
    path = xintel_path(asof, ticker)
    if not path.is_file():
        return {
            "ticker": str(ticker).upper(),
            "asof": asof,
            "tag": "DATA UNAVAILABLE",
            "posts_24h": None,
            "notes": "X not fetched. Skill layer must search $TICKER and write this file.",
            "source": None,
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {
            "ticker": str(ticker).upper(),
            "asof": asof,
            "tag": "DATA UNAVAILABLE",
            "posts_24h": None,
            "notes": "X file unreadable",
            "source": str(path),
        }
    tag = str(payload.get("tag") or "DATA UNAVAILABLE")
    return {
        "ticker": str(ticker).upper(),
        "asof": asof,
        "tag": tag,
        "posts_24h": payload.get("posts_24h"),
        "notes": payload.get("notes") or "",
        "source": payload.get("source") or "xintel_file",
        "crowded": tag.lower() == "crowded",
    }


def missing_x_tickers(rows: Sequence[dict]) -> List[str]:
    """TRADE/WATCH names with no Quiet|Informed|Crowded file."""
    out = []
    seen = set()
    for row in rows or []:
        tag = str(row.get("x") or "DATA UNAVAILABLE")
        if tag not in ("", "DATA UNAVAILABLE"):
            continue
        ticker = str(row.get("ticker") or "").upper()
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        out.append(ticker)
    return out


def write_xintel(asof: str, ticker: str, payload: Dict[str, Any]) -> Path:
    path = xintel_path(asof, ticker)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "ticker": str(ticker).upper(),
        "asof": asof,
        "tag": payload.get("tag") or "DATA UNAVAILABLE",
        "posts_24h": payload.get("posts_24h"),
        "notes": payload.get("notes") or "",
        "source": payload.get("source") or "x_keyword_search",
    }
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def classify_from_counts(posts_24h: Optional[int], promo: bool = False, informed: bool = False) -> str:
    if posts_24h is None:
        return "DATA UNAVAILABLE"
    if promo or (posts_24h >= 80):
        return "Crowded"
    if informed or posts_24h >= 8:
        return "Informed"
    return "Quiet"
