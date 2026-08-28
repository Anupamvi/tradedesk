"""X-HOT overlay. Agent writes var/xhot/DATE/hot.json. Missing → DATA UNAVAILABLE."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from wheelo.config import CODE_DIR


def hot_path(asof: str) -> Path:
    return CODE_DIR / "var" / "xhot" / asof / "hot.json"


def load_hot(asof: str) -> Dict[str, dict]:
    path = hot_path(asof)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    rows = payload.get("names") if isinstance(payload, dict) else payload
    out = {}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        ticker = str(row.get("ticker") or "").upper()
        if not ticker:
            continue
        out[ticker] = {
            "ticker": ticker,
            "heat": str(row.get("heat") or "hot"),
            "bias": str(row.get("bias") or "unknown"),
            "posts_24h": row.get("posts_24h"),
            "narrative": str(row.get("narrative") or ""),
            "tag": str(row.get("tag") or "DATA UNAVAILABLE"),
            "source": str(row.get("source") or "xhot"),
        }
    return out


def write_hot(asof: str, names: List[dict], source: str = "x_keyword_search") -> Path:
    path = hot_path(asof)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {"asof": asof, "source": source, "names": names}
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
