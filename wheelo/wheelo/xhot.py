"""X-HOT overlay. Fresh fetch every run. Disk file is not an execute cache."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from wheelo.config import CODE_DIR


def hot_path(asof: str) -> Path:
    return CODE_DIR / "var" / "xhot" / asof / "hot.json"


def clear_hot(asof: str) -> bool:
    """Drop the dated hot.json so a new run cannot execute off a leftover file."""
    path = hot_path(asof)
    if not path.is_file():
        return False
    path.unlink()
    return True


def hot_is_fresh(asof: str, artifact: Path) -> bool:
    """True when hot.json exists and is not older than the scan artifact."""
    path = hot_path(asof)
    if not path.is_file():
        return False
    if not artifact.is_file():
        return True
    return path.stat().st_mtime >= artifact.stat().st_mtime


def load_hot(asof: str, newer_than: Optional[Path] = None) -> Dict[str, dict]:
    path = hot_path(asof)
    if not path.is_file():
        return {}
    if newer_than is not None and not hot_is_fresh(asof, newer_than):
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if isinstance(payload, dict) and payload.get("asof") and str(payload.get("asof"))[:10] != asof[:10]:
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
    body = {
        "asof": asof,
        "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": source,
        "names": names,
    }
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
