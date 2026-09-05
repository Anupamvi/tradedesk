"""News and filings are agent files, same pattern as X. Never invent a catalyst."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from groat.config import CODE_DIR

UNAVAILABLE = "DATA UNAVAILABLE"


def catalyst_path(kind: str, asof: str, ticker: str) -> Path:
    return CODE_DIR / "var" / kind / asof[:10] / ("%s.json" % str(ticker or "").upper())


def load_catalyst(kind: str, asof: str, ticker: str) -> Dict[str, Any]:
    """kind is news or filings. Missing file stays DATA UNAVAILABLE."""
    name = str(ticker or "").upper()
    empty = {
        "ticker": name,
        "asof": asof,
        "kind": kind,
        "summary": UNAVAILABLE,
        "source": None,
        "notes": "%s not fetched. Skill layer may write var/%s/%s/%s.json." % (kind, kind, asof[:10], name),
    }
    if kind not in ("news", "filings") or not name:
        return empty
    path = catalyst_path(kind, asof, name)
    if not path.is_file():
        return empty
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return dict(empty, notes="%s file unreadable" % kind, source=str(path))
    if not isinstance(payload, dict):
        return dict(empty, notes="%s file unreadable" % kind, source=str(path))
    summary = str(payload.get("summary") or payload.get("notes") or "").strip() or UNAVAILABLE
    return {
        "ticker": name,
        "asof": asof,
        "kind": kind,
        "summary": summary,
        "source": payload.get("source") or ("%s_file" % kind),
        "notes": payload.get("notes") or "",
    }
