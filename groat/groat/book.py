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
