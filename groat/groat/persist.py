"""Prior-session TRADE / analog state. Freshness and analog persist read this."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from groat.num import to_float


def _read_json(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def payload_incomplete(payload: Optional[dict], path: Optional[Path] = None) -> bool:
    """Morning / session_incomplete boards are not the last TRADE session."""
    loc = Path(path) if path is not None else None
    if loc is not None and loc.parent.name == "open":
        return True
    if not isinstance(payload, dict):
        return True
    if payload.get("session_incomplete") is True:
        return True
    if str(payload.get("session") or "") == "open":
        return True
    if (
        loc is not None
        and loc.name == "candidates.json"
        and loc.parent.name not in ("close", "rth", "open")
    ):
        day = loc.parent
        only_open = (day / "open" / "candidates.json").is_file() and not (
            day / "close" / "candidates.json"
        ).is_file() and not (day / "rth" / "candidates.json").is_file()
        if only_open and str(payload.get("session") or "") not in ("rth", "close"):
            return True
    rows = _rows(payload)
    if any(r.get("action") == "TRADE" for r in rows):
        return False
    return any("session_incomplete" in (r.get("reasons") or []) for r in rows)


def _read_complete(path: Path) -> Optional[dict]:
    payload = _read_json(path)
    if payload is None or payload_incomplete(payload, path):
        return None
    return payload


def prior_candidates_path(out_dir: Path, asof: str) -> Optional[Path]:
    root = Path(out_dir)
    if not root.is_dir():
        return None
    try:
        day = datetime.strptime(asof[:10], "%Y-%m-%d")
    except (TypeError, ValueError):
        return None
    for i in range(1, 8):
        prev = (day - timedelta(days=i)).date().isoformat()
        folder = root / prev
        if not folder.is_dir():
            continue
        for rel in ("close/candidates.json", "candidates.json", "open/candidates.json"):
            path = folder / rel
            if path.is_file() and _read_complete(path) is not None:
                return path
    return None


def iter_prior_payloads(out_dir: Optional[Path], asof: str, session: Optional[str] = None) -> List[dict]:
    """Complete boards this run may treat as last session. Newest same-day first, then last complete prior day.

    Incomplete morning open/ is never the TRADE prior. Evening unions yesterday close even if this
    morning already ran and parked everyone for session_incomplete.
    """
    if out_dir is None:
        return []
    root = Path(out_dir)
    day = str(asof or "")[:10]
    found: List[dict] = []
    if session == "close" and day:
        # This evening's DATE/candidates.json is the same session. Do not freshness-park
        # against a close re-run (xintel fill, second print). RTH earlier today still counts.
        payload = _read_complete(root / day / "rth" / "candidates.json")
        if payload is not None:
            found.append(payload)
    elif session == "rth" and day:
        folder = root / day
        for rel in ("candidates.json", "rth/candidates.json"):
            payload = _read_complete(folder / rel)
            if payload is not None:
                found.append(payload)
                break
    path = prior_candidates_path(root, asof)
    if path is not None:
        payload = _read_complete(path)
        if payload is not None:
            found.append(payload)
    return found


def load_prior_payload(out_dir: Optional[Path], asof: str, session: Optional[str] = None) -> Optional[dict]:
    """First complete prior board (same-day complete, else last complete prior day)."""
    payloads = iter_prior_payloads(out_dir, asof, session)
    return payloads[0] if payloads else None


def load_prior_state(
    out_dir: Optional[Path], asof: str, session: Optional[str] = None
) -> Tuple[List[dict], Dict[Tuple[str, str], dict]]:
    """Union TRADE + analog vetoes from complete same-day and last complete prior-day boards."""
    payloads = iter_prior_payloads(out_dir, asof, session)
    trades: List[dict] = []
    seen = set()
    analog: Dict[Tuple[str, str], dict] = {}
    for payload in reversed(payloads):
        analog.update(extract_prior_analog(payload))
    for payload in payloads:
        for row in extract_prior_trades(payload):
            key = (row["ticker"], row["primary"])
            if key in seen:
                continue
            seen.add(key)
            trades.append(row)
    return trades, analog


def _rows(payload: Optional[dict]) -> List[dict]:
    if not isinstance(payload, dict):
        return []
    rows = payload.get("candidates") or payload.get("board") or []
    return [r for r in rows if isinstance(r, dict)]


def extract_prior_trades(payload: Optional[dict]) -> List[dict]:
    out = []
    for row in _rows(payload):
        if row.get("action") != "TRADE":
            continue
        ticker = str(row.get("ticker") or "").upper()
        primary = str(row.get("primary") or "")
        if not ticker or not primary:
            continue
        out.append(
            {
                "ticker": ticker,
                "primary": primary,
                "group_status": row.get("group_status") or "",
                "choice": row.get("choice") or "",
            }
        )
    return out


def extract_prior_analog(payload: Optional[dict]) -> Dict[Tuple[str, str], dict]:
    out = {}
    for row in _rows(payload):
        ticker = str(row.get("ticker") or "").upper()
        primary = str(row.get("primary") or "")
        if not ticker or not primary:
            continue
        ev = row.get("evidence") if isinstance(row.get("evidence"), dict) else {}
        veto = ev.get("analog_veto")
        reasons = row.get("reasons") or []
        if not veto:
            for reason in reasons:
                if str(reason).startswith("analog_"):
                    veto = reason
                    break
        if not veto:
            continue
        stock = ev.get("stock") if isinstance(ev.get("stock"), dict) else ev
        dates = [d for d in (ev.get("analog_dates") or []) if d]
        if not dates:
            dates = [
                h.get("date")
                for h in (ev.get("hits") or [])
                if isinstance(h, dict) and h.get("date")
            ]
        out[(ticker, primary)] = {
            "ticker": ticker,
            "primary": primary,
            "veto": str(veto),
            "n": int(stock.get("n") or 0),
            "wins": int(stock.get("wins") or 0),
            "avg_r": to_float(stock.get("avg_r")),
            "stock": stock,
            "dates": dates,
        }
    return out


def analog_key(ticker: str, primary: str) -> Tuple[str, str]:
    return (str(ticker or "").upper(), str(primary or ""))


def copy_session_artifacts(day: Path, session: str) -> Optional[Path]:
    """Keep open / rth / close copies so a later run does not erase the earlier board."""
    if session not in ("open", "close", "rth"):
        return None
    dest = Path(day) / session
    dest.mkdir(parents=True, exist_ok=True)
    names = (
        "board.md",
        "report.md",
        "regime.md",
        "evidence.md",
        "evidence.json",
        "candidates.json",
        "board.csv",
        "manifest.json",
        "x_queue.json",
        "sectors.md",
        "rejections.csv",
    )
    for name in names:
        src = Path(day) / name
        if src.is_file():
            shutil.copy2(src, dest / name)
    return dest
