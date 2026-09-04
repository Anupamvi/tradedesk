"""Persist planner inputs and actual holdings for the local dashboard."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from compoundcore.projections import path_table_from_rates
from compoundcore.sleeve import (
    ASOF,
    FEE,
    NVDA_WEIGHT,
    ROLES,
    SCENARIOS,
    SLEEVE_NAMES,
    SMH_CRASH,
    TICKER_ORDER,
    bands,
    rates_from_weights,
    weights,
)


STATE_VERSION = 1


def default_state_path() -> Path:
    env = os.environ.get("COMPOUNDCORE_STATE", "").strip()
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent / "var" / "dashboard.json"


def empty_state() -> Dict[str, Any]:
    return {
        "version": STATE_VERSION,
        "planner": {"amount": 0.0, "weekly": 0.0, "monthly": 0.0},
        "book": {
            "holdings": {ticker: 0.0 for ticker in TICKER_ORDER},
            "monthly_add": 0.0,
            "compare_to": "default",
            "submitted_at": None,
        },
    }


def parse_money(raw: Any, field: str = "amount") -> float:
    if raw is None or raw == "":
        return 0.0
    if isinstance(raw, bool):
        raise ValueError("%s must be a dollar amount" % field)
    if isinstance(raw, (int, float)):
        value = float(raw)
    else:
        text = str(raw).strip().replace(",", "").replace("$", "")
        if text == "":
            return 0.0
        try:
            value = float(text)
        except ValueError as exc:
            raise ValueError("%s must be a dollar amount" % field) from exc
    if value < 0:
        raise ValueError("%s must be >= 0" % field)
    return value


def normalize_holdings(raw: Optional[Mapping[str, Any]]) -> Dict[str, float]:
    src = raw or {}
    out = {}
    for ticker in TICKER_ORDER:
        out[ticker] = round(parse_money(src.get(ticker, 0), ticker), 2)
    return out


def _require_sleeve(name: str) -> str:
    key = (name or "default").strip().lower()
    if key not in SLEEVE_NAMES:
        raise ValueError("compare_to must be default or aggressive")
    return key


def load_state(path: Optional[Path] = None) -> Dict[str, Any]:
    dest = Path(path) if path is not None else default_state_path()
    if not dest.exists():
        return empty_state()
    try:
        data = json.loads(dest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return empty_state()
    if not isinstance(data, dict):
        return empty_state()
    blank = empty_state()
    planner = data.get("planner") if isinstance(data.get("planner"), dict) else {}
    book = data.get("book") if isinstance(data.get("book"), dict) else {}
    try:
        holdings = normalize_holdings(book.get("holdings"))
        monthly_add = parse_money(book.get("monthly_add", 0), "monthly_add")
        compare_to = _require_sleeve(str(book.get("compare_to") or "default"))
        amount = parse_money(planner.get("amount", 0), "amount")
        weekly = parse_money(planner.get("weekly", 0), "weekly")
        monthly = parse_money(planner.get("monthly", 0), "monthly")
    except ValueError:
        return empty_state()
    submitted = book.get("submitted_at")
    if submitted is not None:
        submitted = str(submitted)
    return {
        "version": STATE_VERSION,
        "planner": {"amount": amount, "weekly": weekly, "monthly": monthly},
        "book": {
            "holdings": holdings,
            "monthly_add": monthly_add,
            "compare_to": compare_to,
            "submitted_at": submitted,
        },
    }


def save_state(state: Mapping[str, Any], path: Optional[Path] = None) -> Path:
    dest = Path(path) if path is not None else default_state_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": STATE_VERSION,
        "planner": {
            "amount": parse_money(state["planner"]["amount"], "amount"),
            "weekly": parse_money(state["planner"]["weekly"], "weekly"),
            "monthly": parse_money(state["planner"]["monthly"], "monthly"),
        },
        "book": {
            "holdings": normalize_holdings(state["book"]["holdings"]),
            "monthly_add": parse_money(state["book"]["monthly_add"], "monthly_add"),
            "compare_to": _require_sleeve(str(state["book"].get("compare_to") or "default")),
            "submitted_at": state["book"].get("submitted_at"),
        },
    }
    tmp = dest.with_name(dest.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(dest)
    return dest


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _us_of_equity(w: Mapping[str, float]) -> float:
    us = w["VOO"] + w["VGT"] + w["SMH"] + w["VB"]
    intl = w["VXUS"]
    equity = us + intl
    return us / equity if equity else 0.0


def book_view(
    holdings: Mapping[str, float],
    monthly_add: float = 0.0,
    compare_to: str = "default",
    submitted_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Actual mix vs a target sleeve, plus compounded paths on that mix."""
    sleeve = _require_sleeve(compare_to)
    held = normalize_holdings(holdings)
    total = round(sum(held.values()), 2)
    monthly = parse_money(monthly_add, "monthly_add")
    present = total > 0 and submitted_at is not None
    target = weights(sleeve)
    bmap = bands(sleeve)
    if total > 0:
        mix = {ticker: held[ticker] / total for ticker in TICKER_ORDER}
    else:
        mix = {ticker: 0.0 for ticker in TICKER_ORDER}

    rows = []
    for ticker in TICKER_ORDER:
        actual = mix[ticker]
        band = bmap[ticker]
        if total <= 0:
            status = "empty"
        elif actual < band.low:
            status = "low"
        elif actual > band.high:
            status = "high"
        else:
            status = "in"
        rows.append(
            {
                "ticker": ticker,
                "role": ROLES[ticker],
                "dollars": held[ticker],
                "weight": actual,
                "target": target[ticker],
                "gap_dollars": round(target[ticker] * total - held[ticker], 2),
                "band_low": band.low,
                "band_high": band.high,
                "status": status,
            }
        )

    nvda = sum(mix[t] * NVDA_WEIGHT.get(t, 0.0) for t in TICKER_ORDER)
    fee = sum(mix[t] * FEE[t] for t in TICKER_ORDER)
    crash = mix["SMH"] * SMH_CRASH
    rates = rates_from_weights(mix) if total > 0 else {
        "5y": {path: 0.0 for path in SCENARIOS},
        "10y": {path: 0.0 for path in SCENARIOS},
    }
    projections = path_table_from_rates(total, monthly, rates) if present else None

    return {
        "asof": ASOF,
        "present": present,
        "submitted_at": submitted_at,
        "total": total,
        "monthly_add": monthly,
        "compare_to": sleeve,
        "weights": mix,
        "rates": rates,
        "rows": rows,
        "nvda_weight": nvda,
        "nvda_dollars": total * nvda,
        "fee": fee,
        "us_of_equity": _us_of_equity(mix),
        "equity": mix["VOO"] + mix["VGT"] + mix["SMH"] + mix["VB"] + mix["VXUS"],
        "smh_crash_hit": crash,
        "smh_crash_dollars": total * crash,
        "projections": projections,
    }
