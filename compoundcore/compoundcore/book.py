"""Persist planner inputs and the real book: cost, marks, and P&L."""

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


STATE_VERSION = 2


def default_state_path() -> Path:
    env = os.environ.get("COMPOUNDCORE_STATE", "").strip()
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent / "var" / "dashboard.json"


def empty_position() -> Dict[str, float]:
    return {"cost": 0.0, "current": 0.0, "shares": 0.0}


def empty_state() -> Dict[str, Any]:
    return {
        "version": STATE_VERSION,
        "planner": {"amount": 0.0, "weekly": 0.0, "monthly": 0.0},
        "book": {
            "positions": {ticker: empty_position() for ticker in TICKER_ORDER},
            "monthly_add": 0.0,
            "compare_to": "default",
            "submitted_at": None,
            "marked_at": None,
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


def parse_qty(raw: Any, field: str = "shares") -> float:
    value = parse_money(raw, field)
    return round(value, 6)


def normalize_holdings(raw: Optional[Mapping[str, Any]]) -> Dict[str, float]:
    src = raw or {}
    return {ticker: round(parse_money(src.get(ticker, 0), ticker), 2) for ticker in TICKER_ORDER}


def normalize_position(raw: Optional[Mapping[str, Any]], ticker: str) -> Dict[str, float]:
    src = raw if isinstance(raw, dict) else {}
    return {
        "cost": round(parse_money(src.get("cost", 0), "%s cost" % ticker), 2),
        "current": round(parse_money(src.get("current", 0), "%s current" % ticker), 2),
        "shares": parse_qty(src.get("shares", 0), "%s shares" % ticker),
    }


def normalize_positions(
    raw: Optional[Mapping[str, Any]] = None,
    holdings: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Dict[str, float]]:
    src = raw if isinstance(raw, dict) else {}
    legacy = normalize_holdings(holdings) if holdings is not None else None
    out = {}
    for ticker in TICKER_ORDER:
        pos = normalize_position(src.get(ticker), ticker)
        if legacy is not None and pos["cost"] == 0 and pos["current"] == 0 and pos["shares"] == 0:
            pos["cost"] = legacy[ticker]
        out[ticker] = pos
    return out


def holdings_from_positions(positions: Mapping[str, Mapping[str, float]]) -> Dict[str, float]:
    return {ticker: float(positions.get(ticker, {}).get("cost", 0.0) or 0.0) for ticker in TICKER_ORDER}


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
    planner = data.get("planner") if isinstance(data.get("planner"), dict) else {}
    book = data.get("book") if isinstance(data.get("book"), dict) else {}
    try:
        positions = normalize_positions(book.get("positions"), book.get("holdings"))
        monthly_add = parse_money(book.get("monthly_add", 0), "monthly_add")
        compare_to = _require_sleeve(str(book.get("compare_to") or "default"))
        amount = parse_money(planner.get("amount", 0), "amount")
        weekly = parse_money(planner.get("weekly", 0), "weekly")
        monthly = parse_money(planner.get("monthly", 0), "monthly")
    except ValueError:
        return empty_state()
    submitted = book.get("submitted_at")
    marked = book.get("marked_at")
    if submitted is not None:
        submitted = str(submitted)
    if marked is not None:
        marked = str(marked)
    return {
        "version": STATE_VERSION,
        "planner": {"amount": amount, "weekly": weekly, "monthly": monthly},
        "book": {
            "positions": positions,
            "monthly_add": monthly_add,
            "compare_to": compare_to,
            "submitted_at": submitted,
            "marked_at": marked,
        },
    }


def save_state(state: Mapping[str, Any], path: Optional[Path] = None) -> Path:
    dest = Path(path) if path is not None else default_state_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    positions = normalize_positions(state["book"].get("positions"), state["book"].get("holdings"))
    payload = {
        "version": STATE_VERSION,
        "planner": {
            "amount": parse_money(state["planner"]["amount"], "amount"),
            "weekly": parse_money(state["planner"]["weekly"], "weekly"),
            "monthly": parse_money(state["planner"]["monthly"], "monthly"),
        },
        "book": {
            "positions": positions,
            "holdings": holdings_from_positions(positions),
            "monthly_add": parse_money(state["book"]["monthly_add"], "monthly_add"),
            "compare_to": _require_sleeve(str(state["book"].get("compare_to") or "default")),
            "submitted_at": state["book"].get("submitted_at"),
            "marked_at": state["book"].get("marked_at"),
        },
    }
    tmp = dest.with_name(dest.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(dest)
    return dest


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def mark_with_prices(
    positions: Mapping[str, Mapping[str, float]],
    prices: Mapping[str, float],
) -> Dict[str, Dict[str, float]]:
    out = normalize_positions(positions)
    for ticker in TICKER_ORDER:
        shares = out[ticker]["shares"]
        try:
            px = float(prices.get(ticker, 0) or 0)
        except (TypeError, ValueError):
            px = 0.0
        if shares > 0 and px > 0:
            out[ticker]["current"] = round(shares * px, 2)
    return out


def _us_of_equity(w: Mapping[str, float]) -> float:
    us = w["VOO"] + w["VGT"] + w["SMH"] + w["VB"]
    intl = w["VXUS"]
    equity = us + intl
    return us / equity if equity else 0.0


def book_view(
    holdings: Optional[Mapping[str, float]] = None,
    monthly_add: float = 0.0,
    compare_to: str = "default",
    submitted_at: Optional[str] = None,
    positions: Optional[Mapping[str, Mapping[str, float]]] = None,
    marked_at: Optional[str] = None,
    quote_status: Optional[str] = None,
) -> Dict[str, Any]:
    """Saved book: invested vs now, real P&L, mix vs a target sleeve, CMA as a footnote."""
    sleeve = _require_sleeve(compare_to)
    pos = normalize_positions(positions, holdings)
    monthly = parse_money(monthly_add, "monthly_add")
    invested = round(sum(pos[t]["cost"] for t in TICKER_ORDER), 2)
    present = invested > 0 and submitted_at is not None

    rows = []
    marked_cost = 0.0
    marked_now = 0.0
    unmarked_cost = 0.0
    for ticker in TICKER_ORDER:
        cost = pos[ticker]["cost"]
        current = pos[ticker]["current"]
        shares = pos[ticker]["shares"]
        held = cost > 0 or current > 0 or shares > 0
        marked = (not held) or current > 0
        if held and marked:
            marked_cost += cost
            marked_now += current
            pnl = round(current - cost, 2)
            pnl_pct = (pnl / cost) if cost > 0 else None
        elif held and not marked:
            unmarked_cost += cost
            pnl = None
            pnl_pct = None
        else:
            pnl = 0.0
            pnl_pct = None
        rows.append(
            {
                "ticker": ticker,
                "role": ROLES[ticker],
                "cost": cost,
                "current": current,
                "shares": shares,
                "dollars": current if marked and current > 0 else cost,
                "marked": marked and held,
                "held": held,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
            }
        )

    mix_total = round(sum(row["dollars"] for row in rows), 2)
    mix_source = "mark" if unmarked_cost == 0 and marked_now > 0 else "cost"
    if mix_total > 0:
        mix = {ticker: (rows[i]["dollars"] / mix_total) for i, ticker in enumerate(TICKER_ORDER)}
    else:
        mix = {ticker: 0.0 for ticker in TICKER_ORDER}

    target = weights(sleeve)
    bmap = bands(sleeve)
    for i, ticker in enumerate(TICKER_ORDER):
        actual = mix[ticker]
        band = bmap[ticker]
        if mix_total <= 0:
            status = "empty"
        elif actual < band.low:
            status = "low"
        elif actual > band.high:
            status = "high"
        else:
            status = "in"
        rows[i]["weight"] = actual
        rows[i]["target"] = target[ticker]
        rows[i]["gap_dollars"] = round(target[ticker] * mix_total - rows[i]["dollars"], 2)
        rows[i]["band_low"] = band.low
        rows[i]["band_high"] = band.high
        rows[i]["status"] = status

    pnl_ready = present and marked_cost > 0 and unmarked_cost == 0
    pnl_dollars = round(marked_now - marked_cost, 2) if marked_cost > 0 else None
    pnl_pct = (pnl_dollars / marked_cost) if pnl_dollars is not None and marked_cost > 0 else None
    market = marked_now if marked_now > 0 else None

    nvda = sum(mix[t] * NVDA_WEIGHT.get(t, 0.0) for t in TICKER_ORDER)
    fee = sum(mix[t] * FEE[t] for t in TICKER_ORDER)
    crash = mix["SMH"] * SMH_CRASH
    rates = rates_from_weights(mix) if mix_total > 0 else {
        "5y": {path: 0.0 for path in SCENARIOS},
        "10y": {path: 0.0 for path in SCENARIOS},
    }
    path_principal = market if market else invested
    projections = path_table_from_rates(path_principal, monthly, rates) if present else None

    return {
        "asof": ASOF,
        "present": present,
        "submitted_at": submitted_at,
        "marked_at": marked_at,
        "quote_status": quote_status,
        "total": invested,
        "invested": invested,
        "market": market,
        "pnl": pnl_dollars,
        "pnl_pct": pnl_pct,
        "pnl_ready": pnl_ready,
        "unmarked_cost": round(unmarked_cost, 2),
        "mix_source": mix_source,
        "monthly_add": monthly,
        "compare_to": sleeve,
        "weights": mix,
        "rates": rates,
        "rows": rows,
        "positions": pos,
        "nvda_weight": nvda,
        "nvda_dollars": mix_total * nvda,
        "fee": fee,
        "us_of_equity": _us_of_equity(mix),
        "equity": mix["VOO"] + mix["VGT"] + mix["SMH"] + mix["VB"] + mix["VXUS"],
        "smh_crash_hit": crash,
        "smh_crash_dollars": mix_total * crash,
        "projections": projections,
    }
