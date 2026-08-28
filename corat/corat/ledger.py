"""Append-only evidence ledger and original-thesis open-trade review."""

from __future__ import annotations

import fcntl
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from corat.constants import ACTIONABLE_NOW, TARGET_TRADE
from corat.store import canonical_json, read_json, sha256_bytes, utc_now


TRADE_STATUSES = {
    "RECOMMENDED", "SUBMITTED", "FILLED", "OPEN", "REDUCED", "CLOSED", "CANCELED", "EXPIRED", "REVIEW"
}
TERMINAL = {"CLOSED", "CANCELED", "EXPIRED"}
TRANSITIONS = {
    "RECOMMENDED": {"SUBMITTED", "CANCELED", "REVIEW"},
    "SUBMITTED": {"FILLED", "CANCELED", "REVIEW"},
    "FILLED": {"OPEN", "CLOSED", "REVIEW"},
    "OPEN": {"REDUCED", "CLOSED", "EXPIRED", "REVIEW"},
    "REDUCED": {"REDUCED", "CLOSED", "EXPIRED", "REVIEW"},
    "REVIEW": {"REVIEW", "SUBMITTED", "FILLED", "OPEN", "REDUCED", "CLOSED", "CANCELED", "EXPIRED"},
}


def ledger_path(state_root: Path) -> Path:
    return state_root / "trade_ledger.jsonl"


def read_events(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    events = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            event = json.loads(raw)
        except ValueError:
            raise ValueError("invalid trade ledger JSON at line {}".format(line_number)) from None
        if not isinstance(event, dict):
            raise ValueError("invalid trade ledger event at line {}".format(line_number))
        events.append(event)
    return events


def append_event(path: Path, event: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = canonical_json(dict(event)) + "\n"
    descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _validated_scaling(value: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    scaling = dict(value or {})
    if not scaling:
        return {}
    if not bool(scaling.get("enabled")):
        return {"enabled": False}
    required = ("add_entry_low", "add_entry_high", "add_quantity", "maximum_total_quantity")
    missing = [key for key in required if scaling.get(key) is None]
    if missing:
        raise ValueError("enabled scaling plan is missing {}".format(", ".join(missing)))
    try:
        low = float(scaling["add_entry_low"])
        high = float(scaling["add_entry_high"])
        add_quantity = int(scaling["add_quantity"])
        maximum = int(scaling["maximum_total_quantity"])
    except (TypeError, ValueError):
        raise ValueError("enabled scaling plan contains invalid numeric values") from None
    if low <= 0 or high < low or add_quantity <= 0 or maximum < add_quantity:
        raise ValueError("enabled scaling plan has invalid range or quantity")
    return {
        "enabled": True,
        "add_entry_low": low,
        "add_entry_high": high,
        "add_quantity": add_quantity,
        "maximum_total_quantity": maximum,
    }


def _latest_by_trade(events: Iterable[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    latest: Dict[str, Mapping[str, Any]] = {}
    for event in events:
        trade_id = str(event.get("trade_id") or "")
        if trade_id:
            latest[trade_id] = event
    return latest


def record_plan(
    path: Path,
    run_path: Path,
    ticker: str,
    trade_id: Optional[str] = None,
    predefined_scaling: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    run = read_json(run_path)
    if not isinstance(run, dict):
        raise ValueError("CORAT run is unreadable")
    name = ticker.upper()
    candidate = next((row for row in run.get("candidates") or [] if row.get("ticker") == name), None)
    if candidate is None:
        raise ValueError("ticker {} is not in the specified CORAT run".format(name))
    identifier = trade_id or "corat-{}-{}".format(
        name.lower(),
        sha256_bytes(canonical_json({"run": str(run_path), "ticker": name, "time": utc_now()}).encode("utf-8"))[:12],
    )
    event = {
        "schema_version": "corat.trade_event.v1",
        "event_id": sha256_bytes((identifier + utc_now()).encode("utf-8"))[:20],
        "trade_id": identifier,
        "event_time_utc": utc_now(),
        "status": "RECOMMENDED",
        "ticker": name,
        "source_run": str(run_path.expanduser().resolve()),
        "source_run_as_of": run.get("as_of"),
        "original_thesis": {
            "setup": candidate.get("setup"),
            "vehicle": candidate.get("vehicle"),
            "vehicle_reason": candidate.get("vehicle_reason"),
            "stock_plan": candidate.get("stock_plan"),
            "option": candidate.get("option"),
            "score": candidate.get("score"),
            "confidence": candidate.get("confidence"),
            "expected_outcome": (candidate.get("history") or {}).get("expectancy"),
            "predefined_scaling": _validated_scaling(predefined_scaling),
        },
        "actual": {},
    }
    append_event(path, event)
    return event


def record_trade_event(
    path: Path,
    trade_id: str,
    status: str,
    price: Optional[float] = None,
    quantity: Optional[int] = None,
    realized_pnl: Optional[float] = None,
    mfe: Optional[float] = None,
    mae: Optional[float] = None,
    reason: str = "",
    event_time_utc: Optional[str] = None,
    review_horizon_sessions: Optional[int] = None,
) -> Dict[str, Any]:
    state = status.upper()
    if state not in TRADE_STATUSES:
        raise ValueError("invalid trade status: {}".format(status))
    events = read_events(path)
    trade_events = [row for row in events if row.get("trade_id") == trade_id]
    if not trade_events:
        raise ValueError("unknown trade_id: {}".format(trade_id))
    prior = str(trade_events[-1].get("status") or "")
    if prior in TERMINAL:
        raise ValueError("trade {} is already terminal at {}".format(trade_id, prior))
    if state not in TRANSITIONS.get(prior, set()):
        raise ValueError("invalid transition {} -> {}".format(prior, state))
    root = trade_events[0]
    event = {
        "schema_version": "corat.trade_event.v1",
        "event_id": sha256_bytes((trade_id + (event_time_utc or utc_now()) + state).encode("utf-8"))[:20],
        "trade_id": trade_id,
        "event_time_utc": event_time_utc or utc_now(),
        "status": state,
        "ticker": root.get("ticker"),
        "source_run": root.get("source_run"),
        "actual": {
            "price": price,
            "quantity": quantity,
            "realized_pnl": realized_pnl,
            "mfe": mfe,
            "mae": mae,
            "reason": reason,
            "review_horizon_sessions": review_horizon_sessions,
        },
    }
    append_event(path, event)
    return event


def trade_states(events: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for event in events:
        grouped.setdefault(str(event.get("trade_id") or ""), []).append(event)
    states = []
    for trade_id, rows in grouped.items():
        if not trade_id:
            continue
        root = rows[0]
        last_non_review = next((row for row in reversed(rows) if row.get("status") != "REVIEW"), rows[-1])
        states.append(
            {
                "trade_id": trade_id,
                "ticker": root.get("ticker"),
                "status": last_non_review.get("status"),
                "original_thesis": root.get("original_thesis") or {},
                "events": rows,
            }
        )
    return states


def review_open_trades(
    events: Sequence[Mapping[str, Any]],
    current_run: Mapping[str, Any],
) -> Dict[str, Any]:
    candidates = {row.get("ticker"): row for row in current_run.get("candidates") or []}
    reviews = []
    for state in trade_states(events):
        if state["status"] not in {"FILLED", "OPEN", "REDUCED"}:
            continue
        thesis = state["original_thesis"]
        plan = thesis.get("stock_plan") or {}
        original_setup = thesis.get("setup") or {}
        direction = str(original_setup.get("direction") or "")
        current = candidates.get(state["ticker"])
        if current is None:
            reviews.append({"trade_id": state["trade_id"], "ticker": state["ticker"], "action": "EXIT / HUMAN REVIEW", "reason": "Current evidence is unavailable; fail closed.", "current_price": None})
            continue
        price = (current.get("technical") or {}).get("price")
        stop = plan.get("stop")
        target1 = plan.get("target_1")
        target2 = plan.get("target_2")
        action = "HOLD"
        reason = "Original thesis has not reached its objective stop or targets."
        if price is None or stop is None:
            action, reason = "EXIT / HUMAN REVIEW", "Original invalidation cannot be evaluated from current data."
        elif (direction == "BULLISH" and price <= stop) or (direction == "BEARISH" and price >= stop):
            action, reason = "EXIT", "Original technical invalidation was reached; do not replace the thesis."
        elif target2 is not None and ((direction == "BULLISH" and price >= target2) or (direction == "BEARISH" and price <= target2)):
            action, reason = "TAKE PROFIT", "Original second target was reached."
        elif target1 is not None and ((direction == "BULLISH" and price >= target1) or (direction == "BEARISH" and price <= target1)):
            action, reason = "REDUCE / TAKE PROFIT", "Original first target was reached."
        elif str((current.get("setup") or {}).get("direction")) not in {direction, "NEUTRAL"}:
            action, reason = "EXIT", "Current structure is opposite the original direction."
        else:
            scaling = thesis.get("predefined_scaling") or {}
            quantities = [
                (row.get("actual") or {}).get("quantity")
                for row in state["events"]
                if (row.get("actual") or {}).get("quantity") is not None
            ]
            current_quantity = int(quantities[-1]) if quantities else None
            add_quantity = int(scaling.get("add_quantity") or 0)
            maximum_quantity = int(scaling.get("maximum_total_quantity") or 0)
            in_zone = bool(
                price is not None
                and scaling.get("add_entry_low") is not None
                and scaling.get("add_entry_high") is not None
                and float(scaling["add_entry_low"]) <= float(price) <= float(scaling["add_entry_high"])
            )
            clean_current_gate = bool(
                current.get("status") in {ACTIONABLE_NOW, TARGET_TRADE}
                and not (current.get("blockers") or [])
                and not (current.get("hard_rejections") or [])
            )
            capacity = bool(
                current_quantity is not None
                and add_quantity > 0
                and maximum_quantity > 0
                and current_quantity + add_quantity <= maximum_quantity
            )
            if bool(scaling.get("enabled")) and in_zone and clean_current_gate and capacity:
                action = "ADD"
                reason = "Predefined add quantity {} passed its zone, total-quantity cap, aligned-thesis, and current actionability gates.".format(add_quantity)
            elif bool(scaling.get("enabled")) and in_zone and current_quantity is None:
                reason = "HOLD: predefined add zone was reached, but current total quantity is unavailable."
        reviews.append(
            {
                "trade_id": state["trade_id"],
                "ticker": state["ticker"],
                "action": action,
                "reason": reason,
                "current_price": price,
                "original_stop": stop,
                "original_target_1": target1,
                "original_target_2": target2,
                "add_allowed": bool((thesis.get("predefined_scaling") or {}).get("enabled")),
                "recommended_add_quantity": (
                    int((thesis.get("predefined_scaling") or {}).get("add_quantity") or 0)
                    if action == "ADD" else None
                ),
                "current_candidate_status": current.get("status"),
                "current_blockers": current.get("blockers") or current.get("hard_rejections") or [],
            }
        )
    return {
        "schema_version": "corat.open_trade_review.v1",
        "as_of": current_run.get("as_of"),
        "generated_at_utc": utc_now(),
        "reviews": reviews,
    }


def render_open_trade_review(review: Mapping[str, Any]) -> str:
    lines = [
        "# CORAT Open-Trade Review — {}".format(review.get("as_of")),
        "",
        "Actions are evaluated against the original stop and targets. New unrelated information cannot rescue an invalidated thesis. ADD is never emitted unless scaling was predefined before entry.",
        "",
        "| Trade | Ticker | Action | Current | Original stop | Target 1 | Target 2 | Reason |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in review.get("reviews") or []:
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(row.get("trade_id"), row.get("ticker"), row.get("action"), row.get("current_price"), row.get("original_stop"), row.get("original_target_1"), row.get("original_target_2"), str(row.get("reason") or "").replace("|", "\\|")))
    if not review.get("reviews"):
        lines.append("| — | — | NO OPEN TRADES | — | — | — | — | Ledger contains no FILLED/OPEN/REDUCED trade. |")
    return "\n".join(lines) + "\n"


def ledger_summary(events: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    states = trade_states(events)
    closed = [state for state in states if state["status"] == "CLOSED"]
    realized = []
    for state in closed:
        values = [row.get("actual") or {} for row in state["events"] if row.get("status") == "CLOSED"]
        if values and values[-1].get("realized_pnl") is not None:
            realized.append(float(values[-1]["realized_pnl"]))
    return {
        "schema_version": "corat.ledger_summary.v1",
        "trade_count": len(states),
        "status_counts": {status: sum(1 for state in states if state["status"] == status) for status in sorted(TRADE_STATUSES - {"REVIEW"})},
        "closed_with_realized_pnl": len(realized),
        "realized_pnl_total": sum(realized),
        "realized_pnl_expectancy": sum(realized) / len(realized) if realized else None,
        "win_rate": sum(1 for value in realized if value > 0) / float(len(realized)) if realized else None,
    }
