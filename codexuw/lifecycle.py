from __future__ import annotations

import datetime as dt
import math
from typing import Any

import pandas as pd

from .data import safe_float


def _money(value: object) -> str:
    number = safe_float(value)
    return f"${number:.2f}" if math.isfinite(number) else "fresh Schwab recheck"


def _is_debit(row: pd.Series | dict[str, Any]) -> bool:
    text = f"{row.get('strategy', '')} {row.get('direction', '')}".lower()
    return "debit" in text or "bull call" in text or "bear put" in text


def _short_strike(row: pd.Series | dict[str, Any]) -> float:
    return safe_float(row.get("short_strike"))


def build_lifecycle_triggers(row: pd.Series | dict[str, Any], *, asof: dt.date | None = None) -> dict[str, Any]:
    """Build monitor-ready triggers for an executable or scout options spread.

    These are order-ticket instructions only. They intentionally do not imply
    that Codex can place broker orders.
    """
    status = str(row.get("Status") or row.get("trade_status") or row.get("status") or "")
    lane = str(row.get("Lane") or row.get("lane") or "")
    is_actionable = "Execute" in status or "Scout" in status
    if not is_actionable:
        return {
            "profit_take": "",
            "stop_loss": "",
            "roll_trigger": "",
            "short_strike_threat": "",
            "short_leg_delta_threshold": "",
            "dte_warning": "",
            "thesis_invalidation": "",
            "phone_alert_text": "",
        }

    ticker = str(row.get("Ticker") or row.get("ticker") or "").upper().strip() or "POSITION"
    strategy = str(row.get("Trade") or row.get("strategy") or row.get("direction") or "options spread")
    expiry = row.get("Expiry") or row.get("expiry") or row.get("expiration_date") or ""
    dte = safe_float(row.get("dte"))
    entry_limit = row.get("Entry limit") or row.get("entry_limit") or row.get("recommended_limit") or row.get("required_entry")
    max_profit = safe_float(row.get("Max profit") or row.get("max_profit"))
    max_loss = safe_float(row.get("Max loss") or row.get("max_loss"))
    credit = safe_float(row.get("credit"))
    debit = safe_float(row.get("debit"))
    short_delta = abs(safe_float(row.get("short_delta")))
    short_strike = _short_strike(row)
    direction = str(row.get("direction") or strategy)

    if _is_debit(row):
        stop = debit * 0.55 if math.isfinite(debit) and debit > 0 else max_loss * 0.45 / 100.0
        target = min(max_profit * 0.45, max_profit - 5.0) if math.isfinite(max_profit) and max_profit > 0 else math.nan
        profit_take = f"Take profit if spread value gains about 45% of max profit ({_money(target)}) or thesis target hits."
        stop_loss = f"Stop/review if spread value loses about 45% of debit ({_money(stop)}) or closes below thesis level."
    else:
        target_debit = credit * 0.45 if math.isfinite(credit) and credit > 0 else math.nan
        stop_debit = credit * 2.0 if math.isfinite(credit) and credit > 0 else math.nan
        profit_take = f"Buy back around {_money(target_debit)} debit or after 55-65% of credit is captured."
        stop_loss = f"Review/close if debit reaches {_money(stop_debit)} or loss approaches 1x initial credit."

    if math.isfinite(short_strike):
        if "Call" in direction or "call" in strategy.lower():
            strike_threat = f"Alert if spot trades within 1.0% below short call {short_strike:g} or closes above it."
        else:
            strike_threat = f"Alert if spot trades within 1.0% above short put {short_strike:g} or closes below it."
    else:
        strike_threat = "Alert if spot breaches the short-strike buffer from the Schwab snapshot."

    delta_threshold = 0.35
    if math.isfinite(short_delta) and short_delta >= 0.28:
        delta_threshold = max(0.40, round(short_delta + 0.08, 2))
    dte_warning = (
        "Start expiration-week review now; do not carry unmanaged gamma."
        if math.isfinite(dte) and dte <= 7
        else "Warn at 7 DTE; close or roll before unmanaged expiration-week gamma."
    )
    thesis = str(row.get("Required confirmation") or row.get("required_confirmation") or row.get("trade_status_reason") or "")
    thesis_invalid = thesis if thesis else "price action, OI carryover, news/catalyst, or regime confirmation fails on fresh review"
    phone = (
        f"{ticker} {strategy}: monitor {expiry}. Profit trigger: {profit_take} "
        f"Risk trigger: {stop_loss} Short-strike/delta alerts required. Manual order only."
    )
    return {
        "profit_take": profit_take,
        "stop_loss": stop_loss,
        "roll_trigger": "Roll only if thesis still holds, liquidity is tight, and new spread improves risk/reward; otherwise close.",
        "short_strike_threat": strike_threat,
        "short_leg_delta_threshold": f"Alert if absolute short-leg delta reaches {delta_threshold:.2f}.",
        "dte_warning": dte_warning,
        "thesis_invalidation": thesis_invalid,
        "phone_alert_text": phone,
    }


def apply_lifecycle_triggers(board: pd.DataFrame, *, asof: dt.date | None = None) -> pd.DataFrame:
    if board.empty:
        return board.copy()
    out = board.copy()
    rows = [build_lifecycle_triggers(row, asof=asof) for _, row in out.iterrows()]
    triggers = pd.DataFrame(rows, index=out.index)
    for col in triggers.columns:
        out[col] = triggers[col]
    if "Monitor trigger" in out.columns:
        out["Monitor trigger"] = out.apply(
            lambda row: row["phone_alert_text"] if row.get("phone_alert_text") else row.get("Monitor trigger", ""),
            axis=1,
        )
    return out
