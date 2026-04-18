#!/usr/bin/env python3
"""Utilities for reviewing open option spreads as one defined-risk position."""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple


EPS = 1e-9

SCALED_COMPUTED_FIELDS = {
    "theta_pnl_per_day",
    "gamma_risk",
    "vega_exposure",
    "max_profit",
    "max_loss",
    "unrealized_pnl",
}


def safe_float(value: Any, default: Optional[float] = 0.0) -> Optional[float]:
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if out == out else default


def leg_key(pos: Dict[str, Any]) -> str:
    return str(pos.get("symbol") or "UNKNOWN").strip()


def _normalize_right(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text in {"P", "PUT"}:
        return "PUT"
    if text in {"C", "CALL"}:
        return "CALL"
    return text


def _underlying(pos: Dict[str, Any]) -> str:
    symbol = leg_key(pos)
    return str(pos.get("underlying") or symbol.split()[0] or symbol).strip().upper()


def _scaled_position(pos: Dict[str, Any], abs_qty: float) -> Dict[str, Any]:
    """Return a copy of a position allocated to abs_qty contracts/shares."""
    out = copy.deepcopy(pos)
    qty = safe_float(pos.get("qty"), 0.0) or 0.0
    original_abs = abs(qty)
    if original_abs <= EPS:
        return out

    alloc_abs = min(abs_qty, original_abs)
    factor = alloc_abs / original_abs
    out["qty"] = -alloc_abs if qty < 0 else alloc_abs

    market_value = safe_float(out.get("market_value"), None)
    if market_value is not None:
        out["market_value"] = market_value * factor

    computed = out.get("computed")
    if isinstance(computed, dict):
        for field in SCALED_COMPUTED_FIELDS:
            val = safe_float(computed.get(field), None)
            if val is not None:
                computed[field] = val * factor
    return out


def strategy_from_legs(right: str, short_strike: float, long_strike: float) -> Optional[str]:
    right = _normalize_right(right)
    if right == "PUT":
        if short_strike > long_strike:
            return "Bull Put Credit"
        if short_strike < long_strike:
            return "Bear Put Debit"
    if right == "CALL":
        if short_strike > long_strike:
            return "Bull Call Debit"
        if short_strike < long_strike:
            return "Bear Call Credit"
    return None


def net_type_from_strategy(strategy: str) -> str:
    return "credit" if strategy in {"Bull Put Credit", "Bear Call Credit"} else "debit"


def spread_group_key(group: Dict[str, Any]) -> str:
    return (
        f"SPREAD:{group['underlying']}:{group['expiry']}:{group['put_call']}:"
        f"{group['short_symbol']}|{group['long_symbol']}"
    )


def build_position_review_items(positions: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Pair vertical spreads and return spread items plus unpaired positions.

    The pairing is intentionally conservative: same underlying, expiry, option
    right, opposite signed quantities, and nearest strike width first. Unequal
    quantities are split so any leftover contracts still get reviewed on their
    own.
    """
    option_entries: List[Dict[str, Any]] = []
    standalone: List[Dict[str, Any]] = []

    for pos in positions:
        qty = safe_float(pos.get("qty"), 0.0) or 0.0
        right = _normalize_right(pos.get("put_call"))
        strike = safe_float(pos.get("strike"), None)
        expiry = str(pos.get("expiry") or "").strip()
        if (
            pos.get("asset_type") != "OPTION"
            or abs(qty) <= EPS
            or right not in {"PUT", "CALL"}
            or strike is None
            or not expiry
        ):
            standalone.append({"kind": "POSITION", "key": leg_key(pos), "position": pos})
            continue

        option_entries.append(
            {
                "pos": pos,
                "remaining": abs(qty),
                "side": "short" if qty < 0 else "long",
                "underlying": _underlying(pos),
                "expiry": expiry,
                "right": right,
                "strike": float(strike),
            }
        )

    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for entry in option_entries:
        key = (entry["underlying"], entry["expiry"], entry["right"])
        grouped.setdefault(key, []).append(entry)

    spread_items: List[Dict[str, Any]] = []
    for (underlying, expiry, right), entries in sorted(grouped.items()):
        shorts = [e for e in entries if e["side"] == "short"]
        longs = [e for e in entries if e["side"] == "long"]
        while True:
            candidates: List[Tuple[float, int, int, str]] = []
            for si, short in enumerate(shorts):
                if short["remaining"] <= EPS:
                    continue
                for li, long in enumerate(longs):
                    if long["remaining"] <= EPS:
                        continue
                    if leg_key(short["pos"]) == leg_key(long["pos"]):
                        continue
                    width = abs(short["strike"] - long["strike"])
                    if width <= EPS:
                        continue
                    strategy = strategy_from_legs(right, short["strike"], long["strike"])
                    if strategy:
                        candidates.append((width, si, li, strategy))
            if not candidates:
                break

            width, si, li, strategy = sorted(candidates, key=lambda x: x[0])[0]
            short = shorts[si]
            long = longs[li]
            qty = min(short["remaining"], long["remaining"])
            if qty <= EPS:
                break

            group = {
                "kind": "SPREAD",
                "underlying": underlying,
                "expiry": expiry,
                "put_call": right,
                "strategy": strategy,
                "net_type": net_type_from_strategy(strategy),
                "qty": qty,
                "width": width,
                "short_symbol": leg_key(short["pos"]),
                "long_symbol": leg_key(long["pos"]),
                "short_strike": short["strike"],
                "long_strike": long["strike"],
                "short_leg": _scaled_position(short["pos"], qty),
                "long_leg": _scaled_position(long["pos"], qty),
            }
            spread_items.append(
                {
                    "kind": "SPREAD",
                    "key": spread_group_key(group),
                    "group": group,
                }
            )
            short["remaining"] -= qty
            long["remaining"] -= qty

    remainder_items: List[Dict[str, Any]] = []
    for entry in option_entries:
        if entry["remaining"] > EPS:
            pos = _scaled_position(entry["pos"], entry["remaining"])
            remainder_items.append({"kind": "POSITION", "key": leg_key(pos), "position": pos})

    return [*spread_items, *standalone, *remainder_items]


def current_leg_keys(positions: Sequence[Dict[str, Any]]) -> set[str]:
    return {leg_key(pos) for pos in positions}


def _computed(pos: Dict[str, Any], field: str, default: Optional[float] = 0.0) -> Optional[float]:
    return safe_float((pos.get("computed") or {}).get(field), default)


def _quote(pos: Dict[str, Any], field: str, default: Optional[float] = None) -> Optional[float]:
    return safe_float((pos.get("live_quote") or {}).get(field), default)


def _entry_date(short_leg: Dict[str, Any], long_leg: Dict[str, Any]) -> str:
    dates = sorted({str(d) for d in (short_leg.get("entry_date"), long_leg.get("entry_date")) if d})
    if not dates:
        return ""
    if len(dates) == 1:
        return dates[0]
    return f"{dates[0]}..{dates[-1]}"


def compute_spread_metrics(group: Dict[str, Any]) -> Dict[str, Any]:
    short_leg = group["short_leg"]
    long_leg = group["long_leg"]
    qty = safe_float(group.get("qty"), 0.0) or 0.0
    width = safe_float(group.get("width"), 0.0) or 0.0
    short_strike = safe_float(group.get("short_strike"), 0.0) or 0.0
    long_strike = safe_float(group.get("long_strike"), 0.0) or 0.0
    net_type = group.get("net_type", "")
    right = group.get("put_call", "")

    pnl = (_computed(short_leg, "unrealized_pnl") or 0.0) + (_computed(long_leg, "unrealized_pnl") or 0.0)
    theta = (_computed(short_leg, "theta_pnl_per_day") or 0.0) + (_computed(long_leg, "theta_pnl_per_day") or 0.0)
    gamma = (_computed(short_leg, "gamma_risk") or 0.0) + (_computed(long_leg, "gamma_risk") or 0.0)
    dte = _computed(short_leg, "dte", None)
    if dte is None:
        dte = _computed(long_leg, "dte", None)

    short_credit_total = _computed(short_leg, "max_profit", None)
    if short_credit_total is None:
        short_credit_total = (safe_float(short_leg.get("avg_cost"), 0.0) or 0.0) * qty * 100.0
    long_debit_total = _computed(long_leg, "max_loss", None)
    if long_debit_total is None:
        long_debit_total = (safe_float(long_leg.get("avg_cost"), 0.0) or 0.0) * qty * 100.0

    entry_cash_total = (short_credit_total or 0.0) - (long_debit_total or 0.0)
    width_value = width * qty * 100.0
    max_profit = None
    max_loss = None
    if net_type == "credit":
        credit = max(entry_cash_total, 0.0)
        max_profit = credit if credit > EPS else None
        max_loss = max(width_value - credit, 0.0) if width_value > EPS else None
    else:
        debit = max(-entry_cash_total, 0.0)
        max_loss = debit if debit > EPS else None
        max_profit = max(width_value - debit, 0.0) if width_value > EPS else None

    pct_of_max = (pnl / max_profit * 100.0) if max_profit and max_profit > EPS else None
    pnl_on_risk_pct = (pnl / max_loss * 100.0) if max_loss and max_loss > EPS else None

    spot = safe_float((short_leg.get("underlying_quote") or {}).get("last"), None)
    if spot is None:
        spot = safe_float((long_leg.get("underlying_quote") or {}).get("last"), None)

    low_strike = min(short_strike, long_strike)
    high_strike = max(short_strike, long_strike)
    between_strikes = spot is not None and low_strike < spot < high_strike

    short_leg_itm = False
    max_loss_zone = False
    debit_target_zone = False
    debit_failed_zone = False
    if spot is not None:
        if right == "PUT":
            short_leg_itm = spot < short_strike
            max_loss_zone = net_type == "credit" and spot <= long_strike
            debit_target_zone = net_type == "debit" and spot <= short_strike
            debit_failed_zone = net_type == "debit" and spot >= long_strike
        elif right == "CALL":
            short_leg_itm = spot > short_strike
            max_loss_zone = net_type == "credit" and spot >= long_strike
            debit_target_zone = net_type == "debit" and spot >= short_strike
            debit_failed_zone = net_type == "debit" and spot <= long_strike

    close_net = None
    short_ask = _quote(short_leg, "ask")
    long_bid = _quote(long_leg, "bid")
    if short_ask is not None and long_bid is not None:
        if net_type == "credit":
            close_net = max(short_ask - long_bid, 0.0)
        else:
            close_net = max(long_bid - short_ask, 0.0)

    return {
        "underlying": group["underlying"],
        "expiry": group["expiry"],
        "put_call": right,
        "strategy": group["strategy"],
        "net_type": net_type,
        "qty": qty,
        "width": width,
        "short_symbol": group["short_symbol"],
        "long_symbol": group["long_symbol"],
        "short_strike": short_strike,
        "long_strike": long_strike,
        "strike_pair": f"{short_strike:g}/{long_strike:g}",
        "underlying_price": spot,
        "dte": dte,
        "entry_date": _entry_date(short_leg, long_leg),
        "unrealized_pnl": pnl,
        "pnl_on_risk_pct": pnl_on_risk_pct,
        "pct_of_max_profit": pct_of_max,
        "theta_pnl_per_day": theta,
        "gamma_risk": gamma,
        "max_profit": max_profit,
        "max_loss": max_loss,
        "entry_net_per_contract": abs(entry_cash_total) / (qty * 100.0) if qty > EPS else None,
        "current_exit_net": close_net,
        "short_delta": safe_float((short_leg.get("greeks") or {}).get("delta"), None),
        "short_leg_itm": short_leg_itm,
        "between_strikes": between_strikes,
        "max_loss_zone": max_loss_zone,
        "debit_target_zone": debit_target_zone,
        "debit_failed_zone": debit_failed_zone,
    }


def compute_spread_verdict(group: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
    """Return a spread-level verdict. Actionable verdicts apply to both legs."""
    m = compute_spread_metrics(group)
    strategy = m["strategy"]
    net_type = m["net_type"]
    dte = safe_float(m.get("dte"), None)
    pct_max = safe_float(m.get("pct_of_max_profit"), None)
    risk_pct = safe_float(m.get("pnl_on_risk_pct"), None)
    short_delta = safe_float(m.get("short_delta"), 0.0) or 0.0
    unit_action = "close/roll both legs together; do not leg out"
    base = f"{strategy} {m['strike_pair']} spread"

    if dte is not None and dte <= 2 and (m["short_leg_itm"] or m["between_strikes"]):
        return (
            "CLOSE",
            f"{base}: expiration assignment/exercise risk with {dte:.0f} DTE; {unit_action}",
            m,
        )

    if net_type == "credit":
        if pct_max is not None and pct_max >= 75:
            return (
                "CLOSE",
                f"{base}: {pct_max:.0f}% of max profit harvested; remaining theta is not worth gamma/assignment risk",
                m,
            )
        if m["max_loss_zone"]:
            return (
                "ASSESS",
                f"{base}: beyond long hedge and defined-risk loss zone; manage the whole spread, not the short leg alone",
                m,
            )
        if m["short_leg_itm"] and dte is not None and dte <= 14:
            return (
                "ROLL",
                f"{base}: short leg ITM with {dte:.0f} DTE; roll or close both legs together",
                m,
            )
        if m["short_leg_itm"]:
            return (
                "ASSESS",
                f"{base}: short leg ITM but hedge is attached; assess whole spread roll/close, not the short leg alone",
                m,
            )
        if dte is not None and dte <= 7 and (pct_max is None or pct_max < 50):
            return (
                "CLOSE",
                f"{base}: expiration week with limited profit captured; gamma risk now dominates theta, {unit_action}",
                m,
            )
        if risk_pct is not None and risk_pct <= -60:
            return (
                "ASSESS",
                f"{base}: loss is {risk_pct:.0f}% of defined risk; reassess whole spread",
                m,
            )
        pct_text = f"{pct_max:.0f}% max" if pct_max is not None else "defined-risk"
        dte_text = f", {dte:.0f} DTE" if dte is not None else ""
        return ("HOLD", f"{base}: {pct_text}{dte_text}", m)

    if pct_max is not None and pct_max >= 75:
        return (
            "CLOSE",
            f"{base}: {pct_max:.0f}% of max profit available; close before theta/gamma risk gives back gains",
            m,
        )
    if pct_max is not None and pct_max >= 50 and dte is not None and dte <= 21:
        return (
            "ASSESS",
            f"{base}: {pct_max:.0f}% of max profit with {dte:.0f} DTE; protect gains from theta/gamma decay",
            m,
        )
    if m["debit_target_zone"] and dte is not None and dte <= 10:
        return (
            "CLOSE",
            f"{base}: target zone reached with {dte:.0f} DTE; close the spread as one order",
            m,
        )
    if risk_pct is not None and risk_pct <= -60:
        return (
            "CLOSE",
            f"{base}: down {risk_pct:.0f}% of debit risk; close both legs together",
            m,
        )
    if m["debit_failed_zone"] and dte is not None and dte < 35:
        if dte <= 14:
            return (
                "CLOSE",
                f"{base}: outside the profit zone with {dte:.0f} DTE; theta decay makes recovery unlikely",
                m,
            )
        return (
            "ASSESS",
            f"{base}: debit spread is outside the profit zone with {dte:.0f} DTE; evaluate whole-spread exit",
            m,
        )
    if abs(short_delta) > 0.70 and dte is not None and dte <= 7:
        return (
            "ASSESS",
            f"{base}: short leg high-delta near expiry; manage as a spread, never as a naked leg",
            m,
        )
    pct_text = f"{pct_max:.0f}% max" if pct_max is not None else "defined-risk"
    dte_text = f", {dte:.0f} DTE" if dte is not None else ""
    return ("HOLD", f"{base}: {pct_text}{dte_text}", m)
