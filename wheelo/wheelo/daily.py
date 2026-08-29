"""Daily CSP / shares / CC matrix. Prefer Schwab marks so the run is 0 ORATS."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from wheelo.config import BOOK_PATH, CONTRACT_MULTIPLIER
from wheelo.dates import parse_ymd
from wheelo.num import to_float
from wheelo.scoring import earnings_days


def load_book(path: Path = BOOK_PATH) -> Dict[str, Any]:
    if not path.is_file():
        return {"positions": [], "premium_journal": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"positions": [], "premium_journal": []}
    if not isinstance(payload, dict):
        return {"positions": [], "premium_journal": []}
    payload.setdefault("positions", [])
    payload.setdefault("premium_journal", [])
    return payload


def save_book(payload: Dict[str, Any], path: Path = BOOK_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def dte_left(asof: str, expiry: str) -> Optional[int]:
    a = parse_ymd(asof)
    b = parse_ymd(str(expiry or "")[:10])
    if not a or not b:
        return None
    from datetime import datetime

    return (datetime.strptime(b, "%Y-%m-%d").date() - datetime.strptime(a, "%Y-%m-%d").date()).days


def evaluate_csp(
    pos: dict,
    current_premium: Optional[float],
    dte: Optional[int],
    cfg: dict,
    signal: str = "neutral",
    earn_days: Optional[int] = None,
) -> dict:
    mgmt = cfg.get("management") or {}
    entry = to_float(pos.get("entry_premium")) or 0.0
    pnl = None
    if entry > 0 and current_premium is not None:
        pnl = (entry - current_premium) / entry
    close_tgt = float(mgmt.get("close_target_pct") or 0.5)
    roll_dte = int(mgmt.get("dte_roll_threshold") or 14)
    action = {
        "ticker": pos.get("ticker"),
        "phase": "csp",
        "pnl_pct": pnl,
        "current_premium": current_premium,
        "signal": signal,
        "action": "HOLD",
        "detail": "P/L DATA UNAVAILABLE" if pnl is None else "hold",
        "reason": "hold",
    }
    if pnl is not None and pnl >= close_tgt:
        action.update({"action": "CLOSE", "detail": "target %.0f%%" % (100 * pnl), "reason": "profit_target"})
        return action
    if earn_days is not None and earn_days <= 7:
        action.update({"action": "CLOSE", "detail": "earnings in %sd" % earn_days, "reason": "earnings"})
        return action
    if earn_days is not None and dte is not None and earn_days <= dte + 3:
        action.update(
            {"action": "CLOSE", "detail": "earnings in %sd crosses %sd DTE" % (earn_days, dte), "reason": "earnings_in_dte"}
        )
        return action
    if dte is not None and dte <= roll_dte:
        if pnl is not None and pnl < 0:
            action.update({"action": "ROLL", "detail": "losing with %s DTE" % dte, "reason": "low_dte_losing"})
        elif pnl is not None and pnl >= 0:
            action.update({"action": "CLOSE", "detail": "winning with %s DTE" % dte, "reason": "low_dte_winning"})
        else:
            action.update({"action": "ASSESS", "detail": "DTE %s, mark DATA UNAVAILABLE" % dte, "reason": "low_dte"})
        return action
    if signal == "bearish":
        action.update({"action": "ROLL", "detail": "bearish tape — roll down", "reason": "bearish"})
    return action


def evaluate_shares(pos: dict, spot: Optional[float], signal: str = "neutral") -> dict:
    action = {
        "ticker": pos.get("ticker"),
        "phase": "shares",
        "action": "SELL_CC",
        "signal": signal,
        "pnl_pct": None,
        "current_premium": None,
        "reason": "neutral_cc",
        "detail": "sell CC 1-sigma, ~30 DTE",
    }
    if signal == "bullish":
        action.update({"reason": "bullish_cc", "detail": "sell CC 0.5-sigma (higher strike)"})
    elif signal == "bearish":
        action.update({"reason": "bearish_cc", "detail": "sell CC ATM/ITM"})
    cost = to_float(pos.get("cost_basis"))
    if spot is not None and cost and cost > 0 and (spot - cost) / cost <= -0.10:
        action.update({"detail": "down >10% from cost — CC at cost basis", "reason": "drawdown_cc"})
    return action


def evaluate_cc(pos: dict, current_premium: Optional[float], dte: Optional[int], spot: Optional[float], cfg: dict) -> dict:
    mgmt = cfg.get("management") or {}
    entry = to_float(pos.get("entry_premium")) or 0.0
    pnl = None
    if entry > 0 and current_premium is not None:
        pnl = (entry - current_premium) / entry
    close_tgt = float(mgmt.get("close_target_pct") or 0.5)
    strike = to_float(pos.get("strike"))
    action = {
        "ticker": pos.get("ticker"),
        "phase": "cc",
        "pnl_pct": pnl,
        "current_premium": current_premium,
        "action": "HOLD",
        "detail": "hold",
        "reason": "hold",
    }
    if pnl is not None and pnl >= close_tgt:
        action.update({"action": "CLOSE", "detail": "CC target %.0f%%" % (100 * pnl), "reason": "profit_target"})
        return action
    if spot is not None and strike is not None and spot >= strike and dte is not None and dte <= 7:
        action.update({"action": "ALLOW_CALL_AWAY", "detail": "spot through strike", "reason": "called_away"})
        return action
    return action


def evaluate_book(
    asof: str,
    cfg: dict,
    marks: Dict[str, dict],
    spots: Dict[str, float],
    cores: Dict[str, dict],
    book: Optional[Dict[str, Any]] = None,
) -> List[dict]:
    payload = book if book is not None else load_book()
    actions = []
    for pos in payload.get("positions") or []:
        ticker = str(pos.get("ticker") or "").upper()
        phase = str(pos.get("phase") or "csp")
        dte = dte_left(asof, pos.get("expiry") or "")
        mark = marks.get(ticker) or marks.get(pos.get("option_symbol") or "") or {}
        premium = to_float(mark.get("bid"))
        if premium is None:
            premium = to_float(mark.get("mark"))
        spot = spots.get(ticker)
        chg = None
        core = cores.get(ticker) or {}
        chg = to_float(core.get("chg_1w"))
        signal = "neutral"
        if chg is not None and chg < -5:
            signal = "bearish"
        elif chg is not None and chg > 5:
            signal = "bullish"
        if phase == "csp":
            actions.append(
                evaluate_csp(pos, premium, dte, cfg, signal=signal, earn_days=earnings_days(core, asof))
            )
        elif phase == "shares":
            actions.append(evaluate_shares(pos, spot, signal=signal))
        elif phase == "cc":
            actions.append(evaluate_cc(pos, premium, dte, spot, cfg))
        else:
            actions.append(
                {
                    "ticker": ticker,
                    "phase": phase,
                    "action": "HOLD",
                    "detail": "unknown phase",
                    "reason": "unknown",
                    "pnl_pct": None,
                }
            )
    return actions


def capital_required(strike: float, contracts: int = 1) -> float:
    return float(strike) * CONTRACT_MULTIPLIER * int(contracts)
