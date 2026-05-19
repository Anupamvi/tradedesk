from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from .data import safe_float


def _position_underlying(position: dict[str, Any]) -> str:
    underlying = str(position.get("underlying") or "").strip().upper()
    if underlying:
        return underlying
    symbol = str(position.get("symbol") or "").strip().upper()
    return symbol.split()[0] if symbol else ""


def _position_label(position: dict[str, Any]) -> str:
    symbol = str(position.get("symbol") or "").strip()
    qty = safe_float(position.get("qty"), 0.0)
    asset_type = str(position.get("asset_type") or "").upper()
    if asset_type == "OPTION":
        side = "short" if safe_float(position.get("short_qty"), 0.0) > 0 else "long"
        return f"{side} {abs(qty):g} {symbol}".strip()
    return f"{qty:g} {symbol}".strip()


def _risk_action_icon(action: str) -> str:
    text = str(action or "").upper()
    if text in {"CLOSE", "REDUCE", "HEDGE"}:
        return "🔴"
    if text in {"ROLL", "TAKE PROFIT", "SET STOP"}:
        return "🟡"
    if text in {"SELL COVERED INCOME", "SELL CSP"}:
        return "🔵"
    return "🟢"


def _append_action(
    actions: list[dict[str, Any]],
    *,
    lane: str,
    ticker: str,
    action: str,
    position: str,
    reason: str,
    instruction: str,
    exposure_pct: float = math.nan,
    assignment_risk: str = "",
    upside_downside_tradeoff: str = "",
    portfolio_impact: str = "",
) -> None:
    actions.append(
        {
            "icon": _risk_action_icon(action),
            "lane": lane,
            "ticker": ticker,
            "action": action,
            "position": position,
            "exposure_pct": exposure_pct,
            "reason": reason,
            "instruction": instruction,
            "assignment_risk": assignment_risk,
            "upside_downside_tradeoff": upside_downside_tradeoff,
            "portfolio_impact": portfolio_impact,
        }
    )


def _build_position_actions(
    positions: list[dict[str, Any]],
    *,
    total_value: float,
    equity_exposure: Counter[str],
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    option_tickers = {_position_underlying(pos) for pos in positions if str(pos.get("asset_type") or "").upper() == "OPTION"}
    for pos in positions:
        ticker = _position_underlying(pos)
        if not ticker:
            continue
        asset_type = str(pos.get("asset_type") or "").upper()
        qty = safe_float(pos.get("qty"), 0.0)
        short_qty = safe_float(pos.get("short_qty"), 0.0)
        market_value = safe_float(pos.get("market_value"), 0.0)
        day_pnl = safe_float(pos.get("day_pnl"), 0.0)
        avg_cost = abs(safe_float(pos.get("avg_cost"), 0.0))
        exposure_pct = abs(market_value) / total_value if total_value > 0 and math.isfinite(market_value) else math.nan
        label = _position_label(pos)

        if asset_type == "OPTION":
            if short_qty > 0:
                basis = avg_cost * 100.0 * short_qty if avg_cost > 0 else math.nan
                remaining_pct = abs(market_value) / basis if math.isfinite(basis) and basis > 0 else math.nan
                if math.isfinite(remaining_pct) and remaining_pct <= 0.40:
                    action = "TAKE PROFIT"
                    instruction = "Buy back or reduce the short option/spread if the live debit still captures most planned profit."
                    reason = "short premium has limited remaining reward versus gamma/assignment risk"
                elif day_pnl < 0:
                    action = "ROLL"
                    instruction = "Review roll or reduce before adding unrelated risk; use live Schwab mark and open interest."
                    reason = "short option is moving against the account"
                else:
                    action = "HOLD"
                    instruction = "Hold only with an explicit stop/review alert on short-leg delta, mark, and expiration-week gamma."
                    reason = "short option exposure needs a monitor plan"
                _append_action(
                    actions,
                    lane="Manage Existing Risk",
                    ticker=ticker,
                    action=action,
                    position=label,
                    exposure_pct=exposure_pct,
                    reason=reason,
                    instruction=instruction,
                    assignment_risk="short option assignment/gamma risk must be monitored",
                    portfolio_impact="risk management takes priority over new unrelated exposure",
                )
            elif qty > 0:
                action = "TAKE PROFIT" if day_pnl > 0 else "REDUCE" if day_pnl < 0 else "HOLD"
                instruction = (
                    "Scale or close if the original thesis is stale; otherwise keep a max-loss and time-stop alert."
                    if action != "HOLD"
                    else "Keep a time stop and thesis-invalidation alert; do not average down."
                )
                _append_action(
                    actions,
                    lane="Manage Existing Risk",
                    ticker=ticker,
                    action=action,
                    position=label,
                    exposure_pct=exposure_pct,
                    reason="open long option needs lifecycle review",
                    instruction=instruction,
                    portfolio_impact="prevents stale optionality from crowding today's risk budget",
                )
            continue

        if asset_type == "EQUITY":
            equity_value = safe_float(equity_exposure.get(ticker), abs(market_value))
            pct = equity_value / total_value if total_value > 0 else math.nan
            shares = abs(qty)
            if math.isfinite(pct) and pct >= 0.08:
                _append_action(
                    actions,
                    lane="Manage Existing Risk",
                    ticker=ticker,
                    action="HOLD",
                    position=label,
                    exposure_pct=pct,
                    reason=f"single-name equity concentration is elevated at {pct:.1%}",
                    instruction=(
                        "Exposure note only; valid new trades may still execute if live pricing, expectancy, "
                        "and explicit risk-budget gates pass."
                    ),
                    portfolio_impact="surfaces concentration context without vetoing valid trade setups",
                )
            if shares >= 100 and ticker not in option_tickers:
                _append_action(
                    actions,
                    lane="Portfolio Income",
                    ticker=ticker,
                    action="SELL COVERED INCOME",
                    position=label,
                    exposure_pct=pct,
                    reason="shares are available for covered-call or collar income review",
                    instruction="Price a covered call or collar from the live Schwab chain; keep it Conditional until bid/ask, delta, and assignment tradeoff are explicit.",
                    assignment_risk="covered calls can cap upside and may assign shares",
                    upside_downside_tradeoff="income improves basis but gives up upside above the short call; collar adds downside protection at extra cost",
                    portfolio_impact="monetizes existing exposure instead of adding unrelated directional risk",
                )
    return actions


def summarize_positions(payload: dict[str, Any]) -> dict[str, Any]:
    positions = list(payload.get("positions", []) or [])
    total_value = safe_float((payload.get("balances") or {}).get("total_value"), 0.0)
    cash = safe_float((payload.get("balances") or {}).get("cash"), 0.0)
    option_underlyings: set[str] = set()
    short_option_underlyings: set[str] = set()
    equity_exposure: Counter[str] = Counter()
    option_market_value: Counter[str] = Counter()
    day_pnl_total = 0.0
    for pos in positions:
        underlying = _position_underlying(pos)
        if not underlying:
            continue
        asset_type = str(pos.get("asset_type") or "").upper()
        market_value = safe_float(pos.get("market_value"), 0.0)
        day_pnl_total += safe_float(pos.get("day_pnl"), 0.0)
        if asset_type == "OPTION":
            option_underlyings.add(underlying)
            option_market_value[underlying] += abs(market_value)
            if safe_float(pos.get("short_qty"), 0.0) > 0:
                short_option_underlyings.add(underlying)
        elif asset_type == "EQUITY":
            equity_exposure[underlying] += abs(market_value)
    large_equity = {
        ticker: value
        for ticker, value in equity_exposure.items()
        if total_value > 0 and value / total_value >= 0.04
    }
    risk_actions = _build_position_actions(positions, total_value=total_value, equity_exposure=equity_exposure)
    return {
        "status": "ok",
        "total_value": total_value,
        "cash": cash,
        "day_pnl": day_pnl_total,
        "position_count": len(positions),
        "option_underlyings": sorted(option_underlyings),
        "short_option_underlyings": sorted(short_option_underlyings),
        "equity_exposure": dict(sorted(equity_exposure.items())),
        "option_market_value": dict(sorted(option_market_value.items())),
        "large_equity_exposure": dict(sorted(large_equity.items())),
        "risk_actions": risk_actions,
        "portfolio_income_actions": [row for row in risk_actions if row.get("lane") == "Portfolio Income"],
    }


def fetch_portfolio_context(out_dir: Path) -> dict[str, Any]:
    from uwos.schwab_auth import SchwabAuthConfig, SchwabLiveDataService

    out_dir.mkdir(parents=True, exist_ok=True)
    service = SchwabLiveDataService(SchwabAuthConfig.from_env(load_dotenv_file=True), interactive_login=False)
    payload = service.get_account_positions()
    positions = pd.DataFrame(payload.get("positions", []) or [])
    positions.to_csv(out_dir / "codexuw_open_positions_from_schwab.csv", index=False)
    summary = summarize_positions(payload)
    (out_dir / "codexuw_portfolio_context.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def unavailable_portfolio_context(error: str) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "error": str(error),
        "total_value": 0.0,
        "cash": 0.0,
        "day_pnl": 0.0,
        "position_count": 0,
        "option_underlyings": [],
        "short_option_underlyings": [],
        "equity_exposure": {},
        "option_market_value": {},
        "large_equity_exposure": {},
        "risk_actions": [],
        "portfolio_income_actions": [],
    }
