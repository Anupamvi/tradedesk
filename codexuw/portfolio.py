from __future__ import annotations

import json
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


def summarize_positions(payload: dict[str, Any]) -> dict[str, Any]:
    positions = list(payload.get("positions", []) or [])
    total_value = safe_float((payload.get("balances") or {}).get("total_value"), 0.0)
    cash = safe_float((payload.get("balances") or {}).get("cash"), 0.0)
    option_underlyings: set[str] = set()
    short_option_underlyings: set[str] = set()
    equity_exposure: Counter[str] = Counter()
    option_market_value: Counter[str] = Counter()
    for pos in positions:
        underlying = _position_underlying(pos)
        if not underlying:
            continue
        asset_type = str(pos.get("asset_type") or "").upper()
        market_value = safe_float(pos.get("market_value"), 0.0)
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
    return {
        "status": "ok",
        "total_value": total_value,
        "cash": cash,
        "position_count": len(positions),
        "option_underlyings": sorted(option_underlyings),
        "short_option_underlyings": sorted(short_option_underlyings),
        "equity_exposure": dict(sorted(equity_exposure.items())),
        "option_market_value": dict(sorted(option_market_value.items())),
        "large_equity_exposure": dict(sorted(large_equity.items())),
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
        "position_count": 0,
        "option_underlyings": [],
        "short_option_underlyings": [],
        "equity_exposure": {},
        "option_market_value": {},
        "large_equity_exposure": {},
    }
