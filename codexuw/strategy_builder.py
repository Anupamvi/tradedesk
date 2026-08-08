from __future__ import annotations

import datetime as dt
import json
import math
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .data import safe_float


@dataclass(frozen=True)
class LegTemplate:
    right: str
    quantity: int
    moneyness: float
    expiry_slot: str = "primary"


@dataclass(frozen=True)
class GenericStrategySpec:
    key: str
    display_name: str
    direction: str
    strategy_kind: str
    legs: tuple[LegTemplate, ...]
    stock_units: float = 0.0
    requires_equity_shares: int = 0
    requires_cash_secured: bool = False
    requires_margin_model: bool = False
    path_dependent: bool = False


GENERIC_STRATEGY_SPECS: tuple[GenericStrategySpec, ...] = (
    GenericStrategySpec("long_call", "Long Call", "Long Call", "Debit", (LegTemplate("C", 1, 1.05),)),
    GenericStrategySpec("long_put", "Long Put", "Long Put", "Debit", (LegTemplate("P", 1, 0.95),)),
    GenericStrategySpec(
        "covered_call",
        "Covered Call",
        "Covered Call",
        "Credit",
        (LegTemplate("C", -1, 1.05),),
        stock_units=1.0,
        requires_equity_shares=100,
    ),
    GenericStrategySpec(
        "cash_secured_put",
        "Cash-Secured Put",
        "Cash-Secured Put",
        "Credit",
        (LegTemplate("P", -1, 0.95),),
        requires_cash_secured=True,
    ),
    GenericStrategySpec(
        "protective_put",
        "Protective Put",
        "Protective Put",
        "Debit",
        (LegTemplate("P", 1, 0.95),),
        stock_units=1.0,
        requires_equity_shares=100,
    ),
    GenericStrategySpec(
        "collar",
        "Collar",
        "Collar",
        "Mixed",
        (LegTemplate("P", 1, 0.95), LegTemplate("C", -1, 1.05)),
        stock_units=1.0,
        requires_equity_shares=100,
    ),
    GenericStrategySpec(
        "long_straddle",
        "Long Straddle",
        "Long Straddle",
        "Debit",
        (LegTemplate("C", 1, 1.00), LegTemplate("P", 1, 1.00)),
    ),
    GenericStrategySpec(
        "short_straddle",
        "Short Straddle",
        "Short Straddle",
        "Credit",
        (LegTemplate("C", -1, 1.00), LegTemplate("P", -1, 1.00)),
        requires_margin_model=True,
    ),
    GenericStrategySpec(
        "long_strangle",
        "Long Strangle",
        "Long Strangle",
        "Debit",
        (LegTemplate("C", 1, 1.05), LegTemplate("P", 1, 0.95)),
    ),
    GenericStrategySpec(
        "short_strangle",
        "Short Strangle",
        "Short Strangle",
        "Credit",
        (LegTemplate("C", -1, 1.05), LegTemplate("P", -1, 0.95)),
        requires_margin_model=True,
    ),
    GenericStrategySpec(
        "iron_condor",
        "Iron Condor",
        "Iron Condor",
        "Credit",
        (
            LegTemplate("P", 1, 0.90),
            LegTemplate("P", -1, 0.95),
            LegTemplate("C", -1, 1.05),
            LegTemplate("C", 1, 1.10),
        ),
    ),
    GenericStrategySpec(
        "iron_butterfly",
        "Iron Butterfly",
        "Iron Butterfly",
        "Credit",
        (
            LegTemplate("P", 1, 0.90),
            LegTemplate("P", -1, 1.00),
            LegTemplate("C", -1, 1.00),
            LegTemplate("C", 1, 1.10),
        ),
    ),
    GenericStrategySpec(
        "call_butterfly",
        "Call Butterfly",
        "Call Butterfly",
        "Debit",
        (LegTemplate("C", 1, 1.00), LegTemplate("C", -2, 1.05), LegTemplate("C", 1, 1.10)),
    ),
    GenericStrategySpec(
        "put_butterfly",
        "Put Butterfly",
        "Put Butterfly",
        "Debit",
        (LegTemplate("P", 1, 1.00), LegTemplate("P", -2, 0.95), LegTemplate("P", 1, 0.90)),
    ),
    GenericStrategySpec(
        "call_broken_wing_butterfly",
        "Call Broken-Wing Butterfly",
        "Call Broken-Wing Butterfly",
        "Debit",
        (LegTemplate("C", 1, 1.00), LegTemplate("C", -2, 1.05), LegTemplate("C", 1, 1.15)),
    ),
    GenericStrategySpec(
        "put_broken_wing_butterfly",
        "Put Broken-Wing Butterfly",
        "Put Broken-Wing Butterfly",
        "Debit",
        (LegTemplate("P", 1, 1.00), LegTemplate("P", -2, 0.95), LegTemplate("P", 1, 0.85)),
    ),
    GenericStrategySpec(
        "call_calendar",
        "Call Calendar",
        "Call Calendar",
        "Debit",
        (LegTemplate("C", -1, 1.00, "primary"), LegTemplate("C", 1, 1.00, "far")),
        path_dependent=True,
    ),
    GenericStrategySpec(
        "put_calendar",
        "Put Calendar",
        "Put Calendar",
        "Debit",
        (LegTemplate("P", -1, 1.00, "primary"), LegTemplate("P", 1, 1.00, "far")),
        path_dependent=True,
    ),
    GenericStrategySpec(
        "call_diagonal",
        "Call Diagonal",
        "Call Diagonal",
        "Debit",
        (LegTemplate("C", -1, 1.05, "primary"), LegTemplate("C", 1, 1.00, "far")),
        path_dependent=True,
    ),
    GenericStrategySpec(
        "put_diagonal",
        "Put Diagonal",
        "Put Diagonal",
        "Debit",
        (LegTemplate("P", -1, 0.95, "primary"), LegTemplate("P", 1, 1.00, "far")),
        path_dependent=True,
    ),
    GenericStrategySpec(
        "call_ratio_spread",
        "Call Ratio Spread",
        "Call Ratio Spread",
        "Credit",
        (LegTemplate("C", 1, 1.00), LegTemplate("C", -2, 1.05)),
        requires_margin_model=True,
    ),
    GenericStrategySpec(
        "put_ratio_spread",
        "Put Ratio Spread",
        "Put Ratio Spread",
        "Credit",
        (LegTemplate("P", 1, 1.00), LegTemplate("P", -2, 0.95)),
        requires_margin_model=True,
    ),
    GenericStrategySpec(
        "call_ratio_backspread",
        "Call Ratio Backspread",
        "Call Ratio Backspread",
        "Debit",
        (LegTemplate("C", -1, 1.00), LegTemplate("C", 2, 1.10)),
    ),
    GenericStrategySpec(
        "put_ratio_backspread",
        "Put Ratio Backspread",
        "Put Ratio Backspread",
        "Debit",
        (LegTemplate("P", -1, 1.00), LegTemplate("P", 2, 0.90)),
    ),
    GenericStrategySpec(
        "jade_lizard",
        "Jade Lizard",
        "Jade Lizard",
        "Credit",
        (LegTemplate("P", -1, 0.95), LegTemplate("C", -1, 1.05), LegTemplate("C", 1, 1.10)),
        requires_margin_model=True,
    ),
    GenericStrategySpec(
        "reverse_jade_lizard",
        "Reverse Jade Lizard",
        "Reverse Jade Lizard",
        "Credit",
        (LegTemplate("C", -1, 1.05), LegTemplate("P", -1, 0.95), LegTemplate("P", 1, 0.90)),
        requires_margin_model=True,
    ),
    GenericStrategySpec(
        "covered_strangle",
        "Covered Strangle",
        "Covered Strangle",
        "Credit",
        (LegTemplate("P", -1, 0.95), LegTemplate("C", -1, 1.05)),
        stock_units=1.0,
        requires_equity_shares=100,
        requires_cash_secured=True,
    ),
    GenericStrategySpec(
        "wheel",
        "Wheel",
        "Wheel Entry CSP",
        "Credit",
        (LegTemplate("P", -1, 0.95),),
        requires_cash_secured=True,
        path_dependent=True,
    ),
)


GENERIC_STRATEGY_BY_KEY = {spec.key: spec for spec in GENERIC_STRATEGY_SPECS}


def generic_strategy_keys() -> tuple[str, ...]:
    return tuple(spec.key for spec in GENERIC_STRATEGY_SPECS)


def historical_scope_for_strategy(strategy_key: str) -> str:
    if strategy_key in {
        "bull_call_debit_vertical",
        "bear_put_debit_vertical",
        "bull_put_credit_vertical",
        "bear_call_credit_vertical",
    }:
        return "exact_option_exit"
    spec = GENERIC_STRATEGY_BY_KEY.get(strategy_key)
    if spec is None:
        return "unavailable"
    if strategy_key == "wheel":
        return "entry_csp_cycle"
    if spec.path_dependent:
        return "exact_term_structure_exit"
    if spec.requires_margin_model:
        return "exact_pnl_conservative_reg_t_risk"
    if spec.stock_units:
        return "exact_option_and_stock_exit"
    return "exact_option_exit"


def _expiry_map(
    contracts: pd.DataFrame,
    *,
    as_of_date: dt.date,
    preferred_expiry: dt.date | None,
) -> dict[str, dt.date] | None:
    if contracts is None or contracts.empty or "expiry" not in contracts.columns:
        return None
    expiries = sorted(
        value
        for value in set(contracts["expiry"].dropna())
        if isinstance(value, dt.date) and 0 < (value - as_of_date).days <= 160
    )
    if not expiries:
        return None
    primary_candidates = [
        value for value in expiries if 60 <= (value - as_of_date).days <= 110
    ]
    if not primary_candidates:
        return None
    if preferred_expiry in primary_candidates:
        primary = preferred_expiry
    else:
        primary = min(primary_candidates, key=lambda value: abs((value - as_of_date).days - 80))
    far_candidates = [value for value in expiries if (value - primary).days >= 14]
    far = min(far_candidates, key=lambda value: abs((value - primary).days - 28)) if far_candidates else None
    return {"primary": primary, "far": far}


def _select_contract(
    contracts: pd.DataFrame,
    *,
    expiry: dt.date,
    right: str,
    target_strike: float,
) -> pd.Series | None:
    frame = contracts[(contracts["expiry"] == expiry) & contracts["right"].astype(str).str.upper().eq(right)].copy()
    if frame.empty:
        return None
    frame["_distance"] = (pd.to_numeric(frame["strike"], errors="coerce") - target_strike).abs()
    frame["_oi"] = pd.to_numeric(frame.get("open_interest"), errors="coerce").fillna(0.0)
    frame["_volume"] = pd.to_numeric(frame.get("volume"), errors="coerce").fillna(0.0)
    frame = frame.sort_values(["_distance", "_oi", "_volume"], ascending=[True, False, False])
    return frame.iloc[0] if not frame.empty else None


def _option_mid(contract: pd.Series) -> float:
    bid = safe_float(contract.get("bid"))
    ask = safe_float(contract.get("ask"))
    if math.isfinite(bid) and math.isfinite(ask) and bid >= 0 and ask > 0:
        return (bid + ask) / 2.0
    return safe_float(contract.get("mark"))


def _payoff_metrics(
    *,
    option_legs: list[dict[str, Any]],
    natural_cashflow: float,
    spot: float,
    stock_units: float,
    path_dependent: bool,
) -> tuple[float, float]:
    if path_dependent:
        debit = max(0.0, -natural_cashflow)
        return math.nan, debit * 100.0 if debit > 0 else math.nan
    strikes = sorted({safe_float(leg.get("strike")) for leg in option_legs if math.isfinite(safe_float(leg.get("strike")))})
    if not strikes:
        return math.nan, math.nan
    points = [0.0, *strikes]
    values = []
    for terminal in points:
        value = natural_cashflow + stock_units * (terminal - spot)
        for leg in option_legs:
            strike = safe_float(leg.get("strike"))
            quantity = int(leg.get("quantity", 0))
            intrinsic = max(terminal - strike, 0.0) if leg.get("right") == "C" else max(strike - terminal, 0.0)
            value += quantity * intrinsic
        values.append(value)
    upper_slope = stock_units + sum(int(leg.get("quantity", 0)) for leg in option_legs if leg.get("right") == "C")
    max_profit = math.inf if upper_slope > 0 else max(values) * 100.0
    max_loss = math.inf if upper_slope < 0 else max(0.0, -min(values)) * 100.0
    return max_profit, max_loss


def build_generic_strategy_candidate(
    contracts: pd.DataFrame,
    *,
    strategy_key: str,
    spot: float,
    as_of_date: dt.date,
    preferred_expiry: dt.date | None = None,
) -> dict[str, Any]:
    spec = GENERIC_STRATEGY_BY_KEY.get(strategy_key)
    if spec is None:
        return {"live_status": "unsupported_strategy", "live_blocker": f"no generic strategy spec for {strategy_key}"}
    if contracts is None or contracts.empty or not math.isfinite(spot) or spot <= 0:
        return {"live_status": "no_realistic_structure", "live_blocker": "chain or spot unavailable"}
    expiries = _expiry_map(contracts, as_of_date=as_of_date, preferred_expiry=preferred_expiry)
    if expiries is None:
        return {
            "live_status": "no_realistic_structure",
            "live_blocker": "no primary expiry in the 60-110 DTE live/historical parity window",
        }
    if any(leg.expiry_slot == "far" for leg in spec.legs) and expiries.get("far") is None:
        return {"live_status": "no_realistic_structure", "live_blocker": "no second expiry at least 14 days beyond the front expiry"}

    option_legs: list[dict[str, Any]] = []
    option_natural_cashflow = 0.0
    option_mid_cashflow = 0.0
    displayed_sizes: list[float] = []
    quote_widths: list[float] = []
    regular_session = True
    min_oi = math.inf
    min_volume = math.inf
    for template in spec.legs:
        expiry = expiries.get(template.expiry_slot)
        if expiry is None:
            return {"live_status": "no_realistic_structure", "live_blocker": f"missing {template.expiry_slot} expiry"}
        contract = _select_contract(
            contracts,
            expiry=expiry,
            right=template.right,
            target_strike=spot * template.moneyness,
        )
        if contract is None:
            return {"live_status": "no_realistic_structure", "live_blocker": f"missing {template.right} leg for {expiry}"}
        bid = safe_float(contract.get("bid"))
        ask = safe_float(contract.get("ask"))
        mid = _option_mid(contract)
        if not all(math.isfinite(value) for value in (bid, ask, mid)) or bid < 0 or ask <= 0 or ask < bid:
            return {"live_status": "no_realistic_structure", "live_blocker": "invalid leg market"}
        quantity = int(template.quantity)
        natural_price = ask if quantity > 0 else bid
        option_natural_cashflow -= quantity * natural_price
        option_mid_cashflow -= quantity * mid
        side_size = safe_float(contract.get("ask_size" if quantity > 0 else "bid_size"), 0.0)
        displayed_sizes.append(math.floor(side_size / abs(quantity)) if quantity else 0.0)
        quote_widths.append((ask - bid) / mid if mid > 0 else math.inf)
        regular_session = regular_session and bool(contract.get("regular_session_quote"))
        min_oi = min(min_oi, safe_float(contract.get("open_interest"), 0.0))
        min_volume = min(min_volume, safe_float(contract.get("volume"), 0.0))
        option_legs.append(
            {
                "instrument": "option",
                "symbol": str(contract.get("symbol") or ""),
                "right": template.right,
                "quantity": quantity,
                "side": "BUY" if quantity > 0 else "SELL",
                "expiry": expiry.isoformat(),
                "strike": safe_float(contract.get("strike")),
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "bid_size": safe_float(contract.get("bid_size"), 0.0),
                "ask_size": safe_float(contract.get("ask_size"), 0.0),
                "open_interest": safe_float(contract.get("open_interest"), 0.0),
                "volume": safe_float(contract.get("volume"), 0.0),
                "quote_timestamp": str(contract.get("quote_timestamp") or ""),
            }
        )

    max_profit, max_loss = _payoff_metrics(
        option_legs=option_legs,
        natural_cashflow=option_natural_cashflow,
        spot=spot,
        stock_units=spec.stock_units,
        path_dependent=spec.path_dependent,
    )
    entry_type = "credit" if option_natural_cashflow >= 0 else "debit"
    natural_price = abs(option_natural_cashflow)
    mid_price = abs(option_mid_cashflow)
    short_options = [leg for leg in option_legs if int(leg["quantity"]) < 0]
    long_options = [leg for leg in option_legs if int(leg["quantity"]) > 0]
    primary_expiry = expiries["primary"]
    cash_requirement = 0.0
    if spec.requires_cash_secured:
        short_puts = [leg for leg in short_options if leg["right"] == "P"]
        if short_puts:
            cash_requirement = max(float(leg["strike"]) * abs(int(leg["quantity"])) * 100.0 for leg in short_puts)
    all_legs = list(option_legs)
    if spec.stock_units:
        all_legs.insert(
            0,
            {
                "instrument": "stock",
                "symbol": "UNDERLYING",
                "quantity": int(spec.stock_units * 100),
                "side": "BUY_OR_HOLD",
                "price": spot,
            },
        )
    return {
        "live_status": "PASS",
        "strategy_key": spec.key,
        "strategy_registry_key": spec.key,
        "strategy": spec.display_name,
        "direction": spec.direction,
        "strategy_kind": spec.strategy_kind,
        "leg_count": len(option_legs) + (1 if spec.stock_units else 0),
        "legs_json": json.dumps(all_legs, sort_keys=True),
        "option_legs_json": json.dumps(option_legs, sort_keys=True),
        "expiry": primary_expiry,
        "far_expiry": expiries.get("far"),
        "dte": (primary_expiry - as_of_date).days,
        "short_leg": short_options[0]["symbol"] if short_options else "",
        "long_leg": long_options[0]["symbol"] if long_options else "",
        "short_strike": short_options[0]["strike"] if short_options else math.nan,
        "long_strike": long_options[0]["strike"] if long_options else math.nan,
        "natural_credit": round(natural_price, 2) if entry_type == "credit" else math.nan,
        "mid_credit": round(mid_price, 2) if entry_type == "credit" else math.nan,
        "credit": round(natural_price, 2) if entry_type == "credit" else math.nan,
        "natural_debit": round(natural_price, 2) if entry_type == "debit" else math.nan,
        "mid_debit": round(mid_price, 2) if entry_type == "debit" else math.nan,
        "debit": round(natural_price, 2) if entry_type == "debit" else math.nan,
        "entry_type": entry_type,
        "entry_price": round(natural_price, 2),
        "target_entry": round(natural_price, 2),
        "max_profit": max_profit,
        "max_loss": max_loss,
        "displayed_entry_size": min(displayed_sizes) if displayed_sizes else 0.0,
        "regular_session_quote": regular_session,
        "quote_width_pct": max(quote_widths) if quote_widths else math.inf,
        "liq_score": min_oi if math.isfinite(min_oi) else 0.0,
        "min_leg_open_interest": min_oi if math.isfinite(min_oi) else 0.0,
        "min_leg_volume": min_volume if math.isfinite(min_volume) else 0.0,
        "short_oi": min_oi if math.isfinite(min_oi) else 0.0,
        "short_volume": min_volume if math.isfinite(min_volume) else 0.0,
        "long_oi": min_oi if math.isfinite(min_oi) else 0.0,
        "long_volume": min_volume if math.isfinite(min_volume) else 0.0,
        "requires_equity_shares": spec.requires_equity_shares,
        "requires_cash": cash_requirement,
        "requires_margin_model": spec.requires_margin_model,
        "path_dependent_structure": spec.path_dependent,
        "stock_units": spec.stock_units,
        "construction_source": "generic_live_strategy_builder",
        "construction_reason": "arbitrary-leg natural-side construction from the current Schwab chain",
        "execution_authority": "research_only_pending_strategy_validation",
    }
