from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from codexuw.credit_policy import (
    MAX_CREDIT_PCT_WIDTH,
    MAX_DTE,
    MIN_CREDIT_PCT_WIDTH,
    MIN_DTE,
)
from codexuw.debit_policy import DEBIT_POLICY
from codexuw.strategy_builder import GENERIC_STRATEGY_SPECS

MAX_DEBIT_PCT_WIDTH = float(DEBIT_POLICY["Bull Call"]["max_debit_pct_width"])
GENERIC_DTE_BAND = (60, 110)
GENERIC_TARGET_DTE = 80
GENERIC_HOLD_DAYS = 40
CREDIT_HOLD_DAYS = 21
DEBIT_HOLD_DAYS = 14


@dataclass(frozen=True)
class LegSpec:
    option_type: str
    moneyness: float
    quantity: int
    expiry_slot: str = "primary"


@dataclass(frozen=True)
class StrategySpec:
    key: str
    selection_bucket: str
    risk_model: str
    legs: tuple[LegSpec, ...]
    stock_units: float = 0.0
    requires_cash_secured: bool = False
    requires_margin_model: bool = False
    path_dependent: bool = False
    historical_scope: str = "exact_option_exit"
    # Live-parity admission rules. A study run under looser rules than live
    # execution enforces describes a population the pipeline can never trade.
    dte_band: tuple[int, int] = GENERIC_DTE_BAND
    target_dte: int = GENERIC_TARGET_DTE
    entry_side: str = ""
    entry_pct_width_band: tuple[float, float] | None = None
    screen_earnings_before_expiry: bool = False
    # Exit must land inside the structure's life or liquidation silently drops
    # every contract that expired first, biasing the sample to the longest DTE.
    hold_days: int = GENERIC_HOLD_DAYS


# Mirrors codexuw.credit_policy / codexuw.debit_policy so the tested structure is
# the structure the live builder emits.
CREDIT_DTE_BAND = (MIN_DTE, MAX_DTE)
CREDIT_TARGET_DTE = 35
DEBIT_DTE_BAND = (22, 45)
DEBIT_TARGET_DTE = 35

VERTICAL_STRATEGY_SPECS: tuple[StrategySpec, ...] = (
    StrategySpec(
        "bull_call_debit_vertical", "bullish", "vertical",
        (LegSpec("call", 1.02, 1), LegSpec("call", 1.12, -1)),
        dte_band=DEBIT_DTE_BAND, target_dte=DEBIT_TARGET_DTE,
        entry_side="debit", entry_pct_width_band=(0.0, MAX_DEBIT_PCT_WIDTH),
        screen_earnings_before_expiry=True, hold_days=DEBIT_HOLD_DAYS,
    ),
    StrategySpec(
        "bear_put_debit_vertical", "bearish", "vertical",
        (LegSpec("put", 0.98, 1), LegSpec("put", 0.88, -1)),
        dte_band=DEBIT_DTE_BAND, target_dte=DEBIT_TARGET_DTE,
        entry_side="debit", entry_pct_width_band=(0.0, MAX_DEBIT_PCT_WIDTH),
        screen_earnings_before_expiry=True, hold_days=DEBIT_HOLD_DAYS,
    ),
    StrategySpec(
        "bull_put_credit_vertical", "bullish", "vertical",
        (LegSpec("put", 0.95, -1), LegSpec("put", 0.85, 1)),
        dte_band=CREDIT_DTE_BAND, target_dte=CREDIT_TARGET_DTE,
        entry_side="credit",
        entry_pct_width_band=(MIN_CREDIT_PCT_WIDTH, MAX_CREDIT_PCT_WIDTH),
        screen_earnings_before_expiry=True, hold_days=CREDIT_HOLD_DAYS,
    ),
    StrategySpec(
        "bear_call_credit_vertical", "bearish", "vertical",
        (LegSpec("call", 1.05, -1), LegSpec("call", 1.15, 1)),
        dte_band=CREDIT_DTE_BAND, target_dte=CREDIT_TARGET_DTE,
        entry_side="credit",
        entry_pct_width_band=(MIN_CREDIT_PCT_WIDTH, MAX_CREDIT_PCT_WIDTH),
        screen_earnings_before_expiry=True, hold_days=CREDIT_HOLD_DAYS,
    ),
)


def _selection_bucket(key: str, direction: str) -> str:
    if key in {"long_straddle", "long_strangle"}:
        return "volatility"
    if key in {
        "short_straddle",
        "short_strangle",
        "iron_condor",
        "iron_butterfly",
        "covered_strangle",
    }:
        return "range"
    bearish = any(token in key for token in ("put", "reverse_jade")) and key not in {
        "cash_secured_put",
        "protective_put",
        "bull_put_credit_vertical",
    }
    if "bearish" in direction.lower() or bearish:
        return "bearish"
    return "bullish"


def _historical_scope(key: str, *, margin: bool, path_dependent: bool, stock_units: float) -> str:
    if key == "wheel":
        return "entry_csp_cycle"
    if path_dependent:
        return "exact_term_structure_exit"
    if margin:
        return "exact_pnl_conservative_reg_t_risk"
    if stock_units:
        return "exact_option_and_stock_exit"
    return "exact_option_exit"


GENERIC_HISTORICAL_STRATEGY_SPECS: tuple[StrategySpec, ...] = tuple(
    StrategySpec(
        key=spec.key,
        selection_bucket=_selection_bucket(spec.key, spec.direction),
        risk_model=(
            "margin_proxy"
            if spec.requires_margin_model
            else "term_structure"
            if spec.path_dependent and spec.key != "wheel"
            else "stock_backed"
            if spec.stock_units
            else "cash_secured"
            if spec.requires_cash_secured
            else "option_payoff"
        ),
        legs=tuple(
            LegSpec(
                option_type="call" if leg.right == "C" else "put",
                moneyness=leg.moneyness,
                quantity=leg.quantity,
                expiry_slot=leg.expiry_slot,
            )
            for leg in spec.legs
        ),
        stock_units=spec.stock_units,
        requires_cash_secured=spec.requires_cash_secured,
        requires_margin_model=spec.requires_margin_model,
        path_dependent=spec.path_dependent,
        historical_scope=_historical_scope(
            spec.key,
            margin=spec.requires_margin_model,
            path_dependent=spec.path_dependent,
            stock_units=spec.stock_units,
        ),
    )
    for spec in GENERIC_STRATEGY_SPECS
)


HISTORICAL_STRATEGY_SPECS: tuple[StrategySpec, ...] = (
    *GENERIC_HISTORICAL_STRATEGY_SPECS,
    *VERTICAL_STRATEGY_SPECS,
)
MIN_MAX_RISK_PER_SHARE = 0.05


def build_sector_state(panel: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "sector", "ticker", "pos_52w", "ret_1d", "flow_escalation"}
    if panel is None or panel.empty or not required.issubset(panel.columns):
        return pd.DataFrame()
    work = panel.dropna(subset=["date", "sector", "ticker", "pos_52w"]).copy()
    if work.empty:
        return pd.DataFrame()
    grouped = work.groupby(["date", "sector"], dropna=False)
    state = grouped.agg(
        sector_tickers=("ticker", "nunique"),
        sector_median_52w_position=("pos_52w", "median"),
        sector_breadth=("pos_52w", lambda values: float((values >= 0.60).mean())),
        sector_return_1d=("ret_1d", "median"),
        sector_flow_acceleration=("flow_escalation", "median"),
    ).reset_index()
    state = state.sort_values(["sector", "date"]).reset_index(drop=True)
    by_sector = state.groupby("sector", dropna=False)
    state["sector_momentum_change_5s"] = by_sector["sector_median_52w_position"].diff(5)
    state["sector_breadth_change_5s"] = by_sector["sector_breadth"].diff(5)
    rank_columns = {
        "sector_momentum_change_5s": 0.30,
        "sector_breadth_change_5s": 0.25,
        "sector_median_52w_position": 0.15,
        "sector_return_1d": 0.15,
        "sector_flow_acceleration": 0.15,
    }
    state["sector_emergence_score"] = 0.0
    for column, weight in rank_columns.items():
        values = pd.to_numeric(state[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        ranks = values.groupby(state["date"]).rank(pct=True).fillna(0.5)
        state["sector_emergence_score"] += ranks * weight
    state["sector_state"] = np.select(
        [
            state["sector_emergence_score"].ge(0.70)
            & state["sector_momentum_change_5s"].gt(0)
            & state["sector_breadth_change_5s"].ge(0),
            state["sector_median_52w_position"].ge(0.65),
            state["sector_emergence_score"].le(0.30)
            & state["sector_momentum_change_5s"].lt(0),
        ],
        ["emerging", "established_strength", "weakening"],
        default="mixed",
    )
    return state


def selection_buckets(block: pd.DataFrame, *, percentile: float = 0.80) -> dict[str, set[str]]:
    momentum = pd.to_numeric(block["pos_52w"], errors="coerce").rank(pct=True)
    escalation = pd.to_numeric(block["flow_escalation"], errors="coerce").rank(pct=True)
    centered = (pd.to_numeric(block["pos_52w"], errors="coerce") - 0.50).abs()
    range_rank = centered.rank(pct=True, ascending=False)
    return {
        "bullish": set(block.loc[momentum >= percentile, "ticker"]),
        "bearish": set(block.loc[momentum <= 1.0 - percentile, "ticker"]),
        "volatility": set(block.loc[escalation >= percentile, "ticker"]),
        "range": set(block.loc[range_rank >= percentile, "ticker"]),
    }


def _nearest_by_expiry(
    quotes: pd.DataFrame,
    tickers: set[str],
    leg: LegSpec,
    *,
    dte_band: tuple[int, int],
    min_open_interest: int,
    max_spread_pct: float,
    suffix: str,
) -> pd.DataFrame:
    rows = quotes[
        quotes["ticker"].isin(tickers)
        & quotes["option_type"].eq(leg.option_type)
        & quotes["dte"].between(*dte_band)
        & quotes["curr_oi"].ge(min_open_interest)
        & quotes["spread_pct"].le(max_spread_pct)
    ].copy()
    if rows.empty:
        return rows
    rows["_strike_gap"] = (rows["strike"] - rows["stock_price"] * leg.moneyness).abs()
    rows = rows.sort_values("_strike_gap").groupby(["ticker", "expiry"], as_index=False).first()
    return rows[
        ["ticker", "expiry", "dte", "stock_price", "option_symbol", "strike", "last_bid", "last_ask", "_strike_gap"]
    ].rename(
        columns={
            "dte": f"dte_{suffix}",
            "stock_price": f"stock_price_{suffix}",
            "option_symbol": f"symbol_{suffix}",
            "strike": f"strike_{suffix}",
            "last_bid": f"bid_{suffix}",
            "last_ask": f"ask_{suffix}",
            "_strike_gap": f"strike_gap_{suffix}",
        }
    )


def build_structure(
    quotes: pd.DataFrame,
    tickers: Iterable[str],
    spec: StrategySpec,
    *,
    target_dte: int | None = None,
    dte_band: tuple[int, int] | None = None,
    min_open_interest: int = 50,
    max_spread_pct: float = 0.12,
    earnings_by_ticker: pd.Series | None = None,
) -> pd.DataFrame:
    wanted = {str(ticker) for ticker in tickers}
    if not wanted:
        return pd.DataFrame()
    dte_band = spec.dte_band if dte_band is None else dte_band
    target_dte = spec.target_dte if target_dte is None else target_dte
    selected_frames: dict[int, pd.DataFrame] = {}
    for index, leg in enumerate(spec.legs):
        suffix = str(index)
        leg_dte_band = (
            dte_band
            if leg.expiry_slot == "primary"
            else (dte_band[0] + 14, dte_band[1] + 45)
        )
        selected = _nearest_by_expiry(
            quotes,
            wanted,
            leg,
            dte_band=leg_dte_band,
            min_open_interest=min_open_interest,
            max_spread_pct=max_spread_pct,
            suffix=suffix,
        )
        if selected.empty:
            return pd.DataFrame()
        selected_frames[index] = selected.rename(columns={"expiry": f"expiry_{index}"})

    primary_indices = [index for index, leg in enumerate(spec.legs) if leg.expiry_slot == "primary"]
    far_indices = [index for index, leg in enumerate(spec.legs) if leg.expiry_slot == "far"]
    if not primary_indices:
        return pd.DataFrame()
    first_primary = primary_indices[0]
    primary_expiry_column = f"expiry_{first_primary}"
    merged = selected_frames[first_primary]
    for index in primary_indices[1:]:
        merged = merged.merge(
            selected_frames[index],
            left_on=["ticker", primary_expiry_column],
            right_on=["ticker", f"expiry_{index}"],
            how="inner",
        )
    far_expiry_column = ""
    if far_indices and not merged.empty:
        first_far = far_indices[0]
        far_expiry_column = f"expiry_{first_far}"
        merged = merged.merge(selected_frames[first_far], on="ticker", how="inner")
        primary_dates = pd.to_datetime(merged[primary_expiry_column], errors="coerce")
        far_dates = pd.to_datetime(merged[far_expiry_column], errors="coerce")
        far_gap = (far_dates - primary_dates).dt.days
        merged = merged[far_gap.ge(14)].copy()
        if merged.empty:
            return merged
        merged["_far_expiry_score"] = (far_gap[merged.index] - 28).abs()
        merged = (
            merged.sort_values(["ticker", primary_expiry_column, "_far_expiry_score"])
            .groupby(["ticker", primary_expiry_column], as_index=False)
            .head(1)
        )
        for index in far_indices[1:]:
            merged = merged.merge(
                selected_frames[index],
                left_on=["ticker", far_expiry_column],
                right_on=["ticker", f"expiry_{index}"],
                how="inner",
            )
    if merged.empty:
        return merged
    symbol_columns = [f"symbol_{index}" for index in range(len(spec.legs))]
    merged = merged[merged[symbol_columns].nunique(axis=1).eq(len(symbol_columns))].copy()
    if merged.empty:
        return merged
    merged["_expiry_score"] = (
        pd.to_numeric(merged[f"dte_{first_primary}"], errors="coerce") - target_dte
    ).abs()
    for index in range(len(spec.legs)):
        stock = pd.to_numeric(merged[f"stock_price_{index}"], errors="coerce").replace(0, np.nan)
        merged["_expiry_score"] += pd.to_numeric(merged[f"strike_gap_{index}"], errors="coerce") / stock
    if "_far_expiry_score" in merged.columns:
        merged["_expiry_score"] += merged["_far_expiry_score"] / 28.0
    merged = merged.sort_values("_expiry_score").groupby("ticker", as_index=False).first()
    entry_cashflow = pd.Series(0.0, index=merged.index)
    for index, leg in enumerate(spec.legs):
        if leg.quantity > 0:
            entry_cashflow -= leg.quantity * pd.to_numeric(merged[f"ask_{index}"], errors="coerce")
        else:
            entry_cashflow += abs(leg.quantity) * pd.to_numeric(merged[f"bid_{index}"], errors="coerce")
        merged[f"quantity_{index}"] = leg.quantity
    merged["entry_cashflow"] = entry_cashflow
    merged["expiry"] = merged[primary_expiry_column]
    merged["far_expiry"] = merged[far_expiry_column] if far_expiry_column else pd.NaT
    merged["entry_stock_price"] = pd.to_numeric(
        merged[f"stock_price_{first_primary}"], errors="coerce"
    )
    merged = _apply_live_entry_rules(merged, spec, earnings_by_ticker)
    if merged.empty:
        return merged
    risk_rows = merged.apply(lambda row: pd.Series(_historical_risk_metrics(row, spec)), axis=1)
    merged = pd.concat([merged, risk_rows], axis=1)
    merged = merged[
        pd.to_numeric(merged["max_risk_per_share"], errors="coerce").ge(MIN_MAX_RISK_PER_SHARE)
    ].copy()
    merged["strategy"] = spec.key
    merged["leg_count"] = len(spec.legs)
    merged["historical_scope"] = spec.historical_scope
    return merged


def _apply_live_entry_rules(
    merged: pd.DataFrame,
    spec: StrategySpec,
    earnings_by_ticker: pd.Series | None,
) -> pd.DataFrame:
    """Admit only structures the live builder would also accept."""
    if merged.empty:
        return merged
    if spec.entry_pct_width_band is not None and len(spec.legs) == 2:
        width = (
            pd.to_numeric(merged["strike_0"], errors="coerce")
            - pd.to_numeric(merged["strike_1"], errors="coerce")
        ).abs()
        cashflow = pd.to_numeric(merged["entry_cashflow"], errors="coerce")
        signed = cashflow if spec.entry_side == "credit" else -cashflow
        ratio = signed / width.replace(0, np.nan)
        low, high = spec.entry_pct_width_band
        merged = merged[ratio.between(low, high)].copy()
        if merged.empty:
            return merged
    if spec.screen_earnings_before_expiry and earnings_by_ticker is not None:
        earnings = pd.to_datetime(
            merged["ticker"].astype(str).map(earnings_by_ticker), errors="coerce"
        )
        expiry = pd.to_datetime(merged["expiry"], errors="coerce")
        merged = merged[~earnings.notna() | earnings.gt(expiry)].copy()
    return merged


def _terminal_payoff_bounds(row: pd.Series, spec: StrategySpec) -> tuple[float, float]:
    entry_cashflow = float(row["entry_cashflow"])
    entry_spot = float(row["entry_stock_price"])
    strikes = [float(row[f"strike_{index}"]) for index in range(len(spec.legs))]
    high = max([entry_spot, *strikes]) * 3.0
    points = sorted({0.0, entry_spot, high, *strikes})
    values = []
    for terminal in points:
        value = entry_cashflow + spec.stock_units * (terminal - entry_spot)
        for index, leg in enumerate(spec.legs):
            strike = float(row[f"strike_{index}"])
            intrinsic = (
                max(terminal - strike, 0.0)
                if leg.option_type == "call"
                else max(strike - terminal, 0.0)
            )
            value += leg.quantity * intrinsic
        values.append(value)
    upper_slope = spec.stock_units + sum(
        leg.quantity for leg in spec.legs if leg.option_type == "call"
    )
    max_profit = math.inf if upper_slope > 0 else max(values)
    max_loss = math.inf if upper_slope < 0 else max(0.0, -min(values))
    return max_profit, max_loss


def _conservative_reg_t_requirement(row: pd.Series, spec: StrategySpec) -> float:
    spot = float(row["entry_stock_price"])
    requirement = 0.0
    for index, leg in enumerate(spec.legs):
        if leg.quantity >= 0:
            continue
        strike = float(row[f"strike_{index}"])
        premium = float(row[f"bid_{index}"])
        out_of_money = max(strike - spot, 0.0) if leg.option_type == "call" else max(spot - strike, 0.0)
        per_contract = premium + max(0.20 * spot - out_of_money, 0.10 * spot)
        requirement += abs(leg.quantity) * max(per_contract, premium)
    return max(requirement, max(0.0, -float(row["entry_cashflow"])))


def _historical_risk_metrics(row: pd.Series, spec: StrategySpec) -> dict[str, object]:
    if spec.risk_model == "term_structure":
        debit = max(0.0, -float(row["entry_cashflow"]))
        risk = debit if debit > 0 else max(0.20 * float(row["entry_stock_price"]), 0.05)
        return {
            "max_profit_per_share": math.nan,
            "max_risk_per_share": risk,
            "risk_capital_model": "term_structure_debit_or_20pct_spot",
        }
    max_profit, max_loss = _terminal_payoff_bounds(row, spec)
    if math.isfinite(max_loss) and max_loss > 0:
        risk = max_loss
        model = "terminal_max_loss"
    elif spec.requires_margin_model:
        risk = _conservative_reg_t_requirement(row, spec)
        model = "conservative_reg_t_proxy"
    else:
        risk = max(0.0, -float(row["entry_cashflow"]))
        model = "entry_debit"
    return {
        "max_profit_per_share": max_profit,
        "max_risk_per_share": risk,
        "risk_capital_model": model,
    }


def liquidate_structure(
    structures: pd.DataFrame,
    exit_quotes: pd.DataFrame,
    spec: StrategySpec,
    *,
    contract_fee: float = 1.30,
) -> pd.DataFrame:
    if structures is None or structures.empty or exit_quotes is None or exit_quotes.empty:
        return pd.DataFrame()
    out = structures.copy()
    indexed = exit_quotes.drop_duplicates("option_symbol").set_index("option_symbol")
    exit_cashflow = pd.Series(0.0, index=out.index)
    valid = pd.Series(True, index=out.index)
    contracts = 0
    for index, leg in enumerate(spec.legs):
        symbols = out[f"symbol_{index}"]
        prices = (
            pd.to_numeric(indexed["last_bid"].reindex(symbols).set_axis(out.index), errors="coerce")
            if leg.quantity > 0
            else pd.to_numeric(indexed["last_ask"].reindex(symbols).set_axis(out.index), errors="coerce")
        )
        valid &= prices.notna()
        exit_cashflow += leg.quantity * prices
        contracts += abs(leg.quantity)
    exit_stock = pd.to_numeric(
        out["ticker"].map(
            pd.to_numeric(exit_quotes["stock_price"], errors="coerce")
            .groupby(exit_quotes["ticker"].astype(str))
            .median()
        ),
        errors="coerce",
    )
    if spec.stock_units:
        valid &= exit_stock.notna()
    out = out[valid].copy()
    exit_cashflow = exit_cashflow[valid]
    exit_stock = exit_stock[valid]
    out["exit_cashflow"] = exit_cashflow
    out["exit_stock_price"] = exit_stock
    out["stock_pnl_per_share"] = (
        spec.stock_units * (exit_stock - pd.to_numeric(out["entry_stock_price"], errors="coerce"))
        if spec.stock_units
        else 0.0
    )
    out["pnl"] = (
        out["entry_cashflow"] + exit_cashflow + out["stock_pnl_per_share"]
    ) * 100.0 - contract_fee * contracts
    out["max_risk"] = out["max_risk_per_share"] * 100.0
    out["return_on_risk"] = out["pnl"] / out["max_risk"]
    return out