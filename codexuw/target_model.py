from __future__ import annotations

import calendar
import datetime as dt
import math
from typing import Any

import pandas as pd

from .data import safe_float


MIN_LIVE_OUTCOMES_FOR_TARGET_CONFIDENCE = 50
MIN_LIVE_PROFIT_FACTOR_FOR_TARGET_CONFIDENCE = 1.25


def business_days_remaining(asof: dt.date) -> int:
    last_day = dt.date(asof.year, asof.month, calendar.monthrange(asof.year, asof.month)[1])
    day = asof
    count = 0
    while day <= last_day:
        if day.weekday() < 5:
            count += 1
        day += dt.timedelta(days=1)
    return max(1, count)


def _parse_money(value: object) -> float:
    if isinstance(value, str):
        value = value.replace("$", "").replace(",", "").strip()
    return safe_float(value)


def _opportunity_numbers(board: pd.DataFrame) -> dict[str, float]:
    if board.empty:
        return {
            "qualified_expected_profit": 0.0,
            "qualified_target_profit": 0.0,
            "qualified_max_loss": 0.0,
            "avg_profit_per_trade": math.nan,
            "avg_target_profit_per_trade": math.nan,
            "avg_risk_per_trade": math.nan,
            "qualified_trade_count": 0,
            "raw_execute_count": 0,
        }
    executable = board[board["Status"].astype(str).str.contains("Execute", regex=False)].copy()
    if executable.empty:
        return {
            "qualified_expected_profit": 0.0,
            "qualified_target_profit": 0.0,
            "qualified_max_loss": 0.0,
            "avg_profit_per_trade": math.nan,
            "avg_target_profit_per_trade": math.nan,
            "avg_risk_per_trade": math.nan,
            "qualified_trade_count": 0,
            "raw_execute_count": 0,
        }
    raw_execute_count = int(len(executable))
    target = (
        pd.to_numeric(executable["target_profit_total"], errors="coerce")
        if "target_profit_total" in executable.columns
        else executable["Target profit"].map(_parse_money)
    ).fillna(0.0)
    expected = (
        pd.to_numeric(executable["expected_value_total"], errors="coerce")
        if "expected_value_total" in executable.columns
        else pd.Series(math.nan, index=executable.index, dtype=float)
    )
    contracts = (
        pd.to_numeric(executable["contracts"], errors="coerce")
        if "contracts" in executable.columns
        else pd.Series(1.0, index=executable.index, dtype=float)
    ).fillna(1.0).clip(lower=1.0)
    one_lot_loss = (
        pd.to_numeric(executable["max_loss"], errors="coerce")
        if "max_loss" in executable.columns
        else executable["Max loss"].map(_parse_money)
    ).fillna(0.0)
    position_loss = (
        pd.to_numeric(executable["position_max_loss"], errors="coerce")
        if "position_max_loss" in executable.columns
        else pd.Series(math.nan, index=executable.index, dtype=float)
    )
    position_loss = position_loss.where(position_loss.notna(), one_lot_loss * contracts).fillna(0.0)

    executable["_target_profit"] = target
    executable["_expected_profit"] = expected
    executable["_position_max_loss"] = position_loss
    ticker_col = "ticker" if "ticker" in executable.columns else "Ticker" if "Ticker" in executable.columns else ""
    thesis_keys = [col for col in [ticker_col, "direction"] if col and col in executable.columns]
    if thesis_keys:
        executable = executable.sort_values(
            ["_expected_profit", "_target_profit"],
            ascending=[False, False],
            na_position="last",
        ).drop_duplicates(subset=thesis_keys, keep="first")
    target = executable["_target_profit"].fillna(0.0)
    expected = executable["_expected_profit"]
    max_loss = executable["_position_max_loss"].fillna(0.0)
    expected_positive = expected[expected.notna()]
    return {
        "qualified_expected_profit": float(expected_positive.sum()) if not expected_positive.empty else 0.0,
        "qualified_target_profit": float(target.sum()),
        "qualified_max_loss": float(max_loss.sum()),
        "avg_profit_per_trade": float(expected_positive.mean()) if not expected_positive.empty else math.nan,
        "avg_target_profit_per_trade": float(target[target > 0].mean()) if (target > 0).any() else math.nan,
        "avg_risk_per_trade": float(max_loss[max_loss > 0].mean()) if (max_loss > 0).any() else math.nan,
        "qualified_trade_count": int(len(executable)),
        "raw_execute_count": raw_execute_count,
    }


def build_v3_target_model(
    *,
    asof: dt.date,
    board: pd.DataFrame,
    monthly_profit_target: float = 10_000.0,
    month_to_date_realized_pnl: float = 0.0,
    open_unrealized_pnl: float = 0.0,
    account_value: float = math.nan,
    available_cash: float = math.nan,
    risk_budget: float = 0.0,
    max_daily_loss: float = 0.0,
    max_weekly_loss: float = 0.0,
    max_monthly_loss: float = 0.0,
    historical_win_rate: float = math.nan,
    average_realized_win: float = math.nan,
    average_realized_loss: float = math.nan,
    expected_trade_frequency: float = math.nan,
    live_outcome_status: str = "unavailable",
    live_outcome_count: int = 0,
    live_outcome_profit_factor: float = math.nan,
    minimum_live_outcomes: int = MIN_LIVE_OUTCOMES_FOR_TARGET_CONFIDENCE,
) -> dict[str, Any]:
    days = business_days_remaining(asof)
    weeks = max(1.0, days / 5.0)
    realized = safe_float(month_to_date_realized_pnl, 0.0)
    unrealized = safe_float(open_unrealized_pnl, 0.0)
    target = safe_float(monthly_profit_target, 10_000.0)
    remaining = max(0.0, target - realized - unrealized)
    required_daily = remaining / days
    required_weekly = remaining / weeks
    opp = _opportunity_numbers(board)
    avg_profit = safe_float(opp["avg_profit_per_trade"])
    avg_risk = safe_float(opp["avg_risk_per_trade"])
    qualified_expected = opp["qualified_expected_profit"]
    qualified_target = opp["qualified_target_profit"]
    risk_available = safe_float(risk_budget, 0.0)
    if math.isfinite(available_cash):
        risk_available = min(risk_available, max(0.0, available_cash)) if risk_available > 0 else max(0.0, available_cash)

    if math.isfinite(avg_profit) and avg_profit > 0:
        trades_required = math.ceil(remaining / avg_profit) if remaining > 0 else 0
        required_avg_profit_per_trade = required_daily / max(1, opp["qualified_trade_count"])
    else:
        trades_required = math.inf if remaining > 0 else 0
        required_avg_profit_per_trade = required_daily

    if math.isfinite(avg_profit) and avg_profit > 0 and math.isfinite(avg_risk) and avg_risk > 0:
        risk_per_profit = avg_risk / avg_profit
        risk_required = remaining * risk_per_profit
    elif remaining > 0:
        risk_required = math.inf
    else:
        risk_required = 0.0

    if math.isfinite(expected_trade_frequency) and expected_trade_frequency > 0 and math.isfinite(avg_profit):
        expected_monthly_run_rate = expected_trade_frequency * avg_profit
    else:
        expected_monthly_run_rate = qualified_expected * days

    probability_adjusted_by_lane: dict[str, float] = {}
    if not board.empty:
        for lane, part in board.groupby("Lane"):
            lane_execute = part[part["Status"].astype(str).str.contains("Execute", regex=False)]
            if "expected_value_total" in lane_execute.columns:
                lane_expected = pd.to_numeric(lane_execute["expected_value_total"], errors="coerce").dropna()
                probability_adjusted_by_lane[str(lane)] = float(lane_expected.sum()) if not lane_expected.empty else 0.0
            else:
                probability_adjusted_by_lane[str(lane)] = 0.0

    outcome_count = int(safe_float(live_outcome_count, 0.0))
    try:
        outcome_pf = float(live_outcome_profit_factor)
    except (TypeError, ValueError):
        outcome_pf = math.nan
    live_evidence_ok = bool(
        str(live_outcome_status or "").lower() == "ok"
        and outcome_count >= minimum_live_outcomes
        and (math.isinf(outcome_pf) or (math.isfinite(outcome_pf) and outcome_pf >= MIN_LIVE_PROFIT_FACTOR_FOR_TARGET_CONFIDENCE))
    )

    if remaining <= 0:
        feasibility = "feasible"
        reason = "monthly target already met"
    elif not live_evidence_ok:
        feasibility = "not demonstrated"
        reason = (
            f"requires at least {minimum_live_outcomes} closed live outcomes with profit factor "
            f">= {MIN_LIVE_PROFIT_FACTOR_FOR_TARGET_CONFIDENCE:.2f}; currently {outcome_count}"
        )
    elif risk_available <= 0:
        feasibility = "mathematically impossible under current risk caps"
        reason = "risk budget too small"
    elif qualified_expected <= 0:
        feasibility = "infeasible"
        reason = "not enough valid trades"
    elif not math.isfinite(risk_required) or risk_required > risk_available:
        feasibility = "infeasible"
        reason = "risk required exceeds risk available"
    elif qualified_expected >= required_daily and expected_monthly_run_rate >= remaining:
        feasibility = "feasible"
        reason = "current qualified opportunity run-rate can meet remaining target within risk caps"
    elif qualified_expected >= required_daily * 0.50:
        feasibility = "stretched"
        reason = "qualified opportunity expected P/L is below required daily pace"
    else:
        feasibility = "infeasible"
        reason = "average edge too small"

    drawdown_required = risk_required if math.isfinite(risk_required) else None
    return {
        "monthly_profit_target": round(target, 2),
        "month_to_date_realized_pnl": round(realized, 2),
        "open_unrealized_pnl": round(unrealized, 2),
        "remaining_monthly_target": round(remaining, 2),
        "business_days_remaining": days,
        "required_daily_pl": round(required_daily, 2),
        "required_weekly_pl": round(required_weekly, 2),
        "required_average_profit_per_trade": round(required_avg_profit_per_trade, 2) if math.isfinite(required_avg_profit_per_trade) else None,
        "required_number_of_trades_at_current_edge": int(trades_required) if math.isfinite(trades_required) else None,
        "required_risk_to_pursue_target": round(risk_required, 2) if math.isfinite(risk_required) else None,
        "risk_available": round(risk_available, 2),
        "current_qualified_opportunity_expected_pl": round(qualified_expected, 2),
        "current_qualified_opportunity_target_profit": round(qualified_target, 2),
        "current_qualified_opportunity_max_loss": round(opp["qualified_max_loss"], 2),
        "raw_execute_count": int(opp["raw_execute_count"]),
        "deduplicated_execute_thesis_count": int(opp["qualified_trade_count"]),
        "expected_monthly_run_rate_from_current_qualified_opportunities": round(expected_monthly_run_rate, 2),
        "probability_adjusted_expectancy_by_lane": probability_adjusted_by_lane,
        "drawdown_required_to_attempt_target": round(drawdown_required, 2) if drawdown_required is not None else None,
        "target_feasibility": feasibility,
        "target_gap": {
            "dollars_remaining": round(remaining, 2),
            "trades_required": int(trades_required) if math.isfinite(trades_required) else None,
            "risk_required": round(risk_required, 2) if math.isfinite(risk_required) else None,
            "risk_available": round(risk_available, 2),
            "expected_monthly_run_rate": round(expected_monthly_run_rate, 2),
        },
        "explicit_infeasible_reason": "" if feasibility in {"feasible", "stretched"} else reason,
        "binding_constraint": reason,
        "risk_inputs": {
            "account_value": safe_float(account_value),
            "available_cash": safe_float(available_cash),
            "max_daily_loss": safe_float(max_daily_loss, 0.0),
            "max_weekly_loss": safe_float(max_weekly_loss, 0.0),
            "max_monthly_loss": safe_float(max_monthly_loss, 0.0),
            "historical_win_rate": safe_float(historical_win_rate),
            "average_realized_win": safe_float(average_realized_win),
            "average_realized_loss": safe_float(average_realized_loss),
            "live_outcome_status": str(live_outcome_status or "unavailable"),
            "live_outcome_count": outcome_count,
            "live_outcome_profit_factor": outcome_pf,
            "minimum_live_outcomes": int(minimum_live_outcomes),
            "live_target_evidence_ok": live_evidence_ok,
        },
    }
