from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from .confidence_calibration import DEFAULT_EDGE_HISTORY_PATH, wilson_lower_bound


PAYOFF_CALIBRATION_VERSION = "structure-aware-hierarchical-payoff-v2.0"
COST_STRESS_LEVELS = (0.0, 0.05, 0.10)
MIN_GROUP_SAMPLE = 20
MIN_TRAIN_SAMPLE = 10
MIN_OOS_SAMPLE = 5
MIN_PROSPECTIVE_OOS_SAMPLE = 2
MIN_STRESS_PROFIT_FACTOR = 1.25
MIN_TEST_WINDOW_PROFIT_FACTOR = 1.00
MAX_DRAWDOWN_STRESS_MULTIPLE = 1.25
MAX_FAILED_WALK_FORWARD_WINDOW_RATE = 1.0 / 3.0

ROUTE_POLICIES: dict[str, dict[str, Any]] = {
    "base": {
        "key_column": "_route_key_base",
        "minimum_group_sample": 20,
        "minimum_train_sample": 10,
        "minimum_oos_sample": 5,
        "minimum_train_profit_factor": 1.25,
    },
    "flow": {
        "key_column": "_route_key_flow",
        "minimum_group_sample": 15,
        "minimum_train_sample": 8,
        "minimum_oos_sample": 5,
        "minimum_train_profit_factor": 1.00,
    },
    "flow_cost": {
        "key_column": "_route_key_flow_cost",
        "minimum_group_sample": 12,
        "minimum_train_sample": 8,
        "minimum_oos_sample": 5,
        "minimum_train_profit_factor": 1.00,
    },
}
ROUTE_SELECTION_ORDER = ("flow_cost", "flow", "base")


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _number(value: Any, default: float = math.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _family(row: pd.Series | dict[str, Any]) -> str:
    kind = str(row.get("strategy_kind", "")).strip().lower()
    if kind == "credit":
        return "Credit"
    if kind == "debit":
        return "Debit"
    strategy = str(row.get("strategy", "")).lower()
    if "credit" in strategy:
        return "Credit"
    if "debit" in strategy:
        return "Debit"
    return "Unknown"


def _group_key(row: pd.Series | dict[str, Any]) -> str:
    family = _family(row)
    direction = str(row.get("direction", "")).strip()
    regime = str(row.get("regime_trend") or row.get("regime") or "").strip().lower()
    return f"{family}|{direction}|{regime}"


def _flow_bucket(row: pd.Series | dict[str, Any]) -> str:
    raw = str(row.get("flow_quality", "")).strip().lower()
    if raw == "directional":
        return "directional"
    if "contra" in raw:
        return "contra"
    if raw in {"", "nan", "none", "unknown", "missing"}:
        return "unknown"
    return "ambiguous"


def _entry_pct_width(row: pd.Series | dict[str, Any]) -> float:
    direct_fields = ["_entry_pct_width", "entry_pct_width"]
    if _family(row) == "Credit":
        direct_fields.extend(["credit_pct_width", "estimated_credit_pct_width", "estimated_eod_credit_pct_width"])
        price_fields = ["credit", "mid_credit", "natural_credit", "estimated_eod_credit", "entry_price"]
    else:
        direct_fields.extend(["debit_pct_width", "estimated_debit_pct_width", "estimated_eod_debit_pct_width"])
        price_fields = ["debit", "mid_debit", "natural_debit", "estimated_eod_debit", "entry_price"]
    for field in direct_fields:
        value = _number(row.get(field))
        if math.isfinite(value) and value > 0:
            return value / 100.0 if value > 2.0 else value
    width = math.nan
    for field in ["spread_width", "entry_width", "preferred_width"]:
        width = _number(row.get(field))
        if math.isfinite(width) and width > 0:
            break
    if not math.isfinite(width) or width <= 0:
        return math.nan
    for field in price_fields:
        price = _number(row.get(field))
        if math.isfinite(price) and price > 0:
            return price / width
    return math.nan


def _cost_bucket(row: pd.Series | dict[str, Any]) -> str:
    value = _entry_pct_width(row)
    if not math.isfinite(value) or value <= 0:
        return "unknown"
    if value < 0.18:
        return "lt18"
    if value < 0.30:
        return "18to30"
    if value < 0.45:
        return "30to45"
    return "gt45"


def _route_key(row: pd.Series | dict[str, Any], level: str) -> str:
    base = _group_key(row)
    if level == "base":
        return f"base::{base}"
    flow = _flow_bucket(row)
    if level == "flow":
        return f"flow::{base}|flow={flow}"
    if level == "flow_cost":
        return f"flow_cost::{base}|flow={flow}|cost={_cost_bucket(row)}"
    raise ValueError(f"unknown payoff route level: {level}")


def _profit_factor_at_least(value: Any, threshold: float) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return not math.isnan(number) and number >= threshold


def _eligible_history(history: pd.DataFrame, *, asof: object | None) -> pd.DataFrame:
    if history is None or history.empty:
        return pd.DataFrame()
    out = history.copy()
    for column in ["exact_evaluated", "replay_guard_pass"]:
        if column in out.columns:
            out = out[out[column].map(_truthy)]
    out["_asof_dt"] = pd.to_datetime(out.get("asof"), errors="coerce")
    out["_exit_dt"] = pd.to_datetime(out.get("exit_day"), errors="coerce")
    out["_pnl"] = pd.to_numeric(out.get("pnl_1x"), errors="coerce")
    out["_entry_price"] = pd.to_numeric(out.get("entry_price"), errors="coerce")
    out["_entry_width"] = pd.to_numeric(out.get("entry_width"), errors="coerce")
    out["_family"] = out.apply(_family, axis=1)
    out["_group_key"] = out.apply(_group_key, axis=1)
    out = out[
        out["_asof_dt"].notna()
        & out["_exit_dt"].notna()
        & out["_pnl"].notna()
        & out["_entry_price"].gt(0)
        & out["_entry_width"].gt(0)
        & out["_family"].isin({"Credit", "Debit"})
    ]
    cutoff = pd.to_datetime(asof, errors="coerce")
    if pd.notna(cutoff):
        out = out[(out["_asof_dt"] <= cutoff) & (out["_exit_dt"] <= cutoff)]
    debit = out["_family"].eq("Debit")
    out["_risk"] = out["_entry_price"] * 100.0
    out.loc[~debit, "_risk"] = (out.loc[~debit, "_entry_width"] - out.loc[~debit, "_entry_price"]) * 100.0
    out = out[out["_risk"].gt(0)].copy()
    out["_entry_pct_width"] = out["_entry_price"] / out["_entry_width"]
    out["_flow_bucket"] = out.apply(_flow_bucket, axis=1)
    out["_cost_bucket"] = out.apply(_cost_bucket, axis=1)
    for level, policy in ROUTE_POLICIES.items():
        out[policy["key_column"]] = out.apply(lambda row: _route_key(row, level), axis=1)
    return out.sort_values(["_asof_dt", "_exit_dt"]).reset_index(drop=True)


def _metrics(frame: pd.DataFrame, stress: float) -> dict[str, Any]:
    if frame is None or frame.empty:
        return {
            "sample_size": 0,
            "wins": 0,
            "win_rate": math.nan,
            "win_rate_lower_bound": math.nan,
            "average_pnl": math.nan,
            "average_win": math.nan,
            "average_loss": math.nan,
            "average_win_risk_fraction": math.nan,
            "average_loss_risk_fraction": math.nan,
            "profit_factor": math.nan,
            "max_drawdown": math.nan,
            "entry_pct_width_p25": math.nan,
            "entry_pct_width_p75": math.nan,
        }
    pnl = frame["_pnl"].astype(float) - float(stress) * frame["_entry_price"].astype(float) * 100.0
    risk = frame["_risk"].astype(float)
    returns = pnl / risk
    wins = pnl > 0
    losses = pnl < 0
    gross_wins = float(pnl[wins].sum())
    gross_losses = float(abs(pnl[losses].sum()))
    cumulative = pnl.cumsum()
    drawdown = cumulative.cummax().sub(cumulative)
    sample = int(len(frame))
    win_count = int(wins.sum())
    return {
        "sample_size": sample,
        "wins": win_count,
        "win_rate": win_count / sample if sample else math.nan,
        "win_rate_lower_bound": wilson_lower_bound(win_count, sample),
        "average_pnl": float(pnl.mean()),
        "average_win": float(pnl[wins].mean()) if wins.any() else math.nan,
        "average_loss": float(abs(pnl[losses].mean())) if losses.any() else math.nan,
        "average_win_risk_fraction": float(returns[wins].mean()) if wins.any() else math.nan,
        "average_loss_risk_fraction": float(abs(returns[losses].mean())) if losses.any() else math.nan,
        "profit_factor": gross_wins / gross_losses if gross_losses > 0 else math.inf,
        "max_drawdown": float(drawdown.max()) if not drawdown.empty else 0.0,
        "entry_pct_width_p25": float(frame["_entry_pct_width"].quantile(0.25)),
        "entry_pct_width_p75": float(frame["_entry_pct_width"].quantile(0.75)),
    }


def _month_windows(eligible: pd.DataFrame, cutoff: pd.Timestamp) -> list[dict[str, Any]]:
    if eligible.empty:
        return []
    first = eligible["_asof_dt"].min().to_period("M")
    last = cutoff.to_period("M")
    months = list(pd.period_range(first, last, freq="M"))
    windows: list[dict[str, Any]] = []
    for index in range(2, len(months)):
        train_months = set(months[:index])
        test_month = months[index]
        train = eligible[eligible["_asof_dt"].dt.to_period("M").isin(train_months)]
        test = eligible[eligible["_asof_dt"].dt.to_period("M").eq(test_month)]
        windows.append(
            {
                "train_start": str(min(train_months).start_time.date()),
                "train_end": str(max(train_months).end_time.date()),
                "test_start": str(test_month.start_time.date()),
                "test_end": str(min(test_month.end_time, cutoff).date()),
                "train": train,
                "test": test,
            }
        )
    return windows


def build_default_payoff_calibration(
    *,
    asof: object | None,
    history_path: Path = DEFAULT_EDGE_HISTORY_PATH,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    path = Path(history_path)
    history = pd.read_csv(path, compression="infer", low_memory=False) if path.exists() else pd.DataFrame()
    cutoff = pd.to_datetime(asof, errors="coerce")
    if pd.isna(cutoff):
        cutoff = pd.Timestamp(dt.date.today())
    eligible = _eligible_history(history, asof=cutoff)
    walk_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []

    for route_level, policy in ROUTE_POLICIES.items():
        key_column = str(policy["key_column"])
        oos_frames: dict[str, list[pd.DataFrame]] = {}
        active_windows: dict[str, int] = {}
        tested_windows: dict[str, int] = {}
        failed_windows: dict[str, int] = {}
        prospective_frames: dict[str, list[pd.DataFrame]] = {}
        prospective_tested_windows: dict[str, int] = {}
        prospective_failed_windows: dict[str, int] = {}

        for window in _month_windows(eligible, cutoff):
            train = window["train"]
            test = window["test"]
            for key, train_group in train.groupby(key_column, dropna=False):
                train_metrics = _metrics(train_group, 0.10)
                train_base_metrics = _metrics(train_group, 0.0)
                train_eligible = bool(
                    train_metrics["sample_size"] >= int(policy["minimum_train_sample"])
                    and train_metrics["average_pnl"] > 0
                    and _profit_factor_at_least(
                        train_metrics["profit_factor"], float(policy["minimum_train_profit_factor"])
                    )
                )
                test_group = test[test[key_column].eq(key)]
                test_metrics = _metrics(test_group, 0.10)
                test_pass = bool(
                    test_metrics["sample_size"] > 0
                    and test_metrics["average_pnl"] > 0
                    and _profit_factor_at_least(test_metrics["profit_factor"], MIN_TEST_WINDOW_PROFIT_FACTOR)
                )
                prior_oos = (
                    pd.concat(oos_frames.get(key, []), ignore_index=True)
                    if oos_frames.get(key)
                    else pd.DataFrame()
                )
                prior_oos_metrics = _metrics(prior_oos, 0.10)
                prior_tested_windows = tested_windows.get(key, 0)
                prior_failed_windows = failed_windows.get(key, 0)
                prior_allowed_failed_windows = int(
                    math.floor(prior_tested_windows * MAX_FAILED_WALK_FORWARD_WINDOW_RATE)
                )
                prior_drawdown_ok = bool(
                    math.isfinite(train_base_metrics["max_drawdown"])
                    and math.isfinite(train_metrics["max_drawdown"])
                    and (
                        train_base_metrics["max_drawdown"] <= 0
                        or train_metrics["max_drawdown"]
                        <= train_base_metrics["max_drawdown"] * MAX_DRAWDOWN_STRESS_MULTIPLE
                    )
                )
                release_selected = bool(
                    train_base_metrics["sample_size"] >= int(policy["minimum_group_sample"])
                    and train_metrics["average_pnl"] > 0
                    and _profit_factor_at_least(train_metrics["profit_factor"], MIN_STRESS_PROFIT_FACTOR)
                    and prior_drawdown_ok
                    and prior_oos_metrics["sample_size"] >= int(policy["minimum_oos_sample"])
                    and prior_oos_metrics["average_pnl"] > 0
                    and _profit_factor_at_least(
                        prior_oos_metrics["profit_factor"], MIN_STRESS_PROFIT_FACTOR
                    )
                    and prior_failed_windows <= prior_allowed_failed_windows
                )
                if release_selected and not test_group.empty:
                    prospective_tested_windows[key] = prospective_tested_windows.get(key, 0) + 1
                    prospective_frames.setdefault(key, []).append(test_group)
                    if test_metrics["sample_size"] >= 2 and not test_pass:
                        prospective_failed_windows[key] = prospective_failed_windows.get(key, 0) + 1
                if train_eligible:
                    active_windows[key] = active_windows.get(key, 0) + 1
                    if not test_group.empty:
                        tested_windows[key] = tested_windows.get(key, 0) + 1
                        oos_frames.setdefault(key, []).append(test_group)
                        if test_metrics["sample_size"] >= 2 and not test_pass:
                            failed_windows[key] = failed_windows.get(key, 0) + 1
                walk_rows.append(
                    {
                        "route_level": route_level,
                        "group_key": key,
                        "walk_forward_scheme": "expanding_train_monthly_test",
                        "train_start": window["train_start"],
                        "train_end": window["train_end"],
                        "test_start": window["test_start"],
                        "test_end": window["test_end"],
                        "train_selected": train_eligible,
                        "release_selected": release_selected,
                        "train_sample": train_metrics["sample_size"],
                        "train_stress_10_average_pnl": train_metrics["average_pnl"],
                        "train_stress_10_profit_factor": train_metrics["profit_factor"],
                        "test_sample": test_metrics["sample_size"],
                        "test_stress_10_average_pnl": test_metrics["average_pnl"],
                        "test_stress_10_profit_factor": test_metrics["profit_factor"],
                        "test_pass": test_pass if test_metrics["sample_size"] else False,
                    }
                )

        for key, group in eligible.groupby(key_column, dropna=False):
            base = _metrics(group, 0.0)
            stress_5 = _metrics(group, 0.05)
            stress_10 = _metrics(group, 0.10)
            oos = pd.concat(oos_frames.get(key, []), ignore_index=True) if oos_frames.get(key) else pd.DataFrame()
            oos_metrics = _metrics(oos, 0.10)
            prospective = (
                pd.concat(prospective_frames.get(key, []), ignore_index=True)
                if prospective_frames.get(key)
                else pd.DataFrame()
            )
            prospective_metrics = _metrics(prospective, 0.10)
            drawdown_ok = bool(
                math.isfinite(base["max_drawdown"])
                and math.isfinite(stress_10["max_drawdown"])
                and (
                    base["max_drawdown"] <= 0
                    or stress_10["max_drawdown"] <= base["max_drawdown"] * MAX_DRAWDOWN_STRESS_MULTIPLE
                )
            )
            minimum_group_sample = int(policy["minimum_group_sample"])
            minimum_oos_sample = int(policy["minimum_oos_sample"])
            tested_window_count = tested_windows.get(key, 0)
            failed_window_count = failed_windows.get(key, 0)
            allowed_failed_windows = int(math.floor(tested_window_count * MAX_FAILED_WALK_FORWARD_WINDOW_RATE))
            route_pass = bool(
                base["sample_size"] >= minimum_group_sample
                and stress_10["average_pnl"] > 0
                and _profit_factor_at_least(stress_10["profit_factor"], MIN_STRESS_PROFIT_FACTOR)
                and drawdown_ok
                and oos_metrics["sample_size"] >= minimum_oos_sample
                and oos_metrics["average_pnl"] > 0
                and _profit_factor_at_least(oos_metrics["profit_factor"], MIN_STRESS_PROFIT_FACTOR)
                and failed_window_count <= allowed_failed_windows
                and prospective_metrics["sample_size"] >= MIN_PROSPECTIVE_OOS_SAMPLE
                and prospective_metrics["average_pnl"] > 0
                and _profit_factor_at_least(
                    prospective_metrics["profit_factor"], MIN_STRESS_PROFIT_FACTOR
                )
                and prospective_failed_windows.get(key, 0) == 0
            )
            substantive_negative_evidence = bool(
                base["sample_size"] >= int(policy["minimum_train_sample"])
                and (
                    not math.isfinite(stress_10["average_pnl"])
                    or stress_10["average_pnl"] <= 0
                    or not _profit_factor_at_least(stress_10["profit_factor"], MIN_TEST_WINDOW_PROFIT_FACTOR)
                    or not drawdown_ok
                    or (
                        oos_metrics["sample_size"] >= minimum_oos_sample
                        and (
                            not math.isfinite(oos_metrics["average_pnl"])
                            or oos_metrics["average_pnl"] <= 0
                            or not _profit_factor_at_least(
                                oos_metrics["profit_factor"], MIN_TEST_WINDOW_PROFIT_FACTOR
                            )
                        )
                    )
                    or failed_window_count > allowed_failed_windows
                    or (
                        prospective_metrics["sample_size"] >= MIN_PROSPECTIVE_OOS_SAMPLE
                        and (
                            not math.isfinite(prospective_metrics["average_pnl"])
                            or prospective_metrics["average_pnl"] <= 0
                            or not _profit_factor_at_least(
                                prospective_metrics["profit_factor"], MIN_TEST_WINDOW_PROFIT_FACTOR
                            )
                            or prospective_failed_windows.get(key, 0) > 0
                        )
                    )
                )
            )
            status = "PASS" if route_pass else ("VETO" if substantive_negative_evidence else "INSUFFICIENT")
            reasons = []
            if base["sample_size"] < minimum_group_sample:
                reasons.append(f"sample {base['sample_size']} < {minimum_group_sample}")
            if not _profit_factor_at_least(stress_10["profit_factor"], MIN_STRESS_PROFIT_FACTOR):
                reasons.append(f"10% fill-stress PF {stress_10['profit_factor']:.2f} < {MIN_STRESS_PROFIT_FACTOR:.2f}")
            if not math.isfinite(stress_10["average_pnl"]) or stress_10["average_pnl"] <= 0:
                reasons.append("10% fill-stress average P&L is not positive")
            if not drawdown_ok:
                reasons.append("10% fill-stress drawdown worsens by more than 25%")
            if oos_metrics["sample_size"] < minimum_oos_sample:
                reasons.append(f"walk-forward OOS sample {oos_metrics['sample_size']} < {minimum_oos_sample}")
            if oos_metrics["sample_size"] and not _profit_factor_at_least(
                oos_metrics["profit_factor"], MIN_STRESS_PROFIT_FACTOR
            ):
                reasons.append(f"walk-forward OOS PF {oos_metrics['profit_factor']:.2f} < {MIN_STRESS_PROFIT_FACTOR:.2f}")
            if oos_metrics["sample_size"] and (
                not math.isfinite(oos_metrics["average_pnl"]) or oos_metrics["average_pnl"] <= 0
            ):
                reasons.append("walk-forward OOS average P&L is not positive")
            if failed_window_count > allowed_failed_windows:
                reasons.append(
                    f"failed walk-forward windows={failed_window_count} > allowed {allowed_failed_windows}"
                )
            if prospective_metrics["sample_size"] < MIN_PROSPECTIVE_OOS_SAMPLE:
                reasons.append(
                    f"post-activation OOS sample {prospective_metrics['sample_size']} < {MIN_PROSPECTIVE_OOS_SAMPLE}"
                )
            if prospective_metrics["sample_size"] and not _profit_factor_at_least(
                prospective_metrics["profit_factor"], MIN_STRESS_PROFIT_FACTOR
            ):
                reasons.append(
                    f"post-activation OOS PF {prospective_metrics['profit_factor']:.2f} < "
                    f"{MIN_STRESS_PROFIT_FACTOR:.2f}"
                )
            if prospective_metrics["sample_size"] and (
                not math.isfinite(prospective_metrics["average_pnl"])
                or prospective_metrics["average_pnl"] <= 0
            ):
                reasons.append("post-activation OOS average P&L is not positive")
            if prospective_failed_windows.get(key, 0):
                reasons.append(
                    f"failed post-activation windows={prospective_failed_windows[key]}"
                )
            exemplar = group.iloc[0]
            group_rows.append(
                {
                    "route_level": route_level,
                    "route_specificity": ROUTE_SELECTION_ORDER[::-1].index(route_level),
                    "group_key": key,
                    "strategy_family": exemplar["_family"],
                    "direction": str(exemplar.get("direction", "")),
                    "regime": str(exemplar.get("regime_trend") or exemplar.get("regime") or "").lower(),
                    "flow_bucket": exemplar["_flow_bucket"] if route_level != "base" else "all",
                    "cost_bucket": exemplar["_cost_bucket"] if route_level == "flow_cost" else "all",
                    "minimum_group_sample": minimum_group_sample,
                    "payoff_calibration_status": status,
                    "payoff_calibration_reason": (
                        f"validated {route_level} route under realized payoff, expanding walk-forward, and fill stress"
                        if status == "PASS"
                        else (
                            "route-specific negative evidence; " + "; ".join(reasons)
                            if status == "VETO"
                            else "insufficient route-specific validation; " + "; ".join(reasons)
                        )
                    ),
                    "sample_size": base["sample_size"],
                    "base_win_rate": base["win_rate"],
                    "base_win_rate_lower_bound": base["win_rate_lower_bound"],
                    "base_average_pnl": base["average_pnl"],
                    "base_profit_factor": base["profit_factor"],
                    "base_max_drawdown": base["max_drawdown"],
                    "stress_5_average_pnl": stress_5["average_pnl"],
                    "stress_5_profit_factor": stress_5["profit_factor"],
                    "stress_5_max_drawdown": stress_5["max_drawdown"],
                    "stress_10_win_rate": stress_10["win_rate"],
                    "stress_10_win_rate_lower_bound": stress_10["win_rate_lower_bound"],
                    "stress_10_average_pnl": stress_10["average_pnl"],
                    "stress_10_average_win_risk_fraction": stress_10["average_win_risk_fraction"],
                    "stress_10_average_loss_risk_fraction": stress_10["average_loss_risk_fraction"],
                    "stress_10_profit_factor": stress_10["profit_factor"],
                    "stress_10_max_drawdown": stress_10["max_drawdown"],
                    "entry_pct_width_p25": base["entry_pct_width_p25"],
                    "entry_pct_width_p75": base["entry_pct_width_p75"],
                    "walk_forward_active_windows": active_windows.get(key, 0),
                    "walk_forward_tested_windows": tested_window_count,
                    "walk_forward_failed_windows": failed_window_count,
                    "walk_forward_allowed_failed_windows": allowed_failed_windows,
                    "walk_forward_oos_sample": oos_metrics["sample_size"],
                    "walk_forward_oos_average_pnl": oos_metrics["average_pnl"],
                    "walk_forward_oos_profit_factor": oos_metrics["profit_factor"],
                    "walk_forward_oos_max_drawdown": oos_metrics["max_drawdown"],
                    "post_activation_tested_windows": prospective_tested_windows.get(key, 0),
                    "post_activation_failed_windows": prospective_failed_windows.get(key, 0),
                    "post_activation_oos_sample": prospective_metrics["sample_size"],
                    "post_activation_oos_average_pnl": prospective_metrics["average_pnl"],
                    "post_activation_oos_profit_factor": prospective_metrics["profit_factor"],
                    "post_activation_oos_max_drawdown": prospective_metrics["max_drawdown"],
                }
            )

    groups = pd.DataFrame(group_rows)
    walk_forward = pd.DataFrame(walk_rows)
    family_metrics: dict[str, Any] = {}
    for family, frame in eligible.groupby("_family", dropna=False):
        family_metrics[str(family)] = {
            "base": _metrics(frame, 0.0),
            "stress_5": _metrics(frame, 0.05),
            "stress_10": _metrics(frame, 0.10),
        }
    passed = groups[groups["payoff_calibration_status"].eq("PASS")] if not groups.empty else groups
    summary = {
        "version": PAYOFF_CALIBRATION_VERSION,
        "asof": str(cutoff.date()),
        "history_path": str(path),
        "eligible_rows": int(len(eligible)),
        "thresholds": {
            "route_policies": ROUTE_POLICIES,
            "minimum_10pct_stress_profit_factor": MIN_STRESS_PROFIT_FACTOR,
            "maximum_drawdown_stress_multiple": MAX_DRAWDOWN_STRESS_MULTIPLE,
            "maximum_failed_walk_forward_window_rate": MAX_FAILED_WALK_FORWARD_WINDOW_RATE,
            "minimum_post_activation_oos_sample": MIN_PROSPECTIVE_OOS_SAMPLE,
        },
        "walk_forward_scheme": "expanding_train_monthly_test_after_two-month_warmup",
        "family_metrics": family_metrics,
        "passed_lanes": passed["group_key"].tolist() if not passed.empty else [],
        "passed_lane_count": int(len(passed)),
        "status": "PASS" if not passed.empty else "NO_VALIDATED_LANES",
    }
    return summary, groups, walk_forward


def apply_payoff_calibration(scored: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    if scored is None or scored.empty:
        return scored.copy() if scored is not None else pd.DataFrame()
    out = scored.copy()
    columns = {
        "payoff_calibration_version": PAYOFF_CALIBRATION_VERSION,
        "payoff_group_key": "",
        "payoff_route_level": "",
        "payoff_route_key": "",
        "payoff_flow_bucket": "",
        "payoff_cost_bucket": "",
        "payoff_minimum_sample_required": MIN_GROUP_SAMPLE,
        "payoff_calibration_status": "FAIL",
        "payoff_calibration_reason": "no empirical direction/regime payoff lane",
        "payoff_sample_size": 0,
        "payoff_stress_10_win_rate": math.nan,
        "payoff_stress_10_win_rate_lower_bound": math.nan,
        "payoff_stress_10_average_pnl": math.nan,
        "payoff_stress_10_average_win_risk_fraction": math.nan,
        "payoff_stress_10_average_loss_risk_fraction": math.nan,
        "payoff_stress_10_profit_factor": math.nan,
        "payoff_stress_10_max_drawdown": math.nan,
        "payoff_base_max_drawdown": math.nan,
        "payoff_entry_pct_width_p25": math.nan,
        "payoff_entry_pct_width_p75": math.nan,
        "payoff_walk_forward_oos_sample": 0,
        "payoff_walk_forward_oos_average_pnl": math.nan,
        "payoff_walk_forward_oos_profit_factor": math.nan,
        "payoff_post_activation_oos_sample": 0,
        "payoff_post_activation_oos_average_pnl": math.nan,
        "payoff_post_activation_oos_profit_factor": math.nan,
    }
    for column, default in columns.items():
        out[column] = default
    lookup = groups.set_index("group_key", drop=False) if groups is not None and not groups.empty else pd.DataFrame()
    for index, row in out.iterrows():
        candidates = [(level, _route_key(row, level)) for level in ROUTE_SELECTION_ORDER]
        out.at[index, "payoff_flow_bucket"] = _flow_bucket(row)
        out.at[index, "payoff_cost_bucket"] = _cost_bucket(row)
        selected: tuple[str, str, pd.Series] | None = None
        fallback: tuple[str, str, pd.Series] | None = None
        if not lookup.empty:
            for level, key in candidates:
                if key not in lookup.index:
                    continue
                evidence = lookup.loc[key]
                if isinstance(evidence, pd.DataFrame):
                    evidence = evidence.iloc[0]
                if fallback is None:
                    fallback = (level, key, evidence)
                evidence_status = str(evidence.get("payoff_calibration_status", "")).upper()
                if evidence_status == "PASS":
                    selected = (level, key, evidence)
                    break
                if evidence_status == "VETO":
                    selected = (level, key, evidence)
                    break
        selected = selected or fallback
        if selected is None:
            out.at[index, "payoff_group_key"] = candidates[-1][1]
            out.at[index, "payoff_route_key"] = candidates[-1][1]
            out.at[index, "payoff_route_level"] = "base"
            continue
        level, key, evidence = selected
        out.at[index, "payoff_group_key"] = key
        out.at[index, "payoff_route_key"] = key
        out.at[index, "payoff_route_level"] = level
        mapping = {
            "payoff_calibration_status": "payoff_calibration_status",
            "payoff_calibration_reason": "payoff_calibration_reason",
            "payoff_sample_size": "sample_size",
            "payoff_stress_10_win_rate": "stress_10_win_rate",
            "payoff_stress_10_win_rate_lower_bound": "stress_10_win_rate_lower_bound",
            "payoff_stress_10_average_pnl": "stress_10_average_pnl",
            "payoff_stress_10_average_win_risk_fraction": "stress_10_average_win_risk_fraction",
            "payoff_stress_10_average_loss_risk_fraction": "stress_10_average_loss_risk_fraction",
            "payoff_stress_10_profit_factor": "stress_10_profit_factor",
            "payoff_stress_10_max_drawdown": "stress_10_max_drawdown",
            "payoff_base_max_drawdown": "base_max_drawdown",
            "payoff_entry_pct_width_p25": "entry_pct_width_p25",
            "payoff_entry_pct_width_p75": "entry_pct_width_p75",
            "payoff_walk_forward_oos_sample": "walk_forward_oos_sample",
            "payoff_walk_forward_oos_average_pnl": "walk_forward_oos_average_pnl",
            "payoff_walk_forward_oos_profit_factor": "walk_forward_oos_profit_factor",
            "payoff_post_activation_oos_sample": "post_activation_oos_sample",
            "payoff_post_activation_oos_average_pnl": "post_activation_oos_average_pnl",
            "payoff_post_activation_oos_profit_factor": "post_activation_oos_profit_factor",
            "payoff_minimum_sample_required": "minimum_group_sample",
        }
        for target, source in mapping.items():
            out.at[index, target] = evidence.get(source)
    return out


def build_snapshot_replay_summary(
    *,
    asof: object | None,
    selected_dates: list[object] | None = None,
    history_path: Path = DEFAULT_EDGE_HISTORY_PATH,
) -> pd.DataFrame:
    """Summarize stored, point-in-time exact outcomes without live repricing.

    The edge-history rows are created from dated UW/chain snapshots and their
    completed exits.  This helper deliberately does not call Schwab or rebuild
    candidates, so an old-date validation cannot accidentally inherit today's
    option prices, portfolio, or market regime.
    """
    path = Path(history_path)
    history = pd.read_csv(path, compression="infer", low_memory=False) if path.exists() else pd.DataFrame()
    eligible = _eligible_history(history, asof=asof)
    requested = [str(value) for value in (selected_dates or [])]
    if eligible.empty:
        return pd.DataFrame(
            {
                "date": requested,
                "snapshot_rows": [0] * len(requested),
                "snapshot_status": ["NO_STORED_EXACT_OUTCOMES"] * len(requested),
            }
        )
    eligible["_snapshot_date"] = eligible["_asof_dt"].dt.date.astype(str)
    rows: list[dict[str, Any]] = []
    dates = requested or sorted(eligible["_snapshot_date"].unique().tolist())
    for day in dates:
        frame = eligible[eligible["_snapshot_date"].eq(str(day))]
        base = _metrics(frame, 0.0)
        stress_5 = _metrics(frame, 0.05)
        stress_10 = _metrics(frame, 0.10)
        rows.append(
            {
                "date": str(day),
                "snapshot_rows": base["sample_size"],
                "snapshot_wins": base["wins"],
                "snapshot_base_win_rate": base["win_rate"],
                "snapshot_base_average_pnl": base["average_pnl"],
                "snapshot_base_profit_factor": base["profit_factor"],
                "snapshot_base_max_drawdown": base["max_drawdown"],
                "snapshot_stress_5_average_pnl": stress_5["average_pnl"],
                "snapshot_stress_5_profit_factor": stress_5["profit_factor"],
                "snapshot_stress_10_average_pnl": stress_10["average_pnl"],
                "snapshot_stress_10_profit_factor": stress_10["profit_factor"],
                "snapshot_stress_10_max_drawdown": stress_10["max_drawdown"],
                "snapshot_status": "STORED_EXACT_OUTCOMES" if base["sample_size"] else "NO_STORED_EXACT_OUTCOMES",
            }
        )
    return pd.DataFrame(rows)


def write_payoff_calibration_outputs(
    *,
    out_dir: Path,
    asof: object,
    summary: dict[str, Any],
    groups: pd.DataFrame,
    walk_forward: pd.DataFrame,
) -> dict[str, str]:
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    label = str(asof)
    summary_path = directory / f"codexdaily_v4_payoff_calibration_summary_{label}.json"
    groups_path = directory / f"codexdaily_v4_payoff_calibration_groups_{label}.csv"
    walk_path = directory / f"codexdaily_v4_payoff_walk_forward_{label}.csv"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    groups.to_csv(groups_path, index=False)
    walk_forward.to_csv(walk_path, index=False)
    return {
        "payoff_calibration_summary": str(summary_path),
        "payoff_calibration_groups": str(groups_path),
        "payoff_walk_forward": str(walk_path),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build empirical V4 payoff and walk-forward calibration.")
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--history", default=str(DEFAULT_EDGE_HISTORY_PATH))
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    summary, groups, walk_forward = build_default_payoff_calibration(
        asof=args.as_of,
        history_path=Path(args.history),
    )
    paths = write_payoff_calibration_outputs(
        out_dir=Path(args.out_dir),
        asof=args.as_of,
        summary=summary,
        groups=groups,
        walk_forward=walk_forward,
    )
    print(json.dumps({"summary": summary, "artifacts": paths}, indent=2, default=str))


if __name__ == "__main__":
    main()
