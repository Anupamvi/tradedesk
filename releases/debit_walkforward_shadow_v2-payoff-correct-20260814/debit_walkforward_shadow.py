from __future__ import annotations

"""Leakage-safe debit-spread research book with no execution authority.

This module deliberately lives outside the production V4.24 decision path.
It evaluates frozen next-session debit entries without using future outcomes
for eligibility or ranking. Version 2 fixes a material research defect from
version 1: predicted EV used theoretical spread max profit even though the
replay exited at a stated profit target. Selection now uses the actual target
payoff, adverse entry-fill stress, and worst-case debit loss.
"""

import argparse
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


POLICY_VERSION = "debit-walkforward-shadow-v2-20260814"
EXIT_POLICY_VERSION = "debit-exit-target-revalue-v1-20260814"
DEBIT_STRATEGIES = {"Bull Call Debit Spread", "Bear Put Debit Spread"}
DEFAULT_PROFIT_TAKE_PCT = 1.00
DEFAULT_ENTRY_STRESS_PCT = 0.10
NUMERIC_FEATURES = [
    "entry_dte", "debit_pct_width", "entry_quote_width_pct", "reward_risk",
    "breakeven_sigma", "iv_hv_ratio", "aligned_flow", "aligned_dp",
    "ask_bid_imbalance", "source_multileg_ratio", "source_stock_multileg_ratio",
    "log_contract_volume", "log_contract_oi", "aligned_rsi", "aligned_sma5_20",
    "aligned_price_sma20", "aligned_return5", "aligned_return20", "aligned_rs5",
    "aligned_rs20", "aligned_vwap", "atr14_pct", "volume_ratio",
    "technical_confirmation_count", "flow_confirmation_count",
    "market_regime_aligned",
]
CATEGORICAL_FEATURES = [
    "strategy", "regime", "flow_quality", "source_side_bias",
    "oi_carryover_status", "sector",
]


def _truthy(value: Any) -> bool:
    return isinstance(value, bool) and value or str(value).strip().lower() in {
        "1", "true", "yes", "y"
    }


def _as_date(value: Any) -> dt.date | None:
    parsed = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(parsed) else parsed.date()


def revalue_debit_outcomes(
    replay: pd.DataFrame,
    *,
    root: Path,
    profit_take_pct: float = DEFAULT_PROFIT_TAKE_PCT,
    exit_slippage_pct: float = 0.10,
) -> pd.DataFrame:
    """Revalue fixed next-session debit entries under one exit policy.

    Discovery, exact legs, entry day, and entry debit stay fixed. Only the
    profit target changes, preventing an exit-policy experiment from changing
    the candidate universe or entry timing.
    """
    from .replay import (
        _expiry_spread_value,
        _quote_lookup,
        _spread_mid_value,
        dated_folders,
        future_close,
        load_close_history,
        load_hot_history,
    )

    out = replay[replay["strategy"].isin(DEBIT_STRATEGIES)].copy()
    for column in ("asof", "entry_day", "exit_day", "expiry"):
        out[column] = pd.to_datetime(out.get(column), errors="coerce")
    if out.empty:
        return out
    asof_dates = list(out["asof"].dropna().dt.date)
    if not asof_dates:
        raise ValueError("debit replay has no valid as-of dates")
    folders = dated_folders(root, min(asof_dates), None)
    close_history = load_close_history(folders)
    hot_history = load_hot_history(folders)
    quote_history = {day: _quote_lookup(frame) for day, frame in hot_history.items()}

    for index, row in out.iterrows():
        entry_day = _as_date(row.get("entry_day"))
        signal_day = _as_date(row.get("asof"))
        expiry = _as_date(row.get("expiry"))
        entry_debit = pd.to_numeric(row.get("entry_debit"), errors="coerce")
        width = pd.to_numeric(row.get("entry_width"), errors="coerce")
        if (
            not _truthy(row.get("exact_evaluated"))
            or entry_day is None
            or signal_day is None
            or expiry is None
            or entry_day <= signal_day
            or not math.isfinite(entry_debit)
            or entry_debit <= 0
            or not math.isfinite(width)
            or width <= 0
        ):
            out.at[index, "exact_evaluated"] = False
            out.at[index, "exact_reason"] = "invalid_frozen_next_session_entry"
            continue
        target_value = min(width, entry_debit * (1.0 + profit_take_pct))
        exit_day: dt.date | None = None
        exit_reason = ""
        exit_value = math.nan
        quote_days_seen = 0
        for day in sorted(day for day in quote_history if entry_day < day <= expiry):
            value_mid = _spread_mid_value(row, quote_history[day])
            if not math.isfinite(value_mid):
                continue
            quote_days_seen += 1
            stressed_value = value_mid * (1.0 - exit_slippage_pct)
            if stressed_value >= target_value:
                exit_day = day
                exit_reason = f"profit_target_{profit_take_pct:.0%}"
                exit_value = stressed_value
                break
        if exit_day is None:
            settlement_day, close = future_close(
                close_history, str(row.get("ticker", "")).upper(), expiry
            )
            if settlement_day is None or not math.isfinite(close):
                out.at[index, "exact_evaluated"] = False
                out.at[index, "exact_reason"] = "missing_expiry_close_after_revalue"
                continue
            exit_day = settlement_day
            exit_reason = "expiry_settlement"
            exit_value = _expiry_spread_value(row, close)
        out.at[index, "exact_evaluated"] = True
        out.at[index, "exit_day"] = pd.Timestamp(exit_day)
        out.at[index, "exit_reason"] = exit_reason
        out.at[index, "exit_value"] = exit_value
        out.at[index, "target_exit_value"] = target_value
        out.at[index, "pnl_1x"] = (exit_value - entry_debit) * 100.0
        out.at[index, "return_on_risk"] = (exit_value - entry_debit) / entry_debit
        out.at[index, "exact_win"] = exit_value > entry_debit
        out.at[index, "quote_days_seen"] = quote_days_seen
        out.at[index, "debit_profit_take_pct"] = profit_take_pct
        out.at[index, "debit_exit_policy_version"] = EXIT_POLICY_VERSION
    return out


def prepare_history(replay: pd.DataFrame, technical: pd.DataFrame) -> pd.DataFrame:
    out = replay[replay["strategy"].isin(DEBIT_STRATEGIES)].copy()
    for column in ("asof", "entry_day", "exit_day", "expiry", "next_earnings_dt"):
        out[column] = pd.to_datetime(out.get(column), errors="coerce")
    technical = technical.copy()
    technical["date"] = pd.to_datetime(technical["date"], errors="coerce")
    out = out.merge(
        technical,
        left_on=["asof", "ticker"],
        right_on=["date", "ticker"],
        how="left",
        suffixes=("", "_technical"),
    )
    numeric = set(NUMERIC_FEATURES) | {
        "stock_price_entry", "entry_debit", "long_strike_eod", "iv30d",
        "entry_width", "combined_flow_bias", "dp_directional_ratio",
        "source_contract_volume", "source_contract_oi", "source_ask_side_volume",
        "source_bid_side_volume", "exit_value", "rsi14", "sma5", "sma20",
        "close", "return5", "return20", "relative_strength5",
        "relative_strength20", "vwap20",
    }
    for column in numeric:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")

    bull = out["strategy"].eq("Bull Call Debit Spread")
    sign = np.where(bull, 1.0, -1.0)
    breakeven = np.where(
        bull,
        out["long_strike_eod"] + out["entry_debit"],
        out["long_strike_eod"] - out["entry_debit"],
    )
    implied_move = out["stock_price_entry"] * out["iv30d"] * np.sqrt(
        out["entry_dte"] / 365.0
    )
    out["breakeven_sigma"] = (
        pd.Series(breakeven, index=out.index) - out["stock_price_entry"]
    ).abs() / implied_move.replace(0, np.nan)
    out["debit_pct_width"] = out["entry_debit"] / out["entry_width"]
    out["earnings_known"] = out["next_earnings_dt"].notna()
    out["earnings_crosses"] = out["earnings_known"] & out["next_earnings_dt"].between(
        out["entry_day"], out["expiry"]
    )
    out["aligned_flow"] = out["combined_flow_bias"] * sign
    out["aligned_dp"] = out["dp_directional_ratio"] * sign
    out["aligned_rsi"] = (out["rsi14"] - 50.0) * sign
    out["aligned_sma5_20"] = (out["sma5"] / out["sma20"] - 1.0) * sign
    out["aligned_price_sma20"] = (out["close"] / out["sma20"] - 1.0) * sign
    out["aligned_return5"] = out["return5"] * sign
    out["aligned_return20"] = out["return20"] * sign
    out["aligned_rs5"] = out["relative_strength5"] * sign
    out["aligned_rs20"] = out["relative_strength20"] * sign
    out["aligned_vwap"] = (out["close"] / out["vwap20"] - 1.0) * sign
    denominator = (out["source_ask_side_volume"] + out["source_bid_side_volume"]).replace(0, np.nan)
    out["ask_bid_imbalance"] = (
        (out["source_ask_side_volume"] - out["source_bid_side_volume"])
        / denominator * sign
    )
    out["log_contract_volume"] = np.log1p(out["source_contract_volume"].clip(lower=0))
    out["log_contract_oi"] = np.log1p(out["source_contract_oi"].clip(lower=0))
    for stress_pct in (0.00, 0.05, 0.10, 0.15):
        label = int(round(stress_pct * 100))
        out[f"stress_pnl_{label}pct"] = (
            out["exit_value"] - out["entry_debit"] * (1.0 + stress_pct)
        ) * 100.0
    technical_columns = [
        "aligned_sma5_20", "aligned_price_sma20", "aligned_return5",
        "aligned_return20", "aligned_rs20", "aligned_vwap", "aligned_rsi",
    ]
    flow_columns = ["aligned_flow", "ask_bid_imbalance", "aligned_dp"]
    out["technical_confirmation_count"] = sum(
        out[column].gt(0).astype(int) for column in technical_columns
    )
    out["flow_confirmation_count"] = sum(
        out[column].gt(0).astype(int) for column in flow_columns
    )
    out["market_regime_aligned"] = (
        (out["strategy"].eq("Bull Call Debit Spread") & out["regime"].eq("uptrend"))
        | (out["strategy"].eq("Bear Put Debit Spread") & out["regime"].eq("downtrend"))
    ).astype(int)
    out["stress_win_10pct"] = (out["stress_pnl_10pct"] > 0).astype(int)
    out[NUMERIC_FEATURES] = out[NUMERIC_FEATURES].replace([np.inf, -np.inf], np.nan)
    return out


def learning_guard(frame: pd.DataFrame) -> pd.Series:
    integrity = (
        frame["exact_evaluated"].map(_truthy)
        & frame["entry_debit"].gt(0)
        & frame["entry_width"].gt(0)
        & frame["exit_value"].notna()
        & frame["asof"].notna()
        & frame["entry_day"].notna()
        & frame["entry_day"].gt(frame["asof"])
        & frame["exit_day"].ge(frame["entry_day"])
    )
    return (
        integrity & frame["earnings_known"] & ~frame["earnings_crosses"]
        & frame["entry_dte"].between(4, 75)
        & frame["debit_pct_width"].between(0.02, 0.80)
        & frame["entry_quote_width_pct"].le(0.60)
        & frame["reward_risk"].ge(0.25)
        & frame["breakeven_sigma"].le(1.50)
        & frame["iv_hv_ratio"].between(0.25, 2.50)
    )


def candidate_guard(frame: pd.DataFrame) -> pd.Series:
    return (
        learning_guard(frame)
        & frame["entry_dte"].between(14, 60)
        & frame["debit_pct_width"].between(0.05, 0.65)
        & frame["entry_quote_width_pct"].le(0.40)
        & frame["reward_risk"].ge(1.00)
        & frame["breakeven_sigma"].le(1.00)
        & frame["iv_hv_ratio"].between(0.50, 1.50)
    )


def _model() -> Pipeline:
    transformer = ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", RobustScaler()),
        ]), NUMERIC_FEATURES),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL_FEATURES),
    ], sparse_threshold=0.0)
    return Pipeline([
        ("features", transformer),
        ("model", LogisticRegression(C=0.10, max_iter=3000, solver="liblinear")),
    ])


def _predict_probabilities(model: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    transformed = np.asarray(model.named_steps["features"].transform(frame), dtype=float)
    estimator = model.named_steps["model"]
    coefficients = np.asarray(estimator.coef_[0], dtype=float)
    logits = np.einsum("ij,j->i", transformed, coefficients) + float(estimator.intercept_[0])
    logits = np.clip(logits, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-logits))


def payoff_aware_expected_value(
    frame: pd.DataFrame,
    probabilities: pd.Series | np.ndarray,
    *,
    profit_take_pct: float,
    entry_stress_pct: float,
) -> pd.DataFrame:
    """Return conservative payoff inputs and EV for the actual exit policy."""
    probability = pd.Series(probabilities, index=frame.index, dtype=float)
    debit = pd.to_numeric(frame["entry_debit"], errors="coerce")
    width = pd.to_numeric(frame["entry_width"], errors="coerce")
    stressed_cost = debit * (1.0 + entry_stress_pct) * 100.0
    target_value = pd.concat(
        [width, debit * (1.0 + profit_take_pct)], axis=1
    ).min(axis=1) * 100.0
    win_payoff = (target_value - stressed_cost).clip(lower=0.0)
    loss_payoff = stressed_cost
    expected_value = probability * win_payoff - (1.0 - probability) * loss_payoff
    return pd.DataFrame({
        "conservative_win_payoff": win_payoff,
        "conservative_loss_payoff": loss_payoff,
        "predicted_ev_payoff_correct": expected_value,
    }, index=frame.index)


def walk_forward_predictions(
    frame: pd.DataFrame,
    *,
    min_prior: int = 100,
    profit_take_pct: float = DEFAULT_PROFIT_TAKE_PCT,
    entry_stress_pct: float = DEFAULT_ENTRY_STRESS_PCT,
) -> pd.DataFrame:
    learn = frame[learning_guard(frame)].sort_values(["asof", "ticker", "strategy"])
    candidates = frame[candidate_guard(frame)]
    records: list[pd.DataFrame] = []
    for month, test in candidates.groupby(candidates["asof"].dt.to_period("M"), sort=True):
        prior = learn[learn["exit_day"] < month.start_time]
        if len(prior) < min_prior or prior["stress_win_10pct"].nunique() < 2:
            continue
        model = _model()
        model.fit(prior[NUMERIC_FEATURES + CATEGORICAL_FEATURES], prior["stress_win_10pct"])
        scored = test.copy()
        scored["predicted_win_probability"] = _predict_probabilities(
            model, scored[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        )
        payoff = payoff_aware_expected_value(
            scored,
            scored["predicted_win_probability"],
            profit_take_pct=profit_take_pct,
            entry_stress_pct=entry_stress_pct,
        )
        for column in payoff.columns:
            scored[column] = payoff[column]
        scored["predicted_ev_1x"] = scored["predicted_ev_payoff_correct"]
        scored["prior_sample_size"] = len(prior)
        records.append(scored)
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def _wilson_lower_bound(wins: int, total: int, z: float = 1.96) -> float | None:
    if total <= 0:
        return None
    proportion = wins / total
    denominator = 1.0 + z * z / total
    centre = proportion + z * z / (2.0 * total)
    margin = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    )
    return (centre - margin) / denominator


def metrics(frame: pd.DataFrame, *, pnl_column: str = "stress_pnl_10pct") -> dict[str, Any]:
    empty = {
        "n": 0, "wins": 0, "win_rate": None, "wilson_lower_95": None,
        "profit_factor": None, "pnl": 0.0, "max_drawdown": 0.0,
        "positive_month_ratio": None,
    }
    if frame.empty:
        return empty
    pnl = pd.to_numeric(frame[pnl_column], errors="coerce").dropna()
    if pnl.empty:
        return empty
    gross_loss = -float(pnl[pnl < 0].sum())
    equity = pnl.cumsum()
    monthly = frame.loc[pnl.index].assign(
        month=frame.loc[pnl.index, "asof"].dt.to_period("M").astype(str)
    ).groupby("month")[pnl_column].sum()
    wins = int((pnl > 0).sum())
    return {
        "n": int(len(pnl)),
        "wins": wins,
        "win_rate": float((pnl > 0).mean()),
        "wilson_lower_95": _wilson_lower_bound(wins, len(pnl)),
        "profit_factor": float(pnl[pnl > 0].sum() / gross_loss) if gross_loss else None,
        "pnl": float(pnl.sum()),
        "max_drawdown": float((equity - equity.cummax()).min()),
        "positive_month_ratio": float((monthly > 0).mean()),
    }


def _profit_factor_pass(result: dict[str, Any], minimum: float) -> bool:
    """Treat an all-winning sample as infinite PF without serializing Infinity."""
    if result.get("n", 0) <= 0:
        return False
    if result.get("wins", 0) == result.get("n", 0):
        return True
    profit_factor = result.get("profit_factor")
    return profit_factor is not None and profit_factor >= minimum


def breakdown_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"by_month": {}, "by_strategy": {}, "by_regime": {}, "by_entry_timing": {}}
    result: dict[str, Any] = {}
    for output, column in (
        ("by_month", "asof"),
        ("by_strategy", "strategy"),
        ("by_regime", "regime"),
        ("by_entry_timing", "entry_timing"),
    ):
        if column not in frame.columns:
            result[output] = {}
            continue
        key = frame[column].dt.to_period("M").astype(str) if column == "asof" else frame[column]
        result[output] = {
            str(value): metrics(group)
            for value, group in frame.groupby(key, sort=True, dropna=False)
        }
    return result


def calibration_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty or "predicted_win_probability" not in frame.columns:
        return {"n": 0, "brier_score": None, "bins": [], "monotonic_violations": None}
    probability = pd.to_numeric(frame["predicted_win_probability"], errors="coerce")
    actual = pd.to_numeric(frame["stress_win_10pct"], errors="coerce")
    valid = probability.notna() & actual.notna()
    if not valid.any():
        return {"n": 0, "brier_score": None, "bins": [], "monotonic_violations": None}
    work = pd.DataFrame({"probability": probability[valid], "actual": actual[valid]})
    work["bucket"] = pd.cut(
        work["probability"], [0.0, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 1.000001],
        right=False,
    )
    bins = []
    for bucket, group in work.groupby("bucket", observed=True, sort=True):
        bins.append({
            "bucket": str(bucket),
            "n": int(len(group)),
            "mean_prediction": float(group["probability"].mean()),
            "observed_win_rate": float(group["actual"].mean()),
        })
    stable = [item for item in bins if item["n"] >= 5]
    violations = sum(
        right["observed_win_rate"] + 0.05 < left["observed_win_rate"]
        for left, right in zip(stable, stable[1:])
    )
    return {
        "n": int(len(work)),
        "brier_score": float(((work["probability"] - work["actual"]) ** 2).mean()),
        "bins": bins,
        "monotonic_violations": int(violations),
    }


def active_overlap_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    required = {"entry_day", "exit_day", "entry_debit"}
    empty = {
        "peak_active_positions": 0, "peak_defined_risk": 0.0,
        "peak_sector_share": None, "peak_ticker_share": None,
    }
    if frame.empty or not required.issubset(frame.columns):
        return empty
    work = frame.copy()
    work["entry_day"] = pd.to_datetime(work["entry_day"], errors="coerce")
    work["exit_day"] = pd.to_datetime(work["exit_day"], errors="coerce")
    work["defined_risk"] = pd.to_numeric(work["entry_debit"], errors="coerce") * 110.0
    work = work.dropna(subset=["entry_day", "exit_day", "defined_risk"])
    if work.empty:
        return empty
    peak_positions = 0
    peak_risk = 0.0
    peak_sector_share = 0.0
    peak_ticker_share = 0.0
    for day in sorted(set(work["entry_day"]).union(set(work["exit_day"]))):
        active = work[work["entry_day"].le(day) & work["exit_day"].ge(day)]
        total = float(active["defined_risk"].sum())
        peak_positions = max(peak_positions, len(active))
        if total > peak_risk:
            peak_risk = total
            if "sector" in active.columns:
                peak_sector_share = float(
                    active.groupby("sector", dropna=False)["defined_risk"].sum().max() / total
                )
            if "ticker" in active.columns:
                peak_ticker_share = float(
                    active.groupby("ticker")["defined_risk"].sum().max() / total
                )
    return {
        "peak_active_positions": int(peak_positions),
        "peak_defined_risk": peak_risk,
        "peak_sector_share": peak_sector_share,
        "peak_ticker_share": peak_ticker_share,
    }


def select_book(predictions: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if predictions.empty:
        return predictions.copy()
    ev_column = (
        "predicted_ev_payoff_correct"
        if "predicted_ev_payoff_correct" in predictions.columns
        else "predicted_ev_1x"
    )
    eligible = predictions[
        predictions["predicted_win_probability"].ge(threshold)
        & predictions[ev_column].gt(0)
    ].copy()
    return eligible.sort_values(
        ["asof", "predicted_win_probability", ev_column, "breakeven_sigma"],
        ascending=[True, False, False, True],
    ).groupby("asof", as_index=False).head(1)


def evaluate(
    frame: pd.DataFrame,
    *,
    cutoff: object,
    threshold: float = 0.55,
    profit_take_pct: float = DEFAULT_PROFIT_TAKE_PCT,
    entry_stress_pct: float = DEFAULT_ENTRY_STRESS_PCT,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cutoff_dt = pd.Timestamp(cutoff)
    development = select_book(frame[frame["asof"] < cutoff_dt], threshold)
    holdout = select_book(frame[frame["asof"] >= cutoff_dt], threshold)
    development_metrics = metrics(development)
    holdout_metrics = metrics(holdout)
    blockers: list[str] = []
    if development_metrics["n"] < 20:
        blockers.append("development_sample_below_20")
    if holdout_metrics["n"] < 15:
        blockers.append("holdout_sample_below_15")
    if development_metrics["n"] + holdout_metrics["n"] < 50:
        blockers.append("total_selected_sample_below_50")
    for name, result in (("development", development_metrics), ("holdout", holdout_metrics)):
        if not _profit_factor_pass(result, 1.50):
            blockers.append(f"{name}_stress_pf_below_1.50")
        if result["positive_month_ratio"] is None or result["positive_month_ratio"] < 0.67:
            blockers.append(f"{name}_positive_month_ratio_below_0.67")
        if result["max_drawdown"] < -603.50:
            blockers.append(f"{name}_drawdown_worse_than_v4.21")
        if result["wilson_lower_95"] is None or result["wilson_lower_95"] < 0.55:
            blockers.append(f"{name}_wilson_lower_below_0.55")
    for strategy in sorted(DEBIT_STRATEGIES):
        family = holdout[holdout["strategy"].eq(strategy)] if "strategy" in holdout else holdout.iloc[0:0]
        family_metrics = metrics(family)
        key = strategy.lower().replace(" ", "_")
        if family_metrics["n"] < 10:
            blockers.append(f"holdout_{key}_sample_below_10")
        if family_metrics["n"] and not _profit_factor_pass(family_metrics, 1.25):
            blockers.append(f"holdout_{key}_pf_below_1.25")
    combined = pd.concat([development, holdout], ignore_index=True)
    calibration = calibration_metrics(combined)
    if calibration["monotonic_violations"]:
        blockers.append("probability_calibration_non_monotonic")
    summary = {
        "policy_version": POLICY_VERSION,
        "exit_policy_version": EXIT_POLICY_VERSION,
        "status": "SHADOW_EVIDENCE_PASS" if not blockers else "RESEARCH_ONLY",
        "execution_authorized": False,
        "threshold": threshold,
        "profit_take_pct": profit_take_pct,
        "entry_stress_pct": entry_stress_pct,
        "cutoff": cutoff_dt.date().isoformat(),
        "development": development_metrics,
        "holdout": holdout_metrics,
        "development_breakdown": breakdown_metrics(development),
        "holdout_breakdown": breakdown_metrics(holdout),
        "calibration": calibration,
        "active_overlap": active_overlap_metrics(combined),
        "blockers": blockers,
        "note": (
            "Shadow evidence only; production V4.24 behavior is unchanged. "
            "A separate versioned promotion decision is required even if all evidence gates pass."
        ),
    }
    return combined, summary


def write_report(selected: pd.DataFrame, summary: dict[str, Any], path: Path) -> None:
    columns = [
        "asof", "ticker", "strategy", "entry_day", "expiry", "entry_debit",
        "entry_width", "predicted_win_probability", "predicted_ev_payoff_correct",
        "stress_pnl_10pct", "regime", "entry_timing", "oi_carryover_status",
        "technical_confirmation_count", "flow_confirmation_count",
    ]
    rows = selected[[column for column in columns if column in selected.columns]].copy()
    lines = [
        "# Debit Walk-Forward Shadow V2",
        "",
        f"- Status: **{summary['status']}**",
        "- Execution authority: **NO**",
        f"- Exit policy: {summary['profit_take_pct']:.0%} profit target; no hard stop; {summary['entry_stress_pct']:.0%} adverse entry-fill stress",
        f"- Development: n={summary['development']['n']}; PF={summary['development']['profit_factor']}; P/L=${summary['development']['pnl']:,.2f}",
        f"- Holdout: n={summary['holdout']['n']}; PF={summary['holdout']['profit_factor']}; P/L=${summary['holdout']['pnl']:,.2f}",
        f"- Blockers: {', '.join(summary['blockers']) if summary['blockers'] else 'none'}",
        "",
        "## Selected Shadow Trades",
        "",
        rows.to_markdown(index=False) if not rows.empty else "_No shadow trades selected._",
        "",
        "## Interpretation",
        "",
        "This is a research ledger, not an order list. It uses fixed next-session entries and outcome-blind expanding-window predictions. Production V4.24 remains unchanged.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", required=True, type=Path)
    parser.add_argument("--technical", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--revalue-outcomes", action="store_true")
    parser.add_argument("--cutoff", default="2026-05-01")
    parser.add_argument("--threshold", default=0.55, type=float)
    parser.add_argument("--profit-take-pct", default=DEFAULT_PROFIT_TAKE_PCT, type=float)
    parser.add_argument("--entry-stress-pct", default=DEFAULT_ENTRY_STRESS_PCT, type=float)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    replay = pd.read_csv(args.replay, low_memory=False)
    if args.revalue_outcomes:
        if args.root is None:
            parser.error("--root is required with --revalue-outcomes")
        replay = revalue_debit_outcomes(
            replay,
            root=args.root.expanduser().resolve(),
            profit_take_pct=args.profit_take_pct,
        )
        replay.to_csv(args.out_dir / "debit_revalued_history.csv.gz", index=False)
    history = prepare_history(replay, pd.read_csv(args.technical, low_memory=False))
    predictions = walk_forward_predictions(
        history,
        profit_take_pct=args.profit_take_pct,
        entry_stress_pct=args.entry_stress_pct,
    )
    selected, summary = evaluate(
        predictions,
        cutoff=args.cutoff,
        threshold=args.threshold,
        profit_take_pct=args.profit_take_pct,
        entry_stress_pct=args.entry_stress_pct,
    )
    predictions.to_csv(args.out_dir / "debit_walkforward_predictions.csv", index=False)
    selected.to_csv(args.out_dir / "debit_walkforward_selected.csv", index=False)
    (args.out_dir / "debit_walkforward_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_report(selected, summary, args.out_dir / "debit_walkforward_report.md")
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
