from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .catalysts import earnings_crosses_expiry
from .credit_policy import assess_credit_spread
from .data import safe_float
from .debit_policy import assess_debit_spread
from .replay import (
    _date_key,
    _decision_sort_score,
    _distance_expected,
    _entry_fillable,
    _is_debit_strategy,
    _truthy,
)


@dataclasses.dataclass(frozen=True)
class CoveragePolicy:
    max_per_day: int
    eligibility_mode: str
    dark_pool_weight: float
    oi_mode: str
    history_weight: float
    min_prior_sample: int
    model_weight: float

    @property
    def policy_id(self) -> str:
        return (
            f"n{self.max_per_day}_{self.eligibility_mode}_dp{self.dark_pool_weight:.2f}_oi-{self.oi_mode}_"
            f"hist{self.history_weight:.1f}_prior{self.min_prior_sample}_model{self.model_weight:.1f}"
        )


def _family(row: pd.Series | dict[str, Any]) -> str:
    return "Debit" if _is_debit_strategy(row) else "Credit"


def _cost_bucket(row: pd.Series | dict[str, Any]) -> str:
    value = safe_float(
        row.get("entry_debit_pct_width")
        if _is_debit_strategy(row)
        else row.get("entry_credit_pct_width")
    )
    if not math.isfinite(value):
        return "unknown"
    if value < 0.18:
        return "lt18"
    if value < 0.30:
        return "18to30"
    if value < 0.45:
        return "30to45"
    return "gt45"


def _flow_bucket(row: pd.Series | dict[str, Any]) -> str:
    value = str(row.get("flow_quality") or "").strip().lower()
    if value == "directional":
        return "directional"
    if value in {"", "nan", "none", "unknown", "missing"}:
        return "unknown"
    return "ambiguous"


def _oi_bucket(row: pd.Series | dict[str, Any]) -> str:
    value = str(row.get("oi_carryover_status") or "").strip().lower()
    if value == "supportive":
        return "supportive"
    if value == "contrary":
        return "contrary"
    if value in {"mixed", "matched_unconfirmed"}:
        return "mixed"
    return "unknown"


def _route_key(row: pd.Series | dict[str, Any], *, specific: bool) -> str:
    base = "|".join(
        [
            _family(row),
            str(row.get("direction") or ""),
            str(row.get("regime") or row.get("regime_trend") or "").lower(),
        ]
    )
    if not specific:
        return base
    return "|".join([base, _flow_bucket(row), _cost_bucket(row), _oi_bucket(row)])


def _effective_flow_bias(row: pd.Series | dict[str, Any], weight: float) -> float:
    option_bias = safe_float(
        row.get("option_flow_bias"),
        safe_float(row.get("combined_flow_bias"), safe_float(row.get("flow_bias"), 0.0)),
    )
    dp_bias = safe_float(row.get("dp_flow_bias"))
    dp_ratio = safe_float(row.get("dp_directional_ratio"))
    bounded = min(0.25, max(0.0, safe_float(weight, 0.0)))
    if not math.isfinite(dp_bias) or not math.isfinite(dp_ratio) or dp_ratio < 0.25:
        return option_bias
    return option_bias * (1.0 - bounded) + dp_bias * bounded


def _risk_normalized_return(row: pd.Series | dict[str, Any]) -> float:
    direct = safe_float(row.get("return_on_risk"))
    if math.isfinite(direct):
        return direct
    pnl = safe_float(row.get("pnl_1x"))
    entry = safe_float(row.get("entry_price"))
    width = safe_float(row.get("entry_width"))
    if not all(math.isfinite(value) for value in (pnl, entry, width)):
        return math.nan
    risk = entry * 100.0 if _is_debit_strategy(row) else (width - entry) * 100.0
    return pnl / risk if risk > 0 else math.nan


_MODEL_NUMERIC_FEATURES = (
    "dte",
    "entry_credit_pct_width",
    "entry_debit_pct_width",
    "reward_risk",
    "expected_move_ratio",
    "entry_quote_width_pct",
    "iv_rank",
    "iv30d",
    "combined_flow_bias",
    "flow_total_premium",
    "source_contract_volume",
    "source_contract_oi",
    "source_multileg_ratio",
    "bot_multileg_ratio",
    "bot_volume_oi_ratio",
    "dp_flow_bias",
    "dp_directional_ratio",
)
_MODEL_DIRECTIONS = ("Bull Put", "Bear Call", "Bull Call", "Bear Put")
_MODEL_REGIMES = ("uptrend", "downtrend", "range")
_MODEL_FLOW_BUCKETS = ("directional", "ambiguous", "unknown")
_MODEL_OI_BUCKETS = ("supportive", "contrary", "mixed", "unknown")


def _model_matrix(frame: pd.DataFrame) -> np.ndarray:
    if frame.empty:
        width = (
            len(_MODEL_NUMERIC_FEATURES)
            + len(_MODEL_DIRECTIONS)
            + len(_MODEL_REGIMES)
            + len(_MODEL_FLOW_BUCKETS)
            + len(_MODEL_OI_BUCKETS)
            + 3
        )
        return np.empty((0, width), dtype=float)
    columns: list[np.ndarray] = []
    for name in _MODEL_NUMERIC_FEATURES:
        values = pd.to_numeric(frame.get(name, pd.Series(math.nan, index=frame.index)), errors="coerce").to_numpy(dtype=float)
        if name in {"flow_total_premium", "source_contract_volume", "source_contract_oi"}:
            values = np.log1p(np.clip(values, 0.0, None))
        columns.append(values)
    direction = frame.get("direction", pd.Series("", index=frame.index)).astype(str)
    regime = frame.get("regime", pd.Series("", index=frame.index)).astype(str).str.lower()
    flow = frame.apply(_flow_bucket, axis=1)
    oi = frame.apply(_oi_bucket, axis=1)
    columns.extend([(direction == value).to_numpy(dtype=float) for value in _MODEL_DIRECTIONS])
    columns.extend([(regime == value).to_numpy(dtype=float) for value in _MODEL_REGIMES])
    columns.extend([(flow == value).to_numpy(dtype=float) for value in _MODEL_FLOW_BUCKETS])
    columns.extend([(oi == value).to_numpy(dtype=float) for value in _MODEL_OI_BUCKETS])
    bot_status = frame.get("bot_flow_source_status", pd.Series("", index=frame.index)).astype(str).str.lower()
    dp_status = frame.get("dark_pool_source_status", pd.Series("", index=frame.index)).astype(str).str.lower()
    oi_status = frame.get("chain_oi_source_status", pd.Series("", index=frame.index)).astype(str).str.lower()
    columns.append(bot_status.str.startswith("bot_eod").to_numpy(dtype=float))
    columns.append(dp_status.str.startswith("dp_eod").to_numpy(dtype=float))
    columns.append(oi_status.eq("chain_oi_loaded").to_numpy(dtype=float))
    return np.column_stack(columns)


def _ridge_predict_ror(train: pd.DataFrame, test: pd.DataFrame, *, penalty: float = 10.0) -> np.ndarray:
    if train.empty or test.empty:
        return np.zeros(len(test), dtype=float)
    train_x = _model_matrix(train)
    test_x = _model_matrix(test)
    train_x = np.where(np.isfinite(train_x), train_x, np.nan)
    test_x = np.where(np.isfinite(test_x), test_x, np.nan)
    medians = np.asarray(
        [np.nanmedian(column) if np.isfinite(column).any() else 0.0 for column in train_x.T],
        dtype=float,
    )
    train_x = np.where(np.isfinite(train_x), train_x, medians)
    test_x = np.where(np.isfinite(test_x), test_x, medians)
    train_x = np.clip(train_x, -1_000_000.0, 1_000_000.0)
    test_x = np.clip(test_x, -1_000_000.0, 1_000_000.0)
    lower = np.quantile(train_x, 0.01, axis=0)
    upper = np.quantile(train_x, 0.99, axis=0)
    train_x = np.clip(train_x, lower, upper)
    test_x = np.clip(test_x, lower, upper)
    means = train_x.mean(axis=0)
    scales = train_x.std(axis=0)
    scales = np.where(np.isfinite(scales) & (scales > 1e-9), scales, 1.0)
    train_x = (train_x - means) / scales
    test_x = (test_x - means) / scales
    train_x = np.clip(np.nan_to_num(train_x, nan=0.0, posinf=10.0, neginf=-10.0), -10.0, 10.0)
    test_x = np.clip(np.nan_to_num(test_x, nan=0.0, posinf=10.0, neginf=-10.0), -10.0, 10.0)
    train_x = np.column_stack([np.ones(len(train_x)), train_x])
    test_x = np.column_stack([np.ones(len(test_x)), test_x])
    target = train["_ror"].astype(float).clip(lower=-1.0, upper=3.0).to_numpy()
    day_counts = train["_asof_dt"].value_counts()
    weights = train["_asof_dt"].map(lambda value: 1.0 / day_counts[value]).to_numpy(dtype=float)
    weights *= len(weights) / weights.sum()
    root_weight = np.sqrt(weights)
    weighted_x = np.nan_to_num(train_x * root_weight[:, None], nan=0.0, posinf=10.0, neginf=-10.0)
    weighted_y = np.nan_to_num(target * root_weight, nan=0.0, posinf=3.0, neginf=-1.0)
    regularizer = np.eye(train_x.shape[1]) * penalty
    regularizer[0, 0] = 0.0
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        gram = weighted_x.T @ weighted_x + regularizer
        rhs = weighted_x.T @ weighted_y
    if not np.isfinite(gram).all() or not np.isfinite(rhs).all():
        return np.zeros(len(test), dtype=float)
    try:
        coefficients = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        return np.zeros(len(test), dtype=float)
    if not np.isfinite(coefficients).all():
        return np.zeros(len(test), dtype=float)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        predictions = test_x @ coefficients
    if not np.isfinite(predictions).all():
        return np.zeros(len(test), dtype=float)
    return np.clip(predictions, -1.0, 3.0)


def add_walk_forward_model_predictions(detail: pd.DataFrame) -> pd.DataFrame:
    out = detail.copy()
    out["_wf_ev_fillable"] = 0.0
    out["_wf_ev_guarded"] = 0.0
    for raw_day in sorted(out["_asof_dt"].dropna().unique()):
        day = pd.Timestamp(raw_day)
        test_mask = out["_asof_dt"].eq(day)
        known = out[
            out["_exact"]
            & out["_exit_dt"].notna()
            & out["_exit_dt"].lt(day)
            & out["_ror"].notna()
        ]
        if len(known) >= 80:
            out.loc[test_mask, "_wf_ev_fillable"] = _ridge_predict_ror(known, out[test_mask])
        guarded = known[known["_guard"]]
        if len(guarded) >= 30:
            out.loc[test_mask, "_wf_ev_guarded"] = _ridge_predict_ror(guarded, out[test_mask])
    return out


def _known_prior(
    history: pd.DataFrame,
    day: pd.Timestamp,
    *,
    eligibility_mode: str,
) -> pd.DataFrame:
    if history.empty:
        return history
    prior = history[
        history["_exact"]
        & history["_exit_dt"].notna()
        & history["_exit_dt"].lt(day)
        & history["_ror"].notna()
    ]
    if eligibility_mode != "fillable":
        prior = prior[prior["_guard"]]
    return prior


def _history_lower_bound(
    row: pd.Series,
    prior: pd.DataFrame,
    *,
    min_prior_sample: int,
) -> tuple[float, int, str]:
    if prior.empty:
        return 0.0, 0, "none"
    specific_key = _route_key(row, specific=True)
    base_key = _route_key(row, specific=False)
    specific = prior[prior["_route_specific"].eq(specific_key)]
    base = prior[prior["_route_base"].eq(base_key)]
    chosen = specific if len(specific) >= min_prior_sample else base
    level = "specific" if len(specific) >= min_prior_sample else "base"
    if len(chosen) < min_prior_sample:
        family = prior[prior["_family"].eq(_family(row))]
        chosen = family
        level = "family"
    if len(chosen) < min_prior_sample:
        return 0.0, int(len(chosen)), "insufficient"
    returns = chosen["_ror"].astype(float).clip(lower=-1.0, upper=3.0)
    mean = float(returns.mean())
    standard_error = float(returns.std(ddof=1) / math.sqrt(len(returns))) if len(returns) > 1 else 0.0
    return mean - 1.2815515655446004 * standard_error, int(len(returns)), level


def _lower_bound_by_group(frame: pd.DataFrame, key: str) -> dict[str, tuple[float, int]]:
    result: dict[str, tuple[float, int]] = {}
    if frame.empty:
        return result
    for value, group in frame.groupby(key, dropna=False):
        returns = group["_ror"].astype(float).clip(lower=-1.0, upper=3.0)
        mean = float(returns.mean())
        standard_error = float(returns.std(ddof=1) / math.sqrt(len(returns))) if len(returns) > 1 else 0.0
        result[str(value)] = (mean - 1.2815515655446004 * standard_error, int(len(returns)))
    return result


def add_walk_forward_history_estimates(detail: pd.DataFrame, *, min_sample: int = 8) -> pd.DataFrame:
    out = detail.copy()
    for lane in ("fillable", "guarded"):
        out[f"_history_lb_{lane}"] = 0.0
        out[f"_history_n_{lane}"] = 0
        out[f"_history_level_{lane}"] = "none"
    for raw_day in sorted(out["_asof_dt"].dropna().unique()):
        day = pd.Timestamp(raw_day)
        test_indices = out.index[out["_asof_dt"].eq(day)]
        known = out[
            out["_exact"]
            & out["_exit_dt"].notna()
            & out["_exit_dt"].lt(day)
            & out["_ror"].notna()
        ]
        for lane, prior in (("fillable", known), ("guarded", known[known["_guard"]])):
            specific = _lower_bound_by_group(prior, "_route_specific")
            base = _lower_bound_by_group(prior, "_route_base")
            family = _lower_bound_by_group(prior, "_family")
            for index in test_indices:
                row = out.loc[index]
                choices = [
                    ("specific", specific.get(str(row["_route_specific"]))),
                    ("base", base.get(str(row["_route_base"]))),
                    ("family", family.get(str(row["_family"]))),
                ]
                chosen_level = "insufficient"
                chosen = None
                for level, metrics in choices:
                    if metrics is not None and metrics[1] >= min_sample:
                        chosen_level, chosen = level, metrics
                        break
                if chosen is not None:
                    out.at[index, f"_history_lb_{lane}"] = chosen[0]
                    out.at[index, f"_history_n_{lane}"] = chosen[1]
                    out.at[index, f"_history_level_{lane}"] = chosen_level
    return out


def _eligible(row: pd.Series, policy: CoveragePolicy) -> tuple[bool, str, float]:
    if not _entry_fillable(row):
        return False, "not_entry_fillable", math.nan
    if earnings_crosses_expiry(row):
        return False, "earnings_crosses_expiry", math.nan
    if (
        policy.eligibility_mode != "fillable"
        and "replay_guard_pass" in row.index
        and not _truthy(row.get("replay_guard_pass"))
    ):
        return False, "replay_guard_failed", math.nan
    oi = _oi_bucket(row)
    if policy.oi_mode == "reject_contrary" and oi == "contrary":
        return False, "oi_contrary", math.nan
    if policy.oi_mode == "require_confirmed" and oi not in {"supportive", "mixed"}:
        return False, "oi_not_confirmed", math.nan

    candidate = row.copy()
    candidate["combined_flow_bias"] = _effective_flow_bias(row, policy.dark_pool_weight)
    _distance, _expected, ratio = _distance_expected(candidate)
    direction = str(candidate.get("direction") or "")
    sign = 1.0 if direction in {"Bull Put", "Bull Call"} else -1.0
    alignment = safe_float(candidate.get("combined_flow_bias"), 0.0) * sign
    if policy.eligibility_mode != "policy":
        return True, f"{policy.eligibility_mode}_candidate", alignment
    if _is_debit_strategy(candidate):
        ok, reasons = assess_debit_spread(
            candidate,
            live=False,
            expected_move_ratio=ratio,
            flow_alignment=alignment,
        )
    else:
        ok, reasons = assess_credit_spread(
            candidate,
            live=False,
            expected_move_ratio=ratio,
            flow_alignment=alignment,
        )
    return ok, "|".join(reasons), alignment


def prepare_detail(detail: pd.DataFrame) -> pd.DataFrame:
    out = detail.copy()
    out["_asof_dt"] = pd.to_datetime(out.get("asof"), errors="coerce")
    out["_exit_dt"] = pd.to_datetime(out.get("exit_day"), errors="coerce")
    out["_pnl"] = pd.to_numeric(out.get("pnl_1x"), errors="coerce")
    out["_exact"] = out.get("exact_evaluated", pd.Series(False, index=out.index)).map(_truthy)
    out["_guard"] = out.get("replay_guard_pass", pd.Series(False, index=out.index)).map(_truthy)
    out["_ror"] = out.apply(_risk_normalized_return, axis=1)
    out["_family"] = out.apply(_family, axis=1)
    out["_route_base"] = out.apply(lambda row: _route_key(row, specific=False), axis=1)
    out["_route_specific"] = out.apply(lambda row: _route_key(row, specific=True), axis=1)
    out = out[out["_asof_dt"].notna()].copy()
    out["_dedupe_score"] = out["_exact"].astype(int) * 2 + out["_guard"].astype(int)
    dedupe_keys = [
        column
        for column in [
            "_asof_dt",
            "ticker",
            "direction",
            "expiry",
            "short_strike_eod",
            "long_strike_eod",
        ]
        if column in out.columns
    ]
    out = out.sort_values("_dedupe_score", ascending=False).drop_duplicates(dedupe_keys, keep="first")
    out = out.sort_values(["_asof_dt", "ticker"]).drop(columns=["_dedupe_score"]).reset_index(drop=True)
    out = add_walk_forward_history_estimates(out)
    return add_walk_forward_model_predictions(out)


def select_policy_trades(
    detail: pd.DataFrame,
    policy: CoveragePolicy,
    *,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    selected: list[pd.DataFrame] = []
    days = sorted(detail["_asof_dt"].dropna().unique())
    for raw_day in days:
        day = pd.Timestamp(raw_day)
        if start is not None and day < start:
            continue
        if end is not None and day > end:
            continue
        candidates = detail[detail["_asof_dt"].eq(day)].copy()
        scored_rows: list[pd.Series] = []
        for _, row in candidates.iterrows():
            eligible, reason, alignment = _eligible(row, policy)
            if not eligible:
                continue
            candidate = row.copy()
            candidate["combined_flow_bias"] = _effective_flow_bias(row, policy.dark_pool_weight)
            base_score = _decision_sort_score(candidate)
            history_lane = "fillable" if policy.eligibility_mode == "fillable" else "guarded"
            history_lb = safe_float(candidate.get(f"_history_lb_{history_lane}"), 0.0)
            history_n = int(safe_float(candidate.get(f"_history_n_{history_lane}"), 0.0))
            history_level = str(candidate.get(f"_history_level_{history_lane}") or "none")
            oi_adjustment = 0.20 if _oi_bucket(candidate) == "supportive" else -0.20 if _oi_bucket(candidate) == "contrary" else 0.0
            candidate["goal_base_score"] = base_score
            candidate["goal_history_lower_bound"] = history_lb
            candidate["goal_history_sample"] = history_n
            candidate["goal_history_level"] = history_level
            candidate["goal_effective_flow_alignment"] = alignment
            model_column = "_wf_ev_fillable" if policy.eligibility_mode == "fillable" else "_wf_ev_guarded"
            model_value = safe_float(candidate.get(model_column), 0.0)
            candidate["goal_walk_forward_ev_ror"] = model_value
            candidate["goal_score"] = (
                base_score
                + policy.history_weight * history_lb
                + policy.model_weight * max(-1.0, min(1.0, model_value * 4.0))
                + oi_adjustment
            )
            candidate["goal_policy_id"] = policy.policy_id
            candidate["goal_eligibility_reason"] = reason
            scored_rows.append(candidate)
        if not scored_rows:
            continue
        scored = pd.DataFrame(scored_rows).sort_values(
            ["goal_score", "goal_base_score", "ticker"],
            ascending=[False, False, True],
        )
        # Avoid counting multiple same-ticker structures as independent daily coverage.
        scored = scored.drop_duplicates("ticker", keep="first").head(policy.max_per_day)
        selected.append(scored)
    return pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()


def performance(
    selected: pd.DataFrame,
    all_days: Iterable[pd.Timestamp],
) -> dict[str, Any]:
    days = [pd.Timestamp(day) for day in all_days]
    evaluated = selected[selected["_exact"] & selected["_pnl"].notna()].copy() if not selected.empty else selected
    gross_profit = float(evaluated.loc[evaluated["_pnl"].gt(0), "_pnl"].sum()) if not evaluated.empty else 0.0
    gross_loss = float(-evaluated.loc[evaluated["_pnl"].lt(0), "_pnl"].sum()) if not evaluated.empty else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else math.inf if gross_profit > 0 else 0.0
    by_day = (
        evaluated.groupby("_asof_dt")["_pnl"].agg(total="sum", has_win=lambda values: bool((values > 0).any()))
        if not evaluated.empty
        else pd.DataFrame(columns=["total", "has_win"])
    )
    by_day = by_day.reindex(days, fill_value=0)
    equity = by_day["total"].cumsum()
    drawdown = equity - equity.cummax()
    return {
        "calendar_days": int(len(days)),
        "selected_rows": int(len(selected)),
        "evaluated_rows": int(len(evaluated)),
        "evaluation_rate": float(len(evaluated) / len(selected)) if len(selected) else 0.0,
        "trade_days": int(selected["_asof_dt"].nunique()) if not selected.empty else 0,
        "winner_days": int(by_day["has_win"].sum()),
        "winner_day_rate": float(by_day["has_win"].mean()) if len(by_day) else 0.0,
        "positive_net_days": int(by_day["total"].gt(0).sum()),
        "positive_net_day_rate": float(by_day["total"].gt(0).mean()) if len(by_day) else 0.0,
        "win_rate": float(evaluated["_pnl"].gt(0).mean()) if not evaluated.empty else 0.0,
        "average_pnl": float(evaluated["_pnl"].mean()) if not evaluated.empty else 0.0,
        "average_return_on_risk": float(evaluated["_ror"].mean()) if not evaluated.empty else 0.0,
        "total_pnl": float(evaluated["_pnl"].sum()) if not evaluated.empty else 0.0,
        "profit_factor": float(profit_factor),
        "max_drawdown": float(drawdown.min()) if not drawdown.empty else 0.0,
    }


def opportunity_ceiling(detail: pd.DataFrame, all_days: Iterable[pd.Timestamp]) -> dict[str, Any]:
    days = [pd.Timestamp(day) for day in all_days]
    exact = detail[detail["_exact"] & detail["_pnl"].notna()].copy()
    guarded = exact[exact["_guard"]].copy()
    exact_days = set(exact.loc[exact["_pnl"].gt(0), "_asof_dt"])
    guarded_days = set(guarded.loc[guarded["_pnl"].gt(0), "_asof_dt"])
    return {
        "available_profitable_exact_days": int(sum(day in exact_days for day in days)),
        "available_profitable_exact_day_rate": float(sum(day in exact_days for day in days) / len(days)) if days else 0.0,
        "available_profitable_guarded_days": int(sum(day in guarded_days for day in days)),
        "available_profitable_guarded_day_rate": float(sum(day in guarded_days for day in days) / len(days)) if days else 0.0,
    }


def clustered_day_bootstrap(
    selected: pd.DataFrame,
    all_days: Iterable[pd.Timestamp],
    *,
    iterations: int = 1_000,
    seed: int = 20260722,
) -> dict[str, Any]:
    """Bootstrap whole entry days so correlated same-day candidates stay clustered."""

    days = [pd.Timestamp(day) for day in all_days]
    if not days or iterations <= 0:
        return {}
    evaluated = selected[selected["_exact"] & selected["_pnl"].notna()].copy() if not selected.empty else selected
    pnl_by_day = {
        day: evaluated.loc[evaluated["_asof_dt"].eq(day), "_pnl"].astype(float).to_numpy()
        if not evaluated.empty
        else np.array([], dtype=float)
        for day in days
    }
    rng = np.random.default_rng(seed)
    profit_factors: list[float] = []
    winner_rates: list[float] = []
    total_pnls: list[float] = []
    for _ in range(iterations):
        sampled_days = rng.choice(days, size=len(days), replace=True)
        chunks = [pnl_by_day[pd.Timestamp(day)] for day in sampled_days]
        pnl = np.concatenate([chunk for chunk in chunks if len(chunk)]) if any(len(chunk) for chunk in chunks) else np.array([], dtype=float)
        gross_profit = float(pnl[pnl > 0].sum()) if len(pnl) else 0.0
        gross_loss = float(-pnl[pnl < 0].sum()) if len(pnl) else 0.0
        profit_factors.append(gross_profit / gross_loss if gross_loss > 0 else math.inf if gross_profit > 0 else 0.0)
        winner_rates.append(float(np.mean([bool((chunk > 0).any()) for chunk in chunks])))
        total_pnls.append(float(pnl.sum()) if len(pnl) else 0.0)

    finite_pf = np.asarray([value for value in profit_factors if math.isfinite(value)], dtype=float)
    return {
        "bootstrap_iterations": int(iterations),
        "bootstrap_profit_factor_p05": float(np.quantile(finite_pf, 0.05)) if len(finite_pf) else math.inf,
        "bootstrap_profit_factor_median": float(np.quantile(finite_pf, 0.50)) if len(finite_pf) else math.inf,
        "bootstrap_profit_factor_p95": float(np.quantile(finite_pf, 0.95)) if len(finite_pf) else math.inf,
        "bootstrap_winner_day_rate_p05": float(np.quantile(winner_rates, 0.05)),
        "bootstrap_winner_day_rate_median": float(np.quantile(winner_rates, 0.50)),
        "bootstrap_winner_day_rate_p95": float(np.quantile(winner_rates, 0.95)),
        "bootstrap_total_pnl_p05": float(np.quantile(total_pnls, 0.05)),
        "bootstrap_total_pnl_median": float(np.quantile(total_pnls, 0.50)),
        "bootstrap_total_pnl_p95": float(np.quantile(total_pnls, 0.95)),
    }


def portfolio_metrics(
    selected: pd.DataFrame,
    *,
    monthly_target: float = 10_000.0,
) -> dict[str, Any]:
    evaluated = selected[selected["_exact"] & selected["_pnl"].notna()].copy() if not selected.empty else selected
    if evaluated.empty:
        return {
            "months": 0,
            "average_monthly_pnl_1x": 0.0,
            "worst_monthly_pnl_1x": 0.0,
            "months_meeting_target_1x": 0,
            "peak_concurrent_max_loss_1x": 0.0,
            "scale_to_average_monthly_target": math.inf,
            "scaled_peak_concurrent_max_loss": math.inf,
        }
    evaluated = evaluated.copy()
    evaluated["_month"] = evaluated["_asof_dt"].dt.to_period("M").astype(str)
    monthly = evaluated.groupby("_month")["_pnl"].sum()
    average_monthly = float(monthly.mean())
    entry = pd.to_numeric(evaluated.get("entry_price"), errors="coerce")
    width = pd.to_numeric(evaluated.get("entry_width"), errors="coerce")
    debit = evaluated.apply(_is_debit_strategy, axis=1)
    evaluated["_max_loss"] = entry * 100.0
    evaluated.loc[~debit, "_max_loss"] = (width[~debit] - entry[~debit]) * 100.0
    evaluated["_max_loss"] = evaluated["_max_loss"].clip(lower=0).fillna(0.0)
    start = evaluated["_asof_dt"].min()
    end = evaluated["_exit_dt"].max()
    peak_concurrent = 0.0
    if pd.notna(start) and pd.notna(end):
        for day in pd.date_range(start, end, freq="D"):
            active = evaluated[
                evaluated["_asof_dt"].le(day)
                & evaluated["_exit_dt"].fillna(evaluated["_asof_dt"]).ge(day)
            ]
            peak_concurrent = max(peak_concurrent, float(active["_max_loss"].sum()))
    scale = monthly_target / average_monthly if monthly_target > 0 and average_monthly > 0 else math.inf
    return {
        "months": int(len(monthly)),
        "average_monthly_pnl_1x": average_monthly,
        "median_monthly_pnl_1x": float(monthly.median()),
        "worst_monthly_pnl_1x": float(monthly.min()),
        "best_monthly_pnl_1x": float(monthly.max()),
        "positive_months": int(monthly.gt(0).sum()),
        "months_meeting_target_1x": int(monthly.ge(monthly_target).sum()),
        "monthly_target": float(monthly_target),
        "peak_concurrent_max_loss_1x": peak_concurrent,
        "scale_to_average_monthly_target": float(scale),
        "scaled_peak_concurrent_max_loss": float(peak_concurrent * scale) if math.isfinite(scale) else math.inf,
    }


def policy_grid() -> list[CoveragePolicy]:
    return [
        CoveragePolicy(max_per_day, eligibility_mode, dp_weight, oi_mode, history_weight, min_prior, model_weight)
        for max_per_day in (1, 2, 3, 5, 8)
        for eligibility_mode in ("policy", "guarded", "fillable")
        for dp_weight in (0.0, 0.20)
        for oi_mode in ("none", "reject_contrary")
        for history_weight in (0.0, 2.0)
        for min_prior in (8,)
        for model_weight in (0.0, 2.0)
    ]


def _policy_sort_key(metrics: dict[str, Any], minimum_profit_factor: float) -> tuple[float, ...]:
    qualifies = bool(
        metrics["evaluated_rows"] >= 20
        and metrics["profit_factor"] >= minimum_profit_factor
        and metrics["average_return_on_risk"] > 0
    )
    return (
        1.0 if qualifies else 0.0,
        metrics["winner_day_rate"] if qualifies else min(metrics["profit_factor"], minimum_profit_factor),
        metrics["profit_factor"],
        metrics["average_return_on_risk"],
        -abs(metrics["max_drawdown"]),
    )


def run_nested_walk_forward(
    detail: pd.DataFrame,
    *,
    minimum_profit_factor: float,
    bootstrap_iterations: int = 1_000,
    monthly_target: float = 10_000.0,
    require_complete_sources: bool = False,
    min_train_days: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    input_rows = int(len(detail))
    if require_complete_sources:
        bot = detail.get("bot_flow_source_status", pd.Series("", index=detail.index)).astype(str).str.lower()
        dp = detail.get("dark_pool_source_status", pd.Series("", index=detail.index)).astype(str).str.lower()
        oi = detail.get("chain_oi_source_status", pd.Series("", index=detail.index)).astype(str).str.lower()
        detail = detail[
            bot.str.startswith("bot_eod")
            & dp.str.startswith("dp_eod")
            & oi.eq("chain_oi_loaded")
        ].copy()
    prepared = prepare_detail(detail)
    periods = sorted(prepared["_asof_dt"].dt.to_period("M").unique())
    fold_rows: list[dict[str, Any]] = []
    selected_parts: list[pd.DataFrame] = []
    policies = policy_grid()
    policy_selections = {
        policy: select_policy_trades(prepared, policy)
        for policy in policies
    }
    for period_index in range(3, len(periods)):
        test_period = periods[period_index]
        test_start = test_period.start_time
        test_end = test_period.end_time
        train_days = sorted(prepared.loc[prepared["_asof_dt"].lt(test_start), "_asof_dt"].unique())
        test_days = sorted(prepared.loc[prepared["_asof_dt"].between(test_start, test_end), "_asof_dt"].unique())
        if len(train_days) < min_train_days or len(test_days) < 5:
            continue
        candidates: list[tuple[CoveragePolicy, dict[str, Any]]] = []
        for policy in policies:
            policy_selected = policy_selections[policy]
            train_selected = (
                policy_selected[policy_selected["_asof_dt"].lt(test_start)]
                if not policy_selected.empty
                else policy_selected
            )
            metrics = performance(train_selected, train_days)
            candidates.append((policy, metrics))
        chosen, train_metrics = max(
            candidates,
            key=lambda item: (_policy_sort_key(item[1], minimum_profit_factor), item[0].policy_id),
        )
        chosen_selected = policy_selections[chosen]
        test_selected = (
            chosen_selected[chosen_selected["_asof_dt"].between(test_start, test_end)].copy()
            if not chosen_selected.empty
            else chosen_selected
        )
        test_metrics = performance(test_selected, test_days)
        test_ceiling = opportunity_ceiling(
            prepared[prepared["_asof_dt"].between(test_start, test_end)],
            test_days,
        )
        if not test_selected.empty:
            test_selected = test_selected.copy()
            test_selected["goal_test_period"] = str(test_period)
            selected_parts.append(test_selected)
        fold_rows.append(
            {
                "test_period": str(test_period),
                "chosen_policy": chosen.policy_id,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"test_{key}": value for key, value in test_metrics.items()},
                **{f"test_{key}": value for key, value in test_ceiling.items()},
            }
        )
    folds = pd.DataFrame(fold_rows)
    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()
    pooled_days = sorted(
        prepared.loc[
            prepared["_asof_dt"].dt.to_period("M").isin(periods[3:]),
            "_asof_dt",
        ].unique()
    )
    pooled_periods = set(periods[3:])
    frontier_rows: list[dict[str, Any]] = []
    for policy, policy_selected in policy_selections.items():
        pooled_selected = (
            policy_selected[policy_selected["_asof_dt"].dt.to_period("M").isin(pooled_periods)]
            if not policy_selected.empty
            else policy_selected
        )
        frontier_rows.append(
            {
                "policy_id": policy.policy_id,
                **performance(pooled_selected, pooled_days),
            }
        )
    frontier = pd.DataFrame(frontier_rows).sort_values(
        ["winner_day_rate", "profit_factor", "average_return_on_risk"],
        ascending=[False, False, False],
    )
    summary = performance(selected, pooled_days)
    summary.update(
        opportunity_ceiling(
            prepared[prepared["_asof_dt"].dt.to_period("M").isin(periods[3:])],
            pooled_days,
        )
    )
    summary.update(
        {
            "minimum_profit_factor": minimum_profit_factor,
            "folds": int(len(folds)),
            "folds_meeting_profit_factor": int(folds["test_profit_factor"].ge(minimum_profit_factor).sum()) if not folds.empty else 0,
            "folds_with_winner_every_day": int(folds["test_winner_day_rate"].eq(1.0).sum()) if not folds.empty else 0,
            "selection_uses_future_outcomes": False,
            "prior_outcome_rule": "exit_day strictly before candidate asof",
            "require_complete_sources": bool(require_complete_sources),
            "input_rows": input_rows,
            "source_complete_rows": int(len(detail)),
            "minimum_train_days": int(min_train_days),
        }
    )
    summary.update(
        clustered_day_bootstrap(
            selected,
            pooled_days,
            iterations=bootstrap_iterations,
        )
    )
    summary.update(portfolio_metrics(selected, monthly_target=monthly_target))
    summary["daily_winner_goal_met"] = bool(summary["winner_day_rate"] >= 1.0)
    summary["profitability_goal_met"] = bool(
        summary["profit_factor"] >= minimum_profit_factor
        and summary["average_return_on_risk"] > 0
        and summary.get("bootstrap_profit_factor_p05", 0.0) >= minimum_profit_factor
        and summary["folds_meeting_profit_factor"] == summary["folds"]
        and summary["folds"] > 0
    )
    summary["can_promote_goal_selector"] = bool(
        summary["daily_winner_goal_met"] and summary["profitability_goal_met"]
    )
    qualified_frontier = frontier[
        frontier["profit_factor"].ge(minimum_profit_factor)
        & frontier["average_return_on_risk"].gt(0)
    ]
    full_coverage_frontier = frontier[frontier["winner_day_rate"].ge(1.0)]
    summary["frontier_policies"] = int(len(frontier))
    summary["max_winner_day_rate_at_required_pf"] = (
        float(qualified_frontier["winner_day_rate"].max()) if not qualified_frontier.empty else 0.0
    )
    summary["best_required_pf_policy"] = (
        str(qualified_frontier.iloc[0]["policy_id"]) if not qualified_frontier.empty else ""
    )
    summary["full_daily_coverage_policy_count"] = int(len(full_coverage_frontier))
    summary["best_profit_factor_at_full_daily_coverage"] = (
        float(full_coverage_frontier["profit_factor"].max()) if not full_coverage_frontier.empty else 0.0
    )
    return folds, selected, frontier, summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Outcome-blind /goal audit for daily profitable-trade coverage.")
    parser.add_argument("--detail", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--minimum-profit-factor", type=float, default=1.20)
    parser.add_argument("--bootstrap-iterations", type=int, default=1_000)
    parser.add_argument("--monthly-target", type=float, default=10_000.0)
    parser.add_argument("--require-complete-sources", action="store_true")
    parser.add_argument("--min-train-days", type=int, default=30)
    args = parser.parse_args(argv)

    detail = pd.read_csv(args.detail, low_memory=False)
    folds, selected, frontier, summary = run_nested_walk_forward(
        detail,
        minimum_profit_factor=args.minimum_profit_factor,
        bootstrap_iterations=args.bootstrap_iterations,
        monthly_target=args.monthly_target,
        require_complete_sources=args.require_complete_sources,
        min_train_days=args.min_train_days,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    folds.to_csv(out_dir / "codexuw_goal_walk_forward_folds.csv", index=False)
    selected.to_csv(out_dir / "codexuw_goal_selected_trades.csv", index=False)
    frontier.to_csv(out_dir / "codexuw_goal_policy_frontier.csv", index=False)
    (out_dir / "codexuw_goal_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
