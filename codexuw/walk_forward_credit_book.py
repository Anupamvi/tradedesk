from __future__ import annotations

import datetime as dt
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .credit_policy import MAX_QUOTE_WIDTH_PCT, MIN_CREDIT_PCT_WIDTH
from .data import safe_float
WALK_FORWARD_CREDIT_VERSION = "v4-directional-execution-parity-20260813"
WALK_FORWARD_CREDIT_ACTIVATION_DATE = dt.date(2026, 8, 11)
DEFAULT_WALK_FORWARD_CREDIT_HISTORY = (
    Path(__file__).resolve().parent
    / "history"
    / "codexdaily_v4_credit_payoff_history_v1_2026-08-10.csv.gz"
)
VALIDATION_MONTHS = ("2026-04", "2026-05", "2026-06", "2026-07")
HOLDOUT_MONTHS = ("2026-06", "2026-07")
MIN_OOS_TRADES = 20
MIN_OOS_WIN_RATE = 0.75
MIN_OOS_WILSON_LOWER_BOUND = 0.65
MIN_OOS_STRESS_PROFIT_FACTOR = 2.00
MIN_POSITIVE_VALIDATION_MONTHS = 4
MIN_HOLDOUT_TRADES = 15
MIN_HOLDOUT_WIN_RATE = 0.70
MIN_HOLDOUT_STRESS_PROFIT_FACTOR = 1.50
MIN_POSITIVE_HOLDOUT_MONTHS = 2
MAX_DRAWDOWN_TO_STRESSED_PNL = 0.50
MAX_HISTORY_AGE_DAYS = 14
MIN_WIN_PROBABILITY = 0.65
MIN_PREDICTED_STRESS_ROR = 0.01
MAX_MODEL_DTE = 45
RECENT_HEALTH_LOOKBACK_DAYS = 90
MIN_RECENT_DIRECTION_SAMPLE = 30
MIN_RECENT_DIRECTION_REGIME_SAMPLE = 20
MIN_RECENT_STRESS_PROFIT_FACTOR = 1.10
MAX_DAILY_MEDIUM_TARGETS = 2
DIRECTIONAL_CREDIT_VERSION = "v2.2-family-subgroup-evidence-20260813"
DIRECTIONAL_CREDIT_ACTIVATION_DATE = dt.date(2026, 8, 11)
DEFAULT_DIRECTIONAL_CREDIT_HISTORY = (
    Path(__file__).resolve().parent
    / "history"
    / "codexdaily_v4_directional_credit_history_v1_2026-08-12.csv.gz"
)
DIRECTIONAL_CREDIT_TRAIN_HOLDOUT_CUTOFF = pd.Timestamp("2026-05-19")
DIRECTIONAL_CREDIT_MIN_PCT_WIDTH = 0.15
DIRECTIONAL_CREDIT_MAX_PCT_WIDTH = 0.45
DIRECTIONAL_CREDIT_MAX_QUOTE_WIDTH = 0.35
DIRECTIONAL_CREDIT_MAX_EXPECTED_MOVE_RATIO = 0.90
DIRECTIONAL_CREDIT_MIN_DTE = 21
DIRECTIONAL_CREDIT_MAX_DTE = 44
DIRECTIONAL_CREDIT_MIN_SAMPLE = 75
DIRECTIONAL_CREDIT_MIN_WIN_RATE = 0.80
DIRECTIONAL_CREDIT_MIN_WILSON = 0.78
DIRECTIONAL_CREDIT_MIN_PROFIT_FACTOR = 2.00
DIRECTIONAL_CREDIT_MIN_POSITIVE_MONTHS = 7
DIRECTIONAL_CREDIT_MIN_TRAIN_SAMPLE = 40
DIRECTIONAL_CREDIT_MIN_TRAIN_PROFIT_FACTOR = 2.00
DIRECTIONAL_CREDIT_MIN_HOLDOUT_SAMPLE = 20
DIRECTIONAL_CREDIT_MIN_HOLDOUT_WIN_RATE = 0.75
DIRECTIONAL_CREDIT_MIN_HOLDOUT_PROFIT_FACTOR = 1.50
DIRECTIONAL_CREDIT_EXECUTION_MIN_SAMPLE = 50
DIRECTIONAL_CREDIT_EXECUTION_MIN_TRAIN_SAMPLE = 30
DIRECTIONAL_CREDIT_FAMILY_MIN_SAMPLE = 15
DIRECTIONAL_CREDIT_FAMILY_MIN_WILSON = 0.65
DIRECTIONAL_CREDIT_FAMILY_MIN_PROFIT_FACTOR = 1.50
DIRECTIONAL_CREDIT_FAMILY_MIN_POSITIVE_MONTHS = 5
DIRECTIONAL_CREDIT_FAMILY_MIN_HOLDOUT_SAMPLE = 5
DIRECTIONAL_CREDIT_FAMILY_MIN_HOLDOUT_PROFIT_FACTOR = 1.25
DIRECTIONAL_CREDIT_FAMILY_MIN_PROBATIONARY_HOLDOUT = 3
DIRECTIONAL_CREDIT_MAX_HISTORY_AGE_DAYS = 14
DIRECTIONAL_CREDIT_EXECUTION_OI_STATES = frozenset({"supportive", "matched_unconfirmed"})

NUMERIC_FEATURES = (
    "entry_credit_pct_width",
    "entry_quote_width_pct",
    "combined_flow_bias",
    "flow_total_premium",
    "iv_rank",
    "iv30d",
    "realized_volatility_30d",
    "iv_hv_ratio",
    "iv_hv_spread",
    "expected_move_ratio",
    "distance_pct",
    "bot_volume_oi_ratio",
    "source_multileg_ratio",
    "source_stock_multileg_ratio",
    "short_leg_oi_change",
    "long_leg_oi_change",
    "entry_dte",
)
CATEGORICAL_FEATURES = (
    "direction",
    "regime",
    "flow_quality",
    "sector",
    "sell_leg_quote_source",
    "construction_source",
    "oi_carryover_status",
)
FEATURE_INPUT_SCALES = {
    "flow_total_premium": 1_000_000.0,
    "short_leg_oi_change": 1_000.0,
    "long_leg_oi_change": 1_000.0,
}
MAX_NORMALIZED_FEATURE_ABS = 1_000_000.0
MAX_STANDARDIZED_FEATURE_ABS = 25.0


@dataclass
class CreditPayoffModel:
    preprocessor: ColumnTransformer
    coefficients: np.ndarray
    intercept: float
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    win_models: dict[str, Pipeline]
    recent_direction_health: dict[str, dict[str, float]]
    recent_direction_regime_health: dict[tuple[str, str], dict[str, float]]


def _truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _profit_factor(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    gain = float(clean[clean > 0].sum())
    loss = float(-clean[clean < 0].sum())
    if loss > 0:
        return gain / loss
    return math.inf if gain > 0 else math.nan


def _max_drawdown(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if not len(clean):
        return 0.0
    equity = np.concatenate(([0.0], np.cumsum(clean)))
    drawdown = equity - np.maximum.accumulate(equity)
    return float(drawdown.min())


def _wilson_lower_bound(wins: int, sample: int, z: float = 1.959963984540054) -> float:
    if sample <= 0:
        return math.nan
    probability = wins / sample
    denominator = 1.0 + z * z / sample
    center = probability + z * z / (2.0 * sample)
    margin = z * math.sqrt((probability * (1.0 - probability) + z * z / (4.0 * sample)) / sample)
    return max(0.0, (center - margin) / denominator)


def _prepare_history(history: pd.DataFrame) -> pd.DataFrame:
    out = history.copy()
    for column in ["asof", "entry_day", "exit_day", "expiry", "next_earnings_dt", "history_observation_end"]:
        if column in out.columns:
            out[column] = pd.to_datetime(out[column], errors="coerce")
    for column in [*NUMERIC_FEATURES, "entry_credit", "entry_width", "pnl_1x"]:
        out[column] = pd.to_numeric(out.get(column), errors="coerce").replace([np.inf, -np.inf], np.nan)
    exact = _truthy(out.get("exact_evaluated", pd.Series(False, index=out.index)))
    no_earnings_cross = ~(
        out["next_earnings_dt"].notna()
        & (out["next_earnings_dt"] >= out["entry_day"])
        & (out["next_earnings_dt"] <= out["expiry"])
    )
    same_source = out.get("sell_leg_quote_source", pd.Series("", index=out.index)).eq(
        out.get("buy_leg_quote_source", pd.Series("", index=out.index))
    )
    supported_source = out.get("sell_leg_quote_source", pd.Series("", index=out.index)).isin(
        {"bot_eod_first_regular_nbbo", "hot_chain"}
    )
    out = out[
        exact
        & out.get("strategy_kind", pd.Series("", index=out.index)).eq("Credit")
        & out["pnl_1x"].notna()
        & out["entry_credit"].gt(0)
        & out["entry_width"].gt(out["entry_credit"])
        & no_earnings_cross
        & same_source
        & supported_source
    ].copy()
    out["stress_pnl_5pct"] = out["pnl_1x"] - out["entry_credit"] * 5.0
    out["stress_pnl_10pct"] = out["pnl_1x"] - out["entry_credit"] * 10.0
    stressed_risk = ((out["entry_width"] - out["entry_credit"] * 0.90) * 100.0).clip(lower=1.0)
    out["stress_return_on_risk_10pct"] = (out["stress_pnl_10pct"] / stressed_risk).clip(-1.20, 0.50)
    return out


def load_walk_forward_credit_history(path: Path = DEFAULT_WALK_FORWARD_CREDIT_HISTORY) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return _prepare_history(pd.read_csv(path, low_memory=False))


def _sanitize_numeric_features(frame: pd.DataFrame, numeric: tuple[str, ...]) -> pd.DataFrame:
    """Normalize and bound numeric inputs identically for fit and inference."""
    out = frame.copy()
    for column in numeric:
        out[column] = pd.to_numeric(out.get(column), errors="coerce").replace([np.inf, -np.inf], np.nan)
        out[column] = out[column] / FEATURE_INPUT_SCALES.get(column, 1.0)
        out[column] = out[column].clip(-MAX_NORMALIZED_FEATURE_ABS, MAX_NORMALIZED_FEATURE_ABS)
    return out


def _build_model(train: pd.DataFrame) -> CreditPayoffModel | None:
    if train.empty:
        return None
    numeric = tuple(column for column in NUMERIC_FEATURES if column in train and train[column].notna().any())
    categorical = tuple(column for column in CATEGORICAL_FEATURES if column in train)
    if not numeric and not categorical:
        return None
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                list(numeric),
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                list(categorical),
            )
        )
    preprocessor = ColumnTransformer(transformers, sparse_threshold=0.0)
    day_count = train.groupby("asof")["ticker"].transform("size").clip(lower=1)
    sample_weight = (1.0 / day_count).to_numpy(dtype=float)
    columns = list(numeric) + list(categorical)
    work = _sanitize_numeric_features(train[columns], numeric)
    transformed = np.asarray(preprocessor.fit_transform(work), dtype=float)
    transformed = np.clip(transformed, -MAX_STANDARDIZED_FEATURE_ABS, MAX_STANDARDIZED_FEATURE_ABS)
    target = pd.to_numeric(train["stress_return_on_risk_10pct"], errors="coerce").to_numpy(dtype=float)
    if (
        transformed.ndim != 2
        or transformed.shape[1] == 0
        or not np.isfinite(transformed).all()
        or not np.isfinite(target).all()
        or not np.isfinite(sample_weight).all()
    ):
        return None

    weight_sum = float(sample_weight.sum())
    if weight_sum <= 0:
        return None
    feature_mean = np.einsum("ni,n->i", transformed, sample_weight, optimize=False) / weight_sum
    target_mean = float(np.einsum("n,n->", target, sample_weight, optimize=False) / weight_sum)
    centered_features = np.ascontiguousarray(transformed - feature_mean)
    centered_target = np.ascontiguousarray(target - target_mean)
    gram = np.einsum(
        "ni,n,nj->ij",
        centered_features,
        sample_weight,
        centered_features,
        optimize=False,
    )
    response = np.einsum(
        "ni,n,n->i",
        centered_features,
        sample_weight,
        centered_target,
        optimize=False,
    )
    coefficients = np.linalg.solve(gram + 10.0 * np.eye(gram.shape[0]), response)
    intercept = target_mean - float(np.einsum("i,i->", feature_mean, coefficients, optimize=False))
    if not np.isfinite(coefficients).all() or not math.isfinite(intercept):
        return None
    win_models: dict[str, Pipeline] = {}
    for direction in ("Bull Put", "Bear Call"):
        family = train[train.get("direction", pd.Series("", index=train.index)).eq(direction)].copy()
        outcome = pd.to_numeric(family.get("stress_pnl_10pct"), errors="coerce").gt(0).astype(int)
        if len(family) < 100 or outcome.nunique() < 2:
            continue
        family_work = _sanitize_numeric_features(family[columns], numeric)
        family_preprocessor = ColumnTransformer(transformers, sparse_threshold=0.0)
        classifier = Pipeline(
            [
                ("preprocessor", family_preprocessor),
                (
                    "model",
                    LogisticRegression(
                        C=0.1,
                        solver="liblinear",
                        max_iter=1000,
                        random_state=7,
                    ),
                ),
            ]
        )
        classifier.fit(family_work, outcome)
        win_models[direction] = classifier
    return CreditPayoffModel(
        preprocessor=preprocessor,
        coefficients=coefficients,
        intercept=intercept,
        numeric_features=numeric,
        categorical_features=categorical,
        win_models=win_models,
        recent_direction_health={},
        recent_direction_regime_health={},
    )


def _predict(model: CreditPayoffModel, rows: pd.DataFrame) -> np.ndarray:
    work = _sanitize_numeric_features(rows, model.numeric_features)
    for column in model.categorical_features:
        if column not in work:
            work[column] = ""
    transformed = np.asarray(
        model.preprocessor.transform(work[list(model.numeric_features) + list(model.categorical_features)]),
        dtype=float,
    )
    transformed = np.clip(transformed, -MAX_STANDARDIZED_FEATURE_ABS, MAX_STANDARDIZED_FEATURE_ABS)
    if not np.isfinite(transformed).all():
        return np.full(len(work), np.nan, dtype=float)
    return np.einsum("ni,i->n", transformed, model.coefficients, optimize=False) + model.intercept


def _predict_win_probability(model: CreditPayoffModel, rows: pd.DataFrame) -> np.ndarray:
    probabilities = pd.Series(np.nan, index=rows.index, dtype=float)
    columns = list(model.numeric_features) + list(model.categorical_features)
    for direction, classifier in model.win_models.items():
        mask = rows.get("direction", pd.Series("", index=rows.index)).eq(direction)
        if not mask.any():
            continue
        work = _sanitize_numeric_features(rows.loc[mask], model.numeric_features)
        for column in model.categorical_features:
            if column not in work:
                work[column] = ""
        transformed = np.asarray(classifier.named_steps["preprocessor"].transform(work[columns]), dtype=float)
        transformed = np.nan_to_num(
            transformed,
            nan=0.0,
            posinf=MAX_STANDARDIZED_FEATURE_ABS,
            neginf=-MAX_STANDARDIZED_FEATURE_ABS,
        )
        transformed = np.clip(transformed, -MAX_STANDARDIZED_FEATURE_ABS, MAX_STANDARDIZED_FEATURE_ABS)
        estimator = classifier.named_steps["model"]
        coefficients = np.nan_to_num(
            np.asarray(estimator.coef_, dtype=float), nan=0.0, posinf=100.0, neginf=-100.0
        )
        coefficients = np.clip(coefficients, -100.0, 100.0)
        intercept = np.nan_to_num(
            np.asarray(estimator.intercept_, dtype=float), nan=0.0, posinf=100.0, neginf=-100.0
        )
        intercept = np.clip(intercept, -100.0, 100.0)
        logits = np.einsum("ni,ji->nj", transformed, coefficients, optimize=False) + intercept
        logits = np.clip(logits.reshape(-1), -35.0, 35.0)
        probabilities.loc[mask] = 1.0 / (1.0 + np.exp(-logits))
    return probabilities.to_numpy(dtype=float)


def _objective_credit_guard(row: pd.Series | dict[str, Any]) -> tuple[bool, str]:
    direction = str(row.get("direction") or "")
    if direction not in {"Bull Put", "Bear Call"}:
        return False, "not_credit_spread"
    credit_pct = safe_float(row.get("credit_pct_width"), safe_float(row.get("entry_credit_pct_width")))
    if not math.isfinite(credit_pct) or credit_pct < MIN_CREDIT_PCT_WIDTH:
        return False, "credit_below_guard_floor"
    if credit_pct >= 1.0:
        return False, "credit_at_or_above_spread_width"
    quote_width = safe_float(row.get("quote_width_pct"), safe_float(row.get("entry_quote_width_pct")))
    if not math.isfinite(quote_width) or quote_width > MAX_QUOTE_WIDTH_PCT:
        return False, "credit_quote_too_wide"
    dte = safe_float(row.get("dte"), safe_float(row.get("entry_dte")))
    if not math.isfinite(dte) or dte <= 0 or dte > MAX_MODEL_DTE:
        return False, "dte_outside_validated_range"
    hard_raw = row.get("hard_rejects")
    hard_text = "" if hard_raw is None or (not isinstance(hard_raw, str) and pd.isna(hard_raw)) else str(hard_raw)
    hard_parts = [part.strip() for part in re.split(r"[;,]", hard_text) if part.strip()]
    remaining_hard = [
        part
        for part in hard_parts
        if "thin_replay_sample" not in part.lower() and "negative_replay_edge" not in part.lower()
    ]
    if remaining_hard:
        return False, "preexisting_hard_blocker:" + ";".join(remaining_hard)
    natural_credit = safe_float(row.get("natural_credit"))
    if math.isfinite(natural_credit) and natural_credit <= 0:
        return False, "nonpositive_natural_credit"
    mid_credit = safe_float(row.get("mid_credit"))
    if math.isfinite(mid_credit) and mid_credit <= 0:
        return False, "nonpositive_mid_credit"
    width = safe_float(row.get("width"), safe_float(row.get("entry_width")))
    if math.isfinite(width) and width > 0:
        if math.isfinite(natural_credit) and natural_credit >= width:
            return False, "natural_credit_at_or_above_spread_width"
        if math.isfinite(mid_credit) and mid_credit >= width:
            return False, "mid_credit_at_or_above_spread_width"
    expiry = pd.to_datetime(row.get("expiry", row.get("entry_expiry")), errors="coerce")
    next_earnings = pd.to_datetime(row.get("next_earnings_dt"), errors="coerce")
    if pd.notna(expiry) and pd.notna(next_earnings) and next_earnings <= expiry:
        return False, "earnings_crosses_expiry"
    return True, "objective_credit_quality_pass"


def _recent_health_maps(
    train: pd.DataFrame,
    evaluation_start: pd.Timestamp,
) -> tuple[dict[str, dict[str, float]], dict[tuple[str, str], dict[str, float]]]:
    if train.empty:
        return {}, {}
    recent = train[
        train["asof"].ge(evaluation_start - pd.Timedelta(days=RECENT_HEALTH_LOOKBACK_DAYS))
    ].copy()
    if recent.empty:
        return {}, {}
    objective = recent.apply(lambda row: _objective_credit_guard(row)[0], axis=1)
    recent = recent[objective].copy()

    def summarize(frame: pd.DataFrame) -> dict[str, float]:
        stressed = pd.to_numeric(frame.get("stress_pnl_10pct"), errors="coerce")
        return {
            "sample": float(len(frame)),
            "profit_factor": float(_profit_factor(stressed)),
            "average_pnl": float(stressed.mean()),
        }

    direction = {str(key): summarize(part) for key, part in recent.groupby("direction")}
    direction_regime = {
        (str(key[0]), str(key[1])): summarize(part)
        for key, part in recent.groupby(["direction", "regime"])
    }
    return direction, direction_regime


def _attach_recent_health(
    model: CreditPayoffModel,
    train: pd.DataFrame,
    evaluation_start: pd.Timestamp,
) -> CreditPayoffModel:
    direction, direction_regime = _recent_health_maps(train, evaluation_start)
    model.recent_direction_health = direction
    model.recent_direction_regime_health = direction_regime
    return model


def _recent_health_for_row(
    row: pd.Series | dict[str, Any],
    model: CreditPayoffModel,
) -> tuple[bool, str, dict[str, float]]:
    direction = str(row.get("direction") or "")
    regime = str(row.get("regime") or row.get("regime_trend") or "")
    health = model.recent_direction_regime_health.get((direction, regime), {})
    minimum_sample = MIN_RECENT_DIRECTION_REGIME_SAMPLE
    scope = "direction_regime"
    if safe_float(health.get("sample"), 0.0) < minimum_sample:
        health = model.recent_direction_health.get(direction, {})
        minimum_sample = MIN_RECENT_DIRECTION_SAMPLE
        scope = "direction"
    sample = safe_float(health.get("sample"), 0.0)
    profit_factor = safe_float(health.get("profit_factor"), 0.0)
    average_pnl = safe_float(health.get("average_pnl"), -math.inf)
    passed = bool(
        sample >= minimum_sample
        and profit_factor >= MIN_RECENT_STRESS_PROFIT_FACTOR
        and average_pnl > 0
    )
    return passed, scope, health


def _select_one_per_day(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    work = rows.copy()
    work["_walk_forward_rank"] = (
        pd.to_numeric(work.get("walk_forward_credit_win_probability"), errors="coerce")
        * pd.to_numeric(work.get("walk_forward_credit_prediction"), errors="coerce").clip(lower=0)
    )
    work = work.sort_values(
        ["_walk_forward_rank", "walk_forward_credit_win_probability", "entry_quote_width_pct"],
        ascending=[False, False, True],
    ).drop_duplicates(["asof", "ticker", "direction"])
    active_until: dict[str, pd.Timestamp] = {}
    selected: list[int] = []
    for day, day_rows in work.groupby("asof", sort=True):
        used_sectors: set[str] = set()
        selected_today = 0
        for index, candidate in day_rows.iterrows():
            ticker = str(candidate.get("ticker"))
            sector = str(candidate.get("sector") or "Unknown")
            if active_until.get(ticker, pd.Timestamp.min) >= day or sector in used_sectors:
                continue
            if safe_float(candidate.get("walk_forward_credit_prediction"), -math.inf) < MIN_PREDICTED_STRESS_ROR:
                continue
            selected.append(index)
            active_until[ticker] = pd.Timestamp(candidate.get("exit_day"))
            used_sectors.add(sector)
            selected_today += 1
            if selected_today >= MAX_DAILY_MEDIUM_TARGETS:
                break
    return work.loc[selected].drop(columns=["_walk_forward_rank"])


def _validation_metrics(selected: pd.DataFrame) -> dict[str, Any]:
    ordered = selected.sort_values(["asof", "ticker"]) if not selected.empty else selected
    sample = int(len(ordered))
    wins = int(pd.to_numeric(ordered.get("stress_pnl_10pct"), errors="coerce").gt(0).sum()) if sample else 0
    stressed = pd.to_numeric(ordered.get("stress_pnl_10pct"), errors="coerce") if sample else pd.Series(dtype=float)
    base = pd.to_numeric(ordered.get("pnl_1x"), errors="coerce") if sample else pd.Series(dtype=float)
    stress_5 = pd.to_numeric(ordered.get("stress_pnl_5pct"), errors="coerce") if sample else pd.Series(dtype=float)
    monthly = (
        ordered.assign(_month=ordered["asof"].dt.to_period("M").astype(str))
        .groupby("_month")["stress_pnl_10pct"]
        .sum()
        if sample
        else pd.Series(dtype=float)
    )
    total_stressed = float(stressed.sum()) if sample else 0.0
    max_drawdown = _max_drawdown(stressed)
    drawdown_ratio = abs(max_drawdown) / total_stressed if total_stressed > 0 else math.inf
    return {
        "sample_size": sample,
        "wins": wins,
        "win_rate": wins / sample if sample else math.nan,
        "wilson_lower_bound": _wilson_lower_bound(wins, sample),
        "bayesian_win_probability": (wins + 1.0) / (sample + 2.0) if sample else math.nan,
        "base_profit_factor": _profit_factor(base),
        "stress_profit_factor_5pct": _profit_factor(stress_5),
        "stress_profit_factor_10pct": _profit_factor(stressed),
        "stress_average_pnl_10pct": float(stressed.mean()) if sample else math.nan,
        "stress_total_pnl_10pct": total_stressed,
        "max_drawdown_10pct": max_drawdown,
        "max_drawdown_to_stressed_pnl": drawdown_ratio,
        "positive_months": int((monthly > 0).sum()),
        "monthly_stress_pnl_10pct": {str(key): float(value) for key, value in monthly.items()},
    }


def _directional_credit_metrics(rows: pd.DataFrame) -> dict[str, Any]:
    ordered = rows.sort_values(["asof", "ticker"]) if not rows.empty else rows
    stressed = pd.to_numeric(ordered.get("stress_pnl_10pct"), errors="coerce").dropna()
    sample = int(len(stressed))
    wins = int(stressed.gt(0).sum())
    monthly = (
        ordered.assign(_month=ordered["asof"].dt.to_period("M").astype(str))
        .groupby("_month")["stress_pnl_10pct"]
        .sum()
        if sample
        else pd.Series(dtype=float)
    )
    total = float(stressed.sum()) if sample else 0.0
    max_drawdown = _max_drawdown(stressed)
    width = pd.to_numeric(ordered.get("entry_width"), errors="coerce")
    credit = pd.to_numeric(ordered.get("entry_credit"), errors="coerce")
    risk = ((width - credit) * 100.0).where((width > credit) & credit.gt(0))
    stressed_return = (pd.to_numeric(ordered.get("stress_pnl_10pct"), errors="coerce") / risk).replace(
        [np.inf, -np.inf], np.nan
    )
    win_returns = stressed_return[stressed_return > 0]
    loss_returns = -stressed_return[stressed_return < 0]
    return {
        "sample_size": sample,
        "wins": wins,
        "win_rate": wins / sample if sample else math.nan,
        "wilson_lower_bound": _wilson_lower_bound(wins, sample),
        "bayesian_win_probability": (wins + 1.0) / (sample + 2.0) if sample else math.nan,
        "stress_profit_factor_10pct": _profit_factor(stressed),
        "stress_average_pnl_10pct": float(stressed.mean()) if sample else math.nan,
        "stress_average_return_on_risk_10pct": float(stressed_return.mean()) if stressed_return.notna().any() else math.nan,
        "stress_average_win_risk_fraction_10pct": float(win_returns.mean()) if not win_returns.empty else math.nan,
        "stress_average_loss_risk_fraction_10pct": float(loss_returns.mean()) if not loss_returns.empty else math.nan,
        "stress_total_pnl_10pct": total,
        "max_drawdown_10pct": max_drawdown,
        "max_drawdown_to_stressed_pnl": abs(max_drawdown) / total if total > 0 else math.inf,
        "positive_months": int((monthly > 0).sum()),
        "months_observed": int(len(monthly)),
        "monthly_stress_pnl_10pct": {str(key): float(value) for key, value in monthly.items()},
    }


def build_directional_credit_summary(
    *,
    asof: dt.date,
    history_path: Path = DEFAULT_DIRECTIONAL_CREDIT_HISTORY,
) -> dict[str, Any]:
    if not history_path.exists():
        return {
            "version": DIRECTIONAL_CREDIT_VERSION,
            "status": "FAIL",
            "reason": "directional credit evidence ledger is unavailable",
        }
    history = pd.read_csv(history_path, low_memory=False)
    for column in ("asof", "entry_day", "exit_day", "expiry", "next_earnings_dt", "history_observation_end"):
        history[column] = pd.to_datetime(history.get(column), errors="coerce")
    for column in (
        "pnl_1x",
        "entry_credit",
        "entry_width",
        "entry_credit_pct_width",
        "entry_quote_width_pct",
        "expected_move_ratio",
        "entry_dte",
        "combined_flow_bias",
        "stress_pnl_10pct",
    ):
        history[column] = pd.to_numeric(history.get(column), errors="coerce")
    # Never trust a packaged stress column.  The frozen V4.21 ledger copied
    # base P/L into this field, which overstated every stress metric.  Rebuild
    # both scenarios from the exact one-contract P/L and entry credit.
    history["stress_pnl_5pct"] = history["pnl_1x"] - history["entry_credit"] * 5.0
    history["stress_pnl_10pct"] = history["pnl_1x"] - history["entry_credit"] * 10.0
    no_earnings_cross = ~(
        history["next_earnings_dt"].notna()
        & (history["next_earnings_dt"] >= history["entry_day"])
        & (history["next_earnings_dt"] <= history["expiry"])
    )
    flow_aligned = (
        (history.get("direction").eq("Bull Put") & history["combined_flow_bias"].gt(0))
        | (history.get("direction").eq("Bear Call") & history["combined_flow_bias"].lt(0))
    )
    eligible_population = history[
        _truthy(history.get("exact_evaluated", pd.Series(False, index=history.index)))
        & history.get("direction", pd.Series("", index=history.index)).isin({"Bull Put", "Bear Call"})
        & history.get("flow_quality", pd.Series("", index=history.index)).eq("directional")
        & history.get("oi_carryover_status", pd.Series("", index=history.index)).isin(
            {"supportive", "matched_unconfirmed", "contrary"}
        )
        & flow_aligned
        & no_earnings_cross
        & history["entry_credit"].gt(0)
        & history["entry_width"].gt(history["entry_credit"])
        & history["entry_credit_pct_width"].between(
            DIRECTIONAL_CREDIT_MIN_PCT_WIDTH,
            DIRECTIONAL_CREDIT_MAX_PCT_WIDTH,
        )
        & history["entry_quote_width_pct"].le(DIRECTIONAL_CREDIT_MAX_QUOTE_WIDTH)
        & history["expected_move_ratio"].le(DIRECTIONAL_CREDIT_MAX_EXPECTED_MOVE_RATIO)
        & history["entry_dte"].between(DIRECTIONAL_CREDIT_MIN_DTE, DIRECTIONAL_CREDIT_MAX_DTE)
        & history["stress_pnl_10pct"].notna()
    ].copy()
    eligible = eligible_population.sort_values(["asof", "ticker"]).drop_duplicates("asof", keep="first")
    execution_eligible = (
        eligible_population[
            eligible_population.get("oi_carryover_status", pd.Series("", index=eligible_population.index)).isin(
                DIRECTIONAL_CREDIT_EXECUTION_OI_STATES
            )
        ]
        .sort_values(["asof", "ticker"])
        .drop_duplicates("asof", keep="first")
    )
    train = eligible[eligible["exit_day"] < DIRECTIONAL_CREDIT_TRAIN_HOLDOUT_CUTOFF].copy()
    holdout = eligible[eligible["asof"] >= DIRECTIONAL_CREDIT_TRAIN_HOLDOUT_CUTOFF].copy()
    execution_train = execution_eligible[
        execution_eligible["exit_day"] < DIRECTIONAL_CREDIT_TRAIN_HOLDOUT_CUTOFF
    ].copy()
    execution_holdout = execution_eligible[
        execution_eligible["asof"] >= DIRECTIONAL_CREDIT_TRAIN_HOLDOUT_CUTOFF
    ].copy()
    overall_metrics = _directional_credit_metrics(eligible)
    train_metrics = _directional_credit_metrics(train)
    holdout_metrics = _directional_credit_metrics(holdout)
    execution_metrics = _directional_credit_metrics(execution_eligible)
    execution_train_metrics = _directional_credit_metrics(execution_train)
    execution_holdout_metrics = _directional_credit_metrics(execution_holdout)
    execution_family_metrics: dict[str, dict[str, Any]] = {}
    for family_direction in ("Bull Put", "Bear Call"):
        family = execution_eligible[execution_eligible["direction"].eq(family_direction)].copy()
        family_train = family[family["exit_day"] < DIRECTIONAL_CREDIT_TRAIN_HOLDOUT_CUTOFF].copy()
        family_holdout = family[family["asof"] >= DIRECTIONAL_CREDIT_TRAIN_HOLDOUT_CUTOFF].copy()
        family_metrics = _directional_credit_metrics(family)
        family_train_metrics = _directional_credit_metrics(family_train)
        family_holdout_metrics = _directional_credit_metrics(family_holdout)
        family_core_pass = bool(
            family_metrics["sample_size"] >= DIRECTIONAL_CREDIT_FAMILY_MIN_SAMPLE
            and family_metrics["wilson_lower_bound"] >= DIRECTIONAL_CREDIT_FAMILY_MIN_WILSON
            and family_metrics["stress_profit_factor_10pct"]
            >= DIRECTIONAL_CREDIT_FAMILY_MIN_PROFIT_FACTOR
            and family_metrics["stress_average_pnl_10pct"] > 0
            and family_metrics["positive_months"] >= DIRECTIONAL_CREDIT_FAMILY_MIN_POSITIVE_MONTHS
        )
        if family_holdout_metrics["sample_size"] >= DIRECTIONAL_CREDIT_FAMILY_MIN_HOLDOUT_SAMPLE:
            family_holdout_pass = bool(
                family_holdout_metrics["win_rate"] >= DIRECTIONAL_CREDIT_MIN_HOLDOUT_WIN_RATE
                and family_holdout_metrics["stress_profit_factor_10pct"]
                >= DIRECTIONAL_CREDIT_FAMILY_MIN_HOLDOUT_PROFIT_FACTOR
                and family_holdout_metrics["stress_average_pnl_10pct"] > 0
                and family_holdout_metrics["positive_months"] >= MIN_POSITIVE_HOLDOUT_MONTHS
            )
            family_status = "PASS" if family_core_pass and family_holdout_pass else "FAIL"
        else:
            probationary_holdout_pass = bool(
                family_holdout_metrics["sample_size"]
                >= DIRECTIONAL_CREDIT_FAMILY_MIN_PROBATIONARY_HOLDOUT
                and family_holdout_metrics["wins"] == family_holdout_metrics["sample_size"]
                and family_holdout_metrics["stress_average_pnl_10pct"] > 0
                and family_holdout_metrics["positive_months"] >= MIN_POSITIVE_HOLDOUT_MONTHS
            )
            family_status = (
                "PROBATIONARY"
                if family_core_pass and probationary_holdout_pass
                else "FAIL"
            )
        execution_family_metrics[family_direction] = {
            "validation_status": family_status,
            **family_metrics,
            **{f"train_{key}": value for key, value in family_train_metrics.items()},
            **{f"holdout_{key}": value for key, value in family_holdout_metrics.items()},
        }
    execution_family_validation_pass = all(
        metrics.get("validation_status") in {"PASS", "PROBATIONARY"}
        for metrics in execution_family_metrics.values()
    )
    observation_end = eligible["history_observation_end"].max()
    if pd.isna(observation_end):
        observation_end = eligible["exit_day"].max()
    history_age_days = (pd.Timestamp(asof) - observation_end).days if pd.notna(observation_end) else math.inf
    activated = asof >= DIRECTIONAL_CREDIT_ACTIVATION_DATE
    history_fresh = math.isfinite(history_age_days) and 0 <= history_age_days <= DIRECTIONAL_CREDIT_MAX_HISTORY_AGE_DAYS
    validation_pass = bool(
        overall_metrics["sample_size"] >= DIRECTIONAL_CREDIT_MIN_SAMPLE
        and overall_metrics["win_rate"] >= DIRECTIONAL_CREDIT_MIN_WIN_RATE
        and overall_metrics["wilson_lower_bound"] >= DIRECTIONAL_CREDIT_MIN_WILSON
        and overall_metrics["stress_profit_factor_10pct"] >= DIRECTIONAL_CREDIT_MIN_PROFIT_FACTOR
        and overall_metrics["positive_months"] >= DIRECTIONAL_CREDIT_MIN_POSITIVE_MONTHS
        and train_metrics["sample_size"] >= DIRECTIONAL_CREDIT_MIN_TRAIN_SAMPLE
        and train_metrics["stress_profit_factor_10pct"] >= DIRECTIONAL_CREDIT_MIN_TRAIN_PROFIT_FACTOR
        and holdout_metrics["sample_size"] >= DIRECTIONAL_CREDIT_MIN_HOLDOUT_SAMPLE
        and holdout_metrics["win_rate"] >= DIRECTIONAL_CREDIT_MIN_HOLDOUT_WIN_RATE
        and holdout_metrics["stress_profit_factor_10pct"] >= DIRECTIONAL_CREDIT_MIN_HOLDOUT_PROFIT_FACTOR
    )
    execution_validation_pass = bool(
        execution_metrics["sample_size"] >= DIRECTIONAL_CREDIT_EXECUTION_MIN_SAMPLE
        and execution_metrics["wilson_lower_bound"] >= DIRECTIONAL_CREDIT_MIN_WILSON
        and execution_metrics["stress_profit_factor_10pct"] >= DIRECTIONAL_CREDIT_MIN_PROFIT_FACTOR
        and execution_metrics["stress_average_pnl_10pct"] > 0
        and execution_metrics["positive_months"] >= DIRECTIONAL_CREDIT_MIN_POSITIVE_MONTHS
        and execution_train_metrics["sample_size"] >= DIRECTIONAL_CREDIT_EXECUTION_MIN_TRAIN_SAMPLE
        and execution_train_metrics["stress_profit_factor_10pct"] >= DIRECTIONAL_CREDIT_MIN_TRAIN_PROFIT_FACTOR
        and execution_train_metrics["stress_average_pnl_10pct"] > 0
        and execution_holdout_metrics["sample_size"] >= MIN_HOLDOUT_TRADES
        and execution_holdout_metrics["win_rate"] >= DIRECTIONAL_CREDIT_MIN_HOLDOUT_WIN_RATE
        and execution_holdout_metrics["stress_profit_factor_10pct"]
        >= DIRECTIONAL_CREDIT_MIN_HOLDOUT_PROFIT_FACTOR
        and execution_holdout_metrics["stress_average_pnl_10pct"] > 0
        and execution_holdout_metrics["positive_months"] >= MIN_POSITIVE_HOLDOUT_MONTHS
        and execution_family_validation_pass
    )
    status = (
        "PASS"
        if activated and history_fresh and execution_validation_pass
        else "FAIL"
    )
    reasons: list[str] = []
    if not activated:
        reasons.append(f"lane activates on {DIRECTIONAL_CREDIT_ACTIVATION_DATE}")
    if not history_fresh:
        reasons.append(
            f"history age {history_age_days}d exceeds {DIRECTIONAL_CREDIT_MAX_HISTORY_AGE_DAYS}d"
        )
    if not execution_validation_pass:
        reasons.append(
            "supportive/matched-OI execution subgroup or a directional family failed acceptance metrics"
        )
    elif not validation_pass:
        reasons.append(
            "supportive/matched-OI execution population passed; broader contrary-OI "
            "calibration reference failed and remains non-executable"
        )
    return {
        "version": DIRECTIONAL_CREDIT_VERSION,
        "status": status,
        "reason": "; ".join(reasons) if reasons else "maturity-safe train/holdout evidence passed",
        "model_tier": "Medium" if status == "PASS" else "Unavailable",
        "reference_validation_status": "PASS" if validation_pass else "FAIL",
        "execution_validation_status": "PASS" if execution_validation_pass else "FAIL",
        "execution_family_validation_status": (
            "PASS" if execution_family_validation_pass else "FAIL"
        ),
        "execution_family_metrics": execution_family_metrics,
        "high_confidence_available": False,
        "history_path": str(history_path),
        "fill_stress_source": "recomputed_from_pnl_1x_and_entry_credit",
        "history_observation_end": str(observation_end.date()) if pd.notna(observation_end) else "",
        "history_age_days": history_age_days,
        "policy": {
            "directions": ["Bull Put", "Bear Call"],
            "flow_quality": "directional",
            "calibration_oi_states": ["supportive", "matched_unconfirmed", "contrary"],
            "execution_oi_states": sorted(DIRECTIONAL_CREDIT_EXECUTION_OI_STATES),
            "oi_status_excluded_from_execution": ["contrary", "mixed", "no_exact_match"],
            "credit_pct_width": [DIRECTIONAL_CREDIT_MIN_PCT_WIDTH, DIRECTIONAL_CREDIT_MAX_PCT_WIDTH],
            "maximum_quote_width_pct": DIRECTIONAL_CREDIT_MAX_QUOTE_WIDTH,
            "maximum_expected_move_ratio": DIRECTIONAL_CREDIT_MAX_EXPECTED_MOVE_RATIO,
            "dte": [DIRECTIONAL_CREDIT_MIN_DTE, DIRECTIONAL_CREDIT_MAX_DTE],
            "earnings_crossing": False,
            "live_authorization_cap": None,
        },
        **overall_metrics,
        **{f"train_{key}": value for key, value in train_metrics.items()},
        **{f"holdout_{key}": value for key, value in holdout_metrics.items()},
        **{f"execution_{key}": value for key, value in execution_metrics.items()},
        **{f"execution_train_{key}": value for key, value in execution_train_metrics.items()},
        **{f"execution_holdout_{key}": value for key, value in execution_holdout_metrics.items()},
    }


def _directional_credit_live_guard(
    row: pd.Series | dict[str, Any],
    summary: dict[str, Any],
) -> tuple[bool, str]:
    if summary.get("status") != "PASS":
        return False, str(summary.get("reason") or "directional credit lane unavailable")
    direction = str(row.get("direction") or "")
    if direction not in {"Bull Put", "Bear Call"}:
        return False, "not_directional_credit_family"
    family_evidence = summary.get("execution_family_metrics", {}).get(direction, {})
    family_status = str(family_evidence.get("validation_status") or "FAIL").upper()
    if family_status not in {"PASS", "PROBATIONARY"}:
        return False, f"directional_family_evidence_{family_status.lower()}"
    if str(row.get("flow_quality") or "").strip().lower() != "directional":
        return False, "contract_flow_not_directional"
    flow = safe_float(row.get("combined_flow_bias"), 0.0)
    if (direction == "Bull Put" and flow <= 0) or (direction == "Bear Call" and flow >= 0):
        return False, "aggregate_flow_not_aligned"
    oi_status = str(row.get("oi_carryover_status") or "").strip().lower()
    if oi_status not in DIRECTIONAL_CREDIT_EXECUTION_OI_STATES:
        return False, "oi_not_supportive_or_matched"
    credit_pct = safe_float(row.get("credit_pct_width"), safe_float(row.get("entry_credit_pct_width")))
    if not DIRECTIONAL_CREDIT_MIN_PCT_WIDTH <= credit_pct <= DIRECTIONAL_CREDIT_MAX_PCT_WIDTH:
        return False, "credit_outside_directional_book_band"
    quote_width = safe_float(row.get("quote_width_pct"), safe_float(row.get("entry_quote_width_pct")))
    if not math.isfinite(quote_width) or quote_width > DIRECTIONAL_CREDIT_MAX_QUOTE_WIDTH:
        return False, "quote_width_above_directional_book_limit"
    expected_move_ratio = safe_float(row.get("expected_move_ratio"))
    if not math.isfinite(expected_move_ratio) or expected_move_ratio > DIRECTIONAL_CREDIT_MAX_EXPECTED_MOVE_RATIO:
        return False, "expected_move_ratio_above_directional_book_limit"
    dte = safe_float(row.get("dte"), safe_float(row.get("entry_dte")))
    if not DIRECTIONAL_CREDIT_MIN_DTE <= dte <= DIRECTIONAL_CREDIT_MAX_DTE:
        return False, "dte_outside_directional_book_range"
    hard_raw = row.get("hard_rejects")
    hard_text = "" if hard_raw is None or (not isinstance(hard_raw, str) and pd.isna(hard_raw)) else str(hard_raw)
    hard_parts = [part.strip() for part in re.split(r"[;,]", hard_text) if part.strip()]
    remaining_hard = [
        part for part in hard_parts
        if "thin_replay_sample" not in part.lower() and "negative_replay_edge" not in part.lower()
    ]
    if remaining_hard:
        return False, "preexisting_hard_blocker:" + ";".join(remaining_hard)
    expiry = pd.to_datetime(row.get("expiry", row.get("entry_expiry")), errors="coerce")
    next_earnings = pd.to_datetime(row.get("next_earnings_dt"), errors="coerce")
    if pd.notna(expiry) and pd.notna(next_earnings) and next_earnings <= expiry:
        return False, "earnings_crosses_expiry"
    width = safe_float(row.get("width"), safe_float(row.get("entry_width")))
    natural_credit = safe_float(row.get("natural_credit"))
    mid_credit = safe_float(row.get("mid_credit"))
    if math.isfinite(width) and width > 0:
        if math.isfinite(natural_credit) and not 0 < natural_credit < width:
            return False, "invalid_natural_credit"
        if math.isfinite(mid_credit) and not 0 < mid_credit < width:
            return False, "invalid_mid_credit"
    return True, (
        f"execution-policy n={summary.get('execution_sample_size', 0)}, "
        f"Wilson={safe_float(summary.get('execution_wilson_lower_bound')):.1%}, "
        f"PF={safe_float(summary.get('execution_stress_profit_factor_10pct')):.2f}; "
        f"{direction} family {family_status}, n={family_evidence.get('sample_size', 0)}, "
        f"PF={safe_float(family_evidence.get('stress_profit_factor_10pct')):.2f}"
    )


def _directional_credit_rank(row: pd.Series | dict[str, Any]) -> float:
    direction = str(row.get("direction") or "")
    flow = safe_float(row.get("combined_flow_bias"), 0.0)
    flow_alignment = flow if direction == "Bull Put" else -flow
    credit_pct = safe_float(
        row.get("credit_pct_width"),
        safe_float(row.get("entry_credit_pct_width"), 0.0),
    )
    quote_width = safe_float(
        row.get("quote_width_pct"),
        safe_float(row.get("entry_quote_width_pct"), 1.0),
    )
    expected_move_ratio = safe_float(row.get("expected_move_ratio"), 2.0)
    return 2.0 * credit_pct - 0.8 * expected_move_ratio - 0.5 * quote_width + 0.25 * flow_alignment


def build_walk_forward_credit_model(
    *,
    asof: dt.date,
    history_path: Path = DEFAULT_WALK_FORWARD_CREDIT_HISTORY,
    directional_history_path: Path = DEFAULT_DIRECTIONAL_CREDIT_HISTORY,
) -> tuple[dict[str, Any], pd.DataFrame, CreditPayoffModel | None]:
    directional_summary = build_directional_credit_summary(
        asof=asof,
        history_path=directional_history_path,
    )
    history = load_walk_forward_credit_history(history_path)
    if history.empty:
        return (
            {
                "version": WALK_FORWARD_CREDIT_VERSION,
                "status": "FAIL",
                "reason": "maturity-safe credit history is unavailable",
                "directional_credit_lane": directional_summary,
            },
            pd.DataFrame(),
            None,
        )
    observation_end = pd.to_datetime(history.get("history_observation_end"), errors="coerce").max()
    history_age_days = (pd.Timestamp(asof) - observation_end).days if pd.notna(observation_end) else math.inf
    evidence: list[pd.DataFrame] = []
    for month in VALIDATION_MONTHS:
        start = pd.Period(month).start_time
        end = pd.Period(month).end_time
        train = history[history["expiry"] < start].copy()
        test = history[(history["asof"] >= start) & (history["asof"] <= end)].copy()
        model = _build_model(train)
        if model is None or test.empty:
            continue
        model = _attach_recent_health(model, train, start)
        test["walk_forward_credit_prediction"] = _predict(model, test)
        test["walk_forward_credit_win_probability"] = _predict_win_probability(model, test)
        policy = test.apply(lambda row: _live_credit_guard(row, model)[0], axis=1)
        test["walk_forward_credit_policy_pass"] = (
            policy
            & test["walk_forward_credit_prediction"].ge(MIN_PREDICTED_STRESS_ROR)
            & test["walk_forward_credit_win_probability"].ge(MIN_WIN_PROBABILITY)
        )
        selected = _select_one_per_day(test[test["walk_forward_credit_policy_pass"]].copy())
        if not selected.empty:
            selected["validation_month"] = month
            evidence.append(selected)
    selected_evidence = pd.concat(evidence, ignore_index=True) if evidence else pd.DataFrame()
    metrics = _validation_metrics(selected_evidence)
    holdout_evidence = (
        selected_evidence[selected_evidence.get("validation_month", pd.Series(dtype=str)).isin(HOLDOUT_MONTHS)]
        if not selected_evidence.empty
        else pd.DataFrame()
    )
    holdout_metrics = _validation_metrics(holdout_evidence)
    activated = asof >= WALK_FORWARD_CREDIT_ACTIVATION_DATE
    history_fresh = math.isfinite(history_age_days) and 0 <= history_age_days <= MAX_HISTORY_AGE_DAYS
    validation_pass = bool(
        metrics["sample_size"] >= MIN_OOS_TRADES
        and metrics["win_rate"] >= MIN_OOS_WIN_RATE
        and metrics["wilson_lower_bound"] >= MIN_OOS_WILSON_LOWER_BOUND
        and metrics["stress_profit_factor_10pct"] >= MIN_OOS_STRESS_PROFIT_FACTOR
        and metrics["stress_total_pnl_10pct"] > 0
        and metrics["positive_months"] >= MIN_POSITIVE_VALIDATION_MONTHS
        and metrics["max_drawdown_to_stressed_pnl"] <= MAX_DRAWDOWN_TO_STRESSED_PNL
        and holdout_metrics["sample_size"] >= MIN_HOLDOUT_TRADES
        and holdout_metrics["win_rate"] >= MIN_HOLDOUT_WIN_RATE
        and holdout_metrics["stress_profit_factor_10pct"] >= MIN_HOLDOUT_STRESS_PROFIT_FACTOR
        and holdout_metrics["positive_months"] >= MIN_POSITIVE_HOLDOUT_MONTHS
    )
    status = "PASS" if activated and history_fresh and validation_pass else "FAIL"
    reasons: list[str] = []
    if not activated:
        reasons.append(f"lane activates on {WALK_FORWARD_CREDIT_ACTIVATION_DATE}")
    if not history_fresh:
        reasons.append(f"history age {history_age_days}d exceeds {MAX_HISTORY_AGE_DAYS}d")
    if not validation_pass:
        reasons.append("maturity-safe walk-forward acceptance metrics failed")
    train = history[history["expiry"] < pd.Timestamp(asof)].copy()
    final_model = _build_model(train) if status == "PASS" else None
    if final_model is not None:
        final_model = _attach_recent_health(final_model, train, pd.Timestamp(asof))
    strict_model_status = status
    overall_status = (
        "PASS"
        if strict_model_status == "PASS" or directional_summary.get("status") == "PASS"
        else "FAIL"
    )
    if strict_model_status == "PASS":
        overall_reason = "maturity-safe model credit lane passed"
    elif directional_summary.get("status") == "PASS":
        overall_reason = (
            "directional Medium credit lane passed; legacy model lane unavailable: "
            + ("; ".join(reasons) if reasons else "acceptance metrics failed")
        )
    else:
        overall_reason = "; ".join(reasons) if reasons else "credit validation failed"
    summary = {
        "version": WALK_FORWARD_CREDIT_VERSION,
        "status": overall_status,
        "reason": overall_reason,
        "model_tier": "Medium" if overall_status == "PASS" else "Unavailable",
        "high_confidence_available": False,
        "strict_model_status": strict_model_status,
        "strict_model_reason": "; ".join(reasons) if reasons else "maturity-safe payoff regression passed",
        "history_path": str(history_path),
        "history_rows": int(len(history)),
        "training_rows": int(len(train)),
        "history_observation_end": str(observation_end.date()) if pd.notna(observation_end) else "",
        "history_age_days": history_age_days,
        "validation_months": list(VALIDATION_MONTHS),
        "holdout_months": list(HOLDOUT_MONTHS),
        "minimum_live_win_probability": MIN_WIN_PROBABILITY,
        "minimum_predicted_stress_return": MIN_PREDICTED_STRESS_ROR,
        "maximum_validated_dte": MAX_MODEL_DTE,
        "historical_validation_selection_max_per_day": MAX_DAILY_MEDIUM_TARGETS,
        "live_authorization_cap": None,
        **metrics,
        **{f"holdout_{key}": value for key, value in holdout_metrics.items()},
        "directional_credit_lane": directional_summary,
    }
    return summary, selected_evidence, final_model


def _live_credit_guard(
    row: pd.Series | dict[str, Any],
    model: CreditPayoffModel | None = None,
) -> tuple[bool, str]:
    objective_pass, objective_reason = _objective_credit_guard(row)
    if not objective_pass:
        return False, objective_reason
    direction = str(row.get("direction") or "")
    flow = safe_float(row.get("combined_flow_bias"), 0.0)
    if (direction == "Bull Put" and flow <= 0) or (direction == "Bear Call" and flow >= 0):
        return False, "aggregate_flow_not_aligned"
    if str(row.get("oi_carryover_status") or "").strip().lower() not in {
        "supportive",
        "matched_unconfirmed",
    }:
        return False, "oi_not_supportive"
    if model is None:
        return True, objective_reason
    recent_pass, scope, health = _recent_health_for_row(row, model)
    if not recent_pass:
        return False, f"recent_{scope}_edge_not_positive"
    return True, (
        f"recent_{scope}_edge_pass:"
        f"n={int(safe_float(health.get('sample'), 0.0))},"
        f"pf={safe_float(health.get('profit_factor'), 0.0):.2f}"
    )


def apply_walk_forward_credit_model(
    scored: pd.DataFrame,
    summary: dict[str, Any],
    model: CreditPayoffModel | None,
) -> pd.DataFrame:
    if scored is None or scored.empty:
        return scored.copy() if scored is not None else pd.DataFrame()
    out = scored.copy()
    defaults: dict[str, Any] = {
        "walk_forward_credit_version": summary.get("version", WALK_FORWARD_CREDIT_VERSION),
        "walk_forward_credit_calibration_status": summary.get("status", "FAIL"),
        "walk_forward_credit_model_tier": summary.get("model_tier", "Unavailable"),
        "walk_forward_credit_prediction": math.nan,
        "walk_forward_credit_win_probability": math.nan,
        "walk_forward_credit_confidence_score": math.nan,
        "walk_forward_credit_qualified": False,
        "walk_forward_credit_book_selected": False,
        "walk_forward_credit_policy_pass": False,
        "walk_forward_credit_reason": summary.get("reason", "model unavailable"),
        "walk_forward_credit_oos_sample_size": summary.get("sample_size", 0),
        "walk_forward_credit_oos_wins": summary.get("wins", 0),
        "walk_forward_credit_oos_win_rate": summary.get("win_rate", math.nan),
        "walk_forward_credit_bayesian_win_probability": summary.get("bayesian_win_probability", math.nan),
        "walk_forward_credit_wilson_lower_bound": summary.get("wilson_lower_bound", math.nan),
        "walk_forward_credit_stress_profit_factor_10pct": summary.get("stress_profit_factor_10pct", math.nan),
        "walk_forward_credit_stress_average_pnl_10pct": summary.get("stress_average_pnl_10pct", math.nan),
        "walk_forward_credit_positive_months": summary.get("positive_months", 0),
        "walk_forward_credit_max_drawdown_10pct": summary.get("max_drawdown_10pct", math.nan),
        "walk_forward_credit_model_qualified": False,
        "directional_credit_version": summary.get("directional_credit_lane", {}).get(
            "version", DIRECTIONAL_CREDIT_VERSION
        ),
        "directional_credit_calibration_status": summary.get("directional_credit_lane", {}).get(
            "status", "FAIL"
        ),
        "directional_credit_qualified": False,
        "directional_credit_reason": summary.get("directional_credit_lane", {}).get(
            "reason", "directional credit lane unavailable"
        ),
        "directional_credit_oos_sample_size": summary.get("directional_credit_lane", {}).get(
            "sample_size", 0
        ),
        "directional_credit_oos_win_rate": summary.get("directional_credit_lane", {}).get(
            "win_rate", math.nan
        ),
        "directional_credit_wilson_lower_bound": summary.get("directional_credit_lane", {}).get(
            "wilson_lower_bound", math.nan
        ),
        "directional_credit_stress_profit_factor_10pct": summary.get(
            "directional_credit_lane", {}
        ).get("stress_profit_factor_10pct", math.nan),
        "directional_credit_stress_average_pnl_10pct": summary.get(
            "directional_credit_lane", {}
        ).get("stress_average_pnl_10pct", math.nan),
        "directional_credit_stress_average_return_on_risk_10pct": summary.get(
            "directional_credit_lane", {}
        ).get("stress_average_return_on_risk_10pct", math.nan),
        "directional_credit_stress_average_win_risk_fraction_10pct": summary.get(
            "directional_credit_lane", {}
        ).get("stress_average_win_risk_fraction_10pct", math.nan),
        "directional_credit_stress_average_loss_risk_fraction_10pct": summary.get(
            "directional_credit_lane", {}
        ).get("stress_average_loss_risk_fraction_10pct", math.nan),
        "directional_credit_positive_months": summary.get("directional_credit_lane", {}).get(
            "positive_months", 0
        ),
        "directional_credit_holdout_sample_size": summary.get("directional_credit_lane", {}).get(
            "holdout_sample_size", 0
        ),
        "directional_credit_holdout_win_rate": summary.get("directional_credit_lane", {}).get(
            "holdout_win_rate", math.nan
        ),
        "directional_credit_holdout_stress_profit_factor_10pct": summary.get(
            "directional_credit_lane", {}
        ).get("holdout_stress_profit_factor_10pct", math.nan),
        "directional_credit_reference_validation_status": summary.get(
            "directional_credit_lane", {}
        ).get("reference_validation_status", "FAIL"),
        "directional_credit_execution_validation_status": summary.get(
            "directional_credit_lane", {}
        ).get("execution_validation_status", "FAIL"),
        "directional_credit_execution_sample_size": summary.get(
            "directional_credit_lane", {}
        ).get("execution_sample_size", 0),
        "directional_credit_execution_wilson_lower_bound": summary.get(
            "directional_credit_lane", {}
        ).get("execution_wilson_lower_bound", math.nan),
        "directional_credit_execution_stress_profit_factor_10pct": summary.get(
            "directional_credit_lane", {}
        ).get("execution_stress_profit_factor_10pct", math.nan),
        "directional_credit_execution_stress_average_pnl_10pct": summary.get(
            "directional_credit_lane", {}
        ).get("execution_stress_average_pnl_10pct", math.nan),
        "directional_credit_execution_stress_average_return_on_risk_10pct": summary.get(
            "directional_credit_lane", {}
        ).get("execution_stress_average_return_on_risk_10pct", math.nan),
        "directional_credit_execution_stress_average_win_risk_fraction_10pct": summary.get(
            "directional_credit_lane", {}
        ).get("execution_stress_average_win_risk_fraction_10pct", math.nan),
        "directional_credit_execution_stress_average_loss_risk_fraction_10pct": summary.get(
            "directional_credit_lane", {}
        ).get("execution_stress_average_loss_risk_fraction_10pct", math.nan),
        "directional_credit_execution_positive_months": summary.get(
            "directional_credit_lane", {}
        ).get("execution_positive_months", 0),
        "directional_credit_execution_holdout_sample_size": summary.get(
            "directional_credit_lane", {}
        ).get("execution_holdout_sample_size", 0),
        "directional_credit_execution_holdout_win_rate": summary.get(
            "directional_credit_lane", {}
        ).get("execution_holdout_win_rate", math.nan),
        "directional_credit_execution_holdout_stress_profit_factor_10pct": summary.get(
            "directional_credit_lane", {}
        ).get("execution_holdout_stress_profit_factor_10pct", math.nan),
        "directional_credit_family_validation_status": "FAIL",
        "directional_credit_family_sample_size": 0,
        "directional_credit_family_wilson_lower_bound": math.nan,
        "directional_credit_family_stress_profit_factor_10pct": math.nan,
        "directional_credit_family_stress_average_return_on_risk_10pct": math.nan,
        "directional_credit_family_stress_average_win_risk_fraction_10pct": math.nan,
        "directional_credit_family_stress_average_loss_risk_fraction_10pct": math.nan,
        "directional_credit_family_holdout_sample_size": 0,
        "directional_credit_family_holdout_stress_profit_factor_10pct": math.nan,
    }
    for column, value in defaults.items():
        out[column] = value
    directional_summary = summary.get("directional_credit_lane", {})
    strict_available = summary.get("strict_model_status") == "PASS" and model is not None
    directional_available = directional_summary.get("status") == "PASS"
    if not strict_available and not directional_available:
        return out
    features = out.copy()
    mapping = {
        "credit_pct_width": "entry_credit_pct_width",
        "quote_width_pct": "entry_quote_width_pct",
        "dte": "entry_dte",
        "regime_trend": "regime",
    }
    for source, target in mapping.items():
        features[target] = features.get(source)
    features["sell_leg_quote_source"] = "bot_eod_first_regular_nbbo"
    features["buy_leg_quote_source"] = "bot_eod_first_regular_nbbo"
    credit_mask = features.get("direction", pd.Series("", index=features.index)).isin({"Bull Put", "Bear Call"})
    if not credit_mask.any():
        return out
    if strict_available:
        predictions = _predict(model, features.loc[credit_mask])
        win_probabilities = _predict_win_probability(model, features.loc[credit_mask])
        out.loc[credit_mask, "walk_forward_credit_prediction"] = predictions
        out.loc[credit_mask, "walk_forward_credit_win_probability"] = win_probabilities
        out.loc[credit_mask, "walk_forward_credit_confidence_score"] = win_probabilities * 100.0
        for index in out.index[credit_mask]:
            passed, reason = _live_credit_guard(out.loc[index], model)
            prediction = safe_float(out.at[index, "walk_forward_credit_prediction"], -math.inf)
            win_probability = safe_float(out.at[index, "walk_forward_credit_win_probability"], -math.inf)
            qualified = bool(
                passed
                and prediction >= MIN_PREDICTED_STRESS_ROR
                and win_probability >= MIN_WIN_PROBABILITY
            )
            out.at[index, "walk_forward_credit_model_qualified"] = qualified
            out.at[index, "walk_forward_credit_qualified"] = qualified
            out.at[index, "walk_forward_credit_reason"] = (
                f"win probability {win_probability:.1%}; predicted 10%-stress return {prediction:.2%}; {reason}"
            )
    if directional_available:
        conservative_confidence = (
            safe_float(directional_summary.get("execution_wilson_lower_bound"), 0.0) * 100.0
        )
        for index in out.index[credit_mask]:
            direction = str(out.at[index, "direction"] or "")
            family_evidence = directional_summary.get("execution_family_metrics", {}).get(
                direction, {}
            )
            out.at[index, "directional_credit_family_validation_status"] = (
                family_evidence.get("validation_status", "FAIL")
            )
            out.at[index, "directional_credit_family_sample_size"] = family_evidence.get(
                "sample_size", 0
            )
            out.at[index, "directional_credit_family_wilson_lower_bound"] = family_evidence.get(
                "wilson_lower_bound", math.nan
            )
            out.at[index, "directional_credit_family_stress_profit_factor_10pct"] = (
                family_evidence.get("stress_profit_factor_10pct", math.nan)
            )
            out.at[index, "directional_credit_family_stress_average_return_on_risk_10pct"] = (
                family_evidence.get("stress_average_return_on_risk_10pct", math.nan)
            )
            out.at[index, "directional_credit_family_stress_average_win_risk_fraction_10pct"] = (
                family_evidence.get("stress_average_win_risk_fraction_10pct", math.nan)
            )
            out.at[index, "directional_credit_family_stress_average_loss_risk_fraction_10pct"] = (
                family_evidence.get("stress_average_loss_risk_fraction_10pct", math.nan)
            )
            out.at[index, "directional_credit_family_holdout_sample_size"] = (
                family_evidence.get("holdout_sample_size", 0)
            )
            out.at[index, "directional_credit_family_holdout_stress_profit_factor_10pct"] = (
                family_evidence.get("holdout_stress_profit_factor_10pct", math.nan)
            )
            passed, reason = _directional_credit_live_guard(out.loc[index], directional_summary)
            out.at[index, "directional_credit_qualified"] = passed
            out.at[index, "directional_credit_reason"] = reason
            if passed and not bool(out.at[index, "walk_forward_credit_model_qualified"]):
                out.at[index, "walk_forward_credit_qualified"] = True
                family_floor = safe_float(family_evidence.get("wilson_lower_bound"), 0.0)
                out.at[index, "walk_forward_credit_confidence_score"] = (
                    min(conservative_confidence / 100.0, family_floor) * 100.0
                )
                out.at[index, "walk_forward_credit_reason"] = reason + "; Medium directional-credit lane"
    qualified = out[_truthy(out["walk_forward_credit_qualified"])].copy()
    if not qualified.empty:
        # Evidence qualification and portfolio prioritization are different
        # questions.  Every row that passes the frozen live policy remains
        # execution-authorized; downstream same-ticker, risk, and portfolio
        # checks may still constrain actual orders.
        out.loc[qualified.index, "walk_forward_credit_policy_pass"] = True
        out.loc[qualified.index, "walk_forward_credit_book_selected"] = True
    return out


def write_walk_forward_credit_outputs(
    *,
    out_dir: Path,
    asof: object,
    summary: dict[str, Any],
    evidence: pd.DataFrame,
    qualified: pd.DataFrame | None = None,
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = str(asof)
    summary_path = out_dir / f"codexdaily_v4_walk_forward_credit_summary_{suffix}.json"
    evidence_path = out_dir / f"codexdaily_v4_walk_forward_credit_evidence_{suffix}.csv"
    qualified_path = out_dir / f"codexdaily_v4_walk_forward_credit_qualified_{suffix}.csv"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    evidence.to_csv(evidence_path, index=False)
    qualified_rows = qualified if qualified is not None else pd.DataFrame()
    if not qualified_rows.empty and "walk_forward_credit_qualified" in qualified_rows:
        qualified_rows = qualified_rows[_truthy(qualified_rows["walk_forward_credit_qualified"])].copy()
    else:
        qualified_rows = pd.DataFrame()
    qualified_rows.to_csv(qualified_path, index=False)
    return {
        "walk_forward_credit_summary": str(summary_path),
        "walk_forward_credit_evidence": str(evidence_path),
        "walk_forward_credit_qualified": str(qualified_path),
    }
