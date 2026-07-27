from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from .edge_model import EDGE_HISTORY_NAMESPACE, PACKAGED_HISTORY_DIR


CONFIDENCE_CALIBRATION_VERSION = "confidence-walk-forward-v2.0-policy-base"
# Single source of truth: derived from the edge-history namespace so a refresh
# never has to be applied in more than one place.
DEFAULT_EDGE_HISTORY_PATH = PACKAGED_HISTORY_DIR / f"{EDGE_HISTORY_NAMESPACE}.csv.gz"
MIN_PRIOR_SAMPLE = 12
MIN_CALIBRATION_PREDICTIONS = 30
MAX_CALIBRATION_GAP = 0.10
HIGH_MIN_PROBABILITY = 0.60
HIGH_MIN_SAMPLE = 20
HIGH_MIN_LOWER_BOUND = 0.50
WILSON_Z_90_ONE_SIDED = 1.2815515655446004


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _number(value: Any, default: float = math.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _strategy_family(row: pd.Series | dict[str, Any]) -> str:
    kind = str(row.get("strategy_kind", "")).strip().lower()
    if kind == "credit":
        return "Credit"
    if kind == "debit":
        return "Debit"

    strategy = str(row.get("strategy", "")).strip().lower()
    if "credit" in strategy:
        return "Credit"
    if "debit" in strategy:
        return "Debit"

    direction = str(row.get("direction", "")).strip().lower()
    if direction in {"bull put", "bear call"}:
        return "Credit"
    if direction in {"bull call", "bear put"}:
        return "Debit"
    return "Unknown"


def wilson_lower_bound(
    wins: float,
    sample: float,
    *,
    z: float = WILSON_Z_90_ONE_SIDED,
) -> float:
    n = _number(sample, 0.0)
    w = _number(wins, 0.0)
    if n <= 0:
        return math.nan
    p = min(1.0, max(0.0, w / n))
    z2 = z * z
    center = p + z2 / (2.0 * n)
    adjustment = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n)
    return max(0.0, (center - adjustment) / (1.0 + z2 / n))


def _eligible_history(
    history: pd.DataFrame,
    *,
    asof: object | None = None,
) -> pd.DataFrame:
    """Return exact, point-in-time replay-valid outcomes for calibration.

    Final approval is intentionally not an eligibility requirement. Requiring
    decision_pass caused the probability model to learn from only four
    walk-forward predictions even though exact replay-valid outcomes existed.
    The execution book remains independently strict.
    """

    if history is None or history.empty:
        return pd.DataFrame()

    out = history.copy()
    if "exact_evaluated" in out.columns:
        out = out[out["exact_evaluated"].map(_truthy)]
    if "replay_guard_pass" in out.columns:
        out = out[out["replay_guard_pass"].map(_truthy)]

    asof_col = "asof" if "asof" in out.columns else "eval_day"
    exit_col = "exit_day" if "exit_day" in out.columns else "eval_day"
    outcome_col = "exact_win" if "exact_win" in out.columns else "win"
    if asof_col not in out.columns or exit_col not in out.columns:
        return pd.DataFrame()

    out["_asof_dt"] = pd.to_datetime(out[asof_col], errors="coerce")
    out["_exit_dt"] = pd.to_datetime(out[exit_col], errors="coerce")
    if outcome_col in out.columns:
        out["_actual_outcome"] = pd.to_numeric(out[outcome_col], errors="coerce")
    elif "pnl_1x" in out.columns:
        pnl = pd.to_numeric(out["pnl_1x"], errors="coerce")
        out["_actual_outcome"] = pnl.where(pnl.isna(), (pnl > 0).astype(float))
    else:
        return pd.DataFrame()
    out = out[
        out["_asof_dt"].notna()
        & out["_exit_dt"].notna()
        & (out["_exit_dt"] >= out["_asof_dt"])
        & out["_actual_outcome"].isin([0.0, 1.0])
    ]

    if asof is not None:
        cutoff = pd.to_datetime(asof, errors="coerce")
        if pd.notna(cutoff):
            out = out[(out["_asof_dt"] <= cutoff) & (out["_exit_dt"] <= cutoff)]

    out["_strategy_family"] = out.apply(_strategy_family, axis=1)
    out = out[out["_strategy_family"].isin({"Credit", "Debit"})]
    return out.sort_values(["_asof_dt", "_exit_dt"]).reset_index(drop=True)


def _smoothed_probability(wins: float, sample: float) -> float:
    return (_number(wins, 0.0) + 1.0) / (_number(sample, 0.0) + 2.0)


def _brier(probabilities: pd.Series, outcomes: pd.Series) -> float:
    if probabilities.empty:
        return math.nan
    return float(((probabilities.astype(float) - outcomes.astype(float)) ** 2).mean())


def _validation_status(
    prediction_count: int,
    brier_score: float,
    calibration_gap: float,
    *,
    min_predictions: int,
    max_calibration_gap: float,
) -> tuple[str, str]:
    if prediction_count < min_predictions:
        return "INSUFFICIENT", f"walk-forward predictions {prediction_count} < {min_predictions}"
    if not math.isfinite(brier_score) or brier_score >= 0.25:
        return "FAIL", "prior-only policy probability does not beat the no-skill 50% Brier benchmark"
    if not math.isfinite(calibration_gap) or calibration_gap > max_calibration_gap:
        return "FAIL", f"calibration gap {calibration_gap:.3f} > {max_calibration_gap:.3f}"
    return "PASS", "prior-only policy probability beats the no-skill benchmark with acceptable calibration gap"


def _current_estimates(eligible: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    families: dict[str, Any] = {}
    groups: dict[str, Any] = {}
    for family, frame in eligible.groupby("_strategy_family", dropna=False):
        sample = int(len(frame))
        wins = float(frame["_actual_outcome"].sum())
        families[str(family)] = {
            "sample_size": sample,
            "wins": wins,
            "probability": _smoothed_probability(wins, sample),
            "probability_lower_bound": wilson_lower_bound(wins, sample),
        }
    for keys, frame in eligible.groupby(
        ["_strategy_family", "direction", "regime"],
        dropna=False,
    ):
        family, direction, regime = (str(value) for value in keys)
        sample = int(len(frame))
        wins = float(frame["_actual_outcome"].sum())
        groups[f"{family}|{direction}|{regime}"] = {
            "strategy_family": family,
            "direction": direction,
            "regime": regime,
            "sample_size": sample,
            "wins": wins,
            "probability": _smoothed_probability(wins, sample),
            "probability_lower_bound": wilson_lower_bound(wins, sample),
        }
    return families, groups


def build_walk_forward_calibration(
    history: pd.DataFrame,
    *,
    asof: object | None = None,
    min_prior_sample: int = MIN_PRIOR_SAMPLE,
    min_predictions: int = MIN_CALIBRATION_PREDICTIONS,
    max_calibration_gap: float = MAX_CALIBRATION_GAP,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    eligible = _eligible_history(history, asof=asof)
    group_records: list[dict[str, Any]] = []
    family_records: list[dict[str, Any]] = []

    for _, row in eligible.iterrows():
        prior = eligible[eligible["_exit_dt"] < row["_asof_dt"]]
        family = str(row["_strategy_family"])
        prior_family = prior[prior["_strategy_family"] == family]
        if len(prior_family) >= min_prior_sample:
            family_wins = float(prior_family["_actual_outcome"].sum())
            family_records.append(
                {
                    "asof": row["_asof_dt"].date().isoformat(),
                    "ticker": str(row.get("ticker", "")),
                    "strategy_family": family,
                    "prior_sample_size": int(len(prior_family)),
                    "prior_wins": family_wins,
                    "predicted_probability": _smoothed_probability(family_wins, len(prior_family)),
                    "actual_outcome": float(row["_actual_outcome"]),
                }
            )

        prior_group = prior[
            (prior["_strategy_family"] == family)
            & (prior["direction"].astype(str) == str(row.get("direction", "")))
            & (prior["regime"].astype(str) == str(row.get("regime", "")))
        ]
        if len(prior_group) < min_prior_sample:
            continue
        wins = float(prior_group["_actual_outcome"].sum())
        group_records.append(
            {
                "asof": row["_asof_dt"].date().isoformat(),
                "ticker": str(row.get("ticker", "")),
                "direction": str(row.get("direction", "")),
                "regime": str(row.get("regime", "")),
                "strategy_family": family,
                "prior_sample_size": int(len(prior_group)),
                "prior_wins": wins,
                "predicted_probability": _smoothed_probability(wins, len(prior_group)),
                "family_baseline_probability": (
                    _smoothed_probability(
                        float(prior_family["_actual_outcome"].sum()),
                        len(prior_family),
                    )
                    if len(prior_family) >= min_prior_sample
                    else math.nan
                ),
                "probability_lower_bound": wilson_lower_bound(wins, len(prior_group)),
                "actual_outcome": float(row["_actual_outcome"]),
            }
        )

    detail = pd.DataFrame(group_records)
    family_detail = pd.DataFrame(family_records)
    current_families, current_groups = _current_estimates(eligible)

    family_validation: dict[str, Any] = {}
    for family in ("Credit", "Debit"):
        subset = (
            family_detail[family_detail["strategy_family"] == family]
            if not family_detail.empty
            else pd.DataFrame()
        )
        prediction_count = int(len(subset))
        brier_score = (
            _brier(subset["predicted_probability"], subset["actual_outcome"])
            if prediction_count
            else math.nan
        )
        mean_probability = (
            float(subset["predicted_probability"].mean()) if prediction_count else math.nan
        )
        actual_win_rate = (
            float(subset["actual_outcome"].mean()) if prediction_count else math.nan
        )
        calibration_gap = (
            abs(mean_probability - actual_win_rate) if prediction_count else math.nan
        )
        status, reason = _validation_status(
            prediction_count,
            brier_score,
            calibration_gap,
            min_predictions=min_predictions,
            max_calibration_gap=max_calibration_gap,
        )

        group_subset = (
            detail[detail["strategy_family"] == family]
            if not detail.empty
            else pd.DataFrame()
        )
        group_brier = (
            _brier(group_subset["predicted_probability"], group_subset["actual_outcome"])
            if not group_subset.empty
            else math.nan
        )
        group_family_baseline = (
            _brier(
                group_subset["family_baseline_probability"],
                group_subset["actual_outcome"],
            )
            if not group_subset.empty
            else math.nan
        )

        family_validation[family] = {
            "status": status,
            "reason": reason,
            "prediction_count": prediction_count,
            "eligible_history_rows": int((eligible["_strategy_family"] == family).sum()),
            "brier_score": brier_score,
            "baseline_brier_score": 0.25,
            "mean_predicted_probability": mean_probability,
            "actual_win_rate": actual_win_rate,
            "calibration_gap": calibration_gap,
            "group_prediction_count": int(len(group_subset)),
            "group_brier_score": group_brier,
            "group_family_baseline_brier_score": group_family_baseline,
            "group_model_beats_family_base": bool(
                math.isfinite(group_brier)
                and math.isfinite(group_family_baseline)
                and group_brier < group_family_baseline
            ),
        }

    prediction_count = int(len(detail))
    brier_score = (
        _brier(detail["predicted_probability"], detail["actual_outcome"])
        if prediction_count
        else math.nan
    )
    mean_probability = (
        float(detail["predicted_probability"].mean()) if prediction_count else math.nan
    )
    actual_win_rate = float(detail["actual_outcome"].mean()) if prediction_count else math.nan
    calibration_gap = (
        abs(mean_probability - actual_win_rate) if prediction_count else math.nan
    )
    status, reason = _validation_status(
        prediction_count,
        brier_score,
        calibration_gap,
        min_predictions=min_predictions,
        max_calibration_gap=max_calibration_gap,
    )

    actionable_rows = 0
    if "decision_pass" in eligible.columns:
        actionable_rows = int(eligible["decision_pass"].map(_truthy).sum())

    high_available = False
    for family, validation in family_validation.items():
        estimate = current_families.get(family, {})
        high_available = high_available or bool(
            validation.get("status") == "PASS"
            and _number(estimate.get("sample_size"), 0.0) >= HIGH_MIN_SAMPLE
            and _number(estimate.get("probability"), 0.0) >= HIGH_MIN_PROBABILITY
            and _number(estimate.get("probability_lower_bound"), 0.0)
            >= HIGH_MIN_LOWER_BOUND
        )

    summary: dict[str, Any] = {
        "version": CONFIDENCE_CALIBRATION_VERSION,
        "status": status,
        "reason": reason,
        "calibration_scope": "exact_replay_validated_candidates_not_final_approval",
        "eligible_history_rows": int(len(eligible)),
        "actionable_history_rows": actionable_rows,
        "prediction_count": prediction_count,
        "minimum_prior_sample": int(min_prior_sample),
        "minimum_predictions": int(min_predictions),
        "brier_score": brier_score,
        "baseline_brier_score": 0.25,
        "mean_predicted_probability": mean_probability,
        "actual_win_rate": actual_win_rate,
        "calibration_gap": calibration_gap,
        "family_validation": family_validation,
        "current_family_estimates": current_families,
        "current_group_estimates": current_groups,
        "high_confidence_available": high_available,
        "asof": str(asof) if asof is not None else "",
    }
    return detail, summary


def build_default_walk_forward_calibration(
    *,
    asof: object | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not DEFAULT_EDGE_HISTORY_PATH.exists():
        return pd.DataFrame(), {
            "version": CONFIDENCE_CALIBRATION_VERSION,
            "status": "MISSING",
            "reason": f"history file missing: {DEFAULT_EDGE_HISTORY_PATH}",
            "eligible_history_rows": 0,
            "prediction_count": 0,
            "high_confidence_available": False,
            "asof": str(asof) if asof is not None else "",
        }
    history = pd.read_csv(DEFAULT_EDGE_HISTORY_PATH, low_memory=False)
    return build_walk_forward_calibration(history, asof=asof)


def apply_confidence_calibration(
    scored: pd.DataFrame,
    summary: dict[str, Any],
) -> pd.DataFrame:
    out = scored.copy()
    family_validation = summary.get("family_validation", {}) or {}
    family_estimates = summary.get("current_family_estimates", {}) or {}
    group_estimates = summary.get("current_group_estimates", {}) or {}

    probabilities: list[float] = []
    lower_bounds: list[float] = []
    samples: list[float] = []
    wins_values: list[float] = []
    statuses: list[str] = []
    tiers: list[str] = []
    labels: list[str] = []
    sources: list[str] = []
    briers: list[float] = []
    baselines: list[float] = []

    for _, row in out.iterrows():
        family = _strategy_family(row)
        validation = family_validation.get(family, {})
        family_estimate = family_estimates.get(family, {})
        group_key = f"{family}|{str(row.get('direction', ''))}|{str(row.get('regime', ''))}"
        group_estimate = group_estimates.get(group_key, {})

        use_group = bool(
            validation.get("group_model_beats_family_base")
            and _number(group_estimate.get("sample_size"), 0.0) >= MIN_PRIOR_SAMPLE
        )
        estimate = group_estimate if use_group else family_estimate
        source = "direction_regime" if use_group else "strategy_family"
        status = str(validation.get("status", summary.get("status", "INSUFFICIENT")))
        probability = _number(estimate.get("probability"))
        lower_bound = _number(estimate.get("probability_lower_bound"))
        sample = _number(estimate.get("sample_size"))
        wins = _number(estimate.get("wins"))
        legacy_summary = not family_validation
        if legacy_summary:
            sample = _number(row.get("edge_sample_size"))
            probability = _number(
                row.get("edge_effective_win_rate", row.get("edge_win_rate"))
            )
            wins = probability * sample if math.isfinite(probability) and math.isfinite(sample) else math.nan
            lower_bound = wilson_lower_bound(wins, sample)
        validated = status == "PASS" and math.isfinite(probability)

        probabilities.append(probability)
        lower_bounds.append(lower_bound)
        samples.append(sample)
        wins_values.append(wins)
        statuses.append(status)
        if validated:
            tier = f"{source}_validated"
        elif legacy_summary and math.isfinite(probability) and probability >= 0.50:
            tier = "medium"
        else:
            tier = "descriptive_only"
        tiers.append(tier)
        labels.append("walk_forward_validated" if validated else "descriptive_only")
        sources.append(source if math.isfinite(probability) else "unavailable")
        briers.append(_number(validation.get("brier_score", summary.get("brier_score"))))
        baselines.append(
            _number(
                validation.get(
                    "baseline_brier_score",
                    summary.get("baseline_brier_score", 0.25),
                ),
                0.25,
            )
        )

    out["confidence_probability"] = probabilities
    out["confidence_probability_lower_bound"] = lower_bounds
    out["confidence_calibration_sample_size"] = samples
    out["confidence_calibration_wins"] = wins_values
    out["confidence_calibration_status"] = statuses
    out["confidence_model_tier"] = tiers
    out["confidence_probability_label"] = labels
    out["confidence_probability_source"] = sources
    out["confidence_calibration_brier"] = briers
    out["confidence_calibration_baseline_brier"] = baselines
    return out


def confidence_high_ready(row: pd.Series | dict[str, Any]) -> bool:
    return bool(
        str(row.get("confidence_calibration_status", "")).upper() == "PASS"
        and str(row.get("confidence_model_tier", "")) != "descriptive_only"
        and _number(row.get("confidence_calibration_sample_size"), 0.0) >= HIGH_MIN_SAMPLE
        and _number(row.get("confidence_probability"), 0.0) >= HIGH_MIN_PROBABILITY
        and _number(row.get("confidence_probability_lower_bound"), 0.0)
        >= HIGH_MIN_LOWER_BOUND
    )
