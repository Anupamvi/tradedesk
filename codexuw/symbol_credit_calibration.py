from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

import pandas as pd


SYMBOL_CREDIT_CALIBRATION_VERSION = "symbol-trend-credit-v1.1-exact-live-population"
SYMBOL_CREDIT_ACTIVATION_DATE = pd.Timestamp("2026-08-11")
SYMBOL_CREDIT_MIN_SAMPLE = 12
SYMBOL_CREDIT_MIN_OOS_SAMPLE = 5
SYMBOL_CREDIT_MIN_POSTACTIVATION_SAMPLE = 5
SYMBOL_CREDIT_MIN_INDEPENDENT_EPISODES = 5
SYMBOL_CREDIT_MIN_STRESS_PF = 1.25
SYMBOL_CREDIT_MIN_WILSON_LOWER = 0.65
SYMBOL_CREDIT_MAX_HISTORY_LAG_BUSINESS_DAYS = 10
SYMBOL_CREDIT_ALLOWED_OI_STATES = {"supportive", "matched_unconfirmed"}


def _number(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _profit_factor_number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if not math.isnan(result) and result >= 0 else default


def _first_number(
    row: pd.Series | dict[str, Any],
    names: tuple[str, ...],
    default: float = math.nan,
) -> float:
    for name in names:
        value = _number(row.get(name))
        if math.isfinite(value):
            return value
    return default


def _first_text(row: pd.Series | dict[str, Any], names: tuple[str, ...]) -> str:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip() and str(value).lower() != "nan":
            return str(value).strip()
    return ""


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _date_series(frame: pd.DataFrame, names: tuple[str, ...]) -> pd.Series:
    for name in names:
        if name in frame.columns:
            return pd.to_datetime(frame[name], errors="coerce").dt.normalize()
    return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")


def _ticker_series(frame: pd.DataFrame) -> pd.Series:
    for name in ("ticker", "symbol", "underlying_symbol"):
        if name in frame.columns:
            return frame[name].fillna("").astype(str).str.upper().str.strip()
    return pd.Series("", index=frame.index, dtype="object")


def _spot_series(frame: pd.DataFrame) -> pd.Series:
    for name in (
        "technical_close",
        "underlying_price",
        "entry_underlying_price",
        "stock_price",
        "spot",
        "close",
    ):
        if name in frame.columns:
            values = pd.to_numeric(frame[name], errors="coerce")
            if values.notna().any():
                return values
    return pd.Series(math.nan, index=frame.index, dtype="float64")


def _profit_factor(values: pd.Series) -> float:
    pnl = pd.to_numeric(values, errors="coerce").dropna()
    gains = float(pnl[pnl > 0].sum())
    losses = abs(float(pnl[pnl < 0].sum()))
    if losses == 0:
        return math.inf if gains > 0 else 0.0
    return gains / losses


def _max_drawdown(values: pd.Series) -> float:
    pnl = pd.to_numeric(values, errors="coerce").fillna(0.0)
    equity = pnl.cumsum()
    drawdown = equity.cummax() - equity
    return float(drawdown.max()) if not drawdown.empty else 0.0


def _business_day_lag(latest: pd.Timestamp, cutoff: pd.Timestamp) -> int:
    if pd.isna(latest):
        return 999999
    latest = pd.Timestamp(latest).normalize()
    cutoff = pd.Timestamp(cutoff).normalize()
    if latest >= cutoff:
        return 0
    return len(pd.bdate_range(latest + pd.Timedelta(days=1), cutoff))


def _assign_overlap_episodes(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype="Int64", index=frame.index)
    ordered = frame.sort_values(["_signal_day", "_exit_day", "_ticker"], kind="stable")
    episode = 0
    current_end = pd.NaT
    assignments: dict[Any, int] = {}
    for index, row in ordered.iterrows():
        signal_day = pd.Timestamp(row["_signal_day"]).normalize()
        exit_day = pd.Timestamp(row["_exit_day"]).normalize()
        if episode == 0 or pd.isna(current_end) or signal_day > current_end:
            episode += 1
            current_end = exit_day
        else:
            current_end = max(current_end, exit_day)
        assignments[index] = episode
    return pd.Series(assignments, dtype="Int64").reindex(frame.index)


def _calibration_metrics(
    selected: pd.DataFrame,
    cutoff: pd.Timestamp,
    *,
    history_fresh: bool,
) -> dict[str, Any]:
    total = len(selected)
    split_at = max(1, min(total, int(math.floor(total * 0.60)))) if total else 0
    holdout = selected.iloc[split_at:].copy() if total else selected.copy()
    wins = int(selected["stress_pnl_10pct"].gt(0).sum()) if total else 0
    raw_win = wins / total if total else 0.0
    stress_pf = _profit_factor(selected["stress_pnl_10pct"])
    stress_avg = float(selected["stress_pnl_10pct"].mean()) if total else math.nan
    holdout_pf = _profit_factor(holdout["stress_pnl_10pct"])
    holdout_avg = float(holdout["stress_pnl_10pct"].mean()) if not holdout.empty else math.nan
    monthly = (
        selected.assign(month=selected["_signal_day"].dt.to_period("M").astype(str))
        .groupby("month")["stress_pnl_10pct"]
        .agg(["count", "sum", "mean"])
        if total
        else pd.DataFrame(columns=["count", "sum", "mean"])
    )
    monthly_positive = bool(not monthly.empty and monthly["sum"].gt(0).all())
    postactivation = selected[selected["_signal_day"].ge(SYMBOL_CREDIT_ACTIVATION_DATE)]
    episodes = int(selected["_episode_id"].nunique()) if total and "_episode_id" in selected else 0
    wilson = _wilson_lower(wins, total)
    retrospective_pass = bool(
        history_fresh
        and total >= SYMBOL_CREDIT_MIN_SAMPLE
        and stress_pf >= SYMBOL_CREDIT_MIN_STRESS_PF
        and stress_avg > 0
        and len(holdout) >= SYMBOL_CREDIT_MIN_OOS_SAMPLE
        and holdout_pf >= SYMBOL_CREDIT_MIN_STRESS_PF
        and holdout_avg > 0
        and monthly_positive
        and episodes >= SYMBOL_CREDIT_MIN_INDEPENDENT_EPISODES
        and wilson >= SYMBOL_CREDIT_MIN_WILSON_LOWER
    )
    if retrospective_pass and len(postactivation) >= SYMBOL_CREDIT_MIN_POSTACTIVATION_SAMPLE:
        status = "PASS"
        reason = "chronological_holdout_and_postactivation_evidence_pass"
    elif retrospective_pass:
        status = "PROBATIONARY"
        reason = "retrospective_holdout_pass_postactivation_outcomes_pending"
    elif not history_fresh:
        status = "FAIL"
        reason = "history_freshness_failed"
    else:
        status = "FAIL"
        reason = "exact_live_population_evidence_failed"
    return {
        "status": status,
        "reason": reason,
        "sample_size": total,
        "wins": wins,
        "raw_win_rate": raw_win,
        "bayesian_win_probability": (wins + 1.0) / (total + 2.0) if total else 0.0,
        "conservative_win_probability": wilson,
        "wilson_lower_bound": wilson,
        "stress_profit_factor_10pct": stress_pf,
        "stress_average_pnl_10pct": stress_avg,
        "stress_total_pnl_10pct": float(selected["stress_pnl_10pct"].sum()) if total else 0.0,
        "stress_max_drawdown_10pct": _max_drawdown(selected["stress_pnl_10pct"]),
        "oos_sample_size": len(holdout),
        "oos_wins": int(holdout["stress_pnl_10pct"].gt(0).sum()),
        "oos_profit_factor_10pct": holdout_pf,
        "oos_average_pnl_10pct": holdout_avg,
        "postactivation_sample_size": len(postactivation),
        "independent_episode_count": episodes,
        "monthly_positive": monthly_positive,
        "monthly_results": monthly.reset_index().to_dict(orient="records"),
    }


def _wilson_lower(wins: int, total: int, z: float = 1.96) -> float:
    if total <= 0:
        return 0.0
    p = wins / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    centre = p + z2 / (2.0 * total)
    margin = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * total)) / total)
    return max(0.0, (centre - margin) / denominator)


def derive_symbol_regime(row: pd.Series | dict[str, Any]) -> tuple[str, str]:
    close = _first_number(row, ("technical_close", "underlying_price", "stock_price", "spot", "close"))
    sma20 = _first_number(row, ("technical_sma20", "sma20", "ma20"))
    sma50 = _first_number(row, ("technical_sma50", "sma50", "ma50"))
    return20 = _first_number(row, ("technical_return_20d", "return_20d", "ret_20d"))
    if not all(math.isfinite(value) for value in (close, sma20, sma50, return20)):
        return "unknown", "missing_20_50_session_symbol_trend"
    if close > sma20 > sma50 and return20 > 0:
        return "uptrend", "close_gt_sma20_gt_sma50_and_return20_positive"
    if close < sma20 < sma50 and return20 < 0:
        return "downtrend", "close_lt_sma20_lt_sma50_and_return20_negative"
    return "range", "mixed_20_50_session_symbol_trend"


def apply_symbol_regime_context(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["market_regime_trend"] = out.get("market_regime_trend", out.get("regime_trend", "unknown"))
    derived = out.apply(derive_symbol_regime, axis=1)
    out["symbol_regime_trend"] = derived.map(lambda item: item[0])
    out["symbol_regime_reason"] = derived.map(lambda item: item[1])
    out["symbol_regime_method"] = "point_in_time_close_sma20_sma50_return20"
    return out


def _assessment(
    assessor: Callable[..., Any],
    row: dict[str, Any],
    *,
    live: bool,
) -> tuple[bool, list[str]]:
    result = assessor(row, live=live)
    if isinstance(result, tuple):
        passed = bool(result[0])
        raw_reasons = result[1] if len(result) > 1 else []
    elif isinstance(result, dict):
        passed = bool(result.get("pass", result.get("passed", False)))
        raw_reasons = result.get("reasons", result.get("reason", []))
    else:
        return bool(result), []
    if isinstance(raw_reasons, str):
        reasons = [part.strip() for part in raw_reasons.replace(";", "|").split("|") if part.strip()]
    else:
        reasons = [str(part).strip() for part in (raw_reasons or []) if str(part).strip()]
    return passed, reasons


def _evidence_only_reason(reason: str) -> bool:
    lowered = reason.lower()
    return any(token in lowered for token in ("edge", "sample", "profit_factor", "payoff", "replay_guard"))


def _load_symbol_history(
    root: Path,
    cutoff: pd.Timestamp,
    screener_loader: Callable[..., pd.DataFrame],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        try:
            day = pd.Timestamp(folder.name).normalize()
        except (TypeError, ValueError):
            continue
        if day > cutoff:
            continue
        try:
            screen = screener_loader(folder, point_in_time=True)
        except TypeError:
            screen = screener_loader(folder)
        except Exception:
            continue
        if screen is None or screen.empty:
            continue
        current = pd.DataFrame(
            {"ticker": _ticker_series(screen), "day": day, "close": _spot_series(screen)}
        )
        current = current[current["ticker"].ne("") & current["close"].gt(0)]
        if not current.empty:
            rows.append(current)
    if not rows:
        return pd.DataFrame(columns=["ticker", "day", "close", "sma20", "sma50", "return20", "rv30"])
    daily = (
        pd.concat(rows, ignore_index=True)
        .groupby(["ticker", "day"], as_index=False)["close"]
        .median()
        .sort_values(["ticker", "day"])
    )
    grouped = daily.groupby("ticker", group_keys=False)["close"]
    daily["sma20"] = grouped.transform(lambda series: series.rolling(20, min_periods=20).mean())
    daily["sma50"] = grouped.transform(lambda series: series.rolling(50, min_periods=50).mean())
    daily["return20"] = grouped.transform(lambda series: series / series.shift(20) - 1.0)
    # Match codexuw.realized_vol exactly: 21 trailing returns, at least 10
    # observations, sample standard deviation, annualised by sqrt(252).
    daily["rv30"] = grouped.transform(
        lambda series: series.pct_change(fill_method=None).rolling(21, min_periods=10).std(ddof=1) * math.sqrt(252.0)
    )
    return daily


def _prepare_history(
    history: pd.DataFrame,
    symbol_history: pd.DataFrame,
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    frame = history.copy()
    if "exact_evaluated" in frame.columns:
        frame = frame[frame["exact_evaluated"].map(_truthy)]
    frame["_signal_day"] = _date_series(frame, ("asof", "signal_date", "scan_date", "date", "entry_date"))
    frame["_exit_day"] = _date_series(frame, ("exit_day", "close_date", "resolution_date"))
    frame["_ticker"] = _ticker_series(frame)
    direction = frame.get("direction", pd.Series("", index=frame.index)).fillna("").astype(str)
    frame = frame[
        direction.isin({"Bull Put", "Bear Call"})
        & frame["_signal_day"].notna()
        & frame["_exit_day"].notna()
        & frame["_exit_day"].le(cutoff)
        & frame["_ticker"].ne("")
    ].copy()
    if frame.empty or symbol_history.empty:
        return pd.DataFrame()
    return frame.merge(
        symbol_history,
        left_on=["_ticker", "_signal_day"],
        right_on=["ticker", "day"],
        how="left",
    )


def build_symbol_credit_calibration(
    root: str | Path,
    asof: str | pd.Timestamp,
    *,
    assessor: Callable[..., Any],
    screener_loader: Callable[..., pd.DataFrame],
    history_path: str | Path,
    credit_policy_version: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    cutoff = pd.Timestamp(asof).normalize()
    path = Path(history_path)
    summary: dict[str, Any] = {
        "version": SYMBOL_CREDIT_CALIBRATION_VERSION,
        "status": "FAIL",
        "reason": "history_unavailable",
        "history_path": str(path),
        "asof": cutoff.date().isoformat(),
        "sample_size": 0,
        "oos_sample_size": 0,
        "postactivation_sample_size": 0,
        "credit_policy_version": credit_policy_version,
    }
    if not path.exists():
        return summary, pd.DataFrame()
    history = pd.read_csv(path, compression="infer", low_memory=False)
    history_exit_days = _date_series(history, ("exit_day", "close_date", "resolution_date"))
    latest_history_exit = history_exit_days.max()
    history_lag = _business_day_lag(latest_history_exit, cutoff)
    history_fresh = history_lag <= SYMBOL_CREDIT_MAX_HISTORY_LAG_BUSINESS_DAYS
    summary.update(
        {
            "history_latest_exit_day": (
                latest_history_exit.date().isoformat() if pd.notna(latest_history_exit) else None
            ),
            "history_lag_business_days": history_lag,
            "history_fresh": history_fresh,
            "maximum_history_lag_business_days": SYMBOL_CREDIT_MAX_HISTORY_LAG_BUSINESS_DAYS,
        }
    )
    symbol_history = _load_symbol_history(Path(root), cutoff, screener_loader)
    frame = _prepare_history(history, symbol_history, cutoff)
    if frame.empty:
        summary["reason"] = "no_resolved_exact_credit_history_with_symbol_trend"
        return summary, frame

    accepted: list[dict[str, Any]] = []
    for _, source in frame.iterrows():
        rec = source.to_dict()
        rec["technical_close"] = _number(source.get("close"))
        rec["technical_sma20"] = _number(source.get("sma20"))
        rec["technical_sma50"] = _number(source.get("sma50"))
        rec["technical_return_20d"] = _number(source.get("return20"))
        local_rv = _number(source.get("rv30"))
        if math.isfinite(local_rv):
            # Recompute this from the same dated close history used for the
            # symbol trend. Stored replay RV values can come from an older
            # estimator and would make the calibration non-reproducible.
            rec["realized_volatility_30d"] = local_rv
        iv = _first_number(rec, ("iv30d", "implied_volatility", "iv"))
        rv = _first_number(rec, ("realized_volatility_30d", "realized_volatility", "rv30"))
        if math.isfinite(iv) and math.isfinite(rv) and rv > 0:
            rec["iv_hv_ratio"] = iv / rv
        symbol_regime, symbol_reason = derive_symbol_regime(rec)
        if symbol_regime == "unknown":
            continue
        rec["market_regime_trend"] = rec.get("regime_trend", rec.get("regime", "unknown"))
        rec["regime_trend"] = symbol_regime
        rec["regime"] = symbol_regime
        passed, reasons = _assessment(assessor, rec, live=False)
        structural_reasons = [reason for reason in reasons if not _evidence_only_reason(reason)]
        if not passed and (not reasons or structural_reasons):
            continue
        oi_status = _first_text(rec, ("oi_carryover_status", "oi_confirmation", "oi_quality")).lower()
        if oi_status not in SYMBOL_CREDIT_ALLOWED_OI_STATES:
            continue
        rec["symbol_regime_trend"] = symbol_regime
        rec["symbol_regime_reason"] = symbol_reason
        rec["symbol_credit_policy_reasons"] = "|".join(reasons)
        accepted.append(rec)

    selected = pd.DataFrame(accepted)
    if selected.empty:
        summary["reason"] = "no_current_policy_symbol_regime_credit_rows"
        return summary, selected

    selected["_oi_rank"] = 1
    selected["_flow_rank"] = selected.apply(
        lambda row: abs(_first_number(row, ("combined_flow_bias", "flow_bias"), 0.0)), axis=1
    )
    selected["_quote_rank"] = selected.apply(
        lambda row: _first_number(row, ("quote_width_pct", "relative_spread_width"), 999.0), axis=1
    )
    selected = (
        selected.sort_values(
            ["_signal_day", "_oi_rank", "_flow_rank", "_quote_rank", "_ticker"],
            ascending=[True, False, False, True, True],
            kind="stable",
        )
        .drop_duplicates("_signal_day", keep="first")
        .sort_values(["_signal_day", "_ticker"], kind="stable")
        .reset_index(drop=True)
    )
    pnl = pd.to_numeric(selected.get("pnl_1x"), errors="coerce")
    entry_price = pd.to_numeric(selected.get("entry_price"), errors="coerce").abs()
    selected["stress_pnl_10pct"] = pnl - entry_price * 100.0 * 0.10
    selected = selected[selected["stress_pnl_10pct"].notna()].copy()
    selected["_episode_id"] = _assign_overlap_episodes(selected)
    selected["_group_key"] = (
        selected["direction"].astype(str) + "|" + selected["symbol_regime_trend"].astype(str)
    )
    aggregate = _calibration_metrics(selected, cutoff, history_fresh=history_fresh)
    groups = {
        group_key: _calibration_metrics(group.copy(), cutoff, history_fresh=history_fresh)
        for group_key, group in selected.groupby("_group_key", sort=True)
    }
    group_statuses = {metrics["status"] for metrics in groups.values()}
    status = (
        "PASS"
        if "PASS" in group_statuses
        else "PROBATIONARY"
        if "PROBATIONARY" in group_statuses
        else "FAIL"
    )
    if not history_fresh:
        reason = "history_freshness_failed"
    elif status == "PASS":
        reason = "at_least_one_direction_regime_group_passed"
    elif status == "PROBATIONARY":
        reason = "retrospective_group_passed_postactivation_pending"
    else:
        reason = "no_direction_regime_group_passed_exact_live_population"
    summary.update(
        {
            "status": status,
            "reason": reason,
            "activation_date": SYMBOL_CREDIT_ACTIVATION_DATE.date().isoformat(),
            **{key: value for key, value in aggregate.items() if key not in {"status", "reason"}},
            "groups": groups,
            "minimum_postactivation_sample": SYMBOL_CREDIT_MIN_POSTACTIVATION_SAMPLE,
            "minimum_independent_episodes": SYMBOL_CREDIT_MIN_INDEPENDENT_EPISODES,
            "selection_rule": "exact_live_oi_population_then_one_per_signal_day_flow_quote_no_outcome_ranking",
            "validation_method": "retrospective_chronological_holdout_not_true_walk_forward",
        }
    )
    return summary, selected


def apply_symbol_credit_calibration(
    frame: pd.DataFrame,
    summary: dict[str, Any],
    *,
    assessor: Callable[..., Any],
) -> pd.DataFrame:
    out = apply_symbol_regime_context(frame)
    values = {
        "symbol_credit_calibration_version": summary.get("version", SYMBOL_CREDIT_CALIBRATION_VERSION),
        "symbol_credit_calibration_status": summary.get("status", "FAIL"),
        "symbol_credit_policy_pass": False,
        "symbol_credit_policy_reasons": "",
        "symbol_credit_sample_size": int(summary.get("sample_size", 0) or 0),
        "symbol_credit_oos_sample_size": int(summary.get("oos_sample_size", 0) or 0),
        "symbol_credit_postactivation_sample_size": int(summary.get("postactivation_sample_size", 0) or 0),
        "symbol_credit_independent_episode_count": int(summary.get("independent_episode_count", 0) or 0),
        "symbol_credit_raw_win_rate": _number(summary.get("raw_win_rate"), 0.0),
        "symbol_credit_bayesian_win_probability": _number(summary.get("bayesian_win_probability"), 0.0),
        "symbol_credit_wilson_lower_bound": _number(summary.get("wilson_lower_bound"), 0.0),
        "symbol_credit_stress_profit_factor_10pct": _profit_factor_number(summary.get("stress_profit_factor_10pct")),
        "symbol_credit_stress_average_pnl_10pct": _number(summary.get("stress_average_pnl_10pct")),
        "symbol_credit_oos_profit_factor_10pct": _profit_factor_number(summary.get("oos_profit_factor_10pct")),
        "symbol_credit_oos_average_pnl_10pct": _number(summary.get("oos_average_pnl_10pct")),
    }
    for column, value in values.items():
        out[column] = value
    out["symbol_credit_group_key"] = ""
    groups = summary.get("groups", {}) if isinstance(summary.get("groups"), dict) else {}
    for index, row in out.iterrows():
        direction = _first_text(row, ("direction",))
        symbol_regime = _first_text(row, ("symbol_regime_trend",)).lower()
        if direction not in {"Bull Put", "Bear Call"} or symbol_regime == "unknown":
            continue
        group_key = f"{direction}|{symbol_regime}"
        group = groups.get(group_key)
        if not isinstance(group, dict):
            continue
        out.at[index, "symbol_credit_group_key"] = group_key
        out.at[index, "symbol_credit_calibration_status"] = group.get("status", "FAIL")
        out.at[index, "symbol_credit_sample_size"] = int(group.get("sample_size", 0) or 0)
        out.at[index, "symbol_credit_oos_sample_size"] = int(group.get("oos_sample_size", 0) or 0)
        out.at[index, "symbol_credit_postactivation_sample_size"] = int(group.get("postactivation_sample_size", 0) or 0)
        out.at[index, "symbol_credit_independent_episode_count"] = int(group.get("independent_episode_count", 0) or 0)
        out.at[index, "symbol_credit_raw_win_rate"] = _number(group.get("raw_win_rate"), 0.0)
        out.at[index, "symbol_credit_bayesian_win_probability"] = _number(group.get("bayesian_win_probability"), 0.0)
        out.at[index, "symbol_credit_wilson_lower_bound"] = _number(group.get("wilson_lower_bound"), 0.0)
        out.at[index, "symbol_credit_stress_profit_factor_10pct"] = _profit_factor_number(group.get("stress_profit_factor_10pct"))
        out.at[index, "symbol_credit_stress_average_pnl_10pct"] = _number(group.get("stress_average_pnl_10pct"))
        out.at[index, "symbol_credit_oos_profit_factor_10pct"] = _profit_factor_number(group.get("oos_profit_factor_10pct"))
        out.at[index, "symbol_credit_oos_average_pnl_10pct"] = _number(group.get("oos_average_pnl_10pct"))
        if group.get("status") != "PASS":
            continue
        rec = row.to_dict()
        rec["market_regime_trend"] = rec.get("regime_trend", rec.get("regime", "unknown"))
        rec["regime_trend"] = symbol_regime
        rec["regime"] = symbol_regime
        rec["replay_guard_pass"] = True
        rec["edge_sample_size"] = int(group.get("sample_size", 0) or 0)
        group_pf = _profit_factor_number(group.get("stress_profit_factor_10pct"))
        rec["edge_profit_factor"] = 999.0 if math.isinf(group_pf) else group_pf
        rec["edge_avg_pnl"] = _number(group.get("stress_average_pnl_10pct"))
        passed, reasons = _assessment(assessor, rec, live=True)
        policy_pass = bool(passed)
        out.at[index, "symbol_credit_policy_pass"] = policy_pass
        out.at[index, "symbol_credit_policy_reasons"] = "|".join(reasons)
        if not policy_pass:
            continue
        out.at[index, "payoff_route_level"] = "symbol_credit"
        out.at[index, "payoff_route_key"] = f"symbol_credit::Credit|{direction}|{symbol_regime}"
        out.at[index, "payoff_calibration_status"] = group["status"]
        out.at[index, "payoff_minimum_sample_required"] = SYMBOL_CREDIT_MIN_SAMPLE
        out.at[index, "payoff_sample_size"] = int(group.get("sample_size", 0) or 0)
        out.at[index, "payoff_stress_10_win_rate"] = _number(group.get("raw_win_rate"), 0.0)
        out.at[index, "payoff_stress_10_profit_factor"] = group_pf
        out.at[index, "payoff_stress_10_average_pnl"] = _number(group.get("stress_average_pnl_10pct"))
        out.at[index, "payoff_walk_forward_oos_sample"] = int(group.get("oos_sample_size", 0) or 0)
        out.at[index, "payoff_walk_forward_oos_profit_factor"] = _profit_factor_number(group.get("oos_profit_factor_10pct"))
        out.at[index, "payoff_walk_forward_oos_average_pnl"] = _number(group.get("oos_average_pnl_10pct"))
        out.at[index, "payoff_post_activation_oos_sample"] = int(group.get("postactivation_sample_size", 0) or 0)
    return out


def write_symbol_credit_calibration_outputs(
    *,
    out_dir: Path,
    asof: object,
    summary: dict[str, Any],
    evidence: pd.DataFrame,
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"codexdaily_v4_symbol_credit_calibration_{asof}.json"
    evidence_path = out_dir / f"codexdaily_v4_symbol_credit_evidence_{asof}.csv"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    evidence.to_csv(evidence_path, index=False)
    return {
        "symbol_credit_calibration": str(summary_path),
        "symbol_credit_evidence": str(evidence_path),
    }
