from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import debit_walkforward_shadow as debit_model


DEBIT_SHADOW_POLICY_VERSION = "debit-walkforward-shadow-v2-prospective-20260816"
DEBIT_TRAINING_FILE = "codexdaily_v4_debit_shadow_training_v2_2026-08-14.csv.gz"
DEBIT_LEDGER_NAME = "codexdaily_v4_debit_shadow_ledger.csv"
MINIMUM_PRIOR_ROWS = 100
MINIMUM_WIN_PROBABILITY = 0.55
PROFIT_TAKE_PCT = 1.0
ENTRY_STRESS_PCT = 0.10

DEBIT_LEDGER_COLUMNS = [
    "policy_version",
    "signal_date",
    "entry_day",
    "generated_at_utc",
    "evidence_timing",
    "ticker",
    "sector",
    "strategy",
    "direction",
    "expiry",
    "buy_leg",
    "sell_leg",
    "buy_strike",
    "sell_strike",
    "entry_debit",
    "entry_width",
    "target_exit_value",
    "predicted_win_probability",
    "predicted_ev_payoff_correct",
    "prior_sample_size",
    "model_training_through",
    "feature_parity",
    "shadow_only",
    "execution_authorized",
    "no_order_placement",
    "outcome_status",
    "outcome_last_checked",
    "exit_day",
    "exit_reason",
    "exit_value",
    "pnl_1x",
    "return_on_risk",
    "outcome_note",
]

RESOLVED_STATES = {"RESOLVED_WIN", "RESOLVED_LOSS", "RESOLVED_FLAT"}


def _series(frame: pd.DataFrame, name: str, default: Any = np.nan) -> pd.Series:
    if name in frame.columns:
        return frame[name]
    return pd.Series(default, index=frame.index)


def _numeric(frame: pd.DataFrame, name: str, default: Any = np.nan) -> pd.Series:
    return pd.to_numeric(_series(frame, name, default), errors="coerce")


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _first_numeric(frame: pd.DataFrame, names: list[str]) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    for name in names:
        result = result.combine_first(_numeric(frame, name))
    return result


def _first_text(frame: pd.DataFrame, names: list[str]) -> pd.Series:
    result = pd.Series("", index=frame.index, dtype=object)
    for name in names:
        values = _series(frame, name, "").fillna("").astype(str).str.strip()
        result = result.where(result.astype(str).str.len().gt(0), values)
    return result


def _technical_frame(scored: pd.DataFrame, asof: dt.date) -> pd.DataFrame:
    technical = pd.DataFrame(
        {
            "date": pd.Timestamp(asof),
            "ticker": _series(scored, "ticker", "").astype(str).str.upper(),
            "close": _numeric(scored, "technical_close"),
            "rsi14": _numeric(scored, "rsi14"),
            "sma5": _numeric(scored, "sma5"),
            "sma20": _numeric(scored, "sma20"),
            "return5": _first_numeric(scored, ["return_5d", "return5"]),
            "return20": _first_numeric(scored, ["return_20d", "return20"]),
            "relative_strength5": _first_numeric(scored, ["relative_strength_5d_vs_spy", "relative_strength5"]),
            "relative_strength20": _first_numeric(scored, ["relative_strength_20d_vs_spy", "relative_strength20"]),
            "atr14_pct": _numeric(scored, "atr14_pct"),
            "vwap20": _first_numeric(scored, ["anchored_vwap_20d", "vwap20"]),
            "volume_ratio": _first_numeric(scored, ["volume_ratio", "stock_volume_ratio"]),
        }
    )
    return technical.drop_duplicates(["date", "ticker"], keep="first").reset_index(drop=True)


def prepare_live_debit_candidates(scored: pd.DataFrame, *, asof: dt.date) -> pd.DataFrame:
    """Create V2 feature rows from unresolved live candidates without using outcomes."""
    if scored is None or scored.empty:
        return pd.DataFrame()
    source = scored[
        _series(scored, "strategy", "").astype(str).isin(
            {"Bull Call Debit Spread", "Bear Put Debit Spread"}
        )
    ].copy()
    if source.empty:
        return source
    source = source.reset_index(drop=True)
    source["_shadow_row_id"] = np.arange(len(source), dtype=int)
    candidate = source.copy()
    asof_ts = pd.Timestamp(asof).normalize()
    quote_day = pd.to_datetime(_series(source, "quote_observation_date", ""), errors="coerce").dt.normalize()

    candidate["asof"] = asof_ts
    candidate["signal_day"] = asof_ts
    candidate["entry_day"] = quote_day
    candidate["exit_day"] = quote_day
    candidate["entry_debit"] = _first_numeric(source, ["natural_debit", "debit", "mid_debit"])
    candidate["entry_mid_debit"] = _numeric(source, "mid_debit")
    candidate["entry_natural_debit"] = _numeric(source, "natural_debit")
    candidate["entry_price"] = candidate["entry_debit"]
    candidate["entry_width"] = _first_numeric(source, ["spread_width", "preferred_width"])
    candidate["entry_quote_width_pct"] = _numeric(source, "quote_width_pct")
    candidate["entry_dte"] = _numeric(source, "dte")
    candidate["stock_price_entry"] = _first_numeric(source, ["stock_price_live", "stock_price_eod"])
    candidate["regime"] = _first_text(source, ["symbol_regime_trend", "regime_trend", "market_regime_trend"])
    candidate["reward_risk"] = _numeric(source, "reward_risk")
    candidate["breakeven"] = _numeric(source, "breakeven")
    candidate["long_leg_eod"] = _first_text(source, ["long_leg", "long_leg_eod"])
    candidate["short_leg_eod"] = _first_text(source, ["short_leg", "short_leg_eod"])
    candidate["long_strike_eod"] = _first_numeric(source, ["long_strike", "long_strike_eod"])
    candidate["short_strike_eod"] = _first_numeric(source, ["short_strike", "short_strike_eod"])
    regular = _series(source, "regular_session_quote", True).map(_truthy)
    candidate["exact_fillable"] = _series(source, "live_status", "").astype(str).eq("PASS") & regular

    # prepare_history owns the feature formulas. Synthetic outcome fields only
    # satisfy its integrity preconditions and are never model inputs or labels.
    candidate["exact_evaluated"] = True
    candidate["exit_value"] = candidate["entry_debit"]
    candidate["exact_win"] = False
    candidate["pnl_1x"] = 0.0
    prepared = debit_model.prepare_history(candidate, _technical_frame(source, asof))
    if prepared.empty:
        return prepared
    be_map = source.set_index("_shadow_row_id")["breakeven_expected_move_ratio"] if "breakeven_expected_move_ratio" in source else pd.Series(dtype=float)
    if not be_map.empty:
        prepared["breakeven_sigma"] = pd.to_numeric(prepared["_shadow_row_id"].map(be_map), errors="coerce")
    feature_names = [
        "rsi14",
        "sma20",
        "return20",
        "relative_strength20",
        "atr14_pct",
        "vwap20",
        "iv_hv_ratio",
        "combined_flow_bias",
    ]
    parity_count = sum(
        int(
            pd.to_numeric(
                prepared[name] if name in prepared.columns else pd.Series(np.nan, index=prepared.index),
                errors="coerce",
            ).notna().any()
        )
        for name in feature_names
    )
    prepared["feature_parity"] = f"{parity_count}/{len(feature_names)}_daily_features"
    prepared["shadow_live_feature_prep"] = "outcome_fields_synthetic_not_model_inputs"
    guard = debit_model.candidate_guard(prepared)
    return prepared.loc[guard].copy().reset_index(drop=True)


def load_training_history(root: Path, *, asof: dt.date) -> pd.DataFrame:
    path = root / "codexuw" / "history" / DEBIT_TRAINING_FILE
    if not path.exists():
        raise FileNotFoundError(f"Frozen debit shadow training file missing: {path}")
    training = pd.read_csv(path, low_memory=False)
    training["asof"] = pd.to_datetime(training.get("asof"), errors="coerce")
    training["entry_day"] = pd.to_datetime(training.get("entry_day"), errors="coerce")
    training["exit_day"] = pd.to_datetime(training.get("exit_day"), errors="coerce")
    cutoff = pd.Timestamp(asof).normalize()
    training = training[training["exit_day"].lt(cutoff)].copy()
    if not training.empty:
        training = training.loc[debit_model.learning_guard(training)].copy()
    return training.reset_index(drop=True)


def score_debit_shadow(
    scored: pd.DataFrame,
    *,
    root: Path,
    asof: dt.date,
    minimum_prior: int = MINIMUM_PRIOR_ROWS,
    threshold: float = MINIMUM_WIN_PROBABILITY,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    candidates = prepare_live_debit_candidates(scored, asof=asof)
    training = load_training_history(root, asof=asof)
    training_through = training["exit_day"].max() if not training.empty else pd.NaT
    base_summary = {
        "policy_version": DEBIT_SHADOW_POLICY_VERSION,
        "shadow_only": True,
        "execution_authorized": False,
        "no_order_placement": True,
        "training_rows": int(len(training)),
        "training_through": str(training_through.date()) if pd.notna(training_through) else "",
        "live_guard_rows": int(len(candidates)),
        "minimum_prior_rows": int(minimum_prior),
        "minimum_win_probability": float(threshold),
        "entry_stress_pct": ENTRY_STRESS_PCT,
        "profit_take_pct": PROFIT_TAKE_PCT,
    }
    if len(training) < minimum_prior or candidates.empty:
        reason = "insufficient_maturity_safe_prior" if len(training) < minimum_prior else "no_live_candidates_passed_v2_guard"
        base_summary.update({"status": "RESEARCH_ONLY", "reason": reason, "selected_rows": 0})
        return candidates, candidates.head(0).copy(), base_summary

    model = debit_model._model().fit(training, training["stress_win_10pct"].astype(int))
    evaluated = candidates.copy()
    evaluated["predicted_win_probability"] = debit_model._predict_probabilities(model, evaluated)
    payoff = debit_model.payoff_aware_expected_value(
        evaluated,
        evaluated["predicted_win_probability"],
        profit_take_pct=PROFIT_TAKE_PCT,
        entry_stress_pct=ENTRY_STRESS_PCT,
    )
    for column in payoff.columns:
        evaluated[column] = payoff[column]
    evaluated["prior_sample_size"] = int(len(training))
    evaluated["policy_version"] = DEBIT_SHADOW_POLICY_VERSION
    evaluated["model_training_through"] = base_summary["training_through"]
    evaluated["shadow_only"] = True
    evaluated["execution_authorized"] = False
    evaluated["no_order_placement"] = True
    evaluated["shadow_qualified"] = (
        evaluated["predicted_win_probability"].ge(threshold)
        & evaluated["predicted_ev_payoff_correct"].gt(0)
    )
    selected = debit_model.select_book(evaluated, threshold=threshold).copy()
    selected["shadow_selected"] = True
    evaluated = evaluated.sort_values(
        ["shadow_qualified", "predicted_ev_payoff_correct", "predicted_win_probability"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    base_summary.update(
        {
            "status": "SHADOW_ACTIVE",
            "reason": "maturity_safe_v2_model_scored; execution authority intentionally disabled",
            "evaluated_rows": int(len(evaluated)),
            "qualified_rows": int(evaluated["shadow_qualified"].sum()),
            "selected_rows": int(len(selected)),
        }
    )
    return evaluated, selected, base_summary


def _generated_at() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def build_debit_ledger_rows(selected: pd.DataFrame, *, asof: dt.date) -> pd.DataFrame:
    if selected is None or selected.empty:
        return pd.DataFrame(columns=DEBIT_LEDGER_COLUMNS)
    now = _generated_at()
    generated_date = dt.datetime.fromisoformat(now).date()
    timing = "prospective" if generated_date <= asof + dt.timedelta(days=1) else "retrospective_backfill"
    rows = pd.DataFrame(
        {
            "policy_version": DEBIT_SHADOW_POLICY_VERSION,
            "signal_date": str(asof),
            "entry_day": pd.to_datetime(selected["entry_day"], errors="coerce").dt.date.astype(str),
            "generated_at_utc": now,
            "evidence_timing": timing,
            "ticker": selected["ticker"].astype(str).str.upper(),
            "sector": _series(selected, "sector", ""),
            "strategy": selected["strategy"],
            "direction": _series(selected, "direction", ""),
            "expiry": pd.to_datetime(selected["expiry"], errors="coerce").dt.date.astype(str),
            "buy_leg": selected["long_leg_eod"],
            "sell_leg": selected["short_leg_eod"],
            "buy_strike": selected["long_strike_eod"],
            "sell_strike": selected["short_strike_eod"],
            "entry_debit": selected["entry_debit"],
            "entry_width": selected["entry_width"],
            "target_exit_value": np.minimum(selected["entry_width"], selected["entry_debit"] * (1.0 + PROFIT_TAKE_PCT)),
            "predicted_win_probability": selected["predicted_win_probability"],
            "predicted_ev_payoff_correct": selected["predicted_ev_payoff_correct"],
            "prior_sample_size": selected["prior_sample_size"],
            "model_training_through": selected["model_training_through"],
            "feature_parity": _series(selected, "feature_parity", ""),
            "shadow_only": True,
            "execution_authorized": False,
            "no_order_placement": True,
            "outcome_status": "PENDING",
            "outcome_last_checked": "",
            "exit_day": "",
            "exit_reason": "",
            "exit_value": np.nan,
            "pnl_1x": np.nan,
            "return_on_risk": np.nan,
            "outcome_note": "prospective shadow; not an order",
        }
    )
    return rows.reindex(columns=DEBIT_LEDGER_COLUMNS)


def _ledger_key(frame: pd.DataFrame) -> pd.Series:
    columns = ["policy_version", "signal_date", "ticker", "strategy", "expiry", "buy_leg", "sell_leg"]
    return frame.reindex(columns=columns, fill_value="").astype(str).agg("|".join, axis=1)


def update_debit_shadow_ledger(path: Path, incoming: pd.DataFrame) -> pd.DataFrame:
    existing = pd.read_csv(path, low_memory=False).reindex(columns=DEBIT_LEDGER_COLUMNS) if path.exists() else pd.DataFrame(columns=DEBIT_LEDGER_COLUMNS)
    incoming = incoming.reindex(columns=DEBIT_LEDGER_COLUMNS).copy() if incoming is not None else pd.DataFrame(columns=DEBIT_LEDGER_COLUMNS)
    if not existing.empty:
        existing["_key"] = _ledger_key(existing)
        existing["_resolved"] = existing["outcome_status"].astype(str).isin(RESOLVED_STATES).astype(int)
        existing = existing.sort_values(["_key", "_resolved", "generated_at_utc"], ascending=[True, False, True]).drop_duplicates("_key", keep="first").drop(columns="_resolved")
    if not incoming.empty:
        incoming["_key"] = _ledger_key(incoming)
        incoming = incoming.drop_duplicates("_key", keep="first")
        if not existing.empty:
            incoming = incoming.loc[~incoming["_key"].isin(set(existing["_key"]))]
    if existing.empty:
        combined = incoming.copy()
    elif incoming.empty:
        combined = existing.copy()
    else:
        combined = pd.concat([existing, incoming], ignore_index=True)
    combined = combined.drop(columns="_key", errors="ignore").reindex(columns=DEBIT_LEDGER_COLUMNS)
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)
    return combined


def resolve_debit_shadow_ledger(*, root: Path, ledger_path: Path, through_date: dt.date) -> pd.DataFrame:
    from . import goal_shadow

    ledger = pd.read_csv(ledger_path, low_memory=False) if ledger_path.exists() else pd.DataFrame(columns=DEBIT_LEDGER_COLUMNS)
    if ledger.empty:
        return ledger
    for column in ["outcome_status", "outcome_last_checked", "exit_day", "exit_reason", "outcome_note"]:
        ledger[column] = ledger[column].astype(object)
    pending = ~ledger["outcome_status"].astype(str).isin(RESOLVED_STATES)
    entry_dates = pd.to_datetime(ledger.loc[pending, "entry_day"], errors="coerce").dropna()
    if entry_dates.empty:
        return ledger
    folders = goal_shadow.dated_folders(root, entry_dates.min().date(), through_date)
    close_history = goal_shadow.load_close_history(folders)
    hot_history = goal_shadow.load_hot_history(folders)
    quote_history = {day: goal_shadow._quote_lookup(frame) for day, frame in hot_history.items()}
    now = _generated_at()
    for index in ledger.index[pending]:
        source = ledger.loc[index]
        entry_day = pd.to_datetime(source.get("entry_day"), errors="coerce")
        expiry = pd.to_datetime(source.get("expiry"), errors="coerce")
        if pd.isna(entry_day) or pd.isna(expiry):
            continue
        row = pd.Series(
            {
                "ticker": source.get("ticker"),
                "strategy": source.get("strategy"),
                "strategy_kind": "Debit",
                "asof": entry_day.date(),
                "expiry": expiry.date(),
                "long_leg_eod": source.get("buy_leg"),
                "short_leg_eod": source.get("sell_leg"),
                "long_strike_eod": source.get("buy_strike"),
                "short_strike_eod": source.get("sell_strike"),
                "entry_price": source.get("entry_debit"),
                "entry_width": source.get("entry_width"),
                "target_exit_value": source.get("target_exit_value"),
            }
        )
        result = goal_shadow._simulate_locked_spread_exit(
            row,
            close_history,
            quote_history,
            through_date=through_date,
            slippage_pct=ENTRY_STRESS_PCT,
            profit_take_pct=PROFIT_TAKE_PCT,
            stop_loss_mult=None,
        )
        ledger.at[index, "outcome_last_checked"] = now
        if not result.get("exact_evaluated"):
            ledger.at[index, "outcome_note"] = str(result.get("exact_reason", "awaiting future quote or expiry"))
            continue
        pnl = float(result.get("pnl_1x", 0.0))
        ledger.at[index, "outcome_status"] = "RESOLVED_WIN" if pnl > 0 else "RESOLVED_LOSS" if pnl < 0 else "RESOLVED_FLAT"
        ledger.at[index, "exit_day"] = str(result.get("exit_day", ""))
        ledger.at[index, "exit_reason"] = str(result.get("exit_reason", ""))
        ledger.at[index, "exit_value"] = result.get("exit_value", np.nan)
        ledger.at[index, "pnl_1x"] = pnl
        ledger.at[index, "return_on_risk"] = result.get("return_on_risk", np.nan)
        ledger.at[index, "outcome_note"] = "resolved from point-in-time future quotes; remained shadow-only"
    ledger = ledger.reindex(columns=DEBIT_LEDGER_COLUMNS)
    ledger.to_csv(ledger_path, index=False)
    return ledger


def write_debit_shadow_outputs(
    scored: pd.DataFrame,
    *,
    out_dir: Path,
    root: Path,
    asof: dt.date,
    source_scored_file: str = "",
) -> tuple[pd.DataFrame, dict[str, str], dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    evaluated_path = out_dir / f"codexdaily_v4_debit_shadow_candidates_{asof}.csv"
    selected_path = out_dir / f"codexdaily_v4_debit_shadow_selected_{asof}.csv"
    status_path = out_dir / f"codexdaily_v4_debit_shadow_status_{asof}.json"
    ledger_path = out_dir.parent / DEBIT_LEDGER_NAME
    evaluated, selected, summary = score_debit_shadow(scored, root=root, asof=asof)
    evaluated.to_csv(evaluated_path, index=False)
    selected.to_csv(selected_path, index=False)
    ledger = update_debit_shadow_ledger(ledger_path, build_debit_ledger_rows(selected, asof=asof))
    ledger = resolve_debit_shadow_ledger(root=root, ledger_path=ledger_path, through_date=asof)
    summary.update(
        {
            "source_scored_file": source_scored_file,
            "candidate_artifact": str(evaluated_path),
            "selected_artifact": str(selected_path),
            "central_ledger": str(ledger_path),
            "pending_count": int(ledger["outcome_status"].astype(str).eq("PENDING").sum()) if not ledger.empty else 0,
            "resolved_count": int(ledger["outcome_status"].astype(str).isin(RESOLVED_STATES).sum()) if not ledger.empty else 0,
        }
    )
    status_path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    paths = {
        "debit_shadow_candidates": str(evaluated_path),
        "debit_shadow_selected": str(selected_path),
        "debit_shadow_status": str(status_path),
        "debit_shadow_ledger": str(ledger_path),
    }
    return selected, paths, summary


def write_daily_shadow_outputs(
    scored: pd.DataFrame,
    *,
    out_dir: Path,
    root: Path,
    asof: dt.date,
    source_scored_file: str = "",
) -> tuple[dict[str, str], dict[str, Any]]:
    from .range_gex_income_book_v2 import write_prospective_outputs

    _, debit_paths, debit_summary = write_debit_shadow_outputs(
        scored,
        out_dir=out_dir,
        root=root,
        asof=asof,
        source_scored_file=source_scored_file,
    )
    range_paths, range_summary = write_prospective_outputs(scored, out_dir=out_dir, root=root, asof=asof)
    return {**debit_paths, **range_paths}, {"debit": debit_summary, "range_gex": range_summary}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Refresh non-executable Codex Daily shadow books from an existing scored artifact.")
    parser.add_argument("--scored", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("/Users/anuppamvi/uw_root/tradedesk"))
    parser.add_argument("--date", required=True)
    args = parser.parse_args(argv)
    paths, summary = write_daily_shadow_outputs(
        pd.read_csv(args.scored, low_memory=False),
        out_dir=args.out_dir,
        root=args.root,
        asof=dt.date.fromisoformat(args.date),
        source_scored_file=str(args.scored),
    )
    print(json.dumps({"artifacts": paths, "summary": summary}, indent=2, default=str))


if __name__ == "__main__":
    main()
