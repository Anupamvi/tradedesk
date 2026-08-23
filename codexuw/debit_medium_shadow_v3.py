from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


POLICY_VERSION = "codexdaily-v4-debit-bull-call-production-v5-20260816"
MEDIUM_THRESHOLD = 0.45
HIGH_THRESHOLD = 0.55
MEDIUM_SIZE_MULTIPLIER = 0.25
MIN_UNION_ROWS = 15
MIN_UNION_HOLDOUT_ROWS = 5
MIN_STRICT_BULL_ROWS = 5
MIN_STRICT_BULL_HOLDOUT_ROWS = 2
MIN_PROFIT_FACTOR = 1.50
MIN_STRESS_PROFIT_FACTOR = 1.25


def _generated_at() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _series(frame: pd.DataFrame, names: list[str], default: Any = "") -> pd.Series:
    for name in names:
        if name in frame.columns:
            return frame[name]
    return pd.Series(default, index=frame.index)


def _numeric(frame: pd.DataFrame, names: list[str]) -> pd.Series:
    return pd.to_numeric(_series(frame, names, np.nan), errors="coerce")


def _ticket_key(frame: pd.DataFrame) -> pd.Series:
    parts = [
        _series(frame, ["signal_day", "asof"]).astype(str),
        _series(frame, ["ticker"]).astype(str),
        _series(frame, ["strategy", "setup_family"]).astype(str),
        _series(frame, ["expiry", "expiration"]).astype(str),
        _series(frame, ["long_strike_eod", "long_strike", "buy_strike"]).astype(str),
        _series(frame, ["short_strike_eod", "short_strike", "sell_strike"]).astype(str),
    ]
    key = parts[0]
    for part in parts[1:]:
        key = key + "|" + part
    return key


def _signal_days(frame: pd.DataFrame, asof: dt.date | None = None) -> pd.Series:
    if "signal_day" in frame.columns:
        values = pd.to_datetime(frame["signal_day"], errors="coerce")
    elif "asof" in frame.columns:
        values = pd.to_datetime(frame["asof"], errors="coerce")
    else:
        values = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    if asof is not None:
        values = values.fillna(pd.Timestamp(asof))
    return values


def select_medium_bull_candidates(
    candidates: pd.DataFrame,
    *,
    high_selected: pd.DataFrame | None = None,
    threshold: float = MEDIUM_THRESHOLD,
    asof: dt.date | None = None,
) -> pd.DataFrame:
    """Select a research-only bull-call Medium lane without the payoff EV veto."""
    if candidates.empty:
        return candidates.copy()

    work = candidates.copy()
    work["predicted_win_probability"] = _numeric(
        work, ["predicted_win_probability", "model_probability"]
    )
    work["_signal_day"] = _signal_days(work, asof)
    work["_ticket_key"] = _ticket_key(work)
    strategy = _series(work, ["strategy", "setup_family"]).astype(str)
    eligible = strategy.str.contains("Bull Call", case=False, na=False)
    eligible &= work["predicted_win_probability"].ge(float(threshold))
    work = work.loc[eligible].copy()

    if high_selected is not None and not high_selected.empty:
        high_keys = set(_ticket_key(high_selected).astype(str))
        work = work.loc[~work["_ticket_key"].isin(high_keys)].copy()
        high_days = set(_signal_days(high_selected, asof).dropna())
        work = work.loc[~work["_signal_day"].isin(high_days)].copy()

    if work.empty:
        return work.drop(columns=["_signal_day", "_ticket_key"], errors="ignore")

    work["_sort_ev"] = _numeric(
        work, ["predicted_ev_payoff_correct", "predicted_ev_1x"]
    ).fillna(-np.inf)
    work = work.sort_values(
        ["_signal_day", "predicted_win_probability", "_sort_ev"],
        ascending=[True, False, False],
        kind="stable",
    )
    work = work.groupby("_signal_day", group_keys=False).head(1).copy()
    work["shadow_policy_version"] = POLICY_VERSION
    work["shadow_book"] = "MEDIUM_BULL_CALL_DEBIT"
    work["shadow_status"] = "RESEARCH_SHADOW_ONLY"
    work["execution_authorized"] = False
    work["size_multiplier"] = MEDIUM_SIZE_MULTIPLIER
    work["selection_reason"] = (
        "Bull-call debit with walk-forward probability >= "
        f"{threshold:.2f}; existing High tickets excluded; theoretical payoff EV is diagnostic only"
    )
    return work.drop(columns=["_signal_day", "_ticket_key", "_sort_ev"], errors="ignore")


def _metrics(frame: pd.DataFrame, pnl_column: str = "stress_pnl_10pct") -> dict[str, Any]:
    if frame.empty or pnl_column not in frame.columns:
        return {
            "n": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": None,
            "profit_factor": None,
            "total_pnl": 0.0,
            "max_drawdown": 0.0,
        }
    work = frame.copy()
    work["_day"] = _signal_days(work)
    work = work.sort_values("_day", kind="stable")
    pnl = pd.to_numeric(work[pnl_column], errors="coerce").dropna().to_numpy(float)
    wins = int((pnl > 0).sum())
    losses = int((pnl < 0).sum())
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = float(-pnl[pnl < 0].sum())
    curve = np.r_[0.0, np.cumsum(pnl)]
    drawdown = float(np.min(curve - np.maximum.accumulate(curve))) if len(pnl) else 0.0
    return {
        "n": int(len(pnl)),
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / len(pnl), 6) if len(pnl) else None,
        "profit_factor": round(gross_profit / gross_loss, 6) if gross_loss else None,
        "total_pnl": round(float(pnl.sum()), 2),
        "max_drawdown": round(drawdown, 2),
    }


def _profit_factor_pass(metrics: dict[str, Any], minimum: float) -> bool:
    value = metrics.get("profit_factor")
    if value is None:
        return int(metrics.get("wins") or 0) > 0 and int(metrics.get("losses") or 0) == 0
    return float(value) >= float(minimum)


def evaluate_predictions(
    predictions: pd.DataFrame,
    *,
    cutoff: dt.date | str = dt.date(2026, 5, 19),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    work = predictions.copy()
    work["signal_day"] = _signal_days(work)
    work["exit_day"] = pd.to_datetime(_series(work, ["exit_day"]), errors="coerce")
    probability = _numeric(work, ["predicted_win_probability"])
    payoff_ev = _numeric(work, ["predicted_ev_payoff_correct"])
    high = work.loc[probability.ge(HIGH_THRESHOLD) & payoff_ev.gt(0)].copy()
    high["_prob"] = _numeric(high, ["predicted_win_probability"])
    high = high.sort_values(["signal_day", "_prob"], ascending=[True, False])
    high = high.groupby("signal_day", group_keys=False).head(1).drop(columns="_prob")
    medium = select_medium_bull_candidates(work, high_selected=high)
    union = pd.concat([high, medium], ignore_index=True, sort=False)
    union["_ticket_key"] = _ticket_key(union)
    union = union.drop_duplicates("_ticket_key", keep="first").drop(columns="_ticket_key")

    cutoff_ts = pd.Timestamp(cutoff)

    def periods(frame: pd.DataFrame) -> dict[str, Any]:
        return {
            "all": _metrics(frame),
            "development": _metrics(frame.loc[frame["exit_day"] < cutoff_ts]),
            "holdout": _metrics(frame.loc[frame["signal_day"] >= cutoff_ts]),
        }

    stress = {
        column: _metrics(union, column)
        for column in [
            "stress_pnl_0pct",
            "stress_pnl_5pct",
            "stress_pnl_10pct",
            "stress_pnl_15pct",
        ]
        if column in union.columns
    }
    high_metrics = periods(high)
    medium_metrics = periods(medium)
    union_metrics = periods(union)
    strict_bull = high.loc[
        _series(high, ["strategy", "setup_family"]).astype(str).str.contains(
            "Bull Call", case=False, na=False
        )
    ].copy()
    strict_bull_metrics = periods(strict_bull)
    high_pf = high_metrics["all"]["profit_factor"] or 0.0
    union_pf = union_metrics["all"]["profit_factor"] or 0.0
    high_dd = high_metrics["all"]["max_drawdown"]
    union_dd = union_metrics["all"]["max_drawdown"]
    holdout_pnl = union_metrics["holdout"]["total_pnl"]
    stress_15_pnl = stress.get("stress_pnl_15pct", {}).get("total_pnl", 0.0)
    economics_pass = bool(
        union_metrics["all"]["n"] > high_metrics["all"]["n"]
        and union_pf >= high_pf
        and union_dd >= high_dd
        and holdout_pnl > 0
        and stress_15_pnl > 0
    )
    promotion_blockers: list[str] = []
    if not economics_pass:
        promotion_blockers.append("economics_gate_failed")
    if union_metrics["all"]["n"] < MIN_UNION_ROWS:
        promotion_blockers.append(f"union_sample_below_{MIN_UNION_ROWS}")
    if union_metrics["holdout"]["n"] < MIN_UNION_HOLDOUT_ROWS:
        promotion_blockers.append(f"union_holdout_below_{MIN_UNION_HOLDOUT_ROWS}")
    if strict_bull_metrics["all"]["n"] < MIN_STRICT_BULL_ROWS:
        promotion_blockers.append(f"strict_bull_sample_below_{MIN_STRICT_BULL_ROWS}")
    if strict_bull_metrics["holdout"]["n"] < MIN_STRICT_BULL_HOLDOUT_ROWS:
        promotion_blockers.append(
            f"strict_bull_holdout_below_{MIN_STRICT_BULL_HOLDOUT_ROWS}"
        )
    if not _profit_factor_pass(strict_bull_metrics["all"], MIN_PROFIT_FACTOR):
        promotion_blockers.append("strict_bull_profit_factor_failed")
    if strict_bull_metrics["development"]["total_pnl"] <= 0:
        promotion_blockers.append("strict_bull_development_pnl_not_positive")
    if strict_bull_metrics["holdout"]["total_pnl"] <= 0:
        promotion_blockers.append("strict_bull_holdout_pnl_not_positive")
    stress_15_metrics = stress.get("stress_pnl_15pct", {})
    if not _profit_factor_pass(stress_15_metrics, MIN_STRESS_PROFIT_FACTOR):
        promotion_blockers.append("union_15pct_stress_profit_factor_failed")
    if float(stress_15_metrics.get("total_pnl") or 0.0) <= 0:
        promotion_blockers.append("union_15pct_stress_pnl_not_positive")
    production_authorized = not promotion_blockers
    summary = {
        "policy_version": POLICY_VERSION,
        "generated_at": _generated_at(),
        "threshold": MEDIUM_THRESHOLD,
        "medium_size_multiplier": MEDIUM_SIZE_MULTIPLIER,
        "execution_authorized": production_authorized,
        "production_authorized": production_authorized,
        "authority_scope": "one_contract_bull_call_probability_gte_0.55_positive_ev_rr_gte_1.25_quote_width_lte_0.25_noncontra_oi",
        "production_blockers": promotion_blockers,
        "production_blocker": ";".join(promotion_blockers) if promotion_blockers else "none",
        "economics_pass": economics_pass,
        "existing_high": high_metrics,
        "strict_positive_ev_bull_call": strict_bull_metrics,
        "incremental_medium_bull": medium_metrics,
        "union": union_metrics,
        "union_stress": stress,
    }
    return high, medium, union, summary


def write_historical_outputs(
    predictions: pd.DataFrame,
    *,
    out_dir: Path,
    cutoff: dt.date | str = dt.date(2026, 5, 19),
) -> tuple[pd.DataFrame, dict[str, str], dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    high, medium, union, summary = evaluate_predictions(predictions, cutoff=cutoff)
    medium_path = out_dir / "debit_medium_bull_shadow_v3_selected.csv"
    union_path = out_dir / "debit_high_plus_medium_bull_shadow_v3.csv"
    summary_path = out_dir / "debit_medium_bull_shadow_v3_summary.json"
    report_path = out_dir / "debit_medium_bull_shadow_v3_report.md"
    medium.to_csv(medium_path, index=False)
    union.to_csv(union_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    report_path.write_text(
        "# Debit Medium Bull Shadow V3\n\n"
        f"- Policy: `{POLICY_VERSION}`\n"
        f"- Execution authority: **{'ONE-CONTRACT BULL-CALL PILOT' if summary['execution_authorized'] else 'NONE'}**\n"
        f"- Existing High: `{summary['existing_high']['all']}`\n"
        f"- Incremental Medium bull: `{summary['incremental_medium_bull']['all']}`\n"
        f"- Union: `{summary['union']['all']}`\n"
        f"- 15% entry-stress union: `{summary['union_stress'].get('stress_pnl_15pct', {})}`\n"
        f"- Economics gate: `{'PASS' if summary['economics_pass'] else 'FAIL'}`\n"
        f"- Production blocker: {summary['production_blocker']}\n"
    )
    artifacts = {
        "debit_medium_bull_shadow_v3_selected": str(medium_path),
        "debit_high_plus_medium_bull_shadow_v3": str(union_path),
        "debit_medium_bull_shadow_v3_summary": str(summary_path),
        "debit_medium_bull_shadow_v3_report": str(report_path),
    }
    return medium, artifacts, summary


def write_live_medium_outputs(
    scored: pd.DataFrame,
    *,
    out_dir: Path,
    root: Path,
    asof: dt.date,
    source_scored_file: str = "",
) -> tuple[pd.DataFrame, dict[str, str], dict[str, Any]]:
    from codexuw.daily_shadow_books import score_debit_shadow

    candidates, high_selected, base_summary = score_debit_shadow(
        scored,
        root=root,
        asof=asof,
        threshold=HIGH_THRESHOLD,
    )
    medium = select_medium_bull_candidates(
        candidates,
        high_selected=high_selected,
        asof=asof,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_path = out_dir / f"debit_medium_bull_shadow_v3_selected_{asof.isoformat()}.csv"
    summary_path = out_dir / f"debit_medium_bull_shadow_v3_summary_{asof.isoformat()}.json"
    medium.to_csv(selected_path, index=False)
    summary = {
        "policy_version": POLICY_VERSION,
        "generated_at": _generated_at(),
        "asof": asof.isoformat(),
        "source_scored_file": source_scored_file,
        "input_candidate_rows": int(len(candidates)),
        "existing_high_rows": int(len(high_selected)),
        "medium_bull_rows": int(len(medium)),
        "execution_authorized": False,
        "production_authorized": False,
        "production_blocker": "Research shadow only; sample below promotion minimum",
        "base_debit_shadow": base_summary,
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    artifacts = {
        "debit_medium_bull_shadow_v3_selected": str(selected_path),
        "debit_medium_bull_shadow_v3_summary": str(summary_path),
    }
    return medium, artifacts, summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cutoff", default="2026-05-19")
    args = parser.parse_args(argv)
    predictions = pd.read_csv(args.predictions)
    _, artifacts, summary = write_historical_outputs(
        predictions,
        out_dir=args.out_dir,
        cutoff=args.cutoff,
    )
    print(json.dumps({"artifacts": artifacts, "summary": summary}, indent=2, default=str))


if __name__ == "__main__":
    main()
