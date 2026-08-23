from __future__ import annotations

"""Portfolio-capacity audit for a frozen directional-credit outcome ledger."""

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CAPACITY_VERSION = "directional-credit-capacity-v1-20260813"
DEFAULT_CONTRACT_SCALES = (1, 2, 4, 8, 12, 16)
EXECUTION_OI_STATES = {"supportive", "matched_unconfirmed"}


def _number_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame.get(column), errors="coerce")


def prepare_history(history: pd.DataFrame) -> pd.DataFrame:
    out = history.copy()
    out["signal_dt"] = pd.to_datetime(
        out.get("asof", out.get("entry_date", out.get("entry_day"))), errors="coerce"
    ).dt.normalize()
    out["entry_dt"] = pd.to_datetime(
        out.get("entry_date", out.get("entry_day")), errors="coerce"
    ).dt.normalize()
    out["exit_dt"] = pd.to_datetime(
        out.get("exit_date", out.get("exit_day")), errors="coerce"
    ).dt.normalize()
    out["base_pnl_1x"] = _number_series(out, "pnl_1x")
    if out["base_pnl_1x"].isna().all() and "pnl" in out:
        out["base_pnl_1x"] = _number_series(out, "pnl")
    out["reported_stress_pnl_1x"] = _number_series(out, "stress_pnl_10pct")
    entry_credit = _number_series(out, "entry_credit")
    recomputed_stress = out["base_pnl_1x"] - entry_credit * 10.0
    out["stress_pnl_1x"] = recomputed_stress.where(
        entry_credit.notna(), out["reported_stress_pnl_1x"]
    )
    out["risk_1x"] = _number_series(out, "risk_1x")
    derived_risk = (
        _number_series(out, "entry_width") - _number_series(out, "entry_credit")
    ) * 100.0
    out["risk_1x"] = out["risk_1x"].where(out["risk_1x"].gt(0), derived_risk)
    out["ticker"] = out.get("ticker", "UNKNOWN").fillna("UNKNOWN").astype(str)
    out["sector"] = out.get("sector", "Unknown").fillna("Unknown").astype(str)
    return out[
        out["entry_dt"].notna()
        & out["exit_dt"].notna()
        & out["exit_dt"].ge(out["entry_dt"])
        & out["risk_1x"].gt(0)
        & out["base_pnl_1x"].notna()
        & out["stress_pnl_1x"].notna()
    ].sort_values(["entry_dt", "exit_dt", "ticker"]).reset_index(drop=True)


def execution_population(history: pd.DataFrame) -> pd.DataFrame:
    if "oi_carryover_status" not in history.columns:
        return history.iloc[0:0].copy()
    return history[
        history["oi_carryover_status"].fillna("").astype(str).str.lower().isin(EXECUTION_OI_STATES)
    ].copy()


def _max_drawdown(pnl_by_exit_day: pd.Series) -> float:
    if pnl_by_exit_day.empty:
        return 0.0
    cumulative = pnl_by_exit_day.cumsum().to_numpy(dtype=float)
    curve = np.r_[0.0, cumulative]
    return float((curve - np.maximum.accumulate(curve)).min())


def _active_risk(history: pd.DataFrame, contracts: int) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for _, row in history.iterrows():
        risk = float(row["risk_1x"]) * contracts
        for day in pd.date_range(row["entry_dt"], row["exit_dt"], freq="D"):
            records.append({
                "date": day,
                "ticker": row["ticker"],
                "sector": row["sector"],
                "risk": risk,
                "positions": 1,
            })
    return pd.DataFrame(records)


def portfolio_metrics(history: pd.DataFrame, contracts: int) -> dict[str, Any]:
    if history.empty:
        return {"contracts_per_trade": contracts, "trades": 0}
    base = history["base_pnl_1x"] * contracts
    stress = history["stress_pnl_1x"] * contracts
    exit_dates = history["exit_dt"]
    signal_dates = history["signal_dt"]
    base_daily = base.groupby(exit_dates).sum().sort_index()
    stress_daily = stress.groupby(exit_dates).sum().sort_index()
    realized_monthly = pd.DataFrame({"exit_dt": exit_dates, "base": base, "stress": stress})
    realized_monthly["month"] = realized_monthly["exit_dt"].dt.to_period("M").astype(str)
    realized_monthly = realized_monthly.groupby("month")[["base", "stress"]].sum()
    signal_monthly = pd.DataFrame({"signal_dt": signal_dates, "base": base, "stress": stress})
    signal_monthly["month"] = signal_monthly["signal_dt"].dt.to_period("M").astype(str)
    signal_monthly = signal_monthly.groupby("month")[["base", "stress"]].sum()
    signal_order = history.assign(_pnl=base).sort_values(
        ["signal_dt", "ticker"], kind="stable"
    )["_pnl"]
    active = _active_risk(history, contracts)
    daily = active.groupby("date").agg(active_positions=("positions", "sum"), risk=("risk", "sum"))
    peak_risk_day = daily["risk"].idxmax()
    peak_active = active[active["date"].eq(peak_risk_day)]
    peak_total_risk = float(peak_active["risk"].sum())
    peak_sector_share = float(
        peak_active.groupby("sector")["risk"].sum().max() / peak_total_risk
    )
    peak_ticker_share = float(
        peak_active.groupby("ticker")["risk"].sum().max() / peak_total_risk
    )
    gross_loss = -float(stress[stress < 0].sum())
    return {
        "contracts_per_trade": int(contracts),
        "trades": int(len(history)),
        "signal_months": int(len(signal_monthly)),
        "realized_months": int(len(realized_monthly)),
        "base_total_pnl": float(base.sum()),
        "base_average_monthly_pnl": float(realized_monthly["base"].mean()),
        "stress_total_pnl": float(stress.sum()),
        "stress_average_monthly_pnl": float(realized_monthly["stress"].mean()),
        "stress_profit_factor": (
            float(stress[stress > 0].sum() / gross_loss) if gross_loss else None
        ),
        "selection_sequence_max_drawdown": _max_drawdown(signal_order),
        "realized_base_max_drawdown": _max_drawdown(base_daily),
        "realized_stress_max_drawdown": _max_drawdown(stress_daily),
        "positive_signal_month_ratio": float((signal_monthly["stress"] > 0).mean()),
        "positive_realized_month_ratio": float((realized_monthly["stress"] > 0).mean()),
        "minimum_signal_month": float(signal_monthly["stress"].min()),
        "minimum_realized_month": float(realized_monthly["stress"].min()),
        "maximum_active_positions": int(daily["active_positions"].max()),
        "peak_defined_risk": peak_total_risk,
        "peak_risk_date": peak_risk_day.date().isoformat(),
        "peak_sector_risk_share": peak_sector_share,
        "peak_ticker_risk_share": peak_ticker_share,
    }


def capacity_curve(
    history: pd.DataFrame,
    *,
    scales: tuple[int, ...] = DEFAULT_CONTRACT_SCALES,
    monthly_target: float = 10_000.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = [portfolio_metrics(history, contracts) for contracts in scales]
    curve = pd.DataFrame(rows)
    one_contract = portfolio_metrics(history, 1)
    average = float(one_contract.get("stress_average_monthly_pnl", 0.0))
    target_contracts = math.ceil(monthly_target / average) if average > 0 else None
    target = portfolio_metrics(history, target_contracts) if target_contracts else {}
    blockers: list[str] = []
    if target:
        if target["positive_signal_month_ratio"] < 1.0:
            blockers.append("historical_monthly_target_not_consistent")
        if target["peak_sector_risk_share"] > 0.40:
            blockers.append("peak_sector_share_above_40pct")
        if target["peak_ticker_risk_share"] > 0.20:
            blockers.append("peak_ticker_share_above_20pct")
    summary = {
        "version": CAPACITY_VERSION,
        "history_trades": int(len(history)),
        "monthly_target": float(monthly_target),
        "one_contract": one_contract,
        "contracts_required_for_historical_average_target": target_contracts,
        "target_scale": target,
        "stress_pnl_distinct_from_base": bool(
            not history["base_pnl_1x"].round(8).equals(history["stress_pnl_1x"].round(8))
        ),
        "reported_stress_matches_recomputed": bool(
            history["reported_stress_pnl_1x"].round(8).equals(
                history["stress_pnl_1x"].round(8)
            )
        ),
        "target_is_reliably_demonstrated": False,
        "blockers": blockers,
        "note": (
            "Required contracts target the historical average, not a guaranteed monthly return. "
            "No account budget or risk limit is assumed."
        ),
    }
    return curve, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--monthly-target", type=float, default=10_000.0)
    parser.add_argument("--execution-only", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    history = prepare_history(pd.read_csv(args.history, low_memory=False))
    population = "all_calibration_rows"
    if args.execution_only:
        history = execution_population(history)
        population = "supportive_or_matched_oi_execution_rows"
    curve, summary = capacity_curve(history, monthly_target=args.monthly_target)
    summary["population"] = population
    curve.to_csv(args.out_dir / "credit_capacity_curve.csv", index=False)
    (args.out_dir / "credit_capacity_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
