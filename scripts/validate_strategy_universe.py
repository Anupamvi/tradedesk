from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("/Users/anuppamvi/uw_root/tradedesk")
UNIVERSE = ROOT / "out/sector_strategy_universe_v3.csv"
DETAIL_OUT = ROOT / "out/sector_strategy_validation_v3.csv"
SUMMARY_OUT = ROOT / "out/sector_strategy_validation_summary_v3.csv"
MIN_TRAIN = 20
MIN_TEST = 10
MIN_PROFIT_FACTOR = 1.20
MIN_CLUSTERED_P05 = 1.20
MAX_PERMUTATION_P = 0.05


def profit_factor(values: pd.Series | np.ndarray) -> float:
    series = pd.Series(values, dtype=float).dropna()
    losses = -series[series < 0].sum()
    gains = series[series > 0].sum()
    if losses <= 0:
        return float("inf") if gains > 0 else float("nan")
    return float(gains / losses)


def _metrics(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "n": int(len(frame)),
        "days": int(frame["signal_date"].nunique()) if not frame.empty else 0,
        "average_pnl": float(frame["pnl"].mean()) if not frame.empty else float("nan"),
        "total_pnl": float(frame["pnl"].sum()) if not frame.empty else 0.0,
        "profit_factor": profit_factor(frame["pnl"]) if not frame.empty else float("nan"),
    }


def _draw_matched(
    universe: pd.DataFrame,
    signal_counts: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows = []
    keys = ["signal_date", "sector", "strategy"]
    lookup = {key: frame for key, frame in universe.groupby(keys, sort=False, observed=True)}
    for count in signal_counts.itertuples(index=False):
        key = (count.signal_date, count.sector, count.strategy)
        pool = lookup.get(key)
        if pool is None or pool.empty:
            continue
        sample_size = min(int(count.signal_count), len(pool))
        chosen = rng.choice(pool.index.to_numpy(), size=sample_size, replace=False)
        rows.append(pool.loc[chosen])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _compile_matched_pools(
    universe: pd.DataFrame,
    signal_counts: pd.DataFrame,
) -> list[tuple[np.ndarray, int, str]]:
    keys = ["signal_date", "sector", "strategy"]
    lookup = {key: frame for key, frame in universe.groupby(keys, sort=False, observed=True)}
    pools: list[tuple[np.ndarray, int, str]] = []
    for count in signal_counts.itertuples(index=False):
        pool = lookup.get((count.signal_date, count.sector, count.strategy))
        if pool is None or pool.empty:
            continue
        values = pd.to_numeric(pool["pnl"], errors="coerce").dropna().to_numpy(dtype=float)
        sample_size = min(int(count.signal_count), len(values))
        if sample_size <= 0:
            continue
        pools.append((values, sample_size, str(pool["sample"].iloc[0])))
    return pools


def _draw_profit_factors(
    pools: list[tuple[np.ndarray, int, str]],
    rng: np.random.Generator,
) -> tuple[float, float]:
    totals = {
        "TRAIN": [0.0, 0.0],
        "TEST": [0.0, 0.0],
    }
    for values, sample_size, sample in pools:
        chosen = rng.choice(len(values), size=sample_size, replace=False)
        pnl = values[chosen]
        totals[sample][0] += float(pnl[pnl > 0].sum())
        totals[sample][1] += float(-pnl[pnl < 0].sum())
    output = []
    for sample in ("TRAIN", "TEST"):
        gains, losses = totals[sample]
        output.append(gains / losses if losses > 0 else float("inf") if gains > 0 else float("nan"))
    return float(output[0]), float(output[1])


def _assert_test_has_power(family_size: int, permutations: int) -> None:
    """A permutation floor above the Holm threshold rejects everything regardless of data."""
    if family_size <= 0:
        return
    best_possible = family_size / (permutations + 1.0)
    if best_possible > MAX_PERMUTATION_P:
        raise SystemExit(
            f"validation has zero power: {permutations} permutations over {family_size} "
            f"hypotheses floors the Holm-adjusted p at {best_possible:.4f} > "
            f"{MAX_PERMUTATION_P}. Every result would be REJECTED by construction. "
            f"Use at least {math.ceil(family_size / MAX_PERMUTATION_P) - 1} permutations."
        )


def holm_adjust(p_values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(p_values, errors="coerce").fillna(1.0).clip(0.0, 1.0)
    ordered = numeric.sort_values()
    adjusted = pd.Series(1.0, index=numeric.index, dtype=float)
    running = 0.0
    family_size = len(ordered)
    for rank, (index, value) in enumerate(ordered.items()):
        running = max(running, min(1.0, float(value) * (family_size - rank)))
        adjusted.at[index] = running
    return adjusted


def _clustered_p05(frame: pd.DataFrame, rng: np.random.Generator, trials: int) -> float:
    by_day = {day: group["pnl"].to_numpy() for day, group in frame.groupby("signal_date")}
    days = list(by_day)
    if len(days) < 2:
        return float("nan")
    values = []
    for _ in range(trials):
        chosen = rng.choice(days, size=len(days), replace=True)
        values.append(profit_factor(np.concatenate([by_day[day] for day in chosen])))
    finite = np.asarray([value for value in values if np.isfinite(value)])
    return float(np.quantile(finite, 0.05)) if finite.size else float("nan")


def _candidate_definitions(universe: pd.DataFrame) -> list[dict[str, str]]:
    definitions = []
    for sector, strategy in universe[["sector", "strategy"]].drop_duplicates().itertuples(index=False):
        definitions.append({"scope": "sector", "scope_value": str(sector), "strategy": str(strategy)})
    for state, strategy in universe[["sector_state", "strategy"]].drop_duplicates().itertuples(index=False):
        definitions.append({"scope": "sector_state", "scope_value": str(state), "strategy": str(strategy)})
    for strategy in universe["strategy"].drop_duplicates().astype(str):
        definitions.append({"scope": "all_sectors", "scope_value": "all", "strategy": strategy})
    return definitions


def _scope_mask(frame: pd.DataFrame, definition: dict[str, str]) -> pd.Series:
    if definition["scope"] == "sector":
        return frame["sector"].astype(str).eq(definition["scope_value"])
    if definition["scope"] == "sector_state":
        return frame["sector_state"].astype(str).eq(definition["scope_value"])
    return pd.Series(True, index=frame.index)


def validate(
    universe: pd.DataFrame,
    *,
    permutations: int = 20_000,
    bootstrap_trials: int = 4_000,
    seed: int = 20260801,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if universe is None or universe.empty:
        return pd.DataFrame(), pd.DataFrame()
    work = universe.copy()
    work["signal_selected"] = work["signal_selected"].astype(str).str.lower().isin({"true", "1", "yes"})
    rng = np.random.default_rng(seed)
    records = []
    definitions = _candidate_definitions(work)
    for definition in definitions:
        scoped = work[_scope_mask(work, definition) & work["strategy"].eq(definition["strategy"])].copy()
        signal = scoped[scoped["signal_selected"]].copy()
        if signal.empty:
            continue
        signal_counts = (
            signal.groupby(["signal_date", "sector", "strategy"], observed=True)
            .size()
            .rename("signal_count")
            .reset_index()
        )
        control = _draw_matched(scoped, signal_counts, rng)
        row: dict[str, Any] = dict(definition)
        for sample in ("TRAIN", "TEST"):
            signal_metrics = _metrics(signal[signal["sample"].eq(sample)])
            control_metrics = _metrics(control[control["sample"].eq(sample)])
            prefix = sample.lower()
            for key, value in signal_metrics.items():
                row[f"{prefix}_signal_{key}"] = value
            for key, value in control_metrics.items():
                row[f"{prefix}_control_{key}"] = value
        screen_pass = bool(
            row["train_signal_n"] >= MIN_TRAIN
            and row["test_signal_n"] >= MIN_TEST
            and row["train_signal_profit_factor"] >= MIN_PROFIT_FACTOR
            and row["test_signal_profit_factor"] >= MIN_PROFIT_FACTOR
            and row["train_signal_profit_factor"] > row["train_control_profit_factor"]
            and row["test_signal_profit_factor"] > row["test_control_profit_factor"]
        )
        row["screen_pass"] = screen_pass
        row["evaluable_hypothesis"] = bool(
            row["train_signal_n"] >= MIN_TRAIN and row["test_signal_n"] >= MIN_TEST
        )
        if screen_pass:
            row["clustered_pf_p05"] = _clustered_p05(signal, rng, bootstrap_trials)
            pools = _compile_matched_pools(scoped, signal_counts)
            null_train = []
            null_test = []
            for _ in range(permutations):
                train_pf, test_pf = _draw_profit_factors(pools, rng)
                null_train.append(train_pf)
                null_test.append(test_pf)
            train_exceedances = int(np.sum(np.asarray(null_train) >= row["train_signal_profit_factor"]))
            test_exceedances = int(np.sum(np.asarray(null_test) >= row["test_signal_profit_factor"]))
            row["train_permutation_p"] = (train_exceedances + 1.0) / (permutations + 1.0)
            row["test_permutation_p"] = (test_exceedances + 1.0) / (permutations + 1.0)
            row["joint_permutation_p"] = max(
                row["train_permutation_p"],
                row["test_permutation_p"],
            )
        else:
            row["clustered_pf_p05"] = float("nan")
            row["train_permutation_p"] = float("nan")
            row["test_permutation_p"] = float("nan")
            row["joint_permutation_p"] = 1.0
        records.append(row)
    detail = pd.DataFrame(records)
    evaluable = detail["evaluable_hypothesis"].astype(bool)
    family_size = int(evaluable.sum())
    detail["multiplicity_family_size"] = family_size
    _assert_test_has_power(family_size, permutations)
    detail["holm_adjusted_joint_p"] = 1.0
    if evaluable.any():
        detail.loc[evaluable, "holm_adjusted_joint_p"] = holm_adjust(
            detail.loc[evaluable, "joint_permutation_p"]
        )
    detail["release_status"] = np.where(
        detail["screen_pass"].astype(bool)
        & detail["clustered_pf_p05"].ge(MIN_CLUSTERED_P05)
        & detail["holm_adjusted_joint_p"].le(MAX_PERMUTATION_P),
        "VALIDATED",
        "REJECTED",
    )
    summary = detail[detail["screen_pass"]].sort_values(
        ["release_status", "test_signal_profit_factor"],
        ascending=[True, False],
    ) if not detail.empty else detail
    return detail, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate all-sector option strategy lanes")
    parser.add_argument("--universe", default=str(UNIVERSE))
    parser.add_argument("--permutations", type=int, default=20_000)
    parser.add_argument("--bootstrap-trials", type=int, default=4_000)
    args = parser.parse_args()
    universe = pd.read_csv(args.universe, low_memory=False)
    detail, summary = validate(
        universe,
        permutations=args.permutations,
        bootstrap_trials=args.bootstrap_trials,
    )
    detail.to_csv(DETAIL_OUT, index=False)
    summary.to_csv(SUMMARY_OUT, index=False)
    print(summary.to_string(index=False))
    print(f"\nwrote {DETAIL_OUT}")
    print(f"wrote {SUMMARY_OUT}")


if __name__ == "__main__":
    main()