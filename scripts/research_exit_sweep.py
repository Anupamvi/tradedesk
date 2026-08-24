"""Sweep exit-management policies against extracted forward mark paths.

Consumes the output of ``research_exit_paths.py`` and replays every candidate
under alternative take-profit / stop / horizon rules.  Reports train and held-out
performance separately so that a policy which only looks good in-sample is
visible as such.

The sweep is a research tool: it selects nothing and authorises nothing.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

ROUND_TRIP_COMMISSION = 2.60


def _pnl(entry_side: str, entry: float, exit_value: float) -> float:
    gross = entry - exit_value if entry_side == "CREDIT" else exit_value - entry
    return gross * 100.0 - ROUND_TRIP_COMMISSION


def _profit_factor(series: pd.Series) -> float:
    gains = series[series > 0].sum()
    losses = -series[series < 0].sum()
    if losses <= 0:
        return np.nan if gains <= 0 else np.inf
    return gains / losses


def simulate(paths: pd.DataFrame, tp: float, stop: float | None, horizon: int) -> pd.DataFrame:
    """Resolve every trade under one policy. ``tp``/``stop`` are entry multiples."""

    work = paths[paths["session"] <= horizon].copy()
    credit = work["entry_side"] == "CREDIT"

    target_level = np.where(
        credit,
        work["entry_price"] * tp,
        np.minimum(work["entry_width"] * 0.80, work["entry_price"] * tp),
    )
    if stop is None:
        stop_hit = np.zeros(len(work), dtype=bool)
    else:
        stop_level = np.where(
            credit,
            np.minimum(work["entry_width"], work["entry_price"] * stop),
            np.maximum(work["entry_price"] * stop, 0.01),
        )
        stop_hit = np.where(credit, work["value"] >= stop_level, work["value"] <= stop_level)
    target_hit = np.where(credit, work["value"] <= target_level, work["value"] >= target_level)

    work["_target"] = target_hit
    work["_stop"] = stop_hit
    work["_trigger"] = np.where(target_hit, "take_profit", np.where(stop_hit, "stop_loss", ""))

    triggered = work[work["_trigger"] != ""].sort_values(["row_id", "session"])
    first = triggered.groupby("row_id", as_index=False).first()

    last = work.sort_values(["row_id", "session"]).groupby("row_id", as_index=False).last()
    last["_trigger"] = "time_exit"

    resolved = pd.concat([first, last[~last["row_id"].isin(first["row_id"])]], ignore_index=True)
    resolved["pnl"] = [
        _pnl(side, entry, value)
        for side, entry, value in zip(
            resolved["entry_side"], resolved["entry_price"], resolved["value"]
        )
    ]
    return resolved


def summarise(resolved: pd.DataFrame, split_day: str) -> dict:
    asof = pd.to_datetime(resolved["asof"], errors="coerce")
    split = pd.Timestamp(split_day)
    out: dict = {}
    for label, subset in (
        ("all", resolved),
        ("train", resolved[asof <= split]),
        ("heldout", resolved[asof > split]),
    ):
        pnl = subset["pnl"]
        out[f"{label}_n"] = len(pnl)
        out[f"{label}_pf"] = round(_profit_factor(pnl), 3) if len(pnl) else np.nan
        out[f"{label}_avg"] = round(pnl.mean(), 2) if len(pnl) else np.nan
        out[f"{label}_tot"] = round(pnl.sum(), 0) if len(pnl) else np.nan
        out[f"{label}_win"] = round(100 * (pnl > 0).mean(), 1) if len(pnl) else np.nan
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, default=Path("out/research/exit_paths.csv"))
    parser.add_argument("--split-day", default="2026-05-01")
    parser.add_argument("--out", type=Path, default=Path("out/research/exit_sweep.csv"))
    parser.add_argument("--side", choices=["CREDIT", "DEBIT", "BOTH"], default="BOTH")
    args = parser.parse_args()

    paths = pd.read_csv(args.paths, low_memory=False)
    if args.side != "BOTH":
        paths = paths[paths["entry_side"] == args.side]

    credit_tp = [0.25, 0.35, 0.50, 0.65, 0.80]
    credit_stop = [1.5, 2.0, 2.5, 3.0, None]
    debit_tp = [1.3, 1.5, 1.8, 2.2]
    debit_stop = [0.3, 0.5, 0.7, None]
    horizons = [4, 8, 12, 16, 21, 30]

    rows: list[dict] = []
    sides = ["CREDIT", "DEBIT"] if args.side == "BOTH" else [args.side]
    for side in sides:
        subset = paths[paths["entry_side"] == side]
        if subset.empty:
            continue
        tps = credit_tp if side == "CREDIT" else debit_tp
        stops = credit_stop if side == "CREDIT" else debit_stop
        for tp, stop, horizon in itertools.product(tps, stops, horizons):
            resolved = simulate(subset, tp, stop, horizon)
            if resolved.empty:
                continue
            row = {
                "side": side,
                "take_profit": tp,
                "stop": "none" if stop is None else stop,
                "horizon": horizon,
            }
            row.update(summarise(resolved, args.split_day))
            rows.append(row)

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)

    pd.set_option("display.width", 250)
    for side in sides:
        block = frame[frame["side"] == side]
        if block.empty:
            continue
        print(f"\n{'=' * 110}\n{side}: baseline vs best (ranked by held-out profit factor)\n{'=' * 110}")
        cols = [
            "take_profit",
            "stop",
            "horizon",
            "all_n",
            "all_pf",
            "all_avg",
            "train_pf",
            "heldout_n",
            "heldout_pf",
            "heldout_avg",
            "heldout_win",
            "heldout_tot",
        ]
        print(block.sort_values("heldout_pf", ascending=False).head(12)[cols].to_string(index=False))
    print(f"\nwrote {args.out} rows={len(frame)}")


if __name__ == "__main__":
    main()
