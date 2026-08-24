"""Permutation test on the Technology two-directional book.

The put lane is the load-bearing claim -- it is what shows the result is
selection rather than beta -- and it rests on 37 test trades. A single random
draw is one sample from the null and cannot establish that. This rebuilds the
null properly for both directions.

Selection is replaced by a random draw of the same size from the same sector on
the same dates, running the identical entry and managed-exit machinery.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import managed_exit_backtest as base  # noqa: E402
import symmetric_direction_test as sym  # noqa: E402

PERMUTATIONS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
OUT = base.ROOT / "out/tech_permutation_test.csv"


def profit_factor(values: pd.Series) -> float:
    gains = values[values > 0].sum()
    losses = -values[values < 0].sum()
    return gains / losses if losses > 0 else np.nan


def stats(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"n": 0, "mean": np.nan, "win": np.nan, "pf": np.nan, "pnl": np.nan}
    return {
        "n": len(trades),
        "mean": trades.return_on_cost.mean(),
        "win": trades.pnl.gt(0).mean(),
        "pf": profit_factor(trades.pnl),
        "pnl": trades.pnl.sum(),
    }


def main() -> None:
    columns = ["date", "ticker", "sector", "issue_type", "marketcap", "close", "pos_52w"]
    panel = pd.read_csv(base.PANEL, usecols=columns, low_memory=False)
    panel["date"] = pd.to_datetime(panel["date"]).dt.strftime("%Y-%m-%d")
    panel = panel[
        (panel.issue_type == "Common Stock")
        & (panel.marketcap.fillna(0) >= 2e9)
        & (panel.sector == "Technology")
    ].sort_values(["ticker", "date"])

    days = sorted(p.name for p in base.ROOT.iterdir() if p.is_dir() and re.fullmatch(r"2026-\d{2}-\d{2}", p.name))
    position = {d: i for i, d in enumerate(days)}
    cache: dict[str, pd.DataFrame] = {}

    def quote_for(session: str) -> pd.DataFrame:
        if session not in cache:
            slot = position[session]
            cache[session] = (
                base.chain_quotes(session, days[slot + 1]) if slot + 1 < len(days) else pd.DataFrame()
            )
        return cache[session]

    print("[tech-perm] warming quote cache", flush=True)
    for session in days:
        quote_for(session)

    rng = np.random.default_rng(20260728)
    rows = []
    for direction in ("long_call", "long_put"):
        actual = sym.simulate(panel, days, quote_for, direction, rng, randomize=False)
        for sample in ("TRAIN", "TEST"):
            observed = stats(actual[actual.signal_date.ge(base.SPLIT) == (sample == "TEST")])
            print(f"[tech-perm] actual {direction} {sample}: n={observed['n']} pf={observed['pf']:.2f}", flush=True)

        null = []
        for trial in range(PERMUTATIONS):
            trades = sym.simulate(panel, days, quote_for, direction, rng, randomize=True)
            for sample in ("TRAIN", "TEST"):
                record = stats(trades[trades.signal_date.ge(base.SPLIT) == (sample == "TEST")])
                record.update(direction=direction, sample=sample, trial=trial)
                null.append(record)
            if (trial + 1) % 25 == 0:
                print(f"[tech-perm] {direction} {trial + 1}/{PERMUTATIONS}", flush=True)

        null_frame = pd.DataFrame(null)
        rows.append(null_frame)

        print(f"\n=== {direction.upper()} : {PERMUTATIONS} permutations ===")
        for sample in ("TRAIN", "TEST"):
            observed = stats(actual[actual.signal_date.ge(base.SPLIT) == (sample == "TEST")])
            block = null_frame[null_frame["sample"] == sample]
            for metric in ("mean", "win", "pf", "pnl"):
                distribution = block[metric].dropna()
                value = observed[metric]
                if distribution.empty or not np.isfinite(value):
                    continue
                p_value = (distribution >= value).mean()
                print(
                    "  {:<5} {:<5} actual={:>10.3f}  null={:>9.3f}  p95={:>9.3f}  p={:.4f}{}".format(
                        sample, metric, value, distribution.mean(),
                        distribution.quantile(0.95), p_value,
                        "  <<<" if p_value <= 0.05 else "",
                    )
                )
        print(flush=True)

    pd.concat(rows, ignore_index=True).to_csv(OUT, index=False)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
