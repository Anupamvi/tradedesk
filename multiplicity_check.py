"""Does the Technology put lane survive correction for everything else I tested?

Finding one significant result after testing 22 sector-direction combinations is
roughly what chance alone produces. Before treating that lane as real it has to
clear the multiplicity bar, and the honest way to set that bar is empirically:
count how often the SEARCH PROCEDURE -- not a single hypothesis -- turns up a
combo that looks this good when nothing is there.

Each replication draws random selections for every sector and direction, applies
the same "significant in both halves" screen, and records the best combo found.
The distribution of those best-of-search results is the null the real finding
must beat.
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

REPLICATIONS = int(sys.argv[1]) if len(sys.argv) > 1 else 60
OUT = base.ROOT / "out/multiplicity_check.csv"
MIN_TRAIN = 40
MIN_TEST = 15


def profit_factor(values: pd.Series) -> float:
    gains = values[values > 0].sum()
    losses = -values[values < 0].sum()
    return gains / losses if losses > 0 else np.nan


def evaluate(trades: pd.DataFrame) -> pd.DataFrame:
    """Per sector: profit factor in each half, and whether both clear 1.0."""
    if trades.empty:
        return pd.DataFrame()
    trades = trades.copy()
    trades["sample"] = np.where(trades.signal_date >= base.SPLIT, "TEST", "TRAIN")
    rows = []
    for sector, block in trades.groupby("sector"):
        train = block[block["sample"] == "TRAIN"]
        test = block[block["sample"] == "TEST"]
        if len(train) < MIN_TRAIN or len(test) < MIN_TEST:
            continue
        rows.append(
            {
                "sector": sector,
                "train_pf": profit_factor(train.pnl),
                "test_pf": profit_factor(test.pnl),
                "train_mean": train.return_on_cost.mean(),
                "test_mean": test.return_on_cost.mean(),
                "n_test": len(test),
            }
        )
    return pd.DataFrame(rows)


def best_of_search(frame: pd.DataFrame) -> float:
    """Best test PF among combos that are profitable in BOTH halves."""
    if frame.empty:
        return np.nan
    qualifying = frame[(frame.train_pf > 1.0) & (frame.test_pf > 1.0)]
    return qualifying.test_pf.max() if len(qualifying) else 0.0


def main() -> None:
    columns = ["date", "ticker", "sector", "issue_type", "marketcap", "close", "pos_52w"]
    panel = pd.read_csv(base.PANEL, usecols=columns, low_memory=False)
    panel["date"] = pd.to_datetime(panel["date"]).dt.strftime("%Y-%m-%d")
    panel = panel[
        (panel.issue_type == "Common Stock")
        & (panel.marketcap.fillna(0) >= 2e9)
        & panel.sector.notna()
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

    print("[multi] warming quote cache", flush=True)
    for session in days:
        quote_for(session)

    rng = np.random.default_rng(20260728)

    observed = []
    for direction in ("long_call", "long_put"):
        frame = evaluate(sym.simulate(panel, days, quote_for, direction, rng, randomize=False))
        if not frame.empty:
            frame["direction"] = direction
            observed.append(frame)
    actual = pd.concat(observed, ignore_index=True)
    actual_best = best_of_search(actual)
    print("\n=== WHAT THE REAL SEARCH FOUND ===")
    qualifying = actual[(actual.train_pf > 1.0) & (actual.test_pf > 1.0)]
    print(actual.round(3).to_string(index=False))
    print(f"\ncombos profitable in both halves: {len(qualifying)} of {len(actual)}")
    print(f"best test PF among them: {actual_best:.3f}")

    nulls = []
    for trial in range(REPLICATIONS):
        found = []
        for direction in ("long_call", "long_put"):
            frame = evaluate(sym.simulate(panel, days, quote_for, direction, rng, randomize=True))
            if not frame.empty:
                frame["direction"] = direction
                found.append(frame)
        if not found:
            continue
        combined = pd.concat(found, ignore_index=True)
        winners = combined[(combined.train_pf > 1.0) & (combined.test_pf > 1.0)]
        nulls.append(
            {
                "trial": trial,
                "best_test_pf": best_of_search(combined),
                "n_qualifying": len(winners),
                "n_combos": len(combined),
            }
        )
        if (trial + 1) % 10 == 0:
            print(f"[multi] {trial + 1}/{REPLICATIONS}", flush=True)

    null = pd.DataFrame(nulls)
    null.to_csv(OUT, index=False)

    print(f"\n=== SEARCH-CORRECTED NULL, {len(null)} replications ===")
    print("Each replication runs the SAME search over random selections and keeps its best find.")
    print(f"  combos surviving 'profitable in both halves' by chance: "
          f"mean {null.n_qualifying.mean():.2f}  max {null.n_qualifying.max():.0f}")
    print(f"  best test PF found by chance: mean {null.best_test_pf.mean():.3f}  "
          f"p95 {null.best_test_pf.quantile(0.95):.3f}  max {null.best_test_pf.max():.3f}")
    p_value = (null.best_test_pf >= actual_best).mean()
    print(f"\n  actual best test PF = {actual_best:.3f}")
    print(f"  search-corrected p-value = {p_value:.4f}")
    print("  " + ("SURVIVES multiplicity correction" if p_value <= 0.05 else "DOES NOT survive -- consistent with chance"))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
