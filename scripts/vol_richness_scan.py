"""Is the outsized movement of extreme-flow names already priced into options?

The direction tests failed: large ask-side blocks are mostly hedging flow and the
underlying goes their way only ~46% of the time. But those same names realise
roughly 2.3x the universe's absolute move. That is only tradeable if the market
has not already charged for it.

Benchmark is iv30d scaled to the holding window. The screener's implied_move_perc
CANNOT be used here: its horizon is each name's nearest expiry, which inverts to
~9 trading days for the median name but under 1 day for TSLA/NVDA/AMZN, so it is
not comparable across the cross-section.

A 1-sigma move is not the expected absolute move. For a normal, E|X| = 0.798 sigma,
so realised mean |move| is compared against 0.798 * implied sigma.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/anuppamvi/uw_root/tradedesk")
PANEL = ROOT / "out/uw_all_feeds.csv"
OUT = ROOT / "out/vol_richness_scan.csv"
SPLIT = pd.Timestamp("2026-04-14")

SELECTORS = {
    "oi_built_premium": "oi_built_premium",
    "hot_chain_premium": "hc_premium",
    "dark_pool_premium": "dp_premium",
    "call_volume_surge": "call_vol_surge",
    "stock_volume_surge": "stock_vol_surge",
    "iv_rank": "iv_rank",
}


def main() -> None:
    panel = pd.read_csv(PANEL, low_memory=False)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel[
        (panel.issue_type == "Common Stock")
        & (panel.marketcap.fillna(0) >= 2e9)
        & (panel.iv30d > 0.01)
    ].copy()
    panel = panel.sort_values(["ticker", "date"])

    dates = np.sort(panel.date.unique())
    position = {pd.Timestamp(d): i for i, d in enumerate(dates)}
    close = panel.set_index(["ticker", "date"])["close"]

    horizon = 5
    entry = panel.date.map(lambda d: dates[position[d] + 1] if position[d] + 1 < len(dates) else pd.NaT)
    exit_ = panel.date.map(
        lambda d: dates[position[d] + horizon + 1] if position[d] + horizon + 1 < len(dates) else pd.NaT
    )
    entry_px = close.reindex(pd.MultiIndex.from_arrays([panel.ticker, entry])).to_numpy()
    exit_px = close.reindex(pd.MultiIndex.from_arrays([panel.ticker, exit_])).to_numpy()
    panel["abs_move"] = np.abs(exit_px / entry_px - 1.0)

    # iv30d is an annualised sigma. Scale to the holding window, then convert to
    # an expected ABSOLUTE move: E|X| = sigma * sqrt(2/pi).
    panel["implied_sigma_5d"] = panel.iv30d * np.sqrt(horizon / 252.0)
    panel["expected_abs_move"] = panel.implied_sigma_5d * np.sqrt(2.0 / np.pi)
    panel["realised_over_implied"] = panel.abs_move / panel.expected_abs_move.replace(0, np.nan)

    rows = []
    for name, column in SELECTORS.items():
        for top_n in (1, 3, 5, 10):
            picks = []
            for date, day in panel.groupby("date"):
                day = day.dropna(subset=[column, "abs_move", "expected_abs_move"])
                if len(day) < 100:
                    continue
                picks.append(day.nlargest(top_n, column).assign(signal_date=date))
            if not picks:
                continue
            picked = pd.concat(picks, ignore_index=True)
            for sample, frame in (
                ("TRAIN", picked[picked.signal_date < SPLIT]),
                ("TEST", picked[picked.signal_date >= SPLIT]),
            ):
                if len(frame) < 20:
                    continue
                rows.append(
                    {
                        "selector": name,
                        "top_n_per_day": top_n,
                        "sample": sample,
                        "trades": len(frame),
                        "mean_realised": frame.abs_move.mean(),
                        "mean_expected_abs": frame.expected_abs_move.mean(),
                        "realised_over_implied": frame.realised_over_implied.mean(),
                        "median_ratio": frame.realised_over_implied.median(),
                        "pct_exceeding_implied": frame.realised_over_implied.gt(1).mean(),
                    }
                )
    result = pd.DataFrame(rows)
    result.to_csv(OUT, index=False)

    baseline = panel.dropna(subset=["realised_over_implied"])
    train_base = baseline[baseline.date < SPLIT].realised_over_implied
    test_base = baseline[baseline.date >= SPLIT].realised_over_implied
    print("=== BASELINE: every large-cap name ===")
    print(f"TRAIN mean realised/implied {train_base.mean():.3f}  exceed rate {train_base.gt(1).mean():.3f}")
    print(f"TEST  mean realised/implied {test_base.mean():.3f}  exceed rate {test_base.gt(1).mean():.3f}")
    print("\n=== SELECTED NAMES, untouched second half, sorted by realised/implied ===")
    test = result[result["sample"] == "TEST"].sort_values("realised_over_implied", ascending=False)
    print(test.round(4).to_string(index=False))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
