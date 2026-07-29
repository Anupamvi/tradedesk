"""Extreme single-name event test: the top 1-3 names per day, not deciles.

The decile studies answered "does this feature rank the cross-section." That is
the wrong question for a book that takes 10-20 trades a month. This asks the
right one: when a name is a genuine outlier on the day, what does it do next?

Sample sizes here are deliberately small, because that is the actual trade count.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PANEL = Path("/Users/anuppamvi/uw_root/tradedesk/out/uw_all_feeds.csv")
OUT = Path("/Users/anuppamvi/uw_root/tradedesk/out/extreme_event_scan.csv")
SPLIT = pd.Timestamp("2026-04-14")

# Each is a plausible "this is an unusual event" measure available end of day.
EVENTS = {
    "hot_chain_premium": ("hc_premium", False),
    "oi_built_premium": ("oi_built_premium", False),
    "oi_signed_premium_bull": ("oi_signed_premium", False),
    "oi_signed_premium_bear": ("oi_signed_premium", True),
    "tape_net_premium_bull": ("tape_net_premium", False),
    "tape_net_premium_bear": ("tape_net_premium", True),
    "dark_pool_premium": ("dp_premium", False),
    "call_volume_surge": ("call_vol_surge", False),
    "put_volume_surge": ("put_vol_surge", False),
    "stock_volume_surge": ("stock_vol_surge", False),
    "implied_move": ("implied_move_perc", False),
    "iv_rank_high": ("iv_rank", False),
}


def main() -> None:
    panel = pd.read_csv(PANEL, low_memory=False)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel[
        (panel["issue_type"] == "Common Stock")
        & (panel["marketcap"].fillna(0) >= 2e9)
    ].copy()
    panel = panel.sort_values(["ticker", "date"])

    dates = np.sort(panel["date"].unique())
    position = {pd.Timestamp(d): i for i, d in enumerate(dates)}
    close = panel.set_index(["ticker", "date"])["close"]

    # Enter next close. Measure signed and absolute moves.
    for horizon in (1, 5, 10):
        entry = panel["date"].map(
            lambda d: dates[position[d] + 1] if position[d] + 1 < len(dates) else pd.NaT
        )
        exit_ = panel["date"].map(
            lambda d: dates[position[d] + horizon + 1]
            if position[d] + horizon + 1 < len(dates)
            else pd.NaT
        )
        entry_px = close.reindex(pd.MultiIndex.from_arrays([panel["ticker"], entry])).to_numpy()
        exit_px = close.reindex(pd.MultiIndex.from_arrays([panel["ticker"], exit_])).to_numpy()
        panel[f"ret_{horizon}"] = exit_px / entry_px - 1.0
        panel[f"abs_{horizon}"] = np.abs(panel[f"ret_{horizon}"])

    rows = []
    for name, (column, ascending) in EVENTS.items():
        for top_n in (1, 3, 5):
            picks = []
            for date, day in panel.groupby("date"):
                day = day.dropna(subset=[column, "ret_5"])
                if len(day) < 100:
                    continue
                chosen = day.nsmallest(top_n, column) if ascending else day.nlargest(top_n, column)
                picks.append(chosen.assign(signal_date=date))
            if not picks:
                continue
            picked = pd.concat(picks, ignore_index=True)
            for sample, frame in (
                ("TRAIN", picked[picked.signal_date < SPLIT]),
                ("TEST", picked[picked.signal_date >= SPLIT]),
            ):
                if len(frame) < 10:
                    continue
                universe_abs = panel.loc[panel.date.isin(frame.signal_date.unique()), "abs_5"].mean()
                rows.append(
                    {
                        "event": name,
                        "top_n_per_day": top_n,
                        "sample": sample,
                        "trades": len(frame),
                        "trades_per_month": len(frame) / max(frame.signal_date.nunique() / 21.0, 1e-9),
                        "mean_ret_5": frame.ret_5.mean(),
                        "mean_abs_5": frame.abs_5.mean(),
                        "universe_abs_5": universe_abs,
                        "abs_vs_universe": frame.abs_5.mean() / universe_abs if universe_abs else np.nan,
                        "pct_move_over_5": (frame.abs_5 > 0.05).mean(),
                        "pct_move_over_10": (frame.abs_5 > 0.10).mean(),
                        "up_rate": (frame.ret_5 > 0).mean(),
                    }
                )
    result = pd.DataFrame(rows)
    result.to_csv(OUT, index=False)

    test = result[result["sample"] == "TEST"].copy()
    print("=== EXTREME EVENTS, UNTOUCHED SECOND HALF, ranked by move size vs universe ===")
    print(
        test.sort_values("abs_vs_universe", ascending=False)
        .head(20)
        .round(4)
        .to_string(index=False)
    )
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
