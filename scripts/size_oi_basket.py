"""Capacity/concentration test for the OI-direction long-short equity basket."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

PANEL = Path("/Users/anuppamvi/uw_root/tradedesk/out/uw_all_feeds.csv")
OUT = Path("/Users/anuppamvi/uw_root/tradedesk/out/oi_basket_sizing.csv")
HORIZON = 5
SPLIT = pd.Timestamp("2026-04-14")


def stable_order(date: pd.Timestamp, ticker: str, seed: int) -> int:
    text = f"{date.date()}|{ticker}|{seed}".encode()
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "big")


def nw_t(values: pd.Series, lag: int = HORIZON) -> float:
    values = values.dropna().to_numpy()
    centered = values - values.mean()
    n = len(centered)
    if n < 10:
        return np.nan
    variance = centered @ centered / n
    for step in range(1, min(lag, n - 1) + 1):
        covariance = centered[step:] @ centered[:-step] / n
        variance += 2.0 * (1.0 - step / (lag + 1.0)) * covariance
    return values.mean() / np.sqrt(max(variance, 1e-12) / n)


def load() -> pd.DataFrame:
    columns = ["date", "ticker", "issue_type", "marketcap", "close", "oi_dir_bias"]
    panel = pd.read_csv(PANEL, usecols=columns, low_memory=False)
    panel.date = pd.to_datetime(panel.date)
    panel = panel[
        (panel.issue_type == "Common Stock")
        & (panel.marketcap.fillna(0) >= 2e9)
    ].copy()
    dates = np.sort(panel.date.unique())
    position = {pd.Timestamp(date): index for index, date in enumerate(dates)}
    lookup = panel.set_index(["ticker", "date"])["close"]
    entry_dates = panel.date.map(lambda date: dates[position[date] + 1] if position[date] + 1 < len(dates) else pd.NaT)
    exit_dates = panel.date.map(
        lambda date: dates[position[date] + HORIZON + 1]
        if position[date] + HORIZON + 1 < len(dates)
        else pd.NaT
    )
    panel["return"] = (
        lookup.reindex(pd.MultiIndex.from_arrays([panel.ticker, exit_dates])).to_numpy()
        / lookup.reindex(pd.MultiIndex.from_arrays([panel.ticker, entry_dates])).to_numpy()
        - 1.0
    )
    return panel[panel.date >= SPLIT]


def main() -> None:
    panel = load()
    rows = []
    for basket_size in (5, 10, 20, 40, 80, 160):
        for seed in range(30):
            daily = []
            for date, day in panel.groupby("date"):
                day = day.dropna(subset=["oi_dir_bias", "return"]).copy()
                if len(day) < 100:
                    continue
                rank = day.oi_dir_bias.rank(pct=True, method="average")
                longs = day[rank >= 0.9].copy()
                shorts = day[rank <= 0.1].copy()
                if len(longs) < basket_size or len(shorts) < basket_size:
                    continue
                longs["order"] = [stable_order(date, ticker, seed) for ticker in longs.ticker]
                shorts["order"] = [stable_order(date, ticker, seed) for ticker in shorts.ticker]
                long_return = longs.nsmallest(basket_size, "order")["return"].mean()
                short_return = shorts.nsmallest(basket_size, "order")["return"].mean()
                daily.append({"date": date, "spread": long_return - short_return})
            daily = pd.DataFrame(daily)
            rows.append(
                {
                    "basket_size_per_side": basket_size,
                    "seed": seed,
                    "days": len(daily),
                    "gross_spread": daily.spread.mean(),
                    "net_spread_5bps_round_trip_each_side": daily.spread.mean() - 0.001,
                    "nw_t": nw_t(daily.spread),
                    "hit_rate": (daily.spread > 0).mean(),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(OUT, index=False)
    summary = result.groupby("basket_size_per_side").agg(
        seeds=("seed", "size"),
        mean_gross=("gross_spread", "mean"),
        p05_gross=("gross_spread", lambda x: x.quantile(0.05)),
        p95_gross=("gross_spread", lambda x: x.quantile(0.95)),
        mean_net=("net_spread_5bps_round_trip_each_side", "mean"),
        median_t=("nw_t", "median"),
        p05_t=("nw_t", lambda x: x.quantile(0.05)),
        mean_hit=("hit_rate", "mean"),
    )
    print(summary.round(4).to_string())
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
