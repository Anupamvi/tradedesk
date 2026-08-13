"""Chronological validation of the matched call-put IV spread."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/anuppamvi/uw_root/tradedesk")
PANEL = ROOT / "out/uw_all_feeds.csv"
FEATURES = ROOT / "out/volatility_spread_features.csv"
OUT = ROOT / "out/volatility_spread_validation.csv"
MIN_NAMES = 75
COST = 0.001  # 5 bps per long/short leg, round trip


def nw_t(values: pd.Series, lag: int) -> float:
    values = values.dropna().to_numpy()
    if len(values) < 10:
        return np.nan
    centered = values - values.mean()
    n = len(centered)
    variance = centered @ centered / n
    for step in range(1, min(lag, n - 1) + 1):
        covariance = centered[step:] @ centered[:-step] / n
        variance += 2.0 * (1.0 - step / (lag + 1.0)) * covariance
    return values.mean() / np.sqrt(max(variance, 1e-12) / n)


def main() -> None:
    panel = pd.read_csv(
        PANEL,
        usecols=["date", "ticker", "close", "marketcap", "issue_type"],
        low_memory=False,
    )
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel[
        (panel["marketcap"].fillna(0) >= 2e9)
        & (panel["issue_type"] == "Common Stock")
    ].copy()
    if panel.duplicated(["ticker", "date"]).any():
        raise ValueError("duplicate ticker/date rows")
    features = pd.read_csv(FEATURES, low_memory=False)
    features["date"] = pd.to_datetime(features["date"])
    panel = panel.merge(features, on=["date", "ticker"], how="left", validate="one_to_one")
    panel = panel.sort_values(["ticker", "date"])

    dates = np.sort(panel["date"].unique())
    position = {pd.Timestamp(date): index for index, date in enumerate(dates)}
    lookup = panel.set_index(["ticker", "date"])["close"]
    for horizon in (1, 5):
        entry_dates = panel.date.map(
            lambda date: dates[position[date] + 1] if position[date] + 1 < len(dates) else pd.NaT
        )
        exit_dates = panel.date.map(
            lambda date: dates[position[date] + horizon + 1]
            if position[date] + horizon + 1 < len(dates)
            else pd.NaT
        )
        entry_index = pd.MultiIndex.from_arrays([panel.ticker, entry_dates])
        exit_index = pd.MultiIndex.from_arrays([panel.ticker, exit_dates])
        panel[f"fwd_{horizon}d"] = lookup.reindex(exit_index).to_numpy() / lookup.reindex(entry_index).to_numpy() - 1.0

    feature_dates = np.sort(panel.loc[panel.volatility_spread.notna(), "date"].unique())
    split = pd.Timestamp(feature_dates[len(feature_dates) // 2])
    rows = []
    daily_by_horizon = {}
    for horizon in (1, 5):
        target = f"fwd_{horizon}d"
        daily = []
        for date, day in panel.groupby("date"):
            day = day.dropna(subset=["volatility_spread", target]).copy()
            if len(day) < MIN_NAMES or day.volatility_spread.nunique() < 10:
                continue
            rank = day.volatility_spread.rank(pct=True)
            long_return = day.loc[rank >= 0.9, target].mean()
            short_return = day.loc[rank <= 0.1, target].mean()
            daily.append(
                {
                    "date": date,
                    "long": long_return,
                    "short": short_return,
                    "spread": long_return - short_return,
                }
            )
        daily = pd.DataFrame(daily)
        daily_by_horizon[horizon] = daily
        for sample, frame in (
            ("FULL", daily),
            ("TRAIN", daily[daily.date < split]),
            ("TEST", daily[daily.date >= split]),
        ):
            gross = frame.spread
            net = gross - COST
            rows.append(
                {
                    "horizon": f"{horizon}d",
                    "sample": sample,
                    "days": len(frame),
                    "long": frame.long.mean(),
                    "short": frame.short.mean(),
                    "gross_spread": gross.mean(),
                    "gross_nw_t": nw_t(gross, horizon),
                    "net_spread": net.mean(),
                    "net_nw_t": nw_t(net, horizon),
                    "gross_hit": (gross > 0).mean(),
                    "net_hit": (net > 0).mean(),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(OUT, index=False)
    print(f"feature_dates={len(feature_dates)} split={split.date()}")
    print(result.round(4).to_string(index=False))

    test = daily_by_horizon[5]
    test = test[test.date >= split].copy()
    test["month"] = test.date.dt.to_period("M").astype(str)
    print("\n=== 5D TEST MONTHS ===")
    print(test.groupby("month").agg(days=("spread", "size"), spread=("spread", "mean"), hit=("spread", lambda x: (x > 0).mean())).round(4).to_string())
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
