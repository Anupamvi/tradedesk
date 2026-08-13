"""Test whether constituent OI-direction breadth predicts sector ETF returns."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PANEL = Path("/Users/anuppamvi/uw_root/tradedesk/out/uw_all_feeds.csv")
OUT = Path("/Users/anuppamvi/uw_root/tradedesk/out/oi_sector_rotation.csv")
HORIZON = 5
SPLIT = pd.Timestamp("2026-04-14")
COST = 0.001  # 5 bps round trip on each of the long and short ETF legs

SECTOR_ETF = {
    "Basic Materials": "XLB",
    "Communication Services": "XLC",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Energy": "XLE",
    "Financial Services": "XLF",
    "Healthcare": "XLV",
    "Industrials": "XLI",
    "Real Estate": "XLRE",
    "Technology": "XLK",
    "Utilities": "XLU",
}


def nw_t(values: pd.Series, lag: int = HORIZON) -> float:
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
    columns = ["date", "ticker", "sector", "issue_type", "marketcap", "close", "oi_dir_bias"]
    panel = pd.read_csv(PANEL, usecols=columns, low_memory=False)
    panel.date = pd.to_datetime(panel.date)
    etf_closes = panel[panel.ticker.isin(SECTOR_ETF.values())].pivot(
        index="date", columns="ticker", values="close"
    )
    stocks = panel[
        (panel.issue_type == "Common Stock")
        & (panel.marketcap.fillna(0) >= 2e9)
        & panel.sector.isin(SECTOR_ETF)
        & panel.oi_dir_bias.notna()
    ].copy()

    dates = np.sort(panel.date.unique())
    position = {pd.Timestamp(date): index for index, date in enumerate(dates)}
    rows = []
    for (date, sector), group in stocks.groupby(["date", "sector"]):
        if len(group) < 20:
            continue
        entry_position = position[date] + 1
        exit_position = entry_position + HORIZON
        if exit_position >= len(dates):
            continue
        etf = SECTOR_ETF[sector]
        entry_date = pd.Timestamp(dates[entry_position])
        exit_date = pd.Timestamp(dates[exit_position])
        if etf not in etf_closes.columns or entry_date not in etf_closes.index or exit_date not in etf_closes.index:
            continue
        entry = etf_closes.at[entry_date, etf]
        exit_ = etf_closes.at[exit_date, etf]
        if not np.isfinite(entry) or not np.isfinite(exit_):
            continue
        positive = (group.oi_dir_bias > 0).mean()
        negative = (group.oi_dir_bias < 0).mean()
        rows.append(
            {
                "date": date,
                "sector": sector,
                "etf": etf,
                "names": len(group),
                "oi_breadth": positive - negative,
                "oi_mean": group.oi_dir_bias.mean(),
                "oi_median": group.oi_dir_bias.median(),
                "return": exit_ / entry - 1.0,
            }
        )
    frame = pd.DataFrame(rows)

    summaries = []
    daily_results = {}
    for feature in ("oi_breadth", "oi_mean", "oi_median"):
        daily = []
        for date, day in frame.groupby("date"):
            if len(day) < 9:
                continue
            ordered = day.sort_values(feature)
            long_row = ordered.iloc[-1]
            short_row = ordered.iloc[0]
            daily.append(
                {
                    "date": date,
                    "long_etf": long_row.etf,
                    "short_etf": short_row.etf,
                    "long_return": long_row["return"],
                    "short_return": short_row["return"],
                    "spread": long_row["return"] - short_row["return"],
                }
            )
        daily = pd.DataFrame(daily)
        daily_results[feature] = daily
        for sample, subset in (
            ("FULL", daily),
            ("TRAIN", daily[daily.date < SPLIT]),
            ("TEST", daily[daily.date >= SPLIT]),
        ):
            gross = subset.spread
            net = gross - COST
            summaries.append(
                {
                    "feature": feature,
                    "sample": sample,
                    "days": len(subset),
                    "gross_spread": gross.mean(),
                    "gross_nw_t": nw_t(gross),
                    "net_spread": net.mean(),
                    "net_nw_t": nw_t(net),
                    "gross_hit": (gross > 0).mean(),
                    "net_hit": (net > 0).mean(),
                }
            )
    result = pd.DataFrame(summaries)
    result.to_csv(OUT, index=False)
    print(result.round(4).to_string(index=False))

    best = daily_results["oi_breadth"]
    test = best[best.date >= SPLIT].copy()
    test["month"] = test.date.dt.to_period("M").astype(str)
    print("\n=== OI-BREADTH TEST MONTHS ===")
    print(test.groupby("month").agg(days=("spread", "size"), spread=("spread", "mean"), hit=("spread", lambda x: (x > 0).mean())).round(4).to_string())
    print("\n=== 2026-07-24 SECTOR SIGNAL ===")
    latest = frame[frame.date == frame.date.max()].sort_values("oi_breadth", ascending=False)
    print(latest[["sector", "etf", "names", "oi_breadth", "oi_mean"]].round(4).to_string(index=False))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
