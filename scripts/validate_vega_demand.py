"""Test whether customer vega/gamma demand predicts future realized magnitude."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/anuppamvi/uw_root/tradedesk")
PANEL = ROOT / "out/uw_all_feeds.csv"
OUT = ROOT / "out/vega_demand_validation.csv"
MIN_NAMES = 75


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
    columns = [
        "date", "ticker", "close", "marketcap", "issue_type", "implied_move_perc",
        "tape_vega_flow", "tape_gamma_flow", "tape_gross_premium",
    ]
    panel = pd.read_csv(PANEL, usecols=columns, low_memory=False)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel[
        (panel.marketcap.fillna(0) >= 2e9)
        & (panel.issue_type == "Common Stock")
    ].copy()
    if panel.duplicated(["ticker", "date"]).any():
        raise ValueError("duplicate ticker/date rows")
    panel = panel.sort_values(["ticker", "date"])
    panel["vega_demand"] = panel.tape_vega_flow / panel.tape_gross_premium.replace(0, np.nan)
    panel["gamma_demand"] = panel.tape_gamma_flow / panel.tape_gross_premium.replace(0, np.nan)

    dates = np.sort(panel.date.unique())
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
        entry = lookup.reindex(entry_index).to_numpy()
        exit_ = lookup.reindex(exit_index).to_numpy()
        panel[f"abs_{horizon}d"] = np.abs(exit_ / entry - 1.0)
        panel[f"move_ratio_{horizon}d"] = panel[f"abs_{horizon}d"] / panel.implied_move_perc.replace(0, np.nan)

    tape_dates = np.sort(panel.loc[panel.tape_gross_premium.notna(), "date"].unique())
    split = pd.Timestamp(tape_dates[len(tape_dates) // 2])
    rows = []
    for feature in ("vega_demand", "gamma_demand"):
        for horizon in (1, 5):
            for target in (f"abs_{horizon}d", f"move_ratio_{horizon}d"):
                daily = []
                for date, day in panel.groupby("date"):
                    day = day.dropna(subset=[feature, target]).copy()
                    if len(day) < MIN_NAMES or day[feature].nunique() < 10:
                        continue
                    rank = day[feature].rank(pct=True)
                    high = day.loc[rank >= 0.9, target].mean()
                    low = day.loc[rank <= 0.1, target].mean()
                    daily.append({"date": date, "high": high, "low": low, "spread": high - low})
                daily = pd.DataFrame(daily)
                for sample, frame in (
                    ("FULL", daily),
                    ("TRAIN", daily[daily.date < split]),
                    ("TEST", daily[daily.date >= split]),
                ):
                    rows.append(
                        {
                            "feature": feature,
                            "horizon": f"{horizon}d",
                            "target": target,
                            "sample": sample,
                            "days": len(frame),
                            "high": frame.high.mean(),
                            "low": frame.low.mean(),
                            "spread": frame.spread.mean(),
                            "nw_t": nw_t(frame.spread, horizon),
                            "hit_rate": (frame.spread > 0).mean(),
                        }
                    )
    result = pd.DataFrame(rows)
    result.to_csv(OUT, index=False)
    print(f"tape_dates={len(tape_dates)} split={split.date()}")
    print(result[result["sample"] == "TEST"].sort_values("nw_t", ascending=False).round(4).to_string(index=False))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
