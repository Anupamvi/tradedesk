"""Falsification suite for the OI-confirmed directional-flow pattern."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PANEL = Path("/Users/anuppamvi/uw_root/tradedesk/out/uw_all_feeds.csv")
OUT = Path("/Users/anuppamvi/uw_root/tradedesk/out/oi_direction_validation.csv")
HORIZON = 5
MIN_MCAP = 2e9


def nw_t(values: pd.Series, lag: int = HORIZON) -> float:
    x = values.dropna().to_numpy()
    if len(x) < 10:
        return np.nan
    centered = x - x.mean()
    n = len(centered)
    variance = centered @ centered / n
    for step in range(1, min(lag, n - 1) + 1):
        covariance = centered[step:] @ centered[:-step] / n
        variance += 2.0 * (1.0 - step / (lag + 1.0)) * covariance
    return x.mean() / np.sqrt(max(variance, 1e-12) / n)


def load_panel() -> tuple[pd.DataFrame, pd.Timestamp]:
    columns = [
        "date", "ticker", "sector", "issue_type", "marketcap", "close",
        "oi_dir_bias", "oi_built_premium", "oi_n_chains", "oi_nearmoney_share",
    ]
    panel = pd.read_csv(PANEL, usecols=columns, low_memory=False)
    panel["date"] = pd.to_datetime(panel["date"])
    spy_close = panel.loc[panel["ticker"] == "SPY"].set_index("date")["close"]
    panel = panel[
        (panel["marketcap"].fillna(0) >= MIN_MCAP)
        & (panel["issue_type"] == "Common Stock")
    ].copy()
    if panel.duplicated(["ticker", "date"]).any():
        raise ValueError("duplicate ticker/date rows in research universe")
    panel = panel.sort_values(["ticker", "date"])
    dates = np.sort(panel["date"].unique())
    positions = {pd.Timestamp(date): index for index, date in enumerate(dates)}
    lookup = panel.set_index(["ticker", "date"])["close"]
    entry_dates = panel["date"].map(
        lambda date: dates[positions[date] + 1] if positions[date] + 1 < len(dates) else pd.NaT
    )
    exit_dates = panel["date"].map(
        lambda date: dates[positions[date] + HORIZON + 1]
        if positions[date] + HORIZON + 1 < len(dates)
        else pd.NaT
    )
    entry_index = pd.MultiIndex.from_arrays([panel["ticker"], entry_dates])
    exit_index = pd.MultiIndex.from_arrays([panel["ticker"], exit_dates])
    panel["return"] = lookup.reindex(exit_index).to_numpy() / lookup.reindex(entry_index).to_numpy() - 1.0
    panel["spy_return"] = (
        spy_close.reindex(exit_dates).to_numpy()
        / spy_close.reindex(entry_dates).to_numpy()
        - 1.0
    )
    split = pd.Timestamp(dates[len(dates) // 2])
    return panel, split


def portfolio_rows(frame: pd.DataFrame, sector_neutral: bool = False) -> pd.DataFrame:
    rows = []
    for date, day in frame.groupby("date"):
        day = day.dropna(subset=["oi_dir_bias", "return"]).copy()
        if len(day) < 100:
            continue
        if sector_neutral:
            selected = []
            for _, sector in day.groupby("sector"):
                if len(sector) < 20 or sector["oi_dir_bias"].nunique() < 10:
                    continue
                rank = sector["oi_dir_bias"].rank(pct=True, method="average")
                long_ret = sector.loc[rank >= 0.9, "return"].mean()
                short_ret = sector.loc[rank <= 0.1, "return"].mean()
                selected.append((long_ret, short_ret))
            if len(selected) < 5:
                continue
            long_ret = np.mean([item[0] for item in selected])
            short_ret = np.mean([item[1] for item in selected])
        else:
            rank = day["oi_dir_bias"].rank(pct=True, method="average")
            long_ret = day.loc[rank >= 0.9, "return"].mean()
            short_ret = day.loc[rank <= 0.1, "return"].mean()
        benchmark_return = day["spy_return"].iloc[0]
        universe_return = day["return"].mean()
        rows.append(
            {
                "date": date,
                "long": long_ret,
                "short": short_ret,
                "spread": long_ret - short_ret,
                "spy_return": benchmark_return,
                "long_spy_alpha": long_ret - benchmark_return,
                "long_universe_alpha": long_ret - universe_return,
            }
        )
    return pd.DataFrame(rows)


def summary(label: str, rows: pd.DataFrame) -> dict:
    spread = rows["spread"]
    return {
        "test": label,
        "days": len(rows),
        "long": rows["long"].mean(),
        "short": rows["short"].mean(),
        "spread": spread.mean(),
        "nw_t": nw_t(spread),
        "hit_rate": (spread > 0).mean(),
        "long_spy_alpha": rows["long_spy_alpha"].mean(),
        "long_spy_alpha_nw_t": nw_t(rows["long_spy_alpha"]),
        "long_universe_alpha": rows["long_universe_alpha"].mean(),
        "long_universe_alpha_nw_t": nw_t(rows["long_universe_alpha"]),
        "break_even_cost_per_leg_bps": spread.mean() * 10000 / 2.0,
    }


def main() -> None:
    panel, split = load_panel()
    train = panel[panel.date < split]
    test = panel[panel.date >= split]
    summaries = []
    portfolios: dict[str, pd.DataFrame] = {}

    filters = {
        "all": lambda frame: frame,
        "premium_above_daily_median": lambda frame: frame[
            frame["oi_built_premium"] >= frame.groupby("date")["oi_built_premium"].transform("median")
        ],
        "at_least_10_chains": lambda frame: frame[frame["oi_n_chains"] >= 10],
        "near_money_share_ge_25pct": lambda frame: frame[frame["oi_nearmoney_share"] >= 0.25],
    }
    for sample_name, sample in (("TRAIN", train), ("TEST", test), ("FULL", panel)):
        for filter_name, apply_filter in filters.items():
            filtered = apply_filter(sample)
            for sector_neutral in (False, True):
                key = f"{sample_name}:{filter_name}:{'sector_neutral' if sector_neutral else 'global'}"
                rows = portfolio_rows(filtered, sector_neutral=sector_neutral)
                portfolios[key] = rows
                summaries.append(summary(key, rows))

    result = pd.DataFrame(summaries)
    result.to_csv(OUT, index=False)
    print(f"split={split.date()}\n")
    print("=== UNTOUCHED TEST HALF ===")
    print(result[result.test.str.startswith("TEST")].sort_values("nw_t", ascending=False).round(4).to_string(index=False))

    base = portfolios["TEST:all:global"].copy()
    base["month"] = base["date"].dt.to_period("M").astype(str)
    print("\n=== TEST-HALF MONTHS: GLOBAL DECILE SPREAD ===")
    print(base.groupby("month").agg(days=("spread", "size"), spread=("spread", "mean"), hit=("spread", lambda x: (x > 0).mean())).round(4).to_string())

    print("\n=== FIVE NON-OVERLAPPING ENTRY SCHEDULES ===")
    base = base.sort_values("date").reset_index(drop=True)
    offset_rows = []
    for offset in range(HORIZON):
        cohort = base.iloc[offset::HORIZON]
        offset_rows.append({
            "offset": offset,
            "periods": len(cohort),
            "spread": cohort.spread.mean(),
            "t": cohort.spread.mean() / (cohort.spread.std(ddof=1) / np.sqrt(len(cohort))),
            "hit": (cohort.spread > 0).mean(),
        })
    print(pd.DataFrame(offset_rows).round(4).to_string(index=False))

    print("\n=== TEST-HALF MONOTONICITY (daily decile returns) ===")
    deciles = []
    for date, day in test.groupby("date"):
        day = day.dropna(subset=["oi_dir_bias", "return"]).copy()
        if len(day) < 100 or day.oi_dir_bias.nunique() < 10:
            continue
        day["decile"] = pd.qcut(day.oi_dir_bias.rank(method="first"), 10, labels=False)
        daily = day.groupby("decile")["return"].mean()
        for decile, value in daily.items():
            deciles.append({"date": date, "decile": decile + 1, "return": value})
    print(pd.DataFrame(deciles).groupby("decile")["return"].agg(["mean", "count"]).round(4).to_string())
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
