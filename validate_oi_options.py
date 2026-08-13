"""Map prior-day OI-direction ranks to next-day executable option outcomes."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/anuppamvi/uw_root/tradedesk")
PANEL = ROOT / "out/uw_all_feeds.csv"
DETAILS = ROOT / "out/pattern_analysis_v2/2026-07-24/validation_details.csv"
OUT = ROOT / "out/oi_direction_option_validation.csv"
OOS_START = pd.Timestamp("2026-04-14")


def profit_factor(values: pd.Series) -> float:
    gains = values[values > 0].sum()
    losses = -values[values < 0].sum()
    return gains / losses if losses > 0 else np.nan


def summarize(label: str, frame: pd.DataFrame) -> dict:
    values = pd.to_numeric(frame["net_r"], errors="coerce").dropna()
    return {
        "cohort": label,
        "trades": len(values),
        "days": frame.loc[values.index, "signal_date"].nunique(),
        "tickers": frame.loc[values.index, "ticker"].nunique(),
        "avg_r": values.mean(),
        "median_r": values.median(),
        "win_rate": (values > 0).mean(),
        "profit_factor": profit_factor(values),
        "sum_r": values.sum(),
    }


def main() -> None:
    panel = pd.read_csv(
        PANEL,
        usecols=["date", "ticker", "issue_type", "marketcap", "oi_dir_bias"],
        low_memory=False,
    )
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel[
        (panel["issue_type"] == "Common Stock")
        & (panel["marketcap"].fillna(0) >= 2e9)
    ].copy()
    panel["oi_rank"] = panel.groupby("date")["oi_dir_bias"].rank(pct=True)
    dates = np.sort(panel["date"].unique())
    next_date = {pd.Timestamp(date): pd.Timestamp(dates[index + 1]) for index, date in enumerate(dates[:-1])}
    panel["signal_date"] = panel["date"].map(next_date)
    lagged = panel[["signal_date", "ticker", "oi_rank", "oi_dir_bias"]].dropna(subset=["signal_date"])

    details = pd.read_csv(DETAILS, low_memory=False)
    details["signal_date"] = pd.to_datetime(details["signal_date"])
    details = details[
        (details["status"] == "SCORED")
        & (details["horizon"] == "5d")
        & (details["signal_date"] >= OOS_START)
    ].copy()
    details = details.merge(lagged, on=["signal_date", "ticker"], how="inner", validate="many_to_one")

    # The same contract is emitted under several mined pattern-family labels.
    # Count the actual trade once.
    details["legs_key"] = details["legs_json"].fillna("")
    details = details.drop_duplicates(
        [
            "signal_date", "ticker", "direction", "strategy_kind",
            "lead_option_symbol", "legs_key",
        ]
    )

    rows = []
    for strategy in ("long_option", "credit_spread"):
        strategy_rows = details[details["strategy_kind"] == strategy]
        rows.append(summarize(f"{strategy}:all_bullish", strategy_rows[strategy_rows.direction == "bullish"]))
        rows.append(
            summarize(
                f"{strategy}:top_oi_bullish",
                strategy_rows[(strategy_rows.direction == "bullish") & (strategy_rows.oi_rank >= 0.9)],
            )
        )
        rows.append(
            summarize(
                f"{strategy}:bottom_oi_bearish",
                strategy_rows[(strategy_rows.direction == "bearish") & (strategy_rows.oi_rank <= 0.1)],
            )
        )
        rows.append(
            summarize(
                f"{strategy}:wrong_way_controls",
                strategy_rows[
                    ((strategy_rows.direction == "bearish") & (strategy_rows.oi_rank >= 0.9))
                    | ((strategy_rows.direction == "bullish") & (strategy_rows.oi_rank <= 0.1))
                ],
            )
        )

    result = pd.DataFrame(rows)
    result.to_csv(OUT, index=False)
    print("=== LAGGED OI RANK -> NEXT-DAY OPTION ENTRY, OOS ONLY ===")
    print(result.round(4).to_string(index=False))

    selected = details[
        ((details.direction == "bullish") & (details.oi_rank >= 0.9))
        | ((details.direction == "bearish") & (details.oi_rank <= 0.1))
    ].copy()
    selected["month"] = selected["signal_date"].dt.to_period("M").astype(str)
    print("\n=== SELECTED OPTION OUTCOMES BY MONTH / STRUCTURE ===")
    monthly = selected.groupby(["strategy_kind", "month"]).agg(
        trades=("net_r", "size"),
        avg_r=("net_r", "mean"),
        win_rate=("win", "mean"),
        sum_r=("net_r", "sum"),
    )
    print(monthly.round(4).to_string())
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
