"""Small, pre-declared 1d/5d hypothesis test over the five UW feeds.

No family mining. Each feature is tied to a market-microstructure hypothesis.
Rules are fixed on the first half, then reported on the untouched second half.
Daily long-short portfolios are equal-weighted; Newey-West errors correct overlap.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PANEL = Path("/Users/anuppamvi/uw_root/tradedesk/out/uw_all_feeds.csv")
OPENING_FLOW = Path("/Users/anuppamvi/uw_root/tradedesk/out/opening_flow_features.csv")
SCREENER_FLOW = Path("/Users/anuppamvi/uw_root/tradedesk/out/screener_flow_features.csv")
OUT = Path("/Users/anuppamvi/uw_root/tradedesk/out/short_horizon_hypotheses.csv")
MIN_MCAP = 2e9
MIN_NAMES = 100
QUANTILE = 0.10
ROUND_TRIP_COST_PER_LEG = 0.0005

# Positive score means buy high-score / sell low-score.
HYPOTHESES = {
    # Dark-pool prints above the midpoint are often dealer short sales against
    # non-aggressive institutional buying; the observed next-day sign is reversal.
    "dp_late_reversal": ("dp_late_bias", -1.0),
    "dp_all_reversal": ("dp_bias", -1.0),
    "dp_block_reversal": ("dp_block_bias", -1.0),
    # Buyer-initiated opening option flow should predict the underlying direction.
    "oi_open_direction": ("oi_dir_bias", 1.0),
    "oi_signed_premium": ("oi_signed_premium", 1.0),
    "oi_direction_shrunk": ("oi_direction_shrunk", 1.0),
    "oi_direction_nearmoney": ("oi_direction_nearmoney", 1.0),
    "buyer_open_low_pcr": ("buyer_open_pcr", -1.0),
    "buyer_open_direction": ("buyer_open_direction", 1.0),
    "seller_open_direction": ("seller_open_direction", 1.0),
    "oi_chain_breadth": ("oi_chain_breadth", 1.0),
    "oi_chain_breadth_shrunk": ("oi_chain_breadth_shrunk", 1.0),
    "oi_near_chain_breadth_shrunk": ("oi_near_chain_breadth_shrunk", 1.0),
    "option_stock_volume_reversal": ("option_stock_volume_ratio", -1.0),
    "screener_flow_direction": ("screener_directional_volume_bias", 1.0),
    "buyer_low_put_call": ("buyer_put_call_ratio", -1.0),
    "hot_chain_direction": ("hc_dir_bias", 1.0),
    # Full tape customer delta is directional; customer vega may predict magnitude,
    # but here signed vega is tested only as a directional diagnostic.
    "tape_delta_direction": ("tape_delta_notional_xs", 1.0),
    "tape_premium_direction": ("tape_prem_bias", 1.0),
    # Volume shock / IV rank are common crowding hypotheses; signs are explicitly tested.
    "stock_volume_continuation": ("stock_vol_surge_xs", 1.0),
    "iv_rank_reversal": ("iv_rank_xs", -1.0),
}


def nw_t(values: np.ndarray, lag: int) -> float:
    values = values[np.isfinite(values)]
    if len(values) < 10:
        return np.nan
    centered = values - values.mean()
    n = len(centered)
    var = centered @ centered / n
    for step in range(1, min(lag, n - 1) + 1):
        covariance = centered[step:] @ centered[:-step] / n
        var += 2.0 * (1.0 - step / (lag + 1.0)) * covariance
    return values.mean() / np.sqrt(max(var, 1e-12) / n)


def daily_spreads(frame: pd.DataFrame, feature: str, sign: float, horizon: int) -> pd.DataFrame:
    rows = []
    target = f"fwd_{horizon}d"
    for date, group in frame.groupby("date"):
        sample = group[[feature, target]].dropna()
        if len(sample) < MIN_NAMES or sample[feature].nunique() < 10:
            continue
        score = sign * sample[feature]
        low_cut = score.quantile(QUANTILE)
        high_cut = score.quantile(1.0 - QUANTILE)
        high = sample.loc[score >= high_cut, target]
        low = sample.loc[score <= low_cut, target]
        if len(high) < 10 or len(low) < 10:
            continue
        rows.append(
            {
                "date": date,
                "long": high.mean(),
                "short": low.mean(),
                "spread": high.mean() - low.mean(),
                "long_n": len(high),
                "short_n": len(low),
            }
        )
    return pd.DataFrame(rows)


def summarize(name: str, horizon: int, sample: str, spreads: pd.DataFrame) -> dict:
    if spreads.empty:
        return {"hypothesis": name, "horizon": f"{horizon}d", "sample": sample, "days": 0}
    values = spreads["spread"].to_numpy()
    net_values = values - 2.0 * ROUND_TRIP_COST_PER_LEG
    return {
        "hypothesis": name,
        "horizon": f"{horizon}d",
        "sample": sample,
        "days": len(spreads),
        "long_return": spreads["long"].mean(),
        "short_return": spreads["short"].mean(),
        "long_short": values.mean(),
        "net_long_short": net_values.mean(),
        "nw_t": nw_t(values, horizon),
        "net_nw_t": nw_t(net_values, horizon),
        "hit_rate": (values > 0).mean(),
        "net_hit_rate": (net_values > 0).mean(),
        "worst_day": values.min(),
        "best_day": values.max(),
    }


def main() -> None:
    opening_features = {
        "buyer_open_pcr",
        "buyer_open_direction",
        "seller_open_direction",
        "oi_chain_breadth",
        "oi_chain_breadth_shrunk",
        "oi_near_chain_breadth_shrunk",
    }
    screener_features = {
        "option_stock_volume_ratio",
        "screener_directional_volume_bias",
        "buyer_put_call_ratio",
    }
    external_features = opening_features | screener_features
    derived_features = {"oi_direction_shrunk", "oi_direction_nearmoney"}
    usecols = ["date", "ticker", "close", "marketcap", "oi_built_premium", "oi_nearmoney_share"] + sorted(
        {feature for feature, _ in HYPOTHESES.values()} - external_features - derived_features
    )
    panel = pd.read_csv(PANEL, usecols=usecols, low_memory=False)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel[panel["marketcap"].fillna(0) >= MIN_MCAP]
    opening = pd.read_csv(
        OPENING_FLOW,
        usecols=["date", "ticker"] + sorted(opening_features),
        low_memory=False,
    )
    opening["date"] = pd.to_datetime(opening["date"])
    panel = panel.merge(opening, on=["date", "ticker"], how="left", validate="one_to_one")
    screener = pd.read_csv(
        SCREENER_FLOW,
        usecols=["date", "ticker"] + sorted(screener_features),
        low_memory=False,
    )
    screener["date"] = pd.to_datetime(screener["date"])
    panel = panel.merge(screener, on=["date", "ticker"], how="left", validate="one_to_one")
    daily_median_premium = panel.groupby("date")["oi_built_premium"].transform("median").clip(lower=1.0)
    evidence_weight = panel["oi_built_premium"] / (panel["oi_built_premium"] + daily_median_premium)
    panel["oi_direction_shrunk"] = panel["oi_dir_bias"] * evidence_weight
    panel["oi_direction_nearmoney"] = (
        panel["oi_direction_shrunk"] * panel["oi_nearmoney_share"].fillna(0.0)
    )
    panel = panel.sort_values(["ticker", "date"])
    if panel.duplicated(["ticker", "date"]).any():
        raise ValueError("panel contains duplicate ticker/date rows")

    # The EOD files are not available until after the signal-date close. Enter
    # at the next session's close and exit h trading sessions later. This also
    # uses exact calendar dates, rather than treating a missing ticker row as a
    # trading session.
    dates = np.sort(panel["date"].unique())
    date_position = {pd.Timestamp(date): index for index, date in enumerate(dates)}
    close_lookup = panel.set_index(["ticker", "date"])["close"]
    for horizon in (1, 5):
        entry_dates = panel["date"].map(
            lambda date: dates[date_position[date] + 1]
            if date_position[date] + 1 < len(dates)
            else pd.NaT
        )
        exit_dates = panel["date"].map(
            lambda date: dates[date_position[date] + horizon + 1]
            if date_position[date] + horizon + 1 < len(dates)
            else pd.NaT
        )
        entry_index = pd.MultiIndex.from_arrays([panel["ticker"], entry_dates])
        exit_index = pd.MultiIndex.from_arrays([panel["ticker"], exit_dates])
        entry_close = close_lookup.reindex(entry_index).to_numpy()
        exit_close = close_lookup.reindex(exit_index).to_numpy()
        panel[f"fwd_{horizon}d"] = exit_close / entry_close - 1.0
    split_date = pd.Timestamp(dates[len(dates) // 2])
    print(f"rows={len(panel)} dates={len(dates)} split={split_date.date()}")

    rows = []
    for name, (feature, sign) in HYPOTHESES.items():
        for horizon in (1, 5):
            spreads = daily_spreads(panel, feature, sign, horizon)
            rows.append(summarize(name, horizon, "FULL", spreads))
            rows.append(summarize(name, horizon, "TRAIN", spreads[spreads.date < split_date]))
            rows.append(summarize(name, horizon, "TEST", spreads[spreads.date >= split_date]))

    result = pd.DataFrame(rows)
    result.to_csv(OUT, index=False)
    test = result[(result["sample"] == "TEST") & (result["days"] >= 20)].copy()
    test["abs_t"] = test["nw_t"].abs()
    print("\n=== UNTOUCHED SECOND HALF, sorted by |Newey-West t| ===")
    print(test.sort_values("abs_t", ascending=False).head(30).round(4).to_string(index=False))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
