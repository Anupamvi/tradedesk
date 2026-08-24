"""Trend persistence: do established movers keep moving, and does flow confirm?

Every earlier test used a 5-day horizon and cross-sectional deciles, which is
structurally blind to multi-month trends like MU +192%, SNDK +422%, INTC +134%.

This tests the opposite structure: rank on TRAILING strength, hold for weeks, and
ask whether options-flow escalation adds anything on top of price momentum.

Signs are never fitted here. Momentum is tested long-high/short-low as published;
flow escalation is tested as a conditioner, not a free parameter.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/anuppamvi/uw_root/tradedesk")
PANEL = ROOT / "out/uw_all_feeds.csv"
OUT = ROOT / "out/trend_persistence.csv"
SPLIT = pd.Timestamp("2026-04-14")
COST = 0.001  # 5 bps per leg round trip


def nw_t(values: pd.Series, lag: int) -> float:
    x = values.dropna().to_numpy()
    if len(x) < 10:
        return np.nan
    c = x - x.mean()
    n = len(c)
    var = c @ c / n
    for step in range(1, min(lag, n - 1) + 1):
        var += 2.0 * (1.0 - step / (lag + 1.0)) * (c[step:] @ c[:-step] / n)
    return x.mean() / np.sqrt(max(var, 1e-12) / n)


def main() -> None:
    columns = [
        "date", "ticker", "issue_type", "marketcap", "close", "pos_52w",
        "hc_premium", "oi_built_premium", "iv_rank",
    ]
    panel = pd.read_csv(PANEL, usecols=columns, low_memory=False)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel[
        (panel.issue_type == "Common Stock") & (panel.marketcap.fillna(0) >= 2e9)
    ].copy()
    panel = panel.sort_values(["ticker", "date"])

    grouped = panel.groupby("ticker")
    # Trailing strength, all strictly backward looking.
    panel["mom_20"] = grouped.close.pct_change(20)
    panel["mom_60"] = grouped.close.pct_change(60)
    # Flow escalation: today's premium against its own trailing month.
    panel["flow_avg_20"] = grouped.hc_premium.transform(lambda s: s.rolling(20, min_periods=10).mean())
    panel["flow_escalation"] = panel.hc_premium / panel.flow_avg_20.replace(0, np.nan)
    panel["oi_avg_20"] = grouped.oi_built_premium.transform(lambda s: s.rolling(20, min_periods=10).mean())
    panel["oi_escalation"] = panel.oi_built_premium / panel.oi_avg_20.replace(0, np.nan)

    dates = np.sort(panel.date.unique())
    position = {pd.Timestamp(d): i for i, d in enumerate(dates)}
    close = panel.set_index(["ticker", "date"])["close"]

    results = []
    for horizon in (10, 20, 40):
        entry = panel.date.map(lambda d: dates[position[d] + 1] if position[d] + 1 < len(dates) else pd.NaT)
        exit_ = panel.date.map(
            lambda d: dates[position[d] + horizon + 1] if position[d] + horizon + 1 < len(dates) else pd.NaT
        )
        entry_px = close.reindex(pd.MultiIndex.from_arrays([panel.ticker, entry])).to_numpy()
        exit_px = close.reindex(pd.MultiIndex.from_arrays([panel.ticker, exit_])).to_numpy()
        panel[f"fwd_{horizon}"] = exit_px / entry_px - 1.0

        for signal in ("mom_20", "mom_60", "pos_52w"):
            for conditioner in (None, "flow_escalation", "oi_escalation"):
                daily = []
                for date, day in panel.groupby("date"):
                    day = day.dropna(subset=[signal, f"fwd_{horizon}"])
                    if conditioner:
                        day = day.dropna(subset=[conditioner])
                        day = day[day[conditioner] >= 1.5]  # flow running hot vs its own month
                    if len(day) < 60:
                        continue
                    rank = day[signal].rank(pct=True)
                    long_ret = day.loc[rank >= 0.9, f"fwd_{horizon}"].mean()
                    short_ret = day.loc[rank <= 0.1, f"fwd_{horizon}"].mean()
                    daily.append(
                        {"date": date, "long": long_ret, "short": short_ret, "spread": long_ret - short_ret}
                    )
                if len(daily) < 20:
                    continue
                frame = pd.DataFrame(daily)
                for sample, subset in (
                    ("TRAIN", frame[frame.date < SPLIT]),
                    ("TEST", frame[frame.date >= SPLIT]),
                ):
                    if len(subset) < 20:
                        continue
                    gross = subset.spread
                    results.append(
                        {
                            "signal": signal,
                            "conditioner": conditioner or "none",
                            "horizon": f"{horizon}d",
                            "sample": sample,
                            "days": len(subset),
                            "long_leg": subset.long.mean(),
                            "short_leg": subset.short.mean(),
                            "spread": gross.mean(),
                            "net_spread": gross.mean() - COST,
                            "nw_t": nw_t(gross, horizon),
                            "hit_rate": gross.gt(0).mean(),
                        }
                    )

    result = pd.DataFrame(results)
    result.to_csv(OUT, index=False)
    test = result[result["sample"] == "TEST"].sort_values("nw_t", ascending=False)
    print("=== TREND PERSISTENCE, untouched second half ===")
    print(test.round(4).to_string(index=False))
    print("\n=== matching TRAIN rows for the top 5 ===")
    for _, row in test.head(5).iterrows():
        match = result[
            (result.signal == row.signal)
            & (result.conditioner == row.conditioner)
            & (result.horizon == row.horizon)
            & (result["sample"] == "TRAIN")
        ]
        if not match.empty:
            m = match.iloc[0]
            print(
                f"  {row.signal:<9} {row.conditioner:<16} {row.horizon:<4} "
                f"TRAIN spread={m.spread:+.4f} t={m.nw_t:+.2f} | TEST spread={row.spread:+.4f} t={row.nw_t:+.2f}"
            )
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
