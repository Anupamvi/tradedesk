"""Information coefficient of every UW feed signal against forward STOCK returns.

Answers the prior question the pipeline never asked: is there predictive signal
in these feeds at all, before deciding how to express it as an option trade?

Method: per-day cross-sectional Spearman rank IC vs forward 1d/5d/21d stock
returns, then a t-stat across days (each day is one observation, so overlapping
forward windows do not inflate significance the way pooled rows would).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PANEL = Path("/Users/anuppamvi/uw_root/tradedesk/out/uw_all_feeds.csv")
HORIZONS = (1, 5, 21)
MIN_NAMES_PER_DAY = 30

SKIP = {
    "date",
    "ticker",
    "sector",
    "issue_type",
    "next_earnings_date",
    "close",
    "marketcap",
}


def main() -> int:
    panel = pd.read_csv(PANEL, low_memory=False)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["ticker", "date"])

    # Forward stock returns per ticker.
    for h in HORIZONS:
        panel[f"fwd_{h}d"] = (
            panel.groupby("ticker")["close"].shift(-h) / panel["close"] - 1.0
        )

    # Liquid, real equities only: avoid microcaps where the IC is noise.
    panel = panel[panel["marketcap"].fillna(0) > 2e9]

    signals = [
        c
        for c in panel.columns
        if c not in SKIP
        and not c.startswith("fwd_")
        and pd.api.types.is_numeric_dtype(panel[c])
    ]
    print(f"panel rows={len(panel)} days={panel.date.nunique()} signals={len(signals)}")

    rows = []
    for sig in signals:
        for h in HORIZONS:
            daily = []
            for _, g in panel.groupby("date"):
                sub = g[[sig, f"fwd_{h}d"]].dropna()
                if len(sub) < MIN_NAMES_PER_DAY or sub[sig].nunique() < 5:
                    continue
                ic = stats.spearmanr(sub[sig], sub[f"fwd_{h}d"]).correlation
                if np.isfinite(ic):
                    daily.append(ic)
            if len(daily) < 20:
                continue
            arr = np.array(daily)
            # Newey-West: consecutive daily ICs share h-1 days of the same
            # forward window, so treating them as independent overstates t by
            # roughly sqrt(h). Correct for it.
            x = arr - arr.mean()
            n = len(x)
            var = (x @ x) / n
            for lag in range(1, min(h, n - 1) + 1):
                var += 2.0 * (1.0 - lag / (h + 1.0)) * ((x[lag:] @ x[:-lag]) / n)
            se = np.sqrt(max(var, 1e-12) / n)
            t = arr.mean() / se
            rows.append(
                {
                    "signal": sig,
                    "horizon": f"{h}d",
                    "n_days": len(arr),
                    "mean_ic": arr.mean(),
                    "t_stat_NW": t,
                    "hit_rate": (arr > 0).mean(),
                }
            )

    out = pd.DataFrame(rows)
    out["abs_t"] = out.t_stat_NW.abs()
    out = out.sort_values("abs_t", ascending=False)
    dest = Path("/Users/anuppamvi/uw_root/tradedesk/out/stock_level_ic.csv")
    out.to_csv(dest, index=False)

    print("\n=== TOP 25 SIGNALS BY |t| (forward STOCK return IC) ===")
    print(out.head(25).round(4).to_string(index=False))
    print(f"\nwrote {dest}")
    print(f"\nsignals with |t| > 3: {(out.abs_t > 3).sum()} of {len(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
