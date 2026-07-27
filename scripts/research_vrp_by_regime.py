"""Does the variance-risk-premium edge depend on trend regime?

The live credit policy blocks `range` sessions in both directions. That map was
derived from directional P&L under the OLD stop-loss exit policy. Since the edge
is now understood to be volatility richness (direction was shown to be
unpredictable), this re-tests whether premium capture actually varies by regime.

Regime here is reconstructed per (asof, ticker) from the research price panel so
it is point-in-time: the 21d trend of the underlying, normalised by its own
realised vol, bucketed the same way the pipeline buckets uptrend/range/downtrend.
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

PANEL = "/Users/anuppamvi/uw_root/tradedesk/out/research/price_panel.csv.gz"


def load(min_dollar_vol: float, min_mcap: float) -> pd.DataFrame:
    d = pd.read_csv(PANEL, low_memory=False)
    d = d[(d["dollar_vol"] >= min_dollar_vol) & (d["marketcap"] >= min_mcap)]
    d = d.dropna(subset=["vrp_ratio", "vrp_capture", "rv21_ann", "iv30d"]).copy()

    # point-in-time earnings exclusion, matching MAX_DTE_EARNINGS_EXCLUSION
    ed = pd.to_datetime(d.get("next_earnings_date"), errors="coerce")
    ao = pd.to_datetime(d["asof"], errors="coerce")
    days = (ed - ao).dt.days
    d = d[~((days >= 0) & (days <= 21))]
    return d


def add_regime(d: pd.DataFrame, band: float) -> pd.DataFrame:
    """Trend strength = 21d return / (daily vol * sqrt(21)), i.e. how many sigma
    the underlying has travelled. Same construction as the pipeline's ax_P input."""
    d = d.sort_values(["ticker", "asof"]).copy()
    g = d.groupby("ticker", sort=False)
    ret21 = g["close"].transform(lambda s: s.pct_change(21, fill_method=None))
    denom = d["adm21"] * np.sqrt(21.0)
    d["trend_sigma"] = ret21 / denom.replace(0.0, np.nan)
    d["regime"] = np.where(
        d["trend_sigma"] >= band,
        "uptrend",
        np.where(d["trend_sigma"] <= -band, "downtrend", "range"),
    )
    return d.dropna(subset=["trend_sigma"])


def report(d: pd.DataFrame, label: str) -> None:
    print(f"\n=== {label} ===")
    print(f"{'regime':<11}{'n':>7}{'capture':>10}{'median':>9}{'win%':>8}{'p05':>9}{'iv':>7}")
    for reg in ["uptrend", "range", "downtrend"]:
        s = d[d["regime"] == reg]
        if s.empty:
            continue
        c = s["vrp_capture"]
        print(
            f"{reg:<11}{len(s):>7}{c.mean():>+10.4f}{c.median():>+9.4f}"
            f"{100*(c>0).mean():>7.1f}%{c.quantile(0.05):>+9.4f}{s['iv30d'].mean():>7.3f}"
        )


def monthly(d: pd.DataFrame, reg: str) -> None:
    s = d[d["regime"] == reg]
    if s.empty:
        return
    m = s.groupby(s["asof"].str[:7])["vrp_capture"].agg(
        ["size", "mean", lambda x: (x > 0).mean()]
    )
    m.columns = ["n", "cap", "win"]
    txt = " | ".join(f"{k} {r['cap']:+.3f}/{100*r['win']:.0f}%" for k, r in m.iterrows())
    print(f"  {reg} by month: {txt}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-ratio", type=float, default=1.30)
    ap.add_argument("--min-rv", type=float, default=0.15)
    ap.add_argument("--band", type=float, default=0.5, help="sigma band for range")
    ap.add_argument("--min-dollar-vol", type=float, default=25e6)
    ap.add_argument("--min-mcap", type=float, default=1e9)
    args = ap.parse_args()

    d = load(args.min_dollar_vol, args.min_mcap)
    d = add_regime(d, args.band)
    print(f"universe {len(d):,} rows  days {d['asof'].nunique()}  tickers {d['ticker'].nunique()}")
    print(f"regime mix: {d['regime'].value_counts(normalize=True).round(3).to_dict()}")

    report(d, "ALL candidates (no richness gate)")

    rich = d[(d["vrp_ratio"] >= args.min_ratio) & (d["rv21_ann"] >= args.min_rv)]
    report(rich, f"RICH only: IV/HV>={args.min_ratio}, RV>={args.min_rv}")
    print()
    for reg in ["uptrend", "range", "downtrend"]:
        monthly(rich, reg)

    # Day-level: is the range spread reliable across sessions, not just pooled?
    print("\n--- per-day mean capture, RICH rows, by regime ---")
    for reg in ["uptrend", "range", "downtrend"]:
        s = rich[rich["regime"] == reg]
        if s.empty:
            continue
        daily = s.groupby("asof")["vrp_capture"].mean()
        print(
            f"  {reg:<10} days {len(daily):>4}  mean {daily.mean():+.4f}  "
            f"positive on {100*(daily>0).mean():.1f}% of days"
        )


if __name__ == "__main__":
    main()
