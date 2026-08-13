"""Decisive test of the one two-sided effect the VRP scan surfaced.

Within-day cross-sectional ranking removes market-wide term structure and regime,
so the q5-q1 SPREAD is the clean statistic, not the level.

H (pre-registered, sign stated before testing):
  names whose 30d IV rose most over the past week subsequently realize a SMALLER
  fraction of that implied than names whose IV rose least  ->  spread < 0.
"""
from __future__ import annotations

import argparse
import math
import numpy as np
import pandas as pd

E_ABS = math.sqrt(2.0 / math.pi)
VOL_ETPS = {"UVXY", "VXX", "SVIX", "SVXY", "VIXY", "UVIX", "VIXM", "VXZ",
            "SQQQ", "TQQQ", "SOXL", "SOXS", "SPXU", "UPRO", "TNA", "TZA"}


def build(panel: str, min_mcap: float, horizon: int, feature: str) -> pd.DataFrame:
    keep = {"date", "ticker", "sector", "issue_type", "marketcap", "close",
            "iv30d", "iv_rank", "next_earnings_date", feature}
    df = pd.read_csv(panel, usecols=lambda c: c in keep, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["ticker"].notna()]
    df = df[~df["ticker"].astype(str).str.upper().isin(VOL_ETPS)]
    df = df[df["issue_type"].astype(str).str.contains("Common", case=False, na=False)]
    df = df[pd.to_numeric(df["marketcap"], errors="coerce").fillna(0) >= min_mcap]
    df = df[pd.to_numeric(df["iv30d"], errors="coerce") > 0.01]
    df = df.sort_values(["ticker", "date"])

    g = df.groupby("ticker", sort=False)["close"]
    entry = g.shift(-1)
    exit_ = g.shift(-(1 + horizon))
    df["abs_move"] = (exit_ / entry - 1.0).abs()
    df["implied"] = pd.to_numeric(df["iv30d"], errors="coerce") * math.sqrt(horizon / 252.0)
    df["ratio"] = df["abs_move"] / (df["implied"] * E_ABS)

    if "next_earnings_date" in df.columns:
        ern = pd.to_datetime(df["next_earnings_date"], errors="coerce")
        df["earn_dte"] = (ern - df["date"]).dt.days
    else:
        df["earn_dte"] = np.nan

    df = df[df["ratio"].notna() & np.isfinite(df["ratio"]) & df[feature].notna()]
    df["ratio"] = df["ratio"].clip(upper=df["ratio"].quantile(0.995))
    return df


def quintile(df: pd.DataFrame, feature: str, n: int = 5) -> pd.DataFrame:
    r = df.groupby("date")[feature].rank(pct=True, method="average")
    return df.assign(q=np.ceil(r * n).clip(1, n).astype(int))


def daily_spread(df: pd.DataFrame, n: int = 5) -> pd.Series:
    """One observation per day: mean(ratio | q5) - mean(ratio | q1)."""
    piv = df.groupby(["date", "q"])["ratio"].mean().unstack("q")
    if n not in piv.columns or 1 not in piv.columns:
        return pd.Series(dtype=float)
    return (piv[n] - piv[1]).dropna()


def report(name: str, s: pd.Series) -> None:
    if s.empty:
        print(f"  {name:<34} no data")
        return
    m, sd, k = s.mean(), s.std(ddof=1), len(s)
    t = m / (sd / math.sqrt(k)) if sd > 0 else float("nan")
    rng = np.random.default_rng(11)
    boot = [np.mean(rng.choice(s.to_numpy(), k, replace=True)) for _ in range(2000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"  {name:<34} spread {m:+.4f}  t {t:+.2f}  95% [{lo:+.4f},{hi:+.4f}]  "
          f"days {k}  days<0 {(s < 0).mean():.0%}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="/Users/anuppamvi/uw_root/tradedesk/out/uw_all_feeds.csv")
    ap.add_argument("--feature", default="iv_chg_1w")
    ap.add_argument("--horizon", type=int, default=21)
    ap.add_argument("--min-mcap", type=float, default=2e9)
    args = ap.parse_args()

    df = build(args.panel, args.min_mcap, args.horizon, args.feature)
    df = quintile(df, args.feature)
    print(f"feature={args.feature}  horizon={args.horizon}d  rows={len(df):,}  "
          f"days={df['date'].nunique()}  tickers={df['ticker'].nunique()}")

    print("\nQUINTILE MEANS (within-day rank of the feature)")
    agg = df.groupby("q")["ratio"].agg(["mean", "median", "count"])
    for q, r in agg.iterrows():
        print(f"  q{q}  realized/implied {r['mean']:.4f}  median {r['median']:.4f}  n {int(r['count']):,}")

    print("\nDAY-CLUSTERED q5-q1 SPREAD  (one observation per trading day)")
    report("ALL", daily_spread(df))

    print("\nSTABILITY BY MONTH")
    for mo, sub in df.groupby(df["date"].dt.to_period("M")):
        report(str(mo), daily_spread(sub))

    print("\nSTABILITY BY SECTOR (>=20 days)")
    for sec, sub in df.groupby("sector"):
        s = daily_spread(sub)
        if len(s) >= 20:
            report(str(sec)[:32], s)

    print("\nEARNINGS CONFOUND: is the effect just IV rising into a print?")
    near = df[df["earn_dte"].between(0, args.horizon + 5)]
    far = df[~df["earn_dte"].between(0, args.horizon + 5)]
    report(f"earnings within {args.horizon + 5}d", daily_spread(near))
    report("no earnings in window", daily_spread(far))

    print("\nIS IT JUST HIGH-IV NAMES? (double sort: within iv_rank tercile)")
    ivr = df.groupby("date")["iv_rank"].rank(pct=True, method="average")
    df3 = df.assign(ivt=np.ceil(ivr * 3).clip(1, 3).astype(int))
    for t3, sub in df3.groupby("ivt"):
        sub2 = quintile(sub.drop(columns=["q"]), args.feature)
        report(f"iv_rank tercile {t3}", daily_spread(sub2))


if __name__ == "__main__":
    main()
