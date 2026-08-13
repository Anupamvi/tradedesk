"""Does market-level (SPY) dealer-gamma regime separate winning vs losing tape?

Join SPY net_gex_norm per day onto the replay detail, bucket days by SPY gamma
magnitude, and compare trade PF across buckets. Local UW replay only.
"""
import sys
import numpy as np
import pandas as pd

DETAIL = ("out/options_agent_independent_replay/v1_56_live_selector_dte_parity_ytd_full/"
          "options_agent_replay_detail.csv")

df = pd.read_csv(DETAIL, low_memory=False)
df = df[df["next_session_reprice_approved"].astype(str).str.lower().isin(["true", "1", "yes"])].copy()
df["pnl_1x"] = pd.to_numeric(df["pnl_1x"], errors="coerce")
df = df[df["pnl_1x"].notna()].copy()
df["asof"] = df["asof"].astype(str)

gex = pd.read_csv("gex_reconstructed.csv")
gex["asof"] = gex["asof"].astype(str)
spy = gex[gex["ticker"] == "SPY"][["asof", "net_gex_norm", "abs_gex"]].rename(
    columns={"net_gex_norm": "spy_gex", "abs_gex": "spy_absgex"})
qqq = gex[gex["ticker"] == "QQQ"][["asof", "net_gex_norm"]].rename(columns={"net_gex_norm": "qqq_gex"})
df = df.merge(spy, on="asof", how="left").merge(qqq, on="asof", how="left")
df = df[df["spy_gex"].notna()].copy()


def pf(pnl):
    pos = pnl[pnl > 0].sum(); neg = -pnl[pnl < 0].sum()
    return pos / neg if neg > 0 else float("inf")


print(f"rows with SPY GEX: {len(df)}  days: {df['asof'].nunique()}\n")

# SPY gamma is always negative (dealer short). Magnitude = how short = how unstable.
df["spy_mag"] = df["spy_gex"].abs()
print("=== Buckets by |SPY net_gex_norm| (more negative = more dealer short gamma = trendier) ===")
df["bkt"] = pd.qcut(df["spy_mag"], 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"])
for b, g in df.groupby("bkt", observed=True):
    print(f"{b:>8}  n={len(g):4d}  win={ (g['pnl_1x']>0).mean():.0%}  avgP/L={g['pnl_1x'].mean():7.2f}  PF={pf(g['pnl_1x']):.2f}  total={g['pnl_1x'].sum():.0f}")

print("\n=== Same, split pre/post 2026-05-01 (OOS check) ===")
for lbl, mask in [("PRE ", df["asof"] < "2026-05-01"), ("POST", df["asof"] >= "2026-05-01")]:
    sub = df[mask]
    hi = sub[sub["spy_mag"] >= df["spy_mag"].median()]
    lo = sub[sub["spy_mag"] < df["spy_mag"].median()]
    print(f"{lbl} high-mag n={len(hi):4d} PF={pf(hi['pnl_1x']):.2f} | low-mag n={len(lo):4d} PF={pf(lo['pnl_1x']):.2f}")

print("\n=== Correlation: does spy_mag rank days by profitability? ===")
day = df.groupby("asof").agg(spy_mag=("spy_mag", "first"), pnl=("pnl_1x", "sum"), n=("pnl_1x", "size"))
print(f"day-level corr(spy_mag, day_total_pnl) = {day['spy_mag'].corr(day['pnl']):.3f}")
print(f"day-level corr(spy_mag, day_mean_pnl)  = {day['spy_mag'].corr(day['pnl']/day['n']):.3f}")
