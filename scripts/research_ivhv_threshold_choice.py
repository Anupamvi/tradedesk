"""How high should MIN_IV_HV_RATIO actually be?

The 1.30 threshold came from `vrp_capture`, which measures pure variance-premium
capture. Replayed vertical P&L does not reproduce it: inside the regime map the
ordering is non-monotone and a permutation test cannot distinguish the rich
subset from a random subset of the same size.

Trades entered on the same session share regime, macro and vol shocks, so they
are not independent. This resamples whole DAYS (block bootstrap) rather than
individual trades, which is the honest unit of observation, and reports the
threshold sweep with confidence intervals plus the cost in trade count.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HISTORY = "codexuw/history/codexdaily_v4_edge_history_v4_2026-07-26.csv.gz"
PANEL = "/Users/anuppamvi/uw_root/tradedesk/out/research/price_panel.csv.gz"
RNG = np.random.default_rng(11)
BOOT = 3000


def pf(p: pd.Series) -> float:
    w, l = p[p > 0].sum(), -p[p < 0].sum()
    return float(w / l) if l > 0 else float("inf")


def load() -> pd.DataFrame:
    h = pd.read_csv(HISTORY, low_memory=False)
    h = h[(h["evaluated"] == True) & h["pnl_1x"].notna()]  # noqa: E712
    h = h[h["strategy_kind"] == "Credit"].copy()
    panel = pd.read_csv(PANEL, usecols=["asof", "ticker", "rv21_ann"], low_memory=False)
    panel = panel.dropna(subset=["rv21_ann"]).drop_duplicates(["asof", "ticker"])
    h["asof"] = h["asof"].astype(str)
    panel["asof"] = panel["asof"].astype(str)
    m = h.merge(panel, on=["asof", "ticker"], how="left")
    iv = pd.to_numeric(m["iv30d"], errors="coerce")
    rv = pd.to_numeric(m["rv21_ann"], errors="coerce")
    m["rv_true"] = rv
    m["ratio"] = iv / rv.where(rv > 0)
    m["allowed"] = (((m["direction"] == "Bull Put") & (m["regime"] == "downtrend"))
                    | ((m["direction"] == "Bear Call") & (m["regime"] == "uptrend")))
    return m[m["ratio"].notna() & m["rv_true"].notna()]


def day_bootstrap(d: pd.DataFrame) -> tuple:
    """Resample sessions with replacement; a day is the independent unit."""
    days = d["asof"].unique()
    by_day = {k: v["pnl_1x"].to_numpy() for k, v in d.groupby("asof")}
    means = np.empty(BOOT)
    for i in range(BOOT):
        pick = RNG.choice(days, len(days), replace=True)
        means[i] = np.concatenate([by_day[k] for k in pick]).mean()
    return float(np.percentile(means, 5)), float(np.percentile(means, 95)), float((means <= 0).mean())


def main() -> None:
    m = load()
    a = m[m["allowed"] & (m["rv_true"] >= 0.15)]
    print(f"map-allowed credit trades with realised vol >= 0.15: {len(a):,}  days {a['asof'].nunique()}")

    print("\n=== MIN_IV_HV_RATIO sweep INSIDE the regime map (day-clustered CI) ===")
    print(f"{'threshold':<11}{'n':>5}{'kept%':>7}{'days':>6}{'win%':>7}{'avg':>8}"
          f"{'PF':>6}{'total':>8}{'  90% CI on avg':>22}{'p(avg<=0)':>11}")
    for thr in [0.00, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30, 1.40, 1.50]:
        s = a[a["ratio"] >= thr]
        if len(s) < 25:
            continue
        p = s["pnl_1x"]
        lo, hi, pneg = day_bootstrap(s)
        print(f"{thr:<11.2f}{len(s):>5}{100*len(s)/len(a):>6.0f}%{s['asof'].nunique():>6}"
              f"{100*(p>0).mean():>6.1f}%{p.mean():>+8.1f}{pf(p):>6.2f}{p.sum():>+8.0f}"
              f"   [{lo:>+7.1f}, {hi:>+7.1f}]{pneg:>11.3f}")

    print("\n=== same sweep, but does it BEAT no-threshold? ===")
    print("  difference in mean P&L vs the full map-allowed pool, day-clustered")
    base_days = {k: v["pnl_1x"].to_numpy() for k, v in a.groupby("asof")}
    days = a["asof"].unique()
    for thr in [0.90, 1.00, 1.15, 1.30, 1.50]:
        s = a[a["ratio"] >= thr]
        if len(s) < 25:
            continue
        sub_days = {k: v["pnl_1x"].to_numpy() for k, v in s.groupby("asof")}
        diffs = np.empty(BOOT)
        for i in range(BOOT):
            pick = RNG.choice(days, len(days), replace=True)
            b = np.concatenate([base_days[k] for k in pick])
            sv = [sub_days[k] for k in pick if k in sub_days]
            diffs[i] = (np.concatenate(sv).mean() - b.mean()) if sv else np.nan
        diffs = diffs[~np.isnan(diffs)]
        print(f"  >= {thr:.2f}   delta {np.mean(diffs):>+7.1f}   "
              f"90% CI [{np.percentile(diffs,5):>+7.1f}, {np.percentile(diffs,95):>+7.1f}]   "
              f"p(delta<=0) {np.mean(diffs <= 0):.3f}")

    print("\n=== the low end: is selling CHEAP premium inside the map actually fine? ===")
    for lo, hi, lab in [(0.0, 0.80, "ratio < 0.80"), (0.80, 0.90, "0.80 - 0.90"),
                        (0.90, 1.00, "0.90 - 1.00"), (1.00, 1.30, "1.00 - 1.30"),
                        (1.30, 99, ">= 1.30")]:
        s = a[(a["ratio"] >= lo) & (a["ratio"] < hi)]
        if len(s) < 15:
            continue
        p = s["pnl_1x"]
        print(f"  {lab:<14} n {len(s):>4}  days {s['asof'].nunique():>3}  "
              f"win {100*(p>0).mean():>5.1f}%  avg {p.mean():>+7.1f}  PF {pf(p):>5.2f}")

    print("\n=== how often would each threshold leave a session with NO trade? ===")
    all_days = sorted(m["asof"].unique())
    for thr in [0.00, 0.90, 1.15, 1.30]:
        s = a[a["ratio"] >= thr]
        have = s["asof"].nunique()
        print(f"  >= {thr:.2f}: {have}/{len(all_days)} sessions have at least one candidate "
              f"({100*have/len(all_days):.0f}%)")


if __name__ == "__main__":
    main()
