"""Walk-forward test of the long-vol gate plus the all-feed flow overlays.

The base gate (iv_rank>=50 & vrp_ratio>1.0) was discovered in-sample on this same
history, and ~20 overlay features have now been looked at, so the in-sample PF of
1.661 is optimistic by an unknown amount. Two guards here:

  1. WALK-FORWARD. Thresholds are chosen only on months strictly before the month
     being scored, then applied forward. No future information reaches a decision.
  2. THEORY-FIRST OVERLAYS. The overlays are restricted to the ones with a stated
     mechanism (dealer positioning, multileg contamination, sweep urgency) rather
     than the ones that happened to score best.

Deployment bar: pooled OOS PF >= 1.2 AND day-clustered bootstrap p05 >= 1.2 AND
every fold profitable.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from uw_feed_edge_test import add_xs, boot_p05, load_lane, profit_factor

# overlay -> (+1 means "high is good", -1 means "low is good"), with the reason.
OVERLAYS = {
    "tape_vega_flow": (+1, "customer +vega => dealers short vega => vol amplified"),
    "tape_gamma_flow": (+1, "customer +gamma => dealers short gamma => vol amplified"),
    "hc_multileg_share": (-1, "multileg prints are spread legs, not directional conviction"),
    "hc_sweep_share": (+1, "sweeps = urgency = informed flow"),
    "hc_opening_share": (+1, "volume>OI = new position rather than an unwind"),
    "dp_bias": (-1, "dark-pool print location does NOT classify like a lit tape"),
}


def monthly(df: pd.DataFrame, col: str = "r50") -> pd.DataFrame:
    m = df.groupby(df["signal_date"].str[:7])[col].agg(
        n="size", PF=profit_factor, avgR="mean", win=lambda x: (x > 0).mean())
    return m.round(3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default="/Users/anuppamvi/uw_root/tradedesk")
    ap.add_argument("--feeds", default="out/uw_all_feeds.csv")
    ap.add_argument("--k", type=float, default=0.5)
    ap.add_argument("--min-train", type=int, default=2, help="months of history before scoring")
    args = ap.parse_args()

    base = Path(args.base_dir)
    lane = load_lane(base / "out/vol_lane_long.csv", args.k)
    feeds = add_xs(pd.read_csv(base / args.feeds, low_memory=False))
    feeds["date"] = pd.to_datetime(feeds["date"]).dt.strftime("%Y-%m-%d")
    df = lane.merge(feeds, left_on=["signal_date", "ticker"],
                    right_on=["date", "ticker"], how="left")
    df["mo"] = df["signal_date"].str[:7]

    gate = df[(df["iv_rank"] >= 50) & (df["vrp_ratio"] > 1.0)].copy()
    print(f"=== BASE GATE iv_rank>=50 & vrp_ratio>1.0   n={len(gate)}  "
          f"PF={profit_factor(gate['r50']):.3f}  p05={boot_p05(gate, 'r50'):.3f}")
    print(monthly(gate).to_string())
    bad = monthly(gate)
    print(f"  months profitable: {int((bad['PF'] > 1).sum())}/{len(bad)}  "
          f"-> deployment bar requires ALL")

    # ---- how much of the bad months is explained by each overlay? ----
    print("\n=== does each overlay rescue the losing months? (in-sample, median split) ===")
    print(f"{'overlay':22s} {'dir':>4s} {'n_keep':>7s} {'PF':>7s} {'p05':>7s} {'mo_pos':>8s}  reason")
    for f, (sign, why) in OVERLAYS.items():
        if f not in gate.columns:
            continue
        s = gate[gate[f].notna()]
        if len(s) < 120:
            print(f"{f:22s} {sign:+4d} {'sparse':>7s} (n={len(s)})")
            continue
        med = s[f].median()
        keep = s[s[f] > med] if sign > 0 else s[s[f] <= med]
        mm = monthly(keep)
        mm = mm[mm["n"] >= 5]
        print(f"{f:22s} {sign:+4d} {len(keep):7d} {profit_factor(keep['r50']):7.3f} "
              f"{boot_p05(keep, 'r50'):7.3f} {int((mm['PF'] > 1).sum())}/{len(mm):<6d}  {why}")

    # ---- walk-forward: thresholds from the past only ----
    print(f"\n=== WALK-FORWARD (median thresholds fit on prior months only) ===")
    months = sorted(gate["mo"].unique())
    for name, feats in [("base", []),
                        ("+vega", ["tape_vega_flow"]),
                        ("+sweep", ["hc_sweep_share"]),
                        ("+vega+sweep", ["tape_vega_flow", "hc_sweep_share"]),
                        ("+noMultileg", ["hc_multileg_share"]),
                        ("+vega+noMultileg", ["tape_vega_flow", "hc_multileg_share"])]:
        picks = []
        for i, mo in enumerate(months):
            if i < args.min_train:
                continue
            train = gate[gate["mo"] < mo]
            test = gate[gate["mo"] == mo]
            if train.empty or test.empty:
                continue
            sel = test
            for f in feats:
                if f not in train.columns or train[f].notna().sum() < 40:
                    continue
                thr = train[f].median()
                sign = OVERLAYS[f][0]
                sel = sel[sel[f] > thr] if sign > 0 else sel[sel[f] <= thr]
            if len(sel):
                picks.append(sel)
        if not picks:
            print(f"  {name:20s} no OOS rows")
            continue
        oos = pd.concat(picks)
        mm = monthly(oos)
        mm = mm[mm["n"] >= 5]
        print(f"  {name:20s} n={len(oos):5d}  PF={profit_factor(oos['r50']):6.3f}  "
              f"p05={boot_p05(oos, 'r50'):6.3f}  avgR={oos['r50'].mean():+.4f}  "
              f"months_pos={int((mm['PF'] > 1).sum())}/{len(mm)}")

    print("\n=== monthly detail, walk-forward base gate ===")
    picks = [gate[gate["mo"] == mo] for i, mo in enumerate(months) if i >= args.min_train]
    if picks:
        print(monthly(pd.concat(picks)).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
