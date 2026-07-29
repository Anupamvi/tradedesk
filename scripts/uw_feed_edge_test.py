"""Test whether the all-feed UW signals add edge on top of the iv_rank/VRP vol gate.

Cost model (see vol_lane_backtest.score_forward / settle_at_expiry):
  net_r already contains a FULL bid-ask crossing on entry (buys at ask, sells at
  bid), a FULL crossing on exit when the structure is unwound, plus an explicit
  modelled slippage charge on top. That explicit charge is double-counting.

  mid_r = net_r + spread_pct*0.5*(1+xc) + (entry_slip+exit_slip)/max_risk
  r_k   = mid_r - k*spread_pct*0.5*(1+xc)        k = fraction of half-spread paid

  xc = 1 when the position was unwound before expiry, 0 when it settled.

r50 (k=0.5) is the honest central case: you cross half the spread each way.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FEATS = [
    "iv_rank", "vrp_ratio",
    "hc_dir_bias", "hc_sweep_share", "hc_opening_share", "hc_multileg_share",
    "oi_dir_bias", "oi_open_conviction", "oi_nearmoney_share", "oi_built_premium",
    "dp_bias", "dp_block_bias", "dp_late_bias", "dp_block_share",
    "tape_prem_bias", "tape_late_bias",
    "tape_vega_flow", "tape_gamma_flow", "tape_delta_notional", "tape_net_premium",
]


def profit_factor(r: pd.Series) -> float:
    w = r[r > 0].sum()
    l = -r[r < 0].sum()
    return float(w / l) if l > 0 else np.inf


def boot_p05(df: pd.DataFrame, col: str, n: int = 2000, seed: int = 7) -> float:
    """Day-clustered bootstrap: resample whole signal_dates, not rows."""
    days = df["signal_date"].unique()
    if len(days) < 5:
        return float("nan")
    rng = np.random.default_rng(seed)
    by_day = {d: g[col].to_numpy() for d, g in df.groupby("signal_date")}
    out = np.empty(n)
    for i in range(n):
        pick = rng.choice(days, size=len(days), replace=True)
        r = np.concatenate([by_day[d] for d in pick])
        w = r[r > 0].sum()
        l = -r[r < 0].sum()
        out[i] = w / l if l > 0 else np.nan
    return float(np.nanpercentile(out, 5))


def load_lane(path: Path, k: float = 0.5) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = df[df["status"].astype(str).str.upper() == "SCORED"].copy()
    df["signal_date"] = pd.to_datetime(df["signal_date"]).dt.strftime("%Y-%m-%d")
    # unwound (crossed the spread a second time) vs settled at expiry
    xc = (pd.to_datetime(df["target_date"]) < pd.to_datetime(df["expiry"])).astype(float)
    addback = df["combined_spread_pct"] * 0.5 * (1.0 + xc)
    slip_r = (df["entry_slippage"].fillna(0) + df["exit_slippage"].fillna(0)) / df["max_risk"]
    df["mid_r"] = df["net_r"] + addback + slip_r
    df["r50"] = df["mid_r"] - k * addback
    df["_addback"] = addback
    return df


def rk(df: pd.DataFrame, k: float) -> pd.Series:
    return df["mid_r"] - k * df["_addback"]


def add_xs(feeds: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional daily percentile ranks.

    Raw greek flow scales with the size of the name - NVDA's gamma flow dwarfs a
    small cap's regardless of how aggressive the positioning is. Desks compare a
    name against that day's field, so rank within the day before using it.
    """
    for c in ["tape_gamma_flow", "tape_vega_flow", "tape_delta_notional",
              "tape_net_premium", "tape_gross_premium", "oi_signed_premium"]:
        if c in feeds.columns and f"{c}_xs" not in feeds.columns:
            feeds[f"{c}_xs"] = feeds.groupby("date")[c].rank(pct=True)
    return feeds


def dealer_test(df: pd.DataFrame, lane: str, col: str = "r50") -> None:
    """PRE-REGISTERED test of the dealer-positioning hypothesis.

    The signs were fixed from the literature BEFORE looking at any result, so this
    is a real test rather than a search. tape_*_flow is CUSTOMER-signed (+1 when
    the customer lifts the ask); dealers hold the mirror image.

        customer flow > 0 -> dealers SHORT gamma/vega -> hedging amplifies vol -> LONG vol wins
        customer flow < 0 -> dealers LONG  gamma/vega -> hedging damps vol     -> SHORT vol wins
    """
    print(f"\n=== PRE-REGISTERED dealer-positioning test ({lane} lane) ===")
    want_pos = lane == "long"
    print(f"    prediction: {lane} vol does BETTER when customer greek flow is "
          f"{'POSITIVE (dealers short vol)' if want_pos else 'NEGATIVE (dealers long vol)'}")
    print(f"    {'greek':22s} {'n_neg':>6s} {'PF_neg':>8s} {'n_pos':>6s} {'PF_pos':>8s}  verdict")
    for g in ("tape_gamma_flow", "tape_vega_flow"):
        if g not in df.columns:
            continue
        s = df[df[g].notna()]
        if len(s) < 60:
            print(f"    {g:22s} insufficient coverage (n={len(s)})")
            continue
        neg, pos = s[s[g] < 0], s[s[g] > 0]
        if len(neg) < 30 or len(pos) < 30:
            continue
        pf_n, pf_p = profit_factor(neg[col]), profit_factor(pos[col])
        confirmed = (pf_p > pf_n) if want_pos else (pf_n > pf_p)
        print(f"    {g:22s} {len(neg):6d} {pf_n:8.3f} {len(pos):6d} {pf_p:8.3f}  "
              f"{'CONFIRMED' if confirmed else 'REJECTED'}")


def report(name: str, d: pd.DataFrame, col: str = "r50") -> dict:
    if d.empty:
        return {"gate": name, "n": 0}
    return {
        "gate": name,
        "n": len(d),
        "PF": round(profit_factor(d[col]), 3),
        "avgR": round(float(d[col].mean()), 4),
        "win%": round(100.0 * float((d[col] > 0).mean()), 1),
        "days": d["signal_date"].nunique(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default="/Users/anuppamvi/uw_root/tradedesk")
    ap.add_argument("--feeds", default="out/uw_all_feeds.csv")
    ap.add_argument("--lane", default="long", choices=["long", "short"])
    ap.add_argument("--k", type=float, default=0.5)
    ap.add_argument("--min-n", type=int, default=120)
    args = ap.parse_args()

    base = Path(args.base_dir)
    lane = load_lane(base / f"out/vol_lane_{args.lane}.csv", args.k)
    feeds = pd.read_csv(base / args.feeds, low_memory=False)
    feeds["date"] = pd.to_datetime(feeds["date"]).dt.strftime("%Y-%m-%d")
    feeds = add_xs(feeds)

    keep = ["date", "ticker"] + [c for c in feeds.columns if c not in ("date", "ticker")]
    df = lane.merge(feeds[keep], left_on=["signal_date", "ticker"],
                    right_on=["date", "ticker"], how="left")

    print(f"=== lane={args.lane}  k={args.k}  rows={len(df)}  matched={df['date'].notna().sum()}")
    print(report("ALL", df))

    dealer_test(df, args.lane)

    have = [f for f in FEATS if f in df.columns]
    print(f"\n=== coverage of feed features on lane rows ===")
    for f in have:
        print(f"  {f:24s} {100.0*df[f].notna().mean():5.1f}%")

    # baseline vol gate
    gate = df[(df["iv_rank"] >= 50) & (df["vrp_ratio"] > 1.0)]
    base_row = report("iv_rank>=50 & vrp>1.0", gate)
    print(f"\n=== baseline gate ===\n{base_row}")
    if len(gate) >= args.min_n:
        print(f"  day-clustered bootstrap p05 = {boot_p05(gate, 'r50'):.3f}")

    # each feature, standalone terciles, on the FULL lane
    print(f"\n=== standalone: PF by feature tercile (full lane, k={args.k}) ===")
    print(f"{'feature':24s} {'n':>6s} {'lo':>7s} {'mid':>7s} {'hi':>7s}")
    for f in have:
        s = df[df[f].notna()]
        if len(s) < args.min_n * 3:
            continue
        try:
            q = pd.qcut(s[f], 3, labels=["lo", "mid", "hi"], duplicates="drop")
        except ValueError:
            continue
        pf = {str(g): profit_factor(x["r50"]) for g, x in s.groupby(q, observed=True)}
        print(f"{f:24s} {len(s):6d} " + " ".join(
            f"{pf.get(g, float('nan')):7.3f}" for g in ("lo", "mid", "hi")))

    # each feature, INSIDE the vol gate -> does it add on top?
    print(f"\n=== incremental: PF by feature median split INSIDE the vol gate ===")
    print(f"{'feature':24s} {'n':>6s} {'below':>8s} {'above':>8s} {'above_p05':>10s}")
    for f in have:
        s = gate[gate[f].notna()]
        if len(s) < args.min_n:
            continue
        med = s[f].median()
        lo, hi = s[s[f] <= med], s[s[f] > med]
        if len(lo) < 30 or len(hi) < 30:
            continue
        p05 = boot_p05(hi, "r50") if len(hi) >= 60 else float("nan")
        print(f"{f:24s} {len(s):6d} {profit_factor(lo['r50']):8.3f} "
              f"{profit_factor(hi['r50']):8.3f} {p05:10.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
