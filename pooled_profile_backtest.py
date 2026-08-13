"""Exhaustive walk-forward backtest of a POOLED-PROFILE actionability lane for
Pattern Analysis V2.

Hypothesis: a proven *directional* pattern family can be traded through a
pooled option ticket (across all DTE/moneyness contract profiles) instead of
requiring a separately validated contract profile. This tests whether that lane
is profitable out-of-sample, versus the current strict contract-profile gate.

Data: the pipeline's own per-signal OOS-scored outcomes
(out/pattern_analysis_v2/<date>/validation_details.csv). Each row is a 5d option
ticket outcome. net_r is outcome-only (independent of tiering), so we can
re-decide selection with a proper nested walk-forward with NO leakage:
  - fold = calendar month of signal_date
  - training pool for fold m = all SCORED rows from months strictly before m
  - proven-direction tier learned on training only
  - test = SCORED rows in month m, collapsed to (date,ticker,direction) events
No outcome data is used to select trades. Only prior-fold outcomes build tiers.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def month_of(date_str: str) -> str:
    return str(date_str)[:7]


def profit_factor(r: np.ndarray) -> float:
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    if losses <= 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def family_tier(train: pd.DataFrame, group_cols: List[str], min_n: int, min_dates: int,
                min_pf: float) -> set:
    """Return set of group keys that qualify as proven on TRAINING data only."""
    proven = set()
    for key, g in train.groupby(group_cols):
        n = len(g)
        dates = g["signal_date"].nunique()
        pf = profit_factor(g["net_r"].to_numpy())
        avg = g["net_r"].mean()
        if n >= min_n and dates >= min_dates and pf >= min_pf and avg > 0:
            proven.add(key if isinstance(key, tuple) else (key,))
    return proven


def collapse_events(df: pd.DataFrame, agg: str) -> pd.DataFrame:
    """One row per (signal_date,ticker,direction) event.

    agg controls how multiple contract implementations of the same event are
    reduced: mean (unbiased), best (optimistic), worst (pessimistic), median.
    """
    grp = df.groupby(["signal_date", "ticker", "direction"], as_index=False)
    if agg == "mean":
        ev = grp["net_r"].mean()
    elif agg == "median":
        ev = grp["net_r"].median()
    elif agg == "best":
        ev = grp["net_r"].max()
    elif agg == "worst":
        ev = grp["net_r"].min()
    else:
        raise ValueError(agg)
    return ev


def day_cluster_bootstrap_pf(ev: pd.DataFrame, n_boot: int = 2000, seed: int = 7) -> Tuple[float, float]:
    """Day-clustered bootstrap of PF: resample distinct signal_dates with
    replacement. Returns (p05, p50)."""
    if ev.empty:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    dates = ev["signal_date"].unique()
    by_date = {d: ev.loc[ev["signal_date"] == d, "net_r"].to_numpy() for d in dates}
    pfs = []
    for _ in range(n_boot):
        pick = rng.choice(dates, size=len(dates), replace=True)
        r = np.concatenate([by_date[d] for d in pick])
        pfs.append(profit_factor(r))
    finite = np.array([p for p in pfs if np.isfinite(p)])
    if finite.size == 0:
        return (0.0, 0.0)
    return (float(np.percentile(finite, 5)), float(np.percentile(finite, 50)))


def summarize(ev: pd.DataFrame) -> Dict[str, float]:
    r = ev["net_r"].to_numpy()
    return {
        "events": int(len(ev)),
        "dates": int(ev["signal_date"].nunique()),
        "win": float((r > 0).mean()) if len(r) else 0.0,
        "avg_r": float(r.mean()) if len(r) else 0.0,
        "pf": profit_factor(r),
        "gross_r": float(r.sum()),
    }


def run_walkforward(df: pd.DataFrame, *, lane: str, min_n: int, min_dates: int,
                    min_pf: float, agg: str, regime_gate: bool,
                    bootstrap: bool = False) -> Dict:
    df = df.copy()
    df["month"] = df["signal_date"].map(month_of)
    months = sorted(df["month"].unique())
    per_fold = []
    all_ev = []
    for i, m in enumerate(months):
        if i == 0:
            continue  # no prior training pool
        train = df[df["month"] < m]
        test = df[df["month"] == m]
        if lane == "direction":
            gcols = ["directional_pattern_family"]
        elif lane == "profile":
            gcols = ["pattern_family", "contract_profile"]
        else:
            raise ValueError(lane)
        proven = family_tier(train, gcols, min_n, min_dates, min_pf)
        keyfn = lambda row: tuple(row[c] for c in gcols)
        mask = test.apply(lambda row: keyfn(row) in proven, axis=1)
        sel = test[mask]
        if regime_gate:
            # stand down in MIXED regime (edge historically breaks there)
            sel = sel[sel["market_regime"].astype(str).str.upper() != "MIXED"]
        ev = collapse_events(sel, agg)
        if ev.empty:
            per_fold.append({"month": m, **summarize(ev)})
            continue
        s = summarize(ev)
        per_fold.append({"month": m, **s})
        all_ev.append(ev.assign(month=m))
    pooled = pd.concat(all_ev, ignore_index=True) if all_ev else pd.DataFrame(columns=["signal_date", "net_r"])
    pooled_summary = summarize(pooled) if not pooled.empty else {"events": 0, "dates": 0, "win": 0, "avg_r": 0, "pf": 0, "gross_r": 0}
    if bootstrap and not pooled.empty:
        p05, p50 = day_cluster_bootstrap_pf(pooled)
        pooled_summary["boot_pf_p05"] = p05
        pooled_summary["boot_pf_p50"] = p50
    profitable_folds = sum(1 for f in per_fold if f["events"] > 0 and f["pf"] >= 1.0)
    evaluated_folds = sum(1 for f in per_fold if f["events"] > 0)
    return {
        "lane": lane, "min_n": min_n, "min_dates": min_dates, "min_pf": min_pf,
        "agg": agg, "regime_gate": regime_gate,
        "per_fold": per_fold, "pooled": pooled_summary,
        "profitable_folds": profitable_folds, "evaluated_folds": evaluated_folds,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--details", default="out/pattern_analysis_v2/2026-07-20/validation_details.csv")
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.details, low_memory=False)
    df = df[(df["status"] == "SCORED") & df["net_r"].notna()].copy()
    df["net_r"] = pd.to_numeric(df["net_r"], errors="coerce")
    df = df[df["net_r"].notna()]
    print(f"loaded SCORED rows: {len(df)}  events: "
          f"{df.groupby(['signal_date','ticker','direction']).ngroups}")

    # Baseline reference: whole universe, per-event mean.
    base_ev = collapse_events(df, "mean")
    print("\n=== RAW UNIVERSE (all events, no selection) ===")
    print(summarize(base_ev))

    if not args.sweep:
        for lane in ("profile", "direction"):
            res = run_walkforward(df, lane=lane, min_n=30, min_dates=15, min_pf=1.2,
                                  agg="mean", regime_gate=False, bootstrap=True)
            print(f"\n=== LANE={lane} (min_n=30,dates=15,pf=1.2,agg=mean) ===")
            print("pooled:", res["pooled"], "profitable_folds",
                  f"{res['profitable_folds']}/{res['evaluated_folds']}")
            for f in res["per_fold"]:
                print("  ", f)
        return 0

    # Exhaustive sweep.
    rows = []
    grid = itertools.product(
        ("direction", "profile"),
        (20, 30, 50),      # min_n
        (8, 15),           # min_dates
        (1.2, 1.5),        # min_pf
        ("mean", "best", "worst"),
        (False, True),     # regime_gate
    )
    for lane, mn, md, pf, agg, rg in grid:
        res = run_walkforward(df, lane=lane, min_n=mn, min_dates=md, min_pf=pf,
                              agg=agg, regime_gate=rg, bootstrap=True)
        p = res["pooled"]
        rows.append({
            "lane": lane, "min_n": mn, "min_dates": md, "min_pf": pf, "agg": agg,
            "regime_gate": rg, "events": p["events"], "dates": p["dates"],
            "win": round(p["win"], 3), "avg_r": round(p["avg_r"], 4),
            "pf": round(p["pf"], 3), "gross_r": round(p["gross_r"], 1),
            "boot_pf_p05": round(p.get("boot_pf_p05", 0.0), 3),
            "prof_folds": f"{res['profitable_folds']}/{res['evaluated_folds']}",
        })
    out = pd.DataFrame(rows).sort_values("pf", ascending=False)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print("\n=== EXHAUSTIVE SWEEP (sorted by pooled PF) ===")
    print(out.to_string(index=False))
    out.to_csv("pooled_profile_backtest_sweep.csv", index=False)
    print("\nwrote pooled_profile_backtest_sweep.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
