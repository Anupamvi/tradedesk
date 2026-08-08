"""Is the option-relevant moment predictable at all?

The pipeline has measured two things:
  direction              AUC 0.53  (noise)
  |move| > median        AUC 0.71  (real)
Neither is what an option pays on. A long option pays when |move| exceeds the
BREAKEVEN; a short vertical loses when |move| exceeds the SHORT STRIKE. Both are
tail events at roughly 1-2 implied sigma, not median events.

This measures AUC on the exceedance events that actually price the structures,
using walk-forward folds and features known strictly at entry.
"""
from __future__ import annotations

import argparse
import math
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

VOL_ETPS = {"UVXY", "VXX", "SVIX", "SVXY", "VIXY", "UVIX", "VIXM", "VXZ",
            "SQQQ", "TQQQ", "SOXL", "SOXS", "SPXU", "UPRO", "TNA", "TZA"}

FEATURES = [
    "iv_rank", "iv30d", "vrp_ratio", "iv_chg_1w", "iv_chg_1m", "pos_52w",
    "stock_vol_surge", "call_vol_surge", "put_vol_surge", "put_call_ratio",
    "prem_tilt", "net_prem_tilt", "call_oi_chg", "put_oi_chg",
    "hc_multileg_share", "hc_sweep_share", "hc_opening_share", "hc_quote_churn",
    "hc_premium", "hc_chains", "hc_dir_bias",
    "oi_built_contracts", "oi_built_premium", "oi_signed_premium", "oi_n_chains",
    "oi_median_dte", "oi_nearmoney_premium", "oi_dir_bias", "oi_open_conviction",
    "oi_nearmoney_share",
    "dp_premium", "dp_block_premium", "dp_prints", "dp_bias", "dp_block_bias",
    "dp_block_share",
    "tape_net_premium", "tape_delta_notional", "tape_vega_flow", "tape_gamma_flow",
    "tape_gross_premium", "tape_prem_bias",
    "realized_vol_30d", "marketcap", "earn_dte",
]


def load(panel: str, horizon: int, min_mcap: float) -> pd.DataFrame:
    need = set(FEATURES) | {"date", "ticker", "sector", "issue_type", "marketcap",
                            "close", "iv30d", "next_earnings_date"}
    df = pd.read_csv(panel, usecols=lambda c: c in need, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["ticker"].notna()]
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df = df[~df["ticker"].isin(VOL_ETPS)]
    df = df[df["issue_type"].astype(str).str.contains("Common", case=False, na=False)]
    df = df[pd.to_numeric(df["marketcap"], errors="coerce").fillna(0) >= min_mcap]
    df = df[pd.to_numeric(df["iv30d"], errors="coerce") > 0.01]
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    ern = pd.to_datetime(df["next_earnings_date"], errors="coerce")
    df["earn_dte"] = (ern - df["date"]).dt.days

    g = df.groupby("ticker", sort=False)["close"]
    entry = g.shift(-1)
    exit_ = g.shift(-(1 + horizon))
    df["abs_move"] = (exit_ / entry - 1.0).abs()
    df["sigma"] = pd.to_numeric(df["iv30d"], errors="coerce") * math.sqrt(horizon / 252.0)
    df["z"] = df["abs_move"] / df["sigma"]
    return df[df["z"].notna() & np.isfinite(df["z"])].reset_index(drop=True)


def walk_forward_auc(df: pd.DataFrame, target: str) -> tuple[float, list]:
    months = sorted(df["month"].unique())
    feats = [f for f in FEATURES if f in df.columns]
    per_fold = []
    for i in range(2, len(months)):
        tr = df[df["month"].isin(months[:i])]
        te = df[df["month"] == months[i]]
        if te[target].nunique() < 2 or len(te) < 300 or tr[target].nunique() < 2:
            continue
        m = HistGradientBoostingClassifier(
            max_iter=180, learning_rate=0.06, max_depth=4,
            min_samples_leaf=60, l2_regularization=1.0, random_state=7,
        )
        m.fit(tr[feats], tr[target])
        p = m.predict_proba(te[feats])[:, 1]
        per_fold.append((str(months[i]), roc_auc_score(te[target], p),
                         float(te[target].mean()), len(te)))
    mean_auc = float(np.mean([a for _, a, _, _ in per_fold])) if per_fold else float("nan")
    return mean_auc, per_fold


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="/Users/anuppamvi/uw_root/tradedesk/out/uw_all_feeds.csv")
    ap.add_argument("--horizon", type=int, default=21)
    ap.add_argument("--min-mcap", type=float, default=2e9)
    args = ap.parse_args()

    df = load(args.panel, args.horizon, args.min_mcap)
    df["month"] = df["date"].dt.to_period("M")
    print(f"rows={len(df):,}  days={df['date'].nunique()}  horizon={args.horizon}d\n")

    targets = {
        "|move| > median            (known 0.71)": (df["z"] > df["z"].median()).astype(int),
        "|move| > 1.0 sigma  short-strike touch ": (df["z"] > 1.00).astype(int),
        "|move| > 1.5 sigma  vertical blowup    ": (df["z"] > 1.50).astype(int),
        "|move| > 2.0 sigma  deep tail          ": (df["z"] > 2.00).astype(int),
        "|move| > 0.8 sigma  long-option payoff ": (df["z"] > 0.80).astype(int),
        "DIRECTION up               (known 0.53)": None,
    }
    g = df.groupby("ticker", sort=False)["close"]
    signed = (g.shift(-(1 + args.horizon)) / g.shift(-1) - 1.0)
    targets["DIRECTION up               (known 0.53)"] = (signed > 0).astype(int)

    print(f"{'target':<42}{'base rate':>11}{'mean OOS AUC':>14}{'folds>0.55':>12}{'min':>7}{'max':>7}")
    for name, y in targets.items():
        d = df.assign(y=y)
        d = d[d["y"].notna()]
        auc, folds = walk_forward_auc(d, "y")
        if not folds:
            print(f"{name:<42}{'--':>11}{'--':>14}")
            continue
        aucs = [a for _, a, _, _ in folds]
        base = float(d["y"].mean())
        print(f"{name:<42}{base:>11.1%}{auc:>14.4f}{sum(a > 0.55 for a in aucs):>7}/{len(aucs):<4}"
              f"{min(aucs):>7.3f}{max(aucs):>7.3f}")

    print("\nPER-FOLD, vertical-blowup target (the moment a short-premium book is exposed to):")
    d = df.assign(y=(df["z"] > 1.50).astype(int))
    _, folds = walk_forward_auc(d, "y")
    for mo, a, base, n in folds:
        print(f"  {mo}  AUC {a:.4f}  base {base:.1%}  n {n:,}")


if __name__ == "__main__":
    main()
