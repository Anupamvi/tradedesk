"""Magnitude-ranking model + walk-forward evaluation for the volatility lane.

Why this shape:
  - direction predictability : OOS AUC 0.53  -> noise, do not trade it
  - |move| predictability    : OOS AUC 0.71  -> real, trade THIS

Every structure in vol_lane_outcomes.csv carries a `breakeven_move_pct` that is
known at entry. So the single question worth modelling is:

        P( realized_abs_move > breakeven_move_pct )

One model, both directions of vol:
  - long_straddle / long_strangle -> take when p_exceed is HIGH  (options too cheap)
  - iron_butterfly / iron_condor  -> take when p_exceed is LOW   (options too rich)

Strictly walk-forward by month. Nothing is fit on data at or after the month
being predicted. Read-only; no order placement.
"""
import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

NUM = [
    "dte", "stock", "entry_net", "max_risk_points", "combined_spread_pct",
    "spread_to_premium", "min_volume", "min_oi", "breakeven_move_pct",
    "implied_move_h", "n_signals_for_ticker",
]
CAT = ["market_regime", "sector", "pattern_family"]

LONG_VOL = ("long_straddle", "long_strangle")
SHORT_VOL = ("iron_butterfly", "iron_condor")
HORIZON = 5


def pf(x):
    x = np.asarray(x, dtype=float)
    w = x[x > 0].sum()
    l = -x[x < 0].sum()
    return w / l if l > 0 else np.nan


def make_model(objective):
    pre = ColumnTransformer([
        ("num", "passthrough", NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), CAT),
    ])
    if objective == "ev":
        # A credit structure wins often and loses big. Ranking by P(win)
        # actively selects the worst tail, so rank by expected R instead.
        est = GradientBoostingRegressor(
            random_state=0, n_estimators=200, max_depth=3,
            learning_rate=0.05, subsample=0.8,
        )
    else:
        est = GradientBoostingClassifier(
            random_state=0, n_estimators=200, max_depth=3,
            learning_rate=0.05, subsample=0.8,
        )
    return Pipeline([("pre", pre), ("est", est)])


def prep(df, target):
    d = df.copy()
    d["dte"] = pd.to_numeric(d["dte"], errors="coerce")
    d["breakeven_move_pct"] = pd.to_numeric(d["breakeven_move_pct"], errors="coerce")
    # A 43-DTE straddle's breakeven is an EXPIRY number; comparing it to a 5-day
    # move is meaningless. Scale it to the holding period before using it.
    d["implied_move_h"] = d["breakeven_move_pct"] * np.sqrt(
        HORIZON / d["dte"].clip(lower=1)
    )
    for c in NUM:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    for c in CAT:
        d[c] = d[c].fillna("").astype(str)
    d["realized_abs_move"] = pd.to_numeric(d["realized_abs_move"], errors="coerce")
    d["net_r"] = pd.to_numeric(d["net_r"], errors="coerce")
    d = d.dropna(subset=["realized_abs_move", "net_r", "breakeven_move_pct"])
    d[NUM] = d[NUM].fillna(0.0)
    d["exceeded"] = (d["realized_abs_move"] > d["breakeven_move_pct"]).astype(int)
    # `exceeded` is only apples-to-apples when the structure is held to expiry
    # (short lane). For a mid-life exit, score what actually happened.
    d["y"] = d["exceeded"] if target == "exceeded" else d["win"].astype(int)
    d["month"] = d["signal_date"].str[:7]
    return d


def walk_forward(d, min_train, objective, verbose=True):
    """Fit on strictly-prior months, predict the held-out month."""
    months = sorted(d["month"].unique())
    out = []
    fold_auc = []
    for m in months:
        train = d[d["month"] < m]
        test = d[d["month"] == m]
        if len(train) < min_train or test.empty:
            continue
        if objective != "ev" and train["y"].nunique() < 2:
            continue
        model = make_model(objective)
        if objective == "ev":
            model.fit(train[NUM + CAT], train["net_r"])
            p = model.predict(test[NUM + CAT])
        else:
            model.fit(train[NUM + CAT], train["y"])
            p = model.predict_proba(test[NUM + CAT])[:, 1]
        t = test.copy()
        t["p_exceed"] = p
        out.append(t)
        auc = (roc_auc_score(t["y"], p)
               if t["y"].nunique() > 1 else np.nan)
        fold_auc.append((m, len(train), len(t), auc))
        if verbose:
            print(f"[wf] {m} train={len(train):6d} test={len(t):5d} AUC={auc:.4f}", flush=True)
    if not out:
        return pd.DataFrame(), pd.DataFrame()
    return pd.concat(out, ignore_index=True), pd.DataFrame(
        fold_auc, columns=["month", "n_train", "n_test", "auc"]
    )


def day_bootstrap_p05(d, n_boot=2000, seed=0):
    """Resample whole signal_dates, not rows - trades on one day are correlated."""
    if d.empty:
        return np.nan
    rng = np.random.default_rng(seed)
    days = d["signal_date"].unique()
    by_day = {k: g["net_r"].to_numpy(dtype=float) for k, g in d.groupby("signal_date")}
    stats = []
    for _ in range(n_boot):
        pick = rng.choice(days, size=len(days), replace=True)
        vals = np.concatenate([by_day[k] for k in pick])
        stats.append(pf(vals))
    stats = np.asarray(stats, dtype=float)
    stats = stats[~np.isnan(stats)]
    return float(np.percentile(stats, 5)) if stats.size else np.nan


def sweep(oos, kinds, pick, max_spread_pct):
    """Threshold sweep on the model score. pick = 'top' or 'bottom'."""
    sub = oos[oos["kind"].isin(kinds)].copy()
    sub = sub[sub["combined_spread_pct"] <= max_spread_pct]
    if sub.empty:
        print(f"  (no rows at spread<={max_spread_pct})")
        return pd.DataFrame()
    rows = []
    for q in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        if pick == "top":
            thr = sub["p_exceed"].quantile(1 - q) if q > 0 else -1.0
            sel = sub[sub["p_exceed"] >= thr]
            label = f"top {int(q*100)}%" if q > 0 else "all"
        else:
            thr = sub["p_exceed"].quantile(q) if q > 0 else 2.0
            sel = sub[sub["p_exceed"] <= thr]
            label = f"bottom {int(q*100)}%" if q > 0 else "all"
        if len(sel) < 30:
            continue
        per_fold = sel.groupby("month")["net_r"].apply(pf)
        rows.append({
            "select": label,
            "n": len(sel),
            "avgR": sel["net_r"].mean(),
            "win": sel["win"].mean(),
            "PF": pf(sel["net_r"]),
            "folds": len(per_fold),
            "folds_pos": int((per_fold > 1.0).sum()),
            "worst_fold_PF": per_fold.min(),
            "hit_rate_move": sel["exceeded"].mean(),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default="/Users/anuppamvi/uw_root/tradedesk/out/vol_lane_outcomes.csv")
    ap.add_argument("--out-preds", default="/Users/anuppamvi/uw_root/tradedesk/out/vol_lane_oos_preds.csv")
    ap.add_argument("--min-train", type=int, default=800)
    ap.add_argument("--target", choices=("win", "exceeded"), default="win")
    ap.add_argument("--objective", choices=("win", "ev"), default="ev",
                    help="'ev' ranks by predicted net_r; 'win' ranks by P(target)")
    ap.add_argument("--max-spread-pct", type=float, default=0.15)
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()

    raw = pd.read_csv(args.infile)
    raw = raw[raw["status"] == "SCORED"]
    d = prep(raw, args.target)
    print(f"[data] scored rows={len(d)} months={d['month'].nunique()} "
          f"({d['month'].min()}..{d['month'].max()}) target={args.target}")
    print("[data] by kind:")
    print(d.groupby("kind").agg(
        n=("net_r", "size"), avgR=("net_r", "mean"), win=("win", "mean"),
        be=("breakeven_move_pct", "median"), impl_h=("implied_move_h", "median"),
        mv=("realized_abs_move", "median"), exceeded=("exceeded", "mean"),
    ).round(4).to_string())
    print()

    oos, folds = walk_forward(d, args.min_train, args.objective)
    if oos.empty:
        print("[wf] no out-of-sample folds produced - not enough history")
        return 1
    print(f"\n[wf] OOS rows={len(oos)} mean AUC={folds['auc'].mean():.4f} "
          f"stable_folds={(folds['auc'] > 0.55).sum()}/{len(folds)} "
          f"objective={args.objective}")
    oos.to_csv(args.out_preds, index=False)

    # ranking by predicted EV always wants the top tail
    present = [("long", LONG_VOL), ("short", SHORT_VOL)]
    present = [(s, k) for s, k in present if oos["kind"].isin(k).any()]
    for side, kinds in present:
        pick = "top" if (args.objective == "ev" or args.target == "win"
                         or side == "long") else "bottom"
        print(f"\n=== {side.upper()} VOL  (pick {pick}, spread<={args.max_spread_pct}) ===")
        t = sweep(oos, kinds, pick, args.max_spread_pct)
        if not t.empty:
            print(t.round(4).to_string(index=False))

    print("\n=== DEPLOYMENT BAR (PF>=1.2 pooled, p05>=1.2, every fold profitable) ===")
    for side, kinds in present:
        pick = "top" if (args.objective == "ev" or args.target == "win"
                         or side == "long") else "bottom"
        sub = oos[oos["kind"].isin(kinds)]
        sub = sub[sub["combined_spread_pct"] <= args.max_spread_pct]
        if sub.empty:
            continue
        thr = sub["p_exceed"].quantile(0.7 if pick == "top" else 0.3)
        sel = sub[sub["p_exceed"] >= thr] if pick == "top" else sub[sub["p_exceed"] <= thr]
        if len(sel) < 30:
            continue
        per_fold = sel.groupby("month")["net_r"].apply(pf)
        p05 = day_bootstrap_p05(sel, args.bootstrap)
        pooled = pf(sel["net_r"])
        ok = (pooled >= 1.2) and (p05 >= 1.2) and bool((per_fold > 1.0).all())
        print(f"  {side:5s} {pick} 30%: n={len(sel):5d} PF={pooled:.3f} "
              f"p05={p05:.3f} folds_pos={(per_fold > 1.0).sum()}/{len(per_fold)} "
              f"-> {'PASS' if ok else 'FAIL'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
