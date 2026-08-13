"""Permutation null for the move detector.

Repo lesson, learned the hard way and recorded: a single random control draw is
one sample from the null and says nothing about its spread. Every claim here is
judged against the full distribution of N random draws using the identical
universe, dates and pick count.

Also reports the universe mean forward return per fold, because a top-20 book
averaging +9% in a tape that averaged +9% has found the tape.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .detector import _fit_predict, purge
from .features import feature_sets


def run(df: pd.DataFrame, target: str, feature_set: str = "all", k: int = 20,
        n_perm: int = 200, min_train_months: int = 2, seed: int = 7,
        sets_fn=None) -> dict:
    feats = (sets_fn or feature_sets)(df)[feature_set]
    months = sorted(df["month"].unique())
    usable = df[df[target].notna()].copy()
    h = int(target.split("_")[1])
    fwd_col = "fwd_" + target.split("_")[1]
    rng = np.random.default_rng(seed)

    folds, sig_prec, sig_fwd = [], [], []
    null_prec, null_fwd = [], []
    for i in range(min_train_months, len(months)):
        te = usable[usable["month"] == months[i]]
        tr = purge(usable[usable["month"].isin(months[:i])], te, h)
        if len(te) < 500 or tr[target].nunique() < 2 or te[target].nunique() < 2:
            continue
        _, p = _fit_predict(tr, te, feats, target, seed)
        t = te.assign(_s=p)
        picks = t.sort_values("_s", ascending=False).groupby("date").head(k)

        # Null: same days, same pick count, random names.
        per_day = {d: g for d, g in t.groupby("date")}
        draws_prec, draws_fwd = [], []
        for _ in range(n_perm):
            chunks = [g.iloc[rng.choice(len(g), size=min(k, len(g)), replace=False)]
                      for g in per_day.values()]
            r = pd.concat(chunks)
            draws_prec.append(float(r[target].mean()))
            draws_fwd.append(float(r[fwd_col].mean()))
        folds.append({
            "fold": str(months[i]),
            "base_rate": float(te[target].mean()),
            "universe_mean_fwd": float(te[fwd_col].mean()),
            "signal_prec": float(picks[target].mean()),
            "null_prec_mean": float(np.mean(draws_prec)),
            "null_prec_p95": float(np.percentile(draws_prec, 95)),
            "signal_fwd": float(picks[fwd_col].mean()),
            "null_fwd_mean": float(np.mean(draws_fwd)),
            "null_fwd_p95": float(np.percentile(draws_fwd, 95)),
            "null_fwd_p05": float(np.percentile(draws_fwd, 5)),
            "p_prec": float(np.mean(np.array(draws_prec) >= picks[target].mean())),
            "p_fwd": float(np.mean(np.array(draws_fwd) >= picks[fwd_col].mean())),
        })
        sig_prec.append(picks[target].mean())
        sig_fwd.append(picks[fwd_col].mean())
        null_prec.append(np.mean(draws_prec))
        null_fwd.append(np.mean(draws_fwd))

    fd = pd.DataFrame(folds)
    return {
        "folds": fd,
        "target": target,
        "feature_set": feature_set,
        "mean_signal_prec": float(np.mean(sig_prec)) if sig_prec else np.nan,
        "mean_null_prec": float(np.mean(null_prec)) if null_prec else np.nan,
        "mean_signal_fwd": float(np.mean(sig_fwd)) if sig_fwd else np.nan,
        "mean_null_fwd": float(np.mean(null_fwd)) if null_fwd else np.nan,
        "folds_prec_sig": int((fd["p_prec"] <= 0.05).sum()) if len(fd) else 0,
        "folds_fwd_sig": int((fd["p_fwd"] <= 0.05).sum()) if len(fd) else 0,
        "n_folds": len(fd),
    }


def report(res: dict) -> None:
    print(f"\n=== {res['target']} | {res['feature_set']} | {res['n_folds']} folds ===")
    print(f"  precision  signal {res['mean_signal_prec']:.4f}  vs null {res['mean_null_prec']:.4f}"
          f"   significant folds {res['folds_prec_sig']}/{res['n_folds']}")
    print(f"  fwd return signal {res['mean_signal_fwd']:+.4f}  vs null {res['mean_null_fwd']:+.4f}"
          f"   significant folds {res['folds_fwd_sig']}/{res['n_folds']}")
    f = res["folds"]
    if f.empty:
        return
    print(f"  {'fold':<9}{'uni_fwd':>9}{'sig_fwd':>9}{'null_fwd':>10}{'p_fwd':>8}"
          f"{'sig_prec':>10}{'null_prec':>11}{'p_prec':>8}")
    for _, r in f.iterrows():
        print(f"  {r.fold:<9}{r.universe_mean_fwd:>+9.4f}{r.signal_fwd:>+9.4f}"
              f"{r.null_fwd_mean:>+10.4f}{r.p_fwd:>8.3f}{r.signal_prec:>10.4f}"
              f"{r.null_prec_mean:>11.4f}{r.p_prec:>8.3f}")
