"""Does the detector have DIRECTION skill, or does it just find volatile names?

The decisive diagnostic. If the up model and the down model pick the same names,
the system is a magnitude detector and must be traded with non-directional
structures. If they pick different names and each lane's own direction dominates,
directional structures are justified.

Reports, for each lane's picks:
    P(up 20%) vs P(down 20%)      the conditional direction split
    overlap                        share of names both models picked
    directional edge               P(own direction) - P(opposite direction)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .detector import _fit_predict, purge
from .features import feature_sets


def run(df: pd.DataFrame, horizon: int = 21, pct: int = 20, k: int = 20,
        feature_set: str = "all", min_train_months: int = 2, seed: int = 7,
        sets_fn=None) -> pd.DataFrame:
    feats = (sets_fn or feature_sets)(df)[feature_set]
    up_t, dn_t = f"up_{horizon}_{pct}", f"dn_{horizon}_{pct}"
    fwd_col = f"fwd_{horizon}"
    months = sorted(df["month"].unique())
    usable = df[df[up_t].notna() & df[dn_t].notna()].copy()

    rows = []
    for i in range(min_train_months, len(months)):
        te = usable[usable["month"] == months[i]]
        tr = purge(usable[usable["month"].isin(months[:i])], te, horizon)
        if len(te) < 500 or tr[up_t].nunique() < 2 or tr[dn_t].nunique() < 2:
            continue
        _, p_up = _fit_predict(tr, te, feats, up_t, seed)
        _, p_dn = _fit_predict(tr, te, feats, dn_t, seed)
        t = te.assign(_up=p_up, _dn=p_dn)
        up_picks = t.sort_values("_up", ascending=False).groupby("date").head(k)
        dn_picks = t.sort_values("_dn", ascending=False).groupby("date").head(k)

        up_key = set(zip(up_picks["date"], up_picks["ticker"]))
        dn_key = set(zip(dn_picks["date"], dn_picks["ticker"]))
        overlap = len(up_key & dn_key) / max(len(up_key), 1)

        rows.append({
            "fold": str(months[i]),
            "universe_fwd": float(te[fwd_col].mean()),
            "up_p_up": float(up_picks[up_t].mean()),
            "up_p_dn": float(up_picks[dn_t].mean()),
            "up_fwd": float(up_picks[fwd_col].mean()),
            "up_absfwd": float(up_picks[fwd_col].abs().mean()),
            "dn_p_dn": float(dn_picks[dn_t].mean()),
            "dn_p_up": float(dn_picks[up_t].mean()),
            "dn_fwd": float(dn_picks[fwd_col].mean()),
            "dn_absfwd": float(dn_picks[fwd_col].abs().mean()),
            "universe_absfwd": float(te[fwd_col].abs().mean()),
            "overlap": overlap,
            # A long/short book: long the up picks, short the down picks.
            "spread_fwd": float(up_picks[fwd_col].mean() - dn_picks[fwd_col].mean()),
        })
    return pd.DataFrame(rows)


def report(fd: pd.DataFrame) -> None:
    if fd.empty:
        print("no folds")
        return
    print(f"  {'fold':<9}{'uni_fwd':>9}{'up_fwd':>9}{'dn_fwd':>9}{'L/S':>9}"
          f"{'up:P(up)':>10}{'up:P(dn)':>10}{'dn:P(dn)':>10}{'dn:P(up)':>10}{'overlap':>9}")
    for _, r in fd.iterrows():
        print(f"  {r.fold:<9}{r.universe_fwd:>+9.4f}{r.up_fwd:>+9.4f}{r.dn_fwd:>+9.4f}"
              f"{r.spread_fwd:>+9.4f}{r.up_p_up:>10.3f}{r.up_p_dn:>10.3f}"
              f"{r.dn_p_dn:>10.3f}{r.dn_p_up:>10.3f}{r.overlap:>9.1%}")
    print(f"  {'MEAN':<9}{fd.universe_fwd.mean():>+9.4f}{fd.up_fwd.mean():>+9.4f}"
          f"{fd.dn_fwd.mean():>+9.4f}{fd.spread_fwd.mean():>+9.4f}"
          f"{fd.up_p_up.mean():>10.3f}{fd.up_p_dn.mean():>10.3f}"
          f"{fd.dn_p_dn.mean():>10.3f}{fd.dn_p_up.mean():>10.3f}{fd.overlap.mean():>9.1%}")
    print(f"\n  MAGNITUDE: universe |fwd| {fd.universe_absfwd.mean():.4f}  "
          f"up picks {fd.up_absfwd.mean():.4f}  dn picks {fd.dn_absfwd.mean():.4f}")
    up_edge = fd.up_p_up.mean() - fd.up_p_dn.mean()
    dn_edge = fd.dn_p_dn.mean() - fd.dn_p_up.mean()
    print(f"  DIRECTIONAL EDGE: up lane {up_edge:+.3f}   down lane {dn_edge:+.3f}")
    print(f"  L/S spread positive in {int((fd.spread_fwd > 0).sum())}/{len(fd)} folds")
