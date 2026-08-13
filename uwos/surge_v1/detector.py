"""Move detection: can the system find INTC/MU class rallies AND ORCL/IBM class drops?

Judged on what a 10-20 trade/month book actually consumes:
  precision@K   of the top K names ranked each day, how many moved as predicted
  lift          precision@K divided by the base rate
  capture       what share of all moves appeared in the top K
AUC is reported but is NOT the decision metric -- a book trades the top of the
ranking, not the whole cross-section.

Both directions are always run. A detector that only finds rallies in a rising
tape has found the tape, not a pattern, so the up and down lanes are reported
side by side and neither is allowed to stand alone.

Every fold is strictly walk-forward: train on all prior months, test on the next.
Feature sets are run separately so the options feeds have to prove they add
something over price alone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from .features import feature_sets


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, feats: list[str],
                 target: str, seed: int = 7):
    model = HistGradientBoostingClassifier(
        max_iter=250, learning_rate=0.05, max_depth=5,
        min_samples_leaf=80, l2_regularization=1.0, random_state=seed,
    )
    model.fit(train[feats], train[target].astype(int))
    return model, model.predict_proba(test[feats])[:, 1]


def purge(train: pd.DataFrame, test: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Drop training rows whose label window reaches into the test period.

    An h-day forward label on the last h training sessions is resolved by prices
    inside the test month, which lets the model see the fold it is scored on.
    """
    if train.empty or test.empty:
        return train
    cutoff = test["date"].min() - pd.tseries.offsets.BDay(horizon + 1)
    return train[train["date"] <= cutoff]


def _precision_at_k(test: pd.DataFrame, score: np.ndarray, target: str, k: int):
    """Rank within each day and take the top k, the way a daily book would."""
    t = test.assign(_s=score)
    picks = t.sort_values("_s", ascending=False).groupby("date").head(k)
    if picks.empty:
        return np.nan, np.nan, 0, np.nan
    hit = float(picks[target].mean())
    captured = float(picks[target].sum() / max(t[target].sum(), 1))
    fwd_col = "fwd_" + target.split("_")[1]
    avg_fwd = float(picks[fwd_col].mean()) if fwd_col in picks else np.nan
    return hit, captured, len(picks), avg_fwd


def walk_forward(df: pd.DataFrame, target: str, k: int = 20,
                 feature_sets_to_run: list[str] | None = None,
                 min_train_months: int = 2, seed: int = 7,
                 sets_fn=None) -> pd.DataFrame:
    sets = (sets_fn or feature_sets)(df)
    if feature_sets_to_run:
        sets = {n: f for n, f in sets.items() if n in feature_sets_to_run}
    h = int(target.split("_")[1])
    months = sorted(df["month"].unique())
    usable = df[df[target].notna()].copy()
    rows = []
    for name, feats in sets.items():
        for i in range(min_train_months, len(months)):
            te = usable[usable["month"] == months[i]]
            tr = purge(usable[usable["month"].isin(months[:i])], te, h)
            if len(te) < 500 or tr[target].nunique() < 2 or te[target].nunique() < 2:
                continue
            _, p = _fit_predict(tr, te, feats, target, seed)
            prec, cap, n, avg_fwd = _precision_at_k(te, p, target, k)
            base = float(te[target].mean())
            rows.append({
                "target": target, "feature_set": name, "fold": str(months[i]),
                "n_test": len(te), "base_rate": base,
                "auc": float(roc_auc_score(te[target].astype(int), p)),
                "precision_at_k": prec, "capture_at_k": cap, "picks": n,
                "avg_fwd_at_k": avg_fwd,
                "lift": prec / base if base > 0 else np.nan,
            })
    return pd.DataFrame(rows)


def summarize(res: pd.DataFrame) -> pd.DataFrame:
    if res.empty:
        return res
    keys = ["target", "feature_set"] if "target" in res.columns else ["feature_set"]
    return (res.groupby(keys)
            .agg(folds=("fold", "count"),
                 base=("base_rate", "mean"),
                 auc=("auc", "mean"),
                 prec=("precision_at_k", "mean"),
                 lift=("lift", "mean"),
                 capture=("capture_at_k", "mean"),
                 avg_fwd=("avg_fwd_at_k", "mean"),
                 worst_prec=("precision_at_k", "min"),
                 folds_gt_base=("lift", lambda s: int((s > 1.0).sum())))
            .sort_values("lift", ascending=False))


def bidirectional(df: pd.DataFrame, horizon: int = 21, pct: int = 20, k: int = 20,
                  feature_sets_to_run: list[str] | None = None,
                  seed: int = 7, sets_fn=None) -> pd.DataFrame:
    """Run the identical harness on the up lane and the down lane."""
    frames = []
    for side in ("up", "dn"):
        tgt = f"{side}_{horizon}_{pct}"
        if tgt not in df.columns:
            continue
        frames.append(walk_forward(df, tgt, k=k,
                                   feature_sets_to_run=feature_sets_to_run,
                                   seed=seed, sets_fn=sets_fn))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def name_level_detection(df: pd.DataFrame, target: str, tickers: list[str],
                         feature_set: str = "all", k: int = 20,
                         min_train_months: int = 2, seed: int = 7,
                         sets_fn=None) -> pd.DataFrame:
    """Did the model rank the named tickers highly on the days they were about to move?"""
    feats = (sets_fn or feature_sets)(df)[feature_set]
    months = sorted(df["month"].unique())
    usable = df[df[target].notna()].copy()
    h = int(target.split("_")[1])
    fwd_col = "fwd_" + target.split("_")[1]
    out = []
    for i in range(min_train_months, len(months)):
        te = usable[usable["month"] == months[i]]
        tr = purge(usable[usable["month"].isin(months[:i])], te, h)
        if len(te) < 500 or tr[target].nunique() < 2:
            continue
        _, p = _fit_predict(tr, te, feats, target, seed)
        te = te.assign(_s=p)
        te["day_rank"] = te.groupby("date")["_s"].rank(ascending=False, method="min")
        te["universe"] = te.groupby("date")["_s"].transform("size")
        for _, r in te[te["ticker"].isin(tickers)].iterrows():
            out.append({
                "date": r["date"], "ticker": r["ticker"], "fold": str(months[i]),
                "target": target, "rank": int(r["day_rank"]),
                "universe": int(r["universe"]),
                "pctile": float(1 - r["day_rank"] / r["universe"]),
                "in_top_k": bool(r["day_rank"] <= k),
                "hit": bool(r[target] == 1),
                "fwd": float(r[fwd_col]) if pd.notna(r.get(fwd_col)) else np.nan,
            })
    return pd.DataFrame(out)
