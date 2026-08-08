"""Market-neutral equity book from the magnitude detector.

Every option expression measured in this repo loses to friction: a random long-vol
book runs PF 0.62, and the detector's +0.4 PF selection edge only reaches
breakeven. Equity round-trip friction is ~10bp against ~800bp for an option round
trip, so this asks whether the same picks pay when the premium is removed.

Long the up-lane top-K, short the dn-lane top-K, equal weight, held `horizon`
sessions. Entry is the NEXT close. Costs are charged explicitly and the whole book
is measured against random long/short books drawn from the same daily universe.

The book is judged on the per-cohort return distribution, not a profit factor:
an equity book's relevant statistics are mean, t against a day-clustered
bootstrap, and fold consistency.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .detector import _fit_predict, purge
from .features import feature_sets

DEFAULT_COSTS = {
    "commission_bps": 1.0,       # per side
    "spread_bps": 4.0,           # half-spread paid per side on liquid names
    "borrow_bps_annual": 200.0,  # charged on the short leg only
}


def _net_return(gross: float, is_short: bool, horizon: int, costs: dict) -> float:
    rt = 2.0 * (costs["commission_bps"] + costs["spread_bps"]) / 1e4
    r = -gross if is_short else gross
    if is_short:
        r -= costs["borrow_bps_annual"] / 1e4 * (horizon / 252.0)
    return r - rt


def run(df: pd.DataFrame, horizon: int = 21, pct: int = 20, k: int = 25,
        feature_set: str = "everything", costs: dict | None = None,
        n_controls: int = 200, min_train_months: int = 2, seed: int = 7,
        sets_fn=None) -> dict:
    costs = {**DEFAULT_COSTS, **(costs or {})}
    feats = (sets_fn or feature_sets)(df)[feature_set]
    up_t, dn_t, fwd = f"up_{horizon}_{pct}", f"dn_{horizon}_{pct}", f"fwd_{horizon}"
    months = sorted(df["month"].unique())
    usable = df[df[up_t].notna() & df[dn_t].notna() & df[fwd].notna()].copy()
    rng = np.random.default_rng(seed)

    legs, fold_rows = [], []
    for i in range(min_train_months, len(months)):
        te = usable[usable["month"] == months[i]]
        tr = purge(usable[usable["month"].isin(months[:i])], te, horizon)
        if len(te) < 500 or tr[up_t].nunique() < 2 or tr[dn_t].nunique() < 2:
            continue
        _, p_up = _fit_predict(tr, te, feats, up_t, seed)
        _, p_dn = _fit_predict(tr, te, feats, dn_t, seed)
        t = te.assign(_up=p_up, _dn=p_dn)

        longs = t.sort_values("_up", ascending=False).groupby("date").head(k)
        shorts = t.sort_values("_dn", ascending=False).groupby("date").head(k)
        for frame, is_short in ((longs, False), (shorts, True)):
            for _, r in frame.iterrows():
                legs.append({
                    "date": r["date"], "ticker": r["ticker"], "fold": str(months[i]),
                    "side": "short" if is_short else "long",
                    "gross": float(r[fwd]),
                    "net": _net_return(float(r[fwd]), is_short, horizon, costs),
                })

        # Null: same days, same leg counts, random names from the same universe.
        per_day = {d: g for d, g in t.groupby("date")}
        draws = []
        for _ in range(n_controls):
            tot = 0.0
            n_leg = 0
            for g in per_day.values():
                take = min(k, len(g))
                idx = rng.choice(len(g), size=min(2 * take, len(g)), replace=False)
                pick = g.iloc[idx]
                half = len(pick) // 2
                for j, (_, r) in enumerate(pick.iterrows()):
                    tot += _net_return(float(r[fwd]), j >= half, horizon, costs)
                    n_leg += 1
            draws.append(tot / max(n_leg, 1))
        f_legs = [x for x in legs if x["fold"] == str(months[i])]
        obs = float(np.mean([x["net"] for x in f_legs])) if f_legs else np.nan
        null_mean = float(np.mean(draws)) if draws else np.nan
        null_p95 = float(np.percentile(draws, 95)) if draws else np.nan
        p_value = float(np.mean(np.array(draws) >= obs)) if draws else np.nan
        fold_rows.append({
            "fold": str(months[i]), "n_legs": len(f_legs),
            "book_net": obs,
            "universe_gross": float(te[fwd].mean()),
            "null_mean": null_mean,
            "null_p95": null_p95,
            "p_value": p_value,
        })

    lf = pd.DataFrame(legs)
    return {"legs": lf, "folds": pd.DataFrame(fold_rows),
            "horizon": horizon, "k": k, "costs": costs}


def day_clustered_t(legs: pd.DataFrame, col: str = "net",
                    n: int = 2000, seed: int = 3) -> tuple[float, float, float]:
    """Resample whole entry dates; legs sharing a date are not independent."""
    rng = np.random.default_rng(seed)
    days = legs["date"].unique()
    by = {d: g[col].to_numpy() for d, g in legs.groupby("date")}
    out = []
    for _ in range(n):
        pick = rng.choice(len(days), len(days), replace=True)
        out.append(np.mean(np.concatenate([by[days[i]] for i in pick])))
    arr = np.array(out)
    return float(np.mean(arr)), float(np.percentile(arr, 5)), float(np.percentile(arr, 95))


def report(res: dict) -> None:
    lf, fd = res["legs"], res["folds"]
    if lf.empty:
        print("no legs")
        return
    h, k, c = res["horizon"], res["k"], res["costs"]
    rt = 2 * (c["commission_bps"] + c["spread_bps"])
    print(f"\n=== MARKET-NEUTRAL EQUITY BOOK  hold={h}d  k={k}/side  "
          f"round-trip {rt:.0f}bp/leg + {c['borrow_bps_annual']:.0f}bp/yr borrow ===")
    for side, g in lf.groupby("side"):
        print(f"  {side:<6} legs {len(g):>6}  gross {g.gross.mean():+.4f}  "
              f"net {g.net.mean():+.4f}  win {(g.net > 0).mean():.1%}")
    mean, p05, p95 = day_clustered_t(lf)
    print(f"  BOOK   legs {len(lf):>6}  net/leg {lf.net.mean():+.4f}  "
          f"day-clustered 90% CI [{p05:+.4f}, {p95:+.4f}]")
    per_period = lf.net.mean()
    print(f"  annualized (naive, {252 / h:.1f} periods/yr): {per_period * 252 / h:+.1%}")
    print(f"\n  {'fold':<9}{'legs':>7}{'book_net':>10}{'null':>9}{'uni_gross':>11}{'p':>8}")
    for _, r in fd.iterrows():
        print(f"  {r.fold:<9}{int(r.n_legs):>7}{r.book_net:>+10.4f}{r.null_mean:>+9.4f}"
              f"{r.universe_gross:>+11.4f}{r.p_value:>8.3f}")
    pos = int((fd.book_net > 0).sum())
    beat = int((fd.p_value <= 0.05).sum())
    print(f"\n  folds positive {pos}/{len(fd)}   folds beating null at p<=0.05 {beat}/{len(fd)}")
