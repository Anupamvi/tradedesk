"""Deployment gates for a book whose payoff is not symmetric.

Profit factor is a first-moment ratio. On a concave book it is dominated by a
tail that six months of data has probably not shown yet -- an 86% win rate over
six months tells you almost nothing about the 14%. On a convex book PF is
dominated by one or two winners, so it overstates repeatability.

These gates are therefore built on the loss distribution and on survival, not on
the mean:
    CVaR(5%)          average of the worst 5% of trade outcomes
    day-clustered p05 bootstrap lower bound, resampling whole entry dates
    concentration     share of profit from the single best name and best day
    ruin              probability of hitting a drawdown limit under resampling
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def profit_factor(r: pd.Series | np.ndarray) -> float:
    r = pd.Series(r).dropna()
    w, l = r[r > 0].sum(), -r[r < 0].sum()
    return float(w / l) if l > 0 else float("inf")


def cvar(r: pd.Series, q: float = 0.05) -> float:
    r = pd.Series(r).dropna()
    if r.empty:
        return float("nan")
    cutoff = r.quantile(q)
    tail = r[r <= cutoff]
    return float(tail.mean()) if len(tail) else float(cutoff)


def day_clustered_bootstrap(df: pd.DataFrame, col: str = "r",
                            date_col: str = "entry_date",
                            stat=profit_factor, n: int = 800,
                            seed: int = 5) -> tuple[float, float]:
    """Resample whole entry dates. Trades sharing a date are not independent."""
    rng = np.random.default_rng(seed)
    days = df[date_col].unique()
    by = {d: g[col].to_numpy() for d, g in df.groupby(date_col)}
    out = []
    for _ in range(n):
        pick = rng.choice(len(days), len(days), replace=True)
        vals = np.concatenate([by[days[i]] for i in pick])
        out.append(stat(pd.Series(vals)))
    arr = np.array(out, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.percentile(arr, 5)), float(np.percentile(arr, 50))


def ruin_probability(df: pd.DataFrame, col: str = "r", date_col: str = "entry_date",
                     risk_per_trade: float = 0.01, dd_limit: float = 0.25,
                     n: int = 600, seed: int = 5) -> float:
    """Share of resampled orderings that breach the drawdown limit."""
    rng = np.random.default_rng(seed)
    days = sorted(df[date_col].unique())
    by = {d: g[col].to_numpy() for d, g in df.groupby(date_col)}
    breaches = 0
    for _ in range(n):
        order = rng.permutation(len(days))
        equity, peak, ruined = 1.0, 1.0, False
        for i in order:
            for x in by[days[i]]:
                equity *= (1.0 + risk_per_trade * float(x))
                peak = max(peak, equity)
                if equity / peak - 1.0 <= -dd_limit:
                    ruined = True
                    break
            if ruined:
                break
        breaches += int(ruined)
    return breaches / n


def concentration(df: pd.DataFrame, col: str = "r", by: str = "ticker") -> dict:
    tot = df[col].sum()
    if tot == 0:
        return {"top1": float("nan"), "top3": float("nan"), "top_day": float("nan")}
    g = df.groupby(by)[col].sum().sort_values(ascending=False)
    day = df.groupby("entry_date")[col].sum().sort_values(ascending=False)
    return {
        "top1": float(g.iloc[0] / tot) if len(g) else float("nan"),
        "top3": float(g.head(3).sum() / tot) if len(g) else float("nan"),
        "top_day": float(day.iloc[0] / tot) if len(day) else float("nan"),
        "profitable_names": float((g > 0).mean()) if len(g) else float("nan"),
    }


GATES = {
    "min_trades": 100,
    "min_unique_dates": 20,
    "min_pf": 1.20,
    "min_pf_p05": 1.00,
    "max_ruin": 0.10,           # <=10% of orderings may breach a 25% drawdown
    "max_top1_share": 0.40,     # one name may not be 40%+ of profit
    "min_folds_positive": 0.75,
    "max_maxloss_rate": 0.35,   # share of trades losing >=95% of premium
}


def evaluate(df: pd.DataFrame, fold_col: str = "fold", col: str = "r",
             floor_r: float = -1.0) -> dict:
    """`floor_r` is the structural max loss. For long premium it is -1.0R by
    construction, so CVaR is gated on the FREQUENCY of max loss, not its size."""
    if df.empty:
        return {"deployable": False, "reason": "no trades"}
    pf = profit_factor(df[col])
    p05, p50 = day_clustered_bootstrap(df, col=col)
    cv = cvar(df[col])
    ruin = ruin_probability(df, col=col)
    conc = concentration(df, col=col)
    folds = df.groupby(fold_col)[col].sum() if fold_col in df.columns else pd.Series(dtype=float)
    frac_pos = float((folds > 0).mean()) if len(folds) else float("nan")
    maxloss_rate = float((df[col] <= floor_r * 0.95).mean())

    checks = {
        "trades": (len(df) >= GATES["min_trades"], f"{len(df)} >= {GATES['min_trades']}"),
        "unique_dates": (df["entry_date"].nunique() >= GATES["min_unique_dates"],
                         f"{df['entry_date'].nunique()} >= {GATES['min_unique_dates']}"),
        "profit_factor": (pf >= GATES["min_pf"], f"{pf:.3f} >= {GATES['min_pf']}"),
        "pf_p05": (p05 >= GATES["min_pf_p05"], f"{p05:.3f} >= {GATES['min_pf_p05']}"),
        "maxloss_rate": (maxloss_rate <= GATES["max_maxloss_rate"],
                         f"{maxloss_rate:.3f} <= {GATES['max_maxloss_rate']}"),
        "ruin_prob": (ruin <= GATES["max_ruin"], f"{ruin:.3f} <= {GATES['max_ruin']}"),
        "top1_share": (conc["top1"] <= GATES["max_top1_share"],
                       f"{conc['top1']:.3f} <= {GATES['max_top1_share']}"),
        "folds_positive": (frac_pos >= GATES["min_folds_positive"],
                           f"{frac_pos:.2f} >= {GATES['min_folds_positive']}"),
    }
    failed = [k for k, (ok, _) in checks.items() if not ok]
    return {
        "deployable": not failed, "failed_gates": failed, "checks": checks,
        "n": len(df), "pf": pf, "pf_p05": p05, "pf_median_boot": p50,
        "avg_r": float(df[col].mean()), "win_rate": float((df[col] > 0).mean()),
        "cvar_5pct": cv, "maxloss_rate": maxloss_rate, "ruin_prob": ruin,
        "folds_positive": frac_pos, **conc,
    }


def report(name: str, res: dict) -> None:
    if not res.get("checks"):
        print(f"\n=== {name} === {res.get('reason', 'no data')}")
        return
    verdict = "DEPLOYABLE" if res["deployable"] else "BLOCKED"
    print(f"\n=== {name} === {verdict}")
    print(f"  n={res['n']}  win={res['win_rate']:.1%}  avgR={res['avg_r']:+.4f}  "
          f"PF={res['pf']:.3f}  PF_p05={res['pf_p05']:.3f}")
    print(f"  CVaR5%={res['cvar_5pct']:.3f}  maxloss_rate={res['maxloss_rate']:.1%}  "
          f"ruin={res['ruin_prob']:.1%}  top1={res['top1']:.1%}  top3={res['top3']:.1%}  "
          f"names_profitable={res['profitable_names']:.1%}")
    for k, (ok, detail) in res["checks"].items():
        print(f"    [{'PASS' if ok else 'FAIL'}] {k:<16} {detail}")
