"""P(profit)/EV selection model + leakage-free walk-forward backtest.

Trains an entry-time-only model on the pipeline's own scored option-ticket
outcomes and asks the honest question: if we RANK the fillable universe by the
model's predicted expected R and trade only the top slice, do we clear a robust
profit-factor bar out-of-sample?

Design (no leakage):
  - fold = calendar month of signal_date; train on all STRICTLY earlier months,
    test on the current month. Model never sees the test fold or any future data.
  - features are entry-time only (direction, strategy, DTE, moneyness, spread,
    quotes, fees, slippage, regime, sector). net_r / win are NEVER features.
  - deployment sim: within each test fold, for each (date,ticker,direction)
    EVENT the model picks its single highest predicted-EV ticket (one trade per
    event, as live), then we rank events by predicted EV and keep the top slice.
  - honest gate: pooled OOS PF >= 1.2 AND day-clustered bootstrap PF p05 >= 1.2
    AND every evaluated fold profitable.
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

NUMERIC = [
    "dte",
    "contract_directional_moneyness",
    "bid_ask_spread_pct",
    "entry_ask",
    "entry_bid",
    "round_trip_fees",
    "slippage_pct_of_spread",
]
CATEGORICAL = ["direction", "strategy_kind", "market_regime", "sector"]


def month_of(s: str) -> str:
    return str(s)[:7]


def profit_factor(r: np.ndarray) -> float:
    g = r[r > 0].sum()
    l = -r[r < 0].sum()
    if l <= 0:
        return float("inf") if g > 0 else 0.0
    return float(g / l)


def summarize(ev: pd.DataFrame) -> dict:
    r = ev["net_r"].to_numpy()
    return {
        "events": int(len(ev)),
        "dates": int(ev["signal_date"].nunique()),
        "win": round(float((r > 0).mean()), 3) if len(r) else 0.0,
        "avg_r": round(float(r.mean()), 4) if len(r) else 0.0,
        "pf": round(profit_factor(r), 3),
        "gross_r": round(float(r.sum()), 1),
    }


def boot_pf_p05(ev: pd.DataFrame, n: int = 2000, seed: int = 7) -> float:
    if ev.empty:
        return 0.0
    rng = np.random.default_rng(seed)
    dates = ev["signal_date"].unique()
    by = {d: ev.loc[ev["signal_date"] == d, "net_r"].to_numpy() for d in dates}
    pfs = []
    for _ in range(n):
        pick = rng.choice(dates, size=len(dates), replace=True)
        pfs.append(profit_factor(np.concatenate([by[d] for d in pick])))
    fin = np.array([p for p in pfs if np.isfinite(p)])
    return round(float(np.percentile(fin, 5)), 3) if fin.size else 0.0


def build_model() -> Pipeline:
    pre = ColumnTransformer(
        [
            ("num", "passthrough", NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    return Pipeline(
        [
            ("pre", pre),
            ("gbr", GradientBoostingRegressor(random_state=0, n_estimators=300,
                                              max_depth=3, learning_rate=0.05,
                                              subsample=0.8)),
        ]
    )


def prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["status"] == "SCORED"].copy()
    df["net_r"] = pd.to_numeric(df["net_r"], errors="coerce")
    df = df[df["net_r"].notna()].copy()
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    for c in CATEGORICAL:
        df[c] = df[c].fillna("NA").astype(str)
    df["month"] = df["signal_date"].map(month_of)
    return df


def deployment_events(test: pd.DataFrame, ehat: np.ndarray) -> pd.DataFrame:
    """One trade per event: pick the model's top predicted-EV ticket."""
    t = test.copy()
    t["ehat"] = ehat
    idx = t.groupby(["signal_date", "ticker", "direction"])["ehat"].idxmax()
    return t.loc[idx, ["signal_date", "ticker", "direction", "ehat", "net_r"]]


def run(df: pd.DataFrame, top_q: float, min_pf_gate: float = 1.2) -> dict:
    months = sorted(df["month"].unique())
    per_fold = []
    pooled = []
    for i, m in enumerate(months):
        if i == 0:
            continue
        train = df[df["month"] < m]
        test = df[df["month"] == m]
        if len(train) < 200 or test.empty:
            continue
        model = build_model()
        model.fit(train[NUMERIC + CATEGORICAL], train["net_r"])
        ehat = model.predict(test[NUMERIC + CATEGORICAL])
        ev = deployment_events(test, ehat)
        if ev.empty:
            continue
        cut = ev["ehat"].quantile(top_q)
        sel = ev[ev["ehat"] >= cut]
        s = summarize(sel)
        per_fold.append({"month": m, **s})
        if not sel.empty:
            pooled.append(sel.assign(month=m))
    pooledf = pd.concat(pooled, ignore_index=True) if pooled else pd.DataFrame(columns=["signal_date", "net_r"])
    psum = summarize(pooledf)
    psum["boot_pf_p05"] = boot_pf_p05(pooledf)
    ev_folds = [f for f in per_fold if f["events"] > 0]
    psum["profitable_folds"] = f"{sum(1 for f in ev_folds if f['pf'] >= 1.0)}/{len(ev_folds)}"
    psum["deployable"] = bool(
        psum["pf"] >= min_pf_gate and psum["boot_pf_p05"] >= min_pf_gate
        and ev_folds and all(f["pf"] >= 1.0 for f in ev_folds)
    )
    return {"top_q": top_q, "pooled": psum, "per_fold": per_fold}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--details", default="out/pattern_analysis_v2/2026-07-23/validation_details.csv")
    args = ap.parse_args()
    df = prep(pd.read_csv(args.details, low_memory=False))
    print(f"SCORED rows {len(df)}  events "
          f"{df.groupby(['signal_date','ticker','direction']).ngroups}  months {sorted(df['month'].unique())}")

    # Reference: raw universe (all events, mean over tickets).
    raw = df.groupby(["signal_date", "ticker", "direction"], as_index=False)["net_r"].mean()
    print("RAW UNIVERSE:", summarize(raw))

    for q in (0.80, 0.90, 0.95):
        res = run(df, q)
        p = res["pooled"]
        print(f"\n=== EV-model rank, top {int((1-q)*100)}% (top_q={q}) ===")
        print("pooled:", p)
        for f in res["per_fold"]:
            print("   ", f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
