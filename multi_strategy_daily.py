"""Daily EV-selected multi-strategy slate for a given date.

Trains the E[pnl] model on all backtested trades strictly before --date, then
constructs fresh candidates for --date across all strategy families, scores them,
and emits the top-N ranked slate (ticker, strategy, legs, entry, predicted EV).
No future data used for scoring; local UW chains only; no live orders.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import multi_strategy_backtest as mb
import multi_strategy_walkforward as wf
from uwos.exact_spread_backtester import HistoricalOptionQuoteStore, UnderlyingCloseStore

ROOT = Path("/Users/anuppamvi/uw_root/tradedesk")
DATE = next((a.split("=")[1] for a in sys.argv if a.startswith("--date=")), "2026-07-22")
TOPN = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--n=")), 15))
FILL = float(next((a.split("=")[1] for a in sys.argv if a.startswith("--fill=")), 0.75))
mb.FILL_FRAC = FILL


def prep_features(df):
    df = df.copy()
    df["log_spot"] = np.log(pd.to_numeric(df["spot"], errors="coerce").clip(1, None))
    df["log_liq"] = np.log1p(pd.to_numeric(df["min_leg_liquidity"], errors="coerce").clip(0, None))
    for src, cols in [("gex_reconstructed.csv", wf.GEX), ("iv_skew_reconstructed.csv", wf.IVF),
                      ("flow_features.csv", wf.FLOW)]:
        try:
            e = pd.read_csv(src); e["asof"] = e["asof"].astype(str); e["ticker"] = e["ticker"].astype(str)
            use = [c for c in cols if c in e.columns]
            df = df.merge(e[["asof", "ticker"] + use], on=["asof", "ticker"], how="left")
        except FileNotFoundError:
            for c in cols:
                df[c] = np.nan
    for c in wf.GEX + wf.IVF + wf.FLOW:
        df[c] = pd.to_numeric(df.get(c), errors="coerce").replace([np.inf, -np.inf], np.nan)
        df[c] = df[c].fillna(df[c].median() if df[c].notna().any() else 0.0)
    return df


def main():
    target = dt.date.fromisoformat(DATE)
    # 1) Train EV model on history strictly before target.
    hist = wf.load()
    tr = hist["asof"] < DATE
    if tr.sum() < 2000:
        print(f"insufficient history before {DATE} ({tr.sum()} trades)")
        return
    Xtr_full = wf.build_X(hist, None)
    cols = Xtr_full.columns
    model = wf.fit_reg(Xtr_full[tr.values].values.astype(float),
                       hist["pnl_1x"][tr.values].values.astype(float))
    print(f"Trained E[pnl] model on {tr.sum()} trades before {DATE} (fill={FILL}).\n")

    # 2) Construct fresh candidates for target date.
    store = HistoricalOptionQuoteStore(ROOT, use_oi=False)
    closes = UnderlyingCloseStore(ROOT, allow_web_fallback=False)
    dates = store.available_dates()
    if target not in dates:
        print(f"{DATE} not in available dated folders."); return
    dq = store.get_quotes_for_date(target)
    chain = mb.parse_chain(dq)
    liq = chain.groupby("under")["liq"].sum().sort_values(ascending=False)
    universe = list(liq.head(mb.UNIVERSE_PER_DAY).index)
    rows, meta = [], []
    for under in universe:
        spot = closes.get_close_on_or_before(under, target)
        cu = chain[chain["under"] == under]
        if spot is None or not np.isfinite(spot) or spot <= 0:
            spot = float(np.median(cu["strike"]))
        for route, legs in mb.build_strategies(under, spot, cu, target).items():
            od, cv, liq0, ok = mb.price_legs(store, target, under, legs)
            if not ok or od is None or abs(od) < 0.05:
                continue
            rows.append({
                "asof": DATE, "ticker": under, "strategy_route": route,
                "entry_type": "DEBIT" if od > 0 else "CREDIT", "spot": round(spot, 2),
                "dte": (legs[0].expiry - target).days, "entry_net": round(od, 2),
                "entry_cost": round(abs(od), 2), "min_leg_liquidity": round(liq0, 0), "n_legs": len(legs),
            })
            meta.append("; ".join(f"{'+' if l.qty>0 else ''}{l.qty} {under} {l.expiry} {l.strike:g}{l.right}" for l in legs))
    if not rows:
        print("no candidates constructed."); return
    cand = pd.DataFrame(rows)
    cand["legs"] = meta
    cand = wf.sanity_filter(cand)
    if cand.empty:
        print("no legit candidates after sanity filter."); return
    cand = prep_features(cand)
    Xc = wf.build_X(cand, cols)
    cand["pred_ev_1x"] = np.round(wf.predict_reg(model, Xc.values.astype(float)), 2)
    cand = cand.sort_values("pred_ev_1x", ascending=False).reset_index(drop=True)

    show = ["ticker", "strategy_route", "entry_type", "entry_net", "dte",
            "min_leg_liquidity", "pred_ev_1x", "legs"]
    out = cand.head(TOPN)[show]
    print(f"=== Top {TOPN} EV-ranked candidates for {DATE} ===\n")
    print(out.to_string(index=False))
    csv_path = ROOT / f"multi_strategy_slate_{DATE}.csv"
    cand[["asof"] + show].to_csv(csv_path, index=False)
    print(f"\nFull ranked slate ({len(cand)} candidates) -> {csv_path}")


if __name__ == "__main__":
    main()
