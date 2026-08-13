"""Walk-forward confidence model over the multi-strategy backtest dataset.

Trains family-specific P(profit) or E[P/L] models on entry-time features.
Training outcomes must mature before each test window, and selection uses a
fixed ex-ante threshold rather than a quantile learned from the test window.
Local data only.
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer

DETAIL = next((a.split("=")[1] for a in sys.argv if a.startswith("--detail=")), "multi_strategy_detail.csv")
NUMERIC = ["dte", "entry_cost", "n_legs", "min_leg_liquidity", "log_spot", "log_liq"]
GEX = ["net_gex_norm", "gex_sign", "atm_oi_frac"]
IVF = ["atm_iv", "put_skew", "call_skew", "risk_reversal", "iv_term_slope"]
# Real directional signal from stock-screener + bot-eod + dp-eod bundle files.
FLOW = ["flow_dir_premium", "net_prem_dir", "call_vol_surge", "put_vol_surge",
        "vol_surge_dir", "pcr", "day_ret", "log_mktcap",
        "bot_cp_prem_dir", "bot_aggr_dir_prem", "bot_delta_prem", "bot_avg_iv",
        "bot_trade_count", "dp_premium", "dp_prints"]
CATEG = ["strategy_route", "entry_type"]


def pf(p):
    pos = p[p > 0].sum(); neg = -p[p < 0].sum()
    return pos / neg if neg > 0 else (float("inf") if pos > 0 else 0.0)


# Expected net sign per family: DEBIT structures must cost money (entry_net>0),
# CREDIT structures must pay (entry_net<0). Rows violating this are stale/illiquid
# quote artifacts and are dropped.
DEBIT_FAMILIES = {"long_call", "long_put", "bull_call_debit", "bear_put_debit",
                  "straddle", "strangle", "call_butterfly", "put_butterfly",
                  "calendar_call", "diagonal_call"}
CREDIT_FAMILIES = {"short_put", "bull_put_credit", "bear_call_credit", "iron_butterfly"}
MIN_LEG_LIQ = 200.0


def sanity_filter(d):
    net = pd.to_numeric(d["entry_net"], errors="coerce")
    is_debit_fam = d["strategy_route"].isin(DEBIT_FAMILIES)
    is_credit_fam = d["strategy_route"].isin(CREDIT_FAMILIES)
    sign_ok = (is_debit_fam & (net > 0)) | (is_credit_fam & (net < 0))
    liq_ok = pd.to_numeric(d["min_leg_liquidity"], errors="coerce").fillna(0) >= MIN_LEG_LIQ
    return d[sign_ok & liq_ok].copy()


def load():
    d = pd.read_csv(DETAIL, low_memory=False)
    d["asof"] = d["asof"].astype(str)
    d["exit_day"] = d["exit_day"].astype(str)
    d["ticker"] = d["ticker"].astype(str)
    d["pnl_1x"] = pd.to_numeric(d["pnl_1x"], errors="coerce")
    d = d[d["pnl_1x"].notna()].copy()
    d["entry_cost"] = pd.to_numeric(d["entry_cost"], errors="coerce")
    d["pnl_stress_10"] = d["pnl_1x"] - 0.10 * d["entry_cost"].clip(lower=0) * 100.0
    d["y"] = (d["pnl_stress_10"] > 0).astype(float)
    d["log_spot"] = np.log(pd.to_numeric(d["spot"], errors="coerce").clip(1, None))
    d["log_liq"] = np.log1p(pd.to_numeric(d["min_leg_liquidity"], errors="coerce").clip(0, None))
    try:
        g = pd.read_csv("gex_reconstructed.csv"); g["asof"] = g["asof"].astype(str); g["ticker"] = g["ticker"].astype(str)
        d = d.merge(g[["asof", "ticker"] + GEX], on=["asof", "ticker"], how="left")
    except FileNotFoundError:
        for c in GEX: d[c] = 0.0
    try:
        iv = pd.read_csv("iv_skew_reconstructed.csv"); iv["asof"] = iv["asof"].astype(str); iv["ticker"] = iv["ticker"].astype(str)
        d = d.merge(iv[["asof", "ticker"] + IVF], on=["asof", "ticker"], how="left")
    except FileNotFoundError:
        for c in IVF: d[c] = np.nan
    try:
        fl = pd.read_csv("flow_features.csv"); fl["asof"] = fl["asof"].astype(str); fl["ticker"] = fl["ticker"].astype(str)
        d = d.merge(fl[["asof", "ticker"] + [c for c in FLOW if c in fl.columns]], on=["asof", "ticker"], how="left")
    except FileNotFoundError:
        for c in FLOW: d[c] = np.nan
    for c in GEX + IVF + FLOW:
        d[c] = pd.to_numeric(d.get(c), errors="coerce")
        d[c] = d[c].replace([np.inf, -np.inf], np.nan)
        d[c] = d[c].fillna(d[c].median() if d[c].notna().any() else 0.0)
    if "--raw" not in sys.argv:
        before = len(d)
        d = sanity_filter(d)
        print(f"sanity filter: kept {len(d)}/{before} legit structures "
              f"(family-correct sign + min leg liquidity {MIN_LEG_LIQ:.0f}).")
    return d.sort_values("asof").reset_index(drop=True)


def build_X(df, cols):
    feats = []
    for c in NUMERIC + GEX + IVF + FLOW:
        s = pd.to_numeric(df.get(c), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        # Do not derive clipping bounds from a future/test window.  The model
        # standardization below is fitted only on the training fold.
        feats.append(s.rename(c))
    X = pd.concat(feats, axis=1)
    for c in CATEG:
        dd = pd.get_dummies(df.get(c, pd.Series("na", index=df.index)).astype(str), prefix=c)
        X = pd.concat([X, dd.astype(float)], axis=1)
    if cols is not None:
        X = X.reindex(columns=cols, fill_value=0.0)
    return X


def fit(X, y):
    imputer = SimpleImputer(strategy="median")
    Xi = imputer.fit_transform(X)
    model = HistGradientBoostingClassifier(
        max_iter=100,
        learning_rate=0.05,
        max_leaf_nodes=7,
        min_samples_leaf=100,
        l2_regularization=2.0,
        random_state=41,
    )
    model.fit(Xi, y)
    return imputer, model


def predict(m, X):
    imputer, model = m
    return model.predict_proba(imputer.transform(X))[:, 1]


def fit_reg(X, y):
    imputer = SimpleImputer(strategy="median")
    Xi = imputer.fit_transform(X)
    model = HistGradientBoostingRegressor(
        max_iter=100,
        learning_rate=0.05,
        max_leaf_nodes=7,
        min_samples_leaf=100,
        l2_regularization=2.0,
        random_state=41,
    )
    model.fit(Xi, y)
    return imputer, model


def predict_reg(m, X):
    imputer, model = m
    return model.predict(imputer.transform(X))


def main():
    mode = "ev" if "--ev" in sys.argv else "pwin"
    min_pwin = float(next((a.split("=")[1] for a in sys.argv if a.startswith("--min-pwin=")), 0.60))
    min_ev = float(next((a.split("=")[1] for a in sys.argv if a.startswith("--min-ev=")), 0.0))
    d = load()
    print(f"{len(d)} trades, {d['asof'].nunique()} days, "
          f"10%-stress base PF {pf(d['pnl_stress_10']):.2f}  mode={mode}\n")
    cutoffs = ["2026-03-01", "2026-03-20", "2026-04-08", "2026-04-27",
               "2026-05-15", "2026-06-03", "2026-06-22"]
    gate = f"E[stress P/L] >= {min_ev:.2f}" if mode == "ev" else f"P(stress profit) >= {min_pwin:.2f}"
    print(f"Gate: fixed {gate}; family-specific, maturity-safe rolling folds.\n")
    print("%-12s %6s %6s %6s %8s %6s %10s" % ("test_from", "train", "n", "win%", "avgPL", "PF", "total"))
    allsel = []
    for i, cut in enumerate(cutoffs):
        end = cutoffs[i + 1] if i + 1 < len(cutoffs) else "2026-12-31"
        fold_rows = []
        fold_train = 0
        for family in sorted(d["strategy_route"].dropna().unique()):
            family_mask = d["strategy_route"].eq(family)
            # Maturity safety: a signal before the cutoff is not training data
            # unless its realized outcome was also known before the cutoff.
            tr = (family_mask & (d["asof"] < cut) & (d["exit_day"] < cut)).values
            te = (family_mask & (d["asof"] >= cut) & (d["asof"] < end)).values
            if tr.sum() < 300 or te.sum() < 20 or d.loc[tr, "y"].nunique() < 2:
                continue
            train_frame = d.loc[tr]
            test_frame = d.loc[te]
            Xtrain = build_X(train_frame, None)
            Xtest = build_X(test_frame, Xtrain.columns)
            Xtr = Xtrain.values.astype(float)
            Xte = Xtest.values.astype(float)
            if mode == "ev":
                m = fit_reg(Xtr, train_frame["pnl_stress_10"].values.astype(float))
                scores = predict_reg(m, Xte)
                selected = scores >= min_ev
            else:
                m = fit(Xtr, train_frame["y"].values.astype(float))
                scores = predict(m, Xte)
                selected = scores >= min_pwin
            sub = test_frame.iloc[selected].copy()
            if sub.empty:
                continue
            sub["model_score"] = scores[selected]
            sub = sub.sort_values(["asof", "model_score"], ascending=[True, False]).groupby("asof", as_index=False).head(1)
            fold_rows.append(sub)
            fold_train += int(tr.sum())
        if not fold_rows:
            continue
        sub = pd.concat(fold_rows, ignore_index=True)
        sub = sub.sort_values(["asof", "model_score"], ascending=[True, False]).groupby("asof", as_index=False).head(2)
        allsel.append(sub)
        pnl = sub["pnl_stress_10"]
        print("%-12s %6d %6d %6.1f %8.2f %6.2f %10.0f" % (
            cut, fold_train, len(sub), (pnl > 0).mean() * 100,
            pnl.mean(), pf(pnl), pnl.sum()))
    if allsel:
        a = pd.concat(allsel)
        pnl = a["pnl_stress_10"]
        print(f"\nPOOLED 10%-STRESS: n={len(a)} win={(pnl>0).mean()*100:.1f}% avgPL={pnl.mean():.2f} "
              f"PF={pf(pnl):.2f} total={pnl.sum():.0f}")
        print("\nSelected mix by strategy:")
        print(a.groupby("strategy_route")["pnl_stress_10"].agg(n="size", PF=pf, total="sum").sort_values("total", ascending=False).to_string())


if __name__ == "__main__":
    main()
