"""Multi-split walk-forward validation of the P(profit) model + regime gate.

Rolling folds: for each fold cutoff, train on all fillable trades before the
cutoff and test on the next ~4 weeks. Apply a market-direction (regime) gate so
we only take high-confidence trades in decisive tape (risk_on/risk_off) and
stand down in 'mixed'. All data is the local UW backtest replay detail; no
Schwab account data is used.
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd

DETAIL = next((a for a in sys.argv[1:] if not a.startswith("--")), None) or (
    "out/options_agent_independent_replay/v1_56_live_selector_dte_parity_ytd_full/"
    "options_agent_replay_detail.csv"
)

NUMERIC = [
    "dte", "entry_width", "entry_credit", "entry_debit",
    "entry_credit_pct_width", "entry_debit_pct_width", "entry_quote_width_pct",
    "reward_risk", "expected_move_ratio", "combined_flow_bias",
    "flow_total_premium", "iv_rank", "iv30d",
    "source_contract_oi", "source_contract_volume", "price_move_pct",
    "macro_event_count_before_expiry",
]
CATEG = ["strategy_route", "regime", "underlying_quality_tier", "entry_side", "sector"]
BOOL = ["core_universe_member", "macro_tape_candidate", "earnings_before_expiry"]
DECISIVE = {"risk_on", "risk_off"}


def pf_stats(pnl):
    pos = pnl[pnl > 0].sum()
    neg = -pnl[pnl < 0].sum()
    n = len(pnl)
    win = float((pnl > 0).mean()) if n else 0.0
    pf = pos / neg if neg > 0 else (float("inf") if pos > 0 else 0.0)
    return n, win, (pnl.mean() if n else 0.0), pf, pnl.sum()


def build_X(df):
    feats = []
    for c in NUMERIC:
        s = pd.to_numeric(df.get(c), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        lo, hi = s.quantile(0.01), s.quantile(0.99)
        feats.append(s.clip(lo, hi).rename(c))
    for c in BOOL:
        feats.append(df.get(c, pd.Series(False, index=df.index)).astype(str).str.lower().isin(["true", "1", "yes"]).astype(float).rename(c))
    X = pd.concat(feats, axis=1)
    for c in CATEG:
        d = pd.get_dummies(df.get(c, pd.Series("na", index=df.index)).astype(str), prefix=c)
        X = pd.concat([X, d.astype(float)], axis=1)
    return X


def fit_logit(Xtr, ytr, epochs=400, lr=0.1, lam=1e-3):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    Xs = (Xtr - mu) / sd
    n, d = Xs.shape
    w = np.zeros(d)
    b = 0.0

    def sig(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    for _ in range(epochs):
        p = sig(Xs @ w + b)
        g = p - ytr
        w -= lr * (Xs.T @ g / n + lam * w)
        b -= lr * g.mean()
    return (w, b, mu, sd, sig)


def predict(model, X):
    w, b, mu, sd, sig = model
    return sig(((X - mu) / sd) @ w + b)


def main():
    df = pd.read_csv(DETAIL, low_memory=False)
    df = df[df["next_session_reprice_approved"].astype(str).str.lower().isin(["true", "1", "yes"])].copy()
    df = df[pd.to_numeric(df["pnl_1x"], errors="coerce").notna()].copy()
    df["pnl_1x"] = pd.to_numeric(df["pnl_1x"], errors="coerce")
    df["y"] = (df["pnl_1x"] > 0).astype(float)
    df["asof"] = df["asof"].astype(str)
    df["regime"] = df["regime"].astype(str)
    df = df.sort_values("asof").reset_index(drop=True)

    # --- Merge reconstructed historical GEX (dealer-gamma proxy) on (asof, ticker) ---
    gex_path = "gex_reconstructed.csv"
    try:
        if "--no-gex" in sys.argv:
            raise FileNotFoundError
        gex = pd.read_csv(gex_path)
        gex["asof"] = gex["asof"].astype(str)
        gex["ticker"] = gex["ticker"].astype(str)
        df["ticker"] = df["ticker"].astype(str)
        df = df.merge(gex[["asof", "ticker", "net_gex_norm", "gex_sign", "atm_oi_frac"]],
                      on=["asof", "ticker"], how="left")
        for c in ["net_gex_norm", "gex_sign", "atm_oi_frac"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        NUMERIC.extend(["net_gex_norm", "gex_sign", "atm_oi_frac"])
        print(f"GEX merged: {(df['net_gex_norm'] != 0).mean():.1%} of rows have GEX.\n")
    except FileNotFoundError:
        print("gex_reconstructed.csv not found; running without GEX features.\n")

    # --- Merge reconstructed IV-surface / skew features on (asof, ticker) ---
    if "--no-iv" not in sys.argv:
        try:
            iv = pd.read_csv("iv_skew_reconstructed.csv")
            iv["asof"] = iv["asof"].astype(str)
            iv["ticker"] = iv["ticker"].astype(str)
            ivcols = ["atm_iv", "put_skew", "call_skew", "risk_reversal", "iv_term_slope"]
            df = df.merge(iv[["asof", "ticker"] + ivcols], on=["asof", "ticker"], how="left")
            for c in ivcols:
                df[c] = pd.to_numeric(df[c], errors="coerce")
                df[c] = df[c].fillna(df[c].median())
            NUMERIC.extend(ivcols)
            print(f"IV-skew merged: {(df['atm_iv'].notna()).mean():.1%} of rows.\n")
        except FileNotFoundError:
            print("iv_skew_reconstructed.csv not found; running without IV features.\n")

    # --- Day-level market-direction features (same-day, no future leakage) ---
    fb = pd.to_numeric(df.get("combined_flow_bias"), errors="coerce").fillna(0.0)
    pm = pd.to_numeric(df.get("price_move_pct"), errors="coerce").fillna(0.0)
    df["_fb"] = fb
    df["_pm"] = pm
    g = df.groupby("asof")
    day_feat = pd.DataFrame({
        "day_mean_flow_bias": g["_fb"].transform("mean"),
        "day_mean_price_move": g["_pm"].transform("mean"),
        "day_frac_risk_on": g["regime"].transform(lambda s: (s == "risk_on").mean()),
        "day_frac_risk_off": g["regime"].transform(lambda s: (s == "risk_off").mean()),
        "day_breadth": g["_pm"].transform(lambda s: (s > 0).mean()),
    })
    # regime-direction interaction: flow bias aligned with day breadth
    day_feat["flow_x_breadth"] = df["_fb"] * (day_feat["day_breadth"] - 0.5)

    use_day = "--day-feats" in sys.argv
    if use_day:
        X_all = pd.concat([build_X(df).reset_index(drop=True), day_feat.reset_index(drop=True)], axis=1)
    else:
        X_all = build_X(df).reset_index(drop=True)
    cols = X_all.columns

    cutoffs = ["2026-04-01", "2026-04-20", "2026-05-08", "2026-05-27", "2026-06-15", "2026-07-01"]
    print("Rolling walk-forward folds. Gate: top-10%% model confidence, DECISIVE regime only (risk_on/risk_off).\n")
    print("%-12s %6s %6s %7s %8s %6s %10s" % ("test_from", "train", "n", "win%", "avgP/L", "PF", "totalP/L"))
    agg = []
    for i, cut in enumerate(cutoffs):
        end = cutoffs[i + 1] if i + 1 < len(cutoffs) else "2026-12-31"
        tr = (df["asof"] < cut).values
        te = ((df["asof"] >= cut) & (df["asof"] < end)).values
        if tr.sum() < 200 or te.sum() < 20:
            continue
        model = fit_logit(X_all[tr].values.astype(float), df["y"][tr].values.astype(float))
        pte = predict(model, X_all[te].values.astype(float))
        reg = df["regime"][te].values
        pnl = df["pnl_1x"][te].values
        thr = np.quantile(pte, 0.90)
        keep = (pte >= thr) & np.isin(reg, list(DECISIVE))
        if keep.sum() == 0:
            print("%-12s %6d %6d %7s" % (cut, tr.sum(), 0, "n/a"))
            continue
        n_, w_, a_, pf_, tot_ = pf_stats(pnl[keep])
        agg.append(pnl[keep])
        print("%-12s %6d %6d %6.1f %8.2f %6s %10.0f" % (cut, tr.sum(), n_, 100 * w_, a_, ("inf" if np.isinf(pf_) else f"{pf_:.2f}"), tot_))

    if agg:
        allp = np.concatenate(agg)
        n_, w_, a_, pf_, tot_ = pf_stats(allp)
        print("\nPOOLED across folds: n=%d win=%.1f%% avgP/L=%.2f PF=%s total=%.0f" % (
            n_, 100 * w_, a_, ("inf" if np.isinf(pf_) else f"{pf_:.2f}"), tot_))


if __name__ == "__main__":
    main()
