"""Walk-forward probability-of-profit model POC for the Options Agent.

Reads the local replay detail (dated UW backtest outcomes), trains a calibrated
logistic regression on entry-time-only features on the pre-split window, and
evaluates held-out (post-split) profit factor when ranking candidates by the
model's predicted probability of profit. No Schwab account data is used.
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd

DETAIL = sys.argv[1] if len(sys.argv) > 1 else (
    "out/options_agent_independent_replay/v1_56_live_selector_dte_parity_ytd_full/"
    "options_agent_replay_detail.csv"
)
SPLIT = "2026-05-01"

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


def pf_stats(pnl: np.ndarray):
    pos = pnl[pnl > 0].sum()
    neg = -pnl[pnl < 0].sum()
    n = len(pnl)
    win = float((pnl > 0).mean()) if n else 0.0
    pf = pos / neg if neg > 0 else float("inf")
    return n, win, (pnl.mean() if n else 0.0), pf, pnl.sum()


def main() -> None:
    df = pd.read_csv(DETAIL, low_memory=False)
    df = df[df["next_session_reprice_approved"].astype(str).str.lower().isin(["true", "1", "yes"])].copy()
    df = df[pd.to_numeric(df["pnl_1x"], errors="coerce").notna()].copy()
    df["pnl_1x"] = pd.to_numeric(df["pnl_1x"], errors="coerce")
    df["y"] = (df["pnl_1x"] > 0).astype(float)
    df["asof"] = df["asof"].astype(str)

    # Build feature matrix (entry-time only; no next_session/executed/exit/pnl leakage)
    feats = []
    for c in NUMERIC:
        s = pd.to_numeric(df.get(c), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        # clip extreme values to 1st/99th pct to stabilize the linear model
        lo, hi = s.quantile(0.01), s.quantile(0.99)
        feats.append(s.clip(lo, hi).rename(c))
    for c in BOOL:
        feats.append(df.get(c, pd.Series(False, index=df.index)).astype(str).str.lower().isin(["true", "1", "yes"]).astype(float).rename(c))
    X = pd.concat(feats, axis=1)
    for c in CATEG:
        d = pd.get_dummies(df.get(c, pd.Series("na", index=df.index)).astype(str), prefix=c)
        X = pd.concat([X, d.astype(float)], axis=1)

    train_mask = (df["asof"] < SPLIT).values
    test_mask = ~train_mask
    Xtr, ytr = X[train_mask].values, df["y"][train_mask].values
    Xte, yte = X[test_mask].values, df["y"][test_mask].values
    pnl_te = df["pnl_1x"][test_mask].values
    ds_te = pd.to_numeric(df["decision_score"][test_mask], errors="coerce").fillna(0.0).values

    # Standardize using train stats
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    Xtr_s = (Xtr - mu) / sd
    Xte_s = (Xte - mu) / sd

    # Logistic regression via gradient descent with L2
    rng = np.random.default_rng(0)
    n, d = Xtr_s.shape
    w = np.zeros(d)
    b = 0.0
    lr, lam, epochs = 0.1, 1e-3, 400

    def sig(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    for _ in range(epochs):
        p = sig(Xtr_s @ w + b)
        g = p - ytr
        gw = Xtr_s.T @ g / n + lam * w
        gb = g.mean()
        w -= lr * gw
        b -= lr * gb

    ptr = sig(Xtr_s @ w + b)
    pte = sig(Xte_s @ w + b)

    # AUC (rank-based) on test
    def auc(y, p):
        order = np.argsort(p)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(p) + 1)
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

    print(f"rows: train={train_mask.sum()} test={test_mask.sum()} features={d}")
    print(f"train base win-rate={ytr.mean():.3f}  test base win-rate={yte.mean():.3f}")
    print(f"test AUC (model P(profit)) = {auc(yte, pte):.3f}")
    print(f"test AUC (existing decision_score) = {auc(yte, ds_te):.3f}")

    print("\n== HELD-OUT PF: rank test by MODEL P(profit), keep top fraction ==")
    print("%6s %6s %7s %8s %6s %10s" % ("topFrac", "n", "win%", "avgP/L", "PF", "totalP/L"))
    order = np.argsort(-pte)
    for frac in [1.0, 0.5, 0.3, 0.2, 0.1, 0.05]:
        k = max(1, int(frac * len(pte)))
        sub = pnl_te[order[:k]]
        n_, w_, a_, pf_, tot_ = pf_stats(sub)
        print("%6.2f %6d %6.1f %8.2f %6s %10.0f" % (frac, n_, 100 * w_, a_, ("inf" if np.isinf(pf_) else f"{pf_:.2f}"), tot_))

    print("\n== HELD-OUT PF: rank test by EXISTING decision_score, keep top fraction ==")
    print("%6s %6s %7s %8s %6s %10s" % ("topFrac", "n", "win%", "avgP/L", "PF", "totalP/L"))
    order_ds = np.argsort(-ds_te)
    for frac in [1.0, 0.5, 0.3, 0.2, 0.1, 0.05]:
        k = max(1, int(frac * len(ds_te)))
        sub = pnl_te[order_ds[:k]]
        n_, w_, a_, pf_, tot_ = pf_stats(sub)
        print("%6.2f %6d %6.1f %8.2f %6s %10.0f" % (frac, n_, 100 * w_, a_, ("inf" if np.isinf(pf_) else f"{pf_:.2f}"), tot_))

    # Regime robustness: held-out top-decile PF within each market regime
    reg_te = df["regime"][test_mask].astype(str).values
    thr10 = np.quantile(pte, 0.90)
    keep = pte >= thr10
    print("\n== REGIME ROBUSTNESS: held-out top-10%% confidence trades, split by market regime ==")
    print("%-10s %6s %7s %8s %6s %10s" % ("regime", "n", "win%", "avgP/L", "PF", "totalP/L"))
    for reg in ["risk_on", "mixed", "risk_off"]:
        m = keep & (reg_te == reg)
        if m.sum() == 0:
            print("%-10s %6d %7s" % (reg, 0, "n/a"))
            continue
        n_, w_, a_, pf_, tot_ = pf_stats(pnl_te[m])
        print("%-10s %6d %6.1f %8.2f %6s %10.0f" % (reg, n_, 100 * w_, a_, ("inf" if np.isinf(pf_) else f"{pf_:.2f}"), tot_))


if __name__ == "__main__":
    main()
