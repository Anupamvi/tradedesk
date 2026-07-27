"""Walk-forward selection test on top of a chosen exit policy.

The question this answers: once the exit leak is removed, does ANY selection
rule clear PF >= 1.25 out-of-sample, on rolling folds, with per-fold sample
sizes large enough to believe?

Everything is expanding-window walk-forward: a rule's threshold is chosen on
data strictly before a fold and then scored on the fold. Pooled numbers are
reported alongside per-fold numbers, because pooled-only numbers have hidden a
mirage in this codebase before.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

HISTORY = Path("codexuw/history/codexdaily_v4_edge_history_v3_2026-07-23.csv.gz")
GRID_DIR = Path("out/redesign_exit_grid")


def truthy(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "1.0", "yes"})


def pf(pnl: pd.Series) -> float:
    pnl = pd.Series(pnl).dropna()
    if pnl.empty:
        return float("nan")
    g = pnl[pnl > 0].sum()
    l = -pnl[pnl < 0].sum()
    return g / l if l > 0 else float("inf")


def load(exit_csv: str) -> pd.DataFrame:
    d = pd.read_csv(HISTORY, low_memory=False)
    d["row_id"] = d.index
    num = [
        "entry_credit_pct_width", "entry_debit_pct_width", "expected_move_ratio",
        "entry_quote_width_pct", "entry_width", "distance_pct", "reward_risk", "dte",
        "iv_rank", "iv_hv_ratio", "combined_flow_bias", "flow_bias", "entry_credit",
        "entry_debit", "bot_volume_oi_ratio", "source_contract_oi", "iv30d",
        "realized_volatility_30d", "dp_flow_bias", "bot_flow_bias", "pnl_1x",
    ]
    for c in num:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    ex = pd.read_csv(GRID_DIR / exit_csv)
    ex = ex[["row_id", "pnl", "reason"]].rename(columns={"pnl": "pnl_exit", "reason": "exit_reason_new"})
    m = d.merge(ex, on="row_id", how="inner")
    m["asof"] = pd.to_datetime(m["asof"], errors="coerce")
    m["is_credit"] = m["direction"].isin(["Bull Put", "Bear Call"])
    sign = m["direction"].map({"Bull Put": 1, "Bull Call": 1, "Bear Call": -1, "Bear Put": -1})
    m["flow_align"] = m["combined_flow_bias"].fillna(m["flow_bias"]) * sign
    m["is_guard"] = truthy(m["replay_guard_pass"])
    m["premium_pct"] = np.where(m["is_credit"], m["entry_credit_pct_width"], m["entry_debit_pct_width"])
    return m


def folds(d: pd.DataFrame, n_folds: int = 5, min_train_days: int = 40):
    days = sorted(d["asof"].dropna().unique())
    if len(days) < min_train_days + n_folds:
        return []
    tail = days[min_train_days:]
    chunks = np.array_split(np.array(tail), n_folds)
    out = []
    for ch in chunks:
        if len(ch) == 0:
            continue
        lo, hi = ch[0], ch[-1]
        out.append((d[d["asof"] < lo], d[(d["asof"] >= lo) & (d["asof"] <= hi)], pd.Timestamp(lo), pd.Timestamp(hi)))
    return out


def report(name: str, per_fold: list[tuple[str, pd.Series]]) -> dict:
    all_pnl = pd.concat([p for _, p in per_fold]) if per_fold else pd.Series(dtype=float)
    pooled = pf(all_pnl)
    ok_folds = sum(1 for _, p in per_fold if len(p) > 0 and pf(p) >= 1.25)
    live_folds = sum(1 for _, p in per_fold if len(p) > 0)
    print(f"\n--- {name} ---")
    print(f"  pooled OOS: n={len(all_pnl):>5}  PF={pooled:>6.3f}  win={(all_pnl > 0).mean() if len(all_pnl) else float('nan'):>6.1%}  total=${all_pnl.sum():>9.0f}")
    print(f"  folds clearing PF>=1.25: {ok_folds}/{live_folds}")
    for label, p in per_fold:
        if len(p) == 0:
            print(f"    {label}  n=    0")
        else:
            print(f"    {label}  n={len(p):>4}  PF={pf(p):>6.3f}  total=${p.sum():>8.0f}")
    return {"n": len(all_pnl), "pf": pooled, "ok_folds": ok_folds, "live_folds": live_folds,
            "total": float(all_pnl.sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exit", default="hold_to_expiry_slip10pct.csv")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    d = load(args.exit)
    d = d[d["pnl_exit"].notna()]
    print(f"exit policy file : {args.exit}")
    print(f"rows             : {len(d)}   sessions: {d['asof'].nunique()}")
    print(f"baseline pooled  : PF={pf(d['pnl_exit']):.3f}  total=${d['pnl_exit'].sum():.0f}  avg=${d['pnl_exit'].mean():.2f}")

    fs = folds(d, n_folds=args.folds)
    print(f"folds            : {len(fs)}")
    for tr, te, lo, hi in fs:
        print(f"    fold {lo.date()}..{hi.date()}  train={len(tr):>5}  test={len(te):>5}")

    summaries = {}

    # 0. No selection at all -- the floor everything must beat.
    summaries["no selection"] = report(
        "NO SELECTION (whole fillable universe)",
        [(f"{lo.date()}..{hi.date()}", te["pnl_exit"]) for tr, te, lo, hi in fs],
    )

    # 1. The current live guard, re-scored under the new exit.
    summaries["current guard"] = report(
        "CURRENT GUARD (replay_guard_pass) under new exit",
        [(f"{lo.date()}..{hi.date()}", te[te["is_guard"]]["pnl_exit"]) for tr, te, lo, hi in fs],
    )

    # 2. Structural single-factor rules, threshold picked on train each fold.
    factor_specs = [
        ("premium_pct", "high", [0.20, 0.25, 0.30, 0.35, 0.40]),
        ("expected_move_ratio", "high", [0.50, 0.75, 1.00, 1.25, 1.50]),
        ("flow_align", "high", [0.0, 0.10, 0.20, 0.30]),
        ("entry_quote_width_pct", "low", [0.10, 0.15, 0.20, 0.25, 0.35]),
        ("iv_hv_ratio", "high", [0.8, 0.9, 1.0, 1.1, 1.2]),
        ("dte", "high", [7, 14, 21, 30]),
        ("reward_risk", "high", [0.8, 1.0, 1.25, 1.5, 2.0]),
    ]
    for col, side, thresholds in factor_specs:
        if col not in d.columns:
            continue
        per_fold = []
        for tr, te, lo, hi in fs:
            best_t, best_pf = None, -math.inf
            for t in thresholds:
                sel = tr[tr[col] >= t] if side == "high" else tr[tr[col] <= t]
                if len(sel) < 60:
                    continue
                v = pf(sel["pnl_exit"])
                if math.isfinite(v) and v > best_pf:
                    best_pf, best_t = v, t
            if best_t is None:
                per_fold.append((f"{lo.date()}..{hi.date()} (no thr)", pd.Series(dtype=float)))
                continue
            sel = te[te[col] >= best_t] if side == "high" else te[te[col] <= best_t]
            per_fold.append((f"{lo.date()}..{hi.date()} thr={best_t}", sel["pnl_exit"]))
        summaries[f"{col} {side}"] = report(f"WALK-FORWARD FACTOR: {col} ({side})", per_fold)

    print("\n" + "=" * 90)
    print(f"{'rule':<38}{'n':>7}{'pooled PF':>12}{'folds>=1.25':>14}{'total$':>12}")
    print("=" * 90)
    for k, v in sorted(summaries.items(), key=lambda kv: -(kv[1]["pf"] if math.isfinite(kv[1]["pf"]) else -9)):
        print(f"{k:<38}{v['n']:>7}{v['pf']:>12.3f}{str(v['ok_folds']) + '/' + str(v['live_folds']):>14}{v['total']:>12.0f}")


if __name__ == "__main__":
    main()
