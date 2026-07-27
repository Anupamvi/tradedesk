"""Combined-rule walk-forward under the corrected exit policy.

Tests whether a small, economically-motivated rule set clears PF >= 1.25
out-of-sample on rolling folds, and reports the realistic monthly P/L and
drawdown that follows from it -- including a per-day selection cap, because the
live pipeline takes a handful of trades per day, not the whole universe.
"""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd

HISTORY = Path("codexuw/history/codexdaily_v4_edge_history_v3_2026-07-23.csv.gz")
GRID_DIR = Path("out/redesign_exit_grid")


def truthy(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "1.0", "yes"})


def pf(pnl) -> float:
    p = pd.Series(pnl).dropna()
    if p.empty:
        return float("nan")
    g = p[p > 0].sum()
    l = -p[p < 0].sum()
    return g / l if l > 0 else float("inf")


def load(exit_csv: str) -> pd.DataFrame:
    d = pd.read_csv(HISTORY, low_memory=False)
    d["row_id"] = d.index
    for c in [
        "entry_credit_pct_width", "entry_debit_pct_width", "expected_move_ratio",
        "entry_quote_width_pct", "entry_width", "distance_pct", "reward_risk", "dte",
        "iv_rank", "iv_hv_ratio", "combined_flow_bias", "flow_bias", "entry_credit",
        "entry_debit", "source_contract_oi", "iv30d", "realized_volatility_30d",
        "bot_volume_oi_ratio", "stock_price_eod",
    ]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    ex = pd.read_csv(GRID_DIR / exit_csv)[["row_id", "pnl"]].rename(columns={"pnl": "pnl_exit"})
    m = d.merge(ex, on="row_id", how="inner")
    m = m[m["pnl_exit"].notna()].copy()
    m["asof"] = pd.to_datetime(m["asof"], errors="coerce")
    m["is_credit"] = m["direction"].isin(["Bull Put", "Bear Call"])
    sign = m["direction"].map({"Bull Put": 1, "Bull Call": 1, "Bear Call": -1, "Bear Put": -1})
    m["flow_align"] = m["combined_flow_bias"].fillna(m["flow_bias"]) * sign
    m["premium_pct"] = np.where(m["is_credit"], m["entry_credit_pct_width"], m["entry_debit_pct_width"])
    m["is_guard"] = truthy(m["replay_guard_pass"])
    # Max loss per contract, used for risk-normalised sizing and drawdown.
    m["max_loss"] = np.where(
        m["is_credit"],
        (m["entry_width"] - m["entry_credit"]) * 100.0,
        m["entry_debit"] * 100.0,
    )
    return m


def coverage(d: pd.DataFrame) -> None:
    print("\nfeature coverage (non-null share of rows):")
    for c in ["dte", "entry_quote_width_pct", "premium_pct", "expected_move_ratio",
              "iv_rank", "iv_hv_ratio", "flow_align", "source_contract_oi", "reward_risk"]:
        if c in d.columns:
            print(f"  {c:<26}{d[c].notna().mean():>7.1%}   median={d[c].median():.4f}")


def folds(d: pd.DataFrame, n_folds: int = 5, min_train_days: int = 40):
    days = sorted(d["asof"].dropna().unique())
    tail = days[min_train_days:]
    out = []
    for ch in np.array_split(np.array(tail), n_folds):
        if len(ch) == 0:
            continue
        lo, hi = ch[0], ch[-1]
        out.append((d[d["asof"] < lo], d[(d["asof"] >= lo) & (d["asof"] <= hi)],
                    pd.Timestamp(lo), pd.Timestamp(hi)))
    return out


def apply_rule(frame: pd.DataFrame, rule: dict) -> pd.DataFrame:
    sel = frame
    if "min_dte" in rule:
        sel = sel[sel["dte"] >= rule["min_dte"]]
    if "max_dte" in rule:
        sel = sel[sel["dte"] <= rule["max_dte"]]
    if "max_quote_width" in rule:
        sel = sel[sel["entry_quote_width_pct"] <= rule["max_quote_width"]]
    if "min_premium" in rule:
        sel = sel[sel["premium_pct"] >= rule["min_premium"]]
    if "max_premium" in rule:
        sel = sel[sel["premium_pct"] <= rule["max_premium"]]
    if "min_oi" in rule:
        sel = sel[sel["source_contract_oi"] >= rule["min_oi"]]
    if "credit_only" in rule and rule["credit_only"]:
        sel = sel[sel["is_credit"]]
    return sel


def cap_per_day(frame: pd.DataFrame, k: int, score: str) -> pd.DataFrame:
    if k <= 0 or frame.empty:
        return frame
    return (frame.sort_values(score, ascending=False)
                 .groupby("asof", group_keys=False)
                 .head(k))


def evaluate(frame: pd.DataFrame, label: str, verbose: bool = True) -> dict:
    p = frame["pnl_exit"]
    if p.empty:
        return {"label": label, "n": 0, "pf": float("nan"), "total": 0.0, "per_month": 0.0}
    months = max(frame["asof"].dt.to_period("M").nunique(), 1)
    res = {
        "label": label,
        "n": int(len(p)),
        "pf": pf(p),
        "win": float((p > 0).mean()),
        "avg": float(p.mean()),
        "total": float(p.sum()),
        "per_month": float(p.sum()) / months,
        "trades_per_month": len(p) / months,
        "avg_max_loss": float(frame["max_loss"].median()),
    }
    if verbose:
        print(f"  {label:<44} n={res['n']:>5} PF={res['pf']:>6.3f} win={res['win']:>6.1%} "
              f"avg=${res['avg']:>7.2f} $/mo={res['per_month']:>8.0f} n/mo={res['trades_per_month']:>5.1f}")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exit", default="tp50_-_no_stop____________________slip10pct.csv")
    ap.add_argument("--cap", type=int, default=0, help="max trades per day (0 = uncapped)")
    args = ap.parse_args()

    d = load(args.exit)
    print(f"exit policy      : {args.exit}")
    print(f"rows {len(d)}  sessions {d['asof'].nunique()}  baseline PF {pf(d['pnl_exit']):.3f}")
    coverage(d)

    rules = {
        "baseline (no rule)": {},
        "dte>=21": {"min_dte": 21},
        "dte>=21 + qw<=0.10": {"min_dte": 21, "max_quote_width": 0.10},
        "dte>=21 + qw<=0.10 + prem>=0.20": {"min_dte": 21, "max_quote_width": 0.10, "min_premium": 0.20},
        "dte>=21 + qw<=0.15 + prem>=0.20": {"min_dte": 21, "max_quote_width": 0.15, "min_premium": 0.20},
        "dte 21-45 + qw<=0.10 + credit": {"min_dte": 21, "max_dte": 45, "max_quote_width": 0.10, "credit_only": True},
        "dte>=21 + qw<=0.10 + credit + prem>=0.20": {"min_dte": 21, "max_quote_width": 0.10,
                                                      "credit_only": True, "min_premium": 0.20},
        "dte>=28 + qw<=0.10 + credit + prem>=0.20": {"min_dte": 28, "max_quote_width": 0.10,
                                                      "credit_only": True, "min_premium": 0.20},
        "dte>=21 + qw<=0.10 + oi>=500": {"min_dte": 21, "max_quote_width": 0.10, "min_oi": 500},
        "CURRENT LIVE GUARD": None,
    }

    print("\n=== FULL SAMPLE (in-sample, for orientation only) ===")
    for label, rule in rules.items():
        sel = d[d["is_guard"]] if rule is None else apply_rule(d, rule)
        if args.cap:
            sel = cap_per_day(sel, args.cap, "premium_pct")
        evaluate(sel, label)

    fs = folds(d)
    print(f"\n=== ROLLING OUT-OF-SAMPLE FOLDS (n={len(fs)}) ===")
    summary = []
    for label, rule in rules.items():
        per_fold = []
        for _tr, te, lo, hi in fs:
            sel = te[te["is_guard"]] if rule is None else apply_rule(te, rule)
            if args.cap:
                sel = cap_per_day(sel, args.cap, "premium_pct")
            per_fold.append((f"{lo.date()}..{hi.date()}", sel))
        allp = pd.concat([s for _, s in per_fold]) if per_fold else pd.DataFrame()
        if allp.empty:
            continue
        ok = sum(1 for _, s in per_fold if len(s) and pf(s["pnl_exit"]) >= 1.25)
        live = sum(1 for _, s in per_fold if len(s))
        r = evaluate(allp, label, verbose=False)
        r["ok_folds"], r["live_folds"] = ok, live
        r["fold_pfs"] = [pf(s["pnl_exit"]) if len(s) else float("nan") for _, s in per_fold]
        summary.append(r)

    summary.sort(key=lambda r: -(r["pf"] if math.isfinite(r["pf"]) else -9))
    print(f"\n{'rule':<44}{'n':>6}{'PF':>8}{'win':>7}{'$/mo':>9}{'n/mo':>7}{'folds':>8}")
    print("-" * 92)
    for r in summary:
        print(f"{r['label']:<44}{r['n']:>6}{r['pf']:>8.3f}{r['win']:>7.1%}"
              f"{r['per_month']:>9.0f}{r['trades_per_month']:>7.1f}"
              f"{str(r['ok_folds']) + '/' + str(r['live_folds']):>8}")
        print("      per-fold PF: " + "  ".join(f"{v:5.2f}" if math.isfinite(v) else "  --  " for v in r["fold_pfs"]))


if __name__ == "__main__":
    main()
