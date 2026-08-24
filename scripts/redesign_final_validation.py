"""Final honest validation of the redesign, with survivorship and capital checks.

Two integrity issues the earlier passes did not handle:

  1. TRUNCATION BIAS. Trades entered near the end of the data window can only be
     recorded if they resolved before the window ends. With a 50% profit target
     and no stop, winners resolve early and losers run to expiry -- so the last
     weeks of the sample are biased toward winners. Any month whose trades could
     not all have resolved is excluded from the headline.

  2. CAPITAL. Concurrency is computed from actual exit days, not a flat 30-day
     assumption, so the buying-power requirement is real rather than notional.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from redesign_combined_wf import apply_rule, folds, load, pf  # noqa: E402

RULES = {
    "A  dte>=28 only": {"min_dte": 28},
    "B  dte>=28 + credit": {"min_dte": 28, "credit_only": True},
    "C  dte>=28 + credit + prem>=0.20": {"min_dte": 28, "credit_only": True, "min_premium": 0.20},
    "D  dte>=28 + credit + prem 0.25-0.30": {"min_dte": 28, "credit_only": True,
                                              "min_premium": 0.25, "max_premium": 0.30},
    "E  dte>=28 + credit + prem>=0.20 + qw<=0.10": {"min_dte": 28, "credit_only": True,
                                                     "min_premium": 0.20, "max_quote_width": 0.10},
}


def attach_exit_dates(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["exit_dt"] = pd.to_datetime(d["exit_day"], errors="coerce")
    d["expiry_dt"] = pd.to_datetime(d["expiry"], errors="coerce")
    d["hold_days"] = (d["exit_dt"] - d["asof"]).dt.days
    return d


def clean_window(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Drop entries that could not have resolved inside the data window."""
    last_day = d["asof"].max()
    # A trade is only unbiased if its expiry falls inside the observed window.
    ok = d["expiry_dt"] <= last_day
    return d[ok].copy(), last_day


def concurrency(sel: pd.DataFrame) -> pd.Series:
    """Open-position risk per calendar day, from entry to actual exit."""
    if sel.empty:
        return pd.Series(dtype=float)
    days = pd.date_range(sel["asof"].min(), sel["exit_dt"].max(), freq="D")
    risk = pd.Series(0.0, index=days)
    for _, r in sel.iterrows():
        if pd.isna(r["exit_dt"]):
            continue
        risk.loc[r["asof"]:r["exit_dt"]] += r["max_loss"]
    return risk


def block(title: str) -> None:
    print("\n" + "=" * 94)
    print(title)
    print("=" * 94)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exit", default="tp50_-_no_stop____________________slip10pct.csv")
    args = ap.parse_args()

    raw = attach_exit_dates(load(args.exit))
    clean, last_day = clean_window(raw)
    print(f"exit policy   : {args.exit}")
    print(f"all rows      : {len(raw)}   last session {last_day.date()}")
    print(f"unbiased rows : {len(clean)}  (expiry inside window)  dropped {len(raw) - len(clean)}")

    block("1. TRUNCATION CHECK -- headline PF with and without unresolved entries")
    print(f"{'rule':<44}{'raw n':>7}{'raw PF':>9}{'clean n':>9}{'clean PF':>10}{'clean $':>10}")
    for label, rule in RULES.items():
        r_all, r_cln = apply_rule(raw, rule), apply_rule(clean, rule)
        print(f"{label:<44}{len(r_all):>7}{pf(r_all['pnl_exit']):>9.3f}"
              f"{len(r_cln):>9}{pf(r_cln['pnl_exit']):>10.3f}{r_cln['pnl_exit'].sum():>10.0f}")

    block("2. ROLLING OUT-OF-SAMPLE FOLDS on the unbiased sample")
    fs = folds(clean)
    print(f"{'rule':<44}{'n':>7}{'PF':>8}{'win':>8}{'folds':>8}{'total$':>10}")
    for label, rule in RULES.items():
        parts = [apply_rule(te, rule) for _tr, te, _lo, _hi in fs]
        allp = pd.concat(parts) if parts else pd.DataFrame()
        if allp.empty:
            continue
        ok = sum(1 for s in parts if len(s) and pf(s["pnl_exit"]) >= 1.25)
        live = sum(1 for s in parts if len(s))
        print(f"{label:<44}{len(allp):>7}{pf(allp['pnl_exit']):>8.3f}"
              f"{(allp['pnl_exit'] > 0).mean():>8.1%}{str(ok) + '/' + str(live):>8}"
              f"{allp['pnl_exit'].sum():>10.0f}")
        print("      per-fold PF: " + "  ".join(
            f"{pf(s['pnl_exit']):5.2f}" if len(s) else "  --  " for s in parts))

    block("3. MONTHLY P/L on the unbiased sample (1 contract per trade)")
    for label, rule in RULES.items():
        sel = apply_rule(clean, rule).copy()
        if sel.empty:
            continue
        sel["month"] = sel["asof"].dt.to_period("M")
        agg = sel.groupby("month")["pnl_exit"].agg(["count", "sum"])
        agg["PF"] = sel.groupby("month")["pnl_exit"].apply(pf)
        pos = int((agg["sum"] > 0).sum())
        print(f"\n{label}   profitable months {pos}/{len(agg)}")
        print("   " + "  ".join(f"{str(m)[-2:]}:${v:>7,.0f}" for m, v in agg["sum"].items()))

    block("4. PER-DAY SELECTION CAP -- does the edge survive taking only the best few?")
    best_rule = RULES["C  dte>=28 + credit + prem>=0.20"]
    print(f"{'cap (ranked by premium %)':<44}{'n':>7}{'PF':>8}{'win':>8}{'$/mo':>9}{'n/day':>8}")
    for cap in (0, 1, 2, 3, 5, 8):
        sel = apply_rule(clean, best_rule)
        if cap:
            sel = (sel.sort_values("premium_pct", ascending=False)
                      .groupby("asof", group_keys=False).head(cap))
        if sel.empty:
            continue
        months = max(sel["asof"].dt.to_period("M").nunique(), 1)
        label = "uncapped" if cap == 0 else f"top {cap} per session"
        print(f"{label:<44}{len(sel):>7}{pf(sel['pnl_exit']):>8.3f}"
              f"{(sel['pnl_exit'] > 0).mean():>8.1%}{sel['pnl_exit'].sum() / months:>9.0f}"
              f"{len(sel) / sel['asof'].nunique():>8.1f}")

    block("5. CAPITAL REQUIREMENT -- real concurrency from actual exit days")
    for cap in (0, 2, 3, 5):
        sel = apply_rule(clean, best_rule)
        if cap:
            sel = (sel.sort_values("premium_pct", ascending=False)
                      .groupby("asof", group_keys=False).head(cap))
        if sel.empty:
            continue
        risk = concurrency(sel)
        months = max(sel["asof"].dt.to_period("M").nunique(), 1)
        pm = sel["pnl_exit"].sum() / months
        peak = risk.max()
        label = "uncapped" if cap == 0 else f"top {cap}/session"
        contracts_for_10k = 10_000 / pm if pm > 0 else float("inf")
        print(f"\n  {label}")
        print(f"    median hold days       : {sel['hold_days'].median():.0f}")
        print(f"    peak concurrent risk 1x: ${peak:,.0f}   median ${risk.median():,.0f}")
        print(f"    P/L per month at 1x    : ${pm:,.0f}")
        print(f"    contracts for $10k/mo  : {contracts_for_10k:,.1f}")
        print(f"    peak buying power then : ${peak * contracts_for_10k:,.0f}")
        eq = sel.sort_values("asof")["pnl_exit"].cumsum()
        print(f"    max drawdown 1x        : ${(eq - eq.cummax()).min():,.0f}"
              f"   -> at $10k/mo scale ${(eq - eq.cummax()).min() * contracts_for_10k:,.0f}")


if __name__ == "__main__":
    main()
