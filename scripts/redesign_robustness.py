"""Robustness stress test for the candidate redesign rule.

A rule that only works at one exact set of thresholds is an artifact. This
script attacks the candidate from five directions:

  1. Threshold sensitivity  -- does PF survive small parameter changes?
  2. Slippage sensitivity   -- does it survive worse fills than modelled?
  3. Monthly consistency    -- is P/L steady or driven by one month?
  4. Component ablation     -- which conditions actually carry the result?
  5. Drawdown / concurrency -- what does it cost to run, and what is the worst
                               peak-to-trough on a risk-normalised basis?
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

CANDIDATE = {"min_dte": 28, "max_quote_width": 0.10, "credit_only": True, "min_premium": 0.20}


def summarize(frame: pd.DataFrame) -> tuple[int, float, float, float]:
    if frame.empty:
        return 0, float("nan"), float("nan"), 0.0
    p = frame["pnl_exit"]
    return len(p), pf(p), float((p > 0).mean()), float(p.sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exit", default="tp50_-_no_stop____________________slip10pct.csv")
    args = ap.parse_args()

    d = load(args.exit)
    fs = folds(d)

    def oos(rule: dict) -> tuple[int, float, float, float, int, int]:
        parts = [apply_rule(te, rule) for _tr, te, _lo, _hi in fs]
        allp = pd.concat(parts) if parts else pd.DataFrame()
        ok = sum(1 for s in parts if len(s) and pf(s["pnl_exit"]) >= 1.25)
        live = sum(1 for s in parts if len(s))
        n, p, w, t = summarize(allp)
        return n, p, w, t, ok, live

    print("=" * 96)
    print("1. THRESHOLD SENSITIVITY (out-of-sample; candidate is dte>=28, qw<=0.10, credit, prem>=0.20)")
    print("=" * 96)
    print(f"{'variant':<46}{'n':>7}{'PF':>8}{'win':>8}{'total$':>11}{'folds':>9}")
    for label, rule in [
        ("CANDIDATE", CANDIDATE),
        ("min_dte 21", {**CANDIDATE, "min_dte": 21}),
        ("min_dte 25", {**CANDIDATE, "min_dte": 25}),
        ("min_dte 30", {**CANDIDATE, "min_dte": 30}),
        ("min_dte 35", {**CANDIDATE, "min_dte": 35}),
        ("qw <= 0.06", {**CANDIDATE, "max_quote_width": 0.06}),
        ("qw <= 0.08", {**CANDIDATE, "max_quote_width": 0.08}),
        ("qw <= 0.15", {**CANDIDATE, "max_quote_width": 0.15}),
        ("qw <= 0.20", {**CANDIDATE, "max_quote_width": 0.20}),
        ("qw <= 0.35 (live cap)", {**CANDIDATE, "max_quote_width": 0.35}),
        ("prem >= 0.15", {**CANDIDATE, "min_premium": 0.15}),
        ("prem >= 0.18", {**CANDIDATE, "min_premium": 0.18}),
        ("prem >= 0.22", {**CANDIDATE, "min_premium": 0.22}),
        ("prem >= 0.25", {**CANDIDATE, "min_premium": 0.25}),
        ("prem 0.20-0.30 (live band top)", {**CANDIDATE, "min_premium": 0.20, "max_premium": 0.30}),
        ("prem 0.25-0.30 (LIVE BAND)", {**CANDIDATE, "min_premium": 0.25, "max_premium": 0.30}),
    ]:
        n, p, w, t, ok, live = oos(rule)
        print(f"{label:<46}{n:>7}{p:>8.3f}{w:>8.1%}{t:>11.0f}{str(ok) + '/' + str(live):>9}")

    print("\n" + "=" * 96)
    print("2. COMPONENT ABLATION (out-of-sample) -- drop one condition at a time")
    print("=" * 96)
    print(f"{'variant':<46}{'n':>7}{'PF':>8}{'win':>8}{'total$':>11}{'folds':>9}")
    base = dict(CANDIDATE)
    for label, rule in [
        ("CANDIDATE (all four)", base),
        ("drop dte gate", {k: v for k, v in base.items() if k != "min_dte"}),
        ("drop quote-width gate", {k: v for k, v in base.items() if k != "max_quote_width"}),
        ("drop credit-only", {k: v for k, v in base.items() if k != "credit_only"}),
        ("drop premium gate", {k: v for k, v in base.items() if k != "min_premium"}),
        ("dte gate ONLY", {"min_dte": 28}),
        ("quote-width gate ONLY", {"max_quote_width": 0.10}),
        ("credit-only ONLY", {"credit_only": True}),
        ("premium gate ONLY", {"min_premium": 0.20}),
        ("nothing (baseline)", {}),
    ]:
        n, p, w, t, ok, live = oos(rule)
        print(f"{label:<46}{n:>7}{p:>8.3f}{w:>8.1%}{t:>11.0f}{str(ok) + '/' + str(live):>9}")

    print("\n" + "=" * 96)
    print("3. SLIPPAGE SENSITIVITY (full sample, candidate rule)")
    print("=" * 96)
    print(f"{'exit / fill assumption':<46}{'n':>7}{'PF':>8}{'win':>8}{'avg$':>9}{'total$':>11}")
    for f in sorted(Path("out/redesign_exit_grid").glob("*.csv")):
        try:
            dd = load(f.name)
        except Exception:
            continue
        sel = apply_rule(dd, CANDIDATE)
        n, p, w, t = summarize(sel)
        avg = t / n if n else float("nan")
        print(f"{f.stem.replace('_', ' ').strip():<46}{n:>7}{p:>8.3f}{w:>8.1%}{avg:>9.2f}{t:>11.0f}")

    print("\n" + "=" * 96)
    print("4. MONTHLY CONSISTENCY (candidate rule, full sample, 1 contract per trade)")
    print("=" * 96)
    sel = apply_rule(d, CANDIDATE).copy()
    sel["month"] = sel["asof"].dt.to_period("M")
    rows = []
    for mth, grp in sel.groupby("month"):
        n, p, w, t = summarize(grp)
        rows.append({"month": str(mth), "n": n, "PF": round(p, 3), "win": f"{w:.1%}",
                     "total$": round(t), "days": grp["asof"].nunique()})
    md = pd.DataFrame(rows)
    print(md.to_string(index=False))
    pos = (md["total$"] > 0).sum()
    print(f"\nprofitable months: {pos}/{len(md)}")

    print("\n" + "=" * 96)
    print("5. RISK / DRAWDOWN (candidate rule, 1 contract per trade, chronological)")
    print("=" * 96)
    chron = sel.sort_values("asof")
    eq = chron["pnl_exit"].cumsum()
    peak = eq.cummax()
    dd = eq - peak
    print(f"  trades                : {len(chron)}")
    print(f"  final equity (1x)     : ${eq.iloc[-1]:,.0f}")
    print(f"  max drawdown (1x)     : ${dd.min():,.0f}")
    print(f"  median max-loss/contract: ${chron['max_loss'].median():,.0f}")
    print(f"  worst single trade    : ${chron['pnl_exit'].min():,.0f}")
    print(f"  best  single trade    : ${chron['pnl_exit'].max():,.0f}")
    per_day = chron.groupby("asof").size()
    print(f"  trades per session    : median {per_day.median():.0f}  p90 {per_day.quantile(0.9):.0f}  max {per_day.max():.0f}")
    print(f"  sessions with >=1     : {chron['asof'].nunique()} of {d['asof'].nunique()}")
    # Concurrency: open positions on any given day, assuming held to exit.
    print(f"  concurrent risk (1x, 30d overlap): ${per_day.median() * 30 * chron['max_loss'].median():,.0f}")


if __name__ == "__main__":
    main()
