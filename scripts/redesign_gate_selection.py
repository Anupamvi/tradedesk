"""Decide which of the remaining live gates to keep on top of the winning rule.

Winning core (validated in redesign_final_validation.py):
    credit vertical, DTE >= 28, credit 25-30% of width, take profit 50%, no stop.

The live policy layers three more conditions on credit spreads: a regime map, a
flow-alignment floor, and the expected-move buffer. Each is tested here for
whether it adds or destroys out-of-sample performance, so the redesign keeps
only conditions that earn their place.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from redesign_combined_wf import folds, load, pf  # noqa: E402
from redesign_final_validation import attach_exit_dates, clean_window, concurrency  # noqa: E402

CORE = {"min_dte": 28, "credit_only": True, "min_premium": 0.25, "max_premium": 0.30}
ALLOWED_REGIMES = {"Bull Put": {"uptrend"}, "Bear Call": {"downtrend"}}


def core(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[
        frame["is_credit"]
        & (frame["dte"] >= 28)
        & frame["entry_credit_pct_width"].between(0.25, 0.30)
    ]


def gate(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    if name == "none":
        return frame
    if name == "regime":
        keep = frame.apply(
            lambda r: str(r.get("regime", "")).lower() in ALLOWED_REGIMES.get(r["direction"], set()),
            axis=1,
        )
        return frame[keep]
    if name == "regime_inverted":
        keep = frame.apply(
            lambda r: str(r.get("regime", "")).lower() not in ALLOWED_REGIMES.get(r["direction"], set()),
            axis=1,
        )
        return frame[keep]
    if name.startswith("flow>="):
        return frame[frame["flow_align"] >= float(name.split(">=")[1])]
    if name.startswith("em>="):
        return frame[frame["expected_move_ratio"] >= float(name.split(">=")[1])]
    if name.startswith("qw<="):
        return frame[frame["entry_quote_width_pct"] <= float(name.split("<=")[1])]
    if name.startswith("ivr<="):
        return frame[frame["iv_rank"] <= float(name.split("<=")[1])]
    if name.startswith("ivr>="):
        return frame[frame["iv_rank"] >= float(name.split(">=")[1])]
    if name.startswith("maxdte<="):
        return frame[frame["dte"] <= float(name.split("<=")[1])]
    raise ValueError(name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exit", default="tp50_-_no_stop____________________slip10pct.csv")
    args = ap.parse_args()

    clean, last_day = clean_window(attach_exit_dates(load(args.exit)))
    base = core(clean)
    fs = folds(clean)
    print(f"core population (unbiased): n={len(base)}  PF={pf(base['pnl_exit']):.3f}  "
          f"total=${base['pnl_exit'].sum():,.0f}")

    print("\n" + "=" * 100)
    print("GATE-BY-GATE TEST on top of the core (out-of-sample folds)")
    print("=" * 100)
    print(f"{'additional gate':<34}{'n':>7}{'PF':>8}{'win':>8}{'folds':>8}{'total$':>10}   per-fold PF")
    names = ["none", "regime", "regime_inverted",
             "flow>=0.0", "flow>=0.10", "flow>=0.20",
             "em>=0.50", "em>=0.75", "em>=1.00",
             "qw<=0.05", "qw<=0.10", "qw<=0.20",
             "ivr>=30", "ivr>=40", "ivr<=55",
             "maxdte<=45", "maxdte<=60"]
    for name in names:
        parts = []
        for _tr, te, _lo, _hi in fs:
            parts.append(gate(core(te), name))
        allp = pd.concat(parts) if parts else pd.DataFrame()
        if allp.empty:
            print(f"{name:<34}{0:>7}")
            continue
        ok = sum(1 for s in parts if len(s) and pf(s["pnl_exit"]) >= 1.25)
        live = sum(1 for s in parts if len(s))
        fold_str = " ".join(f"{pf(s['pnl_exit']):5.2f}" if len(s) else "  -- " for s in parts)
        print(f"{name:<34}{len(allp):>7}{pf(allp['pnl_exit']):>8.3f}"
              f"{(allp['pnl_exit'] > 0).mean():>8.1%}{str(ok) + '/' + str(live):>8}"
              f"{allp['pnl_exit'].sum():>10.0f}   {fold_str}")

    print("\n" + "=" * 100)
    print("REGIME BREAKDOWN of the core -- are range days really untradeable?")
    print("=" * 100)
    print(f"{'direction | regime':<34}{'n':>7}{'PF':>8}{'win':>8}{'avg$':>9}{'total$':>10}")
    for (dirn, reg), grp in base.groupby(["direction", "regime"]):
        print(f"{dirn + ' | ' + str(reg):<34}{len(grp):>7}{pf(grp['pnl_exit']):>8.3f}"
              f"{(grp['pnl_exit'] > 0).mean():>8.1%}{grp['pnl_exit'].mean():>9.2f}"
              f"{grp['pnl_exit'].sum():>10.0f}")

    print("\n" + "=" * 100)
    print("MONTHLY CONSISTENCY of the core, no regime gate (1 contract)")
    print("=" * 100)
    b = base.copy()
    b["month"] = b["asof"].dt.to_period("M")
    agg = b.groupby("month")["pnl_exit"].agg(["count", "sum"])
    agg["PF"] = b.groupby("month")["pnl_exit"].apply(pf)
    agg["win"] = b.groupby("month")["pnl_exit"].apply(lambda s: (s > 0).mean())
    print(agg.to_string(float_format=lambda v: f"{v:,.3f}"))
    print(f"profitable months: {(agg['sum'] > 0).sum()}/{len(agg)}")

    print("\n" + "=" * 100)
    print("CAPITAL & SCALE for the core rule")
    print("=" * 100)
    for cap in (0, 2, 3, 5):
        sel = base
        if cap:
            sel = (sel.sort_values("entry_credit_pct_width", ascending=False)
                      .groupby("asof", group_keys=False).head(cap))
        if sel.empty:
            continue
        risk = concurrency(sel)
        months = max(sel["asof"].dt.to_period("M").nunique(), 1)
        pm = sel["pnl_exit"].sum() / months
        peak = risk.max()
        eq = sel.sort_values("asof")["pnl_exit"].cumsum()
        dd = (eq - eq.cummax()).min()
        need = 10_000 / pm if pm > 0 else float("inf")
        label = "uncapped" if cap == 0 else f"top {cap}/session"
        print(f"\n  {label}:  n={len(sel)}  PF={pf(sel['pnl_exit']):.3f}  "
              f"sessions with a trade {sel['asof'].nunique()}/{clean['asof'].nunique()}")
        print(f"    P/L per month @1 contract : ${pm:,.0f}")
        print(f"    peak concurrent risk @1x  : ${peak:,.0f}   median ${risk.median():,.0f}")
        print(f"    max drawdown @1x          : ${dd:,.0f}")
        print(f"    contracts needed for $10k : {need:,.1f}")
        print(f"    -> peak buying power      : ${peak * need:,.0f}")
        print(f"    -> drawdown at that scale : ${dd * need:,.0f}")
        print(f"    what ${15_000:,.0f} of risk buys : ${pm * (15_000 / peak):,.0f}/month")


if __name__ == "__main__":
    main()
