"""Replayed spread P&L by regime, conditioned on volatility richness.

The live credit policy blocks `range` sessions. That map was fit on directional
P&L. This re-tests it on the recorded replay history using the SAME richness gate
the live path now applies (IV/HV >= 1.30 and RV >= 15%).

The recorded history was produced before realised vol was wired in, so
`iv_hv_ratio` is null there. Realised vol is joined back from the research price
panel on (asof, ticker) -- the same point-in-time 21d close-to-close annualised
series that `codexuw.realized_vol.attach_realized_vol` computes live -- and the
ratio is rebuilt against the history's own `iv30d`, i.e. the value the pipeline
actually saw that morning.
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

HISTORY = "codexuw/history/codexdaily_v4_edge_history_v4_2026-07-26.csv.gz"
PANEL = "/Users/anuppamvi/uw_root/tradedesk/out/research/price_panel.csv.gz"


def profit_factor(p: pd.Series) -> float:
    win = p[p > 0].sum()
    loss = -p[p < 0].sum()
    return float(win / loss) if loss > 0 else float("inf")


def load() -> pd.DataFrame:
    h = pd.read_csv(HISTORY, low_memory=False)
    h = h[(h["evaluated"] == True) & h["pnl_1x"].notna()].copy()  # noqa: E712

    panel = pd.read_csv(PANEL, usecols=["asof", "ticker", "rv21_ann"], low_memory=False)
    panel = panel.dropna(subset=["rv21_ann"]).drop_duplicates(["asof", "ticker"])

    h["asof"] = h["asof"].astype(str)
    panel["asof"] = panel["asof"].astype(str)
    m = h.merge(panel, on=["asof", "ticker"], how="left")

    iv = pd.to_numeric(m["iv30d"], errors="coerce")
    rv = pd.to_numeric(m["rv21_ann"], errors="coerce")
    m["rv_true"] = rv
    m["ratio"] = iv / rv.where(rv > 0)
    return m


def block(d: pd.DataFrame, label: str, by: str = "regime") -> None:
    print(f"\n=== {label} ===")
    print(f"{by:<12}{'n':>6}{'win%':>8}{'avg':>9}{'median':>9}{'PF':>7}{'total':>10}{'days':>6}")
    order = ["uptrend", "range", "downtrend"]
    keys = [k for k in order if k in set(d[by])] if by == "regime" else sorted(set(d[by].dropna()))
    for k in keys:
        s = d[d[by] == k]
        if len(s) < 20:
            continue
        p = s["pnl_1x"]
        print(
            f"{str(k):<12}{len(s):>6}{100*(p>0).mean():>7.1f}%{p.mean():>+9.2f}"
            f"{p.median():>+9.2f}{profit_factor(p):>7.2f}{p.sum():>+10.0f}{s['asof'].nunique():>6}"
        )
    p = d["pnl_1x"]
    print(f"{'ALL':<12}{len(d):>6}{100*(p>0).mean():>7.1f}%{p.mean():>+9.2f}"
          f"{p.median():>+9.2f}{profit_factor(p):>7.2f}{p.sum():>+10.0f}{d['asof'].nunique():>6}")


def monthly(d: pd.DataFrame, reg: str) -> None:
    s = d[d["regime"] == reg]
    if s.empty:
        return
    g = s.groupby(s["asof"].str[:7])["pnl_1x"]
    txt = " | ".join(f"{k} {v.mean():+.0f}(n{len(v)})" for k, v in g)
    print(f"  {reg:<10} {txt}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-ratio", type=float, default=1.30)
    ap.add_argument("--min-rv", type=float, default=0.15)
    args = ap.parse_args()

    m = load()
    credit = m[m["strategy_kind"] == "Credit"].copy()
    cov = credit["ratio"].notna().mean()
    print(f"credit replay rows {len(credit):,}  days {credit['asof'].nunique()}  "
          f"realised-vol join coverage {100*cov:.1f}%")

    block(credit, "CREDIT, ALL (this is what the regime map was fit on)")

    rich = credit[(credit["ratio"] >= args.min_ratio) & (credit["rv_true"] >= args.min_rv)]
    block(rich, f"CREDIT, RICH ONLY: IV/HV>={args.min_ratio} and RV>={args.min_rv}")

    cheap = credit[(credit["ratio"] < args.min_ratio) & credit["ratio"].notna()]
    block(cheap, f"CREDIT, CHEAP (IV/HV<{args.min_ratio}) -- what the old gates let through")

    print("\n--- RICH credit, monthly mean P&L by regime ---")
    for reg in ["uptrend", "range", "downtrend"]:
        monthly(rich, reg)

    print("\n--- RICH credit, by direction ---")
    block(rich, "RICH credit by direction", by="direction")

    print("\n--- ratio sweep on credit replay P&L (all regimes) ---")
    print(f"{'min ratio':<11}{'n':>6}{'win%':>8}{'avg':>9}{'PF':>7}")
    for thr in [0.0, 0.90, 1.00, 1.10, 1.20, 1.30, 1.40, 1.50]:
        s = credit[(credit["ratio"] >= thr) & (credit["rv_true"] >= args.min_rv)]
        if len(s) < 20:
            continue
        p = s["pnl_1x"]
        print(f"{thr:<11.2f}{len(s):>6}{100*(p>0).mean():>7.1f}%{p.mean():>+9.2f}{profit_factor(p):>7.2f}")


if __name__ == "__main__":
    main()
