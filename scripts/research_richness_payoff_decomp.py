"""Why does replayed vertical-spread P&L get WORSE as IV/HV richness rises?

`vrp_capture` (= (iv30d - realised_fwd_vol)/iv30d) says richness is strongly
profitable and monotone. Replayed vertical P&L says the opposite. Both cannot be
right about the same trade, so this decomposes the difference.

A variance swap monetises the whole premium continuously. A defined-risk vertical
caps the gain at the credit and caps the loss at (width - credit), typically 3x
the credit. So win RATE can stay flat while the payoff still deteriorates, if the
losses that do occur become full-width instead of partial. That is the first
thing checked here.

Second, a strike-placement confound: if strikes are chosen by percentage distance
rather than by expected-move multiples, then a high-IV name gets a short strike
that is closer in SIGMA terms, mechanically raising breach severity. That would
make this an execution bug rather than a statement about the variance premium.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HISTORY = "codexuw/history/codexdaily_v4_edge_history_v4_2026-07-26.csv.gz"
PANEL = "/Users/anuppamvi/uw_root/tradedesk/out/research/price_panel.csv.gz"


def profit_factor(p: pd.Series) -> float:
    w, l = p[p > 0].sum(), -p[p < 0].sum()
    return float(w / l) if l > 0 else float("inf")


def load() -> pd.DataFrame:
    h = pd.read_csv(HISTORY, low_memory=False)
    h = h[(h["evaluated"] == True) & h["pnl_1x"].notna()].copy()  # noqa: E712
    h = h[h["strategy_kind"] == "Credit"]

    panel = pd.read_csv(PANEL, usecols=["asof", "ticker", "rv21_ann"], low_memory=False)
    panel = panel.dropna(subset=["rv21_ann"]).drop_duplicates(["asof", "ticker"])
    h["asof"] = h["asof"].astype(str)
    panel["asof"] = panel["asof"].astype(str)
    m = h.merge(panel, on=["asof", "ticker"], how="left")

    iv = pd.to_numeric(m["iv30d"], errors="coerce")
    rv = pd.to_numeric(m["rv21_ann"], errors="coerce")
    m["rv_true"] = rv
    m["ratio"] = iv / rv.where(rv > 0)
    m = m[m["ratio"].notna() & (m["rv_true"] >= 0.15)]

    # how far is the short strike, measured in standard deviations of the move
    dte = pd.to_numeric(m["dte"], errors="coerce")
    px = pd.to_numeric(m["stock_price_eod"], errors="coerce")
    dist = (pd.to_numeric(m["short_strike_eod"], errors="coerce") - px).abs()
    m["dist_pct"] = dist / px.where(px > 0)
    m["em_sigma"] = m["dist_pct"] / (iv * np.sqrt(dte.clip(lower=1) / 365.0))

    credit = pd.to_numeric(m["entry_credit"], errors="coerce")
    width = pd.to_numeric(m["entry_width"], errors="coerce")
    m["credit_"] = credit
    m["width_"] = width
    m["max_loss_"] = (width - credit) * 100.0
    m["loss_frac"] = np.where(m["pnl_1x"] < 0, -m["pnl_1x"] / m["max_loss_"].where(m["max_loss_"] > 0), np.nan)
    m["bucket"] = pd.cut(m["ratio"], [0, 1.0, 1.15, 1.30, 1.5, 99],
                         labels=["<1.00", "1.00-1.15", "1.15-1.30", "1.30-1.50", ">1.50"])
    m = m[m["em_sigma"].notna()]
    return m


def main() -> None:
    m = load()
    print(f"credit rows with usable ratio and RV>=15%: {len(m):,}  days {m['asof'].nunique()}")

    print("\n--- payoff decomposition by IV/HV bucket ---")
    print(f"{'bucket':<11}{'n':>5}{'win%':>7}{'avgWin':>8}{'avgLoss':>9}{'lossFrac':>9}"
          f"{'maxLossHit':>11}{'credit%w':>9}{'emSigma':>8}{'PF':>6}")
    for b in m["bucket"].cat.categories:
        s = m[m["bucket"] == b]
        if len(s) < 20:
            continue
        p = s["pnl_1x"]
        wins, losses = p[p > 0], p[p < 0]
        full = (s["loss_frac"] >= 0.95).mean()
        cw = (s["credit_"] / s["width_"].where(s["width_"] > 0)).median()
        print(f"{str(b):<11}{len(s):>5}{100*(p>0).mean():>6.1f}%{wins.mean():>+8.0f}"
              f"{losses.mean():>+9.0f}{s['loss_frac'].mean():>9.2f}{100*full:>10.1f}%"
              f"{cw:>9.3f}{s['em_sigma'].median():>8.2f}{profit_factor(p):>6.2f}")

    print("\n--- is it a strike-placement artifact? P&L by expected-move sigma ---")
    m["em_b"] = pd.qcut(m["em_sigma"], 4, labels=["Q1 closest", "Q2", "Q3", "Q4 furthest"])
    print(f"{'em bucket':<13}{'n':>5}{'medSigma':>10}{'win%':>7}{'avg':>8}{'PF':>6}")
    for b in m["em_b"].cat.categories:
        s = m[m["em_b"] == b]
        p = s["pnl_1x"]
        print(f"{str(b):<13}{len(s):>5}{s['em_sigma'].median():>10.2f}"
              f"{100*(p>0).mean():>6.1f}%{p.mean():>+8.0f}{profit_factor(p):>6.2f}")

    print("\n--- richness effect HOLDING expected-move distance fixed ---")
    print("(if richness is still bad inside every distance bucket, it is not a strike bug)")
    print(f"{'em bucket':<13}{'rich n':>7}{'rich PF':>9}{'rich avg':>10}"
          f"{'cheap n':>9}{'cheap PF':>10}{'cheap avg':>11}")
    for b in m["em_b"].cat.categories:
        s = m[m["em_b"] == b]
        r, c = s[s["ratio"] >= 1.30], s[s["ratio"] < 1.30]
        if len(r) < 15 or len(c) < 15:
            continue
        print(f"{str(b):<13}{len(r):>7}{profit_factor(r['pnl_1x']):>9.2f}{r['pnl_1x'].mean():>+10.0f}"
              f"{len(c):>9}{profit_factor(c['pnl_1x']):>10.2f}{c['pnl_1x'].mean():>+11.0f}")

    print("\n--- regime x direction on real P&L (the live map is contrarian) ---")
    print(f"{'regime':<11}{'direction':<11}{'n':>5}{'win%':>7}{'avg':>8}{'PF':>6}{'total':>9}")
    for reg in ["uptrend", "range", "downtrend"]:
        for d in ["Bull Put", "Bear Call"]:
            s = m[(m["regime"] == reg) & (m["direction"] == d)]
            if len(s) < 20:
                continue
            p = s["pnl_1x"]
            print(f"{reg:<11}{d:<11}{len(s):>5}{100*(p>0).mean():>6.1f}%"
                  f"{p.mean():>+8.0f}{profit_factor(p):>6.2f}{p.sum():>+9.0f}")

    print("\n--- what the CURRENT live regime map would have traded ---")
    allowed = ((m["direction"] == "Bull Put") & (m["regime"] == "downtrend")) | \
              ((m["direction"] == "Bear Call") & (m["regime"] == "uptrend"))
    for name, s in [("map-allowed", m[allowed]), ("map-blocked", m[~allowed])]:
        p = s["pnl_1x"]
        print(f"  {name:<13} n {len(s):>5}  win {100*(p>0).mean():>5.1f}%  "
              f"avg {p.mean():>+7.0f}  PF {profit_factor(p):>5.2f}  total {p.sum():>+8.0f}")
    print("\n  map-allowed, split by richness:")
    for lo, hi, lab in [(0, 1.15, "cheap <1.15"), (1.15, 1.30, "mid"), (1.30, 99, "rich >=1.30")]:
        s = m[allowed & (m["ratio"] >= lo) & (m["ratio"] < hi)]
        if len(s) < 15:
            continue
        p = s["pnl_1x"]
        print(f"    {lab:<13} n {len(s):>4}  win {100*(p>0).mean():>5.1f}%  "
              f"avg {p.mean():>+7.0f}  PF {profit_factor(p):>5.2f}")


if __name__ == "__main__":
    main()
