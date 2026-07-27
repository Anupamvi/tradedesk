"""Validate the two remaining new credit gates on replayed P&L.

Both were justified from `vrp_capture`, the proxy that has already failed to
reproduce on real vertical P&L, so neither can be shipped on that basis alone.

  1. MAX_DTE_EARNINGS_EXCLUSION = 21 -- skip names with earnings inside the hold.
     The pipeline already hard-rejects `earnings_crosses_expiry`, so this may be
     redundant.
  2. MIN_FLOW_ALIGNMENT = 0.10 -- removed earlier this session because flow was
     shown to carry no directional information. Removing a filter is a loosening,
     so it has to be justified on P&L or restored.

Everything is measured inside the validated regime map, since that is the only
population the strategy actually trades, and resampled by session.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HISTORY = "codexuw/history/codexdaily_v4_edge_history_v4_2026-07-26.csv.gz"
PANEL = "/Users/anuppamvi/uw_root/tradedesk/out/research/price_panel.csv.gz"
RNG = np.random.default_rng(23)
BOOT = 3000


def pf(p: pd.Series) -> float:
    w, l = p[p > 0].sum(), -p[p < 0].sum()
    return float(w / l) if l > 0 else float("inf")


def load() -> pd.DataFrame:
    h = pd.read_csv(HISTORY, low_memory=False)
    h = h[(h["evaluated"] == True) & h["pnl_1x"].notna()]  # noqa: E712
    h = h[h["strategy_kind"] == "Credit"].copy()
    panel = pd.read_csv(PANEL, usecols=["asof", "ticker", "rv21_ann"], low_memory=False)
    panel = panel.dropna(subset=["rv21_ann"]).drop_duplicates(["asof", "ticker"])
    h["asof"] = h["asof"].astype(str)
    panel["asof"] = panel["asof"].astype(str)
    m = h.merge(panel, on=["asof", "ticker"], how="left")
    m["rv_true"] = pd.to_numeric(m["rv21_ann"], errors="coerce")
    m["allowed"] = (((m["direction"] == "Bull Put") & (m["regime"] == "downtrend"))
                    | ((m["direction"] == "Bear Call") & (m["regime"] == "uptrend")))
    ed = pd.to_datetime(m["next_earnings_dt"], errors="coerce")
    ao = pd.to_datetime(m["asof"], errors="coerce")
    m["days_to_earn"] = (ed - ao).dt.days
    m["align"] = pd.to_numeric(m["combined_flow_bias"], errors="coerce") * np.where(
        m["direction"] == "Bull Put", 1.0, -1.0)
    return m


def delta_test(base: pd.DataFrame, sub: pd.DataFrame, label: str) -> None:
    """Does `sub` beat `base` in mean P&L, resampling whole sessions?"""
    if len(sub) < 20:
        print(f"  {label:<40} n {len(sub):>4}  (too few to test)")
        return
    days = base["asof"].unique()
    bd = {k: v["pnl_1x"].to_numpy() for k, v in base.groupby("asof")}
    sd = {k: v["pnl_1x"].to_numpy() for k, v in sub.groupby("asof")}
    diffs = []
    for _ in range(BOOT):
        pick = RNG.choice(days, len(days), replace=True)
        sv = [sd[k] for k in pick if k in sd]
        if not sv:
            continue
        diffs.append(np.concatenate(sv).mean() - np.concatenate([bd[k] for k in pick]).mean())
    diffs = np.array(diffs)
    p = sub["pnl_1x"]
    print(f"  {label:<40} n {len(sub):>4}  win {100*(p>0).mean():>5.1f}%  "
          f"avg {p.mean():>+7.1f}  PF {pf(p):>5.2f}   delta {diffs.mean():>+6.1f}  "
          f"90% CI [{np.percentile(diffs,5):>+6.1f},{np.percentile(diffs,95):>+6.1f}]  "
          f"p(no gain) {np.mean(diffs <= 0):.3f}")


def main() -> None:
    m = load()
    a = m[m["allowed"] & (m["rv_true"] >= 0.15)].copy()
    p = a["pnl_1x"]
    print(f"baseline = regime map + RV>=0.15 : n {len(a)}  days {a['asof'].nunique()}  "
          f"win {100*(p>0).mean():.1f}%  avg {p.mean():+.1f}  PF {pf(p):.2f}")

    print("\n=== 1. earnings exclusion ===")
    print(f"  rows with a known earnings date: {a['days_to_earn'].notna().mean()*100:.0f}%")
    inwin = a["days_to_earn"].between(0, 21)
    print(f"  rows with earnings inside 21d  : {inwin.sum()} of {len(a)}")
    s = a[inwin]
    if len(s) >= 20:
        pp = s["pnl_1x"]
        print(f"  earnings-in-window trades themselves: n {len(s)}  "
              f"win {100*(pp>0).mean():.1f}%  avg {pp.mean():+.1f}  PF {pf(pp):.2f}")
    delta_test(a, a[~inwin], "exclude earnings within 21d")
    for d in (7, 14, 30):
        delta_test(a, a[~a["days_to_earn"].between(0, d)], f"exclude earnings within {d}d")

    print("\n=== 2. flow alignment gate ===")
    print(f"  rows with usable combined_flow_bias: {a['align'].notna().mean()*100:.0f}%")
    for thr in (0.05, 0.10, 0.20):
        delta_test(a, a[a["align"] >= thr], f"require align >= {thr:.2f}")
    delta_test(a, a[a["align"] < 0.10], "  (control) align < 0.10 instead")

    print("\n=== 3. combined: what the shipping config should be ===")
    keep = a[~inwin]
    delta_test(a, keep, "regime map + RV floor + no earnings")
    delta_test(a, keep[keep["align"] >= 0.10], "  ... + align >= 0.10")


if __name__ == "__main__":
    main()
