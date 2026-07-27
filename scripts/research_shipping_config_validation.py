"""Final validation of the shipping credit configuration on replayed P&L.

The unconditional richness sweep said richness HURTS. That was a confound: rich
candidates are disproportionately located in the regimes the live map already
blocks (range, and the wrong-direction leg), which is where the losses are. Once
the regime map is applied, the ordering reverses and richness helps monotonically.

This checks the configuration that is actually being shipped:
    regime map (Bull Put in downtrend, Bear Call in uptrend)
  + IV/HV >= 1.30
  + realised vol >= 0.15

and asks whether the richness ordering is stable rather than a small-sample
artefact -- by month, by leg of the map, and against a shuffled-richness control.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HISTORY = "codexuw/history/codexdaily_v4_edge_history_v4_2026-07-26.csv.gz"
PANEL = "/Users/anuppamvi/uw_root/tradedesk/out/research/price_panel.csv.gz"
RNG = np.random.default_rng(7)


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
    iv = pd.to_numeric(m["iv30d"], errors="coerce")
    rv = pd.to_numeric(m["rv21_ann"], errors="coerce")
    m["rv_true"] = rv
    m["ratio"] = iv / rv.where(rv > 0)
    m["allowed"] = (((m["direction"] == "Bull Put") & (m["regime"] == "downtrend"))
                    | ((m["direction"] == "Bear Call") & (m["regime"] == "uptrend")))
    return m[m["ratio"].notna() & m["rv_true"].notna()]


def line(tag: str, s: pd.DataFrame) -> None:
    p = s["pnl_1x"]
    if len(s) == 0:
        return
    print(f"  {tag:<34} n {len(s):>4}  win {100*(p>0).mean():>5.1f}%  "
          f"avg {p.mean():>+7.1f}  PF {pf(p):>5.2f}  total {p.sum():>+8.0f}  days {s['asof'].nunique():>3}")


def main() -> None:
    m = load()
    print(f"credit replay rows {len(m):,}  days {m['asof'].nunique()}")

    print("\n=== 1. the confound, stated plainly ===")
    rich = m["ratio"] >= 1.30
    print(f"  share of RICH candidates that sit in a regime the map BLOCKS: "
          f"{100*(~m.loc[rich,'allowed']).mean():.1f}%")
    print(f"  share of CHEAP candidates that sit in a blocked regime:       "
          f"{100*(~m.loc[~rich,'allowed']).mean():.1f}%")
    print("  -> richness looked toxic unconditionally because rich names cluster")
    print("     in the regimes that were already known to lose.")

    print("\n=== 2. richness ordering INSIDE the regime map ===")
    a = m[m["allowed"] & (m["rv_true"] >= 0.15)]
    for lo, hi, lab in [(0, 1.00, "IV/HV < 1.00"), (1.00, 1.15, "1.00 - 1.15"),
                        (1.15, 1.30, "1.15 - 1.30"), (1.30, 1.50, "1.30 - 1.50"),
                        (1.50, 99, ">= 1.50")]:
        line(lab, a[(a["ratio"] >= lo) & (a["ratio"] < hi)])

    print("\n=== 3. richness ordering OUTSIDE the map (should stay bad) ===")
    b = m[~m["allowed"] & (m["rv_true"] >= 0.15)]
    for lo, hi, lab in [(0, 1.15, "blocked, cheap"), (1.15, 1.30, "blocked, mid"),
                        (1.30, 99, "blocked, rich")]:
        line(lab, b[(b["ratio"] >= lo) & (b["ratio"] < hi)])

    print("\n=== 4. the exact shipping config ===")
    ship = m[m["allowed"] & (m["ratio"] >= 1.30) & (m["rv_true"] >= 0.15)]
    base = m[m["allowed"]]
    line("regime map only (baseline)", base)
    line("regime map + RV>=0.15", m[m["allowed"] & (m["rv_true"] >= 0.15)])
    line("regime map + RV>=0.15 + IV/HV>=1.30", ship)
    line("  ... of which Bull Put / downtrend", ship[ship["direction"] == "Bull Put"])
    line("  ... of which Bear Call / uptrend", ship[ship["direction"] == "Bear Call"])

    print("\n=== 5. monthly stability of the shipping config ===")
    g = ship.groupby(ship["asof"].str[:7])["pnl_1x"]
    for k, v in g:
        print(f"  {k}  n {len(v):>3}  win {100*(v>0).mean():>5.1f}%  "
              f"avg {v.mean():>+7.1f}  PF {pf(v):>5.2f}  total {v.sum():>+7.0f}")
    pos = sum(1 for _, v in g if v.sum() > 0)
    print(f"  months with positive total: {pos}/{g.ngroups}")

    print("\n=== 6. control: is the richness lift inside the map real? ===")
    print("  shuffling the IV/HV values within the map-allowed pool and re-selecting")
    print("  the same number of trades, 2000 times:")
    pool = m[m["allowed"] & (m["rv_true"] >= 0.15)].reset_index(drop=True)
    k = len(ship)
    obs = ship["pnl_1x"].mean()
    draws = np.array([pool["pnl_1x"].iloc[RNG.choice(len(pool), k, replace=False)].mean()
                      for _ in range(2000)])
    print(f"    observed mean P&L of rich subset : {obs:+.2f}")
    print(f"    random same-size subsets         : {draws.mean():+.2f} "
          f"(5th {np.percentile(draws,5):+.2f}, 95th {np.percentile(draws,95):+.2f})")
    print(f"    empirical p(random >= observed)  : {(draws >= obs).mean():.3f}")

    print("\n=== 7. what a range day should produce ===")
    rng_ = m[m["regime"] == "range"]
    line("every credit trade on a range day", rng_)
    line("  range + rich >= 1.30", rng_[rng_["ratio"] >= 1.30])
    print("  -> a range session producing zero trades is the correct outcome,")
    print("     not a pipeline failure.")


if __name__ == "__main__":
    main()
