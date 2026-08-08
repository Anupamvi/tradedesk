"""Which parameters were tested, and which did I just pick?

The HIGH/MEDIUM tier turned out to be noise dressed as information. That is a
reason to distrust every other number in the lane until each one is checked the
same way, so this audits all of them:

  - is the split statistically distinguishable, not merely different?
  - does anything actually RANK the qualifying tickets against each other?

A parameter that only ever had a point estimate behind it gets reported as
ASSUMED, not validated.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path("/Users/anuppamvi/uw_root/tradedesk")
TRADES = ROOT / "out/symmetric_direction_test.csv"
SPLIT = "2026-04-14"


def profit_factor(values: pd.Series) -> float:
    gains = values[values > 0].sum()
    losses = -values[values < 0].sum()
    return gains / losses if losses > 0 else np.nan


def compare(name: str, low: pd.DataFrame, high: pd.DataFrame, low_label: str, high_label: str) -> dict:
    """Is the split real, or just two point estimates that differ?"""
    if len(low) < 10 or len(high) < 10:
        return {"split": name, "verdict": "TOO FEW TRADES TO TEST"}
    wins = [[low.pnl.gt(0).sum(), len(low) - low.pnl.gt(0).sum()],
            [high.pnl.gt(0).sum(), len(high) - high.pnl.gt(0).sum()]]
    fisher_p = stats.fisher_exact(wins)[1]
    mw_p = stats.mannwhitneyu(low.return_on_cost, high.return_on_cost).pvalue
    distinguishable = fisher_p <= 0.05 or mw_p <= 0.05
    return {
        "split": name,
        low_label: f"n={len(low)} win={low.pnl.gt(0).mean():.3f} PF={profit_factor(low.pnl):.2f}",
        high_label: f"n={len(high)} win={high.pnl.gt(0).mean():.3f} PF={profit_factor(high.pnl):.2f}",
        "fisher_p": round(fisher_p, 3),
        "mannwhitney_p": round(mw_p, 3),
        "verdict": "REAL" if distinguishable else "NOT DISTINGUISHABLE",
    }


def main() -> None:
    trades = pd.read_csv(TRADES, low_memory=False)
    lane = trades[
        (trades.sector == "Technology")
        & (trades.direction == "long_put")
        & (trades["mode"] == "signal")
    ].copy()
    lane["sample"] = np.where(lane.signal_date >= SPLIT, "TEST", "TRAIN")

    print("=== IS THE $700 COST FLOOR REAL, OR THE SAME KIND OF NOISE? ===")
    rows = [
        compare(
            "cost < $700 vs >= $700 (all)",
            lane[lane.cost < 700], lane[lane.cost >= 700], "cheap", "rich",
        )
    ]
    for sample in ("TRAIN", "TEST"):
        block = lane[lane["sample"] == sample]
        rows.append(
            compare(
                f"cost floor within {sample}",
                block[block.cost < 700], block[block.cost >= 700], "cheap", "rich",
            )
        )
    print(pd.DataFrame(rows).to_string(index=False))

    qualifying = lane[lane.cost >= 700].copy()
    print(f"\n=== CAN ANYTHING RANK THE {len(qualifying)} QUALIFYING TICKETS? ===")
    print("If a feature ranks them, its correlation with realised return is non-zero.")
    print("Spearman across all qualifying trades, plus the TEST half alone:\n")

    features = {
        "momentum rank (pos_52w)": "pos_52w" if "pos_52w" in qualifying else None,
        "contract cost": "cost",
        "held sessions": "held" if "held" in qualifying else None,
    }
    results = []
    for label, column in features.items():
        if column is None or column not in qualifying:
            continue
        subset = qualifying[[column, "return_on_cost"]].dropna()
        if len(subset) < 20:
            continue
        rho_all, p_all = stats.spearmanr(subset[column], subset.return_on_cost)
        test = qualifying[qualifying["sample"] == "TEST"][[column, "return_on_cost"]].dropna()
        if len(test) >= 15:
            rho_test, p_test = stats.spearmanr(test[column], test.return_on_cost)
        else:
            rho_test, p_test = np.nan, np.nan
        results.append(
            {
                "feature": label,
                "n": len(subset),
                "spearman_all": round(rho_all, 3),
                "p_all": round(p_all, 3),
                "spearman_test": round(rho_test, 3) if np.isfinite(rho_test) else None,
                "p_test": round(p_test, 3) if np.isfinite(p_test) else None,
                "ranks?": "YES" if p_all <= 0.05 else "no",
            }
        )
    print(pd.DataFrame(results).to_string(index=False))

    print("\n=== PARAMETER PROVENANCE: TESTED vs ASSUMED ===")
    provenance = [
        ("Technology sector only", "TESTED", "only sector significant in both halves vs permutation null"),
        ("puts on weak momentum", "TESTED", "symmetric test, both directions, all sectors"),
        ("+50% profit target", "TESTED", "compared against none / +100% / +200%"),
        ("no stop loss", "TESTED", "a 50% stop cut TEST PF 7.63 -> 1.39"),
        ("$700 contract floor", "see above", "point estimate was clear; significance tested here"),
        ("bottom 20% momentum", "ASSUMED", "0.90 then 0.80 tried; never swept systematically"),
        ("0.95x strike", "ASSUMED", "never compared against 0.90 / 1.00 with managed exits"),
        ("80 DTE target", "ASSUMED", "chosen to match the 40-session hold, not optimised"),
        ("40-session time stop", "ASSUMED", "inherited from the momentum horizon, never varied"),
        ("spread <= 12%", "ASSUMED", "liquidity hygiene, never tested as a filter"),
        ("open interest >= 50", "ASSUMED", "liquidity hygiene, never tested as a filter"),
    ]
    print(pd.DataFrame(provenance, columns=["parameter", "status", "basis"]).to_string(index=False))


if __name__ == "__main__":
    main()
