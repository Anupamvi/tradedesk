"""Walk-forward validation of the credit/width hypothesis.

The threshold is refitted on past trades only at every fold, so a cut that only
works with hindsight cannot pass here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from claude_pipeline import selection, stats

OUT = Path("/Users/anuppamvi/tradedesk/out/claude_pipeline")
GRID = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]


def _card(label: str, frame: pd.DataFrame) -> dict:
    card = stats.scorecard(frame, label)
    if not card.get("n"):
        return {"population": label, "n": 0}
    return {
        "population": label,
        "n": card["n"],
        "win%": round(100 * card["win_rate"], 1),
        "avg$": round(card["avg_pnl"], 2),
        "total$": round(card["total_pnl"], 0),
        "PF": round(card["profit_factor"], 3),
        "maxDD$": round(card["max_drawdown"], 0),
        "mo+": f"{card['months_profitable']}/{card['months_total']}",
        "boot_p05": round(card["boot_p05"], 2),
        "p(loss)": round(card["boot_p_loss"], 3),
        "passes": card["passes"],
    }


def main(name: str = "backtest_full") -> None:
    enriched = pd.read_csv(OUT / f"{name}_enriched.csv.gz", low_memory=False)
    credit = enriched[enriched["family"].isin(["bull_put_credit", "bear_call_credit"])].copy()
    credit = credit[credit["credit_pct_width"].notna()]

    print(f"=== credit population: {len(credit):,} resolved trades ===")
    print("\nfixed thresholds, FULL SAMPLE (in-sample, for shape only):")
    rows = [_card("all credit", credit)]
    for cut in GRID:
        rows.append(_card(f"credit/width >= {cut:.2f}", credit[credit["credit_pct_width"] >= cut]))
    rows.append(_card("credit/width >= 0.50 (near-arbitrage, suspect)",
                      credit[credit["credit_pct_width"] >= 0.50]))
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n=== WALK-FORWARD: threshold refitted on past folds only ===")
    fitted = stats.walk_forward(credit, selection.threshold_selector("credit_pct_width", GRID), folds=5)
    if fitted.empty:
        print("no folds produced trades")
        return
    print(pd.DataFrame([_card("walk-forward (fitted cut)", fitted)]).to_string(index=False))
    print("\nper fold:")
    per_fold = fitted.groupby("fold").agg(
        n=("pnl", "size"), cut=("threshold", "first"),
        win=("pnl", lambda s: round((s > 0).mean(), 3)),
        avg=("pnl", "mean"), total=("pnl", "sum"),
    )
    per_fold["PF"] = fitted.groupby("fold")["pnl"].apply(
        lambda s: s[s > 0].sum() / max(-s[s < 0].sum(), 1e-9)
    )
    print(per_fold.round(2).to_string())

    print("\n=== FIXED 0.30 cut, no fitting at all (the honest control) ===")
    fixed = stats.walk_forward(
        credit, selection.fixed_rule(lambda df: df["credit_pct_width"] >= 0.30), folds=5
    )
    if not fixed.empty:
        print(pd.DataFrame([_card("walk-forward (fixed 0.30)", fixed)]).to_string(index=False))
        monthly = stats.monthly(fixed)
        print("\nmonthly P&L, out-of-sample folds only:")
        print(monthly.round(0).to_string())
        print(f"\nmonths profitable: {(monthly > 0).sum()}/{len(monthly)}")


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else "backtest_full")
