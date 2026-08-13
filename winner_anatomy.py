"""What did the winners actually look like?

The averages say the stack loses. That is the wrong question for a long-premium
book: buying a call is a positive-skew bet, so what matters is whether the large
payoffs are IDENTIFIABLE in advance, not whether the mean across every candidate
is positive.

This is exploratory and in-sample by construction. Anything it turns up is a
hypothesis to be validated out of sample, not a result.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/anuppamvi/uw_root/tradedesk")
TRADES = ROOT / "out/conviction_stack_backtest.csv"
CATALYSTS = ROOT / "out/catalyst_panel.csv"


def main() -> None:
    trades = pd.read_csv(TRADES, low_memory=False)
    trades = trades[trades.tradeable].copy()
    trades["cost_per_contract"] = trades.entry_price * 100.0
    trades["rt_cost_pct"] = 200.0 * (1.0 - trades.entry_price.rsub(0).abs() * 0)  # placeholder replaced below
    catalysts = pd.read_csv(CATALYSTS, usecols=["date", "ticker", "days_to_earnings"], low_memory=False)
    trades = trades.merge(
        catalysts.rename(columns={"date": "signal_date"}), on=["signal_date", "ticker"], how="left"
    )

    r = trades.return_on_premium
    print("=== RETURN DISTRIBUTION, 2538 trades ===")
    print(r.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).round(3).to_string())
    print()
    for threshold in (0.5, 1.0, 2.0, 3.0):
        hits = r >= threshold
        print(f"  returned >= {threshold*100:>4.0f}% : {hits.sum():>4} trades ({hits.mean()*100:.1f}%)")
    print()
    print(f"total P&L 1 lot each: ${trades.pnl_per_contract.sum():>+12,.0f}")
    top = trades.nlargest(50, "return_on_premium")
    print(f"P&L from the best 50 trades:  ${top.pnl_per_contract.sum():>+12,.0f}")
    print(f"P&L from everything else:     ${trades.pnl_per_contract.sum() - top.pnl_per_contract.sum():>+12,.0f}")

    trades["big_winner"] = r >= 1.0
    print("\n=== WHAT SEPARATES THE >=100% WINNERS? ===")
    features = [
        "days_to_expiry", "cost_per_contract", "premium", "ask_share", "oi_change",
        "tape_ask_share", "tape_block_premium", "tape_largest_ask_print",
        "tape_delta", "iv_rank", "dp_bias", "conviction", "days_to_earnings",
    ]
    rows = []
    for feature in features:
        if feature not in trades.columns:
            continue
        winners = trades.loc[trades.big_winner, feature].dropna()
        rest = trades.loc[~trades.big_winner, feature].dropna()
        if len(winners) < 10 or len(rest) < 10:
            continue
        rows.append(
            {
                "feature": feature,
                "winners_median": winners.median(),
                "others_median": rest.median(),
                "ratio": winners.median() / rest.median() if rest.median() else np.nan,
            }
        )
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    print("\n=== EXPECTANCY BY SLICE (mean return on premium) ===")
    slices = {
        "dte<=21": trades[trades.days_to_expiry <= 21],
        "dte 22-45": trades[trades.days_to_expiry.between(22, 45)],
        "dte>45": trades[trades.days_to_expiry > 45],
        "cheap<$500": trades[trades.cost_per_contract < 500],
        "$500-1500": trades[trades.cost_per_contract.between(500, 1500)],
        "$1500-4000": trades[trades.cost_per_contract.between(1500, 4000)],
        ">$4000": trades[trades.cost_per_contract > 4000],
        "|delta|<0.25": trades[trades.tape_delta.abs() < 0.25],
        "|delta| 0.25-0.45": trades[trades.tape_delta.abs().between(0.25, 0.45)],
        "|delta|>0.45": trades[trades.tape_delta.abs() > 0.45],
        "earnings 0-10d": trades[trades.days_to_earnings.between(0, 10)],
        "earnings 11-30d": trades[trades.days_to_earnings.between(11, 30)],
        "iv_rank<40": trades[trades.iv_rank < 40],
        "iv_rank>70": trades[trades.iv_rank > 70],
    }
    rows = []
    for label, frame in slices.items():
        if len(frame) < 30:
            continue
        values = frame.return_on_premium
        gains = frame.pnl_per_contract[frame.pnl_per_contract > 0].sum()
        losses = -frame.pnl_per_contract[frame.pnl_per_contract < 0].sum()
        rows.append(
            {
                "slice": label,
                "trades": len(frame),
                "mean_return": values.mean(),
                "win_rate": values.gt(0).mean(),
                "pct_over_100": values.ge(1.0).mean(),
                "profit_factor": gains / losses if losses else np.nan,
                "pnl_1_lot": frame.pnl_per_contract.sum(),
            }
        )
    result = pd.DataFrame(rows).sort_values("mean_return", ascending=False)
    print(result.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
