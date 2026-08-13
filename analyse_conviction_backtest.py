"""Analyse the conviction-stack backtest.

Three questions, deliberately separated:

1. Does following the block into the SAME contract make money after real fills?
2. Does the signal predict the UNDERLYING? If the stock moves the right way but
   the option loses, the information is real and the expression is wrong -- the
   block just paid the offer and lifted implied vol (Bollen & Whaley 2004).
3. Do earnings proximity, affordability and conviction rank change the answer?
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/anuppamvi/uw_root/tradedesk")
TRADES = ROOT / "out/conviction_stack_backtest.csv"
PANEL = ROOT / "out/uw_all_feeds.csv"
CATALYSTS = ROOT / "out/catalyst_panel.csv"


def profit_factor(values: pd.Series) -> float:
    gains = values[values > 0].sum()
    losses = -values[values < 0].sum()
    return gains / losses if losses > 0 else np.nan


def describe(label: str, frame: pd.DataFrame) -> dict:
    return {
        "cohort": label,
        "trades": len(frame),
        "win_rate": frame.pnl_per_contract.gt(0).mean(),
        "avg_return_on_premium": frame.return_on_premium.mean(),
        "median_return": frame.return_on_premium.median(),
        "profit_factor": profit_factor(frame.pnl_per_contract),
        "total_pnl_1_lot": frame.pnl_per_contract.sum(),
    }


def main() -> None:
    trades = pd.read_csv(TRADES, low_memory=False)
    trades = trades[trades.tradeable].copy()
    trades["cost_per_contract"] = trades.entry_price * 100.0
    trades["month"] = trades.signal_date.str[:7]

    # Underlying move over the same window, in the direction the block implied.
    panel = pd.read_csv(PANEL, usecols=["date", "ticker", "close"], low_memory=False)
    panel["date"] = pd.to_datetime(panel["date"]).dt.strftime("%Y-%m-%d")
    closes = panel.drop_duplicates(["date", "ticker"]).set_index(["ticker", "date"])["close"]
    entry_close = closes.reindex(pd.MultiIndex.from_arrays([trades.ticker, trades.entry_date])).to_numpy()
    exit_close = closes.reindex(pd.MultiIndex.from_arrays([trades.ticker, trades.exit_date])).to_numpy()
    trades["stock_return"] = exit_close / entry_close - 1.0
    trades["directional_sign"] = np.where(trades.option_type == "call", 1.0, -1.0)
    trades["stock_move_with_block"] = trades.stock_return * trades.directional_sign

    catalysts = pd.read_csv(CATALYSTS, usecols=["date", "ticker", "days_to_earnings"], low_memory=False)
    catalysts = catalysts.rename(columns={"date": "signal_date"})
    trades = trades.merge(catalysts, on=["signal_date", "ticker"], how="left")

    rows = [describe("ALL", trades)]
    for option_type in ("call", "put"):
        rows.append(describe(f"type={option_type}", trades[trades.option_type == option_type]))
    rows.append(describe("affordable<=$1500", trades[trades.cost_per_contract <= 1500]))
    rows.append(describe("affordable<=$3000", trades[trades.cost_per_contract <= 3000]))
    rows.append(describe("block>=$1M", trades[trades.tape_largest_ask_print >= 1e6]))
    rows.append(describe("block>=$5M", trades[trades.tape_largest_ask_print >= 5e6]))
    rows.append(describe("dark pool agrees", trades[trades.dp_agrees == True]))  # noqa: E712
    rows.append(describe("earnings in 0-7d", trades[trades.days_to_earnings.between(0, 7)]))
    rows.append(describe("earnings in 8-30d", trades[trades.days_to_earnings.between(8, 30)]))
    rows.append(describe("no earnings <30d", trades[trades.days_to_earnings > 30]))
    top = trades.sort_values("conviction", ascending=False).groupby("signal_date").head(2)
    rows.append(describe("top 2 conviction per day", top))

    print("=== OPTION P&L, buy at ask / sell at bid, 5 sessions ===")
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    print("\n=== DID THE UNDERLYING MOVE THE BLOCK'S WAY? ===")
    stock = trades.dropna(subset=["stock_move_with_block"])
    print(f"trades with stock data: {len(stock)}")
    print(f"mean move in block direction: {stock.stock_move_with_block.mean():+.4f}")
    print(f"median: {stock.stock_move_with_block.median():+.4f}")
    print(f"right direction rate: {stock.stock_move_with_block.gt(0).mean():.3f}")
    for label, subset in (
        ("block>=$1M", stock[stock.tape_largest_ask_print >= 1e6]),
        ("block>=$5M", stock[stock.tape_largest_ask_print >= 5e6]),
        ("calls", stock[stock.option_type == "call"]),
        ("puts", stock[stock.option_type == "put"]),
    ):
        if len(subset) >= 20:
            print(
                f"  {label:<12} n={len(subset):<5} mean={subset.stock_move_with_block.mean():+.4f} "
                f"right={subset.stock_move_with_block.gt(0).mean():.3f}"
            )

    print("\n=== BY MONTH (all trades) ===")
    print(
        trades.groupby("month")
        .agg(
            trades=("pnl_per_contract", "size"),
            win_rate=("pnl_per_contract", lambda x: (x > 0).mean()),
            avg_return=("return_on_premium", "mean"),
            pnl_1_lot=("pnl_per_contract", "sum"),
        )
        .round(4)
        .to_string()
    )


if __name__ == "__main__":
    main()
