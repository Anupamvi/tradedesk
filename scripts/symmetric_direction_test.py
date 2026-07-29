"""Symmetric test: both directions, every sector, same managed exit.

Three flaws in the earlier work are corrected here.

1. The +50% take-profit was only ever applied to calls. Puts were judged on the
   fixed-40-session hold, which is the method the profit target already proved
   wrong for calls.
2. The sector list was fitted to calls -- Tech and Financials were chosen because
   calls worked there, then puts were tested in those same rising sectors.
3. Nobody asked which sectors were actually falling, which is where a put should
   work at all.

So: calls on the top momentum names and puts on the bottom momentum names, in
every sector, under identical exit rules. A direction that only works because the
tape went one way is not a pattern, and this is what shows that.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import managed_exit_backtest as base  # noqa: E402

OUT = base.ROOT / "out/symmetric_direction_test.csv"
PROFIT_TARGET = 0.5
DECILE = 0.80


def simulate(panel, days, quote_for, direction: str, rng, randomize: bool = False) -> pd.DataFrame:
    option_type = "call" if direction == "long_call" else "put"
    moneyness = 1.05 if direction == "long_call" else 0.95

    open_positions: list[dict] = []
    closed: list[dict] = []
    held: set[str] = set()

    for index, session in enumerate(days):
        quotes = quote_for(session)
        if quotes.empty:
            continue
        bid = quotes.set_index("option_symbol").last_bid

        still_open = []
        for position in open_positions:
            current = bid.get(position["symbol"])
            age = index - position["entry_index"]
            if current is None or not np.isfinite(current):
                if age < base.MAX_HOLD:
                    still_open.append(position)
                    continue
                current = position["last_mark"]
            else:
                position["last_mark"] = current
            gain = current / position["cost"] - 1.0
            if gain >= PROFIT_TARGET:
                reason = "profit_target"
            elif age >= base.MAX_HOLD:
                reason = "time_stop"
            else:
                still_open.append(position)
                continue
            pnl = (current - position["cost"]) * 100.0 - base.CONTRACT_FEE
            closed.append(
                {
                    "signal_date": position["signal_date"],
                    "ticker": position["ticker"],
                    "sector": position["sector"],
                    "direction": direction,
                    "mode": "random" if randomize else "signal",
                    "exit_reason": reason,
                    "held": age,
                    "cost": position["cost"] * 100.0,
                    "pnl": pnl,
                    "return_on_cost": pnl / (position["cost"] * 100.0),
                }
            )
            held.discard(position["ticker"])
        open_positions = still_open

        if index + 1 >= len(days):
            continue
        day = panel[panel.date == session].dropna(subset=["pos_52w"])
        if day.empty:
            continue
        entry_quotes = quote_for(days[index + 1])
        if entry_quotes.empty:
            continue

        for sector, block in day.groupby("sector"):
            if len(block) < base.MIN_PER_SECTOR:
                continue
            momentum = block.pos_52w.rank(pct=True)
            chosen = set(
                block[momentum >= DECILE].ticker
                if direction == "long_call"
                else block[momentum <= 1.0 - DECILE].ticker
            )
            if randomize:
                pool = block.ticker.to_numpy()
                chosen = set(rng.choice(pool, size=min(len(chosen), len(pool)), replace=False))
            chosen -= held
            if not chosen:
                continue

            legs = entry_quotes[
                entry_quotes.ticker.isin(chosen)
                & entry_quotes.option_type.eq(option_type)
                & entry_quotes.dte.between(*base.DTE_BAND)
                & (entry_quotes.curr_oi >= 50)
                & (entry_quotes.spread_pct <= base.MAX_SPREAD_PCT)
            ].copy()
            if legs.empty:
                continue
            legs["strike_gap"] = (legs.strike - legs.stock_price * moneyness).abs()
            legs["dte_gap"] = (legs.dte - base.TARGET_DTE).abs()
            legs = legs.sort_values(["dte_gap", "strike_gap"]).groupby("ticker", as_index=False).first()
            for row in legs.itertuples():
                open_positions.append(
                    {
                        "signal_date": session,
                        "ticker": row.ticker,
                        "sector": sector,
                        "entry_index": index + 1,
                        "cost": row.last_ask,
                        "symbol": row.option_symbol,
                        "last_mark": row.last_bid,
                    }
                )
                held.add(row.ticker)
    return pd.DataFrame(closed)


def profit_factor(values: pd.Series) -> float:
    gains = values[values > 0].sum()
    losses = -values[values < 0].sum()
    return gains / losses if losses > 0 else np.nan


def main() -> None:
    columns = ["date", "ticker", "sector", "issue_type", "marketcap", "close", "pos_52w"]
    panel = pd.read_csv(base.PANEL, usecols=columns, low_memory=False)
    panel["date"] = pd.to_datetime(panel["date"]).dt.strftime("%Y-%m-%d")
    panel = panel[
        (panel.issue_type == "Common Stock")
        & (panel.marketcap.fillna(0) >= 2e9)
        & panel.sector.notna()
    ].sort_values(["ticker", "date"])

    days = sorted(p.name for p in base.ROOT.iterdir() if p.is_dir() and re.fullmatch(r"2026-\d{2}-\d{2}", p.name))
    position = {d: i for i, d in enumerate(days)}
    cache: dict[str, pd.DataFrame] = {}

    def quote_for(session: str) -> pd.DataFrame:
        if session not in cache:
            slot = position[session]
            cache[session] = (
                base.chain_quotes(session, days[slot + 1]) if slot + 1 < len(days) else pd.DataFrame()
            )
        return cache[session]

    print("[symmetric] warming quote cache", flush=True)
    for session in days:
        quote_for(session)

    rng = np.random.default_rng(20260728)
    frames = []
    for direction in ("long_call", "long_put"):
        for randomize in (False, True):
            trades = simulate(panel, days, quote_for, direction, rng, randomize)
            if not trades.empty:
                frames.append(trades)
            label = "random" if randomize else "signal"
            print(f"[symmetric] {direction} {label}: {len(trades)} closed", flush=True)

    trades = pd.concat(frames, ignore_index=True)
    trades["sample"] = np.where(trades.signal_date >= base.SPLIT, "TEST", "TRAIN")
    trades.to_csv(OUT, index=False)

    print("\n=== POOLED, signal vs its own random control ===")
    pooled = trades.groupby(["direction", "mode", "sample"]).apply(
        lambda f: pd.Series(
            {
                "n": len(f),
                "mean": f.return_on_cost.mean(),
                "median": f.return_on_cost.median(),
                "win": f.pnl.gt(0).mean(),
                "PF": profit_factor(f.pnl),
                "pnl": f.pnl.sum(),
            }
        )
    ).round(3)
    print(pooled.to_string())

    print("\n=== BY SECTOR, SIGNAL ONLY, mean return on cost ===")
    signal = trades[trades["mode"] == "signal"]
    for sample in ("TRAIN", "TEST"):
        frame = signal[signal["sample"] == sample]
        table = frame.pivot_table(index="sector", columns="direction", values="return_on_cost", aggfunc="mean")
        counts = frame.pivot_table(index="sector", columns="direction", values="pnl", aggfunc="size")
        table = table.join(counts, rsuffix="_n")
        print(f"\n{sample}:")
        print(table.round(3).to_string())

    print("\n=== PUT LANE: sectors positive in BOTH halves ===")
    puts = signal[signal.direction == "long_put"]
    pivot = puts.pivot_table(index="sector", columns="sample", values="return_on_cost", aggfunc="mean")
    counts = puts.pivot_table(index="sector", columns="sample", values="pnl", aggfunc="size")
    combined = pivot.join(counts, rsuffix="_n")
    if {"TRAIN", "TEST"}.issubset(pivot.columns):
        combined["both_positive"] = (pivot.TRAIN > 0) & (pivot.TEST > 0)
    print(combined.round(3).to_string())
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
