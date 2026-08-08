"""Permutation test: is the managed-exit call result better than luck?

More history is not available, so significance has to come from the data we have.
The selection rule is replaced with a random draw of the same size, from the same
sector, on the same dates, running the identical entry/exit machinery. Repeating
that many times builds the null distribution the real result must beat.

This is a stronger control than the single random draw used earlier, which is one
sample from this distribution and says nothing about its spread.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import managed_exit_backtest as base  # noqa: E402

PERMUTATIONS = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 200
PROFIT_TARGET = 0.5
OUT = base.ROOT / "out/permutation_test.csv"


def run_selection(panel, days, quote_for, rng, randomize: bool) -> pd.DataFrame:
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
            current = bid.get(position["exit_symbol"])
            age = index - position["entry_index"]
            if current is None or not np.isfinite(current):
                if age < base.MAX_HOLD:
                    still_open.append(position)
                    continue
                current = position["last_mark"]
            else:
                position["last_mark"] = current
            gain = current / position["entry_cost"] - 1.0
            if gain >= PROFIT_TARGET:
                reason = "profit_target"
            elif age >= base.MAX_HOLD:
                reason = "time_stop"
            else:
                still_open.append(position)
                continue
            pnl = (current - position["entry_cost"]) * 100.0 - base.CONTRACT_FEE
            closed.append(
                {
                    "signal_date": position["signal_date"],
                    "ticker": position["ticker"],
                    "exit_reason": reason,
                    "pnl": pnl,
                    "return_on_cost": pnl / (position["entry_cost"] * 100.0),
                }
            )
            held.discard(position["ticker"])
        open_positions = still_open

        if index + 1 >= len(days):
            continue
        day = panel[panel.date == session].dropna(subset=["pos_52w", "flow_escalation"])
        if day.empty:
            continue
        entry_quotes = quote_for(days[index + 1])
        if entry_quotes.empty:
            continue

        for sector, block in day.groupby("sector"):
            if sector not in base.SECTORS or len(block) < base.MIN_PER_SECTOR:
                continue
            momentum = block.pos_52w.rank(pct=True)
            chosen = set(block[momentum >= base.DECILE].ticker)
            if randomize:
                pool = block.ticker.to_numpy()
                chosen = set(rng.choice(pool, size=min(len(chosen), len(pool)), replace=False))
            chosen -= held
            if not chosen:
                continue
            legs = base.nearest(entry_quotes, chosen, "call", 1.05)
            if legs.empty:
                continue
            for row in legs.itertuples():
                open_positions.append(
                    {
                        "signal_date": session,
                        "ticker": row.ticker,
                        "entry_index": index + 1,
                        "entry_cost": row.last_ask,
                        "exit_symbol": row.option_symbol,
                        "last_mark": row.last_bid,
                    }
                )
                held.add(row.ticker)
    return pd.DataFrame(closed)


def stats(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"n": 0, "mean": np.nan, "win": np.nan, "pf": np.nan, "pnl": np.nan}
    gains = trades.pnl[trades.pnl > 0].sum()
    losses = -trades.pnl[trades.pnl < 0].sum()
    return {
        "n": len(trades),
        "mean": trades.return_on_cost.mean(),
        "win": trades.pnl.gt(0).mean(),
        "pf": gains / losses if losses else np.nan,
        "pnl": trades.pnl.sum(),
    }


def main() -> None:
    columns = ["date", "ticker", "sector", "issue_type", "marketcap", "close", "pos_52w", "hc_premium"]
    panel = pd.read_csv(base.PANEL, usecols=columns, low_memory=False)
    panel["date"] = pd.to_datetime(panel["date"]).dt.strftime("%Y-%m-%d")
    panel = panel[
        (panel.issue_type == "Common Stock")
        & (panel.marketcap.fillna(0) >= 2e9)
        & panel.sector.isin(base.SECTORS)
    ].sort_values(["ticker", "date"])
    grouped = panel.groupby("ticker")
    panel["flow_avg_20"] = grouped.hc_premium.transform(lambda s: s.rolling(20, min_periods=10).mean())
    panel["flow_escalation"] = panel.hc_premium / panel.flow_avg_20.replace(0, np.nan)

    import re

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

    print("[perm] warming quote cache", flush=True)
    for session in days:
        quote_for(session)

    rng = np.random.default_rng(20260728)
    actual = run_selection(panel, days, quote_for, rng, randomize=False)
    actual_test = actual[actual.signal_date >= base.SPLIT]
    actual_stats = stats(actual_test)
    print(f"[perm] actual TEST: {actual_stats}", flush=True)

    rows = []
    for trial in range(PERMUTATIONS):
        trades = run_selection(panel, days, quote_for, rng, randomize=True)
        test = trades[trades.signal_date >= base.SPLIT]
        record = stats(test)
        record["trial"] = trial
        rows.append(record)
        if (trial + 1) % 25 == 0:
            print(f"[perm] {trial + 1}/{PERMUTATIONS}", flush=True)

    null = pd.DataFrame(rows)
    null.to_csv(OUT, index=False)

    print(f"\n=== PERMUTATION TEST, {PERMUTATIONS} random selections, TEST half ===")
    for metric in ("mean", "win", "pf", "pnl"):
        observed = actual_stats[metric]
        distribution = null[metric].dropna()
        if distribution.empty or not np.isfinite(observed):
            continue
        p_value = (distribution >= observed).mean()
        print(
            "{:<5} actual={:>10.3f}   null mean={:>8.3f}  p05={:>8.3f}  p95={:>8.3f}   "
            "p-value={:.4f}".format(
                metric, observed, distribution.mean(), distribution.quantile(0.05),
                distribution.quantile(0.95), p_value,
            )
        )
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
