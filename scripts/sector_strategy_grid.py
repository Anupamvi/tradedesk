"""Full grid: every sector x every strategy x controls, in one pass.

Testing one lane at a time was the wrong approach. This ranks names WITHIN each
sector, so a sector's own leaders and laggards are compared against each other
rather than against a tech-dominated cross-section, then evaluates every
structure on the same names and the same dates.

Structures
  long_call     top momentum decile, 1.05x strike
  long_put      bottom momentum decile, 0.95x strike
  straddle      highest flow escalation, ATM, direction-free
  call_spread   top decile, long 1.02x / short 1.12x, defined risk

Controls
  signal        the ranked selection
  random        same count, same dates, same sector, drawn at random

All quotes come from chain-oi-changes so any contract with open interest can be
re-quoted. Entry pays the ask, exit sells the bid.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conviction_stack import ROOT, find, open_zip, parse_occ  # noqa: E402

PANEL = ROOT / "out/uw_all_feeds.csv"
OUT = ROOT / "out/sector_strategy_grid.csv"
SPLIT = "2026-04-14"
HOLD = 40
DTE_BAND = (60, 110)
TARGET_DTE = 80
MAX_SPREAD_PCT = 0.12
CONTRACT_FEE = 1.30
MIN_PER_SECTOR = 12
DECILE = 0.80


def chain_quotes(session: str, following: str) -> pd.DataFrame:
    path = find(ROOT / following, "chain-oi-changes")
    if path is None:
        return pd.DataFrame()
    frame = open_zip(
        path,
        ["option_symbol", "last_bid", "last_ask", "last_date", "curr_oi", "stock_price", "dte", "strike"],
    )
    if "last_date" in frame.columns:
        frame = frame[frame.last_date.astype(str).str.startswith(session)]
    for column in ["last_bid", "last_ask", "curr_oi", "stock_price", "dte"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame[(frame.last_ask > 0) & (frame.last_bid >= 0) & (frame.stock_price > 0)]
    frame = frame.drop(columns=["strike"], errors="ignore")
    frame = frame.join(parse_occ(frame.option_symbol.astype(str)))
    frame = frame[frame.ticker.notna()]
    frame["spread_pct"] = (frame.last_ask - frame.last_bid) / frame.last_ask
    return frame.drop_duplicates("option_symbol")


def nearest(quotes: pd.DataFrame, tickers: set[str], option_type: str, moneyness: float) -> pd.DataFrame:
    legs = quotes[
        quotes.ticker.isin(tickers)
        & quotes.option_type.eq(option_type)
        & quotes.dte.between(*DTE_BAND)
        & (quotes.curr_oi >= 50)
        & (quotes.spread_pct <= MAX_SPREAD_PCT)
    ].copy()
    if legs.empty:
        return legs
    legs["strike_gap"] = (legs.strike - legs.stock_price * moneyness).abs()
    legs["dte_gap"] = (legs.dte - TARGET_DTE).abs()
    return legs.sort_values(["dte_gap", "strike_gap"]).groupby("ticker", as_index=False).first()


def main() -> None:
    columns = ["date", "ticker", "sector", "issue_type", "marketcap", "close", "pos_52w", "hc_premium"]
    panel = pd.read_csv(PANEL, usecols=columns, low_memory=False)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel[
        (panel.issue_type == "Common Stock")
        & (panel.marketcap.fillna(0) >= 2e9)
        & panel.sector.notna()
    ].sort_values(["ticker", "date"])
    grouped = panel.groupby("ticker")
    panel["flow_avg_20"] = grouped.hc_premium.transform(lambda s: s.rolling(20, min_periods=10).mean())
    panel["flow_escalation"] = panel.hc_premium / panel.flow_avg_20.replace(0, np.nan)

    days = sorted(p.name for p in ROOT.iterdir() if p.is_dir() and re.fullmatch(r"2026-\d{2}-\d{2}", p.name))
    position = {d: i for i, d in enumerate(days)}
    cache: dict[str, pd.DataFrame] = {}
    rng = np.random.default_rng(20260728)

    def quotes(session: str) -> pd.DataFrame:
        if session not in cache:
            slot = position[session]
            cache[session] = chain_quotes(session, days[slot + 1]) if slot + 1 < len(days) else pd.DataFrame()
            # keep memory bounded across a 139-day sweep
            if len(cache) > 6:
                for old in list(cache)[:-6]:
                    cache.pop(old, None)
        return cache[session]

    records = []
    for signal_date in days:
        slot = position[signal_date]
        if slot + 1 + HOLD >= len(days):
            continue
        day = panel[panel.date == signal_date].dropna(subset=["pos_52w", "flow_escalation"])
        if day.empty:
            continue
        entry_date, exit_date = days[slot + 1], days[slot + 1 + HOLD]
        entry_quotes, exit_quotes = quotes(entry_date), quotes(exit_date)
        if entry_quotes.empty or exit_quotes.empty:
            continue
        exit_bid = exit_quotes.set_index("option_symbol").last_bid
        print(f"[grid] {signal_date}", flush=True)

        for sector, block in day.groupby("sector"):
            if len(block) < MIN_PER_SECTOR:
                continue
            momentum = block.pos_52w.rank(pct=True)
            escalation = block.flow_escalation.rank(pct=True)
            selections = {
                "long_call": set(block[momentum >= DECILE].ticker),
                "long_put": set(block[momentum <= 1.0 - DECILE].ticker),
                "straddle": set(block[escalation >= DECILE].ticker),
                "call_spread": set(block[momentum >= DECILE].ticker),
            }
            pool = block.ticker.to_numpy()

            for strategy, chosen in selections.items():
                if not chosen:
                    continue
                for mode in ("signal", "random"):
                    tickers = (
                        chosen
                        if mode == "signal"
                        else set(rng.choice(pool, size=min(len(chosen), len(pool)), replace=False))
                    )
                    if strategy == "long_call":
                        legs = nearest(entry_quotes, tickers, "call", 1.05)
                        if legs.empty:
                            continue
                        cost = legs.last_ask
                        proceeds = exit_bid.reindex(legs.option_symbol).to_numpy()
                        fees = CONTRACT_FEE
                    elif strategy == "long_put":
                        legs = nearest(entry_quotes, tickers, "put", 0.95)
                        if legs.empty:
                            continue
                        cost = legs.last_ask
                        proceeds = exit_bid.reindex(legs.option_symbol).to_numpy()
                        fees = CONTRACT_FEE
                    elif strategy == "straddle":
                        calls = nearest(entry_quotes, tickers, "call", 1.0)
                        puts = nearest(entry_quotes, tickers, "put", 1.0)
                        if calls.empty or puts.empty:
                            continue
                        legs = calls.merge(puts, on="ticker", suffixes=("_c", "_p"))
                        if legs.empty:
                            continue
                        cost = legs.last_ask_c + legs.last_ask_p
                        proceeds = np.maximum(exit_bid.reindex(legs.option_symbol_c).to_numpy(), 0) + np.maximum(
                            exit_bid.reindex(legs.option_symbol_p).to_numpy(), 0
                        )
                        fees = 2 * CONTRACT_FEE
                    else:  # call_spread
                        longs = nearest(entry_quotes, tickers, "call", 1.02)
                        shorts = nearest(entry_quotes, tickers, "call", 1.12)
                        if longs.empty or shorts.empty:
                            continue
                        legs = longs.merge(shorts, on="ticker", suffixes=("_l", "_s"))
                        legs = legs[legs.strike_s > legs.strike_l]
                        if legs.empty:
                            continue
                        cost = legs.last_ask_l - legs.last_bid_s
                        legs = legs[cost > 0]
                        cost = cost[cost > 0]
                        if legs.empty:
                            continue
                        long_exit = exit_bid.reindex(legs.option_symbol_l).to_numpy()
                        short_exit = exit_quotes.set_index("option_symbol").last_ask.reindex(
                            legs.option_symbol_s
                        ).to_numpy()
                        proceeds = long_exit - short_exit
                        fees = 2 * CONTRACT_FEE

                    cost = np.asarray(cost, dtype=float)
                    proceeds = np.asarray(proceeds, dtype=float)
                    valid = np.isfinite(proceeds) & np.isfinite(cost) & (cost > 0)
                    if valid.sum() == 0:
                        continue
                    pnl = (proceeds[valid] - cost[valid]) * 100.0 - fees
                    records.append(
                        pd.DataFrame(
                            {
                                "signal_date": signal_date,
                                "sector": sector,
                                "strategy": strategy,
                                "mode": mode,
                                "ticker": legs.ticker.to_numpy()[valid],
                                "cost": cost[valid] * 100.0,
                                "pnl": pnl,
                                "return_on_cost": pnl / (cost[valid] * 100.0),
                            }
                        )
                    )

    if not records:
        raise SystemExit("no trades built")
    trades = pd.concat(records, ignore_index=True)
    trades["sample"] = np.where(trades.signal_date >= SPLIT, "TEST", "TRAIN")
    trades.to_csv(OUT, index=False)

    def summarize(frame: pd.DataFrame) -> pd.Series:
        gains = frame.pnl[frame.pnl > 0].sum()
        losses = -frame.pnl[frame.pnl < 0].sum()
        return pd.Series(
            {
                "n": len(frame),
                "mean_ret": frame.return_on_cost.mean(),
                "median_ret": frame.return_on_cost.median(),
                "win": frame.pnl.gt(0).mean(),
                "PF": gains / losses if losses else np.nan,
                "pnl": frame.pnl.sum(),
            }
        )

    print(f"\nbuilt {len(trades)} trades across {trades.signal_date.nunique()} dates")
    for strategy in ["long_call", "long_put", "straddle", "call_spread"]:
        subset = trades[trades.strategy == strategy]
        if subset.empty:
            continue
        table = subset.groupby(["sample", "mode"]).apply(summarize).round(3)
        print(f"\n=== {strategy.upper()} : all sectors pooled ===")
        print(table.to_string())

    print("\n=== SIGNAL BY SECTOR, TEST HALF, mean return on cost ===")
    test_signal = trades[(trades["sample"] == "TEST") & (trades["mode"] == "signal")]
    pivot = test_signal.pivot_table(
        index="sector", columns="strategy", values="return_on_cost", aggfunc="mean"
    ).round(3)
    counts = test_signal.pivot_table(index="sector", columns="strategy", values="pnl", aggfunc="size")
    print(pivot.to_string())
    print("\ntrade counts:")
    print(counts.to_string())
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
