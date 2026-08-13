"""Event-driven backtest with profit taking, stops and a per-name cap.

Every earlier test held a fixed 40 sessions no matter what happened in between,
so a position up 200% on day 5 was carried to day 40 and handed the gain back.
That is not how the book would be run, and it penalised exactly the structures
whose edge is a large move that later mean-reverts.

This walks forward day by day, marks every open position against real quotes,
and closes on the first rule that triggers:

  profit target   exit at the bid the day it is reached
  stop loss       exit at the bid the day it is breached
  time stop       exit at the bid after max_hold sessions

A ticker may hold only one open position at a time, so a single name cannot
become the whole book.
"""
from __future__ import annotations

import itertools
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conviction_stack import ROOT, find, open_zip, parse_occ  # noqa: E402

PANEL = ROOT / "out/uw_all_feeds.csv"
OUT = ROOT / "out/managed_exit_backtest.csv"
SPLIT = "2026-04-14"
DTE_BAND = (60, 110)
TARGET_DTE = 80
MAX_SPREAD_PCT = 0.12
CONTRACT_FEE = 1.30
DECILE = 0.80
MIN_PER_SECTOR = 12
SECTORS = ("Technology", "Financial Services")
MAX_HOLD = 40


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


def run(
    panel: pd.DataFrame,
    days: list[str],
    quote_for: dict,
    strategy: str,
    profit_target: float | None,
    stop_loss: float | None,
    max_hold: int,
    one_per_ticker: bool,
) -> pd.DataFrame:
    open_positions: list[dict] = []
    closed: list[dict] = []
    held: set[str] = set()

    for index, session in enumerate(days):
        quotes = quote_for(session)
        if quotes.empty:
            continue
        bid = quotes.set_index("option_symbol").last_bid

        # 1. mark and manage what is already open
        still_open = []
        for position in open_positions:
            current = bid.get(position["exit_symbol"])
            if position["strategy"] == "straddle":
                second = bid.get(position["exit_symbol_2"])
                current = (
                    (max(current, 0.0) + max(second, 0.0))
                    if current is not None and second is not None and np.isfinite(current) and np.isfinite(second)
                    else None
                )
            age = index - position["entry_index"]
            if current is None or not np.isfinite(current):
                if age < max_hold:
                    still_open.append(position)
                    continue
                current = position["last_mark"]
            else:
                position["last_mark"] = current

            gain = current / position["entry_cost"] - 1.0
            reason = None
            if profit_target is not None and gain >= profit_target:
                reason = "profit_target"
            elif stop_loss is not None and gain <= -stop_loss:
                reason = "stop_loss"
            elif age >= max_hold:
                reason = "time_stop"
            if reason is None:
                still_open.append(position)
                continue

            pnl = (current - position["entry_cost"]) * 100.0 - position["fees"]
            closed.append(
                {
                    **{k: position[k] for k in ("signal_date", "ticker", "sector", "strategy", "entry_cost")},
                    "exit_date": session,
                    "held_sessions": age,
                    "exit_reason": reason,
                    "pnl": pnl,
                    "return_on_cost": pnl / (position["entry_cost"] * 100.0),
                }
            )
            held.discard(position["ticker"])
        open_positions = still_open

        # 2. open new positions on this session's signal
        if index + 1 >= len(days):
            continue
        day = panel[panel.date == session].dropna(subset=["pos_52w", "flow_escalation"])
        if day.empty:
            continue
        entry_session = days[index + 1]
        entry_quotes = quote_for(entry_session)
        if entry_quotes.empty:
            continue

        for sector, block in day.groupby("sector"):
            if sector not in SECTORS or len(block) < MIN_PER_SECTOR:
                continue
            momentum = block.pos_52w.rank(pct=True)
            escalation = block.flow_escalation.rank(pct=True)
            chosen = set(
                block[momentum >= DECILE].ticker
                if strategy in {"long_call", "call_spread"}
                else block[escalation >= DECILE].ticker
            )
            if one_per_ticker:
                chosen -= held
            if not chosen:
                continue

            if strategy == "long_call":
                legs = nearest(entry_quotes, chosen, "call", 1.05)
                if legs.empty:
                    continue
                for row in legs.itertuples():
                    open_positions.append(
                        {
                            "signal_date": session,
                            "ticker": row.ticker,
                            "sector": sector,
                            "strategy": strategy,
                            "entry_index": index + 1,
                            "entry_cost": row.last_ask,
                            "exit_symbol": row.option_symbol,
                            "exit_symbol_2": None,
                            "fees": CONTRACT_FEE,
                            "last_mark": row.last_bid,
                        }
                    )
                    held.add(row.ticker)
            elif strategy == "straddle":
                calls = nearest(entry_quotes, chosen, "call", 1.0)
                puts = nearest(entry_quotes, chosen, "put", 1.0)
                if calls.empty or puts.empty:
                    continue
                merged = calls.merge(puts, on="ticker", suffixes=("_c", "_p"))
                for row in merged.itertuples():
                    open_positions.append(
                        {
                            "signal_date": session,
                            "ticker": row.ticker,
                            "sector": sector,
                            "strategy": strategy,
                            "entry_index": index + 1,
                            "entry_cost": row.last_ask_c + row.last_ask_p,
                            "exit_symbol": row.option_symbol_c,
                            "exit_symbol_2": row.option_symbol_p,
                            "fees": 2 * CONTRACT_FEE,
                            "last_mark": row.last_bid_c + row.last_bid_p,
                        }
                    )
                    held.add(row.ticker)

    return pd.DataFrame(closed)


def main() -> None:
    columns = ["date", "ticker", "sector", "issue_type", "marketcap", "close", "pos_52w", "hc_premium"]
    panel = pd.read_csv(PANEL, usecols=columns, low_memory=False)
    panel["date"] = pd.to_datetime(panel["date"]).dt.strftime("%Y-%m-%d")
    panel = panel[
        (panel.issue_type == "Common Stock")
        & (panel.marketcap.fillna(0) >= 2e9)
        & panel.sector.isin(SECTORS)
    ].sort_values(["ticker", "date"])
    grouped = panel.groupby("ticker")
    panel["flow_avg_20"] = grouped.hc_premium.transform(lambda s: s.rolling(20, min_periods=10).mean())
    panel["flow_escalation"] = panel.hc_premium / panel.flow_avg_20.replace(0, np.nan)

    days = sorted(p.name for p in ROOT.iterdir() if p.is_dir() and re.fullmatch(r"2026-\d{2}-\d{2}", p.name))
    position = {d: i for i, d in enumerate(days)}
    cache: dict[str, pd.DataFrame] = {}

    def quote_for(session: str) -> pd.DataFrame:
        if session not in cache:
            slot = position[session]
            cache[session] = chain_quotes(session, days[slot + 1]) if slot + 1 < len(days) else pd.DataFrame()
        return cache[session]

    print("[managed] warming quote cache", flush=True)
    for session in days:
        quote_for(session)

    results = []
    grid = list(itertools.product(["long_call", "straddle"], [None, 0.5, 1.0, 2.0], [None, 0.5]))
    for strategy, target, stop in grid:
        trades = run(panel, days, quote_for, strategy, target, stop, MAX_HOLD, one_per_ticker=True)
        if trades.empty:
            continue
        trades["sample"] = np.where(trades.signal_date >= SPLIT, "TEST", "TRAIN")
        trades["profit_target"] = target
        trades["stop_loss"] = stop
        results.append(trades)
        for sample in ("TRAIN", "TEST"):
            frame = trades[trades["sample"] == sample]
            if len(frame) < 20:
                continue
            gains = frame.pnl[frame.pnl > 0].sum()
            losses = -frame.pnl[frame.pnl < 0].sum()
            months = pd.to_datetime(frame.signal_date).dt.to_period("M").nunique()
            top = frame.groupby("ticker").pnl.sum().sort_values(ascending=False)
            print(
                "{:<11} tgt={:<5} stop={:<5} {:<6} n={:<4} mean={:+.3f} med={:+.3f} "
                "win={:.2f} PF={:>6.2f} hold={:>4.1f} pnl/mo=${:>+10,.0f} top1={:.0f}%".format(
                    strategy,
                    "none" if target is None else f"{target:.0%}",
                    "none" if stop is None else f"{stop:.0%}",
                    sample,
                    len(frame),
                    frame.return_on_cost.mean(),
                    frame.return_on_cost.median(),
                    frame.pnl.gt(0).mean(),
                    gains / losses if losses else float("nan"),
                    frame.held_sessions.mean(),
                    frame.pnl.sum() / max(months, 1),
                    100 * top.iloc[0] / frame.pnl.sum() if frame.pnl.sum() else float("nan"),
                ),
                flush=True,
            )
        print(flush=True)

    if results:
        pd.concat(results, ignore_index=True).to_csv(OUT, index=False)
        print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
