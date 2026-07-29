"""Straddles on extreme-flow names, built from real chain quotes.

Everything measured so far says the same thing: on these names direction is a
coin flip (46% right) but SIZE is predictable (2.3x the universe's move, 1.08x
implied). A bought call needs both. A straddle needs only size.

Quotes come from chain-oi-changes, which carries every contract with open
interest. In a file dated t, last_date is t-1, so the quote for session X lives
in the file dated X+1. Entry pays both asks, exit sells both bids.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conviction_stack import ROOT, find, open_zip, parse_occ  # noqa: E402

OUT = ROOT / "out/extreme_flow_straddle_backtest.csv"
PANEL = ROOT / "out/uw_all_feeds.csv"
SPLIT = "2026-04-14"
HOLD = 5
TARGET_DTE = 30
DTE_BAND = (20, 45)
CONTRACT_FEE = 1.30
MAX_SPREAD_PCT = 0.10


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
    for column in ["last_bid", "last_ask", "curr_oi", "stock_price", "dte", "strike"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame[(frame.last_ask > 0) & (frame.last_bid >= 0) & (frame.stock_price > 0)]
    # The OCC symbol is the authoritative strike; drop the feed column so the
    # parsed one does not collide with it.
    frame = frame.drop(columns=["strike"], errors="ignore")
    frame = frame.join(parse_occ(frame.option_symbol.astype(str)))
    frame = frame[frame.ticker.notna()]
    frame["spread_pct"] = (frame.last_ask - frame.last_bid) / frame.last_ask
    return frame.drop_duplicates("option_symbol")


def pick_straddles(quotes: pd.DataFrame, tickers: set[str]) -> pd.DataFrame:
    quotes = quotes[
        quotes.ticker.isin(tickers)
        & quotes.dte.between(*DTE_BAND)
        & (quotes.curr_oi >= 100)
        & (quotes.spread_pct <= MAX_SPREAD_PCT)
    ]
    rows = []
    for (ticker, expiry), group in quotes.groupby(["ticker", "expiry"]):
        calls = group[group.option_type == "call"]
        puts = group[group.option_type == "put"]
        if calls.empty or puts.empty:
            continue
        spot = group.stock_price.median()
        call = calls.iloc[(calls.strike - spot).abs().argsort().iloc[0]]
        put = puts.iloc[(puts.strike - spot).abs().argsort().iloc[0]]
        if abs(call.strike - put.strike) > 0.02 * spot:
            continue
        debit = call.last_ask + put.last_ask
        if debit <= 0:
            continue
        rows.append(
            {
                "ticker": ticker,
                "expiry": expiry,
                "dte": call.dte,
                "spot": spot,
                "call_symbol": call.option_symbol,
                "put_symbol": put.option_symbol,
                "call_strike": call.strike,
                "put_strike": put.strike,
                "entry_debit": debit,
                "entry_spread_pct": (call.spread_pct + put.spread_pct) / 2.0,
                "breakeven_pct": debit / spot,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["dte_gap"] = (frame.dte - TARGET_DTE).abs()
    return frame.sort_values("dte_gap").groupby("ticker", as_index=False).first()


def main() -> None:
    selector = sys.argv[1] if len(sys.argv) > 1 else "oi_built_premium"
    top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    panel = pd.read_csv(
        PANEL, usecols=["date", "ticker", "issue_type", "marketcap", selector], low_memory=False
    )
    panel = panel[
        (panel.issue_type == "Common Stock")
        & (panel.marketcap.fillna(0) >= 2e9)
        & panel[selector].notna()
    ]
    picks = {
        date: set(day.nlargest(top_n, selector).ticker)
        for date, day in panel.groupby("date")
    }

    days = sorted(p.name for p in ROOT.iterdir() if p.is_dir() and re.fullmatch(r"2026-\d{2}-\d{2}", p.name))
    position = {d: i for i, d in enumerate(days)}
    cache: dict[str, pd.DataFrame] = {}

    def quotes(session: str) -> pd.DataFrame:
        if session not in cache:
            slot = position[session]
            cache[session] = (
                chain_quotes(session, days[slot + 1]) if slot + 1 < len(days) else pd.DataFrame()
            )
        return cache[session]

    rows = []
    for signal_date in days:
        slot = position[signal_date]
        if slot + 1 + HOLD >= len(days):
            continue
        tickers = picks.get(signal_date)
        if not tickers:
            continue
        entry_date, exit_date = days[slot + 1], days[slot + 1 + HOLD]
        entry_quotes, exit_quotes = quotes(entry_date), quotes(exit_date)
        if entry_quotes.empty or exit_quotes.empty:
            continue
        straddles = pick_straddles(entry_quotes, tickers)
        if straddles.empty:
            continue
        exit_lookup = exit_quotes.set_index("option_symbol")[["last_bid"]]
        call_exit = exit_lookup.reindex(straddles.call_symbol).last_bid.to_numpy()
        put_exit = exit_lookup.reindex(straddles.put_symbol).last_bid.to_numpy()
        straddles["exit_credit"] = np.maximum(call_exit, 0) + np.maximum(put_exit, 0)
        straddles["signal_date"] = signal_date
        straddles["entry_date"] = entry_date
        straddles["exit_date"] = exit_date
        straddles["tradeable"] = np.isfinite(call_exit) & np.isfinite(put_exit)
        straddles["pnl"] = (straddles.exit_credit - straddles.entry_debit) * 100.0 - 2 * CONTRACT_FEE
        straddles["return_on_premium"] = straddles.pnl / (straddles.entry_debit * 100.0)
        rows.append(straddles)
        print(f"[straddle] {signal_date} -> {len(straddles)} straddles", flush=True)

    if not rows:
        raise SystemExit("no straddles built")
    result = pd.concat(rows, ignore_index=True)
    result.to_csv(OUT, index=False)
    scored = result[result.tradeable].copy()
    scored["sample"] = np.where(scored.signal_date >= SPLIT, "TEST", "TRAIN")

    print(f"\n=== STRADDLES ON TOP-{top_n} {selector}, {HOLD}-session hold ===")
    print(f"built {len(result)}  tradeable {len(scored)}  days {result.signal_date.nunique()}")
    for sample in ("TRAIN", "TEST"):
        frame = scored[scored["sample"] == sample]
        if len(frame) < 20:
            continue
        gains = frame.pnl[frame.pnl > 0].sum()
        losses = -frame.pnl[frame.pnl < 0].sum()
        print(
            f"\n{sample}: n={len(frame)}  mean_return={frame.return_on_premium.mean():+.4f}  "
            f"median={frame.return_on_premium.median():+.4f}  win={frame.pnl.gt(0).mean():.3f}  "
            f"PF={gains / losses if losses else float('nan'):.3f}  "
            f"breakeven_req={frame.breakeven_pct.mean():.3f}"
        )
        frame = frame.copy()
        frame["month"] = frame.signal_date.str[:7]
        print(
            frame.groupby("month")
            .agg(n=("pnl", "size"), mean_ret=("return_on_premium", "mean"), pnl=("pnl", "sum"))
            .round(4)
            .to_string()
        )
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
