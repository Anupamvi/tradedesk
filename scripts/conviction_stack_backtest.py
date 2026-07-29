"""Backtest the five-file contract conviction stack with honest option fills.

Entry is the next session's ask on the identical OCC contract, exit is that
contract's bid N sessions later. No mid prices, no model marks.

Quotes come from chain-oi-changes, NOT hot-chains. hot-chains only lists
contracts that traded actively, and only 27% of them are still listed five
sessions later, so using it for exits keeps precisely the contracts that stayed
interesting and silently discards the rest. chain-oi-changes carries a quote for
every contract with open interest (250,642 of 250,812 rows on a sample day). In
a file dated t, last_date is t-1 and last_bid/last_ask are that prior session's
quotes, so the quote for session X is read from the file dated X+1.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conviction_stack import (  # noqa: E402
    ROOT,
    find,
    load_dark_pool,
    load_hot_chains,
    load_oi_confirmation,
    load_screener,
    load_tape,
    open_zip,
)

OUT = ROOT / "out/conviction_stack_backtest.csv"
HOLD_SESSIONS = 5
CONTRACT_FEE = 1.30  # per contract round trip, matching the pipeline convention


def build_stack(signal_date: str, next_date: str) -> pd.DataFrame:
    day = ROOT / signal_date
    hot = load_hot_chains(day)
    screener = load_screener(day)
    universe = hot.merge(
        screener[
            ["ticker", "marketcap", "issue_type", "iv_rank", "implied_move_perc", "next_earnings_date", "close"]
        ].rename(columns={"close": "stock_close"}),
        on="ticker",
        how="inner",
    )
    universe = universe[
        (universe.issue_type == "Common Stock")
        & (universe.marketcap.fillna(0) >= 2e9)
        & (universe.volume >= 250)
        & (universe.open_interest >= 100)
        & (universe.premium >= 250_000)
        & (universe.spread_pct <= 0.15)
        & universe.ask_share.notna()
    ].copy()
    if universe.empty:
        return universe
    universe = universe.merge(load_oi_confirmation(ROOT / next_date), on="option_symbol", how="left")
    universe["oi_confirmed"] = universe.oi_change.fillna(0) > 0
    universe = universe[universe.oi_confirmed & universe.ask_share.ge(0.60)].copy()
    if universe.empty:
        return universe
    tape = load_tape(day, set(universe.option_symbol))
    if tape.empty:
        return tape
    universe = universe.merge(tape, on="option_symbol", how="left")
    universe = universe.merge(load_dark_pool(day), on="ticker", how="left")
    universe["days_to_expiry"] = (universe.expiry - pd.Timestamp(signal_date)).dt.days
    universe["directional_sign"] = np.where(universe.option_type == "call", 1.0, -1.0)
    universe["dp_agrees"] = np.sign(universe.dp_bias.fillna(0)) == universe.directional_sign
    stack = universe[
        universe.tape_ask_share.fillna(0).ge(0.60)
        & universe.tape_largest_ask_print.fillna(0).ge(250_000)
        & universe.volume_to_oi.ge(0.5)
        & universe.days_to_expiry.between(10, 90)
    ].copy()
    if stack.empty:
        return stack
    stack["conviction"] = (
        np.log1p(stack.tape_block_premium.fillna(0) / 1e5)
        + np.log1p(stack.premium / 1e5)
        + 2.0 * (stack.tape_ask_share - 0.5)
        + stack.dp_agrees.astype(float) * 0.5
    )
    stack["signal_date"] = signal_date
    return stack


def session_quotes(session: str, following: str) -> pd.DataFrame:
    """Quotes as of `session`, read from the chain-oi file dated `following`."""
    path = find(ROOT / following, "chain-oi-changes")
    if path is None:
        return pd.DataFrame()
    frame = open_zip(path, ["option_symbol", "last_bid", "last_ask", "last_date", "curr_oi"])
    if "last_date" in frame.columns:
        frame = frame[frame.last_date.astype(str).str.startswith(session)]
    for column in ["last_bid", "last_ask", "curr_oi"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame[frame.last_ask > 0]
    return frame[["option_symbol", "last_bid", "last_ask", "curr_oi"]].drop_duplicates("option_symbol")


def main() -> None:
    days = sorted(p.name for p in ROOT.iterdir() if p.is_dir() and re.fullmatch(r"2026-\d{2}-\d{2}", p.name))
    tape_days = [d for d in days if find(ROOT / d, "bot-eod-report") is not None]
    position = {d: i for i, d in enumerate(days)}

    quote_cache: dict[str, pd.DataFrame] = {}

    def quotes(session: str) -> pd.DataFrame:
        if session not in quote_cache:
            slot = position[session]
            if slot + 1 >= len(days):
                quote_cache[session] = pd.DataFrame()
            else:
                quote_cache[session] = session_quotes(session, days[slot + 1])
        return quote_cache[session]

    rows = []
    for index, signal_date in enumerate(tape_days, 1):
        slot = position[signal_date]
        if slot + 1 + HOLD_SESSIONS >= len(days):
            continue
        entry_date = days[slot + 1]
        exit_date = days[slot + 1 + HOLD_SESSIONS]
        print(f"[backtest] {index}/{len(tape_days)} {signal_date}", flush=True)
        try:
            stack = build_stack(signal_date, days[slot + 1])
        except SystemExit as exc:
            print(f"  skipped: {exc}", flush=True)
            continue
        if stack.empty:
            continue
        entry_quotes = quotes(entry_date)
        exit_quotes = quotes(exit_date)
        if entry_quotes.empty or exit_quotes.empty:
            continue
        merged = stack.merge(
            entry_quotes.rename(
                columns={"last_ask": "entry_ask_quote", "last_bid": "entry_bid_quote"}
            ).drop(columns=["curr_oi"], errors="ignore"),
            on="option_symbol",
            how="left",
        )
        merged = merged.merge(
            exit_quotes.rename(
                columns={"last_bid": "exit_bid", "last_ask": "exit_ask", "curr_oi": "exit_oi"}
            ),
            on="option_symbol",
            how="left",
        )
        merged["entry_date"] = entry_date
        merged["exit_date"] = exit_date
        merged["entry_price"] = merged.entry_ask_quote
        merged["exit_price"] = merged.exit_bid
        merged["tradeable"] = merged.entry_price.gt(0) & merged.exit_price.notna()
        merged["pnl_per_contract"] = (merged.exit_price - merged.entry_price) * 100.0 - CONTRACT_FEE
        merged["return_on_premium"] = merged.pnl_per_contract / (merged.entry_price * 100.0)
        rows.append(merged)

    if not rows:
        raise SystemExit("no backtest rows produced")
    result = pd.concat(rows, ignore_index=True)
    keep = [
        "signal_date", "entry_date", "exit_date", "option_symbol", "ticker", "sector",
        "option_type", "strike", "days_to_expiry", "premium", "ask_share", "oi_change",
        "tape_ask_share", "tape_block_premium", "tape_largest_ask_print", "tape_delta",
        "dp_bias", "dp_agrees", "iv_rank", "conviction", "entry_price", "exit_price",
        "tradeable", "pnl_per_contract", "return_on_premium",
    ]
    keep = [c for c in keep if c in result.columns]
    result[keep].to_csv(OUT, index=False)

    scored = result[result.tradeable].copy()
    print(f"\n=== CONVICTION STACK BACKTEST, {HOLD_SESSIONS}-session hold ===")
    print(f"signals {len(result)}  tradeable {len(scored)}  days {result.signal_date.nunique()}")
    if scored.empty:
        return
    gains = scored.pnl_per_contract[scored.pnl_per_contract > 0].sum()
    losses = -scored.pnl_per_contract[scored.pnl_per_contract < 0].sum()
    print(f"win rate {scored.pnl_per_contract.gt(0).mean():.3f}")
    print(f"avg return on premium {scored.return_on_premium.mean():+.4f}")
    print(f"median return on premium {scored.return_on_premium.median():+.4f}")
    print(f"profit factor {gains / losses if losses else float('nan'):.3f}")
    print(f"avg P&L per contract ${scored.pnl_per_contract.mean():+,.0f}")
    print("\nby month:")
    scored["month"] = scored.signal_date.str[:7]
    print(
        scored.groupby("month")
        .agg(
            trades=("pnl_per_contract", "size"),
            win_rate=("pnl_per_contract", lambda x: (x > 0).mean()),
            avg_return=("return_on_premium", "mean"),
            total_pnl_1_contract=("pnl_per_contract", "sum"),
        )
        .round(4)
        .to_string()
    )
    print("\nby conviction decile:")
    scored["decile"] = pd.qcut(scored.conviction.rank(method="first"), 10, labels=False) + 1
    print(
        scored.groupby("decile")
        .agg(
            trades=("pnl_per_contract", "size"),
            win_rate=("pnl_per_contract", lambda x: (x > 0).mean()),
            avg_return=("return_on_premium", "mean"),
        )
        .round(4)
        .to_string()
    )
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
