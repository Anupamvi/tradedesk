"""Multi-leg structures on the validated signal, with the exit rule that matters.

Honest gap this closes: the +50% managed exit was applied to single legs and
straddles, but spreads were only ever judged on the fixed 40-session hold -- the
method that exit rule had already proven wrong. So the validated view (weak
Technology names fall) was never expressed as a defined-risk spread.

Four structures on the SAME names, dates and exit discipline:

    long_put            buy 0.95x                      -- the validated baseline
    put_debit_spread    buy 0.95x / sell 0.85x         -- cheaper, capped upside
    call_credit_spread  sell 1.02x / buy 1.12x         -- collect premium, same view
    put_ratio_backspread sell 1.00x / buy 2x 0.90x     -- convex if it really falls

Debits take profit at +50% of cost. Credits take profit at 50% of the credit
collected, which is the equivalent milestone rather than a fabricated one.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import managed_exit_backtest as base  # noqa: E402

OUT = base.ROOT / "out/multileg_structures.csv"
PROFIT_TARGET = 0.50
DECILE = 0.20
STRUCTURES = ("long_put", "put_debit_spread", "call_credit_spread", "put_ratio_backspread")


def leg_near(chain: pd.DataFrame, ticker: str, option_type: str, moneyness: float):
    legs = chain[
        chain.ticker.eq(ticker)
        & chain.option_type.eq(option_type)
        & chain.dte.between(*base.DTE_BAND)
        & (chain.curr_oi >= base.MIN_OPEN_INTEREST if hasattr(base, "MIN_OPEN_INTEREST") else chain.curr_oi >= 50)
        & (chain.spread_pct <= base.MAX_SPREAD_PCT)
    ]
    if legs.empty:
        return None
    target = legs.stock_price.iloc[0] * moneyness
    legs = legs.assign(
        gap=(legs.strike - target).abs(),
        dte_gap=(legs.dte - base.TARGET_DTE).abs(),
    )
    return legs.sort_values(["dte_gap", "gap"]).iloc[0]


def build(chain: pd.DataFrame, ticker: str, structure: str):
    """Returns (entry_cost, max_risk, legs, is_credit) or None."""
    if structure == "long_put":
        leg = leg_near(chain, ticker, "put", 0.95)
        if leg is None:
            return None
        return leg.last_ask, leg.last_ask, [(leg.option_symbol, +1)], False

    if structure == "put_debit_spread":
        long_leg = leg_near(chain, ticker, "put", 0.95)
        short_leg = leg_near(chain, ticker, "put", 0.85)
        if long_leg is None or short_leg is None or short_leg.strike >= long_leg.strike:
            return None
        debit = long_leg.last_ask - short_leg.last_bid
        if debit <= 0:
            return None
        return debit, debit, [(long_leg.option_symbol, +1), (short_leg.option_symbol, -1)], False

    if structure == "call_credit_spread":
        short_leg = leg_near(chain, ticker, "call", 1.02)
        long_leg = leg_near(chain, ticker, "call", 1.12)
        if short_leg is None or long_leg is None or long_leg.strike <= short_leg.strike:
            return None
        credit = short_leg.last_bid - long_leg.last_ask
        if credit <= 0:
            return None
        width = long_leg.strike - short_leg.strike
        return credit, width - credit, [(short_leg.option_symbol, -1), (long_leg.option_symbol, +1)], True

    if structure == "put_ratio_backspread":
        short_leg = leg_near(chain, ticker, "put", 1.00)
        long_leg = leg_near(chain, ticker, "put", 0.90)
        if short_leg is None or long_leg is None or long_leg.strike >= short_leg.strike:
            return None
        debit = 2 * long_leg.last_ask - short_leg.last_bid
        if debit <= 0:
            return None
        return debit, debit, [(short_leg.option_symbol, -1), (long_leg.option_symbol, +2)], False

    return None


def mark(bids: pd.Series, asks: pd.Series, legs: list[tuple[str, int]]) -> float | None:
    """Close the position: sell longs at bid, buy back shorts at ask."""
    total = 0.0
    for symbol, quantity in legs:
        price = bids.get(symbol) if quantity > 0 else asks.get(symbol)
        if price is None or not np.isfinite(price):
            return None
        total += quantity * max(float(price), 0.0)
    return total


def run(panel, days, quote_for, structure: str) -> pd.DataFrame:
    open_positions: list[dict] = []
    closed: list[dict] = []
    held: set[str] = set()

    for index, session in enumerate(days):
        chain = quote_for(session)
        if chain.empty:
            continue
        indexed = chain.set_index("option_symbol")
        bids, asks = indexed.last_bid, indexed.last_ask

        still_open = []
        for pos in open_positions:
            value = mark(bids, asks, pos["legs"])
            age = index - pos["entry_index"]
            if value is None:
                if age < base.MAX_HOLD_SESSIONS if hasattr(base, "MAX_HOLD_SESSIONS") else age < base.MAX_HOLD:
                    still_open.append(pos)
                    continue
                value = pos["last_mark"]
            else:
                pos["last_mark"] = value

            if pos["is_credit"]:
                # mark() returns the position's value to the holder: negative for
                # a credit spread, because closing it costs a debit. The credit
                # was received up front, so P&L is credit PLUS that negative
                # value. Subtracting it instead makes every trade a winner --
                # which is exactly what a 100% win rate and a 0-session hold
                # looked like before this was fixed.
                gain_dollars = pos["entry"] + value
                progress = gain_dollars / pos["entry"] if pos["entry"] else 0.0
            else:
                gain_dollars = value - pos["entry"]
                progress = gain_dollars / pos["entry"] if pos["entry"] else 0.0

            if progress >= PROFIT_TARGET:
                reason = "profit_target"
            elif age >= base.MAX_HOLD:
                reason = "time_stop"
            else:
                still_open.append(pos)
                continue

            pnl = gain_dollars * 100.0 - base.CONTRACT_FEE * len(pos["legs"])
            closed.append(
                {
                    "signal_date": pos["signal_date"],
                    "ticker": pos["ticker"],
                    "structure": structure,
                    "entry_cost": pos["max_risk"] * 100.0,
                    "held": age,
                    "exit_reason": reason,
                    "pnl": pnl,
                    "return_on_risk": pnl / (pos["max_risk"] * 100.0),
                }
            )
            held.discard(pos["ticker"])
        open_positions = still_open

        if not base.has_full_observation_horizon(index, days):
            continue
        day = panel[panel.date == session].dropna(subset=["pos_52w"])
        if day.empty or len(day) < base.MIN_PER_SECTOR:
            continue
        entry_chain = quote_for(days[index + 1])
        if entry_chain.empty:
            continue
        momentum = day.pos_52w.rank(pct=True)
        for ticker in set(day[momentum <= DECILE].ticker) - held:
            built = build(entry_chain, ticker, structure)
            if built is None:
                continue
            entry, max_risk, legs, is_credit = built
            if max_risk * 100.0 < 200:
                continue
            open_positions.append(
                {
                    "signal_date": session,
                    "ticker": ticker,
                    "entry_index": index + 1,
                    "entry": entry,
                    "max_risk": max_risk,
                    "legs": legs,
                    "is_credit": is_credit,
                    "last_mark": entry,
                }
            )
            held.add(ticker)
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
        & (panel.sector == "Technology")
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

    print("[multileg] warming quote cache", flush=True)
    for session in days:
        quote_for(session)

    frames = []
    for structure in STRUCTURES:
        trades = run(panel, days, quote_for, structure)
        if trades.empty:
            print(f"{structure:<22} no trades built", flush=True)
            continue
        trades["sample"] = np.where(trades.signal_date >= base.SPLIT, "TEST", "TRAIN")
        frames.append(trades)
        for sample in ("TRAIN", "TEST"):
            frame = trades[trades["sample"] == sample]
            if len(frame) < 10:
                continue
            print(
                "{:<22} {:<6} n={:<4} mean={:+.3f} med={:+.3f} win={:.2f} PF={:>6.2f} "
                "risk=${:>6,.0f} hold={:>4.1f} pnl=${:>+9,.0f}".format(
                    structure, sample, len(frame), frame.return_on_risk.mean(),
                    frame.return_on_risk.median(), frame.pnl.gt(0).mean(),
                    profit_factor(frame.pnl), frame.entry_cost.mean(),
                    frame.held.mean(), frame.pnl.sum(),
                ),
                flush=True,
            )
        print(flush=True)

    if frames:
        pd.concat(frames, ignore_index=True).to_csv(OUT, index=False)
        print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
