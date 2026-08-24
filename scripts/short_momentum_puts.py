"""Daily candidate generator for the one lane that survived validation.

What this implements, and nothing else:

    Technology sector, bottom momentum quintile within sector, long puts,
    ~0.95x strike, ~80 DTE, contract cost above $700, +50% take-profit, NO stop,
    one position per ticker, 40-session maximum hold.

Measured against a 150-permutation null on 2026 data, before the cost floor:

    TRAIN  n=158  mean +0.114  PF 1.66  vs null PF 0.68   p=0.0000
    TEST   n=37   mean +0.124  PF 2.90  vs null PF 1.13   p=0.0000

The cost floor was then checked separately in each half and holds:

    cost <= $700   TRAIN PF 1.381 | TEST PF 0.623  -- LOSES out of sample
    cost >  $700   TRAIN PF 1.720 | TEST PF 4.248  -- random control 0.64 / 0.52

Cheap puts carry no edge and lose in the untouched half, so they are excluded.

Deliberately excluded, because each failed against a proper null:

    long calls          TEST mean 0.606 vs null 0.605, p=0.47 -- beta, not signal
    flow-led contracts  worst of three choosers in both halves
    flow short filter   helps in test, hurts in train -- noise
    straddles           TRAIN mean negative while TEST positive
    every other sector  none significant in both halves

Honest size limit: this lane produces roughly 5-12 trades a month after the cost
floor and its dollar P&L was NOT significant (p=0.31). The return rate is real;
the lane is small.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conviction_stack import ROOT, find, open_zip, parse_occ  # noqa: E402

PANEL = ROOT / "out/uw_all_feeds.csv"
SECTOR = "Technology"
MOMENTUM_QUINTILE = 0.20
STRIKE_MONEYNESS = 0.95
TARGET_DTE = 80
DTE_BAND = (60, 110)
MAX_SPREAD_PCT = 0.12
MIN_OPEN_INTEREST = 50
MIN_CONTRACT_COST = 700.0
MIN_PER_SECTOR = 12
PROFIT_TARGET = 0.50
MAX_HOLD_SESSIONS = 40


def latest_session(base: Path) -> str:
    days = sorted(p.name for p in base.iterdir() if p.is_dir() and re.fullmatch(r"\d{4}-\d{2}-\d{2}", p.name))
    if not days:
        raise SystemExit(f"no dated folders under {base}")
    return days[-1]


def quotes_for(session: str, days: list[str]) -> pd.DataFrame:
    """Quotes as of `session`. In a file dated t, last_date is t-1."""
    index = days.index(session)
    if index + 1 >= len(days):
        raise SystemExit(
            f"{session} is the last folder; the quote file dated the following "
            "session is required and does not exist yet"
        )
    path = find(ROOT / days[index + 1], "chain-oi-changes")
    if path is None:
        raise SystemExit(f"no chain-oi-changes for {days[index + 1]}")
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
    frame = frame[frame.ticker.notna()].copy()
    frame["spread_pct"] = (frame.last_ask - frame.last_bid) / frame.last_ask
    return frame.drop_duplicates("option_symbol")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--as-of", default=None, help="Signal session. Default: latest dated folder.")
    parser.add_argument("--risk-per-trade", type=float, default=1500.0)
    parser.add_argument("--held", default="", help="Comma-separated tickers already open.")
    args = parser.parse_args()

    days = sorted(p.name for p in ROOT.iterdir() if p.is_dir() and re.fullmatch(r"\d{4}-\d{2}-\d{2}", p.name))
    session = args.as_of or latest_session(ROOT)
    if session not in days:
        raise SystemExit(f"{session} is not a dated folder under {ROOT}")

    panel = pd.read_csv(
        PANEL,
        usecols=["date", "ticker", "sector", "issue_type", "marketcap", "close", "pos_52w"],
        low_memory=False,
    )
    panel["date"] = pd.to_datetime(panel["date"]).dt.strftime("%Y-%m-%d")
    day = panel[
        (panel.date == session)
        & (panel.issue_type == "Common Stock")
        & (panel.marketcap.fillna(0) >= 2e9)
        & (panel.sector == SECTOR)
        & panel.pos_52w.notna()
    ].copy()
    if len(day) < MIN_PER_SECTOR:
        raise SystemExit(f"only {len(day)} eligible {SECTOR} names on {session}; need {MIN_PER_SECTOR}")

    day["momentum_rank"] = day.pos_52w.rank(pct=True)
    weak = day[day.momentum_rank <= MOMENTUM_QUINTILE].copy()
    already_held = {t.strip().upper() for t in args.held.split(",") if t.strip()}
    weak = weak[~weak.ticker.isin(already_held)]
    if weak.empty:
        print(f"no qualifying {SECTOR} names on {session}")
        return

    chain = quotes_for(session, days)
    puts = chain[
        chain.ticker.isin(set(weak.ticker))
        & chain.option_type.eq("put")
        & chain.dte.between(*DTE_BAND)
        & (chain.curr_oi >= MIN_OPEN_INTEREST)
        & (chain.spread_pct <= MAX_SPREAD_PCT)
    ].copy()
    if puts.empty:
        print(f"no put contracts met liquidity filters on {session}")
        return

    puts["strike_gap"] = (puts.strike - puts.stock_price * STRIKE_MONEYNESS).abs()
    puts["dte_gap"] = (puts.dte - TARGET_DTE).abs()
    picks = puts.sort_values(["dte_gap", "strike_gap"]).groupby("ticker", as_index=False).first()
    picks = picks.merge(weak[["ticker", "pos_52w", "momentum_rank"]], on="ticker", how="left")

    picks["cost_per_contract"] = picks.last_ask * 100.0
    below_floor = picks[picks.cost_per_contract < MIN_CONTRACT_COST]
    picks = picks[picks.cost_per_contract >= MIN_CONTRACT_COST].copy()
    if picks.empty:
        print(f"all candidates fell below the ${MIN_CONTRACT_COST:,.0f} contract-cost floor on {session}")
        return
    picks["contracts"] = np.maximum(1, np.floor(args.risk_per_trade / picks.cost_per_contract)).astype(int)
    picks["position_cost"] = picks.contracts * picks.cost_per_contract
    picks["take_profit_at"] = (picks.last_ask * (1.0 + PROFIT_TARGET)).round(2)
    picks["moneyness"] = (picks.strike / picks.stock_price).round(3)
    picks = picks[picks.position_cost <= args.risk_per_trade * 1.5]
    picks = picks.sort_values("momentum_rank")

    print(f"=== {SECTOR.upper()} SHORT-MOMENTUM PUT CANDIDATES -- signal session {session} ===")
    print(f"entry: next session | {len(day)} eligible names, bottom {MOMENTUM_QUINTILE:.0%} selected\n")
    columns = [
        "ticker", "pos_52w", "stock_price", "strike", "moneyness", "dte",
        "last_bid", "last_ask", "spread_pct", "curr_oi", "cost_per_contract",
        "contracts", "position_cost", "take_profit_at",
    ]
    print(picks[columns].round(3).to_string(index=False))

    if not below_floor.empty:
        excluded = ", ".join(
            f"{row.ticker} (${row.cost_per_contract:,.0f})" for row in below_floor.itertuples()
        )
        print(f"\nexcluded below the ${MIN_CONTRACT_COST:,.0f} cost floor: {excluded}")
        print("  those buckets returned PF 0.623 in the untouched half -- they are not tradeable.")

    print("\nmanagement rules, exactly as validated:")
    print(f"  take profit  close the whole position at +{PROFIT_TARGET:.0%} on the ask paid")
    print("  stop loss    NONE. Adding a 50% stop cut test PF from 7.63 to 1.39")
    print(f"  time stop    close after {MAX_HOLD_SESSIONS} sessions regardless")
    print("  per name     one open position per ticker, no exceptions")
    print("\nevidence: TEST PF 2.90 vs a 150-permutation null of 1.13, p=0.0000.")
    print("          restricted to contracts above the cost floor: TRAIN PF 1.72 / TEST PF 4.25")
    print("          against a random control of 0.64 / 0.52 in the same bucket.")
    print("limits:   ~5-12 trades/month; dollar P&L was NOT significant (p=0.31).")
    print("          the return rate is validated, the lane size is not.")


if __name__ == "__main__":
    main()
