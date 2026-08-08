"""Daily candidate generator for the one lane that survived validation.

What this implements, and nothing else:

    Technology sector, bottom momentum quintile within sector, long puts,
    ~0.95x strike, ~80 DTE, contract cost above $700, +50% take-profit, NO stop,
    one position per ticker, 40-session maximum hold.

Measured against a 150-permutation null that holds entry date, sector, option
type and number of names fixed, and randomizes only WHICH names are chosen:

    TRAIN  n=158  mean +0.114  PF 1.66  vs null PF 0.69   mean p=0.0000, PF p=0.0000
    TEST   n=37   mean +0.124  PF 2.90  vs null PF 1.04   mean p=0.0250, PF p=0.0000

The selection is real, but read what it does and does not do. Out of sample the
WIN RATE is not significant (p=0.20) and the DOLLAR P&L is not significant
(p=0.31). The edge is in how much you win, not how often, and it has not been
shown to produce a significant dollar result. Size accordingly.

The cost floor was then checked separately in each half:

    cost <= $700   TRAIN PF 1.381 | TEST PF 0.623  -- LOSES out of sample
    cost >  $700   TRAIN PF 1.720 | TEST PF 4.248

It separates pooled (Mann-Whitney p=0.021) and in TEST (p=0.014) but NOT within
TRAIN (p=0.164). Supported, not proven. Cheap puts are excluded on that basis.

KNOWN WEAKNESSES -- these are not hypothetical, they are measured:

    * NOT profitable every month. 4 of 6. March PF 0.41, April PF 0.53. This
      lane loses money outright when Technology rallies.
    * The 114 trades come from only 52 entry dates, and on 52% of those dates
      the whole basket shared one outcome. It herds. Trade-level statistics
      overstate it; the day-clustered bootstrap p05 is 1.24 against a 1.20 bar,
      which is a pass by a hair.
    * January alone contributed $27,287 of $34,958 lifetime P&L.

Deliberately excluded, because each failed against the same matched null:

    long calls          TEST mean p=0.47, PF p=0.35, win p=0.09, P&L p=0.76.
                        Beta. Train looks superb (all p=0.0000) and none of it
                        survives out of sample. Pairing calls with these puts
                        does fill the March/April hole, but it does so by adding
                        market exposure, not a second edge.
    flow-led contracts  worst of three choosers in both halves
    flow short filter   helps in test, hurts in train -- noise
    straddles           TRAIN mean negative while TEST positive
    every other sector  none significant in both halves

Honest size limit: roughly 5-12 trades a month after the cost floor. The return
rate is real; the lane is small and it is not a standalone book.
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

# There is ONE rate, not a tier. Splitting qualifying tickets into HIGH/MEDIUM by
# contract cost was tested and the buckets are not distinguishable: win rates
# 52.9% (n=70) vs 59.5% (n=37) give Fisher exact p=0.546, and the return
# distributions give Mann-Whitney p=0.781. Grading tickets against each other on
# that basis would be inventing a distinction the data does not contain.
#
# The only split the evidence supports is binary: above the cost floor, or out.
LANE_WIN_RATE = 0.553      # n=114, Technology puts, cost >= $700
LANE_PROFIT_FACTOR = 2.01
LANE_SAMPLE = 114


def latest_session(base: Path) -> str:
    days = sorted(p.name for p in base.iterdir() if p.is_dir() and re.fullmatch(r"\d{4}-\d{2}-\d{2}", p.name))
    if not days:
        raise SystemExit(f"no dated folders under {base}")
    return days[-1]


def quotes_for(session: str, days: list[str]) -> tuple[pd.DataFrame, str]:
    """Quotes as of `session`. In a file dated t, last_date is t-1.

    For a signal on the newest session the following day's file does not exist
    yet, so the most recent available quotes are used instead and the session
    they belong to is returned so the caller can say so plainly. Entry prices
    will differ; these are for selection and sizing, not for a limit order.
    """
    index = days.index(session)
    candidates = days[index + 1 :] or []
    for following in candidates:
        path = find(ROOT / following, "chain-oi-changes")
        if path is not None:
            return _load_quotes(path, session), session
    # No later file: fall back to the newest chain-oi we have.
    for following in reversed(days[: index + 1]):
        path = find(ROOT / following, "chain-oi-changes")
        if path is None:
            continue
        frame = _load_quotes(path, None)
        if frame.empty:
            continue
        quoted = str(frame.quote_session.iloc[0])
        return frame, quoted
    raise SystemExit(f"no chain-oi-changes file usable for {session}")


def _load_quotes(path: Path, session: str | None) -> pd.DataFrame:
    frame = open_zip(
        path,
        ["option_symbol", "last_bid", "last_ask", "last_date", "curr_oi", "stock_price", "dte", "strike"],
    )
    if "last_date" in frame.columns:
        frame["quote_session"] = frame.last_date.astype(str).str.slice(0, 10)
        if session is not None:
            frame = frame[frame.quote_session.eq(session)]
    else:
        frame["quote_session"] = "unknown"
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

    chain, quote_session = quotes_for(session, days)
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

    print(f"=== {SECTOR.upper()} SHORT-MOMENTUM PUTS -- signal session {session} ===")
    if quote_session != session:
        print(f"!! QUOTES ARE AS OF {quote_session}, NOT {session}. The next session's file")
        print("   does not exist yet. Names and sizing are valid; RE-PRICE before sending.")
    print(f"entry: next session open | {len(day)} eligible names, weakest {MOMENTUM_QUINTILE:.0%} taken\n")

    for row in picks.itertuples():
        expiry = pd.Timestamp(row.expiry).strftime("%Y-%m-%d")
        print(f"\U0001F7E2 {row.ticker:<6} BUY {row.contracts} x PUT  ${row.strike:g} strike  exp {expiry} ({int(row.dte)} DTE)")
        print(f"     spot ${row.stock_price:,.2f}   strike is {(1 - row.moneyness) * 100:.1f}% below spot   52w position {row.pos_52w:.3f}")
        print(f"     ENTRY   pay debit up to ${row.last_ask:,.2f} per contract   (bid {row.last_bid:,.2f} / ask {row.last_ask:,.2f}, spread {row.spread_pct * 100:.1f}%)")
        print(f"     RISK    ${row.position_cost:,.0f} total, and that is the maximum loss")
        print(f"     EXIT    sell to close at ${row.take_profit_at:,.2f} (+{PROFIT_TARGET:.0%})   no stop   time stop {MAX_HOLD_SESSIONS} sessions")
        print()

    print(f"total risk if all taken: ${picks.position_cost.sum():,.0f} across {len(picks)} positions")

    if not below_floor.empty:
        excluded = ", ".join(
            f"{row.ticker} (${row.cost_per_contract:,.0f})" for row in below_floor.itertuples()
        )
        print(f"\n\U0001F534 excluded below the ${MIN_CONTRACT_COST:,.0f} cost floor: {excluded}")
        print("   that bucket returned PF 0.623 in the untouched half -- not tradeable.")

    print("\nCONFIDENCE -- ONE NUMBER, AND IT APPLIES TO THE LANE, NOT TO ANY NAME")
    print(f"  every \U0001F7E2 ticket above carries the SAME evidence: {LANE_WIN_RATE:.1%} win rate,")
    print(f"  profit factor {LANE_PROFIT_FACTOR:.2f}, over {LANE_SAMPLE} backtested trades.")
    print("  \U0001F534 below the $700 floor is excluded: PF 0.62, loses out of sample.")

    print("\nTHERE IS NO BEST TICKET, AND NO WORST -- THIS WAS TESTED")
    print("  Nothing available ranks these against each other:")
    print("    contract cost vs return   spearman -0.031  p=0.744   does not rank")
    print("    momentum rank vs return   spearman -0.066  p=0.486   does not rank")
    print("  So picking one name out of this list is arbitrary. The edge is a property")
    print("  of the BASKET, not of any ticket in it. Take them all at equal size, or")
    print("  take none. If capital only allows a few, choose at random or by whatever")
    print("  external view you hold -- but do not imagine the list is ordered.")

    print("\nWHAT THIS LANE IS NOT")
    print("  It is not profitable every month -- 4 of 6. It lost outright in March")
    print("  (PF 0.41) and April (PF 0.53), both months Technology rallied. The whole")
    print("  basket tends to share one outcome on any given entry date (52% of dates),")
    print("  so this is closer to ONE position than to seven. Out of sample the dollar")
    print("  P&L is not statistically significant (p=0.31). Treat it as a small")
    print("  satellite, not a book, and do not fund it with capital you need.")
    print("\nWHAT IS TESTED, AND WHAT I SIMPLY CHOSE")
    print("  TESTED   Technology only; puts on weak momentum; +50% target; no stop")
    print("  MIXED    $700 cost floor -- significant in TEST (p=0.014) and pooled")
    print("           (p=0.021), but NOT distinguishable within TRAIN (p=0.164)")
    print("  ASSUMED  bottom 20% quintile, 0.95x strike, 80 DTE, 40-session stop,")
    print("           12% spread cap, OI >= 50. These were chosen, never swept.")
    print("           They are plausible, not validated. Treat them as defaults.")

    print("\nmanagement rules, exactly as validated:")
    print(f"  take profit  close the whole position at +{PROFIT_TARGET:.0%} on the ask paid")
    print("  stop loss    NONE. Adding a 50% stop cut test PF from 7.63 to 1.39")
    print(f"  time stop    close after {MAX_HOLD_SESSIONS} sessions regardless")
    print("  per name     one open position per ticker, no exceptions")
    print("\nevidence, against a null that fixes date/sector/type/count and shuffles")
    print("only which names are picked (200 permutations):")
    print("          TRAIN  PF 1.66 vs null 0.69  p=0.0000   mean p=0.0000")
    print("          TEST   PF 2.90 vs null 1.04  p=0.0000   mean p=0.0250")
    print("          but OUT OF SAMPLE win rate p=0.20 and dollar P&L p=0.31 are")
    print("          NOT significant. The edge is in size of win, not frequency,")
    print("          and has not been shown to produce a significant dollar result.")
    print("limits:   ~5-12 trades/month; day-clustered p05 1.24 against a 1.20 bar.")
    print("          the return rate is validated, the lane size is not.")


if __name__ == "__main__":
    main()
