"""Apply the validated management wrapper to any candidate list.

The measured lesson from this whole exercise is that the management rules carried
far more of the result than the selection did. Random selection with these rules
beat sophisticated selection without them, repeatedly. So the rules are worth
separating from the signal and pointing at whatever candidates you actually
believe in -- your own research, a news catalyst, or the one validated lane.

What is validated, and what is not, is labelled per row. Candidates whose source
has not been permutation-tested are marked UNVALIDATED. That is not a reason to
avoid the trade; it is a reason not to quote the lane's statistics for it.

Usage
    # from the one validated source
    python3 managed_trade_wrapper.py --source short-momentum --as-of 2026-07-23

    # from your own list
    python3 managed_trade_wrapper.py --tickers NVDA,AMD --direction put
    python3 managed_trade_wrapper.py --tickers MU --direction call --risk 2000
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

# Every one of these came out of a test in this repo, not from convention.
PROFIT_TARGET = 0.50          # +50% beat every other target tried
STOP_LOSS = None              # a 50% stop cut test PF from 7.63 to 1.39
MAX_HOLD_SESSIONS = 40        # horizon the momentum signal was measured on
TARGET_DTE = 80               # DTE matched to the hold, not 30d bought for 5d
DTE_BAND = (60, 110)
MIN_CONTRACT_COST = 700.0     # cheap contracts: TEST PF 0.623, no edge
MAX_SPREAD_PCT = 0.12
MIN_OPEN_INTEREST = 50
CALL_MONEYNESS = 1.05
PUT_MONEYNESS = 0.95

VALIDATED_SOURCE = {
    "sector": "Technology",
    "direction": "put",
    "evidence": "TRAIN PF 1.72 / TEST PF 4.25 vs random control 0.64 / 0.52",
}


def sessions() -> list[str]:
    return sorted(p.name for p in ROOT.iterdir() if p.is_dir() and re.fullmatch(r"\d{4}-\d{2}-\d{2}", p.name))


def chain_for(session: str, days: list[str]) -> pd.DataFrame:
    index = days.index(session)
    if index + 1 >= len(days):
        raise SystemExit(f"{session} is the newest folder; quotes come from the following session's file")
    path = find(ROOT / days[index + 1], "chain-oi-changes")
    if path is None:
        raise SystemExit(f"no chain-oi-changes file dated {days[index + 1]}")
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


def short_momentum_candidates(session: str) -> pd.DataFrame:
    panel = pd.read_csv(
        PANEL,
        usecols=["date", "ticker", "sector", "issue_type", "marketcap", "pos_52w"],
        low_memory=False,
    )
    panel["date"] = pd.to_datetime(panel["date"]).dt.strftime("%Y-%m-%d")
    day = panel[
        (panel.date == session)
        & (panel.issue_type == "Common Stock")
        & (panel.marketcap.fillna(0) >= 2e9)
        & (panel.sector == VALIDATED_SOURCE["sector"])
        & panel.pos_52w.notna()
    ].copy()
    if day.empty:
        return day
    day["momentum_rank"] = day.pos_52w.rank(pct=True)
    weak = day[day.momentum_rank <= 0.20].copy()
    weak["direction"] = "put"
    weak["provenance"] = "VALIDATED"
    return weak[["ticker", "direction", "provenance", "pos_52w", "momentum_rank"]]


def build_tickets(candidates: pd.DataFrame, chain: pd.DataFrame, risk: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for candidate in candidates.itertuples():
        option_type = "call" if candidate.direction == "call" else "put"
        moneyness = CALL_MONEYNESS if option_type == "call" else PUT_MONEYNESS
        legs = chain[
            chain.ticker.eq(candidate.ticker)
            & chain.option_type.eq(option_type)
            & chain.dte.between(*DTE_BAND)
            & (chain.curr_oi >= MIN_OPEN_INTEREST)
            & (chain.spread_pct <= MAX_SPREAD_PCT)
        ].copy()
        if legs.empty:
            rows.append({"ticker": candidate.ticker, "direction": option_type, "reject": "no liquid contract"})
            continue
        legs["strike_gap"] = (legs.strike - legs.stock_price * moneyness).abs()
        legs["dte_gap"] = (legs.dte - TARGET_DTE).abs()
        pick = legs.sort_values(["dte_gap", "strike_gap"]).iloc[0]
        cost = pick.last_ask * 100.0
        if cost < MIN_CONTRACT_COST:
            rows.append(
                {"ticker": candidate.ticker, "direction": option_type, "reject": f"below ${MIN_CONTRACT_COST:,.0f} cost floor (${cost:,.0f})"}
            )
            continue
        # One contract is the smallest tradeable unit, so a name whose cheapest
        # qualifying contract already exceeds the risk budget cannot be taken at
        # this size. Sizing to a single lot anyway would silently blow the budget
        # -- on high-priced names that is a 10-20x overshoot.
        if cost > risk:
            rows.append(
                {
                    "ticker": candidate.ticker,
                    "direction": option_type,
                    "reject": f"1 contract costs ${cost:,.0f}, over the ${risk:,.0f} budget",
                }
            )
            continue
        contracts = max(1, int(risk // cost))
        rows.append(
            {
                "ticker": candidate.ticker,
                "direction": option_type,
                "provenance": getattr(candidate, "provenance", "UNVALIDATED"),
                "option_symbol": pick.option_symbol,
                "stock_price": round(pick.stock_price, 2),
                "strike": pick.strike,
                "moneyness": round(pick.strike / pick.stock_price, 3),
                "dte": int(pick.dte),
                "bid": pick.last_bid,
                "ask": pick.last_ask,
                "spread_pct": round(pick.spread_pct, 3),
                "open_interest": int(pick.curr_oi),
                "cost_per_contract": round(cost, 0),
                "contracts": contracts,
                "position_cost": round(contracts * cost, 0),
                "take_profit_at": round(pick.last_ask * (1 + PROFIT_TARGET), 2),
                "reject": "",
            }
        )
    frame = pd.DataFrame(rows)
    accepted = frame[frame.reject.eq("")].drop(columns=["reject"]) if "reject" in frame else frame
    rejected = frame[frame.reject.ne("")] if "reject" in frame else pd.DataFrame()
    return accepted, rejected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source", choices=["short-momentum"], default=None)
    parser.add_argument("--tickers", default="", help="Comma-separated tickers for a manual list.")
    parser.add_argument("--direction", choices=["call", "put"], default="put")
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--risk", type=float, default=1500.0)
    args = parser.parse_args()

    days = sessions()
    session = args.as_of or days[-1]
    if session not in days:
        raise SystemExit(f"{session} is not a dated folder under {ROOT}")

    if args.source == "short-momentum":
        candidates = short_momentum_candidates(session)
        heading = f"VALIDATED LANE -- {VALIDATED_SOURCE['sector']} short momentum puts"
    elif args.tickers:
        names = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        candidates = pd.DataFrame(
            {"ticker": names, "direction": args.direction, "provenance": "UNVALIDATED"}
        )
        heading = f"MANUAL LIST -- {len(names)} name(s), {args.direction}s"
    else:
        raise SystemExit("provide --source short-momentum or --tickers")

    if candidates.empty:
        print(f"no candidates for {session}")
        return

    chain = chain_for(session, days)
    accepted, rejected = build_tickets(candidates, chain, args.risk)

    print(f"=== {heading} ===")
    print(f"signal session {session}, entry next session, ${args.risk:,.0f} risk per trade\n")
    if accepted.empty:
        print("no candidate produced a tradeable contract")
    else:
        columns = [
            "ticker", "direction", "provenance", "stock_price", "strike", "moneyness",
            "dte", "bid", "ask", "spread_pct", "cost_per_contract", "contracts",
            "position_cost", "take_profit_at",
        ]
        print(accepted[[c for c in columns if c in accepted]].to_string(index=False))
        print(f"\ntotal deployed: ${accepted.position_cost.sum():,.0f} across {len(accepted)} positions")

    if not rejected.empty:
        print("\nrejected:")
        for row in rejected.itertuples():
            print(f"  {row.ticker:<6} {row.reject}")

    print("\nMANAGEMENT RULES -- these are what was actually validated:")
    print(f"  take profit   close at +{PROFIT_TARGET:.0%} on the ask paid")
    print("  stop loss     NONE -- a 50% stop cut test PF from 7.63 to 1.39")
    print(f"  time stop     close after {MAX_HOLD_SESSIONS} sessions regardless")
    print("  per name      one open position per ticker")
    print(f"  cost floor    skip contracts under ${MIN_CONTRACT_COST:,.0f} (TEST PF 0.623 below it)")

    if (accepted.get("provenance") == "UNVALIDATED").any() if "provenance" in accepted else False:
        print("\nNOTE: rows marked UNVALIDATED come from a selection this repo has not")
        print("      permutation-tested. The management rules above still apply, but the")
        print("      lane statistics do NOT transfer to them. Size accordingly.")


if __name__ == "__main__":
    main()
