"""Buy calls on confirmed trends: the one structure the evidence supports.

Established movers kept moving over ~40 sessions, and options-flow escalation
roughly doubled the momentum-only spread. The long leg ran +6.5% to +8.3% over
40 sessions, which is large enough to pay for a call.

Structure follows the failure of the earlier straddle test: DTE is matched to the
holding period rather than buying a month of premium to capture a week. Entry
pays the ask, exit sells the bid, both from chain-oi quotes so every contract with
open interest can be re-quoted.
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
OUT = ROOT / "out/trend_call_backtest.csv"
SPLIT = "2026-04-14"
HOLD = 40
DTE_BAND = (60, 110)
TARGET_DTE = 80
MAX_SPREAD_PCT = 0.12
CONTRACT_FEE = 1.30
FLOW_ESCALATION_MIN = 1.5
TOP_DECILE = 0.90


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


def pick_calls(quotes: pd.DataFrame, tickers: set[str], moneyness: float) -> pd.DataFrame:
    calls = quotes[
        quotes.ticker.isin(tickers)
        & quotes.option_type.eq("call")
        & quotes.dte.between(*DTE_BAND)
        & (quotes.curr_oi >= 50)
        & (quotes.spread_pct <= MAX_SPREAD_PCT)
    ].copy()
    if calls.empty:
        return calls
    calls["target_strike"] = calls.stock_price * moneyness
    calls["strike_gap"] = (calls.strike - calls.target_strike).abs()
    calls["dte_gap"] = (calls.dte - TARGET_DTE).abs()
    calls = calls.sort_values(["dte_gap", "strike_gap"])
    return calls.groupby("ticker", as_index=False).first()


def main() -> None:
    moneyness = float(sys.argv[1]) if len(sys.argv) > 1 else 1.05
    mode = sys.argv[2] if len(sys.argv) > 2 else "signal"

    columns = ["date", "ticker", "issue_type", "marketcap", "close", "pos_52w", "hc_premium"]
    panel = pd.read_csv(PANEL, usecols=columns, low_memory=False)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel[
        (panel.issue_type == "Common Stock") & (panel.marketcap.fillna(0) >= 2e9)
    ].sort_values(["ticker", "date"])
    grouped = panel.groupby("ticker")
    panel["flow_avg_20"] = grouped.hc_premium.transform(lambda s: s.rolling(20, min_periods=10).mean())
    panel["flow_escalation"] = panel.hc_premium / panel.flow_avg_20.replace(0, np.nan)

    # A bull tape lifts every call. Unless the selection beats an equally sized
    # basket drawn from the same dates and the same liquid universe, the result
    # is beta, not signal.
    rng = np.random.default_rng(20260728)
    picks: dict[str, set[str]] = {}
    for date, day in panel.groupby("date"):
        day = day.dropna(subset=["pos_52w", "flow_escalation"])
        eligible = day[day.flow_escalation >= FLOW_ESCALATION_MIN]
        if len(eligible) < 30:
            continue
        rank = eligible.pos_52w.rank(pct=True)
        chosen = eligible[rank >= TOP_DECILE]
        if chosen.empty:
            continue
        key = date.strftime("%Y-%m-%d")
        if mode == "signal":
            picks[key] = set(chosen.ticker)
        elif mode == "random":
            pool = day.ticker.to_numpy()
            size = min(len(chosen), len(pool))
            picks[key] = set(rng.choice(pool, size=size, replace=False))
        elif mode == "worst":
            picks[key] = set(eligible[rank <= 1.0 - TOP_DECILE].ticker)
        else:
            raise SystemExit(f"unknown mode {mode}")

    days = sorted(p.name for p in ROOT.iterdir() if p.is_dir() and re.fullmatch(r"2026-\d{2}-\d{2}", p.name))
    position = {d: i for i, d in enumerate(days)}
    cache: dict[str, pd.DataFrame] = {}

    def quotes(session: str) -> pd.DataFrame:
        if session not in cache:
            slot = position[session]
            cache[session] = chain_quotes(session, days[slot + 1]) if slot + 1 < len(days) else pd.DataFrame()
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
        calls = pick_calls(entry_quotes, tickers, moneyness)
        if calls.empty:
            continue
        exit_bid = exit_quotes.set_index("option_symbol").last_bid
        calls["exit_bid"] = exit_bid.reindex(calls.option_symbol).to_numpy()
        calls["signal_date"] = signal_date
        calls["entry_date"] = entry_date
        calls["exit_date"] = exit_date
        calls["entry_ask"] = calls.last_ask
        calls["tradeable"] = calls.exit_bid.notna()
        calls["pnl"] = (calls.exit_bid.fillna(0) - calls.entry_ask) * 100.0 - CONTRACT_FEE
        calls["return_on_premium"] = calls.pnl / (calls.entry_ask * 100.0)
        rows.append(calls)
        print(f"[trend-calls] {signal_date} -> {len(calls)} calls", flush=True)

    if not rows:
        raise SystemExit("no trades built")
    result = pd.concat(rows, ignore_index=True)
    result.to_csv(OUT, index=False)
    scored = result[result.tradeable].copy()
    scored["sample"] = np.where(scored.signal_date >= SPLIT, "TEST", "TRAIN")

    print(f"\n=== TREND CALLS [{mode}], {moneyness:.2f}x strike, {TARGET_DTE}d DTE, {HOLD}-session hold ===")
    print(f"built {len(result)}  tradeable {len(scored)}  days {result.signal_date.nunique()}")
    for sample in ("TRAIN", "TEST"):
        frame = scored[scored["sample"] == sample]
        if len(frame) < 10:
            continue
        gains = frame.pnl[frame.pnl > 0].sum()
        losses = -frame.pnl[frame.pnl < 0].sum()
        print(
            f"\n{sample}: n={len(frame)}  mean={frame.return_on_premium.mean():+.4f}  "
            f"median={frame.return_on_premium.median():+.4f}  win={frame.pnl.gt(0).mean():.3f}  "
            f"PF={gains / losses if losses else float('nan'):.3f}  "
            f"avg_cost=${frame.entry_ask.mean()*100:,.0f}  avg_pnl=${frame.pnl.mean():+,.0f}"
        )
        print(f"  >=+100%: {frame.return_on_premium.ge(1.0).mean():.3f}   total P&L 1 lot: ${frame.pnl.sum():+,.0f}")
        month = frame.copy()
        month["month"] = month.signal_date.str[:7]
        print(
            month.groupby("month")
            .agg(n=("pnl", "size"), mean_ret=("return_on_premium", "mean"), pnl=("pnl", "sum"))
            .round(4)
            .to_string()
        )
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
