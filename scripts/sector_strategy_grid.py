"""Full grid: every sector x every historically constructible strategy x controls.

Testing one lane at a time was the wrong approach. This ranks names WITHIN each
sector, so a sector's own leaders and laggards are compared against each other
rather than against a tech-dominated cross-section, then evaluates every
structure on the same names and the same dates.

The canonical structure list lives in strategy_universe.py. It covers all 32
registered families. Stock-backed strategies include underlying P/L, calendars
and diagonals use distinct expiries, and undefined-risk structures use exact
observed P/L with a conservative Reg-T risk-capital proxy.

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
from strategy_universe import (  # noqa: E402
    HISTORICAL_STRATEGY_SPECS,
    build_sector_state,
    build_structure,
    liquidate_structure,
    selection_buckets,
)

PANEL = ROOT / "out/uw_all_feeds.csv"
OUT = ROOT / "out/sector_strategy_grid_v3.csv"
UNIVERSE_OUT = ROOT / "out/sector_strategy_universe_v3.csv"
SPLIT = "2026-04-14"
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


def build_matched_grid(universe: pd.DataFrame, *, seed: int = 20260728) -> pd.DataFrame:
    if universe is None or universe.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    signals = universe[universe["signal_selected"].astype(bool)].copy()
    signals["mode"] = "signal"
    controls = []
    group_columns = ["signal_date", "sector", "strategy"]
    for keys, pool in universe.groupby(group_columns, sort=True, observed=True):
        signal_count = int(pool["signal_selected"].astype(bool).sum())
        if signal_count <= 0:
            continue
        sample_size = min(signal_count, len(pool))
        chosen = rng.choice(pool.index.to_numpy(), size=sample_size, replace=False)
        control = pool.loc[chosen].copy()
        control["mode"] = "random"
        controls.append(control)
    pieces = [signals] + controls
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def main() -> None:
    columns = ["date", "ticker", "sector", "issue_type", "marketcap", "close", "pos_52w", "ret_1d", "hc_premium"]
    panel = pd.read_csv(PANEL, usecols=columns, low_memory=False)
    panel["date"] = pd.to_datetime(panel["date"]).dt.strftime("%Y-%m-%d")
    panel = panel[
        (panel.issue_type == "Common Stock")
        & (panel.marketcap.fillna(0) >= 2e9)
        & panel.sector.notna()
    ].sort_values(["ticker", "date"])
    grouped = panel.groupby("ticker")
    panel["flow_avg_20"] = grouped.hc_premium.transform(lambda s: s.rolling(20, min_periods=10).mean())
    panel["flow_escalation"] = panel.hc_premium / panel.flow_avg_20.replace(0, np.nan)
    sector_state = build_sector_state(panel)

    days = sorted(p.name for p in ROOT.iterdir() if p.is_dir() and re.fullmatch(r"2026-\d{2}-\d{2}", p.name))
    position = {d: i for i, d in enumerate(days)}
    cache: dict[str, pd.DataFrame] = {}
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
    min_hold = min(spec.hold_days for spec in HISTORICAL_STRATEGY_SPECS)
    for signal_date in days:
        slot = position[signal_date]
        if slot + 1 + min_hold >= len(days):
            continue
        day = panel[panel.date == signal_date].dropna(subset=["pos_52w", "flow_escalation"])
        if day.empty:
            continue
        entry_date = days[slot + 1]
        entry_quotes = quotes(entry_date)
        if entry_quotes.empty:
            continue
        print(f"[grid] {signal_date}", flush=True)

        selection_rows: list[dict[str, object]] = []
        for sector, block in day.groupby("sector"):
            if len(block) < MIN_PER_SECTOR:
                continue
            buckets = selection_buckets(block, percentile=DECILE)
            for spec in HISTORICAL_STRATEGY_SPECS:
                chosen = buckets[spec.selection_bucket]
                if not chosen:
                    continue
                selection_rows.extend(
                    {
                        "signal_date": signal_date,
                        "sector": sector,
                        "strategy": spec.key,
                        "ticker": ticker,
                        "signal_selected": ticker in chosen,
                    }
                    for ticker in block["ticker"].astype(str)
                )

        if not selection_rows:
            continue
        selections = pd.DataFrame(selection_rows)
        earnings_for_day = None
        if "next_earnings_date" in day.columns:
            dated = day.dropna(subset=["next_earnings_date"]).drop_duplicates("ticker")
            earnings_for_day = pd.Series(
                pd.to_datetime(dated["next_earnings_date"], errors="coerce").values,
                index=dated["ticker"].astype(str).values,
            )
        for spec in HISTORICAL_STRATEGY_SPECS:
            selected = selections[selections["strategy"].eq(spec.key)]
            if selected.empty:
                continue
            # Each family is gated on its own horizon so a long-hold spec cannot
            # truncate the usable date range of a short-hold one.
            exit_slot = slot + 1 + spec.hold_days
            if exit_slot >= len(days):
                continue
            exit_quotes = quotes(days[exit_slot])
            if exit_quotes.empty:
                continue
            structures = build_structure(
                entry_quotes,
                selected["ticker"].unique(),
                spec,
                min_open_interest=50,
                max_spread_pct=MAX_SPREAD_PCT,
                earnings_by_ticker=earnings_for_day,
            )
            outcomes = liquidate_structure(
                structures,
                exit_quotes,
                spec,
                contract_fee=CONTRACT_FEE,
            )
            if outcomes.empty:
                continue
            keep = [
                "ticker",
                "expiry",
                "entry_cashflow",
                "exit_cashflow",
                "far_expiry",
                "historical_scope",
                "risk_capital_model",
                "max_risk",
                "pnl",
                "return_on_risk",
            ] + [f"symbol_{index}" for index in range(len(spec.legs))]
            result = selected.merge(outcomes[keep], on="ticker", how="inner")
            if result.empty:
                continue
            state_day = sector_state[sector_state["date"].eq(signal_date)].drop(columns=["date"])
            result = result.merge(state_day, on="sector", how="left")
            result["cost"] = result["max_risk"]
            result["return_on_cost"] = result["return_on_risk"]
            records.append(result)

    if not records:
        raise SystemExit("no trades built")
    universe = pd.concat(records, ignore_index=True)
    universe["sample"] = np.where(universe.signal_date >= SPLIT, "TEST", "TRAIN")
    universe.to_csv(UNIVERSE_OUT, index=False)
    trades = build_matched_grid(universe)
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

    print(
        f"\nbuilt {len(universe)} constructible outcomes and {len(trades)} matched-grid rows "
        f"across {trades.signal_date.nunique()} dates"
    )
    for strategy in [spec.key for spec in HISTORICAL_STRATEGY_SPECS]:
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
    print(f"wrote {UNIVERSE_OUT}")


if __name__ == "__main__":
    main()
