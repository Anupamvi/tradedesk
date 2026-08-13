"""Direct OI-direction -> defined-risk credit-spread backtest.

Signal is measured on date t and entry uses date t+1 option quotes. Outcomes are
marked after 5/10/20 trading sessions and at expiration. The mined
pattern-family layer is not used. Fills match the production scorer: short bid
- long ask at entry, short ask - long bid at marked exits, configured fees, plus
configured spread slippage. Expiration uses intrinsic value without exit spread.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from uwos.options_pattern_pipeline_v1 import core

ROOT = Path("/Users/anuppamvi/uw_root/tradedesk")
PANEL = ROOT / "out/uw_all_feeds.csv"
OUT = ROOT / "out/oi_credit_spread_backtest.csv"
START_SIGNAL = pd.Timestamp("2026-04-14")
HORIZONS = (5, 10, 20)


def build_entry_ranks() -> tuple[list[str], dict[str, dict[str, float]]]:
    panel = pd.read_csv(
        PANEL,
        usecols=["date", "ticker", "issue_type", "marketcap", "oi_dir_bias"],
        low_memory=False,
    )
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel[
        (panel["issue_type"] == "Common Stock")
        & (panel["marketcap"].fillna(0) >= 2e9)
        & panel["oi_dir_bias"].notna()
    ].copy()
    dates = sorted(panel["date"].dt.strftime("%Y-%m-%d").unique())
    next_date = {dates[index]: dates[index + 1] for index in range(len(dates) - 1)}
    panel["rank"] = panel.groupby("date")["oi_dir_bias"].rank(pct=True)
    panel["signal_date_text"] = panel["date"].dt.strftime("%Y-%m-%d")
    panel = panel[panel["date"] >= START_SIGNAL]
    panel["entry_date"] = panel["signal_date_text"].map(next_date)
    ranks: dict[str, dict[str, float]] = defaultdict(dict)
    for row in panel.dropna(subset=["entry_date"]).itertuples():
        ranks[row.entry_date][row.ticker] = float(row.rank)
    return dates, dict(ranks)


def cohort(rank: float, direction: str) -> str:
    if rank >= 0.9 and direction == "bullish":
        return "aligned_top_bullish"
    if rank <= 0.1 and direction == "bearish":
        return "aligned_bottom_bearish"
    if rank >= 0.9 and direction == "bearish":
        return "wrong_top_bearish"
    if rank <= 0.1 and direction == "bullish":
        return "wrong_bottom_bullish"
    if 0.45 <= rank <= 0.55:
        return "middle_control"
    return "other"


def score_exit(trade: dict, snapshot: core.Snapshot, risk_config: dict) -> dict:
    short_leg = next(leg for leg in trade["legs"] if leg["action"] == "SELL")
    long_leg = next(leg for leg in trade["legs"] if leg["action"] == "BUY")
    future_short = snapshot.option_quotes.get(short_leg["option_symbol"])
    future_long = snapshot.option_quotes.get(long_leg["option_symbol"])
    result = dict(trade)
    if not future_short or not future_long or future_short.get("ask", 0.0) <= 0 or future_long.get("bid", 0.0) < 0:
        result.update(status="UNSCORABLE", outcome_note="future_spread_leg_quotes_missing")
        return result
    exit_debit = max(0.0, future_short["ask"] - future_long["bid"])
    exit_slippage = core.credit_spread_exit_slippage_dollars(future_short, future_long, risk_config)
    round_trip_fees = core.configured_round_trip_spread_fees(risk_config)
    net_dollars = (
        (trade["entry_credit"] - exit_debit) * 100.0
        - round_trip_fees
        - trade["entry_slippage"]
        - exit_slippage
    )
    result.update(
        status="SCORED",
        exit_debit=exit_debit,
        exit_slippage=exit_slippage,
        round_trip_fees=round_trip_fees,
        net_dollars=net_dollars,
        net_r=net_dollars / trade["max_risk"],
        win=int(net_dollars > 0),
        outcome_note="short_bid_long_ask_to_short_ask_long_bid_after_costs",
    )
    return result


def score_expiration(trade: dict, snapshot: core.Snapshot, risk_config: dict) -> dict:
    result = dict(trade)
    stock = snapshot.features.get(trade["ticker"], {}).get("close")
    if not stock or stock <= 0:
        result.update(status="UNSCORABLE", outcome_note="expiration_stock_close_missing")
        return result
    short_leg = next(leg for leg in trade["legs"] if leg["action"] == "SELL")
    long_leg = next(leg for leg in trade["legs"] if leg["action"] == "BUY")
    if trade["direction"] == "bullish":
        short_intrinsic = max(short_leg["strike"] - stock, 0.0)
        long_intrinsic = max(long_leg["strike"] - stock, 0.0)
    else:
        short_intrinsic = max(stock - short_leg["strike"], 0.0)
        long_intrinsic = max(stock - long_leg["strike"], 0.0)
    exit_debit = min(trade["spread_width"], max(0.0, short_intrinsic - long_intrinsic))
    round_trip_fees = core.configured_round_trip_spread_fees(risk_config)
    net_dollars = (
        (trade["entry_credit"] - exit_debit) * 100.0
        - round_trip_fees
        - trade["entry_slippage"]
    )
    result.update(
        status="SCORED",
        exit_debit=exit_debit,
        exit_slippage=0.0,
        round_trip_fees=round_trip_fees,
        net_dollars=net_dollars,
        net_r=net_dollars / trade["max_risk"],
        win=int(net_dollars > 0),
        outcome_note="expiration_intrinsic_after_entry_costs",
    )
    return result


def profit_factor(values: pd.Series) -> float:
    gains = values[values > 0].sum()
    losses = -values[values < 0].sum()
    return gains / losses if losses > 0 else np.nan


def main() -> None:
    all_dates, ranks_by_entry = build_entry_ranks()
    first_entry = min(ranks_by_entry)
    last_date = all_dates[-1]
    calendar_position = {date: index for index, date in enumerate(all_dates)}
    run_dates = [date for date in all_dates if first_entry <= date <= last_date]

    args = core.parse_args(["--base-dir", str(ROOT), "--as-of", last_date, "--no-validation"])
    cache_dir = ROOT / "out/options_pattern_pipeline_v1/cache/bot_eod"
    config = core.base_run_config(args, ROOT, last_date, cache_dir)
    risk_config = config["risk_config"]

    pending: dict[str, list[dict]] = defaultdict(list)
    outcomes = []
    entries = 0
    for index, date in enumerate(run_dates, 1):
        if index == 1 or index % 10 == 0 or index == len(run_dates):
            print(f"[oi-credit] snapshot {index}/{len(run_dates)} {date}", flush=True)
        snapshot = core.build_daily_snapshot(ROOT, date, config)

        for trade in pending.pop(date, []):
            if trade["horizon"] == "expiry":
                outcomes.append(score_expiration(trade, snapshot, risk_config))
            else:
                outcomes.append(score_exit(trade, snapshot, risk_config))

        ranks = ranks_by_entry.get(date)
        if not ranks:
            continue
        position = calendar_position[date]
        spreads = core.select_best_vertical_spreads(snapshot.option_quotes, risk_config)
        for (ticker, direction), spread in spreads.items():
            rank = ranks.get(ticker)
            if rank is None:
                continue
            group = cohort(rank, direction)
            if group == "other":
                continue
            entry_slippage = core.signal_entry_slippage_dollars(spread, risk_config)
            base_trade = {
                "signal_date": all_dates[position - 1],
                "entry_date": date,
                "ticker": ticker,
                "direction": direction,
                "cohort": group,
                "oi_rank": rank,
                "strategy_kind": "credit_spread",
                "strategy_type": spread["strategy_type"],
                "expiry": spread["expiry"],
                "dte": spread["dte"],
                "entry_credit": spread["entry_credit"],
                "spread_width": spread["spread_width"],
                "max_risk": spread["max_risk"],
                "entry_slippage": entry_slippage,
                "quote_spread": spread["quote_spread"],
                "legs": spread["legs"],
            }
            entries += 1
            for horizon in HORIZONS:
                if position + horizon >= len(all_dates):
                    continue
                target_date = all_dates[position + horizon]
                if target_date >= spread["expiry"]:
                    continue
                trade = dict(base_trade)
                trade.update(target_date=target_date, horizon=f"{horizon}d")
                pending[target_date].append(trade)

            expiry_dates = [candidate for candidate in all_dates[position + 1 :] if candidate <= spread["expiry"]]
            if expiry_dates and spread["expiry"] <= last_date:
                trade = dict(base_trade)
                trade.update(target_date=expiry_dates[-1], horizon="expiry")
                pending[expiry_dates[-1]].append(trade)

    for trades in pending.values():
        for trade in trades:
            trade = dict(trade)
            trade.update(status="UNSCORABLE", outcome_note="not_enough_future_dates")
            outcomes.append(trade)

    result = pd.DataFrame(outcomes)
    result.drop(columns=["legs"], errors="ignore").to_csv(OUT, index=False)
    print(f"[oi-credit] entries={entries} outcomes={len(result)} scored={(result.status == 'SCORED').sum()}")
    scored = result[result.status == "SCORED"].copy()
    summary = scored.groupby(["horizon", "cohort"]).agg(
        trades=("net_r", "size"),
        days=("entry_date", "nunique"),
        tickers=("ticker", "nunique"),
        avg_r=("net_r", "mean"),
        win_rate=("win", "mean"),
        sum_r=("net_r", "sum"),
        profit_factor=("net_r", profit_factor),
    )
    print("\n=== HONEST MULTI-HORIZON CREDIT-SPREAD OUTCOMES ===")
    print(summary.round(4).to_string())
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
