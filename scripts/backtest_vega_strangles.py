"""Direct customer-vega-demand -> long-strangle backtest."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from uwos.options_pattern_pipeline_v1 import core

ROOT = Path("/Users/anuppamvi/uw_root/tradedesk")
PANEL = ROOT / "out/uw_all_feeds.csv"
OUT = ROOT / "out/vega_strangle_backtest.csv"
SPLIT = "2026-06-05"
HORIZONS = (1, 5)


def build_entry_ranks() -> tuple[list[str], dict[str, dict[str, float]]]:
    panel = pd.read_csv(
        PANEL,
        usecols=[
            "date", "ticker", "issue_type", "marketcap",
            "tape_vega_flow", "tape_gross_premium",
        ],
        low_memory=False,
    )
    panel = panel[
        (panel.issue_type == "Common Stock")
        & (panel.marketcap.fillna(0) >= 2e9)
        & panel.tape_vega_flow.notna()
        & (panel.tape_gross_premium > 0)
    ].copy()
    panel["vega_demand"] = panel.tape_vega_flow / panel.tape_gross_premium
    panel["rank"] = panel.groupby("date")["vega_demand"].rank(pct=True)
    dates = sorted(pd.read_csv(PANEL, usecols=["date"])["date"].unique())
    next_date = {dates[index]: dates[index + 1] for index in range(len(dates) - 1)}
    panel["entry_date"] = panel.date.map(next_date)
    ranks: dict[str, dict[str, float]] = defaultdict(dict)
    for row in panel.dropna(subset=["entry_date"]).itertuples():
        ranks[row.entry_date][row.ticker] = float(row.rank)
    return dates, dict(ranks)


def cohort(rank: float) -> str:
    if rank >= 0.9:
        return "high_vega_demand"
    if rank <= 0.1:
        return "low_vega_demand"
    if 0.45 <= rank <= 0.55:
        return "middle_control"
    return "other"


def score_exit(trade: dict, snapshot: core.Snapshot, risk_config: dict) -> dict:
    quotes = []
    for leg in trade["legs"]:
        quote = snapshot.option_quotes.get(leg["option_symbol"])
        if not quote or quote.get("bid", 0.0) < 0:
            result = dict(trade)
            result.update(status="UNSCORABLE", outcome_note="future_strangle_leg_quotes_missing")
            return result
        quotes.append(quote)
    exit_value = sum(max(0.0, quote["bid"]) for quote in quotes)
    exit_slippage = sum(
        core.bid_ask_slippage_dollars(quote.get("bid"), quote.get("ask"), risk_config)
        for quote in quotes
    )
    round_trip_fees = core.configured_round_trip_spread_fees(risk_config)
    net_dollars = (
        (exit_value - trade["entry_debit"]) * 100.0
        - round_trip_fees
        - trade["entry_slippage"]
        - exit_slippage
    )
    result = dict(trade)
    result.update(
        status="SCORED",
        exit_value=exit_value,
        exit_slippage=exit_slippage,
        round_trip_fees=round_trip_fees,
        net_dollars=net_dollars,
        net_r=net_dollars / trade["max_risk"],
        win=int(net_dollars > 0),
        outcome_note="both_legs_entry_ask_exit_bid_after_costs_slippage",
    )
    return result


def profit_factor(values: pd.Series) -> float:
    gains = values[values > 0].sum()
    losses = -values[values < 0].sum()
    return gains / losses if losses > 0 else np.nan


def main() -> None:
    all_dates, ranks_by_entry = build_entry_ranks()
    positions = {date: index for index, date in enumerate(all_dates)}
    first_entry = min(ranks_by_entry)
    run_dates = [date for date in all_dates if date >= first_entry]

    args = core.parse_args(["--base-dir", str(ROOT), "--as-of", all_dates[-1], "--no-validation"])
    cache_dir = ROOT / "out/options_pattern_pipeline_v1/cache/bot_eod"
    config = core.base_run_config(args, ROOT, all_dates[-1], cache_dir)
    risk_config = config["risk_config"]

    pending: dict[str, list[dict]] = defaultdict(list)
    outcomes = []
    for index, date in enumerate(run_dates, 1):
        if index == 1 or index % 10 == 0 or index == len(run_dates):
            print(f"[vega-strangle] snapshot {index}/{len(run_dates)} {date}", flush=True)
        snapshot = core.build_daily_snapshot(ROOT, date, config)
        for trade in pending.pop(date, []):
            outcomes.append(score_exit(trade, snapshot, risk_config))

        ranks = ranks_by_entry.get(date)
        if not ranks:
            continue
        strangles = core.select_best_long_strangles(snapshot.option_quotes, risk_config)
        for (ticker, _), strangle in strangles.items():
            rank = ranks.get(ticker)
            if rank is None:
                continue
            group = cohort(rank)
            if group == "other":
                continue
            entry_slippage = core.signal_entry_slippage_dollars(strangle, risk_config)
            for horizon in HORIZONS:
                target_position = positions[date] + horizon
                if target_position >= len(all_dates):
                    continue
                trade = {
                    "signal_date": all_dates[positions[date] - 1],
                    "entry_date": date,
                    "target_date": all_dates[target_position],
                    "sample": "TEST" if all_dates[positions[date] - 1] >= SPLIT else "TRAIN",
                    "horizon": f"{horizon}d",
                    "ticker": ticker,
                    "cohort": group,
                    "vega_rank": rank,
                    "expiry": strangle["expiry"],
                    "dte": strangle["dte"],
                    "entry_debit": strangle["ask"],
                    "max_risk": strangle["max_risk"],
                    "entry_slippage": entry_slippage,
                    "quote_spread": strangle["quote_spread"],
                    "legs": strangle["legs"],
                }
                pending[trade["target_date"]].append(trade)

    for trades in pending.values():
        for trade in trades:
            result = dict(trade)
            result.update(status="UNSCORABLE", outcome_note="not_enough_future_dates")
            outcomes.append(result)

    result = pd.DataFrame(outcomes)
    result.drop(columns=["legs"], errors="ignore").to_csv(OUT, index=False)
    scored = result[result.status == "SCORED"].copy()
    summary = scored.groupby(["sample", "horizon", "cohort"]).agg(
        trades=("net_r", "size"),
        days=("entry_date", "nunique"),
        avg_r=("net_r", "mean"),
        win_rate=("win", "mean"),
        sum_r=("net_r", "sum"),
        profit_factor=("net_r", profit_factor),
    )
    print(f"[vega-strangle] outcomes={len(result)} scored={len(scored)}")
    print("\n=== HONEST VEGA-STRANGLE OUTCOMES ===")
    print(summary.round(4).to_string())
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
