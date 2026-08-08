"""Executable option test for the buyer-to-open direction hypothesis.

The chain-OI file dated t confirms which positions opened during t-1 and carries
t-1 ask/bid/multileg volume. Selection is therefore made on t and entered on the
next quoted session. The primary feature is predeclared from the buyer-to-open
literature; no feature search happens here.

Each null draw preserves signal date, direction, sector, and number of names.
The final 40 sessions are excluded from signal generation so unresolved losses
cannot disappear through right censoring.
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import managed_exit_backtest as base  # noqa: E402

TRADES_OUT = base.ROOT / "out/opening_flow_option_trades.csv"
NULL_OUT = base.ROOT / "out/opening_flow_option_permutations.csv"
FEATURE = "oi_dir_bias"
PICKS_PER_DAY = 1
PROFIT_TARGET = 0.50
PERMUTATIONS = int(sys.argv[1]) if len(sys.argv) > 1 else 100
RNG = np.random.default_rng(20260729)


def profit_factor(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    gains = array[array > 0].sum()
    losses = -array[array < 0].sum()
    return gains / losses if losses > 0 else np.inf if gains > 0 else np.nan


def stats(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"n": 0, "dates": 0, "mean": np.nan, "win": np.nan, "pf": np.nan, "pnl": 0.0}
    return {
        "n": len(frame),
        "dates": frame.signal_date.nunique(),
        "mean": frame.return_on_cost.mean(),
        "win": frame.pnl.gt(0).mean(),
        "pf": profit_factor(frame.pnl),
        "pnl": frame.pnl.sum(),
    }


def select_names(
    day: pd.DataFrame,
    direction: str,
    held: set[str],
    executable: set[str],
    randomize: bool,
    rng: np.random.Generator,
) -> set[str]:
    available = day[
        ~day.ticker.isin(held) & day.ticker.isin(executable)
    ].dropna(subset=[FEATURE])
    if available.empty:
        return set()
    ascending = direction == "long_put"
    signal = available.sort_values(FEATURE, ascending=ascending).head(PICKS_PER_DAY)
    if not randomize:
        return set(signal.ticker)

    chosen: set[str] = set()
    sector_counts = Counter(signal.sector.astype(str))
    for sector, count in sector_counts.items():
        pool = available.loc[available.sector.astype(str).eq(sector), "ticker"].to_numpy()
        if len(pool):
            chosen.update(rng.choice(pool, size=min(count, len(pool)), replace=False))
    return chosen


def simulate(
    panel_by_day: dict[str, pd.DataFrame],
    sessions: list[str],
    quote_for,
    direction: str,
    randomize: bool,
    rng: np.random.Generator,
) -> pd.DataFrame:
    option_type = "call" if direction == "long_call" else "put"
    moneyness = 1.05 if direction == "long_call" else 0.95
    last_signal_index = len(sessions) - base.MAX_HOLD - 1

    open_positions: list[dict] = []
    closed: list[dict] = []
    held: set[str] = set()

    for index, session in enumerate(sessions):
        quotes = quote_for(session)
        if quotes.empty:
            continue
        bid = quotes.set_index("option_symbol").last_bid

        still_open = []
        for position in open_positions:
            current = bid.get(position["symbol"])
            age = index - position["entry_index"]
            if current is None or not np.isfinite(current):
                if age < base.MAX_HOLD:
                    still_open.append(position)
                    continue
                current = position["last_mark"]
            else:
                position["last_mark"] = current
            gain = current / position["cost"] - 1.0
            if gain >= PROFIT_TARGET:
                reason = "profit_target"
            elif age >= base.MAX_HOLD:
                reason = "time_stop"
            else:
                still_open.append(position)
                continue
            pnl = (current - position["cost"]) * 100.0 - base.CONTRACT_FEE
            closed.append(
                {
                    "signal_date": position["signal_date"],
                    "ticker": position["ticker"],
                    "sector": position["sector"],
                    "direction": direction,
                    "mode": "random" if randomize else "signal",
                    "exit_date": session,
                    "exit_reason": reason,
                    "held": age,
                    "cost": position["cost"] * 100.0,
                    "pnl": pnl,
                    "return_on_cost": pnl / (position["cost"] * 100.0),
                }
            )
            held.discard(position["ticker"])
        open_positions = still_open

        if index >= last_signal_index or index + 1 >= len(sessions):
            continue
        day = panel_by_day.get(session)
        if day is None or day.empty:
            continue
        entry_quotes = quote_for(sessions[index + 1])
        if entry_quotes.empty:
            continue
        legs = entry_quotes[
            entry_quotes.option_type.eq(option_type)
            & entry_quotes.dte.between(*base.DTE_BAND)
            & entry_quotes.curr_oi.ge(50)
            & entry_quotes.spread_pct.le(base.MAX_SPREAD_PCT)
        ].copy()
        if legs.empty:
            continue
        legs["strike_gap"] = (legs.strike - legs.stock_price * moneyness).abs()
        legs["dte_gap"] = (legs.dte - base.TARGET_DTE).abs()
        legs = legs.sort_values(["dte_gap", "strike_gap"]).groupby("ticker", as_index=False).first()
        chosen = select_names(day, direction, held, set(legs.ticker), randomize, rng)
        if not chosen:
            continue
        legs = legs[legs.ticker.isin(chosen)]
        sector_by_ticker = day.set_index("ticker").sector
        for row in legs.itertuples():
            open_positions.append(
                {
                    "signal_date": session,
                    "ticker": row.ticker,
                    "sector": str(sector_by_ticker.get(row.ticker, "")),
                    "entry_index": index + 1,
                    "cost": row.last_ask,
                    "symbol": row.option_symbol,
                    "last_mark": row.last_bid,
                }
            )
            held.add(row.ticker)

    return pd.DataFrame(closed)


def clustered_pf_p05(frame: pd.DataFrame, iterations: int = 4000) -> tuple[float, float]:
    by_date = {date: group.pnl.to_numpy() for date, group in frame.groupby("signal_date")}
    dates = np.array(list(by_date))
    if not len(dates):
        return np.nan, np.nan
    values = []
    for _ in range(iterations):
        selected = RNG.choice(dates, size=len(dates), replace=True)
        value = profit_factor(np.concatenate([by_date[date] for date in selected]))
        if np.isfinite(value):
            values.append(value)
    return tuple(np.percentile(values, [5, 50])) if values else (np.nan, np.nan)


def main() -> None:
    panel = pd.read_csv(
        base.PANEL,
        usecols=["date", "ticker", "sector", "issue_type", "marketcap", FEATURE],
        low_memory=False,
    )
    panel["date"] = pd.to_datetime(panel.date).dt.strftime("%Y-%m-%d")
    panel = panel[
        panel.issue_type.eq("Common Stock")
        & panel.marketcap.fillna(0).ge(2e9)
        & panel.sector.notna()
        & panel.ticker.notna()
        & panel.ticker.ne("NAN")
    ].copy()
    if panel.duplicated(["date", "ticker"]).any():
        raise ValueError("duplicate ticker/date rows in eligible stock universe")
    panel_by_day = {date: day for date, day in panel.groupby("date", sort=False)}
    sessions = sorted(panel.date.unique())

    all_folders = sorted(
        path.name
        for path in base.ROOT.iterdir()
        if path.is_dir() and re.fullmatch(r"2026-\d{2}-\d{2}", path.name)
    )
    folder_position = {date: index for index, date in enumerate(all_folders)}
    quote_cache: dict[str, pd.DataFrame] = {}

    def quote_for(session: str) -> pd.DataFrame:
        if session not in quote_cache:
            slot = folder_position.get(session)
            quote_cache[session] = (
                base.chain_quotes(session, all_folders[slot + 1])
                if slot is not None and slot + 1 < len(all_folders)
                else pd.DataFrame()
            )
        return quote_cache[session]

    print("[opening-option] warming quote cache", flush=True)
    for session in sessions:
        quote_for(session)

    actual_frames = []
    null_rows = []
    for direction in ("long_call", "long_put"):
        actual = simulate(panel_by_day, sessions, quote_for, direction, False, RNG)
        if not actual.empty:
            actual_frames.append(actual)
        for sample in ("TRAIN", "TEST"):
            observed = stats(actual[actual.signal_date.ge(base.SPLIT) == (sample == "TEST")])
            print(f"[opening-option] {direction} {sample}: {observed}", flush=True)

        for trial in range(PERMUTATIONS):
            random = simulate(panel_by_day, sessions, quote_for, direction, True, RNG)
            for sample in ("TRAIN", "TEST"):
                record = stats(random[random.signal_date.ge(base.SPLIT) == (sample == "TEST")])
                record.update(direction=direction, sample=sample, trial=trial)
                null_rows.append(record)
            if (trial + 1) % 25 == 0:
                print(f"[opening-option] {direction} permutations {trial + 1}/{PERMUTATIONS}", flush=True)

    trades = pd.concat(actual_frames, ignore_index=True) if actual_frames else pd.DataFrame()
    null = pd.DataFrame(null_rows)
    trades.to_csv(TRADES_OUT, index=False)
    null.to_csv(NULL_OUT, index=False)

    print("\n=== BUYER-TO-OPEN OPTIONS: ACTUAL VS MATCHED NULL ===")
    for direction in ("long_call", "long_put"):
        for sample in ("TRAIN", "TEST"):
            actual = trades[(trades.direction == direction) & (trades.signal_date.ge(base.SPLIT) == (sample == "TEST"))]
            observed = stats(actual)
            block = null[(null.direction == direction) & (null["sample"] == sample)]
            print(f"\n{direction} {sample} n={observed['n']} dates={observed['dates']}")
            for metric in ("mean", "win", "pf", "pnl"):
                distribution = block[metric].replace([np.inf, -np.inf], np.nan).dropna()
                value = observed[metric]
                p_value = (distribution >= value).mean() if len(distribution) and np.isfinite(value) else np.nan
                median = distribution.median() if len(distribution) else np.nan
                print(f"  {metric:4s} actual={value:>10.3f} null_median={median:>10.3f} p={p_value:.4f}")
            p05, p50 = clustered_pf_p05(actual)
            print(f"  clustered PF p05={p05:.2f} median={p50:.2f}")

    if not trades.empty:
        trades["month"] = trades.signal_date.str[:7]
        monthly = trades.groupby(["direction", "month"]).pnl.agg(n="size", pnl="sum")
        monthly["pf"] = trades.groupby(["direction", "month"]).pnl.apply(profit_factor)
        print("\n=== MONTHLY ===")
        print(monthly.to_string(float_format=lambda value: f"{value:.2f}"))
    print(f"\nwrote {TRADES_OUT} and {NULL_OUT}")


if __name__ == "__main__":
    main()
