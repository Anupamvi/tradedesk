"""Translate the fixed bullish shock-reversal signal into actual option P&L.

Research only. This does not modify or feed a live pipeline.

Signal (fixed before option testing):
  * common stock, market cap >= $2B, dollar volume >= $50M
  * one-day return <= -3 average daily moves
  * at least 12% below the trailing 63-session high
  * shock volume >= 1.25x the trailing 20-session median
  * exclude earnings shocks (earnings day through three sessions after)

Structures (fixed, no sweep):
  * ATM long call
  * ATM / 8%-OTM bull-call debit spread
    * 5%-OTM / 10%-OTM bull-put credit spread, 16-30% credit/width
  * 30-75 DTE, target 50 DTE, OI >= 50, quote width <= 15%

Entry is the next session's natural price (long ask, short bid). Exit is the
natural price (long bid, short ask), at +50% or 21 sessions, with no stop. A
missing required exit quote makes the trade unresolved; stale marks are never
substituted for missing losers.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uwos.exact_spread_backtester import HistoricalOptionQuoteStore, parse_occ_symbol


SPLIT_DATE = "2026-04-14"
TARGET_DTE = 50
DTE_MIN = 30
DTE_MAX = 75
MIN_OPEN_INTEREST = 50
MAX_SPREAD_PCT = 0.15
SHORT_CALL_MONEYNESS = 1.08
SHORT_PUT_MONEYNESS = 0.95
LONG_PUT_MONEYNESS = 0.90
MIN_CREDIT = 0.50
MIN_CREDIT_WIDTH_RATIO = 0.16
MAX_CREDIT_WIDTH_RATIO = 0.30
PROFIT_TARGET = 0.50
CREDIT_PROFIT_TARGET = 0.30
MAX_HOLD = 21
ROUND_TRIP_FEE_PER_LEG = 1.30
STRUCTURES = ("long_call", "bull_call_debit", "bull_put_credit")


def parse_chain(quotes: pd.DataFrame, tickers: set[str], asof: dt.date, spot: dict[str, float]) -> pd.DataFrame:
    columns = [
        "ticker",
        "option_symbol",
        "right",
        "expiry",
        "strike",
        "dte",
        "bid",
        "ask",
        "open_interest",
        "spread_pct",
        "spot",
    ]
    if quotes.empty or not tickers:
        return pd.DataFrame(columns=columns)
    records = []
    for row in quotes.itertuples(index=False):
        parsed = parse_occ_symbol(row.option_symbol)
        if parsed is None:
            continue
        root, expiry, right, strike = parsed
        if root not in tickers or right not in {"C", "P"}:
            continue
        underlying = spot.get(root)
        if underlying is None or not np.isfinite(underlying) or underlying <= 0:
            continue
        dte = (expiry - asof).days
        bid = float(row.bid)
        ask = float(row.ask)
        width_pct = (ask - bid) / ask if ask > 0 else np.nan
        oi = float(row.open_interest) if np.isfinite(row.open_interest) else 0.0
        if not DTE_MIN <= dte <= DTE_MAX or bid < 0 or ask <= 0:
            continue
        if oi < MIN_OPEN_INTEREST or not np.isfinite(width_pct) or width_pct > MAX_SPREAD_PCT:
            continue
        records.append(
            {
                "ticker": root,
                "option_symbol": row.option_symbol,
                "right": right,
                "expiry": expiry,
                "strike": strike,
                "dte": dte,
                "bid": bid,
                "ask": ask,
                "open_interest": oi,
                "spread_pct": width_pct,
                "spot": underlying,
            }
        )
    return pd.DataFrame(records, columns=columns)


def select_structure(chain: pd.DataFrame, ticker: str, structure: str) -> dict | None:
    ticker_chain = chain[chain["ticker"].eq(ticker)].copy()
    if ticker_chain.empty:
        return None
    calls = ticker_chain[ticker_chain["right"].eq("C")].copy()
    puts = ticker_chain[ticker_chain["right"].eq("P")].copy()
    if structure in {"long_call", "bull_call_debit"} and calls.empty:
        return None
    calls["dte_gap"] = (calls["dte"] - TARGET_DTE).abs()
    calls["atm_gap"] = (calls["strike"] - calls["spot"]).abs()

    if structure == "long_call":
        leg = calls.sort_values(["dte_gap", "atm_gap", "spread_pct"]).iloc[0]
        return {
            "entry": float(leg.ask),
            "max_risk": float(leg.ask),
            "legs": [(str(leg.option_symbol), 1)],
            "expiry": leg.expiry,
            "long_strike": float(leg.strike),
            "short_strike": np.nan,
            "entry_spread_pct": float(leg.spread_pct),
            "is_credit": False,
        }

    if structure == "bull_call_debit":
        candidates = []
        for expiry, expiry_calls in calls.groupby("expiry", observed=True):
            long_leg = expiry_calls.assign(
                gap=(expiry_calls["strike"] - expiry_calls["spot"]).abs()
            ).sort_values(["gap", "spread_pct"]).iloc[0]
            short_pool = expiry_calls[expiry_calls["strike"].gt(long_leg.strike)].copy()
            if short_pool.empty:
                continue
            target = float(long_leg.spot) * SHORT_CALL_MONEYNESS
            short_leg = short_pool.assign(gap=(short_pool["strike"] - target).abs()).sort_values(
                ["gap", "spread_pct"]
            ).iloc[0]
            debit = float(long_leg.ask) - float(short_leg.bid)
            spread_width = float(short_leg.strike) - float(long_leg.strike)
            if debit <= 0 or spread_width <= debit:
                continue
            candidates.append(
                {
                    "entry": debit,
                    "max_risk": debit,
                    "legs": [(str(long_leg.option_symbol), 1), (str(short_leg.option_symbol), -1)],
                    "expiry": expiry,
                    "long_strike": float(long_leg.strike),
                    "short_strike": float(short_leg.strike),
                    "entry_spread_pct": max(float(long_leg.spread_pct), float(short_leg.spread_pct)),
                    "dte_gap": abs(int(long_leg.dte) - TARGET_DTE),
                    "is_credit": False,
                }
            )
        if not candidates:
            return None
        return sorted(candidates, key=lambda value: (value["dte_gap"], value["entry_spread_pct"]))[0]

    if structure == "bull_put_credit":
        if puts.empty:
            return None
        candidates = []
        for expiry, expiry_puts in puts.groupby("expiry", observed=True):
            short_target = float(expiry_puts["spot"].iloc[0]) * SHORT_PUT_MONEYNESS
            short_leg = expiry_puts.assign(gap=(expiry_puts["strike"] - short_target).abs()).sort_values(
                ["gap", "spread_pct"]
            ).iloc[0]
            long_pool = expiry_puts[expiry_puts["strike"].lt(short_leg.strike)].copy()
            if long_pool.empty:
                continue
            long_target = float(short_leg.spot) * LONG_PUT_MONEYNESS
            long_leg = long_pool.assign(gap=(long_pool["strike"] - long_target).abs()).sort_values(
                ["gap", "spread_pct"]
            ).iloc[0]
            credit = float(short_leg.bid) - float(long_leg.ask)
            spread_width = float(short_leg.strike) - float(long_leg.strike)
            credit_width = credit / spread_width if spread_width > 0 else np.nan
            if (
                credit < MIN_CREDIT
                or spread_width <= credit
                or not MIN_CREDIT_WIDTH_RATIO <= credit_width <= MAX_CREDIT_WIDTH_RATIO
            ):
                continue
            candidates.append(
                {
                    "entry": credit,
                    "max_risk": spread_width - credit,
                    "legs": [(str(short_leg.option_symbol), -1), (str(long_leg.option_symbol), 1)],
                    "expiry": expiry,
                    "long_strike": float(long_leg.strike),
                    "short_strike": float(short_leg.strike),
                    "entry_spread_pct": max(float(long_leg.spread_pct), float(short_leg.spread_pct)),
                    "dte_gap": abs(int(short_leg.dte) - TARGET_DTE),
                    "is_credit": True,
                }
            )
        if not candidates:
            return None
        return sorted(candidates, key=lambda value: (value["dte_gap"], value["entry_spread_pct"]))[0]
    return None


def natural_exit(indexed: pd.DataFrame, legs: list[tuple[str, int]]) -> float | None:
    total = 0.0
    for symbol, quantity in legs:
        if symbol not in indexed.index:
            return None
        quote = indexed.loc[symbol]
        if isinstance(quote, pd.DataFrame):
            quote = quote.iloc[0]
        price = quote.bid if quantity > 0 else quote.ask
        if not np.isfinite(price):
            return None
        total += quantity * max(float(price), 0.0)
    return total


def settlement_value(spot: float, legs: list[tuple[str, int]]) -> float | None:
    if not np.isfinite(spot) or spot <= 0:
        return None
    total = 0.0
    for symbol, quantity in legs:
        parsed = parse_occ_symbol(symbol)
        if parsed is None:
            return None
        _, _, right, strike = parsed
        intrinsic = max(spot - strike, 0.0) if right == "C" else max(strike - spot, 0.0)
        total += quantity * intrinsic
    return total


def simulate(root: Path, panel: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    store = HistoricalOptionQuoteStore(root, use_hot=True, use_oi=True)
    dates = store.available_dates()
    date_index = {date.isoformat(): index for index, date in enumerate(dates)}
    panel = panel.copy()
    panel["asof"] = pd.to_datetime(panel["asof"]).dt.strftime("%Y-%m-%d")
    close_map = panel.set_index(["asof", "ticker"])["close"].to_dict()

    # Signals are created at the shock close and entered on the next available session.
    entries: dict[dt.date, list[dict]] = {}
    for row in signals.itertuples(index=False):
        slot = date_index.get(str(row.signal_date))
        if slot is None or slot + 1 >= len(dates):
            continue
        entry_date = dates[slot + 1]
        for structure in STRUCTURES:
            entries.setdefault(entry_date, []).append(
                {
                    "event_id": row.event_id,
                    "ticker": row.ticker,
                    "sector": row.sector,
                    "shock_date": row.shock_date,
                    "signal_date": row.signal_date,
                    "post_earnings_shock": bool(row.post_earnings_shock),
                    "shock_z": row.shock_z,
                    "drawdown_63": row.drawdown_63,
                    "shock_volume_ratio": row.shock_volume_ratio,
                    "structure": structure,
                }
            )

    open_positions: list[dict] = []
    closed: list[dict] = []
    for session_index, session in enumerate(dates):
        quotes = store.get_quotes_for_date(session)
        indexed = quotes.set_index("option_symbol") if not quotes.empty else pd.DataFrame()

        still_open = []
        for position in open_positions:
            age = session_index - position["entry_index"]
            value = natural_exit(indexed, position["legs"]) if not quotes.empty else None
            settled = False
            if value is None and session >= position["expiry"]:
                spot = close_map.get((session.isoformat(), position["ticker"]))
                value = settlement_value(float(spot) if spot is not None else np.nan, position["legs"])
                settled = value is not None
            if value is not None:
                if position["is_credit"]:
                    gain = position["entry"] + value
                    progress = gain / position["entry"]
                    target = CREDIT_PROFIT_TARGET
                else:
                    gain = value - position["entry"]
                    progress = gain / position["entry"]
                    target = PROFIT_TARGET
                if settled:
                    reason = "expiry_settlement"
                elif progress >= target:
                    reason = "profit_target"
                elif age >= MAX_HOLD:
                    reason = "time_exit"
                else:
                    still_open.append(position)
                    continue
                fees = ROUND_TRIP_FEE_PER_LEG * len(position["legs"])
                pnl = gain * 100.0 - fees
                closed.append(
                    {
                        **{key: position[key] for key in (
                            "event_id",
                            "ticker",
                            "sector",
                            "shock_date",
                            "signal_date",
                            "entry_date",
                            "post_earnings_shock",
                            "shock_z",
                            "drawdown_63",
                            "shock_volume_ratio",
                            "structure",
                            "entry",
                            "max_risk",
                            "long_strike",
                            "short_strike",
                            "entry_spread_pct",
                        )},
                        "exit_date": session.isoformat(),
                        "held_sessions": age,
                        "exit_reason": reason,
                        "exit_value": value,
                        "pnl": pnl,
                        "return_on_risk": pnl / (position["max_risk"] * 100.0),
                        "resolved": True,
                    }
                )
            elif age >= MAX_HOLD:
                closed.append(
                    {
                        **{key: position[key] for key in (
                            "event_id",
                            "ticker",
                            "sector",
                            "shock_date",
                            "signal_date",
                            "entry_date",
                            "post_earnings_shock",
                            "shock_z",
                            "drawdown_63",
                            "shock_volume_ratio",
                            "structure",
                            "entry",
                            "max_risk",
                            "long_strike",
                            "short_strike",
                            "entry_spread_pct",
                        )},
                        "exit_date": session.isoformat(),
                        "held_sessions": age,
                        "exit_reason": "missing_exit_quote",
                        "exit_value": np.nan,
                        "pnl": np.nan,
                        "return_on_risk": np.nan,
                        "resolved": False,
                    }
                )
            else:
                still_open.append(position)
        open_positions = still_open

        pending = entries.get(session, [])
        if pending and not quotes.empty:
            tickers = {item["ticker"] for item in pending}
            spot = {ticker: close_map.get((session.isoformat(), ticker)) for ticker in tickers}
            chain = parse_chain(quotes, tickers, session, spot)
            for item in pending:
                selected = select_structure(chain, item["ticker"], item["structure"])
                if selected is None:
                    continue
                open_positions.append(
                    {
                        **item,
                        **selected,
                        "entry_date": session.isoformat(),
                        "entry_index": session_index,
                    }
                )

        # Keep memory bounded across the full history.
        store._cache.pop(session, None)
        store._leg_quote_cache.pop(session, None)

    for position in open_positions:
        closed.append(
            {
                **{key: position[key] for key in (
                    "event_id",
                    "ticker",
                    "sector",
                    "shock_date",
                    "signal_date",
                    "entry_date",
                    "post_earnings_shock",
                    "shock_z",
                    "drawdown_63",
                    "shock_volume_ratio",
                    "structure",
                    "entry",
                    "max_risk",
                    "long_strike",
                    "short_strike",
                    "entry_spread_pct",
                )},
                "exit_date": "",
                "held_sessions": len(dates) - 1 - position["entry_index"],
                "exit_reason": "outcome_not_yet_observable",
                "exit_value": np.nan,
                "pnl": np.nan,
                "return_on_risk": np.nan,
                "resolved": False,
            }
        )
    return pd.DataFrame(closed)


def profit_factor(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    gains = values[values > 0].sum()
    losses = -values[values < 0].sum()
    return gains / losses if losses > 0 else np.inf if gains > 0 else np.nan


def clustered_pf_interval(frame: pd.DataFrame, trials: int = 3000) -> tuple[float, float]:
    by_day = {day: block for day, block in frame.groupby("signal_date", observed=True)}
    days = list(by_day)
    if len(days) < 3:
        return np.nan, np.nan
    rng = np.random.default_rng(20260729)
    values = []
    for _ in range(trials):
        sample = [by_day[days[index]] for index in rng.integers(0, len(days), len(days))]
        values.append(profit_factor(pd.concat(sample, ignore_index=True)["pnl"]))
    finite = np.asarray([value for value in values if np.isfinite(value)])
    return (
        float(np.quantile(finite, 0.05)) if len(finite) else np.nan,
        float(np.quantile(finite, 0.95)) if len(finite) else np.nan,
    )


def summarize(trades: pd.DataFrame, split: str) -> None:
    print("\n=== FIXED BULLISH SHOCK-REVERSAL OPTION STRUCTURES ===")
    rows = []
    for structure in STRUCTURES:
        structure_rows = trades[trades["structure"].eq(structure)]
        for cohort, cohort_rows in (
            ("non_earnings", structure_rows[~structure_rows["post_earnings_shock"]]),
            ("post_earnings", structure_rows[structure_rows["post_earnings_shock"]]),
        ):
            for sample, block in (
                ("TRAIN", cohort_rows[cohort_rows["signal_date"].lt(split)]),
                ("TEST", cohort_rows[cohort_rows["signal_date"].ge(split)]),
                ("ALL", cohort_rows),
            ):
                resolved = block[block["resolved"]].dropna(subset=["pnl"])
                low, high = clustered_pf_interval(resolved)
                rows.append(
                    {
                        "structure": structure,
                        "cohort": cohort,
                        "sample": sample,
                        "attempted": len(block),
                        "resolved": len(resolved),
                        "days": resolved["signal_date"].nunique(),
                        "win": resolved["pnl"].gt(0).mean(),
                        "mean_ror": resolved["return_on_risk"].mean(),
                        "pf": profit_factor(resolved["pnl"]),
                        "pf_p05": low,
                        "pf_p95": high,
                        "pnl": resolved["pnl"].sum(),
                        "avg_hold": resolved["held_sessions"].mean(),
                    }
                )
    table = pd.DataFrame(rows)
    print(table.round(3).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/Users/anuppamvi/uw_root/tradedesk")
    parser.add_argument(
        "--panel",
        default="/Users/anuppamvi/uw_root/tradedesk/out/research/price_panel_20260728.csv.gz",
    )
    parser.add_argument(
        "--signals",
        default="/Users/anuppamvi/uw_root/tradedesk/out/research/bull_reversal_avwap_signals.csv",
    )
    parser.add_argument(
        "--out",
        default="/Users/anuppamvi/uw_root/tradedesk/out/research/bull_reversal_option_trades.csv",
    )
    parser.add_argument("--split", default=SPLIT_DATE)
    args = parser.parse_args()

    panel = pd.read_csv(args.panel, low_memory=False)
    signals = pd.read_csv(args.signals, low_memory=False)
    shocks = signals[signals["variant"].eq("shock_close")].copy()
    trades = simulate(Path(args.root), panel, shocks)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(args.out, index=False)
    summarize(trades, args.split)

    tsla = trades[trades["ticker"].eq("TSLA")].sort_values(["signal_date", "structure"])
    print("\n=== TSLA HISTORICAL SHOCK STRUCTURES ===")
    if tsla.empty:
        print("no TSLA option structure met the fixed liquidity rules")
    else:
        columns = [
            "shock_date",
            "entry_date",
            "structure",
            "entry",
            "long_strike",
            "short_strike",
            "resolved",
            "exit_reason",
            "pnl",
            "return_on_risk",
        ]
        print(tsla[columns].to_string(index=False))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()