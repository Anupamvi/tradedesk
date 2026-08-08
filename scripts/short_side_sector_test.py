"""Where does short-side selection carry signal, sector by sector?

The Technology put lane was the only thing that survived a permutation null out
of sample, so the short side is where the information is. That test covered one
sector. This runs every sector, and builds the per-sector null from the SAME
permutation runs rather than re-simulating per sector, which keeps it affordable.

It also asks the second open question: does UW flow escalation add anything on
the short side? Flow was proven useless on the long side, but was never tested
where the signal actually lives.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import managed_exit_backtest as base  # noqa: E402

PERMUTATIONS = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 150
OUT = base.ROOT / "out/short_side_sector_test.csv"
PROFIT_TARGET = 0.5
DECILE = 0.80
MIN_TRADES = 25


def simulate(panel, days, quote_for, rng, mode: str) -> pd.DataFrame:
    """mode: signal | signal_flow | random"""
    open_positions: list[dict] = []
    closed: list[dict] = []
    held: set[str] = set()

    for index, session in enumerate(days):
        quotes = quote_for(session)
        if quotes.empty:
            continue
        bid = quotes.set_index("option_symbol").last_bid

        still_open = []
        for pos in open_positions:
            current = bid.get(pos["symbol"])
            age = index - pos["entry_index"]
            if current is None or not np.isfinite(current):
                if age < base.MAX_HOLD:
                    still_open.append(pos)
                    continue
                current = pos["last_mark"]
            else:
                pos["last_mark"] = current
            gain = current / pos["cost"] - 1.0
            if gain >= PROFIT_TARGET:
                reason = "profit_target"
            elif age >= base.MAX_HOLD:
                reason = "time_stop"
            else:
                still_open.append(pos)
                continue
            pnl = (current - pos["cost"]) * 100.0 - base.CONTRACT_FEE
            closed.append(
                {
                    "signal_date": pos["signal_date"],
                    "sector": pos["sector"],
                    "ticker": pos["ticker"],
                    "pnl": pnl,
                    "return_on_cost": pnl / (pos["cost"] * 100.0),
                }
            )
            held.discard(pos["ticker"])
        open_positions = still_open

        if index + 1 >= len(days):
            continue
        day = panel[panel.date == session].dropna(subset=["pos_52w"])
        if day.empty:
            continue
        entry_quotes = quote_for(days[index + 1])
        if entry_quotes.empty:
            continue

        for sector, block in day.groupby("sector"):
            if len(block) < base.MIN_PER_SECTOR:
                continue
            momentum = block.pos_52w.rank(pct=True)
            weak = block[momentum <= 1.0 - DECILE]
            if mode == "signal_flow":
                escalation = weak.flow_escalation.rank(pct=True)
                weak = weak[escalation >= 0.5]
            chosen = set(weak.ticker)
            if mode == "random":
                pool = block.ticker.to_numpy()
                chosen = set(rng.choice(pool, size=min(len(chosen), len(pool)), replace=False))
            chosen -= held
            if not chosen:
                continue
            legs = entry_quotes[
                entry_quotes.ticker.isin(chosen)
                & entry_quotes.option_type.eq("put")
                & entry_quotes.dte.between(*base.DTE_BAND)
                & (entry_quotes.curr_oi >= 50)
                & (entry_quotes.spread_pct <= base.MAX_SPREAD_PCT)
            ].copy()
            if legs.empty:
                continue
            legs["strike_gap"] = (legs.strike - legs.stock_price * 0.95).abs()
            legs["dte_gap"] = (legs.dte - base.TARGET_DTE).abs()
            legs = legs.sort_values(["dte_gap", "strike_gap"]).groupby("ticker", as_index=False).first()
            for row in legs.itertuples():
                open_positions.append(
                    {
                        "signal_date": session,
                        "ticker": row.ticker,
                        "sector": sector,
                        "entry_index": index + 1,
                        "cost": row.last_ask,
                        "symbol": row.option_symbol,
                        "last_mark": row.last_bid,
                    }
                )
                held.add(row.ticker)
    return pd.DataFrame(closed)


def profit_factor(values: pd.Series) -> float:
    gains = values[values > 0].sum()
    losses = -values[values < 0].sum()
    return gains / losses if losses > 0 else np.nan


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["sample"] = np.where(frame.signal_date >= base.SPLIT, "TEST", "TRAIN")
    return frame.groupby(["sector", "sample"]).apply(
        lambda f: pd.Series(
            {"n": len(f), "mean": f.return_on_cost.mean(), "pf": profit_factor(f.pnl), "pnl": f.pnl.sum()}
        )
    )


def main() -> None:
    columns = ["date", "ticker", "sector", "issue_type", "marketcap", "close", "pos_52w", "hc_premium"]
    panel = pd.read_csv(base.PANEL, usecols=columns, low_memory=False)
    panel["date"] = pd.to_datetime(panel["date"]).dt.strftime("%Y-%m-%d")
    panel = panel[
        (panel.issue_type == "Common Stock")
        & (panel.marketcap.fillna(0) >= 2e9)
        & panel.sector.notna()
    ].sort_values(["ticker", "date"])
    grouped = panel.groupby("ticker")
    panel["flow_avg_20"] = grouped.hc_premium.transform(lambda s: s.rolling(20, min_periods=10).mean())
    panel["flow_escalation"] = panel.hc_premium / panel.flow_avg_20.replace(0, np.nan)

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

    print("[short] warming quote cache", flush=True)
    for session in days:
        quote_for(session)

    rng = np.random.default_rng(20260728)
    actual = summarize(simulate(panel, days, quote_for, rng, "signal"))
    with_flow = summarize(simulate(panel, days, quote_for, rng, "signal_flow"))

    nulls = []
    for trial in range(PERMUTATIONS):
        trades = simulate(panel, days, quote_for, rng, "random")
        if trades.empty:
            continue
        stats = summarize(trades).reset_index()
        stats["trial"] = trial
        nulls.append(stats)
        if (trial + 1) % 25 == 0:
            print(f"[short] {trial + 1}/{PERMUTATIONS}", flush=True)

    null = pd.concat(nulls, ignore_index=True)
    null.to_csv(OUT, index=False)

    print(f"\n=== SHORT SIDE BY SECTOR vs {PERMUTATIONS}-permutation null ===")
    rows = []
    for (sector, sample), observed in actual.iterrows():
        if observed["n"] < MIN_TRADES:
            continue
        block = null[(null.sector == sector) & (null["sample"] == sample)]
        if len(block) < 20:
            continue
        rows.append(
            {
                "sector": sector,
                "sample": sample,
                "n": int(observed["n"]),
                "mean": round(observed["mean"], 3),
                "pf": round(observed["pf"], 2),
                "null_pf": round(block.pf.mean(), 2),
                "p_pf": round((block.pf.dropna() >= observed["pf"]).mean(), 4),
                "p_mean": round((block["mean"].dropna() >= observed["mean"]).mean(), 4),
            }
        )
    table = pd.DataFrame(rows).sort_values(["sector", "sample"])
    print(table.to_string(index=False))

    print("\nsectors significant (p_pf <= 0.05) in BOTH halves:")
    pivot = table.pivot_table(index="sector", columns="sample", values="p_pf")
    if {"TRAIN", "TEST"}.issubset(pivot.columns):
        winners = pivot[(pivot.TRAIN <= 0.05) & (pivot.TEST <= 0.05)]
        print(winners.to_string() if len(winners) else "NONE")

    print("\n=== DOES FLOW ADD ON THE SHORT SIDE? ===")
    comparison = actual.join(with_flow, rsuffix="_flow")
    keep = comparison[comparison.n >= MIN_TRADES]
    print(keep[["n", "mean", "pf", "n_flow", "mean_flow", "pf_flow"]].round(3).to_string())
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
