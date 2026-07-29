"""Does UW flow pick a better CONTRACT, given the same name?

Random name selection matched the momentum signal on dollars, so the selection
rule is not where the edge is. The remaining question is whether the five files
add value one level down: given that a name is being traded, do they identify a
better strike and expiry than a mechanical rule?

Three contract choosers compete on the SAME names, SAME dates, SAME managed exit:

  fixed       1.05x strike, 80 DTE -- the mechanical baseline
  flow_led    the call that actually absorbed the most ask-side premium that day
  random      any eligible call on the name

Any difference is attributable to contract choice alone.
"""
from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import managed_exit_backtest as base  # noqa: E402
from conviction_stack import parse_occ  # noqa: E402

OUT = base.ROOT / "out/contract_selection_test.csv"
PROFIT_TARGET = 0.5


def hot_chain_flow(session: str) -> pd.DataFrame:
    """Ask-side premium per contract for one session."""
    day = base.ROOT / session
    hits = [p for p in sorted(day.glob("hot-chains-*.zip")) if p.is_file() and session in p.name]
    if not hits:
        return pd.DataFrame()
    archive = zipfile.ZipFile(hits[0])
    member = archive.namelist()[0]
    frame = pd.read_csv(
        archive.open(member),
        usecols=["option_symbol", "premium", "ask_side_volume", "bid_side_volume", "multileg_volume", "volume"],
        low_memory=False,
    )
    for column in ["premium", "ask_side_volume", "bid_side_volume", "multileg_volume", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    half = frame.multileg_volume / 2.0
    ask = (frame.ask_side_volume - half).clip(lower=0)
    bid = (frame.bid_side_volume - half).clip(lower=0)
    total = (ask + bid).replace(0, np.nan)
    frame["ask_share"] = ask / total
    frame["ask_premium"] = frame.premium * frame.ask_share.fillna(0)
    return frame[["option_symbol", "ask_premium", "ask_share"]]


def choose(quotes: pd.DataFrame, tickers: set[str], how: str, flow: pd.DataFrame, rng) -> pd.DataFrame:
    eligible = quotes[
        quotes.ticker.isin(tickers)
        & quotes.option_type.eq("call")
        & quotes.dte.between(*base.DTE_BAND)
        & (quotes.curr_oi >= 50)
        & (quotes.spread_pct <= base.MAX_SPREAD_PCT)
    ].copy()
    if eligible.empty:
        return eligible
    if how == "fixed":
        eligible["strike_gap"] = (eligible.strike - eligible.stock_price * 1.05).abs()
        eligible["dte_gap"] = (eligible.dte - base.TARGET_DTE).abs()
        return eligible.sort_values(["dte_gap", "strike_gap"]).groupby("ticker", as_index=False).first()
    if how == "flow_led":
        if flow.empty:
            return flow
        merged = eligible.merge(flow, on="option_symbol", how="inner")
        merged = merged[merged.ask_premium > 0]
        if merged.empty:
            return merged
        return merged.sort_values("ask_premium", ascending=False).groupby("ticker", as_index=False).first()
    order = rng.permutation(len(eligible))
    return eligible.iloc[order].groupby("ticker", as_index=False).first()


def simulate(panel, days, quote_for, flow_for, how: str, rng) -> pd.DataFrame:
    open_positions: list[dict] = []
    closed: list[dict] = []
    held: set[str] = set()

    for index, session in enumerate(days):
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
                    "chooser": how,
                    "dte": position["dte"],
                    "moneyness": position["moneyness"],
                    "cost": position["cost"] * 100.0,
                    "exit_reason": reason,
                    "held": age,
                    "pnl": pnl,
                    "return_on_cost": pnl / (position["cost"] * 100.0),
                }
            )
            held.discard(position["ticker"])
        open_positions = still_open

        if index + 1 >= len(days):
            continue
        day = panel[panel.date == session].dropna(subset=["pos_52w", "flow_escalation"])
        if day.empty:
            continue
        entry_session = days[index + 1]
        entry_quotes = quote_for(entry_session)
        if entry_quotes.empty:
            continue
        flow = flow_for(entry_session) if how == "flow_led" else pd.DataFrame()

        for sector, block in day.groupby("sector"):
            if sector not in base.SECTORS or len(block) < base.MIN_PER_SECTOR:
                continue
            momentum = block.pos_52w.rank(pct=True)
            chosen = set(block[momentum >= base.DECILE].ticker) - held
            if not chosen:
                continue
            legs = choose(entry_quotes, chosen, how, flow, rng)
            if legs.empty:
                continue
            for row in legs.itertuples():
                open_positions.append(
                    {
                        "signal_date": session,
                        "ticker": row.ticker,
                        "entry_index": index + 1,
                        "cost": row.last_ask,
                        "symbol": row.option_symbol,
                        "last_mark": row.last_bid,
                        "dte": row.dte,
                        "moneyness": row.strike / row.stock_price,
                    }
                )
                held.add(row.ticker)
    return pd.DataFrame(closed)


def main() -> None:
    columns = ["date", "ticker", "sector", "issue_type", "marketcap", "close", "pos_52w", "hc_premium"]
    panel = pd.read_csv(base.PANEL, usecols=columns, low_memory=False)
    panel["date"] = pd.to_datetime(panel["date"]).dt.strftime("%Y-%m-%d")
    panel = panel[
        (panel.issue_type == "Common Stock")
        & (panel.marketcap.fillna(0) >= 2e9)
        & panel.sector.isin(base.SECTORS)
    ].sort_values(["ticker", "date"])
    grouped = panel.groupby("ticker")
    panel["flow_avg_20"] = grouped.hc_premium.transform(lambda s: s.rolling(20, min_periods=10).mean())
    panel["flow_escalation"] = panel.hc_premium / panel.flow_avg_20.replace(0, np.nan)

    days = sorted(p.name for p in base.ROOT.iterdir() if p.is_dir() and re.fullmatch(r"2026-\d{2}-\d{2}", p.name))
    position = {d: i for i, d in enumerate(days)}
    quote_cache: dict[str, pd.DataFrame] = {}
    flow_cache: dict[str, pd.DataFrame] = {}

    def quote_for(session: str) -> pd.DataFrame:
        if session not in quote_cache:
            slot = position[session]
            quote_cache[session] = (
                base.chain_quotes(session, days[slot + 1]) if slot + 1 < len(days) else pd.DataFrame()
            )
        return quote_cache[session]

    def flow_for(session: str) -> pd.DataFrame:
        if session not in flow_cache:
            flow_cache[session] = hot_chain_flow(session)
        return flow_cache[session]

    print("[contract] warming quote cache", flush=True)
    for session in days:
        quote_for(session)

    rng = np.random.default_rng(20260728)
    frames = []
    for how in ("fixed", "flow_led", "random"):
        trades = simulate(panel, days, quote_for, flow_for, how, rng)
        if trades.empty:
            print(f"[contract] {how}: no trades")
            continue
        trades["sample"] = np.where(trades.signal_date >= base.SPLIT, "TEST", "TRAIN")
        frames.append(trades)
        for sample in ("TRAIN", "TEST"):
            frame = trades[trades["sample"] == sample]
            if len(frame) < 20:
                continue
            gains = frame.pnl[frame.pnl > 0].sum()
            losses = -frame.pnl[frame.pnl < 0].sum()
            print(
                "{:<9} {:<6} n={:<4} mean={:+.3f} med={:+.3f} win={:.2f} PF={:>6.2f} "
                "pnl=${:>+10,.0f} cost=${:>6,.0f} dte={:>4.0f} mny={:.2f}".format(
                    how, sample, len(frame), frame.return_on_cost.mean(), frame.return_on_cost.median(),
                    frame.pnl.gt(0).mean(), gains / losses if losses else float("nan"),
                    frame.pnl.sum(), frame.cost.mean(), frame.dte.mean(), frame.moneyness.mean(),
                ),
                flush=True,
            )
        print(flush=True)

    if frames:
        pd.concat(frames, ignore_index=True).to_csv(OUT, index=False)
        print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
