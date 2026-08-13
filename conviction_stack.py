"""Contract-level conviction stack across all five UW files.

This is the join the ticker-day panel destroys. A candidate must be the SAME
OCC contract in every file:

  hot-chains        the contract traded real size today, ask-side led
  chain-oi-changes  next session confirms open interest actually rose
  bot-eod tape      individual prints show large aggressive buyers, with greeks
  dp-eod            the underlying saw dark-pool accumulation the same way
  stock-screener    implied vol and earnings context for the underlying

Output is intentionally a handful of contracts per day, not a ranked universe.
"""
from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/anuppamvi/uw_root/tradedesk")
OCC = re.compile(r"^([A-Z]+)(\d{6})([CP])(\d{8})$")


def open_zip(path: Path, usecols=None, chunksize=None):
    archive = zipfile.ZipFile(path)
    member = archive.namelist()[0]
    if usecols is not None:
        with archive.open(member) as handle:
            head = pd.read_csv(handle, nrows=0)
        usecols = [column for column in usecols if column in head.columns]
    return pd.read_csv(archive.open(member), usecols=usecols, chunksize=chunksize, low_memory=False)


def find(day: Path, stem: str) -> Path | None:
    hits = [p for p in sorted(day.glob(f"{stem}-*.zip")) if p.is_file() and day.name in p.name]
    return hits[0] if hits else None


def parse_occ(symbol: pd.Series) -> pd.DataFrame:
    extracted = symbol.str.upper().str.extract(OCC)
    extracted.columns = ["ticker", "yymmdd", "cp", "strike8"]
    return pd.DataFrame(
        {
            "ticker": extracted.ticker,
            "expiry": pd.to_datetime(extracted.yymmdd, format="%y%m%d", errors="coerce"),
            "option_type": extracted.cp.map({"C": "call", "P": "put"}),
            "strike": pd.to_numeric(extracted.strike8, errors="coerce") / 1000.0,
        }
    )


def load_hot_chains(day: Path) -> pd.DataFrame:
    path = find(day, "hot-chains")
    if path is None:
        raise SystemExit(f"no hot-chains for {day.name}")
    columns = [
        "option_symbol", "volume", "open_interest", "premium", "ask_side_volume",
        "bid_side_volume", "mid_volume", "multileg_volume", "sweep_volume",
        "floor_volume", "iv", "bid", "ask", "trades", "close", "sector",
    ]
    frame = open_zip(path, columns)
    numeric = [c for c in columns if c not in {"option_symbol", "sector"}]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.join(parse_occ(frame.option_symbol.astype(str)))
    frame = frame[frame.ticker.notna()].copy()

    # Strip spread legs before calling anything directional.
    half_multileg = frame.multileg_volume.fillna(0) / 2.0
    frame["directional_ask"] = (frame.ask_side_volume.fillna(0) - half_multileg).clip(lower=0)
    frame["directional_bid"] = (frame.bid_side_volume.fillna(0) - half_multileg).clip(lower=0)
    directional = frame.directional_ask + frame.directional_bid
    frame["ask_share"] = frame.directional_ask / directional.replace(0, np.nan)
    frame["spread_pct"] = (frame.ask - frame.bid) / frame.ask.replace(0, np.nan)
    frame["volume_to_oi"] = frame.volume / frame.open_interest.replace(0, np.nan)
    return frame


def load_oi_confirmation(next_day: Path) -> pd.DataFrame:
    path = find(next_day, "chain-oi-changes")
    if path is None:
        raise SystemExit(f"no chain-oi-changes for {next_day.name}")
    frame = open_zip(path, ["option_symbol", "oi_diff_plain", "curr_oi", "last_oi"])
    for column in ["oi_diff_plain", "curr_oi", "last_oi"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.rename(columns={"oi_diff_plain": "oi_change"})
    return frame[["option_symbol", "oi_change", "curr_oi", "last_oi"]]


def load_tape(day: Path, wanted: set[str]) -> pd.DataFrame:
    path = find(day, "bot-eod-report")
    if path is None:
        raise SystemExit(f"no bot-eod-report for {day.name}")
    columns = [
        "underlying_symbol", "option_chain_id", "side", "strike", "option_type",
        "expiry", "size", "premium", "delta", "vega", "gamma",
        "implied_volatility", "underlying_price", "canceled",
    ]
    parts = []
    for chunk in open_zip(path, columns, chunksize=2_000_000):
        chunk = chunk[chunk.option_chain_id.isin(wanted)]
        if chunk.empty:
            continue
        chunk = chunk[chunk.canceled.astype(str).str.lower() != "t"]
        for column in ["size", "premium", "delta", "vega", "gamma", "implied_volatility", "underlying_price"]:
            chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
        chunk["is_ask"] = chunk.side.eq("ask")
        chunk["ask_premium"] = np.where(chunk.is_ask, chunk.premium, 0.0)
        chunk["bid_premium"] = np.where(chunk.side.eq("bid"), chunk.premium, 0.0)
        chunk["block_premium"] = np.where(chunk.is_ask & chunk.premium.ge(100_000), chunk.premium, 0.0)
        chunk["max_single_ask"] = np.where(chunk.is_ask, chunk.premium, 0.0)
        parts.append(chunk)
    if not parts:
        return pd.DataFrame()
    tape = pd.concat(parts, ignore_index=True)
    grouped = tape.groupby("option_chain_id").agg(
        tape_trades=("premium", "size"),
        tape_ask_premium=("ask_premium", "sum"),
        tape_bid_premium=("bid_premium", "sum"),
        tape_block_premium=("block_premium", "sum"),
        tape_largest_ask_print=("max_single_ask", "max"),
        tape_delta=("delta", "median"),
        tape_iv=("implied_volatility", "median"),
        underlying_price=("underlying_price", "median"),
    )
    grouped["tape_net_premium"] = grouped.tape_ask_premium - grouped.tape_bid_premium
    grouped["tape_ask_share"] = grouped.tape_ask_premium / (
        grouped.tape_ask_premium + grouped.tape_bid_premium
    ).replace(0, np.nan)
    return grouped.reset_index().rename(columns={"option_chain_id": "option_symbol"})


def load_dark_pool(day: Path) -> pd.DataFrame:
    path = find(day, "dp-eod-report")
    if path is None:
        return pd.DataFrame(columns=["ticker", "dp_premium", "dp_bias"])
    frame = open_zip(path, ["ticker", "nbbo_ask", "nbbo_bid", "price", "premium", "size", "canceled"])
    for column in ["nbbo_ask", "nbbo_bid", "price", "premium", "size"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame[frame.canceled.astype(str).str.lower() != "t"]
    frame = frame[(frame.nbbo_ask > 0) & (frame.nbbo_bid > 0) & (frame.nbbo_ask >= frame.nbbo_bid)]
    midpoint = (frame.nbbo_ask + frame.nbbo_bid) / 2.0
    spread = (frame.nbbo_ask - frame.nbbo_bid).replace(0, np.nan)
    frame["location"] = ((frame.price - midpoint) / spread).clip(-1, 1)
    frame["signed"] = frame.location * frame.premium
    frame["ticker"] = frame.ticker.astype(str).str.upper()
    grouped = frame.groupby("ticker").agg(
        dp_premium=("premium", "sum"),
        dp_signed=("signed", "sum"),
        dp_prints=("premium", "size"),
    )
    grouped["dp_bias"] = grouped.dp_signed / grouped.dp_premium.replace(0, np.nan)
    return grouped.reset_index()


def load_screener(day: Path) -> pd.DataFrame:
    path = find(day, "stock-screener")
    if path is None:
        raise SystemExit(f"no stock-screener for {day.name}")
    columns = [
        "ticker", "close", "marketcap", "sector", "issue_type", "iv_rank", "iv30d",
        "volatility", "implied_move_perc", "next_earnings_date", "total_volume",
        "avg30_volume",
    ]
    frame = open_zip(path, columns)
    frame = frame[frame.ticker.notna()].copy()
    for column in ["close", "marketcap", "iv_rank", "iv30d", "volatility", "implied_move_perc", "total_volume", "avg30_volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["ticker"] = frame.ticker.astype(str).str.upper()
    frame["stock_volume_surge"] = frame.total_volume / frame.avg30_volume.replace(0, np.nan)
    return frame.drop_duplicates("ticker")


def main() -> None:
    signal_date = sys.argv[1] if len(sys.argv) > 1 else "2026-07-23"
    day = ROOT / signal_date
    days = sorted(p.name for p in ROOT.iterdir() if p.is_dir() and re.fullmatch(r"2026-\d{2}-\d{2}", p.name))
    following = [d for d in days if d > signal_date]
    if not following:
        raise SystemExit("need a following session to confirm open interest")
    next_day = ROOT / following[0]

    print(f"[stack] hot chains {signal_date}", flush=True)
    hot = load_hot_chains(day)
    screener = load_screener(day)

    # Liquid, real, tradeable contracts only.
    universe = hot.merge(
        screener[["ticker", "marketcap", "issue_type", "iv_rank", "implied_move_perc", "next_earnings_date", "stock_volume_surge", "close"]].rename(columns={"close": "stock_close"}),
        on="ticker",
        how="inner",
    )
    universe = universe[
        (universe.issue_type == "Common Stock")
        & (universe.marketcap.fillna(0) >= 2e9)
        & (universe.volume >= 250)
        & (universe.open_interest >= 100)
        & (universe.premium >= 250_000)
        & (universe.spread_pct <= 0.15)
        & universe.ask_share.notna()
    ].copy()
    print(f"[stack] liquid contracts today: {len(universe)}", flush=True)

    print(f"[stack] open-interest confirmation from {next_day.name}", flush=True)
    confirmation = load_oi_confirmation(next_day)
    universe = universe.merge(confirmation, on="option_symbol", how="left")
    universe["oi_confirmed"] = universe.oi_change.fillna(0) > 0

    print("[stack] scanning full option tape for these contracts", flush=True)
    tape = load_tape(day, set(universe.option_symbol))
    universe = universe.merge(tape, on="option_symbol", how="left") if not tape.empty else universe

    dark = load_dark_pool(day)
    universe = universe.merge(dark, on="ticker", how="left")

    universe["days_to_expiry"] = (universe.expiry - pd.Timestamp(signal_date)).dt.days
    universe["directional_sign"] = np.where(universe.option_type == "call", 1.0, -1.0)
    universe["dp_agrees"] = np.sign(universe.dp_bias.fillna(0)) == universe.directional_sign

    stack = universe[
        universe.oi_confirmed
        & universe.ask_share.ge(0.60)
        & universe.tape_ask_share.fillna(0).ge(0.60)
        & universe.tape_largest_ask_print.fillna(0).ge(250_000)
        & universe.volume_to_oi.ge(0.5)
        & universe.days_to_expiry.between(10, 90)
    ].copy()

    stack["conviction"] = (
        np.log1p(stack.tape_block_premium.fillna(0) / 1e5)
        + np.log1p(stack.premium / 1e5)
        + 2.0 * (stack.tape_ask_share - 0.5)
        + stack.dp_agrees.astype(float) * 0.5
    )
    stack = stack.sort_values("conviction", ascending=False)

    output = ROOT / f"out/conviction_stack_{signal_date}.csv"
    columns = [
        "option_symbol", "ticker", "sector", "option_type", "strike", "expiry",
        "days_to_expiry", "stock_close", "bid", "ask", "spread_pct", "volume",
        "open_interest", "volume_to_oi", "premium", "ask_share", "oi_change",
        "tape_trades", "tape_ask_share", "tape_block_premium",
        "tape_largest_ask_print", "tape_delta", "tape_iv", "dp_bias", "dp_agrees",
        "iv_rank", "implied_move_perc", "next_earnings_date", "conviction",
    ]
    columns = [c for c in columns if c in stack.columns]
    stack[columns].to_csv(output, index=False)

    print(f"\n=== FIVE-FILE CONVICTION STACK {signal_date} ===")
    print(f"liquid contracts {len(universe)} -> survive all five files: {len(stack)}")
    show = [
        "ticker", "option_type", "strike", "days_to_expiry", "premium",
        "ask_share", "oi_change", "tape_ask_share", "tape_largest_ask_print",
        "dp_bias", "iv_rank", "conviction",
    ]
    show = [c for c in show if c in stack.columns]
    print(stack[show].head(20).round(3).to_string(index=False))
    print(f"\nwrote {output}")


if __name__ == "__main__":
    main()
