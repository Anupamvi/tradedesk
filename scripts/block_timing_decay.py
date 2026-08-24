"""When does the move actually happen relative to the block print?

Every backtest so far enters at the NEXT session's close. If the underlying
reprices within minutes of a large aggressive print, then an end-of-day pipeline
can never capture it and no amount of signal work will fix that.

The tape carries underlying_price on every print, so the intraday path is
reconstructed directly from it.
"""
from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/anuppamvi/uw_root/tradedesk")
OUT = ROOT / "out/block_timing_decay.csv"
MIN_BLOCK_PREMIUM = 1_000_000


def tape_frame(day: str) -> pd.DataFrame:
    path = ROOT / day / f"bot-eod-report-{day}.zip"
    if not path.exists():
        return pd.DataFrame()
    archive = zipfile.ZipFile(path)
    member = archive.namelist()[0]
    columns = [
        "executed_at", "underlying_symbol", "option_chain_id", "side", "option_type",
        "size", "premium", "underlying_price", "canceled", "delta",
    ]
    parts = []
    for chunk in pd.read_csv(archive.open(member), usecols=columns, chunksize=2_000_000, low_memory=False):
        chunk = chunk[chunk.canceled.astype(str).str.lower() != "t"]
        chunk["premium"] = pd.to_numeric(chunk.premium, errors="coerce")
        chunk["underlying_price"] = pd.to_numeric(chunk.underlying_price, errors="coerce")
        chunk = chunk[chunk.underlying_price > 0]
        parts.append(chunk)
    if not parts:
        return pd.DataFrame()
    tape = pd.concat(parts, ignore_index=True)
    tape["timestamp"] = pd.to_datetime(tape.executed_at, utc=True, errors="coerce")
    return tape.dropna(subset=["timestamp"])


def main() -> None:
    days = sorted(
        p.name for p in ROOT.iterdir()
        if p.is_dir() and re.fullmatch(r"2026-\d{2}-\d{2}", p.name)
        and (p / f"bot-eod-report-{p.name}.zip").exists()
    )
    sample = days[-int(sys.argv[1]) :] if len(sys.argv) > 1 else days[-10:]
    print(f"[timing] analysing {len(sample)} sessions", flush=True)

    rows = []
    for day in sample:
        tape = tape_frame(day)
        if tape.empty:
            continue
        blocks = tape[(tape.side == "ask") & (tape.premium >= MIN_BLOCK_PREMIUM)]
        if blocks.empty:
            continue
        print(f"[timing] {day}: {len(blocks)} blocks >= ${MIN_BLOCK_PREMIUM/1e6:.0f}M", flush=True)
        # Group once. Filtering the full tape per ticker inside the loop is a
        # full scan of ~24M rows for every block and is unusably slow.
        block_tickers = set(blocks.underlying_symbol)
        relevant = tape[tape.underlying_symbol.isin(block_tickers)].sort_values("timestamp")
        by_ticker = {name: frame for name, frame in relevant.groupby("underlying_symbol")}
        for ticker, group in blocks.groupby("underlying_symbol"):
            block = group.loc[group.premium.idxmax()]
            prints = by_ticker.get(ticker)
            if prints is None or len(prints) < 50:
                continue
            block_time = block.timestamp
            price_at_block = block.underlying_price
            session_close = prints.underlying_price.iloc[-1]
            direction = 1.0 if block.option_type == "call" else -1.0

            record = {
                "date": day,
                "ticker": ticker,
                "block_premium": block.premium,
                "option_type": block.option_type,
                "block_time_utc": block_time.strftime("%H:%M"),
                "price_at_block": price_at_block,
                "session_close": session_close,
            }
            for minutes in (5, 15, 30, 60, 120):
                window = prints[
                    (prints.timestamp > block_time)
                    & (prints.timestamp <= block_time + pd.Timedelta(minutes=minutes))
                ]
                if window.empty:
                    record[f"move_{minutes}m"] = np.nan
                    continue
                record[f"move_{minutes}m"] = direction * (
                    window.underlying_price.iloc[-1] / price_at_block - 1.0
                )
            record["move_to_close"] = direction * (session_close / price_at_block - 1.0)
            rows.append(record)

    if not rows:
        raise SystemExit("no blocks found")
    result = pd.DataFrame(rows)
    result.to_csv(OUT, index=False)

    print(f"\n=== MOVE IN THE BLOCK'S DIRECTION, from the print onward (n={len(result)}) ===")
    for column in ["move_5m", "move_15m", "move_30m", "move_60m", "move_120m", "move_to_close"]:
        series = result[column].dropna()
        if series.empty:
            continue
        print(
            f"  {column:<14} n={len(series):<4} mean={series.mean():+.4f} "
            f"median={series.median():+.4f} right_dir={series.gt(0).mean():.3f}"
        )
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
