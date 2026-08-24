"""Build stock-option relative volume and aggressor-side flow from screener files."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/anuppamvi/uw_root/tradedesk")
OUT = ROOT / "out/screener_flow_features.csv"
COLS = [
    "ticker", "call_volume", "put_volume", "total_volume",
    "call_volume_ask_side", "call_volume_bid_side",
    "put_volume_ask_side", "put_volume_bid_side",
]


def main() -> None:
    frames = []
    days = sorted(path for path in ROOT.iterdir() if path.is_dir() and re.fullmatch(r"2026-\d{2}-\d{2}", path.name))
    for day in days:
        files = sorted(
            path
            for path in day.glob("stock-screener*.zip")
            if path.is_file() and day.name in path.name
        )
        if not files:
            continue
        frame = pd.concat((pd.read_csv(path, usecols=COLS, low_memory=False) for path in files), ignore_index=True)
        frame = frame[frame.ticker.notna()].copy()
        for column in COLS[1:]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
        option_contracts = frame.call_volume + frame.put_volume
        frame["option_stock_volume_ratio"] = 100.0 * option_contracts / frame.total_volume.replace(0, np.nan)
        bullish = frame.call_volume_ask_side + frame.put_volume_bid_side
        bearish = frame.call_volume_bid_side + frame.put_volume_ask_side
        frame["screener_directional_volume_bias"] = (bullish - bearish) / (bullish + bearish).replace(0, np.nan)
        frame["buyer_put_call_ratio"] = frame.put_volume_ask_side / frame.call_volume_ask_side.replace(0, np.nan)
        frame["date"] = day.name
        frames.append(frame[["date", "ticker", "option_stock_volume_ratio", "screener_directional_volume_bias", "buyer_put_call_ratio"]])
    result = pd.concat(frames, ignore_index=True)
    if result.duplicated(["date", "ticker"]).any():
        raise ValueError("duplicate ticker/date rows in screener features")
    result.to_csv(OUT, index=False)
    print(f"days={result.date.nunique()} rows={len(result)} -> {OUT}")


if __name__ == "__main__":
    main()
