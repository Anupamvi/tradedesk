"""Build buyer-to-open option-flow features from daily OI confirmation files.

The file dated t confirms which contracts' OI increased from t-1 to t and
contains t-1 ask/bid/multileg volume. This is a conservative public-data proxy
for Pan-Poteshman's buyer-to-open put/call ratio; it is not true account-level
open/close data.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/anuppamvi/uw_root/tradedesk")
OUT = ROOT / "out/opening_flow_features.csv"
COLS = [
    "option_symbol",
    "underlying_symbol",
    "oi_diff_plain",
    "prev_ask_volume",
    "prev_bid_volume",
    "prev_multi_leg_volume",
    "strike",
    "stock_price",
    "dte",
]


def build_day(day: Path) -> pd.DataFrame | None:
    paths = sorted(path for path in day.glob("chain-oi-changes*.zip") if path.is_file())
    if not paths:
        return None
    frame = pd.concat((pd.read_csv(path, usecols=COLS, low_memory=False) for path in paths), ignore_index=True)
    for column in COLS[2:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    frame = frame[(frame["oi_diff_plain"] > 0) & frame["underlying_symbol"].notna()].copy()
    if frame.empty:
        return None

    # Remove half of multileg volume from each side. A spread contributes legs
    # to both ask and bid buckets but is not a naked directional opinion.
    half_multileg = frame["prev_multi_leg_volume"].clip(lower=0) / 2.0
    ask = (frame["prev_ask_volume"] - half_multileg).clip(lower=0)
    bid = (frame["prev_bid_volume"] - half_multileg).clip(lower=0)
    directional = ask + bid
    confirmed_open = np.minimum(frame["oi_diff_plain"], directional)
    buyer_share = ask / directional.replace(0, np.nan)
    frame["buyer_open"] = confirmed_open * buyer_share
    frame["seller_open"] = confirmed_open * (1.0 - buyer_share)
    frame["ticker"] = frame["underlying_symbol"].astype(str).str.upper()
    frame["is_call"] = frame["option_symbol"].astype(str).str.upper().str.contains(
        r"\d{6}C\d{8}$", regex=True
    )
    frame["buyer_open_call"] = np.where(frame["is_call"], frame["buyer_open"], 0.0)
    frame["buyer_open_put"] = np.where(~frame["is_call"], frame["buyer_open"], 0.0)
    frame["seller_open_call"] = np.where(frame["is_call"], frame["seller_open"], 0.0)
    frame["seller_open_put"] = np.where(~frame["is_call"], frame["seller_open"], 0.0)
    option_sign = np.where(frame["is_call"], 1.0, -1.0)
    frame["direction_score"] = option_sign * (buyer_share - (1.0 - buyer_share))
    frame["directional_chain"] = frame["direction_score"].abs() >= 0.20
    frame["bullish_chain"] = (
        frame["directional_chain"] & (frame["direction_score"] > 0)
    ).astype(int)
    frame["bearish_chain"] = (
        frame["directional_chain"] & (frame["direction_score"] < 0)
    ).astype(int)
    frame["near_money"] = (
        ((frame["strike"] - frame["stock_price"]).abs() / frame["stock_price"].replace(0, np.nan) <= 0.10)
        & frame["dte"].between(7, 60)
    )
    frame["bullish_near_chain"] = (frame["bullish_chain"].astype(bool) & frame["near_money"]).astype(int)
    frame["bearish_near_chain"] = (frame["bearish_chain"].astype(bool) & frame["near_money"]).astype(int)

    grouped = frame.groupby("ticker", sort=False).agg(
        buyer_open_call=("buyer_open_call", "sum"),
        buyer_open_put=("buyer_open_put", "sum"),
        seller_open_call=("seller_open_call", "sum"),
        seller_open_put=("seller_open_put", "sum"),
        confirmed_open_chains=("buyer_open", "count"),
        bullish_chains=("bullish_chain", "sum"),
        bearish_chains=("bearish_chain", "sum"),
        bullish_near_chains=("bullish_near_chain", "sum"),
        bearish_near_chains=("bearish_near_chain", "sum"),
    )
    grouped["buyer_open_pcr"] = np.log1p(grouped["buyer_open_put"]) - np.log1p(
        grouped["buyer_open_call"]
    )
    grouped["buyer_open_call_share"] = grouped["buyer_open_call"] / (
        grouped["buyer_open_call"] + grouped["buyer_open_put"]
    ).replace(0, np.nan)
    grouped["buyer_open_direction"] = (
        grouped["buyer_open_call"] - grouped["buyer_open_put"]
    ) / (grouped["buyer_open_call"] + grouped["buyer_open_put"]).replace(0, np.nan)
    grouped["seller_open_direction"] = (
        grouped["seller_open_put"] - grouped["seller_open_call"]
    ) / (grouped["seller_open_put"] + grouped["seller_open_call"]).replace(0, np.nan)
    directional_chains = grouped["bullish_chains"] + grouped["bearish_chains"]
    grouped["oi_chain_breadth"] = (
        grouped["bullish_chains"] - grouped["bearish_chains"]
    ) / directional_chains.replace(0, np.nan)
    grouped["oi_chain_breadth_shrunk"] = (
        grouped["bullish_chains"] - grouped["bearish_chains"]
    ) / (directional_chains + 10.0)
    near_chains = grouped["bullish_near_chains"] + grouped["bearish_near_chains"]
    grouped["oi_near_chain_breadth_shrunk"] = (
        grouped["bullish_near_chains"] - grouped["bearish_near_chains"]
    ) / (near_chains + 5.0)
    grouped.insert(0, "date", day.name)
    return grouped.reset_index()


def main() -> None:
    days = sorted(path for path in ROOT.iterdir() if path.is_dir() and re.fullmatch(r"2026-\d{2}-\d{2}", path.name))
    frames = []
    for index, day in enumerate(days, 1):
        frame = build_day(day)
        if frame is not None:
            frames.append(frame)
        if index % 20 == 0:
            print(f"[opening-flow] {index}/{len(days)}", flush=True)
    if not frames:
        raise SystemExit("no OI files found")
    result = pd.concat(frames, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT, index=False)
    print(f"[opening-flow] days={result.date.nunique()} rows={len(result)} -> {OUT}")


if __name__ == "__main__":
    main()
