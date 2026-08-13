"""Build the Cremers-Weinbaum matched call-put IV spread from full-tape quotes."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/anuppamvi/uw_root/tradedesk")
CACHE = ROOT / "out/options_pattern_pipeline_v1/cache/bot_eod"
OUT = ROOT / "out/volatility_spread_features.csv"


def weighted_median(values: pd.Series, weights: pd.Series) -> float:
    order = np.argsort(values.to_numpy())
    ordered_values = values.to_numpy()[order]
    ordered_weights = weights.to_numpy()[order]
    cutoff = ordered_weights.sum() / 2.0
    return float(ordered_values[np.searchsorted(np.cumsum(ordered_weights), cutoff)])


def build_file(path: Path) -> pd.DataFrame | None:
    columns = [
        "date", "ticker", "expiry", "strike", "option_type", "dte", "bid", "ask",
        "iv", "spread_pct", "volume", "open_interest", "stock_close",
    ]
    quotes = pd.read_csv(path, usecols=columns, low_memory=False)
    numeric = [column for column in columns if column not in {"date", "ticker", "expiry", "option_type"}]
    for column in numeric:
        quotes[column] = pd.to_numeric(quotes[column], errors="coerce")
    quotes = quotes[
        quotes["ticker"].notna()
        & quotes["iv"].between(0.01, 5.0)
        & quotes["dte"].between(20, 60)
        & (quotes["bid"] > 0)
        & (quotes["ask"] >= quotes["bid"])
        & (quotes["spread_pct"] <= 0.20)
        & (quotes["volume"] >= 10)
        & (quotes["open_interest"] >= 10)
        & (quotes["stock_close"] > 0)
        & (((quotes["strike"] - quotes["stock_close"]).abs() / quotes["stock_close"]) <= 0.10)
    ].copy()
    if quotes.empty:
        return None
    quotes["weight"] = np.sqrt(quotes["volume"] * quotes["open_interest"]) / (
        quotes["spread_pct"].clip(lower=0.005)
    )
    pair = quotes.pivot_table(
        index=["date", "ticker", "expiry", "strike"],
        columns="option_type",
        values=["iv", "weight", "stock_close"],
        aggfunc="first",
    )
    required = [("iv", "call"), ("iv", "put"), ("weight", "call"), ("weight", "put")]
    if any(column not in pair.columns for column in required):
        return None
    pair = pair.dropna(subset=required).reset_index()
    if pair.empty:
        return None
    pair["iv_spread"] = pair[("iv", "call")] - pair[("iv", "put")]
    pair["pair_weight"] = np.minimum(pair[("weight", "call")], pair[("weight", "put")])
    pair.columns = [column[0] if column[1] == "" else f"{column[0]}_{column[1]}" for column in pair.columns]

    rows = []
    for (date, ticker), group in pair.groupby(["date", "ticker"]):
        rows.append(
            {
                "date": date,
                "ticker": ticker,
                "volatility_spread": weighted_median(group["iv_spread"], group["pair_weight"]),
                "matched_pairs": len(group),
                "matched_pair_weight": group["pair_weight"].sum(),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    files = sorted(CACHE.glob("bot_eod_quotes_*.csv"))
    frames = []
    for index, path in enumerate(files, 1):
        frame = build_file(path)
        if frame is not None:
            frames.append(frame)
        if index % 10 == 0:
            print(f"[vol-spread] {index}/{len(files)}", flush=True)
    if not frames:
        raise SystemExit("no matched call-put pairs")
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(OUT, index=False)
    print(
        f"[vol-spread] days={result.date.nunique()} rows={len(result)} "
        f"tickers={result.ticker.nunique()} -> {OUT}"
    )


if __name__ == "__main__":
    main()
