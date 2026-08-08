"""Self-normalization and compression features.

Two defects found by tracing why INTC/MU were ranked ~1000 during the months they
rose 54-72% and ranked ~5 after the move was over:

1. FLOW WAS ABSOLUTE, NOT UNUSUAL. A $2M call build is enormous for a $16bn name
   and noise for NVDA, but the model saw the same number. "Unusual" is the entire
   premise of this data and it was never computed. Every deep flow feature is
   z-scored against that ticker's OWN trailing distribution.

2. EVERY PRICE FEATURE PEAKED AFTER THE MOVE. ret_21 / pos_52w / dist_ma50 are
   all high once a stock has already tripled, so the model learned to buy
   exhaustion. INTC was FLAT at $45 for weeks before going to $139. The setup is
   compression then expansion, so compression is measured explicitly.

Rolling windows include the current session and nothing after it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def self_z(df: pd.DataFrame, cols: list[str], window: int = 20,
           min_periods: int = 10, clip: float = 6.0) -> pd.DataFrame:
    """z-score each column against the same ticker's own trailing distribution."""
    g = df.groupby("ticker", sort=False)
    out = {}
    for c in cols:
        s = df[c]
        mu = g[c].transform(lambda x: x.rolling(window, min_periods=min_periods).mean())
        sd = g[c].transform(lambda x: x.rolling(window, min_periods=min_periods).std())
        out[f"{c}_z"] = ((s - mu) / sd.replace(0, np.nan)).clip(-clip, clip)
    return pd.DataFrame(out, index=df.index)


def add_compression(df: pd.DataFrame) -> pd.DataFrame:
    """Features that are HIGH before a move and LOW after it."""
    g = df.groupby("ticker", sort=False)["close"]
    r1 = g.pct_change(1)

    v20 = r1.groupby(df["ticker"]).transform(lambda s: s.rolling(20, min_periods=10).std())
    v60 = r1.groupby(df["ticker"]).transform(lambda s: s.rolling(60, min_periods=25).std())
    df["cmp_vol_squeeze"] = v20 / v60.replace(0, np.nan)
    # Where does today's 20d vol sit in this name's own year? Low = coiled.
    df["cmp_vol_pctile"] = v20.groupby(df["ticker"]).transform(
        lambda s: s.rolling(252, min_periods=40).rank(pct=True))

    hi20 = g.transform(lambda s: s.rolling(20, min_periods=10).max())
    lo20 = g.transform(lambda s: s.rolling(20, min_periods=10).min())
    df["cmp_range_20"] = (hi20 - lo20) / df["close"].replace(0, np.nan)
    df["cmp_range_pctile"] = df["cmp_range_20"].groupby(df["ticker"]).transform(
        lambda s: s.rolling(252, min_periods=40).rank(pct=True))
    df["cmp_pos_in_range"] = (df["close"] - lo20) / (hi20 - lo20).replace(0, np.nan)

    # Flatness: the INTC-in-March state. Small trailing move, not a large one.
    df["cmp_flat_21"] = -g.pct_change(21).abs()
    df["cmp_flat_63"] = -g.pct_change(63).abs()

    # How long since the last 20d high -- a long base, not an extended run.
    at_high = (df["close"] >= hi20 * 0.999).astype(float)
    df["cmp_days_since_high"] = at_high.groupby(df["ticker"]).transform(
        lambda s: s.groupby((s == 1).cumsum()).cumcount())

    # Volume building while price does not: accumulation inside the base.
    if "hc_premium" in df.columns:
        opt = df["hc_premium"]
        opt_mu = opt.groupby(df["ticker"]).transform(
            lambda s: s.rolling(20, min_periods=10).mean())
        df["cmp_opt_vs_base"] = (opt / opt_mu.replace(0, np.nan)) * df["cmp_vol_squeeze"]
    return df


COMPRESSION_FEATURES = [
    "cmp_vol_squeeze", "cmp_vol_pctile", "cmp_range_20", "cmp_range_pctile",
    "cmp_pos_in_range", "cmp_flat_21", "cmp_flat_63", "cmp_days_since_high",
    "cmp_opt_vs_base",
]


def build_normalized(df: pd.DataFrame, deep_cols: list[str],
                     window: int = 20) -> tuple[pd.DataFrame, list[str]]:
    """Return the panel with `_z` flow columns and compression features added."""
    df = df.sort_values(["ticker", "date"]).copy()
    z = self_z(df, deep_cols, window=window)
    df = pd.concat([df, z], axis=1)
    df = add_compression(df)
    return df, list(z.columns)


def normalized_feature_sets(df: pd.DataFrame, z_cols: list[str],
                            price_cols: list[str], vol_cols: list[str],
                            ctx_cols: list[str], raw_deep: list[str]) -> dict:
    def keep(cols):
        return [c for c in cols if c in df.columns]
    cmp_ = keep(COMPRESSION_FEATURES)
    z = keep(z_cols)
    return {
        "raw_momentum": keep(price_cols) + keep(ctx_cols),
        "compression_only": cmp_ + keep(ctx_cols),
        "unusual_flow_only": z + keep(ctx_cols),
        "compression_plus_unusual": cmp_ + z + keep(ctx_cols),
        "everything": keep(price_cols) + keep(vol_cols) + keep(raw_deep) + z + cmp_ + keep(ctx_cols),
    }
