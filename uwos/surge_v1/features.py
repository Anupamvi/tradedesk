"""Feature construction for the surge engine.

Deliberately separates three feature families so their contribution can be
measured independently, because the repo's history is full of results that
turned out to be one family wearing another's name:

  price_*   pure price/trend, derivable without any options data
  vol_*     what the option market implies -- the thing you must PAY
  flow_*    the five UW feeds

Every feature is known at the close of the signal date. Entry is the NEXT close.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

VOL_ETPS = {
    "UVXY", "VXX", "SVIX", "SVXY", "VIXY", "UVIX", "VIXM", "VXZ", "VIXP",
    "SQQQ", "TQQQ", "SOXL", "SOXS", "SPXU", "UPRO", "TNA", "TZA", "LABU", "LABD",
    "SPXL", "SPXS", "UDOW", "SDOW", "FAS", "FAZ", "YINN", "YANG", "NUGT", "DUST",
}

PRICE_FEATURES = [
    "price_ret_5", "price_ret_10", "price_ret_21", "price_ret_63",
    "price_pos_52w", "price_dist_ma20", "price_dist_ma50",
    "price_vol_20", "price_vol_ratio", "price_up_days_10",
    "price_max_drawup_21", "price_gap_ret", "price_range_pct",
    "price_rvol_adj_mom", "price_new_high_20", "price_accel",
]
VOL_FEATURES = ["iv_rank", "iv30d", "iv_chg_1w", "iv_chg_1m", "vrp_ratio", "vol_iv_to_rv"]
FLOW_FEATURES = [
    "call_vol_surge", "put_vol_surge", "stock_vol_surge", "put_call_ratio",
    "prem_tilt", "net_prem_tilt", "call_oi_chg", "put_oi_chg",
    "hc_sweep_share", "hc_opening_share", "hc_dir_bias", "hc_premium",
    "oi_dir_bias", "oi_open_conviction", "oi_built_premium", "oi_nearmoney_share",
    "dp_bias", "dp_block_bias", "dp_block_share", "dp_premium",
    "tape_prem_bias", "tape_delta_notional", "tape_vega_flow", "tape_gamma_flow",
]
CONTEXT_FEATURES = ["earn_dte", "log_mcap"]

PANEL_COLS = sorted(
    set(VOL_FEATURES) | set(FLOW_FEATURES)
    | {"date", "ticker", "sector", "issue_type", "marketcap", "close",
       "next_earnings_date", "realized_vol_30d", "ret_1d"}
)


def load_panel(path: str, min_mcap: float = 2e9, min_price: float = 5.0) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=lambda c: c in set(PANEL_COLS), low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["ticker"].notna()]
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df = df[~df["ticker"].isin(VOL_ETPS)]
    df = df[df["issue_type"].astype(str).str.contains("Common", case=False, na=False)]
    df = df[pd.to_numeric(df["marketcap"], errors="coerce").fillna(0) >= min_mcap]
    df = df[pd.to_numeric(df["close"], errors="coerce") > min_price]
    df = df.drop_duplicates(subset=["ticker", "date"], keep="last")
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("ticker", sort=False)["close"]
    for h in (5, 10, 21, 63):
        df[f"price_ret_{h}"] = g.pct_change(h)

    ma20 = g.transform(lambda s: s.rolling(20, min_periods=10).mean())
    ma50 = g.transform(lambda s: s.rolling(50, min_periods=20).mean())
    df["price_dist_ma20"] = df["close"] / ma20 - 1.0
    df["price_dist_ma50"] = df["close"] / ma50 - 1.0

    hi52 = g.transform(lambda s: s.rolling(252, min_periods=40).max())
    lo52 = g.transform(lambda s: s.rolling(252, min_periods=40).min())
    df["price_pos_52w"] = (df["close"] - lo52) / (hi52 - lo52).replace(0, np.nan)
    hi20 = g.transform(lambda s: s.rolling(20, min_periods=10).max())
    df["price_new_high_20"] = (df["close"] >= hi20 * 0.999).astype(float)

    r1 = g.pct_change(1)
    df["price_gap_ret"] = r1
    vol20 = r1.groupby(df["ticker"]).transform(lambda s: s.rolling(20, min_periods=10).std())
    vol60 = r1.groupby(df["ticker"]).transform(lambda s: s.rolling(60, min_periods=25).std())
    df["price_vol_20"] = vol20 * np.sqrt(252)
    df["price_vol_ratio"] = vol20 / vol60.replace(0, np.nan)
    df["price_up_days_10"] = (r1 > 0).groupby(df["ticker"]).transform(
        lambda s: s.rolling(10, min_periods=5).mean())

    # Trend measured in units of the name's own noise, so a 10% move in a quiet
    # name is not compared against 10% in a 90-vol name.
    df["price_rvol_adj_mom"] = df["price_ret_21"] / (df["price_vol_20"] * np.sqrt(21 / 252)).replace(0, np.nan)
    df["price_accel"] = df["price_ret_5"] - df["price_ret_21"] * (5.0 / 21.0)
    df["price_max_drawup_21"] = g.transform(
        lambda s: s.rolling(21, min_periods=10).apply(lambda w: w[-1] / w.min() - 1.0, raw=True))
    df["price_range_pct"] = (hi20 / lo52.replace(0, np.nan)) - 1.0
    return df


def add_context(df: pd.DataFrame) -> pd.DataFrame:
    ern = pd.to_datetime(df.get("next_earnings_date"), errors="coerce")
    df["earn_dte"] = (ern - df["date"]).dt.days
    df["log_mcap"] = np.log10(pd.to_numeric(df["marketcap"], errors="coerce").clip(lower=1))
    rv = pd.to_numeric(df.get("realized_vol_30d"), errors="coerce")
    df["vol_iv_to_rv"] = pd.to_numeric(df["iv30d"], errors="coerce") / rv.replace(0, np.nan)
    return df


def add_targets(df: pd.DataFrame, horizons=(21, 63), thresholds=(0.20, 0.30)) -> pd.DataFrame:
    """Entry is the NEXT close; EOD data is not actionable at the same close.

    Targets are symmetric. A move is a move: INTC +130% and ORCL -39% are the
    same event class and a detector that only finds one side is not a detector.
    """
    g = df.groupby("ticker", sort=False)["close"]
    entry = g.shift(-1)
    df["entry_close"] = entry
    for h in horizons:
        fwd = g.shift(-(1 + h)) / entry - 1.0
        df[f"fwd_{h}"] = fwd
        df[f"abs_fwd_{h}"] = fwd.abs()
        for thr in thresholds:
            pct = int(thr * 100)
            df[f"up_{h}_{pct}"] = (fwd >= thr).astype("float")
            df[f"dn_{h}_{pct}"] = (fwd <= -thr).astype("float")
            df[f"move_{h}_{pct}"] = (fwd.abs() >= thr).astype("float")
        # Path-aware: a managed trade is filled on the path, not the terminal price.
        shifted = g.shift(-2)
        run_max = shifted.groupby(df["ticker"]).transform(
            lambda s: s.rolling(h, min_periods=max(3, h // 3)).max())
        run_min = shifted.groupby(df["ticker"]).transform(
            lambda s: s.rolling(h, min_periods=max(3, h // 3)).min())
        df[f"fwd_max_{h}"] = run_max / entry - 1.0
        df[f"fwd_min_{h}"] = run_min / entry - 1.0
    return df


def build(path: str, min_mcap: float = 2e9, horizons=(21, 63)) -> pd.DataFrame:
    df = load_panel(path, min_mcap=min_mcap)
    df = add_price_features(df)
    df = add_context(df)
    df = add_targets(df, horizons=horizons)
    df["month"] = df["date"].dt.to_period("M")
    return df


def feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    def present(cols):
        return [c for c in cols if c in df.columns]
    price = present(PRICE_FEATURES)
    vol = present(VOL_FEATURES)
    flow = present(FLOW_FEATURES)
    ctx = present(CONTEXT_FEATURES)
    return {
        "price_only": price + ctx,
        "price_vol": price + vol + ctx,
        "flow_only": flow + ctx,
        "all": price + vol + flow + ctx,
    }


DEEP_PREFIXES = ("tp_", "hc_", "oi_", "dp_")


def load_deep(deep_dir: str) -> pd.DataFrame:
    """Concatenate the per-date native-resolution feature caches."""
    files = sorted(Path(deep_dir).glob("*.pkl"))
    if not files:
        raise FileNotFoundError(f"no deep feature caches under {deep_dir}")
    d = pd.concat([pd.read_pickle(f) for f in files], ignore_index=True)
    d["ticker"] = d["ticker"].astype(str).str.upper()
    d["date"] = pd.to_datetime(d["date"])
    return d.drop_duplicates(subset=["ticker", "date"], keep="last")


def build_deep(path: str, deep_dir: str, min_mcap: float = 2e9,
               horizons=(21, 63)) -> pd.DataFrame:
    """Price + implied-vol context from the screener panel, flow at native resolution.

    The shallow flow columns are dropped entirely so the two cannot be confused;
    every flow feature in the returned frame comes from the deep extractor.
    """
    base = load_panel(path, min_mcap=min_mcap)
    base = base.drop(columns=[c for c in FLOW_FEATURES if c in base.columns], errors="ignore")
    deep = load_deep(deep_dir)
    df = base.merge(deep, on=["ticker", "date"], how="left")
    df = add_price_features(df)
    df = add_context(df)
    df = add_targets(df, horizons=horizons)
    df["month"] = df["date"].dt.to_period("M")
    return df


def deep_feature_columns(df: pd.DataFrame) -> list[str]:
    skip = {"tp_premium", "tp_size", "tp_prints", "hc_premium", "hc_volume",
            "oi_built_prem", "dp_prem", "dp_dir_prem"}
    return [c for c in df.columns
            if c.startswith(DEEP_PREFIXES) and c not in skip
            and pd.api.types.is_numeric_dtype(df[c])]


def deep_feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    """Feature sets for the deep panel; flow blocks are separable by feed."""
    def present(cols):
        return [c for c in cols if c in df.columns]
    price = present(PRICE_FEATURES)
    vol = present(VOL_FEATURES)
    ctx = present(CONTEXT_FEATURES)
    deep = deep_feature_columns(df)
    by_feed = {p: [c for c in deep if c.startswith(p)] for p in DEEP_PREFIXES}
    return {
        "price_only": price + ctx,
        "price_vol": price + vol + ctx,
        "deep_flow_only": deep + ctx,
        "tape_only": by_feed["tp_"] + ctx,
        "nontape_deep": by_feed["hc_"] + by_feed["oi_"] + by_feed["dp_"] + ctx,
        "all_deep": price + vol + deep + ctx,
    }
