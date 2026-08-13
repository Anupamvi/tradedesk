"""Reconstruct per-(day,ticker) predictive flow/greek/dark-pool features from the
full UW download bundle (stock-screener + bot-eod-report + dp-eod-report).

These are the directional-conviction signals the pricing files (hot-chains,
chain-oi) do not carry. Output: flow_features.csv keyed on (asof, ticker).
Local data only; no Schwab account; no live orders.
"""
from __future__ import annotations

import io
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/Users/anuppamvi/uw_root/tradedesk")
OUT = ROOT / "flow_features.csv"


def _read_zip(z: Path, usecols=None) -> pd.DataFrame:
    if not z.exists():
        return pd.DataFrame()
    try:
        with zipfile.ZipFile(z) as zf:
            name = next((n for n in zf.namelist() if n.lower().endswith(".csv")), None)
            if not name:
                return pd.DataFrame()
            with zf.open(name) as fh:
                return pd.read_csv(io.TextIOWrapper(fh, "utf-8"), usecols=usecols, low_memory=False)
    except Exception as exc:
        print(f"  skip {z.name}: {exc}", file=sys.stderr)
        return pd.DataFrame()


def screener_features(fld: Path) -> pd.DataFrame:
    cols = ["ticker", "call_volume", "put_volume", "call_premium", "put_premium",
            "put_call_ratio", "bearish_premium", "bullish_premium",
            "avg_30_day_call_volume", "avg_30_day_put_volume",
            "net_call_premium", "net_put_premium", "total_open_interest",
            "marketcap", "close", "prev_close"]
    df = _read_zip(fld / f"stock-screener-{fld.name}.zip", usecols=lambda c: c in cols)
    if df.empty:
        return pd.DataFrame()
    df["ticker"] = df["ticker"].astype(str).str.upper()
    for c in df.columns:
        if c != "ticker":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    bull = df["bullish_premium"].fillna(0); bear = df["bearish_premium"].fillna(0)
    df["flow_dir_premium"] = (bull - bear) / (bull + bear).replace(0, np.nan)
    df["net_prem_dir"] = (df["net_call_premium"].fillna(0) - df["net_put_premium"].fillna(0))
    df["call_vol_surge"] = df["call_volume"] / df["avg_30_day_call_volume"].replace(0, np.nan)
    df["put_vol_surge"] = df["put_volume"] / df["avg_30_day_put_volume"].replace(0, np.nan)
    df["vol_surge_dir"] = np.log((df["call_vol_surge"].clip(0.01, 100)) /
                                 (df["put_vol_surge"].clip(0.01, 100)))
    df["pcr"] = df["put_call_ratio"]
    df["day_ret"] = df["close"] / df["prev_close"] - 1.0
    df["log_mktcap"] = np.log(df["marketcap"].clip(1e6, None))
    keep = ["ticker", "flow_dir_premium", "net_prem_dir", "call_vol_surge",
            "put_vol_surge", "vol_surge_dir", "pcr", "day_ret", "log_mktcap",
            "total_open_interest"]
    return df[keep]


def bot_eod_features(fld: Path) -> pd.DataFrame:
    cols = ["underlying_symbol", "option_type", "side", "premium", "size",
            "delta", "gamma", "vega", "implied_volatility"]
    df = _read_zip(fld / f"bot-eod-report-{fld.name}.zip", usecols=lambda c: c in cols)
    if df.empty:
        return pd.DataFrame()
    df["ticker"] = df["underlying_symbol"].astype(str).str.upper()
    for c in ["premium", "size", "delta", "gamma", "vega", "implied_volatility"]:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    df["premium"] = df["premium"].fillna(0)
    ot = df["option_type"].astype(str).str.lower()
    df["call_prem"] = np.where(ot.str.startswith("c"), df["premium"], 0.0)
    df["put_prem"] = np.where(ot.str.startswith("p"), df["premium"], 0.0)
    # ask-side = aggressive buyer; bid-side = aggressive seller
    side = df["side"].astype(str).str.lower()
    df["aggr_dir_prem"] = np.where(side.str.contains("ask"), df["premium"],
                                   np.where(side.str.contains("bid"), -df["premium"], 0.0))
    df["dprem"] = df["delta"].fillna(0) * df["premium"]
    g = df.groupby("ticker")
    out = pd.DataFrame({
        "bot_call_prem": g["call_prem"].sum(),
        "bot_put_prem": g["put_prem"].sum(),
        "bot_aggr_dir_prem": g["aggr_dir_prem"].sum(),
        "bot_delta_prem": g["dprem"].sum(),
        "bot_avg_iv": g["implied_volatility"].mean(),
        "bot_trade_count": g.size(),
    }).reset_index()
    tot = (out["bot_call_prem"] + out["bot_put_prem"]).replace(0, np.nan)
    out["bot_cp_prem_dir"] = (out["bot_call_prem"] - out["bot_put_prem"]) / tot
    return out


def dp_features(fld: Path) -> pd.DataFrame:
    cols = ["ticker", "premium", "size", "volume"]
    df = _read_zip(fld / f"dp-eod-report-{fld.name}.zip", usecols=lambda c: c in cols)
    if df.empty:
        return pd.DataFrame()
    df["ticker"] = df["ticker"].astype(str).str.upper()
    for c in ["premium", "size", "volume"]:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    g = df.groupby("ticker")
    return pd.DataFrame({
        "dp_premium": g["premium"].sum(),
        "dp_prints": g.size(),
    }).reset_index()


def main():
    folders = sorted(p for p in ROOT.glob("2026-*")
                     if p.is_dir() and re.fullmatch(r"2026-\d\d-\d\d", p.name))
    rows = []
    for fld in folders:
        sc = screener_features(fld)
        if sc.empty:
            continue
        m = sc
        for extra in (bot_eod_features(fld), dp_features(fld)):
            if not extra.empty:
                m = m.merge(extra, on="ticker", how="left")
        m.insert(0, "asof", fld.name)
        rows.append(m)
        print(f"{fld.name}: {len(m)} tickers  cols={m.shape[1]}")
    if not rows:
        print("no flow features built"); return
    allf = pd.concat(rows, ignore_index=True)
    allf.to_csv(OUT, index=False)
    print(f"\nWrote {len(allf)} (asof,ticker) rows -> {OUT}")
    print("columns:", [c for c in allf.columns if c not in ("asof", "ticker")])


if __name__ == "__main__":
    main()
