"""Reconstruct a historical dealer-gamma (GEX) proxy from local UW chain-oi data.

For each dated folder, unzip chain-oi-changes, and for every underlying compute a
gamma-concentration net GEX proxy: gamma peaks at-the-money and decays with a
width scaled by sqrt(dte). Net dealer gamma ~ sum over strikes of
gamma_weight * (call_OI - put_OI). No IV needed; no browser; local data only.

Outputs one row per (asof, ticker): net_gex, abs_gex, gex_sign, atm_oi_frac.
Also emits market-level GEX from SPY/QQQ/IWM.
"""
from __future__ import annotations

import io
import math
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/Users/anuppamvi/uw_root/tradedesk")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/Users/anuppamvi/uw_root/tradedesk/gex_reconstructed.csv")

# option_symbol OCC: e.g. AAPL  260710C00190000 -> ...C.../P... marks call/put
CP_RE = re.compile(r"\d{6}([CP])\d{8}")


def day_gex(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cp = df["option_symbol"].astype(str).str.extract(CP_RE)[0]
    df["is_call"] = (cp == "C").astype(float)
    df["is_put"] = (cp == "P").astype(float)
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df["oi"] = pd.to_numeric(df["curr_oi"], errors="coerce").fillna(0.0)
    df["dte"] = pd.to_numeric(df["dte"], errors="coerce")
    df["spot"] = pd.to_numeric(df["stock_price"], errors="coerce")
    df = df[(df["strike"] > 0) & (df["spot"] > 0) & df["dte"].notna() & (cp.notna())]
    if df.empty:
        return pd.DataFrame()
    # gamma-concentration kernel: peak ATM, width ~ spot * 0.02 * sqrt(max(dte,1))
    width = (df["spot"] * 0.02 * np.sqrt(np.clip(df["dte"], 1, 90))).clip(lower=1e-6)
    z = (df["strike"] - df["spot"]) / width
    gk = np.exp(-0.5 * z * z)  # relative gamma weight
    signed = gk * df["oi"] * (df["is_call"] - df["is_put"])  # dealer GEX sign
    df["_signed"] = signed
    df["_absgk"] = gk * df["oi"]
    df["_atm"] = df["_absgk"] * (np.abs(z) < 0.5)
    g = df.groupby("underlying_symbol")
    out = pd.DataFrame({
        "net_gex": g["_signed"].sum(),
        "abs_gex": g["_absgk"].sum(),
        "atm_oi": g["_atm"].sum(),
    }).reset_index().rename(columns={"underlying_symbol": "ticker"})
    out["gex_sign"] = np.sign(out["net_gex"])
    out["atm_oi_frac"] = out["atm_oi"] / out["abs_gex"].replace(0, np.nan)
    out["net_gex_norm"] = out["net_gex"] / out["abs_gex"].replace(0, np.nan)
    return out[["ticker", "net_gex", "abs_gex", "gex_sign", "atm_oi_frac", "net_gex_norm"]]


def main():
    folders = sorted(p for p in ROOT.glob("2026-*") if p.is_dir() and re.fullmatch(r"2026-\d\d-\d\d", p.name))
    rows = []
    for fld in folders:
        z = fld / f"chain-oi-changes-{fld.name}.zip"
        if not z.exists():
            continue
        try:
            with zipfile.ZipFile(z) as zf:
                name = next((n for n in zf.namelist() if n.endswith(".csv")), None)
                if not name:
                    continue
                with zf.open(name) as fh:
                    df = pd.read_csv(io.TextIOWrapper(fh, "utf-8"), low_memory=False,
                                     usecols=["option_symbol", "underlying_symbol", "strike", "curr_oi", "dte", "stock_price"])
        except Exception as exc:
            print(f"skip {fld.name}: {exc}", file=sys.stderr)
            continue
        gx = day_gex(df)
        if gx.empty:
            continue
        gx.insert(0, "asof", fld.name)
        rows.append(gx)
        print(f"{fld.name}: {len(gx)} tickers")
    if not rows:
        print("no GEX reconstructed")
        return
    allg = pd.concat(rows, ignore_index=True)
    allg.to_csv(OUT, index=False)
    print(f"\nWrote {len(allg)} (asof,ticker) GEX rows across {allg['asof'].nunique()} days -> {OUT}")
    # market-level sanity: SPY net_gex_norm over time
    spy = allg[allg["ticker"] == "SPY"][["asof", "net_gex_norm", "gex_sign"]]
    print("\nSPY GEX sample (first/last 3 days):")
    print(spy.head(3).to_string(index=False))
    print(spy.tail(3).to_string(index=False))
    print("SPY net-gamma days: positive=%d negative=%d" % ((spy["gex_sign"] > 0).sum(), (spy["gex_sign"] < 0).sum()))


if __name__ == "__main__":
    main()
