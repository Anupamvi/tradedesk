"""Reconstruct per-day, per-ticker IV-surface / skew signals from local hot-chains.

hot-chains has per-contract `iv` + OCC `option_symbol` (underlying, strike, C/P).
Spot comes from chain-oi-changes (stock_price per underlying). We build genuine
forward-looking risk features: ATM IV level, put skew, call skew, risk-reversal
(25d proxy), and IV term slope. Local data only; no browser, no Schwab.

Output: one row per (asof, ticker).
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
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "iv_skew_reconstructed.csv"

# OCC: ROOT (alpha, may include digits after) + YYMMDD + C/P + strike(8) => strike/1000
OCC_RE = re.compile(r"^([A-Z]+)(\d{6})([CP])(\d{8})$")


def parse_occ(sym: pd.Series):
    s = sym.astype(str).str.replace(" ", "", regex=False)
    ext = s.str.extract(OCC_RE)
    ext.columns = ["under", "exp", "cp", "strike_raw"]
    ext["strike"] = pd.to_numeric(ext["strike_raw"], errors="coerce") / 1000.0
    ext["exp_dt"] = pd.to_datetime(ext["exp"], format="%y%m%d", errors="coerce")
    return ext


def spot_map(fld: Path) -> dict:
    z = fld / f"chain-oi-changes-{fld.name}.zip"
    if not z.exists():
        return {}
    try:
        with zipfile.ZipFile(z) as zf:
            name = next((n for n in zf.namelist() if n.endswith(".csv")), None)
            with zf.open(name) as fh:
                df = pd.read_csv(io.TextIOWrapper(fh, "utf-8"), low_memory=False,
                                 usecols=["underlying_symbol", "stock_price"])
        df["stock_price"] = pd.to_numeric(df["stock_price"], errors="coerce")
        return df.dropna().groupby("underlying_symbol")["stock_price"].median().to_dict()
    except Exception:
        return {}


def day_skew(fld: Path) -> pd.DataFrame:
    z = fld / f"hot-chains-{fld.name}.zip"
    if not z.exists():
        return pd.DataFrame()
    try:
        with zipfile.ZipFile(z) as zf:
            name = next((n for n in zf.namelist() if n.endswith(".csv")), None)
            with zf.open(name) as fh:
                df = pd.read_csv(io.TextIOWrapper(fh, "utf-8"), low_memory=False,
                                 usecols=["option_symbol", "iv", "open_interest"])
    except Exception as exc:
        print(f"skip {fld.name}: {exc}", file=sys.stderr)
        return pd.DataFrame()
    occ = parse_occ(df["option_symbol"])
    df = pd.concat([df, occ], axis=1)
    df["iv"] = pd.to_numeric(df["iv"], errors="coerce")
    df["oi"] = pd.to_numeric(df["open_interest"], errors="coerce").fillna(0.0)
    asof = pd.to_datetime(fld.name)
    df["dte"] = (df["exp_dt"] - asof).dt.days
    df = df[df["under"].notna() & df["iv"].between(0.01, 5.0) & df["strike"].gt(0)
            & df["dte"].between(3, 90)]
    if df.empty:
        return pd.DataFrame()
    sm = spot_map(fld)
    df["spot"] = df["under"].map(sm)
    df = df[df["spot"].gt(0)]
    if df.empty:
        return pd.DataFrame()
    df["m"] = df["strike"] / df["spot"] - 1.0  # moneyness (0=ATM)
    out = []
    for u, g in df.groupby("under"):
        atm = g[g["m"].abs() <= 0.02]
        atm_iv = np.average(atm["iv"], weights=atm["oi"] + 1) if len(atm) else np.nan
        if not np.isfinite(atm_iv):
            atm_iv = g["iv"].median()
        otm_put = g[(g["cp"] == "P") & (g["m"].between(-0.12, -0.03))]
        otm_call = g[(g["cp"] == "C") & (g["m"].between(0.03, 0.12))]
        pv = np.average(otm_put["iv"], weights=otm_put["oi"] + 1) if len(otm_put) else np.nan
        cv = np.average(otm_call["iv"], weights=otm_call["oi"] + 1) if len(otm_call) else np.nan
        near = g[g["dte"] <= 30]["iv"].median()
        far = g[g["dte"] > 30]["iv"].median()
        out.append({
            "ticker": u,
            "atm_iv": atm_iv,
            "put_skew": (pv - atm_iv) if np.isfinite(pv) else np.nan,
            "call_skew": (cv - atm_iv) if np.isfinite(cv) else np.nan,
            "risk_reversal": (pv - cv) if np.isfinite(pv) and np.isfinite(cv) else np.nan,
            "iv_term_slope": (far - near) if np.isfinite(far) and np.isfinite(near) else np.nan,
            "n_contracts": len(g),
        })
    res = pd.DataFrame(out)
    res.insert(0, "asof", fld.name)
    return res


def main():
    folders = sorted(p for p in ROOT.glob("2026-*")
                     if p.is_dir() and re.fullmatch(r"2026-\d\d-\d\d", p.name))
    rows = []
    for fld in folders:
        r = day_skew(fld)
        if not r.empty:
            rows.append(r)
            print(f"{fld.name}: {len(r)} tickers")
    if not rows:
        print("no IV skew reconstructed")
        return
    allg = pd.concat(rows, ignore_index=True)
    allg.to_csv(OUT, index=False)
    print(f"\nWrote {len(allg)} (asof,ticker) rows across {allg['asof'].nunique()} days -> {OUT}")
    spy = allg[allg["ticker"] == "SPY"][["asof", "atm_iv", "put_skew", "risk_reversal"]]
    print("\nSPY IV/skew sample:")
    print(spy.head(3).to_string(index=False))
    print(spy.tail(3).to_string(index=False))


if __name__ == "__main__":
    main()
