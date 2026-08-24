"""Extract per-ticker/day UW signal features that the Options Agent currently ignores.

Reads the dated stock-screener and hot-chains exports and builds a point-in-time
feature panel keyed by (asof, ticker). Features follow the interpretation rules
published in the Unusual Whales FAQ and API skill doc:

  * CALL at ASK / PUT at BID  -> bullish opening pressure
  * CALL at BID / PUT at ASK  -> bearish opening pressure
  * volume > open_interest    -> opening (new position) rather than an unwind
  * multi-leg volume          -> spread prints, NOT directional conviction
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

SCREENER_COLS = [
    "ticker",
    "date",
    "call_volume",
    "put_volume",
    "call_premium",
    "put_premium",
    "bullish_premium",
    "bearish_premium",
    "net_call_premium",
    "net_put_premium",
    "call_volume_ask_side",
    "call_volume_bid_side",
    "put_volume_ask_side",
    "put_volume_bid_side",
    "avg_30_day_call_volume",
    "avg_30_day_put_volume",
    "call_open_interest",
    "put_open_interest",
    "prev_call_oi",
    "prev_put_oi",
    "iv30d",
    "iv30d_1d",
    "iv30d_1w",
    "iv30d_1m",
    "iv_rank",
    "volatility",
    "implied_move",
    "implied_move_perc",
    "close",
    "week_52_high",
    "week_52_low",
    "marketcap",
    "total_volume",
    "avg30_volume",
]

HOT_COLS = [
    "ticker",
    "option_symbol",
    "volume",
    "open_interest",
    "premium",
    "ask_side_volume",
    "bid_side_volume",
    "multileg_volume",
    "sweep_volume",
    "floor_volume",
    "cross_volume",
    "neutral_volume",
    "total_bid_changes",
    "total_ask_changes",
]


def _read_zip(path: Path, usecols: list[str]) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as fh:
            head = pd.read_csv(fh, nrows=0)
        available = [c for c in usecols if c in head.columns]
        with zf.open(name) as fh:
            return pd.read_csv(fh, usecols=available, low_memory=False)


def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    den = den.replace(0, np.nan)
    return (num / den).replace([np.inf, -np.inf], np.nan)


def screener_features(day_dir: Path, asof: str) -> pd.DataFrame | None:
    hits = list(day_dir.glob("stock-screener-*.zip"))
    if not hits:
        return None
    df = _read_zip(sorted(hits)[-1], SCREENER_COLS)
    if df.empty or "ticker" not in df.columns:
        return None
    for c in df.columns:
        if c not in {"ticker", "date"}:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    out = pd.DataFrame({"asof": asof, "ticker": df["ticker"].astype(str).str.upper()})

    # --- directional pressure from the four side-tagged volumes (all unused today)
    call_ask, call_bid = df.get("call_volume_ask_side"), df.get("call_volume_bid_side")
    put_ask, put_bid = df.get("put_volume_ask_side"), df.get("put_volume_bid_side")
    tot = df["call_volume"].fillna(0) + df["put_volume"].fillna(0)
    out["call_ask_ratio"] = _safe_ratio(call_ask, df["call_volume"])
    out["put_ask_ratio"] = _safe_ratio(put_ask, df["put_volume"])
    # UW rule: call@ask bullish, call@bid bearish, put@ask bearish, put@bid bullish
    bull = call_ask.fillna(0) + put_bid.fillna(0)
    bear = call_bid.fillna(0) + put_ask.fillna(0)
    out["side_bull_ratio"] = _safe_ratio(bull, bull + bear)
    out["side_net_pressure"] = _safe_ratio(bull - bear, tot)

    # --- net premium (UW "Market Tide" construction), scale-normalised
    ncp, npp = df.get("net_call_premium"), df.get("net_put_premium")
    out["net_call_premium"] = ncp
    out["net_put_premium"] = npp
    out["net_prem_dir"] = _safe_ratio(ncp.fillna(0) - npp.fillna(0), ncp.abs().fillna(0) + npp.abs().fillna(0))
    out["net_call_prem_per_mcap"] = _safe_ratio(ncp, df.get("marketcap"))
    out["bull_prem_ratio"] = _safe_ratio(
        df.get("bullish_premium"), df.get("bullish_premium").fillna(0) + df.get("bearish_premium").fillna(0)
    )

    # --- IV term structure / momentum (entirely unused today)
    iv = df.get("iv30d")
    out["iv30d"] = iv
    out["iv_rank"] = df.get("iv_rank")
    out["iv_chg_1d"] = iv - df.get("iv30d_1d")
    out["iv_chg_1w"] = iv - df.get("iv30d_1w")
    out["iv_chg_1m"] = iv - df.get("iv30d_1m")
    out["iv_vs_rv"] = _safe_ratio(iv, df.get("volatility"))
    out["implied_move_perc"] = df.get("implied_move_perc")

    # --- activity surge vs 30d baseline
    out["call_vol_surge"] = _safe_ratio(df["call_volume"], df.get("avg_30_day_call_volume"))
    out["put_vol_surge"] = _safe_ratio(df["put_volume"], df.get("avg_30_day_put_volume"))
    out["stock_vol_surge"] = _safe_ratio(df.get("total_volume"), df.get("avg30_volume"))
    out["put_call_vol"] = _safe_ratio(df["put_volume"], df["call_volume"])

    # --- open interest build (positioning carry-over)
    out["call_oi_chg"] = _safe_ratio(
        df.get("call_open_interest") - df.get("prev_call_oi"), df.get("prev_call_oi")
    )
    out["put_oi_chg"] = _safe_ratio(
        df.get("put_open_interest") - df.get("prev_put_oi"), df.get("prev_put_oi")
    )

    # --- location in 52w range (upside room for a short call spread)
    hi, lo, close = df.get("week_52_high"), df.get("week_52_low"), df.get("close")
    out["pct_to_52w_high"] = _safe_ratio(hi - close, close)
    out["range_pos_52w"] = _safe_ratio(close - lo, hi - lo)
    out["marketcap"] = df.get("marketcap")
    return out


def hot_chain_features(day_dir: Path, asof: str) -> pd.DataFrame | None:
    hits = list(day_dir.glob("hot-chains-*.zip"))
    if not hits:
        return None
    df = _read_zip(sorted(hits)[-1], HOT_COLS)
    if df.empty or "option_symbol" not in df.columns:
        return None
    # hot-chains has no ticker column; the root of the OCC symbol is the underlying
    df["ticker"] = (
        df["option_symbol"].astype(str).str.upper().str.extract(r"^([A-Z]+)\d{6}[CP]\d+$")[0]
    )
    df = df[df["ticker"].notna()]
    if df.empty:
        return None
    for c in df.columns:
        if c not in {"ticker", "option_symbol"}:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    vol = df["volume"].fillna(0)
    oi = df["open_interest"].fillna(0)
    # UW screener recipe: opening conviction = vol > OI, at-ask, and NOT multi-leg
    df["_opening_vol"] = np.where(vol > oi, vol, 0.0)
    df["_vol"] = vol
    for src, dst in [
        ("ask_side_volume", "_ask"),
        ("multileg_volume", "_ml"),
        ("sweep_volume", "_sweep"),
        ("floor_volume", "_floor"),
        ("total_ask_changes", "_askchg"),
        ("total_bid_changes", "_bidchg"),
    ]:
        df[dst] = df[src].fillna(0) if src in df.columns else 0.0

    g = df.groupby("ticker", sort=False).agg(
        _v=("_vol", "sum"),
        _a=("_ask", "sum"),
        _m=("_ml", "sum"),
        _s=("_sweep", "sum"),
        _f=("_floor", "sum"),
        _o=("_opening_vol", "sum"),
        _ac=("_askchg", "sum"),
        _bc=("_bidchg", "sum"),
        chain_premium=("premium", "sum"),
        chain_n=("_vol", "size"),
    )
    v = g["_v"].replace(0, np.nan)
    out = pd.DataFrame(
        {
            "asof": asof,
            "ticker": g.index,
            "chain_ask_perc": (g["_a"] / v).values,
            "chain_multileg_ratio": (g["_m"] / v).values,
            "chain_sweep_ratio": (g["_s"] / v).values,
            "chain_floor_ratio": (g["_f"] / v).values,
            "chain_opening_ratio": (g["_o"] / v).values,
            "chain_quote_churn": ((g["_ac"] + g["_bc"]) / v).values,
            "chain_premium": g["chain_premium"].values,
            "chain_n": g["chain_n"].values,
        }
    )
    return out.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/Users/anuppamvi/uw_root/tradedesk")
    ap.add_argument("--start", default="2026-01-02")
    ap.add_argument("--end", default="2026-07-24")
    ap.add_argument("--out", default="/Users/anuppamvi/uw_root/tradedesk/out/research/uw_features.csv.gz")
    args = ap.parse_args()

    root = Path(args.root)
    days = sorted(p for p in root.glob("2026-*") if p.is_dir() and args.start <= p.name <= args.end)
    print(f"scanning {len(days)} dated folders")

    frames = []
    for i, day in enumerate(days, 1):
        asof = day.name
        try:
            sc = screener_features(day, asof)
            if sc is None:
                continue
            hc = hot_chain_features(day, asof)
            if hc is not None:
                sc = sc.merge(hc.drop(columns=["asof"]), on="ticker", how="left")
            frames.append(sc)
        except Exception as exc:  # noqa: BLE001
            print(f"  {asof}: {exc}")
        if i % 25 == 0:
            print(f"  ...{i}/{len(days)}")

    panel = pd.concat(frames, ignore_index=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out, index=False, compression="gzip")
    print(f"wrote {len(panel):,} rows x {panel.shape[1]} cols -> {out}")
    print(f"days: {panel['asof'].nunique()}  tickers: {panel['ticker'].nunique()}")


if __name__ == "__main__":
    main()
