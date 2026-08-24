"""Build a per-(date, ticker) feature panel from the UW stock-screener feed.

The pattern pipeline never reads iv_rank / iv30d / implied_move / 52w range /
aggressor-side volume, even though every one of them ships daily in
stock-screener-YYYY-MM-DD.zip. This extracts them into a single panel so they
can be joined onto any backtest by (signal_date, ticker).
"""

from __future__ import annotations

import argparse
import io
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

USECOLS = [
    "date", "ticker", "sector", "issue_type", "is_index", "marketcap",
    "close", "prev_close", "high", "low", "total_volume", "avg30_volume",
    "week_52_high", "week_52_low",
    "implied_move", "implied_move_perc", "volatility",
    "iv30d", "iv30d_1d", "iv30d_1w", "iv30d_1m", "iv_rank",
    "call_volume", "put_volume", "put_call_ratio",
    "call_premium", "put_premium", "bullish_premium", "bearish_premium",
    "net_call_premium", "net_put_premium",
    "call_volume_ask_side", "call_volume_bid_side",
    "put_volume_ask_side", "put_volume_bid_side",
    "avg_3_day_call_volume", "avg_3_day_put_volume",
    "avg_30_day_call_volume", "avg_30_day_put_volume",
    "call_open_interest", "put_open_interest", "total_open_interest",
    "prev_call_oi", "prev_put_oi",
    "avg_30_day_call_oi", "avg_30_day_put_oi",
    "next_earnings_date", "er_time",
]

NUMERIC = [c for c in USECOLS if c not in (
    "date", "ticker", "sector", "issue_type", "is_index",
    "next_earnings_date", "er_time")]


def _safe_div(a, b):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    return a.div(b.replace(0, np.nan))


def read_day(zip_path: Path) -> pd.DataFrame | None:
    try:
        with zipfile.ZipFile(zip_path) as zf:
            name = zf.namelist()[0]
            with zf.open(name) as fh:
                head = pd.read_csv(io.BytesIO(fh.read(1 << 16)), nrows=0)
        cols = [c for c in USECOLS if c in head.columns]
        with zipfile.ZipFile(zip_path) as zf:
            name = zf.namelist()[0]
            with zf.open(name) as fh:
                df = pd.read_csv(fh, usecols=cols, low_memory=False)
    except Exception as exc:  # noqa: BLE001 - feed files are occasionally truncated
        print(f"  !! {zip_path.name}: {exc}")
        return None
    for c in NUMERIC:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    out["ticker"] = df["ticker"].astype(str)
    out["sector"] = df.get("sector")
    out["issue_type"] = df.get("issue_type")
    out["marketcap"] = df.get("marketcap")
    out["close"] = df.get("close")

    # --- volatility pricing state (the entire missing dimension) ---
    out["iv_rank"] = df.get("iv_rank")
    out["iv30d"] = df.get("iv30d")
    out["rv30d"] = df.get("volatility")
    out["vrp"] = df.get("iv30d") - df.get("volatility")
    out["vrp_ratio"] = _safe_div(df.get("iv30d"), df.get("volatility"))
    out["iv_chg_1d"] = df.get("iv30d") - df.get("iv30d_1d")
    out["iv_chg_1w"] = df.get("iv30d") - df.get("iv30d_1w")
    out["iv_chg_1m"] = df.get("iv30d") - df.get("iv30d_1m")
    out["implied_move_perc"] = df.get("implied_move_perc")

    # --- price location / momentum ---
    rng = df.get("week_52_high") - df.get("week_52_low")
    out["pos_52w"] = _safe_div(df.get("close") - df.get("week_52_low"), rng)
    out["ret_1d"] = _safe_div(df.get("close"), df.get("prev_close")) - 1.0
    out["range_pct"] = _safe_div(df.get("high") - df.get("low"), df.get("close"))
    out["rel_volume"] = _safe_div(df.get("total_volume"), df.get("avg30_volume"))

    # --- option flow intensity (vs own baseline, not absolute) ---
    out["call_vol_ratio_30d"] = _safe_div(df.get("call_volume"), df.get("avg_30_day_call_volume"))
    out["put_vol_ratio_30d"] = _safe_div(df.get("put_volume"), df.get("avg_30_day_put_volume"))
    out["call_vol_ratio_3d"] = _safe_div(df.get("call_volume"), df.get("avg_3_day_call_volume"))
    out["put_vol_ratio_3d"] = _safe_div(df.get("put_volume"), df.get("avg_3_day_put_volume"))
    out["put_call_ratio"] = df.get("put_call_ratio")

    # --- aggressor imbalance: who is lifting offers vs hitting bids ---
    out["call_aggr"] = _safe_div(
        df.get("call_volume_ask_side") - df.get("call_volume_bid_side"), df.get("call_volume"))
    out["put_aggr"] = _safe_div(
        df.get("put_volume_ask_side") - df.get("put_volume_bid_side"), df.get("put_volume"))
    out["net_aggr"] = out["call_aggr"] - out["put_aggr"]

    # --- premium-weighted directional tilt ---
    tot_prem = df.get("bullish_premium").abs() + df.get("bearish_premium").abs()
    out["bull_bear_tilt"] = _safe_div(
        df.get("bullish_premium") - df.get("bearish_premium"), tot_prem)
    out["net_prem"] = df.get("net_call_premium") - df.get("net_put_premium")
    out["net_prem_norm"] = _safe_div(out["net_prem"], tot_prem)

    # --- open interest build (positioning, not just churn) ---
    out["call_oi_chg"] = _safe_div(df.get("call_open_interest"), df.get("prev_call_oi")) - 1.0
    out["put_oi_chg"] = _safe_div(df.get("put_open_interest"), df.get("prev_put_oi")) - 1.0
    out["call_oi_vs_30d"] = _safe_div(df.get("call_open_interest"), df.get("avg_30_day_call_oi"))
    out["put_oi_vs_30d"] = _safe_div(df.get("put_open_interest"), df.get("avg_30_day_put_oi"))
    # volume relative to existing OI => how much of today's trade is new money
    out["call_vol_to_oi"] = _safe_div(df.get("call_volume"), df.get("call_open_interest"))
    out["put_vol_to_oi"] = _safe_div(df.get("put_volume"), df.get("put_open_interest"))

    # --- events ---
    ed = pd.to_datetime(df.get("next_earnings_date"), errors="coerce")
    d0 = pd.to_datetime(df["date"], errors="coerce")
    out["days_to_earnings"] = (ed - d0).dt.days
    out["er_time"] = df.get("er_time")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default="/Users/anuppamvi/uw_root/tradedesk")
    ap.add_argument("--out", default="out/uw_panel.csv")
    ap.add_argument("--start", default="2026-01-01")
    ap.add_argument("--end", default="2026-12-31")
    args = ap.parse_args()

    base = Path(args.base_dir)
    days = sorted(
        p for p in base.glob("20??-??-??")
        if p.is_dir() and args.start <= p.name <= args.end
    )
    frames = []
    for d in days:
        z = d / f"stock-screener-{d.name}.zip"
        if not z.exists():
            continue
        raw = read_day(z)
        if raw is None or raw.empty:
            continue
        frames.append(engineer(raw))
        if len(frames) % 20 == 0:
            print(f"  {d.name}  days={len(frames)}  rows={sum(len(f) for f in frames)}")

    if not frames:
        raise SystemExit("no stock-screener files found")
    panel = pd.concat(frames, ignore_index=True)

    # cross-sectional percentile ranks computed per day -- relative value, the
    # way a desk would look at it, rather than absolute thresholds
    for col in ["vrp", "vrp_ratio", "iv_rank", "call_vol_ratio_30d", "put_vol_ratio_30d",
                "net_aggr", "bull_bear_tilt", "net_prem_norm", "rel_volume",
                "call_oi_chg", "put_oi_chg", "iv_chg_1w"]:
        panel[f"{col}_xs"] = panel.groupby("date")[col].rank(pct=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out, index=False)
    print(f"[panel] days={panel['date'].nunique()} rows={len(panel)} cols={panel.shape[1]} -> {out}")
    print(panel[["iv_rank", "vrp", "vrp_ratio", "pos_52w", "net_aggr"]].describe().round(4).to_string())


if __name__ == "__main__":
    main()
