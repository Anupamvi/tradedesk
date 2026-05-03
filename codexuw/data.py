from __future__ import annotations

import datetime as dt
import math
import zipfile
from pathlib import Path
from typing import Iterable

import pandas as pd

from .occ import parse_occ_symbol


def safe_float(value: object, default: float = math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def infer_asof_date(base_dir: Path) -> dt.date:
    try:
        return dt.datetime.strptime(base_dir.name[:10], "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"Cannot infer YYYY-MM-DD asof date from {base_dir}") from exc


def dte_from_expiry(value: object, asof: dt.date) -> float:
    if pd.isna(value):
        return math.nan
    if isinstance(value, dt.datetime):
        return float((value.date() - asof).days)
    if isinstance(value, dt.date):
        return float((value - asof).days)
    return math.nan


def find_export(base_dir: Path, prefix: str) -> Path:
    candidates = sorted(base_dir.glob(f"{prefix}*.csv")) + sorted(base_dir.glob(f"{prefix}*.zip"))
    if not candidates:
        unzipped = base_dir / "_unzipped_mode_a"
        candidates = sorted(unzipped.glob(f"{prefix}*.csv")) + sorted(unzipped.glob(f"{prefix}*.zip"))
    if not candidates:
        raise FileNotFoundError(f"No {prefix}*.csv or {prefix}*.zip found under {base_dir}")
    live_names = ("latest", "current", "live", "next")
    live_candidates = [path for path in candidates if any(token in path.name.lower() for token in live_names)]
    if live_candidates:
        return sorted(live_candidates, key=lambda path: (path.stat().st_mtime, path.name), reverse=True)[0]
    return candidates[0]


def read_csv_export(path: Path, **kwargs) -> pd.DataFrame:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            members = [name for name in zf.namelist() if name.lower().endswith(".csv")]
            if not members:
                raise FileNotFoundError(f"No CSV member in {path}")
            with zf.open(members[0]) as handle:
                return pd.read_csv(handle, **kwargs)
    return pd.read_csv(path, **kwargs)


def iter_csv_export(path: Path, **kwargs):
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            members = [name for name in zf.namelist() if name.lower().endswith(".csv")]
            if not members:
                raise FileNotFoundError(f"No CSV member in {path}")
            with zf.open(members[0]) as handle:
                yield from pd.read_csv(handle, **kwargs)
    else:
        yield from pd.read_csv(path, **kwargs)


def load_stock_screener(base_dir: Path) -> pd.DataFrame:
    path = find_export(base_dir, "stock-screener-")
    df = read_csv_export(path)
    numeric_cols = [
        "call_volume",
        "put_volume",
        "call_premium",
        "put_premium",
        "bearish_premium",
        "bullish_premium",
        "net_call_premium",
        "net_put_premium",
        "total_open_interest",
        "close",
        "high",
        "low",
        "total_volume",
        "avg30_volume",
        "prev_close",
        "week_52_high",
        "week_52_low",
        "implied_move",
        "implied_move_perc",
        "volatility",
        "iv30d",
        "iv_rank",
        "marketcap",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["flow_total_premium"] = df.get("bullish_premium", 0).fillna(0) + df.get("bearish_premium", 0).fillna(0)
    denom = df["flow_total_premium"].where(df["flow_total_premium"].abs() > 0)
    df["flow_bias"] = (df.get("bullish_premium", 0).fillna(0) - df.get("bearish_premium", 0).fillna(0)) / denom
    df["next_earnings_dt"] = pd.to_datetime(df.get("next_earnings_date", pd.Series(index=df.index)), errors="coerce").dt.date
    return df


def load_hot_chains(base_dir: Path, asof: dt.date) -> pd.DataFrame:
    path = find_export(base_dir, "hot-chains-")
    df = read_csv_export(path)
    parsed = df["option_symbol"].map(parse_occ_symbol)
    df["ticker"] = parsed.map(lambda x: x.root if x else "")
    df["expiry_dt"] = parsed.map(lambda x: x.expiry if x else pd.NaT)
    df["right"] = parsed.map(lambda x: x.right if x else "")
    df["strike"] = parsed.map(lambda x: x.strike if x else math.nan)
    df = df[df["ticker"].astype(bool)].copy()
    df["dte"] = df["expiry_dt"].map(lambda x: dte_from_expiry(x, asof))
    for col in ["volume", "open_interest", "premium", "bid", "ask", "iv", "close.1"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["mid"] = (df["bid"].fillna(0) + df["ask"].fillna(0)) / 2.0
    df["spread"] = df["ask"].fillna(0) - df["bid"].fillna(0)
    df["spread_pct_mid"] = df["spread"] / df["mid"].where(df["mid"].abs() > 0)
    df["next_earnings_dt"] = pd.to_datetime(df.get("next_earnings_date", pd.Series(index=df.index)), errors="coerce").dt.date
    return df


def load_chain_oi(base_dir: Path, asof: dt.date) -> pd.DataFrame:
    path = find_export(base_dir, "chain-oi-changes-")
    df = read_csv_export(path)
    parsed = df["option_symbol"].map(parse_occ_symbol)
    df["ticker"] = parsed.map(lambda x: x.root if x else df.get("underlying_symbol", ""))
    df["expiry_dt"] = parsed.map(lambda x: x.expiry if x else pd.NaT)
    df["right"] = parsed.map(lambda x: x.right if x else "")
    df["strike"] = parsed.map(lambda x: x.strike if x else math.nan)
    df["dte"] = df["expiry_dt"].map(lambda x: dte_from_expiry(x, asof))
    for col in ["oi_change", "curr_oi", "volume", "last_fill", "last_bid", "last_ask", "prev_total_premium"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def aggregate_bot_flow(
    base_dir: Path,
    tickers: Iterable[str],
    *,
    chunksize: int = 750_000,
    max_rows: int | None = None,
) -> pd.DataFrame:
    path = find_export(base_dir, "bot-eod-report-")
    wanted = {str(t).upper().strip() for t in tickers if str(t).strip()}
    usecols = [
        "underlying_symbol",
        "side",
        "option_type",
        "expiry",
        "strike",
        "premium",
        "size",
        "volume",
        "open_interest",
        "delta",
        "canceled",
    ]
    rows_seen = 0
    parts = []
    for chunk in iter_csv_export(path, usecols=usecols, chunksize=chunksize):
        rows_seen += len(chunk)
        chunk["underlying_symbol"] = chunk["underlying_symbol"].astype(str).str.upper().str.strip()
        if wanted:
            chunk = chunk[chunk["underlying_symbol"].isin(wanted)]
        if "canceled" in chunk.columns:
            chunk = chunk[chunk["canceled"].astype(str).str.lower().ne("t")]
        if chunk.empty:
            if max_rows and rows_seen >= max_rows:
                break
            continue
        chunk["bot_premium"] = pd.to_numeric(chunk["premium"], errors="coerce").fillna(0)
        side = chunk["side"].astype(str).str.lower()
        opt_type = chunk["option_type"].astype(str).str.lower()
        bull_mask = ((opt_type == "call") & (side == "ask")) | ((opt_type == "put") & (side == "bid"))
        bear_mask = ((opt_type == "call") & (side == "bid")) | ((opt_type == "put") & (side == "ask"))
        chunk["bot_bull_premium"] = chunk["bot_premium"].where(bull_mask, 0.0)
        chunk["bot_bear_premium"] = chunk["bot_premium"].where(bear_mask, 0.0)
        chunk["bot_trades"] = 1
        agg = chunk.groupby("underlying_symbol", as_index=False).agg(
            bot_bull_premium=("bot_bull_premium", "sum"),
            bot_bear_premium=("bot_bear_premium", "sum"),
            bot_total_premium=("bot_premium", "sum"),
            bot_trades=("bot_trades", "sum"),
        )
        parts.append(agg)
        if max_rows and rows_seen >= max_rows:
            break
    if not parts:
        return pd.DataFrame(columns=["ticker", "bot_bull_premium", "bot_bear_premium", "bot_total_premium", "bot_trades", "bot_flow_bias"])
    out = pd.concat(parts, ignore_index=True).groupby("underlying_symbol", as_index=False).sum()
    out = out.rename(columns={"underlying_symbol": "ticker"})
    denom = out["bot_total_premium"].where(out["bot_total_premium"].abs() > 0)
    out["bot_flow_bias"] = (out["bot_bull_premium"] - out["bot_bear_premium"]) / denom
    return out
