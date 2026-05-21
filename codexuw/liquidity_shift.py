from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from .data import find_export, iter_csv_export, load_stock_screener, safe_float


INDEX_FLOW_TICKERS = {
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    "SMH",
    "SOXX",
    "XLK",
    "XLF",
    "XLE",
    "XLV",
    "XLY",
    "XLI",
    "XLC",
    "XBI",
}
FIXED_LIQUID_UNIVERSE = {
    "AAPL",
    "AMD",
    "AMZN",
    "AVGO",
    "COIN",
    "GOOGL",
    "IWM",
    "META",
    "MSFT",
    "NVDA",
    "QQQ",
    "SMCI",
    "SMH",
    "SPY",
    "TSLA",
    "XLK",
}
SECTOR_BENCHMARKS = {
    "technology": "XLK",
    "communication": "XLC",
    "consumer cyclical": "XLY",
    "consumer defensive": "XLP",
    "financial": "XLF",
    "energy": "XLE",
    "healthcare": "XLV",
    "industrial": "XLI",
    "materials": "XLB",
    "utilities": "XLU",
    "semiconductor": "SMH",
}
SEMI_TICKERS = {"NVDA", "AMD", "AVGO", "TSM", "MU", "SMCI", "ARM", "ASML", "AMAT", "LRCX", "KLAC"}
BOT_USECOLS = {
    "executed_at",
    "underlying_symbol",
    "side",
    "strike",
    "option_type",
    "expiry",
    "underlying_price",
    "size",
    "premium",
    "volume",
    "open_interest",
    "delta",
    "gamma",
    "report_flags",
    "upstream_condition_detail",
    "canceled",
    "sector",
}


def _clean_ticker(value: object) -> str:
    return str(value or "").upper().strip()


def _date_value(value: object) -> dt.date | None:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def _flow_direction(side: object, option_type: object) -> str:
    side_text = str(side or "").lower()
    opt = str(option_type or "").lower()
    is_ask = "ask" in side_text
    is_bid = "bid" in side_text
    if opt == "call" and is_ask:
        return "bullish"
    if opt == "call" and is_bid:
        return "bearish"
    if opt == "put" and is_ask:
        return "bearish"
    if opt == "put" and is_bid:
        return "bullish"
    return "unclear"


def _strategy_family(row: pd.Series) -> str:
    flags = f"{row.get('report_flags', '')} {row.get('upstream_condition_detail', '')}".lower()
    opt = str(row.get("option_type") or "").lower()
    side = str(row.get("side") or "").lower()
    if "stock" in flags:
        return "stock_linked"
    if any(token in flags for token in ["multi", "spread", "floor", "cross"]):
        return "spread_leg"
    if opt in {"call", "put"} and "ask" in side:
        return f"buyer_{opt}"
    if opt in {"call", "put"} and "bid" in side:
        return f"seller_{opt}"
    return "unclear"


def _prepare_bot_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    if "canceled" in chunk.columns:
        chunk = chunk[chunk["canceled"].astype(str).str.lower().ne("t")].copy()
    if chunk.empty:
        return chunk
    for col in BOT_USECOLS:
        if col not in chunk.columns:
            chunk[col] = ""
    chunk["ticker"] = chunk["underlying_symbol"].map(_clean_ticker)
    chunk = chunk[chunk["ticker"].astype(bool)].copy()
    if chunk.empty:
        return chunk
    chunk["executed_at"] = pd.to_datetime(chunk.get("executed_at"), errors="coerce", utc=True)
    chunk["minute"] = chunk["executed_at"].dt.floor("min")
    chunk["expiry_dt"] = pd.to_datetime(chunk.get("expiry"), errors="coerce").dt.date
    for col in ["premium", "size", "volume", "open_interest", "strike", "underlying_price", "delta", "gamma"]:
        chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
    chunk["abs_premium"] = chunk["premium"].abs().fillna(0.0)
    directions = chunk.apply(lambda row: _flow_direction(row.get("side"), row.get("option_type")), axis=1)
    chunk["flow_direction"] = directions
    sign = directions.map({"bullish": 1.0, "bearish": -1.0}).fillna(0.0)
    chunk["signed_premium"] = chunk["abs_premium"] * sign
    chunk["side_key"] = (
        chunk["option_type"].astype(str).str.lower().str[:1].str.upper()
        + "_"
        + chunk["side"].astype(str).str.lower()
    )
    chunk["strategy_family"] = chunk.apply(_strategy_family, axis=1)
    return chunk


def load_bot_flow_events(base_dir: Path, *, max_rows: int | None = None, chunksize: int = 500_000) -> pd.DataFrame:
    path = find_export(base_dir, "bot-eod-report-")
    parts: list[pd.DataFrame] = []
    rows_seen = 0
    usecols = lambda col: col in BOT_USECOLS
    for chunk in iter_csv_export(path, usecols=usecols, chunksize=chunksize):
        rows_seen += len(chunk)
        if max_rows is not None:
            remaining = max_rows - sum(len(part) for part in parts)
            if remaining <= 0:
                break
            chunk = chunk.head(remaining).copy()
        chunk = _prepare_bot_chunk(chunk)
        if chunk.empty:
            if max_rows is not None and rows_seen >= max_rows:
                break
            continue
        parts.append(chunk)
        if max_rows is not None and sum(len(part) for part in parts) >= max_rows:
            break
    if not parts:
        return pd.DataFrame(columns=["ticker", "executed_at", "minute", "expiry_dt", "strike", "abs_premium", "signed_premium"])
    return pd.concat(parts, ignore_index=True, sort=False)


def scan_bot_flow_tape(base_dir: Path, *, asof: dt.date, max_rows: int | None = None, chunksize: int = 500_000) -> dict[str, Any]:
    path = find_export(base_dir, "bot-eod-report-")
    usecols = lambda col: col in BOT_USECOLS
    ticker_parts: list[pd.DataFrame] = []
    minute_parts: list[pd.DataFrame] = []
    zero_dte_parts: list[pd.DataFrame] = []
    kept_rows = 0
    raw_rows = 0
    for chunk in iter_csv_export(path, usecols=usecols, chunksize=chunksize):
        raw_rows += len(chunk)
        if max_rows is not None:
            remaining = max_rows - kept_rows
            if remaining <= 0:
                break
            chunk = chunk.head(remaining).copy()
        chunk = _prepare_bot_chunk(chunk)
        if chunk.empty:
            continue
        kept_rows += len(chunk)
        chunk["vwap_num"] = chunk["underlying_price"].fillna(0.0) * chunk["abs_premium"].fillna(0.0)
        ticker_parts.append(
            chunk.groupby("ticker", as_index=False).agg(
                net_premium=("signed_premium", "sum"),
                total_premium=("abs_premium", "sum"),
                trade_count=("abs_premium", "size"),
                volume=("volume", "sum"),
                open_interest=("open_interest", "sum"),
                vwap_num=("vwap_num", "sum"),
                vwap_weight=("abs_premium", "sum"),
                last_underlying_price=("underlying_price", "last"),
                sector=("sector", "last"),
            )
        )
        minute_parts.append(
            chunk.groupby(["ticker", "expiry_dt", "strike", "side_key", "strategy_family", "minute"], dropna=False, as_index=False).agg(
                abs_premium=("abs_premium", "sum"),
                signed_premium=("signed_premium", "sum"),
                trade_count=("abs_premium", "size"),
                max_trade_premium=("abs_premium", "max"),
                first_timestamp=("executed_at", "min"),
                last_timestamp=("executed_at", "max"),
            )
        )
        z = chunk[chunk["ticker"].isin(INDEX_FLOW_TICKERS) & chunk["expiry_dt"].eq(asof)].copy()
        if not z.empty:
            zero_dte_parts.append(
                z.groupby(["ticker", "strike"], dropna=False, as_index=False).agg(
                    total_premium=("abs_premium", "sum"),
                    net_premium=("signed_premium", "sum"),
                    volume=("volume", "sum"),
                    open_interest=("open_interest", "sum"),
                    gamma=("gamma", "sum"),
                    spot=("underlying_price", "last"),
                )
            )
        if max_rows is not None and kept_rows >= max_rows:
            break

    if ticker_parts:
        ticker_summary = pd.concat(ticker_parts, ignore_index=True).groupby("ticker", as_index=False).agg(
            net_premium=("net_premium", "sum"),
            total_premium=("total_premium", "sum"),
            trade_count=("trade_count", "sum"),
            volume=("volume", "sum"),
            open_interest=("open_interest", "sum"),
            vwap_num=("vwap_num", "sum"),
            vwap_weight=("vwap_weight", "sum"),
            last_underlying_price=("last_underlying_price", "last"),
            sector=("sector", "last"),
        )
    else:
        ticker_summary = pd.DataFrame()
    if minute_parts:
        minute_bars = pd.concat(minute_parts, ignore_index=True).groupby(
            ["ticker", "expiry_dt", "strike", "side_key", "strategy_family", "minute"], dropna=False, as_index=False
        ).agg(
            abs_premium=("abs_premium", "sum"),
            signed_premium=("signed_premium", "sum"),
            trade_count=("trade_count", "sum"),
            max_trade_premium=("max_trade_premium", "max"),
            first_timestamp=("first_timestamp", "min"),
            last_timestamp=("last_timestamp", "max"),
        )
    else:
        minute_bars = pd.DataFrame()
    if zero_dte_parts:
        zero_dte_bars = pd.concat(zero_dte_parts, ignore_index=True).groupby(["ticker", "strike"], dropna=False, as_index=False).agg(
            total_premium=("total_premium", "sum"),
            net_premium=("net_premium", "sum"),
            volume=("volume", "sum"),
            open_interest=("open_interest", "sum"),
            gamma=("gamma", "sum"),
            spot=("spot", "last"),
        )
    else:
        zero_dte_bars = pd.DataFrame()
    return {
        "raw_rows": int(raw_rows),
        "event_rows": int(kept_rows),
        "ticker_summary": ticker_summary,
        "minute_bars": minute_bars,
        "zero_dte_bars": zero_dte_bars,
    }


def compute_volatility_thresholds(regime: dict[str, Any] | None) -> dict[str, Any]:
    vix = safe_float((regime or {}).get("vix_proxy"))
    vol_label = str((regime or {}).get("volatility") or "").lower()
    if math.isfinite(vix):
        if vix < 18:
            label = "low"
        elif vix < 25:
            label = "medium"
        else:
            label = "high"
    elif vol_label in {"low", "medium", "high"}:
        label = vol_label
    else:
        label = "medium"
    if label == "low":
        premium_5m = 350_000.0
        premium_15m = 750_000.0
        child_count = 6
        volume_oi = 0.25
        why = "low VIX/volatility regime lowers unusual-flow thresholds because baseline movement is smaller"
    elif label == "high":
        premium_5m = 1_250_000.0
        premium_15m = 2_500_000.0
        child_count = 14
        volume_oi = 0.75
        why = "high VIX/volatility regime raises unusual-flow thresholds to avoid noise"
    else:
        premium_5m = 750_000.0
        premium_15m = 1_500_000.0
        child_count = 10
        volume_oi = 0.45
        why = "medium volatility regime uses baseline unusual-flow thresholds"
    return {
        "volatility_regime": label,
        "vix_proxy": round(vix, 2) if math.isfinite(vix) else None,
        "premium_5m_threshold": premium_5m,
        "premium_15m_threshold": premium_15m,
        "child_order_count_threshold": child_count,
        "volume_oi_ratio_threshold": volume_oi,
        "why": why,
    }


def build_flow_velocity(events: pd.DataFrame, thresholds: dict[str, Any]) -> pd.DataFrame:
    columns = [
        "ticker",
        "expiry",
        "strike",
        "side",
        "strategy_family",
        "total_premium",
        "net_premium",
        "trade_count",
        "premium_per_minute",
        "rolling_5m_premium",
        "rolling_15m_premium",
        "child_order_accumulation",
        "flow_velocity_signal",
        "flow_direction",
        "first_timestamp",
        "last_timestamp",
    ]
    if events.empty:
        return pd.DataFrame(columns=columns)
    df = events.copy()
    df = df[df["abs_premium"].fillna(0.0) > 0].copy()
    if df.empty:
        return pd.DataFrame(columns=columns)
    ticker_totals = df.groupby("ticker")["abs_premium"].sum().sort_values(ascending=False)
    if len(ticker_totals) > 100:
        df = df[df["ticker"].isin(set(ticker_totals.head(100).index))].copy()
    keys = ["ticker", "expiry_dt", "strike", "side_key", "strategy_family"]
    group_totals = (
        df.groupby(keys, dropna=False)
        .agg(total_premium=("abs_premium", "sum"), trade_count=("abs_premium", "size"))
        .reset_index()
        .sort_values("total_premium", ascending=False)
    )
    eligible = group_totals[
        (group_totals["total_premium"] >= safe_float(thresholds["premium_5m_threshold"]) * 0.25)
        | (group_totals["trade_count"] >= int(thresholds["child_order_count_threshold"]))
    ].head(5000)
    if eligible.empty:
        eligible = group_totals.head(500)
    df = df.merge(eligible[keys], on=keys, how="inner")
    rows: list[dict[str, Any]] = []
    for key, part in df.groupby(keys, dropna=False):
        part = part.sort_values("executed_at")
        total = float(part["abs_premium"].sum())
        net = float(part["signed_premium"].sum())
        trade_count = int(len(part))
        valid_time = part["executed_at"].notna().all()
        if valid_time:
            indexed = part.set_index("executed_at").sort_index()
            rolling_5m = float(indexed["abs_premium"].rolling("5min").sum().max())
            rolling_15m = float(indexed["abs_premium"].rolling("15min").sum().max())
            minutes = max(1.0, (indexed.index.max() - indexed.index.min()).total_seconds() / 60.0)
            first_ts = indexed.index.min().isoformat()
            last_ts = indexed.index.max().isoformat()
        else:
            rolling_5m = total
            rolling_15m = total
            minutes = 1.0
            first_ts = ""
            last_ts = ""
        max_trade = safe_float(part["abs_premium"].max(), 0.0)
        avg_trade = total / max(1, trade_count)
        child_order = (
            trade_count >= int(thresholds["child_order_count_threshold"])
            and rolling_15m >= safe_float(thresholds["premium_15m_threshold"])
            and max_trade <= max(total * 0.35, avg_trade * 4.0)
        )
        velocity_signal = rolling_5m >= safe_float(thresholds["premium_5m_threshold"]) or rolling_15m >= safe_float(
            thresholds["premium_15m_threshold"]
        )
        direction = "bullish" if net > 0 else "bearish" if net < 0 else "unclear"
        rows.append(
            {
                "ticker": key[0],
                "expiry": str(key[1]) if key[1] else "",
                "strike": key[2],
                "side": key[3],
                "strategy_family": key[4],
                "total_premium": round(total, 2),
                "net_premium": round(net, 2),
                "trade_count": trade_count,
                "premium_per_minute": round(total / minutes, 2),
                "rolling_5m_premium": round(rolling_5m, 2),
                "rolling_15m_premium": round(rolling_15m, 2),
                "child_order_accumulation": bool(child_order),
                "flow_velocity_signal": bool(velocity_signal),
                "flow_direction": direction,
                "first_timestamp": first_ts,
                "last_timestamp": last_ts,
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["flow_velocity_signal", "rolling_15m_premium", "total_premium"], ascending=[False, False, False]
    )


def build_flow_velocity_from_minute_bars(minute_bars: pd.DataFrame, thresholds: dict[str, Any]) -> pd.DataFrame:
    columns = [
        "ticker",
        "expiry",
        "strike",
        "side",
        "strategy_family",
        "total_premium",
        "net_premium",
        "trade_count",
        "premium_per_minute",
        "rolling_5m_premium",
        "rolling_15m_premium",
        "child_order_accumulation",
        "flow_velocity_signal",
        "flow_direction",
        "first_timestamp",
        "last_timestamp",
    ]
    if minute_bars.empty:
        return pd.DataFrame(columns=columns)
    df = minute_bars.copy()
    ticker_totals = df.groupby("ticker")["abs_premium"].sum().sort_values(ascending=False)
    if len(ticker_totals) > 100:
        df = df[df["ticker"].isin(set(ticker_totals.head(100).index))].copy()
    keys = ["ticker", "expiry_dt", "strike", "side_key", "strategy_family"]
    group_totals = (
        df.groupby(keys, dropna=False)
        .agg(total_premium=("abs_premium", "sum"), trade_count=("trade_count", "sum"))
        .reset_index()
        .sort_values("total_premium", ascending=False)
    )
    eligible = group_totals[
        (group_totals["total_premium"] >= safe_float(thresholds["premium_5m_threshold"]) * 0.25)
        | (group_totals["trade_count"] >= int(thresholds["child_order_count_threshold"]))
    ].head(5000)
    if eligible.empty:
        eligible = group_totals.head(500)
    df = df.merge(eligible[keys], on=keys, how="inner")
    rows: list[dict[str, Any]] = []
    for key, part in df.groupby(keys, dropna=False):
        part = part.sort_values("minute")
        total = float(part["abs_premium"].sum())
        net = float(part["signed_premium"].sum())
        trade_count = int(part["trade_count"].sum())
        valid_time = part["minute"].notna().all()
        if valid_time:
            indexed = part.set_index("minute").sort_index()
            rolling_5m = float(indexed["abs_premium"].rolling("5min").sum().max())
            rolling_15m = float(indexed["abs_premium"].rolling("15min").sum().max())
            minutes = max(1.0, (indexed.index.max() - indexed.index.min()).total_seconds() / 60.0 + 1.0)
            first_ts = part["first_timestamp"].min()
            last_ts = part["last_timestamp"].max()
            first_text = first_ts.isoformat() if pd.notna(first_ts) else ""
            last_text = last_ts.isoformat() if pd.notna(last_ts) else ""
        else:
            rolling_5m = total
            rolling_15m = total
            minutes = max(1.0, float(len(part)))
            first_text = ""
            last_text = ""
        max_trade = safe_float(part["max_trade_premium"].max(), 0.0)
        avg_trade = total / max(1, trade_count)
        child_order = (
            trade_count >= int(thresholds["child_order_count_threshold"])
            and rolling_15m >= safe_float(thresholds["premium_15m_threshold"])
            and max_trade <= max(total * 0.35, avg_trade * 4.0)
        )
        velocity_signal = rolling_5m >= safe_float(thresholds["premium_5m_threshold"]) or rolling_15m >= safe_float(
            thresholds["premium_15m_threshold"]
        )
        direction = "bullish" if net > 0 else "bearish" if net < 0 else "unclear"
        rows.append(
            {
                "ticker": key[0],
                "expiry": str(key[1]) if key[1] else "",
                "strike": key[2],
                "side": key[3],
                "strategy_family": key[4],
                "total_premium": round(total, 2),
                "net_premium": round(net, 2),
                "trade_count": trade_count,
                "premium_per_minute": round(total / minutes, 2),
                "rolling_5m_premium": round(rolling_5m, 2),
                "rolling_15m_premium": round(rolling_15m, 2),
                "child_order_accumulation": bool(child_order),
                "flow_velocity_signal": bool(velocity_signal),
                "flow_direction": direction,
                "first_timestamp": first_text,
                "last_timestamp": last_text,
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["flow_velocity_signal", "rolling_15m_premium", "total_premium"], ascending=[False, False, False]
    )


def build_top_flow_universe(events: pd.DataFrame, stock_screener: pd.DataFrame, flow_velocity: pd.DataFrame, *, top_n: int = 50) -> pd.DataFrame:
    columns = [
        "rank",
        "ticker",
        "source",
        "net_premium",
        "abs_net_premium",
        "total_premium",
        "flow_direction",
        "max_rolling_5m_premium",
        "max_rolling_15m_premium",
        "volume_oi_ratio",
        "liquidity_quality",
        "tape_vwap_proxy",
        "last_underlying_price",
        "vwap_confirmation",
        "sector",
        "rank_score",
    ]
    if events.empty:
        fixed = pd.DataFrame({"ticker": sorted(FIXED_LIQUID_UNIVERSE), "source": "fixed_liquid_universe"})
        for col in columns:
            if col not in fixed.columns:
                fixed[col] = "" if col not in {"rank", "rank_score"} else 0
        fixed["rank"] = range(1, len(fixed) + 1)
        return fixed[columns].head(top_n)
    df = events.copy()
    grouped = df.groupby("ticker", as_index=False).agg(
        net_premium=("signed_premium", "sum"),
        total_premium=("abs_premium", "sum"),
        trade_count=("abs_premium", "size"),
        volume=("volume", "sum"),
        open_interest=("open_interest", "sum"),
        last_underlying_price=("underlying_price", "last"),
        sector=("sector", "last"),
    )
    grouped["abs_net_premium"] = grouped["net_premium"].abs()
    grouped["flow_direction"] = grouped["net_premium"].map(lambda x: "bullish" if x > 0 else "bearish" if x < 0 else "unclear")
    grouped["volume_oi_ratio"] = grouped["volume"] / grouped["open_interest"].where(grouped["open_interest"].abs() > 0)
    if not flow_velocity.empty:
        velocity = flow_velocity.groupby("ticker", as_index=False).agg(
            max_rolling_5m_premium=("rolling_5m_premium", "max"),
            max_rolling_15m_premium=("rolling_15m_premium", "max"),
            child_order_accumulation=("child_order_accumulation", "max"),
            flow_velocity_signal=("flow_velocity_signal", "max"),
        )
        grouped = grouped.merge(velocity, on="ticker", how="left")
    else:
        grouped["max_rolling_5m_premium"] = 0.0
        grouped["max_rolling_15m_premium"] = 0.0
    if not stock_screener.empty and "ticker" in stock_screener.columns:
        sc = stock_screener.copy()
        sc["ticker"] = sc["ticker"].astype(str).str.upper()
        sc_cols = [c for c in ["ticker", "total_open_interest", "avg30_volume", "sector", "close"] if c in sc.columns]
        grouped = grouped.merge(sc[sc_cols].drop_duplicates("ticker"), on="ticker", how="left", suffixes=("", "_screener"))
        grouped["sector"] = grouped.get("sector", "").fillna(grouped.get("sector_screener", ""))
        grouped["liquidity_quality"] = (
            pd.to_numeric(grouped.get("total_open_interest"), errors="coerce").fillna(0).clip(upper=1_000_000) / 1_000_000
            + pd.to_numeric(grouped.get("avg30_volume"), errors="coerce").fillna(0).clip(upper=100_000_000) / 100_000_000
        )
    else:
        grouped["liquidity_quality"] = 0.0

    price_rows = []
    for ticker, part in df[df["underlying_price"].notna()].sort_values("executed_at").groupby("ticker"):
        weights = part["abs_premium"].replace(0, math.nan)
        weighted = (part["underlying_price"] * weights).sum()
        weight_sum = weights.sum()
        vwap = weighted / weight_sum if weight_sum and math.isfinite(weight_sum) else math.nan
        last = safe_float(part["underlying_price"].iloc[-1])
        direction = "bullish" if part["signed_premium"].sum() > 0 else "bearish" if part["signed_premium"].sum() < 0 else "unclear"
        if not math.isfinite(vwap) or not math.isfinite(last):
            confirmation = "unavailable"
        elif direction == "bullish" and last >= vwap:
            confirmation = "bullish_above_tape_vwap"
        elif direction == "bearish" and last <= vwap:
            confirmation = "bearish_below_tape_vwap"
        elif direction in {"bullish", "bearish"}:
            confirmation = f"{direction}_vwap_not_confirmed"
        else:
            confirmation = "unclear"
        price_rows.append({"ticker": ticker, "tape_vwap_proxy": vwap, "last_underlying_price": last, "vwap_confirmation": confirmation})
    price = pd.DataFrame(price_rows)
    if not price.empty:
        grouped = grouped.drop(columns=["last_underlying_price"], errors="ignore").merge(price, on="ticker", how="left")
    grouped["vwap_confirmation"] = grouped.get("vwap_confirmation", pd.Series("", index=grouped.index)).fillna("unavailable")
    grouped["source"] = grouped["ticker"].map(lambda ticker: "fixed+uw_flow" if ticker in FIXED_LIQUID_UNIVERSE else "uw_discovered")
    for col in ["abs_net_premium", "max_rolling_15m_premium", "volume_oi_ratio", "liquidity_quality"]:
        numeric = pd.to_numeric(grouped[col], errors="coerce").fillna(0.0)
        grouped[f"_{col}_rank"] = numeric.rank(pct=True)
    grouped["rank_score"] = (
        grouped["_abs_net_premium_rank"] * 0.34
        + grouped["_max_rolling_15m_premium_rank"] * 0.28
        + grouped["_volume_oi_ratio_rank"] * 0.18
        + grouped["_liquidity_quality_rank"] * 0.20
    )
    grouped = grouped.sort_values(["rank_score", "abs_net_premium"], ascending=False).head(top_n).copy()
    grouped.insert(0, "rank", range(1, len(grouped) + 1))
    for col in columns:
        if col not in grouped.columns:
            grouped[col] = ""
    numeric_cols = [
        "net_premium",
        "abs_net_premium",
        "total_premium",
        "max_rolling_5m_premium",
        "max_rolling_15m_premium",
        "volume_oi_ratio",
        "liquidity_quality",
        "tape_vwap_proxy",
        "last_underlying_price",
        "rank_score",
    ]
    for col in numeric_cols:
        grouped[col] = pd.to_numeric(grouped[col], errors="coerce").round(4)
    return grouped[columns]


def build_top_flow_universe_from_summary(
    ticker_summary: pd.DataFrame,
    stock_screener: pd.DataFrame,
    flow_velocity: pd.DataFrame,
    *,
    top_n: int = 50,
) -> pd.DataFrame:
    columns = [
        "rank",
        "ticker",
        "source",
        "net_premium",
        "abs_net_premium",
        "total_premium",
        "flow_direction",
        "max_rolling_5m_premium",
        "max_rolling_15m_premium",
        "volume_oi_ratio",
        "liquidity_quality",
        "tape_vwap_proxy",
        "last_underlying_price",
        "vwap_confirmation",
        "sector",
        "rank_score",
    ]
    if ticker_summary.empty:
        fixed = pd.DataFrame({"ticker": sorted(FIXED_LIQUID_UNIVERSE), "source": "fixed_liquid_universe"})
        for col in columns:
            if col not in fixed.columns:
                fixed[col] = "" if col not in {"rank", "rank_score"} else 0
        fixed["rank"] = range(1, len(fixed) + 1)
        return fixed[columns].head(top_n)
    grouped = ticker_summary.copy()
    grouped["ticker"] = grouped["ticker"].astype(str).str.upper()
    grouped["abs_net_premium"] = pd.to_numeric(grouped["net_premium"], errors="coerce").abs()
    grouped["flow_direction"] = grouped["net_premium"].map(lambda x: "bullish" if x > 0 else "bearish" if x < 0 else "unclear")
    grouped["volume_oi_ratio"] = grouped["volume"] / grouped["open_interest"].where(grouped["open_interest"].abs() > 0)
    grouped["tape_vwap_proxy"] = grouped["vwap_num"] / grouped["vwap_weight"].where(grouped["vwap_weight"].abs() > 0)
    if not flow_velocity.empty:
        velocity = flow_velocity.groupby("ticker", as_index=False).agg(
            max_rolling_5m_premium=("rolling_5m_premium", "max"),
            max_rolling_15m_premium=("rolling_15m_premium", "max"),
        )
        grouped = grouped.merge(velocity, on="ticker", how="left")
    else:
        grouped["max_rolling_5m_premium"] = 0.0
        grouped["max_rolling_15m_premium"] = 0.0
    if not stock_screener.empty and "ticker" in stock_screener.columns:
        sc = stock_screener.copy()
        sc["ticker"] = sc["ticker"].astype(str).str.upper()
        sc_cols = [c for c in ["ticker", "total_open_interest", "avg30_volume", "sector"] if c in sc.columns]
        grouped = grouped.merge(sc[sc_cols].drop_duplicates("ticker"), on="ticker", how="left", suffixes=("", "_screener"))
        if "sector_screener" in grouped.columns:
            grouped["sector"] = grouped["sector"].fillna(grouped["sector_screener"])
        grouped["liquidity_quality"] = (
            pd.to_numeric(grouped.get("total_open_interest"), errors="coerce").fillna(0).clip(upper=1_000_000) / 1_000_000
            + pd.to_numeric(grouped.get("avg30_volume"), errors="coerce").fillna(0).clip(upper=100_000_000) / 100_000_000
        )
    else:
        grouped["liquidity_quality"] = 0.0
    grouped["vwap_confirmation"] = "unavailable"
    has_vwap = grouped["tape_vwap_proxy"].notna() & grouped["last_underlying_price"].notna()
    bullish = grouped["flow_direction"].eq("bullish")
    bearish = grouped["flow_direction"].eq("bearish")
    grouped.loc[has_vwap & bullish & (grouped["last_underlying_price"] >= grouped["tape_vwap_proxy"]), "vwap_confirmation"] = "bullish_above_tape_vwap"
    grouped.loc[has_vwap & bullish & (grouped["last_underlying_price"] < grouped["tape_vwap_proxy"]), "vwap_confirmation"] = "bullish_vwap_not_confirmed"
    grouped.loc[has_vwap & bearish & (grouped["last_underlying_price"] <= grouped["tape_vwap_proxy"]), "vwap_confirmation"] = "bearish_below_tape_vwap"
    grouped.loc[has_vwap & bearish & (grouped["last_underlying_price"] > grouped["tape_vwap_proxy"]), "vwap_confirmation"] = "bearish_vwap_not_confirmed"
    grouped["source"] = grouped["ticker"].map(lambda ticker: "fixed+uw_flow" if ticker in FIXED_LIQUID_UNIVERSE else "uw_discovered")
    for col in ["abs_net_premium", "max_rolling_15m_premium", "volume_oi_ratio", "liquidity_quality"]:
        numeric = pd.to_numeric(grouped[col], errors="coerce").fillna(0.0)
        grouped[f"_{col}_rank"] = numeric.rank(pct=True)
    grouped["rank_score"] = (
        grouped["_abs_net_premium_rank"] * 0.34
        + grouped["_max_rolling_15m_premium_rank"] * 0.28
        + grouped["_volume_oi_ratio_rank"] * 0.18
        + grouped["_liquidity_quality_rank"] * 0.20
    )
    grouped = grouped.sort_values(["rank_score", "abs_net_premium"], ascending=False).head(top_n).copy()
    grouped.insert(0, "rank", range(1, len(grouped) + 1))
    for col in columns:
        if col not in grouped.columns:
            grouped[col] = ""
    numeric_cols = [
        "net_premium",
        "abs_net_premium",
        "total_premium",
        "max_rolling_5m_premium",
        "max_rolling_15m_premium",
        "volume_oi_ratio",
        "liquidity_quality",
        "tape_vwap_proxy",
        "last_underlying_price",
        "rank_score",
    ]
    for col in numeric_cols:
        grouped[col] = pd.to_numeric(grouped[col], errors="coerce").round(4)
    return grouped[columns]


def _benchmark_for(row: pd.Series) -> str:
    ticker = _clean_ticker(row.get("ticker"))
    if ticker in SEMI_TICKERS:
        return "SMH"
    sector = str(row.get("sector") or "").lower()
    for key, benchmark in SECTOR_BENCHMARKS.items():
        if key in sector:
            return benchmark
    return "SPY"


def _price_history(root: Path, tickers: set[str], *, asof: dt.date, lookback: int = 30) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    dated: list[tuple[dt.date, Path]] = []
    for child in root.iterdir() if root.exists() else []:
        if not child.is_dir():
            continue
        day = _date_value(child.name[:10])
        if day is None or day > asof:
            continue
        dated.append((day, child))
    dated.sort(key=lambda item: item[0])
    for day, folder in dated[-lookback:]:
        try:
            sc = load_stock_screener(folder)
        except Exception:
            continue
        part = sc[sc["ticker"].astype(str).str.upper().isin(tickers)].copy()
        if part.empty:
            continue
        for _, row in part.iterrows():
            rows.append({"date": day, "ticker": _clean_ticker(row.get("ticker")), "close": safe_float(row.get("close"))})
    return pd.DataFrame(rows)


def build_correlation_anomalies(top_flow: pd.DataFrame, root: Path, *, asof: dt.date) -> pd.DataFrame:
    columns = [
        "ticker",
        "benchmark",
        "rolling_correlation",
        "beta",
        "ticker_return_1d",
        "benchmark_return_1d",
        "relative_return_divergence",
        "flow_direction",
        "anomaly",
        "sector_leader_signal",
        "reason",
    ]
    if top_flow.empty:
        return pd.DataFrame(columns=columns)
    top = top_flow.copy()
    top["benchmark"] = top.apply(_benchmark_for, axis=1)
    tickers = set(top["ticker"].astype(str).str.upper()) | set(top["benchmark"].astype(str).str.upper()) | {"SPY", "QQQ", "IWM", "SMH", "XLK", "XLF", "XLE"}
    hist = _price_history(root, tickers, asof=asof)
    if hist.empty:
        return pd.DataFrame(columns=columns)
    pivot = hist.pivot_table(index="date", columns="ticker", values="close", aggfunc="last").sort_index()
    returns = pivot.pct_change()
    rows: list[dict[str, Any]] = []
    for _, row in top.iterrows():
        ticker = _clean_ticker(row.get("ticker"))
        benchmark = _clean_ticker(row.get("benchmark")) or "SPY"
        if benchmark == ticker:
            benchmark = "QQQ" if ticker != "QQQ" and "QQQ" in returns.columns else "SPY"
        if benchmark == ticker:
            continue
        if ticker not in returns.columns or benchmark not in returns.columns:
            continue
        pair = returns[[ticker, benchmark]].dropna().tail(20)
        if len(pair) < 3:
            continue
        ticker_series = pair[ticker]
        benchmark_series = pair[benchmark]
        if isinstance(ticker_series, pd.DataFrame):
            ticker_series = ticker_series.iloc[:, 0]
        if isinstance(benchmark_series, pd.DataFrame):
            benchmark_series = benchmark_series.iloc[:, 0]
        corr = ticker_series.corr(benchmark_series)
        bench_var = benchmark_series.var()
        beta = ticker_series.cov(benchmark_series) / bench_var if bench_var and math.isfinite(bench_var) else math.nan
        ticker_ret = safe_float(ticker_series.iloc[-1])
        bench_ret = safe_float(benchmark_series.iloc[-1])
        divergence = ticker_ret - bench_ret if math.isfinite(ticker_ret) and math.isfinite(bench_ret) else math.nan
        flow_dir = str(row.get("flow_direction") or "unclear")
        anomaly = math.isfinite(divergence) and abs(divergence) >= 0.02 and abs(safe_float(row.get("net_premium"), 0.0)) > 0
        sector_signal = ""
        if anomaly and benchmark in {"SMH", "XLK", "XLF", "XLE"}:
            sector_signal = f"{benchmark} divergence leader"
        reason = (
            f"{ticker} diverged {divergence:+.2%} from {benchmark} while UW net flow was {flow_dir}"
            if anomaly
            else "no material ticker/index divergence"
        )
        rows.append(
            {
                "ticker": ticker,
                "benchmark": benchmark,
                "rolling_correlation": round(corr, 4) if math.isfinite(corr) else math.nan,
                "beta": round(beta, 4) if math.isfinite(beta) else math.nan,
                "ticker_return_1d": round(ticker_ret, 4) if math.isfinite(ticker_ret) else math.nan,
                "benchmark_return_1d": round(bench_ret, 4) if math.isfinite(bench_ret) else math.nan,
                "relative_return_divergence": round(divergence, 4) if math.isfinite(divergence) else math.nan,
                "flow_direction": flow_dir,
                "anomaly": bool(anomaly),
                "sector_leader_signal": sector_signal,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(["anomaly", "relative_return_divergence"], ascending=[False, False])


def build_zero_dte_gamma_context(events: pd.DataFrame, *, asof: dt.date) -> pd.DataFrame:
    columns = [
        "ticker",
        "spot",
        "high_volume_strike_magnet",
        "pinning_level",
        "gamma_flip_zone",
        "net_0dte_premium",
        "total_0dte_premium",
        "dominant_flow_direction",
        "setup_type",
        "reason",
    ]
    if events.empty or "expiry_dt" not in events.columns:
        return pd.DataFrame(columns=columns)
    df = events[events["ticker"].isin(INDEX_FLOW_TICKERS) & events["expiry_dt"].eq(asof)].copy()
    if df.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for ticker, part in df.groupby("ticker"):
        part = part.copy()
        spot = safe_float(part["underlying_price"].dropna().iloc[-1]) if part["underlying_price"].notna().any() else math.nan
        by_strike = part.groupby("strike", as_index=False).agg(
            total_premium=("abs_premium", "sum"),
            net_premium=("signed_premium", "sum"),
            volume=("volume", "sum"),
            open_interest=("open_interest", "sum"),
            gamma=("gamma", "sum"),
        )
        if by_strike.empty:
            continue
        by_strike["strike_pressure"] = by_strike["volume"].fillna(0.0) + by_strike["open_interest"].fillna(0.0)
        magnet = by_strike.sort_values(["strike_pressure", "total_premium"], ascending=False).iloc[0]
        by_strike = by_strike.sort_values("strike")
        by_strike["gamma_exposure_proxy"] = by_strike["gamma"].fillna(0.0) * by_strike["open_interest"].fillna(0.0)
        by_strike["cum_gamma_proxy"] = by_strike["gamma_exposure_proxy"].cumsum()
        flip_row = by_strike.iloc[(by_strike["cum_gamma_proxy"].abs()).argmin()]
        pinning = safe_float(magnet.get("strike"))
        gamma_flip = safe_float(flip_row.get("strike"))
        total = safe_float(part["abs_premium"].sum(), 0.0)
        net = safe_float(part["signed_premium"].sum(), 0.0)
        direction = "bullish" if net > 0 else "bearish" if net < 0 else "balanced"
        magnet_distance = abs(pinning / spot - 1.0) if math.isfinite(pinning) and math.isfinite(spot) and spot else math.nan
        flip_distance = abs(gamma_flip / spot - 1.0) if math.isfinite(gamma_flip) and math.isfinite(spot) and spot else math.nan
        if math.isfinite(magnet_distance) and magnet_distance <= 0.004 and abs(net) <= total * 0.25:
            setup_type = "pinning"
            reason = "0DTE volume/OI magnet is near spot and net premium is balanced"
        elif math.isfinite(flip_distance) and flip_distance <= 0.006 and abs(net) > total * 0.25:
            setup_type = "reversal"
            reason = "estimated gamma flip is near spot with directional 0DTE premium"
        elif abs(net) > total * 0.45:
            setup_type = "directional"
            reason = "0DTE net premium is directionally imbalanced"
        else:
            setup_type = "liquidity-trap"
            reason = "0DTE flow is concentrated but does not confirm clean direction"
        rows.append(
            {
                "ticker": ticker,
                "spot": round(spot, 4) if math.isfinite(spot) else math.nan,
                "high_volume_strike_magnet": round(pinning, 4) if math.isfinite(pinning) else math.nan,
                "pinning_level": round(pinning, 4) if math.isfinite(pinning) else math.nan,
                "gamma_flip_zone": round(gamma_flip, 4) if math.isfinite(gamma_flip) else math.nan,
                "net_0dte_premium": round(net, 2),
                "total_0dte_premium": round(total, 2),
                "dominant_flow_direction": direction,
                "setup_type": setup_type,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values("total_0dte_premium", ascending=False)


def build_zero_dte_gamma_context_from_bars(zero_dte_bars: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "ticker",
        "spot",
        "high_volume_strike_magnet",
        "pinning_level",
        "gamma_flip_zone",
        "net_0dte_premium",
        "total_0dte_premium",
        "dominant_flow_direction",
        "setup_type",
        "reason",
    ]
    if zero_dte_bars.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for ticker, part in zero_dte_bars.groupby("ticker"):
        part = part.copy()
        spot = safe_float(part["spot"].dropna().iloc[-1]) if part["spot"].notna().any() else math.nan
        part["strike_pressure"] = part["volume"].fillna(0.0) + part["open_interest"].fillna(0.0)
        magnet = part.sort_values(["strike_pressure", "total_premium"], ascending=False).iloc[0]
        by_strike = part.sort_values("strike")
        by_strike["gamma_exposure_proxy"] = by_strike["gamma"].fillna(0.0) * by_strike["open_interest"].fillna(0.0)
        by_strike["cum_gamma_proxy"] = by_strike["gamma_exposure_proxy"].cumsum()
        flip_row = by_strike.iloc[(by_strike["cum_gamma_proxy"].abs()).argmin()]
        pinning = safe_float(magnet.get("strike"))
        gamma_flip = safe_float(flip_row.get("strike"))
        total = safe_float(part["total_premium"].sum(), 0.0)
        net = safe_float(part["net_premium"].sum(), 0.0)
        direction = "bullish" if net > 0 else "bearish" if net < 0 else "balanced"
        magnet_distance = abs(pinning / spot - 1.0) if math.isfinite(pinning) and math.isfinite(spot) and spot else math.nan
        flip_distance = abs(gamma_flip / spot - 1.0) if math.isfinite(gamma_flip) and math.isfinite(spot) and spot else math.nan
        if math.isfinite(magnet_distance) and magnet_distance <= 0.004 and abs(net) <= total * 0.25:
            setup_type = "pinning"
            reason = "0DTE volume/OI magnet is near spot and net premium is balanced"
        elif math.isfinite(flip_distance) and flip_distance <= 0.006 and abs(net) > total * 0.25:
            setup_type = "reversal"
            reason = "estimated gamma flip is near spot with directional 0DTE premium"
        elif abs(net) > total * 0.45:
            setup_type = "directional"
            reason = "0DTE net premium is directionally imbalanced"
        else:
            setup_type = "liquidity-trap"
            reason = "0DTE flow is concentrated but does not confirm clean direction"
        rows.append(
            {
                "ticker": ticker,
                "spot": round(spot, 4) if math.isfinite(spot) else math.nan,
                "high_volume_strike_magnet": round(pinning, 4) if math.isfinite(pinning) else math.nan,
                "pinning_level": round(pinning, 4) if math.isfinite(pinning) else math.nan,
                "gamma_flip_zone": round(gamma_flip, 4) if math.isfinite(gamma_flip) else math.nan,
                "net_0dte_premium": round(net, 2),
                "total_0dte_premium": round(total, 2),
                "dominant_flow_direction": direction,
                "setup_type": setup_type,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values("total_0dte_premium", ascending=False)


def build_liquidity_shift_signals(
    *,
    base_dir: Path,
    root: Path,
    asof: dt.date,
    stock_screener: pd.DataFrame,
    hot_chains: pd.DataFrame | None = None,
    chain_oi: pd.DataFrame | None = None,
    regime: dict[str, Any] | None = None,
    max_rows: int | None = None,
) -> dict[str, Any]:
    del hot_chains, chain_oi
    generated_at = dt.datetime.now(dt.timezone.utc).isoformat()
    thresholds = compute_volatility_thresholds(regime)
    try:
        scan = scan_bot_flow_tape(base_dir, asof=asof, max_rows=max_rows)
        status = "ok"
        error = ""
    except Exception as exc:
        scan = {
            "event_rows": 0,
            "ticker_summary": pd.DataFrame(),
            "minute_bars": pd.DataFrame(),
            "zero_dte_bars": pd.DataFrame(),
        }
        status = "unavailable"
        error = str(exc)
    velocity = build_flow_velocity_from_minute_bars(scan["minute_bars"], thresholds)
    top_flow = build_top_flow_universe_from_summary(scan["ticker_summary"], stock_screener, velocity, top_n=50)
    anomalies = build_correlation_anomalies(top_flow, root, asof=asof)
    zero_dte = build_zero_dte_gamma_context_from_bars(scan["zero_dte_bars"])
    summary = {
        "status": status,
        "error": error,
        "generated_at_utc": generated_at,
        "threshold_regime": thresholds,
        "event_rows": int(scan.get("event_rows", 0)),
        "flow_velocity_rows": int(len(velocity)),
        "flow_velocity_signals": int(velocity["flow_velocity_signal"].sum()) if not velocity.empty else 0,
        "child_order_accumulation_signals": int(velocity["child_order_accumulation"].sum()) if not velocity.empty else 0,
        "top_flow_tickers_scanned": int(len(top_flow)),
        "correlation_anomaly_count": int(anomalies["anomaly"].sum()) if not anomalies.empty else 0,
        "zero_dte_index_signal_count": int(len(zero_dte)),
        "top_flow_tickers": top_flow["ticker"].head(10).tolist() if not top_flow.empty else [],
    }
    return {
        "status": status,
        "generated_at_utc": generated_at,
        "thresholds": thresholds,
        "events": pd.DataFrame(),
        "ticker_summary": scan["ticker_summary"],
        "minute_bars": scan["minute_bars"],
        "flow_velocity": velocity,
        "top_flow_universe": top_flow,
        "correlation_anomalies": anomalies,
        "zero_dte_gamma": zero_dte,
        "summary": summary,
    }


def expand_pool_with_top_flow(pool: pd.DataFrame, stock_screener: pd.DataFrame, signals: dict[str, Any], *, max_top_flow: int = 50) -> pd.DataFrame:
    if stock_screener.empty:
        return pool
    top = signals.get("top_flow_universe")
    if not isinstance(top, pd.DataFrame) or top.empty:
        return pool
    tickers = set(top["ticker"].astype(str).str.upper().head(max_top_flow))
    if not tickers:
        return pool
    expanded = stock_screener[stock_screener["ticker"].astype(str).str.upper().isin(tickers)].copy()
    if pool is not None and not pool.empty:
        expanded = pd.concat([pool, expanded], ignore_index=True, sort=False)
    return expanded.drop_duplicates("ticker", keep="first")


def _direction_confirms_candidate(candidate_direction: object, vwap_confirmation: object) -> bool:
    direction = str(candidate_direction or "")
    confirmation = str(vwap_confirmation or "")
    bullish = direction in {"Bull Put", "Bull Call"} or "bull" in direction.lower()
    bearish = direction in {"Bear Call", "Bear Put"} or "bear" in direction.lower()
    return (bullish and confirmation.startswith("bullish_above")) or (bearish and confirmation.startswith("bearish_below"))


def _append_token(value: object, token: str) -> str:
    parts = [x.strip() for x in str(value or "").split(";") if x.strip() and x.strip().lower() != "nan"]
    if token and token not in parts:
        parts.append(token)
    return ";".join(parts)


def apply_liquidity_shift_context(scored: pd.DataFrame, signals: dict[str, Any], *, require_intraday_vwap: bool = False) -> pd.DataFrame:
    if scored.empty:
        return scored.copy()
    top = signals.get("top_flow_universe")
    velocity = signals.get("flow_velocity")
    if not isinstance(top, pd.DataFrame) or top.empty:
        out = scored.copy()
        out["alpha_tier"] = "unclassified"
        out["liquidity_shift_note"] = "liquidity-shift scan unavailable"
        return out
    out = scored.copy()
    top_map = top.set_index("ticker", drop=False)
    if isinstance(velocity, pd.DataFrame) and not velocity.empty:
        velocity_by_ticker = velocity.groupby("ticker", as_index=False).agg(
            rolling_5m_premium=("rolling_5m_premium", "max"),
            rolling_15m_premium=("rolling_15m_premium", "max"),
            child_order_accumulation=("child_order_accumulation", "max"),
            flow_velocity_signal=("flow_velocity_signal", "max"),
        ).set_index("ticker", drop=False)
    else:
        velocity_by_ticker = pd.DataFrame()
    for idx, row in out.iterrows():
        ticker = _clean_ticker(row.get("ticker"))
        if ticker not in top_map.index:
            out.at[idx, "alpha_tier"] = "unclassified"
            out.at[idx, "liquidity_shift_note"] = "ticker not in top-50 UW liquidity-shift sweep"
            continue
        top_row = top_map.loc[ticker]
        if isinstance(top_row, pd.DataFrame):
            top_row = top_row.iloc[0]
        velocity_row = velocity_by_ticker.loc[ticker] if ticker in velocity_by_ticker.index else pd.Series(dtype=object)
        rank = int(safe_float(top_row.get("rank"), 999))
        flow_signal = bool(velocity_row.get("flow_velocity_signal", False)) if not velocity_row.empty else False
        child = bool(velocity_row.get("child_order_accumulation", False)) if not velocity_row.empty else False
        vwap_confirmation = str(top_row.get("vwap_confirmation") or "unavailable")
        vwap_ok = _direction_confirms_candidate(row.get("direction"), vwap_confirmation)
        alpha_tier = "Tier 1" if rank <= 10 and flow_signal and vwap_ok else "Tier 2" if (flow_signal or child) else "Watchlist"
        out.at[idx, "liquidity_shift_rank"] = rank
        out.at[idx, "liquidity_shift_score"] = top_row.get("rank_score")
        out.at[idx, "liquidity_shift_flow_direction"] = top_row.get("flow_direction")
        out.at[idx, "flow_velocity_signal"] = flow_signal
        out.at[idx, "rolling_5m_premium"] = velocity_row.get("rolling_5m_premium", math.nan) if not velocity_row.empty else math.nan
        out.at[idx, "rolling_15m_premium"] = velocity_row.get("rolling_15m_premium", math.nan) if not velocity_row.empty else math.nan
        out.at[idx, "child_order_accumulation"] = child
        out.at[idx, "vwap_confirmation"] = vwap_confirmation
        out.at[idx, "alpha_tier"] = alpha_tier
        out.at[idx, "liquidity_shift_note"] = (
            f"{alpha_tier}: top-flow rank {rank}, 5m/15m premium "
            f"{safe_float(out.at[idx, 'rolling_5m_premium'], 0):,.0f}/{safe_float(out.at[idx, 'rolling_15m_premium'], 0):,.0f}; "
            f"VWAP={vwap_confirmation}"
        )
        if alpha_tier == "Tier 2":
            out.at[idx, "penalties"] = _append_token(row.get("penalties"), "tier2_reduced_size_until_live_outcome_proof")
            if require_intraday_vwap and not vwap_ok:
                out.at[idx, "hard_rejects"] = _append_token(row.get("hard_rejects"), "vwap_unconfirmed_tier2_intraday")
                out.at[idx, "penalties"] = _append_token(out.at[idx, "penalties"], "vwap_unconfirmed_tier2")
    return out


def write_liquidity_shift_artifacts(out_dir: Path, asof: dt.date, signals: dict[str, Any]) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Any] = {}
    frames = {
        "flow_velocity": signals.get("flow_velocity"),
        "top_flow_universe": signals.get("top_flow_universe"),
        "correlation_anomalies": signals.get("correlation_anomalies"),
        "zero_dte_gamma": signals.get("zero_dte_gamma"),
    }
    for name, frame in frames.items():
        path = out_dir / f"codexdaily_v3_{name}_{asof}.csv"
        if isinstance(frame, pd.DataFrame):
            frame.to_csv(path, index=False)
        else:
            pd.DataFrame().to_csv(path, index=False)
        artifacts[f"{name}_csv"] = str(path)
    summary = dict(signals.get("summary") or {})
    summary["artifact_paths"] = artifacts
    summary_path = out_dir / f"codexdaily_v3_liquidity_shift_summary_{asof}.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    artifacts["summary_json"] = str(summary_path)
    return {"summary": summary, "artifacts": artifacts}
