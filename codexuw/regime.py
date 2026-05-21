from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from .data import safe_float


INDEXES = ["SPY", "QQQ", "IWM"]
SECTOR_ETFS = ["SMH", "XLK", "XLF", "XLE", "XLV", "XLY", "XLI", "XLC"]
MAG7 = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA"]
SEMIS = ["NVDA", "AMD", "AVGO", "TSM", "MU", "SMCI", "SMH"]


def _return_1d(row: pd.Series) -> float:
    close = safe_float(row.get("close"))
    prev = safe_float(row.get("prev_close"))
    if not math.isfinite(close) or not math.isfinite(prev) or prev == 0:
        return math.nan
    return close / prev - 1.0


def _row_map(stock_screener: pd.DataFrame) -> dict[str, pd.Series]:
    if stock_screener.empty or "ticker" not in stock_screener.columns:
        return {}
    df = stock_screener.copy()
    df["ticker"] = df["ticker"].astype(str).str.upper()
    return {str(row["ticker"]): row for _, row in df.iterrows()}


def _breadth(stock_screener: pd.DataFrame) -> dict[str, Any]:
    if stock_screener.empty or "close" not in stock_screener.columns or "prev_close" not in stock_screener.columns:
        return {"status": "unavailable", "advance_ratio": None}
    df = stock_screener.copy()
    close = pd.to_numeric(df["close"], errors="coerce")
    prev = pd.to_numeric(df["prev_close"], errors="coerce")
    valid = close.notna() & prev.notna() & prev.ne(0)
    returns = close[valid] / prev[valid] - 1.0
    if returns.empty:
        return {"status": "unavailable", "advance_ratio": None}
    return {
        "status": "ok",
        "advance_ratio": round(float((returns > 0).mean()), 4),
        "median_return_1d": round(float(returns.median()), 4),
        "up_count": int((returns > 0).sum()),
        "down_count": int((returns < 0).sum()),
    }


def _leadership(rows: dict[str, pd.Series], tickers: list[str]) -> dict[str, Any]:
    values = []
    for ticker in tickers:
        row = rows.get(ticker)
        if row is None:
            continue
        ret = _return_1d(row)
        if math.isfinite(ret):
            values.append(ret)
    if not values:
        return {"status": "unavailable", "avg_return_1d": None}
    avg = sum(values) / len(values)
    return {"status": "ok", "avg_return_1d": round(avg, 4), "positive_ratio": round(sum(v > 0 for v in values) / len(values), 4)}


def build_v3_regime_context(
    *,
    stock_screener: pd.DataFrame,
    base_regime: dict[str, Any],
    liquidity_shift: dict[str, Any] | None,
    asof: dt.date,
    run_mode: str,
) -> dict[str, Any]:
    rows = _row_map(stock_screener)
    index_context: dict[str, Any] = {}
    for ticker in INDEXES:
        row = rows.get(ticker)
        if row is None:
            index_context[ticker] = {"status": "unavailable"}
            continue
        index_context[ticker] = {
            "close": safe_float(row.get("close")),
            "prev_close": safe_float(row.get("prev_close")),
            "return_1d": round(_return_1d(row), 4) if math.isfinite(_return_1d(row)) else None,
            "flow_bias": safe_float(row.get("flow_bias")),
            "iv30d": safe_float(row.get("iv30d")),
        }
    sector_context: dict[str, Any] = {}
    for ticker in SECTOR_ETFS:
        row = rows.get(ticker)
        if row is None:
            continue
        ret = _return_1d(row)
        sector_context[ticker] = {
            "return_1d": round(ret, 4) if math.isfinite(ret) else None,
            "flow_bias": safe_float(row.get("flow_bias")),
        }
    zero_dte = liquidity_shift.get("zero_dte_gamma") if liquidity_shift else None
    if isinstance(zero_dte, pd.DataFrame) and not zero_dte.empty:
        zero_dte_summary = zero_dte[["ticker", "setup_type", "pinning_level", "gamma_flip_zone", "dominant_flow_direction"]].head(10).to_dict(
            orient="records"
        )
    else:
        zero_dte_summary = []
    thresholds = (liquidity_shift or {}).get("thresholds") or {}
    top_flow = (liquidity_shift or {}).get("top_flow_universe")
    top_flow_tickers = top_flow["ticker"].head(10).tolist() if isinstance(top_flow, pd.DataFrame) and not top_flow.empty else []
    return {
        "pipeline": "Codex Daily V3",
        "asof": str(asof),
        "run_mode": run_mode,
        "base_regime": base_regime,
        "indices": index_context,
        "vix": {"proxy": base_regime.get("vix_proxy"), "volatility_regime": base_regime.get("volatility")},
        "sector_breadth": _breadth(stock_screener),
        "sector_etfs": sector_context,
        "rates_yields": {"status": "unavailable", "reason": "no local rates/yields feed found in UW exports"},
        "mag7_leadership": _leadership(rows, MAG7),
        "semi_leadership": _leadership(rows, SEMIS),
        "liquidity_shift_thresholds": thresholds,
        "top_flow_tickers": top_flow_tickers,
        "vwap_context": {
            "source": "UW tape underlying_price premium-weighted proxy when available",
            "intraday_requirement": "Tier 2 intraday candidates require VWAP confirmation before Execute",
        },
        "zero_dte_gamma_context": zero_dte_summary,
    }


def write_v3_regime_artifact(out_dir: Path, asof: dt.date, context: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"codexdaily_v3_regime_context_{asof}.json"
    path.write_text(json.dumps(context, indent=2, sort_keys=True, default=str), encoding="utf-8")
    summary = {
        "status": "ok",
        "path": str(path),
        "trend": (context.get("base_regime") or {}).get("trend"),
        "volatility": (context.get("base_regime") or {}).get("volatility"),
        "top_flow_tickers": context.get("top_flow_tickers") or [],
        "zero_dte_gamma_rows": len(context.get("zero_dte_gamma_context") or []),
    }
    return path, summary
