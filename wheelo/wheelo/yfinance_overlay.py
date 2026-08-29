"""Optional yfinance overlay. Rate-limits are not a pipeline failure."""

from __future__ import annotations

from typing import Any, Dict


def fetch_yfinance(ticker: str) -> Dict[str, Any]:
    try:
        import yfinance as yf
    except ImportError:
        return {"ok": False, "error": "yfinance_unavailable"}
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}
        roe = info.get("returnOnEquity")
        de = info.get("debtToEquity")
        fcf = info.get("freeCashflow")
        mcap = info.get("marketCap")
        fcf_yield = None
        if fcf is not None and mcap:
            fcf_yield = 100.0 * float(fcf) / float(mcap)
        return {
            "ok": True,
            "roe": None if roe is None else float(roe) * 100.0,
            "debt_equity": None if de is None else float(de) / 100.0,
            "fcf_yield": fcf_yield,
        }
    except Exception:
        return {"ok": False, "error": "yfinance_unavailable"}
