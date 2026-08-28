"""ORATS volatility, earnings, and positioning normalization."""

from __future__ import annotations

import math
from datetime import date
from typing import Any, Dict, Iterable, Mapping, Optional

from corat.constants import DATA_UNAVAILABLE


def _float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _percent(value: Any) -> Optional[float]:
    parsed = _float(value)
    if parsed is None:
        return None
    return parsed * 100.0 if abs(parsed) <= 3.0 else parsed


def _index(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    result: Dict[str, Mapping[str, Any]] = {}
    for row in rows:
        ticker = str(row.get("ticker") or "").strip().upper()
        if not ticker:
            continue
        existing = result.get(ticker)
        if existing is None or str(row.get("updatedAt") or row.get("tradeDate") or "") >= str(
            existing.get("updatedAt") or existing.get("tradeDate") or ""
        ):
            result[ticker] = row
    return result


def _date_value(value: Any) -> str:
    text = str(value or "")[:10]
    try:
        date.fromisoformat(text)
    except ValueError:
        return ""
    return text


def normalize_volatility(
    tickers: Iterable[str],
    core_rows: Iterable[Mapping[str, Any]],
    ivrank_rows: Iterable[Mapping[str, Any]],
    summary_rows: Iterable[Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    cores = _index(core_rows)
    ranks = _index(ivrank_rows)
    summaries = _index(summary_rows)
    result: Dict[str, Dict[str, Any]] = {}
    for raw in tickers:
        ticker = str(raw).upper()
        core = cores.get(ticker, {})
        rank = ranks.get(ticker, {})
        summary = summaries.get(ticker, {})
        atm_iv = _percent(core.get("orIvXern20d")) or _percent(summary.get("iv20d")) or _percent(rank.get("iv"))
        hv = _percent(core.get("orHv20d")) or _percent(core.get("clsHv20d"))
        forecast_realized = _percent(core.get("orFcst20d"))
        forecast_iv = _percent(core.get("orIvFcst20d"))
        ex_earnings_iv = _percent(core.get("exErnIv20d")) or _percent(summary.get("exErnIv20d"))
        m1 = _percent(core.get("atmIvM1"))
        m2 = _percent(core.get("atmIvM2"))
        term_structure = DATA_UNAVAILABLE
        if m1 is not None and m2 is not None:
            difference = m2 - m1
            term_structure = "CONTANGO" if difference > 0.5 else "BACKWARDATION" if difference < -0.5 else "FLAT"
        iv_hv_ratio = atm_iv / hv if atm_iv is not None and hv and hv > 0 else None
        iv_forecast_ratio = atm_iv / forecast_realized if atm_iv is not None and forecast_realized and forecast_realized > 0 else None
        missing = []
        for label, value in (
            ("ATM IV", atm_iv),
            ("IV rank", _float(rank.get("ivRank1y"))),
            ("IV percentile", _float(rank.get("ivPct1y")) or _float(core.get("ivPctile1y"))),
            ("historical volatility", hv),
            ("ORATS forecast volatility", forecast_realized),
            ("term structure", None if term_structure == DATA_UNAVAILABLE else 1.0),
        ):
            if value is None:
                missing.append(label)
        implied_move = _percent(core.get("impliedEarningsMove")) or _percent(summary.get("impliedEarningsMove")) or _percent(summary.get("impliedMove"))
        next_earnings_date = _date_value(core.get("nextErn"))
        weeks_to_next = _float(core.get("wksNextErn"))
        earnings_reference = next_earnings_date
        if not earnings_reference and weeks_to_next and weeks_to_next > 0:
            earnings_reference = "ORATS estimate: {:.0f} weeks; exact date unavailable".format(weeks_to_next)
        result[ticker] = {
            "status": "AVAILABLE" if core or rank or summary else DATA_UNAVAILABLE,
            "trade_date": str(core.get("tradeDate") or rank.get("tradeDate") or summary.get("tradeDate") or "")[:10],
            "updated_at": str(core.get("updatedAt") or rank.get("updatedAt") or summary.get("updatedAt") or ""),
            "stock_price": _float(core.get("pxCls")) or _float(core.get("pxAtmIv")) or _float(summary.get("stockPrice")),
            "atm_iv_pct": atm_iv,
            "iv_rank_1y": _float(rank.get("ivRank1y")),
            "iv_percentile_1y": _float(rank.get("ivPct1y")) or _float(core.get("ivPctile1y")),
            "historical_volatility_20d_pct": hv,
            "historical_volatility_60d_pct": _percent(core.get("orHv60d")) or _percent(core.get("clsHv60d")),
            "ex_earnings_iv_20d_pct": ex_earnings_iv,
            "orats_forecast_realized_20d_pct": forecast_realized,
            "orats_forecast_iv_20d_pct": forecast_iv,
            "iv_hv_ratio": iv_hv_ratio,
            "iv_forecast_realized_ratio": iv_forecast_ratio,
            "term_structure": term_structure,
            "atm_iv_m1_pct": m1,
            "atm_iv_m2_pct": m2,
            "skew": _float(core.get("slope")) or _float(summary.get("skewing")),
            "skew_forecast": _float(core.get("slopeFcst")),
            "orats_confidence": _float(summary.get("confidence")),
            "market_width_vol": _float(core.get("mktWidthVol")),
            "implied_earnings_move_pct": implied_move,
            "historical_average_earnings_move_pct": _percent(core.get("absAvgErnMv")),
            "historical_earnings_move_std_pct": _percent(core.get("ernMvStdv")),
            "next_earnings_date": next_earnings_date,
            "next_earnings_reference": earnings_reference or DATA_UNAVAILABLE,
            "weeks_to_next_earnings": weeks_to_next,
            "last_earnings_date": _date_value(core.get("lastErn")),
            "days_to_next_earnings": _float(core.get("daysToNextErn")),
            "risk_free_rate_pct": _percent(core.get("iRate5wk")),
            "dividend_yield_pct": _percent(core.get("divYield")),
            "call_volume": int(_float(core.get("cVolu")) or 0),
            "put_volume": int(_float(core.get("pVolu")) or 0),
            "call_open_interest": int(_float(core.get("cOi")) or 0),
            "put_open_interest": int(_float(core.get("pOi")) or 0),
            "average_option_volume_20d": int(_float(core.get("avgOptVolu20d")) or 0),
            "sector": str(core.get("sectorName") or core.get("sector") or ""),
            "missing_metrics": missing,
        }
    return result
