"""Broad ORATS universe discovery before expensive chain requests."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


@dataclass(frozen=True)
class UniverseCandidate:
    ticker: str
    asset_class: str
    direction: str
    discovery_score: float
    price: float
    average_option_volume_20d: float
    option_open_interest: float
    data_confidence: float
    return_1w_pct: float
    return_1m_pct: float
    realized_vol_forecast_20d_pct: float
    implied_vol_30d_pct: float
    implied_vol_forecast_20d_pct: float
    implied_forecast_wedge: float
    implied_vs_realized_wedge: float
    next_earnings_date: str
    source: str = "ORATS_CORES_FULL_UNIVERSE"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _confidence(value: Any) -> float:
    result = _number(value)
    return result / 100.0 if result > 1.0 else result


def discover_optionable_universe(
    core_rows: Iterable[Mapping[str, Any]],
    *,
    limit: int = 250,
    minimum_average_option_volume: float = 5_000.0,
    minimum_option_open_interest: float = 10_000.0,
    minimum_price: float = 5.0,
    minimum_confidence: float = 0.80,
) -> Tuple[Sequence[UniverseCandidate], Mapping[str, int]]:
    """Rank liquid U.S. equities and ETFs from the complete ORATS cores feed.

    This stage is intentionally cheap. It narrows thousands of underlyings
    before any historical or Schwab option-chain request is made.
    """

    if limit < 1:
        raise ValueError("limit must be positive")
    accepted: List[UniverseCandidate] = []
    rejections: Dict[str, int] = {}
    seen = set()
    for row in core_rows:
        ticker = str(row.get("ticker") or "").strip().upper()
        if not ticker or ticker in seen:
            if ticker in seen:
                rejections["duplicate_ticker"] = rejections.get("duplicate_ticker", 0) + 1
            else:
                rejections["missing_ticker"] = rejections.get("missing_ticker", 0) + 1
            continue
        seen.add(ticker)
        if not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,14}", ticker):
            rejections["nontradable_symbol"] = rejections.get("nontradable_symbol", 0) + 1
            continue
        asset_type = int(_number(row.get("assetType"), -1))
        if asset_type not in {3, 7}:
            rejections["unsupported_asset_type"] = rejections.get("unsupported_asset_type", 0) + 1
            continue
        price = _number(row.get("pxCls") or row.get("priorCls"))
        option_volume = _number(row.get("avgOptVolu20d"))
        open_interest = _number(row.get("oi"))
        confidence = _confidence(row.get("confidence"))
        if price < minimum_price:
            rejections["price_below_floor"] = rejections.get("price_below_floor", 0) + 1
            continue
        if option_volume < minimum_average_option_volume:
            rejections["thin_option_volume"] = rejections.get("thin_option_volume", 0) + 1
            continue
        if open_interest < minimum_option_open_interest:
            rejections["thin_open_interest"] = rejections.get("thin_open_interest", 0) + 1
            continue
        if confidence < minimum_confidence:
            rejections["low_orats_confidence"] = rejections.get("low_orats_confidence", 0) + 1
            continue

        return_1w = _number(row.get("stkPxChng1wk"))
        return_1m = _number(row.get("stkPxChng1m"))
        trend = 0.65 * return_1w + 0.35 * return_1m
        realized_forecast = _number(row.get("orFcst20d"))
        current_iv = _number(row.get("iv30d"))
        implied_forecast = _number(row.get("orIvFcst20d"))
        implied_wedge = (
            implied_forecast / current_iv - 1.0 if current_iv > 0 else 0.0
        )
        realized_wedge = current_iv / realized_forecast - 1.0 if realized_forecast > 0 else 0.0
        liquidity_score = math.log10(max(option_volume, 1.0)) + 0.5 * math.log10(
            max(open_interest, 1.0)
        )
        opportunity = min(abs(trend) / 5.0, 3.0) + min(
            max(abs(implied_wedge), abs(realized_wedge)), 1.0
        )
        quality = 2.0 * confidence
        accepted.append(
            UniverseCandidate(
                ticker=ticker,
                asset_class="EQUITY" if asset_type == 3 else "ETF",
                direction="LONG" if trend >= 0 else "SHORT",
                discovery_score=liquidity_score + quality + opportunity,
                price=price,
                average_option_volume_20d=option_volume,
                option_open_interest=open_interest,
                data_confidence=confidence,
                return_1w_pct=return_1w,
                return_1m_pct=return_1m,
                realized_vol_forecast_20d_pct=realized_forecast,
                implied_vol_30d_pct=current_iv,
                implied_vol_forecast_20d_pct=implied_forecast,
                implied_forecast_wedge=implied_wedge,
                implied_vs_realized_wedge=realized_wedge,
                next_earnings_date=str(row.get("nextErn") or ""),
            )
        )

    accepted.sort(
        key=lambda item: (
            -item.discovery_score,
            -item.average_option_volume_20d,
            item.ticker,
        )
    )
    rejections["accepted_before_limit"] = len(accepted)
    rejections["returned"] = min(len(accepted), limit)
    return tuple(accepted[:limit]), dict(sorted(rejections.items()))
