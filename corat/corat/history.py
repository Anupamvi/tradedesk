"""Leakage-safe price analogue validation for frozen setup definitions."""

from __future__ import annotations

import statistics
from datetime import date
from typing import Any, Dict, List, Mapping, Optional, Sequence

from corat.models import Bar, HistoricalStats
from corat.technical import atr14, avwap_levels, ema_series, return_n, sma


def _float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _earnings_anchor_index(bars: Sequence[Bar], earnings_events: Sequence[Mapping[str, Any]]) -> Optional[int]:
    if not bars or not earnings_events:
        return None
    signal_date = bars[-1].date
    eligible = []
    for row in earnings_events:
        event_date = str(row.get("earnDate") or "")[:10]
        if not event_date or event_date > signal_date:
            continue
        try:
            age = (date.fromisoformat(signal_date) - date.fromisoformat(event_date)).days
        except ValueError:
            continue
        if 0 <= age <= 35:
            eligible.append((event_date, str(row.get("anncTod") or "")))
    if not eligible:
        return None
    event_date, announcement_time = max(eligible)
    after_hours = announcement_time in {"3", "1630", "AFTER", "AMC"}
    return next(
        (
            index
            for index, bar in enumerate(bars)
            if bar.date > event_date or (bar.date == event_date and not after_hours)
        ),
        None,
    )


def _anchored_vwap(bars: Sequence[Bar], start_index: int) -> Optional[float]:
    weighted = 0.0
    volume = 0.0
    for bar in bars[start_index:]:
        if bar.volume <= 0:
            continue
        weighted += ((bar.high + bar.low + bar.close) / 3.0) * bar.volume
        volume += bar.volume
    return weighted / volume if volume > 0 else None


def _match_setup(
    name: str,
    bars: Sequence[Bar],
    spy_bars: Sequence[Bar],
    sector_bars: Sequence[Bar] = (),
    earnings_events: Sequence[Mapping[str, Any]] = (),
) -> bool:
    if len(bars) < 61:
        return False
    closes = [bar.close for bar in bars]
    ema20 = ema_series(closes, 20)[-1]
    average50 = sma(closes, 50)
    atr = atr14(bars)
    prior_high = max(bar.high for bar in bars[-21:-1])
    relative_volume = 0.0
    prior_volumes = [bar.volume for bar in bars[-21:-1] if bar.volume > 0]
    if prior_volumes and bars[-1].volume > 0:
        relative_volume = bars[-1].volume / (sum(prior_volumes) / len(prior_volumes))
    stock20 = return_n(closes, 20) or 0.0
    spy20 = return_n([bar.close for bar in spy_bars], 20) or 0.0
    sector20 = return_n([bar.close for bar in sector_bars], 20) if len(sector_bars) >= 21 else None
    trend = average50 is not None and bars[-1].close > ema20 > average50
    if name == "TREND PULLBACK":
        levels = avwap_levels(bars)
        near_structure = bool(
            atr
            and (
                abs(bars[-1].close - ema20) <= 0.8 * atr
                or any(abs(bars[-1].close - level.value) <= 0.8 * atr for level in levels)
            )
        )
        return bool(trend and near_structure and relative_volume <= 1.2)
    if name == "BREAKOUT + CONFIRMATION":
        return bool(bars[-1].close > prior_high and relative_volume >= 1.15 and trend)
    if name == "POST-EARNINGS DRIFT":
        anchor_index = _earnings_anchor_index(bars, earnings_events)
        if anchor_index is None or anchor_index <= 0:
            return False
        gap = bars[anchor_index].open / bars[anchor_index - 1].close - 1.0
        earnings_avwap = _anchored_vwap(bars, anchor_index)
        return bool(
            gap >= 0.025
            and earnings_avwap is not None
            and bars[-1].close > earnings_avwap
            and stock20 > 0
            and bars[-1].close > bars[-2].high
        )
    if name == "RELATIVE-STRENGTH LEADER":
        sector_confirmed = sector20 is None or stock20 - sector20 >= 0.015
        return bool(trend and stock20 - spy20 >= 0.04 and sector_confirmed)
    if name == "EMERGING SECTOR ROTATION":
        sector_confirmed = sector20 is None or sector20 - spy20 > 0
        return bool(trend and stock20 - spy20 >= 0.02 and (return_n(closes, 5) or 0) > 0 and sector_confirmed)
    if name == "OVERSOLD REVERSAL":
        recent_low = min(bar.low for bar in bars[-20:])
        return bool(atr and bars[-1].low <= recent_low + 0.25 * atr and bars[-1].close > bars[-2].high and relative_volume >= 1.1)
    if name == "FAILED BREAKOUT / TREND BREAKDOWN":
        was_above = any(bar.close > prior_high for bar in bars[-6:-1])
        return bool(was_above and bars[-1].close < ema20 and bars[-1].close < bars[-2].low)
    return False


def _profit_factor(values: Sequence[float]) -> Optional[float]:
    gains = sum(value for value in values if value > 0)
    losses = abs(sum(value for value in values if value < 0))
    if losses == 0:
        return None if gains == 0 else float("inf")
    return gains / losses


def _max_drawdown(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    equity = 0.0
    peak = 0.0
    worst = 0.0
    for value in values:
        equity += value
        peak = max(peak, equity)
        worst = min(worst, equity - peak)
    return worst


def analyze_analogues(
    setup_name: str,
    direction: str,
    bars: Sequence[Bar],
    spy_bars: Sequence[Bar],
    as_of: str,
    horizons: Sequence[int],
    primary_horizon: int,
    minimum_sample: int,
    maximum_sample: int,
    signal_spacing: int,
    signal_start_date: str = "",
    signal_end_date: str = "",
    sector_bars: Sequence[Bar] = (),
    earnings_events: Sequence[Mapping[str, Any]] = (),
    historical_volatility_rows: Sequence[Mapping[str, Any]] = (),
    current_iv_hv_ratio: Optional[float] = None,
) -> HistoricalStats:
    complete = [bar for bar in bars if bar.complete and bar.date < as_of]
    spy_by_date = {bar.date: bar for bar in spy_bars if bar.complete and bar.date < as_of}
    spy_dates = sorted(spy_by_date)
    sector_by_date = {bar.date: bar for bar in sector_bars if bar.complete and bar.date < as_of}
    sector_dates = sorted(sector_by_date)
    volatility_by_date: Dict[str, float] = {}
    for row in historical_volatility_rows:
        row_date = str(row.get("tradeDate") or "")[:10]
        iv = _float(row.get("orIvXern20d")) or _float(row.get("iv20d"))
        hv = _float(row.get("orHv20d")) or _float(row.get("clsHv20d"))
        if row_date and iv is not None and hv is not None:
            volatility_by_date[row_date] = iv / hv
    volatility_dates = sorted(volatility_by_date)
    direction_sign = -1.0 if direction == "BEARISH" else 1.0
    max_horizon = max(horizons)
    observations: List[Dict[str, object]] = []
    last_signal_index = -1000
    effective_spacing = max(int(signal_spacing), int(primary_horizon))
    for index in range(60, len(complete) - max_horizon):
        if index - last_signal_index < effective_spacing:
            continue
        history = complete[: index + 1]
        signal_date = history[-1].date
        if signal_start_date and signal_date < signal_start_date:
            continue
        if signal_end_date and signal_date > signal_end_date:
            continue
        matching_spy = [spy_by_date[key] for key in spy_dates if key <= history[-1].date]
        matching_sector = [sector_by_date[key] for key in sector_dates if key <= history[-1].date]
        if len(matching_spy) < 61 or not _match_setup(
            setup_name,
            history,
            matching_spy,
            matching_sector,
            earnings_events,
        ):
            continue
        entry = complete[index].close
        returns = {
            horizon: direction_sign * (complete[index + horizon].close / entry - 1.0)
            for horizon in horizons
        }
        forward_bars = complete[index + 1 : index + primary_horizon + 1]
        path = [direction_sign * (bar.close / entry - 1.0) for bar in forward_bars]
        if direction == "BEARISH":
            adverse_path = [-(bar.high / entry - 1.0) for bar in forward_bars]
            favorable_path = [-(bar.low / entry - 1.0) for bar in forward_bars]
        else:
            adverse_path = [bar.low / entry - 1.0 for bar in forward_bars]
            favorable_path = [bar.high / entry - 1.0 for bar in forward_bars]
        prior_vol_dates = [key for key in volatility_dates if key <= signal_date]
        historical_iv_hv = volatility_by_date[prior_vol_dates[-1]] if prior_vol_dates else None
        iv_distance = (
            abs(historical_iv_hv - float(current_iv_hv_ratio))
            if historical_iv_hv is not None and current_iv_hv_ratio is not None
            else None
        )
        observations.append(
            {
                "date": complete[index].date,
                "returns": returns,
                "mae": min(adverse_path) if adverse_path else 0.0,
                "mfe": max(favorable_path) if favorable_path else 0.0,
                "path": path,
                "adverse_path": adverse_path,
                "favorable_path": favorable_path,
                "iv_distance": iv_distance,
            }
        )
        last_signal_index = index
    if current_iv_hv_ratio is not None and any(row.get("iv_distance") is not None for row in observations):
        observations = sorted(
            observations,
            key=lambda row: (
                float(row["iv_distance"]) if row.get("iv_distance") is not None else float("inf"),
                str(row["date"]),
            ),
        )[:maximum_sample]
        observations.sort(key=lambda row: str(row["date"]))
    else:
        observations = observations[-maximum_sample:]
    horizon_stats: Dict[str, Mapping[str, Optional[float]]] = {}
    for horizon in horizons:
        values = [float((row["returns"])[horizon]) for row in observations]  # type: ignore[index]
        horizon_stats[str(horizon)] = {
            "average": statistics.mean(values) if values else None,
            "median": statistics.median(values) if values else None,
            "win_rate": sum(1 for value in values if value > 0) / float(len(values)) if values else None,
        }
    primary = [float((row["returns"])[primary_horizon]) for row in observations]  # type: ignore[index]
    winners = [value for value in primary if value > 0]
    losers = [value for value in primary if value < 0]
    reliable = len(primary) >= minimum_sample
    return HistoricalStats(
        method=(
            "Frozen same-ticker setup analogue using price, volume, relative strength, event AVWAP where applicable, "
            "sector confirmation when available, and nearest historical IV/HV regime when ORATS history is available; "
            "all features use signal-date data. Forward returns are direction-adjusted, intraday high/low ranges drive "
            "conservative stop-before-target sequencing, and signals do not overlap the primary horizon."
        ),
        sample_size=len(primary),
        reliable=reliable,
        horizon_returns=horizon_stats,
        primary_horizon=primary_horizon,
        win_rate=sum(1 for value in primary if value > 0) / float(len(primary)) if primary else None,
        expectancy=statistics.mean(primary) if primary else None,
        average_winner=statistics.mean(winners) if winners else None,
        average_loser=statistics.mean(losers) if losers else None,
        profit_factor=_profit_factor(primary),
        mae=statistics.mean(float(row["mae"]) for row in observations) if observations else None,
        mfe=statistics.mean(float(row["mfe"]) for row in observations) if observations else None,
        max_drawdown=_max_drawdown(primary),
        signal_dates=[str(row["date"]) for row in observations[-20:]],
        primary_returns=primary,
        primary_paths=[[float(value) for value in row["path"]] for row in observations],
        primary_adverse_paths=[[float(value) for value in row["adverse_path"]] for row in observations],
        primary_favorable_paths=[[float(value) for value in row["favorable_path"]] for row in observations],
        similarity_dimensions=[
            "price/trend",
            "volume",
            "SPY relative strength",
            *( ["sector relative strength"] if sector_bars else [] ),
            *( ["event AVWAP and earnings catalyst"] if setup_name == "POST-EARNINGS DRIFT" and earnings_events else [] ),
            *( ["ORATS IV/HV regime"] if volatility_by_date and current_iv_hv_ratio is not None else [] ),
        ],
        missing_dimensions=[
            *( ["historical sector regime"] if not sector_bars else [] ),
            *( ["historical IV regime"] if not volatility_by_date or current_iv_hv_ratio is None else [] ),
            *( ["historical catalyst classification"] if setup_name != "POST-EARNINGS DRIFT" else [] ),
        ],
    )
