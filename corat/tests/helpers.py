from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, List

from corat.models import Bar, HistoricalStats, OptionStructure, SetupSignal, TechnicalSnapshot


def trend_bars(ticker: str = "AAA", count: int = 260, breakout: bool = False) -> List[Bar]:
    start = date(2025, 1, 2)
    bars = []
    price = 100.0
    for index in range(count):
        price *= 1.0012
        volume = 1_000_000.0
        if index % 30 == 0 and index > 60:
            price *= 1.025
            volume = 2_000_000.0
        day = (start + timedelta(days=index)).isoformat()
        bars.append(Bar(day, price - 0.3, price + 1.5, price - 1.5, price, volume, True, day + "T21:00:00Z", "fixture"))
    if breakout:
        prior = max(item.high for item in bars[-20:])
        price = prior + 1.0
        day = (start + timedelta(days=count)).isoformat()
        bars.append(Bar(day, price - 0.5, price + 1.0, price - 1.0, price, 3_000_000.0, True, day + "T21:00:00Z", "fixture"))
    return bars


def dailies_rows(ticker: str, count: int = 260, breakout: bool = False):
    return [
        {
            "ticker": ticker,
            "tradeDate": bar.date,
            "open": bar.open,
            "hiPx": bar.high,
            "loPx": bar.low,
            "clsPx": bar.close,
            "stockVolume": bar.volume,
            "updatedAt": bar.updated_at,
        }
        for bar in trend_bars(ticker, count=count, breakout=breakout)
    ]


def snapshot(direction: str = "BULLISH") -> TechnicalSnapshot:
    price = 120.0
    return TechnicalSnapshot(
        ticker="AAA", as_of="2026-08-27", price=price, price_date="2026-08-27", price_complete=True,
        ema20=115.0, sma50=110.0, sma200=100.0, atr14=3.0, rsi14=60.0,
        return_5d=0.03, return_20d=0.10, return_60d=0.20, relative_volume_20d=1.5,
        average_dollar_volume_20d=500_000_000.0, prior_high_20d=118.0, prior_low_20d=105.0,
        support=114.0, resistance=130.0, extension_from_ema_atr=1.2, avwaps=[],
    )


def setup(direction: str = "BULLISH", triggered: bool = True) -> SetupSignal:
    return SetupSignal("BREAKOUT + CONFIRMATION", direction, 0.85, triggered, "fixture reason", "fixture trigger", "fixture invalidation")


def history(reliable: bool = True, expectancy: float = 0.02) -> HistoricalStats:
    count = 30 if reliable else 5
    returns = ([0.05] * int(count * 0.6)) + ([-0.025] * (count - int(count * 0.6)))
    return HistoricalStats(
        method="fixture", sample_size=count, reliable=reliable, horizon_returns={},
        primary_horizon=10, win_rate=0.6, expectancy=expectancy, average_winner=0.05,
        average_loser=-0.025, profit_factor=2.0, mae=-0.02, mfe=0.06, max_drawdown=-0.08,
        signal_dates=[], primary_returns=returns, primary_paths=[[value] for value in returns],
    )


def empty_option() -> OptionStructure:
    return OptionStructure(False, "NONE", "", 0, [], "", None, None, None, None, None, None, None, None, None, None, None, None, None, None, "", "", ["fixture"])
