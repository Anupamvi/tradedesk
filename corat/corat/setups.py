"""Underlying-first setup detection."""

from __future__ import annotations

from datetime import date
from typing import List, Optional, Sequence

from corat.models import Bar, SetupSignal, TechnicalSnapshot


def _fmt_price(value: Optional[float]) -> str:
    return "DATA UNAVAILABLE" if value is None else "${:.2f}".format(value)


def _days_between(left: str, right: str) -> Optional[int]:
    try:
        return (date.fromisoformat(right) - date.fromisoformat(left)).days
    except ValueError:
        return None


def detect_setups(
    snapshot: TechnicalSnapshot,
    bars: Sequence[Bar],
    spy: Optional[TechnicalSnapshot],
    sector: Optional[TechnicalSnapshot],
    sector_state: str,
    last_earnings_date: str = "",
) -> List[SetupSignal]:
    signals: List[SetupSignal] = []
    atr = snapshot.atr14 or 0.0
    price = snapshot.price
    prior_bar = bars[-2] if len(bars) >= 2 else bars[-1]
    trend_up = (
        snapshot.ema20 is not None
        and snapshot.sma50 is not None
        and price > snapshot.ema20 > snapshot.sma50
        and (snapshot.sma200 is None or snapshot.sma50 > snapshot.sma200)
    )
    rs20 = (snapshot.return_20d or 0.0) - ((spy.return_20d or 0.0) if spy else 0.0)
    sector_rs20 = (snapshot.return_20d or 0.0) - ((sector.return_20d or 0.0) if sector else 0.0)
    nearby_levels = [snapshot.ema20, snapshot.support]
    nearby_levels.extend(level.value for level in snapshot.avwaps)
    near_support = bool(
        atr > 0
        and any(level is not None and abs(price - level) <= 0.75 * atr for level in nearby_levels)
    )
    if trend_up and near_support and (snapshot.relative_volume_20d or 1.0) <= 1.15:
        triggered = price > prior_bar.high and (snapshot.relative_volume_20d or 0.0) >= 0.8
        signals.append(
            SetupSignal(
                "TREND PULLBACK",
                "BULLISH",
                min(1.0, 0.68 + max(0.0, rs20) * 2.0 + (0.08 if triggered else 0.0)),
                triggered,
                "Established uptrend is testing EMA/AVWAP/support with non-expanding pullback volume.",
                "Reclaim {} with improving volume and sector confirmation.".format(_fmt_price(prior_bar.high)),
                "Close below {} or loss of the supporting AVWAP cluster.".format(_fmt_price(snapshot.support)),
            )
        )
    breakout = bool(snapshot.prior_high_20d is not None and price > snapshot.prior_high_20d)
    not_extended = snapshot.extension_from_ema_atr is None or snapshot.extension_from_ema_atr <= 1.75
    if breakout and (snapshot.relative_volume_20d or 0.0) >= 1.15 and not_extended:
        signals.append(
            SetupSignal(
                "BREAKOUT + CONFIRMATION",
                "BULLISH",
                min(1.0, 0.72 + min(0.18, ((snapshot.relative_volume_20d or 1.0) - 1.0) * 0.25)),
                True,
                "Price cleared the prior 20-session high with volume confirmation without an extreme ATR extension.",
                "Hold above {} or retest it successfully.".format(_fmt_price(snapshot.prior_high_20d)),
                "Failed breakout and close below {}.".format(_fmt_price(snapshot.prior_high_20d)),
            )
        )
    if last_earnings_date:
        age = _days_between(last_earnings_date, snapshot.as_of)
        earnings_index = next((i for i, bar in enumerate(bars) if bar.date >= last_earnings_date), None)
        if age is not None and 0 <= age <= 35 and earnings_index is not None and earnings_index > 0:
            gap = bars[earnings_index].open / bars[earnings_index - 1].close - 1.0
            earnings_avwap = next((level for level in snapshot.avwaps if level.anchor_reason == "most recent earnings"), None)
            holds = bool(earnings_avwap and price > earnings_avwap.value)
            if gap >= 0.025 and holds and (snapshot.return_20d or 0.0) > 0:
                triggered = price > prior_bar.high
                signals.append(
                    SetupSignal(
                        "POST-EARNINGS DRIFT",
                        "BULLISH",
                        min(1.0, 0.70 + min(0.20, gap)),
                        triggered,
                        "A material earnings gap remains above the earnings AVWAP rather than filling immediately.",
                        "Break the post-earnings consolidation high with confirming volume.",
                        "Close below earnings AVWAP at {}.".format(_fmt_price(earnings_avwap.value if earnings_avwap else None)),
                    )
                )
    if trend_up and rs20 >= 0.04 and sector_rs20 >= 0.015:
        triggered = price > prior_bar.high and not_extended
        signals.append(
            SetupSignal(
                "RELATIVE-STRENGTH LEADER",
                "BULLISH",
                min(1.0, 0.65 + rs20 + sector_rs20),
                triggered,
                "The stock is outperforming SPY and its sector while maintaining accumulation structure.",
                "New short-term high or controlled pullback reclaim above EMA20.",
                "Relative strength rolls over and price closes below {}.".format(_fmt_price(snapshot.ema20)),
            )
        )
    if sector_state in {"ACCELERATING LEADER", "EMERGING LEADER"} and trend_up and sector_rs20 > 0:
        triggered = price > prior_bar.high and not_extended
        signals.append(
            SetupSignal(
                "EMERGING SECTOR ROTATION",
                "BULLISH",
                min(1.0, 0.64 + max(0.0, sector_rs20) * 2.0),
                triggered,
                "The security is a leader inside an accelerating or newly emerging group.",
                "Sector ETF and stock confirm above their short-term trigger levels.",
                "Sector state deteriorates or price closes below {}.".format(_fmt_price(snapshot.ema20)),
            )
        )
    if (
        snapshot.rsi14 is not None
        and snapshot.rsi14 < 45
        and snapshot.support is not None
        and atr > 0
        and abs(price - snapshot.support) <= atr
        and price > prior_bar.high
        and (snapshot.relative_volume_20d or 0) >= 1.1
    ):
        signals.append(
            SetupSignal(
                "OVERSOLD REVERSAL",
                "BULLISH",
                0.58,
                True,
                "Price and volume show an objective reversal at support; RSI is supporting context only.",
                "Hold the reversal above {}.".format(_fmt_price(prior_bar.high)),
                "Close below support at {}.".format(_fmt_price(snapshot.support)),
            )
        )
    recently_above = any(
        bar.close > (snapshot.prior_high_20d or float("inf")) for bar in bars[-6:-1]
    )
    bearish_break = bool(
        recently_above
        and snapshot.ema20 is not None
        and price < snapshot.ema20
        and price < prior_bar.low
        and (snapshot.relative_volume_20d or 0.0) >= 1.0
    )
    if bearish_break:
        signals.append(
            SetupSignal(
                "FAILED BREAKOUT / TREND BREAKDOWN",
                "BEARISH",
                min(1.0, 0.68 + ((snapshot.relative_volume_20d or 1.0) - 1.0) * 0.1),
                True,
                "A recent breakout failed and price lost EMA20 plus the prior session low on selling volume.",
                "Remain below {} after a weak retest.".format(_fmt_price(snapshot.ema20)),
                "Reclaim and hold above {}.".format(_fmt_price(snapshot.prior_high_20d)),
            )
        )
    signals.sort(key=lambda item: (item.triggered, item.strength), reverse=True)
    if not signals:
        signals.append(
            SetupSignal(
                "NO QUALIFYING SETUP",
                "NEUTRAL",
                0.0,
                False,
                "No frozen CORAT setup definition is currently satisfied.",
                "Wait for a valid pullback, breakout, post-event drift, rotation, reversal, or breakdown trigger.",
                "Not applicable.",
            )
        )
    return signals

