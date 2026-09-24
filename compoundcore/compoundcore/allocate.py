"""Split a dollar amount across a sleeve. Largest-remainder cents so rows sum."""

from __future__ import annotations

from typing import Dict, List

from compoundcore.sleeve import TICKER_ORDER, bands, nvda_lookthrough, smh_crash_hit, weights


def _cents(amount: float) -> int:
    if amount < 0:
        raise ValueError("amount must be >= 0")
    return int(round(amount * 100.0))


def allocate_cents(amount: float, sleeve: str = "default") -> Dict[str, int]:
    """Return integer cents per ticker that sum to the rounded amount."""
    total = _cents(amount)
    w = weights(sleeve)
    raw = {t: total * w[t] for t in TICKER_ORDER}
    floors = {t: int(raw[t]) for t in TICKER_ORDER}
    leftover = total - sum(floors.values())
    ranked = sorted(
        TICKER_ORDER,
        key=lambda t: (raw[t] - floors[t], TICKER_ORDER.index(t)),
        reverse=True,
    )
    for i in range(leftover):
        floors[ranked[i]] += 1
    return floors


def allocate_dollars(amount: float, sleeve: str = "default") -> Dict[str, float]:
    return {t: c / 100.0 for t, c in allocate_cents(amount, sleeve).items()}


def distribution(amount: float, sleeve: str = "default", weekly: float = 0.0) -> Dict[str, object]:
    """Full allocation view: lump, optional weekly, bands, look-through."""
    if weekly < 0:
        raise ValueError("weekly must be >= 0")
    lump = allocate_dollars(amount, sleeve)
    week = allocate_dollars(weekly, sleeve) if weekly else {t: 0.0 for t in TICKER_ORDER}
    per_thousand = allocate_dollars(1000.0, sleeve)
    bmap = bands(sleeve)
    rows: List[Dict[str, object]] = []
    for ticker in TICKER_ORDER:
        band = bmap[ticker]
        rows.append(
            {
                "ticker": ticker,
                "weight": weights(sleeve)[ticker],
                "dollars": lump[ticker],
                "weekly": week[ticker],
                "per_1000": per_thousand[ticker],
                "band_low": band.low * amount,
                "band_high": band.high * amount,
                "band_low_pct": band.low,
                "band_high_pct": band.high,
            }
        )
    return {
        "sleeve": sleeve,
        "amount": round(amount, 2),
        "weekly": round(weekly, 2),
        "rows": rows,
        "nvda_dollars": amount * nvda_lookthrough(sleeve),
        "nvda_weight": nvda_lookthrough(sleeve),
        "vgsh_dollars": lump["VGSH"],
        "smh_crash_dollars": amount * smh_crash_hit(sleeve),
    }
