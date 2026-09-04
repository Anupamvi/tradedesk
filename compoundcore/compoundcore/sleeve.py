"""Sleeve weights, bands, fees, and forward-return building blocks.

Numbers are labeled with as-of dates. They are not live quotes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple


ASOF = "2026-09-04"
RESEARCH_ASOF = "2026-09-04"

# Holdings used only for look-through NVDA (not an operating trigger).
NVDA_WEIGHT = {
    "VOO": 0.0755,  # stockanalysis / ETFDB, holdings as of 2026-07-31
    "VGT": 0.1620,  # Vanguard VGT fact sheet, 2026-06-30
    "SMH": 0.2194,  # VanEck SMH fact sheet, 2026-08-24
}

FEE = {
    "VOO": 0.0003,
    "VGT": 0.0009,
    "SMH": 0.0035,
    "VB": 0.0003,
    "VXUS": 0.0005,
    "GLDM": 0.0010,
    "VGSH": 0.0004,
}

ROLES = {
    "VOO": "US compounding engine",
    "VGT": "Tech overweight",
    "SMH": "Chip satellite",
    "VB": "Small-cap kicker",
    "VXUS": "Non-US stocks (US-overweight book)",
    "GLDM": "Inflation / geopolitics hedge",
    "VGSH": "Crash-protocol reserve",
}

TICKER_ORDER = ("VOO", "VGT", "SMH", "VB", "VXUS", "GLDM", "VGSH")

WEIGHTS: Dict[str, Dict[str, float]] = {
    "default": {
        "VOO": 0.48,
        "VGT": 0.10,
        "SMH": 0.07,
        "VB": 0.05,
        "VXUS": 0.20,
        "GLDM": 0.05,
        "VGSH": 0.05,
    },
    "aggressive": {
        "VOO": 0.45,
        "VGT": 0.15,
        "SMH": 0.10,
        "VB": 0.05,
        "VXUS": 0.15,
        "GLDM": 0.05,
        "VGSH": 0.05,
    },
}

BAND_RELATIVE = 0.25
BAND_FLOOR_PP = 2.0  # percentage points

# 10-year geometric nominal building blocks, before inflation and taxes.
# Equity/gold: plan blend (Vanguard range + J.P. Morgan + house median).
# VGSH: Vanguard US short-term Treasury 3.5–4.5% range midpoint (30 Jun 2026).
BUILDING_BLOCKS_10Y = {
    "VOO": {"stress": 0.004, "bear": 0.032, "base": 0.058, "bull": 0.081},
    "VGT": {"stress": -0.020, "bear": 0.028, "base": 0.052, "bull": 0.076},
    "SMH": {"stress": -0.050, "bear": 0.000, "base": 0.055, "bull": 0.110},
    "VB": {"stress": 0.010, "bear": 0.035, "base": 0.061, "bull": 0.088},
    "VXUS": {"stress": 0.030, "bear": 0.052, "base": 0.069, "bull": 0.085},
    "GLDM": {"stress": 0.010, "bear": 0.010, "base": 0.045, "bull": 0.070},
    "VGSH": {"stress": 0.030, "bear": 0.040, "base": 0.040, "bull": 0.040},
}

# Default sleeve 5y/10y rates are locked to the reviewed playbook tables.
# 10y base equals the weighted building blocks (5.8%). 10y bear is the
# published 3.2% (weighted blocks with VGSH at 4% are 3.3%; we keep the
# published table). 5y is wider than 10y by design (rich CAPE + AI path).
DEFAULT_PORTFOLIO_RATES = {
    "5y": {"stress": -0.010, "bear": 0.020, "base": 0.050, "bull": 0.095},
    "10y": {"stress": 0.005, "bear": 0.032, "base": 0.058, "bull": 0.081},
}

# Same 5y-vs-10y spread applied to the aggressive mix.
_FIVE_VS_TEN = {
    path: DEFAULT_PORTFOLIO_RATES["5y"][path] - DEFAULT_PORTFOLIO_RATES["10y"][path]
    for path in ("stress", "bear", "base", "bull")
}

INFLATION = 0.020  # Vanguard US inflation range 1.5–2.5% (30 Jun 2026), midpoint
FANTASY_ANNUAL = 0.40
VOO_ONLY_10Y = 0.052  # Vanguard US-equity 4.2–6.2% range midpoint, not a median
SMH_CRASH = -0.45
VXUS_YIELD = 0.0273  # PortfoliosLab TTM as of early Sep 2026; 2025 income return was 4.33%
VXUS_YIELD_PLANNING = 0.03

SCENARIOS = ("stress", "bear", "base", "bull")
SLEEVE_NAMES = ("default", "aggressive")


@dataclass(frozen=True)
class Band:
    ticker: str
    target: float
    low: float
    high: float
    half_pp: float


def _require_sleeve(name: str) -> str:
    key = (name or "default").strip().lower()
    if key not in WEIGHTS:
        raise ValueError("unknown sleeve %r; use default or aggressive" % name)
    return key


def weights(sleeve: str = "default") -> Dict[str, float]:
    key = _require_sleeve(sleeve)
    return dict(WEIGHTS[key])


def blended_fee(sleeve: str = "default") -> float:
    w = weights(sleeve)
    return sum(w[t] * FEE[t] for t in TICKER_ORDER)


def nvda_lookthrough(sleeve: str = "default") -> float:
    w = weights(sleeve)
    return sum(w[t] * NVDA_WEIGHT.get(t, 0.0) for t in TICKER_ORDER)


def us_share_of_equities(sleeve: str = "default") -> float:
    w = weights(sleeve)
    us = w["VOO"] + w["VGT"] + w["SMH"] + w["VB"]
    intl = w["VXUS"]
    equity = us + intl
    return us / equity if equity else 0.0


def equity_weight(sleeve: str = "default") -> float:
    w = weights(sleeve)
    return w["VOO"] + w["VGT"] + w["SMH"] + w["VB"] + w["VXUS"]


def smh_crash_hit(sleeve: str = "default") -> float:
    return weights(sleeve)["SMH"] * SMH_CRASH


def band_half_pp(target_pct: float) -> float:
    """Half-width in percentage points: max(25% of target, 2pp)."""
    return max(target_pct * BAND_RELATIVE, BAND_FLOOR_PP)


def bands(sleeve: str = "default") -> Dict[str, Band]:
    w = weights(sleeve)
    out = {}
    for ticker in TICKER_ORDER:
        target_pct = w[ticker] * 100.0
        half = band_half_pp(target_pct)
        out[ticker] = Band(
            ticker=ticker,
            target=w[ticker],
            low=(target_pct - half) / 100.0,
            high=(target_pct + half) / 100.0,
            half_pp=half,
        )
    return out


def weighted_block(sleeve: str, scenario: str) -> float:
    w = weights(sleeve)
    return sum(w[t] * BUILDING_BLOCKS_10Y[t][scenario] for t in TICKER_ORDER)


def _aggressive_rates() -> Dict[str, Dict[str, float]]:
    ten = {path: weighted_block("aggressive", path) for path in SCENARIOS}
    five = {path: ten[path] + _FIVE_VS_TEN[path] for path in SCENARIOS}
    return {"5y": five, "10y": ten}


PORTFOLIO_RATES = {
    "default": DEFAULT_PORTFOLIO_RATES,
    "aggressive": _aggressive_rates(),
}


def portfolio_rate(sleeve: str, horizon: str, scenario: str) -> float:
    key = _require_sleeve(sleeve)
    if horizon not in ("5y", "10y"):
        raise ValueError("horizon must be 5y or 10y")
    if scenario not in SCENARIOS:
        raise ValueError("unknown scenario %r" % scenario)
    return PORTFOLIO_RATES[key][horizon][scenario]


def vxus_tax_drag_bps(ordinary_rate: float, yield_: float = VXUS_YIELD_PLANNING, sleeve: str = "default") -> float:
    """Portfolio bps of ordinary-tax drag if VXUS sits in taxable."""
    return weights(sleeve)["VXUS"] * yield_ * ordinary_rate * 10000.0


def public_snapshot() -> Mapping[str, object]:
    """JSON-safe constants for the HTML calculator."""
    sleeve_out = {}
    for name in SLEEVE_NAMES:
        sleeve_out[name] = {
            "weights": weights(name),
            "fee": blended_fee(name),
            "nvda": nvda_lookthrough(name),
            "us_of_equity": us_share_of_equities(name),
            "equity": equity_weight(name),
            "smh_crash_hit": smh_crash_hit(name),
            "bands": {
                t: {"low": b.low, "high": b.high, "half_pp": b.half_pp}
                for t, b in bands(name).items()
            },
            "rates": {
                horizon: dict(PORTFOLIO_RATES[name][horizon])
                for horizon in ("5y", "10y")
            },
        }
    return {
        "asof": ASOF,
        "tickers": list(TICKER_ORDER),
        "roles": dict(ROLES),
        "fees": dict(FEE),
        "nvda_weight": dict(NVDA_WEIGHT),
        "inflation": INFLATION,
        "fantasy": FANTASY_ANNUAL,
        "voo_only_10y": VOO_ONLY_10Y,
        "sleeves": sleeve_out,
    }
