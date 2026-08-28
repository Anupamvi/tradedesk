"""Dependency-free Black-Scholes-Merton pricing primitives."""

from __future__ import annotations

import math


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def black_scholes_price(
    spot: float,
    strike: float,
    time_years: float,
    rate: float,
    volatility: float,
    right: str,
    dividend_yield: float = 0.0,
) -> float:
    normalized_right = right.strip().lower()
    if normalized_right not in {"call", "put"}:
        raise ValueError("right must be call or put")
    if spot <= 0 or strike <= 0:
        raise ValueError("spot and strike must be positive")
    if time_years <= 0 or volatility <= 0:
        return max(spot - strike, 0.0) if normalized_right == "call" else max(strike - spot, 0.0)
    root_time = math.sqrt(time_years)
    d1 = (
        math.log(spot / strike)
        + (rate - dividend_yield + 0.5 * volatility * volatility) * time_years
    ) / (volatility * root_time)
    d2 = d1 - volatility * root_time
    discounted_spot = spot * math.exp(-dividend_yield * time_years)
    discounted_strike = strike * math.exp(-rate * time_years)
    if normalized_right == "call":
        return discounted_spot * normal_cdf(d1) - discounted_strike * normal_cdf(d2)
    return discounted_strike * normal_cdf(-d2) - discounted_spot * normal_cdf(-d1)

