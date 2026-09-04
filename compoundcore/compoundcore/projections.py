"""5-year and 10-year dollar paths. Geometric annual rates, monthly compounding.

End-of-month contributions. Monthly rate is (1+r)^(1/12)-1 so a 5% annual
path on a lump sum is exactly (1.05)^5.
"""

from __future__ import annotations

from typing import Dict

from compoundcore.sleeve import (
    FANTASY_ANNUAL,
    INFLATION,
    SCENARIOS,
    VOO_ONLY_10Y,
    portfolio_rate,
)


def monthly_rate(annual: float) -> float:
    return (1.0 + annual) ** (1.0 / 12.0) - 1.0


def fv_lump(principal: float, annual: float, years: float) -> float:
    if principal < 0:
        raise ValueError("principal must be >= 0")
    return principal * ((1.0 + annual) ** years)


def fv_annuity(pmt: float, annual: float, years: float) -> float:
    """Ordinary annuity, payment at end of each month."""
    if pmt < 0:
        raise ValueError("pmt must be >= 0")
    n = int(round(years * 12))
    r = monthly_rate(annual)
    if abs(r) < 1e-15:
        return pmt * n
    return pmt * (((1.0 + r) ** n) - 1.0) / r


def fv_dca(principal: float, monthly: float, annual: float, years: float) -> float:
    return fv_lump(principal, annual, years) + fv_annuity(monthly, annual, years)


def real_value(nominal: float, years: float, inflation: float = INFLATION) -> float:
    return nominal / ((1.0 + inflation) ** years)


def path_table(
    principal: float,
    monthly: float,
    sleeve: str = "default",
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Nested [horizon][scenario] -> {nominal, real, contributed}."""
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for years, horizon in ((5, "5y"), (10, "10y")):
        out[horizon] = {}
        for scenario in SCENARIOS:
            annual = portfolio_rate(sleeve, horizon, scenario)
            nominal = fv_dca(principal, monthly, annual, years)
            contributed = principal + monthly * years * 12
            out[horizon][scenario] = {
                "annual": annual,
                "nominal": nominal,
                "real": real_value(nominal, years),
                "contributed": contributed,
            }
        out[horizon]["fantasy"] = {
            "annual": FANTASY_ANNUAL,
            "nominal": fv_dca(principal, monthly, FANTASY_ANNUAL, years),
            "real": real_value(fv_dca(principal, monthly, FANTASY_ANNUAL, years), years),
            "contributed": principal + monthly * years * 12,
        }
    voo10 = fv_dca(principal, monthly, VOO_ONLY_10Y, 10)
    out["10y"]["voo_only"] = {
        "annual": VOO_ONLY_10Y,
        "nominal": voo10,
        "real": real_value(voo10, 10),
        "contributed": principal + monthly * 10 * 12,
    }
    return out


def round_thousands(value: float) -> int:
    return int(round(value / 1000.0))
