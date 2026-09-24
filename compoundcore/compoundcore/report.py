"""Markdown tables for the CLI and playbook-shaped output."""

from __future__ import annotations

from typing import List

from compoundcore.allocate import distribution
from compoundcore.projections import path_table, round_thousands
from compoundcore.sleeve import (
    ROLES,
    SLEEVE_NAMES,
    blended_fee,
    nvda_lookthrough,
    smh_crash_hit,
    us_share_of_equities,
    vxus_tax_drag_bps,
    weights,
)


def _usd(n: float) -> str:
    if abs(n) >= 1000000:
        return "$" + format(n, ",.0f")
    if abs(n) >= 1000:
        return "$" + format(n, ",.0f")
    return "$" + format(n, ",.2f")


def _pct(x: float, digits: int = 1) -> str:
    return ("%." + str(digits) + "f%%") % (x * 100.0)


def _k(n: float) -> str:
    k = round_thousands(n)
    if abs(k) >= 1000:
        return "$%.2fM" % (k / 1000.0)
    return "$%dk" % k


SLEEVE_TITLE = {
    "default": "Compound Core (default)",
    "aggressive": "Aggressive variant",
}


def allocation_markdown(amount: float, sleeve: str, weekly: float = 0.0) -> str:
    dist = distribution(amount, sleeve, weekly)
    lines = [
        "### %s — %s" % (SLEEVE_TITLE[sleeve], _usd(amount)),
        "",
        "| Ticker | Role | Weight | Dollars | Band | Weekly |" if weekly
        else "| Ticker | Role | Weight | Dollars | Band | Per $1,000 |",
        "|---|---|---:|---:|---|---:|",
    ]
    w = weights(sleeve)
    for row in dist["rows"]:
        t = row["ticker"]
        band = "%s–%s" % (
            _pct(row["band_low_pct"], 1),
            _pct(row["band_high_pct"], 1),
        )
        extra = _usd(row["weekly"]) if weekly else _usd(row["per_1000"])
        lines.append(
            "| **%s** | %s | %s | %s | %s | %s |"
            % (t, ROLES[t], _pct(w[t], 0), _usd(row["dollars"]), band, extra)
        )
    lines.extend(
        [
            "",
            "- Blended fee **%.3f%%**/yr. Look-through NVDA **%s** (%s)."
            % (
                blended_fee(sleeve) * 100.0,
                _pct(nvda_lookthrough(sleeve), 1),
                _usd(dist["nvda_dollars"]),
            ),
            "- Equity book is **%.0f%% US** (world cap-weight is ~60–65%%). That is a labeled home-bias, not cap-neutral."
            % (us_share_of_equities(sleeve) * 100),
            "- SMH −45%% ≈ **%s** of this core. Crash-protocol cash (VGSH) = **%s**."
            % (_pct(smh_crash_hit(sleeve), 1), _usd(dist["vgsh_dollars"])),
            "- VXUS in taxable at 15–24%% ordinary tax on a ~3%% yield ≈ **%.0f–%.0f bps** of portfolio drag."
            % (
                vxus_tax_drag_bps(0.15, sleeve=sleeve),
                vxus_tax_drag_bps(0.24, sleeve=sleeve),
            ),
        ]
    )
    return "\n".join(lines)


def weekly_recipe_markdown(weekly: float, sleeve: str = "default") -> str:
    if weekly <= 0:
        weekly = 1000.0
        label = "per $1,000 weekly"
    else:
        label = "weekly " + _usd(weekly)
    dist = distribution(weekly, sleeve, 0.0)
    lines = [
        "Weekly buy recipe (%s, %s) — split every contribution. Do not wait for a dip on VOO / VXUS / VGSH."
        % (SLEEVE_TITLE[sleeve], label),
        "",
        "| Ticker | Weight | Amount |",
        "|---|---:|---:|",
    ]
    for row in dist["rows"]:
        lines.append(
            "| %s | %s | %s |"
            % (row["ticker"], _pct(row["weight"], 0), _usd(row["dollars"]))
        )
    return "\n".join(lines)


def _path_row(label: str, five: Dict, ten: Dict) -> str:
    return "| %s | %s | %s |" % (label, _k(five["nominal"]), _k(ten["nominal"]))


def projection_markdown(principal: float, monthly: float, sleeve: str) -> str:
    paths = path_table(principal, monthly, sleeve)
    title = "lump %s" % _usd(principal)
    if monthly:
        title += " + %s/month (adds %s / %s of cash)" % (
            _usd(monthly),
            _usd(monthly * 60),
            _usd(monthly * 120),
        )
    lines = [
        "### %s — 5y / 10y (%s)" % (SLEEVE_TITLE[sleeve], title),
        "",
        "Monthly compounding, end-of-month adds. **Not a guarantee.**",
        "",
        "| Path | 5-year | 10-year |",
        "|---|---:|---:|",
        _path_row("Stress", paths["5y"]["stress"], paths["10y"]["stress"]),
        _path_row("Bear", paths["5y"]["bear"], paths["10y"]["bear"]),
        _path_row("**Base**", paths["5y"]["base"], paths["10y"]["base"]),
        _path_row("Bull", paths["5y"]["bull"], paths["10y"]["bull"]),
        _path_row("Fantasy 40%/yr", paths["5y"]["fantasy"], paths["10y"]["fantasy"]),
        "| VOO-only at VG 4.2–6.2 midpoint (5.2%% 10y) | — | %s |"
        % _k(paths["10y"]["voo_only"]["nominal"]),
        "",
        "Base 10y real (2%% inflation) ≈ **%s**. Stress 10y real ≈ **%s**."
        % (
            _k(paths["10y"]["base"]["real"]),
            _k(paths["10y"]["stress"]["real"]),
        ),
    ]
    if monthly:
        lines.append(
            "New cash in the window: 5y **%s**, 10y **%s** (included in the totals)."
            % (_usd(monthly * 60), _usd(monthly * 120))
        )
    else:
        lines.append("Contributed capital stays **%s** (no additional buys)." % _usd(principal))
    return "\n".join(lines)


def calc_markdown(
    amount: float,
    weekly: float = 0.0,
    monthly: float = 0.0,
    sleeve: str = "both",
) -> str:
    names: List[str]
    if sleeve == "both":
        names = list(SLEEVE_NAMES)
    else:
        names = [sleeve]
    chunks = [
        "# Compound Core calculator",
        "",
        "Long-term **core**. No options, no leverage, no orders. Trading desks never touch this bucket.",
        "",
    ]
    for name in names:
        chunks.append(allocation_markdown(amount, name, weekly))
        chunks.append("")
        chunks.append(weekly_recipe_markdown(weekly if weekly else 1000.0, name))
        chunks.append("")
        chunks.append(projection_markdown(amount, monthly, name))
        chunks.append("")
    chunks.append(
        "Not financial advice. Capital-market assumptions are hypothetical, not guarantees. "
        "Past 40% years are not a budget."
    )
    return "\n".join(chunks).rstrip() + "\n"
