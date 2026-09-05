"""Underlying thesis from tape. No invented catalysts or posts."""

from __future__ import annotations

from groat.num import fmt, fmt_pct
from groat.setups import SETUP_NAMES


def build_thesis(row: dict) -> dict:
    ticker = row.get("ticker") or ""
    direction = row.get("direction") or "neutral"
    setup = row.get("primary") or ""
    setup_name = SETUP_NAMES.get(setup, "no named setup")
    regime = row.get("regime") or "unknown"
    group = row.get("group") or "other"
    gstat = row.get("group_status") or "DATA UNAVAILABLE"
    trend = row.get("trend") or "unknown"
    fire = row.get("fire") or {}
    earn = row.get("earnings") or {}
    picked = row.get("picked") or {}
    choice = row.get("choice") or "NO TRADE"

    why_now = []
    if setup:
        why_now.append("%s (%s)" % (setup_name, setup))
    if fire.get("kind"):
        why_now.append(fire.get("note") or ("FIRE %s" % fire.get("kind")))
    if row.get("rs_20") is not None:
        why_now.append("20d RS vs SPY %s" % fmt_pct(row.get("rs_20")))
    if gstat in ("accelerating", "emerging"):
        why_now.append("%s group is %s" % (group, gstat))

    invalidation = picked.get("invalidation") if isinstance(picked, dict) else None
    if not invalidation:
        if direction == "bullish":
            invalidation = "close back below 20 EMA / swing-low AVWAP"
        elif direction == "bearish":
            invalidation = "close back above 20 EMA / failed-breakdown reclaim"
        else:
            invalidation = "no directional thesis"

    earn_line = "earnings %s (%s)" % (earn.get("date") or "DATA UNAVAILABLE", earn.get("source") or "DATA UNAVAILABLE")
    if earn.get("overlaps_hold"):
        earn_line += "; ordinary options blocked through the print"

    lines = [
        "%s is a %s idea in a %s tape because %s."
        % (ticker, direction, regime.replace("_", " "), ", ".join(why_now) or "the screen did not produce a named edge"),
        "Price is %s, trend %s, 20 EMA %s / 50 %s / 200 %s, relative volume %s."
        % (
            fmt(row.get("close")),
            trend,
            fmt(row.get("ema20")),
            fmt(row.get("sma50")),
            fmt(row.get("sma200")),
            fmt(row.get("rvol"), 1),
        ),
        "Group %s (%s) is %s. AVWAP year %s, swing-low %s."
        % (
            group,
            row.get("etf") or "",
            gstat,
            fmt(row.get("avwap_year")),
            fmt(row.get("avwap_swing_low")),
        ),
        earn_line[0].upper() + earn_line[1:] + ".",
        "Instrument shortlist picked **%s**. Invalidation: %s."
        % (choice, invalidation),
    ]
    news = str(row.get("news") or "").strip()
    filings = str(row.get("filings") or "").strip()
    if news and news != "DATA UNAVAILABLE":
        lines.append("News: %s." % news)
    if filings and filings != "DATA UNAVAILABLE":
        lines.append("Filings: %s." % filings)
    if choice == "NO TRADE":
        lines.append("No trade: edge is not objective enough to risk capital today.")
    headline = "%s — %s %s in %s %s" % (ticker, direction, setup_name, gstat, group)
    return {
        "headline": headline,
        "paragraphs": lines,
        "invalidation": invalidation,
        "why_now": why_now,
    }
