"""Minervini 8-point Trend Template on daily bars. Diagnostic, not the ON_BOARD gate."""

from __future__ import annotations

from typing import Optional


def eight_point(daily: dict, rs_pctile: Optional[float]) -> dict:
    px = daily.get("close")
    sma50 = daily.get("sma50")
    sma150 = daily.get("sma150")
    sma200 = daily.get("sma200")
    sma200_prev = daily.get("sma200_prev_21")
    hi252 = daily.get("hi252")
    lo252 = daily.get("lo252")
    checks = []

    def add(name: str, ok: Optional[bool], detail: str) -> None:
        checks.append({"name": name, "ok": ok, "detail": detail})

    add(
        "price_above_150_and_200",
        None if px is None or sma150 is None or sma200 is None else (px > sma150 and px > sma200),
        "close vs 150/200 SMA",
    )
    add(
        "sma150_above_sma200",
        None if sma150 is None or sma200 is None else sma150 > sma200,
        "150 > 200",
    )
    add(
        "sma200_rising_1m",
        None if sma200 is None or sma200_prev is None else sma200 > sma200_prev,
        "200 SMA up vs 21 sessions ago",
    )
    add(
        "sma50_above_150_and_200",
        None if sma50 is None or sma150 is None or sma200 is None else (sma50 > sma150 and sma50 > sma200),
        "50 > 150 and 200",
    )
    add(
        "price_above_50",
        None if px is None or sma50 is None else px > sma50,
        "close > 50 SMA",
    )
    above_low = None
    if px is not None and lo252 not in (None, 0):
        above_low = px >= 1.30 * lo252
    add("pct_above_52w_low", above_low, ">= 30% above 52w low")
    near_high = None
    if px is not None and hi252 not in (None, 0):
        near_high = px >= 0.75 * hi252
    add("within_25pct_52w_high", near_high, "within 25% of 52w high")
    rs_ok_flag = None if rs_pctile is None else rs_pctile >= 70.0
    add("rs_pctile_70", rs_ok_flag, "RS percentile >= 70")

    known = [c for c in checks if c["ok"] is not None]
    passed = sum(1 for c in known if c["ok"])
    return {
        "checks": checks,
        "passed": passed,
        "known": len(known),
        "pass": len(known) >= 6 and passed == len(known),
        "n_pass": passed,
    }
