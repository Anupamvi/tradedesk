"""Sector / theme relative-strength ranking."""

from __future__ import annotations

from typing import Dict, List

from groat.config import SECTOR_ETFS, ticker_etf, ticker_group
from groat.num import fmt, fmt_pct, to_float
from groat.technicals import snapshot


def rank_groups(asof: str, bars_map: Dict[str, list], spy_bars: list) -> List[dict]:
    rows = []
    for etf, group in SECTOR_ETFS.items():
        snap = snapshot(bars_map.get(etf) or [], asof, bench_bars=spy_bars)
        if not snap.get("ok"):
            rows.append(
                {
                    "etf": etf,
                    "group": group,
                    "status": "DATA UNAVAILABLE",
                    "ok": False,
                    "rs_5": None,
                    "rs_20": None,
                    "rs_60": None,
                    "ret_20": None,
                    "rvol": None,
                    "trend": "unknown",
                    "accel": None,
                }
            )
            continue
        rs5 = to_float(snap.get("rs_5"))
        rs20 = to_float(snap.get("rs_20"))
        rs60 = to_float(snap.get("rs_60"))
        accel = None
        if rs20 is not None and rs60 is not None:
            accel = rs20 - rs60
        status = classify_group(rs20, rs60, accel, snap.get("trend") or "")
        rows.append(
            {
                "etf": etf,
                "group": group,
                "status": status,
                "ok": True,
                "rs_5": rs5,
                "rs_20": rs20,
                "rs_60": rs60,
                "ret_20": snap.get("ret_20"),
                "rvol": snap.get("rvol"),
                "trend": snap.get("trend"),
                "accel": accel,
                "close": snap.get("close"),
            }
        )
    rows.sort(key=lambda r: (to_float(r.get("rs_20")) is not None, to_float(r.get("rs_20")) or -999), reverse=True)
    return rows


def classify_group(rs20, rs60, accel, trend: str) -> str:
    if rs20 is None:
        return "DATA UNAVAILABLE"
    if rs20 > 0.02 and accel is not None and accel > 0.01:
        return "accelerating"
    if rs20 > 0 and (rs60 is None or rs20 > rs60) and trend in ("up", "strong_up"):
        return "emerging"
    if rs20 > 0.03 and accel is not None and accel < -0.01:
        return "mature"
    if rs20 < -0.02:
        return "deteriorating"
    return "neutral"


def group_status_map(rows: List[dict]) -> Dict[str, str]:
    out = {}
    for row in rows:
        out[row["group"]] = row.get("status") or "DATA UNAVAILABLE"
        out[row["etf"]] = row.get("status") or "DATA UNAVAILABLE"
    return out


def name_group_row(ticker: str, group_rows: List[dict]) -> dict:
    etf = ticker_etf(ticker)
    group = ticker_group(ticker)
    for row in group_rows:
        if row.get("etf") == etf or row.get("group") == group:
            return row
    return {"etf": etf, "group": group, "status": "DATA UNAVAILABLE", "ok": False}


def render_rotation(rows: List[dict]) -> List[str]:
    lines = [
        "# Sector rotation",
        "",
        "| etf | group | status | RS 5d vs SPY | RS 20d | RS 60d | 20d ret | trend |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s | %s |"
            % (
                row.get("etf") or "",
                row.get("group") or "",
                row.get("status") or "",
                fmt_pct(row.get("rs_5")),
                fmt_pct(row.get("rs_20")),
                fmt_pct(row.get("rs_60")),
                fmt_pct(row.get("ret_20")),
                row.get("trend") or "",
            )
        )
    lines.append("")
    return lines
