from __future__ import annotations

import math
import re
from typing import Any

import pandas as pd


LEG_RE = re.compile(
    r"\b(?P<side>sell|sold|sto|sell_to_open|buy|bought|bto|buy_to_open)\s+"
    r"(?P<ticker>[A-Z][A-Z0-9./-]*)\s+"
    r"(?P<expiry>20\d{2}-\d{2}-\d{2})\s+"
    r"\$?(?P<strike>\d+(?:\.\d+)?)\s*(?P<right>P|PUT|C|CALL)\b",
    re.IGNORECASE,
)


def _clean(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _side(value: str) -> str:
    text = value.lower()
    if text.startswith("s"):
        return "sell"
    return "buy"


def _right(value: str) -> str:
    text = value.upper()
    return "P" if text.startswith("P") else "C"


def extract_trade_legs(text: object) -> list[dict[str, Any]]:
    """Extract simple buy/sell option legs from report or ledger text."""
    legs: list[dict[str, Any]] = []
    for match in LEG_RE.finditer(_clean(text).replace("/", " / ")):
        legs.append(
            {
                "side": _side(match.group("side")),
                "ticker": match.group("ticker").upper().replace("-", "/"),
                "expiry": match.group("expiry"),
                "strike": float(match.group("strike")),
                "right": _right(match.group("right")),
            }
        )
    return legs


def _leg_by_side(legs: list[dict[str, Any]], side: str) -> dict[str, Any] | None:
    for leg in legs:
        if leg.get("side") == side:
            return leg
    return None


def _width(legs: list[dict[str, Any]]) -> float:
    sell = _leg_by_side(legs, "sell")
    buy = _leg_by_side(legs, "buy")
    if not sell or not buy:
        return math.nan
    return abs(float(sell["strike"]) - float(buy["strike"]))


def compare_leg_drift(recommended_text: object, actual_text: object) -> dict[str, Any]:
    recommended = extract_trade_legs(recommended_text)
    actual = extract_trade_legs(actual_text)
    reasons: list[str] = []
    if len(recommended) != len(actual) or len(recommended) < 2:
        reasons.append("leg_count_changed")
    for side in ("sell", "buy"):
        rec = _leg_by_side(recommended, side)
        act = _leg_by_side(actual, side)
        if not rec or not act:
            continue
        for key in ("ticker", "expiry", "right"):
            if rec.get(key) != act.get(key):
                reasons.append(f"{side}_{key}_changed:{rec.get(key)}->{act.get(key)}")
        if float(rec.get("strike", math.nan)) != float(act.get("strike", math.nan)):
            reasons.append(f"{side}_strike_changed:{rec.get('strike'):g}->{act.get('strike'):g}")
    rec_width = _width(recommended)
    act_width = _width(actual)
    if math.isfinite(rec_width) and math.isfinite(act_width) and rec_width != act_width:
        reasons.append(f"width_changed:{rec_width:g}->{act_width:g}")
    return {
        "recommended_legs": recommended,
        "actual_legs": actual,
        "drift_detected": bool(reasons),
        "drift_reason": ";".join(reasons),
        "status": "UNAPPROVED LEG DRIFT - re-score required" if reasons else "matched",
    }


def build_leg_drift_audit(recommendations: pd.DataFrame, fills: pd.DataFrame) -> pd.DataFrame:
    """Compare actual filled tickets against the latest same-ticker recommendation."""
    columns = [
        "ticker",
        "recommended_trade",
        "actual_trade",
        "drift_detected",
        "drift_reason",
        "status",
    ]
    if recommendations is None or recommendations.empty or fills is None or fills.empty:
        return pd.DataFrame(columns=columns)
    recs = recommendations.copy()
    fills = fills.copy()
    recs["_ticker"] = recs.get("ticker", recs.get("Ticker", "")).astype(str).str.upper().str.strip()
    fills["_ticker"] = fills.get("ticker", fills.get("Ticker", "")).astype(str).str.upper().str.strip()
    if "generated_at" in recs.columns:
        recs = recs.sort_values("generated_at")
    elif "report_date" in recs.columns:
        recs = recs.sort_values("report_date")

    rows: list[dict[str, Any]] = []
    for _, fill in fills.iterrows():
        ticker = _clean(fill.get("_ticker")).upper()
        if not ticker:
            continue
        matches = recs[recs["_ticker"].eq(ticker)]
        if matches.empty:
            rows.append(
                {
                    "ticker": ticker,
                    "recommended_trade": "",
                    "actual_trade": _clean(fill.get("trade") or fill.get("Trade") or fill.get("actual_trade")),
                    "drift_detected": True,
                    "drift_reason": "no_recent_recommendation",
                    "status": "UNAPPROVED LEG DRIFT - re-score required",
                }
            )
            continue
        rec = matches.iloc[-1]
        recommended_text = rec.get("trade") or rec.get("Trade") or rec.get("recommendation_text") or rec.get("recommendation") or ""
        actual_text = fill.get("trade") or fill.get("Trade") or fill.get("actual_trade") or ""
        comparison = compare_leg_drift(recommended_text, actual_text)
        rows.append(
            {
                "ticker": ticker,
                "recommended_trade": _clean(recommended_text),
                "actual_trade": _clean(actual_text),
                "drift_detected": comparison["drift_detected"],
                "drift_reason": comparison["drift_reason"],
                "status": comparison["status"],
            }
        )
    return pd.DataFrame(rows, columns=columns)
