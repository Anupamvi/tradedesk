from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from .catalysts import earnings_crosses_expiry, earnings_event_date
from .data import safe_float


BULLISH_DIRECTIONS = {"Bull Put", "Bull Call"}
BEARISH_DIRECTIONS = {"Bear Call", "Bear Put"}
SOFT_FLOW_BLOCKERS = {"flow_not_directional:unclear", "flow_not_directional:spread_leg"}


def _token_set(value: object) -> set[str]:
    return {x.strip() for x in str(value or "").split(";") if x.strip() and x.strip().lower() not in {"", "nan", "none"}}


def _set_tokens(tokens: set[str]) -> str:
    return ";".join(sorted(tokens))


def _date_value(value: object) -> dt.date | None:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def _direction_from_candidate(row: pd.Series | dict[str, Any]) -> str:
    direction = str(row.get("direction") or "")
    if direction in BULLISH_DIRECTIONS:
        return "bullish"
    if direction in BEARISH_DIRECTIONS:
        return "bearish"
    return "unclear"


def _vwap_confirms(row: pd.Series | dict[str, Any]) -> bool:
    expected = _direction_from_candidate(row)
    confirmation = str(row.get("vwap_confirmation") or "")
    if expected == "bullish":
        return confirmation.startswith("bullish_above")
    if expected == "bearish":
        return confirmation.startswith("bearish_below")
    return False


def _news_confirmation(row: pd.Series | dict[str, Any], *, asof: dt.date, browser_news_present: bool) -> tuple[str, str]:
    catalyst = str(row.get("catalyst_status") or "").strip().lower()
    if earnings_crosses_expiry(row, asof=asof):
        return "blocked", f"earnings/event {earnings_event_date(row)} occurs on or before expiry"
    if catalyst == "caution":
        return "blocked", "catalyst_status=caution"
    if catalyst in {"supportive", "mixed"}:
        return "cleared", f"local catalyst status is {catalyst}"
    earnings = _date_value(row.get("next_earnings_dt") or row.get("catalyst_earnings_date"))
    if earnings is not None:
        days = (earnings - asof).days
        if 0 <= days <= 7:
            return "blocked", f"earnings/event within 7 days ({earnings})"
        if days > 7 and browser_news_present:
            return "cleared", f"UW earnings date {earnings} is outside 7 days and local macro/news capture exists"
    if browser_news_present and catalyst == "unknown":
        return "manual", "local news exists but no ticker-specific clearance was captured"
    return "unconfirmed", "no local news/catalyst evidence"


def _flow_confirmation(row: pd.Series | dict[str, Any]) -> tuple[str, str]:
    flow_quality = str(row.get("flow_quality") or "")
    oi = str(row.get("oi_carryover_status") or "")
    if flow_quality == "hedge" or flow_quality == "roll":
        return "blocked", f"flow classified as {flow_quality}"
    if oi == "contrary":
        return "blocked", "OI carryover conflicts with candidate direction"
    if flow_quality == "directional":
        return "cleared", "UW flow classified directional"
    if bool(row.get("flow_velocity_signal")) and _vwap_confirms(row) and oi in {"supportive", "matched_unconfirmed", "", "unavailable", "no_exact_match"}:
        return "cleared", "flow velocity plus tape-VWAP confirmation clears ambiguous flow"
    if bool(row.get("child_order_accumulation")) and _vwap_confirms(row):
        return "cleared", "child-order accumulation plus tape-VWAP confirmation clears ambiguous flow"
    return "manual", f"flow_quality={flow_quality or 'unknown'}; vwap={row.get('vwap_confirmation', '')}; oi={oi or 'unknown'}"


def build_confirmation_evidence(
    *,
    scored: pd.DataFrame,
    asof: dt.date,
    input_provenance: dict[str, Any] | None = None,
) -> pd.DataFrame:
    columns = [
        "ticker",
        "strategy",
        "direction",
        "expiry",
        "confirmation_status",
        "news_confirmation",
        "flow_confirmation",
        "news_reason",
        "flow_reason",
        "can_clear_scout_blockers",
    ]
    if scored.empty:
        return pd.DataFrame(columns=columns)
    browser_news_present = int((input_provenance or {}).get("browser_text_count", 0) or 0) > 0
    rows: list[dict[str, Any]] = []
    for _, row in scored.iterrows():
        news_status, news_reason = _news_confirmation(row, asof=asof, browser_news_present=browser_news_present)
        flow_status, flow_reason = _flow_confirmation(row)
        status = "cleared" if news_status == "cleared" and flow_status == "cleared" else "blocked" if "blocked" in {news_status, flow_status} else "manual"
        rows.append(
            {
                "ticker": row.get("ticker"),
                "strategy": row.get("strategy"),
                "direction": row.get("direction"),
                "expiry": row.get("expiry"),
                "confirmation_status": status,
                "news_confirmation": news_status,
                "flow_confirmation": flow_status,
                "news_reason": news_reason,
                "flow_reason": flow_reason,
                "can_clear_scout_blockers": bool(status == "cleared"),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def apply_confirmation_evidence(scored: pd.DataFrame, evidence: pd.DataFrame) -> pd.DataFrame:
    if scored.empty or evidence.empty:
        return scored.copy()
    out = scored.copy()
    keys = ["ticker", "strategy", "direction", "expiry"]
    ev = evidence.copy()
    ev["_key"] = ev[keys].astype(str).agg("|".join, axis=1)
    ev_map = ev.set_index("_key", drop=False)
    out["_confirm_key"] = out[keys].astype(str).agg("|".join, axis=1)
    for idx, row in out.iterrows():
        key = row["_confirm_key"]
        if key not in ev_map.index:
            continue
        match = ev_map.loc[key]
        if isinstance(match, pd.DataFrame):
            match = match.iloc[0]
        out.at[idx, "v3_confirmation_status"] = match.get("confirmation_status", "")
        out.at[idx, "v3_news_confirmation"] = match.get("news_confirmation", "")
        out.at[idx, "v3_flow_confirmation"] = match.get("flow_confirmation", "")
        out.at[idx, "v3_confirmation_reason"] = f"{match.get('news_reason', '')}; {match.get('flow_reason', '')}".strip("; ")
        penalties = _token_set(row.get("penalties"))
        news_cleared = str(match.get("news_confirmation") or "") == "cleared"
        flow_cleared = str(match.get("flow_confirmation") or "") == "cleared"
        if news_cleared:
            penalties.discard("news_unconfirmed")
        if flow_cleared:
            penalties -= SOFT_FLOW_BLOCKERS
        out.at[idx, "penalties"] = _set_tokens(penalties)
        if news_cleared and str(row.get("catalyst_status") or "").lower() == "unknown":
            out.at[idx, "catalyst_status"] = "mixed"
            out.at[idx, "catalyst_note"] = "V3 confirmation: no near-term event found in UW data; local macro/news capture present."
        if flow_cleared and str(row.get("flow_quality") or "") in {"unclear", "spread_leg", ""}:
            out.at[idx, "flow_quality"] = "directional"
            out.at[idx, "flow_quality_reason"] = str(match.get("flow_reason") or "V3 confirmation cleared ambiguous flow")
    return out.drop(columns=["_confirm_key"], errors="ignore")


def write_confirmation_evidence(out_dir: Path, asof: dt.date, evidence: pd.DataFrame) -> tuple[Path, Path, dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if "confirmation_status" not in evidence.columns:
        evidence = pd.DataFrame(columns=[
            "ticker",
            "strategy",
            "direction",
            "expiry",
            "confirmation_status",
            "news_confirmation",
            "flow_confirmation",
            "news_reason",
            "flow_reason",
            "can_clear_scout_blockers",
        ])
    csv_path = out_dir / f"codexdaily_v3_confirmation_evidence_{asof}.csv"
    evidence.to_csv(csv_path, index=False)
    summary = {
        "status": "ok",
        "rows": int(len(evidence)),
        "cleared": int(evidence["confirmation_status"].eq("cleared").sum()) if not evidence.empty else 0,
        "manual": int(evidence["confirmation_status"].eq("manual").sum()) if not evidence.empty else 0,
        "blocked": int(evidence["confirmation_status"].eq("blocked").sum()) if not evidence.empty else 0,
        "csv": str(csv_path),
    }
    json_path = out_dir / f"codexdaily_v3_confirmation_evidence_{asof}.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return csv_path, json_path, summary
