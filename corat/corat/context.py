"""Evidence-bound catalyst, event, X, and flow context ingestion."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from corat.constants import DATA_UNAVAILABLE
from corat.store import sha256_file


ALLOWED_CLASSIFICATIONS = {"FACT", "REPORTED INFORMATION", "RUMOR / X SPECULATION"}
ALLOWED_CREDIBILITY = {"PRIMARY", "HIGH", "MEDIUM", "LOW"}
ALLOWED_DIRECTIONS = {"BULLISH", "BEARISH", "NEUTRAL", "MIXED", "UNKNOWN"}
ALLOWED_FRESHNESS = {
    "NEW",
    "DEVELOPING",
    "KNOWN BUT UNDER-APPRECIATED",
    "KNOWN BUT POTENTIALLY UNDER-APPRECIATED",
    "FULLY PRICED",
    "STALE",
    DATA_UNAVAILABLE,
}


def _day(text: Any) -> Optional[date]:
    value = str(text or "")[:10]
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _freshness(published: Optional[date], as_of: date) -> str:
    if published is None:
        return DATA_UNAVAILABLE
    age = (as_of - published).days
    if age < 0:
        return "FUTURE-DATED / INVALID"
    if age <= 2:
        return "NEW"
    if age <= 7:
        return "DEVELOPING"
    if age <= 21:
        return "KNOWN BUT POTENTIALLY UNDER-APPRECIATED"
    return "STALE"


def empty_context(as_of: str, reason: str = "No structured evidence file supplied") -> Dict[str, Any]:
    return {
        "schema_version": "corat.context.v1",
        "as_of": as_of,
        "status": DATA_UNAVAILABLE,
        "reason": reason,
        "market_events": [],
        "tickers": {},
        "source_path": "",
        "source_sha256": "",
    }


def load_context(path: Optional[Path], as_of: str) -> Dict[str, Any]:
    if path is None:
        return empty_context(as_of)
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        return empty_context(as_of, "Context file not found: {}".format(resolved))
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != "corat.context.v1":
        raise ValueError("context must use schema_version corat.context.v1")
    context_day = _day(payload.get("as_of"))
    decision_day = date.fromisoformat(as_of)
    if context_day is None or context_day > decision_day:
        raise ValueError("context as_of is missing or after the decision date")
    market_events = payload.get("market_events", [])
    if not isinstance(market_events, list):
        raise ValueError("context market_events must be a list")
    tickers = payload.get("tickers")
    if not isinstance(tickers, dict):
        raise ValueError("context tickers must be an object")
    evidence_families = [("MARKET", "market_events", market_events)]
    for ticker, value in tickers.items():
        if not isinstance(value, dict):
            raise ValueError("context entry for {} must be an object".format(ticker))
        for family in ("catalysts", "x_intelligence", "events", "options_flow"):
            rows = value.get(family, [])
            if not isinstance(rows, list):
                raise ValueError("{}.{} must be a list".format(ticker, family))
            evidence_families.append((str(ticker), family, rows))
    for owner, family, rows in evidence_families:
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("{}.{} entries must be objects".format(owner, family))
            classification = str(row.get("classification") or "")
            credibility = str(row.get("credibility") or "")
            if classification not in ALLOWED_CLASSIFICATIONS:
                raise ValueError("invalid or missing context classification for {}".format(owner))
            if credibility not in ALLOWED_CREDIBILITY:
                raise ValueError("invalid or missing context credibility for {}".format(owner))
            if not str(row.get("source") or "").strip():
                raise ValueError("context source name required for {}".format(owner))
            source_url = str(row.get("source_url") or "")
            if not source_url.startswith(("https://", "http://")):
                raise ValueError("context source_url required for {}".format(owner))
            published = _day(row.get("published_at"))
            if published is None:
                raise ValueError("context published_at required for {}".format(owner))
            if published > decision_day:
                raise ValueError("future context evidence for {}".format(owner))
            if not str(row.get("title") or row.get("claim") or "").strip():
                raise ValueError("context title or claim required for {}".format(owner))
            direction = str(row.get("direction") or "UNKNOWN").upper()
            if direction not in ALLOWED_DIRECTIONS:
                raise ValueError("invalid context direction for {}".format(owner))
            row["direction"] = direction
            freshness = str(row.get("freshness") or _freshness(published, decision_day))
            if freshness == "KNOWN BUT UNDER-APPRECIATED":
                freshness = "KNOWN BUT POTENTIALLY UNDER-APPRECIATED"
            if freshness not in ALLOWED_FRESHNESS:
                raise ValueError("invalid context freshness for {}".format(owner))
            row["freshness"] = freshness
            event_date = row.get("event_date")
            if family in {"market_events", "events"} and _day(event_date) is None:
                raise ValueError("context event_date required for {}".format(owner))
            if event_date not in (None, "") and _day(event_date) is None:
                raise ValueError("invalid event_date for {}".format(owner))
    payload = dict(payload)
    payload["status"] = "AVAILABLE"
    payload["source_path"] = str(resolved)
    payload["source_sha256"] = sha256_file(resolved)
    return payload


def ticker_context(context: Mapping[str, Any], ticker: str, as_of: str) -> Dict[str, Any]:
    raw = (context.get("tickers") or {}).get(ticker, {}) if isinstance(context.get("tickers"), dict) else {}
    if not isinstance(raw, dict):
        raw = {}
    catalysts = list(raw.get("catalysts") or [])
    x_rows = list(raw.get("x_intelligence") or [])
    events = list(raw.get("events") or [])
    flows = list(raw.get("options_flow") or [])
    catalyst_points = 0.0
    catalyst_strength_by_direction = {"BULLISH": 0.0, "BEARISH": 0.0}
    actionable_catalysts = []
    for row in catalysts:
        credibility = {"PRIMARY": 1.0, "HIGH": 0.85, "MEDIUM": 0.55, "LOW": 0.2}.get(str(row.get("credibility")), 0.0)
        classification = {"FACT": 1.0, "REPORTED INFORMATION": 0.7, "RUMOR / X SPECULATION": 0.15}.get(str(row.get("classification")), 0.0)
        freshness = {"NEW": 1.0, "DEVELOPING": 0.85, "KNOWN BUT POTENTIALLY UNDER-APPRECIATED": 0.55, "FULLY PRICED": 0.15, "STALE": 0.0}.get(str(row.get("freshness")), 0.0)
        direction_name = str(row.get("direction") or "UNKNOWN").upper()
        direction_weight = 1.0 if direction_name in {"BULLISH", "BEARISH"} else 0.25
        points = credibility * classification * freshness * direction_weight
        catalyst_points = max(catalyst_points, points)
        if direction_name in catalyst_strength_by_direction:
            catalyst_strength_by_direction[direction_name] = max(catalyst_strength_by_direction[direction_name], points)
        if (
            str(row.get("classification")) in {"FACT", "REPORTED INFORMATION"}
            and str(row.get("credibility")) in {"PRIMARY", "HIGH"}
            and str(row.get("freshness")) in {"NEW", "DEVELOPING", "KNOWN BUT POTENTIALLY UNDER-APPRECIATED"}
            and direction_name in {"BULLISH", "BEARISH"}
        ):
            actionable_catalysts.append(row)
    credible_x = [row for row in x_rows if str(row.get("credibility")) in {"PRIMARY", "HIGH"} and not bool(row.get("spam_risk"))]
    x_score = min(1.0, len(credible_x) / 3.0)
    credible_flow = [row for row in flows if str(row.get("credibility")) in {"PRIMARY", "HIGH"}]
    flow_score = min(1.0, len(credible_flow) / 2.0)
    status = "AVAILABLE" if catalysts or x_rows or events or flows else DATA_UNAVAILABLE
    return {
        "status": status,
        "catalysts": catalysts,
        "actionable_catalysts": actionable_catalysts,
        "catalyst_strength_by_direction": catalyst_strength_by_direction,
        "x_intelligence": x_rows,
        "events": events,
        "options_flow": flows,
        "catalyst_strength": catalyst_points,
        "x_strength": x_score,
        "flow_strength": flow_score,
        "x_spam_risk": any(bool(row.get("spam_risk")) for row in x_rows),
        "mention_acceleration": raw.get("mention_acceleration", DATA_UNAVAILABLE),
        "source_count": sum(len(rows) for rows in (catalysts, x_rows, events, flows)),
    }


def event_risks(
    context: Mapping[str, Any],
    ticker_context_value: Mapping[str, Any],
    as_of: str,
    holding_sessions: int,
) -> List[Mapping[str, Any]]:
    """Return sourced events whose event date falls inside the planned hold.

    Events are disclosed rather than mechanically rejected. The scoring layer
    may still reject earnings separately because that is an explicit hard rule.
    """

    start = date.fromisoformat(as_of)
    end = start + timedelta(days=max(1, int(holding_sessions * 1.6)))
    rows = list(context.get("market_events") or []) + list(ticker_context_value.get("events") or [])
    result = []
    for row in rows:
        event_day = _day(row.get("event_date"))
        if event_day is not None and start <= event_day <= end:
            result.append(row)
    return sorted(result, key=lambda row: str(row.get("event_date") or ""))


def context_template(as_of: str, tickers: Iterable[str]) -> Dict[str, Any]:
    return {
        "schema_version": "corat.context.v1",
        "as_of": as_of,
        "market_events": [],
        "tickers": {
            str(ticker).upper(): {
                "catalysts": [],
                "x_intelligence": [],
                "events": [],
                "options_flow": [],
                "mention_acceleration": DATA_UNAVAILABLE,
            }
            for ticker in tickers
        },
    }
