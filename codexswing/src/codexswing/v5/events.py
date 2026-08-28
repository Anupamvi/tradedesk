"""Conservative earnings and dividend-assignment exclusions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Iterable, List, Mapping, Sequence, Tuple


EARNINGS_KEYS = ("earnDate", "earningsDate", "nextErn", "nextEarningsDate", "ernDate")
EX_DIVIDEND_KEYS = ("exDivDate", "exDividendDate", "divDate", "nextDivDate")
SHORT_CALL_STRATEGIES = {"BULL_CALL_DEBIT", "BEAR_CALL_CREDIT"}


def _date_text(value: Any, label: str) -> str:
    candidate = str(value or "")[:10]
    try:
        date.fromisoformat(candidate)
    except ValueError:
        raise ValueError("invalid {}".format(label)) from None
    return candidate


def _optional_date_text(value: Any, label: str) -> str:
    candidate = str(value or "")[:10]
    if candidate in {"", "0000-00-00"}:
        return ""
    return _date_text(candidate, label)


@dataclass(frozen=True, order=True)
class CorporateEvent:
    ticker: str
    event_date: str
    event_type: str
    source_field: str

    def __post_init__(self) -> None:
        if not self.ticker or self.ticker != self.ticker.upper():
            raise ValueError("event ticker must be uppercase")
        _date_text(self.event_date, "event_date")
        if self.event_type not in {"EARNINGS", "EX_DIVIDEND"}:
            raise ValueError("unsupported event type")


def parse_orats_events(
    ticker: str, rows: Iterable[Mapping[str, Any]]
) -> Tuple[CorporateEvent, ...]:
    normalized = ticker.strip().upper()
    events = set()
    for row in rows:
        row_ticker = str(row.get("ticker") or row.get("symbol") or normalized).upper()
        if row_ticker != normalized:
            continue
        generic_type = str(row.get("eventType") or "").strip().upper()
        generic_date = _optional_date_text(row.get("eventDate"), "eventDate")
        if generic_type in {"EARNINGS", "EARN", "ERN"} and generic_date:
            events.add(
                CorporateEvent(
                    normalized, generic_date, "EARNINGS", "eventDate"
                )
            )
        if generic_type in {"EX_DIVIDEND", "EX-DIVIDEND", "DIVIDEND"} and generic_date:
            events.add(
                CorporateEvent(
                    normalized,
                    generic_date,
                    "EX_DIVIDEND",
                    "eventDate",
                )
            )
        for key in EARNINGS_KEYS:
            event_date = _optional_date_text(row.get(key), key)
            if event_date:
                events.add(
                    CorporateEvent(
                        normalized, event_date, "EARNINGS", key
                    )
                )
        for key in EX_DIVIDEND_KEYS:
            event_date = _optional_date_text(row.get(key), key)
            if event_date:
                events.add(
                    CorporateEvent(
                        normalized, event_date, "EX_DIVIDEND", key
                    )
                )
    return tuple(sorted(events))


@dataclass(frozen=True)
class EventExclusionDecision:
    eligible: bool
    reasons: Tuple[str, ...]
    blocked_events: Tuple[CorporateEvent, ...]


def evaluate_event_exclusions(
    ticker: str,
    strategy: str,
    entry_date: str,
    planned_exit_date: str,
    events: Sequence[CorporateEvent],
) -> EventExclusionDecision:
    """Fail closed for earnings and short-call ex-dividend assignment windows."""

    normalized = ticker.strip().upper()
    entry = date.fromisoformat(_date_text(entry_date, "entry_date"))
    exit_day = date.fromisoformat(_date_text(planned_exit_date, "planned_exit_date"))
    if exit_day < entry:
        raise ValueError("planned exit cannot precede entry")
    blocked: List[CorporateEvent] = []
    reasons: List[str] = []
    for event in sorted(events):
        if event.ticker != normalized:
            continue
        event_day = date.fromisoformat(event.event_date)
        if event.event_type == "EARNINGS" and entry <= event_day <= exit_day:
            blocked.append(event)
            reasons.append("EARNINGS_WINDOW:{}".format(event.event_date))
        elif event.event_type == "EX_DIVIDEND" and strategy in SHORT_CALL_STRATEGIES:
            assignment_exposure_start = event_day - timedelta(days=1)
            if entry <= event_day and exit_day >= assignment_exposure_start:
                blocked.append(event)
                reasons.append("SHORT_CALL_DIVIDEND_ASSIGNMENT_WINDOW:{}".format(event.event_date))
    return EventExclusionDecision(
        eligible=not blocked,
        reasons=tuple(reasons),
        blocked_events=tuple(blocked),
    )
