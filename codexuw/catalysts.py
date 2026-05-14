from __future__ import annotations

import datetime as dt
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


POSITIVE_WORDS = {
    "beat",
    "beats",
    "bullish",
    "climb",
    "higher",
    "rally",
    "risk-on",
    "strength",
    "strong",
    "upgrade",
}
NEGATIVE_WORDS = {
    "blockade",
    "crash",
    "drawdown",
    "inflation",
    "lower",
    "miss",
    "recession",
    "risk",
    "sell",
    "shock",
    "surges",
    "war",
    "weakening",
}
MACRO_RISK_WORDS = {"fed", "iran", "oil", "cpi", "vix", "rates", "inflation", "war"}
STRUCTURED_EVENT_FILES = [
    "catalysts*.csv",
    "catalysts*.json",
    "earnings*.csv",
    "earnings*.json",
    "earnings-calendar*.csv",
    "earnings-calendar*.json",
]
EVENT_DATE_TOKENS = {
    "conference call",
    "earnings",
    "financial results",
    "financial-results",
    "monthly revenue",
    "monthly sales",
    "revenue report",
    "results",
}
PRIMARY_EVENT_TOKENS = {
    "conference call",
    "earnings release",
    "earnings released",
    "earnings were released",
    "financial results",
    "financial-results",
    "monthly revenue",
    "monthly sales",
    "revenue report",
    "results released",
}


def _read_browser_texts(base_dir: Path) -> list[tuple[str, str]]:
    browser_dir = base_dir / "browser_text"
    if not browser_dir.is_dir():
        return []
    rows: list[tuple[str, str]] = []
    for path in sorted(browser_dir.glob("browser-text-capture-*")):
        if path.suffix.lower() not in {".txt", ".csv"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if text.strip():
            rows.append((path.name, text))
    return rows


def _mentions(text: str, ticker: str) -> bool:
    if ticker in {"SPY", "QQQ", "IWM"}:
        if re.search(r"\b(SPY|QQQ|IWM|SPX|S&P|NASDAQ|VIX)\b", text, re.IGNORECASE):
            return True
    return bool(re.search(rf"(?<![A-Z0-9]){re.escape(ticker)}(?![A-Z0-9])", text, re.IGNORECASE))


def _source_mentions_ticker(source_name: str, ticker: str) -> bool:
    return bool(re.search(rf"(?<![A-Z0-9]){re.escape(ticker)}(?![A-Z0-9])", source_name, re.IGNORECASE))


def _snippet(text: str, ticker: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines:
        if _mentions(line, ticker) or any(word in line.lower() for word in MACRO_RISK_WORDS):
            return line[:180]
    return ""


def _parse_date(value: object) -> dt.date | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    text = str(value)
    iso = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", text)
    if iso:
        try:
            return dt.datetime.strptime(iso.group(1), "%Y-%m-%d").date()
        except ValueError:
            return None
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def _parse_dates_from_text(text: str) -> list[dt.date]:
    dates: list[dt.date] = []
    for match in re.finditer(r"\b(20\d{2}-\d{2}-\d{2})\b", str(text)):
        try:
            dates.append(dt.datetime.strptime(match.group(1), "%Y-%m-%d").date())
        except ValueError:
            continue
    return dates


def _first_present(row: pd.Series, keys: list[str]) -> object:
    lower = {str(k).lower(): k for k in row.index}
    for key in keys:
        actual = lower.get(key.lower())
        if actual is not None:
            value = row.get(actual)
            if value is not None and not pd.isna(value) and str(value).strip():
                return value
    return None


def _load_structured_events(base_dir: Path) -> dict[str, dict[str, Any]]:
    events: dict[str, dict[str, Any]] = {}
    files: list[Path] = []
    for pattern in STRUCTURED_EVENT_FILES:
        files.extend(sorted(base_dir.glob(pattern)))
        browser_dir = base_dir / "browser_text"
        if browser_dir.is_dir():
            files.extend(sorted(browser_dir.glob(pattern)))
    for path in files:
        try:
            if path.suffix.lower() == ".csv":
                df = pd.read_csv(path)
                records = df.to_dict("records")
            elif path.suffix.lower() == ".json":
                payload = json.loads(path.read_text(encoding="utf-8"))
                records = payload if isinstance(payload, list) else payload.get("events", [])
            else:
                continue
        except Exception:
            continue
        for record in records:
            row = pd.Series(record)
            ticker = str(_first_present(row, ["ticker", "symbol", "underlying", "underlying_symbol"]) or "").upper().strip()
            if not ticker:
                continue
            event_date = _parse_date(
                _first_present(row, ["earnings_date", "next_earnings_date", "event_date", "date", "report_date"])
            )
            existing = events.get(ticker, {})
            if existing.get("event_date") and event_date and existing["event_date"] <= event_date:
                continue
            events[ticker] = {
                "event_date": event_date,
                "status": str(_first_present(row, ["catalyst_status", "status", "risk_status"]) or "").lower().strip(),
                "note": str(_first_present(row, ["catalyst_note", "note", "event", "description"]) or ""),
                "source": str(path),
            }
    return events


def _extract_earnings_date_from_text(text: str, ticker: str, *, source_name: str = "") -> dt.date | None:
    source_scoped = _source_mentions_ticker(source_name, ticker)
    if not _mentions(text, ticker) and not source_scoped:
        return None
    event_lines: list[str] = []
    for line in text.splitlines():
        lower = line.lower()
        if not any(token in lower for token in EVENT_DATE_TOKENS):
            continue
        if any(
            token in lower
            for token in [
                "block",
                "blocked",
                "blocker",
                "candidate",
                "expires",
                "expiry",
                "option",
                "reject",
                "setup",
                "spread",
                "trade",
            ]
        ) and not any(
            token in lower for token in PRIMARY_EVENT_TOKENS
        ):
            continue
        event_lines.append(line)
    event_lines = sorted(
        event_lines,
        key=lambda line: 0
        if any(token in line.lower() for token in PRIMARY_EVENT_TOKENS)
        else 1,
    )
    for line in event_lines:
        if not _mentions(line, ticker):
            continue
        dates = _parse_dates_from_text(line)
        if dates:
            return dates[-1]
        date_value = _parse_date(line)
        if date_value:
            return date_value
    if source_scoped:
        for line in event_lines:
            dates = _parse_dates_from_text(line)
            if dates:
                return dates[-1]
            date_value = _parse_date(line)
            if date_value:
                return date_value
    return None


def _status_from_event(event_date: dt.date | None, asof: dt.date | None, explicit_status: str) -> tuple[str, str, float]:
    if explicit_status in {"supportive", "mixed", "caution", "unknown"}:
        days = float((event_date - asof).days) if event_date and asof else math.nan
        return explicit_status, "Structured catalyst status supplied.", days
    if event_date and asof:
        days = float((event_date - asof).days)
        if days < 0:
            return "mixed", f"Structured earnings/event date has already passed ({event_date}).", days
        if 0 <= days <= 7:
            return "caution", f"Structured earnings/event date is within 7 days ({event_date}).", days
        if days > 7:
            return "mixed", f"Structured earnings/event date is known and outside 7 days ({event_date}).", days
    return "unknown", "No structured catalyst date/status available.", math.nan


def load_catalyst_context(base_dir: Path, tickers: Iterable[str], *, asof: dt.date | None = None) -> pd.DataFrame:
    texts = _read_browser_texts(base_dir)
    structured_events = _load_structured_events(base_dir)
    rows = []
    for ticker_raw in tickers:
        ticker = str(ticker_raw or "").strip().upper()
        if not ticker:
            continue
        source_hits: list[str] = []
        snippets: list[str] = []
        word_counts: Counter[str] = Counter()
        structured_event = structured_events.get(ticker, {})
        event_date = structured_event.get("event_date")
        if not event_date:
            for name, text in texts:
                extracted = _extract_earnings_date_from_text(text, ticker, source_name=name)
                if extracted:
                    event_date = extracted
                    structured_event = {
                        "event_date": event_date,
                        "status": "",
                        "note": "Earnings/event date extracted from local browser capture.",
                        "source": name,
                    }
                    break
        for name, text in texts:
            lower = text.lower()
            if _mentions(text, ticker):
                source_hits.append(name)
                snippets.append(_snippet(text, ticker))
                for word in POSITIVE_WORDS | NEGATIVE_WORDS | MACRO_RISK_WORDS:
                    word_counts[word] += lower.count(word)
        pos = sum(word_counts[w] for w in POSITIVE_WORDS)
        neg = sum(word_counts[w] for w in NEGATIVE_WORDS)
        macro = sum(word_counts[w] for w in MACRO_RISK_WORDS)
        event_source = str(structured_event.get("source") or "")
        if event_date or structured_event.get("status"):
            status, note, days = _status_from_event(event_date, asof, str(structured_event.get("status") or ""))
            if structured_event.get("note"):
                note = f"{note} {structured_event.get('note')}".strip()
            if event_source and event_source not in source_hits:
                source_hits.append(event_source)
        elif not source_hits:
            status = "unknown"
            note = "No local browser/news capture matched ticker."
            days = math.nan
        elif neg > pos + 2:
            status = "caution"
            note = "Local capture has more negative/macro-risk terms than positive terms."
            days = math.nan
        elif pos > neg:
            status = "supportive"
            note = "Local capture is net supportive."
            days = math.nan
        else:
            status = "mixed"
            note = "Local capture is mixed; do not treat news as a primary edge."
            days = math.nan
        rows.append(
            {
                "ticker": ticker,
                "catalyst_status": status,
                "catalyst_note": note,
                "catalyst_earnings_date": event_date,
                "catalyst_earnings_days": days,
                "news_hits": len(source_hits),
                "macro_risk_hits": int(macro),
                "positive_hits": int(pos),
                "negative_hits": int(neg),
                "catalyst_sources": ";".join(source_hits[:5]),
                "catalyst_snippet": " | ".join(x for x in snippets[:2] if x),
            }
        )
    return pd.DataFrame(rows)
