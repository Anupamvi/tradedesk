from __future__ import annotations

import datetime as dt
import json
import math
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests


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
NASDAQ_EARNINGS_URL = "https://api.nasdaq.com/api/calendar/earnings"
NASDAQ_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.nasdaq.com/",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
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


def earnings_event_date(row: pd.Series | dict[str, Any]) -> dt.date | None:
    for key in ["catalyst_earnings_date", "next_earnings_dt", "next_earnings_date", "earnings_date"]:
        value = row.get(key)
        parsed = _parse_date(value)
        if parsed is not None:
            return parsed
    return None


def earnings_crosses_expiry(
    row: pd.Series | dict[str, Any],
    *,
    asof: dt.date | None = None,
) -> bool:
    event_date = earnings_event_date(row)
    expiry = _parse_date(row.get("expiry"))
    if event_date is None or expiry is None:
        return False
    if asof is not None and event_date < asof:
        return False
    return event_date <= expiry


def _event_from_record(record: dict[str, Any], *, default_source: str = "") -> dict[str, Any] | None:
    ticker = str(record.get("ticker") or record.get("symbol") or "").upper().strip()
    event_date = _parse_date(
        record.get("earnings_date")
        or record.get("next_earnings_date")
        or record.get("event_date")
        or record.get("date")
        or record.get("report_date")
    )
    if not ticker or event_date is None:
        return None
    return {
        "ticker": ticker,
        "event_date": event_date,
        "status": str(record.get("catalyst_status") or record.get("status") or "").lower().strip(),
        "note": str(record.get("catalyst_note") or record.get("note") or record.get("event") or ""),
        "source": str(record.get("source") or record.get("source_url") or default_source),
        "confidence": str(record.get("confidence") or record.get("source_confidence") or "unknown"),
        "resolution": str(record.get("resolution") or "structured"),
    }


def _fetch_nasdaq_day(day: dt.date) -> list[dict[str, Any]]:
    try:
        response = requests.get(
            NASDAQ_EARNINGS_URL,
            params={"date": day.isoformat()},
            headers=NASDAQ_HEADERS,
            timeout=12,
        )
        response.raise_for_status()
        return list(((response.json().get("data") or {}).get("rows") or []))
    except Exception:
        return []


def _fetch_nasdaq_earnings_events(
    tickers: Iterable[str],
    *,
    start: dt.date,
    through: dt.date,
    cache_path: Path | None = None,
) -> tuple[dict[str, dict[str, Any]], str]:
    wanted = {str(ticker).upper().strip() for ticker in tickers if str(ticker).strip()}
    through = min(through, start + dt.timedelta(days=120))
    if not wanted or through < start:
        return {}, "not_needed"

    if cache_path is not None and cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            cached_start = _parse_date(payload.get("start"))
            cached_through = _parse_date(payload.get("through"))
            cached_tickers = {str(ticker).upper() for ticker in payload.get("queried_tickers", [])}
            if cached_start and cached_through and cached_start <= start and cached_through >= through and wanted <= cached_tickers:
                events: dict[str, dict[str, Any]] = {}
                for record in payload.get("events", []):
                    event = _event_from_record(record, default_source=NASDAQ_EARNINGS_URL)
                    if event and event["ticker"] in wanted:
                        events[event["ticker"]] = event
                return events, "cache"
        except Exception:
            pass

    days = []
    cursor = start
    while cursor <= through:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor += dt.timedelta(days=1)

    matched: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    successful_days = 0
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(days)))) as pool:
        futures = {pool.submit(_fetch_nasdaq_day, day): day for day in days}
        for future in as_completed(futures):
            day = futures[future]
            rows = future.result()
            if rows:
                successful_days += 1
            for raw in rows:
                ticker = str(raw.get("symbol") or "").upper().strip()
                if ticker not in wanted:
                    continue
                record = {
                    "ticker": ticker,
                    "earnings_date": day.isoformat(),
                    "status": "",
                    "note": f"Nasdaq web earnings calendar; timing={raw.get('time') or 'not supplied'}.",
                    "source": f"{NASDAQ_EARNINGS_URL}?date={day.isoformat()}",
                    "confidence": "secondary_calendar",
                    "resolution": "web_nasdaq",
                }
                records.append(record)
                event = _event_from_record(record, default_source=NASDAQ_EARNINGS_URL)
                existing = matched.get(ticker)
                if event and (existing is None or event["event_date"] < existing["event_date"]):
                    matched[ticker] = event

    if cache_path is not None and successful_days:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {
                    "source": NASDAQ_EARNINGS_URL,
                    "start": start.isoformat(),
                    "through": through.isoformat(),
                    "queried_tickers": sorted(wanted),
                    "events": records,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    return matched, "web" if successful_days else "unavailable"


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
                "source": str(_first_present(row, ["source", "source_url"]) or path),
                "confidence": str(_first_present(row, ["confidence", "source_confidence"]) or "structured"),
                "resolution": str(_first_present(row, ["resolution"]) or "structured"),
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


def load_catalyst_context(
    base_dir: Path,
    tickers: Iterable[str],
    *,
    asof: dt.date | None = None,
    fallback_earnings: dict[str, object] | None = None,
    resolve_web: bool = False,
    web_through: dt.date | None = None,
    event_exempt_tickers: Iterable[str] | None = None,
) -> pd.DataFrame:
    texts = _read_browser_texts(base_dir)
    structured_events = _load_structured_events(base_dir)
    ticker_list = sorted({str(ticker or "").strip().upper() for ticker in tickers if str(ticker or "").strip()})
    exempt = {str(ticker).upper().strip() for ticker in (event_exempt_tickers or [])}
    for ticker, value in (fallback_earnings or {}).items():
        ticker = str(ticker).upper().strip()
        event_date = _parse_date(value)
        if not ticker or event_date is None:
            continue
        existing = structured_events.get(ticker)
        if existing is None or existing.get("event_date") is None or event_date < existing["event_date"]:
            structured_events[ticker] = {
                "event_date": event_date,
                "status": "",
                "note": "Earnings date carried from UW stock-screener next_earnings_dt.",
                "source": "stock_screener.next_earnings_dt",
                "confidence": "UW_estimated_or_scheduled",
                "resolution": "stock_screener",
            }

    web_status = "not_requested"
    if resolve_web and asof is not None and web_through is not None:
        web_events, web_status = _fetch_nasdaq_earnings_events(
            [ticker for ticker in ticker_list if ticker not in exempt],
            start=asof,
            through=web_through,
            cache_path=base_dir / "browser_text" / f"earnings-calendar-web-{asof}.json",
        )
        for ticker, event in web_events.items():
            existing = structured_events.get(ticker)
            if existing is None or existing.get("event_date") is None or event["event_date"] < existing["event_date"]:
                structured_events[ticker] = event

    rows = []
    for ticker_raw in ticker_list:
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
                        "confidence": "local_capture",
                        "resolution": "local_browser_capture",
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
            note = (
                "No structured, local, or web earnings/news evidence matched ticker."
                if resolve_web
                else "No local browser/news capture matched ticker."
            )
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
                "catalyst_resolution": str(structured_event.get("resolution") or ("event_exempt" if ticker in exempt else "unresolved")),
                "catalyst_source_confidence": str(structured_event.get("confidence") or ("not_applicable" if ticker in exempt else "unresolved")),
                "catalyst_web_lookup": web_status,
            }
        )
    return pd.DataFrame(rows)
