"""Point-in-time GDELT article ingestion for event and geopolitical research."""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from codexswing.clock import NEW_YORK, UTC, iso_utc, utc_now
from codexswing.schemas.source import SourceRecord, canonical_json


GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search"


class GDELTError(RuntimeError):
    pass


class GoogleNewsRSSError(RuntimeError):
    pass


Transport = Callable[[Mapping[str, str]], Mapping[str, Any]]
RSS_TRANSPORT = Callable[[Mapping[str, str]], bytes]


def parse_gdelt_timestamp(value: str) -> datetime:
    text = value.strip()
    for pattern in ("%Y%m%dT%H%M%SZ", "%Y%m%d%H%M%S"):
        try:
            return datetime.strptime(text, pattern).replace(tzinfo=UTC)
        except ValueError:
            continue
    raise ValueError("unsupported GDELT timestamp")


class GDELTClient:
    def __init__(self, timeout_seconds: int = 30, transport: Optional[Transport] = None) -> None:
        self.timeout_seconds = timeout_seconds
        self._transport = transport or self._default_transport

    def _default_transport(self, params: Mapping[str, str]) -> Mapping[str, Any]:
        url = "{}?{}".format(GDELT_DOC_URL, urllib.parse.urlencode(dict(params)))
        request = urllib.request.Request(
            url,
            headers={"Accept": "application/json", "User-Agent": "codexswing/0.1"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            raise GDELTError("GDELT returned HTTP {}".format(exc.code)) from None
        except urllib.error.URLError as exc:
            raise GDELTError("GDELT request failed: {}".format(exc.reason)) from None
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise GDELTError("GDELT returned invalid JSON") from None
        if not isinstance(payload, Mapping):
            raise GDELTError("GDELT returned an unexpected payload")
        return payload

    def fetch_articles(
        self,
        query: str,
        start_utc: datetime,
        end_utc: datetime,
        max_records: int = 100,
    ) -> Tuple[Mapping[str, Any], ...]:
        if not query.strip():
            raise ValueError("GDELT query is required")
        if start_utc.tzinfo is None or end_utc.tzinfo is None:
            raise ValueError("GDELT window must be timezone-aware")
        start = start_utc.astimezone(UTC)
        end = end_utc.astimezone(UTC)
        if start >= end:
            raise ValueError("GDELT start must be before end")
        if max_records < 1 or max_records > 250:
            raise ValueError("GDELT max_records must be between 1 and 250")
        params = {
            "query": query.strip(),
            "mode": "ArtList",
            "format": "json",
            "sort": "DateDesc",
            "maxrecords": str(max_records),
            "startdatetime": start.strftime("%Y%m%d%H%M%S"),
            "enddatetime": end.strftime("%Y%m%d%H%M%S"),
        }
        payload = self._transport(params)
        articles = payload.get("articles")
        if not isinstance(articles, list):
            raise GDELTError("GDELT response is missing an articles array")
        if not all(isinstance(article, Mapping) for article in articles):
            raise GDELTError("GDELT returned a non-object article")
        return tuple(articles)

    def articles_to_records(
        self,
        articles: Iterable[Mapping[str, Any]],
        ingested_at: Optional[datetime] = None,
    ) -> Tuple[SourceRecord, ...]:
        ingestion_time = (ingested_at or utc_now()).astimezone(UTC)
        records: List[SourceRecord] = []
        for article in articles:
            url = str(article.get("url") or "").strip()
            title = str(article.get("title") or "").strip()
            seen = str(article.get("seendate") or "").strip()
            if not url or not title or not seen:
                raise GDELTError("GDELT article requires url, title, and seendate")
            published = parse_gdelt_timestamp(seen)
            if published > ingestion_time + timedelta(minutes=5):
                raise GDELTError("GDELT article is future-dated relative to ingestion")
            session_date = published.astimezone(NEW_YORK).date().isoformat()
            digest = hashlib.sha256(
                canonical_json({"url": url, "title": title, "seendate": seen}).encode("utf-8")
            ).hexdigest()
            records.append(
                SourceRecord(
                    source="gdelt_articles",
                    source_id=digest,
                    session_date=session_date,
                    event_time_utc=iso_utc(published),
                    published_at_utc=iso_utc(published),
                    first_seen_at_utc=iso_utc(ingestion_time),
                    available_at_utc=iso_utc(published),
                    ingested_at_utc=iso_utc(ingestion_time),
                    source_uri=GDELT_DOC_URL,
                    revision=digest,
                    payload=dict(article),
                )
            )
        return tuple(records)


class GoogleNewsRSSClient:
    """Time-windowed, source-cited news fallback for shadow context only."""

    def __init__(
        self,
        timeout_seconds: int = 30,
        transport: Optional[RSS_TRANSPORT] = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self._transport = transport or self._default_transport

    def _default_transport(self, params: Mapping[str, str]) -> bytes:
        url = "{}?{}".format(GOOGLE_NEWS_RSS_URL, urllib.parse.urlencode(dict(params)))
        request = urllib.request.Request(
            url,
            headers={"Accept": "application/rss+xml, application/xml", "User-Agent": "codexswing/0.3"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            raise GoogleNewsRSSError(
                "Google News RSS returned HTTP {}".format(exc.code)
            ) from None
        except urllib.error.URLError as exc:
            raise GoogleNewsRSSError(
                "Google News RSS request failed: {}".format(exc.reason)
            ) from None

    def fetch_articles(
        self,
        query: str,
        start_utc: datetime,
        end_utc: datetime,
        max_records: int = 75,
    ) -> Tuple[Mapping[str, Any], ...]:
        if not query.strip():
            raise ValueError("Google News query is required")
        if start_utc.tzinfo is None or end_utc.tzinfo is None:
            raise ValueError("Google News window must be timezone-aware")
        start = start_utc.astimezone(UTC)
        end = end_utc.astimezone(UTC)
        if start >= end:
            raise ValueError("Google News start must be before end")
        if max_records < 1 or max_records > 100:
            raise ValueError("Google News max_records must be between 1 and 100")
        before_date = (end + timedelta(days=1)).date().isoformat()
        dated_query = "{} after:{} before:{}".format(
            query.strip(), start.date().isoformat(), before_date
        )
        body = self._transport(
            {
                "q": dated_query,
                "hl": "en-US",
                "gl": "US",
                "ceid": "US:en",
            }
        )
        try:
            root = ET.fromstring(body)
        except (ET.ParseError, TypeError, ValueError):
            raise GoogleNewsRSSError("Google News RSS returned invalid XML") from None
        articles: List[Mapping[str, Any]] = []
        for item in root.findall("./channel/item"):
            title = (item.findtext("title") or "").strip()
            url = (item.findtext("link") or "").strip()
            published_text = (item.findtext("pubDate") or "").strip()
            source_node = item.find("source")
            source_name = (source_node.text or "").strip() if source_node is not None else ""
            source_url = (source_node.attrib.get("url") or "").strip() if source_node is not None else ""
            try:
                published = parsedate_to_datetime(published_text).astimezone(UTC)
            except (TypeError, ValueError, OverflowError):
                continue
            if not title or not url or not (start <= published <= end):
                continue
            articles.append(
                {
                    "title": title,
                    "url": url,
                    "publishedAt": iso_utc(published),
                    "source": source_name,
                    "sourceUrl": source_url,
                }
            )
            if len(articles) >= max_records:
                break
        return tuple(articles)

    def articles_to_records(
        self,
        articles: Iterable[Mapping[str, Any]],
        ingested_at: Optional[datetime] = None,
    ) -> Tuple[SourceRecord, ...]:
        ingestion_time = (ingested_at or utc_now()).astimezone(UTC)
        records: List[SourceRecord] = []
        for article in articles:
            url = str(article.get("url") or "").strip()
            title = str(article.get("title") or "").strip()
            try:
                published = datetime.fromisoformat(
                    str(article.get("publishedAt") or "").replace("Z", "+00:00")
                ).astimezone(UTC)
            except (TypeError, ValueError):
                raise GoogleNewsRSSError(
                    "Google News article requires a valid publication time"
                ) from None
            if not url or not title:
                raise GoogleNewsRSSError("Google News article requires url and title")
            if published > ingestion_time + timedelta(minutes=5):
                raise GoogleNewsRSSError(
                    "Google News article is future-dated relative to ingestion"
                )
            digest = hashlib.sha256(
                canonical_json(
                    {"url": url, "title": title, "publishedAt": iso_utc(published)}
                ).encode("utf-8")
            ).hexdigest()
            records.append(
                SourceRecord(
                    source="google_news_rss",
                    source_id=digest,
                    session_date=published.astimezone(NEW_YORK).date().isoformat(),
                    event_time_utc=iso_utc(published),
                    published_at_utc=iso_utc(published),
                    first_seen_at_utc=iso_utc(ingestion_time),
                    available_at_utc=iso_utc(published),
                    ingested_at_utc=iso_utc(ingestion_time),
                    source_uri=GOOGLE_NEWS_RSS_URL,
                    revision=digest,
                    payload=dict(article),
                )
            )
        return tuple(records)
