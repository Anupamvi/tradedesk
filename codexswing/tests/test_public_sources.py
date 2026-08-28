from datetime import datetime, timezone
from typing import Any, Mapping

import pytest

from codexswing.sources.events import GDELTClient, GoogleNewsRSSClient
from codexswing.sources.sec import SECSubmissionsClient


UTC = timezone.utc


def test_gdelt_articles_preserve_publication_time_and_evidence() -> None:
    seen_params = {}

    def transport(params: Mapping[str, str]) -> Mapping[str, Any]:
        seen_params.update(params)
        return {
            "articles": [
                {
                    "url": "https://news.example/geopolitical-event",
                    "title": "Shipping route disrupted",
                    "seendate": "20260827T120000Z",
                    "domain": "news.example",
                    "sourcecountry": "United States",
                    "language": "English",
                }
            ]
        }

    client = GDELTClient(transport=transport)
    articles = client.fetch_articles(
        query="shipping disruption",
        start_utc=datetime(2026, 8, 27, 0, 0, tzinfo=UTC),
        end_utc=datetime(2026, 8, 27, 13, 0, tzinfo=UTC),
    )
    records = client.articles_to_records(
        articles,
        ingested_at=datetime(2026, 8, 27, 12, 5, tzinfo=UTC),
    )
    assert seen_params["mode"] == "ArtList"
    assert len(records) == 1
    assert records[0].available_at_utc == "2026-08-27T12:00:00Z"
    assert records[0].payload["domain"] == "news.example"
    assert records[0].source_uri == "https://api.gdeltproject.org/api/v2/doc/doc"


def test_gdelt_future_article_fails_closed() -> None:
    client = GDELTClient(transport=lambda params: {"articles": []})
    with pytest.raises(Exception, match="future-dated"):
        client.articles_to_records(
            [
                {
                    "url": "https://news.example/future",
                    "title": "Future",
                    "seendate": "20260827T130000Z",
                }
            ],
            ingested_at=datetime(2026, 8, 27, 12, 0, tzinfo=UTC),
        )


def test_google_news_rss_fallback_is_time_filtered_and_source_cited() -> None:
    seen_params = {}
    xml = b"""<?xml version="1.0" encoding="UTF-8"?>
    <rss><channel>
      <item><title>Market shipping update</title><link>https://news.google.com/example</link>
      <pubDate>Thu, 27 Aug 2026 12:00:00 GMT</pubDate>
      <source url="https://example.com">Example News</source></item>
      <item><title>Outside window</title><link>https://news.google.com/old</link>
      <pubDate>Mon, 17 Aug 2026 12:00:00 GMT</pubDate></item>
    </channel></rss>"""

    def transport(params: Mapping[str, str]) -> bytes:
        seen_params.update(params)
        return xml

    client = GoogleNewsRSSClient(transport=transport)
    articles = client.fetch_articles(
        "markets geopolitics",
        datetime(2026, 8, 24, 0, 0, tzinfo=UTC),
        datetime(2026, 8, 27, 23, 59, tzinfo=UTC),
    )
    records = client.articles_to_records(
        articles,
        ingested_at=datetime(2026, 8, 28, 0, 5, tzinfo=UTC),
    )
    assert "after:2026-08-24" in seen_params["q"]
    assert len(records) == 1
    assert records[0].source == "google_news_rss"
    assert records[0].payload["source"] == "Example News"
    assert records[0].published_at_utc == "2026-08-27T12:00:00Z"


def test_sec_recent_filings_normalization_and_timestamp() -> None:
    company_payload = {
        "filings": {
            "recent": {
                "accessionNumber": ["0000320193-26-000100"],
                "filingDate": ["2026-08-27"],
                "reportDate": ["2026-08-27"],
                "acceptanceDateTime": ["2026-08-27T17:30:00"],
                "act": ["34"],
                "form": ["8-K"],
                "fileNumber": ["001-36743"],
                "filmNumber": ["26123456"],
                "items": ["2.02"],
                "size": [12345],
                "isXBRL": [1],
                "isInlineXBRL": [1],
                "primaryDocument": ["sample-8k.htm"],
                "primaryDocDescription": ["Current report"],
            }
        }
    }
    client = SECSubmissionsClient(
        "CodexSwing Research contact@example.com",
        transport=lambda url: company_payload,
    )
    fetched = client.fetch_company("320193")
    filings = client.recent_filings(fetched)
    records = client.filings_to_records(
        "320193",
        filings,
        forms=["8-K"],
        ingested_at=datetime(2026, 8, 27, 22, 0, tzinfo=UTC),
    )
    assert len(records) == 1
    assert records[0].source_id == "0000320193:0000320193-26-000100"
    assert records[0].available_at_utc == "2026-08-27T21:30:00Z"
    assert records[0].source_uri.endswith("/320193/000032019326000100/sample-8k.htm")


def test_sec_requires_contact_user_agent() -> None:
    with pytest.raises(ValueError, match="contact email"):
        SECSubmissionsClient("codexswing")
