import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from corat.research import _ticker_news, build_auto_context, classify_headline_direction, parse_news_rss


class ResearchTest(unittest.TestCase):
    def test_headline_classifier_requires_explicit_event_language(self):
        self.assertEqual(classify_headline_direction("Company raises guidance after earnings"), "BULLISH")
        self.assertEqual(classify_headline_direction("Company cuts outlook after weak demand"), "BEARISH")
        self.assertEqual(classify_headline_direction("Nvidia jumps after blockbuster earnings"), "BULLISH")
        self.assertEqual(
            classify_headline_direction(
                "Palantir likely to plunge after its historic run higher",
                "Revenue rose and prior guidance was strong.",
            ),
            "BEARISH",
        )
        self.assertEqual(
            classify_headline_direction("Will Nvidia stock soar after earnings?", "History shows prior gains."),
            "NEUTRAL",
        )
        self.assertEqual(
            classify_headline_direction("Company holds investor day", "The company raises guidance."),
            "NEUTRAL",
        )
        self.assertEqual(classify_headline_direction("Is this stock ready to rally?"), "NEUTRAL")

    def test_rss_is_date_filtered_and_keeps_direct_source_link(self):
        rss = b'''<?xml version="1.0"?><rss xmlns:News="urn:test"><channel>
        <item><title>AAA raises guidance after earnings</title>
        <link>https://www.bing.com/news/apiclick.aspx?url=https%3A%2F%2Fwww.reuters.com%2Faaa</link>
        <description>AAA beat expectations.</description><pubDate>Thu, 27 Aug 2026 12:00:00 GMT</pubDate>
        <News:Source>Reuters on MSN</News:Source></item>
        <item><title>Future item</title><link>https://example.com/future</link>
        <pubDate>Fri, 28 Aug 2026 12:00:00 GMT</pubDate><News:Source>Reuters</News:Source></item>
        </channel></rss>'''
        rows = parse_news_rss(rss, "2026-08-27", 21)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["direction"], "BULLISH")
        self.assertEqual(rows[0]["credibility"], "HIGH")
        self.assertEqual(rows[0]["source_url"], "https://www.reuters.com/aaa")

    def test_auto_context_merges_existing_evidence(self):
        existing_payload = {
            "schema_version":"corat.context.v1",
            "as_of":"2026-08-27",
            "market_events":[],
            "tickers":{"AAA":{"catalysts":[{"classification":"FACT","credibility":"PRIMARY","source":"AAA IR","source_url":"https://example.com/ir","published_at":"2026-08-27","title":"Existing fact","direction":"BULLISH"}],"x_intelligence":[],"events":[],"options_flow":[]}},
        }
        researched = [{"classification":"REPORTED INFORMATION","credibility":"HIGH","source":"Reuters","source_url":"https://reuters.com/aaa","published_at":"2026-08-27","title":"AAA beats estimates","direction":"BULLISH"}]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            existing = root / "existing.json"
            output = root / "auto.json"
            existing.write_text(json.dumps(existing_payload), encoding="utf-8")
            with mock.patch("corat.research._ticker_news", return_value=(researched, [])):
                payload = build_auto_context(
                    "2026-08-27",
                    [{"ticker":"AAA","name":"AAA Corp"}],
                    output,
                    existing_path=existing,
                    maximum_tickers=1,
                )
            self.assertEqual(len(payload["tickers"]["AAA"]["catalysts"]), 2)
            self.assertEqual(payload["research_metadata"]["researched_tickers"], ["AAA"])
            self.assertTrue(output.is_file())

    def test_live_query_shape_does_not_use_empty_feed_parenthesized_or_syntax(self):
        empty_rss = b'<?xml version="1.0"?><rss><channel></channel></rss>'
        with mock.patch("corat.research._fetch_query", return_value=empty_rss) as fetch:
            rows, errors = _ticker_news("AAA", "AAA Corp", "2026-08-27", 21, 1)
        queries = [call.args[0] for call in fetch.call_args_list]
        self.assertEqual(rows, [])
        self.assertEqual(errors, [])
        self.assertEqual(len(queries), 2)
        self.assertTrue(all("(" not in query and " OR " not in query for query in queries))
        self.assertTrue(any("latest stock news" in query for query in queries))


if __name__ == "__main__":
    unittest.main()
