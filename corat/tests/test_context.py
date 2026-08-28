import json
import tempfile
import unittest
from pathlib import Path

from corat.context import event_risks, load_context, ticker_context


def evidence(**overrides):
    row = {
        "classification": "FACT",
        "credibility": "PRIMARY",
        "source": "Issuer IR",
        "source_url": "https://example.com/release",
        "published_at": "2026-08-27",
        "title": "Sourced development",
        "direction": "BULLISH",
    }
    row.update(overrides)
    return row


class ContextTest(unittest.TestCase):
    def _load(self, payload):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "context.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            return load_context(path, "2026-08-27")

    def test_rumor_does_not_become_actionable_catalyst(self):
        context = self._load({
            "schema_version": "corat.context.v1",
            "as_of": "2026-08-27",
            "market_events": [],
            "tickers": {"AAA": {
                "catalysts": [evidence(classification="RUMOR / X SPECULATION")],
                "x_intelligence": [], "events": [], "options_flow": [],
            }},
        })
        value = ticker_context(context, "AAA", "2026-08-27")
        self.assertEqual(value["actionable_catalysts"], [])

    def test_directional_strength_is_kept_separate(self):
        context = self._load({
            "schema_version": "corat.context.v1",
            "as_of": "2026-08-27",
            "market_events": [],
            "tickers": {"AAA": {
                "catalysts": [evidence(direction="BEARISH")],
                "x_intelligence": [], "events": [], "options_flow": [],
            }},
        })
        value = ticker_context(context, "AAA", "2026-08-27")
        self.assertEqual(value["catalyst_strength_by_direction"]["BULLISH"], 0.0)
        self.assertGreater(value["catalyst_strength_by_direction"]["BEARISH"], 0.0)

    def test_sourced_event_is_flagged_during_hold(self):
        market = evidence(event_date="2026-09-02", title="Macro release")
        context = self._load({
            "schema_version": "corat.context.v1",
            "as_of": "2026-08-27",
            "market_events": [market],
            "tickers": {},
        })
        self.assertEqual(len(event_risks(context, {}, "2026-08-27", 10)), 1)

    def test_missing_classification_is_rejected(self):
        row = evidence()
        del row["classification"]
        with self.assertRaisesRegex(ValueError, "classification"):
            self._load({
                "schema_version": "corat.context.v1", "as_of": "2026-08-27", "market_events": [],
                "tickers": {"AAA": {"catalysts": [row], "x_intelligence": [], "events": [], "options_flow": []}},
            })

    def test_market_event_requires_event_date(self):
        with self.assertRaisesRegex(ValueError, "event_date"):
            self._load({
                "schema_version": "corat.context.v1",
                "as_of": "2026-08-27",
                "market_events": [evidence(title="Macro event")],
                "tickers": {},
            })


if __name__ == "__main__":
    unittest.main()
