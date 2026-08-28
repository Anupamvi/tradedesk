import unittest

from groat.dates import parse_any_date
from groat.earnings import apply_web, from_cores


class TestDateSchema(unittest.TestCase):
    def test_iso_and_us_slash(self):
        self.assertEqual(parse_any_date("2026-08-03"), "2026-08-03")
        self.assertEqual(parse_any_date("8/3/2026"), "2026-08-03")
        self.assertEqual(parse_any_date("08/03/2026"), "2026-08-03")
        self.assertIsNone(parse_any_date("0000-00-00"))
        self.assertIsNone(parse_any_date(""))


class TestCoresEarnings(unittest.TestCase):
    def test_next_ern_placeholder_is_not_a_date(self):
        core = {
            "nextErn": "0000-00-00",
            "daysToNextErn": 0,
            "wksNextErn": 9,
            "lastErn": "2026-08-03",
            "ernDate1": "8/3/2026",
            "ernDate2": "5/4/2026",
        }
        info = from_cores("PLTR", core, "2026-08-27")
        self.assertTrue(info["usable"])
        self.assertEqual(info["last"], "2026-08-03")
        self.assertEqual(info["source"], "orats.wksNextErn")
        self.assertGreaterEqual(info["days"], 50)
        self.assertFalse(info["overlaps_hold"])

    def test_days_to_next_zero_is_ignored(self):
        core = {
            "nextErn": "0000-00-00",
            "daysToNextErn": 0,
            "wksNextErn": 0,
            "lastErn": "2026-08-03",
            "ernDate1": "8/3/2026",
            "ernDate2": "5/4/2026",
            "ernDate3": "2/2/2026",
        }
        info = from_cores("PLTR", core, "2026-08-27")
        self.assertTrue(info["usable"])
        self.assertEqual(info["source"], "orats.ernDate_cadence")
        self.assertGreater(info["days"], 40)

    def test_web_overrides_orats_weeks(self):
        core = {"nextErn": "0000-00-00", "wksNextErn": 9, "lastErn": "2026-08-03"}
        base = from_cores("PLTR", core, "2026-08-27")
        merged = apply_web(
            base,
            {"web_next": "2026-11-02", "web_last": "2026-08-03", "web_source": "web.alphaquery"},
            "2026-08-27",
        )
        self.assertEqual(merged["date"], "2026-11-02")
        self.assertEqual(merged["source"], "web.alphaquery")
        self.assertFalse(merged["overlaps_hold"])

    def test_near_earnings_overlaps_hold(self):
        core = {"nextErn": "0000-00-00", "wksNextErn": 2, "lastErn": "2026-06-11"}
        info = from_cores("ADBE", core, "2026-08-27")
        self.assertTrue(info["usable"])
        self.assertTrue(info["overlaps_hold"])


if __name__ == "__main__":
    unittest.main()
