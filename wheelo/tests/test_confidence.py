import unittest

from wheelo.confidence import ticket_confidence


def _core(**over):
    row = {
        "px": 52.0,
        "iv30": 40.0,
        "iv_pctile_1y": 65.0,
        "iv_hv": 1.20,
        "next_ern": "2026-11-15",
        "wks_next_ern": 11,
        "days_to_ern": 80,
    }
    row.update(over)
    return row


def _prem(**over):
    row = {
        "csp_strike": 48.0,
        "csp_bid": 1.40,
        "csp_premium": 1.40,
        "dte": 28,
        "spread_pct": 0.06,
        "iv_rank": 65.0,
    }
    row.update(over)
    return row


class TestTicketConfidence(unittest.TestCase):
    def test_not_a_win_probability(self):
        pack = ticket_confidence(_core(), _prem(), "2026-08-28", x_status="Informed")
        self.assertIn("not P(win)", pack["note"])
        self.assertLessEqual(pack["conf"], 85)

    def test_missing_earnings_cannot_be_trade(self):
        pack = ticket_confidence(_core(next_ern="0000-00-00", days_to_ern=0, wks_next_ern=0), _prem(), "2026-08-28")
        self.assertEqual(pack["earn_status"], "unknown")
        self.assertIn("earnings_unknown", pack["hard"])
        self.assertEqual(pack["label"], "NO_TRADE")
        self.assertLessEqual(pack["conf"], 45)

    def test_penny_credit_is_no_trade(self):
        pack = ticket_confidence(_core(px=81.6), _prem(csp_strike=75.0, csp_bid=0.47, csp_premium=0.47), "2026-08-28")
        self.assertIn("credit_too_small", pack["hard"])
        self.assertEqual(pack["label"], "NO_TRADE")

    def test_earnings_this_week_is_no_trade(self):
        pack = ticket_confidence(_core(next_ern="2026-09-02", days_to_ern=5, wks_next_ern=1), _prem(), "2026-08-28")
        self.assertEqual(pack["earn_status"], "soon")
        self.assertIn("earnings", pack["hard"])
        self.assertEqual(pack["label"], "NO_TRADE")

    def test_wks_next_ern_is_usable_when_date_is_placeholder(self):
        pack = ticket_confidence(
            _core(next_ern="0000-00-00", days_to_ern=0, wks_next_ern=8),
            _prem(),
            "2026-08-28",
            x_status="Informed",
            live_validated=True,
        )
        self.assertEqual(pack["earn_status"], "known")
        self.assertNotIn("earnings_unknown", pack["hard"])
        self.assertEqual(pack["label"], "TRADE")
        self.assertGreaterEqual(pack["conf"], 63)

    def test_wks_one_is_earnings_week(self):
        pack = ticket_confidence(_core(next_ern="0000-00-00", days_to_ern=0, wks_next_ern=1), _prem(), "2026-08-28")
        self.assertEqual(pack["earn_status"], "soon")
        self.assertEqual(pack["label"], "NO_TRADE")

    def test_earnings_inside_dte_is_no_trade(self):
        pack = ticket_confidence(
            _core(next_ern="2026-09-18", days_to_ern=21, wks_next_ern=3),
            _prem(dte=28),
            "2026-08-28",
        )
        self.assertEqual(pack["earn_status"], "known")
        self.assertIn("earnings_in_dte", pack["hard"])
        self.assertEqual(pack["label"], "NO_TRADE")

    def test_earnings_after_expiry_can_be_trade(self):
        pack = ticket_confidence(
            _core(next_ern="0000-00-00", days_to_ern=0, wks_next_ern=7),
            _prem(dte=28),
            "2026-08-28",
            x_status="Informed",
            live_validated=True,
        )
        self.assertEqual(pack["earn_status"], "known")
        self.assertGreaterEqual(pack["earn_days"], 32)
        self.assertNotIn("earnings_in_dte", pack["hard"])
        self.assertEqual(pack["label"], "TRADE")
