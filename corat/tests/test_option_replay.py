import unittest

from corat.option_replay import conservative_exit_credit


class OptionReplayTest(unittest.TestCase):
    def test_exact_exit_uses_bid_for_long_and_ask_for_short_then_partial_mid_improvement(self):
        rows=[
            {"expirDate":"2026-10-02","strike":120,"callBidPrice":8.0,"callAskPrice":8.4,"putBidPrice":1,"putAskPrice":1.2},
            {"expirDate":"2026-10-02","strike":130,"callBidPrice":3.0,"callAskPrice":3.4,"putBidPrice":4,"putAskPrice":4.2},
        ]
        credit=conservative_exit_credit(rows,"BULLISH","2026-10-02",120,130)
        natural=8.0-3.4
        midpoint=(8.0+8.4)/2-(3.0+3.4)/2
        self.assertAlmostEqual(credit,natural+0.25*(midpoint-natural))

    def test_missing_exact_leg_is_not_reconstructed(self):
        rows=[{"expirDate":"2026-10-02","strike":120,"callBidPrice":8,"callAskPrice":8.4}]
        self.assertIsNone(conservative_exit_credit(rows,"BULLISH","2026-10-02",120,130))

