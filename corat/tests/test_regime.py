import unittest
from dataclasses import replace

from corat.regime import classify_market
from tests.helpers import snapshot


class RegimeTest(unittest.TestCase):
    def test_narrow_scan_does_not_masquerade_as_market_breadth(self):
        spy = replace(snapshot(), ticker="SPY")
        qqq = replace(snapshot(), ticker="QQQ")
        iwm = replace(snapshot(), ticker="IWM")
        candidates = [replace(snapshot(), ticker="AAA"), replace(snapshot(), ticker="BBB")]
        result = classify_market({"SPY": spy, "QQQ": qqq, "IWM": iwm}, candidates)
        self.assertIsNone(result["breadth_above_50"])
        self.assertFalse(result["breadth_reliable"])
        self.assertEqual(result["label"], "WEAK RISK-ON")


if __name__ == "__main__":
    unittest.main()
