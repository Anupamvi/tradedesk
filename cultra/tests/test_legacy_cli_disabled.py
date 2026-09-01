import argparse
import unittest

from cultra.cli import (
    _build_opportunities,
    _rebuild_patterns,
    _research_orders,
    _validate_history,
)


class LegacyCliDisabledTests(unittest.TestCase):
    def test_invalid_v1_research_surfaces_cannot_execute(self):
        for handler in (
            _validate_history,
            _research_orders,
            _build_opportunities,
            _rebuild_patterns,
        ):
            with self.subTest(handler=handler.__name__):
                with self.assertRaisesRegex(ValueError, "disabled"):
                    handler(argparse.Namespace())


if __name__ == "__main__":
    unittest.main()
