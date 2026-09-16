import unittest
from pathlib import Path

from xhigh.config import TRADEDESK_ROOT
from xhigh.envload import schwab_credentials
from xhigh.schwab import _mover_index


class TestSchwabToken(unittest.TestCase):
    def test_uses_tradedesk_shared_token(self):
        creds = schwab_credentials()
        self.assertIsNotNone(creds)
        got = Path(creds["token_path"]).resolve()
        want = (TRADEDESK_ROOT / "tokens" / "schwab_token.json").resolve()
        self.assertEqual(got, want)
        self.assertTrue(got.is_file())
        self.assertNotEqual(got, (TRADEDESK_ROOT / "xhigh" / "tokens" / "schwab_token.json").resolve())


class TestMoverIndex(unittest.TestCase):
    def test_spx_alias(self):
        self.assertEqual(_mover_index("$SPX.X"), "$SPX")
        self.assertEqual(_mover_index("$SPX"), "$SPX")
        self.assertEqual(_mover_index("$DJI"), "$DJI")


if __name__ == "__main__":
    unittest.main()
