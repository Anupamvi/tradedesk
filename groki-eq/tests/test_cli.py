import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from groki_eq.cli import run
from tests.test_breakout import _bars


class TestBreakoutRun(unittest.TestCase):
    def test_selector_and_no_token_in_manifest(self):
        spy = _bars(40, breakout=True)
        qqq = _bars(40, breakout=False)
        iwm = _bars(40, breakout=False)
        asof = spy[-1]["date"]
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "archive"
            out = Path(tmp) / "out"
            with mock.patch("groki_eq.orats.archive_dir", return_value=archive):
                result = run(
                    asof,
                    out,
                    token="secret-token",
                    no_schwab=True,
                    today="2099-01-01",
                    liquid=["SPY", "QQQ", "IWM"],
                    schwab_bars={"SPY": spy, "QQQ": qqq, "IWM": iwm},
                )
            self.assertEqual(result["selector"], "breakout_eq")
            self.assertLessEqual(result["execute_count"], 1)
            manifest = json.loads((out / asof / "manifest.json").read_text(encoding="utf-8"))
            self.assertNotIn("secret-token", json.dumps(manifest))
            self.assertEqual(manifest["selector"], "breakout_eq")


if __name__ == "__main__":
    unittest.main()
