import tempfile
import unittest
from pathlib import Path
from unittest import mock

from wheelo import xhot as xhot_mod
from wheelo.xhot import clear_hot, hot_path, load_hot, write_hot


class TestXhotFreshness(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.p = mock.patch.object(xhot_mod, "CODE_DIR", self.root)
        self.p.start()

    def tearDown(self):
        self.p.stop()
        self.tmp.cleanup()

    def test_clear_hot_removes_file(self):
        write_hot("2026-08-31", [{"ticker": "AMAT", "tag": "Informed"}])
        self.assertTrue(hot_path("2026-08-31").is_file())
        self.assertTrue(clear_hot("2026-08-31"))
        self.assertFalse(hot_path("2026-08-31").is_file())
        self.assertEqual(load_hot("2026-08-31"), {})

    def test_wrong_asof_payload_is_ignored(self):
        path = hot_path("2026-08-31")
        path.parent.mkdir(parents=True)
        path.write_text('{"asof":"2026-08-01","names":[{"ticker":"AMAT","tag":"Crowded"}]}', encoding="utf-8")
        self.assertEqual(load_hot("2026-08-31"), {})
