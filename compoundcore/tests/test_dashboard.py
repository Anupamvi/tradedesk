import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from compoundcore.dashboard import make_server


class TestDashboardHttp(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.state = Path(self.tmp.name) / "dashboard.json"
        self.httpd = make_server("127.0.0.1", 0, self.state)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.base = "http://127.0.0.1:%d" % self.port

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.tmp.cleanup()

    def _json(self, path, payload=None, method=None):
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(self.base + path, data=data, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = resp.read().decode("utf-8")
            ctype = resp.headers.get("Content-Type", "")
            if "json" in ctype:
                return resp.status, json.loads(body)
            return resp.status, body

    def test_get_dashboard_and_raw_calculator(self):
        status, html = self._json("/")
        self.assertEqual(status, 200)
        self.assertIn("Compound Core", html)
        self.assertIn("Save my book", html)
        self.assertIn("Cost $", html)
        self.assertIn("Now $", html)
        self.assertIn("e.g. 100000", html)
        self.assertIn("My book", html)
        status, calc = self._json("/calculator.html")
        self.assertEqual(status, 200)
        self.assertIn("Compound Core calculator", calc)

    def test_empty_book_until_submit(self):
        status, data = self._json("/api/state")
        self.assertEqual(status, 200)
        self.assertFalse(data["book"]["present"])
        self.assertIsNone(data["book"]["projections"])

    def test_planner_persists_and_splits_both_sleeves(self):
        status, data = self._json("/api/planner", {"amount": 100000, "weekly": 250, "monthly": 1000}, method="POST")
        self.assertEqual(status, 200)
        voo = data["planner"]["sleeves"]["default"]["allocation"]["rows"][0]
        self.assertEqual(voo["ticker"], "VOO")
        self.assertEqual(voo["dollars"], 48000.0)
        smh_agg = [r for r in data["planner"]["sleeves"]["aggressive"]["allocation"]["rows"] if r["ticker"] == "SMH"][0]
        self.assertEqual(smh_agg["dollars"], 10000.0)
        self.assertEqual(
            int(round(data["planner"]["sleeves"]["default"]["projections"]["10y"]["base"]["nominal"] / 1000.0)) * 1000,
            337000,
        )
        status, again = self._json("/api/state")
        self.assertEqual(again["saved"]["planner"]["amount"], 100000.0)
        self.assertEqual(
            again["planner"]["sleeves"]["default"]["projections"]["10y"]["base"]["nominal"],
            data["planner"]["sleeves"]["default"]["projections"]["10y"]["base"]["nominal"],
        )

    def test_submit_book_then_refresh_shows_growth(self):
        holdings = {
            "VOO": 48000,
            "VGT": 10000,
            "SMH": 7000,
            "VB": 5000,
            "VXUS": 20000,
            "GLDM": 5000,
            "VGSH": 5000,
        }
        status, data = self._json(
            "/api/book",
            {"holdings": holdings, "monthly_add": 1000, "compare_to": "default"},
            method="POST",
        )
        self.assertEqual(status, 200)
        self.assertTrue(data["book"]["present"])
        self.assertFalse(data["book"]["pnl_ready"])
        self.assertEqual(
            int(round(data["book"]["projections"]["10y"]["base"]["nominal"] / 1000.0)) * 1000,
            337000,
        )
        status, again = self._json("/api/state")
        self.assertTrue(again["book"]["present"])
        self.assertEqual(again["book"]["invested"], 100000.0)

    def test_saved_book_shows_real_gain_and_loss(self):
        positions = {
            "VOO": {"cost": 48000, "current": 52800, "shares": 80},
            "VGT": {"cost": 10000, "current": 9000, "shares": 20},
            "SMH": {"cost": 7000, "current": 7000, "shares": 12},
            "VB": {"cost": 5000, "current": 5000, "shares": 0},
            "VXUS": {"cost": 20000, "current": 20000, "shares": 0},
            "GLDM": {"cost": 5000, "current": 5000, "shares": 0},
            "VGSH": {"cost": 5000, "current": 5000, "shares": 0},
        }
        status, data = self._json(
            "/api/book",
            {"positions": positions, "monthly_add": 0, "compare_to": "default"},
            method="POST",
        )
        self.assertEqual(status, 200)
        self.assertTrue(data["book"]["pnl_ready"])
        self.assertEqual(data["book"]["invested"], 100000.0)
        self.assertEqual(data["book"]["market"], 103800.0)
        self.assertEqual(data["book"]["pnl"], 3800.0)
        status, again = self._json("/api/state")
        self.assertEqual(again["book"]["pnl"], 3800.0)
        self.assertEqual(again["saved"]["book"]["positions"]["VOO"]["cost"], 48000.0)
        self.assertEqual(again["saved"]["book"]["positions"]["VGT"]["current"], 9000.0)

    def test_refresh_marks_from_quotes_keeps_cost(self):
        from unittest.mock import patch

        positions = {
            "VOO": {"cost": 48000, "current": 48000, "shares": 80},
            "VGT": {"cost": 0, "current": 0, "shares": 0},
            "SMH": {"cost": 0, "current": 0, "shares": 0},
            "VB": {"cost": 0, "current": 0, "shares": 0},
            "VXUS": {"cost": 0, "current": 0, "shares": 0},
            "GLDM": {"cost": 0, "current": 0, "shares": 0},
            "VGSH": {"cost": 0, "current": 0, "shares": 0},
        }
        self._json("/api/book", {"positions": positions}, method="POST")
        with patch("compoundcore.quotes.last_prices", return_value={"VOO": 650.0}):
            status, data = self._json("/api/book/refresh", {}, method="POST")
        self.assertEqual(status, 200)
        self.assertEqual(data["book"]["positions"]["VOO"]["cost"], 48000.0)
        self.assertEqual(data["book"]["positions"]["VOO"]["current"], 52000.0)
        self.assertEqual(data["book"]["pnl"], 4000.0)

    def test_rejects_negative_holdings(self):
        req = urllib.request.Request(
            self.base + "/api/book",
            data=json.dumps({"VOO": -1}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req, timeout=5)
        self.assertEqual(ctx.exception.code, 400)
