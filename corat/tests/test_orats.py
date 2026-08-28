import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

from corat.orats import OratsClient


class FakeResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def getcode(self):
        return 200
    def read(self):
        return self.payload


class OratsTest(unittest.TestCase):
    def client(self, tmp, offline=False):
        return OratsClient("SECRET_TOKEN_XYZ", "https://api.orats.io/datav2", Path(tmp) / "cache", Path(tmp) / "state", max_requests=5, requests_per_minute=100000, offline=offline)

    def test_request_caches_without_token(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = {"data": [{"ticker": "SPY", "tradeDate": "2026-08-27", "pxCls": 700.0}]}
            with mock.patch("corat.orats.urllib.request.urlopen", return_value=FakeResponse(payload)):
                result = self.client(tmp).request_rows("/cores", {"ticker": "SPY"})
            self.assertEqual(len(result.rows), 1)
            cache = Path(result.traces[0].cache_path)
            text = cache.read_text(encoding="utf-8")
            self.assertNotIn("SECRET_TOKEN_XYZ", text)
            self.assertNotIn("token", result.traces[0].params)

    def test_offline_cache_hit(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = {"data": [{"ticker": "SPY", "tradeDate": "2026-08-27"}]}
            with mock.patch("corat.orats.urllib.request.urlopen", return_value=FakeResponse(payload)):
                self.client(tmp).request_rows("/cores", {"ticker": "SPY"})
            result = self.client(tmp, offline=True).request_rows("/cores", {"ticker": "SPY"})
            self.assertEqual(result.traces[0].status, "CACHED")
            self.assertEqual(len(result.rows), 1)

    def test_offline_cache_miss_is_data_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.client(tmp, offline=True).request_rows("/cores", {"ticker": "SPY"})
            self.assertFalse(result.rows)
            self.assertEqual(result.traces[0].status, "DATA UNAVAILABLE")

    def test_rejects_invalid_ticker(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                self.client(tmp).fetch_chain("SPY&token=bad", "2026-08-27", 21, 75)

    def test_monthly_reserve_is_enforced_before_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "state"
            state.mkdir(parents=True)
            (state / "orats_usage.json").write_text(
                json.dumps({"month": datetime.now().strftime("%Y-%m"), "used": 95, "cap": 100}),
                encoding="utf-8",
            )
            client = OratsClient(
                "SECRET_TOKEN_XYZ",
                "https://api.orats.io/datav2",
                Path(tmp) / "cache",
                state,
                max_requests=5,
                monthly_cap=100,
                requests_per_minute=100000,
                monthly_reserve=5,
            )
            with mock.patch("corat.orats.urllib.request.urlopen", side_effect=AssertionError("network called")):
                result = client.request_rows("/cores", {"ticker": "SPY"})
            self.assertFalse(result.rows)
            self.assertIn("reserve", result.errors[0])
