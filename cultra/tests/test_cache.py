import hashlib
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from cultra.cache import (
    CacheError,
    ContentAddressedCache,
    EntityClaimStore,
    SingleFlight,
    VintageExpectation,
    cache_key_for,
)
from cultra.requesting import Endpoint, RunType, make_planned_request


def request():
    return make_planned_request(
        logical_request_id="cache-request",
        endpoint=Endpoint.CORES,
        run_type=RunType.EOD,
        entities=["AAPL", "MSFT"],
        fields=["ticker", "tradeDate", "updatedAt"],
        field_profile="CORE_V1",
        purpose="offline cache test",
        expected_vintage="2026-08-29",
        expected_rows=2,
        expected_bytes=1000,
        retry_limit=0,
    )


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.allowed_root = Path(self.temp.name) / "cultra-cache-root"
        self.root_patch = mock.patch(
            "cultra.cache.CULTRA_CACHE_ROOT", self.allowed_root
        )
        self.root_patch.start()
        self.root = self.allowed_root / "cache"

    def tearDown(self):
        self.root_patch.stop()
        self.temp.cleanup()

    def test_paths_outside_cultra_cache_root_fail_closed(self):
        outside = Path(self.temp.name) / "outside"
        with self.assertRaises(CacheError):
            ContentAddressedCache(outside)
        with self.assertRaises(CacheError):
            EntityClaimStore(outside / "claims.sqlite3")

    def test_content_addressed_publish_lookup_and_permissions(self):
        item = request()
        expectation = VintageExpectation.from_request(item)
        raw = (
            b'{"data":[{"ticker":"AAPL","tradeDate":"2026-08-29"},'
            b'{"ticker":"MSFT","tradeDate":"2026-08-29"}]}'
        )
        cache = ContentAddressedCache(self.root)
        manifest = cache.publish(
            request=item,
            expectation=expectation,
            raw=raw,
            provider_trade_dates=["2026-08-29"],
            returned_entities=["AAPL", "MSFT"],
            row_count=2,
        )
        loaded_manifest, loaded_raw = cache.lookup(item, expectation)
        self.assertEqual(manifest.snapshot_id, loaded_manifest.snapshot_id)
        self.assertEqual(raw, loaded_raw)
        direct_manifest, direct_raw = cache.load_snapshot(manifest.snapshot_id)
        self.assertEqual(manifest.snapshot_id, direct_manifest.snapshot_id)
        self.assertEqual(raw, direct_raw)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), manifest.raw_sha256)
        raw_path = cache.raw_root / manifest.raw_sha256[:2] / (manifest.raw_sha256 + ".bin")
        self.assertEqual(0o600, os.stat(raw_path).st_mode & 0o777)
        self.assertEqual({"indexes": 1, "manifests": 1, "raw_blobs": 1}, cache.verify())

    def test_vintage_mismatch_is_rejected_before_publish(self):
        item = request()
        expectation = VintageExpectation.from_request(item)
        cache = ContentAddressedCache(self.root)
        with self.assertRaises(CacheError):
            cache.publish(
                request=item,
                expectation=expectation,
                raw=b'{"data":[]}',
                provider_trade_dates=["2026-08-28"],
                returned_entities=[],
                row_count=0,
            )

    def test_full_history_accepts_multiple_past_dates_but_rejects_future_rows(self):
        item = make_planned_request(
            logical_request_id="historical-series",
            endpoint=Endpoint.HIST_CORES,
            run_type=RunType.HISTORICAL_BACKFILL,
            entities=["AAPL"],
            fields=["ticker", "tradeDate"],
            field_profile="HIST_CORE_V2",
            purpose="full-history cache contract",
            expected_vintage="2026-08-28",
            expected_rows=100,
            expected_bytes=1000,
            retry_limit=0,
        )
        expectation = VintageExpectation.from_request(item)
        cache = ContentAddressedCache(self.root)
        with self.assertRaises(CacheError):
            cache.publish(
                request=item,
                expectation=expectation,
                raw=b'{"data":[]}',
                provider_trade_dates=["2026-08-29"],
                returned_entities=[],
                row_count=0,
            )
        manifest = cache.publish(
            request=item,
            expectation=expectation,
            raw=b'{"data":[{"ticker":"AAPL"}]}',
            provider_trade_dates=["2026-01-02", "2026-08-28"],
            returned_entities=["AAPL"],
            row_count=1,
        )
        self.assertEqual(
            ("2026-01-02", "2026-08-28"), manifest.provider_trade_dates
        )

    def test_split_history_absent_rows_cover_the_entire_requested_batch(self):
        item = make_planned_request(
            logical_request_id="historical-splits",
            endpoint=Endpoint.HIST_SPLITS,
            run_type=RunType.HISTORICAL_BACKFILL,
            entities=["AAPL", "MSFT"],
            fields=["divisor", "splitDate", "ticker"],
            field_profile="HIST_SPLITS_V2",
            purpose="event-history cache contract",
            expected_vintage="2026-08-28",
            expected_rows=100,
            expected_bytes=1000,
            retry_limit=0,
        )
        expectation = VintageExpectation.from_request(item)
        cache = ContentAddressedCache(self.root)
        manifest = cache.publish(
            request=item,
            expectation=expectation,
            raw=b'{"data":[{"ticker":"AAPL","splitDate":"2026-01-02","divisor":2}]}',
            provider_trade_dates=[],
            returned_entities=["AAPL"],
            row_count=1,
        )
        self.assertEqual(("AAPL", "MSFT"), manifest.returned_entities)
        self.assertEqual((), manifest.missing_entities)

    def test_cache_key_is_stable_and_contains_no_raw_identifiers(self):
        item = request()
        expectation = VintageExpectation.from_request(item)
        key = cache_key_for(item, expectation)
        self.assertEqual(key, cache_key_for(item, expectation))
        self.assertEqual(64, len(key))
        self.assertNotIn("AAPL", key)

    def test_corrupt_raw_blob_fails_closed(self):
        item = request()
        expectation = VintageExpectation.from_request(item)
        cache = ContentAddressedCache(self.root)
        manifest = cache.publish(
            request=item,
            expectation=expectation,
            raw=b'{"data":[{"ticker":"AAPL"},{"ticker":"MSFT"}]}',
            provider_trade_dates=["2026-08-29"],
            returned_entities=["AAPL", "MSFT"],
            row_count=2,
        )
        raw_path = cache.raw_root / manifest.raw_sha256[:2] / (manifest.raw_sha256 + ".bin")
        raw_path.write_bytes(b"corrupt")
        with self.assertRaises(CacheError):
            cache.lookup(item, expectation)

    def test_manifest_tampering_and_missing_provider_date_fail_closed(self):
        item = request()
        expectation = VintageExpectation.from_request(item)
        cache = ContentAddressedCache(self.root)
        with self.assertRaises(CacheError):
            cache.publish(
                request=item,
                expectation=expectation,
                raw=b'{"data":[]}',
                provider_trade_dates=[],
                returned_entities=[],
                row_count=0,
            )

        manifest = cache.publish(
            request=item,
            expectation=expectation,
            raw=b'{"data":[{"ticker":"AAPL"},{"ticker":"MSFT"}]}',
            provider_trade_dates=["2026-08-29"],
            returned_entities=["AAPL", "MSFT"],
            row_count=2,
        )
        manifest_path = (
            cache.manifest_root
            / manifest.snapshot_id[:2]
            / (manifest.snapshot_id + ".json")
        )
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["field_profile"] = "TAMPERED"
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        manifest_path.chmod(0o600)
        with self.assertRaises(CacheError):
            cache.lookup(item, expectation)

    def test_single_flight_runs_one_leader(self):
        flight = SingleFlight()
        barrier = threading.Barrier(17)
        calls = []
        values = []

        def work():
            calls.append(1)
            time.sleep(0.05)
            return "shared"

        def run():
            barrier.wait()
            values.append(flight.run("fingerprint", work))

        threads = [threading.Thread(target=run) for _ in range(16)]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join()
        self.assertEqual(1, len(calls))
        self.assertEqual(["shared"] * 16, sorted(values))

    def test_entity_claims_deduplicate_overlapping_batches(self):
        store = EntityClaimStore(self.allowed_root / "claims.sqlite3")
        first = store.claim("vintage-group", ["AAPL", "MSFT"], owner_id="first")
        second = store.claim("vintage-group", ["MSFT", "NVDA"], owner_id="second")
        self.assertEqual(("AAPL", "MSFT"), first.claimed)
        self.assertEqual(("NVDA",), second.claimed)
        self.assertEqual(("MSFT",), second.pending)
        store.complete(first, snapshot_id="snapshot-one")
        third = store.claim("vintage-group", ["AAPL", "MSFT"], owner_id="third")
        self.assertEqual(("AAPL", "MSFT"), third.ready)
        self.assertFalse(third.claimed)


if __name__ == "__main__":
    unittest.main()
