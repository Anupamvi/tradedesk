import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from cultra.cache import ContentAddressedCache
from cultra.gateway import (
    EnvFileTokenSource,
    GatewayError,
    GatewayRequestError,
    GatewayResult,
    OratsGateway,
    OratsGatewayClient,
    OratsGatewayServer,
    SafeTransportError,
    TokenSourceError,
    TransportResponse,
    UrllibTransport,
    execute_plan_via_local_daemon,
    redact_text,
)
from cultra.ledger import (
    ORATS_ACCOUNT_SCOPE,
    ActiveRunError,
    CircuitOpen,
    RequestLedger,
)
from cultra.requesting import Endpoint, RequestPlan, RunType, make_planned_request


SECRET = "test-secret-should-never-escape"


class FakeTokenSource:
    def __init__(self):
        self.loads = []

    def load(self, *, force_reload=False):
        self.loads.append(force_reload)
        return SECRET


class FakeTransport:
    def __init__(self, responses, delay=0.0):
        self.responses = list(responses)
        self.delay = delay
        self.calls = 0
        self.tokens_seen = []
        self.lock = threading.Lock()

    def send(self, request, token):
        with self.lock:
            self.calls += 1
            self.tokens_seen.append(token)
            response = self.responses.pop(0)
        if self.delay:
            time.sleep(self.delay)
        if isinstance(response, BaseException):
            raise response
        return response


def response(status=200, body=None, headers=()):
    if body is None:
        body = json.dumps(
            {
                "data": [
                    {
                        "ticker": "AAPL",
                        "tradeDate": "2026-08-29",
                        "updatedAt": "2026-08-29T22:00:00Z",
                    }
                ]
            }
        ).encode()
    return TransportResponse(status, body, tuple(headers), 1.0)


def request_plan(run_id="gateway-run"):
    item = make_planned_request(
        logical_request_id="core-one",
        endpoint=Endpoint.CORES,
        run_type=RunType.EOD,
        entities=["AAPL"],
        fields=["ticker", "tradeDate", "updatedAt"],
        field_profile="CORE_V1",
        purpose="offline gateway test",
        expected_vintage="2026-08-29",
        expected_rows=1,
        expected_bytes=1000,
        retry_limit=0,
    )
    return RequestPlan(
        run_id=run_id,
        run_type=RunType.EOD,
        requests=(item,),
        target=1,
        hard_cap=1,
        retry_reserve=0,
    )


class GatewayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.ledger_root = self.root / "cultra-ledger-root"
        self.cache_root = self.root / "cultra-cache-root"
        self.env_path = self.root / ".env"
        self.patches = (
            mock.patch("cultra.ledger.CULTRA_LEDGER_ROOT", self.ledger_root),
            mock.patch("cultra.cache.CULTRA_CACHE_ROOT", self.cache_root),
            mock.patch("cultra.gateway.CULTRA_ENV_PATH", self.env_path),
        )
        for patcher in self.patches:
            patcher.start()

    def tearDown(self):
        for patcher in reversed(self.patches):
            patcher.stop()
        self.temp.cleanup()

    def gateway(self, transport, run_id="gateway-run"):
        plan = request_plan(run_id)
        ledger = RequestLedger(self.ledger_root / "account.sqlite3")
        cache = ContentAddressedCache(self.cache_root / (run_id + "-cache"))
        gateway = OratsGateway(
            plan=plan,
            ledger=ledger,
            cache=cache,
            token_source=FakeTokenSource(),
            transport=transport,
            sleeper=lambda unused: None,
            jitter=lambda: 0.0,
        )
        return gateway, ledger, plan

    def custom_gateway(self, plan, transport, *, ledger_name="custom.sqlite3"):
        del ledger_name
        ledger = RequestLedger(self.ledger_root / "account.sqlite3")
        cache = ContentAddressedCache(self.cache_root / (plan.run_id + "-cache"))
        gateway = OratsGateway(
            plan=plan,
            ledger=ledger,
            cache=cache,
            token_source=FakeTokenSource(),
            transport=transport,
            sleeper=lambda unused: None,
            jitter=lambda: 0.0,
        )
        return gateway, ledger

    def test_success_then_cache_hit_uses_one_attempt(self):
        transport = FakeTransport([response()])
        gateway, ledger, plan = self.gateway(transport)
        first = gateway.execute("core-one")
        second = gateway.execute("core-one")
        self.assertFalse(first.cache_hit)
        self.assertTrue(second.cache_hit)
        self.assertEqual(1, transport.calls)
        self.assertEqual([SECRET], transport.tokens_seen)
        self.assertEqual(1, ledger.summary(plan.run_id)["charged_attempts"])

    def test_validated_warm_cache_hit_does_not_load_token(self):
        transport = FakeTransport([response()])
        gateway, ledger, plan = self.gateway(transport, "warm-tokenless")
        gateway.execute("core-one")
        gateway._token_cache = None
        gateway._token_source.loads.clear()
        result = gateway.execute("core-one")
        self.assertTrue(result.cache_hit)
        self.assertEqual([], gateway._token_source.loads)
        self.assertEqual(1, ledger.summary(plan.run_id)["charged_attempts"])

    def test_malformed_success_row_is_burned_and_never_cached(self):
        malformed = response(
            body=json.dumps(
                {
                    "data": [
                        {
                            "ticker": ["AAPL"],
                            "tradeDate": "2026-08-29",
                            "updatedAt": "2026-08-29T22:00:00Z",
                        }
                    ]
                }
            ).encode()
        )
        transport = FakeTransport([malformed])
        gateway, ledger, plan = self.gateway(transport, "malformed-success")
        with self.assertRaisesRegex(GatewayRequestError, "validation failed"):
            gateway.execute("core-one")
        self.assertEqual(1, transport.calls)
        self.assertEqual(1, ledger.summary(plan.run_id)["charged_attempts"])
        self.assertFalse(any(self.cache_root.rglob("*.bin")))

    def test_non_object_provider_row_is_not_silently_dropped(self):
        transport = FakeTransport(
            [response(body=b'{"data":[{"ticker":"AAPL","tradeDate":"2026-08-29","updatedAt":"2026-08-29T22:00:00Z"},7]}')]
        )
        gateway, ledger, plan = self.gateway(transport, "non-object-row")
        with self.assertRaisesRegex(GatewayRequestError, "validation failed"):
            gateway.execute("core-one")
        self.assertEqual(1, ledger.summary(plan.run_id)["charged_attempts"])

    def test_provider_fields_outside_the_frozen_profile_are_rejected(self):
        transport = FakeTransport(
            [
                response(
                    body=b'{"data":[{"ticker":"AAPL","tradeDate":"2026-08-29","updatedAt":"2026-08-29T22:00:00Z","futureProfit":99}]}'
                )
            ]
        )
        gateway, ledger, plan = self.gateway(transport, "unknown-provider-field")
        with self.assertRaisesRegex(GatewayRequestError, "validation failed"):
            gateway.execute("core-one")
        self.assertEqual(1, ledger.summary(plan.run_id)["charged_attempts"])

    def test_daemon_result_rejects_tampered_duplicate_metadata(self):
        transport = FakeTransport([response()])
        gateway, _ledger, _plan = self.gateway(transport, "daemon-integrity")
        public = gateway.execute("core-one").to_public_dict(include_raw=True)
        public["row_count"] += 1
        with self.assertRaisesRegex(GatewayRequestError, "malformed"):
            GatewayResult.from_public_dict(public)

    def test_daemon_auth_canary_prevents_parallel_fanout(self):
        requests = tuple(
            make_planned_request(
                logical_request_id="auth-%d" % index,
                endpoint=Endpoint.CORES,
                run_type=RunType.EOD,
                entities=[ticker],
                fields=["ticker", "tradeDate", "updatedAt"],
                field_profile="CORE_V1",
                purpose="authorization canary test",
                expected_vintage="2026-08-29",
                expected_rows=1,
                expected_bytes=1000,
                retry_limit=0,
            )
            for index, ticker in enumerate(("AAPL", "MSFT", "NVDA", "TSLA"), 1)
        )
        plan = RequestPlan(
            run_id="auth-canary",
            run_type=RunType.EOD,
            requests=requests,
            target=4,
            hard_cap=4,
            retry_reserve=0,
        )
        transport = FakeTransport(
            [response(401, b'{"error":"unauthorized"}')] + [response()] * 3
        )
        gateway, ledger = self.custom_gateway(plan, transport)
        completed, errors = execute_plan_via_local_daemon(
            gateway,
            [item.logical_request_id for item in requests],
            socket_path=self.root / "auth-canary.sock",
            workers=4,
        )
        self.assertEqual({}, completed)
        self.assertEqual(["auth-1"], list(errors))
        self.assertEqual(1, transport.calls)
        self.assertEqual(1, ledger.summary(plan.run_id)["charged_attempts"])

    def test_official_query_keys_are_endpoint_specific(self):
        core = request_plan("query-core").requests[0]
        core_params = UrllibTransport._request_parameters(core)
        self.assertEqual("AAPL", core_params["ticker"])
        self.assertNotIn("tickers", core_params)
        exact = make_planned_request(
            logical_request_id="exact",
            endpoint=Endpoint.EXACT_OPTIONS,
            run_type=RunType.EOD,
            entities=["AAPL  270101C00100000", "MSFT  270101P00200000"],
            fields=["ticker", "tradeDate"],
            field_profile="EXACT_OPTION_V1",
            purpose="query-key contract",
            expected_vintage="2026-08-29",
            expected_rows=2,
            expected_bytes=1000,
            retry_limit=0,
        )
        exact_params = UrllibTransport._request_parameters(exact)
        self.assertEqual(
            "AAPL  270101C00100000,MSFT  270101P00200000",
            exact_params["tickers"],
        )
        self.assertNotIn("ticker", exact_params)
        self.assertNotIn("optionSymbols", exact_params)

    def test_concurrent_identical_calls_single_flight(self):
        transport = FakeTransport([response()], delay=0.05)
        gateway, ledger, plan = self.gateway(transport, "concurrent")
        barrier = threading.Barrier(11)
        results = []

        def call():
            barrier.wait()
            results.append(gateway.execute("core-one"))

        threads = [threading.Thread(target=call) for _ in range(10)]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join()
        self.assertEqual(10, len(results))
        self.assertEqual(1, transport.calls)
        self.assertEqual(1, ledger.summary(plan.run_id)["charged_attempts"])

    def test_retryable_failure_is_not_automatically_retried(self):
        transport = FakeTransport([response(503, b'{"error":"busy"}'), response()])
        gateway, ledger, plan = self.gateway(transport, "retry")
        with self.assertRaises(GatewayRequestError):
            gateway.execute("core-one")
        self.assertEqual(1, transport.calls)
        summary = ledger.summary(plan.run_id)
        self.assertEqual(1, summary["charged_attempts"])
        self.assertEqual(0, summary["retries"])

    def test_non_idempotent_post_is_never_retried(self):
        item = make_planned_request(
            logical_request_id="backtest-post",
            endpoint=Endpoint.BACKTEST,
            run_type=RunType.BACKTEST_VALIDATION,
            entities=["JOB1"],
            fields=["status"],
            field_profile="BACKTEST_JOB_V1",
            purpose="offline POST retry contract test",
            expected_vintage="2026-08-29",
            expected_rows=1,
            expected_bytes=1000,
            retry_limit=0,
        )
        plan = RequestPlan(
            run_id="post-no-retry",
            run_type=RunType.BACKTEST_VALIDATION,
            requests=(item,),
            target=1,
            hard_cap=1,
            retry_reserve=0,
        )
        transport = FakeTransport(
            [response(503, b'{"error":"busy"}'), response()]
        )
        gateway, ledger = self.custom_gateway(plan, transport)
        with self.assertRaises(GatewayRequestError):
            gateway.execute("backtest-post")
        self.assertEqual(1, transport.calls)
        self.assertEqual(1, ledger.summary(plan.run_id)["charged_attempts"])

    def test_non_idempotent_post_transport_failure_is_never_retried(self):
        item = make_planned_request(
            logical_request_id="scanner-post",
            endpoint=Endpoint.SCANNER,
            run_type=RunType.SCANNER_RESEARCH,
            entities=["JOB1"],
            fields=["status"],
            field_profile="SCANNER_JOB_V1",
            purpose="offline POST transport retry contract test",
            expected_vintage="2026-08-29",
            expected_rows=1,
            expected_bytes=1000,
            retry_limit=0,
        )
        plan = RequestPlan(
            run_id="post-transport-no-retry",
            run_type=RunType.SCANNER_RESEARCH,
            requests=(item,),
            target=1,
            hard_cap=1,
            retry_reserve=0,
        )
        transport = FakeTransport(
            [
                SafeTransportError("network", retryable=True, ambiguous=True),
                response(),
            ]
        )
        gateway, ledger = self.custom_gateway(
            plan, transport, ledger_name="post-transport.sqlite3"
        )
        with self.assertRaises(GatewayRequestError):
            gateway.execute("scanner-post")
        self.assertEqual(1, transport.calls)
        self.assertEqual(1, ledger.summary(plan.run_id)["charged_attempts"])

    def test_circuit_breaker_stops_a_fourth_provider_attempt(self):
        requests = tuple(
            make_planned_request(
                logical_request_id="request-%d" % index,
                endpoint=Endpoint.CORES,
                run_type=RunType.EOD,
                entities=[ticker],
                fields=["ticker", "tradeDate"],
                field_profile="CORE_V1",
                purpose="offline circuit test",
                expected_vintage="2026-08-29",
                expected_rows=1,
                expected_bytes=1000,
                retry_limit=0,
            )
            for index, ticker in enumerate(("AAPL", "MSFT", "NVDA", "TSLA"), 1)
        )
        plan = RequestPlan(
            run_id="gateway-circuit",
            run_type=RunType.EOD,
            requests=requests,
            target=4,
            hard_cap=4,
            retry_reserve=0,
        )
        transport = FakeTransport(
            [response(503, b'{"error":"busy"}') for unused in range(4)]
        )
        gateway, ledger = self.custom_gateway(
            plan, transport, ledger_name="circuit.sqlite3"
        )
        for item in requests[:3]:
            with self.assertRaises(GatewayRequestError):
                gateway.execute(item.logical_request_id)
        with self.assertRaises(CircuitOpen):
            gateway.execute(requests[3].logical_request_id)
        self.assertEqual(3, transport.calls)
        summary = ledger.summary(plan.run_id)
        self.assertEqual("open", summary["provider_circuit_state"])
        self.assertEqual(3, summary["charged_attempts"])

    def test_redirect_is_not_followed_or_retried(self):
        transport = FakeTransport([response(302, b'{"redirect":true}')])
        gateway, ledger, plan = self.gateway(transport, "redirect")
        with self.assertRaises(GatewayRequestError):
            gateway.execute("core-one")
        self.assertEqual(1, transport.calls)
        self.assertEqual(1, ledger.summary(plan.run_id)["charged_attempts"])

    def test_unplanned_id_never_reaches_transport(self):
        transport = FakeTransport([response()])
        gateway, ledger, plan = self.gateway(transport, "unplanned")
        with self.assertRaises(GatewayRequestError):
            gateway.execute("not-frozen")
        self.assertEqual(0, transport.calls)
        self.assertEqual(0, ledger.summary(plan.run_id)["charged_attempts"])

    def test_token_echo_is_rejected_and_not_persisted(self):
        transport = FakeTransport([response(body=(b'{"error":"' + SECRET.encode() + b'"}'))])
        gateway, ledger, plan = self.gateway(transport, "echo")
        with self.assertRaises(GatewayRequestError) as caught:
            gateway.execute("core-one")
        self.assertNotIn(SECRET, str(caught.exception))
        for path in self.root.rglob("*"):
            if path.is_file() and not path.name.endswith((".sqlite3", "-wal", "-shm")):
                self.assertNotIn(SECRET.encode(), path.read_bytes())

    def test_account_lease_uses_fixed_token_free_scope(self):
        first_plan = request_plan("credential-one")
        ledger_path = self.ledger_root / "account.sqlite3"
        ledger = RequestLedger(ledger_path)
        OratsGateway(
            plan=first_plan,
            ledger=ledger,
            cache=ContentAddressedCache(self.cache_root / "credential-one-cache"),
            token_source=FakeTokenSource(),
            transport=FakeTransport([response()]),
        )
        second_plan = request_plan("credential-two")
        with self.assertRaises(ActiveRunError):
            OratsGateway(
                plan=second_plan,
                ledger=RequestLedger(ledger_path),
                cache=ContentAddressedCache(self.cache_root / "credential-two-cache"),
                token_source=FakeTokenSource(),
                transport=FakeTransport([response()]),
            )
        import sqlite3

        connection = sqlite3.connect(str(ledger_path))
        scope = connection.execute("SELECT credential_key FROM runs").fetchone()[0]
        connection.close()
        self.assertEqual(ORATS_ACCOUNT_SCOPE, scope)
        self.assertNotEqual(SECRET, scope)
        for path in self.ledger_root.rglob("*"):
            if path.is_file():
                self.assertNotIn(SECRET.encode(), path.read_bytes())

    def test_gateway_rejects_a_per_run_ledger(self):
        plan = request_plan("per-run-ledger")
        with self.assertRaisesRegex(GatewayError, "shared account ledger"):
            OratsGateway(
                plan=plan,
                ledger=RequestLedger(self.ledger_root / "per-run.sqlite3"),
                cache=ContentAddressedCache(self.cache_root / "per-run-cache"),
                token_source=FakeTokenSource(),
                transport=FakeTransport([response()]),
            )

    def test_redaction_covers_literal_and_url_encoded_value(self):
        encoded = "test-secret-should-never-escape"
        text = "token=%s and again %s" % (SECRET, encoded)
        clean = redact_text(text, [SECRET])
        self.assertNotIn(SECRET, clean)
        self.assertIn("[REDACTED]", clean)

    def test_env_source_requires_private_single_key_file(self):
        path = self.env_path
        path.write_text("ORATS_TOKEN=%s\n" % SECRET)
        os.chmod(path, 0o600)
        self.assertEqual(SECRET, EnvFileTokenSource(path).load())
        path.write_text("ORATS_TOKEN=%s\nEXTRA=no\n" % SECRET)
        with self.assertRaises(TokenSourceError):
            EnvFileTokenSource(path).load()

    def test_env_source_rejects_non_cultra_path(self):
        outside = self.root / "outside.env"
        outside.write_text("ORATS_TOKEN=%s\n" % SECRET)
        os.chmod(outside, 0o600)
        with self.assertRaises(TokenSourceError):
            EnvFileTokenSource(outside)

    def test_unix_client_can_only_execute_frozen_id(self):
        transport = FakeTransport([response()])
        gateway, ledger, plan = self.gateway(transport, "socket")
        socket_path = self.root / "gateway.sock"
        server = OratsGatewayServer(socket_path, gateway)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            result = OratsGatewayClient(socket_path).execute("core-one")
            self.assertEqual("core-one", result["logical_request_id"])
            self.assertEqual(0o600, os.stat(socket_path).st_mode & 0o777)
            with self.assertRaises(GatewayRequestError):
                OratsGatewayClient(socket_path).execute("unplanned")
        finally:
            server.shutdown()
            thread.join()
            server.close()
        self.assertEqual(1, transport.calls)


if __name__ == "__main__":
    unittest.main()
