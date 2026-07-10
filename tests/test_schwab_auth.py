import datetime as dt
import sys
import time

import pytest

from codexuw.schwab_live import SchwabChainValidator
from uwos.schwab_auth import SchwabAuthConfig, SchwabLiveDataService


@pytest.fixture(autouse=True)
def _disable_option_chain_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UWOS_SCHWAB_OPTION_CHAIN_SUBPROCESS", "0")


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, response=None, exc=None):
        self.timeout = "original-timeout"
        self.response = response or _FakeResponse({"ok": True})
        self.exc = exc
        self.request_timeout = None
        self.request_params = None

    def get(self, url, **kwargs):
        self.request_timeout = kwargs.get("timeout")
        self.request_params = kwargs.get("params", {})
        if self.exc is not None:
            raise self.exc
        return self.response


class _FakeClient:
    def __init__(self, response=None, exc=None):
        self.session = _FakeSession(response=response, exc=exc)
        self.api_key = "fake-api-key"
        self.response = response or _FakeResponse({"ok": True})
        self.exc = exc
        self.timeout_seen_by_chain = None

    def set_timeout(self, timeout):
        self.session.timeout = timeout

    def get_option_chain(self, symbol, **kwargs):
        self.timeout_seen_by_chain = self.session.timeout
        if self.exc is not None:
            raise self.exc
        return self.response


class _FakeReadTimeout(Exception):
    pass


class _SlowSession(_FakeSession):
    def get(self, url, **kwargs):
        self.request_timeout = kwargs.get("timeout")
        self.request_params = kwargs.get("params", {})
        time.sleep(0.2)
        return self.response


def test_get_option_chain_applies_bounded_timeout_and_restores(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UWOS_SCHWAB_OPTION_CHAIN_TIMEOUT_SECONDS", "7.5")
    client = _FakeClient(response=_FakeResponse({"symbol": "AAPL"}))
    service = SchwabLiveDataService(SchwabAuthConfig(api_key="key", app_secret="secret"))
    monkeypatch.setattr(service, "connect", lambda: client)

    payload = service.get_option_chain("AAPL", strike_count=4)

    assert payload == {"symbol": "AAPL"}
    assert client.session.request_timeout == 7.5
    assert client.session.request_params["apikey"] == "fake-api-key"
    assert client.session.request_params["strikeCount"] == 4
    assert client.session.timeout == "original-timeout"


def test_get_option_chain_raises_actionable_timeout_and_restores(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UWOS_SCHWAB_OPTION_CHAIN_TIMEOUT_SECONDS", "3")
    client = _FakeClient(exc=_FakeReadTimeout("read timed out"))
    service = SchwabLiveDataService(SchwabAuthConfig(api_key="key", app_secret="secret"))
    monkeypatch.setattr(service, "connect", lambda: client)

    with pytest.raises(RuntimeError, match=r"timed out for MSFT after 3\.0s"):
        service.get_option_chain("MSFT")

    assert client.session.request_timeout == 3.0
    assert client.session.timeout == "original-timeout"


def test_get_option_chain_interrupts_uncooperative_blocking_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UWOS_SCHWAB_OPTION_CHAIN_TIMEOUT_SECONDS", "0.05")
    client = _FakeClient(response=_FakeResponse({"symbol": "TSLA"}))
    client.session = _SlowSession(response=_FakeResponse({"symbol": "TSLA"}))
    service = SchwabLiveDataService(SchwabAuthConfig(api_key="key", app_secret="secret"))
    monkeypatch.setattr(service, "connect", lambda: client)

    started = time.monotonic()
    with pytest.raises(RuntimeError, match=r"timed out for TSLA after 0\.1s"):
        service.get_option_chain("TSLA")

    assert time.monotonic() - started < 0.15
    assert client.session.request_timeout == 0.05
    assert client.session.timeout == "original-timeout"


def test_get_option_chain_uses_killable_subprocess_before_connect(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    class Completed:
        returncode = 0
        stdout = '{"symbol": "META"}'
        stderr = ""

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return Completed()

    monkeypatch.setenv("UWOS_SCHWAB_OPTION_CHAIN_SUBPROCESS", "1")
    monkeypatch.setenv("UWOS_SCHWAB_OPTION_CHAIN_TIMEOUT_SECONDS", "4")
    monkeypatch.setattr("uwos.schwab_auth.subprocess.run", fake_run)
    service = SchwabLiveDataService(
        SchwabAuthConfig(api_key="key", app_secret="secret", callback_url="https://cb", token_path="/tmp/token.json"),
        interactive_login=False,
    )
    monkeypatch.setattr(service, "connect", lambda: (_ for _ in ()).throw(AssertionError("connect should not run")))

    payload = service.get_option_chain("META", strike_count=5, from_date="2026-06-19", to_date="2026-07-17")

    assert payload == {"symbol": "META"}
    assert captured["cmd"][:3] == [sys.executable, "-m", "uwos.schwab_chain_fetch"]
    assert "--symbol" in captured["cmd"]
    assert "META" in captured["cmd"]
    assert captured["kwargs"]["env"]["UWOS_SCHWAB_OPTION_CHAIN_CHILD"] == "1"
    assert captured["kwargs"]["env"]["SCHWAB_TOKEN_PATH"] == "/tmp/token.json"
    assert captured["kwargs"]["timeout"] == 7.0


def test_chain_validator_skips_undated_fallback_after_timeout(tmp_path) -> None:
    class TimeoutService:
        def __init__(self):
            self.calls = []

        def get_option_chain(self, symbol, **kwargs):
            self.calls.append((symbol, kwargs))
            raise RuntimeError("Schwab option-chain request timed out for NVDA after 3.0s")

    service = TimeoutService()
    validator = SchwabChainValidator(tmp_path)
    validator.service = service

    chain = validator.get_chain("NVDA", dt.date(2026, 6, 11), dt.date(2026, 7, 17))

    assert chain is None
    assert len(service.calls) == 1
    assert "undated fallback skipped after timeout" in validator.errors["NVDA"]
