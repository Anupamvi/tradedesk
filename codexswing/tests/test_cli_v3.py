import json
from pathlib import Path
from typing import Any, Mapping

from codexswing.cli import _fetch_hist_strikes, build_parser, main
from codexswing.sources.orats import ORATSClient, ORATSHTTPError
from codexswing.store.immutable import ContentAddressedStore


def _env(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o600)
    return path


def test_doctor_uses_only_configured_api_sources(tmp_path: Path, capsys) -> None:
    orats = _env(tmp_path / "orats.env", "ORATS_TOKEN=test-token\n")
    schwab = _env(tmp_path / "schwab.env", "SCHWAB_ACCESS_TOKEN=test-access\n")
    code = main(
        [
            "doctor",
            "--env-file",
            str(orats),
            "--schwab-env-file",
            str(schwab),
            "--output-root",
            str(tmp_path / "out"),
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["architecture"] == "ORATS_FIRST_SCHWAB_EXECUTION_TRUTH"
    assert payload["broker_mutation_surface"] is False
    assert set(payload["configured_sources"]) == {"ORATS", "Schwab", "public_context"}


def test_active_cli_exposes_daily_and_backtest_controls() -> None:
    parser = build_parser()
    args = parser.parse_args(["run-daily", "--date", "2026-08-27"])
    assert args.backtest_top == 1
    assert args.backtest_workers == 6
    assert args.finalists == 8


def test_historical_404_is_cached_as_rejection_without_hiding_valid_ticker(
    tmp_path: Path,
) -> None:
    calls = []

    def transport(endpoint: str, params: Mapping[str, str]) -> Mapping[str, Any]:
        ticker = params["ticker"]
        calls.append(ticker)
        if ticker == "MISS":
            raise ORATSHTTPError(endpoint, 404, '{"message":"Not Found."}')
        return {
            "data": [
                {
                    "ticker": ticker,
                    "tradeDate": "2026-01-05",
                    "expirDate": "2026-02-20",
                    "strike": 100,
                }
            ]
        }

    store = ContentAddressedStore(tmp_path / "out")
    records, coverage = _fetch_hist_strikes(
        ORATSClient("test-token", transport=transport),
        store,
        {"2026-01-05": ("MISS", "SPY")},
        workers=1,
    )
    assert calls == ["MISS", "SPY"]
    assert {record.source for record in records} == {"orats_hist_strikes"}
    assert {record.source for record in coverage} == {"orats_hist_strikes_unavailable"}
    unavailable = coverage[0]
    assert unavailable.payload["ticker"] == "MISS"
    assert unavailable.payload["httpStatus"] == 404

    def unexpected_transport(endpoint: str, params: Mapping[str, str]) -> Mapping[str, Any]:
        raise AssertionError("fresh request should use positive and negative cache")

    cached, cached_coverage = _fetch_hist_strikes(
        ORATSClient("test-token", transport=unexpected_transport),
        store,
        {"2026-01-05": ("MISS", "SPY")},
        workers=1,
    )
    assert {record.source for record in cached} == {"orats_hist_strikes"}
    assert {record.source for record in cached_coverage} == {
        "orats_hist_strikes_unavailable"
    }
