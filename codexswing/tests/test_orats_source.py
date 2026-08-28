from datetime import datetime, timezone
from typing import Any, Dict, Mapping

from codexswing.sources.orats import ORATSClient


UTC = timezone.utc


def test_orats_transport_never_receives_token_parameter() -> None:
    calls = []

    def transport(endpoint: str, params: Mapping[str, str]) -> Mapping[str, Any]:
        calls.append((endpoint, dict(params)))
        assert "token" not in params
        tickers = params["ticker"].split(",")
        return {
            "data": [
                {
                    "ticker": ticker,
                    "tradeDate": "2026-08-26",
                    "updatedAt": "2026-08-26T20:10:00Z",
                    "iv30d": 0.22,
                }
                for ticker in tickers
            ]
        }

    client = ORATSClient("top-secret-token", transport=transport)
    rows = client.fetch_tickers("cores", ["AAPL", "SPY"])
    assert len(rows) == 2
    records = client.rows_to_records(
        "cores",
        rows,
        ingested_at=datetime(2026, 8, 26, 21, 0, tzinfo=UTC),
    )
    assert len(records) == 2
    assert all("top-secret-token" not in str(record.to_dict()) for record in records)
    assert all(record.source_uri == "https://api.orats.io/datav2/cores" for record in records)
    assert calls == [("cores", {"ticker": "AAPL,SPY"})]


def test_orats_ticker_requests_are_chunked_at_ten() -> None:
    calls = []

    def transport(endpoint: str, params: Mapping[str, str]) -> Mapping[str, Any]:
        calls.append(params["ticker"])
        return {"data": []}

    client = ORATSClient("token-value", transport=transport)
    client.fetch_tickers("cores", ["T{}".format(index) for index in range(11)])
    assert len(calls) == 2
    assert len(calls[0].split(",")) == 10
    assert len(calls[1].split(",")) == 1

