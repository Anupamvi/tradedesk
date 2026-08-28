import ast
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pytest

from codexswing.sources.schwab import SchwabCredentialUnavailable, SchwabReadOnlyClient


PROHIBITED_IMPORTS = {
    "uwos",
    "codexuw",
    "solcodex",
    "swingdesk",
    "groko",
    "groki",
    "wheel",
    "pattern",
}


def test_schwab_fails_closed_without_authorized_env_token() -> None:
    with pytest.raises(SchwabCredentialUnavailable):
        SchwabReadOnlyClient(None)


def test_schwab_surface_is_read_only() -> None:
    calls = []

    def transport(base_url: str, path: str, params: Mapping[str, str]) -> Any:
        calls.append((base_url, path, dict(params)))
        return {"SPY": {"quote": {"lastPrice": 1.0}}}

    client = SchwabReadOnlyClient("access-value", transport=transport)
    response = client.quotes(["SPY"])
    assert "SPY" in response
    assert calls[0][1] == "/quotes"
    public_methods = {name for name in dir(client) if not name.startswith("_")}
    assert not public_methods.intersection({"submit_order", "place_order", "cancel_order", "replace_order"})


def test_working_orders_are_read_only_account_constraints() -> None:
    calls = []

    def transport(base_url: str, path: str, params: Mapping[str, str]) -> Any:
        calls.append((base_url, path, dict(params)))
        return []

    client = SchwabReadOnlyClient("access-value", transport=transport)
    client.working_orders("hash_1234", "2026-08-01T00:00:00Z", "2026-08-28T00:00:00Z")
    assert calls[0][1] == "/accounts/hash_1234/orders"
    assert calls[0][2]["status"] == "WORKING"


def test_schwab_quote_payload_becomes_point_in_time_records() -> None:
    client = SchwabReadOnlyClient("access-value", transport=lambda *_: {})
    records = client.quote_records(
        {
            "SPY": {
                "assetMainType": "EQUITY",
                "quote": {"lastPrice": 600.0, "quoteTime": 1787848200000},
            }
        },
        ingested_at=datetime(2026, 8, 27, 21, 0, tzinfo=timezone.utc),
    )
    assert len(records) == 1
    assert records[0].source == "schwab_quotes"
    assert records[0].payload["quote"]["lastPrice"] == 600.0
    assert "access-value" not in str(records[0].to_dict())


def test_chain_session_uses_embedded_quote_time_not_after_midnight_ingestion() -> None:
    client = SchwabReadOnlyClient("access-value", transport=lambda *_: {})
    record = client.option_chain_record(
        "SPY",
        {
            "symbol": "SPY",
            "callExpDateMap": {
                "2026-10-16:49": {
                    "600.0": [
                        {
                            "symbol": "SPY   261016C00600000",
                            "quoteTimeInLong": 1787860787912,
                        }
                    ]
                }
            },
            "putExpDateMap": {},
        },
        ingested_at=datetime(2026, 8, 28, 4, 5, tzinfo=timezone.utc),
    )
    assert record.session_date == "2026-08-27"
    assert record.event_time_utc == "2026-08-27T19:59:47.912000Z"


def test_clean_room_package_has_no_prohibited_imports() -> None:
    source_root = Path(__file__).resolve().parents[1] / "src" / "codexswing"
    violations = []
    for path in source_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = {alias.name.split(".", 1)[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots = {node.module.split(".", 1)[0]}
            else:
                continue
            bad = roots.intersection(PROHIBITED_IMPORTS)
            if bad:
                violations.append((str(path), sorted(bad)))
    assert violations == []


def test_schwab_implementation_contains_no_mutating_http_method() -> None:
    source_path = Path(__file__).resolve().parents[1] / "src" / "codexswing" / "sources" / "schwab.py"
    source = source_path.read_text(encoding="utf-8")
    assert 'method="POST"' not in source
    assert 'method="PUT"' not in source
    assert 'method="DELETE"' not in source
