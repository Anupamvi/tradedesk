import ast
import hashlib
import json
from pathlib import Path

import pytest

from codexswing.v5.budget import CacheInventory, CacheKey, plan_cache_only
from codexswing.v5.cli import build_parser
from codexswing.schemas.source import SourceRecord
from codexswing.store.immutable import ContentAddressedStore
from codexswing.v5.spec import V5ResearchSpec


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = PROJECT_ROOT / "research_specs" / "ORATS_SWING_RESEARCH_V5.json"


def test_v5_spec_is_inactive_zero_request_and_predeclares_72_variants():
    spec = V5ResearchSpec.from_json_file(SPEC_PATH)

    assert spec.status == "IMPLEMENTED_NOT_EXECUTED"
    assert spec.network_policy == "DENY"
    assert spec.authorized_orats_requests_this_execution == 0
    assert spec.reported_remaining_orats_requests == 12_000
    assert spec.minimum_reserved_orats_requests == 12_000
    assert spec.hypothesis_count == 72
    assert spec.public_summary()["validation_status"] == "NO_REPLAY_RUN"


def test_cache_plan_never_fetches_and_reports_missing_slices():
    spec = V5ResearchSpec.from_json_file(SPEC_PATH)
    existing = CacheKey("hist/cores", "SPY", "2026-01-02")
    missing = CacheKey("hist/strikes", "SPY", "2026-01-05")
    plan = plan_cache_only(
        spec,
        CacheInventory.from_keys([existing]),
        [existing, missing, missing],
    )

    assert plan.status == "BLOCKED_MISSING_CACHE"
    assert plan.requests_executed == 0
    assert plan.conservative_request_upper_bound_if_later_authorized == 1
    assert plan.missing_keys == (missing,)


def test_local_store_inventory_tracks_available_and_known_unavailable(tmp_path):
    store = ContentAddressedStore(tmp_path / "store")
    common = {
        "session_date": "2026-01-02",
        "available_at_utc": "2026-01-02T21:00:00Z",
        "ingested_at_utc": "2026-01-02T21:01:00Z",
    }
    store.put(
        SourceRecord(
            source="orats_hist_dailies",
            source_id="SPY:2026-01-02",
            payload={"ticker": "SPY", "tradeDate": "2026-01-02"},
            **common,
        )
    )
    store.put(
        SourceRecord(
            source="orats_hist_strikes_unavailable",
            source_id="QQQ:2026-01-02:unavailable",
            payload={
                "ticker": "QQQ",
                "requestedTradeDate": "2026-01-02",
                "availability": "NO_ARCHIVED_CHAIN_RETURNED",
            },
            **common,
        )
    )

    inventory = CacheInventory.from_store(store.root)
    assert inventory.contains(CacheKey("hist/dailies", "SPY", "2026-01-02"))
    assert inventory.is_known_unavailable(
        CacheKey("hist/strikes", "QQQ", "2026-01-02")
    )


def test_multi_session_batch_is_recognized_as_full_history_cache(tmp_path):
    store = ContentAddressedStore(tmp_path / "store")
    records = []
    for session_date in ("2026-01-02", "2026-01-03"):
        records.append(
            SourceRecord(
                source="orats_hist_dailies",
                source_id="AAPL:{}".format(session_date),
                session_date=session_date,
                available_at_utc="{}T21:00:00Z".format(session_date),
                ingested_at_utc="{}T21:01:00Z".format(session_date),
                payload={"ticker": "AAPL", "tradeDate": session_date},
            )
        )
    store.put_batch(records)

    inventory = CacheInventory.from_store(store.root)
    assert inventory.contains(CacheKey("hist/dailies", "AAPL", "2025-12-01"))


def test_request_upper_bound_coalesces_full_history_ticker_queries():
    spec = V5ResearchSpec.from_json_file(SPEC_PATH)
    requirements = [
        CacheKey("hist/dailies", "SPY", "2026-01-02"),
        CacheKey("hist/dailies", "SPY", "2026-01-03"),
        CacheKey("hist/cores", "QQQ", "2026-01-02"),
        CacheKey("hist/strikes", "SPY", "2026-01-02"),
        CacheKey("hist/strikes", "SPY", "2026-01-03"),
    ]
    plan = plan_cache_only(spec, CacheInventory.from_keys([]), requirements)
    assert plan.conservative_request_upper_bound_if_later_authorized == 4


def test_cache_plan_rejects_an_undeclared_source():
    spec = V5ResearchSpec.from_json_file(SPEC_PATH)
    with pytest.raises(ValueError, match="undeclared endpoints"):
        plan_cache_only(
            spec,
            CacheInventory.from_keys([]),
            [CacheKey("strikes", "SPY", "2026-01-02")],
        )


def test_v5_package_has_no_network_or_live_adapter_imports():
    package = PROJECT_ROOT / "src" / "codexswing" / "v5"
    forbidden_prefixes = (
        "urllib",
        "http.client",
        "requests",
        "codexswing.sources.orats",
        "codexswing.sources.schwab",
        "codexswing.sources.schwab_auth",
    )
    for path in package.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
        live_imports = [
            name
            for name in imports
            if any(name.startswith(prefix) for prefix in forbidden_prefixes)
        ]
        assert not live_imports, path


def test_v5_cli_parser_exposes_only_read_and_plan_commands():
    parser = build_parser()
    assert parser.parse_args(["describe"]).command == "describe"
    assert parser.parse_args(
        ["plan-cache", "--inventory", "inventory.json", "--paths", "paths.json"]
    ).command == "plan-cache"
    assert parser.parse_args(
        ["plan-cache", "--store-root", "store", "--paths", "paths.json"]
    ).command == "plan-cache"
    assert parser.parse_args(["ledger-verify", "--ledger", "ledger.jsonl"]).command == "ledger-verify"


def test_v4_frozen_manifest_matches_active_source_files():
    manifest = json.loads(
        (PROJECT_ROOT / "research_specs" / "CODEXSWING_V4_FROZEN.json").read_text(
            encoding="utf-8"
        )
    )
    actual = {
        relative: hashlib.sha256((PROJECT_ROOT / relative).read_bytes()).hexdigest()
        for relative in manifest["files"]
    }
    assert actual == manifest["files"]
