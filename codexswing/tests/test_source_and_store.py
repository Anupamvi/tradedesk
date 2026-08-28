from datetime import datetime, timezone
from pathlib import Path

import pytest

from codexswing.schemas.source import SourceRecord
from codexswing.store.immutable import (
    ContentAddressedStore,
    ImmutableWriteConflict,
    SecretLeakError,
    audit_store,
    write_once_json,
)
from codexswing.store.manifest import RunManifest


UTC = timezone.utc


def _record(payload=None) -> SourceRecord:
    return SourceRecord(
        source="test_source",
        source_id="row-1",
        session_date="2026-08-26",
        event_time_utc="2026-08-26T20:00:00Z",
        published_at_utc="2026-08-26T20:05:00Z",
        first_seen_at_utc="2026-08-26T20:06:00Z",
        available_at_utc="2026-08-26T20:05:00Z",
        ingested_at_utc="2026-08-26T20:06:00Z",
        source_uri="https://example.test/data",
        payload=payload or {"b": 2, "a": 1},
    )


def test_hash_is_stable_across_mapping_order() -> None:
    first = _record({"a": 1, "b": 2})
    second = _record({"b": 2, "a": 1})
    assert first.content_hash == second.content_hash


def test_future_availability_is_rejected() -> None:
    with pytest.raises(ValueError, match="after ingestion"):
        SourceRecord(
            source="test_source",
            source_id="future",
            session_date="2026-08-26",
            available_at_utc="2026-08-26T21:00:00Z",
            ingested_at_utc="2026-08-26T20:00:00Z",
            payload={"value": 1},
        )


def test_sensitive_payload_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="sensitive key"):
        _record({"access_token": "never-store-this"})


def test_content_addressed_store_is_idempotent(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path)
    record = _record()
    first_path = store.put(record)
    second_path = store.put(record)
    assert first_path == second_path
    assert first_path.is_file()
    assert record.content_hash in first_path.name


def test_write_once_conflict_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    write_once_json(path, {"value": 1})
    with pytest.raises(ImmutableWriteConflict):
        write_once_json(path, {"value": 2})


def test_store_refuses_configured_secret_value(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path, secret_values=("secret-value-123",))
    record = _record({"ordinary_field": "secret-value-123"})
    with pytest.raises(SecretLeakError):
        store.put(record)


def test_manifest_is_research_only_and_write_once(tmp_path: Path) -> None:
    record = _record()
    store = ContentAddressedStore(tmp_path)
    record_path = store.put(record)
    manifest = RunManifest.create(
        mode="test_run",
        configuration={"as_of": "2026-08-26"},
        input_records=[record],
        output_paths=[record_path],
    )
    path = manifest.write(tmp_path)
    assert path.is_file()
    assert manifest.status == "RESEARCH_ONLY"
    assert record.content_hash in path.read_text(encoding="utf-8")
    assert manifest.schema_version == "codexswing.run_manifest.v2"
    assert len(manifest.code_tree_sha256) == 64
    assert manifest.output_file_hashes[str(record_path.resolve())]
    audit = audit_store(tmp_path)
    assert audit.valid is True
    assert audit.record_count == 1
    assert audit.manifest_count == 1
    assert audit.trial_count == 0


def test_store_audit_detects_record_tampering(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path)
    path = store.put(_record())
    path.write_text(path.read_text(encoding="utf-8").replace('"a":1', '"a":9'), encoding="utf-8")
    audit = audit_store(tmp_path)
    assert audit.valid is False
    assert any("hash mismatch" in error for error in audit.errors)


def test_deterministic_batch_is_auditable(tmp_path: Path) -> None:
    records = [
        SourceRecord(
            source="batch_source",
            source_id="row-{}".format(index),
            session_date="2026-08-26",
            available_at_utc="2026-08-26T20:00:00Z",
            ingested_at_utc="2026-08-26T20:01:00Z",
            payload={"index": index},
        )
        for index in range(3)
    ]
    store = ContentAddressedStore(tmp_path)
    first = store.put_batch(records)
    second = store.put_batch(reversed(records))
    assert first.path == second.path
    assert first.record_count == 3
    assert first.session_scope == "2026-08-26"
    audit = audit_store(tmp_path)
    assert audit.valid is True
    assert audit.batch_count == 1
    assert audit.record_count == 3


def test_multi_session_batch_is_deterministic_and_auditable(tmp_path: Path) -> None:
    records = [
        SourceRecord(
            source="history_source",
            source_id="row-{}".format(index),
            session_date="2026-08-{:02d}".format(25 + index),
            available_at_utc="2026-08-27T20:00:00Z",
            ingested_at_utc="2026-08-27T20:01:00Z",
            payload={"index": index},
        )
        for index in range(2)
    ]
    batch = ContentAddressedStore(tmp_path).put_batch(records)
    assert batch.session_scope == "_multi_session"
    assert batch.path.parent.name == "_multi_session"
    audit = audit_store(tmp_path)
    assert audit.valid is True
    assert audit.record_count == 2
