"""Append-only, hash-chained prospective shadow ledger.

The ledger stores hypothetical research events only.  It has no market-data or
broker client and cannot construct or transmit an order.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from codexswing.clock import parse_timestamp
from codexswing.schemas.source import canonical_json


LEDGER_SCHEMA_VERSION = "codexswing.v5.shadow_ledger.v1"
EVENT_TYPES = {"SIGNAL", "QUOTE", "TRIGGER", "EXIT", "OUTCOME"}
SENSITIVE_KEY_PARTS = ("token", "secret", "password", "authorization", "credential")


def _reject_sensitive_keys(value: Any, path: str = "payload") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            text = str(key).lower()
            if any(marker in text for marker in SENSITIVE_KEY_PARTS):
                raise ValueError("sensitive key prohibited at {}.{}".format(path, key))
            _reject_sensitive_keys(child, "{}.{}".format(path, key))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_sensitive_keys(child, "{}[{}]".format(path, index))


@dataclass(frozen=True)
class LedgerEvent:
    event_id: str
    event_type: str
    occurred_at_utc: str
    model_version: str
    spec_sha256: str
    candidate_id: str
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.event_id or not self.candidate_id:
            raise ValueError("event_id and candidate_id are required")
        if self.event_type not in EVENT_TYPES:
            raise ValueError("unsupported shadow-ledger event type")
        parse_timestamp(self.occurred_at_utc)
        if self.model_version != "codexswing-v0.5":
            raise ValueError("shadow ledger only accepts codexswing-v0.5 events")
        if len(self.spec_sha256) != 64:
            raise ValueError("spec_sha256 must be a SHA-256 hex digest")
        try:
            int(self.spec_sha256, 16)
        except ValueError:
            raise ValueError("spec_sha256 must be hexadecimal") from None
        _reject_sensitive_keys(self.payload)
        canonical_json(self.payload)

    def event_body(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "occurred_at_utc": self.occurred_at_utc,
            "model_version": self.model_version,
            "spec_sha256": self.spec_sha256,
            "candidate_id": self.candidate_id,
            "payload": json.loads(canonical_json(self.payload)),
        }


@dataclass(frozen=True)
class LedgerVerification:
    valid: bool
    record_count: int
    head_hash: Optional[str]
    errors: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "record_count": self.record_count,
            "head_hash": self.head_hash,
            "errors": list(self.errors),
        }


def _record_hash(body: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


def _validate_records(records: Sequence[Mapping[str, Any]]) -> LedgerVerification:
    previous = "GENESIS"
    errors: List[str] = []
    seen_event_ids = set()
    expected_keys = {
        "schema_version",
        "sequence",
        "previous_hash",
        "event_id",
        "event_type",
        "occurred_at_utc",
        "model_version",
        "spec_sha256",
        "candidate_id",
        "payload",
        "record_hash",
    }
    for expected_sequence, record in enumerate(records, start=1):
        try:
            if set(record) != expected_keys:
                raise ValueError("record fields do not match ledger schema")
            body = dict(record)
            record_hash = str(body.pop("record_hash"))
            if body.get("schema_version") != LEDGER_SCHEMA_VERSION:
                raise ValueError("unexpected schema")
            if body.get("sequence") != expected_sequence:
                raise ValueError("non-contiguous sequence")
            if body.get("previous_hash") != previous:
                raise ValueError("broken previous_hash link")
            if _record_hash(body) != record_hash:
                raise ValueError("record hash mismatch")
            event_id = str(body.get("event_id") or "")
            if not event_id or event_id in seen_event_ids:
                raise ValueError("missing or duplicate event_id")
            LedgerEvent(
                event_id=event_id,
                event_type=str(body["event_type"]),
                occurred_at_utc=str(body["occurred_at_utc"]),
                model_version=str(body["model_version"]),
                spec_sha256=str(body["spec_sha256"]),
                candidate_id=str(body["candidate_id"]),
                payload=body["payload"],
            )
            seen_event_ids.add(event_id)
            previous = record_hash
        except Exception as exc:
            errors.append("record {}: {}".format(expected_sequence, str(exc)))
            break
    return LedgerVerification(
        valid=not errors,
        record_count=len(records),
        head_hash=(previous if records and not errors else None),
        errors=tuple(errors),
    )


def _read_records(handle: Any) -> List[Mapping[str, Any]]:
    handle.seek(0)
    records = []
    for line_number, line in enumerate(handle, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            raise ValueError("invalid JSON at ledger line {}".format(line_number)) from None
        if not isinstance(value, Mapping):
            raise ValueError("ledger line {} is not an object".format(line_number))
        records.append(value)
    return records


class ProspectiveLedger:
    def __init__(self, path: Path, model_version: str, spec_sha256: str) -> None:
        self.path = path.expanduser().resolve()
        self.model_version = model_version
        self.spec_sha256 = spec_sha256
        if model_version != "codexswing-v0.5":
            raise ValueError("prospective ledger is isolated to codexswing-v0.5")

    def append(self, event: LedgerEvent) -> Mapping[str, Any]:
        if event.model_version != self.model_version or event.spec_sha256 != self.spec_sha256:
            raise ValueError("event model/spec does not match this ledger")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            str(self.path), os.O_RDWR | os.O_APPEND | os.O_CREAT, 0o600
        )
        with os.fdopen(descriptor, "a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                records = _read_records(handle)
                verification = _validate_records(records)
                if not verification.valid:
                    raise ValueError(
                        "refusing to append to invalid ledger: {}".format(verification.errors)
                    )
                desired_event = event.event_body()
                for record in records:
                    if record.get("event_id") != event.event_id:
                        continue
                    existing_event = {
                        key: record[key]
                        for key in desired_event
                    }
                    if existing_event != desired_event:
                        raise ValueError("event_id already exists with different content")
                    return record
                body: Dict[str, Any] = {
                    "schema_version": LEDGER_SCHEMA_VERSION,
                    "sequence": len(records) + 1,
                    "previous_hash": verification.head_hash or "GENESIS",
                }
                body.update(desired_event)
                output = dict(body)
                output["record_hash"] = _record_hash(body)
                handle.seek(0, os.SEEK_END)
                handle.write(canonical_json(output) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                return output
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def verify(self) -> LedgerVerification:
        if not self.path.exists():
            return LedgerVerification(True, 0, None, ())
        with self.path.open("r", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            try:
                records = _read_records(handle)
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return _validate_records(records)

    def events(self) -> Tuple[Mapping[str, Any], ...]:
        if not self.path.exists():
            return ()
        with self.path.open("r", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            try:
                records = _read_records(handle)
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        verification = _validate_records(records)
        if not verification.valid:
            raise ValueError("invalid shadow ledger: {}".format(verification.errors))
        return tuple(records)


class ShadowLedgerRecorder:
    """Idempotent event hooks for a future scheduled local research run."""

    def __init__(self, ledger: ProspectiveLedger) -> None:
        self.ledger = ledger

    def record(
        self,
        event_type: str,
        candidate_id: str,
        occurred_at_utc: str,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        identity = {
            "event_type": event_type,
            "candidate_id": candidate_id,
            "occurred_at_utc": occurred_at_utc,
            "model_version": self.ledger.model_version,
            "spec_sha256": self.ledger.spec_sha256,
            "payload": json.loads(canonical_json(payload)),
        }
        event_id = hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()
        return self.ledger.append(
            LedgerEvent(
                event_id=event_id,
                event_type=event_type,
                occurred_at_utc=occurred_at_utc,
                model_version=self.ledger.model_version,
                spec_sha256=self.ledger.spec_sha256,
                candidate_id=candidate_id,
                payload=payload,
            )
        )

    def signal(self, candidate_id: str, occurred_at_utc: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        return self.record("SIGNAL", candidate_id, occurred_at_utc, payload)

    def quote(self, candidate_id: str, occurred_at_utc: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        return self.record("QUOTE", candidate_id, occurred_at_utc, payload)

    def trigger(self, candidate_id: str, occurred_at_utc: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        return self.record("TRIGGER", candidate_id, occurred_at_utc, payload)

    def exit(self, candidate_id: str, occurred_at_utc: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        return self.record("EXIT", candidate_id, occurred_at_utc, payload)

    def outcome(self, candidate_id: str, occurred_at_utc: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        return self.record("OUTCOME", candidate_id, occurred_at_utc, payload)
