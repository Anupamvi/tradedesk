"""Point-in-time source record used by every CodexSwing adapter."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Dict, Mapping, Optional

from codexswing.clock import parse_timestamp


SCHEMA_VERSION = "codexswing.source_record.v1"
_SESSION_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_SENSITIVE_KEY_PARTS = ("token", "secret", "password", "authorization", "credential")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _reject_sensitive_keys(value: Any, path: str = "payload") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key).lower()
            if any(part in key_text for part in _SENSITIVE_KEY_PARTS):
                raise ValueError("sensitive key is prohibited at {}.{}".format(path, key))
            _reject_sensitive_keys(child, "{}.{}".format(path, key))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_sensitive_keys(child, "{}[{}]".format(path, index))


@dataclass(frozen=True)
class SourceRecord:
    source: str
    source_id: str
    session_date: str
    available_at_utc: str
    ingested_at_utc: str
    payload: Mapping[str, Any] = field(repr=False)
    event_time_utc: Optional[str] = None
    published_at_utc: Optional[str] = None
    first_seen_at_utc: Optional[str] = None
    source_uri: Optional[str] = None
    revision: Optional[str] = None
    schema_version: str = SCHEMA_VERSION
    _canonical_payload: str = field(init=False, repr=False, compare=False)
    _content_hash: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.source or not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", self.source):
            raise ValueError("source must be a lowercase safe identifier")
        if not self.source_id.strip():
            raise ValueError("source_id is required")
        if not _SESSION_RE.fullmatch(self.session_date):
            raise ValueError("session_date must be YYYY-MM-DD")
        if self.source_uri and any(marker in self.source_uri.lower() for marker in ("token=", "access_token=", "authorization=")):
            raise ValueError("source_uri must not contain credentials")

        available = parse_timestamp(self.available_at_utc)
        ingested = parse_timestamp(self.ingested_at_utc)
        if available > ingested + timedelta(minutes=5):
            raise ValueError("available_at_utc cannot be after ingestion")
        for optional_value in (self.event_time_utc, self.published_at_utc, self.first_seen_at_utc):
            if optional_value is not None:
                parse_timestamp(optional_value)

        _reject_sensitive_keys(self.payload)
        payload_text = canonical_json(self.payload)
        payload_copy = json.loads(payload_text)
        body = {
            "schema_version": self.schema_version,
            "source": self.source,
            "source_id": self.source_id,
            "session_date": self.session_date,
            "event_time_utc": self.event_time_utc,
            "published_at_utc": self.published_at_utc,
            "first_seen_at_utc": self.first_seen_at_utc,
            "available_at_utc": self.available_at_utc,
            "ingested_at_utc": self.ingested_at_utc,
            "source_uri": self.source_uri,
            "revision": self.revision,
            "payload": payload_copy,
        }
        content_hash = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
        object.__setattr__(self, "_canonical_payload", payload_text)
        object.__setattr__(self, "_content_hash", content_hash)

    @property
    def content_hash(self) -> str:
        return self._content_hash

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "content_hash": self.content_hash,
            "source": self.source,
            "source_id": self.source_id,
            "session_date": self.session_date,
            "event_time_utc": self.event_time_utc,
            "published_at_utc": self.published_at_utc,
            "first_seen_at_utc": self.first_seen_at_utc,
            "available_at_utc": self.available_at_utc,
            "ingested_at_utc": self.ingested_at_utc,
            "source_uri": self.source_uri,
            "revision": self.revision,
            "payload": json.loads(self._canonical_payload),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SourceRecord":
        expected_hash = str(value.get("content_hash") or "")
        record = cls(
            schema_version=str(value.get("schema_version") or SCHEMA_VERSION),
            source=str(value["source"]),
            source_id=str(value["source_id"]),
            session_date=str(value["session_date"]),
            event_time_utc=value.get("event_time_utc"),
            published_at_utc=value.get("published_at_utc"),
            first_seen_at_utc=value.get("first_seen_at_utc"),
            available_at_utc=str(value["available_at_utc"]),
            ingested_at_utc=str(value["ingested_at_utc"]),
            source_uri=value.get("source_uri"),
            revision=value.get("revision"),
            payload=value["payload"],
        )
        if expected_hash and record.content_hash != expected_hash:
            raise ValueError("source-record content hash mismatch")
        return record
