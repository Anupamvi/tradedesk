"""Cultra-only, content-addressed ORATS cache and concurrency controls."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import stat
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable, Dict, Generic, Iterable, Mapping, Optional, Tuple, TypeVar

from .requesting import PlannedRequest, SecretMaterialError


class CacheError(RuntimeError):
    """Cache data failed validation or durable publication."""


class CacheMiss(CacheError):
    """No validated snapshot satisfies an expected-vintage key."""


_SECRET_KEYS = frozenset(
    {
        "token",
        "apikey",
        "api_key",
        "authorization",
        "access_token",
        "orats_token",
    }
)
CULTRA_PROJECT_ROOT = Path(__file__).resolve().parents[1]
CULTRA_CACHE_ROOT = (CULTRA_PROJECT_ROOT / "state" / "orats_cache").resolve()
_FULL_HISTORY_ENDPOINTS = frozenset(
    {
        "/datav2/hist/cores",
        "/datav2/hist/dailies",
        "/datav2/hist/summaries",
        "/datav2/hist/strikes/options",
    }
)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _reject_secret_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key).lower()
            if key_text in _SECRET_KEYS or "token" in key_text:
                raise SecretMaterialError("cache identity contains credential-like material")
            _reject_secret_keys(nested)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for nested in value:
            _reject_secret_keys(nested)


@dataclass(frozen=True)
class VintageExpectation:
    """Pre-fetch identity; it never pretends provider timestamps are known."""

    endpoint: str
    method: str
    publication_cycle: str
    expected_trade_date: str
    field_profile: str
    schema_version: str
    representation: str
    entities: Tuple[str, ...]

    def __post_init__(self) -> None:
        if self.method != self.method.upper():
            raise CacheError("vintage method must be uppercase")
        if not all(
            (
                self.endpoint,
                self.publication_cycle,
                self.expected_trade_date,
                self.field_profile,
                self.schema_version,
                self.representation,
            )
        ):
            raise CacheError("vintage identity is incomplete")
        normalized = tuple(sorted(set(self.entities)))
        if not normalized or normalized != self.entities:
            raise CacheError("vintage entities must be non-empty, sorted, and unique")

    @property
    def expectation_id(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict())).hexdigest()

    @property
    def entity_group_id(self) -> str:
        payload = self.to_dict()
        payload.pop("entities")
        return hashlib.sha256(_canonical_json(payload)).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "endpoint": self.endpoint,
            "method": self.method,
            "publication_cycle": self.publication_cycle,
            "expected_trade_date": self.expected_trade_date,
            "field_profile": self.field_profile,
            "schema_version": self.schema_version,
            "representation": self.representation,
            "entities": list(self.entities),
        }

    @classmethod
    def from_request(
        cls,
        request: PlannedRequest,
        *,
        expected_trade_date: Optional[str] = None,
        publication_cycle: str = "EOD_DELAYED",
        schema_version: str = "ORATS_NORMALIZED_V1",
        representation: str = "json",
    ) -> "VintageExpectation":
        return cls(
            endpoint=request.endpoint.value,
            method=request.method,
            publication_cycle=publication_cycle,
            expected_trade_date=expected_trade_date or request.expected_vintage,
            field_profile=request.field_profile,
            schema_version=schema_version,
            representation=representation,
            entities=request.entities,
        )


@dataclass(frozen=True)
class SnapshotManifest:
    snapshot_id: str
    cache_key: str
    request_fingerprint: str
    expectation_id: str
    endpoint: str
    method: str
    publication_cycle: str
    expected_trade_date: str
    provider_trade_dates: Tuple[str, ...]
    updated_at_min: Optional[str]
    updated_at_max: Optional[str]
    field_profile: str
    schema_version: str
    representation: str
    requested_entities: Tuple[str, ...]
    returned_entities: Tuple[str, ...]
    missing_entities: Tuple[str, ...]
    raw_sha256: str
    raw_bytes: int
    row_count: int
    created_at: float

    def __post_init__(self) -> None:
        for field_name in (
            "snapshot_id",
            "cache_key",
            "request_fingerprint",
            "expectation_id",
            "raw_sha256",
        ):
            value = getattr(self, field_name)
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise CacheError("snapshot manifest contains an invalid %s" % field_name)
        for field_name in (
            "endpoint",
            "method",
            "publication_cycle",
            "expected_trade_date",
            "field_profile",
            "schema_version",
            "representation",
        ):
            if not str(getattr(self, field_name)).strip():
                raise CacheError("snapshot manifest is missing %s" % field_name)
        if self.method != self.method.upper():
            raise CacheError("snapshot method must be uppercase")
        for field_name in (
            "provider_trade_dates",
            "requested_entities",
            "returned_entities",
            "missing_entities",
        ):
            values = tuple(getattr(self, field_name))
            if values != tuple(sorted(set(values))):
                raise CacheError("snapshot %s must be sorted and unique" % field_name)
        if not self.requested_entities:
            raise CacheError("snapshot requested_entities cannot be empty")
        requested = set(self.requested_entities)
        returned = set(self.returned_entities)
        missing = set(self.missing_entities)
        if returned & missing or returned | missing != requested:
            raise CacheError("snapshot returned/missing entity reconciliation failed")
        if self.raw_bytes <= 0 or self.row_count < 0:
            raise CacheError("snapshot byte and row counts are invalid")
        if not isinstance(self.created_at, (int, float)) or not math.isfinite(
            float(self.created_at)
        ):
            raise CacheError("snapshot created_at is invalid")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "cache_key": self.cache_key,
            "request_fingerprint": self.request_fingerprint,
            "expectation_id": self.expectation_id,
            "endpoint": self.endpoint,
            "method": self.method,
            "publication_cycle": self.publication_cycle,
            "expected_trade_date": self.expected_trade_date,
            "provider_trade_dates": list(self.provider_trade_dates),
            "updated_at_min": self.updated_at_min,
            "updated_at_max": self.updated_at_max,
            "field_profile": self.field_profile,
            "schema_version": self.schema_version,
            "representation": self.representation,
            "requested_entities": list(self.requested_entities),
            "returned_entities": list(self.returned_entities),
            "missing_entities": list(self.missing_entities),
            "raw_sha256": self.raw_sha256,
            "raw_bytes": self.raw_bytes,
            "row_count": self.row_count,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SnapshotManifest":
        return cls(
            snapshot_id=str(value["snapshot_id"]),
            cache_key=str(value["cache_key"]),
            request_fingerprint=str(value["request_fingerprint"]),
            expectation_id=str(value["expectation_id"]),
            endpoint=str(value["endpoint"]),
            method=str(value["method"]),
            publication_cycle=str(value["publication_cycle"]),
            expected_trade_date=str(value["expected_trade_date"]),
            provider_trade_dates=tuple(value["provider_trade_dates"]),
            updated_at_min=value.get("updated_at_min"),
            updated_at_max=value.get("updated_at_max"),
            field_profile=str(value["field_profile"]),
            schema_version=str(value["schema_version"]),
            representation=str(value["representation"]),
            requested_entities=tuple(value["requested_entities"]),
            returned_entities=tuple(value["returned_entities"]),
            missing_entities=tuple(value["missing_entities"]),
            raw_sha256=str(value["raw_sha256"]),
            raw_bytes=int(value["raw_bytes"]),
            row_count=int(value["row_count"]),
            created_at=float(value["created_at"]),
        )


def _snapshot_identity(manifest: SnapshotManifest) -> Dict[str, Any]:
    return {
        "cache_key": manifest.cache_key,
        "request_fingerprint": manifest.request_fingerprint,
        "expectation_id": manifest.expectation_id,
        "endpoint": manifest.endpoint,
        "method": manifest.method,
        "publication_cycle": manifest.publication_cycle,
        "expected_trade_date": manifest.expected_trade_date,
        "provider_trade_dates": manifest.provider_trade_dates,
        "updated_at_min": manifest.updated_at_min,
        "updated_at_max": manifest.updated_at_max,
        "field_profile": manifest.field_profile,
        "schema_version": manifest.schema_version,
        "representation": manifest.representation,
        "requested_entities": manifest.requested_entities,
        "returned_entities": manifest.returned_entities,
        "missing_entities": manifest.missing_entities,
        "raw_sha256": manifest.raw_sha256,
        "raw_bytes": manifest.raw_bytes,
        "row_count": manifest.row_count,
    }


def _snapshot_id(manifest: SnapshotManifest) -> str:
    return hashlib.sha256(_canonical_json(_snapshot_identity(manifest))).hexdigest()


def cache_key_for(request: PlannedRequest, expectation: VintageExpectation) -> str:
    """Return the complete, token-free pre-fetch cache identity."""

    payload = {
        "endpoint": request.endpoint.value,
        "method": request.method,
        "params": dict(request.params),
        "body": dict(request.body),
        "field_profile": request.field_profile,
        "fields": list(request.fields),
        "vintage": expectation.to_dict(),
        "schema_version": expectation.schema_version,
    }
    _reject_secret_keys(payload)
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


class ContentAddressedCache:
    """Immutable raw blobs plus atomically published snapshot manifests."""

    def __init__(self, root: Path) -> None:
        supplied_root = Path(root).expanduser()
        if supplied_root.is_symlink():
            raise CacheError("cache root may not be a symlink")
        self.root = supplied_root.resolve()
        if not _is_within(self.root, CULTRA_CACHE_ROOT.resolve()):
            raise CacheError("cache root must remain inside Cultra's ORATS cache root")
        self.raw_root = self.root / "raw"
        self.manifest_root = self.root / "manifests"
        self.index_root = self.root / "index"
        for directory in (self.root, self.raw_root, self.manifest_root, self.index_root):
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(directory, 0o700)

    @staticmethod
    def _validate_hash(value: str, label: str) -> None:
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise CacheError("invalid %s" % label)

    def _raw_path(self, digest: str) -> Path:
        self._validate_hash(digest, "raw digest")
        return self.raw_root / digest[:2] / (digest + ".bin")

    def _manifest_path(self, snapshot_id: str) -> Path:
        self._validate_hash(snapshot_id, "snapshot ID")
        return self.manifest_root / snapshot_id[:2] / (snapshot_id + ".json")

    def _index_path(self, cache_key: str) -> Path:
        self._validate_hash(cache_key, "cache key")
        return self.index_root / cache_key[:2] / (cache_key + ".json")

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(path.parent, 0o700)
        temporary = path.with_name(
            ".%s.tmp-%d-%s" % (path.name, os.getpid(), uuid.uuid4().hex)
        )
        try:
            with open(temporary, "xb") as handle:
                os.chmod(temporary, 0o600)
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, path)
                os.unlink(temporary)
            except FileExistsError:
                os.unlink(temporary)
            directory_fd = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    @staticmethod
    def _atomic_replace(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(path.parent, 0o700)
        temporary = path.with_name(
            ".%s.tmp-%d-%s" % (path.name, os.getpid(), uuid.uuid4().hex)
        )
        try:
            with open(temporary, "xb") as handle:
                os.chmod(temporary, 0o600)
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            directory_fd = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def publish(
        self,
        *,
        request: PlannedRequest,
        expectation: VintageExpectation,
        raw: bytes,
        provider_trade_dates: Iterable[str],
        returned_entities: Iterable[str],
        row_count: int,
        updated_at_min: Optional[str] = None,
        updated_at_max: Optional[str] = None,
    ) -> SnapshotManifest:
        if not isinstance(raw, bytes) or not raw:
            raise CacheError("empty or non-byte provider response")
        if len(raw) > request.max_response_bytes:
            raise CacheError("provider response exceeds the planned byte safeguard")
        if row_count < 0 or row_count > request.expected_rows * 10:
            raise CacheError("provider row count is invalid or unexpectedly unbounded")
        if request.endpoint.value != expectation.endpoint or request.method != expectation.method:
            raise CacheError("request and expected-vintage endpoint identity disagree")
        if request.field_profile != expectation.field_profile:
            raise CacheError("request and expected-vintage field profiles disagree")
        if request.entities != expectation.entities:
            raise CacheError("request and expected-vintage coverage disagree")
        dates = tuple(sorted(set(str(value) for value in provider_trade_dates if value)))
        event_history_endpoint = request.endpoint.value in {
            "/datav2/hist/splits",
        }
        full_history_endpoint = request.endpoint.value in _FULL_HISTORY_ENDPOINTS
        if (
            request.endpoint.value.startswith("/datav2/")
            and not event_history_endpoint
            and not dates
        ):
            raise CacheError("delayed-data response is missing provider tradeDate")
        if (
            dates
            and not event_history_endpoint
            and not full_history_endpoint
            and expectation.expected_trade_date not in dates
        ):
            raise CacheError("provider trade date does not satisfy expected vintage")
        if full_history_endpoint and dates:
            try:
                expected_through = date.fromisoformat(expectation.expected_trade_date)
                provider_dates = tuple(date.fromisoformat(value) for value in dates)
            except ValueError as exc:
                raise CacheError("full-history vintage contains an invalid date") from exc
            if max(provider_dates) != expected_through:
                raise CacheError(
                    "full-history response does not reach its frozen through-date"
                )
        returned = tuple(sorted(set(str(value).upper() for value in returned_entities)))
        requested_set = set(expectation.entities)
        if not set(returned).issubset(requested_set):
            raise CacheError("response contains entities outside the planned request")
        # Split-history is an event query, not a one-row-per-entity snapshot.
        # Whether zero, one, or several requested names have records, the
        # successful response covers the full query batch; absent rows mean no
        # split in the requested history rather than missing entity coverage.
        if event_history_endpoint:
            returned = tuple(sorted(requested_set))
        missing = tuple(sorted(requested_set.difference(returned)))
        if request.required and missing:
            raise CacheError("required response is missing planned entities")
        raw_digest = hashlib.sha256(raw).hexdigest()
        cache_key = cache_key_for(request, expectation)
        provisional_snapshot_id = "0" * 64
        manifest = SnapshotManifest(
            snapshot_id=provisional_snapshot_id,
            cache_key=cache_key,
            request_fingerprint=request.fingerprint,
            expectation_id=expectation.expectation_id,
            endpoint=expectation.endpoint,
            method=expectation.method,
            publication_cycle=expectation.publication_cycle,
            expected_trade_date=expectation.expected_trade_date,
            provider_trade_dates=dates,
            updated_at_min=updated_at_min,
            updated_at_max=updated_at_max,
            field_profile=expectation.field_profile,
            schema_version=expectation.schema_version,
            representation=expectation.representation,
            requested_entities=expectation.entities,
            returned_entities=returned,
            missing_entities=missing,
            raw_sha256=raw_digest,
            raw_bytes=len(raw),
            row_count=int(row_count),
            created_at=time.time(),
        )
        snapshot_id = hashlib.sha256(
            _canonical_json(_snapshot_identity(manifest))
        ).hexdigest()
        manifest = SnapshotManifest.from_dict(
            dict(manifest.to_dict(), snapshot_id=snapshot_id)
        )
        raw_path = self._raw_path(raw_digest)
        manifest_path = self._manifest_path(snapshot_id)
        self._atomic_write(raw_path, raw)
        self._atomic_write(
            manifest_path,
            _canonical_json(manifest.to_dict()) + b"\n",
        )
        index = {
            "cache_key": cache_key,
            "snapshot_id": snapshot_id,
            "expectation_id": expectation.expectation_id,
        }
        self._atomic_replace(self._index_path(cache_key), _canonical_json(index) + b"\n")
        return manifest

    def lookup(
        self, request: PlannedRequest, expectation: VintageExpectation
    ) -> Tuple[SnapshotManifest, bytes]:
        cache_key = cache_key_for(request, expectation)
        index_path = self._index_path(cache_key)
        try:
            if index_path.is_symlink() or stat.S_IMODE(index_path.stat().st_mode) & 0o077:
                raise CacheError("cache index permissions or identity are unsafe")
            index = json.loads(index_path.read_text(encoding="utf-8"))
            if index.get("cache_key") != cache_key:
                raise CacheError("cache index identity mismatch")
            manifest_path = self._manifest_path(str(index["snapshot_id"]))
            if manifest_path.is_symlink() or stat.S_IMODE(manifest_path.stat().st_mode) & 0o077:
                raise CacheError("cache manifest permissions or identity are unsafe")
            manifest = SnapshotManifest.from_dict(
                json.loads(manifest_path.read_text(encoding="utf-8"))
            )
            if _snapshot_id(manifest) != manifest.snapshot_id:
                raise CacheError("snapshot manifest identity hash mismatch")
            if manifest.cache_key != cache_key:
                raise CacheError("snapshot cache-key mismatch")
            if manifest.expectation_id != expectation.expectation_id:
                raise CacheError("snapshot expected-vintage mismatch")
            if request.required and manifest.missing_entities:
                raise CacheError("required cached snapshot has incomplete entity coverage")
            raw_path = self._raw_path(manifest.raw_sha256)
            if raw_path.is_symlink() or stat.S_IMODE(raw_path.stat().st_mode) & 0o077:
                raise CacheError("cached raw response permissions or identity are unsafe")
            raw = raw_path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != manifest.raw_sha256:
                raise CacheError("cached raw response hash mismatch")
            if len(raw) != manifest.raw_bytes:
                raise CacheError("cached raw response length mismatch")
            return manifest, raw
        except FileNotFoundError as exc:
            raise CacheMiss("no validated snapshot for expected vintage") from exc
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise CacheError("cache metadata is malformed") from exc

    def load_snapshot(self, snapshot_id: str) -> Tuple[SnapshotManifest, bytes]:
        """Load one immutable snapshot by ID without following a mutable index."""

        self._validate_hash(str(snapshot_id), "snapshot ID")
        try:
            manifest_path = self._manifest_path(str(snapshot_id))
            if manifest_path.is_symlink() or stat.S_IMODE(
                manifest_path.stat().st_mode
            ) & 0o077:
                raise CacheError("cache manifest permissions or identity are unsafe")
            manifest = SnapshotManifest.from_dict(
                json.loads(manifest_path.read_text(encoding="utf-8"))
            )
            if manifest.snapshot_id != str(snapshot_id) or _snapshot_id(manifest) != str(
                snapshot_id
            ):
                raise CacheError("snapshot manifest identity hash mismatch")
            raw_path = self._raw_path(manifest.raw_sha256)
            if raw_path.is_symlink() or stat.S_IMODE(raw_path.stat().st_mode) & 0o077:
                raise CacheError("cached raw response permissions or identity are unsafe")
            raw = raw_path.read_bytes()
            if (
                hashlib.sha256(raw).hexdigest() != manifest.raw_sha256
                or len(raw) != manifest.raw_bytes
            ):
                raise CacheError("cached raw response failed integrity verification")
            return manifest, raw
        except FileNotFoundError as exc:
            raise CacheMiss("immutable snapshot is unavailable") from exc
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise CacheError("cache snapshot metadata is malformed") from exc

    def verify(self) -> Dict[str, int]:
        indexes = manifests = blobs = 0
        for index_path in self.index_root.glob("*/*.json"):
            indexes += 1
            try:
                index = json.loads(index_path.read_text(encoding="utf-8"))
                manifest_path = self._manifest_path(str(index["snapshot_id"]))
                manifest = SnapshotManifest.from_dict(
                    json.loads(manifest_path.read_text(encoding="utf-8"))
                )
                raw = self._raw_path(manifest.raw_sha256).read_bytes()
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                raise CacheError("cache verification failed") from exc
            if hashlib.sha256(raw).hexdigest() != manifest.raw_sha256:
                raise CacheError("cache verification found a corrupt raw blob")
            manifests += 1
            blobs += 1
        return {"indexes": indexes, "manifests": manifests, "raw_blobs": blobs}


T = TypeVar("T")


class _Flight(Generic[T]):
    def __init__(self) -> None:
        self.event = threading.Event()
        self.value: Optional[T] = None
        self.error: Optional[BaseException] = None


class SingleFlight:
    """Collapse identical concurrent work into one leader invocation."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._flights: Dict[str, _Flight[Any]] = {}

    def run(self, key: str, function: Callable[[], T]) -> T:
        with self._lock:
            flight = self._flights.get(key)
            if flight is None:
                flight = _Flight[T]()
                self._flights[key] = flight
                leader = True
            else:
                leader = False
        if leader:
            try:
                flight.value = function()
            except BaseException as exc:
                flight.error = exc
            finally:
                flight.event.set()
                with self._lock:
                    self._flights.pop(key, None)
        else:
            flight.event.wait()
        if flight.error is not None:
            raise flight.error
        return flight.value  # type: ignore[return-value]


@dataclass(frozen=True)
class EntityClaim:
    group_key: str
    owner_id: str
    claimed: Tuple[str, ...]
    ready: Tuple[str, ...]
    pending: Tuple[str, ...]


class EntityClaimStore:
    """Process-safe overlapping-batch deduplication by expected vintage."""

    def __init__(self, path: Path, *, timeout_seconds: float = 10.0) -> None:
        supplied = Path(path).expanduser()
        if supplied.is_symlink():
            raise CacheError("entity-claim path may not be a symlink")
        self.path = supplied.resolve()
        if (
            not _is_within(self.path, CULTRA_CACHE_ROOT.resolve())
            or self.path.suffix != ".sqlite3"
        ):
            raise CacheError("entity claims must remain inside Cultra's cache root")
        self.timeout_seconds = timeout_seconds
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        connection = sqlite3.connect(str(self.path), isolation_level=None)
        try:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA synchronous=FULL;
                CREATE TABLE IF NOT EXISTS entity_claims (
                    group_key TEXT NOT NULL,
                    entity TEXT NOT NULL,
                    state TEXT NOT NULL CHECK(state IN ('in_flight','ready','failed')),
                    owner_id TEXT,
                    snapshot_id TEXT,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY(group_key, entity)
                );
                """
            )
        finally:
            connection.close()
        os.chmod(self.path, 0o600)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            str(self.path), timeout=self.timeout_seconds, isolation_level=None
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA busy_timeout=%d" % int(self.timeout_seconds * 1000))
        return connection

    def claim(
        self, group_key: str, entities: Iterable[str], *, owner_id: Optional[str] = None
    ) -> EntityClaim:
        owner = owner_id or uuid.uuid4().hex
        normalized = tuple(sorted(set(str(entity).upper() for entity in entities)))
        if not group_key or not normalized:
            raise CacheError("entity claim needs a group and bounded entities")
        claimed = []
        ready = []
        pending = []
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            now = time.time()
            for entity in normalized:
                row = connection.execute(
                    "SELECT state, owner_id FROM entity_claims WHERE group_key = ? AND entity = ?",
                    (group_key, entity),
                ).fetchone()
                if row is None:
                    connection.execute(
                        """
                        INSERT INTO entity_claims(
                            group_key, entity, state, owner_id, updated_at
                        ) VALUES (?, ?, 'in_flight', ?, ?)
                        """,
                        (group_key, entity, owner, now),
                    )
                    claimed.append(entity)
                elif row["state"] == "ready":
                    ready.append(entity)
                elif row["state"] == "failed":
                    connection.execute(
                        """
                        UPDATE entity_claims
                        SET state = 'in_flight', owner_id = ?, snapshot_id = NULL, updated_at = ?
                        WHERE group_key = ? AND entity = ?
                        """,
                        (owner, now, group_key, entity),
                    )
                    claimed.append(entity)
                elif row["owner_id"] == owner:
                    claimed.append(entity)
                else:
                    pending.append(entity)
            connection.execute("COMMIT")
        except sqlite3.Error as exc:
            try:
                connection.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise CacheError("could not acquire entity claims") from exc
        finally:
            connection.close()
        return EntityClaim(group_key, owner, tuple(claimed), tuple(ready), tuple(pending))

    def complete(
        self,
        claim: EntityClaim,
        *,
        snapshot_id: str,
        ready_entities: Optional[Iterable[str]] = None,
    ) -> None:
        ready = set(
            str(entity).upper()
            for entity in (ready_entities if ready_entities is not None else claim.claimed)
        )
        if not ready.issubset(set(claim.claimed)):
            raise CacheError("cannot complete entities not owned by this claim")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            now = time.time()
            for entity in claim.claimed:
                state = "ready" if entity in ready else "failed"
                cursor = connection.execute(
                    """
                    UPDATE entity_claims
                    SET state = ?, snapshot_id = ?, owner_id = NULL, updated_at = ?
                    WHERE group_key = ? AND entity = ?
                      AND state = 'in_flight' AND owner_id = ?
                    """,
                    (
                        state,
                        snapshot_id if state == "ready" else None,
                        now,
                        claim.group_key,
                        entity,
                        claim.owner_id,
                    ),
                )
                if cursor.rowcount != 1:
                    raise CacheError("entity claim ownership changed before completion")
            connection.execute("COMMIT")
        except (sqlite3.Error, CacheError) as exc:
            try:
                connection.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            if isinstance(exc, CacheError):
                raise
            raise CacheError("could not complete entity claims") from exc
        finally:
            connection.close()

    def fail(self, claim: EntityClaim) -> None:
        self.complete(claim, snapshot_id="", ready_entities=())

    def ready_snapshot(self, group_key: str, entity: str) -> Optional[str]:
        connection = self._connect()
        try:
            row = connection.execute(
                """
                SELECT snapshot_id FROM entity_claims
                WHERE group_key = ? AND entity = ? AND state = 'ready'
                """,
                (group_key, entity.upper()),
            ).fetchone()
            return None if row is None else str(row["snapshot_id"])
        finally:
            connection.close()
