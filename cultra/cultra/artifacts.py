"""Immutable, secret-safe artifacts for Cultra runs.

This module deliberately has no knowledge of credentials or transports.  Its
only job is to turn already-computed run evidence into reproducible files.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


MANIFEST_SCHEMA = "cultra.run-manifest.v1"
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SECRET_KEYS = {
    "authorization",
    "token",
    "api_key",
    "apikey",
    "access_token",
    "refresh_token",
    "client_secret",
    "orats_token",
}
_SECRET_TEXT_PATTERNS = (
    re.compile(r"(?i)authorization\s*:\s*bearer\s+(?P<value>\S+)"),
    re.compile(r"(?i)\bORATS_TOKEN\s*=\s*(?P<value>[^\s&#;]+)"),
    re.compile(
        r"(?i)(?:^|[?&;\s])(?:token|orats[_-]?token|access[_-]?token|"
        r"api[_-]?key|apikey|authorization)\s*=\s*(?P<value>[^\s&#;]+)"
    ),
    re.compile(
        r'''(?ix)["'](?:token|orats[_-]?token|access[_-]?token|api[_-]?key|'''
        r'''apikey|authorization)["']\s*:\s*["'](?P<value>[^"']+)["']'''
    ),
)
_SAFE_REDACTED_VALUES = frozenset(
    {
        "",
        "***",
        "[redacted]",
        "%5bredacted%5d",
        "<redacted>",
        "%3credacted%3e",
        "redacted",
        "not_set",
    }
)
_SOURCE_FINGERPRINT_RE = re.compile(r"^[0-9a-f]{64}$")


class ArtifactError(ValueError):
    """Raised when an artifact would violate Cultra's immutability rules."""


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _json_value(value: Any) -> Any:
    """Convert supported Cultra values to deterministic JSON data."""

    if dataclasses.is_dataclass(value):
        return {
            item.name: _json_value(getattr(value, item.name))
            for item in dataclasses.fields(value)
        }
    if isinstance(value, Enum):
        return _json_value(value.value)
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ArtifactError("naive datetimes are not reproducible")
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        converted = [_json_value(item) for item in value]
        return sorted(converted, key=lambda item: json.dumps(item, sort_keys=True))
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ArtifactError("unsupported artifact value type: %s" % type(value).__name__)


def assert_secret_free(value: Any, path: str = "$") -> None:
    """Fail closed when a payload contains a credential-shaped key or value."""

    converted = _json_value(value)

    def walk(item: Any, location: str) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                normalized = str(key).strip().lower().replace("-", "_")
                if normalized in _SECRET_KEYS:
                    raise ArtifactError("secret-bearing key rejected at %s.%s" % (location, key))
                walk(child, "%s.%s" % (location, key))
        elif isinstance(item, list):
            for index, child in enumerate(item):
                walk(child, "%s[%d]" % (location, index))
        elif isinstance(item, str):
            for pattern in _SECRET_TEXT_PATTERNS:
                match = pattern.search(item)
                if match and match.group("value").strip().lower() not in _SAFE_REDACTED_VALUES:
                    raise ArtifactError("credential-shaped text rejected at %s" % location)

    walk(converted, path)


def assert_secret_free_bytes(payload: bytes, path: str = "$") -> None:
    """Scan public byte artifacts for textual credential assignments.

    Binary data remains supported. UTF-8/ASCII fragments are inspected without
    rejecting harmless prose containing words such as ``token`` or ``api key``.
    """

    if not isinstance(payload, bytes):
        raise ArtifactError("artifact payload must be bytes")
    text = payload.decode("utf-8", errors="ignore")
    for pattern in _SECRET_TEXT_PATTERNS:
        match = pattern.search(text)
        if match and match.group("value").strip().lower() not in _SAFE_REDACTED_VALUES:
            raise ArtifactError("credential-shaped byte payload rejected at %s" % path)


def source_fingerprint(project_root: Optional[Path] = None) -> str:
    """Hash every runtime Python/config source file without reading credentials."""

    root = Path(project_root or Path(__file__).resolve().parents[1]).resolve()
    candidates = []
    package = root / "cultra"
    if package.is_dir():
        candidates.extend(path for path in package.rglob("*.py") if path.is_file())
    configs = root / "configs"
    if configs.is_dir():
        candidates.extend(
            path
            for path in configs.rglob("*")
            if path.is_file() and path.suffix.lower() in {".json", ".yaml", ".yml"}
        )
    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        candidates.append(pyproject)
    if not candidates:
        raise ArtifactError("no Cultra source files found for fingerprinting")

    digest = hashlib.sha256()
    for path in sorted(set(candidates), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    assert_secret_free(value)
    return (
        json.dumps(
            _json_value(value),
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


@dataclass(frozen=True)
class ArtifactRecord:
    path: str
    media_type: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        if not self.path or Path(self.path).is_absolute() or ".." in Path(self.path).parts:
            raise ArtifactError("artifact record path must be relative and contained")
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ArtifactError("artifact sha256 is malformed")
        if self.size_bytes < 0:
            raise ArtifactError("artifact size cannot be negative")


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    as_of: date
    created_at: datetime
    overall_status: str
    artifacts: Tuple[ArtifactRecord, ...]
    code_version: str = "0.1.0"
    source_fingerprint: str = ""
    request_plan_id: Optional[str] = None
    snapshot_ids: Tuple[str, ...] = ()
    model_versions: Mapping[str, str] = field(default_factory=dict)
    field_profile_versions: Mapping[str, str] = field(default_factory=dict)
    field_profile_statuses: Mapping[str, str] = field(default_factory=dict)
    strategy_states: Mapping[str, str] = field(default_factory=dict)
    source_trade_dates: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        validate_run_id(self.run_id)
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ArtifactError("manifest created_at must be timezone-aware")
        if not _SOURCE_FINGERPRINT_RE.fullmatch(self.source_fingerprint):
            raise ArtifactError("manifest source_fingerprint must be a sha256 digest")
        paths = [record.path for record in self.artifacts]
        if len(paths) != len(set(paths)):
            raise ArtifactError("manifest contains duplicate artifact paths")
        assert_secret_free(self)

    def to_dict(self) -> Dict[str, Any]:
        return _json_value(self)


def validate_run_id(run_id: str) -> str:
    if not _RUN_ID_RE.fullmatch(run_id):
        raise ArtifactError(
            "run_id must contain only letters, digits, dot, underscore, and hyphen"
        )
    return run_id


class ArtifactWriter:
    """Create one immutable, permission-restricted run directory."""

    def __init__(self, output_root: Path, run_id: str) -> None:
        self.output_root = Path(output_root).resolve()
        self.run_id = validate_run_id(run_id)
        self.run_dir = self.output_root / self.run_id
        self._records: Dict[str, ArtifactRecord] = {}
        self._finalized = False

        self.output_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.run_dir.exists():
            raise ArtifactError("run directory already exists: %s" % self.run_dir)
        self.run_dir.mkdir(mode=0o700)
        os.chmod(str(self.run_dir), 0o700)

    @property
    def records(self) -> Tuple[ArtifactRecord, ...]:
        return tuple(self._records[key] for key in sorted(self._records))

    def _destination(self, relative_path: str) -> Path:
        if self._finalized:
            raise ArtifactError("run artifacts are already finalized")
        candidate = Path(relative_path)
        if candidate.is_absolute() or not candidate.parts or ".." in candidate.parts:
            raise ArtifactError("artifact path must remain inside the run directory")
        if candidate.name in {"", ".", ".."}:
            raise ArtifactError("artifact path must name a file")
        destination = self.run_dir.joinpath(*candidate.parts)
        resolved_parent = destination.parent.resolve()
        try:
            resolved_parent.relative_to(self.run_dir.resolve())
        except ValueError as exc:
            raise ArtifactError("artifact path escapes run directory") from exc
        if destination.is_symlink() or destination.exists():
            raise ArtifactError("artifact paths are immutable: %s" % relative_path)
        return destination

    def write_bytes(self, relative_path: str, payload: bytes, media_type: str) -> ArtifactRecord:
        if not isinstance(payload, bytes):
            raise ArtifactError("artifact payload must be bytes")
        assert_secret_free_bytes(payload, path=relative_path)
        destination = self._destination(relative_path)
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(str(destination.parent), 0o700)

        handle, temporary_name = tempfile.mkstemp(prefix=".cultra-", dir=str(destination.parent))
        try:
            os.fchmod(handle, 0o600)
            with os.fdopen(handle, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, destination)
            os.chmod(str(destination), 0o600)
        except BaseException:
            try:
                os.close(handle)
            except OSError:
                pass
            try:
                os.unlink(temporary_name)
            except OSError:
                pass
            raise

        relative = destination.relative_to(self.run_dir).as_posix()
        record = ArtifactRecord(
            path=relative,
            media_type=media_type,
            sha256=hashlib.sha256(payload).hexdigest(),
            size_bytes=len(payload),
        )
        self._records[relative] = record
        return record

    def write_json(self, relative_path: str, value: Any) -> ArtifactRecord:
        return self.write_bytes(relative_path, canonical_json_bytes(value), "application/json")

    def write_text(self, relative_path: str, text: str, media_type: str = "text/plain") -> ArtifactRecord:
        if not isinstance(text, str):
            raise ArtifactError("text artifact must be a string")
        assert_secret_free(text)
        return self.write_bytes(relative_path, text.encode("utf-8"), media_type)

    def finalize(
        self,
        *,
        as_of: date,
        overall_status: str,
        created_at: Optional[datetime] = None,
        code_version: str = "0.1.0",
        source_fingerprint_value: Optional[str] = None,
        request_plan_id: Optional[str] = None,
        snapshot_ids: Sequence[str] = (),
        model_versions: Optional[Mapping[str, str]] = None,
        field_profile_versions: Optional[Mapping[str, str]] = None,
        field_profile_statuses: Optional[Mapping[str, str]] = None,
        strategy_states: Optional[Mapping[str, str]] = None,
        source_trade_dates: Optional[Mapping[str, str]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> RunManifest:
        if self._finalized:
            raise ArtifactError("run artifacts are already finalized")
        manifest = RunManifest(
            run_id=self.run_id,
            as_of=as_of,
            created_at=created_at or utc_now(),
            overall_status=overall_status,
            artifacts=self.records,
            code_version=code_version,
            source_fingerprint=source_fingerprint_value or source_fingerprint(),
            request_plan_id=request_plan_id,
            snapshot_ids=tuple(snapshot_ids),
            model_versions=dict(model_versions or {}),
            field_profile_versions=dict(field_profile_versions or {}),
            field_profile_statuses=dict(field_profile_statuses or {}),
            strategy_states=dict(strategy_states or {}),
            source_trade_dates=dict(source_trade_dates or {}),
            metadata=dict(metadata or {}),
        )
        self.write_json("manifest.json", manifest.to_dict())
        self._finalized = True
        return manifest


def verify_manifest(run_dir: Path) -> Tuple[str, ...]:
    """Return integrity/reconciliation errors for a complete immutable run."""

    root = Path(run_dir).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return ("manifest.json is missing",)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return ("manifest.json is unreadable: %s" % exc,)

    errors = []
    if not isinstance(payload, Mapping):
        return ("manifest.json must contain a JSON object",)
    if payload.get("schema") != MANIFEST_SCHEMA:
        errors.append("manifest schema is missing or unsupported")
    if not _SOURCE_FINGERPRINT_RE.fullmatch(str(payload.get("source_fingerprint", ""))):
        errors.append("manifest source_fingerprint is missing or malformed")
    if payload.get("run_id") != root.name:
        errors.append("manifest run_id does not match its run directory")
    try:
        assert_secret_free(payload)
    except ArtifactError as exc:
        errors.append("manifest contains credential-shaped material: %s" % exc)

    raw_records = payload.get("artifacts")
    if not isinstance(raw_records, list):
        return tuple(errors + ["manifest artifacts must be a list"])
    listed_paths = []
    for raw_record in raw_records:
        if not isinstance(raw_record, Mapping):
            errors.append("manifest artifact record is not an object")
            continue
        try:
            record = ArtifactRecord(
                path=str(raw_record.get("path", "")),
                media_type=str(raw_record.get("media_type", "")),
                sha256=str(raw_record.get("sha256", "")),
                size_bytes=int(raw_record.get("size_bytes", -1)),
            )
        except (ArtifactError, TypeError, ValueError) as exc:
            errors.append("malformed artifact record: %s" % exc)
            continue
        relative = record.path
        listed_paths.append(relative)
        if relative == "manifest.json":
            errors.append("manifest.json cannot list itself as an artifact")
            continue
        candidate = root / relative
        try:
            if candidate.is_symlink():
                raise ValueError("artifact is a symlink")
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, ValueError):
            errors.append("artifact path is missing or escapes run: %s" % relative)
            continue
        if not resolved.is_file():
            errors.append("artifact is not a regular file: %s" % relative)
            continue
        data = resolved.read_bytes()
        if len(data) != record.size_bytes:
            errors.append("artifact size mismatch: %s" % relative)
        if hashlib.sha256(data).hexdigest() != record.sha256:
            errors.append("artifact hash mismatch: %s" % relative)
    if len(listed_paths) != len(set(listed_paths)):
        errors.append("manifest contains duplicate artifact paths")

    actual_paths = set()
    try:
        for candidate in root.rglob("*"):
            if candidate.is_symlink():
                errors.append(
                    "run directory contains a symlink: %s"
                    % candidate.relative_to(root).as_posix()
                )
                continue
            if candidate.is_file():
                relative = candidate.relative_to(root).as_posix()
                if relative != "manifest.json":
                    actual_paths.add(relative)
    except OSError as exc:
        errors.append("could not reconcile run directory: %s" % exc)
    for relative in sorted(actual_paths.difference(listed_paths)):
        errors.append("unlisted artifact present: %s" % relative)
    for relative in sorted(set(listed_paths).difference(actual_paths)):
        errors.append("listed artifact missing: %s" % relative)
    return tuple(errors)
