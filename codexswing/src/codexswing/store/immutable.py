"""Content-addressed, write-once JSON storage."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from codexswing.schemas.source import SourceRecord, canonical_json
from codexswing.secrets import find_secret_leaks


class ImmutableWriteConflict(RuntimeError):
    pass


class SecretLeakError(RuntimeError):
    pass


@dataclass(frozen=True)
class StoreAudit:
    root: Path
    record_count: int
    batch_count: int
    manifest_count: int
    trial_count: int
    source_counts: Mapping[str, int]
    errors: Sequence[str]

    @property
    def valid(self) -> bool:
        return self.record_count > 0 and not self.errors

    def public_dict(self) -> Dict[str, Any]:
        return {
            "root": str(self.root),
            "valid": self.valid,
            "record_count": self.record_count,
            "batch_count": self.batch_count,
            "manifest_count": self.manifest_count,
            "trial_count": self.trial_count,
            "source_counts": dict(sorted(self.source_counts.items())),
            "errors": list(self.errors),
        }


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def write_once_bytes(path: Path, content: bytes, secret_values: Iterable[str] = ()) -> Path:
    text = content.decode("utf-8", errors="ignore")
    if find_secret_leaks(text, secret_values):
        raise SecretLeakError("refusing to write content containing a configured secret")

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        existing = path.read_bytes()
        if existing != content:
            raise ImmutableWriteConflict("write-once artifact already exists with different content: {}".format(path))
        return path

    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise
    return path


def write_once_json(path: Path, payload: Mapping[str, Any], secret_values: Iterable[str] = ()) -> Path:
    rendered = (canonical_json(payload) + "\n").encode("utf-8")
    return write_once_bytes(path, rendered, secret_values=secret_values)


class ContentAddressedStore:
    def __init__(self, root: Path, secret_values: Iterable[str] = ()) -> None:
        self.root = root.expanduser().resolve()
        self.secret_values = tuple(secret_values)

    def put(self, record: SourceRecord) -> Path:
        if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", record.source):
            raise ValueError("unsafe source identifier")
        destination = (
            self.root
            / "records"
            / record.source
            / record.session_date
            / "{}.json".format(record.content_hash)
        )
        write_once_json(destination, record.to_dict(), secret_values=self.secret_values)
        return destination

    def put_batch(self, records: Iterable[SourceRecord]) -> "BatchArtifact":
        materialized = tuple(records)
        if not materialized:
            raise ValueError("record batch cannot be empty")
        sources = {record.source for record in materialized}
        session_dates = {record.session_date for record in materialized}
        if len(sources) != 1:
            raise ValueError("record batch must contain one source")
        source = next(iter(sources))
        session_scope = next(iter(session_dates)) if len(session_dates) == 1 else "_multi_session"
        lines = sorted(canonical_json(record.to_dict()) for record in materialized)
        raw = ("\n".join(lines) + "\n").encode("utf-8")
        if find_secret_leaks(raw.decode("utf-8"), self.secret_values):
            raise SecretLeakError("refusing to write batch containing a configured secret")
        batch_hash = hashlib.sha256(raw).hexdigest()
        buffer = io.BytesIO()
        with gzip.GzipFile(fileobj=buffer, mode="wb", filename="", mtime=0) as handle:
            handle.write(raw)
        destination = self.root / "batches" / source / session_scope / "{}.jsonl.gz".format(batch_hash)
        write_once_bytes(destination, buffer.getvalue())
        return BatchArtifact(
            path=destination,
            batch_hash=batch_hash,
            source=source,
            session_scope=session_scope,
            record_count=len(materialized),
        )


@dataclass(frozen=True)
class BatchArtifact:
    path: Path
    batch_hash: str
    source: str
    session_scope: str
    record_count: int


def read_batch(path: Path) -> Tuple[SourceRecord, ...]:
    """Read and verify a deterministic CodexSwing JSONL batch."""

    resolved = path.expanduser().resolve()
    with gzip.open(str(resolved), mode="rb") as handle:
        raw = handle.read()
    if hashlib.sha256(raw).hexdigest() != resolved.name[:-9]:
        raise ValueError("batch filename does not equal uncompressed content hash")
    records = tuple(
        SourceRecord.from_dict(json.loads(line))
        for line in raw.decode("utf-8").splitlines()
        if line
    )
    if not records:
        raise ValueError("batch is empty")
    return records


def _path_within(root: Path, path: Path) -> bool:
    try:
        return os.path.commonpath([str(root), str(path)]) == str(root)
    except ValueError:
        return False


def audit_store(root: Path, secret_values: Iterable[str] = ()) -> StoreAudit:
    resolved_root = root.expanduser().resolve()
    errors: List[str] = []
    source_counts: Dict[str, int] = {}
    record_hashes = set()
    record_files = sorted((resolved_root / "records").glob("*/*/*.json"))
    secret_tuple = tuple(secret_values)

    for path in record_files:
        try:
            text = path.read_text(encoding="utf-8")
            if find_secret_leaks(text, secret_tuple):
                raise SecretLeakError("configured secret found")
            payload = json.loads(text)
            record = SourceRecord.from_dict(payload)
            if path.stem != record.content_hash:
                raise ValueError("filename does not equal content hash")
            if path.parent.name != record.session_date or path.parent.parent.name != record.source:
                raise ValueError("record path does not match source/session")
            record_hashes.add(record.content_hash)
            source_counts[record.source] = source_counts.get(record.source, 0) + 1
        except Exception as exc:
            errors.append("record {}: {}".format(path, str(exc)))

    batch_files = sorted((resolved_root / "batches").glob("*/*/*.jsonl.gz"))
    batch_record_count = 0
    for path in batch_files:
        try:
            with gzip.open(str(path), mode="rb") as handle:
                raw = handle.read()
            text = raw.decode("utf-8")
            if find_secret_leaks(text, secret_tuple):
                raise SecretLeakError("configured secret found")
            if hashlib.sha256(raw).hexdigest() != path.name[:-9]:
                raise ValueError("batch filename does not equal uncompressed content hash")
            lines = [line for line in text.splitlines() if line]
            if not lines:
                raise ValueError("batch is empty")
            for line in lines:
                record = SourceRecord.from_dict(json.loads(line))
                session_matches = (
                    path.parent.name == "_multi_session" or path.parent.name == record.session_date
                )
                if not session_matches or path.parent.parent.name != record.source:
                    raise ValueError("batch path does not match record source/session")
                record_hashes.add(record.content_hash)
                source_counts[record.source] = source_counts.get(record.source, 0) + 1
                batch_record_count += 1
        except Exception as exc:
            errors.append("batch {}: {}".format(path, str(exc)))

    manifest_files = sorted((resolved_root / "runs").glob("*/manifest.json"))
    for path in manifest_files:
        try:
            text = path.read_text(encoding="utf-8")
            if find_secret_leaks(text, secret_tuple):
                raise SecretLeakError("configured secret found")
            manifest = json.loads(text)
            schema_version = manifest.get("schema_version")
            if schema_version not in {
                "codexswing.run_manifest.v1",
                "codexswing.run_manifest.v2",
            }:
                raise ValueError("unsupported manifest schema")
            if manifest.get("status") != "RESEARCH_ONLY":
                raise ValueError("manifest posture is not RESEARCH_ONLY")
            if manifest.get("run_id") != path.parent.name:
                raise ValueError("manifest run_id does not match directory")
            missing_hashes = sorted(set(manifest.get("input_record_hashes", ())) - record_hashes)
            if missing_hashes:
                raise ValueError("manifest references {} missing record hashes".format(len(missing_hashes)))
            for raw_input, expected_hash in manifest.get("input_file_hashes", {}).items():
                input_path = Path(raw_input).expanduser().resolve()
                if not input_path.is_file():
                    raise ValueError("manifest input file is unavailable")
                if sha256_file(input_path) != expected_hash:
                    raise ValueError("manifest input file hash mismatch")
            for raw_output in manifest.get("output_paths", ()): 
                output_path = Path(raw_output).expanduser().resolve()
                if not _path_within(resolved_root, output_path):
                    raise ValueError("manifest output is outside CodexSwing root")
                if not output_path.is_file():
                    raise ValueError("manifest output is unavailable")
            if schema_version == "codexswing.run_manifest.v2":
                expected_configuration_hash = hashlib.sha256(
                    canonical_json(manifest.get("configuration", {})).encode("utf-8")
                ).hexdigest()
                if manifest.get("configuration_sha256") != expected_configuration_hash:
                    raise ValueError("manifest configuration hash mismatch")
                code_hash = str(manifest.get("code_tree_sha256") or "")
                if not re.fullmatch(r"[0-9a-f]{64}", code_hash):
                    raise ValueError("manifest code tree hash is invalid")
                output_hashes = manifest.get("output_file_hashes")
                if not isinstance(output_hashes, Mapping):
                    raise ValueError("manifest output hashes are missing")
                if set(output_hashes) != set(manifest.get("output_paths", ())):
                    raise ValueError("manifest output hash paths do not match outputs")
                for raw_output, expected_hash in output_hashes.items():
                    output_path = Path(raw_output).expanduser().resolve()
                    if sha256_file(output_path) != expected_hash:
                        raise ValueError("manifest output file hash mismatch")
        except Exception as exc:
            errors.append("manifest {}: {}".format(path, str(exc)))

    trial_files = sorted((resolved_root / "trials").glob("*/*/declaration.json"))
    for path in trial_files:
        try:
            text = path.read_text(encoding="utf-8")
            if find_secret_leaks(text, secret_tuple):
                raise SecretLeakError("configured secret found")
            payload = json.loads(text)
            expected_hash = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
            if path.parent.name != expected_hash:
                raise ValueError("trial directory does not equal declaration hash")
            if path.parent.parent.name != payload.get("hypothesis_id"):
                raise ValueError("trial path does not match hypothesis_id")
            if payload.get("schema_version") != "codexswing.trial_declaration.v1":
                raise ValueError("unsupported trial declaration schema")
            if payload.get("posture") not in {
                "PROSPECTIVE_SHADOW_RESEARCH",
                "EXPLORATORY_ONLY",
            }:
                raise ValueError("invalid trial posture")
        except Exception as exc:
            errors.append("trial {}: {}".format(path, str(exc)))

    return StoreAudit(
        root=resolved_root,
        record_count=len(record_files) + batch_record_count,
        batch_count=len(batch_files),
        manifest_count=len(manifest_files),
        trial_count=len(trial_files),
        source_counts=source_counts,
        errors=tuple(errors[:100]),
    )
