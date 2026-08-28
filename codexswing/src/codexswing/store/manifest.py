"""Immutable run manifest with explicit research posture."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from codexswing import __version__
from codexswing.clock import iso_utc, utc_now
from codexswing.schemas.source import SourceRecord, canonical_json
from codexswing.store.immutable import sha256_file, write_once_json


MANIFEST_VERSION = "codexswing.run_manifest.v2"


def code_tree_sha256() -> str:
    """Fingerprint the exact Python source tree used to create a run."""

    package_root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(package_root.rglob("*.py")):
        relative = path.relative_to(package_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    created_at_utc: str
    mode: str
    configuration: Mapping[str, Any]
    configuration_sha256: str
    code_tree_sha256: str
    input_record_hashes: Sequence[str]
    input_file_hashes: Mapping[str, str]
    output_paths: Sequence[str]
    output_file_hashes: Mapping[str, str]
    warnings: Sequence[str] = field(default_factory=tuple)
    status: str = "RESEARCH_ONLY"
    schema_version: str = MANIFEST_VERSION
    code_version: str = __version__

    @classmethod
    def create(
        cls,
        mode: str,
        configuration: Mapping[str, Any],
        input_records: Iterable[SourceRecord],
        output_paths: Iterable[Path],
        warnings: Iterable[str] = (),
        input_file_hashes: Optional[Mapping[str, str]] = None,
    ) -> "RunManifest":
        if not mode or not mode.replace("_", "").isalnum():
            raise ValueError("mode must be an alphanumeric identifier")
        configuration_copy = dict(configuration)
        materialized_outputs = tuple(sorted(Path(path).expanduser().resolve() for path in output_paths))
        for path in materialized_outputs:
            if not path.is_file():
                raise ValueError("manifest output does not exist: {}".format(path))
        return cls(
            run_id=uuid.uuid4().hex,
            created_at_utc=iso_utc(utc_now()),
            mode=mode,
            configuration=configuration_copy,
            configuration_sha256=hashlib.sha256(
                canonical_json(configuration_copy).encode("utf-8")
            ).hexdigest(),
            code_tree_sha256=code_tree_sha256(),
            input_record_hashes=tuple(sorted(record.content_hash for record in input_records)),
            input_file_hashes=dict(sorted((input_file_hashes or {}).items())),
            output_paths=tuple(str(path) for path in materialized_outputs),
            output_file_hashes={str(path): sha256_file(path) for path in materialized_outputs},
            warnings=tuple(warnings),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "created_at_utc": self.created_at_utc,
            "mode": self.mode,
            "status": self.status,
            "code_version": self.code_version,
            "configuration": dict(self.configuration),
            "configuration_sha256": self.configuration_sha256,
            "code_tree_sha256": self.code_tree_sha256,
            "input_record_hashes": list(self.input_record_hashes),
            "input_file_hashes": dict(self.input_file_hashes),
            "output_paths": list(self.output_paths),
            "output_file_hashes": dict(self.output_file_hashes),
            "warnings": list(self.warnings),
        }

    def write(self, output_root: Path, secret_values: Iterable[str] = ()) -> Path:
        destination = output_root.expanduser().resolve() / "runs" / self.run_id / "manifest.json"
        return write_once_json(destination, self.to_dict(), secret_values=secret_values)
