"""Offline reconciliation of every immutable Cultra historical slice."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .cache import CULTRA_CACHE_ROOT, CacheError, ContentAddressedCache
from .campaign import load_historical_campaign_freeze
from .requesting import Endpoint


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = (PROJECT_ROOT / "out").resolve()


class CampaignCompletionError(RuntimeError):
    """A saved campaign is incomplete, corrupt, or not reproducible."""


def _load_object(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CampaignCompletionError("%s is unavailable" % label) from exc
    if not isinstance(value, Mapping):
        raise CampaignCompletionError("%s is malformed" % label)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _owned_directory(path: Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    try:
        resolved.relative_to(OUT_ROOT)
    except ValueError as exc:
        raise CampaignCompletionError("%s must remain inside Cultra/out" % label) from exc
    if not resolved.is_dir():
        raise CampaignCompletionError("%s is unavailable" % label)
    return resolved


def _verify_run_manifest(run_dir: Path, expected: Mapping[str, Any]) -> None:
    manifest = _load_object(run_dir / "manifest.json", "historical slice run manifest")
    if manifest.get("schema") != "cultra.historical-slice-run-manifest.v1":
        raise CampaignCompletionError("historical slice run manifest schema is unsupported")
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise CampaignCompletionError("historical slice run identity drifted: %s" % key)
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise CampaignCompletionError("historical slice artifact manifest is missing")
    expected_names = {
        "orats_request_plan.json",
        "campaign_execution_proof.json",
        "orats_request_ledger.json",
        "partition_manifest.json",
        "cache_report.json",
    }
    names = {str(item.get("path", "")) for item in artifacts if isinstance(item, Mapping)}
    if names != expected_names or len(artifacts) != len(expected_names):
        raise CampaignCompletionError("historical slice artifact set is incomplete")
    for item in artifacts:
        if not isinstance(item, Mapping):
            raise CampaignCompletionError("historical slice artifact record is malformed")
        artifact = run_dir / str(item["path"])
        if (
            not artifact.is_file()
            or artifact.stat().st_size != int(item["bytes"])
            or _sha256(artifact) != str(item["sha256"])
        ):
            raise CampaignCompletionError(
                "historical slice artifact failed integrity verification"
            )


def verify_historical_campaign_completion(
    *,
    campaign_freeze_path: Path,
    runs_root: Path,
    cache_root: Optional[Path] = None,
) -> Mapping[str, Any]:
    """Verify all 474 request results and their immutable cached snapshots."""

    campaign = load_historical_campaign_freeze(campaign_freeze_path)
    root = _owned_directory(runs_root, "historical slice root")
    cache = ContentAddressedCache(
        Path(cache_root) if cache_root is not None else CULTRA_CACHE_ROOT / "historical"
    )
    completed_ids = set()
    snapshot_ids = set()
    charged_attempts = 0
    cache_hits = 0
    slice_rows = []
    for index, plan in enumerate(campaign.slices):
        run_dir = root / plan.run_id
        if run_dir.resolve().parent != root:
            raise CampaignCompletionError("historical slice run path is invalid")
        _verify_run_manifest(
            run_dir,
            {
                "run_id": plan.run_id,
                "campaign_id": campaign.campaign_id,
                "campaign_freeze_hash": campaign.payload["freeze_hash"],
                "slice_index": index,
                "plan_hash": plan.plan_hash,
                "complete": True,
            },
        )
        proof = _load_object(
            run_dir / "campaign_execution_proof.json", "campaign execution proof"
        )
        if (
            proof.get("campaign_id") != campaign.campaign_id
            or proof.get("campaign_freeze_hash") != campaign.payload["freeze_hash"]
            or proof.get("campaign_receipt_hash")
            != campaign.payload["receipt"]["receipt_hash"]
            or proof.get("slice_index") != index
            or proof.get("slice_plan_hash") != plan.plan_hash
        ):
            raise CampaignCompletionError("campaign execution proof does not reconcile")
        partition = _load_object(
            run_dir / "partition_manifest.json", "historical partition manifest"
        )
        if (
            partition.get("schema") != "cultra.historical-chain-partitions.v2"
            or partition.get("run_id") != plan.run_id
            or partition.get("plan_hash") != plan.plan_hash
            or partition.get("complete") is not True
            or partition.get("campaign_proof") != proof
            or partition.get("failed_requests") != {}
        ):
            raise CampaignCompletionError("historical partition did not complete exactly")
        completed = partition.get("completed_requests")
        if not isinstance(completed, Mapping) or set(completed) != {
            item.logical_request_id for item in plan.requests
        }:
            raise CampaignCompletionError("historical partition request coverage is incomplete")
        ledger_summary = partition.get("ledger_summary")
        if not isinstance(ledger_summary, Mapping):
            raise CampaignCompletionError("historical partition ledger summary is missing")
        slice_charged = int(ledger_summary.get("charged_attempts", -1))
        if not 0 <= slice_charged <= plan.logical_count:
            raise CampaignCompletionError("historical slice charged-attempt count is invalid")
        charged_attempts += slice_charged
        for request in plan.requests:
            row = completed[request.logical_request_id]
            if not isinstance(row, Mapping):
                raise CampaignCompletionError("historical completion row is malformed")
            if (
                row.get("logical_request_id") != request.logical_request_id
                or row.get("trade_date") != request.expected_vintage
                or tuple(row.get("entities", ())) != request.entities
                or row.get("request_fingerprint") != request.fingerprint
            ):
                raise CampaignCompletionError("historical completion row identity drifted")
            try:
                manifest, _raw = cache.load_snapshot(str(row.get("snapshot_id", "")))
            except CacheError as exc:
                raise CampaignCompletionError(
                    "historical cached snapshot failed verification"
                ) from exc
            if (
                manifest.request_fingerprint != request.fingerprint
                or manifest.endpoint != request.endpoint.value
                or manifest.field_profile != request.field_profile
                or manifest.expected_trade_date != request.expected_vintage
                or manifest.requested_entities != request.entities
                or manifest.raw_sha256 != row.get("raw_sha256")
                or manifest.raw_bytes != row.get("raw_bytes")
                or manifest.row_count != row.get("row_count")
            ):
                raise CampaignCompletionError("historical snapshot provenance drifted")
            if request.endpoint is not Endpoint.HIST_SPLITS and manifest.missing_entities:
                raise CampaignCompletionError("historical snapshot is missing required entities")
            if manifest.snapshot_id in snapshot_ids:
                raise CampaignCompletionError("one snapshot was reused by distinct requests")
            snapshot_ids.add(manifest.snapshot_id)
            completed_ids.add(request.logical_request_id)
            cache_hits += int(bool(row.get("cache_hit")))
        slice_rows.append(
            {
                "slice_index": index,
                "run_id": plan.run_id,
                "plan_hash": plan.plan_hash,
                "requests": plan.logical_count,
                "charged_attempts": slice_charged,
            }
        )
    expected_ids = {
        request.logical_request_id
        for plan in campaign.slices
        for request in plan.requests
    }
    if completed_ids != expected_ids:
        raise CampaignCompletionError("historical campaign request set is incomplete")
    return {
        "schema": "cultra.historical-campaign-completion.v1",
        "campaign_id": campaign.campaign_id,
        "campaign_freeze_path": str(Path(campaign_freeze_path).expanduser().resolve()),
        "runs_root": str(root),
        "campaign_freeze_hash": campaign.payload["freeze_hash"],
        "campaign_receipt_hash": campaign.payload["receipt"]["receipt_hash"],
        "expected_requests": len(expected_ids),
        "completed_requests": len(completed_ids),
        "verified_snapshots": len(snapshot_ids),
        "charged_attempts": charged_attempts,
        "cache_hits": cache_hits,
        "slices": slice_rows,
        "complete": True,
        "verification_network_attempted": False,
    }


def save_historical_campaign_completion(
    output_dir: Path, completion: Mapping[str, Any]
) -> Path:
    root = Path(output_dir).expanduser().resolve()
    try:
        root.relative_to(OUT_ROOT)
    except ValueError as exc:
        raise CampaignCompletionError("campaign completion must remain inside Cultra/out") from exc
    if root.exists():
        raise CampaignCompletionError("campaign completion directory already exists")
    root.mkdir(parents=True, mode=0o700)
    os.chmod(root, 0o700)
    artifact = root / "campaign_completion.json"
    encoded = json.dumps(completion, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    with open(artifact, "xb") as handle:
        os.chmod(artifact, 0o600)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    manifest = root / "manifest.json"
    manifest_payload = {
        "schema": "cultra.historical-campaign-completion-manifest.v1",
        "campaign_id": completion["campaign_id"],
        "network_attempted": False,
        "artifacts": [
            {
                "path": artifact.name,
                "bytes": artifact.stat().st_size,
                "sha256": _sha256(artifact),
            }
        ],
    }
    with open(manifest, "xb") as handle:
        os.chmod(manifest, 0o600)
        handle.write(
            json.dumps(manifest_payload, indent=2, sort_keys=True).encode("utf-8")
            + b"\n"
        )
        handle.flush()
        os.fsync(handle.fileno())
    return artifact


def load_historical_campaign_completion(
    path: Path, *, cache_root: Optional[Path] = None
) -> Mapping[str, Any]:
    """Load a saved receipt and reproduce its complete offline verification."""

    supplied = Path(path).expanduser().resolve()
    try:
        supplied.relative_to(OUT_ROOT)
    except ValueError as exc:
        raise CampaignCompletionError(
            "campaign completion must remain inside Cultra/out"
        ) from exc
    value = _load_object(supplied, "historical campaign completion")
    if value.get("schema") != "cultra.historical-campaign-completion.v1":
        raise CampaignCompletionError("historical campaign completion schema is unsupported")
    envelope = _load_object(supplied.parent / "manifest.json", "campaign completion manifest")
    artifacts = envelope.get("artifacts")
    if (
        envelope.get("schema")
        != "cultra.historical-campaign-completion-manifest.v1"
        or not isinstance(artifacts, list)
        or len(artifacts) != 1
        or artifacts[0].get("path") != supplied.name
        or int(artifacts[0].get("bytes", -1)) != supplied.stat().st_size
        or artifacts[0].get("sha256") != _sha256(supplied)
    ):
        raise CampaignCompletionError("campaign completion manifest does not reconcile")
    reproduced = verify_historical_campaign_completion(
        campaign_freeze_path=Path(str(value.get("campaign_freeze_path", ""))),
        runs_root=Path(str(value.get("runs_root", ""))),
        cache_root=cache_root,
    )
    if reproduced != value:
        raise CampaignCompletionError("campaign completion cannot be reproduced")
    return value


__all__ = [
    "CampaignCompletionError",
    "load_historical_campaign_completion",
    "save_historical_campaign_completion",
    "verify_historical_campaign_completion",
]
