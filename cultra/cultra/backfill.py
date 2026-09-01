"""Checkpointed Cultra-only ORATS historical chain backfill.

The planner is tokenless and freezes every trade date before the gateway is
constructed.  Execution is an explicitly requested ``historical_backfill``
run.  Each successful date is immediately durable in Cultra's content-
addressed cache and can be reused without another provider request.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .cache import CULTRA_CACHE_ROOT, ContentAddressedCache
from .gateway import (
    CULTRA_ENV_PATH,
    EnvFileTokenSource,
    OratsGateway,
    UrllibTransport,
    execute_plan_via_local_daemon,
)
from .ledger import LedgerError, RequestLedger, account_ledger_path
from .requesting import RequestPlan, RunType


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VALIDATION_CONFIG = PROJECT_ROOT / "configs" / "historical_validation.v1.json"
DEFAULT_SESSION_CALENDAR = (
    PROJECT_ROOT
    / "out"
    / "cultra-backfill-sessions-2026-08-30-v1"
    / "session_calendar.json"
)

HISTORICAL_STRIKE_FIELDS = (
    "callAskPrice",
    "callBidIv",
    "callBidPrice",
    "callMidIv",
    "callOpenInterest",
    "callVolume",
    "delta",
    "dte",
    "expirDate",
    "gamma",
    "putAskPrice",
    "putBidIv",
    "putBidPrice",
    "putMidIv",
    "putOpenInterest",
    "putVolume",
    "rho",
    "smvVol",
    "stockPrice",
    "strike",
    "theta",
    "ticker",
    "tradeDate",
    "updatedAt",
    "vega",
)


class BackfillError(RuntimeError):
    """The frozen backfill could not be planned or completed safely."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _private_json(path: Path, value: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    temporary = path.with_name(".%s.tmp-%d" % (path.name, os.getpid()))
    data = json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    try:
        with open(temporary, "xb") as handle:
            os.chmod(temporary, 0o600)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return path


def _private_json_once(path: Path, value: Any) -> Path:
    """Create an immutable JSON artifact or verify its exact prior value."""

    destination = Path(path)
    if destination.exists():
        try:
            existing = json.loads(destination.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise BackfillError("existing checkpoint artifact is unreadable") from exc
        if existing != value:
            raise BackfillError("existing checkpoint identity has drifted")
        return destination
    return _private_json(destination, value)


def load_validation_config(path: Path = DEFAULT_VALIDATION_CONFIG) -> Mapping[str, Any]:
    supplied = Path(path).expanduser().resolve()
    if supplied != DEFAULT_VALIDATION_CONFIG.resolve():
        raise BackfillError("historical validation must use the frozen Cultra V1 config")
    try:
        value = json.loads(supplied.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BackfillError("frozen historical validation config is unavailable") from exc
    if value.get("schema") != "cultra.historical-validation.v1":
        raise BackfillError("historical validation config schema is not frozen V1")
    universe = value.get("universe")
    if not isinstance(universe, list) or len(universe) != 10:
        raise BackfillError("frozen historical universe must contain ten symbols")
    if universe != sorted(set(str(item) for item in universe)):
        raise BackfillError("frozen historical universe must be sorted and unique")
    data = value.get("data")
    if not isinstance(data, dict) or int(data.get("session_count", 0)) != 450:
        raise BackfillError("frozen historical session count must be 450")
    return value


def load_recent_sessions(
    path: Path = DEFAULT_SESSION_CALENDAR,
    *,
    required_count: int = 450,
) -> Tuple[str, ...]:
    try:
        value = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
        sessions = tuple(str(item) for item in value["recent_sessions"])
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise BackfillError("validated historical session calendar is unavailable") from exc
    if len(sessions) < required_count:
        raise BackfillError("historical session calendar is incomplete")
    selected = sessions[-required_count:]
    if selected != tuple(sorted(set(selected))):
        raise BackfillError("historical sessions must be sorted and unique")
    try:
        tuple(date.fromisoformat(item) for item in selected)
    except ValueError as exc:
        raise BackfillError("historical session calendar contains an invalid date") from exc
    return selected


def build_chain_backfill_plan(
    *,
    run_id: str,
    sessions: Sequence[str],
    slice_index: int,
    slice_size: int = 75,
    config: Mapping[str, Any],
) -> RequestPlan:
    """Reject the superseded one-full-chain-request-per-date design."""

    del run_id, sessions, slice_index, slice_size, config
    raise BackfillError(
        "date-grid backfill is disabled; use the staged request_optimization plans"
    )


def build_broad_cohort_backfill_plan(
    *,
    run_id: str,
    sessions: Sequence[str],
    symbols: Sequence[str],
    slice_index: int,
    sessions_per_slice: int = 45,
) -> RequestPlan:
    """Reject the superseded broad date-grid design."""

    del run_id, sessions, symbols, slice_index, sessions_per_slice
    raise BackfillError(
        "broad date-grid backfill is disabled; use staged request_optimization plans"
    )


@dataclass(frozen=True)
class BackfillResult:
    run_id: str
    run_dir: Path
    plan_hash: str
    completed_dates: Tuple[str, ...]
    failed_dates: Tuple[str, ...]
    cache_hits: int
    charged_attempts: int


def _partition_manifest_payload(
    plan: RequestPlan,
    completed: Mapping[str, Mapping[str, Any]],
    failed: Mapping[str, str],
    ledger_summary: Mapping[str, Any],
    campaign_proof: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    """Reconcile requests, not dates; broad cohorts have two batches/date."""

    planned_ids = {item.logical_request_id for item in plan.requests}
    if set(completed).intersection(failed):
        raise BackfillError("a backfill request is both completed and failed")
    if not set(completed).union(failed).issubset(planned_ids):
        raise BackfillError("backfill result contains an unplanned request")
    dates: Dict[str, Dict[str, Any]] = {}
    for request in plan.requests:
        row = dates.setdefault(
            request.expected_vintage,
            {"planned_request_ids": [], "completed_request_ids": [], "failed_request_ids": []},
        )
        row["planned_request_ids"].append(request.logical_request_id)
        if request.logical_request_id in completed:
            row["completed_request_ids"].append(request.logical_request_id)
        if request.logical_request_id in failed:
            row["failed_request_ids"].append(request.logical_request_id)
    for row in dates.values():
        row["complete"] = (
            len(row["completed_request_ids"]) == len(row["planned_request_ids"])
            and not row["failed_request_ids"]
        )
    return {
        "schema": "cultra.historical-chain-partitions.v2",
        "run_id": plan.run_id,
        "run_type": plan.run_type.value,
        "plan_hash": plan.plan_hash,
        "campaign_proof": None if campaign_proof is None else dict(campaign_proof),
        "legacy_validation_config_sha256": (
            hashlib.sha256(DEFAULT_VALIDATION_CONFIG.read_bytes()).hexdigest()
            if all(
                item.field_profile == "HIST_STRIKES_EXACT_V1"
                for item in plan.requests
            )
            else None
        ),
        "completed_requests": {key: completed[key] for key in sorted(completed)},
        "failed_requests": {key: failed[key] for key in sorted(failed)},
        "date_coverage": {key: dates[key] for key in sorted(dates)},
        "complete": (
            not failed
            and set(completed) == planned_ids
            and len(completed) == len(plan.requests)
        ),
        "ledger_summary": dict(ledger_summary),
    }


def execute_chain_backfill(
    plan: RequestPlan,
    *,
    output_root: Path,
    workers: int = 3,
    campaign_freeze_path: Path,
    slice_index: int,
) -> BackfillResult:
    """Execute one slice only after reproducing its immutable V2 campaign."""

    if plan.run_type is not RunType.HISTORICAL_BACKFILL:
        raise BackfillError("only a historical_backfill plan can use this executor")
    # Local import avoids a module cycle: campaign owns the freeze and imports
    # this module's pure session-calendar loader.  The execution boundary must
    # nevertheless reproduce that freeze before it can construct a gateway.
    from .campaign import load_historical_campaign_freeze

    campaign = load_historical_campaign_freeze(campaign_freeze_path)
    if (
        isinstance(slice_index, bool)
        or not isinstance(slice_index, int)
        or not 0 <= slice_index < len(campaign.slices)
    ):
        raise BackfillError("campaign slice index is invalid")
    frozen_plan = campaign.slices[slice_index]
    if frozen_plan.to_dict() != plan.to_dict():
        raise BackfillError("execution plan does not match the immutable campaign slice")
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 4:
        raise BackfillError("workers must be between 1 and 4")
    root = Path(output_root).expanduser().resolve()
    allowed = (PROJECT_ROOT / "out").resolve()
    try:
        root.relative_to(allowed)
    except ValueError as exc:
        raise BackfillError("backfill output must remain inside Cultra/out") from exc
    run_dir = root / plan.run_id
    if (run_dir / "manifest.json").exists():
        raise BackfillError("completed historical slice is immutable")
    run_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(run_dir, 0o700)
    campaign_proof = {
        "schema": "cultra.historical-campaign-execution-proof.v1",
        "campaign_id": campaign.campaign_id,
        "campaign_freeze_path": str(Path(campaign_freeze_path).expanduser().resolve()),
        "campaign_freeze_hash": campaign.payload["freeze_hash"],
        "campaign_receipt_hash": campaign.payload["receipt"]["receipt_hash"],
        "slice_index": slice_index,
        "slice_plan_hash": plan.plan_hash,
    }
    _private_json_once(run_dir / "frozen_request_plan.json", plan.to_dict())
    _private_json_once(
        run_dir / "campaign_execution_proof.json", campaign_proof
    )
    executions_root = run_dir / "executions"
    executions_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(executions_root, 0o700)
    prior_executions = tuple(sorted(executions_root.glob("execution-*")))
    execution_dir = executions_root / ("execution-%03d" % len(prior_executions))
    execution_dir.mkdir(mode=0o700)
    os.chmod(execution_dir, 0o700)
    _private_json_once(execution_dir / "orats_request_plan.json", plan.to_dict())
    _private_json_once(
        execution_dir / "campaign_execution_proof.json", campaign_proof
    )

    ledger = RequestLedger(account_ledger_path())
    try:
        prior_summary = ledger.summary(plan.run_id)
    except LedgerError:
        prior_summary = None
    if prior_summary is not None:
        if prior_summary["state"] == "aborted":
            ledger.reactivate_aborted_run(plan)
        elif prior_summary["state"] != "active":
            raise BackfillError("historical slice ledger is already terminal")
    cache = ContentAddressedCache(CULTRA_CACHE_ROOT / "historical")
    gateway = OratsGateway(
        plan=plan,
        ledger=ledger,
        cache=cache,
        token_source=EnvFileTokenSource(CULTRA_ENV_PATH),
        transport=UrllibTransport(timeout_seconds=90.0),
    )
    completed: Dict[str, Dict[str, Any]] = {}
    failed: Dict[str, str] = {}
    aborted = True
    try:
        results, failed = execute_plan_via_local_daemon(
            gateway,
            tuple(item.logical_request_id for item in plan.requests),
            socket_path=execution_dir / "orats-gateway.sock",
            workers=workers,
            client_timeout_seconds=120.0,
        )
        for request in plan.requests:
            result = results.get(request.logical_request_id)
            if result is None:
                continue
            trade_date = request.expected_vintage
            completed[request.logical_request_id] = {
                "logical_request_id": request.logical_request_id,
                "trade_date": trade_date,
                "entities": list(request.entities),
                "request_fingerprint": request.fingerprint,
                "snapshot_id": result.manifest.snapshot_id,
                "raw_sha256": result.manifest.raw_sha256,
                "row_count": result.manifest.row_count,
                "raw_bytes": result.manifest.raw_bytes,
                "cache_hit": result.cache_hit,
                "charged_attempts": result.charged_attempts,
                "provider_trade_dates": list(result.manifest.provider_trade_dates),
                "returned_entities": list(result.manifest.returned_entities),
                "missing_entities": list(result.manifest.missing_entities),
            }
            print(
                "BACKFILL %s rows=%d bytes=%d cache=%s"
                % (
                    trade_date,
                    result.manifest.row_count,
                    result.manifest.raw_bytes,
                    str(result.cache_hit).lower(),
                ),
                flush=True,
            )
        for request_id in sorted(failed):
            request = plan.get(request_id)
            print(
                "BACKFILL %s %s FAILED"
                % (request.expected_vintage, request.logical_request_id),
                flush=True,
            )
        aborted = bool(failed)
    finally:
        ledger.finish_run(plan.run_id, aborted=aborted)
        ledger_path = ledger.export(
            plan.run_id, execution_dir / "orats_request_ledger.json"
        )

    summary = ledger.summary(plan.run_id)
    manifest = _partition_manifest_payload(
        plan, completed, failed, summary, campaign_proof=campaign_proof
    )
    partition_path = _private_json(
        execution_dir / "partition_manifest.json", manifest
    )
    cache_report_path = _private_json(
        execution_dir / "cache_report.json",
        {
            "planned": len(plan.requests),
            "cache_hits": sum(bool(item["cache_hit"]) for item in completed.values()),
            "network_misses": sum(
                not bool(item["cache_hit"]) for item in completed.values()
            ),
            "failed": len(failed),
        },
    )
    artifact_paths = (
        execution_dir / "orats_request_plan.json",
        execution_dir / "campaign_execution_proof.json",
        ledger_path,
        partition_path,
        cache_report_path,
    )
    _private_json(
        execution_dir / "manifest.json",
        {
            "schema": "cultra.historical-slice-execution-manifest.v1",
            "run_id": plan.run_id,
            "campaign_id": campaign.campaign_id,
            "campaign_freeze_hash": campaign.payload["freeze_hash"],
            "slice_index": slice_index,
            "plan_hash": plan.plan_hash,
            "network_attempted": bool(summary["charged_attempts"]),
            "complete": bool(manifest["complete"]),
            "artifacts": [
                {
                    "path": item.name,
                    "bytes": item.stat().st_size,
                    "sha256": _sha256(item),
                }
                for item in artifact_paths
            ],
        },
    )
    if manifest["complete"]:
        canonical_plan = _private_json_once(
            run_dir / "orats_request_plan.json", plan.to_dict()
        )
        canonical_ledger = ledger.export(
            plan.run_id, run_dir / "orats_request_ledger.json"
        )
        canonical_partition = _private_json_once(
            run_dir / "partition_manifest.json", manifest
        )
        canonical_cache = _private_json_once(
            run_dir / "cache_report.json",
            {
                "planned": len(plan.requests),
                "cache_hits": sum(
                    bool(item["cache_hit"]) for item in completed.values()
                ),
                "network_misses": sum(
                    not bool(item["cache_hit"]) for item in completed.values()
                ),
                "failed": 0,
                "execution_count": len(prior_executions) + 1,
            },
        )
        canonical_artifacts = (
            canonical_plan,
            run_dir / "campaign_execution_proof.json",
            canonical_ledger,
            canonical_partition,
            canonical_cache,
        )
        _private_json_once(
            run_dir / "manifest.json",
            {
                "schema": "cultra.historical-slice-run-manifest.v1",
                "run_id": plan.run_id,
                "campaign_id": campaign.campaign_id,
                "campaign_freeze_hash": campaign.payload["freeze_hash"],
                "slice_index": slice_index,
                "plan_hash": plan.plan_hash,
                "network_attempted": bool(summary["charged_attempts"]),
                "complete": True,
                "execution_count": len(prior_executions) + 1,
                "artifacts": [
                    {
                        "path": item.name,
                        "bytes": item.stat().st_size,
                        "sha256": _sha256(item),
                    }
                    for item in canonical_artifacts
                ],
            },
        )
    return BackfillResult(
        run_id=plan.run_id,
        run_dir=run_dir,
        plan_hash=plan.plan_hash,
        completed_dates=tuple(
            sorted(
                {
                    str(item["trade_date"])
                    for item in completed.values()
                }
            )
        ),
        failed_dates=tuple(
            sorted(
                {
                    request.expected_vintage
                    for request in plan.requests
                    if request.logical_request_id in failed
                }
            )
        ),
        cache_hits=sum(bool(item["cache_hit"]) for item in completed.values()),
        charged_attempts=int(summary["charged_attempts"]),
    )


def plan_chain_slices(
    sessions: Sequence[str], *, slice_size: int = 75
) -> Tuple[Tuple[str, ...], ...]:
    if not 2 <= slice_size <= 90:
        raise BackfillError("slice_size must be between 2 and 90")
    return tuple(
        tuple(sessions[index : index + slice_size])
        for index in range(0, len(sessions), slice_size)
    )


__all__ = [
    "BackfillError",
    "BackfillResult",
    "DEFAULT_SESSION_CALENDAR",
    "DEFAULT_VALIDATION_CONFIG",
    "HISTORICAL_STRIKE_FIELDS",
    "build_chain_backfill_plan",
    "build_broad_cohort_backfill_plan",
    "execute_chain_backfill",
    "_partition_manifest_payload",
    "load_recent_sessions",
    "load_validation_config",
    "plan_chain_slices",
]
