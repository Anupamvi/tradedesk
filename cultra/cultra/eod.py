"""Execute a frozen, bounded Cultra ORATS EOD plan and save its evidence."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
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
from .ledger import RequestLedger, account_ledger_path
from .requesting import RequestPlan, RunType, build_reference_eod_plan


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_FIELDS = (
    "confidence",
    "contango",
    "iv30d",
    "orFcst20d",
    "orHv20d",
    "pxAtmIv",
    "slope",
    "ticker",
    "tradeDate",
    "updatedAt",
)
CORE_SCREEN_FIELDS = (
    "absAvgErnMv",
    "assetType",
    "atmFcstIvM1",
    "atmFcstIvM2",
    "atmFcstIvM3",
    "atmFcstIvM4",
    "atmIvM1",
    "atmIvM2",
    "atmIvM3",
    "atmIvM4",
    "beta1m",
    "beta1y",
    "cOi",
    "cVolu",
    "confidence",
    "contango",
    "correlSpy1m",
    "correlSpy1y",
    "deriv",
    "derivFcst",
    "dtExM1",
    "dtExM2",
    "dtExM3",
    "dtExM4",
    "ernMvStdv",
    "exErnIv20d",
    "exErnIv30d",
    "fcstErnEffct",
    "fcstR2",
    "fcstR2Imp",
    "impErnMv",
    "iv20d",
    "iv30d",
    "iv60d",
    "iv200Ma",
    "ivHvXernRatio",
    "ivHvXernRatio1m",
    "ivHvXernRatio1y",
    "ivHvXernRatioStdv1y",
    "ivPctile1m",
    "ivPctile1y",
    "ivStdvMean",
    "mktCap",
    "orFcst20d",
    "orHvXern5d",
    "orHvXern10d",
    "orHvXern20d",
    "orHvXern60d",
    "orIvFcst20d",
    "orIvXern20d",
    "orIvXernInf",
    "pOi",
    "pVolu",
    "priorCls",
    "pxAtmIv",
    "slope",
    "slopeFcst",
    "slopepctile",
    "ticker",
    "tradeDate",
    "updatedAt",
    "wksNextErn",
)


class EodError(RuntimeError):
    """A bounded EOD enrichment could not be completed or reconciled."""


def _private_json(path: Path, value: Any) -> Path:
    resolved = Path(path).expanduser().resolve()
    try:
        resolved.relative_to((PROJECT_ROOT / "out").resolve())
    except ValueError as exc:
        raise EodError("EOD artifacts must remain inside Cultra/out") from exc
    resolved.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(resolved.parent, 0o700)
    temporary = resolved.with_name(".%s.tmp-%d" % (resolved.name, os.getpid()))
    data = json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    try:
        with open(temporary, "xb") as handle:
            os.chmod(temporary, 0o600)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, resolved)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return resolved


def _records(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, list):
        for item in value:
            if isinstance(item, Mapping):
                yield item
        return
    if isinstance(value, Mapping):
        for key in ("data", "rows", "results"):
            nested = value.get(key)
            if isinstance(nested, list):
                for item in nested:
                    if isinstance(item, Mapping):
                        yield item
                return
        if value and all(not isinstance(item, (list, Mapping)) for item in value.values()):
            yield value


def decode_rows(raw: bytes) -> Tuple[Mapping[str, Any], ...]:
    try:
        parsed = json.loads(raw.decode("utf-8"))
        rows = tuple(dict(item) for item in _records(parsed))
    except (UnicodeError, json.JSONDecodeError):
        try:
            rows = tuple(dict(item) for item in csv.DictReader(io.StringIO(raw.decode("utf-8"))))
        except (UnicodeError, csv.Error) as exc:
            raise EodError("ORATS response cannot be normalized") from exc
    if not rows:
        raise EodError("ORATS response contained no rows")
    return rows


def build_core_plan(
    *,
    run_id: str,
    symbols: Sequence[str],
    expected_vintage: str,
    fields: Sequence[str] = CORE_FIELDS,
) -> RequestPlan:
    return build_reference_eod_plan(
        run_id=run_id,
        core_symbols=tuple(symbols),
        expected_vintage=expected_vintage,
        core_fields=tuple(fields),
        hard_cap=99,
    )


def execute_eod_plan(
    plan: RequestPlan,
    *,
    output_root: Path = PROJECT_ROOT / "out",
    workers: int = 3,
) -> Mapping[str, Any]:
    if plan.run_type is not RunType.EOD:
        raise EodError("only a frozen EOD plan can use this executor")
    if not 1 <= int(workers) <= 4:
        raise EodError("workers must be between 1 and 4")
    run_dir = Path(output_root).expanduser().resolve() / plan.run_id
    run_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    os.chmod(run_dir, 0o700)
    _private_json(run_dir / "orats_request_plan.json", plan.to_dict())
    ledger = RequestLedger(account_ledger_path())
    gateway = OratsGateway(
        plan=plan,
        ledger=ledger,
        cache=ContentAddressedCache(CULTRA_CACHE_ROOT / "eod"),
        token_source=EnvFileTokenSource(CULTRA_ENV_PATH),
        transport=UrllibTransport(timeout_seconds=45.0),
    )
    snapshots: Dict[str, Any] = {}
    request_rows: Dict[str, Tuple[Mapping[str, Any], ...]] = {}
    errors: Dict[str, str] = {}
    try:
        completed, errors = execute_plan_via_local_daemon(
            gateway,
            tuple(item.logical_request_id for item in plan.requests),
            socket_path=run_dir / "orats-gateway.sock",
            workers=workers,
            client_timeout_seconds=60.0,
        )
        for request in plan.requests:
            result = completed.get(request.logical_request_id)
            if result is None:
                continue
            request_rows[request.logical_request_id] = decode_rows(result.raw)
            snapshots[request.logical_request_id] = dict(
                result.manifest.to_dict(),
                cache_hit=result.cache_hit,
                charged_attempts=result.charged_attempts,
            )
    finally:
        ledger.finish_run(plan.run_id, aborted=bool(errors))
        ledger.export(plan.run_id, run_dir / "orats_request_ledger.json")
    endpoint_tables: Dict[str, Dict[str, Any]] = {}
    for request in plan.requests:
        key = "%s|%s" % (request.endpoint.value, request.field_profile)
        table = endpoint_tables.setdefault(
            key,
            {
                "endpoint": request.endpoint.value,
                "field_profile": request.field_profile,
                "requested_entities": set(),
                "returned_entities": set(),
                "rows": [],
            },
        )
        table["requested_entities"].update(request.entities)
        result_rows = request_rows.get(request.logical_request_id, ())
        table["rows"].extend(result_rows)
        manifest = snapshots.get(request.logical_request_id)
        if manifest is not None:
            table["returned_entities"].update(manifest["returned_entities"])
    normalized_tables = {}
    for key, table in sorted(endpoint_tables.items()):
        requested_entities = set(table.pop("requested_entities"))
        returned_entities = set(table.pop("returned_entities"))
        rows = sorted(
            table.pop("rows"),
            key=lambda row: (
                str(row.get("ticker", row.get("optionSymbol", ""))),
                str(row.get("tradeDate", "")),
                str(row.get("expirDate", "")),
                str(row.get("strike", "")),
            ),
        )
        normalized_tables[key] = dict(
            table,
            requested_entities=sorted(requested_entities),
            returned_entities=sorted(returned_entities),
            unresolved_entities=sorted(requested_entities - returned_entities),
            rows=rows,
        )
    core_rows = []
    for table in normalized_tables.values():
        if table["endpoint"] == "/datav2/cores":
            core_rows.extend(table["rows"])
    core_rows_by_ticker = {
        str(row["ticker"]).strip().upper(): row
        for row in core_rows
        if row.get("ticker")
    }
    requested = tuple(
        sorted({entity for item in plan.requests for entity in item.entities})
    )
    unresolved = tuple(
        sorted(
            {
                entity
                for table in normalized_tables.values()
                for entity in table["unresolved_entities"]
            }
        )
    )
    result = {
        "schema": "cultra.orats-eod-enrichment.v2",
        "run_id": plan.run_id,
        "plan_hash": plan.plan_hash,
        "expected_vintage": plan.requests[0].expected_vintage,
        "counts": {
            "logical_requests": plan.logical_count,
            "charged_attempts": ledger.summary(plan.run_id)["charged_attempts"],
            "requested_symbols": len(requested),
            "resolved_symbols": len(core_rows_by_ticker),
            "resolved_core_symbols": len(core_rows_by_ticker),
            "unresolved_symbols": len(unresolved),
            "failed_requests": len(errors),
        },
        # Compatibility view for the local Core screener. Other endpoint rows
        # remain in their own typed tables and can never overwrite Core fields.
        "rows": [core_rows_by_ticker[key] for key in sorted(core_rows_by_ticker)],
        "endpoint_tables": normalized_tables,
        "unresolved_symbols": list(unresolved),
        "errors": errors,
        "snapshots": snapshots,
    }
    enrichment_path = _private_json(run_dir / "orats_enrichment.json", result)
    cache_report_path = _private_json(
        run_dir / "orats_cache_report.json",
        {
            "schema": "cultra.orats-cache-report.v1",
            "planned_requests": plan.logical_count,
            "cache_hits": sum(bool(item.get("cache_hit")) for item in snapshots.values()),
            "network_misses": sum(
                not bool(item.get("cache_hit")) for item in snapshots.values()
            ),
            "charged_attempts": ledger.summary(plan.run_id)["charged_attempts"],
            "failed_requests": len(errors),
        },
    )
    vintage_path = _private_json(
        run_dir / "orats_data_vintage_manifest.json",
        {
            "schema": "cultra.orats-data-vintage.v1",
            "expected_vintage": plan.requests[0].expected_vintage,
            "provider_trade_dates": sorted(
                {
                    trade_date
                    for item in snapshots.values()
                    for trade_date in item.get("provider_trade_dates", [])
                }
            ),
            "snapshot_ids": sorted(
                str(item["snapshot_id"]) for item in snapshots.values()
            ),
            "updated_at_min": min(
                (
                    str(item["updated_at_min"])
                    for item in snapshots.values()
                    if item.get("updated_at_min")
                ),
                default=None,
            ),
            "updated_at_max": max(
                (
                    str(item["updated_at_max"])
                    for item in snapshots.values()
                    if item.get("updated_at_max")
                ),
                default=None,
            ),
        },
    )
    plan_path = run_dir / "orats_request_plan.json"
    ledger_path = run_dir / "orats_request_ledger.json"
    _private_json(
        run_dir / "manifest.json",
        {
            "schema": "cultra.orats-eod-run-manifest.v1",
            "run_id": plan.run_id,
            "plan_hash": plan.plan_hash,
            "files": [
                {
                    "path": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for path in (
                    plan_path,
                    ledger_path,
                    enrichment_path,
                    cache_report_path,
                    vintage_path,
                )
            ],
            "complete": not errors,
        },
    )
    if errors:
        raise EodError("one or more frozen ORATS EOD requests failed")
    return result


def select_chain_finalists(
    *, history_screen: Path, orats_enrichment: Path, capacity: Optional[int] = None
) -> Mapping[str, Any]:
    """Freeze exact-chain coverage without silently imposing a top-N universe.

    ``capacity=None`` is the production default and admits every symbol that has
    both history and ORATS Core data.  A finite capacity is retained only for an
    explicitly budgeted diagnostic run, and every omitted symbol is preserved
    as unresolved.
    """

    if capacity is not None and not 1 <= int(capacity) <= 80:
        raise EodError("explicit chain capacity must be between 1 and 80")
    history_rows = {
        str(item["ticker"]): item
        for item in json.loads(Path(history_screen).read_text(encoding="utf-8"))["rows"]
    }
    orats_rows = {
        str(item["ticker"]): item
        for item in json.loads(Path(orats_enrichment).read_text(encoding="utf-8"))["rows"]
    }
    input_fingerprints = {
        "history_screen_sha256": hashlib.sha256(Path(history_screen).read_bytes()).hexdigest(),
        "orats_enrichment_sha256": hashlib.sha256(
            Path(orats_enrichment).read_bytes()
        ).hexdigest(),
    }
    scored = []
    for ticker in sorted(set(history_rows).intersection(orats_rows)):
        history = history_rows[ticker]
        analytics = orats_rows[ticker]
        iv = max(float(analytics.get("iv30d") or 0.0), 0.01)
        forecast = max(float(analytics.get("orFcst20d") or 0.0), 0.01)
        confidence = max(0.0, min(100.0, float(analytics.get("confidence") or 0.0)))
        directional = min(
            1.25,
            abs(float(history["trend_score"]))
            * (0.5 + confidence / 200.0)
            * max(0.5, min(1.5, forecast / iv)),
        )
        atm = max(float(analytics.get("atmIvM1") or iv), 0.01)
        atm_forecast = float(analytics.get("atmFcstIvM1") or forecast)
        reliability = max(0.0, min(1.0, float(analytics.get("fcstR2") or 0.0)))
        forecast_dislocation = abs(atm_forecast / atm - 1.0) * math.sqrt(reliability)
        option_interest = max(
            0.0,
            float(analytics.get("cOi") or 0.0) + float(analytics.get("pOi") or 0.0),
        )
        liquidity = math.log10(option_interest + 1.0)
        scored.append(
            {
                "ticker": ticker,
                "directional_score": directional,
                "forecast_dislocation_score": forecast_dislocation,
                "option_liquidity_score": liquidity,
            }
        )
    if capacity is None:
        selected_values = sorted(scored, key=lambda item: item["ticker"])
        return {
            "schema": "cultra.chain-finalist-selection.v2",
            "selection_policy": "ALL_CORE_AND_HISTORY_RESOLVED_NO_RANK_SUPPRESSION",
            "capacity": None,
            "input_fingerprints": input_fingerprints,
            "selected_symbols": [item["ticker"] for item in selected_values],
            "selected": selected_values,
            "budget_unresolved": [],
        }

    selected: Dict[str, Mapping[str, Any]] = {}

    def admit(values: Sequence[Mapping[str, Any]], count: int) -> None:
        for item in values:
            if len(selected) >= capacity or count <= 0:
                return
            if item["ticker"] not in selected:
                selected[item["ticker"]] = item
                count -= 1

    directional_slots = max(1, int(capacity) - 6)
    admit(
        sorted(scored, key=lambda item: (-float(item["directional_score"]), item["ticker"])),
        directional_slots,
    )
    admit(
        sorted(
            scored,
            key=lambda item: (-float(item["forecast_dislocation_score"]), item["ticker"]),
        ),
        3,
    )
    admit(
        sorted(scored, key=lambda item: (-float(item["option_liquidity_score"]), item["ticker"])),
        int(capacity) - len(selected),
    )
    selected_values = sorted(
        selected.values(),
        key=lambda item: (-float(item["directional_score"]), item["ticker"]),
    )
    unresolved = [
        dict(
            item,
            disposition="NOT_FULLY_EVALUATED_BUDGET",
            reason="passed Core and history screens but fell outside the frozen exact-chain capacity",
        )
        for item in scored
        if item["ticker"] not in selected
    ]
    return {
        "schema": "cultra.chain-finalist-selection.v2",
        "selection_policy": "EXPLICIT_CALLER_CAPACITY_WITH_COMPLETE_UNRESOLVED_SET",
        "capacity": capacity,
        "input_fingerprints": input_fingerprints,
        "selected_symbols": [item["ticker"] for item in selected_values],
        "selected": selected_values,
        "budget_unresolved": sorted(unresolved, key=lambda item: item["ticker"]),
    }


def save_chain_finalists(
    *,
    history_screen: Path,
    orats_enrichment: Path,
    output_path: Path,
    capacity: Optional[int] = None,
) -> Mapping[str, Any]:
    result = select_chain_finalists(
        history_screen=history_screen,
        orats_enrichment=orats_enrichment,
        capacity=capacity,
    )
    _private_json(output_path, result)
    return result


__all__ = [
    "CORE_FIELDS",
    "CORE_SCREEN_FIELDS",
    "EodError",
    "build_core_plan",
    "decode_rows",
    "execute_eod_plan",
    "save_chain_finalists",
    "select_chain_finalists",
]
