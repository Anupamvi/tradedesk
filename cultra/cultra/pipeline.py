"""Offline-first orchestration for the Cultra clean-room pipeline."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import stat
import sys
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .artifacts import ArtifactError, ArtifactWriter, RunManifest
from .catalog import CATALOG_VERSION, FROZEN_STRATEGY_CATALOG
from .reports import CandidateRow, DailyBoardData, render_daily_board, sorted_eligible_tickets
from .protocol import ProtocolError, load_historical_campaign_protocol
from .requesting import ENDPOINT_RULES, Endpoint
from .schwab import DEFAULT_SCHWAB_TOKEN_PATH
from .tickets import (
    ManualTicket,
    TicketCandidate,
    TicketRejection,
    build_manual_ticket,
    revalidate_manual_ticket,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "out"
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"

_DOCUMENTED_PROFILE_ENDPOINTS = {
    "CORE_SCREEN_V1": Endpoint.CORES,
    "SUMMARY_ENRICH_V1": Endpoint.SUMMARIES,
    "MONEY_IMPLIED_V1": Endpoint.MONIES_IMPLIED,
    "MONEY_FORECAST_V1": Endpoint.MONIES_FORECAST,
    "EXACT_OPTION_V1": Endpoint.EXACT_OPTIONS,
}


def documented_field_profiles() -> Dict[str, Dict[str, Any]]:
    """Return the machine-readable, unprobed profile registry for offline runs."""

    result: Dict[str, Dict[str, Any]] = {}
    for profile_name, endpoint in _DOCUMENTED_PROFILE_ENDPOINTS.items():
        rule = ENDPOINT_RULES[endpoint]
        result[profile_name] = {
            "profile_name": profile_name,
            "version": "V" + profile_name.rsplit("_V", 1)[-1],
            "status": rule.contract_status,
            "endpoint": endpoint.value,
            "method": rule.method,
            "entity_kind": rule.entity_kind,
            "batch_size": rule.batch_size,
            "entitlement_verified": False,
        }
    return result


class PipelineError(RuntimeError):
    """Raised when a run would bypass an evidence or safety gate."""


class LiveExecutionDisabled(PipelineError):
    """The initial implementation does not have authorization to fetch ORATS."""


class CheckStatus(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    status: CheckStatus
    detail: str


@dataclass(frozen=True)
class DoctorReport:
    checks: Tuple[DoctorCheck, ...]

    @property
    def ok(self) -> bool:
        return all(item.status != CheckStatus.FAIL for item in self.checks)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "scope": "OFFLINE_ENGINEERING_ONLY",
            "production_ready": False,
            "profit_confidence": "UNPROVEN",
            "manual_ticket_enabled": False,
            "checks": [
                {
                    "name": item.name,
                    "status": item.status.value,
                    "detail": item.detail,
                }
                for item in self.checks
            ],
        }


@dataclass(frozen=True)
class IsolationViolation:
    path: str
    line: int
    rule: str
    detail: str


_STDLIB_MODULES = frozenset(
    {
        "__future__",
        "argparse",
        "ast",
        "base64",
        "bisect",
        "collections",
        "concurrent",
        "contextlib",
        "copy",
        "csv",
        "dataclasses",
        "datetime",
        "decimal",
        "enum",
        "errno",
        "fcntl",
        "functools",
        "hashlib",
        "http",
        "io",
        "itertools",
        "json",
        "math",
        "os",
        "pathlib",
        "queue",
        "random",
        "re",
        "secrets",
        "shutil",
        "signal",
        "socket",
        "socketserver",
        "sqlite3",
        "stat",
        "statistics",
        "string",
        "struct",
        "sys",
        "tempfile",
        "threading",
        "time",
        "types",
        "typing",
        "urllib",
        "uuid",
        "xml",
        "zipfile",
        "zoneinfo",
    }
)
_NETWORK_IMPORTS = frozenset({"http.client", "urllib.request"})
_TRADE_DESK_PREFIX = "/Users/anuppamvi" + "/tradedesk/"
_FORBIDDEN_CALL_NAMES = frozenset(
    {
        "get_account",
        "get_accounts",
        "get_positions",
        "place_order",
        "submit_order",
        "replace_order",
        "cancel_order",
    }
)


def _call_name(node: ast.Call) -> str:
    target = node.func
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    return ""


def scan_clean_room(
    package_root: Path = PACKAGE_ROOT,
    *,
    project_root: Path = PROJECT_ROOT,
) -> Tuple[IsolationViolation, ...]:
    """Statically reject cross-pipeline imports, paths, and broker mutations."""

    package = Path(package_root).resolve()
    project = Path(project_root).resolve()
    allowed_absolute_paths = {
        str(project),
        str(DEFAULT_SCHWAB_TOKEN_PATH.resolve()),
    }
    violations = []
    for source_path in sorted(package.rglob("*.py")):
        try:
            source = source_path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(source_path))
        except (OSError, SyntaxError) as exc:
            violations.append(
                IsolationViolation(
                    str(source_path),
                    getattr(exc, "lineno", 0) or 0,
                    "parse",
                    str(exc),
                )
            )
            continue
        relative = source_path.relative_to(package).as_posix()
        for node in ast.walk(tree):
            imported = []
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    imported = []
                elif node.module:
                    imported = [node.module]
            for module in imported:
                top = module.split(".", 1)[0]
                if top != "cultra" and top not in _STDLIB_MODULES:
                    violations.append(
                        IsolationViolation(
                            relative,
                            getattr(node, "lineno", 0),
                            "stdlib-only",
                            "non-stdlib or cross-pipeline import: %s" % module,
                        )
                    )
                if module in _NETWORK_IMPORTS and source_path.name != "gateway.py":
                    violations.append(
                        IsolationViolation(
                            relative,
                            getattr(node, "lineno", 0),
                            "network-boundary",
                            "%s may be imported only by gateway.py" % module,
                        )
                    )
                if top in {"socket", "socketserver"} and source_path.name != "gateway.py":
                    violations.append(
                        IsolationViolation(
                            relative,
                            getattr(node, "lineno", 0),
                            "network-boundary",
                            "socket may be imported only by gateway.py",
                        )
                    )
            if isinstance(node, ast.Call) and _call_name(node) in _FORBIDDEN_CALL_NAMES:
                violations.append(
                    IsolationViolation(
                        relative,
                        getattr(node, "lineno", 0),
                        "manual-only",
                        "broker account/order operation is prohibited: %s" % _call_name(node),
                    )
                )
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                raw = node.value
                if raw.startswith(_TRADE_DESK_PREFIX):
                    normalized = str(Path(raw).resolve())
                    if not any(
                        normalized == allowed or normalized.startswith(allowed + os.sep)
                        for allowed in allowed_absolute_paths
                    ):
                        violations.append(
                            IsolationViolation(
                                relative,
                                getattr(node, "lineno", 0),
                                "path-isolation",
                                "absolute path leaves the Cultra/credential allowlist",
                            )
                        )
    return tuple(violations)


def _file_metadata_check(name: str, path: Path) -> DoctorCheck:
    try:
        info = path.stat()
    except FileNotFoundError:
        return DoctorCheck(name, CheckStatus.WARN, "%s is not present" % path)
    except OSError as exc:
        return DoctorCheck(name, CheckStatus.FAIL, "cannot stat %s: %s" % (path, exc))
    if not stat.S_ISREG(info.st_mode):
        return DoctorCheck(name, CheckStatus.FAIL, "%s is not a regular file" % path)
    mode = stat.S_IMODE(info.st_mode)
    if mode & 0o077:
        return DoctorCheck(
            name,
            CheckStatus.FAIL,
            "%s has overly broad permissions %03o" % (path, mode),
        )
    return DoctorCheck(name, CheckStatus.PASS, "%s exists with private permissions" % path)


def run_doctor(project_root: Path = PROJECT_ROOT) -> DoctorReport:
    """Perform zero-request checks and inspect only credential file metadata."""

    root = Path(project_root).resolve()
    package = root / "cultra"
    checks = []
    checks.append(
        DoctorCheck(
            "python",
            CheckStatus.PASS if sys.version_info >= (3, 9) else CheckStatus.FAIL,
            "Python %d.%d.%d" % sys.version_info[:3],
        )
    )
    checks.append(
        DoctorCheck(
            "package-root",
            CheckStatus.PASS if package.is_dir() else CheckStatus.FAIL,
            "clean-room package: %s" % package,
        )
    )
    violations = scan_clean_room(package, project_root=root) if package.is_dir() else ()
    checks.append(
        DoctorCheck(
            "clean-room-static-scan",
            CheckStatus.FAIL if violations else CheckStatus.PASS,
            (
                "%d violation(s): %s"
                % (
                    len(violations),
                    "; ".join(
                        "%s:%d %s" % (item.path, item.line, item.detail)
                        for item in violations[:8]
                    ),
                )
                if violations
                else "stdlib-only imports and path/network boundaries passed"
            ),
        )
    )
    checks.append(_file_metadata_check("cultra-env", root / ".env"))
    checks.append(_file_metadata_check("schwab-token", DEFAULT_SCHWAB_TOKEN_PATH))
    required_docs = (
        "00_CLEAN_ROOM_CHARTER.md",
        "15_ORATS_ENDPOINT_OWNERSHIP.md",
        "16_ORATS_REQUEST_EFFICIENCY_PLAN.md",
        "17_ORATS_FIELD_PROFILES.md",
        "18_ORATS_CACHE_AND_VINTAGE_SPEC.md",
    )
    missing = [name for name in required_docs if not (root / "docs" / name).is_file()]
    checks.append(
        DoctorCheck(
            "architecture-docs",
            CheckStatus.FAIL if missing else CheckStatus.PASS,
            "missing: %s" % ", ".join(missing) if missing else "required architecture documents present",
        )
    )
    try:
        protocol = load_historical_campaign_protocol()
    except (ProtocolError, KeyError, TypeError, ValueError) as exc:
        checks.append(
            DoctorCheck(
                "historical-protocol-v2",
                CheckStatus.FAIL,
                "canonical protocol failed closed: %s" % exc,
            )
        )
    else:
        checks.append(
            DoctorCheck(
                "historical-protocol-v2",
                CheckStatus.PASS,
                "%s; expected cold campaign %d attempts"
                % (
                    protocol["version"],
                    int(protocol["acquisition"]["expected_cold_attempts"]),
                ),
            )
        )
    checks.append(
        DoctorCheck(
            "network-mode",
            CheckStatus.PASS,
            "orchestration defaults to offline and live ORATS execution is disabled",
        )
    )
    checks.append(
        DoctorCheck(
            "production-evidence",
            CheckStatus.WARN,
            "not evaluated by doctor; production ready NO, profit confidence UNPROVEN",
        )
    )
    return DoctorReport(tuple(checks))


def reference_request_budget() -> Dict[str, Any]:
    """Return the frozen request envelope without reading credentials or data."""

    return {
        "schema": "cultra.reference-request-budget.v3",
        "run_type": "eod",
        "funnel": {
            "core": {"maximum_entities": 600, "batch_size": 10},
            "summary": {"maximum_entities": 120, "batch_size": 10},
            "monies_implied": {"maximum_entities": 40, "batch_size": 10},
            "monies_forecast": {"maximum_entities": 40, "batch_size": 10},
            "exact_options": {"maximum_entities": 250, "batch_size": 100},
        },
        "target_logical_requests": 25,
        "base_logical_request_ceiling": 60,
        "logical_request_ceiling": 60,
        "maximum_retry_reserve": 0,
        "maximum_planned_charged_attempts": 60,
        "protocol_attempt_ceiling": 99,
        "arbitrary_universe_cap": None,
        "same_vintage_warm_requests": 0,
        "morning_delta_requests": 0,
        "automatic_retries": 0,
        "credential_loaded": False,
        "network_attempted": False,
    }


@dataclass(frozen=True)
class PipelineInputs:
    strategy_evidence: Sequence[Any] = ()
    ticket_candidates: Sequence[Any] = ()
    tickets: Sequence[Any] = ()
    watchlist: Sequence[Any] = ()
    rejected: Sequence[Any] = ()
    data_unavailable: Sequence[Any] = ()
    budget_unresolved: Sequence[Any] = ()
    request_plan: Optional[Any] = None
    request_ledger: Optional[Any] = None
    cache_report: Optional[Any] = None
    data_vintage_manifest: Optional[Any] = None
    promotion_decisions: Sequence[Any] = ()
    model_artifacts: Mapping[str, Any] = field(default_factory=dict)
    field_profiles: Mapping[str, Any] = field(default_factory=dict)
    snapshot_ids: Sequence[str] = ()
    source_trade_dates: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineRunConfig:
    as_of: date
    output_root: Path = DEFAULT_OUTPUT_ROOT
    run_id: Optional[str] = None
    overall_status: str = "UNPROVEN"
    execute_orats: bool = False
    created_at: Optional[datetime] = None

    def resolved_run_id(self) -> str:
        if self.run_id:
            return self.run_id
        stamp = (self.created_at or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
        return "cultra-%s-%s-%s" % (
            self.as_of.isoformat(),
            stamp,
            uuid.uuid4().hex[:8],
        )


@dataclass(frozen=True)
class PipelineResult:
    run_id: str
    run_dir: Path
    board_path: Path
    manifest: RunManifest
    ticket_count: int


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def _strategy_states(evidence: Sequence[Any]) -> Dict[str, str]:
    states = {
        definition.strategy_id: "UNPROVEN"
        for definition in FROZEN_STRATEGY_CATALOG
    }
    supplied = set()
    for item in evidence:
        family = str(getattr(item, "strategy_family", "")).strip()
        if not family:
            raise PipelineError("strategy evidence is missing strategy_family")
        if family in supplied:
            raise PipelineError("duplicate strategy evidence: %s" % family)
        if family not in states:
            raise PipelineError(
                "strategy evidence is outside the frozen catalog: %s" % family
            )
        supplied.add(family)
        states[family] = _enum_value(getattr(item, "state", "UNPROVEN"))
    return states


def _validate_ticket(ticket: Any, now: datetime) -> None:
    if not isinstance(ticket, ManualTicket):
        raise PipelineError("ticket must be produced by Cultra's gated ticket builder")
    try:
        revalidate_manual_ticket(ticket, now)
    except (TicketRejection, ValueError) as exc:
        raise PipelineError(str(exc)) from exc
    state = _enum_value(getattr(ticket, "evidence_state", ""))
    if state != "MANUAL_TICKET_ENABLED":
        raise PipelineError("ticket has not passed the manual-ticket evidence gate")
    quantity = getattr(ticket, "quantity", None)
    if quantity != "USER DETERMINED":
        raise PipelineError("Cultra must not choose ticket quantity")
    for name in (
        "candidate_id",
        "symbol",
        "thesis",
        "signal",
        "strategy_id",
        "orats_snapshot_id",
    ):
        if not str(getattr(ticket, name, "")).strip():
            raise PipelineError("ticket is missing mandatory field %s" % name)
    if getattr(ticket, "quote_source", None) != "SCHWAB":
        raise PipelineError("ticket quote source must be SCHWAB")
    if not getattr(ticket, "analytical_fields", ()):
        raise PipelineError("ticket requires named analytical fields")
    model_calculation = getattr(ticket, "model_calculation", None)
    if model_calculation is None or not getattr(model_calculation, "features", ()):
        raise PipelineError(
            "ticket requires the saved current model feature/score calculation"
        )
    if not str(getattr(model_calculation, "calculation_id", "")).strip():
        raise PipelineError("ticket current model calculation is not content-addressed")
    if getattr(ticket, "policy", None) is None or getattr(ticket, "underlying_quote", None) is None:
        raise PipelineError("ticket requires an entry/exit policy and underlying quote")
    edge = getattr(ticket, "edge", None)
    if edge is None:
        raise PipelineError("ticket is missing edge evidence")
    net_ev = getattr(edge, "net_expected_value", None)
    conservative_ev = getattr(edge, "conservative_net_expected_value", None)
    maximum_loss = getattr(edge, "maximum_loss", None)
    if net_ev is None or not float(net_ev) > 0:
        raise PipelineError("ticket net expected value must be positive")
    if conservative_ev is None or not float(conservative_ev) > 0:
        raise PipelineError("ticket conservative net expected value must be positive")
    if maximum_loss is None or not (float(maximum_loss) > 0 and math.isfinite(float(maximum_loss))):
        raise PipelineError("ticket maximum loss must be finite and positive")
    try:
        ranking_score = float(getattr(ticket, "ranking_score"))
        expected_rank = float(getattr(edge, "conservative_return_on_max_loss"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise PipelineError("ticket requires a finite conservative EV/risk rank") from exc
    if not math.isfinite(ranking_score) or not math.isclose(
        ranking_score, expected_rank, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise PipelineError("ticket ranking score must equal conservative EV/max-loss")
    if not getattr(edge, "point_scenarios", ()) or not getattr(
        edge, "conservative_scenarios", ()
    ):
        raise PipelineError("ticket edge must preserve reproducible scenario inputs")
    if getattr(edge, "conservative_gross_expected_value", None) is None:
        raise PipelineError("ticket edge is missing conservative gross EV")
    if not getattr(ticket, "legs", ()) or not getattr(ticket, "leg_quotes", ()):
        raise PipelineError("ticket requires exact legs and executable quotes")
    if not getattr(ticket, "orats_snapshot_id", ""):
        raise PipelineError("ticket requires an ORATS snapshot ID")
    probabilities = getattr(ticket, "probabilities", None)
    for name in ("pop_net", "p_target", "p_stop", "p_max_loss"):
        if probabilities is None or getattr(probabilities, name, None) is None:
            raise PipelineError("ticket is missing calibrated probability %s" % name)
    evidence = getattr(ticket, "evidence", None)
    if evidence is None:
        raise PipelineError("ticket is missing family evidence")
    if _enum_value(getattr(evidence, "state", "")) not in {
        "HOLDOUT_PASS",
        "SHADOW_PASS",
    }:
        raise PipelineError("ticket family evidence must be HOLDOUT_PASS or SHADOW_PASS")
    if float(getattr(evidence, "pop_ece", 1.0)) > 0.05:
        raise PipelineError("ticket POP calibration exceeds the ECE tolerance")
    if not float(getattr(evidence, "pop_brier_score", 1.0)) < float(
        getattr(evidence, "base_rate_brier_score", 0.0)
    ):
        raise PipelineError("ticket POP model does not beat its base rate")


def _as_payloads(values: Sequence[Any]) -> Sequence[Any]:
    result = []
    for value in values:
        converter = getattr(value, "to_dict", None)
        result.append(converter() if callable(converter) else value)
    return result


def _as_payload(value: Any) -> Any:
    converter = getattr(value, "to_dict", None)
    return converter() if callable(converter) else value


def _get(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _strategy_rejections(decisions: Sequence[Any]) -> Dict[str, str]:
    """Map explicitly family-labeled failed decisions onto the daily board."""

    catalog_ids = {definition.strategy_id for definition in FROZEN_STRATEGY_CATALOG}
    result: Dict[str, str] = {}
    for decision in decisions:
        family = str(_get(decision, "strategy_family", "") or "").strip()
        passed = _get(decision, "passed", None)
        if not family or passed is not False:
            continue
        if family not in catalog_ids:
            raise PipelineError("promotion rejection is outside the frozen catalog: %s" % family)
        raw_reasons = _get(decision, "reasons", ()) or ()
        if isinstance(raw_reasons, str):
            reasons = (raw_reasons,)
        else:
            reasons = tuple(str(item) for item in raw_reasons if str(item).strip())
        result[family] = "; ".join(reasons) or "promotion gate failed"
    return result


def _integer_metric(value: Any, *names: str) -> int:
    current = value
    if isinstance(current, Mapping) and "summary" in current:
        current = current["summary"]
    for name in names:
        if isinstance(current, Mapping) and name in current:
            try:
                return int(current[name])
            except (TypeError, ValueError):
                return 0
    return 0


def _render_data_health(
    *,
    request_plan: Mapping[str, Any],
    request_ledger: Mapping[str, Any],
    cache_report: Mapping[str, Any],
) -> str:
    """Render explicit ORATS telemetry; never hide calls in provider totals."""

    planned = _integer_metric(request_plan, "logical_count")
    actual_logical = _integer_metric(
        request_ledger, "actual_logical_requests", "logical_requests_executed"
    )
    attempts = _integer_metric(request_ledger, "charged_attempts")
    retries = _integer_metric(request_ledger, "retries")
    redirects = _integer_metric(request_ledger, "redirects")
    cache_hits = _integer_metric(cache_report, "cache_hits")
    deduplicated = _integer_metric(cache_report, "deduplicated_requests")
    symbols = _integer_metric(request_ledger, "symbols_requested")
    contracts = _integer_metric(request_ledger, "contracts_requested")
    rows = _integer_metric(request_ledger, "rows_downloaded", "rows_returned")
    response_bytes = _integer_metric(
        request_ledger, "total_response_bytes", "bytes_returned"
    )
    recoveries = _integer_metric(request_ledger, "missing_symbol_recoveries")
    hard_cap = _integer_metric(request_plan, "hard_cap")
    remaining = _integer_metric(request_ledger, "remaining")
    target = _integer_metric(request_plan, "target")
    target_met = attempts <= target if target else attempts == 0
    approached = bool(hard_cap and attempts >= int(hard_cap * 0.8))
    return "\n".join(
        (
            "# Cultra ORATS Data Health",
            "",
            "- Planned logical requests: %d" % planned,
            "- Actual logical requests: %d" % actual_logical,
            "- Actual outbound HTTP attempts: %d" % attempts,
            "- Retries: %d" % retries,
            "- Redirects followed: %d" % redirects,
            "- Cache hits: %d" % cache_hits,
            "- Deduplicated requests: %d" % deduplicated,
            "- Symbols requested: %d" % symbols,
            "- Contracts requested: %d" % contracts,
            "- Rows downloaded: %d" % rows,
            "- Total response bytes: %d" % response_bytes,
            "- Missing-symbol recoveries: %d" % recoveries,
            "- Remaining charged-attempt budget: %d" % remaining,
            "- Request target met: %s" % ("YES" if target_met else "NO"),
            "- Hard cap approached: %s" % ("YES" if approached else "NO"),
            "- Variance from plan: %s"
            % (
                "zero-request offline run"
                if attempts == 0
                else "see the immutable request ledger for every charged attempt"
            ),
            "",
        )
    )


class CultraPipeline:
    """Artifact-complete orchestration; network execution remains fail-closed."""

    def run(
        self,
        config: PipelineRunConfig,
        inputs: Optional[PipelineInputs] = None,
    ) -> PipelineResult:
        if config.execute_orats:
            raise LiveExecutionDisabled(
                "ORATS execution requires a separately authorized discovery/backfill stage"
            )
        supplied = inputs or PipelineInputs()
        validation_now = config.created_at or datetime.now(timezone.utc)
        built_tickets = []
        candidate_rejected = []
        candidate_unavailable = []
        for candidate in supplied.ticket_candidates:
            if not isinstance(candidate, TicketCandidate):
                raise PipelineError(
                    "ticket_candidates must be Cultra TicketCandidate values"
                )
            try:
                built_tickets.append(build_manual_ticket(candidate, validation_now))
            except TicketRejection as exc:
                reason = "; ".join(exc.reasons)
                row = CandidateRow(
                    candidate_id=candidate.candidate_id,
                    symbol=candidate.symbol,
                    strategy_family=candidate.strategy_id,
                    reason=reason,
                    disposition=(
                        "DATA_UNAVAILABLE"
                        if any(
                            marker in reason.lower()
                            for marker in ("stale", "missing", "unavailable")
                        )
                        else "REJECTED"
                    ),
                )
                if row.disposition == "DATA_UNAVAILABLE":
                    candidate_unavailable.append(row)
                else:
                    candidate_rejected.append(row)
        tickets = tuple(supplied.tickets) + tuple(built_tickets)
        rejected_values = tuple(supplied.rejected) + tuple(candidate_rejected)
        unavailable_values = tuple(supplied.data_unavailable) + tuple(
            candidate_unavailable
        )
        for ticket in tickets:
            _validate_ticket(ticket, validation_now)
        ordered_tickets = sorted_eligible_tickets(tickets)
        states = _strategy_states(supplied.strategy_evidence)
        rejection_reasons = _strategy_rejections(supplied.promotion_decisions)
        for family in rejection_reasons:
            states[family] = "REJECTED"
        ticket_families = {
            str(getattr(ticket, "strategy_id", "")) for ticket in tickets
        }
        conflicts = ticket_families.intersection(rejection_reasons)
        if conflicts:
            raise PipelineError(
                "rejected strategy cannot publish a ticket: %s"
                % ", ".join(sorted(conflicts))
            )
        if tickets and config.overall_status == "UNPROVEN":
            raise PipelineError("an UNPROVEN run cannot publish a manual ticket")

        run_id = config.resolved_run_id()
        created_at = validation_now
        writer = ArtifactWriter(Path(config.output_root), run_id)
        request_payload: Any
        request_plan_id: Optional[str]
        if supplied.request_plan is None:
            request_payload = {
                "schema": "cultra.offline-request-plan.v1",
                "run_id": run_id,
                "run_type": "OFFLINE",
                "mode": "OFFLINE",
                "target": 0,
                "hard_cap": 0,
                "retry_reserve": 0,
                "base_count": 0,
                "contingency_count": 0,
                "logical_count": 0,
                "charged_attempts": 0,
                "network_attempted": False,
            }
            request_plan_id = hashlib.sha256(
                json.dumps(
                    request_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            request_payload["plan_hash"] = request_plan_id
        else:
            converter = getattr(supplied.request_plan, "to_dict", None)
            if not callable(converter):
                raise PipelineError("request_plan must provide to_dict()")
            request_payload = converter()
            planned_run_id = str(getattr(supplied.request_plan, "run_id", ""))
            if planned_run_id and planned_run_id != run_id:
                raise PipelineError("request plan and artifact run IDs must match")
            request_plan_id = str(
                getattr(supplied.request_plan, "plan_hash", "") or ""
            ) or None

        if not isinstance(request_payload, Mapping):
            raise PipelineError("request plan payload must be a mapping")

        evidence_payloads = tuple(_as_payloads(supplied.strategy_evidence))
        if not evidence_payloads:
            evidence_payloads = tuple(
                {
                    "strategy_family": definition.strategy_id,
                    "state": "UNPROVEN",
                    "catalog_version": CATALOG_VERSION,
                    "validated_observations": 0,
                    "reason": "NO_AUTHORIZED_HISTORICAL_OR_SHADOW_EVIDENCE",
                }
                for definition in FROZEN_STRATEGY_CATALOG
            )
        ticket_payloads = tuple(_as_payloads(ordered_tickets))
        watchlist_payloads = tuple(_as_payloads(supplied.watchlist))
        rejected_payloads = tuple(_as_payloads(rejected_values))
        unavailable_payloads = tuple(_as_payloads(unavailable_values))
        budget_payloads = tuple(_as_payloads(supplied.budget_unresolved))

        model_versions = {
            str(getattr(item, "strategy_family")): str(getattr(item, "model_version"))
            for item in supplied.strategy_evidence
        }
        field_profiles_payload: Dict[str, Any] = documented_field_profiles()
        field_profiles_payload.update(dict(supplied.field_profiles))
        field_profile_versions = {
            name: str(_get(profile, "version", name))
            for name, profile in field_profiles_payload.items()
        }
        field_profile_statuses = {
            name: str(_get(profile, "status", "UNKNOWN"))
            for name, profile in field_profiles_payload.items()
        }
        ticket_snapshot_ids = tuple(
            str(getattr(ticket, "orats_snapshot_id")) for ticket in ordered_tickets
        )
        snapshot_ids = tuple(
            dict.fromkeys(tuple(supplied.snapshot_ids) + ticket_snapshot_ids)
        )
        source_trade_dates = dict(supplied.source_trade_dates)
        for ticket in ordered_tickets:
            snapshot_id = str(getattr(ticket, "orats_snapshot_id"))
            trade_date = getattr(ticket, "provider_trade_date", None)
            if trade_date is not None:
                source_trade_dates.setdefault("ORATS:%s" % snapshot_id, str(trade_date))

        ledger_payload = (
            _as_payload(supplied.request_ledger)
            if supplied.request_ledger is not None
            else {
                "schema": "cultra.orats-request-ledger.v1",
                "run_id": run_id,
                "state": "NOT_STARTED_OFFLINE",
                "actual_logical_requests": 0,
                "charged_attempts": 0,
                "retries": 0,
                "redirects": 0,
                "remaining": int(request_payload.get("hard_cap", 0) or 0),
                "attempts": [],
                "network_attempted": False,
            }
        )
        cache_payload = (
            _as_payload(supplied.cache_report)
            if supplied.cache_report is not None
            else {
                "schema": "cultra.orats-cache-report.v1",
                "run_id": run_id,
                "cache_hits": 0,
                "cache_misses": 0,
                "deduplicated_requests": 0,
                "published_snapshots": 0,
                "network_attempted": False,
            }
        )
        vintage_payload = (
            _as_payload(supplied.data_vintage_manifest)
            if supplied.data_vintage_manifest is not None
            else {
                "schema": "cultra.orats-data-vintage-manifest.v1",
                "run_id": run_id,
                "snapshot_ids": list(snapshot_ids),
                "source_trade_dates": source_trade_dates,
                "vintages": [],
                "network_attempted": False,
            }
        )
        if not all(
            isinstance(item, Mapping)
            for item in (ledger_payload, cache_payload, vintage_payload)
        ):
            raise PipelineError("ORATS telemetry payloads must be mappings")

        quotes_payload = [
            {
                "candidate_id": _get(ticket, "candidate_id"),
                "quote_source": _get(ticket, "quote_source"),
                "underlying_quote": _get(ticket, "underlying_quote"),
                "leg_quotes": _get(ticket, "leg_quotes", ()),
            }
            for ticket in ordered_tickets
        ]
        pop_payload = [
            {
                "candidate_id": _get(ticket, "candidate_id"),
                "probabilities": _get(ticket, "probabilities"),
            }
            for ticket in ordered_tickets
        ]
        edge_payload = [
            {
                "candidate_id": _get(ticket, "candidate_id"),
                "edge": _get(ticket, "edge"),
            }
            for ticket in ordered_tickets
        ]
        promotion_payload = (
            tuple(_as_payloads(supplied.promotion_decisions))
            if supplied.promotion_decisions
            else tuple(
                {
                    "strategy_family": _get(item, "strategy_family"),
                    "state": _enum_value(_get(item, "state", "UNPROVEN")),
                    "model_version": _get(item, "model_version"),
                }
                for item in supplied.strategy_evidence
            )
        )
        candidates_payload = {
            "eligible_manual_tickets": ticket_payloads,
            "watchlist": watchlist_payloads,
            "rejected": rejected_payloads,
            "data_unavailable": unavailable_payloads,
            "not_fully_evaluated_budget": budget_payloads,
        }

        board = DailyBoardData(
            as_of=config.as_of,
            run_id=run_id,
            overall_status=config.overall_status,
            strategy_states=states,
            strategy_rejection_reasons=rejection_reasons,
            tickets=ticket_payloads,
            watchlist=watchlist_payloads,
            rejected=rejected_payloads,
            data_unavailable=unavailable_payloads,
            budget_unresolved=budget_payloads,
            generated_at=created_at,
        )

        try:
            writer.write_json("orats_request_plan.json", request_payload)
            writer.write_json("request_plan.json", request_payload)
            writer.write_json("orats_request_ledger.json", ledger_payload)
            writer.write_json("orats_cache_report.json", cache_payload)
            writer.write_json("orats_data_vintage_manifest.json", vintage_payload)
            writer.write_text(
                "data_health.md",
                _render_data_health(
                    request_plan=request_payload,
                    request_ledger=ledger_payload,
                    cache_report=cache_payload,
                ),
                media_type="text/markdown",
            )
            writer.write_json("strategy_evidence.json", evidence_payloads)
            writer.write_json("promotion_decisions.json", promotion_payload)
            writer.write_json("model_artifacts.json", dict(supplied.model_artifacts))
            writer.write_json("field_profiles.json", field_profiles_payload)
            writer.write_json("quotes.json", quotes_payload)
            writer.write_json("candidates.json", candidates_payload)
            writer.write_json("pop_calculations.json", pop_payload)
            writer.write_json("edge_calculations.json", edge_payload)
            writer.write_json("manual_tickets.json", ticket_payloads)
            writer.write_json("watchlist.json", watchlist_payloads)
            writer.write_json("rejected.json", rejected_payloads)
            writer.write_json("data_unavailable.json", unavailable_payloads)
            writer.write_json(
                "not_fully_evaluated_budget.json",
                budget_payloads,
            )
            writer.write_json("daily_board.json", board.to_dict())
            writer.write_text(
                "daily_board.md",
                render_daily_board(board),
                media_type="text/markdown",
            )
            manifest = writer.finalize(
                as_of=config.as_of,
                overall_status=config.overall_status,
                created_at=created_at,
                request_plan_id=request_plan_id,
                snapshot_ids=snapshot_ids,
                model_versions=model_versions,
                field_profile_versions=field_profile_versions,
                field_profile_statuses=field_profile_statuses,
                strategy_states=states,
                source_trade_dates=source_trade_dates,
                metadata={
                    "manual_ticket_count": len(ordered_tickets),
                    "watchlist_count": len(supplied.watchlist),
                    "rejected_count": len(rejected_values),
                    "data_unavailable_count": len(unavailable_values),
                    "budget_unresolved_count": len(supplied.budget_unresolved),
                    "quantity_policy": "USER DETERMINED",
                    "order_submission_surface": False,
                    "network_attempted": False,
                },
            )
        except BaseException:
            # Leave the incomplete run directory as evidence; never overwrite it.
            raise
        return PipelineResult(
            run_id=run_id,
            run_dir=writer.run_dir,
            board_path=writer.run_dir / "daily_board.md",
            manifest=manifest,
            ticket_count=len(ordered_tickets),
        )
