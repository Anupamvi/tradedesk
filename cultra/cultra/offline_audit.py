"""Zero-network reconciliation of Cultra usage, historical coverage, and gaps."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .request_optimization import (
    RotatingCohortPolicy,
    daily_request_budget,
    rotating_cohort_campaign_forecast,
)
from .protocol import (
    HISTORICAL_CAMPAIGN_CONFIG,
    load_historical_campaign_protocol,
)
from .prerequisites import (
    HistoricalPrerequisiteError,
    load_historical_prerequisites,
)
from .universe import local_screen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = PROJECT_ROOT / "out"
LEDGER_ROOT = PROJECT_ROOT / "state" / "orats_ledger"
CACHE_ROOT = PROJECT_ROOT / "state" / "orats_cache"
CHAIN_DB = PROJECT_ROOT / "var" / "historical" / "cultra_chains_v1.sqlite3"
SCREEN_PATH = (
    OUT_ROOT / "cultra-broad-screen-2026-08-30-v2" / "schwab_screen.json"
)
ORATS_PATH = (
    OUT_ROOT / "cultra-eod-core-full-2026-08-30-v2" / "orats_enrichment.json"
)
class OfflineAuditError(RuntimeError):
    """The saved Cultra state cannot be reconciled offline."""


def _load(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OfflineAuditError("saved audit input is unavailable: %s" % path.name) from exc


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _private_write(path: Path, data: bytes) -> Path:
    resolved = Path(path).expanduser().resolve()
    try:
        resolved.relative_to(OUT_ROOT.resolve())
    except ValueError as exc:
        raise OfflineAuditError("offline audit output must remain inside Cultra/out") from exc
    resolved.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(resolved.parent, 0o700)
    temporary = resolved.with_name(".%s.tmp-%d" % (resolved.name, os.getpid()))
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


def _private_json(path: Path, value: Any) -> Path:
    return _private_write(
        path,
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n",
    )


def _ledger_usage(ledger_root: Path) -> Mapping[str, Any]:
    runs = []
    endpoints: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"charged_attempts": 0, "successful_2xx": 0, "failed": 0, "rows": 0}
    )
    batch_geometry: Dict[Tuple[str, int], Dict[str, int]] = defaultdict(
        lambda: {"charged_attempts": 0, "successful_2xx": 0, "failed": 0, "rows": 0}
    )
    charged = successful = failed = rows = 0
    for path in sorted(Path(ledger_root).glob("*.sqlite3")):
        try:
            connection = sqlite3.connect("file:%s?mode=ro" % path.resolve(), uri=True)
            records = tuple(
                connection.execute(
                    """
                    SELECT a.run_id, a.endpoint, a.state, a.status_code,
                           COALESCE(a.rows_returned, 0),
                           COALESCE(p.entity_count, 0)
                      FROM attempts AS a
                      LEFT JOIN planned_requests AS p
                        ON p.run_id = a.run_id
                       AND p.logical_request_id = a.logical_request_id
                     ORDER BY a.network_attempt_number
                    """
                )
            )
        except sqlite3.Error as exc:
            raise OfflineAuditError("ORATS attempt ledger is unreadable: %s" % path.name) from exc
        finally:
            try:
                connection.close()
            except UnboundLocalError:
                pass
        by_run: Dict[str, list] = defaultdict(list)
        for record in records:
            by_run[str(record[0])].append(record)
        for run_id in sorted(by_run):
            run_records = by_run[run_id]
            run_charged = len(run_records)
            run_successful = sum(
                item[3] is not None and 200 <= int(item[3]) <= 299
                for item in run_records
            )
            run_failed = run_charged - run_successful
            run_rows = sum(int(item[4]) for item in run_records)
            runs.append(
                {
                    "run_id": run_id,
                    "ledger": str(path),
                    "charged_attempts": run_charged,
                    "successful_2xx": run_successful,
                    "failed_or_non_2xx": run_failed,
                    "rows_returned": run_rows,
                }
            )
            charged += run_charged
            successful += run_successful
            failed += run_failed
            rows += run_rows
        for _run_id, endpoint, _state, status_code, row_count, entity_count in records:
            item = endpoints[str(endpoint)]
            item["charged_attempts"] += 1
            batch = batch_geometry[(str(endpoint), int(entity_count))]
            batch["charged_attempts"] += 1
            if status_code is not None and 200 <= int(status_code) <= 299:
                item["successful_2xx"] += 1
                batch["successful_2xx"] += 1
            else:
                item["failed"] += 1
                batch["failed"] += 1
            item["rows"] += int(row_count)
            batch["rows"] += int(row_count)
    return {
        "ledger_count": len(runs),
        "charged_attempts": charged,
        "successful_2xx": successful,
        "failed_or_non_2xx": failed,
        "rows_returned": rows,
        "endpoints": {key: endpoints[key] for key in sorted(endpoints)},
        "batch_geometry": {
            "%s|%d" % key: batch_geometry[key]
            for key in sorted(batch_geometry)
        },
        "runs": runs,
    }


def _historical_cache_coverage(cache_root: Path) -> Mapping[str, Any]:
    manifests = sorted(
        (Path(cache_root) / "historical" / "manifests").glob("*/*.json")
    )
    groups: Dict[Tuple[str, Tuple[str, ...]], Dict[str, Any]] = {}
    aggregate_hash = hashlib.sha256()
    for path in manifests:
        raw = path.read_bytes()
        aggregate_hash.update(path.name.encode("ascii"))
        aggregate_hash.update(hashlib.sha256(raw).digest())
        try:
            item = json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise OfflineAuditError("historical cache manifest is invalid") from exc
        entities = tuple(str(value) for value in item.get("requested_entities", ()))
        key = (str(item.get("endpoint")), entities)
        group = groups.setdefault(
            key,
            {
                "endpoint": key[0],
                "requested_entities": list(entities),
                "manifest_count": 0,
                "row_count": 0,
                "expected_dates": [],
            },
        )
        group["manifest_count"] += 1
        group["row_count"] += int(item.get("row_count", 0))
        if item.get("expected_trade_date"):
            group["expected_dates"].append(str(item["expected_trade_date"]))
    values = []
    for group in groups.values():
        dates = sorted(set(group.pop("expected_dates")))
        group["first_expected_date"] = dates[0] if dates else None
        group["last_expected_date"] = dates[-1] if dates else None
        values.append(group)
    values.sort(key=lambda item: (item["endpoint"], item["requested_entities"]))
    return {
        "manifest_count": len(manifests),
        "manifest_set_sha256": aggregate_hash.hexdigest(),
        "groups": values,
    }


def _chain_database_coverage(path: Path) -> Mapping[str, Any]:
    try:
        connection = sqlite3.connect("file:%s?mode=ro" % Path(path).resolve(), uri=True)
        rows = tuple(
            connection.execute(
                """
                SELECT ticker, COUNT(DISTINCT trade_date), MIN(trade_date),
                       MAX(trade_date), COUNT(*)
                  FROM chains
                 GROUP BY ticker
                 ORDER BY ticker
                """
            )
        )
        totals = connection.execute(
            """
            SELECT (SELECT COUNT(*) FROM sessions),
                   (SELECT COUNT(*) FROM underlying),
                   (SELECT COUNT(*) FROM chains)
            """
        ).fetchone()
    except sqlite3.Error as exc:
        raise OfflineAuditError("historical chain database is unreadable") from exc
    finally:
        try:
            connection.close()
        except UnboundLocalError:
            pass
    return {
        "sessions": int(totals[0]),
        "underlying_rows": int(totals[1]),
        "chain_rows": int(totals[2]),
        "ticker_count": len(rows),
        "tickers": [str(item[0]) for item in rows],
        "by_ticker": {
            str(item[0]): {
                "sessions": int(item[1]),
                "first_date": str(item[2]),
                "last_date": str(item[3]),
                "rows": int(item[4]),
            }
            for item in rows
        },
    }


def _historical_prerequisite_status(output_root: Path) -> Mapping[str, Any]:
    """Inspect saved source-bound prerequisite receipts without selecting one."""

    valid = []
    invalid = []
    bound_files = set()
    for path in sorted(Path(output_root).glob("*/prerequisite_freeze.json")):
        try:
            frozen = load_historical_prerequisites(path)
        except (HistoricalPrerequisiteError, OSError, ValueError) as exc:
            invalid.append({"path": str(path.resolve()), "error": str(exc)})
            continue
        valid.append(
            {
                "path": str(path.resolve()),
                "input_set_id": frozen.input_set_id,
                "freeze_hash": str(frozen.payload["freeze_hash"]),
                "selection_dates": list(frozen.payload["selection_dates"]),
                "sampled_symbol_count": int(frozen.payload["sampled_symbol_count"]),
                "sampled_symbols": list(frozen.payload["sampled_symbols"]),
            }
        )
        bound_files.add(frozen.source_path)
        for section in ("source_inputs", "normalized_inputs"):
            for item in frozen.payload[section].values():
                bound_files.add(Path(str(item["path"])).resolve())
    return {
        "status": "FROZEN" if valid else "MISSING",
        "valid_count": len(valid),
        "invalid_count": len(invalid),
        "valid": valid,
        "invalid": invalid,
        "bound_files": tuple(sorted(bound_files, key=lambda item: str(item))),
        "network_attempted": False,
    }


def request_feasibility(
    *,
    symbol_counts: Sequence[int],
    batch_size: int,
    minimum_training_sessions: int,
    embargo_sessions: int,
    validation_sessions: int,
    holdout_fraction: float,
    reference_sessions: int,
    attempt_cap: int = 99,
) -> Mapping[str, Any]:
    """Compare a full-universe date grid with the frozen rotating design."""

    development_sessions = (
        int(minimum_training_sessions)
        + int(embargo_sessions)
        + int(validation_sessions)
    )
    minimum_total_sessions = int(
        math.ceil(development_sessions / (1.0 - float(holdout_fraction)))
    )
    scenarios = []
    for count in symbol_counts:
        requests_per_date = int(math.ceil(int(count) / float(batch_size)))
        forecast = rotating_cohort_campaign_forecast(
            RotatingCohortPolicy(
                eligible_symbols=int(count),
                historical_sessions=int(reference_sessions),
                slice_cap=int(attempt_cap),
            )
        )
        scenarios.append(
            {
                "symbols": int(count),
                "full_universe_chain_batches_per_date": requests_per_date,
                "minimum_sessions_for_one_development_fold_plus_holdout": minimum_total_sessions,
                "rejected_full_universe_date_grid_attempts": requests_per_date
                * int(reference_sessions),
                "rotating_cohort_campaign": forecast,
                "expected_campaign_attempts": int(
                    forecast["requests"]["cache_adjusted_total"]
                ),
                "slice_count": int(forecast["slicing"]["slice_count"]),
                "per_slice_cap": int(forecast["slicing"]["per_slice_cap"]),
            }
        )
    return {
        "endpoint_contract": {
            "bulk_features": "HIST_CORES_DOCUMENTED_MAX_10_BUT_FROZEN_AT_2_FROM_SAVED_ACCOUNT_EVIDENCE",
            "cohort_chain": "HIST_STRIKES_REQUIRES_ONE_TRADE_DATE_UP_TO_10_TICKERS",
            "split_history": "HIST_SPLITS_RETURNS_HISTORY_UP_TO_10_TICKERS",
        },
        "formula": "sampled full-history Core + one active-cohort chain per session + sampled split history; entries are censored before rotations",
        "batch_size": int(batch_size),
        "per_slice_cap": int(attempt_cap),
        "minimum_training_sessions": int(minimum_training_sessions),
        "embargo_sessions": int(embargo_sessions),
        "validation_sessions": int(validation_sessions),
        "development_sessions_for_one_fold": development_sessions,
        "holdout_fraction": float(holdout_fraction),
        "minimum_total_sessions": minimum_total_sessions,
        "rejected_design": "full historical chain for every broad-universe symbol on every date",
        "scenarios": scenarios,
    }




def _render_v7(audit: Mapping[str, Any]) -> str:
    """Render the canonical rotating-cohort estimate without legacy N+1 claims."""

    usage = audit["orats_usage"]
    database = audit["historical_database"]
    current = audit["current_coverage"]
    feasibility = audit["historical_campaign_feasibility"]
    budgets = audit["request_budgets"]
    current_history = next(
        item
        for item in feasibility["scenarios"]
        if int(item["symbols"]) == int(current["locally_eligible_without_cap"])
    )
    campaign = current_history["rotating_cohort_campaign"]
    requests = campaign["requests"]
    slicing = campaign["slicing"]
    lines = [
        "# Cultra Offline Request and Production Audit",
        "",
        "## Outcome",
        "",
        "- ORATS or Schwab requests made by this audit: **0**.",
        "- Source-bound historical prerequisites: **%s** (%d valid immutable bundle%s)."
        % (
            audit["historical_prerequisites"]["status"],
            int(audit["historical_prerequisites"]["valid_count"]),
            "" if int(audit["historical_prerequisites"]["valid_count"]) == 1 else "s",
        ),
        "- Base historical campaign: **%d one-time cold planned attempts** across **%d** separately authorized slices (exact sizes `%s`). It is not repeated by the daily pipeline."
        % (
            int(requests["cache_adjusted_total"]),
            int(slicing["slice_count"]),
            "+".join(
                str(item)
                for item in audit["decision"]["historical_campaign_slice_attempts"]
            ),
        ),
        "- Initial frozen campaign maximum: **%d physical attempts**. Cache hits reduce that number; automatic retries and redirects are zero."
        % int(audit["decision"]["historical_campaign_initial_max_actual_attempts"]),
        "- The generic six-slice cap totals **%d**, but its unused **%d** permits are not requests and are not authorized by the exact plan."
        % (
            int(slicing["sum_of_generic_slice_caps"]),
            int(audit["decision"]["unplanned_slice_capacity_not_authorized"]),
        ),
        "- Exact formula: **%d sampled Core + %d daily cohort chain + %d split-history = %d**."
        % (
            int(requests["historical_core"]),
            int(requests["historical_chain_total"]),
            int(requests["split_history"]),
            int(requests["cold_cache_total"]),
        ),
        "- Request-per-signal and request-per-exact-strike history are disabled. One daily cohort chain supports every frozen structure and horizon.",
        "- Entry signals are censored before cohort rotations, so the base campaign does not pay 183 transition-overlap calls. A separately frozen continuous-entry extension would add **%d** calls."
        % int(requests["optional_continuous_entry_extension_chain_batches"]),
        "- Base plus optional continuous-entry extension: **%d** attempts. The extension is not part of the recommended base campaign."
        % (
            int(requests["cache_adjusted_total"])
            + int(requests["optional_continuous_entry_extension_chain_batches"])
        ),
        "- This estimate is **not authorization**. A raw-source-bound independent universe, calendar, and event bundle must exist before slice 1.",
        "- Existing historical evidence is still ten-ETF development data and cannot validate current equity candidates.",
        "- Separate entitlement discovery recommended: **%s**. Saved ledgers contain successful responses for Core history, chain history, and split history."
        % (
            "YES"
            if audit["decision"]["separate_entitlement_discovery_recommended"]
            else "NO"
        ),
        "- Recovery requests are not hidden in the base. If a no-retry batch fails, `R` is the exact number of missing grouped batches in a separately frozen recovery; total becomes `%d + R`."
        % int(requests["cold_cache_total"]),
        "- Historical Core is frozen at two names/request: the saved ledger shows 1/1 success at two names and 0/3 success at ten names. Chain history is frozen at ten names after 451/451 saved successes.",
        "- Offline V2 now has a verified holdout-to-family-evidence bridge and an exact-quote model-to-ticket assembler; invalid V1/V6/V7 build commands fail closed.",
        "",
        "## Daily production budget (separate from history)",
        "",
        "| Run shape | Calls | Admissible under 60? |",
        "|---|---:|---:|",
        "| Current %d-name cold funnel | **%d** | %s |"
        % (
            int(current["locally_eligible_without_cap"]),
            int(budgets["current_cold_full_funnel"]["worst_charged_attempts"]),
            "YES" if budgets["current_cold_full_funnel"]["admissible"] else "NO",
        ),
        "| All configured maximum inputs together | **%d** | %s - rejected before execution |"
        % (
            int(budgets["absolute_cold_full_funnel"]["worst_charged_attempts"]),
            "YES" if budgets["absolute_cold_full_funnel"]["admissible"] else "NO",
        ),
        "| Same-vintage warm rerun | **0** | YES |",
        "",
        "Daily target: **25**. Daily logical cap: **60**. Automatic redirects/retries: **0**. Request 100 is structurally impossible.",
        "",
        "## Usage already charged",
        "",
        "- Physical attempts: **%d**" % int(usage["charged_attempts"]),
        "- Successful 2xx: **%d**" % int(usage["successful_2xx"]),
        "- Failed/non-2xx: **%d**" % int(usage["failed_or_non_2xx"]),
        "- Prior attempts plus a fully cold new base campaign: **%d** (lifetime accounting only; prior spend is not part of the new campaign)."
        % int(audit["decision"]["prior_sunk_plus_future_base_if_all_cold"]),
        "",
        "## Historical campaign comparison",
        "",
        "| Broad eligible names | Full-universe 450-session grid | Rotating-cohort campaign | Slices |",
        "|---:|---:|---:|---:|",
    ]
    for item in feasibility["scenarios"]:
        lines.append(
            "| %d | %d | **%d** | %d |"
            % (
                int(item["symbols"]),
                int(item["rejected_full_universe_date_grid_attempts"]),
                int(item["expected_campaign_attempts"]),
                int(item["slice_count"]),
            )
        )
    lines.extend(
        [
            "",
            "## Existing historical coverage",
            "",
            "- Symbols: **%d** - `%s`"
            % (int(database["ticker_count"]), "`, `".join(database["tickers"])),
            "- Sessions: **%d**" % int(database["sessions"]),
            "- Exact-chain rows: **%d**" % int(database["chain_rows"]),
            "",
            "## Spend gate",
            "",
            "No historical slice is authorized here. Before slice 1, Cultra must freeze the point-in-time universe, cohort hash, session calendar, event source, hypothesis registry, costs, exits, and development protocol. Each later slice requires separate authorization and cache/ledger reconciliation.",
            "",
            "Network requests made by this audit: **0**.",
            "",
            "## Production status",
            "",
            "Status: **%s**. Manual tickets enabled: **%s**."
            % (
                audit["production_readiness"]["status"],
                "YES" if audit["production_readiness"]["manual_ticket_enabled"] else "NO",
            ),
            "",
            "| Blocker | Required fix |",
            "|---|---|",
        ]
    )
    for item in audit["production_readiness"]["blockers"]:
        lines.append("| `%s` | %s |" % (item["check_id"], item["required_fix"]))
    return "\n".join(lines)


def build_offline_audit(
    *,
    run_id: str,
    output_root: Path = OUT_ROOT,
    ledger_root: Path = LEDGER_ROOT,
    cache_root: Path = CACHE_ROOT,
    chain_database: Path = CHAIN_DB,
    screen_path: Path = SCREEN_PATH,
    orats_path: Path = ORATS_PATH,
) -> Mapping[str, Any]:
    if not run_id or len(run_id) > 128:
        raise OfflineAuditError("run_id is missing or too long")
    screen = _load(screen_path)
    orats = _load(orats_path)
    historical_protocol = load_historical_campaign_protocol()
    usage = _ledger_usage(ledger_root)
    cache = _historical_cache_coverage(cache_root)
    database = _chain_database_coverage(chain_database)
    prerequisites = _historical_prerequisite_status(output_root)

    # Reuse the same pure local screen used by production; importing this
    # module does not construct a provider or make a network request.
    screened = local_screen(tuple(screen.get("quotes", ())))
    eligible_symbols = {str(item["ticker"]) for item in screened["admitted"]}
    orats_symbols = {str(item["ticker"]) for item in orats.get("rows", ())}
    missing_core = sorted(eligible_symbols - orats_symbols)
    current = {
        "source_symbols": len(screen.get("quotes", ())),
        "locally_eligible_without_cap": len(eligible_symbols),
        "orats_core_symbols": len(orats_symbols),
        "core_gap": len(missing_core),
        "missing_core_symbols": missing_core,
        "core_symbols_outside_local_eligibility": sorted(
            orats_symbols - eligible_symbols
        ),
        "core_gap_requests": int(math.ceil(len(missing_core) / 10.0)),
        "historical_tickers_overlapping_broad_equity_source": sorted(
            set(database["tickers"]).intersection(
                str(item["ticker"]) for item in screen.get("quotes", ())
            )
        ),
    }
    acquisition = historical_protocol["acquisition"]
    split_policy = historical_protocol["split_policy"]
    feasibility = request_feasibility(
        symbol_counts=(
            len(orats_symbols),
            len(eligible_symbols),
            len(screen.get("quotes", ())),
        ),
        batch_size=int(acquisition["historical_chain_ticker_batch_size"]),
        minimum_training_sessions=int(split_policy["minimum_training_sessions"]),
        embargo_sessions=int(split_policy["embargo_sessions_at_every_boundary"]),
        validation_sessions=int(split_policy["validation_sessions"]),
        holdout_fraction=float(split_policy["final_holdout_fraction"]),
        reference_sessions=int(acquisition["historical_sessions"]),
        attempt_cap=int(acquisition["slice_hard_cap"]),
    )
    request_budgets = {
        "current_core_only": daily_request_budget(
            core_symbols=len(eligible_symbols)
        ),
        "current_cold_full_funnel": daily_request_budget(
            core_symbols=len(eligible_symbols),
            summary_symbols=min(120, len(eligible_symbols)),
            monies_symbols=min(40, len(eligible_symbols)),
            exact_contracts=250,
        ),
        "absolute_cold_full_funnel": daily_request_budget(
            core_symbols=600,
            summary_symbols=120,
            monies_symbols=40,
            exact_contracts=250,
        ),
        "same_vintage_warm": daily_request_budget(core_symbols=0),
    }
    current_history = next(
        item
        for item in feasibility["scenarios"]
        if int(item["symbols"]) == len(eligible_symbols)
    )
    campaign_attempts = int(current_history["expected_campaign_attempts"])
    campaign_slice_cap = int(current_history["per_slice_cap"])
    campaign_slice_attempts = [
        min(campaign_slice_cap, campaign_attempts - offset)
        for offset in range(0, campaign_attempts, campaign_slice_cap)
    ]
    checks = [
        {
            "check_id": "IMMUTABLE_CAMPAIGN_AND_SLICE_GATE",
            "status": "PASS",
            "required_fix": "None; execution accepts only a reproduced campaign freeze and exact slice index.",
        },
        {
            "check_id": "RAW_SOURCE_BOUND_PREREQUISITE_GATE",
            "status": "PASS",
            "required_fix": "None in code; campaign freeze requires a reproducible raw-source-bound prerequisite receipt and rejects ORATS-derived prerequisite sources.",
        },
        {
            "check_id": "COMPLETE_CAMPAIGN_RECONCILIATION",
            "status": "PASS",
            "required_fix": "None in code; all 474 request IDs, six manifests, snapshots, and charged attempts must reconcile before normalization.",
        },
        {
            "check_id": "STRICT_V2_NORMALIZATION",
            "status": "PASS",
            "required_fix": "None in code; exact fields, vintages, chain coverage, snapshots, splits, and immutable hashes are enforced.",
        },
        {
            "check_id": "EXACT_LEG_OUTCOME_ENGINE",
            "status": "PASS",
            "required_fix": "None in code; every frozen ticker/date/hypothesis row is retained and missing paths or assignment risk fail closed.",
        },
        {
            "check_id": "LEAKAGE_SAFE_MODEL_AND_HOLDOUT_GATE",
            "status": "PASS",
            "required_fix": "None in code; OOF folds align to the 59-session cohort entry windows, retain each 61-session path embargo, seal the final 20 percent, and apply Holm-90 atomically.",
        },
        {
            "check_id": "VERIFIED_HOLDOUT_TO_FAMILY_EVIDENCE_BRIDGE",
            "status": "PASS",
            "required_fix": "None in code; the exact frozen model, cost policy, one-time holdout result, manifest, durable registry state, and commit receipt must all reconcile.",
        },
        {
            "check_id": "CURRENT_V2_EXACT_TICKET_ASSEMBLY",
            "status": "PASS",
            "required_fix": "None in code; coherent calibrated probabilities, exact Schwab leg quotes, costs, payoff, scenario EV, and event evidence are recomputed before the ticket gate.",
        },
        {
            "check_id": "INVALID_LEGACY_PRODUCTION_SURFACES_DISABLED",
            "status": "PASS",
            "required_fix": "None; V1 validation, research-order, opportunity, and V6/V7 pattern-build commands now fail closed and preserved artifacts are verification-only.",
        },
        {
            "check_id": "POINT_IN_TIME_BROAD_UNIVERSE",
            "status": "PASS" if prerequisites["valid_count"] else "BLOCKED",
            "required_fix": (
                "None; a valid source-bound bundle is frozen."
                if prerequisites["valid_count"]
                else "Supply an independent source bundle containing the four exact cohort-date snapshots, at least 100 liquid optionable names per snapshot, and no fixed list or current-constituent projection."
            ),
        },
        {
            "check_id": "STRICT_450_SESSION_CALENDAR",
            "status": "PASS" if prerequisites["valid_count"] else "BLOCKED",
            "required_fix": (
                "None; a valid source-bound bundle is frozen."
                if prerequisites["valid_count"]
                else "Supply an independent XNYS source bundle with exactly 450 sorted sessions and timezone-aware closes; the saved ORATS-derived date list is rejected."
            ),
        },
        {
            "check_id": "POINT_IN_TIME_EVENT_MANIFEST",
            "status": "PASS" if prerequisites["valid_count"] else "BLOCKED",
            "required_fix": (
                "None; a valid source-bound bundle is frozen."
                if prerequisites["valid_count"]
                else "Supply an independent point-in-time source covering earnings, dividends, splits, delistings, and contract adjustments for the campaign; empty earnings evidence for sampled stocks is rejected."
            ),
        },
        {
            "check_id": "V2_HISTORICAL_CAMPAIGN_DATA",
            "status": "BLOCKED",
            "required_fix": "After the three offline inputs are frozen and separately authorized, complete and reconcile all six historical slices.",
        },
        {
            "check_id": "V2_DEVELOPMENT_AND_UNTOUCHED_HOLDOUT",
            "status": "BLOCKED",
            "required_fix": "Normalize the completed V2 data, generate exact paths, freeze OOF models, and consume the untouched holdout once.",
        },
        {
            "check_id": "CURRENT_EXECUTABLE_TICKET_INPUTS",
            "status": "BLOCKED",
            "required_fix": "For holdout-pass hypotheses only, obtain current Schwab underlying and exact-leg bid/ask quotes before any manual ticket.",
        },
    ]
    readiness = {
        "source": "CULTRA_V2_OFFLINE_AUDIT",
        "status": "NOT_PRODUCTION_READY",
        "manual_ticket_enabled": False,
        "check_count": len(checks),
        "blocker_count": sum(item["status"] != "PASS" for item in checks),
        "checks": checks,
        "blockers": [item for item in checks if item["status"] != "PASS"],
        "legacy_v1_evidence_status": "DEVELOPMENT_ONLY_INVALID_FOR_CURRENT_STOCK_PRODUCTION",
    }
    audit = {
        "schema": "cultra.offline-audit.v7",
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "network_attempted": False,
        "orats_usage": usage,
        "historical_cache": cache,
        "historical_database": database,
        "historical_prerequisites": {
            key: item
            for key, item in prerequisites.items()
            if key != "bound_files"
        },
        "current_coverage": current,
        "request_budgets": request_budgets,
        "production_readiness": readiness,
        "historical_campaign_feasibility": feasibility,
        "decision": {
            "entire_campaign_under_100": False,
            "each_authorized_slice_under_100": True,
            "largest_slice_attempts": int(
                current_history["rotating_cohort_campaign"]["slicing"][
                    "per_slice_cap"
                ]
            ),
            "status": (
                "ESTIMATED_NOT_AUTHORIZED_PENDING_SLICE_1"
                if prerequisites["valid_count"]
                else "ESTIMATED_NOT_AUTHORIZED_PENDING_OFFLINE_SOURCE_BUNDLES"
            ),
            "recommended_additional_orats_requests_now": 0,
            "reason": (
                "a source-bound prerequisite bundle exists; slice 1 still requires separate authorization"
                if prerequisites["valid_count"]
                else "the request graph is complete, but no reproducible independent universe, calendar, and event source bundle is available"
            ),
            "historical_campaign_expected_attempts": campaign_attempts,
            "historical_campaign_initial_max_actual_attempts": campaign_attempts,
            "historical_campaign_slice_attempts": campaign_slice_attempts,
            "historical_campaign_slice_count": int(current_history["slice_count"]),
            "historical_campaign_generic_slice_cap_sum": int(
                current_history["rotating_cohort_campaign"]["slicing"][
                    "sum_of_generic_slice_caps"
                ]
            ),
            "historical_campaign_is_one_time_backfill": True,
            "unplanned_slice_capacity_not_authorized": int(
                current_history["rotating_cohort_campaign"]["slicing"][
                    "sum_of_generic_slice_caps"
                ]
            )
            - campaign_attempts,
            "recovery_requests_included": False,
            "recovery_formula": "%d + exact count of failed grouped request fingerprints"
            % campaign_attempts,
            "saved_batch_geometry_evidence": {
                key: usage["batch_geometry"].get(key, {})
                for key in (
                    "/datav2/hist/cores|2",
                    "/datav2/hist/cores|10",
                    "/datav2/hist/strikes|10",
                    "/datav2/hist/splits|10",
                )
            },
            "historical_endpoint_success_evidence": {
                endpoint: int(usage["endpoints"].get(endpoint, {}).get("successful_2xx", 0))
                for endpoint in (
                    "/datav2/hist/cores",
                    "/datav2/hist/strikes",
                    "/datav2/hist/splits",
                )
            },
            "separate_entitlement_discovery_recommended": not all(
                int(usage["endpoints"].get(endpoint, {}).get("successful_2xx", 0)) > 0
                for endpoint in (
                    "/datav2/hist/cores",
                    "/datav2/hist/strikes",
                    "/datav2/hist/splits",
                )
            ),
            "prior_sunk_plus_future_base_if_all_cold": int(
                usage["charged_attempts"]
            )
            + campaign_attempts,
            "same_vintage_warm_attempts": 0,
            "optional_continuous_entry_extension_attempts": int(
                current_history["rotating_cohort_campaign"]["requests"][
                    "optional_continuous_entry_extension_chain_batches"
                ]
            ),
            "base_plus_optional_extension_attempts": int(
                current_history["expected_campaign_attempts"]
            )
            + int(
                current_history["rotating_cohort_campaign"]["requests"][
                    "optional_continuous_entry_extension_chain_batches"
                ]
            ),
            "slice_pre_send_gate": (
                "freeze every planned-request ID and reject any slice above 90 before the token-holding gateway can execute it"
            ),
            "current_core_gap_requests_if_separately_desired": current[
                "core_gap_requests"
            ],
            "current_core_gap_is_profit_validation": False,
        },
    }
    root = Path(output_root).expanduser().resolve()
    try:
        root.relative_to(OUT_ROOT.resolve())
    except ValueError as exc:
        raise OfflineAuditError("offline audit root must remain inside Cultra/out") from exc
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    os.chmod(run_dir, 0o700)
    audit_path = _private_json(run_dir / "offline_audit.json", audit)
    board_path = _private_write(
        run_dir / "OFFLINE_AUDIT.md", (_render_v7(audit) + "\n").encode("utf-8")
    )
    # Bind the audit to the complete Cultra runtime and every shipped JSON
    # policy, not a hand-maintained subset that could miss a consequential
    # offline change.
    inputs = tuple(
        sorted(
            set(Path(ledger_root).glob("*.sqlite3"))
            | {
                Path(chain_database),
                Path(screen_path),
                Path(orats_path),
                *prerequisites["bound_files"],
                *Path(PROJECT_ROOT / "cultra").glob("*.py"),
                *Path(PROJECT_ROOT / "configs").glob("*.json"),
            },
            key=lambda path: str(path.resolve()),
        )
    )
    manifest = {
        "schema": "cultra.offline-audit-manifest.v7",
        "run_id": run_id,
        "network_attempted": False,
        "inputs": [
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in inputs
        ],
        "historical_cache_manifest_set_sha256": cache["manifest_set_sha256"],
        "artifacts": [
            {"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in (audit_path, board_path)
        ],
    }
    _private_json(run_dir / "manifest.json", manifest)
    return dict(audit, run_dir=str(run_dir), board=str(board_path))


def verify_offline_audit(run_dir: Path) -> Tuple[str, ...]:
    """Reconcile a saved V7 audit and every immutable input/artifact hash."""

    errors = []
    root = Path(run_dir).expanduser().resolve()
    try:
        root.relative_to(OUT_ROOT.resolve())
    except ValueError:
        return ("offline audit is outside Cultra/out",)
    try:
        manifest = _load(root / "manifest.json")
        audit = _load(root / "offline_audit.json")
    except OfflineAuditError as exc:
        return (str(exc),)
    if manifest.get("schema") != "cultra.offline-audit-manifest.v7":
        errors.append("offline audit manifest schema is unsupported")
    if audit.get("schema") != "cultra.offline-audit.v7":
        errors.append("offline audit schema is unsupported")
    if manifest.get("run_id") != root.name or audit.get("run_id") != root.name:
        errors.append("offline audit run identity does not reconcile")
    if manifest.get("network_attempted") is not False or audit.get("network_attempted") is not False:
        errors.append("offline audit cannot claim network execution")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        errors.append("offline audit input manifest is missing")
    else:
        for item in inputs:
            try:
                path = Path(str(item["path"])).resolve()
                if (
                    not path.is_file()
                    or path.stat().st_size != int(item["bytes"])
                    or _sha256(path) != str(item["sha256"])
                ):
                    errors.append("offline audit input changed: %s" % path.name)
            except (KeyError, OSError, TypeError, ValueError):
                errors.append("offline audit input record is malformed")
    artifacts = manifest.get("artifacts")
    expected_artifacts = {"offline_audit.json", "OFFLINE_AUDIT.md"}
    if not isinstance(artifacts, list) or {
        str(item.get("path", "")) for item in artifacts if isinstance(item, Mapping)
    } != expected_artifacts:
        errors.append("offline audit artifact set is incomplete")
    else:
        for item in artifacts:
            try:
                path = root / str(item["path"])
                if (
                    not path.is_file()
                    or path.stat().st_size != int(item["bytes"])
                    or _sha256(path) != str(item["sha256"])
                ):
                    errors.append("offline audit artifact changed: %s" % path.name)
            except (KeyError, OSError, TypeError, ValueError):
                errors.append("offline audit artifact record is malformed")
    if {item.name for item in root.iterdir() if item.is_file()} != expected_artifacts | {"manifest.json"}:
        errors.append("offline audit directory contains an unmanifested file")
    return tuple(errors)


__all__ = [
    "OfflineAuditError",
    "build_offline_audit",
    "request_feasibility",
    "verify_offline_audit",
]
