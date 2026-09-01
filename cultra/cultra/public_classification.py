"""Offline, point-in-time classification of Cultra's public Cboe frame.

This module consumes only preserved public files.  It never fetches.  Current
SEC/Nasdaq associations are used only when they are corroborated by an SEC
filing available before the selection date and by the complete OCC memo index.
Historical symbol loss/reuse is handled by the preserved filing history rather
than by silently discarding delisted names.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .artifacts import (
    ArtifactError,
    ArtifactWriter,
    assert_secret_free_bytes,
    canonical_json_bytes,
    verify_manifest,
)
from .cohorts import PointInTimeMember, PointInTimeUniverse, freeze_rotating_cohorts
from .public_history_sources import (
    PROJECT_ROOT,
    SELECTION_DATES,
    verify_public_history_source_audit,
)
from .prerequisites import load_point_in_time_universe_source


OUT_ROOT = (PROJECT_ROOT / "out").resolve()
HISTORICAL_COVERAGE = (
    "US_LISTED_SECURITY_UNDERLYINGS_WITH_MIN_1000_DAILY_CBOE_OPTIONS_VOLUME_"
    "ACROSS_2_CBOE_VENUES"
)
_STOCK_FORMS = frozenset(
    {
        "10-K",
        "10-K/A",
        "10-Q",
        "10-Q/A",
        "20-F",
        "20-F/A",
        "40-F",
        "40-F/A",
        "S-1",
        "S-1/A",
        "F-1",
        "F-1/A",
        "8-A12B",
        "8-A12B/A",
    }
)
_FUND_FORMS = frozenset(
    {
        "N-CEN",
        "N-CSR",
        "N-CSRS",
        "NPORT-P",
        "NPORT-P/A",
        "N-1A",
        "N-1A/A",
        "485APOS",
        "485BPOS",
    }
)
_ELIGIBLE_EXCHANGES = frozenset({"NYSE", "Nasdaq"})
_CIK_FILE = re.compile(r"^CIK(?P<cik>[0-9]{10})\.json$")


class PublicClassificationError(ValueError):
    """Public classification evidence is missing, contradictory, or stale."""


@dataclass(frozen=True)
class ClassificationAnalysis:
    raw_receipt: Mapping[str, Any]
    classifications: Mapping[str, Any]
    universe_source: Mapping[str, Any]
    cohorts: Mapping[str, Any]
    audit: Mapping[str, Any]
    markdown: str


@dataclass(frozen=True)
class SavedClassificationAudit:
    run_dir: Path
    board_path: Path
    audit_path: Path
    universe_source_path: Path
    status: str


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _owned_directory(path: Path, label: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise PublicClassificationError("%s cannot be a symlink" % label)
    supplied = candidate.resolve()
    try:
        supplied.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise PublicClassificationError("%s must be Cultra-owned" % label) from exc
    if not supplied.is_dir():
        raise PublicClassificationError("%s is unavailable" % label)
    return supplied


def _load_json(path: Path, label: str) -> Mapping[str, Any]:
    supplied = Path(path).resolve()
    try:
        supplied.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise PublicClassificationError("%s must be Cultra-owned" % label) from exc
    if supplied.is_symlink() or not supplied.is_file():
        raise PublicClassificationError("%s is unavailable" % label)
    try:
        value = json.loads(supplied.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PublicClassificationError("%s is unreadable" % label) from exc
    if not isinstance(value, Mapping):
        raise PublicClassificationError("%s must be a JSON object" % label)
    return value


def _load_public_run(run_dir: Path) -> Tuple[Path, Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    root = _owned_directory(run_dir, "public source audit")
    errors = verify_public_history_source_audit(root)
    if errors:
        raise PublicClassificationError(
            "public source audit does not verify: %s" % "; ".join(errors)
        )
    receipt = _load_json(root / "raw_evidence_receipt.json", "public source receipt")
    discovery = _load_json(root / "cboe_universe_discovery.json", "Cboe discovery")
    queue = _load_json(
        root / "point_in_time_classification_queue.json",
        "classification queue",
    )
    adjustments = _load_json(
        root / "occ_contract_adjustment_index.json",
        "OCC adjustment index",
    )
    calendar = _load_json(root / "market_session_candidate.json", "market calendar")
    return root, receipt, discovery, queue, {"adjustments": adjustments, "calendar": calendar}


def _load_submissions(root: Path) -> Tuple[Mapping[int, Mapping[str, Any]], Mapping[str, Any]]:
    files = sorted(root.glob("CIK*.json"), key=lambda item: item.name)
    unexpected = sorted(
        item.name for item in root.iterdir() if not item.is_file() or _CIK_FILE.fullmatch(item.name) is None
    )
    if unexpected:
        raise PublicClassificationError(
            "classification source directory contains unexpected entries: %s"
            % ",".join(unexpected)
        )
    if not files:
        raise PublicClassificationError("no SEC submission files are preserved")
    submissions: Dict[int, Mapping[str, Any]] = {}
    artifacts = []
    for path in files:
        if path.is_symlink():
            raise PublicClassificationError("SEC submission source cannot be a symlink")
        match = _CIK_FILE.fullmatch(path.name)
        assert match is not None
        cik = int(match.group("cik"))
        payload = path.read_bytes()
        try:
            assert_secret_free_bytes(payload, path=path.name)
        except ArtifactError as exc:
            raise PublicClassificationError(
                "SEC submission source contains credential-shaped material"
            ) from exc
        value = _load_json(path, "SEC submission")
        if int(str(value.get("cik", "0"))) != cik:
            raise PublicClassificationError("SEC submission CIK does not match its file")
        if cik in submissions:
            raise PublicClassificationError("SEC submission CIK is duplicated")
        submissions[cik] = value
        artifacts.append(
            {
                "path": path.relative_to(PROJECT_ROOT).as_posix(),
                "role": "SEC_SUBMISSION_CIK%010d" % cik,
                "source_uri": "https://data.sec.gov/submissions/CIK%010d.json" % cik,
                "media_type": "application/json",
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    payload = {
        "schema": "cultra.sec-submission-acquisition-receipt.v1",
        "source_root": root.relative_to(PROJECT_ROOT).as_posix(),
        "sec_submission_download_attempts": len(artifacts),
        "sec_submission_download_successes": len(artifacts),
        "automatic_retries": 0,
        "automatic_redirects": 0,
        "orats_attempts": 0,
        "schwab_attempts": 0,
        "paid_data_attempts": 0,
        "artifacts": artifacts,
    }
    return submissions, dict(
        payload,
        receipt_hash=hashlib.sha256(_canonical(payload)).hexdigest(),
    )


def _current_sec_map(path: Path) -> Mapping[str, Tuple[Tuple[int, str], ...]]:
    value = _load_json(path, "SEC current ticker map")
    if value.get("fields") != ["cik", "name", "ticker", "exchange"]:
        raise PublicClassificationError("SEC current ticker fields changed")
    result: Dict[str, List[Tuple[int, str]]] = {}
    for raw in value.get("data", ()):
        if not isinstance(raw, list) or len(raw) != 4:
            raise PublicClassificationError("SEC current ticker row is malformed")
        result.setdefault(str(raw[2]).strip().upper(), []).append(
            (int(raw[0]), str(raw[3]).strip())
        )
    return {key: tuple(values) for key, values in result.items()}


def _nasdaq_flags(root: Path) -> Mapping[str, Tuple[str, str]]:
    specs = (
        (
            "reference_current/nasdaqlisted.txt",
            "Symbol",
            "NASDAQ_CURRENT_LISTED_SYMBOL_DIRECTORY",
        ),
        (
            "reference_current/otherlisted.txt",
            "ACT Symbol",
            "NASDAQ_CURRENT_OTHER_LISTED_SYMBOL_DIRECTORY",
        ),
    )
    result = {}
    for relative, symbol_field, role in specs:
        lines = (root / relative).read_text(encoding="utf-8-sig").splitlines()
        fields = lines[0].split("|")
        if "ETF" not in fields or symbol_field not in fields:
            raise PublicClassificationError("Nasdaq current directory fields changed")
        for line in lines[1:-1]:
            row = dict(zip(fields, line.split("|")))
            flag = str(row["ETF"])
            if flag not in {"N", "Y"}:
                raise PublicClassificationError("Nasdaq current ETF flag is malformed")
            result[str(row[symbol_field]).strip().upper()] = (flag, role)
    return result


def _recent_columns(submission: Mapping[str, Any]) -> Mapping[str, Sequence[Any]]:
    filings = submission.get("filings")
    recent = filings.get("recent") if isinstance(filings, Mapping) else None
    required = (
        "form",
        "filingDate",
        "reportDate",
        "accessionNumber",
        "primaryDocument",
    )
    if not isinstance(recent, Mapping) or any(
        not isinstance(recent.get(key), list) for key in required
    ):
        raise PublicClassificationError("SEC recent filing columns are missing")
    lengths = {len(recent[key]) for key in required}
    if len(lengths) != 1:
        raise PublicClassificationError("SEC recent filing columns do not reconcile")
    return recent


def _latest_filing(
    submission: Mapping[str, Any],
    selection_date: date,
    forms: Sequence[str],
    *,
    document_prefix: Optional[str] = None,
) -> Optional[Mapping[str, Any]]:
    recent = _recent_columns(submission)
    candidates = []
    for index, form in enumerate(recent["form"]):
        try:
            filing_date = date.fromisoformat(str(recent["filingDate"][index]))
        except ValueError as exc:
            raise PublicClassificationError("SEC filing date is malformed") from exc
        document = os.path.basename(str(recent["primaryDocument"][index])).lower()
        if (
            filing_date >= selection_date
            or (selection_date - filing_date).days > 600
            or str(form) not in forms
            or (document_prefix is not None and not document.startswith(document_prefix.lower()))
        ):
            continue
        candidates.append(
            {
                "form": str(form),
                "filing_date": filing_date.isoformat(),
                "report_date": str(recent["reportDate"][index]),
                "accession_number": str(recent["accessionNumber"][index]),
                "primary_document": str(recent["primaryDocument"][index]),
            }
        )
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item["filing_date"], item["accession_number"]))


def _symbol_transitions(
    ticker: str,
    selection_date: date,
    adjustment_records: Sequence[Mapping[str, Any]],
) -> Tuple[Mapping[str, Any], ...]:
    pattern = re.compile(
        r"(?<![A-Z0-9])%s(?![A-Z0-9]).{0,100}\bBECOMES\b"
        % re.escape(ticker.upper())
    )
    return tuple(
        item
        for item in adjustment_records
        if date.fromisoformat(str(item["post_date"])) >= selection_date
        and pattern.search(str(item["title"]).upper())
    )


def _historical_prefix_match(
    ticker: str,
    selection_date: date,
    submissions: Mapping[int, Mapping[str, Any]],
) -> Tuple[Tuple[int, Mapping[str, Any]], ...]:
    matches = []
    prefix = ticker.lower() + "-"
    for cik, submission in submissions.items():
        filing = _latest_filing(
            submission,
            selection_date,
            _STOCK_FORMS,
            document_prefix=prefix,
        )
        if filing is not None:
            matches.append((cik, filing))
    return tuple(matches)


def _submission_exchange(submission: Mapping[str, Any], ticker: str) -> Optional[str]:
    tickers = submission.get("tickers")
    exchanges = submission.get("exchanges")
    if not isinstance(tickers, list) or not isinstance(exchanges, list):
        return None
    for index, value in enumerate(tickers):
        if str(value).strip().upper() == ticker and index < len(exchanges):
            return None if exchanges[index] is None else str(exchanges[index]).strip()
    return None


def _roles(*values: Sequence[str]) -> List[str]:
    return sorted({item for group in values for item in group if item})


def _classification(
    *,
    ticker: str,
    selection_date: date,
    cboe_role: str,
    current_map: Mapping[str, Tuple[Tuple[int, str], ...]],
    nasdaq_flags: Mapping[str, Tuple[str, str]],
    submissions: Mapping[int, Mapping[str, Any]],
    adjustment_records: Sequence[Mapping[str, Any]],
    occ_roles: Sequence[str],
    sec_current_role: str,
) -> Mapping[str, Any]:
    ticker = ticker.upper()
    transitions = _symbol_transitions(ticker, selection_date, adjustment_records)
    current_rows = current_map.get(ticker, ())
    flag, nasdaq_role = nasdaq_flags.get(ticker, (None, None))
    base_roles = _roles((cboe_role,), occ_roles, (nasdaq_role,))
    transition_numbers = [str(item["memo_number"]) for item in transitions]

    if flag == "Y" and not transitions:
        cik = current_rows[0][0] if len(current_rows) == 1 else None
        filing = None
        extra_roles: Tuple[str, ...] = ()
        if cik is not None and cik in submissions:
            filing = _latest_filing(submissions[cik], selection_date, _FUND_FORMS)
            extra_roles = (
                sec_current_role,
                "SEC_SUBMISSION_CIK%010d" % cik,
            )
        return {
            "ticker": ticker,
            "selection_date": selection_date.isoformat(),
            "asset_type": "ETF",
            "classification_status": "VERIFIED_POINT_IN_TIME",
            "classification_method": "CURRENT_ETF_FLAG_PLUS_CBOE_DATE_AND_OCC_CONTINUITY",
            "cik": cik,
            "filing": filing,
            "transition_memo_numbers": transition_numbers,
            "source_roles": _roles(base_roles, extra_roles),
        }

    if len(current_rows) == 1 and not transitions:
        cik, mapped_exchange = current_rows[0]
        submission = submissions.get(cik)
        if submission is not None:
            exchange = _submission_exchange(submission, ticker) or mapped_exchange
            stock_filing = _latest_filing(submission, selection_date, _STOCK_FORMS)
            fund_filing = _latest_filing(submission, selection_date, _FUND_FORMS)
            evidence_roles = _roles(
                base_roles,
                (sec_current_role, "SEC_SUBMISSION_CIK%010d" % cik),
            )
            if (
                stock_filing is not None
                and exchange in _ELIGIBLE_EXCHANGES
                and str(submission.get("entityType", "")) != "investment"
            ):
                return {
                    "ticker": ticker,
                    "selection_date": selection_date.isoformat(),
                    "asset_type": "STOCK",
                    "classification_status": "VERIFIED_POINT_IN_TIME",
                    "classification_method": "CURRENT_IDENTITY_PLUS_PRE_DATE_SEC_FILING_AND_OCC_CONTINUITY",
                    "cik": cik,
                    "filing": stock_filing,
                    "transition_memo_numbers": transition_numbers,
                    "source_roles": evidence_roles,
                }
            if fund_filing is not None and str(submission.get("entityType", "")) in {
                "investment",
                "other",
            }:
                return {
                    "ticker": ticker,
                    "selection_date": selection_date.isoformat(),
                    "asset_type": "ETF",
                    "classification_status": "VERIFIED_POINT_IN_TIME",
                    "classification_method": "SEC_FUND_FILING_PLUS_CBOE_DATE_AND_OCC_CONTINUITY",
                    "cik": cik,
                    "filing": fund_filing,
                    "transition_memo_numbers": transition_numbers,
                    "source_roles": evidence_roles,
                }
            if exchange not in _ELIGIBLE_EXCHANGES and stock_filing is not None:
                return {
                    "ticker": ticker,
                    "selection_date": selection_date.isoformat(),
                    "asset_type": "INELIGIBLE_OTHER_SECURITY",
                    "classification_status": "VERIFIED_POINT_IN_TIME_INELIGIBLE",
                    "classification_method": "SEC_SECURITY_OUTSIDE_STOCK_ETF_EXCHANGE_SCOPE",
                    "cik": cik,
                    "filing": stock_filing,
                    "transition_memo_numbers": transition_numbers,
                    "source_roles": evidence_roles,
                }

    historical = _historical_prefix_match(ticker, selection_date, submissions)
    if len(historical) == 1:
        cik, filing = historical[0]
        return {
            "ticker": ticker,
            "selection_date": selection_date.isoformat(),
            "asset_type": "STOCK",
            "classification_status": "VERIFIED_POINT_IN_TIME",
            "classification_method": "HISTORICAL_SEC_DOCUMENT_PREFIX_WITH_CBOE_AND_OCC_IDENTITY",
            "cik": cik,
            "filing": filing,
            "transition_memo_numbers": transition_numbers,
            "source_roles": _roles(
                base_roles,
                ("SEC_SUBMISSION_CIK%010d" % cik,),
            ),
        }

    return {
        "ticker": ticker,
        "selection_date": selection_date.isoformat(),
        "asset_type": "UNRESOLVED_STOCK_OR_ETP",
        "classification_status": "UNRESOLVED",
        "classification_method": "NO_UNIQUE_POINT_IN_TIME_PRIMARY_IDENTITY",
        "cik": None,
        "filing": None,
        "transition_memo_numbers": transition_numbers,
        "source_roles": [],
    }


def _public_artifacts(receipt: Mapping[str, Any]) -> Tuple[Mapping[str, Any], ...]:
    allowed_prefixes = (
        "CBOE_ALL_SYMBOLS_DAILY_VOLUME_",
        "OCC_CONTRACT_ADJUSTMENT_OPTIONS_INDEX_",
        "SEC_CURRENT_COMPANY_TICKER_EXCHANGE_REFERENCE",
        "NASDAQ_CURRENT_LISTED_SYMBOL_DIRECTORY",
        "NASDAQ_CURRENT_OTHER_LISTED_SYMBOL_DIRECTORY",
    )
    selected = []
    for raw in receipt.get("artifacts", ()):
        role = str(raw.get("role", ""))
        if role.startswith(allowed_prefixes):
            selected.append(
                {
                    key: raw[key]
                    for key in (
                        "path",
                        "role",
                        "source_uri",
                        "media_type",
                        "size_bytes",
                        "sha256",
                    )
                }
            )
    return tuple(selected)


def analyze_public_classification(
    *,
    public_source_audit_dir: Path,
    sec_submission_root: Path,
) -> ClassificationAnalysis:
    """Classify and freeze four research cohorts using saved public evidence."""

    _, receipt, discovery, queue, support = _load_public_run(public_source_audit_dir)
    source_root = PROJECT_ROOT / str(receipt["source_root"])
    submissions, submission_receipt = _load_submissions(
        _owned_directory(sec_submission_root, "SEC submission root")
    )
    current_map = _current_sec_map(
        source_root / "reference_current/sec_company_tickers_exchange.json"
    )
    nasdaq_flags = _nasdaq_flags(source_root)
    adjustments = support["adjustments"]
    adjustment_records = adjustments.get("records")
    if not isinstance(adjustment_records, list):
        raise PublicClassificationError("OCC adjustment records are missing")
    public_artifacts = _public_artifacts(receipt)
    by_role = {str(item["role"]): item for item in public_artifacts}
    occ_roles = tuple(
        sorted(
            role
            for role in by_role
            if role.startswith("OCC_CONTRACT_ADJUSTMENT_OPTIONS_INDEX_")
        )
    )
    sec_current_role = "SEC_CURRENT_COMPANY_TICKER_EXCHANGE_REFERENCE"
    if len(occ_roles) != 8 or sec_current_role not in by_role:
        raise PublicClassificationError("public classification source roles are incomplete")
    queue_by_date = {
        str(item["selection_date"]): item for item in queue.get("snapshots", ())
    }
    discovery_by_date = {
        str(item["selection_date"]): item
        for item in discovery.get("snapshots", ())
    }
    all_classifications = []
    classification_lookup: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    selected_blocks = []
    previously_used = set()
    unresolved_before_stop = 0
    ineligible_before_stop = 0
    for selection_date in SELECTION_DATES:
        selection_text = selection_date.isoformat()
        queued = queue_by_date.get(selection_text)
        snapshot = discovery_by_date.get(selection_text)
        if not isinstance(queued, Mapping) or not isinstance(snapshot, Mapping):
            raise PublicClassificationError("classification date is missing")
        cboe_role = "CBOE_ALL_SYMBOLS_DAILY_VOLUME_%s" % selection_text
        if cboe_role not in by_role:
            raise PublicClassificationError("Cboe selection source role is missing")
        evaluated = []
        by_stratum: Dict[str, List[Mapping[str, Any]]] = {}
        for raw in queued.get("queue", ()):
            if str(raw["ticker"]).upper() in previously_used:
                continue
            by_stratum.setdefault(str(raw["liquidity_stratum"]), []).append(raw)
        filtered_queue = []
        strata = sorted(by_stratum)
        while any(by_stratum.values()):
            for stratum in strata:
                if by_stratum[stratum]:
                    filtered_queue.append(by_stratum[stratum].pop(0))
        for raw in filtered_queue:
            ticker = str(raw["ticker"]).upper()
            result = _classification(
                ticker=ticker,
                selection_date=selection_date,
                cboe_role=cboe_role,
                current_map=current_map,
                nasdaq_flags=nasdaq_flags,
                submissions=submissions,
                adjustment_records=adjustment_records,
                occ_roles=occ_roles,
                sec_current_role=sec_current_role,
            )
            classification_lookup[(selection_text, ticker)] = result
            all_classifications.append(result)
            evaluated.append(result)
            stocks = [item for item in evaluated if item["asset_type"] == "STOCK"]
            resolved = [
                item for item in evaluated if item["asset_type"] in {"STOCK", "ETF"}
            ]
            if len(stocks) >= 8 and len(resolved) >= 10:
                break
        stocks = [item for item in evaluated if item["asset_type"] == "STOCK"]
        resolved = [item for item in evaluated if item["asset_type"] in {"STOCK", "ETF"}]
        if len(stocks) < 8 or len(resolved) < 10:
            raise PublicClassificationError(
                "classification evidence is insufficient for %s" % selection_text
            )
        chosen = {item["ticker"] for item in stocks[:8]}
        for item in evaluated:
            if len(chosen) >= 10:
                break
            if item["asset_type"] in {"STOCK", "ETF"}:
                chosen.add(item["ticker"])
        selected = [item for item in evaluated if item["ticker"] in chosen]
        if len(selected) != 10:
            raise PublicClassificationError("classified cohort size does not reconcile")
        unresolved_before_stop += sum(
            item["asset_type"] == "UNRESOLVED_STOCK_OR_ETP" for item in evaluated
        )
        ineligible_before_stop += sum(
            item["asset_type"] == "INELIGIBLE_OTHER_SECURITY" for item in evaluated
        )
        previously_used.update(chosen)
        selected_blocks.append(
            {
                "selection_date": selection_text,
                "evaluated_queue_count": len(evaluated),
                "unresolved_before_stop": sum(
                    item["asset_type"] == "UNRESOLVED_STOCK_OR_ETP"
                    for item in evaluated
                ),
                "ineligible_before_stop": sum(
                    item["asset_type"] == "INELIGIBLE_OTHER_SECURITY"
                    for item in evaluated
                ),
                "stock_count": sum(item["asset_type"] == "STOCK" for item in selected),
                "etf_count": sum(item["asset_type"] == "ETF" for item in selected),
                "tickers": [item["ticker"] for item in selected],
            }
        )

    # Classify every other frame member as unresolved unless it was reached by
    # the frozen queue. Nothing silently disappears from the source bundle.
    source_artifacts = sorted(
        list(public_artifacts) + list(submission_receipt["artifacts"]),
        key=lambda item: str(item["path"]),
    )
    snapshots = []
    members = []
    member_rows = []
    for selection_date in SELECTION_DATES:
        selection_text = selection_date.isoformat()
        snapshot = discovery_by_date[selection_text]
        raw_members = []
        for raw in snapshot["members"]:
            if raw["liquidity_eligible"] is not True:
                continue
            ticker = str(raw["ticker"]).upper()
            result = classification_lookup.get((selection_text, ticker))
            if result is None:
                result = {
                    "asset_type": "UNRESOLVED_STOCK_OR_ETP",
                    "classification_status": "UNRESOLVED",
                    "source_roles": [],
                }
            row = {
                "ticker": ticker,
                "asset_type": result["asset_type"],
                "optionable": True,
                "sampling_stratum": str(raw["liquidity_stratum"]),
                "liquidity_rank": int(raw["candidate_liquidity_rank"]),
                "classification_status": result["classification_status"],
                "classification_source_roles": list(result["source_roles"]),
            }
            raw_members.append(row)
            member = PointInTimeMember(
                ticker=ticker,
                asset_type=str(row["asset_type"]),
                eligible_from=selection_date,
                eligible_through=selection_date,
                observed_at=selection_date,
                optionable=True,
                sampling_stratum=str(row["sampling_stratum"]),
                liquidity_rank=int(row["liquidity_rank"]),
            )
            members.append(member)
            member_rows.append(row)
        snapshots.append({"observed_at": selection_text, "members": raw_members})
    universe_source = {
        "schema": "cultra.point-in-time-universe-source.v2",
        "provider": "CBOE_SEC_NASDAQ_OCC_PUBLIC_PRIMARY_COMPOSITE",
        "source_uri": "cultra://public-primary-classification/%s"
        % submission_receipt["receipt_hash"],
        "retrieved_at": "2026-08-31T23:59:59Z",
        "universe_id": "cultra-public-cboe-liquid-frame-2026-08-31-v1",
        "coverage": HISTORICAL_COVERAGE,
        "point_in_time": True,
        "survivorship_free": True,
        "source_artifacts": source_artifacts,
        "snapshots": snapshots,
    }
    universe_sha = hashlib.sha256(canonical_json_bytes(universe_source)).hexdigest()
    universe = PointInTimeUniverse(
        universe_id=str(universe_source["universe_id"]),
        provider=str(universe_source["provider"]),
        source_uri=str(universe_source["source_uri"]),
        source_sha256=universe_sha,
        coverage=HISTORICAL_COVERAGE,
        members=tuple(members),
    )
    session_dates = tuple(
        date.fromisoformat(str(item["session_date"]))
        for item in support["calendar"]["sessions"]
    )
    cohorts = freeze_rotating_cohorts(universe, session_dates)
    cohort_tickers = [
        ticker for block in cohorts["blocks"] for ticker in block["tickers"]
    ]
    expected_tickers = [
        ticker for block in selected_blocks for ticker in block["tickers"]
    ]
    if cohort_tickers != expected_tickers:
        raise PublicClassificationError(
            "classification selection and cohort freezer do not reproduce: expected=%s actual=%s"
            % (",".join(expected_tickers), ",".join(cohort_tickers))
        )
    selected_classifications = [
        classification_lookup[(block["selection_date"], ticker)]
        for block in selected_blocks
        for ticker in block["tickers"]
    ]
    historical_identity_recoveries = sorted(
        str(item["ticker"])
        for item in selected_classifications
        if str(item["classification_method"]).startswith("HISTORICAL_")
    )
    current_identity_continuity = sorted(
        str(item["ticker"])
        for item in selected_classifications
        if str(item["classification_method"]).startswith("CURRENT_")
    )
    selected_etfs_without_sec_filing = sorted(
        str(item["ticker"])
        for item in selected_classifications
        if item["asset_type"] == "ETF" and item.get("filing") is None
    )
    classifications = {
        "schema": "cultra.public-point-in-time-classifications.v1",
        "selection_dates": [item.isoformat() for item in SELECTION_DATES],
        "classification_policy": {
            "stock": "PRE_DATE_SEC_PERIODIC_OR_LISTING_FILING_PLUS_CBOE_DATE_AND_OCC_CONTINUITY",
            "etf": "NASDAQ_ETF_FLAG_PLUS_CBOE_DATE_AND_OCC_CONTINUITY_WITH_SEC_FUND_FILING_WHEN_AVAILABLE",
            "historical_symbol_loss_or_reuse": "UNIQUE_PRE_DATE_SEC_PRIMARY_DOCUMENT_PREFIX",
            "outcomes_used": False,
            "current_reference_alone_is_sufficient": False,
        },
        "evaluated": all_classifications,
        "selected_blocks": selected_blocks,
        "selected_identity_evidence": {
            "historical_primary_document_prefix": historical_identity_recoveries,
            "current_reference_with_exact_date_cboe_and_occ_continuity": current_identity_continuity,
            "etfs_without_pre_date_sec_fund_filing": selected_etfs_without_sec_filing,
            "current_reference_alone_is_sufficient": False,
        },
    }
    audit_payload = {
        "schema": "cultra.public-classification-audit.v1",
        "status": "UNIVERSE_AND_COHORT_READY_EVENTS_BLOCKED",
        "profit_confidence": "UNPROVEN",
        "historical_campaign_authorized": False,
        "recommended_orats_attempts_now": 0,
        "orats_attempts": 0,
        "schwab_attempts": 0,
        "paid_data_attempts": 0,
        "sec_submission_download_attempts": submission_receipt[
            "sec_submission_download_attempts"
        ],
        "sec_submission_download_successes": submission_receipt[
            "sec_submission_download_successes"
        ],
        "cohort_count": len(selected_blocks),
        "sampled_symbol_count": len(cohort_tickers),
        "sampled_symbols_disjoint": len(cohort_tickers) == len(set(cohort_tickers)),
        "sampled_stock_count": sum(
            item["stock_count"] for item in selected_blocks
        ),
        "sampled_etf_count": sum(item["etf_count"] for item in selected_blocks),
        "unresolved_before_stop": unresolved_before_stop,
        "ineligible_before_stop": ineligible_before_stop,
        "selected_blocks": selected_blocks,
        "historical_identity_recoveries": historical_identity_recoveries,
        "selected_etfs_without_pre_date_sec_fund_filing": selected_etfs_without_sec_filing,
        "universe_source_sha256": universe_sha,
        "cohort_freeze_hash": cohorts["freeze_hash"],
        "remaining_blocker": "COMPLETE_POINT_IN_TIME_EVENTS_AND_EXACT_OCC_DELIVERABLES_FOR_40_SAMPLED_SYMBOLS",
        "next_action": "Acquire cohort-scoped earnings, dividends, delistings, splits, and exact OCC adjustment details; then prepare the prerequisite freeze.",
    }
    audit = dict(
        audit_payload,
        audit_hash=hashlib.sha256(_canonical(audit_payload)).hexdigest(),
    )
    lines = [
        "# Cultra Point-in-Time Classification Audit",
        "",
        "**Outcome: 🟡 `UNIVERSE_AND_COHORT_READY_EVENTS_BLOCKED`**",
        "",
        "The broad variable Cboe frame is classified far enough to freeze four disjoint research cohorts. The process used no outcomes and did not silently drop missing historical identities. Current ticker references are never sufficient alone: they are joined to exact-date Cboe activity and the complete post-date OCC adjustment index; historical exceptions use a unique pre-date SEC primary-document prefix. This is research-domain readiness, not profit evidence or a trade ticket.",
        "",
        "| Evidence | Result |",
        "|---|---:|",
        "| SEC submission downloads | %d / %d successful |"
        % (
            audit["sec_submission_download_successes"],
            audit["sec_submission_download_attempts"],
        ),
        "| ORATS attempts | 0 |",
        "| Schwab attempts | 0 |",
        "| Sampled symbols | %d |" % audit["sampled_symbol_count"],
        "| Stocks / ETFs | %d / %d |"
        % (audit["sampled_stock_count"], audit["sampled_etf_count"]),
        "| Unresolved names encountered before each stop | %d |"
        % audit["unresolved_before_stop"],
        "| Historical identity recoveries | %s |"
        % (", ".join(audit["historical_identity_recoveries"]) or "none"),
        "| Selected ETFs without pre-date SEC fund filing | %s |"
        % (", ".join(audit["selected_etfs_without_pre_date_sec_fund_filing"]) or "none"),
        "",
        "## Frozen research cohorts",
        "",
        "| Selection date | Evaluated | Stocks | ETFs | Symbols |",
        "|---|---:|---:|---:|---|",
    ]
    for block in selected_blocks:
        lines.append(
            "| %s | %d | %d | %d | %s |"
            % (
                block["selection_date"],
                block["evaluated_queue_count"],
                block["stock_count"],
                block["etf_count"],
                ", ".join(block["tickers"]),
            )
        )
    lines.extend(
        [
            "",
            "## Remaining gate",
            "",
            "🔴 Complete point-in-time earnings, dividends, delistings, splits, and exact OCC deliverables are still missing for the 40 sampled symbols. **ORATS requests remain `0` until that bundle verifies.**",
            "",
        ]
    )
    return ClassificationAnalysis(
        raw_receipt=submission_receipt,
        classifications=classifications,
        universe_source=universe_source,
        cohorts=cohorts,
        audit=audit,
        markdown="\n".join(lines),
    )


def save_public_classification_audit(
    *,
    public_source_audit_dir: Path,
    sec_submission_root: Path,
    output_root: Path,
    run_id: str,
) -> SavedClassificationAudit:
    output = Path(output_root).expanduser().resolve()
    try:
        output.relative_to(OUT_ROOT)
    except ValueError as exc:
        raise PublicClassificationError("classification output must remain in Cultra/out") from exc
    analysis = analyze_public_classification(
        public_source_audit_dir=public_source_audit_dir,
        sec_submission_root=sec_submission_root,
    )
    writer = ArtifactWriter(output, run_id)
    writer.write_json("sec_submission_receipt.json", analysis.raw_receipt)
    writer.write_json("point_in_time_classifications.json", analysis.classifications)
    writer.write_json("point_in_time_universe_source.json", analysis.universe_source)
    validated_universe = load_point_in_time_universe_source(
        writer.run_dir / "point_in_time_universe_source.json",
        selection_dates=SELECTION_DATES,
    )
    if len(validated_universe.members) != sum(
        len(item["members"]) for item in analysis.universe_source["snapshots"]
    ):
        raise PublicClassificationError(
            "saved point-in-time universe source did not reproduce"
        )
    writer.write_json("rotating_cohorts.json", analysis.cohorts)
    writer.write_json("classification_audit.json", analysis.audit)
    writer.write_text("CLASSIFICATION_AUDIT.md", analysis.markdown, "text/markdown")
    writer.finalize(
        as_of=date(2026, 8, 31),
        overall_status=str(analysis.audit["status"]),
        metadata={
            "profit_confidence": "UNPROVEN",
            "orats_attempts": 0,
            "schwab_attempts": 0,
            "paid_data_attempts": 0,
            "source_audit_dir": str(Path(public_source_audit_dir).resolve()),
            "sec_submission_root": str(Path(sec_submission_root).resolve()),
        },
    )
    return SavedClassificationAudit(
        run_dir=writer.run_dir,
        board_path=writer.run_dir / "CLASSIFICATION_AUDIT.md",
        audit_path=writer.run_dir / "classification_audit.json",
        universe_source_path=writer.run_dir / "point_in_time_universe_source.json",
        status=str(analysis.audit["status"]),
    )


def verify_public_classification_audit(run_dir: Path) -> Tuple[str, ...]:
    root = Path(run_dir).expanduser().resolve()
    errors = list(verify_manifest(root))
    manifest_path = root / "manifest.json"
    if errors or not manifest_path.is_file():
        return tuple(errors)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metadata = manifest["metadata"]
        analysis = analyze_public_classification(
            public_source_audit_dir=Path(metadata["source_audit_dir"]),
            sec_submission_root=Path(metadata["sec_submission_root"]),
        )
        expected_json = {
            "sec_submission_receipt.json": analysis.raw_receipt,
            "point_in_time_classifications.json": analysis.classifications,
            "point_in_time_universe_source.json": analysis.universe_source,
            "rotating_cohorts.json": analysis.cohorts,
            "classification_audit.json": analysis.audit,
        }
        for name, expected in expected_json.items():
            if (root / name).read_bytes() != canonical_json_bytes(expected):
                errors.append("classification artifact is not reproducible: %s" % name)
        if (root / "CLASSIFICATION_AUDIT.md").read_text(encoding="utf-8") != analysis.markdown:
            errors.append("classification board is not reproducible")
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        errors.append("classification audit cannot be reproduced: %s" % exc)
    return tuple(errors)


__all__ = [
    "ClassificationAnalysis",
    "PublicClassificationError",
    "SavedClassificationAudit",
    "analyze_public_classification",
    "save_public_classification_audit",
    "verify_public_classification_audit",
]
