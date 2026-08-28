"""Read-only SEC submissions adapter with explicit user-agent requirement."""

from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from codexswing.clock import NEW_YORK, UTC, iso_utc, utc_now
from codexswing.schemas.source import SourceRecord, canonical_json


SEC_SUBMISSIONS_BASE = "https://data.sec.gov/submissions"
SEC_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"
CIK_RE = re.compile(r"^\d{1,10}$")


class SECError(RuntimeError):
    pass


Transport = Callable[[str], Mapping[str, Any]]


def _sec_acceptance_timestamp(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z") or re.search(r"[+-]\d{2}:\d{2}$", text):
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    else:
        parsed = datetime.fromisoformat(text).replace(tzinfo=NEW_YORK)
    return parsed.astimezone(UTC)


class SECSubmissionsClient:
    def __init__(
        self,
        user_agent: str,
        timeout_seconds: int = 30,
        transport: Optional[Transport] = None,
    ) -> None:
        if not user_agent.strip() or "@" not in user_agent:
            raise ValueError("SEC user_agent must identify the application and a contact email")
        self.user_agent = user_agent.strip()
        self.timeout_seconds = timeout_seconds
        self._transport = transport or self._default_transport

    def _default_transport(self, url: str) -> Mapping[str, Any]:
        request = urllib.request.Request(
            url,
            headers={"Accept": "application/json", "User-Agent": self.user_agent},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            raise SECError("SEC submissions request returned HTTP {}".format(exc.code)) from None
        except urllib.error.URLError as exc:
            raise SECError("SEC submissions request failed: {}".format(exc.reason)) from None
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise SECError("SEC submissions response was invalid JSON") from None
        if not isinstance(payload, Mapping):
            raise SECError("SEC submissions response had an unexpected shape")
        return payload

    @staticmethod
    def normalize_cik(cik: str) -> str:
        value = cik.strip()
        if not CIK_RE.fullmatch(value):
            raise ValueError("CIK must contain 1 to 10 digits")
        return value.zfill(10)

    def fetch_company(self, cik: str) -> Mapping[str, Any]:
        normalized = self.normalize_cik(cik)
        return self._transport("{}/CIK{}.json".format(SEC_SUBMISSIONS_BASE, normalized))

    def recent_filings(self, company_payload: Mapping[str, Any]) -> Tuple[Mapping[str, Any], ...]:
        filings = company_payload.get("filings")
        recent = filings.get("recent") if isinstance(filings, Mapping) else None
        if not isinstance(recent, Mapping):
            raise SECError("SEC company payload is missing filings.recent")
        accession_numbers = recent.get("accessionNumber")
        if not isinstance(accession_numbers, list):
            raise SECError("SEC recent filings are missing accessionNumber")
        rows: List[Mapping[str, Any]] = []
        keys = tuple(recent.keys())
        for index in range(len(accession_numbers)):
            row: Dict[str, Any] = {}
            for key in keys:
                values = recent.get(key)
                row[key] = values[index] if isinstance(values, list) and index < len(values) else None
            rows.append(row)
        return tuple(rows)

    def filings_to_records(
        self,
        cik: str,
        filings: Iterable[Mapping[str, Any]],
        forms: Optional[Sequence[str]] = None,
        since_date: Optional[str] = None,
        ingested_at: Optional[datetime] = None,
    ) -> Tuple[SourceRecord, ...]:
        normalized_cik = self.normalize_cik(cik)
        allowed_forms = {form.strip().upper() for form in forms or () if form.strip()}
        ingestion_time = (ingested_at or utc_now()).astimezone(UTC)
        records: List[SourceRecord] = []
        for filing in filings:
            form = str(filing.get("form") or "").upper()
            filing_date = str(filing.get("filingDate") or "")
            if allowed_forms and form not in allowed_forms:
                continue
            if since_date and filing_date < since_date:
                continue
            accession = str(filing.get("accessionNumber") or "")
            primary_document = str(filing.get("primaryDocument") or "")
            acceptance_text = str(filing.get("acceptanceDateTime") or "")
            if not accession or not primary_document or not acceptance_text or not filing_date:
                raise SECError("SEC filing row is missing required identity/timestamp fields")
            accepted = _sec_acceptance_timestamp(acceptance_text)
            if accepted > ingestion_time + timedelta(minutes=5):
                raise SECError("SEC filing is future-dated relative to ingestion")
            cik_int = str(int(normalized_cik))
            accession_compact = accession.replace("-", "")
            filing_url = "{}/{}/{}/{}".format(
                SEC_ARCHIVES_BASE,
                cik_int,
                accession_compact,
                primary_document,
            )
            digest = hashlib.sha256(canonical_json(filing).encode("utf-8")).hexdigest()
            records.append(
                SourceRecord(
                    source="sec_filings",
                    source_id="{}:{}".format(normalized_cik, accession),
                    session_date=accepted.astimezone(NEW_YORK).date().isoformat(),
                    event_time_utc=iso_utc(accepted),
                    published_at_utc=iso_utc(accepted),
                    first_seen_at_utc=iso_utc(ingestion_time),
                    available_at_utc=iso_utc(accepted),
                    ingested_at_utc=iso_utc(ingestion_time),
                    source_uri=filing_url,
                    revision=digest,
                    payload=dict(filing),
                )
            )
        return tuple(records)

