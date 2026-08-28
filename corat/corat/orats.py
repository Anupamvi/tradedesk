"""Cache-first ORATS Data API adapter.

The API token is only ever attached to the in-memory request URL. Cache keys,
cache payloads, traces, reports, and raised errors exclude it.
"""

from __future__ import annotations

import gzip
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from corat.clock import today_new_york
from corat.models import SourceTrace
from corat.secrets import redact
from corat.store import canonical_json, read_json, sha256_bytes, sha256_file, utc_now, write_json


TICKER_RE = re.compile(r"^[A-Z0-9.^/-]{1,20}$")

DAILY_FIELDS = (
    "ticker,tradeDate,clsPx,hiPx,loPx,open,stockVolume,updatedAt"
)
CORE_FIELDS = (
    "ticker,tradeDate,assetType,mktCap,priorCls,pxAtmIv,pxCls,stkVolu,sector,sectorName,bestEtf,etfIncl,"
    "cVolu,cOi,pVolu,pOi,avgOptVolu20d,orFcst20d,orIvFcst20d,"
    "orIvXern20d,orIvXernInf,iv200Ma,atmIvM1,atmIvM2,atmIvM3,atmIvM4,"
    "dtExM1,dtExM2,dtExM3,dtExM4,slope,slopeFcst,mktWidthVol,"
    "orHv20d,clsHv20d,clsHv60d,exErnIv20d,exErnIv30d,iv20d,iv30d,"
    "impliedEarningsMove,absAvgErnMv,ernMvStdv,nextErn,nextErnTod,lastErn,"
    "lastErnTod,daysToNextErn,wksNextErn,iRate5wk,divYield,updatedAt"
)
IVRANK_FIELDS = "ticker,tradeDate,iv,ivRank1m,ivRank1y,ivPct1m,ivPct1y,updatedAt"
SUMMARY_FIELDS = (
    "ticker,tradeDate,stockPrice,confidence,iv10d,iv20d,iv30d,iv60d,"
    "exErnIv10d,exErnIv20d,exErnIv30d,exErnIv60d,impliedMove,"
    "impliedEarningsMove,skewing,contango,totalErrorConf,updatedAt"
)
STRIKE_FIELDS = (
    "ticker,tradeDate,expirDate,dte,strike,stockPrice,callVolume,"
    "callOpenInterest,callBidSize,callAskSize,putVolume,putOpenInterest,"
    "putBidSize,putAskSize,callBidPrice,callValue,callAskPrice,putBidPrice,"
    "putValue,putAskPrice,callMidIv,putMidIv,smvVol,delta,gamma,theta,vega,updatedAt"
)


class OratsError(RuntimeError):
    pass


@dataclass
class FetchBundle:
    rows: List[Mapping[str, Any]] = field(default_factory=list)
    traces: List[SourceTrace] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def extend(self, other: "FetchBundle") -> None:
        self.rows.extend(other.rows)
        self.traces.extend(other.traces)
        self.errors.extend(other.errors)


def _chunks(values: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for index in range(0, len(values), max(1, size)):
        yield values[index : index + max(1, size)]


def _rows(payload: Any) -> List[Mapping[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return [row for row in payload["data"] if isinstance(row, dict)]
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    return []


def _latest_data(rows: Sequence[Mapping[str, Any]]) -> str:
    values: List[str] = []
    for row in rows:
        for key in ("updatedAt", "quoteDate", "tradeDate", "earnDate"):
            value = str(row.get(key) or "")
            if value:
                values.append(value)
                break
    return max(values) if values else ""


def _age_seconds(stamp: str) -> Optional[float]:
    if not stamp:
        return None
    try:
        parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return max(0.0, (datetime.now(timezone.utc) - parsed).total_seconds())


class OratsClient:
    def __init__(
        self,
        token: str,
        base_url: str,
        cache_root: Path,
        state_root: Path,
        timeout_seconds: float = 120.0,
        max_requests: int = 100,
        monthly_cap: int = 20000,
        requests_per_minute: int = 90,
        offline: bool = False,
        refresh: bool = False,
        monthly_reserve: int = 0,
    ) -> None:
        if not token:
            raise ValueError("ORATS token is required")
        self._token = token
        self.base_url = base_url.rstrip("/")
        self.cache_root = cache_root
        self.state_root = state_root
        self.timeout_seconds = float(timeout_seconds)
        self.max_requests = int(max_requests)
        self.monthly_cap = int(monthly_cap)
        self.requests_per_minute = max(1, int(requests_per_minute))
        self.offline = bool(offline)
        self.refresh = bool(refresh)
        self.monthly_reserve = max(0, int(monthly_reserve))
        self.run_requests = 0
        self._last_request = 0.0

    @property
    def usage_path(self) -> Path:
        return self.state_root / "orats_usage.json"

    def _usage(self) -> Dict[str, Any]:
        month = datetime.now().strftime("%Y-%m")
        payload = read_json(self.usage_path, {}) or {}
        if payload.get("month") != month:
            return {"month": month, "used": 0, "cap": self.monthly_cap}
        used = int(payload.get("used") or 0)
        cap = int(payload.get("cap") or self.monthly_cap)
        return {"month": month, "used": used, "cap": cap}

    def usage(self) -> Dict[str, Any]:
        payload = self._usage()
        payload["left"] = max(0, int(payload["cap"]) - int(payload["used"]))
        payload["run_requests"] = self.run_requests
        payload["run_left"] = max(0, self.max_requests - self.run_requests)
        payload["monthly_reserve"] = self.monthly_reserve
        payload["spendable_left"] = max(0, int(payload["left"]) - self.monthly_reserve)
        return payload

    def _count_request(self) -> None:
        usage = self._usage()
        usage["used"] = int(usage.get("used") or 0) + 1
        write_json(self.usage_path, usage)
        self.run_requests += 1

    def _check_budget(self) -> None:
        usage = self.usage()
        if self.run_requests >= self.max_requests:
            raise OratsError("ORATS per-run request budget exhausted")
        if int(usage.get("left") or 0) <= self.monthly_reserve:
            raise OratsError("ORATS monthly reserve reached")

    def _throttle(self) -> None:
        interval = 60.0 / float(self.requests_per_minute)
        elapsed = time.monotonic() - self._last_request
        if self._last_request and elapsed < interval:
            time.sleep(interval - elapsed)
        self._last_request = time.monotonic()

    def _cache_path(self, endpoint: str, params: Mapping[str, str]) -> Path:
        clean = {str(k): str(v) for k, v in params.items() if str(k).lower() != "token"}
        digest = sha256_bytes(canonical_json({"endpoint": endpoint, "params": clean}).encode("utf-8"))[:24]
        namespace = endpoint.strip("/").replace("/", "_") or "root"
        return self.cache_root / "orats" / namespace / (digest + ".json")

    def request_rows(
        self,
        endpoint: str,
        params: Mapping[str, str],
        max_cache_age_seconds: Optional[float] = None,
    ) -> FetchBundle:
        clean_params = {str(k): str(v) for k, v in params.items() if str(k).lower() != "token"}
        cache_path = self._cache_path(endpoint, clean_params)
        cached = read_json(cache_path)
        cached_fresh = False
        if isinstance(cached, dict):
            age = _age_seconds(str(cached.get("fetched_at_utc") or ""))
            cached_fresh = max_cache_age_seconds is None or (age is not None and age <= max_cache_age_seconds)
        if isinstance(cached, dict) and (self.offline or (cached_fresh and not self.refresh)):
            rows = _rows(cached.get("payload"))
            trace = SourceTrace(
                source="ORATS",
                endpoint=endpoint,
                status="CACHED",
                fetched_at_utc=str(cached.get("fetched_at_utc") or ""),
                latest_data_at=_latest_data(rows),
                rows=len(rows),
                cache_path=str(cache_path),
                cache_sha256=sha256_file(cache_path),
                params=clean_params,
            )
            return FetchBundle(rows=rows, traces=[trace])
        if self.offline:
            message = "ORATS cache miss for {}".format(endpoint)
            trace = SourceTrace("ORATS", endpoint, "DATA UNAVAILABLE", "", "", 0, str(cache_path), "", clean_params, message)
            return FetchBundle(traces=[trace], errors=[message])
        try:
            self._check_budget()
        except OratsError as exc:
            message = str(exc)
            trace = SourceTrace("ORATS", endpoint, "DATA UNAVAILABLE", "", "", 0, str(cache_path), "", clean_params, message)
            return FetchBundle(traces=[trace], errors=[message])
        query = dict(clean_params)
        query["token"] = self._token
        url = self.base_url + endpoint + "?" + urllib.parse.urlencode(query)
        request = urllib.request.Request(
            url,
            headers={"Accept": "application/json", "User-Agent": "CORAT/0.1 research-only"},
        )
        self._throttle()
        fetched_at = utc_now()
        status = 0
        raw = b""
        error = ""
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                status = int(response.getcode() or 0)
                raw = response.read() or b""
        except urllib.error.HTTPError as exc:
            status = int(exc.code or 0)
            try:
                raw = exc.read() or b""
            except Exception:
                raw = b""
            error = "ORATS HTTP {} for {}".format(status, endpoint)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            error = "ORATS {} for {}".format(type(exc).__name__, endpoint)
        if status or raw:
            self._count_request()
        if error:
            safe = redact(error, self._token)
            if isinstance(cached, dict):
                rows = _rows(cached.get("payload"))
                trace = SourceTrace(
                    "ORATS", endpoint, "STALE_CACHE", str(cached.get("fetched_at_utc") or ""),
                    _latest_data(rows), len(rows), str(cache_path), sha256_file(cache_path), clean_params, safe,
                )
                return FetchBundle(rows=rows, traces=[trace], errors=[safe])
            trace = SourceTrace("ORATS", endpoint, "DATA UNAVAILABLE", fetched_at, "", 0, str(cache_path), "", clean_params, safe)
            return FetchBundle(traces=[trace], errors=[safe])
        if raw[:2] == b"\x1f\x8b":
            try:
                raw = gzip.decompress(raw)
            except OSError:
                message = "ORATS returned invalid gzip for {}".format(endpoint)
                return FetchBundle(errors=[message])
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            message = "ORATS returned invalid JSON for {}".format(endpoint)
            return FetchBundle(errors=[message])
        rows = _rows(payload)
        cache_body = {
            "schema_version": "corat.orats_cache.v1",
            "endpoint": endpoint,
            "params": clean_params,
            "fetched_at_utc": fetched_at,
            "http_status": status,
            "payload": {"data": rows},
        }
        write_json(cache_path, cache_body)
        trace = SourceTrace(
            "ORATS", endpoint, "LIVE_FETCH", fetched_at, _latest_data(rows), len(rows),
            str(cache_path), sha256_file(cache_path), clean_params, "",
        )
        return FetchBundle(rows=rows, traces=[trace])

    @staticmethod
    def normalize_tickers(tickers: Iterable[str]) -> List[str]:
        result: List[str] = []
        seen = set()
        for raw in tickers:
            ticker = str(raw).strip().upper()
            if not TICKER_RE.fullmatch(ticker):
                raise ValueError("invalid ticker: {}".format(raw))
            if ticker not in seen:
                seen.add(ticker)
                result.append(ticker)
        return result

    def _batched(
        self,
        endpoint: str,
        tickers: Iterable[str],
        fields: str,
        batch_size: int,
        extra: Optional[Mapping[str, str]] = None,
        max_cache_age_seconds: Optional[float] = None,
    ) -> FetchBundle:
        names = self.normalize_tickers(tickers)
        combined = FetchBundle()
        for chunk in _chunks(names, batch_size):
            params = {"ticker": ",".join(chunk), "fields": fields}
            params.update({str(k): str(v) for k, v in (extra or {}).items()})
            combined.extend(self.request_rows(endpoint, params, max_cache_age_seconds=max_cache_age_seconds))
        return combined

    def fetch_dailies(
        self,
        tickers: Iterable[str],
        start_date: str,
        end_date: str,
        batch_size: int = 10,
    ) -> FetchBundle:
        current_age = 900 if end_date >= today_new_york() else None
        return self._batched(
            "/hist/dailies", tickers, DAILY_FIELDS, batch_size,
            extra={"tradeDate": "{},{}".format(start_date, end_date)},
            max_cache_age_seconds=current_age,
        )

    def fetch_asof(
        self,
        family: str,
        tickers: Iterable[str],
        as_of: str,
        batch_size: int = 10,
        allow_current: bool = True,
    ) -> FetchBundle:
        fields = {"cores": CORE_FIELDS, "ivrank": IVRANK_FIELDS, "summaries": SUMMARY_FIELDS}[family]
        today = today_new_york()
        recent_cutoff = (date.fromisoformat(today) - timedelta(days=4)).isoformat()
        if allow_current and as_of >= recent_cutoff:
            current = self._batched(
                "/{}".format(family), tickers, fields, batch_size,
                max_cache_age_seconds=900,
            )
            current.rows = [row for row in current.rows if str(row.get("tradeDate") or "")[:10] <= as_of]
            if current.rows:
                return current
            historical = self._batched(
                "/hist/{}".format(family), tickers, fields, batch_size,
                extra={"tradeDate": as_of}, max_cache_age_seconds=None,
            )
            historical.traces = current.traces + historical.traces
            historical.errors = current.errors + historical.errors
            return historical
        return self._batched(
            "/hist/{}".format(family), tickers, fields, batch_size,
            extra={"tradeDate": as_of}, max_cache_age_seconds=None,
        )

    def fetch_chain(self, ticker: str, as_of: str, min_dte: int, max_dte: int) -> FetchBundle:
        name = self.normalize_tickers([ticker])[0]
        today = today_new_york()
        extra = {
            "ticker": name,
            "fields": STRIKE_FIELDS,
            "dte": "{},{}".format(int(min_dte), int(max_dte)),
            "delta": ".10,.90",
        }
        recent_cutoff = (date.fromisoformat(today) - timedelta(days=4)).isoformat()
        if as_of >= recent_cutoff:
            result = self.request_rows("/strikes", extra, max_cache_age_seconds=300)
            result.rows = [row for row in result.rows if str(row.get("tradeDate") or "")[:10] <= as_of]
            if result.rows:
                return result
            extra["tradeDate"] = as_of
            historical = self.request_rows("/hist/strikes", extra, max_cache_age_seconds=None)
            historical.traces = result.traces + historical.traces
            historical.errors = result.errors + historical.errors
            return historical
        extra["tradeDate"] = as_of
        return self.request_rows("/hist/strikes", extra, max_cache_age_seconds=None)

    def fetch_market_asof(self, family: str, as_of: str) -> FetchBundle:
        """Retrieve the complete ORATS family for broad-universe discovery."""

        fields = {"cores": CORE_FIELDS, "ivrank": IVRANK_FIELDS, "summaries": SUMMARY_FIELDS}[family]
        today = today_new_york()
        recent_cutoff = (date.fromisoformat(today) - timedelta(days=4)).isoformat()
        if as_of >= recent_cutoff:
            current = self.request_rows(
                "/{}".format(family),
                {"fields": fields},
                max_cache_age_seconds=900,
            )
            current.rows = [row for row in current.rows if str(row.get("tradeDate") or "")[:10] <= as_of]
            if current.rows:
                return current
            historical = self.request_rows(
                "/hist/{}".format(family),
                {"fields": fields, "tradeDate": as_of},
                max_cache_age_seconds=None,
            )
            historical.traces = current.traces + historical.traces
            historical.errors = current.errors + historical.errors
            return historical
        return self.request_rows(
            "/hist/{}".format(family),
            {"fields": fields, "tradeDate": as_of},
            max_cache_age_seconds=None,
        )

    def fetch_core_history(self, ticker: str, start_date: str, end_date: str) -> FetchBundle:
        name = self.normalize_tickers([ticker])[0]
        result = self.request_rows(
            "/hist/cores",
            {
                "ticker": name,
                "fields": "ticker,tradeDate,orIvXern20d,iv20d,orHv20d,clsHv20d,updatedAt",
            },
            max_cache_age_seconds=None,
        )
        result.rows = [
            row for row in result.rows
            if start_date <= str(row.get("tradeDate") or "")[:10] <= end_date
        ]
        return result

    def fetch_historical_chain_full(self, ticker: str, trade_date: str, max_dte: int = 120) -> FetchBundle:
        """Full historical surface used to find exact exit legs during replay."""
        name = self.normalize_tickers([ticker])[0]
        return self.request_rows(
            "/hist/strikes",
            {
                "ticker": name,
                "tradeDate": trade_date,
                "fields": STRIKE_FIELDS,
                "dte": "0,{}".format(int(max_dte)),
            },
            max_cache_age_seconds=None,
        )

    def fetch_earnings(self, ticker: str) -> FetchBundle:
        name = self.normalize_tickers([ticker])[0]
        return self.request_rows("/hist/earnings", {"ticker": name}, max_cache_age_seconds=86400)
