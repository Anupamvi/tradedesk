"""Automatic, dated public-news enrichment for CORAT discovery candidates.

The adapter stores only headline metadata and direct source links. It does not
invent article contents, infer X posts, or label a headline directional unless
it contains an explicit event phrase. Research evidence informs ranking; the
economic POP/expected-profit decision remains separate.
"""

from __future__ import annotations

import copy
import json
import re
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from xml.etree import ElementTree

from corat.context import context_template
from corat.store import write_json


HIGH_CREDIBILITY_SOURCES = (
    "reuters",
    "bloomberg",
    "cnbc",
    "the wall street journal",
    "wall street journal",
    "financial times",
    "associated press",
    "marketwatch",
    "barron's",
    "dow jones",
)

BULLISH_PATTERNS = (
    r"\bbeat(?:s|ing)?\b.{0,35}\b(?:estimate|expectation|forecast)s?\b",
    r"\btop(?:s|ped)?\b.{0,30}\b(?:estimate|expectation)s?\b",
    r"\b(?:raise|raises|raised|boost|boosts|lift|lifts)\b.{0,25}\b(?:guidance|outlook|forecast)\b",
    r"\bforecast(?:s|ed)?\b.{0,40}\babove estimates\b",
    r"\b(?:win|wins|won|awarded)\b.{0,60}\bcontract\b",
    r"\b(?:fda |regulatory )?(?:approve|approves|approved|approval)\b",
    r"\b(?:announce|announces|announced)\b.{0,35}\b(?:buyback|share repurchase)\b",
    r"\brecord\b.{0,20}\b(?:revenue|sales|earnings|profit)\b",
    r"\b(?:revenue|sales|profit|earnings)\b.{0,35}\b(?:surge|jumps|above estimates)\b",
)

BEARISH_PATTERNS = (
    r"\bmiss(?:es|ed|ing)?\b.{0,35}\b(?:estimate|expectation|forecast)s?\b",
    r"\b(?:cut|cuts|cutting|lower|lowers|lowered)\b.{0,25}\b(?:guidance|outlook|forecast)\b",
    r"\bforecast(?:s|ed)?\b.{0,40}\bbelow estimates\b",
    r"\b(?:fda |regulatory )?(?:reject|rejects|rejected|denial|denied)\b",
    r"\b(?:recall|investigation|data breach|accounting probe|profit warning)\b",
    r"\b(?:revenue|sales|profit|earnings)\b.{0,35}\b(?:falls|drops|declines|below estimates)\b",
)

TITLE_BULLISH_PATTERNS = (
    r"\b(?:jump|jumps|jumped|surge|surges|surged|soar|soars|soared|rally|rallies|rallied|rise|rises|rose)\b.{0,70}\b(?:after|following|on|as)\b",
    r"\bblockbuster earnings\b",
)

TITLE_BEARISH_PATTERNS = (
    r"\b(?:plunge|plunges|plunged|tumble|tumbles|tumbled|sink|sinks|sank|slide|slides|slid|crash|crashes|crashed)\b",
    r"\bprofit warning\b",
)


def classify_headline_direction(title: str, description: str = "") -> str:
    title_text = title.lower()
    _ = description  # RSS descriptions are not used to infer direction.
    if re.match(r"^\s*(?:will|can|could|should|would|is|are|does|do|did|prediction:)\b", title_text):
        return "NEUTRAL"
    title_bullish = any(re.search(pattern, title_text) for pattern in TITLE_BULLISH_PATTERNS)
    title_bearish = any(re.search(pattern, title_text) for pattern in TITLE_BEARISH_PATTERNS)
    if title_bullish and not title_bearish:
        return "BULLISH"
    if title_bearish and not title_bullish:
        return "BEARISH"
    text = title_text
    bullish = any(re.search(pattern, text) for pattern in BULLISH_PATTERNS)
    bearish = any(re.search(pattern, text) for pattern in BEARISH_PATTERNS)
    if bullish and not bearish:
        return "BULLISH"
    if bearish and not bullish:
        return "BEARISH"
    return "NEUTRAL"


def source_credibility(source: str) -> str:
    normalized = source.strip().lower().replace(" on msn", "")
    return "HIGH" if any(name in normalized for name in HIGH_CREDIBILITY_SOURCES) else "MEDIUM"


def _direct_link(link: str) -> str:
    parsed = urllib.parse.urlparse(link)
    query = urllib.parse.parse_qs(parsed.query)
    target = (query.get("url") or [""])[0]
    if target.startswith(("https://", "http://")):
        return target
    return link


def _published_day(value: str) -> Optional[date]:
    try:
        return parsedate_to_datetime(value).date()
    except (TypeError, ValueError, OverflowError):
        return None


def _source_name(item: ElementTree.Element) -> str:
    for child in item:
        if str(child.tag).endswith("Source"):
            return str(child.text or "").strip()
    return "Bing News source"


def _fetch_query(query: str, timeout_seconds: float) -> bytes:
    url = "https://www.bing.com/news/search?{}".format(
        urllib.parse.urlencode({"q": query, "format": "rss"})
    )
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "CORAT/0.2 personal research RSS reader"},
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return response.read()


def parse_news_rss(
    payload: bytes,
    as_of: str,
    lookback_days: int,
) -> List[Dict[str, Any]]:
    decision_day = date.fromisoformat(as_of)
    oldest = decision_day - timedelta(days=max(1, lookback_days))
    root = ElementTree.fromstring(payload)
    rows: List[Dict[str, Any]] = []
    for item in root.findall("./channel/item"):
        title = str(item.findtext("title") or "").strip()
        description = re.sub(r"<[^>]+>", " ", str(item.findtext("description") or "")).strip()
        published = _published_day(str(item.findtext("pubDate") or ""))
        link = _direct_link(str(item.findtext("link") or "").strip())
        source = _source_name(item)
        if not title or published is None or not (oldest <= published <= decision_day):
            continue
        if not link.startswith(("https://", "http://")):
            continue
        rows.append(
            {
                "classification": "REPORTED INFORMATION",
                "credibility": source_credibility(source),
                "source": source.replace(" on MSN", ""),
                "source_url": link,
                "published_at": published.isoformat(),
                "title": title,
                "direction": classify_headline_direction(title, description),
                "research_basis": "Headline metadata from Bing News RSS; article contents were not inferred.",
            }
        )
    return rows


def _ticker_news(
    ticker: str,
    name: str,
    as_of: str,
    lookback_days: int,
    timeout_seconds: float,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    decision_day = date.fromisoformat(as_of)
    oldest = decision_day - timedelta(days=max(1, lookback_days))
    base = "{} {}".format(ticker, name).strip()
    queries = [
        '{} earnings guidance forecast contract approval acquisition investigation recall after:{} before:{}'.format(base, oldest.isoformat(), (decision_day + timedelta(days=1)).isoformat()),
        '{} latest stock news after:{} before:{}'.format(base, oldest.isoformat(), (decision_day + timedelta(days=1)).isoformat()),
    ]
    evidence: List[Dict[str, Any]] = []
    errors: List[str] = []
    for query in queries:
        try:
            evidence.extend(parse_news_rss(_fetch_query(query, timeout_seconds), as_of, lookback_days))
        except Exception as exc:  # network/parser failure remains visible and non-fatal
            errors.append("{} news research {}: {}".format(ticker, type(exc).__name__, str(exc)[:160]))
    deduped: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in evidence:
        key = (str(row.get("title") or "").lower(), str(row.get("source_url") or ""))
        deduped[key] = row
    ordered = sorted(
        deduped.values(),
        key=lambda row: (
            str(row.get("credibility")) == "HIGH",
            str(row.get("direction")) in {"BULLISH", "BEARISH"},
            str(row.get("published_at")),
        ),
        reverse=True,
    )
    return ordered[:6], errors


def _read_existing(path: Optional[Path], as_of: str, tickers: Iterable[str]) -> Dict[str, Any]:
    if path is None or not path.is_file():
        return context_template(as_of, tickers)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != "corat.context.v1":
        raise ValueError("existing context must use schema_version corat.context.v1")
    if str(payload.get("as_of"))[:10] > as_of:
        raise ValueError("existing context is future-dated")
    return copy.deepcopy(payload)


def build_auto_context(
    as_of: str,
    candidates: Sequence[Mapping[str, Any]],
    output_path: Path,
    existing_path: Optional[Path] = None,
    maximum_tickers: int = 12,
    lookback_days: int = 21,
    timeout_seconds: float = 15.0,
) -> Dict[str, Any]:
    selected = list(candidates[: max(0, maximum_tickers)])
    names = [str(row.get("ticker") or "").upper() for row in selected if row.get("ticker")]
    payload = _read_existing(existing_path, as_of, names)
    payload["as_of"] = as_of
    payload.setdefault("market_events", [])
    ticker_payload = payload.setdefault("tickers", {})
    errors: List[str] = []
    researched: Dict[str, List[Dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=min(4, max(1, len(selected)))) as executor:
        futures = {
            executor.submit(
                _ticker_news,
                str(row.get("ticker") or "").upper(),
                str(row.get("name") or row.get("ticker") or ""),
                as_of,
                lookback_days,
                timeout_seconds,
            ): str(row.get("ticker") or "").upper()
            for row in selected
            if row.get("ticker")
        }
        for future in as_completed(futures):
            ticker = futures[future]
            rows, ticker_errors = future.result()
            researched[ticker] = rows
            errors.extend(ticker_errors)
    for ticker in names:
        entry = ticker_payload.setdefault(
            ticker,
            {"catalysts": [], "x_intelligence": [], "events": [], "options_flow": [], "mention_acceleration": "DATA UNAVAILABLE"},
        )
        for family in ("catalysts", "x_intelligence", "events", "options_flow"):
            entry.setdefault(family, [])
        existing_keys = {
            (str(row.get("title") or row.get("claim") or "").lower(), str(row.get("source_url") or ""))
            for row in entry["catalysts"]
        }
        for row in researched.get(ticker, []):
            key = (str(row.get("title") or "").lower(), str(row.get("source_url") or ""))
            if key not in existing_keys:
                entry["catalysts"].append(row)
                existing_keys.add(key)
    evidence_count = sum(len(researched.get(ticker, [])) for ticker in names)
    payload["research_metadata"] = {
        "method": "Two Bing News RSS queries per discovery ticker (event terms plus general stock news); dated filtering; direct-link extraction; strict headline-only direction classifier.",
        "researched_tickers": names,
        "evidence_rows_added_or_seen": evidence_count,
        "errors": sorted(set(errors)),
        "x_status": "DATA UNAVAILABLE — no authenticated X search connector is configured in the local CLI.",
    }
    write_json(output_path, payload)
    return payload
