"""Scenario-aware macro/geopolitical catalyst observability.

The pattern engine stays driven by point-in-time UW flow. This module adds a
separate explainability layer that turns local browser/news captures into
structured catalyst records, maps those records to sectors/tickers, joins them
back to UW evidence, and emits scenario buckets for auditability.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SCENARIO_BUCKETS = [
    "APPROVED_TRADE",
    "CATALYST_CONFIRMED_TRADE_REVIEW",
    "CATALYST_WATCH",
    "CATALYST_NO_UW_CONFIRMATION",
    "POINT_IN_TIME_INELIGIBLE_CATALYST",
    "BLOCKED_SOURCE_INCOMPLETE",
    "REGIME_CONFLICTED_SETUP",
    "VALIDATION_BLOCKED_SETUP",
    "LIQUIDITY_OR_QUOTE_BLOCKED_SETUP",
    "SECTOR_INDEX_CONFIRMED_SETUP",
    "MULTI_DAY_CONTINUING_CATALYST",
    "NO_PATTERN_FOUND_WITH_EVIDENCE",
]

SCENARIO_EXPECTED_BEHAVIOR = {
    "APPROVED_TRADE": "Promote only when catalyst, UW flow, validation, regime, quote, liquidity, and event checks all pass.",
    "CATALYST_CONFIRMED_TRADE_REVIEW": "Surface catalyst plus UW confirmation when a trade-review setup exists but is not fully approved.",
    "CATALYST_WATCH": "Surface eligible catalyst plus UW evidence even when no validated trade row is promoted.",
    "CATALYST_NO_UW_CONFIRMATION": "Show eligible catalyst records whose mapped tickers/themes lack UW confirmation.",
    "POINT_IN_TIME_INELIGIBLE_CATALYST": "Skip future-dated captures or published sources while recording that they were skipped.",
    "BLOCKED_SOURCE_INCOMPLETE": "Stop promotion when required local UW source data is missing and list exact missing files/data.",
    "REGIME_CONFLICTED_SETUP": "Keep confirmed catalyst setups blocked when market regime conflicts with the direction.",
    "VALIDATION_BLOCKED_SETUP": "Keep confirmed catalyst setups blocked when validation/sample/baseline evidence is insufficient.",
    "LIQUIDITY_OR_QUOTE_BLOCKED_SETUP": "Keep confirmed catalyst setups blocked when quote, spread, volume, OI, or DTE quality fails.",
    "SECTOR_INDEX_CONFIRMED_SETUP": "Surface sector/index confirmation when the catalyst appears at ETF/index level.",
    "MULTI_DAY_CONTINUING_CATALYST": "Carry forward repeated eligible catalyst state across two to five sessions without future leakage.",
    "NO_PATTERN_FOUND_WITH_EVIDENCE": "Explain no-trade/no-pattern outcomes with explicit catalyst, source, UW, and blocker evidence.",
}

ETF_TICKERS = {
    "SMH",
    "SOXX",
    "QQQ",
    "SPY",
    "IWM",
    "FXI",
    "KWEB",
    "EEM",
    "TLT",
    "HYG",
    "KRE",
    "XLF",
    "XLE",
    "USO",
    "ITA",
}

THEME_MAP = {
    "semiconductors/AI chips": {
        "sectors": ["Technology", "Semiconductors", "AI chips"],
        "tickers": ["NVDA", "AMD", "MU", "AVGO", "QCOM", "INTC", "SMH", "SOXX", "QQQ"],
    },
    "China beta/trade": {
        "sectors": ["China beta", "Trade-sensitive cyclicals", "Consumer/industrial exporters"],
        "tickers": ["TSLA", "AAPL", "BA", "CAT", "NKE", "SBUX", "BABA", "FXI", "KWEB", "EEM"],
    },
    "mega-cap/index risk-on": {
        "sectors": ["Mega-cap technology", "Index risk-on"],
        "tickers": ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA"],
    },
    "rates/Fed": {
        "sectors": ["Rates", "Credit", "Financials", "Duration-sensitive growth"],
        "tickers": ["TLT", "HYG", "IWM", "KRE", "XLF", "QQQ"],
    },
    "oil/Middle East": {
        "sectors": ["Energy", "Oil risk", "Airlines"],
        "tickers": ["XLE", "USO", "XOM", "CVX", "UAL", "DAL", "AAL", "LUV"],
    },
    "defense/geopolitics": {
        "sectors": ["Defense", "Aerospace", "Geopolitics"],
        "tickers": ["LMT", "RTX", "NOC", "GD", "ITA"],
    },
    "earnings/guidance": {
        "sectors": ["Single-name earnings", "Guidance risk"],
        "tickers": [],
    },
    "regulation/antitrust/approval risk": {
        "sectors": ["Regulatory risk", "Antitrust", "Approval risk"],
        "tickers": [],
    },
}

CATALYST_RULES = [
    {
        "event_type": "trade talks/trade truce",
        "keywords": ["trade talk", "trade talks", "trade truce", "trade deal", "business gains"],
        "geography": "US/China",
        "themes": ["China beta/trade", "mega-cap/index risk-on"],
        "direction_bias": "bullish",
    },
    {
        "event_type": "tariffs",
        "keywords": ["tariff", "tariffs", "duties"],
        "geography": "Global trade",
        "themes": ["China beta/trade", "mega-cap/index risk-on"],
        "direction_bias": "bearish",
    },
    {
        "event_type": "export controls",
        "keywords": ["export control", "export controls", "chip export", "h200", "blackwell", "ai chip export"],
        "geography": "US/China",
        "themes": ["semiconductors/AI chips", "China beta/trade"],
        "direction_bias": "mixed",
    },
    {
        "event_type": "China/US diplomacy",
        "keywords": ["trump/xi", "trump and xi", "trump-xi", "xi summit", "china summit", "u.s.-china", "us-china", "beijing talks", "china trip"],
        "geography": "US/China",
        "themes": ["China beta/trade", "mega-cap/index risk-on"],
        "direction_bias": "bullish",
    },
    {
        "event_type": "CEO delegation/business access",
        "keywords": ["ceo", "ceos", "delegation", "business access", "business gains", "elon musk", "executives to join", "joining china trip"],
        "geography": "US/China",
        "themes": ["China beta/trade", "semiconductors/AI chips"],
        "direction_bias": "bullish",
    },
    {
        "event_type": "AI chips/semiconductors",
        "keywords": ["ai chip", "ai chips", "semiconductor", "semiconductors", "nvidia", "micron", "qualcomm", "intel", "hbm", "ai memory", "chip rebound"],
        "geography": "Global technology",
        "themes": ["semiconductors/AI chips", "mega-cap/index risk-on"],
        "direction_bias": "bullish",
    },
    {
        "event_type": "sanctions",
        "keywords": ["sanction", "sanctions"],
        "geography": "Global geopolitics",
        "themes": ["defense/geopolitics", "oil/Middle East"],
        "direction_bias": "bearish",
    },
    {
        "event_type": "war/Middle East/oil risk",
        "keywords": ["middle east", "iran", "oil", "war", "peace talks stalled", "geopolitical risk"],
        "geography": "Middle East",
        "themes": ["oil/Middle East", "defense/geopolitics"],
        "direction_bias": "mixed",
    },
    {
        "event_type": "Fed/rates/CPI/jobs/inflation",
        "keywords": ["fed", "rate", "rates", "cpi", "ppi", "jobs", "inflation", "higher-for-longer", "restrictive"],
        "geography": "United States",
        "themes": ["rates/Fed", "mega-cap/index risk-on"],
        "direction_bias": "bearish",
    },
    {
        "event_type": "USD/yields/credit/VIX",
        "keywords": ["usd", "dollar", "yield", "yields", "credit", "vix", "volatility"],
        "geography": "United States",
        "themes": ["rates/Fed", "mega-cap/index risk-on"],
        "direction_bias": "mixed",
    },
    {
        "event_type": "earnings/guidance",
        "keywords": ["earnings", "guidance", "eps", "revenue", "financial results"],
        "geography": "Company-specific",
        "themes": ["earnings/guidance"],
        "direction_bias": "mixed",
    },
    {
        "event_type": "regulation/antitrust/approval risk",
        "keywords": ["regulatory", "regulation", "antitrust", "approval", "litigation", "legal risk"],
        "geography": "Company-specific",
        "themes": ["regulation/antitrust/approval risk"],
        "direction_bias": "bearish",
    },
    {
        "event_type": "sector rotation",
        "keywords": ["sector rotation", "sector", "rotation", "cyclicals", "defensives"],
        "geography": "Market breadth",
        "themes": ["mega-cap/index risk-on"],
        "direction_bias": "mixed",
    },
    {
        "event_type": "index/breadth risk-on or risk-off",
        "keywords": ["risk-on", "risk off", "risk-off", "breadth", "record high", "fresh highs", "futures", "nasdaq", "s&p 500"],
        "geography": "United States",
        "themes": ["mega-cap/index risk-on"],
        "direction_bias": "mixed",
    },
]

ALL_MAPPED_TICKERS = sorted({ticker for theme in THEME_MAP.values() for ticker in theme["tickers"]})

MARKET_CONTEXT_TERMS = {
    "stock",
    "stocks",
    "market",
    "markets",
    "futures",
    "shares",
    "trading",
    "trade gating",
    "options",
    "nasdaq",
    "s&p",
    "dow",
    "spy",
    "qqq",
    "ticker",
    "earnings",
    "guidance",
    "fed",
    "cpi",
    "ppi",
    "rates",
    "inflation",
    "oil",
    "credit",
    "vix",
    "semiconductor",
    "chip",
    "ceo",
    "reuters",
    "ap",
}

POSITIVE_TERMS = {
    "supportive",
    "risk-on",
    "rebound",
    "rally",
    "rallied",
    "record",
    "boosted",
    "surge",
    "optimism",
    "business gains",
    "trade truce",
    "strength",
    "fresh highs",
}

NEGATIVE_TERMS = {
    "risk-off",
    "hot inflation",
    "restrictive",
    "caution",
    "war",
    "oil risk",
    "tariff",
    "sanction",
    "stalled",
    "regulatory risk",
    "litigation",
    "higher-for-longer",
    "avoid",
}

VALIDATION_BLOCKERS = {
    "PATTERN_VALIDATION_NOT_PROVEN",
    "LIMITED_OUT_OF_SAMPLE_SAMPLE",
    "DOES_NOT_BEAT_TWO_BASELINES",
    "VALIDATION_EXPECTANCY_NEGATIVE",
}
REGIME_BLOCKERS = {"MARKET_REGIME_CONFLICT"}
LIQUIDITY_OR_QUOTE_BLOCKERS = {
    "BID_ASK_SPREAD_TOO_WIDE",
    "MISSING_BID_ASK_SPREAD",
    "NO_TRADEABLE_OPTION_QUOTE",
    "MISSING_ENTRY_ASK",
    "MISSING_ENTRY_CREDIT",
    "OPTION_LIQUIDITY_TOO_LOW",
    "DTE_TOO_SHORT_FOR_VALIDATION_HORIZONS",
}
EVENT_BLOCKERS = {"NEAR_TERM_EARNINGS_EVENT_RISK"}

BLOCKER_LABELS = {
    "LIMITED_OUT_OF_SAMPLE_SAMPLE": "sample size too small",
    "VALIDATION_EXPECTANCY_NEGATIVE": "validation expectancy negative",
    "DOES_NOT_BEAT_TWO_BASELINES": "does not beat baselines",
    "PATTERN_VALIDATION_NOT_PROVEN": "pattern family not proven",
    "MARKET_REGIME_CONFLICT": "market regime conflict",
    "BID_ASK_SPREAD_TOO_WIDE": "bid/ask too wide",
    "MISSING_BID_ASK_SPREAD": "bid/ask too wide",
    "NO_TRADEABLE_OPTION_QUOTE": "no tradeable option quote",
    "MISSING_ENTRY_ASK": "no tradeable option quote",
    "MISSING_ENTRY_CREDIT": "no tradeable option quote",
    "OPTION_LIQUIDITY_TOO_LOW": "volume/open interest too low",
    "DTE_TOO_SHORT_FOR_VALIDATION_HORIZONS": "DTE too short",
    "NEAR_TERM_EARNINGS_EVENT_RISK": "earnings/event risk",
    "SOURCE_INCOMPLETE": "source incomplete",
    "FUTURE_DATED_CATALYST": "future-dated catalyst",
}


def build_macro_geo_bundle(
    base_dir: Path,
    as_of: str,
    snapshots: Mapping[str, Any],
    source_dates: Sequence[str],
    daily_rows: Sequence[Mapping[str, Any]],
    source_complete: bool,
    missing_sources: Sequence[str],
) -> Dict[str, Any]:
    """Build all scenario-aware macro/geopolitical observability rows."""

    catalysts = collect_macro_geo_catalysts(base_dir, as_of)
    as_of_snapshot = snapshots.get(as_of) if snapshots else None
    ticker_map = build_ticker_map_rows(catalysts)
    confirmations = build_uw_confirmation_rows(catalysts, as_of_snapshot)
    promotion_rows = build_promotion_decision_rows(
        catalysts=catalysts,
        confirmations=confirmations,
        daily_rows=daily_rows,
        source_complete=source_complete,
        missing_sources=missing_sources,
        as_of=as_of,
    )
    promotion_rows.extend(
        build_multi_day_continuation_rows(
            catalysts=catalysts,
            snapshots=snapshots,
            as_of=as_of,
            source_dates=source_dates,
        )
    )
    promotion_rows.extend(build_no_pattern_rows(as_of, catalysts, promotion_rows, daily_rows, source_complete))
    observability_rows = build_observability_matrix_rows(as_of, promotion_rows)
    summary = summarize_macro_geo_bundle(catalysts, confirmations, promotion_rows, source_complete, missing_sources)
    missed_rows = build_missed_pattern_audit_rows(as_of, catalysts, promotion_rows, summary)
    return {
        "catalysts": catalysts,
        "ticker_map": ticker_map,
        "uw_confirmation": confirmations,
        "promotion_decisions": promotion_rows,
        "observability_rows": observability_rows,
        "missed_pattern_rows": missed_rows,
        "summary": summary,
    }


def collect_macro_geo_catalysts(base_dir: Path, as_of: str, lookback_sessions: int = 5) -> List[Dict[str, Any]]:
    date_dirs = [
        p
        for p in sorted(base_dir.iterdir(), key=lambda x: x.name)
        if p.is_dir() and re.fullmatch(r"\d{4}-\d{2}-\d{2}", p.name) and p.name <= as_of
    ]
    date_dirs = date_dirs[-lookback_sessions:]
    records: List[Dict[str, Any]] = []
    for date_dir in date_dirs:
        browser_dir = date_dir / "browser_text"
        if not browser_dir.exists():
            continue
        for path in sorted(browser_dir.glob("*.txt")):
            records.extend(parse_catalysts_from_capture(path, as_of, date_dir.name))
    records.sort(key=lambda r: (r["capture_date"], r["source_file"], r["event_type"]))
    return records


def parse_catalysts_from_capture(path: Path, as_of: str, folder_date: str) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lowered = text.lower()
    if not has_market_context(lowered):
        return []

    capture_date = extract_capture_date(text, path.name, folder_date)
    published_dates = extract_published_dates(text)
    urls = sorted(set(re.findall(r"https?://[^\s)>\"]+", text)))[:8]
    explicit_tickers = extract_explicit_tickers(text)
    ineligible_reason = ""
    if capture_date > as_of:
        ineligible_reason = "capture_date_after_as_of"
    elif any(d > as_of for d in published_dates):
        ineligible_reason = "published_source_after_as_of"

    records: List[Dict[str, Any]] = []
    for rule in CATALYST_RULES:
        matched = [kw for kw in rule["keywords"] if kw in lowered]
        if not matched:
            continue
        themes = list(rule["themes"])
        mapped = mapped_tickers_for_themes(themes, explicit_tickers)
        sectors = affected_sectors_for_themes(themes)
        direction = direction_bias_from_text(lowered, str(rule["direction_bias"]))
        rationale = build_rationale(rule, matched, direction, explicit_tickers)
        record_id = stable_id(str(path.resolve()), as_of, capture_date, str(rule["event_type"]))
        records.append(
            {
                "catalyst_id": record_id,
                "as_of": as_of,
                "source_file": str(path.resolve()),
                "folder_date": folder_date,
                "source_date": min(published_dates) if published_dates else capture_date,
                "capture_date": capture_date,
                "as_of_eligible": not ineligible_reason,
                "ineligible_reason": ineligible_reason,
                "event_type": rule["event_type"],
                "geography": rule["geography"],
                "affected_themes": themes,
                "affected_sectors": sectors,
                "mapped_tickers": [t for t in mapped if t not in ETF_TICKERS],
                "mapped_etfs": [t for t in mapped if t in ETF_TICKERS],
                "direction_bias": direction,
                "confidence": catalyst_confidence(matched, explicit_tickers, urls, bool(ineligible_reason)),
                "rationale": rationale,
                "matched_keywords": matched,
                "explicit_tickers": sorted(explicit_tickers),
                "source_urls": urls,
            }
        )
    return records


def has_market_context(lowered_text: str) -> bool:
    for term in MARKET_CONTEXT_TERMS:
        if len(term) <= 3 and term.isalpha():
            if re.search(rf"\b{re.escape(term)}\b", lowered_text):
                return True
        elif term in lowered_text:
            return True
    return any(re.search(rf"\b{re.escape(ticker.lower())}\b", lowered_text) for ticker in ALL_MAPPED_TICKERS)


def extract_capture_date(text: str, filename: str, fallback: str) -> str:
    m = re.search(r"^\s*Capture date:\s*(20\d{2}-\d{2}-\d{2})\s*$", text, flags=re.IGNORECASE | re.MULTILINE)
    if m:
        return m.group(1)
    filename_date = extract_date_from_text(filename)
    if filename_date:
        return filename_date
    leading_date = extract_date_from_text(text[:500])
    return leading_date or fallback


def extract_date_from_text(text: str) -> Optional[str]:
    m = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", text)
    if m:
        return m.group(1)
    month_date = parse_month_date(text)
    return month_date


def parse_month_date(text: str) -> Optional[str]:
    months = {
        "january": "01",
        "february": "02",
        "march": "03",
        "april": "04",
        "may": "05",
        "june": "06",
        "july": "07",
        "august": "08",
        "september": "09",
        "october": "10",
        "november": "11",
        "december": "12",
    }
    m = re.search(
        r"\b("
        + "|".join(months)
        + r")\s+(\d{1,2}),\s*(20\d{2})\b",
        text,
        flags=re.IGNORECASE,
    )
    if not m:
        return None
    month = months[m.group(1).lower()]
    day = int(m.group(2))
    return f"{m.group(3)}-{month}-{day:02d}"


def extract_published_dates(text: str) -> List[str]:
    out = re.findall(r"published\s+(20\d{2}-\d{2}-\d{2})", text, flags=re.IGNORECASE)
    for match in re.finditer(r"published\s+([A-Za-z]+\s+\d{1,2},\s+20\d{2})", text, flags=re.IGNORECASE):
        parsed = parse_month_date(match.group(1))
        if parsed:
            out.append(parsed)
    return sorted(set(out))


def extract_explicit_tickers(text: str) -> set[str]:
    tickers = {clean_ticker(t) for t in re.findall(r"^\s*Ticker:\s*([A-Za-z0-9.\-]+)\s*$", text, re.MULTILINE)}
    for ticker in ALL_MAPPED_TICKERS:
        if re.search(rf"\b{re.escape(ticker)}\b", text, flags=re.IGNORECASE):
            tickers.add(ticker)
    return {t for t in tickers if t}


def mapped_tickers_for_themes(themes: Sequence[str], explicit_tickers: Iterable[str]) -> List[str]:
    ordered: List[str] = []
    for ticker in sorted(set(explicit_tickers)):
        if ticker not in ordered:
            ordered.append(ticker)
    for theme in themes:
        for ticker in THEME_MAP.get(theme, {}).get("tickers", []):
            if ticker not in ordered:
                ordered.append(ticker)
    return ordered


def affected_sectors_for_themes(themes: Sequence[str]) -> List[str]:
    sectors: List[str] = []
    for theme in themes:
        for sector in THEME_MAP.get(theme, {}).get("sectors", []):
            if sector not in sectors:
                sectors.append(sector)
    return sectors


def direction_bias_from_text(lowered: str, default: str) -> str:
    positive = sum(1 for term in POSITIVE_TERMS if term in lowered)
    negative = sum(1 for term in NEGATIVE_TERMS if term in lowered)
    if positive and negative:
        return "mixed"
    if positive:
        return "bullish"
    if negative:
        return "bearish"
    return default


def catalyst_confidence(
    matched_keywords: Sequence[str],
    explicit_tickers: Iterable[str],
    urls: Sequence[str],
    ineligible: bool,
) -> float:
    value = 0.50 + min(len(matched_keywords), 4) * 0.06
    if list(explicit_tickers):
        value += 0.07
    if urls:
        value += 0.05
    if ineligible:
        value -= 0.08
    return round(max(0.10, min(0.95, value)), 2)


def build_rationale(rule: Mapping[str, Any], matched: Sequence[str], direction: str, explicit_tickers: Iterable[str]) -> str:
    ticker_text = ", ".join(sorted(explicit_tickers))
    parts = [
        f"Matched {rule['event_type']} via {', '.join(matched[:5])}.",
        f"Direction bias: {direction}.",
    ]
    if ticker_text:
        parts.append(f"Explicit tickers in capture: {ticker_text}.")
    parts.append(f"Mapped themes: {', '.join(rule['themes'])}.")
    return " ".join(parts)


def build_ticker_map_rows(catalysts: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for catalyst in catalysts:
        mapped = list(catalyst.get("mapped_tickers") or []) + list(catalyst.get("mapped_etfs") or [])
        for ticker in mapped or [""]:
            rows.append(
                {
                    "catalyst_id": catalyst.get("catalyst_id"),
                    "as_of": catalyst.get("as_of"),
                    "event_type": catalyst.get("event_type"),
                    "capture_date": catalyst.get("capture_date"),
                    "as_of_eligible": catalyst.get("as_of_eligible"),
                    "ticker": ticker,
                    "instrument_type": "ETF" if ticker in ETF_TICKERS else ("EQUITY" if ticker else ""),
                    "affected_themes": join_list(catalyst.get("affected_themes")),
                    "affected_sectors": join_list(catalyst.get("affected_sectors")),
                    "direction_bias": catalyst.get("direction_bias"),
                    "confidence": catalyst.get("confidence"),
                    "source_file": catalyst.get("source_file"),
                }
            )
    return rows


def build_uw_confirmation_rows(catalysts: Sequence[Mapping[str, Any]], snapshot: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    features = getattr(snapshot, "features", {}) if snapshot is not None else {}
    best_options = getattr(snapshot, "best_options", {}) if snapshot is not None else {}
    for catalyst in catalysts:
        mapped = list(catalyst.get("mapped_tickers") or []) + list(catalyst.get("mapped_etfs") or [])
        for ticker in mapped:
            rows.append(uw_confirmation_row(catalyst, ticker, features.get(ticker, {}), best_options))
    return rows


def uw_confirmation_row(
    catalyst: Mapping[str, Any],
    ticker: str,
    feature: Mapping[str, Any],
    best_options: Mapping[Any, Mapping[str, Any]],
) -> Dict[str, Any]:
    direction_bias = str(catalyst.get("direction_bias") or "mixed")
    flags = list(feature.get("source_flags") or [])
    hot_premium = max(_num(feature.get("hot_total_premium")) or 0.0, _num(feature.get("flow_total_premium")) or 0.0)
    oi_change = max(_num(feature.get("oi_call_diff")) or 0.0, _num(feature.get("oi_put_diff")) or 0.0)
    flow_bias = first_num(feature.get("flow_premium_bias"), feature.get("premium_bias"))
    uw_direction = infer_uw_direction(feature)
    quote = best_options.get((ticker, direction_for_quote(direction_bias, uw_direction))) or {}
    evidence = []
    score = 0
    if "bot_eod" in flags:
        evidence.append("bot-EOD options flow")
        score += 2
    if "hot_chains" in flags:
        evidence.append("hot chains")
        score += 2
    if "chain_oi" in flags:
        evidence.append("chain OI changes")
        score += 2
    if hot_premium >= 100_000:
        evidence.append(f"premium concentration {hot_premium:.0f}")
        score += 2
    if oi_change >= 5_000:
        evidence.append(f"OI change {oi_change:.0f}")
        score += 1
    direction_confirmed = direction_bias == "mixed" or uw_direction == direction_bias
    if direction_confirmed and uw_direction in {"bullish", "bearish"}:
        evidence.append(f"{uw_direction} direction")
        score += 1
    quote_quality = option_quote_quality(quote)
    if quote_quality == "tradeable":
        evidence.append("liquid expiry/quote")
        score += 1
    has_feature = bool(feature)
    uw_confirmed = has_feature and score >= 3
    return {
        "catalyst_id": catalyst.get("catalyst_id"),
        "as_of": catalyst.get("as_of"),
        "capture_date": catalyst.get("capture_date"),
        "event_type": catalyst.get("event_type"),
        "ticker": ticker,
        "instrument_type": "ETF" if ticker in ETF_TICKERS else "EQUITY",
        "as_of_eligible": catalyst.get("as_of_eligible"),
        "uw_confirmed": uw_confirmed,
        "uw_evidence_score": score,
        "uw_evidence_found": "; ".join(evidence) if evidence else "no mapped UW evidence",
        "source_flags": ";".join(flags),
        "bot_eod_options_flow": "bot_eod" in flags,
        "chain_oi_changes": "chain_oi" in flags,
        "hot_chains": "hot_chains" in flags,
        "option_volume_oi_changes": oi_change,
        "premium_concentration": hot_premium,
        "uw_direction": uw_direction,
        "catalyst_direction_bias": direction_bias,
        "direction_confirmed": direction_confirmed,
        "sector_etf_confirmation": ticker in ETF_TICKERS and uw_confirmed,
        "liquidity_quote_quality": quote_quality,
        "liquidity_volume": quote.get("volume"),
        "liquidity_open_interest": quote.get("open_interest"),
        "bid_ask_spread_pct": quote.get("spread_pct"),
        "dte": quote.get("dte"),
        "flow_premium_bias": flow_bias,
    }


def direction_for_quote(direction_bias: str, uw_direction: str) -> str:
    if direction_bias in {"bullish", "bearish"}:
        return direction_bias
    if uw_direction in {"bullish", "bearish"}:
        return uw_direction
    return "bullish"


def infer_uw_direction(feature: Mapping[str, Any]) -> str:
    flow_bias = first_num(feature.get("flow_premium_bias"), feature.get("premium_bias"))
    if flow_bias is not None:
        if flow_bias > 0.05:
            return "bullish"
        if flow_bias < -0.05:
            return "bearish"
    call_pressure = max(_num(feature.get("hot_call_ask_ratio")) or 0.0, _num(feature.get("flow_call_ask_ratio")) or 0.0)
    put_pressure = max(_num(feature.get("hot_put_ask_ratio")) or 0.0, _num(feature.get("flow_put_ask_ratio")) or 0.0)
    if call_pressure >= 0.55 and call_pressure > put_pressure:
        return "bullish"
    if put_pressure >= 0.55 and put_pressure > call_pressure:
        return "bearish"
    if (_num(feature.get("oi_call_diff")) or 0.0) > (_num(feature.get("oi_put_diff")) or 0.0):
        return "bullish"
    if (_num(feature.get("oi_put_diff")) or 0.0) > (_num(feature.get("oi_call_diff")) or 0.0):
        return "bearish"
    return "mixed"


def option_quote_quality(quote: Mapping[str, Any]) -> str:
    if not quote:
        return "no_tradeable_option_quote"
    spread = _num(quote.get("spread_pct"))
    volume = _num(quote.get("volume")) or 0.0
    open_interest = _num(quote.get("open_interest")) or 0.0
    dte = _num(quote.get("dte"))
    bid = _num(quote.get("bid")) or 0.0
    ask = _num(quote.get("ask")) or 0.0
    if bid <= 0 or ask <= 0:
        return "no_tradeable_option_quote"
    if spread is None or spread > 0.35:
        return "bid/ask too wide"
    if volume < 50 or open_interest < 25:
        return "volume/open interest too low"
    if dte is None or dte < 7:
        return "DTE too short"
    return "tradeable"


def build_promotion_decision_rows(
    catalysts: Sequence[Mapping[str, Any]],
    confirmations: Sequence[Mapping[str, Any]],
    daily_rows: Sequence[Mapping[str, Any]],
    source_complete: bool,
    missing_sources: Sequence[str],
    as_of: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    confirmation_by_key = {(r.get("catalyst_id"), r.get("ticker")): r for r in confirmations}
    daily_by_ticker = best_daily_rows_by_ticker(daily_rows)

    if not source_complete:
        rows.append(
            promotion_row(
                as_of=as_of,
                scenario_bucket="BLOCKED_SOURCE_INCOMPLETE",
                catalyst={},
                ticker="",
                confirmation={},
                daily_row={},
                blocker="source incomplete: " + "; ".join(missing_sources),
                artifact="missed_pattern_audit.md",
            )
        )

    for catalyst in catalysts:
        mapped = list(catalyst.get("mapped_tickers") or []) + list(catalyst.get("mapped_etfs") or [])
        if not catalyst.get("as_of_eligible"):
            rows.append(
                promotion_row(
                    as_of=as_of,
                    scenario_bucket="POINT_IN_TIME_INELIGIBLE_CATALYST",
                    catalyst=catalyst,
                    ticker=";".join(mapped[:8]),
                    confirmation={},
                    daily_row={},
                    blocker=f"future-dated catalyst: {catalyst.get('ineligible_reason')}",
                    artifact="macro_geo_catalysts.json",
                )
            )
            continue
        if not mapped:
            rows.append(
                promotion_row(
                    as_of=as_of,
                    scenario_bucket="CATALYST_NO_UW_CONFIRMATION",
                    catalyst=catalyst,
                    ticker="",
                    confirmation={},
                    daily_row={},
                    blocker="eligible catalyst has no mapped ticker/ETF universe",
                    artifact="macro_geo_ticker_map.csv",
                )
            )
            continue
        for ticker in mapped:
            confirmation = confirmation_by_key.get((catalyst.get("catalyst_id"), ticker), {})
            daily_row = daily_by_ticker.get(ticker, {})
            bucket, blocker = classify_promotion_bucket(
                catalyst=catalyst,
                confirmation=confirmation,
                daily_row=daily_row,
                source_complete=source_complete,
                missing_sources=missing_sources,
            )
            rows.append(
                promotion_row(
                    as_of=as_of,
                    scenario_bucket=bucket,
                    catalyst=catalyst,
                    ticker=ticker,
                    confirmation=confirmation,
                    daily_row=daily_row,
                    blocker=blocker,
                    artifact="macro_geo_promotion_decisions.csv",
                )
            )
    add_approved_trade_rows(rows, daily_rows, as_of)
    return rows


def best_daily_rows_by_ticker(daily_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    ranked = {"TRADE": 4, "WATCH": 3, "AVOID": 2, "BLOCKED": 1}
    out: Dict[str, Mapping[str, Any]] = {}
    for row in daily_rows:
        ticker = str(row.get("ticker") or "")
        if not ticker:
            continue
        existing = out.get(ticker)
        if existing is None or ranked.get(str(row.get("classification")), 0) > ranked.get(str(existing.get("classification")), 0):
            out[ticker] = row
    return out


def classify_promotion_bucket(
    catalyst: Mapping[str, Any],
    confirmation: Mapping[str, Any],
    daily_row: Mapping[str, Any],
    source_complete: bool,
    missing_sources: Sequence[str],
) -> Tuple[str, str]:
    if not source_complete:
        return "BLOCKED_SOURCE_INCOMPLETE", "source incomplete: " + "; ".join(missing_sources)
    if not catalyst.get("as_of_eligible"):
        return "POINT_IN_TIME_INELIGIBLE_CATALYST", f"future-dated catalyst: {catalyst.get('ineligible_reason')}"
    if confirmation.get("sector_etf_confirmation"):
        return "SECTOR_INDEX_CONFIRMED_SETUP", confirmation.get("uw_evidence_found") or "sector/index UW confirmation"
    if not confirmation.get("uw_confirmed"):
        return "CATALYST_NO_UW_CONFIRMATION", "catalyst found but UW did not confirm mapped ticker/theme"
    blockers = set(daily_row.get("block_reasons") or [])
    classification = str(daily_row.get("classification") or "")
    if classification == "TRADE":
        return "APPROVED_TRADE", ""
    if blockers & REGIME_BLOCKERS:
        return "REGIME_CONFLICTED_SETUP", join_list(decompose_blockers(blockers))
    if blockers & LIQUIDITY_OR_QUOTE_BLOCKERS:
        return "LIQUIDITY_OR_QUOTE_BLOCKED_SETUP", join_list(decompose_blockers(blockers))
    if blockers & VALIDATION_BLOCKERS:
        return "VALIDATION_BLOCKED_SETUP", join_list(decompose_blockers(blockers))
    if daily_row:
        return "CATALYST_CONFIRMED_TRADE_REVIEW", "UW-confirmed catalyst setup needs trade review before approval"
    return "CATALYST_WATCH", "eligible catalyst and UW evidence surfaced, but no validated trade row was promoted"


def promotion_row(
    as_of: str,
    scenario_bucket: str,
    catalyst: Mapping[str, Any],
    ticker: str,
    confirmation: Mapping[str, Any],
    daily_row: Mapping[str, Any],
    blocker: str,
    artifact: str,
) -> Dict[str, Any]:
    return {
        "as_of": as_of,
        "scenario_bucket": scenario_bucket,
        "catalyst_id": catalyst.get("catalyst_id", ""),
        "event_type": catalyst.get("event_type", ""),
        "capture_date": catalyst.get("capture_date", ""),
        "source_file": catalyst.get("source_file", ""),
        "ticker": ticker,
        "direction_bias": catalyst.get("direction_bias", daily_row.get("direction", "")),
        "confidence": catalyst.get("confidence", ""),
        "affected_themes": join_list(catalyst.get("affected_themes")),
        "affected_sectors": join_list(catalyst.get("affected_sectors")),
        "uw_confirmed": confirmation.get("uw_confirmed", ""),
        "uw_evidence_found": confirmation.get("uw_evidence_found", ""),
        "uw_evidence_score": confirmation.get("uw_evidence_score", ""),
        "daily_classification": daily_row.get("classification", ""),
        "pattern_family": daily_row.get("pattern_family", ""),
        "block_reasons": join_list(daily_row.get("block_reasons")),
        "blocker_categories": join_list(decompose_blockers(daily_row.get("block_reasons") or [])),
        "promotion_blocker": blocker,
        "artifact_path": artifact,
    }


def add_approved_trade_rows(rows: List[Dict[str, Any]], daily_rows: Sequence[Mapping[str, Any]], as_of: str) -> None:
    tickers_with_rows = {row.get("ticker") for row in rows}
    for daily_row in daily_rows:
        if daily_row.get("classification") != "TRADE" or daily_row.get("ticker") in tickers_with_rows:
            continue
        rows.append(
            promotion_row(
                as_of=as_of,
                scenario_bucket="APPROVED_TRADE",
                catalyst={},
                ticker=str(daily_row.get("ticker") or ""),
                confirmation={},
                daily_row=daily_row,
                blocker="",
                artifact="actionable_trades.csv",
            )
        )


def build_multi_day_continuation_rows(
    catalysts: Sequence[Mapping[str, Any]],
    snapshots: Mapping[str, Any],
    as_of: str,
    source_dates: Sequence[str],
) -> List[Dict[str, Any]]:
    eligible = [c for c in catalysts if c.get("as_of_eligible") and c.get("capture_date") <= as_of]
    grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for catalyst in eligible:
        key = (str(catalyst.get("event_type")), "|".join(catalyst.get("affected_themes") or []))
        grouped[key].append(catalyst)
    rows: List[Dict[str, Any]] = []
    recent_dates = set(sorted([d for d in source_dates if d <= as_of])[-5:])
    for (_, _), items in grouped.items():
        dates = sorted({str(c.get("capture_date")) for c in items if str(c.get("capture_date")) in recent_dates or not recent_dates})
        if len(dates) < 2:
            continue
        mapped = sorted({t for c in items for t in list(c.get("mapped_tickers") or []) + list(c.get("mapped_etfs") or [])})
        scores = [(d, score_group_confirmation(mapped, snapshots.get(d))) for d in dates if d in snapshots]
        trend = confirmation_trend(scores)
        representative = items[-1]
        rows.append(
            {
                "as_of": as_of,
                "scenario_bucket": "MULTI_DAY_CONTINUING_CATALYST",
                "catalyst_id": representative.get("catalyst_id", ""),
                "event_type": representative.get("event_type", ""),
                "capture_date": ",".join(dates),
                "source_file": "multiple local captures",
                "ticker": ";".join(mapped[:12]),
                "direction_bias": representative.get("direction_bias", ""),
                "confidence": representative.get("confidence", ""),
                "affected_themes": join_list(representative.get("affected_themes")),
                "affected_sectors": join_list(representative.get("affected_sectors")),
                "uw_confirmed": any(score > 0 for _, score in scores),
                "uw_evidence_found": f"multi-day confirmation trend: {trend}",
                "uw_evidence_score": scores[-1][1] if scores else "",
                "daily_classification": "",
                "pattern_family": "",
                "block_reasons": "",
                "blocker_categories": "",
                "promotion_blocker": "" if scores else "insufficient snapshot history for trend scoring",
                "artifact_path": "macro_geo_promotion_decisions.csv",
            }
        )
    return rows


def score_group_confirmation(mapped_tickers: Sequence[str], snapshot: Any) -> int:
    if snapshot is None:
        return 0
    features = getattr(snapshot, "features", {})
    score = 0
    for ticker in mapped_tickers:
        feature = features.get(ticker, {})
        if not feature:
            continue
        flags = set(feature.get("source_flags") or [])
        if "bot_eod" in flags:
            score += 1
        if "hot_chains" in flags:
            score += 1
        if "chain_oi" in flags:
            score += 1
        if max(_num(feature.get("hot_total_premium")) or 0.0, _num(feature.get("flow_total_premium")) or 0.0) >= 100_000:
            score += 1
    return score


def confirmation_trend(scores: Sequence[Tuple[str, int]]) -> str:
    if len(scores) < 2:
        return "insufficient_snapshot_history"
    first = scores[0][1]
    last = scores[-1][1]
    if last > first:
        return "improved"
    if last < first:
        return "faded"
    return "stable"


def build_no_pattern_rows(
    as_of: str,
    catalysts: Sequence[Mapping[str, Any]],
    promotion_rows: Sequence[Mapping[str, Any]],
    daily_rows: Sequence[Mapping[str, Any]],
    source_complete: bool,
) -> List[Dict[str, Any]]:
    if any(r.get("scenario_bucket") == "APPROVED_TRADE" for r in promotion_rows):
        return []
    if not source_complete:
        return []
    eligible = [c for c in catalysts if c.get("as_of_eligible")]
    blockers = Counter(str(r.get("scenario_bucket")) for r in promotion_rows if r.get("scenario_bucket"))
    if eligible or daily_rows:
        blocker = "no approved trades; " + ", ".join(f"{k}:{v}" for k, v in blockers.most_common(5))
    else:
        blocker = "no eligible catalyst found and no approved pattern trade"
    return [
        {
            "as_of": as_of,
            "scenario_bucket": "NO_PATTERN_FOUND_WITH_EVIDENCE",
            "catalyst_id": "",
            "event_type": "run_summary",
            "capture_date": as_of,
            "source_file": "",
            "ticker": ";".join(sorted({str(r.get("ticker")) for r in daily_rows if r.get("ticker")})[:12]),
            "direction_bias": "",
            "confidence": "",
            "affected_themes": "",
            "affected_sectors": "",
            "uw_confirmed": any(r.get("uw_confirmed") for r in promotion_rows),
            "uw_evidence_found": "see macro_geo_uw_confirmation.csv and watchlist/blocked CSVs",
            "uw_evidence_score": "",
            "daily_classification": "",
            "pattern_family": "",
            "block_reasons": "",
            "blocker_categories": "",
            "promotion_blocker": blocker,
            "artifact_path": "missed_pattern_audit.md",
        }
    ]


def build_observability_matrix_rows(as_of: str, promotion_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    by_bucket: Dict[str, Mapping[str, Any]] = {}
    for row in promotion_rows:
        bucket = str(row.get("scenario_bucket") or "")
        by_bucket.setdefault(bucket, row)
    for scenario in SCENARIO_BUCKETS:
        match = by_bucket.get(scenario)
        if match:
            actual = scenario
            tickers = str(match.get("ticker") or "")
            uw_evidence = str(match.get("uw_evidence_found") or "")
            blocker = str(match.get("promotion_blocker") or "")
            artifact = str(match.get("artifact_path") or "macro_geo_promotion_decisions.csv")
        else:
            actual = "NO_REAL_DATA_MATCH_THIS_RUN"
            tickers = ""
            uw_evidence = "covered by focused unit tests or absent in this run"
            blocker = ""
            artifact = "pattern_observability_matrix.md"
        rows.append(
            {
                "scenario_name": scenario,
                "date_tested": as_of,
                "expected_behavior": SCENARIO_EXPECTED_BEHAVIOR[scenario],
                "actual_output_bucket": actual,
                "tickers_sectors_involved": tickers,
                "uw_evidence_found": uw_evidence,
                "blocker_if_not_approved": blocker,
                "artifact_proving_it_surfaced": artifact,
            }
        )
    return rows


def build_missed_pattern_audit_rows(
    as_of: str,
    catalysts: Sequence[Mapping[str, Any]],
    promotion_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    eligible = [c for c in catalysts if c.get("as_of_eligible")]
    ineligible = [c for c in catalysts if not c.get("as_of_eligible")]
    final_bucket = "BLOCKED_SOURCE_INCOMPLETE" if not summary.get("source_complete") else primary_bucket(promotion_rows)
    tickers = sorted(
        {
            t
            for c in eligible
            for t in list(c.get("mapped_tickers") or []) + list(c.get("mapped_etfs") or [])
        }
    )
    return [
        {
            "date": as_of,
            "source_completeness": "complete" if summary.get("source_complete") else "incomplete",
            "eligible_catalysts": "; ".join(sorted({str(c.get("event_type")) for c in eligible})) or "none",
            "ineligible_future_dated_catalysts": "; ".join(
                sorted({f"{c.get('event_type')}:{c.get('ineligible_reason')}" for c in ineligible})
            )
            or "none",
            "mapped_tickers_sectors": "; ".join(tickers[:20]) or "none",
            "uw_confirmation": summary.get("uw_confirmed_themes") or "none",
            "final_scenario_bucket": final_bucket,
            "promotion_blocker": first_blocker(promotion_rows),
            "artifact_path": "macro_geo_promotion_decisions.csv",
        }
    ]


def primary_bucket(promotion_rows: Sequence[Mapping[str, Any]]) -> str:
    priority = {bucket: idx for idx, bucket in enumerate(SCENARIO_BUCKETS)}
    if not promotion_rows:
        return "NO_PATTERN_FOUND_WITH_EVIDENCE"
    return min((str(r.get("scenario_bucket") or "") for r in promotion_rows), key=lambda b: priority.get(b, 999))


def first_blocker(promotion_rows: Sequence[Mapping[str, Any]]) -> str:
    for row in promotion_rows:
        blocker = str(row.get("promotion_blocker") or "")
        if blocker:
            return blocker
    return ""


def summarize_macro_geo_bundle(
    catalysts: Sequence[Mapping[str, Any]],
    confirmations: Sequence[Mapping[str, Any]],
    promotion_rows: Sequence[Mapping[str, Any]],
    source_complete: bool,
    missing_sources: Sequence[str],
) -> Dict[str, Any]:
    eligible = [c for c in catalysts if c.get("as_of_eligible")]
    ineligible = [c for c in catalysts if not c.get("as_of_eligible")]
    confirmed = [r for r in confirmations if r.get("uw_confirmed")]
    confirmed_ids = {r.get("catalyst_id") for r in confirmed}
    confirmed_themes = []
    for catalyst in eligible:
        if catalyst.get("catalyst_id") in confirmed_ids:
            confirmed_themes.extend(catalyst.get("affected_themes") or [])
    watch_or_blocked = [
        r.get("ticker")
        for r in promotion_rows
        if r.get("ticker") and r.get("scenario_bucket") not in {"POINT_IN_TIME_INELIGIBLE_CATALYST", "NO_PATTERN_FOUND_WITH_EVIDENCE"}
    ]
    bucket_counts = Counter(str(r.get("scenario_bucket")) for r in promotion_rows if r.get("scenario_bucket"))
    return {
        "source_complete": source_complete,
        "missing_sources": list(missing_sources),
        "eligible_catalyst_count": len(eligible),
        "eligible_event_types": sorted({str(c.get("event_type")) for c in eligible}),
        "future_dated_catalyst_count": len(ineligible),
        "future_dated_event_types": sorted({str(c.get("event_type")) for c in ineligible}),
        "uw_confirmed_catalyst_count": len(confirmed_ids),
        "uw_confirmed_themes": "; ".join(sorted(set(confirmed_themes))),
        "watch_or_blocked_names": "; ".join(list(dict.fromkeys(watch_or_blocked))[:15]),
        "scenario_bucket_counts": dict(bucket_counts),
        "approved_trade_count": bucket_counts.get("APPROVED_TRADE", 0),
        "primary_no_trade_reason": first_blocker([r for r in promotion_rows if r.get("scenario_bucket") == "NO_PATTERN_FOUND_WITH_EVIDENCE"])
        or first_blocker(promotion_rows),
    }


def render_pattern_observability_matrix(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Pattern Observability Matrix",
        "",
        "| Scenario name | Date tested | Expected behavior | Actual output bucket | Tickers/sectors involved | UW evidence found | Blocker if not approved | Artifact proving it surfaced |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                markdown_cell(row.get(key))
                for key in (
                    "scenario_name",
                    "date_tested",
                    "expected_behavior",
                    "actual_output_bucket",
                    "tickers_sectors_involved",
                    "uw_evidence_found",
                    "blocker_if_not_approved",
                    "artifact_proving_it_surfaced",
                )
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def render_missed_pattern_audit(rows: Sequence[Mapping[str, Any]], missing_sources: Sequence[str]) -> str:
    lines = [
        "# Missed Pattern Audit",
        "",
    ]
    if missing_sources:
        lines.append("Missing source data:")
        for item in missing_sources:
            lines.append(f"- {item}")
        lines.append("")
    lines.extend(
        [
            "| Date | Source completeness | Eligible catalysts | Ineligible/future-dated catalysts | Mapped tickers/sectors | UW confirmation | Final scenario bucket | Promotion blocker | Artifact path |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                markdown_cell(row.get(key))
                for key in (
                    "date",
                    "source_completeness",
                    "eligible_catalysts",
                    "ineligible_future_dated_catalysts",
                    "mapped_tickers_sectors",
                    "uw_confirmation",
                    "final_scenario_bucket",
                    "promotion_blocker",
                    "artifact_path",
                )
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def decompose_blockers(blockers: Iterable[str]) -> List[str]:
    labels: List[str] = []
    for blocker in blockers:
        label = BLOCKER_LABELS.get(str(blocker), str(blocker).lower().replace("_", " "))
        if label not in labels:
            labels.append(label)
    return labels


def macro_geo_ticker_map_fieldnames() -> List[str]:
    return [
        "catalyst_id",
        "as_of",
        "event_type",
        "capture_date",
        "as_of_eligible",
        "ticker",
        "instrument_type",
        "affected_themes",
        "affected_sectors",
        "direction_bias",
        "confidence",
        "source_file",
    ]


def macro_geo_confirmation_fieldnames() -> List[str]:
    return [
        "catalyst_id",
        "as_of",
        "capture_date",
        "event_type",
        "ticker",
        "instrument_type",
        "as_of_eligible",
        "uw_confirmed",
        "uw_evidence_score",
        "uw_evidence_found",
        "source_flags",
        "bot_eod_options_flow",
        "chain_oi_changes",
        "hot_chains",
        "option_volume_oi_changes",
        "premium_concentration",
        "uw_direction",
        "catalyst_direction_bias",
        "direction_confirmed",
        "sector_etf_confirmation",
        "liquidity_quote_quality",
        "liquidity_volume",
        "liquidity_open_interest",
        "bid_ask_spread_pct",
        "dte",
        "flow_premium_bias",
    ]


def macro_geo_promotion_fieldnames() -> List[str]:
    return [
        "as_of",
        "scenario_bucket",
        "catalyst_id",
        "event_type",
        "capture_date",
        "source_file",
        "ticker",
        "direction_bias",
        "confidence",
        "affected_themes",
        "affected_sectors",
        "uw_confirmed",
        "uw_evidence_found",
        "uw_evidence_score",
        "daily_classification",
        "pattern_family",
        "block_reasons",
        "blocker_categories",
        "promotion_blocker",
        "artifact_path",
    ]


def missed_pattern_audit_fieldnames() -> List[str]:
    return [
        "date",
        "source_completeness",
        "eligible_catalysts",
        "ineligible_future_dated_catalysts",
        "mapped_tickers_sectors",
        "uw_confirmation",
        "final_scenario_bucket",
        "promotion_blocker",
        "artifact_path",
    ]


def stable_id(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def clean_ticker(value: str) -> str:
    return re.sub(r"[^A-Z0-9.\-]", "", str(value or "").upper())


def join_list(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, set):
        return "; ".join(str(v) for v in sorted(value))
    if isinstance(value, (list, tuple)):
        return "; ".join(str(v) for v in value)
    return str(value)


def first_num(*values: Any) -> Optional[float]:
    for value in values:
        parsed = _num(value)
        if parsed is not None:
            return parsed
    return None


def _num(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def markdown_cell(value: Any) -> str:
    text = join_list(value)
    text = text.replace("\n", " ").replace("|", "\\|")
    return text or " "
