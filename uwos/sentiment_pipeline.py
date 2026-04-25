#!/usr/bin/env python3
"""
Multi-source sentiment and positioning pipeline for UW trade research.

The pipeline is intentionally deterministic and local-first:
- UW dated folders supply options flow, OI changes, dark-pool activity, prices, and sectors.
- Browser/social/news artifacts supply text sentiment when available.
- Existing trend-analysis outputs supply order-ticket readiness; this module does not invent
  exact option trades unless the backtest/live gate has already produced them.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from uwos import paths
from uwos import trend_analysis


DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
OPTION_RE = re.compile(r"^([A-Z][A-Z0-9.]{0,8}?)(\d{6})([CP])")
BATCH_PROOF_RE = re.compile(
    r"^trend-analysis-batch-(?P<kind>.+)-(?P<start>\d{4}-\d{2}-\d{2})_"
    r"(?P<end>\d{4}-\d{2}-\d{2})-L(?P<lookback>\d+)\.(?P<ext>csv|json|md)$"
)
DEFAULT_LOOKBACK = 20
DEFAULT_TOP = 25
DEFAULT_OUT_SUBDIR = "out/sentiment_pipeline"

COMMON_TICKER_WORDS = {
    "A",
    "AI",
    "ALL",
    "AM",
    "AND",
    "ARE",
    "AT",
    "BE",
    "BIG",
    "BUY",
    "CALL",
    "CEO",
    "CFO",
    "CPI",
    "DAY",
    "DYOR",
    "DTE",
    "ETF",
    "EU",
    "EPS",
    "FED",
    "FOMC",
    "FOR",
    "GDP",
    "HAS",
    "HE",
    "HIGH",
    "HOLD",
    "I",
    "IN",
    "IPO",
    "IR",
    "IT",
    "IV",
    "LOW",
    "MACD",
    "NEW",
    "NO",
    "NOT",
    "OI",
    "OIL",
    "ON",
    "OR",
    "OTM",
    "PCR",
    "PM",
    "PUT",
    "RSI",
    "SEC",
    "SELL",
    "SOLD",
    "THE",
    "TO",
    "US",
    "USA",
    "VIX",
    "WAR",
    "WE",
    "WHY",
    "YOY",
}

STOP_WORDS = {
    "about",
    "after",
    "again",
    "against",
    "also",
    "and",
    "are",
    "because",
    "been",
    "before",
    "being",
    "between",
    "from",
    "have",
    "into",
    "market",
    "markets",
    "options",
    "over",
    "stock",
    "stocks",
    "that",
    "their",
    "there",
    "this",
    "through",
    "today",
    "trade",
    "trades",
    "with",
    "would",
}

BULLISH_PHRASES: Dict[str, float] = {
    "call buying": 3.5,
    "calls bought": 3.0,
    "unusual call": 2.5,
    "bullish flow": 3.0,
    "accumulation": 2.5,
    "breakout": 2.2,
    "new highs": 2.2,
    "beat estimates": 2.5,
    "raises guidance": 3.0,
    "upgrade": 2.0,
    "upgraded": 2.0,
    "outperform": 1.8,
    "buy rating": 1.8,
    "price target raised": 2.0,
    "strong demand": 2.0,
    "margin expansion": 1.8,
    "contract win": 2.2,
    "buyback": 1.8,
    "short squeeze": 1.8,
    "momentum": 1.4,
}

BEARISH_PHRASES: Dict[str, float] = {
    "put buying": 3.5,
    "puts bought": 3.0,
    "unusual put": 2.5,
    "bearish flow": 3.0,
    "distribution": 2.5,
    "breakdown": 2.2,
    "new lows": 2.2,
    "missed estimates": 2.5,
    "cuts guidance": 3.0,
    "cut guidance": 3.0,
    "downgrade": 2.0,
    "downgraded": 2.0,
    "underperform": 1.8,
    "sell rating": 1.8,
    "price target cut": 2.0,
    "weak demand": 2.0,
    "margin pressure": 1.8,
    "investigation": 2.2,
    "lawsuit": 1.8,
    "sanctions risk": 2.0,
    "recession risk": 2.0,
    "credit stress": 2.0,
    "profit warning": 2.5,
}

THEME_RULES: List[Dict[str, Any]] = [
    {
        "name": "middle_east_war_energy_defense",
        "keywords": {
            "iran",
            "israel",
            "hormuz",
            "middle east",
            "missile",
            "sanction",
            "sanctions",
            "war",
            "oil shock",
            "geopolitical",
        },
        "bullish_tickers": {
            "XLE",
            "XOP",
            "USO",
            "UCO",
            "XOM",
            "CVX",
            "OXY",
            "COP",
            "SLB",
            "HAL",
            "LMT",
            "RTX",
            "NOC",
            "GD",
            "HII",
            "PLTR",
            "GLD",
            "GDX",
            "VIX",
            "UVXY",
        },
        "bearish_tickers": {
            "SPY",
            "QQQ",
            "IWM",
            "DAL",
            "UAL",
            "AAL",
            "LUV",
            "JBLU",
            "CCL",
            "RCL",
            "NCLH",
            "BA",
        },
        "bullish_sectors": {"Energy"},
        "bearish_sectors": {"Consumer Cyclical"},
    },
    {
        "name": "ai_semis_capex",
        "keywords": {"ai", "artificial intelligence", "gpu", "datacenter", "data center", "accelerator", "capex"},
        "bullish_tickers": {"NVDA", "AMD", "AVGO", "TSM", "MU", "SMH", "SOXX", "MSFT", "AMZN", "GOOGL", "GOOG"},
        "bearish_tickers": set(),
        "bullish_sectors": {"Technology"},
        "bearish_sectors": set(),
    },
    {
        "name": "rates_recession_risk",
        "keywords": {"recession", "stagflation", "rate shock", "yields spike", "hard landing", "credit stress"},
        "bullish_tickers": {"TLT", "GLD", "VIX", "UVXY"},
        "bearish_tickers": {"SPY", "QQQ", "IWM", "XLY", "KRE", "XLF"},
        "bullish_sectors": set(),
        "bearish_sectors": {"Consumer Cyclical", "Financial Services"},
    },
]

SOURCE_WEIGHTS = {
    "schwab_news": 1.45,
    "schwab": 1.45,
    "schwab_web": 1.45,
    "x": 1.50,
    "reddit": 1.35,
    "news": 1.25,
    "analysts": 1.20,
    "sec": 1.15,
    "institutions": 1.05,
    "insiders": 1.05,
    "browser": 1.10,
    "snapshot": 0.60,
    "packet": 0.70,
}

ENERGY_CONTEXT_TERMS = {
    "barrel",
    "brent",
    "crude",
    "energy",
    "gas",
    "hormuz",
    "lng",
    "oil",
    "opec",
    "refiner",
    "shipping lane",
    "strait",
    "tanker",
}
DEFENSE_CONTEXT_TERMS = {
    "defense",
    "destroyer",
    "drone",
    "missile",
    "military",
    "navy",
    "pentagon",
    "ship",
    "warship",
    "weapon",
}
TRAVEL_CONTEXT_TERMS = {
    "airfare",
    "airline",
    "airlines",
    "booking",
    "bookings",
    "cruise",
    "cruises",
    "jet fuel",
    "tourism",
    "travel",
}
MARKET_CONTEXT_TERMS = {"equities", "market", "markets", "risk off", "selloff", "stocks", "volatility"}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score ticker or market-segment sentiment from UW flow, news/social text, macro, and trend outputs."
    )
    parser.add_argument("query", nargs="*", help="Ticker, segment, or catalyst query, e.g. NFLX or 'Iran war'.")
    parser.add_argument("--ticker", action="append", default=[], help="Ticker to analyze. Repeatable.")
    parser.add_argument("--tickers", default="", help="Comma-separated ticker list.")
    parser.add_argument("--as-of", default="", help="YYYY-MM-DD date folder. Defaults to latest dated UW folder.")
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK, help="Usable dated-folder lookback.")
    parser.add_argument("--root-dir", default="", help="UW trade desk root. Defaults to project root.")
    parser.add_argument("--out-dir", default="", help="Output directory. Defaults to <root>/out/sentiment_pipeline.")
    parser.add_argument("--top", type=int, default=DEFAULT_TOP, help="Rows to show in markdown report.")
    parser.add_argument("--max-docs", type=int, default=800, help="Maximum text/news/social documents to inspect.")
    parser.add_argument(
        "--schwab-news",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fetch Schwab/broker news for the candidate universe when Schwab API credentials are available.",
    )
    parser.add_argument("--schwab-news-limit", type=int, default=20, help="Maximum Schwab news items to request.")
    parser.add_argument(
        "--schwab-news-max-tickers",
        type=int,
        default=25,
        help="Maximum tickers to include in one Schwab news request.",
    )
    parser.add_argument("--manual-auth", action="store_true", help="Use manual Schwab OAuth flow for Schwab news.")
    parser.add_argument(
        "--run-trend-analysis",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run/refresh uwos.trend_analysis before attaching trade artifacts to sentiment rows.",
    )
    parser.add_argument(
        "--trend-lookback",
        type=int,
        default=0,
        help="Trend-analysis lookback to run when --run-trend-analysis is enabled. Defaults to --lookback.",
    )
    parser.add_argument(
        "--trend-top",
        type=int,
        default=0,
        help="Trend-analysis actionable top count. Defaults to max(5, --top).",
    )
    parser.add_argument(
        "--trend-candidate-top",
        type=int,
        default=0,
        help="Trend-analysis candidate/workup top count. Defaults to max(12, --top).",
    )
    parser.add_argument(
        "--trend-no-schwab",
        action="store_true",
        help="Run trend-analysis with --no-schwab if Schwab validation is unavailable.",
    )
    parser.add_argument(
        "--trend-reuse-audits",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse existing trend-analysis audit artifacts when refreshing trade workups.",
    )
    parser.add_argument(
        "--include-pattern-trades",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also surface trend-analysis pattern-only rows. Default false.",
    )
    parser.add_argument(
        "--batch-proof-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Require trend-analysis trade artifacts to match a supportive rolling playbook "
            "from trend_analysis_batch before they are shown as trade considerations."
        ),
    )
    parser.add_argument(
        "--batch-proof-dir",
        default="",
        help="Directory containing trend-analysis-batch proof artifacts. Defaults to <root>/out/trend_analysis_batch.",
    )
    return parser.parse_args(argv)


def _safe_float(value: Any) -> float:
    if value is None:
        return math.nan
    if isinstance(value, str):
        text = value.strip().replace(",", "").replace("$", "").replace("%", "")
        if not text or text.lower() in {"nan", "none", "null", "-"}:
            return math.nan
        value = text
    try:
        return float(value)
    except Exception:
        return math.nan


def _safe_int(value: Any) -> int:
    num = _safe_float(value)
    return int(num) if math.isfinite(num) else 0


def _clip(value: float, low: float = -100.0, high: float = 100.0) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(low, min(high, value))


def _fmt_num(value: Any, places: int = 1) -> str:
    num = _safe_float(value)
    if not math.isfinite(num):
        return "-"
    return f"{num:.{places}f}"


def _fmt_money(value: Any) -> str:
    num = _safe_float(value)
    if not math.isfinite(num):
        return "-"
    sign = "-" if num < 0 else ""
    num = abs(num)
    if num >= 1_000_000_000:
        return f"{sign}${num / 1_000_000_000:.1f}B"
    if num >= 1_000_000:
        return f"{sign}${num / 1_000_000:.1f}M"
    if num >= 1_000:
        return f"{sign}${num / 1_000:.1f}K"
    return f"{sign}${num:.0f}"


def _fmt_percent(value: Any) -> str:
    num = _safe_float(value)
    if not math.isfinite(num):
        return "-"
    return f"{num * 100:.0f}%"


def _slug(text: str) -> str:
    out = re.sub(r"[^A-Za-z0-9._-]+", "-", text.strip())
    out = re.sub(r"-{2,}", "-", out).strip("-")
    return (out or "market").lower()[:80]


def _parse_date(text: str) -> dt.date:
    try:
        return dt.date.fromisoformat(str(text).strip()[:10])
    except Exception as exc:
        raise SystemExit(f"Invalid date: {text}") from exc


def _query_text(args: argparse.Namespace) -> str:
    return " ".join(args.query).strip()


def _query_terms(query: str) -> List[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9.+-]{1,}", query.lower())
    return [w for w in words if w not in STOP_WORDS and len(w) >= 3]


def _query_term_in_text(lower_text: str, term: str) -> bool:
    term = str(term or "").lower().strip()
    if not term:
        return False
    if re.search(r"[^a-z0-9]", term):
        return term in lower_text
    return re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", lower_text) is not None


def _query_overlap_count(lower_text: str, query_terms: Sequence[str]) -> int:
    return sum(1 for term in query_terms if _query_term_in_text(lower_text, term))


def _normalize_ticker(text: Any) -> str:
    raw = str(text or "").strip().upper().lstrip("$")
    return re.sub(r"[^A-Z0-9.]", "", raw)


def extract_option_underlying(option_symbol: Any) -> Tuple[str, str]:
    text = _normalize_ticker(option_symbol)
    match = OPTION_RE.match(text)
    if not match:
        return "", ""
    return match.group(1).upper(), match.group(3).upper()


def extract_tickers(text: str) -> List[str]:
    found = set()
    for match in re.finditer(r"\$([A-Za-z][A-Za-z0-9.]{0,7})\b", text or ""):
        ticker = _normalize_ticker(match.group(1))
        if ticker and ticker not in COMMON_TICKER_WORDS:
            found.add(ticker)
    for match in re.finditer(r"\b[A-Z][A-Z0-9.]{1,7}\b", text or ""):
        ticker = _normalize_ticker(match.group(0))
        if ticker and ticker not in COMMON_TICKER_WORDS:
            found.add(ticker)
    return sorted(found)


def iter_day_dirs(root: Path) -> List[Tuple[dt.date, Path]]:
    out: List[Tuple[dt.date, Path]] = []
    for child in root.iterdir():
        if not child.is_dir() or not DATE_RE.match(child.name):
            continue
        try:
            day = dt.date.fromisoformat(child.name)
        except Exception:
            continue
        out.append((day, child))
    out.sort(key=lambda x: x[0])
    return out


def find_csv(day_dir: Path, patterns: Sequence[str], include_unzipped: bool = True) -> Optional[Path]:
    candidates: List[Path] = []
    for pattern in patterns:
        candidates.extend(day_dir.glob(pattern))
    if include_unzipped:
        unzipped = day_dir / "_unzipped_mode_a"
        if unzipped.exists():
            for pattern in patterns:
                candidates.extend(unzipped.glob(pattern))
    candidates = [p for p in dict.fromkeys(candidates) if p.exists() and p.is_file()]
    if not candidates:
        return None

    def rank(path: Path) -> Tuple[int, int]:
        return int(path.stat().st_size), int(path.stat().st_mtime_ns)

    return max(candidates, key=rank)


def read_csv_rows(path: Optional[Path], limit: Optional[int] = None) -> List[Dict[str, Any]]:
    if not path or not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    try:
        with path.open("r", newline="", encoding="utf-8-sig", errors="replace") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rows.append(dict(row))
                if limit is not None and len(rows) >= limit:
                    break
    except Exception:
        return []
    return rows


def latest_data_date(root: Path) -> dt.date:
    dated = []
    for day, day_dir in iter_day_dirs(root):
        if find_csv(day_dir, ["stock-screener-*.csv"]):
            dated.append(day)
    if not dated:
        raise SystemExit(f"No dated UW stock-screener folders found under {root}")
    return dated[-1]


def trading_days_for_lookback(root: Path, as_of: dt.date, lookback: int) -> List[Tuple[dt.date, Path]]:
    days = []
    for day, day_dir in iter_day_dirs(root):
        if day > as_of:
            continue
        if find_csv(day_dir, ["stock-screener-*.csv"]):
            days.append((day, day_dir))
    if not days:
        raise SystemExit(f"No usable UW stock-screener folders found through {as_of}")
    return days[-max(1, int(lookback)) :]


def source_type(path: Path) -> str:
    name = path.name.lower()
    full = str(path).lower()
    if "schwab-web" in name or "schwab_web" in full:
        return "schwab_web"
    if "schwab-news" in name or "schwab_news" in full:
        return "schwab_news"
    if "reddit" in full:
        return "reddit"
    if "x_scrape" in full or "x-profile" in full or "twitter" in full:
        return "x"
    if "browser-text" in name or "browser_text" in name:
        return "browser"
    if "news-feed" in name or "news" in name:
        return "news"
    if "analyst" in name:
        return "analysts"
    if "sec" in name:
        return "sec"
    if "institution" in name:
        return "institutions"
    if "insider" in name:
        return "insiders"
    if "deep_research_packet" in name:
        return "packet"
    return "snapshot"


def row_source_type(path: Path, row: Dict[str, Any]) -> str:
    base = source_type(path)
    if base != "browser":
        return base
    row_source = str(row.get("source") or row.get("source_type") or "").strip().lower()
    if row_source in SOURCE_WEIGHTS:
        return row_source
    if row_source in {"twitter", "twitter/x", "x/twitter"}:
        return "x"
    return base


def is_auth_wall_text(source: str, text: str) -> bool:
    lower = str(text or "").lower()
    if source not in {"schwab", "schwab_web", "schwab_news"}:
        return False
    markers = [
        "enter security code",
        "security code has been sent",
        "use a different method",
        "unauthorized access is prohibited",
        "log in",
    ]
    return sum(1 for marker in markers if marker in lower) >= 2


def load_latest_stock_rows(days: Sequence[Tuple[dt.date, Path]]) -> Tuple[Path, Dict[str, Dict[str, Any]]]:
    latest_day, latest_dir = days[-1]
    path = find_csv(latest_dir, ["stock-screener-*.csv"])
    rows = read_csv_rows(path)
    by_ticker: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        ticker = _normalize_ticker(row.get("ticker"))
        if ticker:
            by_ticker[ticker] = row
    if path is None:
        raise SystemExit(f"Missing stock-screener file for {latest_day}")
    return path, by_ticker


def recency_weight(day: dt.date, as_of: dt.date) -> float:
    age = max(0, (as_of - day).days)
    return 1.0 / (1.0 + 0.12 * age)


def _allowed_filter(allowed_tickers: Optional[Iterable[str]]) -> Optional[set]:
    if allowed_tickers is None:
        return None
    allowed = {_normalize_ticker(t) for t in allowed_tickers if _normalize_ticker(t)}
    return allowed or None


def aggregate_hot_chains(
    days: Sequence[Tuple[dt.date, Path]],
    as_of: dt.date,
    allowed_tickers: Optional[Iterable[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    allowed = _allowed_filter(allowed_tickers)
    out: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"score_num": 0.0, "premium": 0.0, "rows": 0, "days": set()})
    for day, day_dir in days:
        path = find_csv(day_dir, ["hot-chains-*.csv"])
        if path is None:
            continue
        weight = recency_weight(day, as_of)
        try:
            df = pd.read_csv(
                path,
                usecols=lambda c: c
                in {"option_symbol", "premium", "ask_side_volume", "bid_side_volume", "volume", "avg_price"},
                low_memory=False,
            )
        except Exception:
            continue
        if df.empty or "option_symbol" not in df.columns:
            continue
        parts = df["option_symbol"].fillna("").astype(str).str.upper().str.extract(OPTION_RE)
        parts.columns = ["ticker", "expiry", "right"]
        df = df.assign(_ticker=parts["ticker"], _right=parts["right"])
        df = df[df["_ticker"].fillna("").ne("")]
        if allowed is not None:
            df = df[df["_ticker"].isin(allowed)]
        if df.empty:
            continue
        premium = pd.to_numeric(df.get("premium", pd.Series(0, index=df.index)), errors="coerce").abs()
        fallback = (
            pd.to_numeric(df.get("volume", pd.Series(0, index=df.index)), errors="coerce").clip(lower=0)
            * pd.to_numeric(df.get("avg_price", pd.Series(0, index=df.index)), errors="coerce").clip(lower=0)
            * 100.0
        )
        premium = premium.where(premium.gt(0), fallback).fillna(0.0)
        ask = pd.to_numeric(df.get("ask_side_volume", pd.Series(0, index=df.index)), errors="coerce").fillna(0.0).clip(lower=0)
        bid = pd.to_numeric(df.get("bid_side_volume", pd.Series(0, index=df.index)), errors="coerce").fillna(0.0).clip(lower=0)
        denom = ask + bid
        side_imbalance = ((ask - bid) / denom.replace(0, math.nan)).fillna(0.0)
        direction = side_imbalance.where(df["_right"].eq("C"), -side_imbalance)
        df = df.assign(_premium=premium, _score_num=direction * premium * weight)
        grouped = df[df["_premium"].gt(0)].groupby("_ticker", dropna=False)
        for ticker, group in grouped:
            item = out[str(ticker)]
            item["score_num"] += float(group["_score_num"].sum())
            item["premium"] += float(group["_premium"].sum() * weight)
            item["rows"] += int(len(group))
            item["days"].add(day.isoformat())
    for item in out.values():
        total = item["premium"]
        item["score"] = _clip(100.0 * item["score_num"] / total) if total > 0 else 0.0
        item["days_observed"] = len(item["days"])
        item["days"] = sorted(item["days"])
    return dict(out)


def aggregate_oi_changes(
    days: Sequence[Tuple[dt.date, Path]],
    as_of: dt.date,
    allowed_tickers: Optional[Iterable[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    allowed = _allowed_filter(allowed_tickers)
    out: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"score_num": 0.0, "premium": 0.0, "rows": 0, "days": set()})
    for day, day_dir in days:
        path = find_csv(day_dir, ["chain-oi-changes-*.csv"])
        if path is None:
            continue
        weight = recency_weight(day, as_of)
        try:
            df = pd.read_csv(
                path,
                usecols=lambda c: c
                in {
                    "option_symbol",
                    "underlying_symbol",
                    "prev_total_premium",
                    "prev_ask_volume",
                    "prev_bid_volume",
                    "curr_vol",
                    "avg_price",
                    "oi_diff_plain",
                },
                low_memory=False,
            )
        except Exception:
            continue
        if df.empty or "option_symbol" not in df.columns:
            continue
        parts = df["option_symbol"].fillna("").astype(str).str.upper().str.extract(OPTION_RE)
        parts.columns = ["opt_ticker", "expiry", "right"]
        ticker_col = (
            df.get("underlying_symbol", pd.Series("", index=df.index))
            .fillna("")
            .astype(str)
            .str.upper()
            .str.replace(r"[^A-Z0-9.]", "", regex=True)
        )
        df = df.assign(_ticker=ticker_col.where(ticker_col.ne(""), parts["opt_ticker"]), _right=parts["right"])
        df = df[df["_ticker"].fillna("").ne("")]
        if allowed is not None:
            df = df[df["_ticker"].isin(allowed)]
        if df.empty:
            continue
        premium = pd.to_numeric(df.get("prev_total_premium", pd.Series(0, index=df.index)), errors="coerce").abs()
        fallback = (
            pd.to_numeric(df.get("curr_vol", pd.Series(0, index=df.index)), errors="coerce").clip(lower=0)
            * pd.to_numeric(df.get("avg_price", pd.Series(0, index=df.index)), errors="coerce").clip(lower=0)
            * 100.0
        )
        oi_fallback = pd.to_numeric(df.get("oi_diff_plain", pd.Series(0, index=df.index)), errors="coerce").abs().clip(lower=1)
        premium = premium.where(premium.gt(0), fallback).where(lambda s: s.gt(0), oi_fallback).fillna(0.0)
        ask = pd.to_numeric(df.get("prev_ask_volume", pd.Series(0, index=df.index)), errors="coerce").fillna(0.0).clip(lower=0)
        bid = pd.to_numeric(df.get("prev_bid_volume", pd.Series(0, index=df.index)), errors="coerce").fillna(0.0).clip(lower=0)
        denom = ask + bid
        side_imbalance = ((ask - bid) / denom.replace(0, math.nan)).fillna(0.0)
        direction = side_imbalance.where(df["_right"].eq("C"), -side_imbalance)
        df = df.assign(_premium=premium, _score_num=direction * premium * weight)
        grouped = df[df["_premium"].gt(0)].groupby("_ticker", dropna=False)
        for ticker, group in grouped:
            item = out[str(ticker)]
            item["score_num"] += float(group["_score_num"].sum())
            item["premium"] += float(group["_premium"].sum() * weight)
            item["rows"] += int(len(group))
            item["days"].add(day.isoformat())
    for item in out.values():
        total = item["premium"]
        item["score"] = _clip(100.0 * item["score_num"] / total) if total > 0 else 0.0
        item["days_observed"] = len(item["days"])
        item["days"] = sorted(item["days"])
    return dict(out)


def aggregate_dark_pool(
    days: Sequence[Tuple[dt.date, Path]],
    as_of: dt.date,
    allowed_tickers: Optional[Iterable[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    allowed = _allowed_filter(allowed_tickers)
    out: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"premium": 0.0, "rows": 0, "days": set()})
    for day, day_dir in days:
        path = find_csv(day_dir, ["dp-eod-report-*.csv"])
        if path is None:
            continue
        weight = recency_weight(day, as_of)
        try:
            df = pd.read_csv(
                path,
                usecols=lambda c: c in {"ticker", "premium", "volume", "price"},
                low_memory=False,
            )
        except Exception:
            continue
        if df.empty or "ticker" not in df.columns:
            continue
        ticker_col = df["ticker"].fillna("").astype(str).str.upper().str.replace(r"[^A-Z0-9.]", "", regex=True)
        df = df.assign(_ticker=ticker_col)
        df = df[df["_ticker"].ne("")]
        if allowed is not None:
            df = df[df["_ticker"].isin(allowed)]
        if df.empty:
            continue
        premium = pd.to_numeric(df.get("premium", pd.Series(0, index=df.index)), errors="coerce").abs()
        fallback = (
            pd.to_numeric(df.get("volume", pd.Series(0, index=df.index)), errors="coerce").clip(lower=0)
            * pd.to_numeric(df.get("price", pd.Series(0, index=df.index)), errors="coerce").clip(lower=0)
        )
        df = df.assign(_premium=premium.where(premium.gt(0), fallback).fillna(0.0))
        grouped = df[df["_premium"].gt(0)].groupby("_ticker", dropna=False)
        for ticker, group in grouped:
            item = out[str(ticker)]
            item["premium"] += float(group["_premium"].sum() * weight)
            item["rows"] += int(len(group))
            item["days"].add(day.isoformat())
    for item in out.values():
        item["days_observed"] = len(item["days"])
        item["days"] = sorted(item["days"])
    return dict(out)


def row_text(row: Dict[str, Any]) -> str:
    preferred = [
        "text",
        "body",
        "title",
        "headline",
        "summary",
        "description",
        "content",
        "author_handle",
        "author_name",
        "ticker",
        "symbol",
        "url",
    ]
    parts: List[str] = []
    for col in preferred:
        value = row.get(col)
        if value:
            parts.append(str(value))
    if not parts:
        parts = [str(v) for v in row.values() if v]
    return " ".join(parts)


def load_text_documents(
    days: Sequence[Tuple[dt.date, Path]],
    *,
    max_docs: int,
) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    csv_patterns = [
        "*news-feed*.csv",
        "schwab-news*.csv",
        "*analysts*.csv",
        "*sec-filings*.csv",
        "*institutions*.csv",
        "*insiders*.csv",
        "browser-text-capture-*.csv",
        "browser_text_capture-*.csv",
        "reddit*.csv",
        "twitter*.csv",
    ]
    for day, day_dir in reversed(days):
        csv_paths: List[Path] = []
        for pattern in csv_patterns:
            csv_paths.extend(day_dir.glob(pattern))
            csv_paths.extend(day_dir.glob(f"**/{pattern}"))
        csv_paths = [p for p in dict.fromkeys(csv_paths) if p.is_file()]
        for path in csv_paths:
            for row in read_csv_rows(path, limit=400):
                stype = row_source_type(path, row)
                text = row_text(row).strip()
                if len(text) < 20:
                    continue
                if is_auth_wall_text(stype, text):
                    continue
                explicit = _normalize_ticker(
                    row.get("ticker")
                    or row.get("symbol")
                    or row.get("underlying")
                    or row.get("underlying_symbol")
                    or ""
                )
                docs.append(
                    {
                        "date": day.isoformat(),
                        "source_type": stype,
                        "path": str(path),
                        "ticker": explicit,
                        "mentions": extract_tickers(text),
                        "text": text[:8000],
                    }
                )
                if len(docs) >= max_docs:
                    return docs
        txt_paths: List[Path] = []
        txt_paths.extend(day_dir.glob("_snapshots/**/*.txt"))
        txt_paths.extend(day_dir.glob("browser-text-capture-*.txt"))
        txt_paths.extend(day_dir.glob("deep_research_packet_*.md"))
        for path in [p for p in dict.fromkeys(txt_paths) if p.is_file()]:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            if len(text.strip()) < 40:
                continue
            docs.append(
                {
                    "date": day.isoformat(),
                    "source_type": source_type(path),
                    "path": str(path),
                    "ticker": "",
                    "mentions": extract_tickers(text[:50000]),
                    "text": text[:50000],
                }
            )
            if len(docs) >= max_docs:
                return docs
    return docs


def fetch_schwab_news_documents(
    symbols: Sequence[str],
    *,
    root: Path,
    as_of: dt.date,
    out_dir: Path,
    suffix: str,
    enabled: bool,
    limit: int,
    max_tickers: int,
    manual_auth: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    summary: Dict[str, Any] = {
        "enabled": bool(enabled),
        "requested_tickers": 0,
        "items": 0,
        "status": "skipped",
        "csv": "",
        "raw": "",
        "error": "",
    }
    if not enabled:
        return [], summary
    tickers = [_normalize_ticker(t) for t in symbols if _normalize_ticker(t)]
    tickers = [t for t in dict.fromkeys(tickers) if t and t not in COMMON_TICKER_WORDS]
    tickers = tickers[: max(0, int(max_tickers))]
    summary["requested_tickers"] = len(tickers)
    if not tickers:
        summary["status"] = "no_tickers"
        return [], summary
    try:
        from uwos.schwab_auth import SchwabAuthConfig, SchwabLiveDataService
    except Exception as exc:
        summary["status"] = "import_failed"
        summary["error"] = str(exc)
        return [], summary
    try:
        cfg = SchwabAuthConfig.from_env(load_dotenv_file=True)
        svc = SchwabLiveDataService(config=cfg, manual_auth=bool(manual_auth), interactive_login=False)
        response = svc.get_news(tickers, limit=max(1, int(limit)))
    except Exception as exc:
        summary["status"] = "fetch_failed"
        summary["error"] = str(exc)
        return [], summary

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"schwab-news-{suffix}.csv"
    raw_path = out_dir / f"schwab-news-raw-{suffix}.json"
    raw_path.write_text(json.dumps(response, indent=2, default=str) + "\n", encoding="utf-8")

    rows: List[Dict[str, Any]] = []
    docs: List[Dict[str, Any]] = []
    endpoint = str(response.get("endpoint", ""))
    status = str(response.get("status", ""))
    for item in response.get("items", []) or []:
        headline = str(item.get("headline", "") or "")
        summary_text = str(item.get("summary", "") or "")
        related = item.get("symbols") or []
        if isinstance(related, str):
            related_symbols = [_normalize_ticker(related)]
        else:
            related_symbols = [_normalize_ticker(x) for x in related]
        related_symbols = [s for s in related_symbols if s] or tickers
        text = " ".join([headline, summary_text, str(item.get("source", "") or ""), str(item.get("url", "") or "")]).strip()
        for ticker in related_symbols:
            rows.append(
                {
                    "ticker": ticker,
                    "published_at": str(item.get("published_at", "") or ""),
                    "source": str(item.get("source", "Schwab") or "Schwab"),
                    "headline": headline,
                    "summary": summary_text,
                    "url": str(item.get("url", "") or ""),
                    "related_symbols": ",".join(related_symbols),
                    "schwab_endpoint": endpoint,
                    "schwab_status": status,
                }
            )
        docs.append(
            {
                "date": as_of.isoformat(),
                "source_type": "schwab_news",
                "path": str(csv_path),
                "ticker": ",".join(related_symbols),
                "mentions": related_symbols,
                "text": text[:8000],
            }
        )

    columns = [
        "ticker",
        "published_at",
        "source",
        "headline",
        "summary",
        "url",
        "related_symbols",
        "schwab_endpoint",
        "schwab_status",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})
    summary.update(
        {
            "status": status,
            "items": len(docs),
            "csv": str(csv_path),
            "raw": str(raw_path),
            "endpoint": endpoint,
            "errors": response.get("errors", []),
        }
    )
    return docs, summary


def text_sentiment_score(text: str) -> Tuple[float, List[str]]:
    lower = f" {str(text or '').lower()} "
    raw = 0.0
    hits: List[str] = []
    for phrase, weight in BULLISH_PHRASES.items():
        count = lower.count(phrase)
        if count:
            raw += weight * count
            hits.append(f"+{phrase}")
    for phrase, weight in BEARISH_PHRASES.items():
        count = lower.count(phrase)
        if count:
            raw -= weight * count
            hits.append(f"-{phrase}")
    if not hits:
        bullish_words = len(re.findall(r"\b(bullish|rally|surge|strong|buy|upgrade|breakout|accumulate)\b", lower))
        bearish_words = len(re.findall(r"\b(bearish|selloff|weak|downgrade|breakdown|miss|lawsuit|risk)\b", lower))
        raw += bullish_words * 0.8
        raw -= bearish_words * 0.8
        if bullish_words:
            hits.append(f"+{bullish_words} bullish words")
        if bearish_words:
            hits.append(f"-{bearish_words} bearish words")
    if raw == 0:
        return 0.0, hits
    return _clip(math.tanh(raw / 7.0) * 100.0), hits[:6]


def snippet_for(text: str, ticker: str, query_terms: Sequence[str]) -> str:
    target_patterns = [re.escape(ticker)]
    target_patterns.extend(re.escape(t) for t in query_terms[:4])
    lower = text.lower()
    pos = -1
    for term in target_patterns:
        m = re.search(term.lower(), lower)
        if m:
            pos = m.start()
            break
    if pos < 0:
        pos = 0
    start = max(0, pos - 120)
    end = min(len(text), pos + 260)
    return re.sub(r"\s+", " ", text[start:end]).strip()


def _contains_any(lower_text: str, terms: Iterable[str]) -> bool:
    return any(term in lower_text for term in terms)


def contextual_text_relevance(
    ticker: str,
    sector: str,
    lower_text: str,
    query_terms: Sequence[str],
    query_overlap: int,
) -> float:
    if not query_terms or query_overlap < max(1, min(2, len(query_terms))):
        return 0.0

    ticker = _normalize_ticker(ticker)
    sector_lower = str(sector or "").lower()
    middle_east_rule = next((rule for rule in THEME_RULES if rule["name"] == "middle_east_war_energy_defense"), {})
    bullish = set(middle_east_rule.get("bullish_tickers", set()))
    bearish = set(middle_east_rule.get("bearish_tickers", set()))

    if (ticker in bullish or sector_lower == "energy") and _contains_any(lower_text, ENERGY_CONTEXT_TERMS):
        return 0.16
    if (
        ticker in bullish
        or sector_lower == "industrials"
    ) and _contains_any(lower_text, DEFENSE_CONTEXT_TERMS):
        return 0.16
    if (ticker in bearish or sector_lower == "consumer cyclical") and _contains_any(lower_text, TRAVEL_CONTEXT_TERMS):
        return 0.16
    if ticker in {"SPY", "QQQ", "IWM", "VIX", "UVXY"} and _contains_any(lower_text, MARKET_CONTEXT_TERMS):
        return 0.12
    return 0.0


def score_text_for_ticker(
    docs: Sequence[Dict[str, Any]],
    ticker: str,
    query_terms: Sequence[str],
    sector: str = "",
) -> Dict[str, Any]:
    weighted = 0.0
    total_weight = 0.0
    matched = 0
    explicit_matched = 0
    context_matched = 0
    snippets: List[Tuple[float, str]] = []
    ticker = _normalize_ticker(ticker)
    for doc in docs:
        text = str(doc.get("text", "") or "")
        mentions = set(_normalize_ticker(x) for x in doc.get("mentions", []) or [])
        explicit = _normalize_ticker(doc.get("ticker", ""))
        lower = text.lower()
        query_overlap = _query_overlap_count(lower, query_terms)
        explicit_hit = explicit == ticker or ticker in mentions or f"${ticker.lower()}" in lower
        direct_ticker_query = len(query_terms) == 1 and _normalize_ticker(query_terms[0]) == ticker
        if query_terms and query_overlap <= 0 and not (direct_ticker_query and explicit_hit):
            continue
        relevance = 0.0
        if explicit_hit:
            relevance = 1.0
        else:
            relevance = contextual_text_relevance(ticker, sector, lower, query_terms, query_overlap)
        if relevance <= 0:
            continue
        score, hits = text_sentiment_score(text)
        stype = str(doc.get("source_type", "snapshot"))
        weight = SOURCE_WEIGHTS.get(stype, 0.65) * relevance
        weighted += score * weight
        total_weight += weight
        matched += 1
        if explicit_hit:
            explicit_matched += 1
        else:
            context_matched += 1
        if explicit_hit and (hits or abs(score) >= 20):
            snippets.append(
                (
                    abs(score) * weight,
                    f"{doc.get('date', '')} {stype}: {_fmt_num(score, 0)} from {', '.join(hits[:3]) or 'context'}; "
                    + snippet_for(text, ticker, query_terms),
                )
            )
    score = weighted / total_weight if total_weight > 0 else 0.0
    snippets.sort(key=lambda x: x[0], reverse=True)
    return {
        "score": _clip(score),
        "documents": matched,
        "explicit_documents": explicit_matched,
        "context_documents": context_matched,
        "evidence": [s for _, s in snippets[:3]],
    }


def stock_flow_score(row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not row:
        return {"score": 0.0, "drivers": [], "premium": 0.0}
    bull = _safe_float(row.get("bullish_premium"))
    bear = _safe_float(row.get("bearish_premium"))
    call = _safe_float(row.get("call_premium"))
    put = _safe_float(row.get("put_premium"))
    drivers: List[str] = []
    parts: List[Tuple[float, float]] = []
    if math.isfinite(bull) and math.isfinite(bear) and bull + bear > 0:
        score = 100.0 * (bull - bear) / (bull + bear)
        parts.append((score, bull + bear))
        drivers.append(f"UW premium bull/bear {_fmt_money(bull)}/{_fmt_money(bear)}")
    elif math.isfinite(call) and math.isfinite(put) and call + put > 0:
        score = 100.0 * (call - put) / (call + put)
        parts.append((score, call + put))
        drivers.append(f"call/put premium {_fmt_money(call)}/{_fmt_money(put)}")
    pcr = _safe_float(row.get("put_call_ratio"))
    if math.isfinite(pcr) and pcr > 0:
        pcr_score = _clip((1.0 - pcr) * 55.0, -35.0, 35.0)
        parts.append((pcr_score, 1.0))
        drivers.append(f"PCR {_fmt_num(pcr, 2)}")
    if not parts:
        return {"score": 0.0, "drivers": drivers, "premium": 0.0}
    weighted = sum(score * max(1.0, weight) for score, weight in parts)
    total = sum(max(1.0, weight) for _, weight in parts)
    return {"score": _clip(weighted / total), "drivers": drivers, "premium": max(0.0, bull) + max(0.0, bear)}


def price_momentum_score(row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not row:
        return {"score": 0.0, "drivers": []}
    close = _safe_float(row.get("close"))
    prev = _safe_float(row.get("prev_close"))
    high52 = _safe_float(row.get("week_52_high"))
    low52 = _safe_float(row.get("week_52_low"))
    parts: List[float] = []
    drivers: List[str] = []
    if math.isfinite(close) and math.isfinite(prev) and prev > 0:
        ret = (close - prev) / prev
        parts.append(_clip(ret * 1200.0, -45.0, 45.0))
        drivers.append(f"1d {ret:+.1%}")
    if math.isfinite(close) and math.isfinite(high52) and math.isfinite(low52) and high52 > low52:
        loc = (close - low52) / (high52 - low52)
        parts.append(_clip((loc - 0.50) * 70.0, -35.0, 35.0))
        drivers.append(f"52w position {loc:.0%}")
    return {"score": _clip(sum(parts) / len(parts)) if parts else 0.0, "drivers": drivers}


def theme_score(ticker: str, sector: str, query: str) -> Dict[str, Any]:
    query_lower = str(query or "").lower()
    ticker = _normalize_ticker(ticker)
    sector_norm = str(sector or "").strip()
    score = 0.0
    notes: List[str] = []
    for rule in THEME_RULES:
        keywords = rule.get("keywords", set())
        matched = [kw for kw in keywords if str(kw).lower() in query_lower]
        if not matched:
            continue
        if ticker in rule.get("bullish_tickers", set()):
            score += 85.0
            notes.append(f"{rule['name']} maps {ticker} bullish")
        if ticker in rule.get("bearish_tickers", set()):
            score -= 85.0
            notes.append(f"{rule['name']} maps {ticker} bearish")
        if sector_norm in rule.get("bullish_sectors", set()):
            score += 35.0
            notes.append(f"{rule['name']} favors {sector_norm}")
        if sector_norm in rule.get("bearish_sectors", set()):
            score -= 25.0
            notes.append(f"{rule['name']} pressures {sector_norm}")
    return {"score": _clip(score), "notes": notes}


def matched_theme_rules(query: str) -> List[Dict[str, Any]]:
    query_lower = str(query or "").lower()
    matches: List[Dict[str, Any]] = []
    for rule in THEME_RULES:
        if any(str(kw).lower() in query_lower for kw in rule.get("keywords", set())):
            matches.append(rule)
    return matches


def compute_regime(root: Path, days: Sequence[Tuple[dt.date, Path]]) -> Dict[str, Any]:
    try:
        return trend_analysis.compute_market_regime(root, list(days))
    except Exception as exc:
        return {"regime": "unknown", "reason": f"regime computation failed: {exc}"}


def macro_score_for(row: Optional[Dict[str, Any]], regime: Dict[str, Any]) -> Dict[str, Any]:
    label = str(regime.get("regime", "unknown") or "unknown")
    ticker = _normalize_ticker(row.get("ticker") if row else "")
    sector = str(row.get("sector", "") if row else "")
    score = 0.0
    if label == "risk_on":
        score = 12.0
    elif label == "risk_off":
        score = -18.0
    elif label == "mixed":
        score = 0.0
    if label == "risk_off" and (ticker in {"GLD", "GDX", "VIX", "UVXY"} or sector == "Utilities"):
        score += 15.0
    return {"score": _clip(score), "label": label, "reason": str(regime.get("reason", "") or "")}


def _weighted_average(parts: Sequence[Tuple[float, float]]) -> float:
    total = sum(weight for _, weight in parts if weight > 0)
    if total <= 0:
        return 0.0
    return _clip(sum(score * weight for score, weight in parts if weight > 0) / total)


def evidence_weight_from_premium(premium: Any, *, floor: float = 0.20) -> float:
    value = _safe_float(premium)
    if not math.isfinite(value) or value <= 0:
        return floor
    if value >= 25_000_000:
        return 1.0
    if value >= 5_000_000:
        return 0.85
    if value >= 1_000_000:
        return 0.65
    if value >= 250_000:
        return 0.45
    if value >= 50_000:
        return 0.30
    return floor


def _direction(score: float) -> str:
    if score >= 20:
        return "bullish"
    if score <= -20:
        return "bearish"
    return "mixed"


def _agreement(final_score: float, components: Sequence[Tuple[str, float]]) -> Tuple[int, int]:
    sign = 1 if final_score > 0 else -1 if final_score < 0 else 0
    if sign == 0:
        return 0, 0
    agree = 0
    disagree = 0
    for _, value in components:
        if abs(value) < 8:
            continue
        if value * sign > 0:
            agree += 1
        else:
            disagree += 1
    return agree, disagree


def load_trade_artifacts(root: Path, as_of: dt.date, *, include_patterns: bool = False) -> Dict[str, List[Dict[str, Any]]]:
    trade_dir = root / "out" / "trend_analysis"
    if not trade_dir.exists():
        return {}
    kinds = [
        ("ACTIONABLE", f"trend-analysis-actionable-{as_of.isoformat()}-L*.csv"),
        ("MAX_CONVICTION", f"trend-analysis-max-conviction-{as_of.isoformat()}-L*.csv"),
        ("TRADE_WORKUP", f"trend-analysis-trade-workups-{as_of.isoformat()}-L*.csv"),
        ("CURRENT_SETUP", f"trend-analysis-current-setups-{as_of.isoformat()}-L*.csv"),
    ]
    if include_patterns:
        kinds.append(("PATTERN_ONLY", f"trend-analysis-patterns-{as_of.isoformat()}-L*.csv"))
    priority = {"MAX_CONVICTION": 0, "ACTIONABLE": 1, "TRADE_WORKUP": 2, "CURRENT_SETUP": 3, "PATTERN_ONLY": 4}
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    seen = set()
    for status, pattern in kinds:
        for path in sorted(trade_dir.glob(pattern)):
            for row in read_csv_rows(path):
                ticker = _normalize_ticker(row.get("ticker"))
                if not ticker:
                    continue
                key = (
                    ticker,
                    status,
                    str(row.get("strategy", "")),
                    str(row.get("target_expiry", "")),
                    str(row.get("strike_setup", "") or row.get("live_strike_setup", "")),
                )
                if key in seen:
                    continue
                seen.add(key)
                item = dict(row)
                item["_trade_status"] = status
                item["_trade_source"] = str(path)
                out[ticker].append(item)
    for ticker, rows in out.items():
        rows.sort(key=lambda r: (priority.get(str(r.get("_trade_status")), 99), -_safe_float(r.get("swing_score"))))
    return dict(out)


def _proof_key(ticker: Any, direction: Any, strategy: Any) -> Tuple[str, str, str]:
    return (
        _normalize_ticker(ticker),
        str(direction or "").strip().lower(),
        re.sub(r"\s+", " ", str(strategy or "").strip().lower()),
    )


def _parse_batch_proof_file(path: Path) -> Dict[str, Any]:
    match = BATCH_PROOF_RE.match(path.name)
    if not match:
        return {}
    try:
        start = dt.date.fromisoformat(match.group("start"))
        end = dt.date.fromisoformat(match.group("end"))
        lookback = int(match.group("lookback"))
    except Exception:
        return {}
    return {
        "kind": match.group("kind"),
        "start": start,
        "end": end,
        "lookback": lookback,
        "ext": match.group("ext"),
    }


def _select_batch_proof_file(batch_dir: Path, kind: str, ext: str, as_of: dt.date, lookback: int) -> Optional[Path]:
    candidates = [p for p in batch_dir.glob(f"trend-analysis-batch-{kind}-*-L*.{ext}") if p.is_file()]
    if not candidates:
        return None
    ranked = []
    for path in candidates:
        parsed = _parse_batch_proof_file(path)
        if parsed and parsed.get("end") and parsed["end"] > as_of:
            continue
        same_lookback = bool(parsed and parsed.get("lookback") == lookback)
        end_date = parsed.get("end") if parsed else dt.date.min
        ranked.append((same_lookback, end_date, int(path.stat().st_mtime_ns), str(path), path))
    if not ranked:
        return None
    ranked.sort(reverse=True)
    return ranked[0][4]


def _sibling_batch_proof_file(batch_dir: Path, selected: Dict[str, Any], kind: str, ext: str) -> Optional[Path]:
    if not selected:
        return None
    start = selected.get("start")
    end = selected.get("end")
    lookback = selected.get("lookback")
    if not isinstance(start, dt.date) or not isinstance(end, dt.date) or not lookback:
        return None
    path = batch_dir / f"trend-analysis-batch-{kind}-{start.isoformat()}_{end.isoformat()}-L{lookback}.{ext}"
    return path if path.exists() and path.is_file() else None


def _proof_playbook_label(row: Dict[str, Any]) -> str:
    ticker = _normalize_ticker(row.get("ticker"))
    direction = str(row.get("direction") or "").strip().lower()
    strategy = str(row.get("strategy") or "").strip()
    tests = row.get("forward_tests") or row.get("overall_outcomes") or row.get("trades") or ""
    hit = row.get("forward_hit_rate") or row.get("overall_hit_rate") or row.get("win_rate") or ""
    avg = row.get("forward_avg_pnl") or row.get("overall_avg_pnl") or row.get("avg_pnl") or ""
    parts = [p for p in [ticker, direction, strategy] if p]
    label = " ".join(parts)
    metrics = []
    if str(tests).strip():
        metrics.append(f"tests {tests}")
    if str(hit).strip():
        metrics.append(f"hit {_fmt_percent(hit)}")
    if str(avg).strip():
        metrics.append(f"avg {_fmt_money(avg)}")
    if metrics:
        label += " (" + ", ".join(metrics) + ")"
    return label


def load_batch_proof_gate(
    root: Path,
    as_of: dt.date,
    lookback: int,
    *,
    enabled: bool = True,
    proof_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    gate: Dict[str, Any] = {
        "enabled": bool(enabled),
        "status": "disabled" if not enabled else "missing",
        "proof_dir": "",
        "rolling_audit": "",
        "ticker_audit": "",
        "strategy_audit": "",
        "metadata": "",
        "report": "",
        "verdict": "",
        "proof_start": "",
        "proof_end": "",
        "proof_lookback": "",
        "supported_playbooks": {},
        "blocked_playbooks": {},
        "ticker_playbooks": {},
        "supported_count": 0,
        "blocked_count": 0,
        "ticker_playbook_count": 0,
        "error": "",
    }
    if not enabled:
        return gate

    batch_dir = proof_dir or (root / "out" / "trend_analysis_batch")
    batch_dir = batch_dir.expanduser().resolve()
    gate["proof_dir"] = str(batch_dir)
    if not batch_dir.exists():
        gate["error"] = f"missing proof directory: {batch_dir}"
        return gate

    rolling_path = _select_batch_proof_file(batch_dir, "rolling-ticker-playbook-audit", "csv", as_of, lookback)
    metadata_path = _select_batch_proof_file(batch_dir, "metadata", "json", as_of, lookback)
    report_path = _select_batch_proof_file(batch_dir, "proof", "md", as_of, lookback)
    selected = _parse_batch_proof_file(rolling_path) if rolling_path else _parse_batch_proof_file(metadata_path) if metadata_path else {}
    if selected:
        sibling_metadata = _sibling_batch_proof_file(batch_dir, selected, "metadata", "json")
        sibling_report = _sibling_batch_proof_file(batch_dir, selected, "proof", "md")
        sibling_ticker = _sibling_batch_proof_file(batch_dir, selected, "ticker-playbook-audit", "csv")
        sibling_strategy = _sibling_batch_proof_file(batch_dir, selected, "strategy-family-audit", "csv")
        metadata_path = sibling_metadata or metadata_path
        report_path = sibling_report or report_path
        ticker_path = sibling_ticker
        strategy_path = sibling_strategy
        gate["proof_start"] = selected["start"].isoformat()
        gate["proof_end"] = selected["end"].isoformat()
        gate["proof_lookback"] = int(selected["lookback"])
    else:
        ticker_path = _select_batch_proof_file(batch_dir, "ticker-playbook-audit", "csv", as_of, lookback)
        strategy_path = _select_batch_proof_file(batch_dir, "strategy-family-audit", "csv", as_of, lookback)

    if rolling_path:
        gate["rolling_audit"] = str(rolling_path)
    if ticker_path:
        gate["ticker_audit"] = str(ticker_path)
    if strategy_path:
        gate["strategy_audit"] = str(strategy_path)
    if metadata_path:
        gate["metadata"] = str(metadata_path)
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            gate["verdict"] = str(metadata.get("verdict") or "")
        except Exception as exc:
            gate["error"] = f"metadata read failed: {exc}"
    if report_path:
        gate["report"] = str(report_path)

    if not rolling_path:
        gate["error"] = gate["error"] or "missing rolling ticker playbook audit"
        return gate

    try:
        for row in read_csv_rows(rolling_path):
            key = _proof_key(row.get("ticker"), row.get("direction"), row.get("strategy"))
            if not all(key):
                continue
            verdict = str(row.get("verdict") or "").strip().lower()
            if verdict == "supportive":
                gate["supported_playbooks"][key] = dict(row)
            else:
                gate["blocked_playbooks"][key] = dict(row)
    except Exception as exc:
        gate["status"] = "error"
        gate["error"] = f"rolling audit read failed: {exc}"
        return gate

    if ticker_path:
        for row in read_csv_rows(ticker_path):
            key = _proof_key(row.get("ticker"), row.get("direction"), row.get("strategy"))
            if all(key):
                gate["ticker_playbooks"][key] = dict(row)

    gate["supported_count"] = len(gate["supported_playbooks"])
    gate["blocked_count"] = len(gate["blocked_playbooks"])
    gate["ticker_playbook_count"] = len(gate["ticker_playbooks"])
    gate["status"] = "ok" if gate["supported_count"] else "ok_no_supported_playbooks"
    if selected and selected.get("lookback") != lookback:
        gate["status"] += "_lookback_mismatch"
    return gate


def batch_proof_gate_summary(proof_gate: Dict[str, Any]) -> Dict[str, Any]:
    supported = list((proof_gate.get("supported_playbooks") or {}).values())
    return {
        "enabled": bool(proof_gate.get("enabled")),
        "status": proof_gate.get("status", ""),
        "proof_dir": proof_gate.get("proof_dir", ""),
        "rolling_audit": proof_gate.get("rolling_audit", ""),
        "ticker_audit": proof_gate.get("ticker_audit", ""),
        "strategy_audit": proof_gate.get("strategy_audit", ""),
        "metadata": proof_gate.get("metadata", ""),
        "report": proof_gate.get("report", ""),
        "verdict": proof_gate.get("verdict", ""),
        "proof_start": proof_gate.get("proof_start", ""),
        "proof_end": proof_gate.get("proof_end", ""),
        "proof_lookback": proof_gate.get("proof_lookback", ""),
        "supported_count": int(proof_gate.get("supported_count") or 0),
        "blocked_count": int(proof_gate.get("blocked_count") or 0),
        "ticker_playbook_count": int(proof_gate.get("ticker_playbook_count") or 0),
        "supported_playbooks": [_proof_playbook_label(row) for row in supported[:20]],
        "error": proof_gate.get("error", ""),
    }


def _batch_block_reason(key: Tuple[str, str, str], proof_gate: Dict[str, Any]) -> str:
    blocked = (proof_gate.get("blocked_playbooks") or {}).get(key)
    if blocked:
        verdict = str(blocked.get("verdict") or "blocked").strip()
        tests = str(blocked.get("forward_tests") or "0").strip()
        avg = blocked.get("forward_avg_pnl")
        hit = blocked.get("forward_hit_rate")
        metrics = [f"rolling verdict {verdict}", f"tests {tests}"]
        if str(hit).strip():
            metrics.append(f"hit {_fmt_percent(hit)}")
        if str(avg).strip():
            metrics.append(f"avg {_fmt_money(avg)}")
        return ", ".join(metrics)
    ticker_playbook = (proof_gate.get("ticker_playbooks") or {}).get(key)
    if ticker_playbook:
        verdict = str(ticker_playbook.get("verdict") or "diagnostic").strip()
        return f"ticker playbook audit is {verdict}, but rolling-forward proof is not supportive"
    return "no exact supportive rolling ticker playbook in batch proof"


def trade_summary_for(
    ticker: str,
    direction: str,
    trade_map: Dict[str, List[Dict[str, Any]]],
    proof_gate: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    rows = trade_map.get(ticker, [])
    if not rows:
        return {
            "status": "NO_TREND_TRADE",
            "original_status": "NO_TREND_TRADE",
            "summary": "no trend-analysis trade artifact",
            "aligned": False,
            "proof_status": "RESEARCH_ONLY",
            "proof_summary": "no trend-analysis trade artifact",
            "proof_source": "",
        }
    preferred = rows
    aligned = [r for r in rows if str(r.get("direction", "") or "").strip().lower() == direction]
    if aligned:
        preferred = aligned
    row = preferred[0]
    row_direction = str(row.get("direction", "") or "").strip().lower()
    setup = str(row.get("live_strike_setup") or row.get("strike_setup") or "").strip()
    expiry = str(row.get("target_expiry", "") or "").strip()
    strategy = str(row.get("strategy", "") or "").strip()
    status = str(row.get("_trade_status", "TREND_ARTIFACT"))
    tier = str(row.get("position_size_tier") or row.get("setup_tier") or "").strip()
    reject = str(row.get("actionability_reject_reasons") or row.get("setup_reason") or "").strip()
    summary = f"{status}: {strategy}"
    if setup:
        summary += f" | {setup}"
    if expiry:
        summary += f" | exp {expiry}"
    if tier:
        summary += f" | {tier}"
    if reject and status not in {"ACTIONABLE", "MAX_CONVICTION"}:
        summary += f" | {reject[:180]}"
    proof_status = "PROOF_NOT_CHECKED"
    proof_summary = "batch proof gate disabled"
    proof_source = ""
    gated_status = status
    gated_summary = summary
    if proof_gate and proof_gate.get("enabled"):
        proof_source = str(proof_gate.get("rolling_audit") or proof_gate.get("report") or "")
        if status == "PATTERN_ONLY":
            proof_status = "RESEARCH_ONLY"
            gated_status = "RESEARCH_ONLY"
            proof_summary = "pattern-only trend artifact; not a current/live trade gate"
            gated_summary = f"RESEARCH_ONLY: {summary} | {proof_summary}"
        elif not proof_gate.get("rolling_audit"):
            proof_status = "BATCH_PROOF_MISSING"
            gated_status = "BATCH_BLOCKED"
            proof_summary = "missing rolling batch proof; trend artifact is context only"
            gated_summary = f"BATCH_BLOCKED: {summary} | {proof_summary}"
        else:
            key = _proof_key(ticker, row_direction or direction, strategy)
            supported = (proof_gate.get("supported_playbooks") or {}).get(key)
            if supported:
                proof_status = "PROOF_SUPPORTED"
                proof_summary = "supportive rolling playbook: " + _proof_playbook_label(supported)
                gated_summary = f"{summary} | proof supported"
            else:
                proof_status = "BATCH_BLOCKED"
                gated_status = "BATCH_BLOCKED"
                proof_summary = _batch_block_reason(key, proof_gate)
                gated_summary = f"BATCH_BLOCKED: {summary} | {proof_summary}"
    return {
        "status": gated_status,
        "original_status": status,
        "summary": gated_summary,
        "aligned": bool(aligned),
        "source": str(row.get("_trade_source", "")),
        "strategy": strategy,
        "setup": setup,
        "expiry": expiry,
        "row_direction": row_direction,
        "proof_status": proof_status,
        "proof_summary": proof_summary,
        "proof_source": proof_source,
    }


def run_trend_analysis_refresh(
    args: argparse.Namespace,
    *,
    root: Path,
    as_of: dt.date,
    lookback: int,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "enabled": bool(args.run_trend_analysis),
        "status": "skipped",
        "as_of": as_of.isoformat(),
        "lookback": int(args.trend_lookback or lookback),
        "report": "",
        "actionable_csv": "",
        "trade_workup_csv": "",
        "current_setups_csv": "",
        "metadata": "",
        "error": "",
    }
    if not bool(args.run_trend_analysis):
        return summary

    trend_lookback = max(1, int(args.trend_lookback or lookback))
    trend_top = max(5, int(args.trend_top or args.top or 5))
    trend_candidate_top = max(12, int(args.trend_candidate_top or args.top or 12))
    trend_args = [
        as_of.isoformat(),
        str(trend_lookback),
        "--root-dir",
        str(root),
        "--top",
        str(trend_top),
        "--candidate-top",
        str(trend_candidate_top),
    ]
    if bool(args.trend_reuse_audits):
        trend_args.extend(["--reuse-walk-forward-raw", "--reuse-walk-forward-outcomes", "--reuse-research-outcomes"])
    if bool(args.trend_no_schwab):
        trend_args.append("--no-schwab")

    try:
        result = trend_analysis.run(trend_args)
        summary["status"] = "ok"
    except Exception as exc:
        if not bool(args.trend_no_schwab) and "schwab" in str(exc).lower():
            retry_args = list(trend_args) + ["--no-schwab"]
            summary["retried_without_schwab"] = True
            try:
                result = trend_analysis.run(retry_args)
                summary["status"] = "ok_no_schwab_retry"
            except Exception as retry_exc:
                summary["status"] = "error"
                summary["error"] = str(retry_exc)
                return summary
        else:
            summary["status"] = "error"
            summary["error"] = str(exc)
            return summary

    summary.update(
        {
            "lookback": trend_lookback,
            "report": str(result.get("report", "")),
            "actionable_csv": str(result.get("actionable_csv", "")),
            "trade_workup_csv": str(result.get("trade_workup_csv", "")),
            "current_setups_csv": str(result.get("current_setups_csv", "")),
            "metadata": str(result.get("metadata", "")),
        }
    )
    return summary


def query_is_ticker(query: str, tickers: Iterable[str]) -> bool:
    clean = _normalize_ticker(query)
    if " " in query.strip():
        return False
    return bool(clean and clean in set(tickers))


def requested_tickers(args: argparse.Namespace) -> List[str]:
    tickers = [_normalize_ticker(t) for t in args.ticker]
    tickers.extend(_normalize_ticker(t) for t in str(args.tickers or "").split(","))
    query = _query_text(args)
    if query and re.fullmatch(r"\$?[A-Za-z][A-Za-z0-9.]{0,7}", query.strip()):
        tickers.append(_normalize_ticker(query))
    return sorted({t for t in tickers if t})


def top_flow_tickers(
    stock_rows: Dict[str, Dict[str, Any]],
    hot: Dict[str, Dict[str, Any]],
    oi: Dict[str, Dict[str, Any]],
    limit: int,
) -> List[str]:
    ranks: List[Tuple[float, str]] = []
    for ticker, row in stock_rows.items():
        flow = abs(stock_flow_score(row)["score"])
        opt = max(abs(_safe_float(hot.get(ticker, {}).get("score"))), abs(_safe_float(oi.get(ticker, {}).get("score"))))
        prem = math.log10(max(1.0, _safe_float(hot.get(ticker, {}).get("premium")) + _safe_float(oi.get(ticker, {}).get("premium"))))
        ranks.append((flow + opt + prem, ticker))
    ranks.sort(reverse=True)
    return [ticker for _, ticker in ranks[:limit]]


def build_universe(
    args: argparse.Namespace,
    query: str,
    query_terms: Sequence[str],
    stock_rows: Dict[str, Dict[str, Any]],
    docs: Sequence[Dict[str, Any]],
    hot: Dict[str, Dict[str, Any]],
    oi: Dict[str, Dict[str, Any]],
) -> List[str]:
    explicit = requested_tickers(args)
    if explicit:
        return explicit
    if query_is_ticker(query, stock_rows.keys()):
        return [_normalize_ticker(query)]

    universe = set()
    query_lower = query.lower()
    for rule in THEME_RULES:
        if any(str(kw).lower() in query_lower for kw in rule.get("keywords", set())):
            universe.update(rule.get("bullish_tickers", set()))
            universe.update(rule.get("bearish_tickers", set()))

    for ticker, row in stock_rows.items():
        haystack = " ".join(
            [
                str(row.get("ticker", "")),
                str(row.get("sector", "")),
                str(row.get("full_name", "")),
                str(row.get("issue_type", "")),
            ]
        ).lower()
        if query_terms and any(_query_term_in_text(haystack, term) for term in query_terms):
            universe.add(ticker)

    for doc in docs:
        lower = str(doc.get("text", "") or "").lower()
        if query_terms and _query_overlap_count(lower, query_terms) >= max(1, min(2, len(query_terms))):
            for mention in doc.get("mentions", []) or []:
                ticker = _normalize_ticker(mention)
                if ticker in stock_rows:
                    universe.add(ticker)

    if not universe:
        universe.update(top_flow_tickers(stock_rows, hot, oi, max(50, int(args.top) * 3)))
    else:
        universe.update(top_flow_tickers(stock_rows, hot, oi, max(15, int(args.top))))
    filtered = [t for t in universe if t and t not in COMMON_TICKER_WORDS]
    return sorted(filtered)


def build_seed_universe(
    args: argparse.Namespace,
    query: str,
    query_terms: Sequence[str],
    stock_rows: Dict[str, Dict[str, Any]],
    docs: Sequence[Dict[str, Any]],
) -> List[str]:
    explicit = requested_tickers(args)
    if explicit:
        return explicit
    if query_is_ticker(query, stock_rows.keys()):
        return [_normalize_ticker(query)]

    universe = set()
    theme_rules = matched_theme_rules(query)
    for rule in theme_rules:
        universe.update(rule.get("bullish_tickers", set()))
        universe.update(rule.get("bearish_tickers", set()))
        bullish_sectors = set(rule.get("bullish_sectors", set()))
        for ticker, row in stock_rows.items():
            if str(row.get("sector", "") or "").strip() in bullish_sectors:
                universe.add(ticker)

    if not theme_rules:
        for ticker, row in stock_rows.items():
            haystack = " ".join(
                [
                    str(row.get("ticker", "")),
                    str(row.get("sector", "")),
                    str(row.get("full_name", "")),
                    str(row.get("issue_type", "")),
                ]
            ).lower()
            if query_terms and any(_query_term_in_text(haystack, term) for term in query_terms):
                universe.add(ticker)

    for doc in docs:
        lower = str(doc.get("text", "") or "").lower()
        if query_terms and _query_overlap_count(lower, query_terms) >= max(1, min(2, len(query_terms))):
            for mention in doc.get("mentions", []) or []:
                ticker = _normalize_ticker(mention)
                if ticker in stock_rows:
                    universe.add(ticker)

    return sorted(t for t in universe if t and t not in COMMON_TICKER_WORDS)


def score_ticker(
    ticker: str,
    *,
    stock_rows: Dict[str, Dict[str, Any]],
    hot: Dict[str, Dict[str, Any]],
    oi: Dict[str, Dict[str, Any]],
    dp: Dict[str, Dict[str, Any]],
    docs: Sequence[Dict[str, Any]],
    query: str,
    query_terms: Sequence[str],
    regime: Dict[str, Any],
    trade_map: Dict[str, List[Dict[str, Any]]],
    proof_gate: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    row = stock_rows.get(ticker)
    sector = str(row.get("sector", "") if row else "")
    stock_flow = stock_flow_score(row)
    price = price_momentum_score(row)
    hot_score = _safe_float(hot.get(ticker, {}).get("score"))
    oi_score = _safe_float(oi.get(ticker, {}).get("score"))
    stock_premium = _safe_float(stock_flow.get("premium"))
    hot_premium = _safe_float(hot.get(ticker, {}).get("premium"))
    oi_premium = _safe_float(oi.get(ticker, {}).get("premium"))
    option_premium = (hot_premium if math.isfinite(hot_premium) else 0.0) + (oi_premium if math.isfinite(oi_premium) else 0.0)
    options_score = _weighted_average(
        [
            (hot_score, max(1.0, math.log10(max(1.0, _safe_float(hot.get(ticker, {}).get("premium")))))),
            (oi_score, max(1.0, math.log10(max(1.0, _safe_float(oi.get(ticker, {}).get("premium")))))),
        ]
    )
    text = score_text_for_ticker(docs, ticker, query_terms, sector)
    theme = theme_score(ticker, sector, query)
    macro = macro_score_for(row or {"ticker": ticker, "sector": sector}, regime)
    dp_premium = _safe_float(dp.get(ticker, {}).get("premium"))
    dp_activity_bonus = min(8.0, math.log10(max(1.0, dp_premium))) if dp_premium > 0 else 0.0
    if options_score == 0.0:
        options_score = _weighted_average([(hot_score, 1.0), (oi_score, 1.0)])
    combined = _weighted_average(
        [
            (
                _safe_float(text["score"]),
                0.35 if text["explicit_documents"] else (0.10 if text["documents"] else 0.04),
            ),
            (_safe_float(stock_flow["score"]), 0.22 * evidence_weight_from_premium(stock_premium, floor=0.15)),
            (options_score, 0.18 * evidence_weight_from_premium(option_premium, floor=0.20)),
            (_safe_float(price["score"]), 0.10),
            (_safe_float(theme["score"]), 0.10 if theme["notes"] else 0.02),
            (_safe_float(macro["score"]), 0.05),
        ]
    )
    if dp_activity_bonus and abs(combined) >= 15:
        combined = _clip(combined + math.copysign(dp_activity_bonus, combined))

    components = [
        ("uw_stock_flow", _safe_float(stock_flow["score"])),
        ("options_flow", options_score),
        ("price", _safe_float(price["score"])),
        ("text", _safe_float(text["score"])),
        ("theme", _safe_float(theme["score"])),
        ("macro", _safe_float(macro["score"])),
    ]
    agree, disagree = _agreement(combined, components)
    coverage = sum(
        [
            bool(row),
            abs(_safe_float(stock_flow["score"])) > 1,
            hot.get(ticker, {}).get("rows", 0) > 0,
            oi.get(ticker, {}).get("rows", 0) > 0,
            dp.get(ticker, {}).get("rows", 0) > 0,
            text["documents"] > 0,
            bool(theme["notes"]),
        ]
    )
    confidence = _clip(coverage * 9.0 + abs(combined) * 0.45 + agree * 7.0 - disagree * 9.0, 0.0, 100.0)
    direction = _direction(combined)
    trade = trade_summary_for(ticker, direction, trade_map, proof_gate)
    drivers: List[str] = []
    drivers.extend(stock_flow["drivers"][:2])
    drivers.extend(price["drivers"][:2])
    drivers.extend(theme["notes"][:2])
    if text["evidence"]:
        drivers.append(text["evidence"][0])
    if dp_premium > 0:
        drivers.append(f"dark-pool activity {_fmt_money(dp_premium)}")
    return {
        "ticker": ticker,
        "direction": direction,
        "sentiment_score": round(combined, 1),
        "confidence": round(confidence, 1),
        "uw_stock_flow_score": round(_safe_float(stock_flow["score"]), 1),
        "options_flow_score": round(options_score, 1),
        "price_score": round(_safe_float(price["score"]), 1),
        "text_score": round(_safe_float(text["score"]), 1),
        "theme_score": round(_safe_float(theme["score"]), 1),
        "macro_score": round(_safe_float(macro["score"]), 1),
        "sector": sector,
        "latest_close": _safe_float(row.get("close")) if row else math.nan,
        "market_regime": str(macro["label"]),
        "source_coverage": coverage,
        "agreement_count": agree,
        "conflict_count": disagree,
        "text_documents": int(text["documents"]),
        "hot_chain_premium": _safe_float(hot.get(ticker, {}).get("premium")),
        "oi_change_premium": _safe_float(oi.get(ticker, {}).get("premium")),
        "dark_pool_premium": dp_premium,
        "trade_status": trade["status"],
        "trade_original_status": trade.get("original_status", trade["status"]),
        "trade_summary": trade["summary"],
        "trade_aligned": bool(trade["aligned"]),
        "trade_source": trade.get("source", ""),
        "proof_status": trade.get("proof_status", ""),
        "proof_summary": trade.get("proof_summary", ""),
        "proof_source": trade.get("proof_source", ""),
        "drivers": " | ".join(drivers[:5]),
        "evidence": text["evidence"],
    }


def rank_results(results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        results,
        key=lambda r: (abs(_safe_float(r.get("sentiment_score"))), _safe_float(r.get("confidence"))),
        reverse=True,
    )


def write_scores_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "ticker",
        "direction",
        "sentiment_score",
        "confidence",
        "sector",
        "latest_close",
        "uw_stock_flow_score",
        "options_flow_score",
        "price_score",
        "text_score",
        "theme_score",
        "macro_score",
        "market_regime",
        "source_coverage",
        "agreement_count",
        "conflict_count",
        "text_documents",
        "hot_chain_premium",
        "oi_change_premium",
        "dark_pool_premium",
        "trade_status",
        "trade_original_status",
        "trade_aligned",
        "trade_summary",
        "trade_source",
        "proof_status",
        "proof_summary",
        "proof_source",
        "drivers",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def render_table(rows: Sequence[Sequence[str]], headers: Sequence[str]) -> str:
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(str(cell)))
    header = "| " + " | ".join(str(h).ljust(widths[idx]) for idx, h in enumerate(headers)) + " |"
    sep = "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"
    body = ["| " + " | ".join(str(cell).ljust(widths[idx]) for idx, cell in enumerate(row)) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def write_report(
    path: Path,
    *,
    query: str,
    as_of: dt.date,
    lookback: int,
    results: Sequence[Dict[str, Any]],
    top: int,
    regime: Dict[str, Any],
    source_summary: Dict[str, Any],
    scores_csv: Path,
    metadata_path: Path,
) -> None:
    lines: List[str] = []
    title_query = query or "market"
    lines.append(f"# Sentiment Analysis - {title_query} - {as_of.isoformat()}")
    lines.append("")
    lines.append(
        "Research output only. Exact option trades are surfaced only when trend-analysis artifacts exist and the batch proof gate supports them."
    )
    lines.append("")
    lines.append("## Data Quality")
    lines.append(f"- Lookback: {lookback} usable UW folder(s)")
    lines.append(f"- Market regime: {regime.get('regime', 'unknown')} ({regime.get('reason', '')})")
    lines.append(f"- Stock universe rows: {source_summary.get('stock_rows', 0)}")
    lines.append(f"- Text/social/news documents scanned: {source_summary.get('text_docs', 0)}")
    source_counts = source_summary.get("text_source_counts", {})
    if source_counts:
        counts_text = ", ".join(f"{k}={v}" for k, v in sorted(source_counts.items()))
        lines.append(f"- Text source mix: {counts_text}")
    schwab_news = source_summary.get("schwab_news", {}) if isinstance(source_summary.get("schwab_news"), dict) else {}
    if schwab_news.get("enabled"):
        lines.append(
            "- Schwab news: "
            + f"{schwab_news.get('status', 'unknown')}; "
            + f"requested {schwab_news.get('requested_tickers', 0)} ticker(s), "
            + f"items {schwab_news.get('items', 0)}"
        )
        if schwab_news.get("error"):
            lines.append(f"- Schwab news error: {str(schwab_news.get('error'))[:220]}")
    trend_refresh = (
        source_summary.get("trend_refresh", {})
        if isinstance(source_summary.get("trend_refresh"), dict)
        else {}
    )
    if trend_refresh.get("enabled"):
        lines.append(
            "- Trend-analysis refresh: "
            + f"{trend_refresh.get('status', 'unknown')}; "
            + f"lookback {trend_refresh.get('lookback', '-')}"
        )
        if trend_refresh.get("error"):
            lines.append(f"- Trend-analysis refresh error: {str(trend_refresh.get('error'))[:220]}")
    lines.append(f"- Trend trade artifacts loaded: {source_summary.get('trade_artifacts', 0)}")
    batch_gate = (
        source_summary.get("batch_proof_gate", {})
        if isinstance(source_summary.get("batch_proof_gate"), dict)
        else {}
    )
    if batch_gate.get("enabled"):
        proof_desc = f"{batch_gate.get('status', 'unknown')}; supported playbooks {batch_gate.get('supported_count', 0)}"
        if batch_gate.get("proof_end"):
            proof_desc += f"; proof through {batch_gate.get('proof_end')} L{batch_gate.get('proof_lookback', '-')}"
        if batch_gate.get("verdict"):
            proof_desc += f"; verdict {batch_gate.get('verdict')}"
        lines.append(f"- Batch proof gate: {proof_desc}")
        if batch_gate.get("supported_playbooks"):
            lines.append("- Live-eligible playbooks: " + "; ".join(batch_gate.get("supported_playbooks", [])[:5]))
        if batch_gate.get("error"):
            lines.append(f"- Batch proof gate error: {str(batch_gate.get('error'))[:220]}")
    else:
        lines.append("- Batch proof gate: disabled")
    if not source_summary.get("text_docs"):
        lines.append("- Text layer gap: no recent X/Reddit/news/browser artifacts were found; score relies on UW and theme/macro only.")
    lines.append("")

    top_rows = list(results)[: max(1, int(top))]
    table_rows = []
    for row in top_rows:
        table_rows.append(
            [
                str(row["ticker"]),
                str(row["direction"]),
                _fmt_num(row["sentiment_score"], 1),
                _fmt_num(row["confidence"], 0),
                str(row.get("sector") or "-")[:22],
                _fmt_num(row["uw_stock_flow_score"], 0),
                _fmt_num(row["options_flow_score"], 0),
                _fmt_num(row["text_score"], 0),
                str(row.get("trade_status", ""))[:18],
            ]
        )
    lines.append("## Ranked Sentiment")
    lines.append(render_table(table_rows, ["Ticker", "Bias", "Score", "Conf", "Sector", "UW", "Opt", "Text", "Trade"]))
    lines.append("")

    bullish = [r for r in results if r["direction"] == "bullish"][: max(5, min(top, 12))]
    bearish = [r for r in results if r["direction"] == "bearish"][: max(5, min(top, 12))]
    lines.append("## Bullish Watchlist")
    if bullish:
        for row in bullish:
            lines.append(
                f"- **{row['ticker']}** score {_fmt_num(row['sentiment_score'], 1)}, confidence {_fmt_num(row['confidence'], 0)}: {row['drivers'] or 'no dominant driver'}"
            )
    else:
        lines.append("- No bullish rows cleared the directional threshold.")
    lines.append("")

    lines.append("## Bearish Watchlist")
    if bearish:
        for row in bearish:
            lines.append(
                f"- **{row['ticker']}** score {_fmt_num(row['sentiment_score'], 1)}, confidence {_fmt_num(row['confidence'], 0)}: {row['drivers'] or 'no dominant driver'}"
            )
    else:
        lines.append("- No bearish rows cleared the directional threshold.")
    lines.append("")

    if batch_gate.get("enabled"):
        trade_rows = [r for r in top_rows if r.get("proof_status") == "PROOF_SUPPORTED"]
    else:
        trade_rows = [r for r in top_rows if r.get("trade_status") not in {"NO_TREND_TRADE", ""}]
    lines.append("## Trade Considerations")
    if trade_rows:
        for row in trade_rows[:12]:
            prefix = "aligned" if row.get("trade_aligned") else "context only"
            lines.append(f"- **{row['ticker']}** ({prefix}): {row.get('trade_summary', '')}")
    else:
        if batch_gate.get("enabled"):
            lines.append("- No proof-supported trend-analysis trade artifacts matched the ranked sentiment rows.")
        else:
            lines.append("- No existing trend-analysis trade artifacts matched the ranked sentiment rows.")
    lines.append("")

    blocked_rows = [r for r in top_rows if r.get("proof_status") in {"BATCH_BLOCKED", "BATCH_PROOF_MISSING"}]
    if blocked_rows:
        lines.append("## Batch-Proof Blocked Trade Artifacts")
        for row in blocked_rows[:12]:
            lines.append(
                f"- **{row['ticker']}**: {row.get('trade_summary', '')} "
                f"(source status {row.get('trade_original_status', '-')})"
            )
        lines.append("")

    evidence_rows = [r for r in top_rows if r.get("evidence")]
    if evidence_rows:
        lines.append("## Text Evidence")
        for row in evidence_rows[:8]:
            lines.append(f"### {row['ticker']}")
            for item in row.get("evidence", [])[:3]:
                lines.append(f"- {item}")
        lines.append("")

    lines.append("## Files")
    lines.append(f"- Scores CSV: {scores_csv}")
    lines.append(f"- Metadata JSON: {metadata_path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(argv: Optional[Sequence[str]] = None) -> Dict[str, Path]:
    args = parse_args(argv)
    root = Path(args.root_dir).expanduser().resolve() if args.root_dir else paths.project_root()
    as_of = _parse_date(args.as_of) if args.as_of else latest_data_date(root)
    lookback = max(1, int(args.lookback))
    query = _query_text(args)
    query_terms = _query_terms(query)
    days = trading_days_for_lookback(root, as_of, lookback)
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (root / DEFAULT_OUT_SUBDIR)
    suffix = f"{as_of.isoformat()}-L{lookback}-{_slug(query or 'market')}"

    stock_path, stock_rows = load_latest_stock_rows(days)
    docs = load_text_documents(days, max_docs=max(0, int(args.max_docs)))
    regime = compute_regime(root, days)
    trend_refresh_summary = run_trend_analysis_refresh(args, root=root, as_of=as_of, lookback=lookback)
    trade_map = load_trade_artifacts(root, as_of, include_patterns=bool(args.include_pattern_trades))
    proof_dir = Path(args.batch_proof_dir).expanduser().resolve() if args.batch_proof_dir else None
    proof_gate = load_batch_proof_gate(
        root,
        as_of,
        lookback,
        enabled=bool(args.batch_proof_gate),
        proof_dir=proof_dir,
    )
    seed_universe = build_seed_universe(args, query, query_terms, stock_rows, docs)
    aggregate_filter = seed_universe or None
    hot = aggregate_hot_chains(days, as_of, allowed_tickers=aggregate_filter)
    oi = aggregate_oi_changes(days, as_of, allowed_tickers=aggregate_filter)
    dp = aggregate_dark_pool(days, as_of, allowed_tickers=aggregate_filter)
    universe = seed_universe or build_universe(args, query, query_terms, stock_rows, docs, hot, oi)
    schwab_news_docs, schwab_news_summary = fetch_schwab_news_documents(
        universe,
        root=root,
        as_of=as_of,
        out_dir=out_dir,
        suffix=suffix,
        enabled=bool(args.schwab_news),
        limit=int(args.schwab_news_limit),
        max_tickers=int(args.schwab_news_max_tickers),
        manual_auth=bool(args.manual_auth),
    )
    docs = list(docs) + schwab_news_docs

    results = []
    for ticker in universe:
        results.append(
            score_ticker(
                ticker,
                stock_rows=stock_rows,
                hot=hot,
                oi=oi,
                dp=dp,
                docs=docs,
                query=query,
                query_terms=query_terms,
                regime=regime,
                trade_map=trade_map,
                proof_gate=proof_gate,
            )
        )
    ranked = rank_results(results)

    report_path = out_dir / f"sentiment-analysis-{suffix}.md"
    scores_csv = out_dir / f"sentiment-analysis-scores-{suffix}.csv"
    metadata_path = out_dir / f"sentiment-analysis-metadata-{suffix}.json"
    write_scores_csv(scores_csv, ranked)
    source_summary = {
        "stock_screener": str(stock_path),
        "stock_rows": len(stock_rows),
        "hot_tickers": len(hot),
        "oi_tickers": len(oi),
        "dark_pool_tickers": len(dp),
        "text_docs": len(docs),
        "text_source_counts": dict(
            sorted(
                pd.Series([str(doc.get("source_type", "unknown")) for doc in docs], dtype="object")
                .value_counts()
                .to_dict()
                .items()
            )
        )
        if docs
        else {},
        "schwab_news": schwab_news_summary,
        "trend_refresh": trend_refresh_summary,
        "batch_proof_gate": batch_proof_gate_summary(proof_gate),
        "trade_artifacts": sum(len(v) for v in trade_map.values()),
        "universe": len(universe),
    }
    write_report(
        report_path,
        query=query or "market",
        as_of=as_of,
        lookback=lookback,
        results=ranked,
        top=int(args.top),
        regime=regime,
        source_summary=source_summary,
        scores_csv=scores_csv,
        metadata_path=metadata_path,
    )
    metadata = {
        "as_of": as_of.isoformat(),
        "query": query,
        "lookback": lookback,
        "root": str(root),
        "report": str(report_path),
        "scores_csv": str(scores_csv),
        "source_summary": source_summary,
        "market_regime": regime,
        "top": ranked[: min(20, len(ranked))],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Wrote: {report_path}")
    print(f"Wrote: {scores_csv}")
    print(f"Wrote: {metadata_path}")
    return {"report": report_path, "scores_csv": scores_csv, "metadata": metadata_path}


def main() -> int:
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
