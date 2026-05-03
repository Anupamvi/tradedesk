from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd


POSITIVE_WORDS = {
    "beat",
    "beats",
    "bullish",
    "climb",
    "higher",
    "rally",
    "risk-on",
    "strength",
    "strong",
    "upgrade",
}
NEGATIVE_WORDS = {
    "blockade",
    "crash",
    "drawdown",
    "inflation",
    "lower",
    "miss",
    "recession",
    "risk",
    "sell",
    "shock",
    "surges",
    "war",
    "weakening",
}
MACRO_RISK_WORDS = {"fed", "iran", "oil", "cpi", "vix", "rates", "inflation", "war"}


def _read_browser_texts(base_dir: Path) -> list[tuple[str, str]]:
    browser_dir = base_dir / "browser_text"
    if not browser_dir.is_dir():
        return []
    rows: list[tuple[str, str]] = []
    for path in sorted(browser_dir.glob("browser-text-capture-*")):
        if path.suffix.lower() not in {".txt", ".csv"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if text.strip():
            rows.append((path.name, text))
    return rows


def _mentions(text: str, ticker: str) -> bool:
    if ticker in {"SPY", "QQQ", "IWM"}:
        if re.search(r"\b(SPY|QQQ|IWM|SPX|S&P|NASDAQ|VIX)\b", text, re.IGNORECASE):
            return True
    return bool(re.search(rf"(?<![A-Z0-9]){re.escape(ticker)}(?![A-Z0-9])", text, re.IGNORECASE))


def _snippet(text: str, ticker: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines:
        if _mentions(line, ticker) or any(word in line.lower() for word in MACRO_RISK_WORDS):
            return line[:180]
    return ""


def load_catalyst_context(base_dir: Path, tickers: Iterable[str]) -> pd.DataFrame:
    texts = _read_browser_texts(base_dir)
    rows = []
    for ticker_raw in tickers:
        ticker = str(ticker_raw or "").strip().upper()
        if not ticker:
            continue
        source_hits: list[str] = []
        snippets: list[str] = []
        word_counts: Counter[str] = Counter()
        for name, text in texts:
            lower = text.lower()
            if _mentions(text, ticker):
                source_hits.append(name)
                snippets.append(_snippet(text, ticker))
                for word in POSITIVE_WORDS | NEGATIVE_WORDS | MACRO_RISK_WORDS:
                    word_counts[word] += lower.count(word)
        pos = sum(word_counts[w] for w in POSITIVE_WORDS)
        neg = sum(word_counts[w] for w in NEGATIVE_WORDS)
        macro = sum(word_counts[w] for w in MACRO_RISK_WORDS)
        if not source_hits:
            status = "unknown"
            note = "No local browser/news capture matched ticker."
        elif neg > pos + 2:
            status = "caution"
            note = "Local capture has more negative/macro-risk terms than positive terms."
        elif pos > neg:
            status = "supportive"
            note = "Local capture is net supportive."
        else:
            status = "mixed"
            note = "Local capture is mixed; do not treat news as a primary edge."
        rows.append(
            {
                "ticker": ticker,
                "catalyst_status": status,
                "catalyst_note": note,
                "news_hits": len(source_hits),
                "macro_risk_hits": int(macro),
                "positive_hits": int(pos),
                "negative_hits": int(neg),
                "catalyst_sources": ";".join(source_hits[:5]),
                "catalyst_snippet": " | ".join(x for x in snippets[:2] if x),
            }
        )
    return pd.DataFrame(rows)
