"""Point-in-time external context for Pattern Analysis V2.

External sources may only downgrade a ticket. X/social data is shadow-only.
Untimestamped browser/news/SEC captures are context-only. A SEC event becomes
veto-eligible only when the record carries an acceptance/publication timestamp
on or before the signal date; a filing date without dissemination time is not
enough for historical gating.
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import pandas as pd

MATERIAL_SEC_FORMS = {
    "8-K", "8-K/A", "10-Q", "10-Q/A", "10-K", "10-K/A",
    "13D", "13D/A", "SC 13D", "SC 13D/A", "4", "4/A",
}
SUMMARY_FIELDS = [
    "source_type", "source_file_count", "item_count", "timestamped_count",
    "future_item_count", "usable_context_count", "veto_eligible_count",
    "decision_role", "status", "note", "files",
]
TICKER_FIELDS = [
    "ticker", "sec_items", "sec_material_timestamped", "x_mentions",
    "news_mentions", "latest_external_timestamp", "external_event_veto",
    "decision_role", "evidence_files", "note",
]


def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


def _ticker(value: Any) -> str:
    text = str(value or "").strip().upper().lstrip("$")
    return text if re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,9}", text) else ""


def _parse_time(value: Any) -> Optional[pd.Timestamp]:
    raw = str(value or "").strip()
    if not raw:
        return None
    parsed = pd.to_datetime(raw, errors="coerce", utc=True)
    return None if pd.isna(parsed) else parsed


def _read_csv(path: Path) -> List[Dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            return list(csv.DictReader(handle))
    except Exception:
        return []


def _normalized(row: Dict[str, Any]) -> Dict[str, Any]:
    return {_norm(key): value for key, value in row.items()}


def _first(row: Dict[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    return ""


def _cashtags(text: str) -> Set[str]:
    return {_ticker(value) for value in re.findall(r"\$([A-Z][A-Z0-9.\-]{0,9})\b", text.upper()) if _ticker(value)}


def _source_summary(
    source_type: str,
    paths: List[Path],
    item_count: int,
    timestamped_count: int,
    future_count: int,
    usable_count: int,
    veto_count: int,
    role: str,
    status: str,
    note: str,
) -> Dict[str, Any]:
    return {
        "source_type": source_type,
        "source_file_count": len(paths),
        "item_count": item_count,
        "timestamped_count": timestamped_count,
        "future_item_count": future_count,
        "usable_context_count": usable_count,
        "veto_eligible_count": veto_count,
        "decision_role": role,
        "status": status,
        "note": note,
        "files": ";".join(str(path) for path in paths),
    }


def build_external_context(base_dir: Path, out_dir: Path, as_of: str) -> Dict[str, str]:
    day_dir = base_dir / as_of
    cutoff = pd.Timestamp(f"{as_of}T23:59:59Z")
    summaries: List[Dict[str, Any]] = []
    ticker_data: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "sec_items": 0,
            "sec_material_timestamped": 0,
            "x_mentions": 0,
            "news_mentions": 0,
            "timestamps": [],
            "files": set(),
            "notes": set(),
        }
    )
    veto_tickers: Set[str] = set()

    x_paths = sorted(day_dir.glob("x_scrapes/*/posts.csv"))
    x_items = x_timestamped = x_future = x_usable = 0
    for path in x_paths:
        for raw in _read_csv(path):
            row = _normalized(raw)
            x_items += 1
            stamp = _parse_time(_first(row, ("published_at", "created_at", "timestamp")))
            if stamp is not None:
                x_timestamped += 1
                if stamp > cutoff:
                    x_future += 1
                    continue
            else:
                continue
            x_usable += 1
            text = str(_first(row, ("text", "content", "body")) or "")
            for ticker in _cashtags(text):
                ticker_data[ticker]["x_mentions"] += 1
                ticker_data[ticker]["timestamps"].append(stamp)
                ticker_data[ticker]["files"].add(str(path))
                ticker_data[ticker]["notes"].add("X mention is shadow-only and cannot promote or veto")
    summaries.append(
        _source_summary(
            "X_TWITTER", x_paths, x_items, x_timestamped, x_future, x_usable, 0,
            "SHADOW_ONLY", "AVAILABLE" if x_usable else "ABSENT_OR_UNUSABLE",
            "Strict published_at filter; social sentiment is not validated option alpha.",
        )
    )

    sec_paths = sorted(day_dir.glob("sec-filings-scrape-*.csv"))
    sec_items = sec_timestamped = sec_future = sec_usable = sec_veto = 0
    acceptance_fields = (
        "accepted_at", "acceptance_datetime", "acceptance_time", "published_at",
        "disseminated_at", "filed_at",
    )
    for path in sec_paths:
        for raw in _read_csv(path):
            row = _normalized(raw)
            sec_items += 1
            ticker = _ticker(_first(row, ("ticker", "symbol", "issuer_ticker")))
            form = str(_first(row, ("filing_type", "form_type", "form")) or "").strip().upper()
            accepted_raw = _first(row, acceptance_fields)
            stamp = _parse_time(accepted_raw)
            filing_date = _parse_time(_first(row, ("filing_date", "date_filed", "filed_date")))
            if stamp is not None:
                sec_timestamped += 1
                if stamp > cutoff:
                    sec_future += 1
                    continue
            if ticker:
                sec_usable += 1
                ticker_data[ticker]["sec_items"] += 1
                ticker_data[ticker]["files"].add(str(path))
                if stamp is not None:
                    ticker_data[ticker]["timestamps"].append(stamp)
                elif filing_date is not None:
                    ticker_data[ticker]["timestamps"].append(filing_date)
                    ticker_data[ticker]["notes"].add(
                        "SEC filing date present without dissemination timestamp; context only"
                    )
                if stamp is not None and stamp <= cutoff and form in MATERIAL_SEC_FORMS:
                    sec_veto += 1
                    veto_tickers.add(ticker)
                    ticker_data[ticker]["sec_material_timestamped"] += 1
                    ticker_data[ticker]["notes"].add(f"timestamped material SEC form {form}")
    summaries.append(
        _source_summary(
            "SEC_FILINGS", sec_paths, sec_items, sec_timestamped, sec_future, sec_usable, sec_veto,
            "EVENT_VETO_WHEN_TIMESTAMPED", "AVAILABLE" if sec_usable else "ABSENT_OR_UNUSABLE",
            "Filing date alone is context-only; acceptance/dissemination timestamp is required for a veto.",
        )
    )

    news_paths = sorted(day_dir.glob("news-feed-scrape-*.csv"))
    news_items = news_timestamped = news_future = news_usable = 0
    for path in news_paths:
        for raw in _read_csv(path):
            row = _normalized(raw)
            news_items += 1
            stamp = _parse_time(_first(row, ("published_at", "created_at", "timestamp", "published")))
            if stamp is not None:
                news_timestamped += 1
                if stamp > cutoff:
                    news_future += 1
                    continue
            text = " ".join(str(value or "") for value in row.values())
            tickers = _cashtags(text)
            if stamp is not None and tickers:
                news_usable += 1
            for ticker in tickers:
                ticker_data[ticker]["news_mentions"] += 1
                if stamp is not None:
                    ticker_data[ticker]["timestamps"].append(stamp)
                ticker_data[ticker]["files"].add(str(path))
                ticker_data[ticker]["notes"].add("news is context-only; no validated sentiment gate")
    summaries.append(
        _source_summary(
            "UW_NEWS", news_paths, news_items, news_timestamped, news_future, news_usable, 0,
            "CONTEXT_ONLY", "AVAILABLE" if news_usable else "ABSENT_OR_UNUSABLE",
            "Requires published timestamp and explicit cashtag; no sentiment-based promotion or veto.",
        )
    )

    browser_paths = sorted((day_dir / "browser_text").glob("*.txt")) if (day_dir / "browser_text").exists() else []
    summaries.append(
        _source_summary(
            "BROWSER_TEXT", browser_paths, len(browser_paths), 0, 0, len(browser_paths), 0,
            "CONTEXT_ONLY_UNTIMED", "AVAILABLE" if browser_paths else "ABSENT",
            "Capture filenames are dated but individual claims lack machine-verifiable publication timestamps.",
        )
    )

    summaries.append(
        _source_summary(
            "EARNINGS_CALENDAR", sorted(day_dir.glob("stock-screener-*.csv")) + sorted(day_dir.glob("stock-screener-*.zip")),
            0, 0, 0, 0, 0, "PRODUCTION_EVENT_GATE", "INTEGRATED_IN_CORE",
            "next_earnings_date is read from the dated stock-screener source.",
        )
    )

    ticker_rows: List[Dict[str, Any]] = []
    for ticker, values in sorted(ticker_data.items()):
        timestamps = [stamp for stamp in values["timestamps"] if stamp is not None]
        veto = ticker in veto_tickers
        ticker_rows.append(
            {
                "ticker": ticker,
                "sec_items": values["sec_items"],
                "sec_material_timestamped": values["sec_material_timestamped"],
                "x_mentions": values["x_mentions"],
                "news_mentions": values["news_mentions"],
                "latest_external_timestamp": max(timestamps).isoformat() if timestamps else "",
                "external_event_veto": veto,
                "decision_role": "EVENT_VETO" if veto else "CONTEXT_ONLY",
                "evidence_files": ";".join(sorted(values["files"])),
                "note": "; ".join(sorted(values["notes"])),
            }
        )

    summary_path = out_dir / "external_context_audit.csv"
    ticker_path = out_dir / "external_ticker_context.csv"
    pd.DataFrame(summaries, columns=SUMMARY_FIELDS).to_csv(summary_path, index=False)
    pd.DataFrame(ticker_rows, columns=TICKER_FIELDS).to_csv(ticker_path, index=False)

    board_path = out_dir / "decision_board.csv"
    enriched_path = out_dir / "decision_board_context.csv"
    if board_path.exists():
        board = pd.read_csv(board_path, low_memory=False)
        context = pd.DataFrame(ticker_rows)
        if not context.empty:
            board = board.merge(context, on="ticker", how="left", validate="many_to_one")
        else:
            for field in TICKER_FIELDS[1:]:
                board[field] = ""
        board["external_adjusted_status"] = board.get("status", "")
        veto = board.get("external_event_veto", pd.Series(False, index=board.index)).fillna(False).astype(bool)
        board.loc[veto, "external_adjusted_status"] = "EXTERNAL_EVENT_REVIEW_REQUIRED"
        board["external_can_promote"] = False
        board.to_csv(enriched_path, index=False)
    else:
        pd.DataFrame().to_csv(enriched_path, index=False)

    json_path = out_dir / "external_context_summary.json"
    json_path.write_text(
        json.dumps(
            {
                "as_of": as_of,
                "external_data_can_promote": False,
                "veto_tickers": sorted(veto_tickers),
                "source_summary": summaries,
                "ticker_context_count": len(ticker_rows),
            },
            indent=2,
            default=str,
        ) + "\n",
        encoding="utf-8",
    )
    return {
        "external_context_audit": str(summary_path),
        "external_ticker_context": str(ticker_path),
        "decision_board_context": str(enriched_path),
        "external_context_summary": str(json_path),
    }
