from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import pandas as pd


GATES = {
    "Fed": ["fed", "fomc", "powell", "rate decision"],
    "CPI": ["cpi", "consumer price index", "inflation print"],
    "PCE": ["pce", "personal consumption expenditures"],
    "jobs": ["jobs report", "nonfarm", "payroll", "unemployment"],
    "major earnings": ["earnings", "results", "guidance"],
    "geopolitical/macro shock": ["geopolitical", "war", "tariff", "oil shock", "iran", "china", "treasury yield"],
}


def _browser_texts(base_dir: Path) -> list[tuple[str, str]]:
    browser_dir = base_dir / "browser_text"
    if not browser_dir.is_dir():
        return []
    rows: list[tuple[str, str]] = []
    for path in sorted(browser_dir.glob("browser-text-capture-*")):
        if path.suffix.lower() not in {".txt", ".csv", ".json"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if text.strip():
            rows.append((path.name, text))
    return rows


def _snippet(text: str, tokens: list[str]) -> str:
    lower_tokens = [token.lower() for token in tokens]
    for line in text.splitlines():
        lower = line.lower()
        if any(token in lower for token in lower_tokens):
            return line.strip()[:220]
    return ""


def build_macro_event_gates(
    *,
    base_dir: Path,
    asof: dt.date,
    stock_screener: pd.DataFrame | None = None,
    regime: dict[str, Any] | None = None,
) -> pd.DataFrame:
    texts = _browser_texts(base_dir)
    rows: list[dict[str, Any]] = []
    generated_at = dt.datetime.now(dt.timezone.utc).isoformat()
    all_text = "\n".join(text for _, text in texts)
    for gate, tokens in GATES.items():
        hits = [name for name, text in texts if any(token in text.lower() for token in tokens)]
        snippet = _snippet(all_text, tokens) if hits else ""
        rows.append(
            {
                "timestamp": generated_at,
                "asof": str(asof),
                "gate": gate,
                "status": "observed" if hits else "unconfirmed",
                "source_count": len(hits),
                "sources": ";".join(hits[:5]),
                "evidence": snippet,
                "decision_impact": "manual confirmation required if trade thesis depends on this gate" if not hits else "review evidence before Execute",
            }
        )
    if stock_screener is not None and not stock_screener.empty and "next_earnings_dt" in stock_screener.columns:
        upcoming = stock_screener.copy()
        upcoming["next_earnings_dt"] = pd.to_datetime(upcoming["next_earnings_dt"], errors="coerce")
        upcoming["days_to_earnings"] = (upcoming["next_earnings_dt"].dt.date - asof).map(lambda value: value.days if pd.notna(value) else None)
        upcoming = upcoming[
            pd.to_numeric(upcoming["days_to_earnings"], errors="coerce").between(0, 7, inclusive="both")
        ]
        if not upcoming.empty:
            tickers = ",".join(sorted(upcoming["ticker"].astype(str).str.upper().head(25)))
            rows.append(
                {
                    "timestamp": generated_at,
                    "asof": str(asof),
                    "gate": "major earnings",
                    "status": "observed",
                    "source_count": int(len(upcoming)),
                    "sources": "stock_screener.next_earnings_dt",
                    "evidence": f"earnings within 7 days for {tickers}",
                    "decision_impact": "single-name trades must pass earnings/event risk gate",
                }
            )
    if regime:
        rows.append(
            {
                "timestamp": generated_at,
                "asof": str(asof),
                "gate": "market regime",
                "status": "observed",
                "source_count": 1,
                "sources": "stock_screener_index_proxies",
                "evidence": f"trend={regime.get('trend')}; vol={regime.get('volatility')}; flow={regime.get('flow')}; vix={regime.get('vix_proxy')}",
                "decision_impact": "index/ETF trades require explicit regime alignment",
            }
        )
    return pd.DataFrame(rows)


def write_macro_event_gates(out_dir: Path, asof: dt.date, gates: pd.DataFrame) -> tuple[Path, Path, dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"codexdaily_v3_macro_event_gates_{asof}.csv"
    gates.to_csv(csv_path, index=False)
    summary = {
        "status": "ok",
        "rows": int(len(gates)),
        "observed": int(gates["status"].astype(str).eq("observed").sum()) if not gates.empty else 0,
        "unconfirmed": int(gates["status"].astype(str).eq("unconfirmed").sum()) if not gates.empty else 0,
        "csv": str(csv_path),
    }
    json_path = out_dir / f"codexdaily_v3_macro_event_gates_{asof}.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return csv_path, json_path, summary
