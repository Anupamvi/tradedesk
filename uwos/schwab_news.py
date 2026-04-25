#!/usr/bin/env python3
"""Fetch Schwab/broker news into dated sentiment artifacts."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from uwos import paths
from uwos.schwab_auth import SchwabAuthConfig, SchwabLiveDataService, normalize_symbols


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch best-effort Schwab news for tickers.")
    parser.add_argument("symbols", nargs="*", help="Ticker symbols.")
    parser.add_argument("--symbols-csv", default="", help="Comma-separated ticker list.")
    parser.add_argument("--run-date", default=dt.date.today().isoformat(), help="YYYY-MM-DD output folder.")
    parser.add_argument("--base-dir", default="", help="UW trade desk root. Defaults to project root.")
    parser.add_argument("--out-subdir", default="schwab_news", help="Subdirectory inside date folder.")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--manual-auth", action="store_true", help="Use manual Schwab OAuth flow.")
    return parser.parse_args(argv)


def row_from_item(symbol: str, item: Dict[str, Any], endpoint: str, status: str) -> Dict[str, Any]:
    related = item.get("symbols") or []
    if isinstance(related, list):
        related_text = ",".join(str(x) for x in related)
    else:
        related_text = str(related or "")
    return {
        "symbol": symbol,
        "published_at": str(item.get("published_at", "") or ""),
        "source": str(item.get("source", "Schwab") or "Schwab"),
        "headline": str(item.get("headline", "") or ""),
        "summary": str(item.get("summary", "") or ""),
        "url": str(item.get("url", "") or ""),
        "related_symbols": related_text,
        "schwab_endpoint": endpoint,
        "schwab_status": status,
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    columns = [
        "symbol",
        "published_at",
        "source",
        "headline",
        "summary",
        "url",
        "related_symbols",
        "schwab_endpoint",
        "schwab_status",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def fetch_schwab_news(
    symbols: Sequence[str],
    *,
    root: Path,
    run_date: dt.date,
    out_subdir: str = "schwab_news",
    limit: int = 20,
    manual_auth: bool = False,
) -> Dict[str, Path]:
    tickers = normalize_symbols(symbols)
    out_dir = root / run_date.isoformat() / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "-".join(tickers[:12]) if tickers else "none"
    csv_path = out_dir / f"schwab-news-{run_date.isoformat()}-{suffix}.csv"
    raw_path = out_dir / f"schwab-news-raw-{run_date.isoformat()}-{suffix}.json"
    rows: List[Dict[str, Any]] = []
    raw: Dict[str, Any] = {"run_date": run_date.isoformat(), "symbols": tickers, "responses": {}}

    if tickers:
        cfg = SchwabAuthConfig.from_env(load_dotenv_file=True)
        svc = SchwabLiveDataService(config=cfg, manual_auth=manual_auth, interactive_login=False)
        response = svc.get_news(tickers, limit=limit)
        raw["responses"]["batch"] = response
        endpoint = str(response.get("endpoint", ""))
        status = str(response.get("status", ""))
        for item in response.get("items", []) or []:
            related = item.get("symbols") or tickers
            matched = [s for s in tickers if not related or s in related] or tickers
            for symbol in matched:
                rows.append(row_from_item(symbol, item, endpoint, status))

    write_csv(csv_path, rows)
    raw_path.write_text(json.dumps(raw, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {raw_path}")
    return {"csv": csv_path, "raw": raw_path}


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    root = Path(args.base_dir).expanduser().resolve() if args.base_dir else paths.project_root()
    run_date = dt.date.fromisoformat(str(args.run_date).strip()[:10])
    symbols = normalize_symbols(list(args.symbols) + ([args.symbols_csv] if args.symbols_csv else []))
    fetch_schwab_news(
        symbols,
        root=root,
        run_date=run_date,
        out_subdir=str(args.out_subdir),
        limit=int(args.limit),
        manual_auth=bool(args.manual_auth),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
