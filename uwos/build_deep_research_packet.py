#!/usr/bin/env python3
"""
Build a single markdown packet for Deep Research upload.

Output:
- <day-folder>/deep_research_packet_YYYY-MM-DD.md
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


ROUTE_PATTERNS: Dict[str, List[str]] = {
    "stock_screener": ["stock-screener-*.csv"],
    "analysts": ["analysts-scrape-*.csv"],
    "insiders_trades": ["insiders-trades-scrape-*.csv"],
    "institutions": ["institutions-scrape-*.csv"],
    "news_feed": ["news-feed-scrape-*.csv"],
    "earnings": ["earnings-scrape-*.csv"],
    "flow_sectors": ["flow-sectors-scrape-*.csv"],
    "market_statistics": ["market-statistics-scrape-*.csv"],
    "sec_filings": ["sec-filings-scrape-*.csv"],
    "correlations": ["correlations-scrape-*.csv"],
    "smart_money_live": ["smart-money-live-scrape-*.csv"],
    "market_maps": ["market-maps-scrape-*.csv"],
}


DEEP_RESEARCH_PROMPT = """You are an institutional long-only growth PM + forensic catalyst analyst.

Goal:
Build a US equity portfolio designed to outperform the S&P 500 over a 3-5 year horizon, with monthly rebalance rules. Focus on growth names (including emerging names like PLTR-style profiles), while keeping risk-controlled sector diversification.

Hard constraints:
1) Use only date-stamped evidence. No unstated assumptions.
2) If stock-level universe data is missing (sector, market cap, liquidity), STOP and list missing inputs.
3) Exclude ETFs, indices, illiquid names, and names with insufficient data coverage.
4) Each candidate must have at least 3 independent signal groups:
   - Growth/fundamental signal
   - Catalyst signal (SEC/K filings, earnings/news)
   - Market signal (flow/technical/sentiment)
5) Portfolio constraints:
   - 12-20 stocks
   - At least 6 sectors
   - Max 20% sector weight
   - Max 10% single-name weight
6) Add/trim rules must be explicit and monthly executable.

Data to use (priority):
A) Uploaded Unusual Whales exports
B) SEC EDGAR filings (13D/13G/13F, Form 4, 8-K, 10-Q, 10-K)
C) Earnings revisions/transcript catalyst context
D) X/Twitter sentiment (bot-filtered; event-window aware)
E) Technical indicators: 50/200DMA trend, RSI regime, drawdown from 52w high, volatility regime

Required scoring framework (show exact formula):
- Growth quality: 35%
- Catalyst strength: 25%
- Flow/sentiment/positioning: 20%
- Technical regime: 15%
- Risk/valuation penalty: 5%

Required outputs:
1) Data quality report (missing files, stale dates, weak fields).
2) Ranked top 30 candidates with score breakdown by factor.
3) Final portfolio (12-20 names) with weights and sector map.
4) Monthly rebalance actions:
   - Add list (new buys)
   - Increase list
   - Trim list (overextended names)
   - Exit list (thesis broken)
5) Buy-the-dip candidates:
   - materially off highs, but fundamentals/catalysts intact
6) Trim/recycle-capital candidates:
   - overextended price + weakening catalyst stack
7) Benchmark comparison section:
   - expected behavior vs S&P 500 in bull/base/bear scenarios
8) Output both Markdown and machine-readable JSON.

Important:
- Do not claim certainty or guaranteed outperformance.
- Every major conclusion must cite source + date.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one deep research packet markdown from a UW day folder.")
    parser.add_argument(
        "--trade-date",
        default=dt.date.today().isoformat(),
        help="Day folder to process (YYYY-MM-DD). Default: today.",
    )
    parser.add_argument(
        "--base-dir",
        default=".",
        help="Root folder that contains daily YYYY-MM-DD folders.",
    )
    parser.add_argument(
        "--stock-lookback-days",
        type=int,
        default=7,
        help="Fallback lookback window for stock-screener if missing in selected day.",
    )
    parser.add_argument(
        "--route-lookback-days",
        type=int,
        default=14,
        help="Fallback lookback window for all route CSVs if missing in selected day.",
    )
    parser.add_argument(
        "--artifact-lookback-days",
        type=int,
        default=14,
        help="Fallback lookback window for generated artifacts (manifest/reports/candidates).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="How many top scored rows from growth candidates CSV to include.",
    )
    return parser.parse_args()


def parse_trade_date(text: str) -> dt.date:
    try:
        return dt.datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError as exc:
        raise SystemExit(f"--trade-date must be YYYY-MM-DD, got: {text}") from exc


def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").strip().lower())


def count_csv_rows(path: Path) -> int:
    try:
        with path.open("r", newline="", encoding="utf-8", errors="replace") as fh:
            reader = csv.reader(fh)
            rows = list(reader)
        if not rows:
            return 0
        header = [normalize_text(x) for x in rows[0]]
        data = 0
        for row in rows[1:]:
            if not row:
                continue
            if [normalize_text(x) for x in row] == header:
                continue
            if not any(str(c).strip() for c in row):
                continue
            data += 1
        return data
    except Exception:
        return 0


def choose_best_csv(paths: Sequence[Path]) -> Optional[Path]:
    candidates = [p for p in paths if p.exists() and p.is_file()]
    if not candidates:
        return None

    def rank(path: Path) -> Tuple[int, int, int]:
        rows = count_csv_rows(path)
        st = path.stat()
        return rows, int(st.st_size), int(st.st_mtime_ns)

    return max(candidates, key=rank)


def find_best_csv(day_dir: Path, patterns: Sequence[str], include_unzipped: bool) -> Optional[Path]:
    candidates: List[Path] = []
    for pat in patterns:
        candidates.extend(day_dir.glob(pat))
    if include_unzipped:
        uz = day_dir / "_unzipped_mode_a"
        if uz.exists():
            for pat in patterns:
                candidates.extend(uz.glob(pat))
    dedup = list(dict.fromkeys(candidates))
    return choose_best_csv(dedup)


def extract_day_from_path(path: Path) -> Optional[dt.date]:
    for parent in path.parents:
        if re.match(r"^\d{4}-\d{2}-\d{2}$", parent.name):
            try:
                return dt.datetime.strptime(parent.name, "%Y-%m-%d").date()
            except Exception:
                return None
    return None


def find_recent_stock_screener(base_dir: Path, trade_date: dt.date, lookback_days: int) -> Optional[Path]:
    candidates = list(base_dir.glob("*/stock-screener-*.csv")) + list(base_dir.glob("*/_unzipped_mode_a/stock-screener-*.csv"))
    ranked: List[Tuple[int, int, int, Path]] = []
    for p in candidates:
        if not p.exists() or not p.is_file():
            continue
        day = extract_day_from_path(p)
        if day is None:
            continue
        age = (trade_date - day).days
        if age < 0 or age > max(0, lookback_days):
            continue
        rows = count_csv_rows(p)
        st = p.stat()
        ranked.append((rows, -age, int(st.st_mtime_ns), p))
    if not ranked:
        return None
    ranked.sort(reverse=True)
    return ranked[0][3]


def iter_day_dirs(base_dir: Path) -> List[Tuple[dt.date, Path]]:
    out: List[Tuple[dt.date, Path]] = []
    for child in base_dir.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", name):
            continue
        try:
            d = dt.datetime.strptime(name, "%Y-%m-%d").date()
        except Exception:
            continue
        out.append((d, child))
    out.sort(key=lambda x: x[0])
    return out


def find_recent_best_csv(
    base_dir: Path,
    trade_date: dt.date,
    patterns: Sequence[str],
    include_unzipped: bool,
    lookback_days: int,
) -> Optional[Path]:
    best_path: Optional[Path] = None
    best_rank: Optional[Tuple[int, int, int]] = None
    for day, day_dir in iter_day_dirs(base_dir):
        age = (trade_date - day).days
        if age < 0 or age > max(0, int(lookback_days)):
            continue
        p = find_best_csv(day_dir, patterns, include_unzipped=include_unzipped)
        if not p:
            continue
        rows = count_csv_rows(p)
        st = p.stat()
        rank = (rows, -age, int(st.st_mtime_ns))
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_path = p
    return best_path


def find_recent_day_artifact(
    base_dir: Path,
    trade_date: dt.date,
    stem_prefix: str,
    ext: str,
    lookback_days: int,
) -> Optional[Path]:
    best: Optional[Path] = None
    best_rank: Optional[Tuple[int, int]] = None
    for day, day_dir in iter_day_dirs(base_dir):
        age = (trade_date - day).days
        if age < 0 or age > max(0, int(lookback_days)):
            continue
        p = day_dir / f"{stem_prefix}_{day.isoformat()}.{ext}"
        if not p.exists() or not p.is_file():
            continue
        st = p.stat()
        rank = (-age, int(st.st_mtime_ns))
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best = p
    return best


def read_growth_candidates(path: Optional[Path], top_n: int) -> Tuple[List[dict], List[dict]]:
    if not path or not path.exists():
        return [], []
    try:
        with path.open("r", newline="", encoding="utf-8", errors="replace") as fh:
            rows = list(csv.DictReader(fh))
    except Exception:
        return [], []
    for r in rows:
        try:
            r["_score"] = float(r.get("score", "") or 0.0)
        except Exception:
            r["_score"] = 0.0
        try:
            r["_sources"] = int(float(r.get("source_count", "") or 0))
        except Exception:
            r["_sources"] = 0
    ranked = sorted(rows, key=lambda r: (r["_score"], r["_sources"]), reverse=True)
    top = ranked[: max(1, int(top_n))]
    trims = [r for r in ranked if r.get("action") == "Trim / Avoid"]
    return top, trims[:15]


def fmt_path(p: Optional[Path]) -> str:
    if not p:
        return "not found"
    return str(p)


def format_age(age: Optional[int]) -> str:
    if age is None:
        return "-"
    return f"{age}d"


def build_packet(
    trade_date: dt.date,
    day_dir: Path,
    output_path: Path,
    source_info: Dict[str, dict],
    artifact_info: Dict[str, dict],
    capture_report: Optional[Path],
    growth_report: Optional[Path],
    growth_csv: Optional[Path],
    manifest_csv: Optional[Path],
    top_rows: Sequence[dict],
    trim_rows: Sequence[dict],
) -> None:
    now = dt.datetime.now().isoformat(timespec="seconds")
    lines: List[str] = []
    lines.append(f"# Deep Research Packet - {trade_date.isoformat()}")
    lines.append("")
    lines.append(f"- Generated: {now}")
    lines.append(f"- Day folder: `{day_dir}`")
    lines.append("")

    lines.append("## Upload These Files To Deep Research")
    lines.append(f"- `{fmt_path(output_path)}`")
    if capture_report:
        lines.append(f"- `{capture_report}`")
    if growth_report:
        lines.append(f"- `{growth_report}`")
    if growth_csv:
        lines.append(f"- `{growth_csv}`")
    if manifest_csv:
        lines.append(f"- `{manifest_csv}`")
    for key in ROUTE_PATTERNS.keys():
        info = source_info.get(key, {})
        p = info.get("path")
        if p:
            rows = int(info.get("rows", 0))
            lines.append(f"- `{p}` ({rows} rows)")
    lines.append("")

    lines.append("## Data Readiness")
    lines.append("")
    lines.append("| Dataset | Status | Rows | Source Day | Age | File |")
    lines.append("|---|---|---:|---|---:|---|")
    for key in ROUTE_PATTERNS.keys():
        info = source_info.get(key, {})
        p = info.get("path")
        rows = int(info.get("rows", 0))
        day = info.get("day")
        age = info.get("age_days")
        origin = str(info.get("origin", "missing"))
        status = "ok_today" if origin == "today" else ("ok_fallback" if origin == "fallback" else "missing")
        day_txt = day.isoformat() if isinstance(day, dt.date) else "-"
        file_txt = f"`{p}`" if p else "-"
        lines.append(f"| {key} | {status} | {rows} | {day_txt} | {format_age(age)} | {file_txt} |")
    lines.append("")

    lines.append("## Artifact Readiness")
    lines.append("")
    lines.append("| Artifact | Status | Source Day | Age | File |")
    lines.append("|---|---|---|---:|---|")
    for key in ["manifest_csv", "capture_report", "growth_report", "growth_csv"]:
        info = artifact_info.get(key, {})
        p = info.get("path")
        day = info.get("day")
        age = info.get("age_days")
        origin = str(info.get("origin", "missing"))
        status = "ok_today" if origin == "today" else ("ok_fallback" if origin == "fallback" else "missing")
        day_txt = day.isoformat() if isinstance(day, dt.date) else "-"
        file_txt = f"`{p}`" if p else "-"
        lines.append(f"| {key} | {status} | {day_txt} | {format_age(age)} | {file_txt} |")
    lines.append("")

    lines.append("## Current Snapshot")
    if top_rows:
        lines.append("")
        lines.append("| Rank | Ticker | Sector | Score | Action | Confidence | Sources | Positives | Flags |")
        lines.append("|---:|---|---|---:|---|---|---:|---|---|")
        for i, r in enumerate(top_rows, start=1):
            lines.append(
                f"| {i} | {r.get('ticker','')} | {r.get('sector','')} | {float(r.get('_score',0.0)):.2f} | {r.get('action','')} | {r.get('confidence','')} | {int(r.get('_sources',0))} | {r.get('positives','') or '-'} | {r.get('flags','') or '-'} |"
            )
    else:
        lines.append("")
        lines.append("- `growth_portfolio_candidates` CSV missing or unreadable.")
    lines.append("")

    lines.append("## Trim/Reduce Snapshot")
    if trim_rows:
        lines.append("")
        lines.append("| Ticker | Sector | Score | Flags |")
        lines.append("|---|---|---:|---|")
        for r in trim_rows:
            lines.append(
                f"| {r.get('ticker','')} | {r.get('sector','')} | {float(r.get('_score',0.0)):.2f} | {r.get('flags','') or '-'} |"
            )
    else:
        lines.append("")
        lines.append("- No current `Trim / Avoid` rows in candidate CSV.")
    lines.append("")

    lines.append("## Deep Research Prompt")
    lines.append("")
    lines.append("```text")
    lines.extend(DEEP_RESEARCH_PROMPT.rstrip().splitlines())
    lines.append("```")
    lines.append("")

    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    trade_date = parse_trade_date(args.trade_date)
    base_dir = Path(args.base_dir).resolve()
    day_dir = base_dir / trade_date.isoformat()
    if not day_dir.exists():
        print(f"Day folder does not exist: {day_dir}")
        return 2

    artifact_info: Dict[str, dict] = {}

    def resolve_artifact(key: str, prefix: str, ext: str) -> Optional[Path]:
        today = day_dir / f"{prefix}_{trade_date.isoformat()}.{ext}"
        if today.exists() and today.is_file():
            artifact_info[key] = {
                "path": today,
                "origin": "today",
                "day": trade_date,
                "age_days": 0,
            }
            return today
        fallback = find_recent_day_artifact(
            base_dir=base_dir,
            trade_date=trade_date,
            stem_prefix=prefix,
            ext=ext,
            lookback_days=int(args.artifact_lookback_days),
        )
        if fallback:
            day = extract_day_from_path(fallback)
            age = (trade_date - day).days if isinstance(day, dt.date) else None
            artifact_info[key] = {
                "path": fallback,
                "origin": "fallback",
                "day": day,
                "age_days": age,
            }
            return fallback
        artifact_info[key] = {
            "path": None,
            "origin": "missing",
            "day": None,
            "age_days": None,
        }
        return None

    capture_report = resolve_artifact("capture_report", "uw_capture_final_report", "md")
    growth_report = resolve_artifact("growth_report", "growth_portfolio_candidates", "md")
    growth_csv = resolve_artifact("growth_csv", "growth_portfolio_candidates", "csv")
    manifest_csv = resolve_artifact("manifest_csv", "uw_capture_manifest", "csv")

    source_files: Dict[str, Optional[Path]] = {}
    source_info: Dict[str, dict] = {}
    for key, patterns in ROUTE_PATTERNS.items():
        include_unzipped = key == "stock_screener"
        p_today = find_best_csv(day_dir, patterns, include_unzipped=include_unzipped)
        origin = "today"
        p = p_today
        if p is None:
            lb = int(args.stock_lookback_days) if key == "stock_screener" else int(args.route_lookback_days)
            if key == "stock_screener":
                p = find_recent_stock_screener(base_dir=base_dir, trade_date=trade_date, lookback_days=lb)
            else:
                p = find_recent_best_csv(
                    base_dir=base_dir,
                    trade_date=trade_date,
                    patterns=patterns,
                    include_unzipped=include_unzipped,
                    lookback_days=lb,
                )
            origin = "fallback" if p is not None else "missing"

        source_files[key] = p
        if p is None:
            source_info[key] = {
                "path": None,
                "origin": "missing",
                "rows": 0,
                "day": None,
                "age_days": None,
            }
        else:
            day = extract_day_from_path(p)
            age = (trade_date - day).days if isinstance(day, dt.date) else None
            source_info[key] = {
                "path": p,
                "origin": origin,
                "rows": count_csv_rows(p),
                "day": day,
                "age_days": age,
            }

    top_rows, trim_rows = read_growth_candidates(growth_csv, top_n=int(args.top_n))

    out_path = day_dir / f"deep_research_packet_{trade_date.isoformat()}.md"
    build_packet(
        trade_date=trade_date,
        day_dir=day_dir,
        output_path=out_path,
        source_info=source_info,
        artifact_info=artifact_info,
        capture_report=capture_report,
        growth_report=growth_report,
        growth_csv=growth_csv,
        manifest_csv=manifest_csv,
        top_rows=top_rows,
        trim_rows=trim_rows,
    )

    print(f"Deep research packet: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
