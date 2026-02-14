#!/usr/bin/env python3
"""
Guided Unusual Whales dashboard capture tool.

Purpose:
- Reuse your logged-in dashboard session.
- Let you set filters manually.
- Capture downloadable files where available.
- Capture CSV/XHR JSON payloads from page network traffic.
- Scrape visible tables/grids when no export exists.

Output:
- Writes files into <base-dir>/<trade-date>/.
- Writes a manifest CSV with all captured artifacts.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import re
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


ROUTES: Dict[str, Dict[str, str]] = {
    "stock-screener": {
        "url": "https://unusualwhales.com/stock-screener",
        "prefix": "stock-screener",
    },
    "hot-chains": {
        "url": "https://unusualwhales.com/options-screener",
        "prefix": "hot-chains",
    },
    "chain-oi-changes": {
        "url": "https://unusualwhales.com/flow/chain_oi_changes",
        "prefix": "chain-oi-changes",
    },
    "dp-eod-report": {
        "url": "https://unusualwhales.com/dark-pool-flow",
        "prefix": "dp-eod-report",
    },
    "news-feed": {
        "url": "https://unusualwhales.com/news-feed",
        "prefix": "news-feed",
    },
    "earnings": {
        "url": "https://unusualwhales.com/earnings",
        "prefix": "earnings",
    },
    "analysts": {
        "url": "https://unusualwhales.com/analysts",
        "prefix": "analysts",
    },
    "insiders-trades": {
        "url": "https://unusualwhales.com/insiders/trades",
        "prefix": "insiders-trades",
    },
    "institutions": {
        "url": "https://unusualwhales.com/institutions",
        "prefix": "institutions",
    },
    "flow-sectors": {
        "url": "https://unusualwhales.com/flow/sectors",
        "prefix": "flow-sectors",
    },
    "market-statistics": {
        "url": "https://unusualwhales.com/market/statistics",
        "prefix": "market-statistics",
    },
    "market-maps": {
        "url": "https://unusualwhales.com/market/maps",
        "prefix": "market-maps",
    },
    "sec-filings": {
        "url": "https://unusualwhales.com/sec",
        "prefix": "sec-filings",
    },
    "correlations": {
        "url": "https://unusualwhales.com/flow/correlations",
        "prefix": "correlations",
    },
    "smart-money-live": {
        "url": "https://unusualwhales.com/predictions/smart-money/live-trades",
        "prefix": "smart-money-live",
    },
    "economic-calendar": {
        "url": "https://unusualwhales.com/economic-calendar",
        "prefix": "economic-calendar",
    },
    "fda-calendar": {
        "url": "https://unusualwhales.com/fda-calendar",
        "prefix": "fda-calendar",
    },
    "flow-overview": {
        "url": "https://unusualwhales.com/flow/overview",
        "prefix": "flow-overview",
    },
    "interval-flow": {
        "url": "https://unusualwhales.com/interval-flow",
        "prefix": "interval-flow",
    },
    "lit-flow": {
        "url": "https://unusualwhales.com/lit-flow",
        "prefix": "lit-flow",
    },
}

PRESETS: Dict[str, List[str]] = {
    "mode-a-core": [
        "stock-screener",
        "hot-chains",
        "chain-oi-changes",
        "dp-eod-report",
    ],
    "growth-intel": [
        "stock-screener",
        "news-feed",
        "earnings",
        "analysts",
        "insiders-trades",
        "institutions",
        "flow-sectors",
        "market-statistics",
        "sec-filings",
        "correlations",
        "smart-money-live",
        "market-maps",
    ],
    "catalyst-intel": [
        "news-feed",
        "earnings",
        "sec-filings",
        "economic-calendar",
        "fda-calendar",
        "analysts",
        "insiders-trades",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture UW dashboard data into dated daily folders.")
    parser.add_argument(
        "--trade-date",
        default=dt.date.today().isoformat(),
        help="Date folder to write into (YYYY-MM-DD). Default: today.",
    )
    parser.add_argument(
        "--base-dir",
        default=".",
        help="Root folder that contains daily YYYY-MM-DD folders.",
    )
    parser.add_argument(
        "--preset",
        default="mode-a-core",
        choices=sorted(PRESETS.keys()),
        help="Route preset to use when --routes is not provided.",
    )
    parser.add_argument(
        "--routes",
        nargs="+",
        default=None,
        help=f"Optional explicit route keys. Known keys: {', '.join(sorted(ROUTES.keys()))}",
    )
    parser.add_argument(
        "--profile-dir",
        default="tokens/uw_playwright_profile",
        help="Persistent browser profile dir (keeps login/cookies).",
    )
    parser.add_argument(
        "--browser-channel",
        choices=["chromium", "chrome", "msedge"],
        default="chromium",
        help="Browser engine/channel for Playwright persistent profile.",
    )
    parser.add_argument(
        "--manual-login-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Open browser only for manual UW login/session bootstrap, then exit.",
    )
    parser.add_argument(
        "--login-url",
        default="https://unusualwhales.com/login",
        help="URL to open in manual-login-only mode.",
    )
    parser.add_argument(
        "--disable-automation-flags",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass flags that reduce automation fingerprinting on some sites.",
    )
    parser.add_argument(
        "--wait-seconds",
        type=int,
        default=40,
        help="How long to wait for download/CSV/JSON responses after capture starts.",
    )
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run browser headless. Default: false.",
    )
    parser.add_argument(
        "--auto-click",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Attempt to click common Export/Download buttons before waiting.",
    )
    parser.add_argument(
        "--interactive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pause for manual filter setup before each capture.",
    )
    parser.add_argument(
        "--scrape-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If no downloadable output is found, scrape the largest visible table/grid to CSV.",
    )
    parser.add_argument(
        "--also-scrape",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Scrape table/grid even when a download/CSV capture succeeds.",
    )
    parser.add_argument(
        "--extract-zips",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If a ZIP is downloaded, extract CSVs into <day>/_unzipped_mode_a.",
    )
    parser.add_argument(
        "--capture-json",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Capture network JSON payloads for non-downloadable pages.",
    )
    parser.add_argument(
        "--json-min-bytes",
        type=int,
        default=1,
        help="Ignore JSON responses smaller than this payload size.",
    )
    parser.add_argument(
        "--max-json-per-route",
        type=int,
        default=40,
        help="Maximum JSON payload files saved per route.",
    )
    parser.add_argument(
        "--scroll-after-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-scroll page after filters are set (helps load virtualized rows).",
    )
    parser.add_argument(
        "--scroll-steps",
        type=int,
        default=10,
        help="Number of wheel scroll steps after filter setup.",
    )
    parser.add_argument(
        "--scroll-delay-ms",
        type=int,
        default=350,
        help="Delay between scroll steps.",
    )
    parser.add_argument(
        "--scrape-scroll-cycles",
        type=int,
        default=10,
        help="Additional scroll/scrape cycles used to expand virtualized tables.",
    )
    parser.add_argument(
        "--snapshot-html",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save per-route HTML snapshot.",
    )
    parser.add_argument(
        "--snapshot-screenshot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save per-route full-page screenshot.",
    )
    parser.add_argument(
        "--snapshot-text",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save per-route plain text snapshot (document.body.innerText).",
    )
    return parser.parse_args()


def resolve_routes(args: argparse.Namespace) -> List[str]:
    routes = args.routes if args.routes else PRESETS[args.preset]
    unknown = [r for r in routes if r not in ROUTES]
    if unknown:
        raise SystemExit(f"Unknown route key(s): {', '.join(unknown)}")
    return list(routes)


def parse_trade_date(text: str) -> dt.date:
    try:
        return dt.datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError as exc:
        raise SystemExit(f"--trade-date must be YYYY-MM-DD, got: {text}") from exc


def sanitize_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip())
    safe = re.sub(r"-{2,}", "-", safe).strip("-")
    return safe or "download.bin"


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    i = 1
    while True:
        candidate = path.with_name(f"{stem}-{i}{suffix}")
        if not candidate.exists():
            return candidate
        i += 1


def route_output_path(day_dir: Path, route_key: str, trade_date: dt.date, ext: str, label: Optional[str] = None) -> Path:
    prefix = ROUTES[route_key]["prefix"]
    ext_norm = ext if ext.startswith(".") else f".{ext}"
    name_prefix = f"{prefix}-{label}" if label else prefix
    return unique_path(day_dir / f"{name_prefix}-{trade_date.isoformat()}{ext_norm.lower()}")


def try_click_export(page) -> Optional[str]:
    selectors = [
        "button:has-text('Export CSV')",
        "button:has-text('Download CSV')",
        "button:has-text('Export')",
        "a:has-text('Export CSV')",
        "a:has-text('Download CSV')",
        "[role='button']:has-text('Export')",
        "text=/export csv/i",
        "text=/download csv/i",
    ]
    for selector in selectors:
        try:
            loc = page.locator(selector).first
            if loc.count() == 0:
                continue
            if not loc.is_visible():
                continue
            loc.click(timeout=3000)
            return selector
        except Exception:
            continue
    return None


def write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[str]]) -> int:
    if not header and not rows:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if header:
            writer.writerow(header)
        writer.writerows(rows)
    return len(rows)


def scrape_best_table_or_grid(page) -> Tuple[str, List[str], List[List[str]]]:
    payload = page.evaluate(
        """
() => {
  function normalize(text) {
    return (text || "").replace(/\\s+/g, " ").trim();
  }

  function scoreCandidate(header, rows) {
    const h = header ? header.length : 0;
    const r = rows ? rows.length : 0;
    const width = Math.max(1, h, ...rows.map((x) => x.length));
    let score = r * width;

    // Bias toward data-like tables.
    if (width >= 4) score += 1000;
    if (r >= 10) score += 500;
    if (r <= 2 && width <= 2) score -= 1200;

    const headerText = (header || []).join(" ").toLowerCase();
    const navLike =
      headerText.includes("options flow") &&
      headerText.includes("options screener") &&
      headerText.includes("stock screener");
    const filterLike =
      headerText.includes("switches") ||
      headerText.includes("toggles") ||
      headerText.includes("others") ||
      headerText.includes("powered by unusualwhales");
    if (navLike) score -= 2500;
    if (filterLike) score -= 1500;
    return score;
  }

  const candidates = [];

  const htmlTables = Array.from(document.querySelectorAll("table"));
  for (const table of htmlTables) {
    const header = Array.from(table.querySelectorAll("thead th")).map((x) => normalize(x.innerText));
    let rows = Array.from(table.querySelectorAll("tbody tr")).map((tr) =>
      Array.from(tr.querySelectorAll("td,th")).map((x) => normalize(x.innerText))
    );
    rows = rows.filter((r) => r.some((c) => c.length > 0));
    if (!header.length && rows.length) {
      const first = rows[0];
      const hasDistinct = first.some((c) => c.length > 0);
      if (hasDistinct) {
        candidates.push({ kind: "table", header: first, rows: rows.slice(1), score: scoreCandidate(first, rows.slice(1)) });
      }
    }
    candidates.push({ kind: "table", header, rows, score: scoreCandidate(header, rows) });
  }

  const gridRoots = Array.from(document.querySelectorAll("[role='grid'], [role='table']"));
  for (const root of gridRoots) {
    let header = Array.from(root.querySelectorAll("[role='columnheader']")).map((x) => normalize(x.innerText)).filter(Boolean);
    const rowNodes = Array.from(root.querySelectorAll("[role='row']"));
    let rows = rowNodes.map((row) =>
      Array.from(row.querySelectorAll("[role='gridcell'], [role='cell'], [role='rowheader'], [role='columnheader']"))
        .map((x) => normalize(x.innerText))
        .filter(Boolean)
    );
    rows = rows.filter((r) => r.length > 0 && r.some((c) => c.length > 0));
    if (!header.length && rows.length) {
      header = rows[0];
      rows = rows.slice(1);
    }
    candidates.push({ kind: "aria-grid", header, rows, score: scoreCandidate(header, rows) });
  }

  candidates.sort((a, b) => b.score - a.score);
  const best = candidates.find((c) => c.rows && c.rows.length > 0);
  if (!best) return null;
  return best;
}
        """
    )
    if not payload:
        return "none", [], []
    kind = str(payload.get("kind") or "unknown")
    header = payload.get("header") or []
    rows = payload.get("rows") or []
    return kind, header, rows


def extract_zip_csvs(zip_path: Path, target_dir: Path) -> List[Path]:
    extracted: List[Path] = []
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            if member.endswith("/") or not member.lower().endswith(".csv"):
                continue
            out_path = unique_path(target_dir / Path(member).name)
            with zf.open(member, "r") as src, out_path.open("wb") as dst:
                dst.write(src.read())
            extracted.append(out_path)
    return extracted


def print_route_banner(route_key: str) -> None:
    cfg = ROUTES[route_key]
    print("")
    print("=" * 92)
    print(f"Route: {route_key}")
    print(f"URL:   {cfg['url']}")
    print("=" * 92)


def wait_for_capture(page, baseline_downloads: int, baseline_csv: int, downloads, csv_responses, wait_seconds: int) -> str:
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if len(downloads) > baseline_downloads:
            return "download"
        if len(csv_responses) > baseline_csv:
            return "csv_response"
        page.wait_for_timeout(300)
    return "none"


def auto_scroll(page, steps: int, delay_ms: int) -> None:
    for _ in range(max(0, steps)):
        try:
            page.mouse.wheel(0, 1400)
        except Exception:
            pass
        page.wait_for_timeout(max(50, delay_ms))


def scrape_with_progressive_scroll(page, cycles: int, delay_ms: int) -> Tuple[str, List[str], List[List[str]]]:
    best_kind, best_header, best_rows = scrape_best_table_or_grid(page)

    def score(header: Sequence[str], rows: Sequence[Sequence[str]]) -> int:
        width = max([len(header)] + [len(r) for r in rows] + [1])
        return len(rows) * width

    best_score = score(best_header, best_rows)
    header_key = tuple(best_header)
    seen = set(tuple(r) for r in best_rows)

    for _ in range(max(0, cycles)):
        try:
            page.mouse.wheel(0, 1500)
        except Exception:
            pass
        page.wait_for_timeout(max(50, delay_ms))
        kind, header, rows = scrape_best_table_or_grid(page)
        cur_score = score(header, rows)
        if tuple(header) == header_key and rows:
            for row in rows:
                t = tuple(row)
                if t not in seen:
                    seen.add(t)
                    best_rows.append(row)
            best_score = score(best_header, best_rows)
        elif cur_score > best_score:
            best_kind, best_header, best_rows = kind, header, rows
            best_score = cur_score
            header_key = tuple(best_header)
            seen = set(tuple(r) for r in best_rows)
    return best_kind, best_header, best_rows


def save_route_snapshots(
    page,
    day_dir: Path,
    route_key: str,
    trade_date: dt.date,
    url: str,
    manifest_rows: List[dict],
    save_html: bool,
    save_screenshot: bool,
    save_text: bool,
) -> None:
    snap_dir = day_dir / "_snapshots" / route_key
    snap_dir.mkdir(parents=True, exist_ok=True)
    prefix = ROUTES[route_key]["prefix"]

    if save_screenshot:
        png_path = unique_path(snap_dir / f"{prefix}-snapshot-{trade_date.isoformat()}.png")
        try:
            page.screenshot(path=str(png_path), full_page=True)
            manifest_rows.append(
                {
                    "route": route_key,
                    "kind": "snapshot_png",
                    "file": str(png_path.relative_to(day_dir)),
                    "source_url": url,
                    "bytes": png_path.stat().st_size if png_path.exists() else "",
                    "rows": "",
                    "captured_at": dt.datetime.now().isoformat(timespec="seconds"),
                }
            )
        except Exception:
            pass

    if save_html:
        html_path = unique_path(snap_dir / f"{prefix}-snapshot-{trade_date.isoformat()}.html")
        try:
            html = page.content()
            html_path.write_text(html, encoding="utf-8", errors="replace")
            manifest_rows.append(
                {
                    "route": route_key,
                    "kind": "snapshot_html",
                    "file": str(html_path.relative_to(day_dir)),
                    "source_url": url,
                    "bytes": html_path.stat().st_size if html_path.exists() else "",
                    "rows": "",
                    "captured_at": dt.datetime.now().isoformat(timespec="seconds"),
                }
            )
        except Exception:
            pass

    if save_text:
        txt_path = unique_path(snap_dir / f"{prefix}-snapshot-{trade_date.isoformat()}.txt")
        meta_path = unique_path(snap_dir / f"{prefix}-snapshot-{trade_date.isoformat()}-meta.json")
        try:
            text = page.evaluate("() => (document && document.body && document.body.innerText) ? document.body.innerText : ''")
            txt_path.write_text(str(text), encoding="utf-8", errors="replace")
            meta = {"url": page.url, "title": page.title(), "captured_at": dt.datetime.now().isoformat(timespec="seconds")}
            meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=True), encoding="utf-8")
            manifest_rows.append(
                {
                    "route": route_key,
                    "kind": "snapshot_text",
                    "file": str(txt_path.relative_to(day_dir)),
                    "source_url": url,
                    "bytes": txt_path.stat().st_size if txt_path.exists() else "",
                    "rows": "",
                    "captured_at": dt.datetime.now().isoformat(timespec="seconds"),
                }
            )
            manifest_rows.append(
                {
                    "route": route_key,
                    "kind": "snapshot_meta",
                    "file": str(meta_path.relative_to(day_dir)),
                    "source_url": url,
                    "bytes": meta_path.stat().st_size if meta_path.exists() else "",
                    "rows": "",
                    "captured_at": dt.datetime.now().isoformat(timespec="seconds"),
                }
            )
        except Exception:
            pass


def summarize_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.2f} MB"


def save_json_payloads(
    day_dir: Path,
    route_key: str,
    trade_date: dt.date,
    payloads: Sequence[dict],
    max_per_route: int,
    manifest_rows: List[dict],
) -> int:
    if not payloads:
        return 0
    out_dir = day_dir / "_network_json" / route_key
    out_dir.mkdir(parents=True, exist_ok=True)

    seen = set()
    saved = 0
    sorted_payloads = sorted(payloads, key=lambda p: int(p.get("bytes", 0)), reverse=True)
    for entry in sorted_payloads:
        if saved >= max_per_route:
            break
        body = entry["body"]
        url = str(entry.get("url", ""))
        key = (url, len(body), hashlib.sha1(body[:4096]).hexdigest())
        if key in seen:
            continue
        seen.add(key)

        text = None
        parsed = None
        try:
            text = body.decode("utf-8")
        except Exception:
            try:
                text = body.decode("latin-1")
            except Exception:
                text = None
        if text is not None:
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = None

        base_name = f"{ROUTES[route_key]['prefix']}-{trade_date.isoformat()}-json-{saved + 1:02d}"
        out_path = out_dir / f"{base_name}.json"
        if parsed is not None:
            out_path.write_text(json.dumps(parsed, indent=2, ensure_ascii=True), encoding="utf-8")
        else:
            out_path = out_dir / f"{base_name}.txt"
            if text is not None:
                out_path.write_text(text, encoding="utf-8", errors="replace")
            else:
                out_path.write_bytes(body)

        manifest_rows.append(
            {
                "route": route_key,
                "kind": "network_json",
                "file": str(out_path.relative_to(day_dir)),
                "source_url": url,
                "bytes": int(entry.get("bytes", len(body))),
                "rows": "",
                "captured_at": dt.datetime.now().isoformat(timespec="seconds"),
            }
        )
        saved += 1
    return saved


def main() -> int:
    args = parse_args()
    route_keys = resolve_routes(args)
    trade_date = parse_trade_date(args.trade_date)

    base_dir = Path(args.base_dir).resolve()
    day_dir = base_dir / trade_date.isoformat()
    day_dir.mkdir(parents=True, exist_ok=True)
    unzip_dir = day_dir / "_unzipped_mode_a"
    profile_dir = Path(args.profile_dir).resolve()
    profile_dir.mkdir(parents=True, exist_ok=True)

    print("Using route order:")
    for r in route_keys:
        print(f"- {r} -> {ROUTES[r]['url']}")
    print("")
    print("Note: use this only on your own account and within your subscription/ToS permissions.")
    print(f"Output folder:  {day_dir}")
    print(f"Browser profile:{profile_dir}")
    print("")

    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        print("Playwright is required. Install with:")
        print("  pip install playwright")
        print("  python -m playwright install chromium")
        return 2

    captured_summary: List[str] = []
    manifest_rows: List[dict] = []

    with sync_playwright() as pw:
        launch_kwargs = {
            "user_data_dir": str(profile_dir),
            "headless": bool(args.headless),
            "accept_downloads": True,
            "viewport": {"width": 1720, "height": 1040},
        }
        if args.browser_channel in {"chrome", "msedge"}:
            launch_kwargs["channel"] = args.browser_channel
        if args.disable_automation_flags:
            launch_kwargs["args"] = ["--disable-blink-features=AutomationControlled"]
        context = pw.chromium.launch_persistent_context(**launch_kwargs)

        page = context.pages[0] if context.pages else context.new_page()
        page.set_default_timeout(25000)

        if args.manual_login_only:
            print("")
            print("Manual login mode enabled.")
            print(f"Opening: {args.login_url}")
            print("Log in to Unusual Whales in the opened browser window.")
            print("After login succeeds, browse to another normal site to confirm network is healthy.")
            try:
                page.goto(args.login_url, wait_until="domcontentloaded")
            except Exception as exc:
                print(f"Navigation warning in manual-login-only mode: {exc}")
            try:
                input("Press ENTER here after login/session is stable to save profile and exit...")
            except EOFError:
                print("No interactive stdin detected; waiting 120 seconds before closing.")
                page.wait_for_timeout(120000)
            context.close()
            print("Profile session saved.")
            return 0

        active_route = {"key": None}
        downloads: List[dict] = []
        csv_responses: List[dict] = []
        json_responses: List[dict] = []

        def on_download(download):
            downloads.append({"route": active_route["key"], "download": download, "time": time.time()})
            try:
                name = download.suggested_filename
            except Exception:
                name = "unknown"
            print(f"[download-event] {name}")

        def on_response(response):
            route = active_route["key"]
            if not route:
                return

            try:
                ct = (response.headers.get("content-type") or "").lower()
            except Exception:
                ct = ""
            url = response.url
            url_lower = url.lower()
            status = 0
            try:
                status = int(response.status)
            except Exception:
                status = 0

            is_csv = (
                "text/csv" in ct
                or "application/csv" in ct
                or "text/plain" in ct and ("csv" in url_lower or "download" in url_lower)
                or url_lower.endswith(".csv")
                or "format=csv" in url_lower
            )
            is_json = (
                "application/json" in ct
                or "text/json" in ct
                or url_lower.endswith(".json")
                or "/api/" in url_lower
            )

            if is_csv:
                try:
                    body = response.body()
                except Exception:
                    return
                csv_responses.append(
                    {
                        "route": route,
                        "url": url,
                        "status": status,
                        "body": body,
                        "bytes": len(body),
                        "time": time.time(),
                    }
                )
                print(f"[csv-response] {url[:130]} ({summarize_bytes(len(body))})")
                return

            if args.capture_json and is_json and status >= 200 and status < 400:
                try:
                    body = response.body()
                except Exception:
                    return
                if len(body) < int(args.json_min_bytes):
                    return
                json_responses.append(
                    {
                        "route": route,
                        "url": url,
                        "status": status,
                        "body": body,
                        "bytes": len(body),
                        "time": time.time(),
                    }
                )

        page.on("download", on_download)
        page.on("response", on_response)

        for route_key in route_keys:
            print_route_banner(route_key)
            active_route["key"] = route_key
            url = ROUTES[route_key]["url"]

            baseline_downloads = len(downloads)
            baseline_csv = len(csv_responses)
            baseline_json = len(json_responses)

            try:
                page.goto(url, wait_until="domcontentloaded")
                page.wait_for_timeout(1800)
            except Exception as exc:
                msg = f"{route_key}: navigation failed ({exc})"
                print(msg)
                captured_summary.append(msg)
                continue

            if args.interactive:
                print("Set your filters in the browser now.")
                try:
                    input("Press ENTER when ready to capture this route...")
                except EOFError:
                    print("No interactive stdin detected; continuing without pause.")

            if args.scroll_after_filter:
                auto_scroll(page, steps=int(args.scroll_steps), delay_ms=int(args.scroll_delay_ms))

            save_route_snapshots(
                page=page,
                day_dir=day_dir,
                route_key=route_key,
                trade_date=trade_date,
                url=url,
                manifest_rows=manifest_rows,
                save_html=bool(args.snapshot_html),
                save_screenshot=bool(args.snapshot_screenshot),
                save_text=bool(args.snapshot_text),
            )

            clicked_selector = None
            if args.auto_click:
                clicked_selector = try_click_export(page)
                if clicked_selector:
                    print(f"Auto-clicked export/download via selector: {clicked_selector}")
                else:
                    print("Auto-click did not find an export button.")

            if args.interactive and not clicked_selector:
                print("If needed, click Export/Download manually now.")
            print(f"Waiting up to {args.wait_seconds}s for downloadable/csv responses...")

            _ = wait_for_capture(
                page=page,
                baseline_downloads=baseline_downloads,
                baseline_csv=baseline_csv,
                downloads=downloads,
                csv_responses=csv_responses,
                wait_seconds=int(args.wait_seconds),
            )

            new_downloads = [d for d in downloads[baseline_downloads:] if d.get("route") == route_key]
            new_csv = [c for c in csv_responses[baseline_csv:] if c.get("route") == route_key]
            new_json = [j for j in json_responses[baseline_json:] if j.get("route") == route_key]
            nontrivial_json_count = len([j for j in new_json if int(j.get("bytes", 0)) >= 100])

            saved_any_primary = False

            # Save downloads first (highest trust).
            for item in new_downloads:
                download = item["download"]
                suggested = sanitize_filename(download.suggested_filename or f"{ROUTES[route_key]['prefix']}.bin")
                ext = Path(suggested).suffix.lower() or ".bin"
                out_path = route_output_path(day_dir, route_key, trade_date, ext)
                try:
                    download.save_as(str(out_path))
                except Exception as exc:
                    print(f"{route_key}: failed saving download ({exc})")
                    continue
                print(f"Saved download: {out_path}")
                saved_any_primary = True

                manifest_rows.append(
                    {
                        "route": route_key,
                        "kind": "download",
                        "file": str(out_path.relative_to(day_dir)),
                        "source_url": "",
                        "bytes": out_path.stat().st_size if out_path.exists() else "",
                        "rows": "",
                        "captured_at": dt.datetime.now().isoformat(timespec="seconds"),
                    }
                )

                if out_path.suffix.lower() == ".zip" and args.extract_zips:
                    try:
                        extracted = extract_zip_csvs(out_path, unzip_dir)
                    except Exception as exc:
                        print(f"ZIP extraction failed for {out_path.name}: {exc}")
                        extracted = []
                    for p in extracted:
                        print(f"Extracted CSV: {p}")
                        manifest_rows.append(
                            {
                                "route": route_key,
                                "kind": "zip_extract_csv",
                                "file": str(p.relative_to(day_dir)),
                                "source_url": str(out_path.name),
                                "bytes": p.stat().st_size if p.exists() else "",
                                "rows": "",
                                "captured_at": dt.datetime.now().isoformat(timespec="seconds"),
                            }
                        )

            # If no file download, save CSV network responses.
            if not saved_any_primary and new_csv:
                csv_sorted = sorted(new_csv, key=lambda c: int(c.get("bytes", 0)), reverse=True)
                seen_csv = set()
                for idx, entry in enumerate(csv_sorted[:6], start=1):
                    body = entry["body"]
                    url_src = str(entry.get("url", ""))
                    key = (url_src, len(body), hashlib.sha1(body[:4096]).hexdigest())
                    if key in seen_csv:
                        continue
                    seen_csv.add(key)
                    label = "network-csv" if idx == 1 else f"network-csv-{idx}"
                    out_path = route_output_path(day_dir, route_key, trade_date, ".csv", label=label)
                    out_path.write_bytes(body)
                    print(f"Saved CSV response: {out_path} ({summarize_bytes(len(body))})")
                    print(f"Source URL: {url_src}")
                    saved_any_primary = True
                    manifest_rows.append(
                        {
                            "route": route_key,
                            "kind": "network_csv",
                            "file": str(out_path.relative_to(day_dir)),
                            "source_url": url_src,
                            "bytes": len(body),
                            "rows": "",
                            "captured_at": dt.datetime.now().isoformat(timespec="seconds"),
                        }
                    )

            # Save JSON payloads for non-downloadable intelligence pages.
            json_saved = 0
            if args.capture_json and new_json:
                json_saved = save_json_payloads(
                    day_dir=day_dir,
                    route_key=route_key,
                    trade_date=trade_date,
                    payloads=new_json,
                    max_per_route=int(args.max_json_per_route),
                    manifest_rows=manifest_rows,
                )
                if json_saved:
                    print(f"Saved {json_saved} network JSON payload(s) for {route_key}.")

            # Scrape visible table/grid for pages without exports.
            should_scrape = bool(args.scrape_fallback) and (bool(args.also_scrape) or not saved_any_primary)
            scraped_rows = 0
            if should_scrape:
                try:
                    scrape_kind, header, rows = scrape_with_progressive_scroll(
                        page=page,
                        cycles=int(args.scrape_scroll_cycles),
                        delay_ms=int(args.scroll_delay_ms),
                    )
                except Exception as exc:
                    scrape_kind, header, rows = "none", [], []
                    print(f"Table/grid scrape failed: {exc}")
                if rows:
                    out_path = route_output_path(day_dir, route_key, trade_date, ".csv", label="scrape")
                    scraped_rows = write_csv(out_path, header, rows)
                    print(f"Saved {scrape_kind} scrape: {out_path} ({scraped_rows} rows)")
                    manifest_rows.append(
                        {
                            "route": route_key,
                            "kind": f"scrape_{scrape_kind}",
                            "file": str(out_path.relative_to(day_dir)),
                            "source_url": url,
                            "bytes": out_path.stat().st_size if out_path.exists() else "",
                            "rows": scraped_rows,
                            "captured_at": dt.datetime.now().isoformat(timespec="seconds"),
                        }
                    )

            if saved_any_primary or json_saved or scraped_rows:
                suffix = ""
                if not saved_any_primary and scraped_rows <= 2 and nontrivial_json_count == 0:
                    suffix = " [low-signal capture]"
                captured_summary.append(
                    f"{route_key}: ok (download/csv={saved_any_primary}, json={json_saved}, scrape_rows={scraped_rows}){suffix}"
                )
            else:
                msg = f"{route_key}: no capture (download/csv/json/scrape) within window"
                print(msg)
                captured_summary.append(msg)

        active_route["key"] = None
        context.close()

    manifest_path = day_dir / f"uw_capture_manifest_{trade_date.isoformat()}.csv"
    manifest_cols = ["route", "kind", "file", "source_url", "bytes", "rows", "captured_at"]
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=manifest_cols)
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow({k: row.get(k, "") for k in manifest_cols})

    print("")
    print("Capture summary")
    for line in captured_summary:
        print(f"- {line}")
    print("")
    print(f"Manifest: {manifest_path}")
    print(f"Done. Output in: {day_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
