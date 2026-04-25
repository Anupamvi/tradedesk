#!/usr/bin/env python3
"""
Capture visible text from logged-in browser search pages for sentiment analysis.

This is a capture helper, not a trading model. It opens X, Reddit, Google News,
and Schwab research pages in the browser and stores visible page text as dated
artifacts that uwos.sentiment_pipeline can ingest.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence
from urllib.parse import quote_plus

from uwos import paths


SOURCE_URLS = {
    "x": "https://x.com/search?q={query}&src=typed_query&f=live",
    "reddit": "https://www.reddit.com/search/?q={query}&sort=new",
    "news": "https://news.google.com/search?q={query}&hl=en-US&gl=US&ceid=US:en",
    "schwab": "",
}

TICKER_RE = re.compile(r"\$?([A-Za-z][A-Za-z0-9.]{0,7})\b")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture logged-in browser text for sentiment pipeline inputs.")
    parser.add_argument("query", nargs="+", help="Search query, e.g. NFLX stock options or Iran war stocks.")
    parser.add_argument("--run-date", default=dt.date.today().isoformat(), help="YYYY-MM-DD output folder.")
    parser.add_argument("--base-dir", default="", help="UW trade desk root. Defaults to project root.")
    parser.add_argument(
        "--out-subdir",
        default="browser_text",
        help="Subdirectory inside the date folder for captured artifacts.",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["x", "reddit", "news"],
        choices=sorted(SOURCE_URLS.keys()),
        help="Sources to open.",
    )
    parser.add_argument(
        "--profile-dir",
        default="tokens/browser_sentiment_profile",
        help="Persistent Playwright profile dir. Use a logged-in profile for X/Reddit.",
    )
    parser.add_argument(
        "--engine",
        choices=["auto", "playwright", "applescript"],
        default="auto",
        help="Capture engine. applescript drives the logged-in Google Chrome app on macOS.",
    )
    parser.add_argument("--browser-channel", choices=["chromium", "chrome", "msedge"], default="chrome")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wait-ms", type=int, default=5000, help="Wait after page open.")
    parser.add_argument("--scrolls", type=int, default=8, help="Scroll cycles before final text capture.")
    parser.add_argument("--scroll-delay-ms", type=int, default=900)
    parser.add_argument("--screenshot", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--disable-automation-flags",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass --disable-blink-features=AutomationControlled.",
    )
    return parser.parse_args(argv)


def safe_slug(value: str) -> str:
    out = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    out = re.sub(r"-{2,}", "-", out).strip("-")
    return (out or "query").lower()[:90]


def build_url(source: str, query: str) -> str:
    if source not in SOURCE_URLS:
        raise ValueError(f"Unknown source: {source}")
    if source == "schwab":
        ticker = extract_first_ticker(query)
        if ticker:
            return f"https://client.schwab.com/app/research/#/stocks/{ticker}/news"
        return f"https://www.schwab.com/search?query={quote_plus(query)}"
    return SOURCE_URLS[source].format(query=quote_plus(query))


def extract_first_ticker(query: str) -> str:
    common = {"STOCK", "STOCKS", "OPTION", "OPTIONS", "NEWS", "FLOW", "TRADES", "TRADE"}
    for match in TICKER_RE.finditer(query or ""):
        value = re.sub(r"[^A-Za-z0-9.]", "", match.group(1)).upper()
        if 1 <= len(value) <= 8 and value not in common:
            return value
    return ""


def normalize_text(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", str(text or "").replace("\r\n", "\n")).strip()


def write_capture_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["captured_at", "source", "query", "url", "text_file", "screenshot_file", "char_count", "text"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def run_osascript(script: str) -> str:
    proc = subprocess.run(["osascript", "-e", script], text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "osascript failed")
    return proc.stdout


def chrome_execute_javascript(js: str) -> str:
    script = f'''
tell application "Google Chrome"
    if not (exists window 1) then make new window
    tell active tab of front window
        execute javascript {json.dumps(js)}
    end tell
end tell
'''
    return run_osascript(script)


def chrome_scroll_with_keyboard() -> None:
    script = '''
tell application "Google Chrome"
    activate
    if (exists window 1) then set index of front window to 1
end tell
delay 0.4
tell application "System Events"
    tell process "Google Chrome"
        set frontmost to true
        try
            perform action "AXRaise" of window 1
            set {winX, winY} to position of window 1
            set {winW, winH} to size of window 1
            click at {winX + (winW div 2), winY + (winH div 2)}
        end try
    end tell
    delay 0.1
    key code 125
    key code 125
    key code 125
end tell
'''
    run_osascript(script)


def chrome_copy_page_text() -> str:
    script = '''
tell application "Google Chrome"
    activate
    if (exists window 1) then set index of front window to 1
end tell
delay 0.5
tell application "System Events"
    tell process "Google Chrome"
        set frontmost to true
        try
            perform action "AXRaise" of window 1
            set {winX, winY} to position of window 1
            set {winW, winH} to size of window 1
            click at {winX + (winW div 2), winY + (winH div 2)}
        end try
    end tell
    delay 0.2
    keystroke "a" using command down
    delay 0.2
    keystroke "c" using command down
    delay 0.5
end tell
'''
    run_osascript(script)
    proc = subprocess.run(["pbpaste"], text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "pbpaste failed")
    return proc.stdout


def capture_with_applescript(args: argparse.Namespace) -> Dict[str, Path]:
    query = " ".join(args.query).strip()
    run_date = dt.date.fromisoformat(str(args.run_date).strip()[:10])
    root = Path(args.base_dir).expanduser().resolve() if args.base_dir else paths.project_root()
    out_dir = root / run_date.isoformat() / str(args.out_subdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, str]] = []
    captured_at = dt.datetime.now(dt.timezone.utc).isoformat()
    query_slug = safe_slug(query)
    for source in args.sources:
        url = build_url(source, query)
        open_script = f'''
tell application "Google Chrome"
    activate
    if not (exists window 1) then make new window
    set URL of active tab of front window to {json.dumps(url)}
end tell
'''
        run_osascript(open_script)
        subprocess.run(["sleep", str(max(0.5, int(args.wait_ms) / 1000.0))], check=False)
        for _ in range(max(0, int(args.scrolls))):
            try:
                chrome_execute_javascript("window.scrollBy(0, 1300);")
            except Exception:
                try:
                    chrome_scroll_with_keyboard()
                except Exception:
                    pass
            subprocess.run(["sleep", str(max(0.1, int(args.scroll_delay_ms) / 1000.0))], check=False)
        try:
            text = normalize_text(chrome_execute_javascript("document.body ? document.body.innerText : ''"))
        except Exception:
            text = normalize_text(chrome_copy_page_text())
        text_path = out_dir / f"browser-text-capture-{source}-{query_slug}-{run_date.isoformat()}.txt"
        text_path.write_text(text + "\n", encoding="utf-8")
        screenshot_path = ""
        if bool(args.screenshot):
            shot = out_dir / f"browser-text-capture-{source}-{query_slug}-{run_date.isoformat()}.png"
            subprocess.run(["screencapture", "-x", str(shot)], check=False)
            if shot.exists():
                screenshot_path = str(shot)
        rows.append(
            {
                "captured_at": captured_at,
                "source": source,
                "query": query,
                "url": url,
                "text_file": str(text_path),
                "screenshot_file": screenshot_path,
                "char_count": str(len(text)),
                "text": text[:12000],
            }
        )

    csv_path = out_dir / f"browser-text-capture-{query_slug}-{run_date.isoformat()}.csv"
    manifest_path = out_dir / f"browser-text-capture-manifest-{query_slug}-{run_date.isoformat()}.json"
    write_capture_csv(csv_path, rows)
    manifest_path.write_text(
        json.dumps(
            {
                "captured_at": captured_at,
                "query": query,
                "run_date": run_date.isoformat(),
                "sources": list(args.sources),
                "engine": "applescript",
                "csv": str(csv_path),
                "rows": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {manifest_path}")
    return {"csv": csv_path, "manifest": manifest_path}


def capture_with_playwright(args: argparse.Namespace) -> Dict[str, Path]:
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        if str(args.engine) == "auto":
            return capture_with_applescript(args)
        raise SystemExit(
            "Playwright is required for --engine playwright. Use --engine applescript to drive logged-in Chrome without Playwright."
        ) from exc

    query = " ".join(args.query).strip()
    run_date = dt.date.fromisoformat(str(args.run_date).strip()[:10])
    root = Path(args.base_dir).expanduser().resolve() if args.base_dir else paths.project_root()
    out_dir = root / run_date.isoformat() / str(args.out_subdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    profile_dir = Path(args.profile_dir)
    if not profile_dir.is_absolute():
        profile_dir = root / profile_dir
    profile_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, str]] = []
    captured_at = dt.datetime.now(dt.timezone.utc).isoformat()
    query_slug = safe_slug(query)
    with sync_playwright() as pw:
        launch_kwargs = {
            "user_data_dir": str(profile_dir),
            "headless": bool(args.headless),
            "accept_downloads": True,
            "viewport": {"width": 1440, "height": 1100},
        }
        if args.browser_channel != "chromium":
            launch_kwargs["channel"] = args.browser_channel
        if bool(args.disable_automation_flags):
            launch_kwargs["args"] = ["--disable-blink-features=AutomationControlled"]
        context = pw.chromium.launch_persistent_context(**launch_kwargs)
        try:
            page = context.pages[0] if context.pages else context.new_page()
            for source in args.sources:
                url = build_url(source, query)
                page.goto(url, wait_until="domcontentloaded", timeout=60000)
                page.wait_for_timeout(max(500, int(args.wait_ms)))
                for _ in range(max(0, int(args.scrolls))):
                    page.mouse.wheel(0, 1300)
                    page.wait_for_timeout(max(100, int(args.scroll_delay_ms)))
                text = normalize_text(page.evaluate("() => document.body ? document.body.innerText : ''"))
                text_path = out_dir / f"browser-text-capture-{source}-{query_slug}-{run_date.isoformat()}.txt"
                text_path.write_text(text + "\n", encoding="utf-8")
                screenshot_path = ""
                if bool(args.screenshot):
                    shot = out_dir / f"browser-text-capture-{source}-{query_slug}-{run_date.isoformat()}.png"
                    page.screenshot(path=str(shot), full_page=True)
                    screenshot_path = str(shot)
                rows.append(
                    {
                        "captured_at": captured_at,
                        "source": source,
                        "query": query,
                        "url": url,
                        "text_file": str(text_path),
                        "screenshot_file": screenshot_path,
                        "char_count": str(len(text)),
                        "text": text[:12000],
                    }
                )
        finally:
            context.close()

    csv_path = out_dir / f"browser-text-capture-{query_slug}-{run_date.isoformat()}.csv"
    manifest_path = out_dir / f"browser-text-capture-manifest-{query_slug}-{run_date.isoformat()}.json"
    write_capture_csv(csv_path, rows)
    manifest_path.write_text(
        json.dumps(
            {
                "captured_at": captured_at,
                "query": query,
                "run_date": run_date.isoformat(),
                "sources": list(args.sources),
                "engine": "playwright",
                "profile_dir": str(profile_dir),
                "csv": str(csv_path),
                "rows": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {manifest_path}")
    return {"csv": csv_path, "manifest": manifest_path}


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if str(args.engine) == "applescript":
        capture_with_applescript(args)
    else:
        capture_with_playwright(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
