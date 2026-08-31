#!/usr/bin/env python3
"""Fetch a page with persistent Playwright Chrome. Fallback when MCP Playwright is not loaded."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.parse import urlparse

PROFILE = Path.home() / ".grok" / "browser" / "grok-option"
ALLOW = {
    "x.com",
    "www.x.com",
    "twitter.com",
    "www.twitter.com",
    "finance.yahoo.com",
    "finance.google.com",
    "www.marketwatch.com",
    "marketwatch.com",
    "www.cboe.com",
    "cboe.com",
    "www.barchart.com",
    "barchart.com",
    "finviz.com",
    "www.finviz.com",
    "www.investing.com",
    "investing.com",
    "www.reuters.com",
    "reuters.com",
    "apnews.com",
    "www.apnews.com",
    "www.wsj.com",
    "www.bloomberg.com",
    "www.eia.gov",
    "www.cmegroup.com",
    "www.google.com",
    "www.tradingview.com",
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("url")
    p.add_argument("--headed", action="store_true")
    p.add_argument("--cdp", default="")
    p.add_argument("--profile", default=str(PROFILE))
    p.add_argument("--timeout-ms", type=int, default=45000)
    p.add_argument("--allow-host", action="append", default=[])
    args = p.parse_args()
    host = (urlparse(args.url).hostname or "").lower()
    allowed = ALLOW | {h.lower() for h in args.allow_host}
    if host not in allowed:
        print(f"blocked host: {host}", file=sys.stderr)
        print("pass --allow-host HOST if the user named this site", file=sys.stderr)
        return 2

    from playwright.sync_api import sync_playwright

    Path(args.profile).mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        if args.cdp:
            browser = pw.chromium.connect_over_cdp(args.cdp)
            context = browser.contexts[0] if browser.contexts else browser.new_context()
            page = context.pages[0] if context.pages else context.new_page()
        else:
            context = pw.chromium.launch_persistent_context(
                args.profile,
                channel="chrome",
                headless=not args.headed,
                viewport={"width": 1280, "height": 800},
            )
            page = context.new_page()
        page.goto(args.url, wait_until="domcontentloaded", timeout=args.timeout_ms)
        page.wait_for_timeout(1500)
        title = page.title()
        text = page.inner_text("body")
        print(f"TITLE: {title}")
        print(f"URL: {page.url}")
        print("---")
        print(text[:20000])
        if not args.cdp:
            context.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
