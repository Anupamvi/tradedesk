# Browser / Playwright ingest

Use a real browser when search, `open_page`, or X API are blocked or logged-out. **Do not use the browser for ticker quotes or option chains** when Schwab auth works (`references/schwab.md`). Do not launch a browser on every scan.

Never invent IV/OI from a blurry screenshot. Read the DOM (snapshot or extracted text). Credentials: **do not type passwords**. Headed window + user completes login. Do not automate Schwab 2FA.

## What to use (first that works)

1. **MCP Playwright** — `search_tool` for `playwright browser_navigate` (qualified name like `playwright__browser_navigate`). Then `use_tool`. Persistent profile: `~/.grok/browser/grok-option`. Cookies (including x.com) survive sessions once the user has logged in headed.
2. **Python Playwright fallback** — if MCP is not in this session:
   `python3 ~/.grok/skills/grok-option/scripts/browser_fetch.py URL`
   Add `--headed` for a first login. Same profile directory.
3. **Attach to a debug Chrome** — user runs `scripts/open-chrome-debug.sh`, then MCP with CDP `http://127.0.0.1:9222`, or `browser_fetch.py --cdp http://127.0.0.1:9222`. This is the path for “use my already-logged-in Chrome.” Default Chrome without a debug port cannot be taken over while it is running.
4. **Thin fallback** — `web_search` / `open_page` / `x_keyword_search`. Mark **THIN**. No Prime.

If none of 1–3 work, say so. Do not pretend to be logged into x.com.

## When grok-option must open a page

- **Earnings date** missing or unconfirmed (Schwab chain does not replace an earnings calendar)
- Schwab **down** and a public quote page is the only remaining check — still MIXED/THIN, not Prime
- **WTI/Brent** session move for Spike
- Named-outlet **geo** copy behind a cookie wall
- X desk posts the API did not return (logged-in following / search)

Not for: picking tickers off the For You feed, copying a viral strike, filling Schwab orders.

## x.com (logged-in)

Profile cookies are the login. First scan that needs X in-browser: headed, navigate to `https://x.com`, wait, tell the user to finish login/2FA in that window, then snapshot.

After login, search the **same** shock or candidate ticker. X remains **confirm/veto**, never the universe. Do not scrape DMs. Do not post.

## Allowlisted hosts (default)

Stay on market pages unless the user names another host: `x.com`, `twitter.com`, `finance.yahoo.com`, `finance.google.com`, `marketwatch.com`, `cboe.com`, `barchart.com`, `finviz.com`, `investing.com`, `reuters.com`, `apnews.com`, `wsj.com`, `bloomberg.com`, `eia.gov`, `cmegroup.com`. Earnings calendars on those hosts are fine.

## Tool pattern (MCP)

```
search_tool query="playwright browser_navigate"
use_tool playwright__browser_navigate  { url }
use_tool playwright__browser_snapshot  {}
# click/type only if needed for chain/earnings widgets
```

Prefer snapshot text over screenshots. Screenshot only to keep a record in the daily card, not as the quote source.

## First-time setup (user)

1. Restart Grok after Playwright MCP is in `~/.grok/config.toml` so tools load.
2. Headed Chrome opens the grok-option profile (not daily Chrome).
3. Log into x.com once. Optional: Yahoo Finance.
4. Later scans reuse that profile.

To attach **daily** Chrome instead: quit Chrome, or use a separate debug profile via `open-chrome-debug.sh` (port 9222). Do not point `--user-data-dir` at `~/Library/Application Support/Google/Chrome` while that Chrome is open.
