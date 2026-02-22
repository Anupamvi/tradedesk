# Strategy Engine (Robot Layer)

This folder contains a deterministic filter that turns your 5 EOD files into a shortlist of mathematically valid option spreads.

## What it does
- Loads **Hot Chains**, **OI Changes**, **Dark Pool EOD**, **Stock Screener**, **Whale Trades**
- Applies deterministic gates:
  - ETFs excluded (Rule #5) by default
  - Width tiers by price
  - Credit >= 25% of width (credit spreads)
  - Debit <= 55% of width (debit spreads)
  - SHIELD trades cannot cross earnings
  - Ticker must appear in Hot/Whale/OI

- Prices spreads using your preference:
  - **C:** Web chain snapshot (Yahoo via `yfinance`)
  - **B:** Fallback to UnusualWhales bid/ask as *indicative* if web is unavailable

## Install
```bash
pip install pandas numpy pyyaml yfinance
```

## Run
Put the 5 files in a folder (example `./EOD`):

- hot-chains-YYYY-MM-DD.zip
- chain-oi-changes-YYYY-MM-DD.zip
- dp-eod-report-YYYY-MM-DD.zip
- stock-screener-YYYY-MM-DD.zip
- whale_trades_filtered-YYYY-MM-DD.csv

Then:

```bash
python -m uwos.strategy_engine --date 2026-01-23 --input-dir ./EOD --out-dir ./out --config uwos/rulebook_config.yaml
```

Outputs:
- `shortlist_trades_YYYY-MM-DD.csv`
- `reject_log_YYYY-MM-DD.csv`
- `SHORTLIST_YYYY-MM-DD.md`  (copy/paste into AI and ask it to roast the trades)

## Guided UW Dashboard Capture (No API)
If your UW plan is dashboard-only, use the guided browser capture tool to export files into your daily folder:

```bash
python -m uwos.uw_dashboard_capture --trade-date 2026-02-07 --base-dir c:\uw_root --preset mode-a-core
```

Default preset (`mode-a-core`) routes:
- `stock-screener`
- `hot-chains`
- `chain-oi-changes`
- `dp-eod-report`

For portfolio growth research (non-easy-download pages), use:

```bash
python -m uwos.uw_dashboard_capture --trade-date 2026-02-07 --base-dir c:\uw_root --preset growth-intel
```

`growth-intel` includes:
- `stock-screener`
- `news-feed`
- `earnings`
- `analysts`
- `insiders-trades`
- `institutions`
- `flow-sectors`
- `market-statistics`
- `sec-filings`
- `correlations`
- `smart-money-live`
- `market-maps`

Notes:
- The script opens a persistent browser profile (`tokens/uw_playwright_profile`) so your login is reused.
- You can set filters manually on each page.
- It captures downloads, CSV responses, and network JSON payloads (for non-downloadable pages).
- If a page has no direct download, it can scrape the largest visible table/grid to CSV.
- It also stores per-route snapshots under `<day-folder>/_snapshots/<route>/`:
  - full-page `.png`
  - rendered `.html`
  - plain-text `.txt`
- ZIP downloads are extracted to `<day-folder>\_unzipped_mode_a`.
- A manifest is written to `<day-folder>/uw_capture_manifest_YYYY-MM-DD.csv`.

Finalize a capture day into one markdown report and remove duplicate rerun CSVs:

```bash
python -m uwos.uw_capture_finalize --trade-date 2026-02-07 --base-dir c:\uw_root --delete-duplicates
```

This writes:
- `<day-folder>/uw_capture_final_report_YYYY-MM-DD.md`

Build ranked growth candidates + monthly rebalance markdown:

```bash
python -m uwos.build_growth_portfolio_candidates --trade-date 2026-02-07 --base-dir c:\uw_root --top-n 30 --portfolio-size 12 --max-per-sector 2 --min-score 10 --min-sources 2 --require-stock-screener --stock-lookback-days 0
```

This writes:
- `<day-folder>/growth_portfolio_candidates_YYYY-MM-DD.md`
- `<day-folder>/growth_portfolio_candidates_YYYY-MM-DD.csv`

Recommended final pass (capture, then finalize):

```bash
python -m uwos.uw_dashboard_capture --trade-date 2026-02-07 --base-dir c:\uw_root --preset growth-intel --routes stock-screener analysts insiders-trades institutions news-feed earnings flow-sectors market-statistics sec-filings correlations smart-money-live market-maps --profile-dir tokens\uw_playwright_profile_v2 --browser-channel msedge --no-headless --no-disable-automation-flags --also-scrape --scrape-scroll-cycles 20 --wait-seconds 55
python -m uwos.uw_capture_finalize --trade-date 2026-02-07 --base-dir c:\uw_root --delete-duplicates
python -m uwos.build_growth_portfolio_candidates --trade-date 2026-02-07 --base-dir c:\uw_root --top-n 30 --portfolio-size 12 --max-per-sector 2 --min-score 10 --min-sources 2 --require-stock-screener --stock-lookback-days 0
```

Single-command end-to-end pipeline (capture -> finalize -> candidates -> packet):

```bash
python -m uwos.run_uw_deep_research_pipeline --trade-date 2026-02-07 --base-dir c:\uw_root --profile-dir tokens\uw_playwright_profile_v2 --browser-channel msedge --no-headless --no-disable-automation-flags --wait-seconds 55 --scrape-scroll-cycles 20 --stock-lookback-days 7 --route-lookback-days 14 --artifact-lookback-days 14 --top-n 30 --packet-top-n 30
```

If the browser opens but login/session gets stuck:

1. Start with a fresh Playwright profile folder.
2. Bootstrap login only (no scraping):

```bash
python -m uwos.uw_dashboard_capture --manual-login-only --base-dir c:\uw_root --profile-dir tokens\uw_playwright_profile_v2 --browser-channel msedge --no-headless
```

3. After login works in that browser, run capture with the same profile/channel:

```bash
python -m uwos.uw_dashboard_capture --trade-date 2026-02-07 --base-dir c:\uw_root --preset growth-intel --profile-dir tokens\uw_playwright_profile_v2 --browser-channel msedge --no-headless
```

Install browser dependency once:

```bash
pip install playwright
python -m playwright install chromium
```

## How to use with AI (best workflow)
1) Run the script -> get `SHORTLIST_YYYY-MM-DD.md`
2) Upload/paste that into AI and ask:
   - "Audit these trades under the rulebook. Which 3 are most likely to fail and why?"
   - "Which 3 are highest quality Prime and which are Tactical?"

The script is the calculator. The AI is the risk manager.

## Customize
Edit `uwos/rulebook_config.yaml`:
- Width tiers
- SHIELD anchor whitelist
- DTE ranges
- Credit/debit gates

## Live Schwab Service
Use `uwos/schwab_auth.py` as a reusable live data layer for trading queries.

CLI wrapper:

```bash
python -m uwos.schwab_quotes --no-interactive-login --symbols-csv AAPL,MSFT,SPY --chain-symbols-csv AAPL,SPY --strike-count 8 --save-json-dir ./out/schwab
```

Reusable import:

```python
from uwos.schwab_auth import SchwabAuthConfig, SchwabLiveDataService

config = SchwabAuthConfig.from_env()
svc = SchwabLiveDataService(config=config, interactive_login=False)
snapshot = svc.snapshot(symbols=["AAPL", "SPY"], chain_symbols=["AAPL"], strike_count=8)
context = snapshot["trading_query_context"]
```

### Build Final Live Strategy Table From Shortlist
This command takes your shortlist, extracts ticker/leg symbols, fetches live option chains for those tickers, and writes a final table priced with live option quotes.

```bash
python -m uwos.pricer --shortlist-csv ./out/shortlist_trades_2026-02-05.csv --out-dir ./out/live --save-chain-dir ./out/live/chains_full
```

Outputs:
- `out/live/live_trade_table_YYYY-MM-DD.csv` (enriched, all rows)
- `out/live/live_trade_table_YYYY-MM-DD_final.csv` (only `is_final_live_valid=true`)

### Exact Spread Backtest (Ticker + Exact Legs)
Use `uwos/exact_spread_backtester.py` to replay exact spread setups (ticker, strategy, long/short OCC legs, expiry) against historical option snapshots and expiry outcomes.

Example:

```bash
python -m uwos.exact_spread_backtester \
  --setups-csv ./out/shortlist_trades_2026-02-06_mode_a.csv \
  --signal-date 2026-02-06 \
  --root-dir ./ \
  --out-dir ./out/exact_backtest_2026-02-06 \
  --entry-source auto \
  --entry-price-model conservative \
  --exit-mode quotes_then_expiry \
  --exit-price-model conservative
```

Core behavior:
- Entry pricing from historical `hot-chains` / `chain-oi-changes` snapshots (or input `entry_net`).
- Entry gate enforcement from `entry_gate` (rows failing the gate are marked skipped).
- Exit valuation via:
  - `quotes_then_expiry`: use `exit_date` leg quotes if provided, otherwise expiry intrinsic.
  - `expiry_intrinsic`: settle from underlying close on/before expiry.
- Status flags per row (`completed`, `skipped_gate_fail`, `open_not_expired`, `failed_missing_exit_price`).

Outputs:
- `trade_level_results.csv`
- `summary.json`
- `summary_by_strategy.csv`
- `summary_by_ticker_setup.csv`

Optional validation against actual trades:

```bash
python -m uwos.exact_spread_backtester \
  --setups-csv ./my_exact_setups.csv \
  --root-dir ./ \
  --out-dir ./out/exact_backtest_validate \
  --actual-trades-csv ./my_actual_trades.csv
```

Validation output files:
- `validation_matches.csv`
- `validation_summary.json`

Expected setup columns (minimum):
- `ticker`, `strategy`, `expiry`, `short_leg`, `long_leg`
- plus either `signal_date` (or `--signal-date`)
- optional: `entry_gate`, `entry_net`, `exit_date`, `exit_net`, `qty`, `trade_id`

Expected actual-trades validation columns:
- `realized_pnl`
- and either `trade_id` for direct matching, or shared keys such as
  `ticker`, `strategy`, `signal_date`, `expiry`, `short_leg`, `long_leg`

## Trade Playbook Pipeline
Use this to monitor short-term risk and long-term performance drift from your realized trade log.

Run all stages:

```bash
python -m uwos.run_trade_playbook --realized-csv ./out/trade_performance_review_manual_options_full/cleaned_realized_trades.csv --config ./uwos/rulebook_config.yaml --out-dir ./out/playbook
```

Run directly from Google Sheet CSV export URL (auto-builds `cleaned_realized_trades` each run):

```bash
python -m uwos.run_trade_playbook --sheet-csv-url "https://docs.google.com/spreadsheets/d/<SHEET_ID>/export?format=csv&gid=<GID>" --config ./uwos/rulebook_config.yaml --out-dir ./out/playbook
```

Run from Google Sheet but only ingest yellow-highlighted rows (plus month header rows for date context):

```bash
python -m uwos.run_trade_playbook --sheet-csv-url "https://docs.google.com/spreadsheets/d/<SHEET_ID>/export?format=csv&gid=<GID>" --sheet-row-filter yellow --config ./uwos/rulebook_config.yaml --out-dir ./out/playbook
```

Or run each stage separately:

```bash
python -m uwos.daily_risk_monitor --realized-csv ./out/trade_performance_review_manual_options_full/cleaned_realized_trades.csv --open-positions-csv ./out/open_positions.csv --config ./uwos/rulebook_config.yaml --out-dir ./out/playbook
python -m uwos.weekly_edge_report --realized-csv ./out/trade_performance_review_manual_options_full/cleaned_realized_trades.csv --config ./uwos/rulebook_config.yaml --out-dir ./out/playbook
python -m uwos.monthly_longitudinal_review --realized-csv ./out/trade_performance_review_manual_options_full/cleaned_realized_trades.csv --config ./uwos/rulebook_config.yaml --out-dir ./out/playbook
```

Run each stage directly from Google Sheet URL:

```bash
python -m uwos.daily_risk_monitor --sheet-csv-url "https://docs.google.com/spreadsheets/d/<SHEET_ID>/export?format=csv&gid=<GID>" --config ./uwos/rulebook_config.yaml --out-dir ./out/playbook
python -m uwos.weekly_edge_report --sheet-csv-url "https://docs.google.com/spreadsheets/d/<SHEET_ID>/export?format=csv&gid=<GID>" --config ./uwos/rulebook_config.yaml --out-dir ./out/playbook
python -m uwos.monthly_longitudinal_review --sheet-csv-url "https://docs.google.com/spreadsheets/d/<SHEET_ID>/export?format=csv&gid=<GID>" --config ./uwos/rulebook_config.yaml --out-dir ./out/playbook
```

Run each stage directly from Google Sheet URL with yellow-only filtering:

```bash
python -m uwos.daily_risk_monitor --sheet-csv-url "https://docs.google.com/spreadsheets/d/<SHEET_ID>/export?format=csv&gid=<GID>" --sheet-row-filter yellow --config ./uwos/rulebook_config.yaml --out-dir ./out/playbook
python -m uwos.weekly_edge_report --sheet-csv-url "https://docs.google.com/spreadsheets/d/<SHEET_ID>/export?format=csv&gid=<GID>" --sheet-row-filter yellow --config ./uwos/rulebook_config.yaml --out-dir ./out/playbook
python -m uwos.monthly_longitudinal_review --sheet-csv-url "https://docs.google.com/spreadsheets/d/<SHEET_ID>/export?format=csv&gid=<GID>" --sheet-row-filter yellow --config ./uwos/rulebook_config.yaml --out-dir ./out/playbook
```

Main outputs:
- `out/playbook/daily_risk_monitor.md`
- `out/playbook/weekly_edge_report.md`
- `out/playbook/monthly_longitudinal_review.md`
- `out/playbook/playbook_run_summary.md`

Dependency note:
- `--sheet-row-filter yellow` uses XLSX style parsing and requires `openpyxl` (`python -m pip install openpyxl`).

### Windows Task Scheduler (auto daily/weekly/monthly)

Register tasks (default: daily 16:45, weekly Friday 17:15, monthly day-1 18:00):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File c:\uw_root\scripts\register_playbook_tasks.ps1
```

Remove tasks:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File c:\uw_root\scripts\unregister_playbook_tasks.ps1
```



