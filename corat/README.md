# CORAT

CORAT is an isolated, evidence-first stock and options swing-research pipeline. It starts with the underlying, classifies the market and sector backdrop, detects frozen setups, measures same-ticker historical analogues, compares stock exposure with an exact ORATS-defined option structure, and writes an immutable research board.

CORAT is **research-only and manual-only**. It has no order-submission, cancellation, replacement, or broker-account mutation surface. `TARGET TRADE` means the completed as-of data contains a triggered setup and positive modeled expected profit for the selected exact plan. The user alone reviews and authorizes any order.

## What is implemented

- Dynamic discovery from the complete ORATS core universe, with configured benchmarks/theme ETFs preserved and up to 500 qualifying equities selected across market-cap, stock-volume, and option-volume ranks.
- New York trading-date routing with explicit as-of cutoffs.
- Cache-first ORATS ingestion for historical daily prices, cores, IV rank, summaries, strikes, and historical strikes.
- Optional **read-only** Schwab quotes as an additional price observation. Market hours and Schwab availability never authorize or block a target trade; CORAT reads the shared token in place and never refreshes or writes it.
- Market-regime, breadth, cross-asset proxy, and sector-rotation analysis.
- EMA20, SMA50, SMA200, ATR, relative volume, support/resistance, relative strength, and event-based AVWAPs.
- Seven underlying-first setup detectors.
- Leakage-safe, non-overlapping price analogues with 1/3/5/10/20-session outcomes, intraday path-aware stop/target exits, post-earnings-event matching, IV/HV similarity, modeled POP, expected profit, MAE, MFE, profit factor, and drawdown.
- Exact 21–75 DTE search across every returned expiration for directional long calls/puts, bull-call/bear-put debit spreads, and defined-risk bull-put/bear-call credit spreads. The exact contract is selected on older training paths and evaluated on untouched recent holdout paths, with quoted entry/exit friction, commissions, natural-fill and IV stresses, Greeks, theoretical-value comparison, and maximum loss.
- Stock-versus-options selection compares expected return on stock capital required with expected return on the option structure's defined maximum loss; a tight stock stop can no longer mechanically make every option look inferior.
- Exact historical option replay using the same expiry and strikes at entry and exit; a missing leg is not reconstructed.
- Frozen full-pipeline walk-forward replay across historical daily CORAT decisions, all selected stock/long/debit/defined-risk-credit vehicles, next-session entry-zone fills, exact-contract exits, train/validation/test embargoes, POP calibration, and unit-versus-sized P/L. It is plan-only by default and quota-guarded when explicitly executed.
- Automatic two-stage full scans: quantitative discovery, dated public-news enrichment of the leaders, then final reranking. User-supplied primary-source context is preserved and merged.
- Source-bound catalyst, event, X, and options-flow context with fact/report/rumor separation and anti-pump controls. Missing catalyst evidence is disclosed rather than turned into a mechanical veto.
- Risk-based sizing, correlation warnings, setup-versus-trigger separation, and positive-expectancy target selection.
- Immutable reports, boards, manifests, source traces, hashes, and an append-only recommendation/outcome ledger.

The [requirements matrix](docs/REQUIREMENTS_MATRIX.md) distinguishes fully implemented controls from intentionally constrained or external evidence steps.

## Quick start

CORAT has no third-party Python dependencies and requires Python 3.9 or later.

```bash
cd /Users/anuppamvi/tradedesk/corat
python3 -m corat doctor --online
python3 -m unittest discover -s tests -v
```

The ORATS credential is stored in `.env`, copied from the user-specified source with mode `600`, and ignored by Git. Never paste it into a command, report, issue, or chat.

Run a focused validation:

```bash
python3 -m corat run \
  --date 2026-08-27 \
  --tickers NVDA,AAPL,MSFT \
  --context inputs/context/2026-08-27.validation.json \
  --validation \
  --max-requests 24
```

Run the configured universe:

```bash
python3 -m corat full-scan \
  --date YYYY-MM-DD \
  --portfolio-nav 500000
```

A full scan now performs the two stages itself: it writes a discovery-only run, researches the leading names through dated public-news RSS metadata, writes the sourced context, and reruns the final board. Supply `--context inputs/context/YYYY-MM-DD.json` to merge analyst/primary-source evidence. Use `--no-auto-research` only for a deliberate single-pass run, and `--offline` for cache-only work.

If the optional Schwab access token is expired or intentionally unavailable, add `--no-schwab`. The run still identifies target trades from completed ORATS as-of data. The report keeps the quote timestamp visible so the user can review the limit before manual submission.

## Evidence context

The local Python process automatically collects dated public-news headline metadata for discovery leaders. It does not pretend to have authenticated X access or infer facts beyond a headline. Primary-source, X, filing, calendar, or analyst evidence can also enter through a validated `corat.context.v1` file. Generate a template with:

```bash
python3 -m corat context-template \
  --date YYYY-MM-DD \
  --tickers NVDA,AAPL,MSFT
```

Populate it using primary sources first, then reputable reporting where needed. Every row requires a source name, URL, publication date, credibility, classification, title/claim, and direction. Rumor/X-only evidence cannot masquerade as a fact, and a bearish fact cannot boost a bullish setup. See [the context contract](docs/CONTEXT_SCHEMA.md), [JSON Schema](schemas/context.schema.json), and [research prompt](prompts/context_research.md).

Without a valid source, affected fields render as `DATA UNAVAILABLE`. Missing context reduces the ranking and is disclosed; it does not mechanically erase an otherwise triggered, positive-expectancy trade.

## Commands

Analyze one security through the full framework:

```bash
python3 -m corat analyze NVDA --date YYYY-MM-DD --context inputs/context/YYYY-MM-DD.json
```

Compare a new immutable run with a prior run:

```bash
python3 -m corat delta-scan \
  --date YYYY-MM-DD \
  --previous YYYY-MM-DD \
  --context inputs/context/YYYY-MM-DD.json
```

Run frozen price-setup train/test diagnostics:

```bash
python3 -m corat backtest \
  --date YYYY-MM-DD \
  --split-date YYYY-MM-DD \
  --tickers NVDA,AAPL,MSFT
```

Replay exact historical ORATS debit spreads:

```bash
python3 -m corat option-replay NVDA \
  --setup "RELATIVE-STRENGTH LEADER" \
  --direction BULLISH \
  --start YYYY-MM-DD \
  --end YYYY-MM-DD \
  --split-date YYYY-MM-DD \
  --holding-sessions 10 \
  --max-signals 20
```

The backtest and replay commands produce evidence only. They do not promote rules or strategies into production.

Plan the comprehensive replay without starting it or using ORATS quota:

```bash
python3 -m corat full-replay \
  --start 2024-01-02 \
  --end 2025-12-31 \
  --train-end 2024-12-31 \
  --validation-end 2025-06-30
```

`full-replay` is deliberately plan-only unless `--execute` is present. Online execution also requires an explicit hard `--request-budget`, the current ORATS-console `--confirmed-remaining`, and an explicit `--monthly-reserve`; the planning ceiling under the displayed trigger assumptions must fit the lower of the confirmed and locally recorded balance before CORAT reads the token or constructs an API client. The request budget remains the hard stop if actual demand is higher. There is no refresh flag. See [the full replay contract](docs/FULL_REPLAY.md).

## Recommendation and outcome ledger

Copy a candidate from an immutable run into the append-only ledger:

```bash
python3 -m corat record-plan \
  --run /absolute/path/to/run.json \
  --ticker NVDA
```

Optional predefined scaling must exist before entry. Example `scaling.json`:

```json
{
  "enabled": true,
  "add_entry_low": 98.0,
  "add_entry_high": 102.0,
  "add_quantity": 5,
  "maximum_total_quantity": 20
}
```

Pass it with `record-plan --scaling-plan scaling.json`. Subsequent `record-event --quantity` values represent the current total position quantity. A review emits `ADD` only when the original zone/cap, current quantity, aligned thesis, and every current actionability gate pass.

Record external execution evidence explicitly:

```bash
python3 -m corat record-event TRADE_ID SUBMITTED --price 123.45 --quantity 10
python3 -m corat record-event TRADE_ID FILLED --price 123.50 --quantity 10
python3 -m corat record-event TRADE_ID OPEN --review-horizon-sessions 5
python3 -m corat record-event TRADE_ID CLOSED --price 130.00 --realized-pnl 650 --reason "Target reached"
python3 -m corat ledger-report
```

Review open trades against the original thesis:

```bash
python3 -m corat review-open-trades --date YYYY-MM-DD
```

`RECOMMENDED`, `SUBMITTED`, `FILLED`, `OPEN`, and `CLOSED` are deliberately distinct states. CORAT never infers a fill from a recommendation.

## Artifacts

Each scan writes a new immutable directory:

```text
out/YYYY-MM-DD/RUN_ID/
  corat-YYYY-MM-DD.md   human research board
  board.csv             compact ranked board
  run.json              complete structured result
  diagnostics.json      scan funnel
  candidate-audit.json  every scanned name, option coverage, economics, and disposition
  sources.json          source/cache/error ledger
  manifest.json         hashes and safety assertions
```

`out/YYYY-MM-DD/latest.json` is the only mutable run pointer. Cache and state directories live under `var/` and are ignored by Git.

## Important limits

- The current structure router promotes exact directional long options, vertical debit spreads, and defined-risk vertical credit spreads. Calendars, diagonals, and event-volatility structures remain outside normal promotion until they have dedicated construction and replay evidence.
- Historical post-earnings-drift matching uses dated ORATS earnings events, the earnings gap, earnings AVWAP behavior, and the final trigger. Forward ordinary-option eligibility uses a dated earnings calendar and remains fail-closed when timing cannot be established.
- Sector ranking uses price, relative strength, and volume. Revisions, fund flows, and institutional-participation evidence must arrive through a sourced context extension before they influence ranking.
- VIX, duration, dollar, credit, and gold are represented by configured traded proxies; these are not substitutes for a full macro-data terminal.
- Dealer gamma direction is shown as `DATA UNAVAILABLE`; CORAT will not infer dealer inventory from open interest alone.
- Full replay discovers from each historical ORATS core snapshot, but without a separate dated security master/index-membership history it is still not guaranteed survivorship-bias-free. Do not treat it as institutional research proof.
- No profitability, monthly-income, production-readiness, or win-probability claim follows from a successful run or passing tests.

## ORATS references

CORAT uses documented ORATS Data API v2 routes and fields. See the official [API guide](https://docs.orats.io/datav2-api-guide/), [data endpoints](https://docs.orats.io/datav2-api-guide/data.html), and [field definitions](https://docs.orats.io/datav2-api-guide/definitions.html).
