# Sentiment Pipeline

The sentiment pipeline scores a ticker, segment, or catalyst query from four layers:

1. Local UW positioning: stock screener premium skew, hot-chain side bias, chain OI changes, dark-pool activity.
2. Text artifacts: X profile scrape CSVs, Reddit/news CSVs, browser text captures, UW news/feed exports, and optional Schwab broker-news pulls.
3. Macro/catalyst context: market regime plus deterministic theme maps such as Middle East war / energy / defense.
4. Trade readiness: existing `uwos.trend_analysis` artifacts. Exact spread tickets are shown only when the trend gate already produced them.

## Commands

Ticker:

```bash
python3 -m uwos.sentiment_pipeline NFLX --as-of 2026-04-17 --lookback 30 --top 15
```

Catalyst or segment:

```bash
python3 -m uwos.sentiment_pipeline "Iran war" --as-of 2026-04-17 --lookback 30 --top 20
```

The command attempts a best-effort Schwab news pull by default when Schwab credentials are available. Use `--no-schwab-news` for fully offline historical replays.

Outputs are written under:

```text
out/sentiment_pipeline/
```

Each run writes a Markdown report, a machine-readable scores CSV, and metadata JSON.

By default, trade artifacts are proof-gated. The report can still show an unproven trend artifact for context, but it will be marked `BATCH_BLOCKED` unless the latest `trend_analysis_batch` rolling ticker playbook audit has an exact supportive ticker / direction / strategy match.

## Browser Capture

Use this when you want logged-in X/Reddit/news search text to feed the text layer:

```bash
python3 -m uwos.browser_text_capture "NFLX stock options" --run-date 2026-04-17 --sources x reddit news schwab --engine applescript
python3 -m uwos.sentiment_pipeline NFLX --as-of 2026-04-17 --lookback 30 --top 15 --run-trend-analysis
```

The `applescript` engine drives the already logged-in Google Chrome app. It only opens search/research pages, scrolls, and saves visible text/screenshots into the dated folder; it does not post, like, reply, subscribe, or send messages. A Playwright engine remains available for dedicated browser profiles, but it is optional and not required for the normal logged-in Chrome workflow.

The `schwab` browser source opens the logged-in Schwab research news route for the first ticker in the query, for example `client.schwab.com/app/research/#/stocks/NFLX/news`. If Schwab shows a login or entitlement page, the captured text will remain low-value and the metadata will make that visible.

Schwab news can also be fetched directly:

```bash
python3 -m uwos.schwab_news NFLX TSLA --run-date 2026-04-17
```

Schwab Trader API news is best-effort. The currently documented retail API surface exposes quotes, option chains, price history, movers, market hours, instruments, accounts, orders, and transactions; it does not expose a stable documented retail news endpoint. The code records unsupported/404 responses in metadata instead of pretending news was fetched. Use browser capture for Schwab research/news text.

## Trade Workups

Use `--run-trend-analysis` when you want the sentiment run to refresh trade artifacts first and then attach any Actionable, Max Conviction, Trade Workup, or Current Setup rows to sentiment-ranked tickers:

```bash
python3 -m uwos.sentiment_pipeline "AI semis capex" --as-of 2026-04-17 --lookback 30 --top 25 --run-trend-analysis
```

If Schwab live option validation is unavailable but you still want trend workups from local UW data:

```bash
python3 -m uwos.sentiment_pipeline "AI semis capex" --as-of 2026-04-17 --lookback 30 --top 25 --run-trend-analysis --trend-no-schwab
```

## Batch Proof Gate

Run or refresh the historical batch proof before trusting live trade rows:

```bash
python3 -m uwos.trend_analysis_batch --start 2025-12-01 --end 2026-04-17 --lookback 30 --horizons 20 --reuse-raw
```

Then run sentiment with the same lookback:

```bash
python3 -m uwos.sentiment_pipeline NFLX --as-of 2026-04-17 --lookback 30 --top 15 --run-trend-analysis --trend-no-schwab
```

The default gate reads `out/trend_analysis_batch/trend-analysis-batch-rolling-ticker-playbook-audit-...csv`. Only rows with `verdict=supportive` become proof-supported trade considerations. Negative, low-sample, insufficient-forward, missing, or broad policy-only rows remain research leads and are marked `BATCH_BLOCKED`.

Use this only for debugging or historical comparisons:

```bash
python3 -m uwos.sentiment_pipeline NFLX --as-of 2026-04-17 --lookback 30 --top 15 --no-batch-proof-gate
```

## Reading Results

Use `sentiment_score` as directional pressure, not as a standalone trade signal:

- `>= 20`: bullish pressure
- `<= -20`: bearish pressure
- otherwise: mixed/no edge

Use `confidence` to judge source agreement and coverage. X, Reddit, and Schwab broker-news text are intentionally weighted more heavily than generic local text, because they carry most of the real sentiment edge. A high sentiment score with `NO_TREND_TRADE` is a research lead. A row with `PROOF_SUPPORTED` has passed the historical batch proof gate and is linked to the existing backtest/live trend-analysis layer; it still must be re-priced before any order. A row with `BATCH_BLOCKED` is explicitly not a trade recommendation.
