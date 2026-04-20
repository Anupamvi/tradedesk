---
name: trend-analysis
description: Use when the user says "trend-analysis", "historical UW trends", "trend analysis 2026-04-17", or asks to scan dated UW folders for backtest-gated option trade ideas.
---

# Trend Analysis

Run the dated-folder UW trend pipeline, not the old replay-manifest-only historical pipeline.

## Parameters

- **date**: As-of date, `YYYY-MM-DD`. Defaults to latest available dated folder.
- **lookback**: Usable market-data-day lookback. Defaults to `90`.
- **top**: Maximum actionable trades. Defaults to `15`.
- **no-schwab**: Skip Schwab live validation only if requested or auth fails.
- **no-backtest**: Skip backtesting only if explicitly requested.
- **quote-replay**: Defaults to `gate`. Uses local daily UW option snapshots to replay both spread legs before any trade is actionable.
- **no-regime-filter**: Skip broad market regime compatibility checks only when explicitly requested.
- **position-json**: Optional trade-desk `position_data_*.json`; defaults to the latest `out/trade_analysis/position_data_*.json`.
- **no-position-check**: Skip open-position duplicate/conflict checks only when explicitly requested.
- **trade-tracker**: Optional post-trade tracking CSV. Defaults to `out/trend_analysis/trend-analysis-trade-tracker.csv`.
- **no-trade-tracking**: Skip appending actionable trades to the tracker.
- **no-outcome-update**: Skip refreshing tracked trade outcomes from local UW option snapshots.
- **reuse-walk-forward-raw / reuse-walk-forward-outcomes / reuse-research-outcomes**: Reuse completed historical audit files when rerunning a large walk-forward audit in the same output directory.
- **batch proof**: Use `python3 -m uwos.trend_analysis_batch --start YYYY-MM-DD --end YYYY-MM-DD --lookback N --horizons 20 --reuse-raw` when the user asks whether the engine is profitable, proven, a money printer, or wants historical replay across all dated folders.

## Execution

```bash
cd /Users/anuppamvi/uw_root/tradedesk
python3 -m uwos.trend_analysis {date} {lookback}
```

Examples:

```bash
python3 -m uwos.trend_analysis 2026-04-17
python3 -m uwos.trend_analysis 2026-04-17 90
python3 -m uwos.trend_analysis 2026-04-17 90 --top 20
```

Batch proof example:

```bash
python3 -m uwos.trend_analysis_batch --start 2025-12-01 --end 2026-04-17 --lookback 30 --horizons 20 --reuse-raw
```

Use local UW option quote replay for batch proof. Do not use Schwab current/live chains to score old signal dates because that would introduce look-ahead leakage.

## Output

Read and summarize:

- `/Users/anuppamvi/uw_root/tradedesk/out/trend_analysis/trend-analysis-{date}-L{lookback}.md`
- `/Users/anuppamvi/uw_root/tradedesk/out/trend_analysis/trend-analysis-candidates-{date}-L{lookback}.csv`
- `/Users/anuppamvi/uw_root/tradedesk/out/trend_analysis/trend-analysis-current-setups-{date}-L{lookback}.csv`
- `/Users/anuppamvi/uw_root/tradedesk/out/trend_analysis/trend-analysis-actionable-{date}-L{lookback}.csv`
- `/Users/anuppamvi/uw_root/tradedesk/out/trend_analysis/trend-analysis-patterns-{date}-L{lookback}.csv`
- `/Users/anuppamvi/uw_root/tradedesk/out/trend_analysis/trend-analysis-quote-replay-{date}-L{lookback}.csv`
- `/Users/anuppamvi/uw_root/tradedesk/out/trend_analysis/trend-analysis-walk-forward-{date}-L{lookback}.csv`
- `/Users/anuppamvi/uw_root/tradedesk/out/trend_analysis/trend-analysis-research-audit-{date}-L{lookback}.csv`
- `/Users/anuppamvi/uw_root/tradedesk/out/trend_analysis/trend-analysis-research-audit-by-horizon-{date}-L{lookback}.csv`
- `/Users/anuppamvi/uw_root/tradedesk/out/trend_analysis/trend-analysis-strategy-family-audit-{date}-L{lookback}.csv`
- `/Users/anuppamvi/uw_root/tradedesk/out/trend_analysis/trend-analysis-ticker-playbook-audit-{date}-L{lookback}.csv`
- `/Users/anuppamvi/uw_root/tradedesk/out/trend_analysis/trend-analysis-rolling-ticker-playbook-audit-{date}-L{lookback}.csv`
- `/Users/anuppamvi/uw_root/tradedesk/out/trend_analysis/trend-analysis-research-outcomes-{date}-L{lookback}.csv`
- `/Users/anuppamvi/uw_root/tradedesk/out/trend_analysis/trend-analysis-trade-tracker.csv` with outcome fields refreshed on each run unless `--no-outcome-update` is used
- `/Users/anuppamvi/uw_root/tradedesk/out/trend_analysis/trend-analysis-metadata-{date}-L{lookback}.json`
- `/Users/anuppamvi/uw_root/tradedesk/out/trend_analysis_batch/trend-analysis-batch-proof-START_END-L{lookback}.md`
- `/Users/anuppamvi/uw_root/tradedesk/out/trend_analysis_batch/trend-analysis-batch-strict-trades-START_END-L{lookback}.csv`
- `/Users/anuppamvi/uw_root/tradedesk/out/trend_analysis_batch/trend-analysis-batch-research-outcomes-START_END-L{lookback}.csv`

## Reporting Rules

- Present **Backtest-Supported Candidate Shortlist** first, then **Actionable Trades**, then **Max Conviction / Max Planned Risk**, then **Trade Workup**.
- Present **Current Trade Setups** as the practical workbench: `TRADE_SETUP` means research/conditional entry only; `REBUILD` means supported thesis but the spread needs better liquidity, flow, expiry, or strikes before entry.
- Review **Walk-Forward Audit** before trusting the trade list. It replays older signal dates selected without future P&L leakage, then scores future option-quote outcomes.
- Review **Research Confidence Audit** and **Research Horizon Audit** next. They are fixed-bucket sanity checks, not parameter searches; if no bucket/horizon is supportive, do not treat current candidates as Actionable Now. Use the research outcomes CSV when debugging the ticker/setup rows behind a negative bucket.
- Review **Strategy Family Audit** next. This is the broad trade-generation layer: predeclared setup families are split into training and later validation dates. Mixed, negative, validation-negative, and low-sample families are research-only unless a narrower ticker playbook overrides them.
- Review **Ticker Playbook Audit** next. This is the narrow trade-generation layer: ticker/direction/strategy/horizon playbooks are split into training and later validation dates. A `promotable` ticker playbook can unlock Actionable Now even when the broad family is negative, but all live quote, Schwab, earnings, liquidity, trend-quality, and max-risk gates still apply.
- Review **Rolling Ticker Playbook Forward Validation** next. It checks whether a ticker playbook continued to work after it first became promotable using prior data only. Negative rolling validation blocks matching actionable trades; insufficient validation is allowed only with lower position-size tiers.
- If a batch proof was requested, read the batch proof report first and answer the proof question before discussing individual candidates.
- In every trade summary, show the trade setup up front: strategy, legs, and expiry.
- Backtest-Supported Candidate Shortlist is the primary output: a few strong bullish/bearish candidates or patterns with at least one backtest-supported structure to work on. It is acceptable for a run to produce zero actionable trades.
- Candidate rows require high swing score, multiple independent confirmations, few contradictory signals, and at least one structure that passed the backtest gate. Directional price confirmation is preferred; if price diverges, label it clearly and do not treat it as an entry until price starts confirming.
- Treat Actionable Trades as trade ideas only when historical support is positive: either the backtest gate passed with enough signals, or a LOW_SAMPLE/PASS-shortfall row has a matching `promotable` ticker playbook. The matching broad strategy family or narrow ticker playbook must be promotable.
- Also require swing score >= `60.0`, enough historical support, daily option quote replay `PASS`, `PARTIAL_PASS`, or current-day `ENTRY_OK`, Schwab live validation, no earnings through expiry, acceptable bid/ask width, short delta <= `0.30`, no volatile-GEX iron condors, compatible market regime, no same-underlying open option exposure from trade-desk, and no negative rolling ticker-playbook validation. Enough support means `100` historical analog signals for ordinary PASS rows, or a matching promotable ticker playbook for LOW_SAMPLE/PASS-shortfall rows.
- Also require the professional-quality gate: underlying price >= `$20`, debit spread price >= `$0.75`, strong whale/institutional coverage scaled to the lookback and capped at `8` mention days, no bullish/bearish trade whose direction conflicts with options flow, directional debit price/flow scores >= `60`, debit <= `50%` of spread width, and long strike no more than `2%` OTM. Do not override this unless the user explicitly asks for speculative/lotto ideas.
- Max Conviction / Max Planned Risk rows are the highest tier. They must already be Actionable Now, then also require strong score, edge, backtest sample support, whale coverage, tight liquidity, and price/flow/OI/dark-pool alignment. Present this as max pre-defined risk for one defined-risk trade, never as all account capital.
- Daily option quote replay prices both legs from local `hot-chains` / `chain-oi-changes` snapshots on the signal date and a later option snapshot. Missing entry/exit leg quotes block actionability by default; use `--quote-replay diagnostic` only when the user asks to see what would have passed without this gate. For the latest available data date, no later snapshot exists yet, so both-leg entry quote coverage is reported as `ENTRY_OK`.
- The pipeline repairs strong blocked candidates before the final gate by trying pre-earnings expiries, more liquid debit structures, farther-OTM credit spreads, and safer condor/directional alternatives.
- Repaired variants are still actionable only if the full live/backtest/tradeability gate passes.
- Trade Workup rows are quality setups, not order tickets. They pass the professional/live/quote/research-support gates and have positive LOW_SAMPLE or sample-shortfall evidence, but they do not have enough historical support for Actionable Now. Summarize them as candidates to work on with explicit next-step conditions. Do not call family-negative rows Trade Workups unless a ticker playbook is promotable.
- Walk-forward audit rows are the confidence check. Treat a supportive audit as evidence that the pipeline is improving; treat empty, low-sample, or negative audit results as a confidence penalty. The audit must select historical trades using signal-date evidence only; future quotes are for scoring outcomes, not entry gating.
- Research Confidence Audit rows show whether fixed historical candidate buckets made money. The summary requires enough unique historical setups; repeated 5/10/20-day exits cannot masquerade as independent evidence.
- Research Horizon Audit rows split the same buckets by holding horizon. If the aggregate table and the horizon table are negative, keep Actionable Now blocked even if a current setup looks interesting.
- Strategy Family Audit rows are train/validation checks for predeclared setup families. A family with validation wins but negative training is `mixed`, not promotable.
- Ticker Playbook Audit rows are train/validation checks for ticker-specific setup behavior. A promotable ticker playbook can override a broad family block, but it must never override live quote, Schwab, earnings, liquidity, price/flow quality, market regime, open-position awareness, negative rolling-forward validation, or lotto gates.
- Every Actionable Trade must show a position-size tier. `PROBE_ONLY`, `STARTER_RISK`, `STANDARD_RISK`, and `MAX_PLANNED_RISK` are caps; low-sample or insufficient-forward-validation rows should remain starter/probe-sized.
- Post-trade tracking should show whether prior tracked ideas are `OPEN_WIN`, `OPEN_LOSS`, `CLOSED_WIN`, `CLOSED_LOSS`, or unavailable based on local UW option replay. Do not call the pipeline profitable until tracker outcomes support that claim.
- Present rows that pass backtest/Schwab but fail tradeability gates as **Risk-Blocked**, not actionable.
- Pattern Candidates are not trades; summarize them separately.
- If there are no actionable trades, say so clearly.
- Include absolute file links.
