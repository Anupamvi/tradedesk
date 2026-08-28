# CORAT operating instructions

These instructions apply only inside `/Users/anuppamvi/tradedesk/corat`.

## Identity and isolation

- CORAT is a standalone research pipeline. Do not import, call, patch, or substitute Codex Daily, `uwos`, Options-Agent, Wheel, Pattern, SolCodex, `groki-eq`, `groki`, or `groko`.
- Do not write outside the CORAT tree except for the configured read-only Schwab credential reads.
- CORAT has no broker-order authorization. Never add or call submit, cancel, replace, or account-mutation endpoints as part of a normal run.
- A report row, `TARGET TRADE`, validation smoke, backtest pass, or ledger recommendation is not a broker order or evidence of a fill. The user alone reviews and authorizes a trade.
- Market hours and Schwab availability are never trade-selection or authorization gates. Schwab is an optional data source; completed ORATS as-of prices and option chains may produce next-session target trades with visible timestamps.

## Natural-language command mapping

- `run corat YYYY-MM-DD` or `corat YYYY-MM-DD` means `python3 -m corat run --date YYYY-MM-DD`.
- `run full scan` means `python3 -m corat full-scan --date DATE` using the requested date or the New York date when omitted.
- `run delta scan` means `python3 -m corat delta-scan --date DATE`, comparing with the most recent earlier immutable CORAT run unless the user names one.
- `analyze TICKER` means `python3 -m corat analyze TICKER --date DATE`.
- `review open trades` means `python3 -m corat review-open-trades --date DATE`.
- `backtest corat` and `option replay` are evidence jobs, not normal live-planning runs.

## Required evidence posture

- `full-scan` performs discovery, automatic dated public-news enrichment, and final reranking. Merge `inputs/context/DATE.json` when analyst, primary-source, X, filing, calendar, or options-flow evidence is available.
- Prefer issuer filings/IR, regulators, and official calendars. Use reputable reporting only for information not yet available from a primary source.
- Separate `FACT`, `REPORTED INFORMATION`, and `RUMOR / X SPECULATION`; never upgrade a rumor because it fits a setup.
- X is discovery/supporting evidence only. Search multiple windows when available, downweight spam/pumps, and trace material claims to the original source.
- Preserve publication dates, event dates, URLs, credibility, and directional relevance. Never invent unavailable values.
- Missing context, approximate earnings timing, stale/mismatched quotes, small historical samples, and missing triggers must remain visible. Do not turn optional evidence or market closure into a mechanical veto. A target trade requires a present as-of trigger and positive modeled expected profit for the selected exact plan.
- The normal option search must compare every returned 21–75 DTE expiration across directional long options, debit spreads, and defined-risk credit spreads. Compare stock and options on expected return per dollar of capital at risk; never let a tight stock stop mechanically eliminate all options.

For `run full scan`, use a two-stage sequence when the final candidate set is not known in advance:

1. Let `python3 -m corat full-scan --date DATE` run the preliminary discovery pass.
2. Treat that output as discovery-only; do not hand it off as the final board.
3. Let the command research leading names and merge any supplied context using the context schema.
4. Inspect the final target tickets, modeled POP/expected profit, blockers, research evidence, and source ledger.

## Normal validation

After code changes:

1. Run `python3 -m unittest discover -s tests -v`.
2. Run `python3 -m compileall -q corat tests`.
3. Run `python3 -m corat doctor`; use `--online` only when an entitlement probe is relevant.
4. For data-path changes, run a small dated `--validation` scan and inspect the Markdown report, `run.json`, diagnostics, sources, and manifest.
5. Confirm the ORATS secret and Schwab access token do not appear in artifacts or cache keys.

Do not manufacture positive expectancy. Execution must have coherent two-sided quotes and defined loss; earnings crossing remains a hard rule. OI, volume, quote width, catalyst, sample size, score, and Schwab freshness must be shown and modeled appropriately rather than converted into arbitrary vetoes.
