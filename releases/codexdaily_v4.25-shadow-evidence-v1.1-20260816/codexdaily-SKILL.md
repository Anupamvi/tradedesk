---
name: codexdaily
description: Run the latest Codex Daily V4 full-universe Schwab-backed options decision engine for a dated Unusual Whales folder. Use for "codexdaily YYYY-MM-DD", "run codex daily", "Codex V4", daily trade generation, V4 overlays, and V4 validation or replay requests.
---

# Codex Daily V4

Run the professional V4 pipeline from `/Users/anuppamvi/uw_root/tradedesk`.

## Routing rules

- Use only `python3 -m codexuw.daily_v4` for Codex Daily requests.
- Never substitute `python3 -m codexuw.daily`, the old `uwos` daily pipeline, or trend analysis.
- A normal dated request is live planning from that date's EOD UW files with current Schwab validation.
- Use `validate` or historical report mode only when the user explicitly requests replay, backtesting, or validation.
- Scan the full eligible universe: `--max-tickers 0 --max-candidates 0`.
- Do not impose an aggregate `$15k` slate budget: use `--risk-budget 0`. Per-ticket and portfolio safety limits still apply.
- Use Schwab for executable prices, option chains, GEX, and portfolio exposure. Do not replace Schwab with Alpaca or another quote source.
- Never force a trade. Distinguish immediate `Execute` from a valid `Work Limit` and from unvalidated Scouts.

## Standard run

For `DATE`:

```bash
python3 -m codexuw.daily_v4 run \
  --root /Users/anuppamvi/uw_root/tradedesk \
  --date DATE \
  --out-dir /Users/anuppamvi/uw_root/tradedesk/out/codexdaily_v4_DATE \
  --max-tickers 0 \
  --max-candidates 0 \
  --risk-budget 0 \
  --monthly-profit-target 10000 \
  --max-contracts-per-trade 20 \
  --minimum-expected-value-per-dollar-risk 0.01 \
  --risk-mandate target-growth \
  --index-income-mode primary \
  --portfolio-income-mode trading-sleeve-only
```

Use a suffixed output directory for reruns. For deterministic code comparisons, add `--schwab-snapshot-dir` pointing to the original V4 Schwab snapshot. Do not call a deterministic snapshot rerun a live quote refresh.

## Point-in-time range/GEX shadow follow-up

The first pass writes `codexdaily_v4_range_gex_collection_universe_DATE.csv`. If the corresponding range/GEX status is `MISSING_POINT_IN_TIME_GEX` and that universe is non-empty, use the existing logged-in UW collector for only those prequalified tickers, then refresh the shadow artifacts without rerunning or changing the production decision book:

```bash
TICKERS=$(python3 -c 'import pandas as pd,sys; print(",".join(pd.read_csv(sys.argv[1])["ticker"].astype(str)))' \
  /Users/anuppamvi/uw_root/tradedesk/out/codexdaily_v4_DATE/codexdaily_v4_range_gex_collection_universe_DATE.csv)

python3 -m uwos.collect_uw_enrichments_mac \
  --mode collect-gex \
  --date DATE \
  --repo-root /Users/anuppamvi/uw_root/tradedesk \
  --tickers "$TICKERS" \
  --max-tickers 25

python3 -m codexuw.daily_shadow_books \
  --root /Users/anuppamvi/uw_root/tradedesk \
  --date DATE \
  --scored /Users/anuppamvi/uw_root/tradedesk/out/codexdaily_v4_DATE/codexdaily_v4_scored_reference_DATE.csv \
  --out-dir /Users/anuppamvi/uw_root/tradedesk/out/codexdaily_v4_DATE
```

This follow-up is research-only. Browser or GEX collection failure must stay visible as missing point-in-time evidence and must never change an `Execute`, `Work Limit`, or rejection decision.

## Inputs

The dated folder should contain the available EOD UW exports, normally stock screener, hot chains, chain OI changes, and bot EOD report files. Prefer clearly named `latest`, `current`, `live`, or `next` OI overlays when the user explicitly requests an overlay.

Before a live recommendation, verify earnings/catalysts, macro context, stale-data status, portfolio overlap, and fresh Schwab pricing. Missing evidence must remain visible as a blocker; it must not silently remove the ticker.

## Required artifacts

Read the V4 artifacts only:

- `codexdaily_v4_report_DATE.md`
- `codexdaily_v4_manifest_DATE.json`
- `codexdaily_v4_scored_reference_DATE.csv`
- `codexdaily_v4_swing_target_tickets_DATE.csv`
- `codexdaily_v4_candidate_disposition_DATE.csv`
- `codexdaily_v4_no_miss_audit_DATE.csv`

Confirm the manifest says `pipeline_name: Codex Daily V4` and reports the expected current version. Treat a V2 manifest or `codexuw_*` artifact namespace as the wrong pipeline.

## Response contract

Put the action board first. For each actionable or target-only setup show:

- status and confidence lane
- ticker, strategy, explicit sell leg, explicit buy leg, expiry, and DTE
- current Schwab mid and executable natural price
- exact entry credit/debit target
- expected win rate and evidence sample
- maximum profit, maximum loss, reward/risk, and one-contract risk
- flow, exact-leg OI, GEX, earnings, portfolio, and liquidity context
- precise blocker when it is not executable now

Use:

- `🟣 ENTER NOW`: validated and executable at the current natural price
- `🟢 WORK LIMIT`: validated setup; enter only at the displayed limit or better
- `🟡 REVIEW`: useful candidate without production execution authority
- `🔴 REJECT`: hard blocker or negative evidence

Never label a review candidate as a trade. Never present a model confidence score as POP. State plainly when no immediate trade passes.
