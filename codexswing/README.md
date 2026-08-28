# CodexSwing

CodexSwing is an ORATS-first, Schwab-verified stock and options swing-trade
decision-support pipeline. It searches broadly, tests exact option structures
on historical ORATS chains, reprices exact current contracts through Schwab,
checks the current portfolio, and stops before broker submission.

It does not promise profitability. A high current model POP is not enough. A
trade becomes `MANUAL_READY` only when its same-structure historical holdout,
current contract, and current Schwab portfolio all pass.

## Active data contract

- ORATS delayed `cores`: complete optionable-universe discovery, liquidity,
  current IV, future realized volatility (`orFcst20d`), and future implied
  volatility (`orIvFcst20d`). Those two forecasts are never interchanged.
- ORATS `hist/dailies`: split-adjusted stock signal and outcome history.
- ORATS `hist/ivrank`: as-of volatility-regime features.
- ORATS delayed `strikes`: current theoretical values and surface context.
- ORATS `hist/strikes`: exact historical option bids, asks, strikes, deltas,
  volume, and open interest for replay.
- Schwab quotes/chains: current stock truth and exact tradable option symbols,
  bids, asks, Greeks, volume, and open interest.
- Schwab balances/positions/working orders: buying-power, concentration, and
  conflict gates.
- Public news/geopolitical context: GDELT primary with a time-windowed Google
  News RSS fallback. Source-cited shadow context only; it has no numeric vote
  until a time-aligned ablation proves incremental net value.

## Promotion states

1. `DISCOVERED` — liquid broad-universe candidate.
2. `BACKTEST_PASS` — same fixed structure passes chronological validation and
   untouched holdout after package slippage and commissions.
3. `CURRENT_CONTRACT_PASS` — exact Schwab contract has positive modeled EV,
   fresh quotes, defined risk, and acceptable liquidity/spreads.
4. `PORTFOLIO_PASS` — current Schwab buying power, concentration, positions,
   and working orders permit the risk.
5. `TACTICAL_READY` — positive train/validation/holdout economics but a still
   overlapping uncertainty interval; exactly one contract, capped at 0.05% NAV.
6. `MANUAL_READY` — all full-evidence gates passed; the user may evaluate and
   manually submit. CodexSwing still has no order-placement method.

## Inactive v0.5 research lane

An isolated v0.5 package now implements cache-only regime matching,
3/5/10/20-session exact-chain paths, fixed target/stop variants, earnings and
dividend-assignment exclusions, full-family Holm correction, and a prospective
hash-chain shadow ledger. It is `IMPLEMENTED_NOT_EXECUTED`: no replay was run,
no API request was authorized, and it has no validated POP or profit claim.

The v0.5 spec reserves the full user-reported 12,000 remaining ORATS requests.
Its CLI can describe the frozen design, compare local cache coverage, and
verify a local ledger; it cannot fetch data. See [V5_RUNBOOK.md](V5_RUNBOOK.md).

## Run

From `/Users/anuppamvi/tradedesk/codexswing`:

```bash
PYTHONPATH=src python3 -m codexswing doctor --online

PYTHONPATH=src python3 -m codexswing discover-universe \
  --limit 250 \
  --write

PYTHONPATH=src python3 -m codexswing run-daily \
  --date 2026-08-27 \
  --discovery-limit 250 \
  --finalists 10 \
  --backtest-top 6 \
  --backtest-workers 6
```

The run writes immutable, content-addressed source batches, exact-chain
backtests, JSON/HTML reports, and a source/code manifest under:

`/Users/anuppamvi/tradedesk/out/codexswing`

Completed historical chain slices are cached and reused. ORATS 404/empty slices
are stored for 24 hours as explicit no-chain rejection evidence, so one missing
ticker/date cannot abort the replay or be mistaken for a successful trade.

## Report semantics

- Estimated profitable POP: same-structure untouched-holdout POP shrunk toward
  50% according to effective non-overlapping sample size.
- Historical POP: raw exact-chain holdout win rate after entry slippage, natural
  exit, and commissions.
- Current modeled POP: scenario-model probability for today's exact contract;
  useful for comparison but not calibrated and never a substitute for holdout.
- Confidence: evidence-strength label and 0–100 score, capped below 50 if the
  historical gate fails.
- `ACTIONABLE_CANDIDATES`: at least one full-evidence or tightly capped tactical
  candidate has exact manual execution conditions.

## Verification

```bash
python3 -m pytest -q
PYTHONPATH=src python3 -m codexswing audit-store
```

The Schwab client exposes GET requests only. OAuth refresh is isolated in the
credential module; there is no submit/cancel/replace order surface.
