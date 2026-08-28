# CORAT frozen full-pipeline replay

`full-replay` is a walk-forward research harness around the same daily CORAT decision engine used by a normal scan. It is not the older single-ticker debit-spread diagnostic.

## Safety default

Without `--execute`, the command only calculates a conservative request plan. Plan mode:

- makes zero ORATS requests;
- does not read the ORATS token;
- does not construct an ORATS client;
- writes no replay artifacts; and
- prints the expected and worst-case request counts.

Online execution is rejected unless `--request-budget`, `--monthly-reserve`, and the ORATS-console value `--confirmed-remaining` are supplied explicitly. CORAT uses the lower of the console-confirmed balance and its local counter. The planned request ceiling under the displayed trigger assumptions must fit the request cap, and the cap must fit that balance after the reserve. The ORATS client enforces the request cap and local reserve during the run. If actual triggers or historical-universe churn exceed the planning assumptions, the cap still stops additional requests and the incomplete run cannot pass the evidence gate. `--refresh` is intentionally unavailable: normal replay execution is cache-first.

Cache-only execution is available with `--execute --offline`; it has a network budget of zero and fails visibly on cache misses.

## Frozen method

For each real SPY session in the requested window:

1. Historical ORATS cores, IV rank, and summaries are fixed as of session T.
2. The historical universe is discovered for T, or the explicitly supplied ticker subset is used.
3. Price history is sliced to T, and the normal CORAT regime, sector, setup, analogue, option-structure, POP, expected-profit, vehicle, and ranking path runs without Schwab or current public-news calls.
4. Only rows labeled `TARGET TRADE` by that frozen path become replay opportunities.
5. The next session must trade the underlying entry zone. No zone touch means no fill.
6. For an option decision, the expiry and strikes selected at T remain fixed. Entry and exit require the exact legs from the historical ORATS chain. A missing leg is never reconstructed or substituted. Because the historical option quote is EOD, option exit monitoring begins with the following session; the entry-day underlying zone touch is evidence of opportunity, not proof of an intraday option fill.
7. The underlying stop, first target, or holding horizon selects the exit session. For stock, entry-session monitoring is included. When a monitored daily bar contains both stop and target, the stop is charged first because EOD data cannot reveal intraday order.
8. Option entry cashflow uses the natural price plus 50% improvement toward midpoint. Exit cashflow uses the natural price plus 25% improvement toward midpoint. Commissions are included on every leg and side.

A repeated signal is not opened while the prior replay position in the same ticker remains unresolved. That implements CORAT's no-unplanned-averaging lifecycle rather than using duplicate overlapping outcomes to inflate N. Cross-ticker and sector caps are disabled unless the user explicitly freezes `--max-open-positions` or `--max-trades-per-date`.

Long calls/puts, debit spreads, defined-risk credit spreads, and stock are all replayed when selected by the normal decision engine.

## Frozen splits and evidence

The command requires train, validation, and test boundaries. A trade whose exit crosses a boundary is labeled as an embargo row and excluded from the adjacent split's evidence. It reports:

- N, win rate, expectancy, standard error and 95% expectancy interval;
- return on maximum risk, profit factor, total P/L, and drawdown;
- POP calibration/Brier score;
- results by split, vehicle, setup, and exact strategy;
- unit economics and independently risk-sized P/L separately (not a cash-constrained account simulation unless an open-position cap is frozen);
- test-period average, median, best, worst, and positive-month rate by signal cohort, including zero-trade months; and
- every missing quote, entry miss, incomplete path, and optional user cap.

A historical test gate can pass only with adequate test N, positive lower 95% expectancy, profit factor above one, and no source errors. Historical option fills remain EOD approximations tied to an underlying zone touch, not broker-confirmed intraday fills. Even when the gate passes, `production_promotion` remains false. An unchanged prospective shadow period is required before any claim of forward profitability.

The first executed artifact marks its test interval as consumed. If rules are changed after reading test results, that interval cannot be described as untouched again; a later period or prospective shadow becomes the next independent evidence.

## Usage

Safe planning only:

```bash
python3 -m corat full-replay \
  --start 2024-01-02 \
  --end 2025-12-31 \
  --train-end 2024-12-31 \
  --validation-end 2025-06-30
```

The command above does **not** start a replay.

An eventual online run must be deliberately authorized after reviewing the printed plan:

```bash
python3 -m corat full-replay \
  --start 2024-01-02 \
  --end 2025-12-31 \
  --train-end 2024-12-31 \
  --validation-end 2025-06-30 \
  --request-budget REQUESTS_YOU_APPROVE \
  --confirmed-remaining CURRENT_ORATS_CONSOLE_BALANCE \
  --monthly-reserve REQUESTS_TO_KEEP_UNUSED \
  --execute
```

Optional `--max-trades-per-date` and `--max-open-positions` are disabled by default. They are applied only if the user chooses them. `--spacing-sessions` defaults to every session. Planning assumptions affect the budget estimate, not trade qualification.

## Artifacts after an executed run

```text
out/full_replays/START_END/RUN_ID/
  replay.md
  replay.json
  trades.csv
  missed.json
  diagnostics.json
  sources.json
  request_plan.json
  checkpoint.json
  manifest.json
```

Partial execution writes a failed checkpoint and preserves populated caches. Repeating the same frozen policy is cache-first, so completed requests are not intentionally repurchased.
