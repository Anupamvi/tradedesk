# Anu Options Engine — EXECUTION PROTOCOL (Audited FIRE + SHIELD No-GEX Payload)

**Canonical upload filename:** `Anu_Options_Engine_EXECUTION_PROTOCOL_v3_0_7.md`
**Effective logic version:** 3.2.9-r8-audit-fixes
**Date:** 2026-04-26
**Revision:** r8 audit fixes (2026-04-26) — partial-payoff EV/ML, portfolio caps, volatility regime, RV/IV ratio, Confident-Buy tier, PriorityScore composite. Inherits r7.1 split-audit + r6 mixed-flow rescue + scan-date OI default.

## Audit gate zero: canonical executable sync

Before candidate seeding, verify that all five canonical project files are loaded from the same bundle and that the root `anu_analysis_v3_1_7.py` has the same effective logic version as the markdown files.

If the root Python is stale, FIRE-only, missing SHIELD, missing condors, missing the Health Gate adapter, missing the scan-date OI resolver / explicit bounded next-day OI overlay, or missing the `Size != None` publication guard, the scan is **BLOCKED**. Do not continue with sidecar replacement logic.

The remediation is full-bundle replacement only.

## Run order

1. Hash and load all required inputs
2. Verify all five canonical project files before candidate seeding
3. Resolve flow source: full streamed `bot-eod-report-YYYY-MM-DD.zip` first; markdown `whale-YYYY-MM-DD.md` only when full source is absent or explicitly allowed for development fallback
4. Stream full-source rows in chunks, infer FIRE/SHIELD family from raw bot fields when needed, and keep a bounded executable candidate reservoir
5. Compute full-source ticker bull/bear premium balance from native rows only
6. Deduplicate repeated economic seeds
7. Generate same-direction family-flex alternate seeds without inflating same-ticker bias
8. Label minority-flow, split-flow watch, FIRE neutral-conflict, and SHIELD bias-mismatch rows
9. Construct executable FIRE debit verticals from the actual hot chain
10. Construct executable SHIELD credit verticals from the actual hot chain
11. Apply the written file-native SHIELD anchor to each SHIELD credit build
12. Pair anchored opposite-side SHIELD credit rows into iron condors when possible
13. Merge scan-date OI context by default; use bounded next-day OI only when explicitly requested
14. Compute POP, reward/risk, EV/ML, and Conviction
15. Apply event-window governance and force material 0-6 day earnings names into catalyst watch
16. Apply Mixed-Flow Rescue to eligible neutral-conflict FIRE debit rows
17. Apply Health Gate execution label to primary rows while preserving Watch execution labels
18. Publish the primary table and watch table
19. Emit the full diagnostic output packet and audit JSON

## Required inputs

Minimum audited FIRE + SHIELD no-GEX run set:

- whale summary markdown or equivalent flow source
- hot-chain quotes
- stock screener premium imbalance
- chain OI change file for the scan date; bounded next-day follow-through only when explicitly requested
- dark-pool file for context only

## Candidate seeding

- FIRE debit rows create FIRE candidate seeds.
- SHIELD credit rows create SHIELD candidate seeds.
- Rows must be deduplicated before scoring.
- Repeated seeds are disclosed in audit but counted once for execution.

## Same-ticker bias

Compute:

- `whale_bias`
- `screen_bias`
- `combined_bias = 0.7 * whale_bias + 0.3 * screen_bias`
- `dominant_direction`
- `dominance_strength`
- `whale_lead_ratio`

Directional contribution rules:

- FIRE debit call and SHIELD put credit premium contribute to `bull`
- FIRE debit put and SHIELD call credit premium contribute to `bear`

## Blocking and conditional labels

Before final selection, label each seed or build as applicable.

Hard non-promotion labels:

- `minority_flow` against a clearly dominant opposite side
- `split_flow_watch` when applicable
- event block: earnings/catalyst inside 0-10 days
- `shield_bias_mismatch` for SHIELD credit directionality
- fabricated/missing leg, invalid debit/credit geometry, negative EV/ML, failed Health Gate, or `Size = None`

Conditional labels:

- `neutral_conflict` for FIRE debit directionality

A neutral-conflict FIRE debit row may promote only through Mixed-Flow Rescue. If rescue fails, it routes to Watch, Alternates, or Blocked Positive-EV; it may not silently disappear.

## FIRE debit spread construction

For each FIRE debit seed:

- use the exact whale strike when the chain contains it
- otherwise use the nearest liquid strike on the same side and expiry
- find an actual short leg in the profitable direction
- use conservative executable quotes for debit pricing
- require positive debit and positive reward/risk
- reject rows with no executable short leg or invalid debit geometry

## SHIELD credit spread construction

For each SHIELD credit seed:

- use the exact whale strike for the short leg when the chain contains it
- otherwise use the nearest liquid strike on the same risk side and expiry
- find an actual protective long wing farther OTM on the risk side
- use conservative executable quotes for credit pricing
- require positive credit and positive reward/risk
- reject rows with no protective long wing or invalid credit geometry
- compute and store the file-native SHIELD anchor result

## Iron condor construction

An iron condor may be built only when all are true:

- one anchored SHIELD call credit build exists
- one anchored SHIELD put credit build exists
- both builds share the same ticker and expiry
- the short put strike is below the short call strike
- the ticker bias is neutral enough for condor use
- all four legs exist in the actual hot chain

If those conditions fail, no condor is created and the failure belongs in the audit. When more than one eligible condor pair exists on a ticker and expiry, keep the highest EV/ML pair and use Conviction only as a tie-break.

## Selection rules for the primary table

A FIRE debit row is primary-table eligible only when all are true:

- liquidity tier is `MAJOR`, or `MID_PILOT` with Pilot sizing
- not minority-flow
- not split-flow watch
- not neutral-conflict unless Mixed-Flow Rescue is satisfied
- not inside the 0 to 10 day event block window
- common-stock primary focus
- either same-ticker combined bias is material **or** the row qualifies for Mixed-Flow Rescue
- EV/ML is positive

Mixed-Flow Rescue eligibility for neutral-conflict FIRE rows requires positive EV/ML, valid debit geometry, executable size, no hard block, and independent follow-through/quote-side evidence. The material combined-bias gate must not be re-applied to rescued neutral-conflict rows.

A SHIELD credit vertical is primary-table eligible only when all are true:

- market cap is major-liquid, or a written index-volatility exception applies
- not minority-flow
- not split-flow watch
- not inside the 0 to 10 day event block window
- file-native SHIELD anchor is true
- not SHIELD bias-mismatch
- EV/ML is positive

A SHIELD iron condor is primary-table eligible only when all are true:

- market cap is major-liquid, or a written index-volatility exception applies
- not inside the 0 to 10 day event block window
- both SHIELD sides are anchored
- ticker bias is neutral enough for condor use
- EV/ML is positive

Then:

- keep one best row per ticker
- rank by EV/ML first
- use Conviction only as secondary context

## Watch table

The watch table may include:

- blocked positive-EV FIRE rows
- blocked or negative-EV anchored SHIELD rows
- failed or blocked condor pair attempts
- split-flow near-event rows
- rejected rows with no liquid executable leg
- other non-executable but material names

## Audit requirements

Every run must disclose:

- input hashes
- raw seed count and deduped seed count
- FIRE seed count and SHIELD seed count
- duplicate seeds found
- family-flex derived seed count and built row count
- built SHIELD credit count and anchored SHIELD count
- condor pair attempts and condor builds
- blocked minority-flow rows
- split-flow watch rows
- FIRE neutral-conflict rows
- SHIELD bias-mismatch rows
- rejected rows with explicit reasons
- primary table
- watch table
- Health Gate status
- confirmation that automated GEX is disabled

## v3.2.6 family-flex execution additions

### Family-flex candidate generation

After native seed deduplication and same-ticker bias calculation, create derived alternate-family candidates as follows:

- SHIELD call credit => FIRE bear put debit
- SHIELD put credit => FIRE bull call debit
- FIRE put debit => SHIELD bear call credit
- FIRE call debit => SHIELD bull put credit

The derived candidate must keep the source ticker, expiry, and bull/bear thesis direction. It must use a real hot-chain anchor strike on the destination option side, preferably near-ATM/OTM for the destination structure. It may not reuse an absurd opposite-side source strike when that would create a nonsensical structure.

Family-flex rows are derived candidates, not new whale rows. They must not affect same-ticker bull/bear premium balance, whale lead ratio, duplicate-seed counts, or screener imbalance. They may compete in EV/ML ranking only after they are built from actual hot-chain legs and pass the destination family eligibility rules.

A FIRE row derived from a SHIELD seed may publish if it passes the normal FIRE primary rules. A SHIELD row derived from a FIRE seed may publish only if it independently satisfies the written file-native SHIELD anchor; otherwise it is watch-only.

### Catalyst watch surface area

When a high-premium ticker is blocked by the 0-6 day earnings rule, the report must still surface the ticker in the watch table with a catalyst-watch reason. This row is never BUY/SELL and never overrides the earnings gate. It exists to prevent material event names from disappearing from the workflow.

### Watch execution labels

The Health Gate adapter sets `Execution = Strict` for executable primary rows after broker-native `PASS`. Watch rows remain `Execution = Watch` and cannot be relabeled Strict.


## v3.2.4 audit additions

### OI resolver

When scanning an EOD date, resolve OI as:

1. current OI = scan-date OI by default
2. previous OI = the nearest OI file before the current OI handle
3. current OI may become `scan_date + 1` only when the operator explicitly enables next-day overlay mode
4. never jump more than one calendar day forward

The audit JSON must disclose both OI handles, the OI mode, whether next-day overlay was requested, and whether next-day overlay was actually used.

### Broker Health Gate adapter

If a Schwab positions artifact exists, inspect `accounts[].health_gate.status`. If `rows_checked > 0`, apply the broker-native status before derived closed-trade logic. `PASS` should set primary-table `Execution = Strict`.

### Primary-table size guard

After computing the size bucket, exclude any row with `Size = None` from primary BUY/SELL publication. Such rows may still appear in the watch table.


## v3.2.7 execution additions

### Source selection order

Candidate seeding now resolves the flow source as:

1. Full streamed `bot-eod-report-YYYY-MM-DD.zip` / EOD flow source when present
2. Markdown `whale-YYYY-MM-DD.md` Top-200 summary when full source is absent

The full source must be streamed in chunks. The engine keeps a bounded candidate reservoir rather than loading the entire ZIP or building every candidate row. Full-source ticker bull/bear premium balance overrides Top-200-only bias when available.

### Candidate reservoir from full source

For full-source runs, keep:

- global top premium executable candidates
- per-symbol / per-family / per-direction representatives
- full-source top-symbol premium statistics
- full-source direction-balance statistics

The reservoir is for construction only; same-ticker bias comes from the full filtered source.

### Selection routing

After construction and scoring, rows are routed as follows:

- Primary table: EV/ML-first, one best row per ticker, clean event/size/focus/bias gates
- Mid-cap clean rows: Pilot only
- Small/unknown-cap rows: watch-only
- ETF/index rows: ETF/index lane
- Neutral-conflict FIRE rows: mixed-flow watch unless Mixed-Flow Rescue is satisfied
- Positive-EV blocked rows: Blocked Positive-EV table
- Non-primary same-ticker rows: Alternates table

### Mandatory report outputs

The executable must write CSV outputs for primary recommendations, watch rows, ETF/index lane, blocked positive-EV rows, alternates, top-symbol gap, built rows diagnostics, audit JSON, run report markdown, and deep audit markdown.


## v3.2.7 r5 raw bot execution mapping

The full bot/EOD ZIP may expose raw columns such as `executed_at`, `underlying_symbol`, `side`, `strike`, `option_type`, `expiry`, `underlying_price`, `price`, `size`, `premium`, `open_interest`, `implied_volatility`, `delta`, `canceled`, and `equity_type`, without enriched `track`, `net_type`, `width`, or `pct_width` columns.

In that case, before candidate filtering:

1. infer FIRE/SHIELD family from `side`;
2. infer vertical width from underlying price using the 2.5 / 5 / 10 ladder;
3. compute `pct_width = price / width`;
4. apply FIRE and SHIELD DTE / pct-width / OI / distance filters;
5. keep full-source ticker premium balance from the streamed filtered rows;
6. never silently fall back to markdown when the full bot ZIP is present unless `--allow-markdown-seed-fallback` is explicitly passed.

The audit JSON must disclose the schema mapping and selected full-source seed count.


## r6 rescue/audit corrections

- Neutral-conflict is conditional, not a hard block. The executable must not reapply the material-bias threshold to a row that has already satisfied Mixed-Flow Rescue.
- `mixed_flow_rescued_rows` must be counted in audit JSON and rescued rows must carry `mixed-flow rescue` in Notice.
- Exact scan-date OI is the default. The next-day OI overlay is available only when explicitly requested and must be disclosed separately as requested/used.
- Full-source ingestion remains preferred; markdown Top-200 remains fallback only.
- Browser/GEX remains non-operative and cannot satisfy rescue, SHIELD anchoring, ranking, sizing, or promotion.

### Raw bot schema inference

A full `bot-eod-report-YYYY-MM-DD.zip` may contain raw single-leg rows rather than enriched `track`, `net_type`, `width`, or `pct_width` columns. In that case the executable must infer:

- `side = bid` => SHIELD credit candidate;
- `side = ask`, `mid`, or `no_side` => FIRE debit candidate;
- `option_type` => call/put direction;
- missing width from the standard width ladder;
- `pct_width = price / width` when missing.

The full-source parser must not silently fall back to markdown when the raw bot ZIP is present. It must either stream and map the raw schema or fail with a schema diagnostic unless the operator explicitly uses the emergency markdown fallback flag.

### Mixed-Flow Rescue execution

A neutral-conflict FIRE debit row may publish only if it passes all hard gates and the executable spread itself has positive edge. The combined-bias materiality gate is waived only for rows marked `mixed_flow_rescue=True`, because neutral-conflict is by definition weak combined bias. The Notice field must disclose mixed-flow rescue and require live entry-gate validation.

### ETF/index lane action labeling

ETF/index rows are emitted in a separate lane. They must not be labeled as common-stock primary BUY/SELL rows. Use ETF WATCH/CANDIDATE labeling unless a future dedicated ETF primary lane is explicitly enabled.


## r7 execution/reporting additions

### Required run outputs added in r7

In addition to the r6 mandatory outputs, the executable must write:

1. `options_scan_YYYY-MM-DD_audited_all_primary_eligible_ideas.csv`
2. `options_scan_YYYY-MM-DD_audited_high_conviction_ideas.csv`
3. `options_scan_YYYY-MM-DD_audited_order_entry_sheet.csv`

The run report must print, in order:

1. official EV/ML primary table;
2. high-conviction idea lane using the configured threshold;
3. order-entry sheet with dollar max loss / max gain per one-lot;
4. watch / catalyst / blocked / alternates / ETF diagnostics.

### Conviction threshold

Default threshold is `62`, meaning `Conviction > 61`. The command line may override this with:

```bash
--conviction-threshold 62
```

### Health Gate enforcement flag

Health Gate is advisory by default. Missing Schwab data or unresolved broker logs must not block the scan.

Only this flag enables broker-log blocking:

```bash
--enforce-health-gate
```

Without the flag, a broker-native FAIL/WARN is disclosed but does not suppress trade ideas.

### Live quote artifact semantics

Live quote data is an execution artifact. It may recompute net price and EV/ML before order entry. It must not be required to generate EOD trade ideas, and Mixed-Flow Rescue rows that already passed r6 hard gates must remain visible as ideas.

### Non-primary lane labeling

High-conviction ideas, all-primary-eligible ideas, order-entry sheets, alternates, and built-row diagnostics are not the official EV/ML primary table. The report must label them separately and must never call them the primary table unless they are from the official primary output.


### Full bot EOD source handling

The EOD bot source must be the complete `bot-eod-report-YYYY-MM-DD.csv` or `bot-eod-report-YYYY-MM-DD.zip`. Split part files such as `bot-eod-report-YYYY-MM-DD.part-NN-of-MM.zip` are not valid inputs for this pipeline.


## r7.1 source audit correction

Full-source bot ZIP runs must hash the complete main file consumed by the engine. Split part files are rejected so audit completeness is tied to the main full-source report.


## r8 execution additions (2026-04-26)

These additions implement the rulebook r8 fixes at the executor level. See `claude/AUDIT_FINDINGS_2026-04-26.md` for derivations.

### Run-order changes (between primary candidate selection and publication)

After the per-ticker EV/ML-first dedup, the executor now runs three additional passes before publishing the primary table:

1. **Volatility-regime gate** (HIGH-5 + MED-3). Joins `iv_rank`, `iv30d`, `volatility` from the screener, computes `vol_regime` and `rv_iv_ratio`, sets `vol_regime_block` and a per-row notice. Blocked rows are dropped from primary; their notices remain visible in audit/Watch.

2. **Portfolio-cap pass** (CRIT-3). Single-name, sector, direction caps. Walks the EV/ML-sorted primary table; admits rows respecting caps, routes excluded rows to `portfolio_dropped` with `drop_reason`.

3. **Confident-Buy visual tier** (LOW-2). Marks any surviving row with Conv >= 70 AND EV/ML >= 0.40 AND POP >= 0.25 as `🔥🟢` and appends a `Confident-Buy tier` notice.

### Math changes

- `partial_ev_ml_debit` and `partial_ev_ml_credit` now compute closed-form lognormal partial-payoff EV/ML across the strike-to-strike ramp. The legacy binary `pure_ev_ml(POP, R/R)` is retained only for fallback when the partial integral is degenerate (POP=0, sigma_t<=0).

- `compute_pop_*` now uses real-world flat-drift `z = ln(K/S)/sigma_T` (MED-1).

### Audit JSON additions

- `portfolio_caps_dropped_rows`, `portfolio_caps_drop_reasons`
- `vol_regime_blocked_rows`
- `confident_buy_rows`
- `audit_fix_revision: r8-2026-04-26`
- per-row `rv_iv_ratio`, `vol_regime`, `vol_regime_block`, `vol_regime_notice` (in built CSV)

### Order-entry sheet

The order-entry CSV now includes `PriorityScore` (Conv 50% + EV/ML 30% + √POP 20%, scaled to 0-100) and is sorted by PriorityScore descending. EV/ML-first remains the canonical ranking for the primary inline table; PriorityScore is the operator-facing composite for "what to enter first" given backtested edge.

### Conviction calibration note

Backtest of historical engine output (n=849, 2026-03-20 → 2026-04-15, 5d horizon, mid-price marks):

- Conv >= 65: 90% hit, mean +$218/lot.
- Conv 55-64: 59% hit, mean +$119/lot.
- Conv < 55: 58% hit, mean +$52/lot.

The default conviction threshold of 62 (high-conviction lane) is calibrated; operator may surface Conv >= 65 as a tighter "Confident-Buy" sub-tier (already wired by r8.14 above).

### Sizing buckets recap

- FIRE debit: EV/ML >= 1.0 → Tiny; >= 0.40 → Starter; >= 0.10 → Pilot; else None.
- SHIELD credit: EV/ML >= 0.40 → Starter; >= 0.10 → Pilot; >= 0.03 → Tiny; else None. (r8.1 fix: previously the lower band collapsed into duplicate Pilot.)
- Iron condor: EV/ML >= 0.30 → Starter; >= 0.08 → Pilot; else None.

### DTE bands

- FIRE debit: DTE 7-90, pct_width <= 0.35.
- SHIELD credit: DTE 7-75, pct_width 0.20-0.55.
- Earnings/event windows are still enforced via `inside_event_block` (0-10d).

### SHIELD anchor follow-through

`shield_anchor_ok` requires AT LEAST TWO of:

- bid_ask_ratio >= 1.20
- oi_change >= 0.05
- (curr_oi >= 1000 AND oi_change >= 0.0)

### Mixed-Flow Rescue follow-through

At least one MEANINGFUL signal:

- oi_change >= 0.10
- ask_bid_ratio >= 1.25
- (curr_oi >= 1000 AND oi_change >= 0.0)

### Conservative pricing

- Long leg: prefer `ask>0`; else `mid + max(0.05, mid - bid)`.
- Short leg: prefer `bid>0`; else `mid - max(0.05, ask - mid)`.
