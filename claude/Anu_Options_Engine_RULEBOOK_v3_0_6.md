# Anu Options Engine — RULEBOOK (Audited FIRE + SHIELD No-GEX Payload)

**Canonical upload filename:** `Anu_Options_Engine_RULEBOOK_v3_0_6.md`
**Effective logic version:** 3.2.9-r8-audit-fixes
**Date:** 2026-04-26
**Revision:** r8 audit fixes (2026-04-26) — see `claude/AUDIT_FINDINGS_2026-04-26.md`. Adds portfolio caps, volatility-regime gate, RV/IV gate, partial-payoff EV/ML, Confident-Buy tier, PriorityScore composite. Inherits r7.1 split-audit + r6 mixed-flow rescue + scan-date OI default.

This file preserves the canonical project slot while replacing the earlier audited no-GEX payload. Automated GEX remains disabled, but SHIELD is restored through a written file-native non-GEX anchor and a real iron-condor path.

## Core policy

The automated engine is **no-GEX**.

Atlas, browser, Unusual Whales page reads, exported GEX JSON, visible-page gamma summaries, and inferred walls or flips are **non-operative** for automated trade selection. They may not rank, gate, block, anchor, promote, suppress, size, or annotate an automated trade.

## Executable sync gate

The canonical Python file is the executable source for the run. A scan may not proceed by using an alternate in-chat, sidecar, or replacement logic path when the root `anu_analysis_v3_1_7.py` does not match the written project files.

If the four markdown project files and the root Python file do not share the same effective logic version and supported feature set, the audit gate must return **BLOCKED**. The only valid remediation is to replace all five canonical files from one full bundle and rerun.

The engine must not publish recommendations under a disclosed executable mismatch.

## Source of truth order

1. Whale / institutional flow rows for candidate seeding
2. Stock screener premium imbalance for same-ticker direction context
3. Hot chains for executable spread construction
4. Chain OI change files for scan-date follow-through; next-day OI overlay is opt-in only
5. Dark-pool as context only

## Candidate families

The engine supports three automated structure families:

- **FIRE debit verticals**
- **SHIELD credit verticals**
- **SHIELD iron condors** formed only from two anchored SHIELD credit sides on the same ticker and expiry

## Direction lock and same-ticker governance

For FIRE debit candidates, the initiating whale row owns the side:

- debit call row => bullish call vertical
- debit put row => bearish put vertical

For SHIELD credit candidates, the initiating whale row owns the side:

- credit call row => bearish or neutral call credit spread
- credit put row => bullish or neutral put credit spread

For same-ticker direction context:

- FIRE debit calls and SHIELD put credits contribute to **bull** premium
- FIRE debit puts and SHIELD call credits contribute to **bear** premium

Same-ticker flow is reconciled from:

- 70% whale premium balance
- 30% screener premium imbalance

A row is **minority flow** and may not be promoted to BUY or SELL when:

- it opposes the ticker's dominant same-day direction, and
- the combined bias is materially one-sided, and
- whale premium leadership is clear

A ticker is **split-flow** when both bullish and bearish whale premium are present and neither side clearly dominates. Split-flow within 10 calendar days of earnings is watch-only.

A FIRE row is **neutral-conflict** when both sides are present and same-ticker combined bias is too weak to cleanly support a directional debit spread. Neutral-conflict is a risk label and routing signal, not an automatic deletion rule. It may publish only through **Mixed-Flow Rescue** when hard blocks are absent and independent structure/follow-through evidence supports the row.

A SHIELD row is **bias-mismatch** when same-ticker combined bias is materially opposite the proposed credit side. Bias-mismatch SHIELD rows are not promoted.

## Construction rules

The engine must build structures from the **actual hot chain**.

### FIRE debit verticals

- The long leg uses the whale strike when present.
- If the whale strike is absent, the engine may use the nearest liquid strike on the same expiry and option side.
- The short leg must exist in the actual hot chain on the profitable side of the spread.
- The short leg may adjust away from the whale target strike when the target is absent or illiquid, but the adjustment must be disclosed in the Notice field.

### SHIELD credit verticals

- The short leg uses the whale strike when present.
- If the whale strike is absent, the engine may use the nearest liquid strike on the same expiry and option side.
- The protective long wing must exist in the actual hot chain farther OTM on the risk side.
- When the exact short strike is absent, any disclosed short-strike replacement must stay on the same risk side of the whale short strike.
- The protective wing may adjust farther OTM when the target wing is absent, but the adjustment must be disclosed in the Notice field.
- Naked credit exposure is forbidden.

### SHIELD iron condors

- An iron condor may be created only by pairing one anchored SHIELD call credit row and one anchored SHIELD put credit row on the same ticker and expiry.
- All four legs must exist in the actual hot chain.
- The short put strike must remain below the short call strike.
- Fabricated opposite-side legs are forbidden.

If no executable short or protective leg exists, the row is not executable and may only be watch or rejected.

## File-native SHIELD anchor

A SHIELD credit row may be auto-promoted only when a written non-GEX file-native anchor is satisfied.

The anchor requires all of the following:

- the row is `track = SHIELD` and `net_type = credit`
- the initiating flow is seller-led from file data, normally `side = bid`, or `side = mid` with supporting bid-dominant OI context
- the exact short strike exists in the actual hot chain, or a disclosed nearest liquid replacement exists
- the protective long wing exists in the actual hot chain farther OTM
- hot-chain and OI liquidity are real rather than fabricated
- same-ticker combined bias is not materially opposite the proposed credit side

Browser context may not satisfy this anchor. The anchor is file-native only.

## Ranking and scoring

The primary inline table must remain:

`Ticker | Action | Buy leg | Sell leg | Expiry | Net | EV/ML | POP | Conviction | Execution | Notice | Size`

Sort by **EV/ML first**.

`EV/ML` is pure expectancy divided by max loss, using only POP and reward/risk.

- debit POP uses the expiration breakeven of the debit spread
- credit POP uses the expiration breakeven of the credit spread
- condor POP uses the probability of finishing between the two condor breakevens

`Conviction` is secondary context and may break ties, but may not outrank EV/ML. Internal pair-selection logic must follow the same EV/ML-first rule.

## Earnings / catalyst policy

- earnings in 0 to 6 calendar days => no BUY or SELL
- earnings in 7 to 10 calendar days => watch-only
- earnings in 11 to 14 calendar days => only if otherwise clean and conservatively sized
- earnings 15+ calendar days away => normal scoring

## Health Gate policy

Health Gate precedence remains:

1. broker-native `accounts[].health_gate.status`
2. engine-native realized option trade log
3. reconciled broker `closed_trades`
4. `UNKNOWN`

If those artifacts are absent, the scan continues in **Bootstrap** mode. It does not stall.

Schwab broker data does not need FIRE / SHIELD labels for Health Gate.

## Track policy

Automatic publication supports both **FIRE** and **anchored SHIELD** under this payload.

- FIRE debit structures may publish automatically when otherwise eligible.
- SHIELD credit structures may publish automatically only when the written file-native non-GEX anchor is satisfied.
- SHIELD iron condors may publish automatically only when both SHIELD sides are anchored, the ticker bias is neutral enough for a condor, and all four legs are real.

## Focus policy

The engine remains common-stock first.

A narrow exception is allowed for broad volatility or index tickers such as VIX or SPX when the structure is SHIELD credit or SHIELD iron condor and the file-native anchor is satisfied.

## Duplicate-seed policy

Repeated whale rows representing the same economic seed must be deduplicated before scoring. Duplicate seeds may be disclosed in the audit, but they may not inflate candidate counts or distort ranking.

## v3.2.6 family-flex and catalyst-watch additions

### Family-flex translation

The engine may evaluate a same-direction alternate structure family when a file-native seed would otherwise be structurally weak, negative EV, or non-executable. Track is no longer destiny; thesis direction remains file-native.

Allowed same-direction translations are:

- SHIELD call credit seed => FIRE bear put debit candidate
- SHIELD put credit seed => FIRE bull call debit candidate
- FIRE put debit seed => SHIELD bear call credit candidate
- FIRE call debit seed => SHIELD bull put credit candidate

Family-flex candidates must satisfy all of the following:

- same ticker and expiry as the source seed
- same bull/bear thesis direction as the source seed
- all legs must exist in the actual hot chain
- no fabricated short, long, or protective legs
- same-ticker bias is computed only from native file rows, not from derived family-flex rows
- source seed, source family, and translation type must be disclosed in the audit
- ranking remains EV/ML-first

A SHIELD candidate derived from a FIRE source is watch-only unless it independently satisfies the written file-native SHIELD anchor. A FIRE candidate derived from a SHIELD source may publish when it satisfies the normal FIRE primary-table requirements, because it is a debit-risk expression of the same file-native directional thesis.

### High-premium catalyst watch

A ticker with material file-native premium that is blocked by the 0-6 day earnings gate may not silently disappear from the user-facing report. It must appear in a catalyst-watch section or watch row with the reason for the BUY/SELL block and the instruction to rerun a live-chain post-event scan. This does not override the earnings gate and does not publish the row as BUY or SELL.

### Watch execution label cleanup

Broker Health Gate status applies to executable primary rows. WATCH rows must remain `Execution = Watch` even when Health Gate is `PASS`.


## v3.2.4 audit additions

### OI handle policy

For an EOD base scan, the production default is **exact scan-date OI**: `chain-oi-changes-YYYY-MM-DD.zip` matching the scan date is the current OI handle when present. A bounded next-calendar-day OI overlay may be used only when explicitly requested, for example with `--use-next-day-oi` or a user request for next-day follow-through.

The resolver must not silently pull tomorrow's OI into a same-date analysis, and it must not jump multiple days forward by default. The audit JSON must disclose previous OI, current OI, whether next-day overlay was requested, and whether it was actually used.

### Health Gate execution label

When a Schwab artifact exposes broker-native `accounts[].health_gate.status` with `rows_checked > 0`, that status must be used ahead of any derived Health Gate source. `PASS` maps to `Strict`; unresolved status maps to `Bootstrap`; blocking or failing status must not silently publish as normal.

### Size publication guard

A row whose computed `Size` bucket is `None` may not publish as BUY or SELL. Positive EV/ML with no executable size bucket is watch-only.


## v3.2.7 routing fixes

### Full-source flow ingestion

When a full `bot-eod-report-YYYY-MM-DD.zip` / EOD flow source is present, the engine must stream the CSV in bounded chunks and use the full filtered Yes-Prime/rulebook-like source for ticker-level premium balance. The engine must not require loading a 1GB ZIP into memory.

The streamed source must produce:

- total rows scanned
- filtered candidate count
- bounded candidate reservoir for executable construction
- full-source top-symbol premium table
- full-source bull/bear premium balance for same-ticker direction

### Raw bot CSV schema mapping

When the full bot EOD ZIP contains raw single-leg rows rather than enriched rulebook columns, the engine must infer candidate family deterministically from file-native fields:

- `side = bid` => SHIELD credit candidate
- `side = ask`, `mid`, or `no_side` => FIRE debit candidate
- `option_type` sets call/put direction
- `width` is inferred from the engine width ladder when absent
- `pct_width = option_price / vertical_width` when the bot file does not provide `pct_width`

The engine must not require `track`, `net_type`, or `pct_width` to already exist in the raw bot CSV. Schema mapping must be disclosed in audit JSON.

If the full source is absent and only the markdown Top-200 summary is present, the engine may scan from the Top-200 table, but it must emit a **Top-Symbol Gap** watch section for high-premium names whose visible Top-200 share is low. These names may not silently disappear.

### Family-flex and same-direction translation

Family-flex is official under this payload. A seed may test an alternate same-direction structure family:

- SHIELD call credit -> FIRE bear put debit
- SHIELD put credit -> FIRE bull call debit
- FIRE put debit -> SHIELD bear call credit
- FIRE call debit -> SHIELD bull put credit

The translation must preserve ticker, thesis direction, and file-native context. It may not flip bullish to bearish or bearish to bullish. Actual hot-chain legs are still required. Translated SHIELD rows remain subject to the file-native SHIELD anchor before promotion.

### Liquidity tiers instead of hard market-cap deletion

The $80B market-cap rule is no longer a hard delete. It becomes a routing tier:

- `MAJOR`: market cap >= $80B, normal primary eligibility
- `MID_PILOT`: $10B <= market cap < $80B, primary eligible only as Pilot if all other gates are clean
- `SMALL_WATCH`: market cap < $10B, watch-only unless a future special-situation lane explicitly enables it
- `UNKNOWN_WATCH`: missing market cap, watch-only
- `ETF_INDEX`: ETF/index products, separate ETF/index lane, not common-stock primary

### Mixed-flow routing and rescue

Neutral-conflict FIRE rows no longer disappear by default. They route to mixed-flow watch unless they qualify for **Mixed-Flow Rescue**.

A neutral-conflict FIRE debit row may publish through Mixed-Flow Rescue when all are true:

- the row is not minority-flow against a clearly dominant opposite side;
- the row is not split-flow watch and is not inside the 0-10 day event block;
- all legs are actual hot-chain legs and the computed size is not `None`;
- EV/ML is positive and debit-spread reward/risk is valid;
- the row has either practical POP (`POP >= 0.15`) or stronger convexity edge (`EV/ML >= 0.40` plus high Conviction);
- independent follow-through/quote-side evidence exists from OI change, ask/bid imbalance, current OI, or an attached live quote gate;
- the Notice field discloses `mixed-flow rescue` and requires live entry-gate validation.

The material same-ticker bias gate applies to clean directional FIRE rows. It must **not** be re-applied to a rescued neutral-conflict row, because weak combined bias is the definition of neutral-conflict. Minority same-ticker flow still may not be promoted.

### Expanded reporting

Every run must emit, at minimum:

1. Primary BUY/SELL table
2. Watch table
3. Catalyst Watch rows
4. Top-Symbol Gap rows when operating from markdown-only Top-200 source
5. Blocked Positive-EV table
6. Per-ticker Alternates table
7. ETF/index lane table
8. Built rows diagnostic CSV
9. Audit JSON and deep audit markdown

### Bounded same-thesis leg rescue

If the exact same-expiry spread cannot be built, the engine may try a bounded adjacent-expiry rescue within 14 calendar days, using actual hot-chain legs only. The rescue must preserve thesis direction, option side, risk side, and no-fabricated-leg policy. Any expiry rescue must be disclosed in Notice.

### Convexity labels

The EV/ML-first ranking contract remains unchanged. However, user-facing notices must label:

- `POP < 5%`: lottery/convexity only
- `5% <= POP < 15%`: low-POP convexity

These labels do not rank, gate, or suppress; they prevent overconfidence in low-probability structures.

### Hard audit-gate zero

If any of the four canonical markdown files is missing, or if any markdown file and the root Python file do not share the same effective logic version, the scan is BLOCKED. `NOT_CHECKED` is no longer acceptable for executable sync.


## v3.2.7 r5 raw bot schema mapping

When `bot-eod-report-YYYY-MM-DD.zip` contains raw single-leg option-flow rows rather than enriched Yes-Prime rows, the engine must reconstruct rulebook-like candidate fields from the raw source instead of falling back to Top-200 markdown.

Required raw mapping:

- `side = bid` => SHIELD credit seed
- `side = ask`, `mid`, or `no_side` => FIRE debit seed
- vertical width is inferred from underlying spot:
  - spot < 25 => 2.5-wide
  - spot < 75 => 5-wide
  - spot >= 75 => 10-wide
- `pct_width = option price / inferred vertical width`

Full-source mode must still stream chunks, use actual hot-chain legs for executable spreads, and disclose the raw mapping in audit JSON. Markdown fallback is emergency/development only when explicitly requested.


## r6 rescue/audit corrections

These corrections close contradictions found in v3.2.7:

1. **Mixed-flow rescue may publish.** Neutral-conflict is a conditional label, not a hard primary-table block. The material combined-bias gate is bypassed only for rows that satisfy Mixed-Flow Rescue.
2. **Hard blocks remain hard.** Earnings 0-10 day blocks, minority-flow against a dominant opposite side, fabricated legs, invalid geometry, negative EV/ML, failed Health Gate, and `Size = None` remain non-publishable.
3. **Exact-date OI default.** Same-date OI is the default. Next-day OI overlay is explicit, bounded, and disclosed.
4. **Output truthfulness.** The audit must separately disclose `next_day_overlay_requested` and `uses_next_day_overlay`; filename ordering alone is not proof of next-day overlay.
5. **Live quote handling.** A live Schwab/broker quote gate may rescue a mixed-flow row only as an execution validation input; it does not replace EV/ML ranking. When live quote data is available, EV/ML should be recomputed from the executable live net or the row remains watch-only.
6. **ETF/index lane clarity.** ETF/index candidates use ETF WATCH/CANDIDATE labeling unless a dedicated ETF primary lane is explicitly enabled.


## r7 trade-idea routing corrections

These corrections fix the user-facing pipeline without changing the FIRE/SHIELD construction math.

### Idea stage vs execution stage

The engine has two distinct stages:

1. **Trade idea stage**: EOD flow, hot-chain, scan-date OI, same-ticker bias, event policy, size guard, and EV/ML scoring decide which ideas are worth carrying forward.
2. **Execution stage**: live broker quotes are used only to verify the final order price, recompute live net/EV when supplied, and avoid stale or crossed fills.

A row marked `mixed-flow rescue; live entry gate required` is **not blocked at the idea stage**. It is a valid idea when it passed the written Mixed-Flow Rescue gate. The live quote gate is the final order-entry validation, not a reason to hide the idea table.

### Health Gate default is advisory unless explicitly enforced

Missing Schwab/broker artifacts remain `Bootstrap` and must never stall the scan.

By default, broker Health Gate artifacts are **advisory**:

- `PASS` may label executable primary rows as `Strict`.
- `UNKNOWN`, missing artifacts, or absent logs leave rows as `Bootstrap`.
- `FAIL`, `WARN`, or unresolved broker rows are written to audit and Notice but do not suppress trade ideas unless the operator explicitly runs with `--enforce-health-gate`.

When `--enforce-health-gate` is enabled, a broker-native hard fail may block BUY/SELL publication. Without that explicit flag, the pipeline must continue and surface ideas.

### Mandatory high-conviction idea lane

The EV/ML primary table remains official and keeps the exact primary table contract. However, the report must also emit a separate high-conviction idea lane so high-conviction rows are not buried by EV/ML-first ranking.

The high-conviction idea lane includes rows when all are true:

- row is primary-table eligible under normal construction/gating rules;
- common-stock lane only, not ETF/index;
- `EV/ML > 0`;
- `Size != None`;
- not inside the 0-10 day earnings/catalyst block;
- not minority-flow and not split-flow watch;
- `Conviction >= conviction_threshold`, default `62`, equivalent to `>61`.

The high-conviction lane includes all qualifying primary-eligible rows, not only the one-per-ticker EV/ML winner. It is sorted by `Conviction` first, then `EV/ML`, then POP. This does not override the official EV/ML primary ranking; it is an additional idea-discovery table.

### Mandatory order-entry sheet

Every run must emit an order-entry sheet for trade ideas. The sheet must include:

- ticker, action, expiry, buy leg, sell leg;
- EOD entry limit: debit rows use `pay <= Net`; credit rows use `collect >= Net`;
- EV/ML, POP, Conviction;
- max loss and max gain per one-lot in dollars;
- Size bucket retained for audit, but the user-facing sheet must show dollar risk so `Tiny`, `Starter`, and `Pilot` are not the only sizing language;
- live quote instruction: live quote confirms final order entry and is not an idea-stage blocker.

### Output truthfulness

The assistant/report must not present diagnostics, alternates, clean overlays, or hand-filtered rows as the official primary table. Non-primary idea lanes must be clearly labeled as separate outputs from the canonical built rows.


### Full bot EOD source handling

The EOD bot source must be the complete `bot-eod-report-YYYY-MM-DD.csv` or `bot-eod-report-YYYY-MM-DD.zip`. Split part files such as `bot-eod-report-YYYY-MM-DD.part-NN-of-MM.zip` are not valid inputs for this pipeline.


## r7.1 source audit correction

Full-source bot ZIP runs must hash the complete main file consumed by the engine. Split part files are rejected so audit completeness is tied to the main full-source report.


## r8 audit fixes (2026-04-26)

This revision lands the findings of `claude/AUDIT_FINDINGS_2026-04-26.md`. Trade logic now incorporates portfolio risk, volatility regime, partial-payoff expectancy, and a backtested-priority composite.

### r8.1 Sizing buckets (CRIT-1)

Credit-vertical sizing now reaches all three tiers. Previous build collapsed the lower band to a duplicate `Pilot`. Bands:

- `EV/ML >= 0.40` => Starter
- `0.10 <= EV/ML < 0.40` => Pilot
- `0.03 <= EV/ML < 0.10` => Tiny
- `EV/ML < 0.03` => None (not publishable)

### r8.2 Partial-payoff EV/ML (CRIT-2)

EV/ML for FIRE debit and SHIELD credit verticals is now computed via closed-form lognormal integration over three zones (full loss, linear ramp, full profit) instead of binary breakeven payoff. The legacy `pure_ev_ml(POP, R/R)` is retained as a fallback only when the partial integral is degenerate.

### r8.3 Portfolio risk caps (CRIT-3)

After per-ticker EV/ML ranking and before final publication, three portfolio-level caps are applied:

- **Single-name cap**: at most 1 primary row per ticker (already enforced; now explicit).
- **Sector cap**: at most 2 directional FIRE primary rows per GICS sector per day (`screener.sector`). Credit / condor structures are exempt because they are not single-direction sector bets.
- **Direction cap**: at most 4 same-direction FIRE primary rows per day (all bull or all bear).

Rows excluded by a portfolio cap are routed to the alternates collector with `drop_reason=portfolio_cap:<single_name|sector:X|direction:bull|direction:bear>` so they remain visible in audit and reporting.

The defaults are tuned for a small-account retail operator. The constants live in `anu_analysis_v3_1_7.py` (`PORTFOLIO_*_CAP`) and may be raised by an explicit operator override.

### r8.4 OCC root regex (HIGH-1)

The option-symbol parser now accepts roots containing digits and standard OCC punctuation: `[A-Z][A-Z0-9.\-/]{0,5}`. The previous `[A-Z]+` silently dropped post-corp-action symbols.

### r8.5 SHIELD anchor follow-through (HIGH-2)

`shield_anchor_ok` now requires AT LEAST TWO of three meaningful follow-through signals:

- bid_ask_ratio >= 1.20 (real seller-led microstructure)
- oi_change >= 5% (OI growing meaningfully)
- curr_oi >= 1000 AND oi_change >= 0 (liquid AND not declining)

A single OR-branch passing on `curr_oi >= 1000` alone is no longer sufficient.

### r8.6 Conservative pricing (HIGH-3)

`conservative_long_price` now requires a real `ask>0` for the buy quote, then falls back to `mid + half_spread` when ask is missing. Symmetric change for `conservative_short_price` (use real bid; else mid - half_spread). Eliminates over-optimistic mid-print entry assumptions.

### r8.7 DTE bands and pct_width (HIGH-4 + LOW-1)

- FIRE debit: DTE 7-90 (was 21-70). pct_width <= 0.35 (was <= 0.45).
- SHIELD credit: DTE 7-75 (was 28-56). pct_width 0.20-0.55 (was 0.30-0.55).

Earnings/event windows are still enforced downstream via `inside_event_block` (0-10d).

### r8.8 Volatility regime gate (HIGH-5)

Each candidate carries `iv_rank`, `vol_regime`, and `rv_iv_ratio` from the screener. Rules:

- `iv_rank < 30` (low-IV regime): debit-favored. FIRE debits require `POP >= 0.20` for primary publication. SHIELD credits get a Notice tag.
- `iv_rank > 70` (high-IV regime): credit-favored. FIRE debits require `width >= 1.5x ladder` OR `Conviction >= 70` for primary publication.
- `rv_iv_ratio = volatility / iv30d`:
  - `> 1.20` (IV expensive): SHIELD favored, FIRE debit gets a penalty Notice.
  - `< 0.85` (IV cheap): FIRE favored, SHIELD gets a downgrade Notice.

Blocked rows are routed to Alternates / Watch with the `vol_regime:*` notice. They are not deleted from the audit.

### r8.9 Real-world flat-drift POP (MED-1)

`compute_pop_*` now uses real-world flat-drift `z = ln(K/S) / sigma_T` instead of the risk-neutral `+0.5 sigma^2` correction. This removes a 3-5 percentage-point bias between ITM and OTM probabilities and matches the natural-measure assumption used by the EV calculation.

### r8.10 Mixed-Flow Rescue follow-through (MED-2)

The follow-through gate now requires at least one MEANINGFUL signal (was: any of three trivial OR branches; the previous `curr_oi >= 1000` branch passed any liquid contract). New gate:

- oi_change >= 0.10
- ask_bid_ratio >= 1.25
- (curr_oi >= 1000 AND oi_change >= 0.0)

Sum of true >= 1 required.

### r8.11 RV / IV ratio in audit (MED-3)

The audit JSON exposes `rv_iv_ratio` per ticker (computed from screener `volatility / iv30d`). This is informational and used by the volatility-regime gate above; it is not a hard filter on its own.

### r8.12 Conviction spread-quality term (MED-4)

`conviction_raw` now includes a 0.10-weighted spread-quality component derived from the long-leg bid/ask:

- relative spread <= 4% of ask: full credit (1.0)
- relative spread >= 20% of ask: zero credit (0.0)
- linear in between

Other weights rebalanced to sum to 1.0. Wide-spread quotes can no longer publish high-conviction entries.

### r8.13 Defensive size cap for 0-10d earnings (MED-5)

`apply_event_size_cap` now sets `Size = "None"` for any row with `0 <= er_days <= 10`, regardless of computed EV/ML. This is defense-in-depth alongside `inside_event_block`.

### r8.14 Confident-Buy visual tier (LOW-2)

In the primary table, any row meeting ALL of:

- Conviction >= 70
- EV/ML >= 0.40
- POP >= 0.25

is marked `🔥🟢` (FIRE Confident) for visual emphasis. EV/ML-first ranking is unchanged; this is a visual cue only.

### r8.15 PriorityScore composite

The order-entry sheet now includes a `PriorityScore` (0-100) composite of Conviction (50%), EV/ML (30%), and √POP (20%), and is sorted by PriorityScore descending. Backtest at 5d horizon (n=849, 2026-03-20 → 2026-04-15):

- Conviction >= 65 alone: 90% hit rate, mean +$218/lot.
- Conviction 55-64: 59% hit, +$119/lot.
- Conviction < 55: 58% hit, +$52/lot.

Conviction is the strongest single discriminator at the upper tier; PriorityScore extends this by combining all three orthogonal signals.

### r8.16 Audit JSON additions

Audit JSON now exposes:

- `portfolio_caps_dropped_rows` and `portfolio_caps_drop_reasons`
- `vol_regime_blocked_rows`
- `confident_buy_rows`
- `audit_fix_revision: r8-2026-04-26`
