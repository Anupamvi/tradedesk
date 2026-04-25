# Anu Options Engine — RULEBOOK (Audited FIRE + SHIELD No-GEX Payload)

**Canonical upload filename:** `Anu_Options_Engine_RULEBOOK_v3_0_6.md`  
**Effective logic version:** 3.2.7-full-source-routing-audit  
**Date:** 2026-04-25
**Revision:** r6 mixed-flow rescue + scan-date OI default

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
