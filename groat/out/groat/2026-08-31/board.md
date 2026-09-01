# Groat 2026-08-31

Regime **strong_risk_on** · TRADE 7 · WATCH 10 · FIRE 4 · X-HOT 10

You click every Schwab order. Empty board is valid. Prefer 1–3 names.

## Desk pick

**Take options: XLE** — BUY 65.0 call / SELL 75.0 call 2026-10-16. Pay **debit 1.67**. Naive POP **36%**, conf **71**.

Why this one: long strike is near the money (1.5% from last); net delta 0.36; naive POP 36%; conf 71; X Informed. Setup E (Emerging Sector Rotation). Sub-50% naive POP is normal for an OTM/near-OTM debit — conf is structure quality, not P(win).

Act: work the fill at or inside the stated debit/credit. 1 lot first if X is Crowded. Invalidation: thesis/setup break.

Why this one, not the others:

- **XLE** **← take this** — long strike is near the money (1.5% from last); net delta 0.36; naive POP 36%; conf 71; X Informed.
- **CVX** IN BOOK — long strike is near the money (1.5% from last); net delta 0.18; naive POP 38%; conf 75; X Informed.
- **SHOP** — long strike is near the money (1.7% from last); net delta 0.15; naive POP 42%; conf 85; X Informed.

**Stock if you want one: ADBE** — buy ~293.38, stop **266.57**, target **347.02**, 18 shares. Setup Emerging Sector Rotation.

Analog caution (stock setup avg R < 0, n≥5): XLE, NET.

Take **one** options ticket unless you explicitly want two uncorrelated names. Prefer 1–3 positions total.

## Evidence — same setup, this ticker

Same ticker + same setup on cached tape — not this ticket's P(win). Options use hist/strikes when cached or under the daily cap. Option P&L is delta+theta clamped to defined risk, not a live exit mark. Not a system win rate. X-HOT is not tested. Does not change today's gates.

| ticker | setup | stock n | W/L/time | avg R | options n | opt P&L / risk | BE vs naive POP |
|---|---|---:|---|---:|---:|---:|---|
| **XLE** ⚠ | E Emerging Sector Rotation | 8 | 0/5/3 | -0.31 | 0 | — | — |
| **SHOP** | E Emerging Sector Rotation | 3 | 1/2/0 | 0.00 | 1 | -0.52 | 0% vs 45% |
| **CVX** | E Emerging Sector Rotation | 6 | 2/3/1 | 0.28 | 4 | -0.08 | 25% vs 39% |
| **ADBE** | E Emerging Sector Rotation | 1 | 0/1/0 | -1.00 | 0 | — | — |
| **NET** ⚠ | E Emerging Sector Rotation | 6 | 1/4/1 | -0.22 | 0 | — | — |
| **SNOW** | E Emerging Sector Rotation | 4 | 2/0/2 | 1.48 | 0 | — | — |
| **ANET** | E Emerging Sector Rotation | 6 | 1/3/2 | 0.02 | 0 | — | — |

| ticker | analog dates (entry → exit / result / R) |
|---|---|
| **XLE** | 2026-08-10→2026-08-31 time 1.42R; 2026-07-17→2026-08-07 time -0.08R; 2026-06-10→2026-06-15 loss -1.00R; 2026-05-18→2026-05-27 loss -1.00R; 2026-02-05→2026-02-27 time 1.17R; 2025-12-11→2025-12-16 loss -1.00R |
| **SHOP** | 2025-12-29→2026-01-02 loss -1.00R; 2025-10-14→2025-10-28 win 2.00R; 2025-09-22→2025-09-25 loss -1.00R |
| **CVX** | 2026-08-07→2026-08-18 win 2.00R; 2026-07-16→2026-08-06 time 0.69R; 2026-06-10→2026-06-15 loss -1.00R; 2026-05-18→2026-05-26 loss -1.00R; 2026-01-22→2026-02-04 win 2.00R; 2025-09-17→2025-09-30 loss -1.00R |
| **ADBE** | 2025-12-29→2026-01-02 loss -1.00R |
| **NET** | 2026-08-17→2026-08-25 loss -1.00R; 2026-07-24→2026-08-07 win 2.00R; 2026-06-11→2026-07-06 time 0.68R; 2026-05-07→2026-05-08 loss -1.00R; 2026-03-26→2026-04-09 loss -1.00R; 2025-10-02→2025-10-16 loss -1.00R |
| **SNOW** | 2026-07-24→2026-08-04 win 2.00R; 2026-06-11→2026-07-06 time 0.75R; 2025-10-14→2025-11-03 win 2.00R; 2025-09-22→2025-10-13 time 1.16R |
| **ANET** | 2026-04-30→2026-05-06 loss -1.00R; 2026-04-08→2026-04-22 win 2.00R; 2026-03-05→2026-03-26 loss -1.00R; 2026-01-28→2026-02-04 loss -1.00R; 2025-10-13→2025-11-03 time 0.71R; 2025-09-19→2025-10-10 time 0.41R |

## How to read this

| field | what it is | what it is not |
|---|---|---|
| **setup C** | Post-earnings drift: print already out; price holds earnings AVWAP. | Not an earnings lottery. |
| **setup E** | Emerging sector rotation: the *group* is attracting capital; pick a leader in it. | Not “the stock is a great company.” |
| **conf** | 0–85 quality of the *option structure* (quotes, OI, IV, earnings distance, X). | **Not** probability of profit. |
| **naive POP** | P(spot beyond breakeven) from ORATS call deltas. | **Not** a backtested win rate. Stock = n/a. |
| **score** | Rank of the underlying idea. | Not POP. |
| **FIRE** | Tape first: volume + 1–2 day shock, then X confirms or vetoes. | Not “it’s loud on X.” |
| **X-HOT** | Conversation first: loud on X, then tape says dipped / will_rise / will_dip. | Not a trade by itself. Heat without a trigger is Watch. |
| **evidence** | Same ticker + same setup on cached tape; options via hist/strikes if cached/capped. | **Not** a system win rate. Does not change today’s gates. |

Other setups: **A** Trend pullback; **B** Breakout; **D** Relative-strength leader; **F** Oversold reversal; **G** Failed breakout / breakdown; **H** FIRE spike/dip

## TRADE — index

| ticker | setup | vehicle | pay/collect | naive POP | conf | last | book |
|---|---|---|---|---:|---:|---:|---|
| **SHOP** | E Emerging Sector Rotation | **OPTIONS** | debit 4.00 | 42% | 85 | 147.47 | — |
| **ADBE** | E Emerging Sector Rotation | **STOCK** | — | n/a | — | 293.38 | — |
| **XLE** | E Emerging Sector Rotation | **OPTIONS** | debit 1.67 | 36% | 71 | 64.07 | — |
| **NET** | E Emerging Sector Rotation | **STOCK** | — | n/a | — | 306.72 | — |
| **SNOW** | E Emerging Sector Rotation | **STOCK** | — | n/a | — | 332.86 | — |
| **CVX** | E Emerging Sector Rotation | **OPTIONS** | debit 3.76 | 38% | 75 | 206.87 | IN BOOK |
| **ANET** | E Emerging Sector Rotation | **STOCK** | — | n/a | — | 196.84 | — |

| ticker | exact structure | stop / invalidation |
|---|---|---|
| **SHOP** | debit_call_spread BUY 150.0 call / SELL 160.0 call 2026-10-16 | thesis/setup break |
| **ADBE** | STOCK long entry 293.38 stop 266.57 target 347.02 R/R 2.00 shares 18 | close beyond stop 266.57 |
| **XLE** | debit_call_spread BUY 65.0 call / SELL 75.0 call 2026-10-16 | thesis/setup break |
| **NET** | STOCK long entry 306.72 stop 274.52 target 371.13 R/R 2.00 shares 15 | close beyond stop 274.52 |
| **SNOW** | STOCK long entry 332.86 stop 307.03 target 384.52 R/R 2.00 shares 19 | close beyond stop 307.03 |
| **CVX** | debit_call_spread BUY 210.0 call / SELL 220.0 call 2026-10-16 | thesis/setup break |
| **ANET** | STOCK long entry 196.84 stop 179.92 target 230.67 R/R 2.00 shares 29 | close beyond stop 179.92 |

## WATCH

| ticker | setup | last | RS20 | score | parked because |
|---|---|---:|---:|---:|---|
| CRWD | C | 229.57 | 12.04% | 57.0 | setup_C_replay_park |
| PLTR | E | 186.55 | 47.16% | 51.0 | below_trade_score |
| XLK | E | 186.50 | 3.45% | 50.0 | below_trade_score |
| XOM | E | 161.20 | 2.66% | 50.0 | below_trade_score |
| VRTX | A | 545.80 | 14.65% | 47.0 | below_trade_score |
| URNM | D | 56.80 | 10.68% | 47.0 | below_trade_score |
| XBI | A | 162.31 | 8.88% | 47.0 | below_trade_score |
| PANW | A | 381.78 | 8.68% | 47.0 | below_trade_score |
| CIBR | A | 100.29 | 5.49% | 47.0 | below_trade_score |
| CRM | C | 258.06 | 37.47% | 45.0 | below_trade_score |

## FIRE — spike / dip

Needs volume + a 1–2 day shock **first**. X only confirms or vetoes.

| ticker | kind | 1d | rvol | vehicle | pay/collect | board | X |
|---|---|---:|---:|---|---|---|---|
| **CRWD** | spike | 5.11% | 1.9 | **OPTIONS** | debit 3.95 | WATCH · setup_C_replay_park | Informed |
| **TSLA** | spike | 5.35% | 1.8 | **OPTIONS** | debit 3.85 | IGNORE · shortlisted debit_call_spread after revi | Crowded |
| **DE** | spike | 4.07% | 1.5 | **STOCK** | — | WATCH · below_trade_score | Informed |
| **GEV** | dip | -1.42% | 1.8 | **STOCK** | — | IGNORE · stock won the shortlist versus priced op | Informed |

FIRE names are not auto-TRADE. Parked/IGNORE rows still show the ticket for visibility.

## X-HOT — conversation first

Starts from what is loud on X, then asks the tape. **dipped** = already red with heat (buy-the-dip only if 20 EMA/AVWAP holds). **will_rise** = bullish X and tape allows continuation or a later entry. **will_dip** = bearish X, or a spike already extended (the trade is the pullback, not the chase). Heat without a volume/price trigger is Watch, not a trade.

| ticker | move | tape | 1d | rvol | X | vehicle |
|---|---|---|---:|---:|---|---|
| **TSLA** | **will_rise** | spike | 5.35% | 1.8 | Crowded / bullish | OPTIONS |
| **CRWD** | **will_rise** | spike | 5.11% | 1.9 | Informed / bullish | OPTIONS |
| **AMZN** | **dipped** | soft_dip | -2.34% | 1.3 | Informed / unknown | OPTIONS |
| **GOOGL** | **dipped** | soft_dip | -2.01% | 1.4 | Informed / unknown | STOCK |
| **SPY** | **dipped** | soft_dip | -0.23% | 1.0 | Crowded / unknown | STOCK |
| **GLD** | **dipped** | soft_dip | -0.17% | 0.8 | Informed / bullish | STOCK |
| **XOM** | **will_rise** | heat_only | 2.87% | 1.3 | Informed / bullish | OPTIONS |
| **CVX** | **will_rise** | heat_only | 2.48% | 1.5 | Informed / bullish | OPTIONS |
| **MU** | **will_rise** | heat_only | 2.47% | 0.8 | Informed / bullish | STOCK |
| **XLE** | **will_rise** | heat_only | 2.22% | 1.0 | Informed / bullish | OPTIONS |

| ticker | play | X narrative |
|---|---|---|
| **TSLA** | X-hot continuation — defined-risk call debit or stock, 1 lot if Crowded | Close recap ~+4.9% isolated bid vs tape. 0DTE call lotto board (362.5C +1,279%). Crowded rip — do not chase. |
| **CRWD** | X-hot continuation — defined-risk call debit or stock, 1 lot if Crowded | Fal.Con: tagged new ATH in close posts. Already ran. Informed, do not chase. |
| **AMZN** | X-hot, soft red day — watch for a hold of 20 EMA/AVWAP before buying | Mega lag; 0DTE put lotto board (260P / 257.5P). Soft day, not a clean dip. |
| **GOOGL** | X-hot, soft red day — watch for a hold of 20 EMA/AVWAP before buying | Mega lag on oil/yield day. Mixed, not a clean dip. |
| **SPY** | X-hot, soft red day — watch for a hold of 20 EMA/AVWAP before buying | Month-end close ~-0.4%. Oil + hike-odds grind. Not a crash. Crowded. |
| **GLD** | X-hot, soft red day — watch for a hold of 20 EMA/AVWAP before buying | Geo/oil hedge. Not a squeeze. |
| **XOM** | X says up — wait for a dip into 20 EMA/AVWAP or a real volume spike to enter | Close recaps +1.6% to +2.7%. Crude beta. Venezuela field-takeover chatter with CVX. Session-high faded. |
| **CVX** | X says up — wait for a dip into 20 EMA/AVWAP or a real volume spike to enter | EOD +~2% on oil. Venezuela agreements (ONGC/Eni/GEV) still in the tape. Open spike faded into the close — group bid, not a chase. |
| **MU** | X says up — wait for a dip into 20 EMA/AVWAP or a real volume spike to enter | Memory bottleneck after NVDA prepaid supply. Still named with NVDA. Informed. |
| **XLE** | X says up — wait for a dip into 20 EMA/AVWAP or a real volume spike to enter | EOD: oil >$90, XLE tagged new all-time closing highs. Group bid on Hormuz; session faded the open spike but held green. Informed, not a squeeze. |

## Tickets

---

### SHOP · TRADE · **OPTIONS**

SHOP — bullish Emerging Sector Rotation in accelerating software

SHOP is a bullish idea in a strong risk on tape because Emerging Sector Rotation (E), 20d RS vs SPY 24.72%, software group is accelerating.

| | |
|---|---|
| **Setup** | E — Emerging Sector Rotation |
| **Lane** | SWING / bullish |
| **Last** | 147.47 |
| **Trade** | debit_call_spread BUY 150.0 call / SELL 160.0 call 2026-10-16 |
| **Pay / collect** | debit 4.00 |
| **Naive POP** | 42% — naive P(spot > breakeven 154.00) from ORATS call deltas. Not a backtested win rate. |
| **Conf** | 85 high (quality of the *structure*, not P(win)) |
| **Score** | 55.0 |
| **Evidence** | stock 1W/2L/0t n=3 avg R 0.00; options n=1 P&L/risk -0.52 (spot-delta, not a live mark) |
| **Stop / invalidation** | close back below 20 EMA / swing-low AVWAP |
| **RS 20d vs SPY** | 24.72% |
| **Trend / MAs** | strong_up · 20 146.23 / 50 131.53 / 200 131.39 |
| **Group** | software (IGV) · accelerating |
| **AVWAP** | year 123.26 · swing-low 131.14 |
| **ORATS IV/HV** | IV30 43.40 · HV20 55.53 · VRP -12.13 |
| **Earnings** | 2026-11-03 (web.alphaquery) |
| **X** | Informed — Flow: ~$498k 120P 10/16 (18% OTM). Gap-down list. Software hide-out / 9 EMA break, post-earnings gap unfilled. One TA: bullish flag, buy pullback to 8 EMA or reclaim $154. Informed, not crowded. |
| **Fill** | long ask minus short bid (never mid) · schwab_quote |
| **Fill as-of** | 2026-08-31 19:56 ET |
| **Greeks** | Δ 0.15 · Γ 0.0020 · Θ -0.0070 · ν 0.0190 |

More context:

Price is 147.47, trend strong_up, 20 EMA 146.23 / 50 131.53 / 200 131.39, relative volume 0.7.

Group software (IGV) is accelerating. AVWAP year 123.26, swing-low 131.14.

Earnings 2026-11-03 (web.alphaquery).

Instrument shortlist picked **OPTIONS**. Invalidation: close back below 20 EMA / swing-low AVWAP.

Tape notes:

- group accelerating with positive 20d RS
- trend pullback into 20 EMA / AVWAP / 50 DMA
- volume contracted on the pullback
- 20d RS vs SPY +24.7% with accumulation structure

All structures reviewed:

| structure | result | debit | credit | why |
|---|---|---:|---:|---|
| stock | PASS | — | — | stock plan clears R/R |
| long_call | REJECT | — | — | quotes too wide for a conservative fill |
| long_put | REJECT | — | — | against bullish underlying thesis |
| debit_call_spread | PASS | 4.00 | — | defined-risk directional; better than naked long when IV is not cheap |
| debit_put_spread | REJECT | 4.25 | — | against bullish underlying thesis |
| put_credit_spread | REJECT | — | — | no liquid structure in 21-75 DTE |
| call_credit_spread | REJECT | — | — | against bullish underlying thesis |

Macro in hold window: 2026-09-04 Employment Situation; 2026-09-10 PPI; 2026-09-11 CPI; 2026-09-16 FOMC decision

---

### ADBE · TRADE · **STOCK**

ADBE — bullish Emerging Sector Rotation in accelerating software

ADBE is a bullish idea in a strong risk on tape because Emerging Sector Rotation (E), 20d RS vs SPY 15.42%, software group is accelerating.

| | |
|---|---|
| **Setup** | E — Emerging Sector Rotation |
| **Lane** | SWING / bullish |
| **Last** | 293.38 |
| **Trade** | STOCK long entry 293.38 stop 266.57 target 347.02 R/R 2.00 shares 18 |
| **Pay / collect** | stock (no option premium) |
| **Naive POP** | n/a — n/a for stock; no delta-based POP |
| **Conf** | n/a n/a (quality of the *structure*, not P(win)) |
| **Score** | 55.0 |
| **Evidence** | stock 0W/1L/0t n=1 avg R -1.00 |
| **Stop / invalidation** | close beyond stop 266.57 |
| **RS 20d vs SPY** | 15.42% |
| **Trend / MAs** | up · 20 270.22 / 50 242.14 / 200 268.98 |
| **Group** | software (IGV) · accelerating |
| **AVWAP** | year 250.46 · swing-low 235.15 |
| **ORATS IV/HV** | IV30 51.98 · HV20 43.70 · VRP 8.28 |
| **Earnings** | 2026-09-10 (web.alphaquery) — ordinary options blocked |
| **X** | Informed — Value long-pitch: ~11x fwd, AI-first ARR >$500M, Firefly ARR ~$300M. Earnings Thursday AMC after Labor Day; recent distribution skewed downside, avg peak move ~9.2%. Options blocked through the print anyway. |
| **Fill** | stock last; options never mid |
| **Fill as-of** | n/a — revalidate at the open |
| **Greeks** | Δ DATA UNAVAILABLE · Γ DATA UNAVAILABLE · Θ DATA UNAVAILABLE · ν DATA UNAVAILABLE |

More context:

Price is 293.38, trend up, 20 EMA 270.22 / 50 242.14 / 200 268.98, relative volume 1.2.

Group software (IGV) is accelerating. AVWAP year 250.46, swing-low 235.15.

Earnings 2026-09-10 (web.alphaquery); ordinary options blocked through the print.

Instrument shortlist picked **STOCK**. Invalidation: close beyond stop 266.57.

Tape notes:

- group accelerating with positive 20d RS
- close above prior 20-session close high
- 20d RS vs SPY +15.4% with accumulation structure

All structures reviewed:

| structure | result | debit | credit | why |
|---|---|---:|---:|---|
| stock | PASS | — | — | stock plan clears R/R |
| long_call | REJECT | — | — | earnings inside intended hold — ordinary options rejected (not an EVENT TRADE) |
| long_put | REJECT | — | — | earnings inside intended hold — ordinary options rejected (not an EVENT TRADE) |
| debit_call_spread | REJECT | — | — | earnings inside intended hold — ordinary options rejected (not an EVENT TRADE) |
| debit_put_spread | REJECT | — | — | earnings inside intended hold — ordinary options rejected (not an EVENT TRADE) |
| put_credit_spread | REJECT | — | — | earnings inside intended hold — ordinary options rejected (not an EVENT TRADE) |
| call_credit_spread | REJECT | — | — | earnings inside intended hold — ordinary options rejected (not an EVENT TRADE) |

Macro in hold window: 2026-09-04 Employment Situation; 2026-09-10 PPI; 2026-09-11 CPI; 2026-09-16 FOMC decision

---

### XLE · TRADE · **OPTIONS**

XLE — bullish Emerging Sector Rotation in emerging energy

XLE is a bullish idea in a strong risk on tape because Emerging Sector Rotation (E), 20d RS vs SPY 7.68%, energy group is emerging.

| | |
|---|---|
| **Setup** | E — Emerging Sector Rotation |
| **Lane** | SWING / bullish |
| **Last** | 64.07 |
| **Trade** | debit_call_spread BUY 65.0 call / SELL 75.0 call 2026-10-16 |
| **Pay / collect** | debit 1.67 |
| **Naive POP** | 36% — naive P(spot > breakeven 66.67) from ORATS call deltas. Not a backtested win rate. |
| **Conf** | 71 high (quality of the *structure*, not P(win)) |
| **Score** | 55.0 |
| **Evidence** | stock 0W/5L/3t n=8 avg R -0.31; weak analog |
| **Stop / invalidation** | close back below 20 EMA / swing-low AVWAP |
| **RS 20d vs SPY** | 7.68% |
| **Trend / MAs** | strong_up · 20 61.73 / 50 58.36 / 200 54.45 |
| **Group** | energy (XLE) · emerging |
| **AVWAP** | year 55.91 · swing-low 58.82 |
| **ORATS IV/HV** | IV30 24.97 · HV20 23.08 · VRP 1.89 |
| **Earnings** | DATA UNAVAILABLE (exempt) |
| **X** | Informed — THE open: WTI ~$86 +3%, Brent ~$89 after US Hormuz/Larak strikes and Iran reply. Energy the only sector up >2%. Group bid, not a single-name story. |
| **Fill** | long ask minus short bid (never mid) · schwab_quote |
| **Fill as-of** | 2026-08-31 19:59 ET |
| **Greeks** | Δ 0.36 · Γ 0.0520 · Θ -0.0130 · ν 0.0620 |

More context:

Price is 64.07, trend strong_up, 20 EMA 61.73 / 50 58.36 / 200 54.45, relative volume 1.0.

Group energy (XLE) is emerging. AVWAP year 55.91, swing-low 58.82.

Earnings DATA UNAVAILABLE (exempt).

Instrument shortlist picked **OPTIONS**. Invalidation: close back below 20 EMA / swing-low AVWAP.

Tape notes:

- group emerging with positive 20d RS
- close above prior 20-session close high
- 20d RS vs SPY +7.7% with accumulation structure

All structures reviewed:

| structure | result | debit | credit | why |
|---|---|---:|---:|---|
| stock | PASS | — | — | stock plan clears R/R |
| long_call | PASS | 2.79 | — | cheap-to-fair vol directional with 21-75 DTE |
| long_put | REJECT | 3.05 | — | against bullish underlying thesis |
| debit_call_spread | PASS | 1.67 | — | defined-risk directional; better than naked long when IV is not cheap |
| debit_put_spread | REJECT | 2.01 | — | against bullish underlying thesis |
| put_credit_spread | REJECT | — | — | no liquid structure in 21-75 DTE |
| call_credit_spread | REJECT | — | — | against bullish underlying thesis |

Macro in hold window: 2026-09-04 Employment Situation; 2026-09-10 PPI; 2026-09-11 CPI; 2026-09-16 FOMC decision

---

### NET · TRADE · **STOCK**

NET — bullish Emerging Sector Rotation in accelerating software

NET is a bullish idea in a strong risk on tape because Emerging Sector Rotation (E), 20d RS vs SPY 7.18%, software group is accelerating.

| | |
|---|---|
| **Setup** | E — Emerging Sector Rotation |
| **Lane** | SWING / bullish |
| **Last** | 306.72 |
| **Trade** | STOCK long entry 306.72 stop 274.52 target 371.13 R/R 2.00 shares 15 |
| **Pay / collect** | stock (no option premium) |
| **Naive POP** | n/a — n/a for stock; no delta-based POP |
| **Conf** | n/a n/a (quality of the *structure*, not P(win)) |
| **Score** | 55.0 |
| **Evidence** | stock 1W/4L/1t n=6 avg R -0.22; weak analog |
| **Stop / invalidation** | close beyond stop 274.52 |
| **RS 20d vs SPY** | 7.18% |
| **Trend / MAs** | strong_up · 20 293.61 / 50 275.70 / 200 221.01 |
| **Group** | software (IGV) · accelerating |
| **AVWAP** | year 220.70 · swing-low 278.67 |
| **ORATS IV/HV** | IV30 53.54 · HV20 66.62 · VRP -13.08 |
| **Earnings** | 2026-10-29 (web.alphaquery) |
| **X** | Crowded — $NET X is mostly Robinhood-chain crypto/RPG spam (5k/10k bags), not Cloudflare. Downweight. No Cloudflare catalyst in the last 24h. |
| **Fill** | stock last; options never mid |
| **Fill as-of** | n/a — revalidate at the open |
| **Greeks** | Δ DATA UNAVAILABLE · Γ DATA UNAVAILABLE · Θ DATA UNAVAILABLE · ν DATA UNAVAILABLE |

More context:

Price is 306.72, trend strong_up, 20 EMA 293.61 / 50 275.70 / 200 221.01, relative volume 0.9.

Group software (IGV) is accelerating. AVWAP year 220.70, swing-low 278.67.

Earnings 2026-10-29 (web.alphaquery).

Instrument shortlist picked **STOCK**. Invalidation: close beyond stop 274.52.

Tape notes:

- group accelerating with positive 20d RS
- trend pullback into 20 EMA / AVWAP / 50 DMA
- volume contracted on the pullback
- 20d RS vs SPY +7.2% with accumulation structure

All structures reviewed:

| structure | result | debit | credit | why |
|---|---|---:|---:|---|
| stock | PASS | — | — | stock plan clears R/R |
| long_call | REJECT | — | — | debit too large to size 1 lot at 1% of the 50k account |
| long_put | REJECT | — | — | against bullish underlying thesis |
| debit_call_spread | REJECT | — | — | open interest below minimum |
| debit_put_spread | REJECT | — | — | against bullish underlying thesis |
| put_credit_spread | REJECT | — | — | no liquid structure in 21-75 DTE |
| call_credit_spread | REJECT | — | — | against bullish underlying thesis |

Macro in hold window: 2026-09-04 Employment Situation; 2026-09-10 PPI; 2026-09-11 CPI; 2026-09-16 FOMC decision

---

### SNOW · TRADE · **STOCK**

SNOW — bullish Emerging Sector Rotation in accelerating software

SNOW is a bullish idea in a strong risk on tape because Emerging Sector Rotation (E), 20d RS vs SPY 6.93%, software group is accelerating.

| | |
|---|---|
| **Setup** | E — Emerging Sector Rotation |
| **Lane** | SWING / bullish |
| **Last** | 332.86 |
| **Trade** | STOCK long entry 332.86 stop 307.03 target 384.52 R/R 2.00 shares 19 |
| **Pay / collect** | stock (no option premium) |
| **Naive POP** | n/a — n/a for stock; no delta-based POP |
| **Conf** | n/a n/a (quality of the *structure*, not P(win)) |
| **Score** | 55.0 |
| **Evidence** | stock 2W/0L/2t n=4 avg R 1.48 |
| **Stop / invalidation** | close beyond stop 307.03 |
| **RS 20d vs SPY** | 6.93% |
| **Trend / MAs** | strong_up · 20 319.40 / 50 289.16 / 200 217.54 |
| **Group** | software (IGV) · accelerating |
| **AVWAP** | year 205.41 · swing-low 285.31 |
| **ORATS IV/HV** | IV30 68.50 · HV20 45.38 · VRP 23.12 |
| **Earnings** | 2026-09-02 (web.alphaquery) — ordinary options blocked |
| **X** | Informed — On RS/breakout lists. Earnings this week (Labor Day week). Ticker-dump mix. Informed, not a fresh print — ordinary options should stay blocked if the print is inside the hold. |
| **Fill** | stock last; options never mid |
| **Fill as-of** | n/a — revalidate at the open |
| **Greeks** | Δ DATA UNAVAILABLE · Γ DATA UNAVAILABLE · Θ DATA UNAVAILABLE · ν DATA UNAVAILABLE |

More context:

Price is 332.86, trend strong_up, 20 EMA 319.40 / 50 289.16 / 200 217.54, relative volume 1.1.

Group software (IGV) is accelerating. AVWAP year 205.41, swing-low 285.31.

Earnings 2026-09-02 (web.alphaquery); ordinary options blocked through the print.

Instrument shortlist picked **STOCK**. Invalidation: close beyond stop 307.03.

Tape notes:

- group accelerating with positive 20d RS
- trend pullback into 20 EMA / AVWAP / 50 DMA
- volume contracted on the pullback
- 20d RS vs SPY +6.9% with accumulation structure

All structures reviewed:

| structure | result | debit | credit | why |
|---|---|---:|---:|---|
| stock | PASS | — | — | stock plan clears R/R |
| long_call | REJECT | — | — | earnings inside intended hold — ordinary options rejected (not an EVENT TRADE) |
| long_put | REJECT | — | — | earnings inside intended hold — ordinary options rejected (not an EVENT TRADE) |
| debit_call_spread | REJECT | — | — | earnings inside intended hold — ordinary options rejected (not an EVENT TRADE) |
| debit_put_spread | REJECT | — | — | earnings inside intended hold — ordinary options rejected (not an EVENT TRADE) |
| put_credit_spread | REJECT | — | — | earnings inside intended hold — ordinary options rejected (not an EVENT TRADE) |
| call_credit_spread | REJECT | — | — | earnings inside intended hold — ordinary options rejected (not an EVENT TRADE) |

Macro in hold window: 2026-09-04 Employment Situation; 2026-09-10 PPI; 2026-09-11 CPI; 2026-09-16 FOMC decision

---

### CVX · TRADE · **OPTIONS**

CVX — bullish Emerging Sector Rotation in emerging energy

CVX is a bullish idea in a strong risk on tape because Emerging Sector Rotation (E), 20d RS vs SPY 5.78%, energy group is emerging.

| | |
|---|---|
| **Setup** | E — Emerging Sector Rotation |
| **Lane** | SWING / bullish |
| **Last** | 206.87 |
| **Trade** | debit_call_spread BUY 210.0 call / SELL 220.0 call 2026-10-16 |
| **Pay / collect** | debit 3.76 |
| **Naive POP** | 38% — naive P(spot > breakeven 213.76) from ORATS call deltas. Not a backtested win rate. |
| **Conf** | 75 high (quality of the *structure*, not P(win)) |
| **Score** | 55.0 |
| **Evidence** | stock 2W/3L/1t n=6 avg R 0.28; options n=4 P&L/risk -0.08 (spot-delta, not a live mark) |
| **Stop / invalidation** | close back below 20 EMA / swing-low AVWAP |
| **RS 20d vs SPY** | 5.78% |
| **Trend / MAs** | strong_up · 20 199.31 / 50 188.21 / 200 180.02 |
| **Group** | energy (XLE) · emerging |
| **AVWAP** | year 185.58 · swing-low 190.55 |
| **ORATS IV/HV** | IV30 24.99 · HV20 22.96 · VRP 2.03 |
| **Earnings** | 2026-10-30 (web.alphaquery) |
| **X** | Informed — Oil-spike expression: Hormuz/Iran, Brent/WTI bid, tape ~+2%. Venezuela energy-agreement chatter (CVX/ONGC/ENI). Friday $1.1M call flow now marked up. Group bid — do not chase the first hour. |
| **Fill** | long ask minus short bid (never mid) · schwab_quote |
| **Fill as-of** | 2026-08-31 19:55 ET |
| **Greeks** | Δ 0.18 · Γ 0.0030 · Θ -0.0180 · ν 0.0500 |

More context:

Price is 206.87, trend strong_up, 20 EMA 199.31 / 50 188.21 / 200 180.02, relative volume 1.5.

Group energy (XLE) is emerging. AVWAP year 185.58, swing-low 190.55.

Earnings 2026-10-30 (web.alphaquery).

Instrument shortlist picked **OPTIONS**. Invalidation: close back below 20 EMA / swing-low AVWAP.

IN BOOK open: BUY 210.0 call / SELL 220.0 call 2026-09-25 @ 2.73 exp 2026-09-25. Board structure is different — visibility only, not a roll/add.; Schwab holds: CALL 2026-09-25 220.0, CALL 2026-09-25 210.0 Shown for visibility — do not add a second lot unless you have a scale plan.

Tape notes:

- group emerging with positive 20d RS
- close above prior 20-session close high
- 20d RS vs SPY +5.8% with accumulation structure

All structures reviewed:

| structure | result | debit | credit | why |
|---|---|---:|---:|---|
| stock | PASS | — | — | stock plan clears R/R |
| long_call | REJECT | — | — | debit too large to size 1 lot at 1% of the 50k account |
| long_put | REJECT | — | — | against bullish underlying thesis |
| debit_call_spread | PASS | 3.76 | — | defined-risk directional; better than naked long when IV is not cheap |
| debit_put_spread | REJECT | 3.01 | — | against bullish underlying thesis |
| put_credit_spread | REJECT | — | — | no liquid structure in 21-75 DTE |
| call_credit_spread | REJECT | — | — | against bullish underlying thesis |

Macro in hold window: 2026-09-04 Employment Situation; 2026-09-10 PPI; 2026-09-11 CPI; 2026-09-16 FOMC decision

---

### ANET · TRADE · **STOCK**

ANET — bullish Emerging Sector Rotation in accelerating networking

ANET is a bullish idea in a strong risk on tape because Emerging Sector Rotation (E), 20d RS vs SPY 5.16%, networking group is accelerating.

| | |
|---|---|
| **Setup** | E — Emerging Sector Rotation |
| **Lane** | SWING / bullish |
| **Last** | 196.84 |
| **Trade** | STOCK long entry 196.84 stop 179.92 target 230.67 R/R 2.00 shares 29 |
| **Pay / collect** | stock (no option premium) |
| **Naive POP** | n/a — n/a for stock; no delta-based POP |
| **Conf** | n/a n/a (quality of the *structure*, not P(win)) |
| **Score** | 55.0 |
| **Evidence** | stock 1W/3L/2t n=6 avg R 0.02 |
| **Stop / invalidation** | close beyond stop 179.92 |
| **RS 20d vs SPY** | 5.16% |
| **Trend / MAs** | strong_up · 20 192.05 / 50 181.02 / 200 150.76 |
| **Group** | networking (XLK) · accelerating |
| **AVWAP** | year 154.53 · swing-low 181.60 |
| **ORATS IV/HV** | IV30 45.97 · HV20 55.04 · VRP -9.07 |
| **Earnings** | 2026-11-03 (web.alphaquery) |
| **X** | Informed — AI networking watchlists. Three insiders sold this week; Bechtolsheim 300k shares / $60.8M (Form 4 filed Aug 31). Informed, not crowded. Selling is a veto-lean, not a trigger. |
| **Fill** | stock last; options never mid |
| **Fill as-of** | n/a — revalidate at the open |
| **Greeks** | Δ DATA UNAVAILABLE · Γ DATA UNAVAILABLE · Θ DATA UNAVAILABLE · ν DATA UNAVAILABLE |

More context:

Price is 196.84, trend strong_up, 20 EMA 192.05 / 50 181.02 / 200 150.76, relative volume 0.7.

Group networking (XLK) is accelerating. AVWAP year 154.53, swing-low 181.60.

Earnings 2026-11-03 (web.alphaquery).

Instrument shortlist picked **STOCK**. Invalidation: close beyond stop 179.92.

Tape notes:

- group accelerating with positive 20d RS
- trend pullback into 20 EMA / AVWAP / 50 DMA
- volume contracted on the pullback
- 20d RS vs SPY +5.2% with accumulation structure

All structures reviewed:

| structure | result | debit | credit | why |
|---|---|---:|---:|---|
| stock | PASS | — | — | stock plan clears R/R |
| long_call | REJECT | — | — | debit too large to size 1 lot at 1% of the 50k account |
| long_put | REJECT | — | — | against bullish underlying thesis |
| debit_call_spread | REJECT | — | — | quotes too wide for a conservative fill |
| debit_put_spread | REJECT | — | — | against bullish underlying thesis |
| put_credit_spread | REJECT | — | — | no liquid structure in 21-75 DTE |
| call_credit_spread | REJECT | — | — | against bullish underlying thesis |

Macro in hold window: 2026-09-04 Employment Situation; 2026-09-10 PPI; 2026-09-11 CPI; 2026-09-16 FOMC decision

---

### CRWD · WATCH · **OPTIONS**

CRWD — bullish Post-Earnings Drift in mature cybersecurity

CRWD is a bullish idea in a strong risk on tape because Post-Earnings Drift (C), FIRE spike: 1d +5.1% on 1.9x volume, 20d RS vs SPY 12.04%.

| | |
|---|---|
| **Setup** | C — Post-Earnings Drift |
| **Lane** | FIRE / bullish |
| **Last** | 229.57 |
| **Trade** | debit_call_spread BUY 240.0 call / SELL 250.0 call 2026-10-16 |
| **Pay / collect** | debit 3.95 |
| **Naive POP** | 43% — naive P(spot > breakeven 243.95) from ORATS call deltas. Not a backtested win rate. |
| **Conf** | 69 medium (quality of the *structure*, not P(win)) |
| **Score** | 57.0 |
| **Evidence** | no same-setup analog |
| **Stop / invalidation** | close back below 20 EMA / swing-low AVWAP |
| **RS 20d vs SPY** | 12.04% |
| **Trend / MAs** | strong_up · 20 206.63 / 50 197.39 / 200 140.93 |
| **Group** | cybersecurity (CIBR) · mature |
| **AVWAP** | year 136.04 · swing-low 206.56 |
| **ORATS IV/HV** | IV30 51.64 · HV20 71.30 · VRP -19.66 |
| **Earnings** | 2026-12-01 (web.alphaquery) |
| **X** | Informed — Fal.Con: tape ~+3.7–3.8% while indexes red. Software confirmation leftover from last week's print. Already ran — do not chase the open. |
| **Fill** | long ask minus short bid (never mid) · schwab_quote |
| **Fill as-of** | 2026-08-31 19:59 ET |
| **Greeks** | Δ 0.09 · Γ 0.0000 · Θ -0.0090 · ν 0.0140 |

More context:

Price is 229.57, trend strong_up, 20 EMA 206.63 / 50 197.39 / 200 140.93, relative volume 1.9.

Group cybersecurity (CIBR) is mature. AVWAP year 136.04, swing-low 206.56.

Earnings 2026-12-01 (web.alphaquery).

Instrument shortlist picked **OPTIONS**. Invalidation: close back below 20 EMA / swing-low AVWAP.

Tape notes:

- holding earnings AVWAP 5 sessions after lastErn
- close above prior 20-session close high
- 20d RS vs SPY +12.0% with accumulation structure
- FIRE spike: 1d +5.1% on 1.9x volume

All structures reviewed:

| structure | result | debit | credit | why |
|---|---|---:|---:|---|
| stock | PASS | — | — | stock plan clears R/R |
| long_call | REJECT | — | — | debit too large to size 1 lot at 1% of the 50k account |
| long_put | REJECT | — | — | against bullish underlying thesis |
| debit_call_spread | PASS | 3.95 | — | defined-risk directional; better than naked long when IV is not cheap |
| debit_put_spread | REJECT | — | — | against bullish underlying thesis |
| put_credit_spread | REJECT | — | — | no liquid structure in 21-75 DTE |
| call_credit_spread | REJECT | — | — | against bullish underlying thesis |

Macro in hold window: 2026-09-04 Employment Situation; 2026-09-10 PPI; 2026-09-11 CPI; 2026-09-16 FOMC decision

---

### TSLA · IGNORE · **OPTIONS**

TSLA — bullish Breakout + Confirmation in deteriorating consumer_discretionary

TSLA is a bullish idea in a strong risk on tape because Breakout + Confirmation (B), FIRE spike: 1d +5.4% on 1.8x volume, 20d RS vs SPY 12.77%.

| | |
|---|---|
| **Setup** | B — Breakout + Confirmation |
| **Lane** | FIRE / bullish |
| **Last** | 367.42 |
| **Trade** | debit_call_spread BUY 380.0 call / SELL 390.0 call 2026-10-16 |
| **Pay / collect** | debit 3.85 |
| **Naive POP** | 42% — naive P(spot > breakeven 383.85) from ORATS call deltas. Not a backtested win rate. |
| **Conf** | 38 low (quality of the *structure*, not P(win)) |
| **Score** | 37.0 |
| **Evidence** | no same-setup analog |
| **Stop / invalidation** | close back below 20 EMA / swing-low AVWAP |
| **RS 20d vs SPY** | 12.77% |
| **Trend / MAs** | range · 20 347.25 / 50 359.81 / 200 400.57 |
| **Group** | consumer_discretionary (XLY) · deteriorating |
| **AVWAP** | year 394.56 · swing-low 336.51 |
| **ORATS IV/HV** | IV30 41.28 · HV20 40.40 · VRP 0.88 |
| **Earnings** | 2026-10-28 (web.alphaquery) |
| **X** | Crowded — Open recap: TSLA ~+3.4% while indexes soft. Relative-strength leader, already extended vs the tape. Promo mix + 0DTE call flow lists. |
| **Fill** | long ask minus short bid (never mid) · schwab_quote |
| **Fill as-of** | 2026-08-31 19:59 ET |
| **Greeks** | Δ 0.07 · Γ 0.0000 · Θ -0.0090 · ν 0.0190 |

More context:

Price is 367.42, trend range, 20 EMA 347.25 / 50 359.81 / 200 400.57, relative volume 1.8.

Group consumer_discretionary (XLY) is deteriorating. AVWAP year 394.56, swing-low 336.51.

Earnings 2026-10-28 (web.alphaquery).

Instrument shortlist picked **OPTIONS**. Invalidation: close back below 20 EMA / swing-low AVWAP.

Schwab holds: TSLA Shown for visibility — do not add a second lot unless you have a scale plan.

Tape notes:

- close above prior 20-session close high
- 20d RS vs SPY +12.8% with accumulation structure
- FIRE spike: 1d +5.4% on 1.8x volume

All structures reviewed:

| structure | result | debit | credit | why |
|---|---|---:|---:|---|
| stock | PASS | — | — | stock plan clears R/R |
| long_call | REJECT | — | — | debit too large to size 1 lot at 1% of the 50k account |
| long_put | REJECT | — | — | against bullish underlying thesis |
| debit_call_spread | PASS | 3.85 | — | defined-risk directional; better than naked long when IV is not cheap |
| debit_put_spread | REJECT | 4.50 | — | against bullish underlying thesis |
| put_credit_spread | REJECT | — | — | no liquid structure in 21-75 DTE |
| call_credit_spread | REJECT | — | — | against bullish underlying thesis |

Macro in hold window: 2026-09-04 Employment Situation; 2026-09-10 PPI; 2026-09-11 CPI; 2026-09-16 FOMC decision

---

### DE · WATCH · **STOCK**

DE — bullish Post-Earnings Drift in deteriorating industrials

DE is a bullish idea in a strong risk on tape because Post-Earnings Drift (C), FIRE spike: 1d +4.1% on 1.5x volume, 20d RS vs SPY 7.12%.

| | |
|---|---|
| **Setup** | C — Post-Earnings Drift |
| **Lane** | FIRE / bullish |
| **Last** | 656.00 |
| **Trade** | STOCK long entry 656.00 stop 615.63 target 736.74 R/R 2.00 shares 12 |
| **Pay / collect** | stock (no option premium) |
| **Naive POP** | n/a — n/a for stock; no delta-based POP |
| **Conf** | n/a n/a (quality of the *structure*, not P(win)) |
| **Score** | 45.0 |
| **Evidence** | no same-setup analog |
| **Stop / invalidation** | close beyond stop 615.63 |
| **RS 20d vs SPY** | 7.12% |
| **Trend / MAs** | strong_up · 20 622.64 / 50 612.03 / 200 563.91 |
| **Group** | industrials (XLI) · deteriorating |
| **AVWAP** | year 582.73 · swing-low 614.82 |
| **ORATS IV/HV** | IV30 29.56 · HV20 39.49 · VRP -9.93 |
| **Earnings** | 2026-11-25 (web.alphaquery) |
| **X** | Informed — FIRE spike name. Premarket setup recap: $DE STOPPED. Scanner: cup-with-handle within 2% of pivot, no breakout confirmed. Informed, thin. |
| **Fill** | stock last; options never mid |
| **Fill as-of** | n/a — revalidate at the open |
| **Greeks** | Δ DATA UNAVAILABLE · Γ DATA UNAVAILABLE · Θ DATA UNAVAILABLE · ν DATA UNAVAILABLE |

More context:

Price is 656.00, trend strong_up, 20 EMA 622.64 / 50 612.03 / 200 563.91, relative volume 1.5.

Group industrials (XLI) is deteriorating. AVWAP year 582.73, swing-low 614.82.

Earnings 2026-11-25 (web.alphaquery).

Instrument shortlist picked **STOCK**. Invalidation: close beyond stop 615.63.

Tape notes:

- holding earnings AVWAP 11 sessions after lastErn
- close above prior 20-session close high
- 20d RS vs SPY +7.1% with accumulation structure
- FIRE spike: 1d +4.1% on 1.5x volume

All structures reviewed:

| structure | result | debit | credit | why |
|---|---|---:|---:|---|
| stock | PASS | — | — | stock plan clears R/R |
| long_call | REJECT | — | — | quotes too wide for a conservative fill |
| long_put | REJECT | — | — | against bullish underlying thesis |
| debit_call_spread | REJECT | — | — | quotes too wide for a conservative fill |
| debit_put_spread | REJECT | — | — | against bullish underlying thesis |
| put_credit_spread | REJECT | — | — | no liquid structure in 21-75 DTE |
| call_credit_spread | REJECT | — | — | against bullish underlying thesis |

Macro in hold window: 2026-09-04 Employment Situation; 2026-09-10 PPI; 2026-09-11 CPI; 2026-09-16 FOMC decision

---

### GEV · IGNORE · **STOCK**

GEV — bearish Failed Breakout / Trend Breakdown in deteriorating power

GEV is a bearish idea in a strong risk on tape because Failed Breakout / Trend Breakdown (G), FIRE dip: 2d -5.8% on 1.8x volume, 20d RS vs SPY -12.01%.

| | |
|---|---|
| **Setup** | G — Failed Breakout / Trend Breakdown |
| **Lane** | FIRE / bearish |
| **Last** | 898.95 |
| **Trade** | STOCK short entry 898.95 stop 990.12 target 716.61 R/R 2.00 shares 5 |
| **Pay / collect** | stock (no option premium) |
| **Naive POP** | n/a — n/a for stock; no delta-based POP |
| **Conf** | n/a n/a (quality of the *structure*, not P(win)) |
| **Score** | 4.0 |
| **Evidence** | no same-setup analog |
| **Stop / invalidation** | close beyond stop 990.12 |
| **RS 20d vs SPY** | -12.01% |
| **Trend / MAs** | range · 20 973.46 / 50 1027.43 / 200 879.05 |
| **Group** | power (GRID) · deteriorating |
| **AVWAP** | year 923.38 · swing-low 980.52 |
| **ORATS IV/HV** | IV30 43.07 · HV20 44.60 · VRP -1.53 |
| **Earnings** | 2026-10-28 (web.alphaquery) |
| **X** | Informed — AI power/turbine/transformer lists with CAT/ETN/VRT. Q2 rev $11.1B +22% YoY in one recap. Informed, not a dump print. |
| **Fill** | stock last; options never mid |
| **Fill as-of** | n/a — revalidate at the open |
| **Greeks** | Δ DATA UNAVAILABLE · Γ DATA UNAVAILABLE · Θ DATA UNAVAILABLE · ν DATA UNAVAILABLE |

More context:

Price is 898.95, trend range, 20 EMA 973.46 / 50 1027.43 / 200 879.05, relative volume 1.8.

Group power (GRID) is deteriorating. AVWAP year 923.38, swing-low 980.52.

Earnings 2026-10-28 (web.alphaquery).

Instrument shortlist picked **STOCK**. Invalidation: close beyond stop 990.12.

Tape notes:

- oversold reversal evidence (range close / 20 EMA reclaim), not RSI-only
- failed to hold the 20-day high with negative RS
- FIRE dip: 2d -5.8% on 1.8x volume

All structures reviewed:

| structure | result | debit | credit | why |
|---|---|---:|---:|---|
| stock | PASS | — | — | stock plan clears R/R |
| long_call | REJECT | — | — | against bearish underlying thesis |
| long_put | REJECT | — | — | debit too large to size 1 lot at 1% of the 50k account |
| debit_call_spread | REJECT | — | — | against bearish underlying thesis |
| debit_put_spread | REJECT | — | — | open interest below minimum |
| put_credit_spread | REJECT | — | — | against bearish underlying thesis |
| call_credit_spread | REJECT | — | — | no liquid structure in 21-75 DTE |

Macro in hold window: 2026-09-04 Employment Situation; 2026-09-10 PPI; 2026-09-11 CPI; 2026-09-16 FOMC decision

---

### AMZN · IGNORE · **OPTIONS**

AMZN — bullish Trend Pullback in DATA UNAVAILABLE megacap

AMZN is a bullish idea in a strong risk on tape because Trend Pullback (A), 20d RS vs SPY -9.69%.

| | |
|---|---|
| **Setup** | A — Trend Pullback |
| **Lane** | SWING / bullish |
| **Last** | 260.20 |
| **Trade** | debit_call_spread BUY 265.0 call / SELL 275.0 call 2026-10-16 |
| **Pay / collect** | debit 4.15 |
| **Naive POP** | 40% — naive P(spot > breakeven 269.15) from ORATS call deltas. Not a backtested win rate. |
| **Conf** | 56 medium (quality of the *structure*, not P(win)) |
| **Score** | 23.0 |
| **Evidence** | no same-setup analog |
| **Stop / invalidation** | close back below 20 EMA / swing-low AVWAP |
| **RS 20d vs SPY** | -9.69% |
| **Trend / MAs** | up · 20 260.91 / 50 251.98 / 200 238.74 |
| **Group** | megacap (QQQ) · DATA UNAVAILABLE |
| **AVWAP** | year 238.59 · swing-low 251.06 |
| **ORATS IV/HV** | IV30 28.02 · HV20 28.95 · VRP -0.93 |
| **Earnings** | 2026-10-29 (web.alphaquery) |
| **X** | Quiet — Mega lag with GOOGL ~-2%. Isolated $260-node long posts. Not the conversation. |
| **Fill** | long ask minus short bid (never mid) · schwab_quote |
| **Fill as-of** | 2026-08-31 19:59 ET |
| **Greeks** | Δ 0.13 · Γ 0.0010 · Θ -0.0160 · ν 0.0350 |

More context:

Price is 260.20, trend up, 20 EMA 260.91 / 50 251.98 / 200 238.74, relative volume 1.3.

Group megacap (QQQ) is DATA UNAVAILABLE. AVWAP year 238.59, swing-low 251.06.

Earnings 2026-10-29 (web.alphaquery).

Instrument shortlist picked **OPTIONS**. Invalidation: close back below 20 EMA / swing-low AVWAP.

Schwab holds: PUT 2026-11-20 230.0, AMZN Shown for visibility — do not add a second lot unless you have a scale plan.

Tape notes:

- trend pullback into 20 EMA / AVWAP / 50 DMA
- volume contracted on the pullback
- failed to hold the 20-day high with negative RS

All structures reviewed:

| structure | result | debit | credit | why |
|---|---|---:|---:|---|
| stock | PASS | — | — | stock plan clears R/R |
| long_call | REJECT | — | — | debit too large to size 1 lot at 1% of the 50k account |
| long_put | REJECT | — | — | against bullish underlying thesis |
| debit_call_spread | PASS | 4.15 | — | defined-risk directional; better than naked long when IV is not cheap |
| debit_put_spread | REJECT | 3.60 | — | against bullish underlying thesis |
| put_credit_spread | REJECT | — | — | no liquid structure in 21-75 DTE |
| call_credit_spread | REJECT | — | — | against bullish underlying thesis |

Macro in hold window: 2026-09-04 Employment Situation; 2026-09-10 PPI; 2026-09-11 CPI; 2026-09-16 FOMC decision

---

### GOOGL · IGNORE · **STOCK**

GOOGL — bearish Failed Breakout / Trend Breakdown in DATA UNAVAILABLE megacap

GOOGL is a bearish idea in a strong risk on tape because Failed Breakout / Trend Breakdown (G), 20d RS vs SPY -10.38%.

| | |
|---|---|
| **Setup** | G — Failed Breakout / Trend Breakdown |
| **Lane** | SWING / bearish |
| **Last** | 339.61 |
| **Trade** | STOCK short entry 339.61 stop 356.70 target 305.43 R/R 2.00 shares 29 |
| **Pay / collect** | stock (no option premium) |
| **Naive POP** | n/a — n/a for stock; no delta-based POP |
| **Conf** | n/a n/a (quality of the *structure*, not P(win)) |
| **Score** | 8.0 |
| **Evidence** | no same-setup analog |
| **Stop / invalidation** | close beyond stop 356.70 |
| **RS 20d vs SPY** | -10.38% |
| **Trend / MAs** | range · 20 345.83 / 50 349.21 / 200 334.81 |
| **Group** | megacap (QQQ) · DATA UNAVAILABLE |
| **AVWAP** | year 339.03 · swing-low 344.94 |
| **ORATS IV/HV** | IV30 26.34 · HV20 31.12 · VRP -4.78 |
| **Earnings** | 2026-10-26 (orats.wksNextErn) |
| **X** | Informed — Mega lag ~-2% on oil/yield open. WSB #6 mixed. Call OI decrease lists. Soft red day, not a clean dip. |
| **Fill** | stock last; options never mid |
| **Fill as-of** | n/a — revalidate at the open |
| **Greeks** | Δ DATA UNAVAILABLE · Γ DATA UNAVAILABLE · Θ DATA UNAVAILABLE · ν DATA UNAVAILABLE |

More context:

Price is 339.61, trend range, 20 EMA 345.83 / 50 349.21 / 200 334.81, relative volume 1.4.

Group megacap (QQQ) is DATA UNAVAILABLE. AVWAP year 339.03, swing-low 344.94.

Earnings 2026-10-26 (orats.wksNextErn).

Instrument shortlist picked **STOCK**. Invalidation: close beyond stop 356.70.

Tape notes:

- failed to hold the 20-day high with negative RS

All structures reviewed:

| structure | result | debit | credit | why |
|---|---|---:|---:|---|
| stock | PASS | — | — | stock plan clears R/R |
| long_call | REJECT | — | — | option chain DATA UNAVAILABLE |
| long_put | REJECT | — | — | option chain DATA UNAVAILABLE |
| debit_call_spread | REJECT | — | — | option chain DATA UNAVAILABLE |
| debit_put_spread | REJECT | — | — | option chain DATA UNAVAILABLE |
| put_credit_spread | REJECT | — | — | option chain DATA UNAVAILABLE |
| call_credit_spread | REJECT | — | — | option chain DATA UNAVAILABLE |

Macro in hold window: 2026-09-04 Employment Situation; 2026-09-10 PPI; 2026-09-11 CPI; 2026-09-16 FOMC decision

---

### SPY · IGNORE · **STOCK**

SPY — bullish Trend Pullback in DATA UNAVAILABLE index

SPY is a bullish idea in a strong risk on tape because Trend Pullback (A), 20d RS vs SPY 0.00%.

| | |
|---|---|
| **Setup** | A — Trend Pullback |
| **Lane** | SWING / bullish |
| **Last** | 767.55 |
| **Trade** | STOCK long entry 767.55 stop 755.15 target 792.36 R/R 2.00 shares 40 |
| **Pay / collect** | stock (no option premium) |
| **Naive POP** | n/a — n/a for stock; no delta-based POP |
| **Conf** | n/a n/a (quality of the *structure*, not P(win)) |
| **Score** | 35.0 |
| **Evidence** | no same-setup analog |
| **Stop / invalidation** | close beyond stop 755.15 |
| **RS 20d vs SPY** | 0.00% |
| **Trend / MAs** | strong_up · 20 765.44 / 50 754.37 / 200 710.27 |
| **Group** | index (SPY) · DATA UNAVAILABLE |
| **AVWAP** | year 705.82 · swing-low 753.68 |
| **ORATS IV/HV** | IV30 11.68 · HV20 9.46 · VRP 2.22 |
| **Earnings** | DATA UNAVAILABLE (exempt) |
| **X** | Crowded — Month-end. Open ~-0.46% on futures. $2.1B put wall at 765. VIX ~15.5. Oil spike + Warsh hawkish vs AI bid. Indexes soft, not a crash. |
| **Fill** | stock last; options never mid |
| **Fill as-of** | n/a — revalidate at the open |
| **Greeks** | Δ DATA UNAVAILABLE · Γ DATA UNAVAILABLE · Θ DATA UNAVAILABLE · ν DATA UNAVAILABLE |

More context:

Price is 767.55, trend strong_up, 20 EMA 765.44 / 50 754.37 / 200 710.27, relative volume 1.0.

Group index (SPY) is DATA UNAVAILABLE. AVWAP year 705.82, swing-low 753.68.

Earnings DATA UNAVAILABLE (exempt).

Instrument shortlist picked **STOCK**. Invalidation: close beyond stop 755.15.

Tape notes:

- trend pullback into 20 EMA / AVWAP / 50 DMA
- volume contracted on the pullback

All structures reviewed:

| structure | result | debit | credit | why |
|---|---|---:|---:|---|
| stock | PASS | — | — | stock plan clears R/R |
| long_call | REJECT | — | — | option chain DATA UNAVAILABLE |
| long_put | REJECT | — | — | option chain DATA UNAVAILABLE |
| debit_call_spread | REJECT | — | — | option chain DATA UNAVAILABLE |
| debit_put_spread | REJECT | — | — | option chain DATA UNAVAILABLE |
| put_credit_spread | REJECT | — | — | option chain DATA UNAVAILABLE |
| call_credit_spread | REJECT | — | — | option chain DATA UNAVAILABLE |

Macro in hold window: 2026-09-04 Employment Situation; 2026-09-10 PPI; 2026-09-11 CPI; 2026-09-16 FOMC decision

---

### GLD · WATCH · **STOCK**

GLD — bullish Relative-Strength Leader in DATA UNAVAILABLE macro

GLD is a bullish idea in a strong risk on tape because Relative-Strength Leader (D), 20d RS vs SPY 8.51%.

| | |
|---|---|
| **Setup** | D — Relative-Strength Leader |
| **Lane** | SWING / bullish |
| **Last** | 408.18 |
| **Trade** | STOCK long entry 408.18 stop 392.86 target 438.81 R/R 2.00 shares 32 |
| **Pay / collect** | stock (no option premium) |
| **Naive POP** | n/a — n/a for stock; no delta-based POP |
| **Conf** | n/a n/a (quality of the *structure*, not P(win)) |
| **Score** | 39.0 |
| **Evidence** | no same-setup analog |
| **Stop / invalidation** | close beyond stop 392.86 |
| **RS 20d vs SPY** | 8.51% |
| **Trend / MAs** | range · 20 406.78 / 50 386.49 / 200 414.93 |
| **Group** | macro (SPY) · DATA UNAVAILABLE |
| **AVWAP** | year 430.27 · swing-low 392.53 |
| **ORATS IV/HV** | IV30 21.52 · HV20 22.40 · VRP -0.88 |
| **Earnings** | DATA UNAVAILABLE (exempt) |
| **X** | Informed — WSB #4. Gold/dollar/oil geo hedge on Hormuz. ETF Action: GLD shed -$628.6M. Macro hedge bid, not a squeeze. |
| **Fill** | stock last; options never mid |
| **Fill as-of** | n/a — revalidate at the open |
| **Greeks** | Δ DATA UNAVAILABLE · Γ DATA UNAVAILABLE · Θ DATA UNAVAILABLE · ν DATA UNAVAILABLE |

More context:

Price is 408.18, trend range, 20 EMA 406.78 / 50 386.49 / 200 414.93, relative volume 0.8.

Group macro (SPY) is DATA UNAVAILABLE. AVWAP year 430.27, swing-low 392.53.

Earnings DATA UNAVAILABLE (exempt).

Instrument shortlist picked **STOCK**. Invalidation: close beyond stop 392.86.

Tape notes:

- 20d RS vs SPY +8.5% with accumulation structure

All structures reviewed:

| structure | result | debit | credit | why |
|---|---|---:|---:|---|
| stock | PASS | — | — | stock plan clears R/R |
| long_call | REJECT | — | — | option chain DATA UNAVAILABLE |
| long_put | REJECT | — | — | option chain DATA UNAVAILABLE |
| debit_call_spread | REJECT | — | — | option chain DATA UNAVAILABLE |
| debit_put_spread | REJECT | — | — | option chain DATA UNAVAILABLE |
| put_credit_spread | REJECT | — | — | option chain DATA UNAVAILABLE |
| call_credit_spread | REJECT | — | — | option chain DATA UNAVAILABLE |

Macro in hold window: 2026-09-04 Employment Situation; 2026-09-10 PPI; 2026-09-11 CPI; 2026-09-16 FOMC decision

---

### XOM · WATCH · **OPTIONS**

XOM — bullish Emerging Sector Rotation in emerging energy

XOM is a bullish idea in a strong risk on tape because Emerging Sector Rotation (E), 20d RS vs SPY 2.66%, energy group is emerging.

| | |
|---|---|
| **Setup** | E — Emerging Sector Rotation |
| **Lane** | SWING / bullish |
| **Last** | 161.20 |
| **Trade** | debit_call_spread BUY 165.0 call / SELL 175.0 call 2026-10-16 |
| **Pay / collect** | debit 3.25 |
| **Naive POP** | 36% — naive P(spot > breakeven 168.25) from ORATS call deltas. Not a backtested win rate. |
| **Conf** | 75 high (quality of the *structure*, not P(win)) |
| **Score** | 50.0 |
| **Evidence** | no same-setup analog |
| **Stop / invalidation** | close back below 20 EMA / swing-low AVWAP |
| **RS 20d vs SPY** | 2.66% |
| **Trend / MAs** | strong_up · 20 158.99 / 50 151.01 / 200 143.86 |
| **Group** | energy (XLE) · emerging |
| **AVWAP** | year 149.25 · swing-low 151.31 |
| **ORATS IV/HV** | IV30 27.93 · HV20 24.50 · VRP 3.43 |
| **Earnings** | 2026-10-30 (web.alphaquery) |
| **X** | Informed — Named with CVX as the oil-spike expression. Tape ~+2% with the energy group. Morgan Stanley raised Brent/WTI path on Middle East re-escalation. |
| **Fill** | long ask minus short bid (never mid) · schwab_quote |
| **Fill as-of** | 2026-08-31 19:58 ET |
| **Greeks** | Δ 0.20 · Γ 0.0050 · Θ -0.0190 · ν 0.0550 |

More context:

Price is 161.20, trend strong_up, 20 EMA 158.99 / 50 151.01 / 200 143.86, relative volume 1.3.

Group energy (XLE) is emerging. AVWAP year 149.25, swing-low 151.31.

Earnings 2026-10-30 (web.alphaquery).

Instrument shortlist picked **OPTIONS**. Invalidation: close back below 20 EMA / swing-low AVWAP.

Tape notes:

- group emerging with positive 20d RS
- trend pullback into 20 EMA / AVWAP / 50 DMA

All structures reviewed:

| structure | result | debit | credit | why |
|---|---|---:|---:|---|
| stock | PASS | — | — | stock plan clears R/R |
| long_call | REJECT | — | — | debit too large to size 1 lot at 1% of the 50k account |
| long_put | REJECT | 3.20 | — | against bullish underlying thesis |
| debit_call_spread | PASS | 3.25 | — | defined-risk directional; better than naked long when IV is not cheap |
| debit_put_spread | REJECT | 3.94 | — | against bullish underlying thesis |
| put_credit_spread | REJECT | — | — | no liquid structure in 21-75 DTE |
| call_credit_spread | REJECT | — | — | against bullish underlying thesis |

Macro in hold window: 2026-09-04 Employment Situation; 2026-09-10 PPI; 2026-09-11 CPI; 2026-09-16 FOMC decision

---

### MU · WATCH · **STOCK**

MU — bullish Trend Pullback in neutral semiconductors

MU is a bullish idea in a strong risk on tape because Trend Pullback (A), 20d RS vs SPY 13.94%.

| | |
|---|---|
| **Setup** | A — Trend Pullback |
| **Lane** | SWING / bullish |
| **Last** | 955.93 |
| **Trade** | STOCK long entry 955.93 stop 839.51 target 1188.79 R/R 2.00 shares 4 |
| **Pay / collect** | stock (no option premium) |
| **Naive POP** | n/a — n/a for stock; no delta-based POP |
| **Conf** | n/a n/a (quality of the *structure*, not P(win)) |
| **Score** | 45.0 |
| **Evidence** | no same-setup analog |
| **Stop / invalidation** | close beyond stop 839.51 |
| **RS 20d vs SPY** | 13.94% |
| **Trend / MAs** | up · 20 931.87 / 50 951.44 / 200 592.00 |
| **Group** | semiconductors (SMH) · neutral |
| **AVWAP** | year 681.74 · swing-low 895.15 |
| **ORATS IV/HV** | IV30 57.68 · HV20 61.27 · VRP -3.59 |
| **Earnings** | 2026-09-30 (web.alphaquery) |
| **X** | Informed — Memory bottleneck after NVDA prepaid supply. Still named with NVDA. Informed. |
| **Fill** | stock last; options never mid |
| **Fill as-of** | n/a — revalidate at the open |
| **Greeks** | Δ DATA UNAVAILABLE · Γ DATA UNAVAILABLE · Θ DATA UNAVAILABLE · ν DATA UNAVAILABLE |

More context:

Price is 955.93, trend up, 20 EMA 931.87 / 50 951.44 / 200 592.00, relative volume 0.8.

Group semiconductors (SMH) is neutral. AVWAP year 681.74, swing-low 895.15.

Earnings 2026-09-30 (web.alphaquery).

Instrument shortlist picked **STOCK**. Invalidation: close beyond stop 839.51.

Tape notes:

- trend pullback into 20 EMA / AVWAP / 50 DMA
- volume contracted on the pullback
- 20d RS vs SPY +13.9% with accumulation structure

All structures reviewed:

| structure | result | debit | credit | why |
|---|---|---:|---:|---|
| stock | PASS | — | — | stock plan clears R/R |
| long_call | REJECT | — | — | earnings inside expiry/hold |
| long_put | REJECT | — | — | against bullish underlying thesis |
| debit_call_spread | REJECT | — | — | debit takes too much of the spread width |
| debit_put_spread | REJECT | — | — | against bullish underlying thesis |
| put_credit_spread | REJECT | — | — | no liquid structure in 21-75 DTE |
| call_credit_spread | REJECT | — | — | against bullish underlying thesis |

Macro in hold window: 2026-09-04 Employment Situation; 2026-09-10 PPI; 2026-09-11 CPI; 2026-09-16 FOMC decision
