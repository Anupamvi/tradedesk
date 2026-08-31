# Groat full scan 2026-08-31

You click every Schwab order. Empty board is valid. Prefer 1–3 names.

## Desk pick

**Take options: CVX** — BUY 210.0 call / SELL 220.0 call 2026-09-25. Pay **debit 2.90**. Naive POP **37%**, conf **75**.

Why this one: long strike is near the money (1.7% from last); net delta 0.23; naive POP 37%; conf 75; X Informed. Setup E (Emerging Sector Rotation). Sub-50% naive POP is normal for an OTM/near-OTM debit — conf is structure quality, not P(win).

Act: work the fill at or inside the stated debit/credit. 1 lot first if X is Crowded. Invalidation: thesis/setup break.

Why this one, not the others:

- **CVX** **← take this** — long strike is near the money (1.7% from last); net delta 0.23; naive POP 37%; conf 75; X Informed.
- **XLE** — long strike is near the money (1.9% from last); net delta 0.37; naive POP 37%; conf 71; X Informed.
- **PLTR** — long strike is 7.7% OTM; net delta 0.09; naive POP 32%; conf 70; X Crowded.

**Stock if you want one: SHOP** — buy ~146.14, stop **134.09**, target **170.24**, 41 shares. Setup Emerging Sector Rotation.

Caution / size down: PLTR (Crowded X, net delta 0.09).

Analog caution (stock setup avg R < 0, n≥5): XLE.

Take **one** options ticket unless you explicitly want two uncorrelated names. Prefer 1–3 positions total.

## Evidence — same setup, this ticker

Same ticker + same setup on cached tape. Options use hist/strikes when cached or under the daily cap. Option P&L is delta+theta clamped to defined risk, not a live exit mark. Not a system win rate. X-HOT is not tested. Does not change today's gates.

| ticker | setup | stock n | W/L/time | avg R | options n | opt P&L / risk | BE vs naive POP |
|---|---|---:|---|---:|---:|---:|---|
| **CVX** | E Emerging Sector Rotation | 6 | 2/3/1 | 0.28 | 4 | -0.20 | 25% vs 32% |
| **PLTR** | E Emerging Sector Rotation | 4 | 0/2/2 | -0.36 | 3 | -0.55 | 0% vs 36% |
| **XLE** ⚠ | E Emerging Sector Rotation | 8 | 0/5/3 | -0.32 | 0 | — | — |
| **SHOP** | E Emerging Sector Rotation | 3 | 1/2/0 | 0.00 | 0 | — | — |
| **ADBE** | E Emerging Sector Rotation | 1 | 0/1/0 | -1.00 | 0 | — | — |
| **IGV** | E Emerging Sector Rotation | 3 | 0/1/2 | -0.62 | 0 | — | — |

| ticker | analog dates (entry → exit / result / R) |
|---|---|
| **CVX** | 2026-08-07→2026-08-18 win 2.00R; 2026-07-16→2026-08-06 time 0.69R; 2026-06-10→2026-06-15 loss -1.00R; 2026-05-18→2026-05-26 loss -1.00R; 2026-01-22→2026-02-04 win 2.00R; 2025-09-17→2025-09-30 loss -1.00R |
| **PLTR** | 2026-03-25→2026-03-30 loss -1.00R; 2025-12-29→2026-01-02 loss -1.00R; 2025-10-14→2025-11-04 time 0.71R; 2025-09-22→2025-10-13 time -0.15R |
| **XLE** | 2026-08-10→2026-08-31 time 1.32R; 2026-07-17→2026-08-07 time -0.08R; 2026-06-10→2026-06-15 loss -1.00R; 2026-05-18→2026-05-27 loss -1.00R; 2026-02-05→2026-02-27 time 1.17R; 2025-12-11→2025-12-16 loss -1.00R |
| **SHOP** | 2025-12-29→2026-01-02 loss -1.00R; 2025-10-14→2025-10-28 win 2.00R; 2025-09-22→2025-09-25 loss -1.00R |
| **ADBE** | 2025-12-29→2026-01-02 loss -1.00R |
| **IGV** | 2026-06-04→2026-06-09 loss -1.00R; 2025-10-14→2025-11-04 time -0.38R; 2025-09-22→2025-10-13 time -0.49R |

Hierarchy: market regime → underlying thesis → price/AVWAP/volume → catalyst → relative strength → risk/reward → ORATS vol + structure → positioning → X.
X excitement cannot rescue a bad chart. Never invent ORATS, prices, posts, or news.

# Market regime

**Regime:** weak_risk_on

Selective pullbacks in relative-strength names. Smaller size. Skip extended breakouts.

## Why

- Uptrend without full 20>50>200 stack
- VIX DATA UNAVAILABLE

## Tape

| name | close | 5d | 20d | 60d | vs 20 EMA | vs 50 | vs 200 | trend |
|---|---:|---:|---:|---:|---|---|---|---|
| SPY | 764.90 | 0.19% | 0.95% | 1.03% | below | above | above | up |
| QQQ | 713.55 | 1.02% | 1.93% | -3.65% | below | above | above | up |
| IWM | 293.01 | -1.66% | -1.08% | 0.34% | below | below | above | range |
| DIA | 530.82 | -0.53% | -0.08% | 2.73% | below | above | above | up |
| TLT | 82.25 | -0.38% | 0.07% | -3.80% | below | below | below | strong_down |

VIX: DATA UNAVAILABLE (5d DATA UNAVAILABLE)

Universe breadth (n=97): >20 EMA 44.33% · >50 DMA 54.64% · >200 DMA 71.13%

# Sector rotation

| etf | group | status | RS 5d vs SPY | RS 20d | RS 60d | 20d ret | trend |
|---|---|---|---:|---:|---:|---:|---|
| IGV | software | accelerating | 6.58% | 11.32% | 8.28% | 12.28% | strong_up |
| URNM | nuclear | accelerating | -3.45% | 11.04% | -7.98% | 12.00% | range |
| XBI | biotech | mature | -2.57% | 7.83% | 19.08% | 8.78% | up |
| XLE | energy | emerging | 0.87% | 7.53% | 7.53% | 8.49% | strong_up |
| CIBR | cybersecurity | mature | 5.63% | 4.10% | 7.74% | 5.05% | strong_up |
| XLV | healthcare | mature | -2.87% | 3.84% | 10.76% | 4.79% | strong_up |
| XLK | technology | accelerating | 2.87% | 3.26% | -4.98% | 4.22% | strong_up |
| XLB | materials | accelerating | -1.49% | 2.71% | 1.41% | 3.67% | strong_up |
| SMH | semiconductors | neutral | 1.46% | 0.94% | -12.46% | 1.89% | range |
| XLF | financials | neutral | -0.99% | -0.31% | 9.62% | 0.64% | strong_up |
| SOXX | semiconductors | neutral | 0.56% | -0.51% | -16.42% | 0.45% | range |
| XLP | consumer_staples | neutral | -3.05% | -0.85% | 2.51% | 0.10% | up |
| XLC | communication | neutral | -1.00% | -0.89% | -2.53% | 0.06% | range |
| BOTZ | robotics | neutral | 1.64% | -1.05% | -11.47% | -0.10% | strong_down |
| XLY | consumer_discretionary | deteriorating | -1.68% | -2.37% | -1.65% | -1.42% | range |
| GRID | electrification | deteriorating | -0.15% | -3.02% | -10.54% | -2.07% | range |
| XLRE | real_estate | deteriorating | -3.30% | -3.74% | -2.11% | -2.79% | range |
| XLI | industrials | deteriorating | -2.53% | -5.51% | -1.80% | -4.56% | range |
| XLU | utilities | deteriorating | -3.16% | -6.42% | -5.59% | -5.47% | strong_down |
| XAR | aerospace | deteriorating | -3.97% | -7.56% | -9.90% | -6.61% | down |
| ITA | defense | deteriorating | -2.54% | -8.34% | -2.61% | -7.38% | down |

# Groat 2026-08-31

Regime **weak_risk_on** · TRADE 6 · WATCH 10 · FIRE 0 · X-HOT 10

You click every Schwab order. Empty board is valid. Prefer 1–3 names.

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

| ticker | setup | vehicle | pay/collect | naive POP | conf | last |
|---|---|---|---|---:|---:|---:|
| **PLTR** | E Emerging Sector Rotation | **OPTIONS** | debit 2.55 | 32% | 70 | 185.69 |
| **SHOP** | E Emerging Sector Rotation | **STOCK** | — | n/a | — | 146.14 |
| **ADBE** | E Emerging Sector Rotation | **STOCK** | — | n/a | — | 291.20 |
| **IGV** | E Emerging Sector Rotation | **STOCK** | — | n/a | — | 109.38 |
| **XLE** | E Emerging Sector Rotation | **OPTIONS** | debit 1.65 | 37% | 71 | 63.78 |
| **CVX** | E Emerging Sector Rotation | **OPTIONS** | debit 2.90 | 37% | 75 | 206.40 |

| ticker | exact structure | stop / invalidation |
|---|---|---|
| **PLTR** | debit_call_spread BUY 200.0 call / SELL 210.0 call 2026-10-16 | thesis/setup break |
| **SHOP** | STOCK long entry 146.14 stop 134.09 target 170.24 R/R 2.00 shares 41 | close beyond stop 134.09 |
| **ADBE** | STOCK long entry 291.20 stop 266.42 target 340.75 R/R 2.00 shares 20 | close beyond stop 266.42 |
| **IGV** | STOCK long entry 109.38 stop 102.17 target 123.80 R/R 2.00 shares 69 | close beyond stop 102.17 |
| **XLE** | debit_call_spread BUY 65.0 call / SELL 75.0 call 2026-10-16 | thesis/setup break |
| **CVX** | debit_call_spread BUY 210.0 call / SELL 220.0 call 2026-09-25 | thesis/setup break |

## WATCH

| ticker | setup | last | RS20 | score | parked because |
|---|---|---:|---:|---:|---|
| CRWD | C | 226.32 | 10.79% | 51.0 | below_trade_score |
| SNOW | E | 325.00 | 4.73% | 50.0 | below_trade_score |
| NET | E | 297.98 | 4.43% | 50.0 | below_trade_score |
| XLK | E | 185.55 | 3.26% | 50.0 | below_trade_score |
| XLB | E | 52.88 | 2.71% | 50.0 | below_trade_score |
| XOM | E | 160.07 | 2.28% | 50.0 | below_trade_score |
| VRTX | A | 536.25 | 12.97% | 47.0 | below_trade_score |
| PANW | A | 375.09 | 7.10% | 47.0 | below_trade_score |
| ANET | E | 194.19 | 4.07% | 46.0 | below_trade_score |
| CRM | C | 259.94 | 38.84% | 45.0 | below_trade_score |

## FIRE — spike / dip

Needs volume + a 1–2 day shock **first**. X only confirms or vetoes.

No FIRE names. Valid.

## X-HOT — conversation first

Starts from what is loud on X, then asks the tape. **dipped** = already red with heat (buy-the-dip only if 20 EMA/AVWAP holds). **will_rise** = bullish X and tape allows continuation or a later entry. **will_dip** = bearish X, or a spike already extended (the trade is the pullback, not the chase). Heat without a volume/price trigger is Watch, not a trade.

| ticker | move | tape | 1d | rvol | X | vehicle |
|---|---|---|---:|---:|---|---|
| **GOOGL** | **dipped** | soft_dip | -2.59% | 0.2 | Informed / unknown | OPTIONS |
| **AMZN** | **dipped** | soft_dip | -2.37% | 0.2 | Quiet / unknown | OPTIONS |
| **GLD** | **dipped** | soft_dip | -0.88% | 0.2 | Informed / bullish | OPTIONS |
| **SPY** | **dipped** | soft_dip | -0.58% | 0.2 | Crowded / unknown | OPTIONS |
| **TSLA** | **will_rise** | heat_only | 4.51% | 0.5 | Crowded / bullish | OPTIONS |
| **CRWD** | **will_rise** | heat_only | 3.63% | 0.5 | Informed / bullish | OPTIONS |
| **CVX** | **will_rise** | heat_only | 2.25% | 0.4 | Informed / bullish | OPTIONS |
| **XOM** | **will_rise** | heat_only | 2.15% | 0.2 | Informed / bullish | OPTIONS |
| **XLU** | **will_dip** | soft_dip | -1.86% | 0.4 | Informed / bearish | STOCK |
| **XLE** | **will_rise** | heat_only | 1.75% | 0.3 | Informed / bullish | OPTIONS |

| ticker | play | X narrative |
|---|---|---|
| **GOOGL** | X-hot, soft red day — watch for a hold of 20 EMA/AVWAP before buying | Mega lag ~-2% on the oil/yield open. WSB #6 mixed. Call OI decrease lists. Not a clean dip narrative. |
| **AMZN** | X-hot, soft red day — watch for a hold of 20 EMA/AVWAP before buying | Mega lag with GOOGL ~-2%. Isolated $260-node long posts. Not the conversation. |
| **GLD** | X-hot, soft red day — watch for a hold of 20 EMA/AVWAP before buying | WSB #4. Gold/dollar/oil geo hedge chatter on Hormuz. Not a squeeze — macro hedge bid. |
| **SPY** | X-hot, soft red day — watch for a hold of 20 EMA/AVWAP before buying | Month-end session. Open landed on futures (~-0.46%). $2.1B put wall at 765. VIX ~15.5. Oil spike + Warsh hawkish vs AI bid. Indexes soft, not a crash. |
| **TSLA** | X says up — wait for a dip into 20 EMA/AVWAP or a real volume spike to enter | Open recap: TSLA ~+3.4% while indexes soft. Relative-strength leader, already extended vs the tape. Promo mix. |
| **CRWD** | X says up — wait for a dip into 20 EMA/AVWAP or a real volume spike to enter | Fal.Con: tape ~+3.7% while indexes red. Software confirmation leftover from last week's print. Already ran — do not chase the open. |
| **CVX** | X says up — wait for a dip into 20 EMA/AVWAP or a real volume spike to enter | Energy leader in open recaps. Oil >$86–90 on Hormuz; CVX cited as the single-name watch. Already ripping with the group — do not chase the first hour. |
| **XOM** | X says up — wait for a dip into 20 EMA/AVWAP or a real volume spike to enter | Named with CVX as the oil-spike expression. Tape ~+2% with the energy group. Morgan Stanley raised Brent/WTI path on Middle East re-escalation. |
| **XLU** | X-hot, red tape + bearish narrative — wait for a volume dump or failed high before puts | Utilities the worst sector (~-1.7%). CA SB 492 failed to refill wildfire fund / cap liability. PCG ~-19%, EIX ~-20% after Mizuho/BMO cuts. Sector dump is two names, not rates. |
| **XLE** | X says up — wait for a dip into 20 EMA/AVWAP or a real volume spike to enter | THE open: WTI ~$86 +3%, Brent ~$89 after US Hormuz/Larak strikes and Iran reply. Energy the only sector up >2%. Group bid, not a single-name story. |

## Tickets

---

### PLTR · TRADE · **OPTIONS**

PLTR — bullish Emerging Sector Rotation in accelerating software

PLTR is a bullish idea in a weak risk on tape because Emerging Sector Rotation (E), 20d RS vs SPY 46.83%, software group is accelerating.

| | |
|---|---|
| **Setup** | E — Emerging Sector Rotation |
| **Lane** | SWING / bullish |
| **Last** | 185.69 |
| **Trade** | debit_call_spread BUY 200.0 call / SELL 210.0 call 2026-10-16 |
| **Pay / collect** | debit 2.55 |
| **Naive POP** | 32% — naive P(spot > breakeven 202.55) from ORATS call deltas. Not a backtested win rate. |
| **Conf** | 70 high (quality of the *structure*, not P(win)) |
| **Score** | 55.0 |
| **Evidence** | stock 0W/2L/2t n=4 avg R -0.36; options n=3 P&L/risk -0.55 (spot-delta, not a live mark) |
| **Stop / invalidation** | close back below 20 EMA / swing-low AVWAP |
| **RS 20d vs SPY** | 46.83% |
| **Trend / MAs** | up · 20 170.57 / 50 144.89 / 200 151.35 |
| **Group** | software (IGV) · accelerating |
| **AVWAP** | year 145.03 · swing-low 145.05 |
| **ORATS IV/HV** | IV30 45.04 · HV20 79.77 · VRP -34.73 |
| **Earnings** | 2026-11-02 (web.alphaquery) |
| **X** | Crowded — Promo + ticker-spam (XXII dumps, generational-wealth lists, Trump-only-mentioned-PLTR). One real post: ~$186, sales +93% / US commercial +149%, FY guide $8.15–8.16B, rich multiple. Downweight the spam; no fresh catalyst today. |
| **Fill** | long ask minus short bid (never mid) |
| **Greeks** | Δ 0.09 · Γ 0.0021 · Θ -0.0118 · ν 0.0340 |

More context:

Price is 185.69, trend up, 20 EMA 170.57 / 50 144.89 / 200 151.35, relative volume 0.1.

Group software (IGV) is accelerating. AVWAP year 145.03, swing-low 145.05.

Earnings 2026-11-02 (web.alphaquery).

Instrument shortlist picked **OPTIONS**. Invalidation: close back below 20 EMA / swing-low AVWAP.

Tape notes:

- group accelerating with positive 20d RS
- 20d RS vs SPY +46.8% with accumulation structure

All structures reviewed:

| structure | result | debit | credit | why |
|---|---|---:|---:|---|
| stock | PASS | — | — | stock plan clears R/R |
| long_call | REJECT | — | — | no liquid structure in 21-75 DTE |
| long_put | REJECT | — | — | against bullish underlying thesis |
| debit_call_spread | PASS | 2.55 | — | defined-risk directional; better than naked long when IV is not cheap |
| debit_put_spread | REJECT | 4.00 | — | against bullish underlying thesis |
| put_credit_spread | REJECT | — | — | no liquid structure in 21-75 DTE |
| call_credit_spread | REJECT | — | — | against bullish underlying thesis |

Macro in hold window: 2026-09-04 Employment Situation; 2026-09-10 PPI; 2026-09-11 CPI; 2026-09-16 FOMC decision

---

### SHOP · TRADE · **STOCK**

SHOP — bullish Emerging Sector Rotation in accelerating software

SHOP is a bullish idea in a weak risk on tape because Emerging Sector Rotation (E), 20d RS vs SPY 23.94%, software group is accelerating.

| | |
|---|---|
| **Setup** | E — Emerging Sector Rotation |
| **Lane** | SWING / bullish |
| **Last** | 146.14 |
| **Trade** | STOCK long entry 146.14 stop 134.09 target 170.24 R/R 2.00 shares 41 |
| **Pay / collect** | stock (no option premium) |
| **Naive POP** | n/a — n/a for stock; no delta-based POP |
| **Conf** | n/a n/a (quality of the *structure*, not P(win)) |
| **Score** | 55.0 |
| **Evidence** | stock 1W/2L/0t n=3 avg R 0.00 |
| **Stop / invalidation** | close beyond stop 134.09 |
| **RS 20d vs SPY** | 23.94% |
| **Trend / MAs** | strong_up · 20 146.10 / 50 131.50 / 200 131.38 |
| **Group** | software (IGV) · accelerating |
| **AVWAP** | year 123.18 · swing-low 130.94 |
| **ORATS IV/HV** | IV30 43.86 · HV20 55.53 · VRP -11.67 |
| **Earnings** | 2026-11-03 (web.alphaquery) |
| **X** | Informed — Flow: ~$498k 120P 10/16 (18% OTM). Gap-down list. Software hide-out / 9 EMA break, post-earnings gap unfilled. One TA: bullish flag, buy pullback to 8 EMA or reclaim $154. Informed, not crowded. |
| **Fill** | stock last; options never mid |
| **Greeks** | Δ DATA UNAVAILABLE · Γ DATA UNAVAILABLE · Θ DATA UNAVAILABLE · ν DATA UNAVAILABLE |

More context:

Price is 146.14, trend strong_up, 20 EMA 146.10 / 50 131.50 / 200 131.38, relative volume 0.2.

Group software (IGV) is accelerating. AVWAP year 123.18, swing-low 130.94.

Earnings 2026-11-03 (web.alphaquery).

Instrument shortlist picked **STOCK**. Invalidation: close beyond stop 134.09.

Tape notes:

- group accelerating with positive 20d RS
- trend pullback into 20 EMA / AVWAP / 50 DMA
- volume contracted on the pullback
- 20d RS vs SPY +23.9% with accumulation structure

All structures reviewed:

| structure | result | debit | credit | why |
|---|---|---:|---:|---|
| stock | PASS | — | — | stock plan clears R/R |
| long_call | REJECT | — | — | no liquid structure in 21-75 DTE |
| long_put | REJECT | — | — | against bullish underlying thesis |
| debit_call_spread | REJECT | — | — | no liquid structure in 21-75 DTE |
| debit_put_spread | REJECT | — | — | against bullish underlying thesis |
| put_credit_spread | REJECT | — | — | no liquid structure in 21-75 DTE |
| call_credit_spread | REJECT | — | — | against bullish underlying thesis |

Macro in hold window: 2026-09-04 Employment Situation; 2026-09-10 PPI; 2026-09-11 CPI; 2026-09-16 FOMC decision

---

### ADBE · TRADE · **STOCK**

ADBE — bullish Emerging Sector Rotation in accelerating software

ADBE is a bullish idea in a weak risk on tape because Emerging Sector Rotation (E), 20d RS vs SPY 14.90%, software group is accelerating.

| | |
|---|---|
| **Setup** | E — Emerging Sector Rotation |
| **Lane** | SWING / bullish |
| **Last** | 291.20 |
| **Trade** | STOCK long entry 291.20 stop 266.42 target 340.75 R/R 2.00 shares 20 |
| **Pay / collect** | stock (no option premium) |
| **Naive POP** | n/a — n/a for stock; no delta-based POP |
| **Conf** | n/a n/a (quality of the *structure*, not P(win)) |
| **Score** | 55.0 |
| **Evidence** | stock 0W/1L/0t n=1 avg R -1.00 |
| **Stop / invalidation** | close beyond stop 266.42 |
| **RS 20d vs SPY** | 14.90% |
| **Trend / MAs** | up · 20 270.02 / 50 242.10 / 200 268.97 |
| **Group** | software (IGV) · accelerating |
| **AVWAP** | year 250.26 · swing-low 234.21 |
| **ORATS IV/HV** | IV30 52.46 · HV20 43.70 · VRP 8.76 |
| **Earnings** | 2026-09-10 (web.alphaquery) — ordinary options blocked |
| **X** | Informed — Value long-pitch: ~11x fwd, AI-first ARR >$500M, Firefly ARR ~$300M. Earnings Thursday AMC after Labor Day; recent distribution skewed downside, avg peak move ~9.2%. Options blocked through the print anyway. |
| **Fill** | stock last; options never mid |
| **Greeks** | Δ DATA UNAVAILABLE · Γ DATA UNAVAILABLE · Θ DATA UNAVAILABLE · ν DATA UNAVAILABLE |

More context:

Price is 291.20, trend up, 20 EMA 270.02 / 50 242.10 / 200 268.97, relative volume 0.1.

Group software (IGV) is accelerating. AVWAP year 250.26, swing-low 234.21.

Earnings 2026-09-10 (web.alphaquery); ordinary options blocked through the print.

Instrument shortlist picked **STOCK**. Invalidation: close beyond stop 266.42.

Tape notes:

- group accelerating with positive 20d RS
- 20d RS vs SPY +14.9% with accumulation structure

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

### IGV · TRADE · **STOCK**

IGV — bullish Emerging Sector Rotation in accelerating software

IGV is a bullish idea in a weak risk on tape because Emerging Sector Rotation (E), 20d RS vs SPY 11.32%, software group is accelerating.

| | |
|---|---|
| **Setup** | E — Emerging Sector Rotation |
| **Lane** | SWING / bullish |
| **Last** | 109.38 |
| **Trade** | STOCK long entry 109.38 stop 102.17 target 123.80 R/R 2.00 shares 69 |
| **Pay / collect** | stock (no option premium) |
| **Naive POP** | n/a — n/a for stock; no delta-based POP |
| **Conf** | n/a n/a (quality of the *structure*, not P(win)) |
| **Score** | 55.0 |
| **Evidence** | stock 0W/1L/2t n=3 avg R -0.62 |
| **Stop / invalidation** | close beyond stop 102.17 |
| **RS 20d vs SPY** | 11.32% |
| **Trend / MAs** | strong_up · 20 103.13 / 50 96.41 / 200 93.67 |
| **Group** | software (IGV) · accelerating |
| **AVWAP** | year 88.40 · swing-low 96.04 |
| **ORATS IV/HV** | IV30 30.35 · HV20 34.02 · VRP -3.67 |
| **Earnings** | DATA UNAVAILABLE (exempt) |
| **X** | Informed — Software-ETF talk, not a name-level catalyst. ETF Action: IGV +$655M inflows. Algo-rotation caution after the thrust; one TA bearish below 107.50. Sector digest, not a squeeze. |
| **Fill** | stock last; options never mid |
| **Greeks** | Δ DATA UNAVAILABLE · Γ DATA UNAVAILABLE · Θ DATA UNAVAILABLE · ν DATA UNAVAILABLE |

More context:

Price is 109.38, trend strong_up, 20 EMA 103.13 / 50 96.41 / 200 93.67, relative volume 0.2.

Group software (IGV) is accelerating. AVWAP year 88.40, swing-low 96.04.

Earnings DATA UNAVAILABLE (exempt).

Instrument shortlist picked **STOCK**. Invalidation: close beyond stop 102.17.

Tape notes:

- group accelerating with positive 20d RS
- 20d RS vs SPY +11.3% with accumulation structure

All structures reviewed:

| structure | result | debit | credit | why |
|---|---|---:|---:|---|
| stock | PASS | — | — | stock plan clears R/R |
| long_call | REJECT | — | — | no liquid structure in 21-75 DTE |
| long_put | REJECT | — | — | against bullish underlying thesis |
| debit_call_spread | REJECT | — | — | no liquid structure in 21-75 DTE |
| debit_put_spread | REJECT | 2.29 | — | against bullish underlying thesis |
| put_credit_spread | REJECT | — | 1.21 | IV not rich — do not sell premium |
| call_credit_spread | REJECT | — | — | against bullish underlying thesis |

Macro in hold window: 2026-09-04 Employment Situation; 2026-09-10 PPI; 2026-09-11 CPI; 2026-09-16 FOMC decision

---

### XLE · TRADE · **OPTIONS**

XLE — bullish Emerging Sector Rotation in emerging energy

XLE is a bullish idea in a weak risk on tape because Emerging Sector Rotation (E), 20d RS vs SPY 7.53%, energy group is emerging.

| | |
|---|---|
| **Setup** | E — Emerging Sector Rotation |
| **Lane** | SWING / bullish |
| **Last** | 63.78 |
| **Trade** | debit_call_spread BUY 65.0 call / SELL 75.0 call 2026-10-16 |
| **Pay / collect** | debit 1.65 |
| **Naive POP** | 37% — naive P(spot > breakeven 66.65) from ORATS call deltas. Not a backtested win rate. |
| **Conf** | 71 high (quality of the *structure*, not P(win)) |
| **Score** | 55.0 |
| **Evidence** | stock 0W/5L/3t n=8 avg R -0.32; weak analog |
| **Stop / invalidation** | close back below 20 EMA / swing-low AVWAP |
| **RS 20d vs SPY** | 7.53% |
| **Trend / MAs** | strong_up · 20 61.70 / 50 58.35 / 200 54.44 |
| **Group** | energy (XLE) · emerging |
| **AVWAP** | year 55.89 · swing-low 58.75 |
| **ORATS IV/HV** | IV30 25.46 · HV20 23.08 · VRP 2.38 |
| **Earnings** | DATA UNAVAILABLE (exempt) |
| **X** | Informed — THE open: WTI ~$86 +3%, Brent ~$89 after US Hormuz/Larak strikes and Iran reply. Energy the only sector up >2%. Group bid, not a single-name story. |
| **Fill** | long ask minus short bid (never mid) |
| **Greeks** | Δ 0.37 · Γ 0.0511 · Θ -0.0157 · ν 0.0608 |

More context:

Price is 63.78, trend strong_up, 20 EMA 61.70 / 50 58.35 / 200 54.44, relative volume 0.3.

Group energy (XLE) is emerging. AVWAP year 55.89, swing-low 58.75.

Earnings DATA UNAVAILABLE (exempt).

Instrument shortlist picked **OPTIONS**. Invalidation: close back below 20 EMA / swing-low AVWAP.

Tape notes:

- group emerging with positive 20d RS
- close above prior 20-session close high
- 20d RS vs SPY +7.5% with accumulation structure

All structures reviewed:

| structure | result | debit | credit | why |
|---|---|---:|---:|---|
| stock | PASS | — | — | stock plan clears R/R |
| long_call | PASS | 2.75 | — | cheap-to-fair vol directional with 21-75 DTE |
| long_put | REJECT | 3.05 | — | against bullish underlying thesis |
| debit_call_spread | PASS | 1.65 | — | defined-risk directional; better than naked long when IV is not cheap |
| debit_put_spread | REJECT | 1.91 | — | against bullish underlying thesis |
| put_credit_spread | REJECT | — | — | no liquid structure in 21-75 DTE |
| call_credit_spread | REJECT | — | — | against bullish underlying thesis |

Macro in hold window: 2026-09-04 Employment Situation; 2026-09-10 PPI; 2026-09-11 CPI; 2026-09-16 FOMC decision

---

### CVX · TRADE · **OPTIONS**

CVX — bullish Emerging Sector Rotation in emerging energy

CVX is a bullish idea in a weak risk on tape because Emerging Sector Rotation (E), 20d RS vs SPY 5.89%, energy group is emerging.

| | |
|---|---|
| **Setup** | E — Emerging Sector Rotation |
| **Lane** | SWING / bullish |
| **Last** | 206.40 |
| **Trade** | debit_call_spread BUY 210.0 call / SELL 220.0 call 2026-09-25 |
| **Pay / collect** | debit 2.90 |
| **Naive POP** | 37% — naive P(spot > breakeven 212.90) from ORATS call deltas. Not a backtested win rate. |
| **Conf** | 75 high (quality of the *structure*, not P(win)) |
| **Score** | 55.0 |
| **Evidence** | stock 2W/3L/1t n=6 avg R 0.28; options n=4 P&L/risk -0.20 (spot-delta, not a live mark) |
| **Stop / invalidation** | close back below 20 EMA / swing-low AVWAP |
| **RS 20d vs SPY** | 5.89% |
| **Trend / MAs** | strong_up · 20 199.27 / 50 188.20 / 200 180.01 |
| **Group** | energy (XLE) · emerging |
| **AVWAP** | year 185.47 · swing-low 190.10 |
| **ORATS IV/HV** | IV30 25.56 · HV20 22.96 · VRP 2.60 |
| **Earnings** | 2026-10-30 (web.alphaquery) |
| **X** | Informed — Oil-spike expression: Hormuz/Iran, Brent/WTI bid, tape ~+2%. Venezuela energy-agreement chatter (CVX/ONGC/ENI). Friday $1.1M call flow now marked up. Group bid — do not chase the first hour. |
| **Fill** | long ask minus short bid (never mid) |
| **Greeks** | Δ 0.23 · Γ 0.0087 · Θ -0.0335 · ν 0.0644 |

More context:

Price is 206.40, trend strong_up, 20 EMA 199.27 / 50 188.20 / 200 180.01, relative volume 0.4.

Group energy (XLE) is emerging. AVWAP year 185.47, swing-low 190.10.

Earnings 2026-10-30 (web.alphaquery).

Instrument shortlist picked **OPTIONS**. Invalidation: close back below 20 EMA / swing-low AVWAP.

Tape notes:

- group emerging with positive 20d RS
- close above prior 20-session close high
- 20d RS vs SPY +5.9% with accumulation structure

All structures reviewed:

| structure | result | debit | credit | why |
|---|---|---:|---:|---|
| stock | PASS | — | — | stock plan clears R/R |
| long_call | REJECT | — | — | no liquid structure in 21-75 DTE |
| long_put | REJECT | — | — | against bullish underlying thesis |
| debit_call_spread | PASS | 2.90 | — | defined-risk directional; better than naked long when IV is not cheap |
| debit_put_spread | REJECT | 2.77 | — | against bullish underlying thesis |
| put_credit_spread | REJECT | — | — | no liquid structure in 21-75 DTE |
| call_credit_spread | REJECT | — | — | against bullish underlying thesis |

Macro in hold window: 2026-09-04 Employment Situation; 2026-09-10 PPI; 2026-09-11 CPI; 2026-09-16 FOMC decision

---

### GOOGL · IGNORE · **OPTIONS**

GOOGL — bearish Failed Breakout / Trend Breakdown in DATA UNAVAILABLE megacap

GOOGL is a bearish idea in a weak risk on tape because Failed Breakout / Trend Breakdown (G), 20d RS vs SPY -10.56%.

| | |
|---|---|
| **Setup** | G — Failed Breakout / Trend Breakdown |
| **Lane** | SWING / bearish |
| **Last** | 337.63 |
| **Trade** | debit_put_spread BUY 325.0 put / SELL 315.0 put 2026-10-16 |
| **Pay / collect** | debit 3.10 |
| **Naive POP** | 29% — naive P(spot < breakeven 321.90) from ORATS call deltas. Not a backtested win rate. |
| **Conf** | 79 high (quality of the *structure*, not P(win)) |
| **Score** | 16.0 |
| **Evidence** | no same-setup analog |
| **Stop / invalidation** | close back above 20 EMA / failed-breakdown reclaim |
| **RS 20d vs SPY** | -10.56% |
| **Trend / MAs** | range · 20 345.65 / 50 349.17 / 200 334.80 |
| **Group** | megacap (QQQ) · DATA UNAVAILABLE |
| **AVWAP** | year 339.02 · swing-low 345.11 |
| **ORATS IV/HV** | IV30 27.30 · HV20 31.12 · VRP -3.82 |
| **Earnings** | 2026-11-04 (web.alphaquery) |
| **X** | Informed — Mega lag ~-2% on oil/yield open. WSB #6 mixed. Call OI decrease lists. Soft red day, not a clean dip. |
| **Fill** | long ask minus short bid (never mid) |
| **Greeks** | Δ -0.09 · Γ 0.0019 · Θ -0.0155 · ν 0.0704 |

More context:

Price is 337.63, trend range, 20 EMA 345.65 / 50 349.17 / 200 334.80, relative volume 0.2.

Group megacap (QQQ) is DATA UNAVAILABLE. AVWAP year 339.02, swing-low 345.11.

Earnings 2026-11-04 (web.alphaquery).

Instrument shortlist picked **OPTIONS**. Invalidation: close back above 20 EMA / failed-breakdown reclaim.

Tape notes:

- failed to hold the 20-day high with negative RS

All structures reviewed:

| structure | result | debit | credit | why |
|---|---|---:|---:|---|
| stock | PASS | — | — | stock plan clears R/R |
| long_call | REJECT | — | — | against bearish underlying thesis |
| long_put | REJECT | — | — | no liquid structure in 21-75 DTE |
| debit_call_spread | REJECT | 3.30 | — | against bearish underlying thesis |
| debit_put_spread | PASS | 3.10 | — | defined-risk directional; better than naked long when IV is not cheap |
| put_credit_spread | REJECT | — | — | against bearish underlying thesis |
| call_credit_spread | REJECT | — | — | no liquid structure in 21-75 DTE |

Macro in hold window: 2026-09-04 Employment Situation; 2026-09-10 PPI; 2026-09-11 CPI; 2026-09-16 FOMC decision

---

### AMZN · IGNORE · **OPTIONS**

AMZN — bullish Trend Pullback in DATA UNAVAILABLE megacap

AMZN is a bullish idea in a weak risk on tape because Trend Pullback (A), 20d RS vs SPY -9.37%.

| | |
|---|---|
| **Setup** | A — Trend Pullback |
| **Lane** | SWING / bullish |
| **Last** | 260.11 |
| **Trade** | debit_call_spread BUY 270.0 call / SELL 280.0 call 2026-10-16 |
| **Pay / collect** | debit 3.25 |
| **Naive POP** | 36% — naive P(spot > breakeven 273.25) from ORATS call deltas. Not a backtested win rate. |
| **Conf** | 70 high (quality of the *structure*, not P(win)) |
| **Score** | 27.0 |
| **Evidence** | no same-setup analog |
| **Stop / invalidation** | close back below 20 EMA / swing-low AVWAP |
| **RS 20d vs SPY** | -9.37% |
| **Trend / MAs** | up · 20 260.91 / 50 251.97 / 200 238.73 |
| **Group** | megacap (QQQ) · DATA UNAVAILABLE |
| **AVWAP** | year 238.49 · swing-low 250.90 |
| **ORATS IV/HV** | IV30 28.68 · HV20 28.95 · VRP -0.27 |
| **Earnings** | 2026-10-29 (web.alphaquery) |
| **X** | Quiet — Mega lag with GOOGL ~-2%. Isolated $260-node long posts. Not the conversation. |
| **Fill** | long ask minus short bid (never mid) |
| **Greeks** | Δ 0.13 · Γ 0.0020 · Θ -0.0197 · ν 0.0594 |

More context:

Price is 260.11, trend up, 20 EMA 260.91 / 50 251.97 / 200 238.73, relative volume 0.2.

Group megacap (QQQ) is DATA UNAVAILABLE. AVWAP year 238.49, swing-low 250.90.

Earnings 2026-10-29 (web.alphaquery).

Instrument shortlist picked **OPTIONS**. Invalidation: close back below 20 EMA / swing-low AVWAP.

Tape notes:

- trend pullback into 20 EMA / AVWAP / 50 DMA
- volume contracted on the pullback
- failed to hold the 20-day high with negative RS

All structures reviewed:

| structure | result | debit | credit | why |
|---|---|---:|---:|---|
| stock | PASS | — | — | stock plan clears R/R |
| long_call | REJECT | — | — | no liquid structure in 21-75 DTE |
| long_put | REJECT | — | — | against bullish underlying thesis |
| debit_call_spread | PASS | 3.25 | — | defined-risk directional; better than naked long when IV is not cheap |
| debit_put_spread | REJECT | 2.85 | — | against bullish underlying thesis |
| put_credit_spread | REJECT | — | — | no liquid structure in 21-75 DTE |
| call_credit_spread | REJECT | — | — | against bullish underlying thesis |

Macro in hold window: 2026-09-04 Employment Situation; 2026-09-10 PPI; 2026-09-11 CPI; 2026-09-16 FOMC decision

---

### GLD · WATCH · **OPTIONS**

GLD — bullish Relative-Strength Leader in DATA UNAVAILABLE macro

GLD is a bullish idea in a weak risk on tape because Relative-Strength Leader (D), 20d RS vs SPY 8.09%.

| | |
|---|---|
| **Setup** | D — Relative-Strength Leader |
| **Lane** | SWING / bullish |
| **Last** | 405.31 |
| **Trade** | debit_call_spread BUY 415.0 call / SELL 426.0 call 2026-10-16 |
| **Pay / collect** | debit 3.65 |
| **Naive POP** | 39% — naive P(spot > breakeven 418.65) from ORATS call deltas. Not a backtested win rate. |
| **Conf** | 65 medium (quality of the *structure*, not P(win)) |
| **Score** | 39.0 |
| **Evidence** | no same-setup analog |
| **Stop / invalidation** | close back below 20 EMA / swing-low AVWAP |
| **RS 20d vs SPY** | 8.09% |
| **Trend / MAs** | range · 20 406.50 / 50 386.43 / 200 414.92 |
| **Group** | macro (SPY) · DATA UNAVAILABLE |
| **AVWAP** | year 430.35 · swing-low 392.28 |
| **ORATS IV/HV** | IV30 21.60 · HV20 22.40 · VRP -0.80 |
| **Earnings** | DATA UNAVAILABLE (exempt) |
| **X** | Informed — WSB #4. Gold/dollar/oil geo hedge on Hormuz. ETF Action: GLD shed -$628.6M. Macro hedge bid, not a squeeze. |
| **Fill** | long ask minus short bid (never mid) |
| **Greeks** | Δ 0.11 · Γ 0.0019 · Θ -0.0090 · ν 0.0485 |

More context:

Price is 405.31, trend range, 20 EMA 406.50 / 50 386.43 / 200 414.92, relative volume 0.2.

Group macro (SPY) is DATA UNAVAILABLE. AVWAP year 430.35, swing-low 392.28.

Earnings DATA UNAVAILABLE (exempt).

Instrument shortlist picked **OPTIONS**. Invalidation: close back below 20 EMA / swing-low AVWAP.

Tape notes:

- 20d RS vs SPY +8.1% with accumulation structure

All structures reviewed:

| structure | result | debit | credit | why |
|---|---|---:|---:|---|
| stock | PASS | — | — | stock plan clears R/R |
| long_call | REJECT | — | — | no liquid structure in 21-75 DTE |
| long_put | REJECT | — | — | against bullish underlying thesis |
| debit_call_spread | PASS | 3.65 | — | defined-risk directional; better than naked long when IV is not cheap |
| debit_put_spread | REJECT | 3.55 | — | against bullish underlying thesis |
| put_credit_spread | REJECT | — | — | no liquid structure in 21-75 DTE |
| call_credit_spread | REJECT | — | — | against bullish underlying thesis |

Macro in hold window: 2026-09-04 Employment Situation; 2026-09-10 PPI; 2026-09-11 CPI; 2026-09-16 FOMC decision

---

### SPY · IGNORE · **OPTIONS**

SPY — bullish Trend Pullback in DATA UNAVAILABLE index

SPY is a bullish idea in a weak risk on tape because Trend Pullback (A), 20d RS vs SPY 0.00%.

| | |
|---|---|
| **Setup** | A — Trend Pullback |
| **Lane** | SWING / bullish |
| **Last** | 764.90 |
| **Trade** | debit_call_spread BUY 781.0 call / SELL 792.0 call 2026-10-16 |
| **Pay / collect** | debit 3.56 |
| **Naive POP** | 31% — naive P(spot > breakeven 784.56) from ORATS call deltas. Not a backtested win rate. |
| **Conf** | 56 medium (quality of the *structure*, not P(win)) |
| **Score** | 35.0 |
| **Evidence** | no same-setup analog |
| **Stop / invalidation** | close back below 20 EMA / swing-low AVWAP |
| **RS 20d vs SPY** | 0.00% |
| **Trend / MAs** | up · 20 765.19 / 50 754.32 / 200 710.26 |
| **Group** | index (SPY) · DATA UNAVAILABLE |
| **AVWAP** | year 705.63 · swing-low 753.47 |
| **ORATS IV/HV** | IV30 11.97 · HV20 9.46 · VRP 2.51 |
| **Earnings** | DATA UNAVAILABLE (exempt) |
| **X** | Crowded — Month-end. Open ~-0.46% on futures. $2.1B put wall at 765. VIX ~15.5. Oil spike + Warsh hawkish vs AI bid. Indexes soft, not a crash. |
| **Fill** | long ask minus short bid (never mid) |
| **Greeks** | Δ 0.13 · Γ 0.0019 · Θ -0.0350 · ν 0.2101 |

More context:

Price is 764.90, trend up, 20 EMA 765.19 / 50 754.32 / 200 710.26, relative volume 0.2.

Group index (SPY) is DATA UNAVAILABLE. AVWAP year 705.63, swing-low 753.47.

Earnings DATA UNAVAILABLE (exempt).

Instrument shortlist picked **OPTIONS**. Invalidation: close back below 20 EMA / swing-low AVWAP.

Tape notes:

- trend pullback into 20 EMA / AVWAP / 50 DMA
- volume contracted on the pullback

All structures reviewed:

| structure | result | debit | credit | why |
|---|---|---:|---:|---|
| stock | PASS | — | — | stock plan clears R/R |
| long_call | REJECT | — | — | no liquid structure in 21-75 DTE |
| long_put | REJECT | — | — | against bullish underlying thesis |
| debit_call_spread | PASS | 3.56 | — | defined-risk directional; better than naked long when IV is not cheap |
| debit_put_spread | REJECT | 3.48 | — | against bullish underlying thesis |
| put_credit_spread | REJECT | — | — | no liquid structure in 21-75 DTE |
| call_credit_spread | REJECT | — | — | against bullish underlying thesis |

Macro in hold window: 2026-09-04 Employment Situation; 2026-09-10 PPI; 2026-09-11 CPI; 2026-09-16 FOMC decision

---

### TSLA · IGNORE · **OPTIONS**

TSLA — bullish Breakout + Confirmation in deteriorating consumer_discretionary

TSLA is a bullish idea in a weak risk on tape because Breakout + Confirmation (B), 20d RS vs SPY 12.21%.

| | |
|---|---|
| **Setup** | B — Breakout + Confirmation |
| **Lane** | SWING / bullish |
| **Last** | 364.50 |
| **Trade** | debit_call_spread BUY 380.0 call / SELL 390.0 call 2026-10-16 |
| **Pay / collect** | debit 3.30 |
| **Naive POP** | 39% — naive P(spot > breakeven 383.30) from ORATS call deltas. Not a backtested win rate. |
| **Conf** | 54 low (quality of the *structure*, not P(win)) |
| **Score** | 31.0 |
| **Evidence** | no same-setup analog |
| **Stop / invalidation** | close back below 20 EMA / swing-low AVWAP |
| **RS 20d vs SPY** | 12.21% |
| **Trend / MAs** | range · 20 346.97 / 50 359.75 / 200 400.55 |
| **Group** | consumer_discretionary (XLY) · deteriorating |
| **AVWAP** | year 394.74 · swing-low 335.02 |
| **ORATS IV/HV** | IV30 41.16 · HV20 40.40 · VRP 0.76 |
| **Earnings** | 2026-10-28 (web.alphaquery) |
| **X** | Crowded — Open recap: TSLA ~+3.4% while indexes soft. Relative-strength leader, already extended vs the tape. Promo mix + 0DTE call flow lists. |
| **Fill** | long ask minus short bid (never mid) |
| **Greeks** | Δ 0.07 · Γ 0.0004 · Θ -0.0127 · ν 0.0338 |

More context:

Price is 364.50, trend range, 20 EMA 346.97 / 50 359.75 / 200 400.55, relative volume 0.5.

Group consumer_discretionary (XLY) is deteriorating. AVWAP year 394.74, swing-low 335.02.

Earnings 2026-10-28 (web.alphaquery).

Instrument shortlist picked **OPTIONS**. Invalidation: close back below 20 EMA / swing-low AVWAP.

Tape notes:

- close above prior 20-session close high
- 20d RS vs SPY +12.2% with accumulation structure

All structures reviewed:

| structure | result | debit | credit | why |
|---|---|---:|---:|---|
| stock | PASS | — | — | stock plan clears R/R |
| long_call | REJECT | — | — | no liquid structure in 21-75 DTE |
| long_put | REJECT | — | — | against bullish underlying thesis |
| debit_call_spread | PASS | 3.30 | — | defined-risk directional; better than naked long when IV is not cheap |
| debit_put_spread | REJECT | 4.40 | — | against bullish underlying thesis |
| put_credit_spread | PASS | — | 1.05 | IV rich vs realized; defined-risk short premium. R/R is credit/width, not 2:1 di |
| call_credit_spread | REJECT | — | — | against bullish underlying thesis |

Macro in hold window: 2026-09-04 Employment Situation; 2026-09-10 PPI; 2026-09-11 CPI; 2026-09-16 FOMC decision

---

### CRWD · WATCH · **OPTIONS**

CRWD — bullish Post-Earnings Drift in mature cybersecurity

CRWD is a bullish idea in a weak risk on tape because Post-Earnings Drift (C), 20d RS vs SPY 10.79%.

| | |
|---|---|
| **Setup** | C — Post-Earnings Drift |
| **Lane** | SWING / bullish |
| **Last** | 226.32 |
| **Trade** | debit_call_spread BUY 250.0 call / SELL 255.0 call 2026-10-16 |
| **Pay / collect** | debit 1.65 |
| **Naive POP** | 33% — naive P(spot > breakeven 251.65) from ORATS call deltas. Not a backtested win rate. |
| **Conf** | 85 high (quality of the *structure*, not P(win)) |
| **Score** | 51.0 |
| **Evidence** | no same-setup analog |
| **Stop / invalidation** | close back below 20 EMA / swing-low AVWAP |
| **RS 20d vs SPY** | 10.79% |
| **Trend / MAs** | strong_up · 20 206.32 / 50 197.33 / 200 140.91 |
| **Group** | cybersecurity (CIBR) · mature |
| **AVWAP** | year 135.48 · swing-low 205.22 |
| **ORATS IV/HV** | IV30 53.17 · HV20 71.30 · VRP -18.13 |
| **Earnings** | 2026-12-01 (web.alphaquery) |
| **X** | Informed — Fal.Con: tape ~+3.7–3.8% while indexes red. Software confirmation leftover from last week's print. Already ran — do not chase the open. |
| **Fill** | long ask minus short bid (never mid) |
| **Greeks** | Δ 0.04 · Γ 0.0004 · Θ -0.0084 · ν 0.0136 |

More context:

Price is 226.32, trend strong_up, 20 EMA 206.32 / 50 197.33 / 200 140.91, relative volume 0.5.

Group cybersecurity (CIBR) is mature. AVWAP year 135.48, swing-low 205.22.

Earnings 2026-12-01 (web.alphaquery).

Instrument shortlist picked **OPTIONS**. Invalidation: close back below 20 EMA / swing-low AVWAP.

Tape notes:

- holding earnings AVWAP 5 sessions after lastErn
- 20d RS vs SPY +10.8% with accumulation structure

All structures reviewed:

| structure | result | debit | credit | why |
|---|---|---:|---:|---|
| stock | PASS | — | — | stock plan clears R/R |
| long_call | REJECT | — | — | no liquid structure in 21-75 DTE |
| long_put | REJECT | — | — | against bullish underlying thesis |
| debit_call_spread | PASS | 1.65 | — | defined-risk directional; better than naked long when IV is not cheap |
| debit_put_spread | REJECT | 4.50 | — | against bullish underlying thesis |
| put_credit_spread | REJECT | — | — | no liquid structure in 21-75 DTE |
| call_credit_spread | REJECT | — | — | against bullish underlying thesis |

Macro in hold window: 2026-09-04 Employment Situation; 2026-09-10 PPI; 2026-09-11 CPI; 2026-09-16 FOMC decision

---

### XOM · WATCH · **OPTIONS**

XOM — bullish Emerging Sector Rotation in emerging energy

XOM is a bullish idea in a weak risk on tape because Emerging Sector Rotation (E), 20d RS vs SPY 2.28%, energy group is emerging.

| | |
|---|---|
| **Setup** | E — Emerging Sector Rotation |
| **Lane** | SWING / bullish |
| **Last** | 160.07 |
| **Trade** | debit_call_spread BUY 165.0 call / SELL 175.0 call 2026-10-16 |
| **Pay / collect** | debit 2.80 |
| **Naive POP** | 36% — naive P(spot > breakeven 167.80) from ORATS call deltas. Not a backtested win rate. |
| **Conf** | 75 high (quality of the *structure*, not P(win)) |
| **Score** | 50.0 |
| **Evidence** | no same-setup analog |
| **Stop / invalidation** | close back below 20 EMA / swing-low AVWAP |
| **RS 20d vs SPY** | 2.28% |
| **Trend / MAs** | strong_up · 20 158.89 / 50 150.99 / 200 143.85 |
| **Group** | energy (XLE) · emerging |
| **AVWAP** | year 149.19 · swing-low 151.10 |
| **ORATS IV/HV** | IV30 27.30 · HV20 24.50 · VRP 2.80 |
| **Earnings** | 2026-10-30 (web.alphaquery) |
| **X** | Informed — Named with CVX as the oil-spike expression. Tape ~+2% with the energy group. Morgan Stanley raised Brent/WTI path on Middle East re-escalation. |
| **Fill** | long ask minus short bid (never mid) |
| **Greeks** | Δ 0.20 · Γ 0.0069 · Θ -0.0181 · ν 0.0455 |

More context:

Price is 160.07, trend strong_up, 20 EMA 158.89 / 50 150.99 / 200 143.85, relative volume 0.2.

Group energy (XLE) is emerging. AVWAP year 149.19, swing-low 151.10.

Earnings 2026-10-30 (web.alphaquery).

Instrument shortlist picked **OPTIONS**. Invalidation: close back below 20 EMA / swing-low AVWAP.

Tape notes:

- group emerging with positive 20d RS
- trend pullback into 20 EMA / AVWAP / 50 DMA
- volume contracted on the pullback

All structures reviewed:

| structure | result | debit | credit | why |
|---|---|---:|---:|---|
| stock | PASS | — | — | stock plan clears R/R |
| long_call | REJECT | — | — | no liquid structure in 21-75 DTE |
| long_put | REJECT | — | — | against bullish underlying thesis |
| debit_call_spread | PASS | 2.80 | — | defined-risk directional; better than naked long when IV is not cheap |
| debit_put_spread | REJECT | 2.66 | — | against bullish underlying thesis |
| put_credit_spread | REJECT | — | — | no liquid structure in 21-75 DTE |
| call_credit_spread | REJECT | — | — | against bullish underlying thesis |

Macro in hold window: 2026-09-04 Employment Situation; 2026-09-10 PPI; 2026-09-11 CPI; 2026-09-16 FOMC decision

---

### XLU · IGNORE · **STOCK**

XLU — bearish Failed Breakout / Trend Breakdown in deteriorating utilities

XLU is a bearish idea in a weak risk on tape because Failed Breakout / Trend Breakdown (G), 20d RS vs SPY -6.42%.

| | |
|---|---|
| **Setup** | G — Failed Breakout / Trend Breakdown |
| **Lane** | SWING / bearish |
| **Last** | 41.94 |
| **Trade** | STOCK short entry 41.94 stop 43.27 target 39.26 R/R 2.00 shares 373 |
| **Pay / collect** | stock (no option premium) |
| **Naive POP** | n/a — n/a for stock; no delta-based POP |
| **Conf** | n/a n/a (quality of the *structure*, not P(win)) |
| **Score** | 2.0 |
| **Evidence** | no same-setup analog |
| **Stop / invalidation** | close beyond stop 43.27 |
| **RS 20d vs SPY** | -6.42% |
| **Trend / MAs** | strong_down · 20 43.56 / 50 44.63 / 200 44.70 |
| **Group** | utilities (XLU) · deteriorating |
| **AVWAP** | year 44.96 · swing-low 43.54 |
| **ORATS IV/HV** | IV30 16.76 · HV20 16.82 · VRP -0.06 |
| **Earnings** | DATA UNAVAILABLE (exempt) |
| **X** | Informed — Utilities worst sector (~-1.7%). CA SB 492 failed to refill wildfire fund / cap liability. PCG ~-19%, EIX ~-20% after Mizuho/BMO cuts. Sector dump is two names, not rates. |
| **Fill** | stock last; options never mid |
| **Greeks** | Δ DATA UNAVAILABLE · Γ DATA UNAVAILABLE · Θ DATA UNAVAILABLE · ν DATA UNAVAILABLE |

More context:

Price is 41.94, trend strong_down, 20 EMA 43.56 / 50 44.63 / 200 44.70, relative volume 0.4.

Group utilities (XLU) is deteriorating. AVWAP year 44.96, swing-low 43.54.

Earnings DATA UNAVAILABLE (exempt).

Instrument shortlist picked **STOCK**. Invalidation: close beyond stop 43.27.

Tape notes:

- below declining/lost 20 EMA in a downtrend

All structures reviewed:

| structure | result | debit | credit | why |
|---|---|---:|---:|---|
| stock | PASS | — | — | stock plan clears R/R |
| long_call | REJECT | 0.99 | — | against bearish underlying thesis |
| long_put | PASS | 1.02 | — | cheap-to-fair vol directional with 21-75 DTE |
| debit_call_spread | REJECT | 0.97 | — | against bearish underlying thesis |
| debit_put_spread | REJECT | — | — | no liquid structure in 21-75 DTE |
| put_credit_spread | REJECT | — | — | against bearish underlying thesis |
| call_credit_spread | REJECT | — | — | no liquid structure in 21-75 DTE |

Macro in hold window: 2026-09-04 Employment Situation; 2026-09-10 PPI; 2026-09-11 CPI; 2026-09-16 FOMC decision


## Data caveats

- ORATS rows: 97 · error: none
- Option chains pulled only for top underlying theses: CRM, SHOP, XLE, CVX, SNOW, NET, ANET, PLTR, SMCI, NOW, ADBE, VRTX, TSLA, SNPS, IGV, HOOD, CRWD, XBI, PANW, DELL, NVDA, MU, XOM, XLU, GLD, GOOGL, AMZN, SPY
- X is confirm/veto only. Write `var/xintel/DATE/TICKER.json` after searching $TICKER. Missing file → DATA UNAVAILABLE. FIRE needs volume+price first.
- Dealer GEX is not computed. Do not present estimated GEX as fact.
- Conservative option fills: debit at ask, credit at short bid minus long ask. Never assume midpoint.
