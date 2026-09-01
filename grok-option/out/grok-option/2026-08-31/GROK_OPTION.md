# grok-option — 2026-08-31

EOD refresh → **Tue 2026-09-01 open** · no cache · SCAN only

🟢 send-list · 🟡 review (cleared, parked) · 🔴 blocked / failed · 🔵 Fire quote · ⚡ Spike

---

**Equity** $717,281 · **Cash** $23,764 · **Bar** $10k/month
**Regime** 😌 **Calm** · VIX **15.22** (GuruFocus 8/31, +5.47% vs Fri 14.43) · 5-session 8/24 15.85 → 15.22 ≈ **−4.0%** · 52w 13.38–35.30
**Index** SPX ~**7,677** (📉 −0.45% vs Fri) · SPY **767.55** (📉 −0.23%)
**Board** 🟢 Expert **2** · 🟡 Review **6** · 🔵 Fire **12** · 🔴 Fail **6** credits
**Shock** ⚡ sourced Hormuz kinetic · **no Spike row**
**Data** 📒 MIXED (Schwab EOD/AH · Mag7 next print estimated Oct 28–29)

**Calendar** JOLTS Tue 9/1 10:00 (not Index Event) · MDT BMO / NIO / DELL+PANW AMC **name Events** · AVGO+SNOW AMC Wed **theme = semis/AI-infra** · NFP Fri 9/4 · Labor Day 9/7 · CPI 9/11 · FOMC 9/16. Skip expiries that land on 9/4, 9/11, 9/16. Oct-2 is NFP — not used.

> [!IMPORTANT]
> Re-quote both 🟢 wings at **9:35 Tuesday**. Do not 9:31 auto-send. After-hours mids are wide.

> [!WARNING]
> MSFT put 475/465 is **0.12 exact** and ⚠️ worse-fill (mid 1.70 vs net 1.20). If 9:35 frac < 0.12, drop the condor — do not mid it in.

> [!TIP]
> 🟡 TSLA / META / GOOGL / GE **still clear**. They are on the review board, not dropped. Promote order is in Watch.

---

## 🟢 Expert — send-list

1-lot dollars. Rec lots from Conf, then Shield 1.0% ($7,173), then cash/BP.

### 🟢 AMZN · 🛡️ Shield · Sell iron condor

**2026-10-16** · Conf **79% naive · book n/a** · Score 65 · 📒 MIXED · X: 🤫 Quiet · 📦 shares held

| Buy (long) | Sell (short) | Credit | 1-lot P/L | Rec lots |
|------------|--------------|--------|-----------|----------|
| Buy 230 Put + Buy 295 Call | Sell 240 Put + Sell 285 Call | **2.70** | +$270 / −$730 | **7** |

- Spot **259.77** · ATM straddle ask **22.25**
- Put 240/230 net **1.44** (mid 1.60, frac 0.144, Δ −0.208, 0.89σ) ⚠️ worse-fill
- Call 285/295 net **1.26** (mid 1.435, frac 0.126, Δ 0.214, 1.13σ) ⚠️ worse-fill
- Condor Conf from **call-wing** `round(100 × (1 − max(0.208, 0.214))) = 79%`
- 📦 close the spread; do not deliver unless asked. Nov-20 short 230P is a **different expiry**
- Rec 7 → ~+$1,890 / −$5,110 N-lot. Earnings estimated ~Oct 29 · Oct-16 is before that window
- Sector: consumer discretionary

### 🟢 MSFT · 🛡️ Shield · Sell iron condor

**2026-10-16** · Conf **78% naive · book n/a** · Score 65 · 📒 MIXED · X: 🤫 Quiet · no shares

| Buy (long) | Sell (short) | Credit | 1-lot P/L | Rec lots |
|------------|--------------|--------|-----------|----------|
| Buy 465 Put + Buy 560 Call | Sell 475 Put + Sell 550 Call | **2.45** | +$245 / −$755 | **7** |

- Spot **507.29** · ATM straddle ask **37.85**
- Put 475/465 net **1.20** (mid 1.70, frac **0.12 exact**, Δ −0.219, 0.85σ) ⚠️ **fragile wing**
- Call 550/560 net **1.25** (mid 1.60, frac 0.125, Δ 0.205, 1.13σ) ⚠️ worse-fill
- Condor Conf from **put-wing** **78%**
- Rec 7 → ~+$1,715 / −$5,285 N-lot. Earnings estimated ~Oct 28
- Sector: IT · **MSFT stock is not held**

**Do not fill both at Rec 7** into the same cash/BP stack (cash $23.8k, 31 option lines open).

| ● | Ticker | Action | Expiry | 1-lot P/L | Rec | Conf | Data |
|---|--------|--------|--------|-----------|-----|------|------|
| 🟢 | AMZN | Sell iron condor | 2026-10-16 | +$270 / −$730 | 7 | 79% | 📒 |
| 🟢 | MSFT | Sell iron condor | 2026-10-16 | +$245 / −$755 | 7 | 78% | 📒 |

⛔ **Do not add META** here. Jan-27 520/500 is live. Oct 520/505 is 🟡 below.

If 9:35 fails geometry: 🔴 **empty table is the scan.** Do not loosen 0.22Δ / 0.80σ / 0.12.

---

## Shock watch

| ● | Item | Source | Status |
|---|------|--------|--------|
| ⚡ | CENTCOM strike, Larak Island IRGC launchers prepping sea-mine rockets into Hormuz (8/30) | Reuters, NYT, Politico, Axios, Newsweek · X finder | sourced + map · **no ±5% crude** |
| 🔴 | WTI ~$84.9 📈 +1.1% session · Brent ~$90.3–90.6 📈 +0.7–1.3% · 2-session WTI 8/28 $82.78 → $84.9 ≈ +2.6% | Investing / Twelve Data | oil-spike path fail (need +5% session or +8% two sessions) |
| 🟡 | XOM $161.20 📈 +2.87% · CVX $206.87 📈 +2.48% · COP $132.50 📈 +1.65% | Schwab EOD | map moving with oil · not a new row |
| ⚡ | XOM Sep-18 165/170 call debit net **1.54** (mid 1.275, long Δ 0.355, frac 0.308) | Schwab vertical | quoted · 🟡 parked — earnings estimated Oct 30 · CVX Spike already live |
| 🔴 | Defense LMT / RTX / GD | no extra state-on-state invasion print | energy map would win · LMT 📉 −0.15% |

No new 🛡️ Shield on XOM / CVX / COP this session (mapped shock names).

---

## Sleeve board

| Structure | 🟢 In table | 🟡 Cleared, parked |
|-----------|-------------|-------------------|
| 🛡️ Sell put credit | AMZN 240/230 · MSFT 475/465 (condor wings) | TSLA 330/315 · META 520/505 · GOOGL 315/300 · GE 310/300 · NFLX 72.5/70 |
| 🛡️ Sell call credit | AMZN 285/295 · MSFT 550/560 (condor wings) | 🔴 no other call credit met Calm |
| 🛡️ Sell iron condor | **AMZN · MSFT** | none else (need both wings) |
| 🔵 Buy call debit | — | quoted · no opening flow |
| 🔵 Buy put debit | — | quoted · no opening flow |
| ⚡ Spike | — | XOM 165/170 quoted · CVX debit live |

---

## 🟡 Cleared structures — review

These **still printed** on the fresh EOD chain. They are not dropped. Expert table is max 4 Shield, **one per sector**.

| ● | Ticker | Sleeve | Action | Expiry | Buy (long) | Sell (short) | Max profit $ | Max loss $ | Rec lots | Score | Conf | Data | Why |
|---|--------|--------|--------|--------|------------|--------------|--------------|------------|----------|-------|------|------|-----|
| 🟢 | AMZN | 🛡️ Shield | Sell iron condor | 2026-10-16 | Buy 230 Put + Buy 295 Call | Sell 240 Put + Sell 285 Call | 270 | 730 | 7 | 65 | 79% naive · book n/a | 📒 MIXED | send-list |
| 🟢 | MSFT | 🛡️ Shield | Sell iron condor | 2026-10-16 | Buy 465 Put + Buy 560 Call | Sell 475 Put + Sell 550 Call | 245 | 755 | 7 | 65 | 78% naive · book n/a | 📒 MIXED | send-list |
| 🟡 | TSLA | 🛡️ Shield | Sell put credit | 2026-10-16 | Buy 315 Put | Sell 330 Put | 250 | 1250 | 4 | 65 | 79% naive · book n/a | 📒 MIXED | same sector as AMZN · X: 📢 Crowded |
| ⛔ | META | 🛡️ Shield | Sell put credit | 2026-10-16 | Buy 505 Put | Sell 520 Put | 240 | 1260 | 4 | 65 | 80% naive · book n/a | 📒 MIXED | ⛔ do not add · Jan 520/500 live |
| 🟡 | GOOGL | 🛡️ Shield | Sell put credit | 2026-10-16 | Buy 300 Put | Sell 315 Put | 217 | 1283 | 4 | 65 | 79% naive · book n/a | 📒 MIXED | comms slot has META Jan credit |
| 🟡 | GE | 🛡️ Shield | Sell put credit | 2026-10-16 | Buy 300 Put | Sell 310 Put | 135 | 865 | 6 | 50 | 78% naive · book n/a | 📒 MIXED | AH wide (mid 1.875 vs 1.35) |
| 🟡 | NFLX | 🛡️ Shield | Sell put credit | 2026-10-16 | Buy 70 Put | Sell 72.5 Put | 32 | 218 | 1 | 50 | 84% naive · book n/a | 📒 MIXED | $32 credit < $100 floor |
| ⚡ | XOM | ⚡ Spike | Buy call debit | 2026-09-18 | Buy 165 Call | Sell 170 Call | 346 | 154 | 4 | 50 | 36% naive · book n/a | 📒 MIXED | earnings estimated · CVX Spike live |

### 🟡 TSLA · 🛡️ Sell put credit · 2026-10-16

Conf **79%** · Rec **4** · +$250 / −$1,250 · 📦 shares held · X: 📢 Crowded

- Spot **367.95** · sigma **43.60** · net **2.50** (mid 2.725, frac **0.167**, Δ −0.206, 0.87σ) ⚠️ worse-fill
- Call wing **failed** Calm → put-only, not a condor
- 📢 Texas Cybercab/DMV · 📈 +5.35% · Thu 9/3 Austin event · lottery/0DTE chatter
- Crowded does **not** veto Shield · **sector cap vs AMZN condor does**
- Promote only if AMZN is skipped or AMZN’s call wing dies at 9:35
- Rec 4 → ~+$1,000 / −$5,000

### ⛔ META · 🛡️ Sell put credit · 2026-10-16

Conf **80%** · Rec **4** · +$240 / −$1,260

- Spot **572.34** · sigma **57.50** · net **2.40** (mid 2.825, frac **0.160**, Δ −0.199, 0.91σ) ⚠️ worse-fill
- Call wing failed
- Live book: short Jan-27 **520P** / long **500P** — same 520 short family
- User instruction: pick something else. Rec 4 → ~+$960 / −$5,040 **only if you override**

### 🟡 GOOGL · 🛡️ Sell put credit · 2026-10-16

Conf **79%** · Rec **4** · +$217 / −$1,283

- Spot **339.35** · sigma **27.45** · net **2.17** (mid 2.36, frac **0.145**, Δ −0.212, 0.89σ) ⚠️ worse-fill
- Call wing failed
- Communication Services — same sector as live META Shield
- Rec 4 → ~+$868 / −$5,132 if you treat META as not filling comms

### 🟡 GE · 🛡️ Sell put credit · 2026-10-16

Conf **78%** · Rec **6** · +$135 / −$865 · Score **50**

- Spot **335.71** · sigma **30.40** · net **1.35** (mid **1.875**, frac **0.135**, Δ −0.217, 0.85σ) ⚠️ +39% worse-fill
- Industrials is an **open** Expert sector if Tuesday bid/ask tightens
- Rec 6 → ~+$810 / −$5,190

### 🟡 NFLX

Geometry yes · dollar picker no · net 0.32 on 2.5-wide.

### ⚡ XOM Spike · Buy call debit · 2026-09-18

- Debit net **1.54** · width 5 · frac 0.308 · long Δ 0.355
- Spike 0.25% ($1,793) → Rec 4
- 🟡 Not Expert: earnings not IR-confirmed · one Spike already on CVX Sep-25 210/220

---

## 🔵 Fire — quoted, no opening flow

Debit/width 0.25–0.55. Long |Δ| nearest 0.40. Missing flow blocks Score 80, not the quote. Semis frozen until after AVGO.

| ● | Ticker | Action | Expiry | Buy (long) | Sell (short) | Net | Frac | Long Δ | 1-lot P/L | Why |
|---|--------|--------|--------|------------|--------------|-----|------|--------|-----------|-----|
| 🔵 | MSFT | Buy call debit | 2026-10-16 | Buy 520 Call | Sell 525 Call | 2.65 | 0.53 | 0.415 | +$235 / −$265 | no flow |
| 🔵 | MSFT | Buy put debit | 2026-10-16 | Buy 500 Put | Sell 490 Put | 4.45 | 0.445 | −0.416 | +$555 / −$445 | no flow |
| 🔵 | AMZN | Buy call debit | 2026-10-16 | Buy 270 Call | Sell 275 Call | 2.00 | 0.40 | 0.387 | +$300 / −$200 | no flow |
| 🔵 | AMZN | Buy put debit | 2026-10-16 | Buy 255 Put | Sell 250 Put | 2.25 | 0.45 | −0.402 | +$275 / −$225 | no flow |
| 🔴 | TSLA | Buy call debit | 2026-10-16 | Buy 385 Call | Sell 390 Call | 2.05 | 0.41 | 0.414 | +$295 / −$205 | 📢 Crowded veto + no flow |
| 🔴 | TSLA | Buy put debit | 2026-10-16 | Buy 360 Put | Sell 355 Put | 2.45 | 0.49 | −0.411 | +$255 / −$245 | 📢 Crowded veto + no flow |
| 🔵 | AAPL | Buy call debit | 2026-10-16 | Buy 325 Call | Sell 330 Call | 2.35 | 0.47 | 0.411 | +$265 / −$235 | no flow |
| 🔵 | GOOGL | Buy call debit | 2026-10-16 | Buy 350 Call | Sell 355 Call | 2.20 | 0.44 | 0.404 | +$280 / −$220 | no flow |
| 🔵 | META | Buy call debit | 2026-10-16 | Buy 595 Call | Sell 600 Call | 2.45 | 0.49 | 0.405 | +$255 / −$245 | no flow |
| 🔵 | HD | Buy call debit | 2026-10-16 | Buy 335 Call | Sell 340 Call | 2.70 | 0.54 | 0.406 | +$230 / −$270 | no flow |
| 🔵 | GE | Buy call debit | 2026-10-16 | Buy 350 Call | Sell 360 Call | 4.10 | 0.41 | 0.374 | +$590 / −$410 | no flow · wide |
| ⚡ | XOM | Buy call debit | 2026-09-18 | Buy 165 Call | Sell 170 Call | 1.54 | 0.308 | 0.355 | +$346 / −$154 | Spike lane, not Fire |

---

## 🔴 Failed — scanned, no credit row

Calm need **all three**: `|Δ| ≤ 0.22` · `≥ 0.80σ` · credit/width `≥ 0.12`

| ● | Ticker | Expiry | What failed |
|---|--------|--------|-------------|
| 🔴 | AAPL | Oct-16 | put credit no · call credit no (σ 22.45, spot 316.85) |
| 🔴 | HD | Oct-16 | put credit no · call credit no (σ 25.65, spot 327.83) |
| 🔴 | WMT | Oct-16 | put credit no · call credit no (σ 6.80, spot 104.87) |
| 🔴 | COST | Sep-18 | put/call credit no (σ 35.80). Earnings **confirmed 9/24** → Oct-16 would overlap |
| 🔴 | JPM | Sep-18 | put/call credit no (σ 13.25). TipRanks **10/13** → Oct-16 would overlap |
| 🔴 | UNH | Sep-18 | put/call credit no (σ 20.20). TipRanks **10/09** → Oct-16 would overlap |

Call credits also failed on TSLA, META, GOOGL, GE, NFLX → those are **put-only**, not condors.

---

## Quote tape · Schwab EOD

| Ticker | Last | Chg |
|--------|------|-----|
| SPY | 767.55 | 📉 −0.23% |
| MSFT | 507.78 | 📉 −1.12% |
| AMZN | 260.20 | 📉 −2.34% |
| GOOGL | 339.61 | 📉 −2.01% |
| META | 572.00 | 📉 −1.04% |
| TSLA | 367.42 | 📈 +5.35% |
| AAPL | 316.99 | 📉 −0.85% |
| NVDA | 220.27 | 📈 +1.25% |
| AMD | 469.60 | 📈 +0.86% |
| AVGO | 371.31 | 📈 +0.68% |
| HD | 328.40 | 📉 −0.54% |
| JPM | 356.15 | 📉 −0.41% |
| WMT | 104.80 | 📈 +1.66% |
| COST | 943.89 | 📉 −0.17% |
| UNH | 390.50 | 📉 −0.62% |
| NFLX | 81.10 | 📉 −0.76% |
| GE | 335.87 | 📉 −1.96% |
| XOM | 161.20 | 📈 +2.87% |
| CVX | 206.87 | 📈 +2.48% |
| COP | 132.50 | 📈 +1.65% |

---

## Assumptions in force

- 😌 Calm Shield is **0.12 credit/width, |Δ| ≤ 0.22, ≥ 0.80σ**. Do not drop these to fill a Tuesday table.
- No protected-ticker list. Call credit / condor if geometry + earnings + Event/Crowded + quotes pass. 📦 shares held → Notes tag, close the spread rather than deliver.
- 🔵 Fire without opening flow stays on review, not Expert. Score ≠ P(win). Conf = quoted-delta naive POP · book n/a.
- Table dollars are **1 lot**. Rec lots from Conf, then 1.0% of $717k, then cash/BP. Do not size up to chase $10k/month.
- Events scoped: DELL/PANW/MDT/NIO = **name** · AVGO Wed = **semis/AI-infra theme** · JOLTS ≠ Index Event · NFP/CPI/FOMC skip **that date’s expiry**.
- Dated `GROK_OPTION.md` is the deliverable. 🟡 geometry-pass trades stay as full rows.

## Book

Equity **$717,281** · Cash **$23,764** · 31 option lines
🛡️ Shield 1.0% ≈ $7,173 then BP · Aggregate 6–8% ($43k–$57k) · Theme 30% of aggregate

**Live overlay**

| ● | Line |
|---|------|
| ⛔ | META Jan-27 520/500 put credit (short 520P / long 500P) |
| 🟡 | AMZN Nov-20 short 230P −1 |
| ⚡ | CVX Sep-25 210/220 call debit (Spike slot filled) |
| 🟡 | GOOG Nov 280/210 put |
| 🟡 | NFLX Oct 70/50 put |
| 🔴 | NVDA 8/31 242.5C leftover ~$0.50 (expires today) |

**MSFT stock is not held.** LONG_EQ includes AAPL AMD AMZN AVGO COST CRM GOOG HOOD META NFLX NVDA ORCL TSLA UNH.

Cash is **not** the position if 9:35 still prints the two 🟢 condors. Cash **is** the bind on filling both at Rec 7.

## Watch

- 🔴 earnings-overlap: COST confirmed **9/24** · JPM TipRanks **10/13** · UNH **10/09** · Mag7 estimated Oct 28–29 · GE/NFLX ~Oct 20
- 🔴 theme freeze: NVDA / AMD / AVGO until the session after AVGO · DELL/PANW Tuesday name Event
- 🟢 Tuesday morning: refresh the two Expert tickets first. Full rescan only if VIX leaves Calm, crude ±5%, a new sourced shock, or you want a new board
- 🟡 Promote if a 🟢 wing dies at 9:35: **TSLA 330/315** (only if AMZN is also out) → **GE 310/300** if the ask tightens → **GOOGL 315/300** only if META does not fill comms

SCAN only. No orders.
