# Compound Core playbook

Long-term **core** sleeve. Compounding, no stock-picking, no options, no orders. Separate from groat / wheelo / xhigh / groko / Codex Daily. Trading desks never touch this bucket.

As-of **4 Sep 2026**. Mixed taxable + IRA. Money needed inside 5 years does not belong here. Living-expense cash sits **outside** the 5% VGSH.

Calculator (enter $ amount, both sleeves):

```bash
cd /Users/anuppamvi/tradedesk/compoundcore
PYTHONPATH=. python3 -m compoundcore 100000 --weekly 250 --monthly 1000
PYTHONPATH=. python3 -m compoundcore calculator   # writes web/calculator.html
```

Open [`web/calculator.html`](../web/calculator.html) and type an amount. Default and aggressive split at once.

---

## Independent review (this implementation)

Verdict on the reviewed plan: **keep the default 48 / 10 / 7 / 5 / 20 / 5 / 5**. The ~20% the plan refused (capex-quarter SMH sell, silent 35% international book, 40/60 bonds) stays refused.

Three citations were corrected so the playbook does not overclaim Vanguard:

1. **Vanguard publishes 25th–75th ranges, not a 5.7% median.** US equities as of 30 Jun 2026: **4.2–6.2%** (midpoint **5.2%**). The 5.8% sleeve base is a **blend** (Vanguard midpoint 5.2 + J.P. Morgan 6.7 + 14-house median 6.0), labeled as such.
2. **Latest Vanguard total ex-US is not above US.** After the Q2 2026 international rally, global ex-US is **3.9–5.9%** vs US **4.2–6.2%**. DM ex-US is still slightly higher (**4.5–6.5%**). VXUS 20% stays as **diversification + 14-house median** (DM 7.2 vs US 6.0) and a **labeled US home-bias**. It is not Vanguard's latest model output.
3. **VGSH was missing from the building-block table.** Vanguard US short-term Treasury **3.5–4.5%** (midpoint 4.0%) is now the VGSH block. Weighted 10y base with that 4% is **5.80%**, matching the published default table.

X API credits were **0** on this run, so the DeepValueBagger thread body is **DATA UNAVAILABLE** here. TinyToothDDS's public bio independently lists ToothFolio as `$VOO $VGT $VB $SMH $XLE $XAR $SHLD`. VanEck SMH NAV 52-week high **$668.69**; 2 Sep 2026 close **$550.48** (~18% off the high). Look-through NVDA uses dated holdings (VOO 7.55% 31 Jul, VGT 16.2% 30 Jun, SMH 21.94% 24 Aug) → default **6.8%**, not the plan's 6.9%.

The 40.48% screenshot is a trailing 1-year account number during the 2025–2026 chip boom, not a forecast. A 50/30/20 VOO/VGT/SMH mix on TotalRealReturns through 21 Aug 2026 is about **+42% 1Y** and **~22%/yr 10Y**. That 10Y window is the best chip/AI decade on record.

---

## 1. Default sleeve + weekly buy

| Role | ETF | Weight | Fee | Why this size |
|---|---|---:|---:|---|
| US compounding engine | **VOO** | **48%** | 0.03% | Own America. Already ~1/3 tech. |
| Tech overweight | **VGT** | **10%** | 0.09% | Extra software/cloud without cloning VOO. |
| Chip satellite | **SMH** | **7%** | 0.35% | AI/chip cycle. −45% SMH ≈ **−3.2%** of core. DVB's "too late" is respected. |
| Small-cap kicker | **VB** | **5%** | 0.03% | ToothFolio size bet. Broad, cheap, not the hedge. |
| Non-US stocks | **VXUS** | **20%** | 0.05% | Diversifier. Equity book is **~78% US** vs ~60–65% world. Labeled home-bias. Cap-neutral would be ~30–35% VXUS. |
| Inflation / geo hedge | **GLDM** | **5%** | 0.10% | Real diversifier. Cut from 10% because gold is extended after 2025. |
| Crash-protocol reserve | **VGSH** (or SGOV) | **5%** | 0.04% | Short Treasuries. This is the sleeve Rule 4 buys the dip *with*. |

Blended expense **0.066%/yr**. Look-through NVDA **6.8%**.

**Weekly recipe per $1,000** (scale linearly; largest-remainder cents so rows sum):

| VOO | VGT | SMH | VB | VXUS | GLDM | VGSH |
|---:|---:|---:|---:|---:|---:|---:|
| $480 | $100 | $70 | $50 | $200 | $50 | $50 |

Do not wait for a dip on VOO / VXUS / VGSH. If SMH or VGT is already through its **high** band, that week's slice goes to VOO.

**Dropped from ToothFolio:** XAR, SHLD, XLE, individual stocks, BTC, QQQ/SOXX stacked on VGT/SMH.

---

## 2. 5-year and 10-year dollar paths (current market)

Geometric, nominal, **before inflation and taxes**. Monthly compounding, end-of-month adds. Arithmetic: lump = `PV × (1+r)^n`. DCA uses monthly rate `(1+r)^(1/12)−1` so a 5% path on a lump is exactly `1.05^5`.

Default sleeve rates (locked):

| Horizon | Stress | Bear | Base | Bull |
|---|---:|---:|---:|---:|
| **5-year annualized** | **−1.0%** | **2.0%** | **5.0%** | **9.5%** |
| **10-year annualized** | **0.5%** | **3.2%** | **5.8%** | **8.1%** |

Building blocks behind the 10y mix (VGSH added):

| Sleeve | Bear | Base | Bull |
|---|---:|---:|---:|
| VOO | 3.2% | 5.8% | 8.1% |
| VGT | 2.8% | 5.2% | 7.6% |
| SMH | 0.0% | 5.5% | 11.0% |
| VB | 3.5% | 6.1% | 8.8% |
| VXUS | 5.2% | 6.9% | 8.5% |
| GLDM | 1.0% | 4.5% | 7.0% |
| VGSH | 4.0% | 4.0% | 4.0% |

Weighted default 10y base = **5.80%**. Weighted bear with VGSH at 4% is **3.3%**; the table keeps the published **3.2%**. 5y is **wider** than 10y: rich CAPE (mean reversion often front-loaded) and AI can still run.

Two math caveats: these are weighted geometric averages and **omit the rebalancing bonus**. DCA sequence risk runs opposite lump-sum — a year-1 crash **helps** the weekly buyer.

### $100,000 lump, no more contributions

| Path | 5-year | 10-year |
|---|---:|---:|
| Stress (−1% / 0.5%) | **$95k** | **$105k** |
| Bear (2.0% / 3.2%) | $110k | $137k |
| **Base (5.0% / 5.8%)** | **$128k** | **$176k** |
| Bull (9.5% / 8.1%) | $157k | $218k |
| Fantasy: 40%/yr forever | $538k | $2.89M |
| VOO-only at VG 4.2–6.2 midpoint 5.2% 10y | — | $166k |

Fantasy 5y is `100000 × 1.4^5 = $537,824` (rounds to $538k). The viral 40% year is not a budget.

### $1,000,000 lump

Ten times the $100k table: base **$1.28M / $1.76M**.

### $100,000 start + $1,000/month

You add **$60k / $120k** of new cash. This is the tweet's real trick.

| Path | 5-year | 10-year |
|---|---:|---:|
| Stress | $154k | $228k |
| Bear | $173k | $278k |
| **Base** | **$195k** | **$337k** |
| Bull | $233k | $399k |

After 2% inflation, base $100k lump is **~$144k real** in 10 years. Stress lump 10y is **~$105k nominal / ~$86k real** — that is the path ballast is for.

---

## 3. Look-through, fees, tax, SMH history

**Look-through NVDA (dated holdings, not an operating trigger):**

| Fund | NVDA weight | As-of |
|---|---:|---|
| VOO | 7.55% | 31 Jul 2026 |
| VGT | 16.2% | 30 Jun 2026 |
| SMH | 21.94% | 24 Aug 2026 |

Default mix ≈ **6.8% NVDA**. Tweet 50/30/20 ≈ **13.0%**. SMH is 26 names, top 10 ~72%, ER 0.35%, non-diversified.

**Tax location.** Prefer **VXUS in IRA** (TTM yield ~2.7%; 2025 income return 4.33%). Planning yield ~3%. 20% of the book × 15–24% ordinary tax on distributions ≈ **9–14 bps** of portfolio drag in taxable — comparable to the entire 6.6 bp expense ratio. VOO/VGT/SMH/GLDM/VGSH can live in taxable. No wash-sale overlap with the trading sleeve.

**SMH 2000–08 max drawdown** cites the **HOLDRS predecessor** (converted to the current ETF in 2011). Treat that ~−85% as industry history, not this share class. 2022 SMH calendar −33.5%; 2022 peak-to-trough about −45%. A −45% SMH crash is a **single-digit** core hit at 7%.

VGT 10Y realized ~24%/yr and SMH ~34%/yr are the AI/chip cycle, not a law of nature. VOO 10Y ~15.5% vs Vanguard's **forward** US-equity range 4.2–6.2%.

Shiller CAPE **~41–42** as of 1–2 Sep 2026 (Multpl 41.93 on 2 Sep; GuruFocus 40.77 on 1 Sep). 10-year Treasury **~4.79%** (2 Sep). High starting CAPE maps to **weaker 7–10 year returns**, not a crash on a timer.

---

## 4. Operating rules (this is the actual edge)

One band system. No second SMH cap. No NVDA look-through trigger.

1. **Account split.** Named Schwab bucket. Trading desks never touch it. Never sell core to fund groat / wheelo / xhigh tickets.
2. **Weekly DCA.** Split each contribution by target weights. Do not wait for a dip on VOO / VXUS / VGSH. If SMH/VGT is already through the high band, that week's slice goes to VOO.
3. **Relative rebalance bands.** Trigger when a sleeve is **±25% of its target, with a 2 percentage-point floor**. SMH 7% → **5–9%**. VB 5% → **3–7%**. VOO 48% → **36–60%**. VGT 10% → **7.5–12.5%**. VXUS 20% → **15–25%**. GLDM/VGSH 5% → **3–7%**. **Taxable: fill bands with new cash first. Sell only if a high band is still breached at the annual check.** IRA: sell/buy to target at the annual check or on a band hit.
4. **Crash protocol.** If VOO is ≥20% off its 52-week high, spend **VGSH first**, then outside living-expense cash. Buy **VOO, then VXUS, then VB**. SMH last. Do not sell GLDM to fund the dip unless VGSH is already gone.
5. **Core is never abandoned.** VOO / VXUS / VB / VGSH / GLDM are permanent. SMH is trimmed **only by the bands in Rule 3** (and by skipping its weekly slice when it is already through the high band). A two-quarter drop in foundry/equipment capex guidance is **information**, not a sell trigger. Annual review question: "is the chip satellite still a 7% idea?" If no, map SMH into VOO at the annual check, not on a headline.
6. **Tax location.** VXUS in IRA when possible. No wash-sale overlap with the trading sleeve.
7. **Outside cash.** 6–12 months of spending plus trading-desk capital. VGSH is *portfolio* dry powder, not rent money.
8. **No hero trades, no options, no leverage, no margin** on this sleeve.

---

## 5. Aggressive variant

Use only if you accept ~−30% equity drawdowns. SMH −45% still ≈ **−4.5%** of core.

| ETF | Weight | Band |
|---|---:|---|
| VOO | 45% | 33.75–56.25% |
| VGT | 15% | 11.25–18.75% |
| SMH | 10% | 7.5–12.5% |
| VB | 5% | 3–7% |
| VXUS | 15% | 11.25–18.75% |
| GLDM | 5% | 3–7% |
| VGSH | 5% | 3–7% |

Blended fee **0.078%/yr**. US share of equities **~83%**. 10y rates are the same building blocks, re-weighted: stress ~0.1%, bear ~3.1%, base **5.7%**, bull ~8.2%. 5y uses the same width overlay as default (front-loaded CAPE / AI). Slightly worse left tail, slightly better bull. Same base neighborhood.

Weekly per $1,000: VOO $450 / VGT $150 / SMH $100 / VB $50 / VXUS $150 / GLDM $50 / VGSH $50.

---

## 6. What we will not recommend

- 50/30/20 VOO/VGT/SMH as the default
- QQQ + XLK + SOXX stacked on VGT/SMH
- 10% gold after a 64% year (recency)
- 40/60 bonds as the whole book (wrong job for a 10y+ accumulator)
- Covered calls / CSPs / VOOY / SOXL on this sleeve
- Two down capex quarters → sell SMH (that is how you sell the cycle trough)

---

## Why forward returns are not 15–22%

Every serious house's 10-year **nominal** US-equity cluster is about **4–7%**, not the trailing decade.

| House (as-of) | US equities | Intl / ex-US | Notes |
|---|---|---|---|
| **Vanguard VCMM** (30 Jun 2026) | **4.2–6.2%** range (midpoint 5.2%) | Global ex-US **3.9–5.9%**; DM ex-US **4.5–6.5%**; EM **2–4%** | Growth **3.6–5.6%**; value **6.4–8.4%**; short Treasury **3.5–4.5%**. Outlook title: *AI exuberance: economic upside, stock market downside.* |
| **J.P. Morgan LTCMA 2026** (30 Sep 2025) | US large **6.7%** | EAFE higher than US in their tables | Gold **5.5%**. 60/40 **6.4%**. |
| **BlackRock CMA 2026** | **5.2%** | DM intl **6.8%**; EM **5.9%** | Tactically still overweight US/AI; long-run number is muted. |
| **Research Affiliates** (early 2026) | **3.1%** | DM **7.7%**; EM **7.5%** | CAPE mean-reversion. Most cautious US print in the roundup. |
| **Morningstar** (2026 expert survey) | **5.3%** (MAR); cluster **4–7%** | Higher than US at every firm in the roundup | "No one's calling for… anything like the 15% return we've had." |
| **14-house median** (Portfolio Lab, 2026) | **6.0%** | DM **7.2%**; EM **7.7%** | Spread on US: 3.1% to 7.6%. |

Houses would go further toward value and bonds. This sleeve does not, because the job is a ToothFolio-like growth core, not a 40/60.

---

## Calculator

Both sleeves. Enter a dollar amount.

```bash
PYTHONPATH=. python3 -m compoundcore 250000 --weekly 500 --monthly 1000
PYTHONPATH=. python3 -m compoundcore 250000 --sleeve default
PYTHONPATH=. python3 -m compoundcore calculator
```

HTML: [`web/calculator.html`](../web/calculator.html) — type amount, weekly, monthly; default and aggressive update together.

---

## Not financial advice

These projections and capital-market assumptions are **hypothetical**. They are not guarantees, not personalized advice, and not an offer to buy or sell any security. Expense ratios, holdings, CAPE, and CMA tables move. No orders are placed by this package. Past 40% years are a gift that already happened, not a budget.
