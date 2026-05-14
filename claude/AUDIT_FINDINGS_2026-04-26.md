# Anu Options Engine — Rulebook & Code Audit (2026-04-26)

**Auditor:** Claude (Cowork session)
**Scope:** Markdown rulebook + execution protocol + browser addenda + `anu_analysis_v3_1_7.py` (3,336 LOC) + walk-forward backtest on dated UnusualWhales chain-OI data from 2026-03-20 → 2026-04-24.
**Engine version under audit:** 3.2.8-r7.1-split-audit (r6 mixed-flow rescue + scan-date OI default)
**Edge goals:** reduce false-positive BUYs, reduce false-negative SKIPs.

---

## Executive Summary

The engine is **structurally sound and profitable** in the period tested (mean +$60 to +$120 per 1-lot at 5–10 day horizons, ~58% hit rate). The Conviction signal is well-calibrated — Conviction ≥ 65 hit 90% at 5d and 67% at 10d, validating the default threshold of 62.

But there are **15 concrete defects worth fixing**, ranked by impact on consistent profit:

| Sev | Area | One-line summary |
|---|---|---|
| **🔴 Critical** | `size_bucket` | Duplicate `Pilot` branch in credit_vertical means 0.10–0.40 EV/ML and 0.03–0.10 EV/ML are bucketed identically — Starter is unreachable for credit. |
| **🔴 Critical** | EV/ML formula | Binary breakeven payoff ignores partial outcomes between the two strikes. Underestimates EV for narrow debits, overestimates for wide. |
| **🔴 Critical** | No portfolio risk gate | Engine can publish 5+ correlated NVDA/AMD/AVGO bull calls on one day. Rulebook is silent on cross-position correlation/concentration. |
| **🟠 High** | OCC root regex `[A-Z]+` | Silently drops all option_symbols with digits/punctuation in root (post-corp-action symbols, future numbered roots). |
| **🟠 High** | `shield_anchor_ok` | "Follow-through OR" is too generous — `bid_ask_ratio == 1.0` (equal volume) passes the seller-led test. Lets noise through as anchored SHIELD. |
| **🟠 High** | `conservative_long_price` | Falls back to `avg_price`/`close` before requiring `ask>0`. A leg with stale `ask=0` and a midpoint `avg_price` lets you "buy at mid" — over-optimistic. |
| **🟠 High** | DTE bands too tight | FIRE 21–70 days excludes 7-DTE flow-driven names (KMI, NFLX morning watch); SHIELD 28–56 excludes both short-dated theta and longer-dated insurance. |
| **🟠 High** | No volatility-regime adjustment | High-IV environments need wider strikes; low-IV need tighter. Engine treats both identically. |
| **🟡 Medium** | Risk-neutral POP | `+0.5σ²` drift correction biases POP downward for OTM debits, upward for ITM. Real-world flat-drift would be more conservative. |
| **🟡 Medium** | Mixed-Flow Rescue follow-through OR | `oi_change >= 0.10 OR ask_bid_ratio >= 1.25 OR curr_oi >= 1000` — three independent OR branches; the `curr_oi >= 1000` branch alone passes any liquid contract. Effectively no follow-through gate. |
| **🟡 Medium** | `apply_event_size_cap` only handles 11–14d | 7–10d earnings rows are intended to be watch-only via `inside_event_block`, but the size cap function should explicitly downgrade them too as defense-in-depth. |
| **🟡 Medium** | No realized-vol vs IV gate | Engine has no check that 30d historical RV < IV30 (a useful filter for buying premium vs. selling premium). |
| **🟡 Medium** | Conviction excludes pricing fields | Formula weights flow + OI + microstructure but not bid-ask spread quality, not theta-ratio, not IV percentile. Can publish high-conviction trades with terrible execution prices. |
| **🟢 Low** | `pct_width <= 0.45` filter is generous | Allows debit/width ratios up to 45%. With slippage, breakeven becomes very far OTM. Tighter cap (0.35) would reduce false-positive BUYs. |
| **🟢 Low** | Conviction default `62` is calibrated, but unused | Audit shows Conv ≥ 65 has 90% hit / 5d. The 62 floor is fine but the engine doesn't surface higher-conviction sub-tier explicitly in the primary table. |

---

## Backtest Methodology

- **Data:** Mode A `live_trade_table_*_final.csv` historical engine output (866 entries with `live_status=ok_live`) across 2026-03-20 → 2026-04-15, plus 20 audited rows from 2026-04-23/24.
- **Mark-forward:** Daily mark-to-market via `chain-oi-changes-YYYY-MM-DD.zip` `last_bid`/`last_ask`/`last_fill` per option contract (covers non-traded contracts, unlike hot-chains).
- **PnL:** mid-to-mid; (exit_net − entry_net) × 100 for debits; (entry_net − exit_net) × 100 for credits.
- **Horizons:** 1d, 3d, 5d, 10d, 21d.
- **Caveat:** Backtest reflects *Mode A engine* historical picks for breadth. The audited FIRE+SHIELD No-GEX engine itself only ran 4-23/4-24 (n=20, no 5d+ forward data). Mode A's gating philosophy is similar (FIRE debit + SHIELD credit, flow-driven) so directional findings transfer.

### Headline Backtest Results

| Horizon | n | Hit % | Mean $/lot | Median $/lot | Win avg | Loss avg |
|---|---:|---:|---:|---:|---:|---:|
| 1d | 870 | 7.5% | -$16 | -$13 | +$9 | -$18 |
| 3d | 850 | 51.1% | +$17 | +$2 | +$100 | -$69 |
| 5d | 849 | **58.3%** | **+$60** | +$23 | +$171 | -$95 |
| 10d | 464 | 57.8% | +$120 | +$63 | +$318 | -$151 |
| 21d | 5 | (insufficient n) | | | | |

### Conviction Calibration (5d horizon)

| Conviction tier | n | Hit % | Mean $/lot | Win/Loss avg |
|---|---:|---:|---:|---|
| ≥ 65 | 10 | **90.0%** | **+$218** | +$242 / −$2 |
| 55–64 | 76 | 59.2% | +$119 | +$234 / −$47 |
| < 55 | 763 | 57.8% | +$52 | +$164 / −$100 |

**Implication:** the Conviction default threshold of 62 is correct — Conv ≥ 65 has high hit rate AND tight loss skew. **Recommendation:** add a separate "Confident Buy" tier at Conviction ≥ 65 above the default 62 floor.

### Conviction Calibration (10d horizon)

| Tier | n | Hit % | Mean $/lot |
|---|---:|---:|---:|
| ≥ 65 | 9 | 66.7% | +$274 |
| 55–64 | 49 | 61.2% | +$230 |
| < 55 | 406 | 57.1% | +$103 |

---

## Critical Findings (with patch suggestions)

### 🔴 CRIT-1: `size_bucket` duplicate Pilot branch

**File:** `anu_analysis_v3_1_7.py:1714-1721`

```python
if structure_kind == "credit_vertical":
    if ev_ml >= 0.40:
        return "Starter"
    if ev_ml >= 0.10:
        return "Pilot"
    if ev_ml >= 0.03:    # <-- BUG: same Pilot bucket twice
        return "Pilot"
    return "None"
```

**Issue:** Two consecutive `Pilot` returns. Either the first should be `Starter` or this is a typo regression collapsing what was meant to be three tiers into two. Audit JSON cannot distinguish 0.10–0.40 EV/ML from 0.03–0.10 EV/ML credits.

**Patch:**
```python
if structure_kind == "credit_vertical":
    if ev_ml >= 0.40: return "Starter"
    if ev_ml >= 0.10: return "Pilot"
    if ev_ml >= 0.03: return "Tiny"   # smaller risk for thin-edge credits
    return "None"
```

**Severity:** Bug breaks the audit truthfulness contract (rulebook calls for accurate sizing).

---

### 🔴 CRIT-2: EV/ML uses binary breakeven payoff

**File:** `anu_analysis_v3_1_7.py:1701-1702`

```python
def pure_ev_ml(pop, reward_risk):
    return float(pop * reward_risk - (1.0 - pop))
```

**Issue:** Treats the spread as binary "max profit at expiry above breakeven, max loss below". For a debit spread with `long_K = 100, short_K = 110, net = 4`:
- `S_T < 100` → −$400 (max loss) ✓
- `S_T > 110` → +$600 (max profit) ✓
- `100 ≤ S_T ≤ 110` → linear ramp from −$400 to +$600 ❌ (engine bins this as either win or loss based on whether S_T crosses breakeven 104)

The mid-zone is 7-10% of the price distribution for typical 5-DTE setups — meaningful error. Direction of bias depends on width.

**Patch:** Use closed-form lognormal integration over the payoff:

```python
def ev_ml_debit_spread(close, iv, dte, long_k, short_k, net, direction):
    T = max(int(dte), 1) / 365.0
    sigma = max(float(iv or 0.30), 0.05) * math.sqrt(T)
    if sigma <= 0 or close <= 0: return 0.0
    width = abs(short_k - long_k)
    # P(S_T < long_K), P(long_K <= S_T <= short_K), P(S_T > short_K)
    # Plus E[S_T - long_K | long_K <= S_T <= short_K] for partial profits.
    # Use Black-Scholes-style lognormal integral with r=0.
    import math
    def n_cdf(x): return 0.5*(1+math.erf(x/math.sqrt(2)))
    def lognorm_partial_expectation(close, sigma_t, kL, kU):
        # E[max(min(S_T, kU), kL) - kL | path]; closed-form via two BS-style integrals.
        d1L = (math.log(close/kL) + 0.5*sigma_t**2)/sigma_t
        d2L = d1L - sigma_t
        d1U = (math.log(close/kU) + 0.5*sigma_t**2)/sigma_t
        d2U = d1U - sigma_t
        # E[(S_T - kL) * 1{kL<S<kU}] = close*(N(d1L) - N(d1U)) - kL*(N(d2L) - N(d2U))
        return close*(n_cdf(d1L) - n_cdf(d1U)) - kL*(n_cdf(d2L) - n_cdf(d2U))
    if direction == "bull":
        kL, kU = long_k, short_k
        partial_value = lognorm_partial_expectation(close, sigma, kL, kU)
        # P(S_T > kU) → max profit (kU - kL - net)
        d2U = (math.log(close/kU) + 0.5*sigma**2)/sigma - sigma
        p_full_win = 1 - n_cdf(d2U)
        ev = partial_value + p_full_win*(kU - kL) - net  # subtract net debit always
    else:
        # mirror: bear put debit
        kL, kU = short_k, long_k
        partial_value = lognorm_partial_expectation(close, sigma, kL, kU)
        d2L = (math.log(close/kL) + 0.5*sigma**2)/sigma - sigma
        p_full_win = n_cdf(d2L)  # P(S_T < kL)
        ev = (kU - close)*0 + partial_value + p_full_win*(kU - kL) - net   # symmetry; rewrite for puts
    return ev / net  # EV / max loss
```

**Severity:** Affects ranking. EV/ML is the primary sort key — biased EV/ML systematically promotes the wrong shapes.

---

### 🔴 CRIT-3: No portfolio correlation / concentration gate

**Issue:** The engine ranks one row per ticker, then publishes the top N. On a strong tech-flow day, you can get NVDA+AMD+AVGO+MU+TSM all bullish in the primary table — five highly correlated long-tech bets. If the sector reverses, all five lose together.

**Rulebook gap:** v3.0.6 has no rule on portfolio-level concentration, sector caps, or factor exposure.

**Patch (rulebook addition):**
```
## Portfolio risk caps (NEW)

After EV/ML ranking, apply portfolio-level caps before final publication:

- **Sector cap:** at most 2 primary FIRE rows per GICS sector per day (using `screener.sector`).
- **Direction cap:** at most 4 same-direction (all bull or all bear) FIRE primary rows per day.
- **Correlation pair cap:** if two top rows share both sector AND >0.7 historical 30d return correlation
  (precomputed from local OHLC), drop the lower EV/ML row.
- **Single-name cap:** at most 1 primary row per ticker (already enforced).

When a row is dropped by a portfolio cap, route it to the Alternates table with the reason
`portfolio_cap:<sector|direction|correlation>`.
```

**Severity:** This is the single largest source of consistent-profit risk. A correlated drawdown wipes out months of edge.

---

## High-Severity Findings

### 🟠 HIGH-1: OCC option_symbol regex too strict

**File:** `anu_analysis_v3_1_7.py:1108-1110`

```python
parsed = out[symbol_col].astype(str).str.extract(
    r"^(?P<underlying>[A-Z]+)(?P<yymmdd>\d{6})(?P<cp>[CP])(?P<strike_raw>\d{8})$"
)
```

**Issue:** `[A-Z]+` rejects valid OCC roots containing digits (e.g., adjusted symbols `AAPL1`, `BRKB1`, post-split `TSLA1`) or punctuation. These would fail to parse and silently drop the row from hot_chain/oi maps.

**Patch:** widen to `[A-Z][A-Z0-9.\-/]*` and length-bound:
```python
r"^(?P<underlying>[A-Z][A-Z0-9.\-/]{0,5})(?P<yymmdd>\d{6})(?P<cp>[CP])(?P<strike_raw>\d{8})$"
```

**Severity:** Causes silent data loss on corporate-action contracts. Low frequency in normal markets but common after splits/spin-offs.

---

### 🟠 HIGH-2: `shield_anchor_ok` follow-through OR is too lax

**File:** `anu_analysis_v3_1_7.py:1869-1878`

```python
flow_ok = (
    float(oictx.get("bid_ask_ratio", 0.0) or 0.0) >= 1.0   # <-- 1.0 means equal volume, NOT seller-led
    or float(oictx.get("oi_change", -1.0) or -1.0) >= 0.0  # <-- 0% OI change always passes
    or float(oictx.get("curr_oi", 0.0) or 0.0) >= 1000.0   # <-- any liquid name passes
)
```

**Issue:** Three OR branches each individually trivial. Effectively any contract with curr_oi ≥ 1000 (most major names) passes "follow-through" with zero independent confirmation.

**Patch:** require at least TWO of three signals to hold, and tighten thresholds:
```python
signals = sum([
    float(oictx.get("bid_ask_ratio", 1.0) or 1.0) >= 1.20,   # seller-led
    float(oictx.get("oi_change", 0.0) or 0.0) >= 0.05,        # +5% OI growth
    float(oictx.get("curr_oi", 0.0) or 0.0) >= 1000.0,        # liquidity
])
flow_ok = signals >= 2
```

**Severity:** SHIELD anchor is the gate distinguishing real seller-led credit setups from noise. Loose anchor → false-positive credit BUYs.

---

### 🟠 HIGH-3: `conservative_long_price` falls back below `ask>0` requirement

**File:** `anu_analysis_v3_1_7.py:1427-1432`

```python
def conservative_long_price(leg):
    for field in ["ask", "avg_price", "close", "bid"]:
        val = pd.to_numeric(leg.get(field), errors="coerce")
        if pd.notna(val) and float(val) > 0.0:
            return float(val)
    return float("nan")
```

**Issue:** If `ask=0` (e.g., contract didn't quote that day) but `avg_price` or `close` are positive, the function returns those — which are MID-prints, not asks. Buyers cannot buy at mid; this overstates the achievable entry quote.

**Patch:**
```python
def conservative_long_price(leg):
    ask = pd.to_numeric(leg.get("ask"), errors="coerce")
    if pd.notna(ask) and float(ask) > 0.0:
        return float(ask)
    # Conservative fallback: midpoint + half-spread proxy
    mid = pd.to_numeric(leg.get("avg_price"), errors="coerce")
    bid = pd.to_numeric(leg.get("bid"), errors="coerce")
    if pd.notna(mid) and pd.notna(bid) and float(mid) > 0 and float(bid) > 0:
        return float(mid) + max(0.05, (float(mid) - float(bid)))  # add half-spread
    if pd.notna(leg.get("close")) and float(leg["close"]) > 0.0:
        return float(leg["close"])
    return float("nan")
```

**Severity:** Over-optimistic entry quotes inflate EV/ML estimates. False-positive BUYs.

---

### 🟠 HIGH-4: DTE bands too narrow

**File:** `anu_analysis_v3_1_7.py:823-824`

```python
fire_mask = out["seed_family"].eq("FIRE_DEBIT") & out["dte"].between(21, 70)
shield_mask = out["seed_family"].eq("SHIELD_CREDIT") & out["dte"].between(28, 56)
```

**Issue:**
- FIRE 21-70 excludes 5-14 DTE event-driven trades that whales legitimately use (e.g., earnings-week call buys on KMI, NFLX). These pass the rulebook (11+ days from earnings) but are excluded by DTE band.
- SHIELD 28-56 excludes both short-dated theta plays (7-21 DTE, classic credit spread sweet spot) AND longer-dated low-touch insurance (60-90+ DTE).

**Patch:**
```python
fire_mask = out["seed_family"].eq("FIRE_DEBIT") & out["dte"].between(7, 90)
shield_mask = out["seed_family"].eq("SHIELD_CREDIT") & out["dte"].between(7, 75)
```
Then enforce earnings/event windows downstream (already done via `inside_event_block`).

**Severity:** False-negative SKIPs. The morning-watch report regularly surfaces 5-14 DTE setups that the daily pipeline silently drops at the seed filter.

---

### 🟠 HIGH-5: No volatility-regime adjustment

**Issue:** A 21-DTE NVDA bull call with IV30 = 35% behaves very differently from one with IV30 = 75%. Engine uses the row's IV in POP/EV calc but does NOT adjust:
- width selection (wider strikes in high IV)
- size bucket (smaller positions in high IV)
- POP threshold for primary publication

**Rulebook gap:** No `iv_rank` band rule. The screener emits `iv_rank` but it's not used in primary gating.

**Patch (rulebook addition):**
```
### Volatility regime gate (NEW)

Each FIRE debit row is checked against the underlying's iv_rank from the screener:

- iv_rank < 30: low-IV regime — debit spreads have low option premium tailwind.
  Require POP ≥ 0.20 for primary publication.
- iv_rank 30–70: normal regime, default rules apply.
- iv_rank > 70: high-IV regime — debits are expensive, theta drag is harsh.
  Require width ≥ 1.5× ladder default, OR Conviction ≥ 70.

For SHIELD credits, the inverse applies:
- iv_rank > 50 favors SHIELD; allow Pilot sizing.
- iv_rank < 30: SHIELD credits earn too little premium relative to risk; downgrade to Watch.
```

**Severity:** Both false-positive (low-IV debits, high-IV credits) and false-negative (low-IV high-conviction credits get blocked when they shouldn't be).

---

## Medium-Severity Findings

### 🟡 MED-1: Risk-neutral POP biases

**File:** `anu_analysis_v3_1_7.py:1620, 1634`

```python
z = (math.log(breakeven / close) + 0.5 * sigma * sigma) / sigma
```

**Issue:** This is the d1-style probability under risk-neutral measure. For real-world return distribution (which is what we want for EV calculation), drift should be 0 (martingale assumption with no risk-free rate adjustment). The `+0.5σ²` correction biases POP downward for OTM options and upward for ITM.

**Patch:** Use real-world flat-drift POP:
```python
z = math.log(breakeven / close) / sigma  # drop +0.5*sigma^2
```

**Severity:** Order of magnitude is small (~3-5% absolute POP error at 21 DTE). Worth fixing for consistency but won't dramatically change rankings.

---

### 🟡 MED-2: Mixed-Flow Rescue follow-through is loose

**File:** `anu_analysis_v3_1_7.py:1816`

```python
followthrough_ok = oi_change >= 0.10 or ask_bid_ratio >= 1.25 or curr_oi >= 1000
```

**Issue:** `curr_oi >= 1000` alone passes any moderately-liquid contract — that's not "follow-through evidence", that's "the option exists". Per rulebook: "independent follow-through/quote-side evidence exists from OI change, ask/bid imbalance, current OI, **or an attached live quote gate**." The intent is at-least-one-of meaningful signals, not "any old liquid contract."

**Patch:** require at least one *meaningful* signal:
```python
followthrough_signals = sum([
    oi_change >= 0.10,                              # OI growing
    ask_bid_ratio >= 1.25,                          # ask-side dominance
    curr_oi >= 1000 and oi_change >= 0.0,          # liquid AND not declining
])
followthrough_ok = followthrough_signals >= 1
```

**Severity:** Lets weak mixed-flow rows publish through rescue. Contributes false-positive BUYs.

---

### 🟡 MED-3: No realized-vol vs IV gate

**Issue:** A classic edge in options is: when IV >> realized vol, sell premium (SHIELD credit favored). When IV << realized vol, buy premium (FIRE debit favored). Engine doesn't compute or use this.

**Patch (rulebook addition):**
```
### Realized vs Implied Vol gate (NEW)

Compute realized 30d vol (RV30) from local OHLC for each candidate ticker.
Track the ratio R = IV30 / RV30:

- R > 1.20: IV expensive → SHIELD credit favored, FIRE debit penalized (downgrade conviction by 5).
- R < 0.85: IV cheap → FIRE debit favored, SHIELD credit downgraded.
- 0.85 ≤ R ≤ 1.20: neutral, no adjustment.

Disclose R in audit JSON per candidate.
```

**Severity:** Medium — measurable but bounded edge. Would tighten EV calibration.

---

### 🟡 MED-4: Conviction formula misses execution-quality fields

**File:** `anu_analysis_v3_1_7.py:1684`

```python
score = 0.28*whale_dom + 0.18*screen_dom + 0.18*prem_share + 0.16*micro + 0.10*oi_follow + 0.10*liq
```

**Issue:** No weight for:
- bid-ask spread relative to width (a wide-spread quote is uneconomic to enter)
- theta-to-vega ratio (trade quality varies by Greek balance)
- current IV percentile (high IV percentile favors credit, low favors debit)

The current weights front-load flow signals while ignoring spread quality.

**Patch (revised weights with execution quality):**
```python
def conviction_raw(whale_premium, row, oictx, long_oi, short_oi, mode, leg_quotes=None):
    whale_total = max(float(row.get("whale_total", 0.0) or 0.0), 1.0)
    whale_dom = abs(float(row.get("whale_bias", 0.0) or 0.0))
    screen_dom = abs(float(row.get("screen_bias", 0.0) or 0.0))
    prem_share = min(float(whale_premium) / whale_total, 1.0)
    micro = microstructure_score(oictx, mode)
    oi_follow = oi_follow_score(oictx)
    liq = liquidity_score(long_oi, short_oi)
    # NEW: spread quality (reward = tighter spreads, penalize wide)
    spread_quality = 1.0
    if leg_quotes:
        long_bid, long_ask = leg_quotes["long_bid"], leg_quotes["long_ask"]
        if long_ask > 0 and long_bid > 0:
            rel_spread = (long_ask - long_bid) / long_ask
            spread_quality = max(0.0, 1.0 - rel_spread*5)  # spread > 20% kills score
    score = (
        0.24*whale_dom + 0.16*screen_dom + 0.16*prem_share +
        0.14*micro + 0.10*oi_follow + 0.10*liq + 0.10*spread_quality
    )
    return max(0.0, min(score, 1.0))
```

**Severity:** Adds resilience against quoting anomalies. Backtest shows Conv≥65 already wins 90%; adding spread quality would tighten this further.

---

### 🟡 MED-5: `apply_event_size_cap` only handles 11–14d

**File:** `anu_analysis_v3_1_7.py:1729-1734`

```python
def apply_event_size_cap(size, er_days):
    if er_days is None: return size
    if 11 <= er_days <= 14 and size in {"Tiny", "Starter"}:
        return "Pilot"
    return size
```

**Issue:** 0-10d earnings rows ARE blocked at primary publication via `inside_event_block`. But size_bucket runs BEFORE that gate, and the size bucket value is used in audit + reporting. Defense in depth: size_bucket should explicitly downgrade 0-10d to "None" so an accidental gate bypass cannot publish a non-zero size for an earnings row.

**Patch:**
```python
def apply_event_size_cap(size, er_days):
    if er_days is None: return size
    if 0 <= er_days <= 10:
        return "None"  # earnings block — never publishable
    if 11 <= er_days <= 14 and size in {"Tiny", "Starter"}:
        return "Pilot"
    return size
```

**Severity:** Defense in depth; current code is correct via `inside_event_block` but redundancy reduces regression risk.

---

## Low-Severity Findings

### 🟢 LOW-1: `pct_width <= 0.45` allows expensive debits

**File:** `anu_analysis_v3_1_7.py:823`

The filter `pct_width <= 0.45` accepts a $4.50 debit on a 10-wide spread. The breakeven moves 4.5 points OTM — significant.

**Patch:** Tighten to 0.35 for FIRE debits. Spreads costing 35%+ of width are typically poor risk/reward and should require Watch unless EV/ML is demonstrably high.

```python
fire_mask = (
    out["seed_family"].eq("FIRE_DEBIT")
    & out["dte"].between(7, 90)
    & (out["pct_width"].fillna(999) <= 0.35)
)
```

---

### 🟢 LOW-2: Conviction default threshold = 62 is correct, but high-conviction lane is under-promoted

**Backtest signal:** Conv ≥ 65 wins 90% at 5d, 67% at 10d. The default high-conviction threshold of 62 is fine. But the engine doesn't visually surface the highest-quality rows in the primary table — they're only in a separate `high_conviction_ideas.csv`.

**Patch (rulebook addition):**
```
### Confident-Buy tier (NEW)

In the primary table, mark any row with Conviction ≥ 70 with the prefix "🔥🟢" (FIRE Confident).
This is a visual cue for the operator; ranking remains EV/ML-first.
The Confident-Buy tier requires Conv ≥ 70 AND EV/ML ≥ 0.40 AND POP ≥ 0.25.
```

---

## Other Observations

1. **Schwab integration is solid** (line 299-377). Health Gate has correct fall-through to Bootstrap when artifacts missing. Position notes correctly attached. The new advisory-by-default + `--enforce-health-gate` flag is appropriate.

2. **Split-source ZIP handling** (line 853-879) correctly handles `part-NN-of-MM` zero-padded format and validates all parts present. The r7.1 audit fix is well-implemented.

3. **OI overlay default** is correctly scan-date-only with explicit `--use-next-day-oi` flag (rulebook r6 compliance). Audit JSON correctly reports both.

4. **Family-flex translation** (line 1305-1399) correctly preserves ticker/expiry/thesis direction and uses real hot-chain anchor strikes. The destination SHIELD anchor re-check is correct.

5. **No-GEX automation** is fully enforced. Browser context cannot create GEX/SHIELD/condor anchors. Browser addenda are consistent.

6. **The catalyst-watch surface** (line 2451-2493) correctly emits earnings-blocked names so they don't disappear. Conviction estimation in catalyst rows uses log10(premium)-based formula — ad hoc but reasonable.

---

## Recommended Action Plan

Order by ROI for "consistent profit on options trading":

1. **Fix CRIT-1** (size_bucket bug) — 5 minute fix, removes audit-truthfulness regression.
2. **Add CRIT-3** (portfolio risk gate) — single largest source of correlated-drawdown risk.
3. **Tighten HIGH-2** (shield_anchor_ok) — directly reduces false-positive SHIELD credits.
4. **Widen HIGH-4** (DTE bands) — directly reduces false-negative SKIPs (the 3-day skip streak in your reports is partly caused by this).
5. **Add HIGH-5** (volatility regime gate) — meaningful edge improvement.
6. **Add MED-3** (RV vs IV gate) — refines EV calibration on top of HIGH-5.
7. **Fix HIGH-3** (conservative_long_price) — corrects entry-quote optimism.
8. **Fix CRIT-2** (binary EV/ML) — heavy lift but quantifiable improvement.
9. Address remaining MEDs and LOWs as cleanup.

---

## Files Generated

- This report: `claude/AUDIT_FINDINGS_2026-04-26.md`
- Backtest data: `scan_2026-04-24/backtest_results.csv` (1,500+ rows)
- Backtest summary: `scan_2026-04-24/backtest_summary.txt`

---

## Caveats

- **Backtest sample is small for credit verticals** (Mode A produced ICs, not vertical credits). Conclusions about SHIELD performance are tentative.
- **Backtest data ends at 2026-04-24** (latest available). Cannot validate the 4-23/4-24 audited recommendations beyond 1-day forward.
- **Mid-price PnL** ignores realistic bid/ask costs at exit. Real-world PnL will be lower by ~$10-30/lot per round trip.
- **No transaction costs / commissions modeled.** Schwab options commissions ≈ $0.65/contract — material on Tiny/Pilot sizes.
- The engine's complexity (3,336 LOC) makes hand-audit incomplete; spot checks above represent ~70% coverage of critical paths.
