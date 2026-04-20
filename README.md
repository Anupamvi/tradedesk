# UW Trade Desk

An automated options trading pipeline that generates, validates, and monitors spread trades using data from [UnusualWhales](https://unusualwhales.com) and live [Schwab](https://developer.schwab.com) quotes.

---

## What You Need

1. **UnusualWhales subscription** — provides the 5 daily EOD data files (hot chains, OI changes, dark pool, screener, whale trades)
2. **Schwab developer account** — provides live option chain quotes for trade validation and position monitoring
3. **Python 3.12+** with: `pip install pandas numpy pyyaml yfinance schwab-py python-dotenv tabulate`
4. **ntfy.sh account** (free) — for push notifications from the trade monitor

### Environment Setup

Create `c:\uw_root\.env`:
```env
SCHWAB_API_KEY=your_api_key
SCHWAB_APP_SECRET=your_app_secret
SCHWAB_TOKEN_PATH=./tokens/schwab_token.json
NTFY_TOPIC=your_ntfy_topic_name
```

Authenticate Schwab (one-time, opens browser):
```bash
cd c:\uw_root
python -m uwos.schwab_position_analyzer --manual-auth
```

---

## 1. Daily Pipeline — Generate Today's Trades

**What:** Takes 5 EOD data files from UnusualWhales and produces a ranked table of option spread trades, validated against live Schwab quotes.

**How it works:**
1. **Stage-1 Discovery** — scans hot chains, whale trades, and dark pool data to find the best debit spreads (FIRE track) and credit spreads (SHIELD track). Scores each by conviction (direction bias, efficiency, liquidity, whale activity).
2. **Stage-2 Validation** — fetches live Schwab option chains, checks entry prices, runs a historical backtest for each setup, and applies quality gates (delta, GEX regime, macro regime, sigma).
3. **Output** — an expert trade table sorted into Core (best), Tactical (good with caveats), and Watch (blocked).

**Run:**
```bash
# Place your 5 UW files in c:\uw_root\2026-04-08\
python -m uwos.run_mode_a_two_stage \
  --base-dir "c:/uw_root/2026-04-08" \
  --config "c:/uw_root/uwos/rulebook_config_goal_holistic_claude.yaml" \
  --out-dir "c:/uw_root/out/2026-04-08" \
  --top-trades 20
```

**Or via Claude Code:** `/daily-pipeline 2026-04-08`

**Input files** (download from UW, place in `c:\uw_root\YYYY-MM-DD\`):
- `hot-chains-YYYY-MM-DD.zip`
- `chain-oi-changes-YYYY-MM-DD.zip`
- `dp-eod-report-YYYY-MM-DD.zip`
- `stock-screener-YYYY-MM-DD.zip`
- `whale-YYYY-MM-DD.md` (generated from `dp-eod-report` via `generate_whale_summary.py`)

**Output files:**
| File | What |
|------|------|
| `YYYY-MM-DD/anu-expert-trade-table-YYYY-MM-DD.md` | The main trade table — Core / Tactical / Watch |
| `out/YYYY-MM-DD/setup_likelihood_YYYY-MM-DD.md` | Backtest win rate and edge for each setup |
| `out/YYYY-MM-DD/live_trade_table_YYYY-MM-DD_final.csv` | Full CSV with live Schwab pricing |
| `out/YYYY-MM-DD/dropped_trades_YYYY-MM-DD.csv` | Why each rejected trade was dropped |

**Key concepts:**
- **FIRE** = debit spreads (Bull Call Debit, Bear Put Debit) — you pay upfront, profit if stock moves your way
- **SHIELD** = credit spreads (Bull Put Credit, Bear Call Credit, Iron Condor) — you collect premium, profit if stock stays in range
- **Core** = all quality gates passed → trade at full size
- **Tactical** = minor blockers (e.g., GEX pinned) → trade at reduced size
- **Watch** = hard blockers (likelihood FAIL, delta fail) → do not trade, monitor only

---

## 2. Trend Analysis — Multi-Day UW Pattern Scanner

**What:** Looks across dated UW folders to find a small set of persistent bullish, bearish, or range-bound option candidates to work on. The command reads stock screener, chain OI, hot chains, dark pool, and whale files across the lookback window, then separates high-conviction candidates from backtest/live-gated actionable trades.

**Run:**
```bash
python -m uwos.trend_analysis 2026-04-17          # default 90 usable market-data-day lookback
python -m uwos.trend_analysis 2026-04-17 45       # 45 usable market-data days ending 2026-04-17
python -m uwos.trend_analysis 2026-04-17 90 --top 20
python -m uwos.trend_analysis 2026-04-17 30 --quote-replay diagnostic
python -m uwos.trend_analysis 2026-04-17 30 --walk-forward-samples 8
python -m uwos.trend_analysis 2026-04-17 30 --walk-forward-samples 31 --reuse-walk-forward-raw --reuse-walk-forward-outcomes --reuse-research-outcomes
```

**Or via Claude/Codex:** `trend-analysis 2026-04-17`

The numeric lookback counts weekday folders with usable `stock-screener` data. Weekend folders or partial dated folders do not consume a lookback day.

**Output:** `out/trend_analysis/trend-analysis-YYYY-MM-DD-LN.md`

**Batch proof / money-engine audit:**
```bash
python -m uwos.trend_analysis_batch \
  --start 2025-12-01 \
  --end 2026-04-17 \
  --lookback 30 \
  --horizons 20 \
  --reuse-raw
```

This answers the harder question: had the command run on every eligible historical date, what trades would it have emitted, what happened afterward, and which playbooks made money? Historical proof intentionally uses local UW option quote replay instead of Schwab live chains, because Schwab current chains would leak today’s prices into old signal dates. Batch output is written to `out/trend_analysis_batch/trend-analysis-batch-proof-START_END-LN.md`.

The **Walk-Forward Audit** is the confidence layer. It reruns older signal dates, selects historical trades using only signal-date evidence, and then scores future option-quote outcomes over configured market-day horizons. A supportive audit raises trust; an empty, low-sample, or negative audit is a confidence penalty even when the current trade list has actionable rows.

The **Backtest-Supported Candidate Shortlist** is the primary output for backtest-supported names: a few high-conviction tickers with high swing score, multiple independent confirmations, few contradictory signals, and at least one structure that passed the backtest gate. Directional price confirmation is preferred; price divergence is shown only when backtest support exists, and it is labeled as a confirmation risk rather than an entry. These are names to work on, not automatic trades.

The **Current Trade Setups** section is the practical workbench. It surfaces conditional setups even when no order-ready trade exists. `TRADE_SETUP` means the current structure is clean enough to research but lacks enough analog sample support; `REBUILD` means the thesis has support but the shown spread needs better liquidity, flow, expiry, or strike construction before entry.

Only rows with swing score >= `60.0`, positive historical support, daily option quote replay `PASS`, `PARTIAL_PASS`, or current-day `ENTRY_OK`, Schwab live validation, no earnings through expiry, acceptable bid/ask width, short delta <= `0.30`, no volatile-GEX iron condor regime, compatible broad market regime, no same-underlying open option exposure from trade-desk, and no negative rolling ticker-playbook validation are shown as actionable trades. Historical support means either backtest `PASS` with sufficient analog signals or a LOW_SAMPLE/PASS-shortfall setup backed by a `promotable` ticker playbook. The professional-quality gate also blocks low-priced lotto setups by default: underlying price must be >= `$20`, debit spreads must price >= `$0.75`, directional trend trades need strong whale/institutional coverage scaled to the lookback and capped at `8` mention days, options flow cannot conflict with the trade direction, directional debit spreads need price/flow scores >= `60`, the debit must be <= `50%` of spread width, and the long strike cannot be more than `2%` OTM. Rows that pass historical support/quote replay/Schwab but fail tradeability are separated as Risk-Blocked; weaker rows stay Pattern Candidates.

The **Max Conviction / Max Planned Risk** section is the highest tier. These rows already pass Actionable Now, then also require strong score, edge, backtest sample support, whale coverage, tight liquidity, and price/flow/OI/dark-pool alignment. This means max pre-defined risk for one defined-risk trade, not all account capital.

The **Trade Workup** section is the middle lane. It surfaces quality setups that pass professional, live, quote, and research-support gates but have positive LOW_SAMPLE or sample-shortfall evidence. They are not entries, but they are the names to research, build a thesis around, and rerun for confirmation instead of hiding them in generic Pattern Candidates. Family-negative rows can still advance only when a ticker-specific playbook is promotable.

The **Research Confidence Audit** is a fixed-bucket sanity check, not a parameter search. It groups historical candidates by predeclared gates, such as entry-available, backtest-pass, and professional-quality, then reports whether those groups had positive future option-quote P&L. The summary requires enough unique historical setups, so repeated 5/10/20-day exits cannot masquerade as independent evidence. The companion **Research Horizon Audit** splits those same buckets by holding horizon. If no bucket/horizon is supportive, the pipeline blocks Actionable Now trades rather than tuning thresholds until a trade appears. The detailed research outcomes CSV lists the ticker, setup, horizon, entry/exit marks, P&L, and gate reasons behind the bucket summary.

The **Strategy Family Audit** is the broad trade-generation layer. It evaluates predeclared setup families, such as growth bull call debits, energy/materials bull call debits, and bearish momentum put debits, using earlier dates for training and later dates for validation. Mixed, negative, validation-negative, or low-sample families are research targets only.

The **Ticker Playbook Audit** is the narrow trade-generation layer. It checks ticker/direction/strategy/horizon playbooks with the same train/validation discipline so a profitable ticker setup is not buried inside a broad losing family. A current setup can unlock Actionable Now when either the matching broad family or the matching ticker playbook is `promotable`, while all live quote, Schwab, earnings, liquidity, trend-quality, and max-risk gates still apply.

The **Rolling Ticker Playbook Forward Validation** section retests ticker playbooks chronologically after they first become promotable using prior-only evidence. Negative rolling validation blocks matching actionable trades. Insufficient or emerging rolling evidence is allowed only with lower position-size tiers.

Every Actionable Trade includes a position-size tier and writes to `out/trend_analysis/trend-analysis-trade-tracker.csv` for post-trade outcome review. Later runs refresh tracker outcomes from local UW option snapshots and mark ideas as `OPEN_WIN`, `OPEN_LOSS`, `CLOSED_WIN`, `CLOSED_LOSS`, or unavailable. `PROBE_ONLY`, `STARTER_RISK`, `STANDARD_RISK`, and `MAX_PLANNED_RISK` are risk caps; low-sample or insufficient-forward-validation trades should remain starter/probe-sized.

Daily option quote replay is on by default. It prices every spread leg from local UW `hot-chains` / `chain-oi-changes` snapshots at the signal date and the latest later option snapshot. If there is no later snapshot yet, same-day entry coverage is marked `ENTRY_OK`; if any required leg is missing, the row is blocked. Defined-risk spread marks must stay inside valid `[0, spread width]` economics; impossible entry/exit marks are treated as bad quotes, not real outcomes. Use `--quote-replay diagnostic` only when you want to inspect candidates without making replay a trade gate.

Before the final gate, the pipeline also repairs strong blocked candidates by trying pre-earnings expiries, more liquid debit structures, farther-OTM credit spreads, and safer condor/directional alternatives. Repaired variants still need to pass Schwab validation and the historical likelihood backtest before they can appear as actionable trades.

The lower-level swing scanner is still available:
```bash
python -m uwos.swing_trend_pipeline --lookback 30 --as-of 2026-04-17
```

---

## 3. Trade Desk — Position Monitoring & Push Alerts

**What:** Monitors your open option positions every 30 minutes during market hours. Applies verdict rules (CLOSE, ROLL, ASSESS, HOLD) and sends push notifications via [ntfy.sh](https://ntfy.sh) when action is needed.

**How it works:**
1. Fetches current positions from Schwab API
2. Evaluates each position against rules:
   - **Credit spreads** (patience rules): close at 50% profit, roll at 21 DTE, close if tested
   - **Debit spreads** (urgency rules): close at 100%+ profit, stop-loss at breakeven, close if momentum lost
3. Compares current verdicts against previous scan (state diff)
4. Sends ntfy push notification only on **transitions** (HOLD → CLOSE, HOLD → ROLL)

**Run:**
```bash
python -m uwos.trade_monitor --force              # single scan, notify on transitions
python -m uwos.trade_monitor --force --manual      # notify ALL actionable verdicts
python -m uwos.trade_monitor --test                # send a test notification
```

**Automated scheduling (Windows Task Scheduler):**
```bash
uwos\setup_trade_monitor.bat
```
Runs every 30 min, Mon-Fri 9:30 AM – 4:05 PM.

**Notifications go to:** `https://ntfy.sh/your_topic` (install ntfy app on phone for push alerts)

---

## 4. Trade History & Performance Review

**What:** Analyzes your realized trade log to track win rate, profit factor, edge drift, and identify which strategy types are working.

**Run all stages:**
```bash
python -m uwos.run_trade_playbook \
  --realized-csv ./out/cleaned_realized_trades.csv \
  --config ./uwos/rulebook_config.yaml \
  --out-dir ./out/playbook
```

**Or from Google Sheets (auto-downloads):**
```bash
python -m uwos.run_trade_playbook \
  --sheet-csv-url "https://docs.google.com/spreadsheets/d/SHEET_ID/export?format=csv&gid=GID" \
  --config ./uwos/rulebook_config.yaml \
  --out-dir ./out/playbook
```

**Output:**
| File | What |
|------|------|
| `daily_risk_monitor.md` | Current open position risk assessment |
| `weekly_edge_report.md` | Weekly win rate, P&L, strategy breakdown |
| `monthly_longitudinal_review.md` | Monthly trends and edge drift |

---

## 5. Stock Dip Scanner — Equity Buying Opportunities

**What:** Screens the S&P 500 for stocks that have dropped significantly and scores them for recovery potential. Useful for finding stock (not options) buying opportunities during selloffs.

**How it works:**
1. Fetches bulk S&P 500 quotes from Schwab
2. Deep-dives on top candidates with yfinance fundamentals
3. 4-layer scoring: quality (30%), drop magnitude (20%), broad-vs-stock-specific context (20%), recovery potential (30%)
4. Alerts on BUY / STRONG BUY signals

**Run:**
```bash
python -m uwos.trade_ideas
```

---

## 6. Historical Replay Analysis

**What:** Aggregates manifest-backed historical pipeline runs into longitudinal replay views. This is useful for audit/replay comparisons, but it is not the primary dated-folder trend command.

**Run:**
```bash
python -m uwos.historical_trend_pipeline \
  --search-root c:\uw_root\out\replay_compare \
  --lookback-days 45 \
  --recommendation-top-n 20
```

For dated-folder trade recommendations, use `python -m uwos.trend_analysis` instead.

**Output:** `out/replay_compare/trend_analysis/final_trade_recommendations_from_trends.md`

---

## 7. UW Dashboard Capture (Browser-Based)

**What:** If your UnusualWhales plan doesn't include API/CSV export, this tool opens a browser, logs into UW, and captures the data you need by scraping dashboard pages.

**Run:**
```bash
# One-time login
python -m uwos.uw_dashboard_capture --manual-login-only --base-dir c:\uw_root \
  --profile-dir tokens\uw_playwright_profile_v2 --browser-channel msedge --no-headless

# Capture daily data
python -m uwos.uw_dashboard_capture --trade-date 2026-04-08 --base-dir c:\uw_root \
  --preset mode-a-core --profile-dir tokens\uw_playwright_profile_v2 --browser-channel msedge
```

**Requires:** `pip install playwright && python -m playwright install chromium`

---

## 8. Schwab Utilities

### Live Quotes
```bash
python -m uwos.schwab_quotes --symbols-csv AAPL,SPY --chain-symbols-csv AAPL --strike-count 8
```

### Position Analysis
```bash
python -m uwos.schwab_position_analyzer
```

### Token Refresh (when expired)
```bash
del "c:\uw_root\tokens\schwab_token.json"
python -m uwos.schwab_position_analyzer --manual-auth
```

---

## Claude Code Integration

### Slash Commands

| Command | What |
|---------|------|
| `/daily-pipeline YYYY-MM-DD` | Run full 2-stage pipeline for a date |
| `/swing-trend` | Multi-day swing trend analysis |
| `/trend-analysis` | Historical replay trend analysis |
| `/wheel` | Wheel strategy pipeline (CSP/CC) |

### Skills (auto-triggered)

| Skill | Triggers On |
|-------|------------|
| `daily-pipeline` | "run pipeline", "daily analysis", "best trades for" |
| `swing-trend` | "swing trend", "lookback", "weekly trend" |
| `trend-analysis` | "trend-analysis 2026-04-17", "historical UW trends", "backtest-gated trend trades" |
| `trade-desk` | Trade monitoring and position management |

Commands: `.claude/commands/` — Skills: `.claude/skills/`

---

## Configuration

Main config: `uwos/rulebook_config_goal_holistic_claude.yaml`

| Section | What It Controls |
|---------|-----------------|
| `fire:` | Debit spread DTE range, width tiers, OTM caps, breakeven distance |
| `shield:` | Credit spread DTE, short strike OTM% (0.18 targets 20-30δ), sigma factors |
| `gates:` | Width bounds, credit/debit ratios, max risk per trade |
| `approval:` | Likelihood thresholds, dynamic delta cap, GEX regime, entry tolerance |
| `high_beta:` | Special rules for TSLA/NVDA/MU/PLTR |

### Tuning History

| Version | Date | Key Changes |
|---------|------|-------------|
| T6 | Mar 2026 | Macro regime gate (SPY/VIX), bear trade unblock, entry tolerance |
| T7 | Apr 2026 | Dynamic SHIELD delta cap (IVR/DTE/VIX/GEX-aware), OTM 0.12→0.18, macro reuse, sentinel normalization |

---

## Project Structure

```
c:\uw_root/
├── .claude/commands/          # Slash commands (/daily-pipeline, /swing-trend, etc.)
├── .claude/skills/            # Auto-triggered skills (daily-pipeline, trade-desk, etc.)
├── .env                       # API credentials (gitignored)
├── tokens/                    # OAuth tokens (gitignored)
├── uwos/                      # Python package
│   ├── run_mode_a_two_stage.py       # Daily pipeline (Stage-1 + Stage-2)
│   ├── eod_trade_scan_mode_a.py      # Stage-1 discovery engine
│   ├── setup_likelihood_backtest.py  # Historical backtest
│   ├── trade_monitor.py              # Position monitoring + ntfy alerts
│   ├── dip_scanner.py                # S&P 500 dip screener
│   ├── schwab_auth.py                # Schwab OAuth + live data
│   ├── schwab_position_analyzer.py   # Position enrichment
│   ├── swing_trend_pipeline.py       # Multi-day trend pipeline
│   ├── pricer.py                     # Live spread pricer
│   ├── generate_whale_summary.py     # Whale trade summarizer
│   └── rulebook_config_goal_holistic_claude.yaml  # Main config
├── YYYY-MM-DD/                # Daily data folders (5 UW files + expert trade table)
└── out/                       # Pipeline outputs
```

---

## Glossary

| Term | Meaning |
|------|---------|
| **FIRE** | Debit spread track — Bull Call Debit, Bear Put Debit |
| **SHIELD** | Credit spread track — Bull Put Credit, Bear Call Credit, Iron Condor |
| **Core / Tactical / Watch** | Trade quality tiers: full size / reduced / don't trade |
| **IVR** | IV Rank (0–100) — how high current IV is vs its 52-week range |
| **GEX** | Gamma Exposure — dealer hedging creates pinning or amplification |
| **Pinned** | Positive GEX — suppresses moves (good for credit spreads) |
| **Volatile** | Negative GEX — amplifies moves (bad for credit spreads) |
| **Conviction** | 0–100 composite score (direction, efficiency, liquidity, whale) |
| **Edge** | Backtest-derived expected profit percentage |
| **Macro Regime** | SPY 5d return + VIX level → risk_off / neutral / risk_on |
| **Entry Gate** | Max debit or min credit threshold to enter a trade |
| **ntfy** | Push notification service — phone alerts for position verdicts |
| **UW** | UnusualWhales — options flow data provider |
