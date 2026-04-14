# CLAUDE.md

## Behavioral Guidelines

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

## Project: UW Root — Options Trading Platform

### What This Is
Automated options trading platform that screens, analyzes, and monitors trades using Schwab API + UnusualWhales data.

### Key Modules

| Module | Purpose | Command |
|--------|---------|---------|
| `trade_monitor.py` | Position surveillance + ntfy alerts | `python -m uwos.trade_monitor --force --manual` |
| `trade_ideas.py` | New trade discovery (Schwab + UW flow) | `python -m uwos.trade_ideas` |
| `schwab_position_analyzer.py` | Fetch + enrich open positions | `python -m uwos.schwab_position_analyzer --days 90` |
| `run_mode_a_two_stage.py` | Daily 2-stage pipeline | `python -m uwos.run_mode_a_two_stage --base-dir ...` |
| `eod_trade_scan_mode_a.py` | Candidate screening + macro regime | `compute_macro_regime()` |
| `backtest_ideas.py` | March backtest of trade ideas | `python -m uwos.backtest_ideas` |
| `test_verdicts.py` | 63-case verdict test suite | `python -m uwos.test_verdicts` |

### Paths

- Project root: `c:\uw_root`
- UWOS module: `c:\uw_root\uwos`
- Daily data: `c:\uw_root\YYYY-MM-DD\` (UW zips + whale markdown)
- Output: `c:\uw_root\out\` (trade_analysis, trade_ideas, opportunities)
- Config: `c:\uw_root\uwos\rulebook_config_goal_holistic_claude.yaml`
- Token: `c:\uw_root\tokens\schwab_token.json`
- Credentials: `c:\uw_root\.env` (gitignored)

### Environment

- **Schwab API**: credentials in `.env` (SCHWAB_API_KEY, SCHWAB_APP_SECRET, SCHWAB_TOKEN_PATH)
- **Re-auth**: `del "c:\uw_root\tokens\schwab_token.json"` then `python -m uwos.schwab_position_analyzer --manual-auth`
- **ntfy**: topic `uw-trades-transition`, configured in `.env` (NTFY_TOPIC)
- **Scheduler**: Windows Task `TradeMonitor`, every 30 min, runs on battery, catches up after sleep

### Critical Rules

1. **Always read `.env` before giving credential/token file paths** — never guess
2. **Run `python -m uwos.test_verdicts` before shipping any verdict engine changes** — 64 test cases must pass
3. **Schwab API is the primary real-time data source** — not yfinance. yfinance is fallback only for fundamentals
4. **Credit trades get patience, debit trades get urgency** — different verdict matrices
5. **Verdicts use hysteresis** — once escalated (HOLD→ASSESS), requires $300+ P&L improvement to de-escalate. Prevents flip-flopping
6. **Equity verdicts are context-aware** — compare stock drop to SPY 5d return. In broad selloff, widen thresholds
7. **Trade ideas exclude held positions** — auto-reads from monitor_state.json
8. **Notifications go to ntfy only** — SMS was removed (T-Mobile gateway blocked)
9. **All output as .md file references with clickable VSCode-compatible links**

### Verdict Engine Rules (compute_verdict)

**Credit positions:**
- ≥85% max → CLOSE
- ITM + DTE ≤ 5 + delta > 0.85 → CLOSE (assignment risk)
- ITM + DTE ≤ 14 → ROLL
- Within 1.5% of strike + DTE ≤ 3 → CLOSE (pin risk)
- ITM + (>5% or delta > 0.50) → ASSESS
- ITM at all → ASSESS
- DTE ≤ 7 + < 50% max → CLOSE (expiration week gamma)
- Earnings ≤ 7 days + ≥25% of max profit → CLOSE
- ≥75% max → CLOSE
- Delta > 0.45 → ASSESS (approaching ATM)
- ≤ -80% max → ASSESS (deep loss)

**Debit positions:**
- Down > 60% → CLOSE
- OTM > 5% + DTE < 35 → CLOSE
- OTM any + DTE < 14 → CLOSE (theta acceleration)
- Down > 40% → ASSESS
- OTM 3-5% + DTE < 35 → ASSESS
- Earnings ≤ 5 days → CLOSE if profitable, ASSESS if losing

**Equity positions:**
- Today -7%+ → CLOSE (crash)
- Today -5%+ → ASSESS (rapid drop)
- Down > 60% → CLOSE (tax harvest)
- Down > 40% → CLOSE (deep loss; widens to 55% in broad selloffs — compares stock drop to SPY 5d)
- Down > 25% → ASSESS (review thesis; widens to 35% in broad selloffs)
- Up > 100% → ASSESS (trim candidate)

### Output Conventions

- Always present output as `.md` file references with clickable paths
- Format: `[filename.md](relative/path/filename.md)`
- Include both analysis report and position data JSON links
- Use markdown tables for structured data
