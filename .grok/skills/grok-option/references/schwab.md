# Schwab live quotes and chains

**This is the source for ticker last/bid/ask and option-chain bid/ask.** Do not invent those fields. Do not scrape Yahoo for a chain when this works.

## Auth (tradedesk)

Run from, or set, the tradedesk root. Load env the same way `uwos.schwab_auth.SchwabAuthConfig.from_env` does:

- File: `<tradedesk>/.env` — `SCHWAB_API_KEY`, `SCHWAB_APP_SECRET`, `SCHWAB_CALLBACK_URL`, `SCHWAB_TOKEN_PATH`, optional `SCHWAB_STRIKE_COUNT`
- Token: `SCHWAB_TOKEN_PATH` (default `./tokens/schwab_token.json` under tradedesk)
- Override root with `UW_ROOT` if cwd is not tradedesk

Never print keys, secrets, or raw token JSON. If refresh fails, tell the user to re-auth in their terminal:

`python3 -m uwos.schwab_position_analyzer --manual-auth`

(or `python3 -m uwos.schwab_quotes --manual-auth --symbols-csv AAPL --chain-symbols-csv AAPL --strike-count 2`)

## Commands

From any cwd:

```bash
python3 ~/.grok/skills/grok-option/scripts/schwab_market.py quote AAPL XOM
python3 ~/.grok/skills/grok-option/scripts/schwab_market.py chain XOM --from-date YYYY-MM-DD --to-date YYYY-MM-DD --strike-count 12
python3 ~/.grok/skills/grok-option/scripts/schwab_market.py structures AMZN --expiry YYYY-MM-DD
python3 ~/.grok/skills/grok-option/scripts/schwab_market.py vertical --symbol XOM --right P --expiry YYYY-MM-DD --short 110 --long 105 --kind credit
```

`structures` prices all five actions on that expiry (put credit, call credit, iron condor, call debit, put debit). Do not skip it and only run puts.

`vertical` is what fills an Expert table row. Credit net = short bid − long ask. Debit net = long ask − short bid. Missing either bid/ask → that structure is NO ROW.

`atm_straddle.straddle_ask` is the quoted 1-sigma proxy for that expiry. Do not estimate sigma.

Delta / IV / OI in the JSON are Schwab fields. Blank if absent. Never LLM-fill.

## SCAN order

After the universe exists, **quote then `structures` via this file** before browser. Browser is for X, cookie-walled news, and crude/geo copy — not for replacing a live Schwab chain.

Data flag: both legs quoted here → **FULL** (if earnings date is also sourced). Schwab down → MIXED/THIN, no Prime, no invented prints.

## Limits

- Defined-risk verticals only. No naked short from a chain dump.
- Earnings overlap still applies (`expiry_date < earnings_date`).
- Do not place orders. This module is market data only.
