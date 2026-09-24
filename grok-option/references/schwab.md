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
python3 /Users/anuppamvi/tradedesk/grok-option/scripts/schwab_market.py quote AAPL XOM
python3 /Users/anuppamvi/tradedesk/grok-option/scripts/schwab_market.py book
python3 /Users/anuppamvi/tradedesk/grok-option/scripts/schwab_market.py chain XOM --from-date YYYY-MM-DD --to-date YYYY-MM-DD --strike-count 12
python3 /Users/anuppamvi/tradedesk/grok-option/scripts/schwab_market.py structures AMZN --expiry YYYY-MM-DD --regime normal
python3 /Users/anuppamvi/tradedesk/grok-option/scripts/schwab_market.py scan --asof YYYY-MM-DD --session ah --regime auto --out /Users/anuppamvi/tradedesk/grok-option/out/grok-option/YYYY-MM-DD/scan_ah.json
python3 /Users/anuppamvi/tradedesk/grok-option/scripts/schwab_market.py book --session ah --out /Users/anuppamvi/tradedesk/grok-option/out/grok-option/YYYY-MM-DD/book_ah.json
python3 /Users/anuppamvi/tradedesk/grok-option/scripts/schwab_market.py vertical --symbol XOM --right P --expiry YYYY-MM-DD --short 110 --long 105 --kind credit
```

`scan --asof YYYY-MM-DD --session ah|live` quotes `$VIX`, maps Calm/Normal/Elevated, and runs `structures` on the stock universe × listed Fridays in **14–60 DTE**. `--asof` is the user date (DTE window). `--session ah` is after-hours / EOD; `--session live` is RTH / market-open. Date-only defaults to `live`. **Stdout is the compact board**. `--out FILE` writes the full JSON; if omitted: `CODE/out/grok-option/ASOF/scan_ah.json` or `scan_live.json`. Never `cat` that file into chat. `--full` on stdout is a token bomb — do not use it on SCAN. Default `--workers 4` runs **names in parallel**, **one Schwab chain per name** covering the Friday window (score expiries in-process). Do not pull one HTTP per Friday. Stderr is **one line per name**. After `scan`, do **not** re-run `structures` on every name. `structures --regime` applies that bucket’s gates (cheap-vol Shield is 0.12 / 0.22Δ / 0.80σ in Calm **and** Normal). Chain timeout defaults to 45s (`UWOS_SCHWAB_OPTION_CHAIN_TIMEOUT_SECONDS`).

`structures` prices all five actions on that expiry (put credit, call credit, iron condor, call debit, put debit). Do not skip it and only run puts. Use it for a single name/expiry probe, not as a second universe pass.

`vertical` fills an Expert table row. Default stdout is compact (legs + net). `--full` prints the chain blob. Credit net = short bid − long ask. Debit net = long ask − short bid. Missing either bid/ask → that structure is NO ROW.

`book --session ah|live` is live Schwab equity, cash, long names, and option overlay. Default out is `book_ah.json` / `book_live.json` under the dated folder. Use it instead of a one-off `get_account_positions` script.

Board credit rows include naive `pop`, `sb`/`la` (short bid, long ask), and `in_ic` (put/call already inside an IC — Expert the IC, not the wing). A credit is on the board only when the pricer set `quoted`. `worse_fill: false` is not that mark: a 0 ask makes the mid and the conservative fill agree. `tape` has spot and per-expiry ATM straddle ask.

`atm_straddle.straddle_ask` is the quoted 1-sigma proxy for that expiry. Do not estimate sigma.

Delta / IV / OI in the JSON are Schwab fields. Blank if absent. Never LLM-fill.

## SCAN order

After the universe exists, **`scan` then `book`** before browser. `vertical` only for Expert candidates. Do not re-run `structures` on every name. Browser is for X, cookie-walled news, and crude/geo copy — not for replacing a live Schwab chain.

Data flag: both legs quoted here → **FULL** (if earnings date is also sourced). Schwab down → MIXED/THIN, no Prime, no invented prints.

## Limits

- Defined-risk verticals only. No naked short from a chain dump.
- Earnings overlap still applies (`expiry_date < earnings_date`).
- Do not place orders. This module is market data only.
