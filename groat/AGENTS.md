# Groat

CODE is this repository: `/Users/anuppamvi/tradedesk/groat`.

Groat is the GROK MASTER SWING-TRADING RESEARCH AGENT. Independent of groki, groki-eq, groko, Codex Daily, and Pattern Analysis.

Skills live in `./skills/` and are mirrored under tradedesk `.grok/skills/groat` and `.claude/skills/groat`.

When the user says the phrases below, read the matching skill and run it. Do not tell the user to type `python3`.

- `groat`, `gorat`, `groat YYYY-MM-DD`, `groat full`, `RUN FULL SCAN` → `skills/groat/SKILL.md` (full scan)
- `groat delta`, `RUN DELTA SCAN` → same skill, delta mode
- `ANALYZE TICKER`, `groat analyze TICKER` → same skill, analyze mode
- `REVIEW OPEN TRADES`, `groat review` → same skill, review mode

If `ORATS_TOKEN` is missing, tell the user to edit `CODE/.env` or run `read -s ORATS_TOKEN && export ORATS_TOKEN`. Do not ask them to paste the token. Never print the token.

Schwab credentials load from `CODE/.env` then `/Users/anuppamvi/tradedesk/.env`.

The user places every Schwab order. This code does not submit, cancel, or replace orders.
Empty board is a valid day.
Never invent ORATS numbers, prices, X posts, or news. Missing source → **DATA UNAVAILABLE**.
