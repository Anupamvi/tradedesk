# Wheelo

CODE is this repository: `/Users/anuppamvi/tradedesk/wheelo`.

Wheelo is the GROK wheel desk (cash-secured puts → assignment → covered calls). Independent of groat, groko, groki, Codex Daily, and `uwos.wheel_pipeline`.

Skills live in `./skills/` and are mirrored under tradedesk `.grok/skills/wheelo` and `.claude/skills/wheelo`.

When the user says `wheelo`, `wheelo select`, `wheelo daily`, `wheelo YYYY-MM-DD`, read `skills/wheelo/SKILL.md` and run it. Do not tell the user to type `python3`.

If `ORATS_TOKEN` is missing, tell the user to edit `CODE/.env` or run `read -s ORATS_TOKEN && export ORATS_TOKEN`. Do not ask them to paste the token. Never print the token.

Schwab credentials load from `CODE/.env` then `/Users/anuppamvi/tradedesk/.env`.

Schwab shortlists first. ORATS delayed cores/strikes run only after the shortlist (max 80 cores, 20 strikes, 15 HTTP/run). Today/live always refetches; disk JSON is audit only. The user places every Schwab order and sizes cash. This code does not submit, cancel, or replace orders.
Empty TRADE board is a valid day. **conf** is 0-85 structure/research quality, not P(win).
Never invent ORATS, Schwab, or X numbers. Missing source → **DATA UNAVAILABLE**.
