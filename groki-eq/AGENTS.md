# groki-eq

CODE is this repository. Skills live in `./skills/` — not under `.claude` or vendor skill folders.

Grok Build does not auto-scan `./skills/`. When the user says the phrases below, read the matching skill and follow it.

- User says `groki-eq YYYY-MM-DD` → read `skills/groki-eq/SKILL.md` and run it.
- User says `groki-eq` with no date → same skill; date is today America/New_York.
- User says `groki-eq replay` → read `skills/groki-eq-replay/SKILL.md` and run it.

Run the skill's command yourself. Do not tell the user to type `python3` or `--date`.

If `ORATS_TOKEN` is missing, tell the user to edit `CODE/.env` (`/Users/anuppamvi/tradedesk/groki-eq/.env`) or run `read -s ORATS_TOKEN && export ORATS_TOKEN`. Do not ask them to paste the token into Grok. Never print the token.

The user places every Schwab order. This code does not submit, cancel, or replace orders.
Empty board is a valid day.
