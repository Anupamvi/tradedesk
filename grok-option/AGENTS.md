# grok-option

CODE is this repository: `/Users/anuppamvi/tradedesk/grok-option`.

grok-option is the Expert Trade Table / defined-risk swing scanner. Independent of groat, groko, wheelo, and Codex Daily. Schwab live chain first. No ORATS. No order placement.

The skill lives in `CODE/SKILL.md` (plus `references/`, `assets/`, `scripts/`). tradedesk `.grok/skills/grok-option` and `.claude/skills/grok-option` are **symlinks to CODE**. `~/.grok/skills/grok-option` is also a symlink to CODE so the TUI default path does not break.

When the user says `grok-option`, `run today's scan`, `Anu table`, `bull put`, `sell put credit`, `manage open book`, read `SKILL.md` and run it. Do not tell the user to type `python3`.

Schwab: tradedesk `.env` + `SCHWAB_TOKEN_PATH`. Never print tokens.

```bash
python3 /Users/anuppamvi/tradedesk/grok-option/scripts/schwab_market.py structures TICKER --expiry YYYY-MM-DD
python3 /Users/anuppamvi/tradedesk/grok-option/scripts/schwab_market.py vertical --symbol TICKER --right P --expiry YYYY-MM-DD --short STRIKE --long STRIKE --kind credit
```

The user places every Schwab order. Empty table is valid when quotes or geometry fail.
Never invent quotes, IV, OI, or delta. Missing source → skip the structure.
