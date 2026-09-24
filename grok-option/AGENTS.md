# grok-option

CODE is this repository: `/Users/anuppamvi/tradedesk/grok-option`.

grok-option is the Expert Trade Table / defined-risk swing scanner. Independent of groat, groko, wheelo, and Codex Daily. Schwab live chain first. No ORATS. No order placement.

The skill lives in `CODE/SKILL.md` (plus `references/`, `assets/`, `scripts/`). tradedesk `.grok/skills/grok-option` and `.claude/skills/grok-option` are **symlinks to CODE**. `~/.grok/skills/grok-option` is also a symlink to CODE so the TUI default path does not break.

When the user says `grok-option`, `grok-option YYYY-MM-DD`, `after market`, `market live`, `run today's scan`, `Anu table`, `bull put`, `sell put credit`, `manage open book`, read `SKILL.md` and run it. Do not tell the user to type `python3`.

`--asof` is the date they typed. `--session ah` if they said after market / post market / AH / EOD; `--session live` if they said market live / RTH / market-open, or date only.

Schwab: tradedesk `.env` + `SCHWAB_TOKEN_PATH`. Never print tokens.

```bash
python3 /Users/anuppamvi/tradedesk/grok-option/scripts/schwab_market.py scan \
  --asof YYYY-MM-DD --session ah --regime auto --workers 4 \
  --out /Users/anuppamvi/tradedesk/grok-option/out/grok-option/YYYY-MM-DD/scan_ah.json
python3 /Users/anuppamvi/tradedesk/grok-option/scripts/schwab_market.py book \
  --session ah --out /Users/anuppamvi/tradedesk/grok-option/out/grok-option/YYYY-MM-DD/book_ah.json
python3 /Users/anuppamvi/tradedesk/grok-option/scripts/schwab_market.py vertical --symbol TICKER --right P --expiry YYYY-MM-DD --short STRIKE --long STRIKE --kind credit
```

Use `--session live` and `scan_live.json` / `book_live.json` for RTH. `--workers 4` is parallel **names**, one chain per name for the 14–60 DTE Friday window.

The user places every Schwab order. Empty table is valid when quotes or geometry fail.
Never invent quotes, IV, OI, or delta. Missing source → skip the structure.
Before Expert: web-source **name calendar** (company IR earnings, deliveries, unveil, vote) in `(scan, expiry]`. Seasonality folklore is not a veto. A sourced event parks that wing. IR beats aggregator estimates.

Every SCAN writes:

```text
/Users/anuppamvi/tradedesk/grok-option/out/grok-option/YYYY-MM-DD/GROK_OPTION.md
```

Chat is the file link plus 🟢 Expert / 🟡 Review / 🔴 Fail counts. Do not replace the file with an inline-only table. Geometry-pass trades that miss the Expert table still appear as full 🟡 rows. Icons in `assets/report-style.md`. No HTML color tags.
