# trendbook

Independent weekly Stage 2 / relative-strength trend book. Not Groat, not xhigh, not grok-option.

```bash
cd /Users/anuppamvi/tradedesk/trendbook && PYTHONPATH=. python3 -m trendbook full --date YYYY-MM-DD
```

Output: `out/trendbook/YYYY-MM-DD/`

Stock first. Dynamic universe (this desk's memory + live 52-week highs + movers; not a static txt, not other desks). One ADD per campaign. Typical week 1–2 genuine first tickets. Breakout (break-week volume, residual 3m ≥ 0) or first early pullback. No new ADD if SPY not Stage 2. Stop = weekly close below 30-week. NEW = weak tag. LATE = old and dying. Live 52-week-high probe. Five-year tape. Grade A/B/C is structure, not P(win). Empty ADD is valid. No orders. Schwab token stays at `SCHWAB_TOKEN_PATH`. ORATS token from xhigh/.env then groat/.env then tradedesk `.env`.
