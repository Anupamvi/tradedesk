# xhigh

New-setup opportunistic wheel/swing scanner. Independent of groat, wheelo, groko. v1 locked — see `docs/LOCK.md`.

```bash
cd /Users/anuppamvi/tradedesk/xhigh && PYTHONPATH=. python3 -m xhigh full --date YYYY-MM-DD
```

If DATE is not today America/New_York (weekend / next-day after-market), add `--live-schwab`. Movers may still be empty off-session; that is DATA UNAVAILABLE, not empty CLICK.

Output: `out/xhigh/YYYY-MM-DD/`

Schwab token is `SCHWAB_TOKEN_PATH` at tradedesk `tokens/schwab_token.json` (do not keep a private copy under `xhigh/tokens`).

No orders. Empty TRADE is valid. Spot is Schwab lastPrice. CSP 8–15% OTM. Call debit long −2% to +4% vs last.
