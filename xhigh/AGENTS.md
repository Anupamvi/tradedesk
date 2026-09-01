# xhigh

New-setup opportunistic wheel/swing scanner. Independent of groat, wheelo, groko. v1 locked — see `docs/LOCK.md`.

```bash
cd /Users/anuppamvi/tradedesk/xhigh && PYTHONPATH=. python3 -m xhigh full --date YYYY-MM-DD
```

Output: `out/xhigh/YYYY-MM-DD/`

No orders. Empty TRADE is valid. Spot is Schwab lastPrice. CSP 8–15% OTM. Call debit long −2% to +4% vs last.
