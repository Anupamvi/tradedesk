---
name: groki-eq
description: Run the groki-eq index equity breakout board for a date. Triggers: groki-eq, groki-eq YYYY-MM-DD, /groki-eq.
---

# groki-eq daily

CODE=`/Users/anuppamvi/tradedesk/groki-eq`

From CODE, run (do not tell the user to type this):

```bash
python3 -m groki_eq --date DATE
```

- `groki-eq` with no date → `--date` today America/New_York.
- `groki-eq YYYY-MM-DD` → that date.

User clicks every Schwab order. No submit/cancel/replace.
If `ORATS_TOKEN` is missing, tell the user to edit `CODE/.env` or run `read -s ORATS_TOKEN && export ORATS_TOKEN`. Do not ask them to paste the token. Never print it.
