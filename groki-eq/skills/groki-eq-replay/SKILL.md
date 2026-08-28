---
name: groki-eq-replay
description: Offline groki-eq breakout replay. Triggers: groki-eq replay, /groki-eq-replay.
---

# groki-eq replay

CODE=`/Users/anuppamvi/tradedesk/groki-eq`

From CODE, run (do not tell the user to type this):

```bash
python3 -m groki_eq.replay --start START --end END --max-days 0
```

- `groki-eq replay` with no range → `--start 2018-01-02 --end` today America/New_York.
- `groki-eq replay DATE` → `--end DATE`.
- `groki-eq replay START END` → that range.

Cache-first daily bars. No order placement. Entry close. Stop 2*ATR(14). Time stop 15 sessions. No profit-target exit.
Split `2023-01-03`. Promote only TEST PF>=1.2 and n>=30.
If `ORATS_TOKEN` is missing, tell the user to edit `CODE/.env`. Do not ask them to paste the token. Never print it.
