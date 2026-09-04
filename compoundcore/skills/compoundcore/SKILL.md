---
name: compoundcore
description: >
  Run the Compound Core long-term index sleeve, dollar calculator, and dashboard.
  Triggers include compound core, compoundcore, core sleeve, core calculator,
  core dashboard, ToothFolio, VOO VGT SMH VB sleeve, allocate core, /compoundcore.
  Independent of groat, wheelo, xhigh, groko, and Codex Daily.
  No stock-picking, no options, no order placement.
---

# Compound Core

CODE=`/Users/anuppamvi/tradedesk/compoundcore`

Long-term **core**. Default **VOO 48 / VGT 10 / SMH 7 / VB 5 / VXUS 20 / GLDM 5 / VGSH 5**. Aggressive variant is 45/15/10/5/15/5/5. The user places every Schwab order. Never submit, cancel, or replace. Never sell this sleeve to fund a trading-desk ticket.

Never invent CMA numbers, ETF holdings, prices, or X posts. Missing source → **DATA UNAVAILABLE**.

## Parse

| User says | CMD |
|---|---|
| `compound core` / `compoundcore` / `core sleeve` (no dollars) | playbook + `$100,000` example of **both** sleeves |
| `compound core $250000` / `allocate 250000` / a dollar amount | `calc --amount AMOUNT` both sleeves |
| weekly and/or monthly mentioned | add `--weekly` and/or `--monthly` |
| `aggressive` / `default` only | `--sleeve aggressive` or `--sleeve default` |
| `core dashboard` / persistent dashboard / my book | `dashboard` then open `http://127.0.0.1:8765/` |
| `core calculator` | write/open `web/calculator.html` (raw HTML) |

If they give one number and no weekly/monthly, still print both sleeves' dollar split, the per-$1,000 weekly recipe, and lump-sum 5y/10y paths.

## Run (agent runs this)

From CODE, timeout 30000ms.

```bash
PYTHONPATH=. python3 -m compoundcore AMOUNT --weekly WEEKLY --monthly MONTHLY
PYTHONPATH=. python3 -m compoundcore dashboard
PYTHONPATH=. python3 -m compoundcore calculator
```

`python3 -m compoundcore 100000` is `calc --amount 100000 --sleeve both`.

Dashboard (persistent): both sleeves, then **My book** with saved cost vs now and real gain/loss. `http://127.0.0.1:8765/`. Raw calculator stays `CODE/web/calculator.html`. Playbook path: `CODE/docs/PLAYBOOK.md`.

## Reply shape

Lead with the **default** dollar table for their amount, then aggressive, then 5y/10y base (and stress). One line: no orders, not a 40%/yr plan. Link the playbook. If they asked for the dashboard, start it and give the local URL.

Do not lecture sleeve math beyond the table. Do not recommend 50/30/20 VOO/VGT/SMH as the default. Do not place trades.
