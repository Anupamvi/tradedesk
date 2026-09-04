# Compound Core

Long-term index sleeve. Cheap ETFs, weekly buys, no stock-picking, no options, **no orders**. Independent of groat, wheelo, xhigh, groko, and Codex Daily.

Default weights: **VOO 48 / VGT 10 / SMH 7 / VB 5 / VXUS 20 / GLDM 5 / VGSH 5**.

## Run

```bash
cd /Users/anuppamvi/tradedesk/compoundcore
PYTHONPATH=. python3 -m compoundcore 100000
PYTHONPATH=. python3 -m compoundcore 250000 --weekly 500 --monthly 1000
PYTHONPATH=. python3 -m compoundcore dashboard
PYTHONPATH=. python3 -m compoundcore calculator
PYTHONPATH=. python3 -m unittest discover -s tests -v
```

Playbook: [`docs/PLAYBOOK.md`](docs/PLAYBOOK.md)

Persistent dashboard: `python3 -m compoundcore dashboard` then open `http://127.0.0.1:8765/`. Type a dollar amount to split **both** sleeves. In **My book**, save cost vs now; refresh keeps real gain or loss.

Raw HTML calculator: [`web/calculator.html`](web/calculator.html) — open the file as-is.

## Rules in one line

No leverage. Never sell this core to fund a trading ticket. Buy every week. Size SMH so a −45% chip crash is a single-digit hit. Core is never abandoned.

Not financial advice. CMAs are not guarantees.
