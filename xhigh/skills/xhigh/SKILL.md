---
name: xhigh
description: >
  Run the xhigh new-setup wheel/swing scanner. Triggers include xhigh,
  xhigh full, xhigh YYYY-MM-DD, xhigh today, xhigh analyze TICKER,
  xhigh revalidate, xhigh intel. Independent of groat, wheelo, groko,
  and Codex Daily. Schwab last first, then ORATS delayed. No order placement.
---

# xhigh

CODE=`/Users/anuppamvi/tradedesk/xhigh`

New opportunistic tickets only. **No ticket cap.** Empty **CLICK** is valid. Never submit, cancel, or replace an order. v1 is locked (`docs/LOCK.md`).

Never invent ORATS, Schwab, X, news, SEC, or earnings numbers. Missing source → **DATA UNAVAILABLE**.

Do not treat `groat`, `RUN FULL SCAN`, or `wheelo` as this desk.

## Parse

| User says | CMD | DATE |
|---|---|---|
| `xhigh` / `xhigh full` / `xhigh today` | full | today America/New_York |
| `xhigh YYYY-MM-DD` | full | that date |
| `xhigh analyze NVDA` | analyze NVDA | today unless a date is given |
| `xhigh revalidate` | revalidate | today unless a date is given |

## Run

From CODE, timeout 300000ms.

```bash
python3 -m xhigh full --date DATE
```

Analyze: `python3 -m xhigh analyze TICKER --date DATE`.

If `orats=missing`, say so. Do not print the token.

## Intel after the scan

For SPY (macro) and every **CLICK** ticker:

1. X last 7d (`x_keyword_search`). Tag Quiet | Informed | Crowded.
2. News + SEC 8-K headlines (cite URL). Thesis-break → kill.
3. Earnings **content** if a print is in the last ~15d (cite). Date still from the stack; `wksNextErn` is est.
4. Write `CODE/var/intel/DATE/macro.json` and `CODE/var/intel/DATE/TICKER.json`. Also `var/xhot/DATE/hot.json` for X.
5. `python3 -m xhigh intel --date DATE` then `python3 -m xhigh xhot --date DATE`.

Intel never creates a trade or changes last/bid/ask.

## Disprove (CLICK rows only)

KILL / SURVIVE / INSUFFICIENT DATA. New CSP: “Would I still want 100 shares 8–15% lower?” if yes → SURVIVE. Missing earnings date on an option → KILL the option. Missing delta → not CLICK.

## Reply

Copy **Recommendation** from `out/xhigh/DATE/recommendation.md` (also the top of `board.md`) **verbatim at the top**. Then the CLICK table. Do not lead with a 20-row dump. 🟢 CLICK / 🔴 SKIP / 🟡 WATCH. 0 CLICK is valid.

Credits are **bid**. Debits are **ask − short bid**. A 270-call on a $186 stock is a bug. Do not promise profit. POP is delta, not a forecast.
