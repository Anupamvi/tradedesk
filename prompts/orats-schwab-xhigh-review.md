COPY THIS ENTIRE FILE INTO GROK XHIGH. This file is the whole prompt. Do not ask the human for a watchlist, capital, or extra rules.

You are a research architect. Review **what the Schwab Trader API and ORATS Data API v2 actually return**, and how those fields support **opportunistic wheel and swing** (not day trading, not 0DTE, not order placement).

Stay on the two APIs plus public overlays (X, news, SEC, earnings). Do not mention, assume, or critique any local trading package, repo, or product.

Do not invent quotes, IV, OI, or earnings dates. Undocumented or plan-gated fields: **UNVERIFIED**. Empty days are valid. Forcing a table and killing every idea are both failures.

---

## Hypothesis (test it; do not sell it)

ORATS delayed/EOD analytics + Schwab live market/account data + public context (X, news, SEC, earnings) can time:

- **Wheel:** sell cash-secured puts on names you would own at the strike; sell covered calls on shares you are **willing to have called away**.
- **Swing:** stock thesis first, then options if they express it, holding days to a few weeks.

Say what is a harvestable edge, what is research theater, and what makes side income not doable. Do not promise a monthly dollar from API coverage.

---

## Official docs (read; do not guess)

ORATS: https://docs.orats.io/ · https://docs.orats.io/datav2-api-guide/ · https://docs.orats.io/datav2-api-guide/data.html · https://docs.orats.io/datav2-api-guide/definitions.html · https://docs.orats.io/datav2-api-guide/core-research.html · https://docs.orats.io/one-minute-api-guide/ · https://docs.orats.io/datav2-snapshot-api-guide/data.html

Schwab: https://developer.schwab.com/products/trader-api--individual  
Market Data `https://api.schwabapi.com/marketdata/v1` · Trader `https://api.schwabapi.com/trader/v1`  
In scope: quotes, price history, chains, expirations, instruments, market hours; accounts, positions, balances, transactions. **No orders.**

---

## Source roles

| Source | Use for | Never use for |
|---|---|---|
| Schwab live | Bid/ask/size, chain, bars, positions, cash/BP, marks to close | Historical IV surface; earnings dates |
| ORATS delayed/EOD | IV vs HV, percentile/rank, forecast vs implied, skew, earnings *weeks*, implied vs realized event move, borrow | Fills; 9:30 live marks; scanning thousands of names every day |
| X | Quiet / Informed / Crowded. Confirm or veto. May start a watch | Entry, price, IV, earnings date |
| News / SEC / filings / transcript | Thesis break; announcement veto; post-print follow-through | Quote, IV, strike |

Fills: credit at **bid** (or short bid − long ask); debit at **ask**; never mid; never ORATS `callValue` / `putValue` / `smvVol`.

ORATS traps you must handle: `nextErn` often `0000-00-00`; `daysToNextErn` often junk — use `wksNextErn` plus an independent calendar you actually fetch. `ivPctile1y` ≠ `ivRank1y`. Many vol fields are **percent** (17.73), not 0.1773. Write the unit in every gate.

---

## What you must specify (so the human adds nothing)

You invent **no** watchlist. You **do** write numeric screens from API fields so a shortlist can be built without one, e.g. Schwab: quotable, bid > 0, spread vs credit; ORATS cores: market cap, 20d option volume, asset type. Put the exact inequalities in Table C.

You invent **no** vague “harvest sometimes.” You **do** write one buy-to-close rule for **short-premium structures** (CSP, covered call, credit vertical): formula using opening credit vs **ask** to close, with DTE bands. Apply to the **structure**, not a short leg of a debit spread. Taking profit on an open short is not blocked by “I would still want assignment.”

You invent **no** “check earnings.” You **do** write a ranked date stack of **fetchable** sources (issuer IR or exchange/Nasdaq calendar first; then ORATS `wksNextErn` / `ernDate*`; `nextErn` last and only if a real date). Missing date → no options; stock may proceed.

Covered calls: allowed on shares the human is willing to sell, including shares from assignment. Do not invent a protected-core ban.

---

## Produce in this order

0. ≤10 lines: how your recs could reduce closed expectancy.
1. Hypothesis verdict (edge vs theater vs not-doable).
2. **7-day list** (max three field uses, the BTC rule, the earnings stack). Then tables. Do not bury this.
3. Schwab catalog: path, useful fields, cadence (research / morning revalidate / manage opens), do-not-use. Positions must support % of max on shorts. Schwab is not an earnings vendor.
4. ORATS catalog: `/tickers`, `/cores`, `/summaries`, `/strikes`, `/strikes/options`, `/monies/implied`, `/monies/forecast`, `/ivrank`, matching `/hist/*`, `/hist/dailies`, `/hist/hvs`, `/hist/earnings`, `/hist/divs`, `/hist/splits`. Live / 1-minute / snapshot / backtest: default no unless token + a named decision require them.

   Lists: every session / shortlist only / skip.

   Verdict each group USE NOW / SHORTLIST / SKIP / STUDY with an **exact gate** or skip reason (no “monitor”):

   Vol: `iv30d`, `orHv20d`, `orHvXern20d`, `ivHvXernRatio`, `ivPctile1y`, `ivRank1y`, `orFcst20d`, `orIvFcst20d`  
   Earnings: `nextErn`, `wksNextErn`, `lastErn`, `ernDate1-12`, `absAvgErnMv`, `impErnMv`, `impliedEarningsMove`, `ernMv*`, `ernEffct*`  
   Skew: `slope`, `dlt25Iv30d`, `dlt75Iv30d`, `dlt95Iv30d`  
   Quality: `avgOptVolu20d`, `mktWidthVol`, `borrow30`, `confidence`, `tkOver`, `divDate`, `divAmt`, `annActDiv`, `annIdiv`  
   Term: `atmIvM1-M4`, `dtExM1-M4`, `contango`, `fwd*`/`ffwd*`/`fbfwd*` (default one grouped SKIP unless a hard fail not already covered)  
   Strikes: research-only vs replace-at-ticket-with-Schwab

5. Leverage layers (gates, not a blended score):

   0 Schwab liquidity screen → ORATS on that shortlist only  
   1 Schwab tape; stock thesis first  
   2 ORATS vol/earnings; event inside DTE?  
   3 Schwab chain; conservative net; CSP only if 100 shares at the strike is acceptable, else defined-risk put spread; CC only if calling away is acceptable  
   4 Earnings stack + SEC/news; no ordinary short premium through unknown earnings; announcement bets are opt-in and separate; post-print drift is ordinary swing  
   5 X tag; never a click  
   6 Schwab book: close qualifying short-premium structures first, then new paper is allowed  
   7 Disprove on new TRADE rows only  

   Evening = delayed ORATS + complete daily bars. Morning = live Schwab before a click. No 9:30 fill of last night’s option ticket.

6. Disprove: KILL / SURVIVE WITH NAMED RISK / INSUFFICIENT DATA. No invented data. No new trade. New CSP: “still want 100 shares 8–15% lower?” if yes → SURVIVE. That question must not veto a take-profit close.

7. Validation: 20–40 name frozen cohort; conservative fills; pre-register field rules; 20 closed overlay trades before trusting expectancy; missed trades counted only if they passed the gates that day.

8. Budget: Schwab first; ORATS cores on the shortlist; strikes on finalists; hist only as a planned study. Extra `fields=` on an already-fetched cores row is free.

---

## Tables

**A** Field → decision (gates/promotes only): Provider, Endpoint, Field, Unit, Vintage, Lane (wheel / CC / swing-stock / swing-opt / event / harvest), Decision, Hard-fail?, Schwab-replaces-at-ticket?

**B** Anti-patterns: field shopping, X as entry, mids, delayed bids as tickets, kill-all, always-a-table, unit mix, closing a debit by buying in only the short leg.

**C** Daily run: Step, API, how names enter, output, skip if.

**D** ORATS group verdicts: extra HTTP yes/no.

**E** Disprove questions by lane.

No questions to the human. No architecture for some other codebase.

Start at 0–1–2 (7-day list), then Table D, then the rest.
