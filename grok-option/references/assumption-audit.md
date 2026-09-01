# Assumption audit — 2026-08-26

Tape + evidence for grok-option v3.2. Unverified numbers stay unverified. Next AUDIT: when the journal's 20-trade rolling expectancy contradicts a rule, rewrite this file and bump the Assumptions-in-force footnote.

## Tape + Evidence (2026-08-26)

Date is Wednesday 2026-08-26.
Bar is **$10k profit / month**, not a $410k desk. That number was a misread of “410k/month options.”
Live Schwab 2026-08-26: **$715,180**, cash ~$24k, mixed equity + options (90-day pull). $150k is only if unknown and Schwab is down.
VIX last regular close 15.45 on 2026-08-25 (MarketWatch, WSJ).
Live 2026-08-26: ~15.66–15.71. 52-week 13.38–35.30 (WSJ). GuruFocus long-term average 18.6. Tape is **Calm**.
WSJ printed VIX 5-day +5.51%; closes 8/19 14.89 → 8/25 15.45 are about +3.8%. Cite both.
SPX last 7,677.28. SPY Schwab ~765. Spot is on/near the 20-day and above the 50-day. Do not invent SMA.
20-day SMA is unverified to the dollar (SwingTradeBot / EODData / FXEmpire disagree).
Desk X is not a trigger: @spotgamma 0DTE/PCE and NVDA 5.5 vs 7.4; CheddarFlow NVDA week-active.
Vendor GEX/put-wall maps are unverified and conflict with each other.
PCE printed this morning (core ~+0.2% MoM / 3.3% YoY). Done, not still-ahead.
NVDA FY Q2 AMC. Also CRM, CRWD, SNPS, OKTA AMC. Theme Event = semis/AI-infra only.
Jackson Hole; Chair Warsh Friday 8/28 (weekly OPEX same day). Next NFP 9/4, CPI 9/11, FOMC 9/16.
SpotGamma: ~0.25% extra Thursday for NVDA, ~0.50% Friday for Warsh. That is not a book-off switch.
NVDA options imply ~5.5% vs ~7.4% average realized last 12 months — do not short NVDA vol into the print.
Live Schwab, 14 names, VIX ~15.7: 1-sigma AND 25% width = **0 hits**. Calm geometry (≤0.22Δ, ≥0.80σ, ≥12% width) printed AMZN Sep-18 net 0.73 and UNH Oct-16 net 1.20.
Those two rows were undersized off a fake $410k book. Live size is 1.0% of $715k **then** buying power (~$24k cash, 25 option lines already open).
Three over-gates this session: NVDA as market-wide Event; Fire off until Monday after Warsh; “harvest Thursday/Monday” as the only tradable window.
Class of error: skip allowed names. Geometry was not the problem; permission was.
UOA/PCR/dark-pool papers unchanged: Jiang & Strong (SSRN 2024) CNBC-covered UOA reverses; JPM Aug 2026 OptionMetrics edge faded after 2020; Grauer et al. (SSRN 2026) mis-tags demand; FINRA TRF in 10 seconds is not a wall.
No invented IV/OI. $10k/month is the bar, not a promise and not a size-up. Do not drop 0.22Δ / 0.80σ to fill a table.

## Verdicts (KEEP / AMEND / KILL)

| # | Assumption | Verdict | Reason + replacement |
|---|-----------|---------|----------------------|
| 1 | Unusual options flow is a directional leading indicator | **AMEND** | Filtered UOA has some edge; public/copied UOA does not, and it faded after 2020. Replacement: flow confirms or vetoes after price + live chain. Never the entry. Prefer opening, ask-side (debits) / bid-side (credits), vol>OI, liquid names. |
| 2 | PCR extremes on unusual volume predict the next 0–5 day move | **KILL** | PCR without aggressor and open/close is overwrite noise (@ask_volAI 8/24; Grauer et al. 2026). Replacement: ignore raw PCR as a forecast. |
| 3 | Dark-pool prints create tradeable walls | **KILL** | 10-second TRF prints are not walls; direction and intent are unknown. Replacement: context only, never a strike or size input. |
| 4 | A 7- or 9-item mostly-yes checklist equals high conviction | **KILL** | Checklists mint fake 74.0 scores. Replacement: **Score** = gates (80/65/50). **Conf** = naive POP from quoted delta + book 20-trade win rate. Fail a gate = no row. |
| 5 | Credit spreads at ≥25% of width are the default path to $10k/month | **AMEND** | 25%+1-sigma is an empty set in VIX~15. Replacement: Calm min width **0.12** with \|delta\|≤0.22. Elevated keeps 0.25. Calm 12% is **not** the $10k engine; Elevated is when options can carry a month. |
| 6 | $10k/month is a strategy | **AMEND** | It is the **stated bar** ($120k/year), not a strategy and not a size-up. On $715k that is 1.4%/month of the **whole account**. Options in Calm cannot floor $10k every calendar month at 12% width. Replacement: year-average scoreboard; harvest first; Shield that clears; do not chase a green month. |
| 7 | IV rank >70 is required to sell premium | **AMEND** | IVR>70 is rich, not a permission slip. Replacement: IVR selects the sleeve. Calm/Normal may sell Shield on liquid mega-caps if credit/width and sigma hold. Low IV → Fire debit or skip. Crisis → cash. |
| 8 | Copying a whale’s exact strike/expiry is the edge | **KILL** | Multi-leg mis-tags, closing prints, hedges. Replacement: print seeds ticker/side only; build from today’s live chain. |
| 9 | LLM-estimated volume multiples, IV, and OI% are acceptable in the table | **KILL** | Invented microstructure is Anu v1 theater. Replacement: quote it or omit it. THIN cannot be Prime. |
| 10 | The scanner must always emit an executable table | **KILL** | Always-a-table is theater. Replacement: 0–7 rows. Empty is valid when **quotes or geometry** fail on names that are allowed today. Empty is invalid as a response to a later-week event. |
| 11 | "Buy Put Credit" is a valid label | **KILL** | It names the wrong action. Replacement: Sell put credit / Sell call credit / Buy call debit / Buy put debit. |
| 12 | Exclude all ETFs forever | **AMEND** | Primary universe is common stock. Index Shield/hedge stays **off** until the user allows it. Do not auto-enable SPX/XSP to chase $10k. |
| 13 | Core short strike must sit outside 1-sigma | **AMEND** | Keep as Elevated default. Calm: ≥0.80-sigma and \|delta\|≤0.22 so Shield can exist. Never \|delta\| > 0.25. Do not drop these to print more rows. |
| 14 | No new Core credit inside 7 days of earnings | **AMEND** | A 7-day entry window still lets a 30–45 DTE sit through the print. Replacement: skip unless a confirmed earnings date exists and **expiry_date < earnings_date**. Unknown date = skip. Open lines that would live into the print: close or roll in front of it. |
| 15 | Take 50–70% of credit; hard-stop ~2–2.5× | **AMEND** | Default manage **60–65% / 2.0× / 2.5× ceiling** if thesis intact and DTE>14. 50% is a valid tighter take but not this book’s default. |
| 16 | Managed OTM credit win rate is ~70% | **AMEND** | 16-delta expiration POP is often cited near 70% before costs; managed 60% + 2× stop changes the ratio. Unverified for this book. Replacement: publish the journal’s 20-trade rolling stats, never a canned 70%. |
| 17 | X sentiment should be a primary entry | **KILL** | Crowds are late and promotional. Replacement: X is veto/confirm after the chain. Spike still needs a **web** source first; X only confirms the same shock. |
| 18 | X sentiment should be ignored | **AMEND** | Not ignored. Crowded vetoes Fire on that name. Name Event vetoes that ticker’s Fire. Index headlines do not veto single-name Fire. |
| 19 | Correlated AI/semi names can be sized independently | **KILL** | One cluster when the tape is pricing it that way, not a frozen ticker list. Replacement: session theme from the live print/shock (`regime-and-signals.md`); cap 30% of aggregate in `book-and-target.md`. |
| 20 | More trades per month is how you hit $10k | **KILL** | Count is not expectancy. Replacement: 4–8 closes/month. Skip is a close of zero. Adding tickets to chase $10k is how Calm books die. |
| 21 | A perfect options API will always exist in Grok cmd | **AMEND** | Grok has no magic chain. Replacement: **Schwab** via tradedesk `.env` + `SCHWAB_TOKEN_PATH` (`scripts/schwab_market.py`). Browser/web only if Schwab is down. Never invent a chain. |
| 22 | A $25k account can run the same pipeline as $150k | **KILL** | Width and lot size break 1.5–2% Shield. Replacement: under $50k, Shield 1% max, Fire off or 0.5%, one live name, no lotto. Size to **live Schwab** ($715k), then buying power. Do not invent a $410k book. |
| 23 | Fire/debit should be 20–40% of risk in a low-VIX grind | **AMEND** | Fire is half-size when cheap IV + not Crowded + not name/theme Event + not Crisis + quoted debit. Opening flow is required for an **Expert table** row. Missing flow → sleeve board, not a 43% debit next to 80% Conf Shields. VWAP unverified → no Prime. |
| 24 | Conviction tenths (74.0) add information | **KILL** | Tenths are theater. Replacement: Score 80/65/50 is not P(win). Conf is `naive POP% · book n/a` until 20 closes. Formula in `expert-table.md`. |
| 25 | X oil/geo headlines are a Spike entry | **AMEND** | Headlines without a print are rumor and must not become a row. Replacement: X **finder** for Shock watch (review lane). Spike **row** still needs web source + map + quoted debit. |
| 26 | A later-week calendar print zeros Shield/Fire until the next week | **KILL** | That is over-gating. Replacement: name/theme Event blocks that cluster until the next session after the print. Index Event skips **that date’s expiry** and **index Fire that day**. Other names trade today. |
| 27 | In Calm the book sits cash until VIX rises or a binary prints | **KILL** | Cash is valid when geometry fails. It is not the default because VIX is 15. Replacement: Calm 12% / 22-delta Shield **is** the sleeve (Score 65, size 1.0% then BP). Scanner prints it. User may skip. |
| 28 | Shield DTE is 9–45, max 3 names, “optional / not the engine” | **AMEND** | 9 DTE is not this book’s product; 45 DTE excludes the next monthly. Replacement: scan **14–60 DTE**, prefer 21–45. Max **4** Shield, **one per sector**. Score 65. Not optional-as-skip. |
| 29 | Fire requires an attached opening-flow file and a VWAP print | **AMEND** | Schwab does not always give VWAP; flow files are often absent. That combo made Fire structurally dead. Replacement: quoted debit + earnings-clear + not name/theme Event + not Crowded. Flow → 80. VWAP unverified → no Prime. |
| 30 | Over-gating is safer than a wrong row, so default skip | **KILL** | Three corrections this session were the same class. Replacement: **over-gate test** in `regime-and-signals.md` — if the name is allowed today and geometry passes, it is a row. |
| 31 | Shield means sell puts | **KILL** | First 2026-08-26 card only printed put credits. Replacement: score **all five** — sell put credit, sell call credit, iron condor, buy call debit, buy put debit. `structures` CLI. Put-only table is incomplete. |
| 32 | Mix 1-lot credit with N-lot max loss | **KILL** | Card printed Net 0.61 next to Size $2,634 (6 lots). Replacement: dollar columns are always **1-lot** max profit and max loss. Rec lots = `max(1, round(naive_POP/100 × floor(sleeve_cap / 1-lot max loss)))`, then cash/BP. N-lot dollars only in Notes. |
| 33 | Rank credits by credit/width %, prefer 21–45 DTE even when Oct pays 3× | **AMEND** | $5-wides at 13% printed $60 and hid 10-wides. Replacement was “max 1-lot dollars,” which then printed TSLA 15-wide +$56 credit / +$444 max loss. That is not a winning trade. Replacement: **credit/width first** (the edge), 1-lot credit ≥ $100 when it exists, extra width only if frac is within 1.5 pts. 21–45 DTE is a tie-break. Fire skip debit/width < 0.25. |
| 34 | A filled Expert table of 1:6 max-loss credits is a set of losing trades | **KILL** | Max-loss R/R is what you get if you hold to max loss. The plan is **60–65% / 2.0×**. Breakeven on that profile is ~75.5% wins. 20-delta naive POP is ~80%. Getting a row ≠ printing a loser, and filling a table ≠ permission to pick the worst legal wing. Replacement: print only credits that still have edge after conservative fill; harvest 85%+ first; Fire without flow stays off the table. |
| 35 | Hardcoded “core longs” never get call credits or iron condors | **KILL** | 2026-08-31: operator holds the stock and the spread; decides later to close the spread or sell shares. A ticker list is over-gating (same class as #30). Replacement: no protected-name list. Call credit / condor if geometry + earnings + Event/Crowded + quotes pass. Shares held → Notes tag, close rather than deliver unless asked. Naked shorts stay banned. |

## In force until the next 20-trade review

Bar $10k/month ($120k/year average, not a Calm floor). Account live Schwab $715k. Score all five structures; long stock does not veto a call wing or condor. No saved ticker veto list. Score ≠ P(win). Conf = quoted-delta naive POP + book win rate. Table dollars = 1 lot; Rec lots from Conf then sleeve cap then BP. Among gates, pick **credit/width then dollars** (not 15-wide scraps). Fire without flow is not an Expert row. Shield 1.0% then BP, aggregate 6–8%, 14–60 DTE, max 4 one per sector. Event = that cluster / that expiry. Do not size up to chase. Geometry stays. Manage 60–65% / 2×.
