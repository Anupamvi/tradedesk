# Master-prompt requirements matrix

This matrix maps the 45 numbered sections of the supplied swing-research specification to CORAT. Status meanings:

- **Implemented**: present in the normal deterministic path and covered by tests or run artifacts.
- **Guarded**: present as an explicit safety/evidence boundary; unavailable inputs fail closed.
- **Partial**: useful implementation exists, but the full requested breadth is not claimed.
- **External evidence**: a connector/analyst must supply evidence unavailable to the local public-news adapter, such as authenticated X observations.

| # | Requirement | Status | CORAT implementation / boundary |
|---:|---|---|---|
| 1 | Broad universe and theme discovery | Partial | Normal full scans discover from the complete ORATS core universe, preserve configured benchmarks/theme ETFs, and select up to 500 equities across market-cap, stock-volume, and option-volume ranks. This is broad current-universe discovery, not a survivorship-free historical constituent database. |
| 2 | Underlying first | Implemented | `setups.py`, `scoring.py`, and `pipeline.py` build direction, setup, trigger, invalidation, target, and stock plan before fetching/choosing an option. |
| 3 | Market regime | Partial | `regime.py` classifies risk-on, rotation, chop, risk-off, liquidation, and recovery using SPY/QQQ/IWM/DIA, internal breadth, and VIX/TLT/UUP/HYG/GLD traded proxies. It does not claim a full rates/factor terminal. |
| 4 | Sector rotation | Partial | 5/20/60 relative strength, SPY-relative performance, price trend, volume, and maturity states are implemented. Revisions/fund-flow/institutional evidence is not silently inferred. |
| 5 | Anchored VWAP | Implemented | `technical.py` uses event-derived anchors including year start, latest earnings, swing low, high-volume day, material gap, and breakout. No arbitrary cursor-chosen anchor. |
| 6 | Technical structure | Implemented | Price, EMA20, SMA50/200, AVWAP, volume, relative volume, ATR, support/resistance, gaps, highs/lows, relative strength, trend, and RSI-as-context. |
| 7 | Seven primary setups | Implemented within available history | All seven are detected in current scans. Post-earnings drift uses dated ORATS earnings events, the earnings gap, earnings AVWAP behavior, and the final trigger; other setups use frozen price/sector/volume/IV similarity dimensions. |
| 8 | ORATS volatility analysis | Implemented | `volatility.py` normalizes ATM IV, rank/percentile, HV, ex-earnings IV, forecast vol/IV, ratios, term structure, skew, implied/historical earnings move, confidence, OI, and volume when returned by ORATS. |
| 9 | Greeks | Implemented | Exact structure delta/gamma/theta/vega, theta holding cost, and trade-specific explanations are in the report. |
| 10 | 21–75 DTE | Implemented | Configured hard window with holding-period-aware target DTE and comparison across returned expirations. |
| 11 | Earnings rule | Guarded | Ordinary options are rejected if the hold crosses earnings. An exact date is used when present; an ORATS weeks-to-next estimate safely beyond the hold avoids a random exact-date veto. Earnings-event strategies are not promoted. |
| 12 | Other event risk | Guarded | Sourced macro/ticker events inside the planned hold are disclosed. Absence remains `DATA UNAVAILABLE`; events are not mechanically rejected. |
| 13 | Option liquidity | Implemented | Per-leg OI, volume, bid/ask, and relative spread are shown. A coherent two-sided market is required, and observed entry/exit friction plus commissions are charged directly to modeled P/L instead of relying on arbitrary OI/volume cutoffs. |
| 14 | Structure selection | Implemented within normal structure scope | Every returned 21–75 DTE expiration is searched for long calls/puts, bull-call/bear-put debit spreads, and defined-risk bull-put/bear-call credit spreads. The exact contract is selected on older training paths and measured on untouched recent paths. Calendars, diagonals, and event-volatility structures remain outside normal promotion. |
| 15 | Stock vs options | Implemented | Every enriched idea compares path-aware expected return on stock capital required with exact-option expected return on defined maximum loss and explains the selected vehicle. Earnings applicability is waived for funds, not companies. |
| 16 | Option positioning | Partial | Major call/put OI concentrations and expiries are reported. Dealer GEX direction is explicitly unavailable rather than inferred. |
| 17 | Options flow | External evidence | ORATS aggregate volume/OI is labeled ambiguous. Sourced flow rows can be ingested, but opening/closing, spread-leg, or hedge intent is never invented. |
| 18 | X intelligence | External evidence | Dated X observations and window/acceleration fields are supported through context input. The local Python process does not claim an embedded X search entitlement. |
| 19 | X anti-pump | Guarded | Credibility and spam flags downweight X; rumor/promotion cannot receive high-credibility catalyst treatment or create economic edge. |
| 20 | Internet catalyst engine | Partial | `full-scan` automatically researches leaders through dated public-news RSS metadata with direct links and a strict headline-only direction classifier. Primary/analyst evidence can be merged through the same context schema. |
| 21 | Catalyst decay | Implemented | Publication-date freshness becomes new, developing, under-appreciated, fully priced, or stale; stale facts score zero. |
| 22 | Risk/reward gate | Implemented | Entry zone, thesis, invalidation, stop/max loss, targets, horizon, and liquidity are mandatory. R:R is displayed as a preference; realized path-aware expected profit is the selection metric, so an arbitrary ratio cannot veto a positive trade. |
| 23 | Position sizing/correlation | Partial | Stock units derive from invalidation risk; option units derive from exact maximum loss; NAV risk defaults and sector/direction caps exist. Full factor/theme/vol aggregation needs portfolio exposures not presently supplied. |
| 24 | No averaging down | Implemented | Ledger accepts scaling only when predefined at `record-plan`; open-trade review never invents a later add rule. |
| 25 | Setup vs trigger | Implemented | Stored separately in `SetupSignal`; only a trigger present in completed as-of data becomes a target trade. Market closure is never treated as a missing trigger. |
| 26 | Do not chase | Implemented | Extension from EMA/AVWAP/breakout via ATR prevents severely extended breakout entries and keeps them waiting. |
| 27 | Historical validation | Partial | Leakage-safe non-overlapping analogues include intraday adverse/favorable paths, sector confirmation, event AVWAP, IV/HV similarity, and dated post-earnings matching. Normal option candidates use train-only contract selection and a recent held-out evaluation. The frozen `full-replay` harness now walks the complete daily path, carries exact selected option legs into next-session entry and exact exits, and separates train/validation/test with boundary embargoes. It is built but no profitable validation is claimed until an executed artifact exists; dated security-master coverage is still not guaranteed survivorship-free. |
| 28 | Expectancy over win rate | Implemented | Exact plans report modeled POP, expected dollars, sample size, average outcomes, profit factor, and payoff asymmetry. A sub-50% POP can still rank when the measured payoff produces positive expectancy. |
| 29 | Rule discovery lifecycle | Partial | `backtest` and `option-replay` retain narrow diagnostics. `full-replay` adds frozen train/validation/test evidence, uncertainty, POP calibration, vehicle/setup/strategy attribution, and an immutable policy hash. It remains plan-only without explicit execution and always leaves production promotion false pending an unchanged prospective shadow period. |
| 30 | Hard vs soft rules | Implemented | Coherent execution, defined loss, earnings crossing, missing essential price data, and nonpositive modeled profit are hard. Catalyst, RSI, X, flow, AVWAP strength, IV, sector, and the 0–100 score are context rather than mandatory gates. |
| 31 | 0–100 score | Implemented | Prompt weights are explicit components. Score ranks context only and never authorizes or vetoes a trade. Target status comes from trigger plus positive modeled expected profit. |
| 32 | Confidence not probability | Implemented | Historical/model POP is calculated only from the displayed same-setup sample and labeled CORAT model output, never ORATS POP. ORATS confidence remains separately labeled. |
| 33 | No trade is valid | Implemented | Report leads with target trades when positive expectancy exists and reports no trade when expected profit is unavailable or nonpositive; market hours and missing optional context cannot manufacture a zero. |
| 34 | Few-trade diagnostic | Implemented | The scan funnel reconciles every scanned name, triggered setup, historical sample, option-chain request, exact structure count, positive option alternative, vehicle choice, and final target. `candidate-audit.json` preserves every disposition; strongest non-trades appear when fewer than three targets qualify. |
| 35 | Top trade board | Implemented | Immutable Markdown research board contains the required underlying, trade, option, positioning, X, history, risk, and source fields for up to 10 ideas. |
| 36 | Ranking/status table | Implemented | Ranked table separates `TARGET TRADE`, `SETUP ONLY — NOT A TRADE`, `NO TRADE — EDGE NOT POSITIVE`, `WATCHLIST`, and `REJECTED / AVOID`. |
| 37 | Best vehicle | Implemented within structure scope | Every candidate explains stock versus the best exact long-option, debit-spread, or defined-risk credit-spread candidate after costs. Non-promoted calendar/diagonal/event structures are not implied. |
| 38 | Source traceability | Implemented | ORATS/Schwab endpoint, fetch/data timestamps, row counts, cache path/hash, context file hash, and mismatched timestamp warning. |
| 39 | Prohibited behavior | Guarded | Boundary, secret-redaction, no-lookahead, conservative-fill, no-order, and missing-data tests enforce the critical prohibitions. |
| 40 | Learning loop | Partial | Append-only recommendation/execution/outcome events and 5/10/20/final review fields exist. Statistical learning requires a real accumulated sample and never changes rules automatically. |
| 41 | Primary hierarchy | Implemented | Pipeline order is regime → underlying/setup → structure/AVWAP → context → R:R/history → option → flow/X. |
| 42 | Run full scan | Implemented | `python3 -m corat full-scan --date DATE` performs discovery, automatic leader research, context persistence, and final reranking in one command. |
| 43 | Run delta scan | Implemented | Immutable run comparison reports changed status, score, trigger, price, volatility, catalyst, and invalidation fields. |
| 44 | Analyze ticker | Implemented | `python3 -m corat analyze TICKER --date DATE` runs the complete single-name comparison. |
| 45 | Review open trades | Implemented | Ledger review uses the original thesis/stop and returns hold/add/reduce/take-profit/exit. `ADD` requires a pre-entry zone and quantity cap, known current quantity, aligned thesis, and a current `TARGET TRADE` result. |

## Promotion boundary

Passing tests or completing a dated run validates software behavior, not strategy profitability. A strategy becomes promotable only after adequate leakage-safe train/test evidence, exact-chain replay where options are involved, and prospective paper/observational evidence. CORAT currently creates and preserves those evidence artifacts; it does not claim the promotion has occurred.
