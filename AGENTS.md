# Trading Desk Agent Instructions

## Codex Daily V2 Default

- Treat `codexdaily`, `codex daily`, and discussions of the daily Codex pipeline as **Codex Daily V2** unless the user explicitly asks for V1, `uwos`, historical replay, or another pipeline.
- V2 is the clean `codexuw` implementation: `python3 -m codexuw.daily`.
- Default V2 live-planning command for a dated UW folder:

```bash
python3 -m codexuw.daily \
  --base-dir /Users/anuppamvi/uw_root/tradedesk/YYYY-MM-DD \
  --out-dir /Users/anuppamvi/uw_root/tradedesk/out/codexdaily_v2_YYYY-MM-DD \
  --max-tickers 0 \
  --max-candidates 0 \
  --max-final-trades 0 \
  --risk-budget 15000 \
  --monthly-profit-target 10000 \
  --max-contracts-per-trade 20 \
  --minimum-expected-value-per-dollar-risk 0.01 \
  --risk-mandate target-growth \
  --index-income-mode primary \
  --portfolio-income-mode trading-sleeve-only
```

- V2 must still use Schwab live chains/pricing and Schwab portfolio state for execution decisions. Do not force trades when live edge, liquidity, catalyst, or risk controls do not support them.
- Core investment holdings are protected by default. Do not recommend covered calls on long-term holdings unless the user explicitly allows the ticker or asks for `--portfolio-income-mode existing-core-review`.

## Groko Default

- Treat `groko`, `grok o`, and discussions of the Groko pipeline as the **Options Agent v1.84 fork** under `groko/`, not Codex Daily V2 and not live `uwos.options_agent`.
- Groko has **no Python import of `codexuw`**. Codex-named CSVs are optional leftover evidence files on disk only.
- Groko is the independent `groko` package: `python3 -m groko`.
- Default Groko live-planning command for a dated UW folder:

```bash
python3 -m groko \
  --date YYYY-MM-DD \
  --base-dir /Users/anuppamvi/uw_root/tradedesk \
  --out-dir /Users/anuppamvi/uw_root/tradedesk/out/groko/YYYY-MM-DD \
  --live-schwab \
  --live-portfolio
```

- Replay:

```bash
python3 -m groko.replay \
  --root /Users/anuppamvi/uw_root/tradedesk \
  --start YYYY-MM-DD \
  --end YYYY-MM-DD \
  --split-day YYYY-MM-DD \
  --output-dir /Users/anuppamvi/uw_root/tradedesk/out/groko_replay_YYYY-MM-DD
```

- Do not edit `uwos.options_agent` for Groko work. Do not write Groko artifacts under `codexuw_*` names.
- No trade/order placement.

## Groat Default

- Treat `groat`, `RUN FULL SCAN`, `RUN DELTA SCAN`, `ANALYZE [TICKER]`, and `REVIEW OPEN TRADES` as the **Groat** swing-trading research desk under `groat/`.
- Groat is independent: `python3 -m groat` from `/Users/anuppamvi/tradedesk/groat`. Do not import other desks as the execute path.
- Underlying thesis first, then stock vs options. Empty board is valid. No order placement.
- Default full scan:

```bash
python3 -m groat full --date YYYY-MM-DD
```

- Output: `/Users/anuppamvi/tradedesk/groat/out/groat/YYYY-MM-DD/`
- ORATS token lives in `groat/.env`. Schwab fills from `groat/.env` then tradedesk `.env`. Never print the token. Never invent ORATS, prices, X posts, or news.

## Xhigh Default

- Treat `xhigh`, `xhigh full`, and `xhigh YYYY-MM-DD` as the **xhigh** new-setup wheel/swing scanner under `xhigh/`. Not Groat, not Wheelo, not Groko.
- Independent: `python3 -m xhigh` from `/Users/anuppamvi/tradedesk/xhigh`. Do not import other desks as the execute path. Do not steal `groat` / `RUN FULL SCAN` / `wheelo` triggers.
- Schwab movers + `lastPrice` only. Catalog: CSP, put credit, call debit, call credit, put debit, iron condor. No ticket cap. CLICK only if EV > 0. No harvest. No covered calls. Empty CLICK is valid. No order placement. **v1 locked** — `xhigh/docs/LOCK.md`.
- Default:

```bash
python3 -m xhigh full --date YYYY-MM-DD
```

- Output: `/Users/anuppamvi/tradedesk/xhigh/out/xhigh/YYYY-MM-DD/`
- ORATS from `xhigh/.env` then `groat/.env` then tradedesk `.env`. Never print the token. Never invent ORATS, Schwab, X, or earnings numbers.
- CSP short put 8–15% below last. Call debit long −2% to +4% vs last. Credits are bid. Debits are long ask − short bid.

## Wheelo Default

- Treat `wheelo`, `wheelo select`, `wheelo daily`, and `wheelo YYYY-MM-DD` as the **Wheelo** CSP/CC desk under `wheelo/`, not `uwos.wheel_pipeline` and not Groat.
- Wheelo is independent: `python3 -m wheelo` from `/Users/anuppamvi/tradedesk/wheelo`. Do not import other desks as the execute path.
- Schwab quotes shortlist first. ORATS delayed `/cores` and `/strikes` run only after that shortlist (≤80 cores, ≤20 strikes, default 15 HTTP/run). Today/live always refetches; disk JSON is audit only. Empty TRADE board is valid. No order placement. User sizes cash. Ticket **conf** is 0-85 structure/research quality, not P(win); TRADE / WATCH / NO_TRADE. Lead with the rotation pick (highest-conf TRADE CSP). Do not lecture sleeve math.
- Default:

```bash
python3 -m wheelo full --date YYYY-MM-DD --capital 35000
```

- Output: `/Users/anuppamvi/tradedesk/wheelo/out/wheelo/YYYY-MM-DD/`
- ORATS token lives in `wheelo/.env` then tradedesk `.env`. Never print the token. Never invent ORATS, Schwab, or X numbers. Credits are put bid, not mid.

## grok-option Default

- Treat `grok-option`, `Anu table`, `Expert Trade Table`, `run today's scan`, `bull put`, and `sell put credit` as the **grok-option** desk under `grok-option/`, not Groat, not Groko, not Codex Daily.
- Canonical path: `/Users/anuppamvi/tradedesk/grok-option`. Skill + Schwab scripts live there. `.grok/skills/grok-option` and `~/.grok/skills/grok-option` are symlinks to that folder.
- Schwab live chain first (`python3 grok-option/scripts/schwab_market.py`). No ORATS. No order placement. Empty table is valid when quotes or geometry fail.

## Compound Core Default

- Treat `compound core`, `compoundcore`, `core sleeve`, `core calculator`, `ToothFolio`, and VOO/VGT/SMH/VB indexing-sleeve requests as **Compound Core** under `compoundcore/`. Not Groat, not Wheelo, not Groko, not Codex Daily.
- Independent: `python3 -m compoundcore` from `/Users/anuppamvi/tradedesk/compoundcore`. Do not import other desks as the execute path.
- Default weights: VOO 48 / VGT 10 / SMH 7 / VB 5 / VXUS 20 / GLDM 5 / VGSH 5. Aggressive variant is 45/15/10/5/15/5/5. Weekly DCA. No stock-picking, no options, no order placement. Empty of trading-desk tickets is the point. Core is never abandoned; SMH is band-trimmed only.
- Default:

```bash
python3 -m compoundcore 100000 --weekly 250 --monthly 1000
```

- Playbook: `/Users/anuppamvi/tradedesk/compoundcore/docs/PLAYBOOK.md`
- HTML calculator: `/Users/anuppamvi/tradedesk/compoundcore/web/calculator.html`
- Output is a dollar split plus 5-year / 10-year stress–bear–base–bull paths. Not a 40%/yr plan.

## Pattern Analysis V2 Default

- Treat `pattern analysis`, `run pattern analysis`, `pattern-analysis`, and plain `pattern` requests as **Pattern Analysis V2** unless the user explicitly asks for V1, the frozen baseline, or `uwos.options_pattern_pipeline_v1`.
- V2 is the hardened Pattern Analysis entrypoint: `python3 -m uwos.pattern_analysis_v2`.
- The compatibility module `python3 -m uwos.options_pattern_pipeline_v2` routes to the same V2 engine.
- Default V2 command for latest source-complete local UW data:

```bash
python3 -m uwos.pattern_analysis_v2 \
  --base-dir /Users/anuppamvi/uw_root/tradedesk \
  --as-of latest
```

- Default V2 dated output path is:

```text
/Users/anuppamvi/uw_root/tradedesk/out/pattern_analysis_v2/YYYY-MM-DD
```

- V2 emits ticket-first `AUTO_APPROVED`, `TRADE_REVIEW`, `AVOID`, and `NO_TRADE` decisions with decision boards, manifests, walk-forward performance, threshold sensitivity, calibration, shadow ledger, profitability audit, and runbook artifacts.
- No trade/order placement.

## Pattern Pipeline V1 Baseline

- `uwos/options_pattern_pipeline_v1_frozen_v1/` is the immutable backup copy of the Options Pattern Pipeline V1 baseline.
- Do not edit, delete, rename, reformat, regenerate, or bulk-update files under `uwos/options_pattern_pipeline_v1_frozen_v1/` unless the user explicitly asks to update the frozen V1 baseline.
- Future pattern-pipeline work should happen in `uwos/options_pattern_pipeline_v1/` or a newly named successor package such as `uwos/options_pattern_pipeline_v2/`.
- If a future change needs rollback to V1 behavior, restore from `uwos/options_pattern_pipeline_v1_frozen_v1/` instead of reconstructing the code manually.
- Before committing future pattern-pipeline changes, verify that `uwos/options_pattern_pipeline_v1_frozen_v1/` has no accidental diff.
