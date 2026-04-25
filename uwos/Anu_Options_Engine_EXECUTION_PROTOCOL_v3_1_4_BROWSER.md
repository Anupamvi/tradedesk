# Anu Options Engine — BROWSER EXECUTION ADDENDUM (Audited FIRE + SHIELD No-GEX Payload)

**Canonical upload filename:** `Anu_Options_Engine_EXECUTION_PROTOCOL_v3_1_4_BROWSER.md`  
**Effective logic version:** 3.2.7-full-source-routing-audit  
**Date:** 2026-04-25
**Revision:** r6 mixed-flow rescue + scan-date OI default

## Browser execution policy

Browser inputs are optional under this payload and are never required for automated scanning.

The browser layer must not:

- call Atlas / browser GEX
- infer GEX walls or gamma flips from page text
- suppress the primary table because browser context is missing
- create or deny a SHIELD anchor from browser-only context
- fabricate an iron condor from browser-only information

## Browser inheritance

Browser scans inherit the same audited execution rules as file-only scans:

- dedupe repeated seeds
- block minority flow
- use actual hot-chain legs
- keep EV/ML independent from Conviction
- emit the base table
- use Bootstrap when Health Gate is unresolved
- support file-native SHIELD credit spreads
- support file-native paired SHIELD iron condors


## v3.2.4 browser execution additions

The browser layer must not override the OI overlay, Health Gate, or size-publication rules. Browser context remains non-operative for GEX and cannot promote a `Size = None` row to BUY/SELL.

## Executable sync inheritance

Browser execution inherits the canonical executable sync gate. It may not use browser context or sidecar analysis code to bypass a stale root `anu_analysis_v3_1_7.py`.


## v3.2.6 browser execution additions

Browser execution inherits family-flex only from file-native source rows and actual hot-chain legs. Browser text, GEX, pages, or manual observations may not create a derived seed, satisfy SHIELD anchoring, or override EV/ML-first ranking.

Browser execution also inherits the catalyst-watch requirement for material earnings-blocked names and the rule that WATCH rows retain `Execution = Watch` even when Health Gate is PASS.


## v3.2.7 browser inheritance

Browser/Atlas context remains non-operative for automated ranking, gating, sizing, GEX, SHIELD anchors, and condor creation. Browser mode inherits full-source streaming, Top-Symbol Gap reporting, liquidity-tier routing, ETF/index lane separation, blocked-positive-EV output, alternates output, low-POP labels, and hard executable-sync blocking.


## r6 browser execution inheritance

Browser execution inherits the file-only Mixed-Flow Rescue gate, but browser context may not create or satisfy it. The row must still come from file-native flow, actual hot-chain construction, positive EV/ML, executable size, and no hard blocks. Browser execution also inherits scan-date OI by default, explicit next-day overlay only by request, raw bot full-source schema inference, and ETF/index lane WATCH labeling.
