# options-agent-v5 Lesson Pack Changelog

Released: 2026-06-01T15:47:49
Regression run: `/Users/anuppamvi/uw_root/tradedesk/out/options_agent_regression/smoke_goal_check_options-agent-v5_dated`
Digest: `sha256:ae2656b04bada574163db28ceeb7f7043415d041e46252d7d5c6b6f6e50c71cd`

- Promoted OA-002 `EOD_TARGET_SUPPRESSION`: EOD target candidates with valid target credit/debit are next-day planning rows; keep them yellow-visible with ready_to_enter=false until live recheck passes.
- Promoted OA-006 `JUNK_TICKER_PROMOTION`: Do not promote low-quality tail tickers into action rows without explicit strong evidence.
- Promoted OA-003 `MISSING_TARGET_PRICE_GREEN`: Missing target debit/credit must prevent green execution readiness.
- Promoted OA-009 `MULTI_AGENT_NOT_MANDATORY`: Multi-agent dispatch is mandatory for normal Options Agent runs when available.
- Promoted OA-001 `PORTFOLIO_RISK_SUPPRESSION`: Do not suppress otherwise-good trades solely due to portfolio risk; annotate the risk and size-risk it.
- Promoted OA-004 `REPORT_LEG_DETAIL_MISSING`: Reports must show plain-language buy/sell legs, expiration dates, and target debit/credit instead of relying on OCC codes.
- Promoted OA-010 `NEGATIVE_CLOSED_TRADE_OUTCOME`: Repeated or material closed-trade losses should penalize the same ticker/strategy until positive actual outcome evidence returns.
- Promoted OA-005 `ARBITRARY_TOP_N_SUPPRESSION`: Do not use arbitrary top-N cutoffs to hide candidates, coverage rows, or no-trade audits.
- Promoted OA-008 `MAJOR_TICKER_COVERAGE_MISSING`: Major liquid tickers must appear in coverage audit when source data exists.
- Promoted OA-007 `STATUS_VISIBILITY_MISSING`: Preserve clear green, yellow, blocked, and no-action status labels/icons in reports and audits.
