# Options Agent Lessons

- OA-002 [hard] EOD target candidates with valid target credit/debit are next-day planning rows; keep them yellow-visible with ready_to_enter=false until live recheck passes. Applies to: decision_board, regression_gate, report_contract, subagent_prompt, synthesis_scoring.
- OA-006 [hard] Do not promote low-quality tail tickers into action rows without explicit strong evidence. Applies to: decision_board, regression_gate, subagent_prompt, synthesis_scoring.
- OA-003 [hard] Missing target debit/credit must prevent green execution readiness. Applies to: decision_board, regression_gate, report_contract, subagent_prompt, synthesis_scoring.
- OA-009 [hard] Multi-agent dispatch is mandatory for normal Options Agent runs when available. Applies to: regression_gate, subagent_prompt.
- OA-001 [hard] Do not suppress otherwise-good trades solely due to portfolio risk; annotate the risk and size-risk it. Applies to: decision_board, regression_gate, report_contract, subagent_prompt.
- OA-004 [hard] Reports must show plain-language buy/sell legs, expiration dates, and target debit/credit instead of relying on OCC codes. Applies to: regression_gate, report_contract, subagent_prompt.
- OA-010 [medium] Repeated or material closed-trade losses should penalize the same ticker/strategy until positive actual outcome evidence returns. Applies to: regression_gate, subagent_prompt, synthesis_scoring.
- OA-005 [medium] Do not use arbitrary top-N cutoffs to hide candidates, coverage rows, or no-trade audits. Applies to: coverage_audit, regression_gate, subagent_prompt.
- OA-008 [medium] Major liquid tickers must appear in coverage audit when source data exists. Applies to: coverage_audit, regression_gate, subagent_prompt.
- OA-007 [medium] Preserve clear green, yellow, blocked, and no-action status labels/icons in reports and audits. Applies to: coverage_audit, regression_gate, report_contract.
