# structure

Constructed first-pass routed option structures from dated hot-chain bid/ask quotes and wrote `strategy_routing_audit.csv` plus `structure_attempts.csv` to audit dated and live-chain construction outcomes.

## Structured Reviews

- BRKB: supportive (medium) - dated UW EOD debit-spread quote; refresh Schwab chain before entry; live Schwab chain unavailable; keep as visible review row
- T: supportive (medium) - live Schwab chain Bull Call validated at 0.30 debit
- ABT: caution (medium) - live Schwab chain found 2.34 debit above target 2.25
- DDOG: caution (high) - missing debit-spread short leg in dated UW hot-chain source; live chain expansion required; no realistic debit spread with acceptable debit/delta/liquidity
- BKNG: avoid (high) - objective structure blocker: live_quote_width_pct_above_40pct
- UPS: supportive (medium) - live Schwab chain Bear Put validated at 1.41 debit
- XLV: avoid (high) - objective structure blocker: live_quote_width_pct_above_40pct; live_leg_liquidity_below_100
- PEP: supportive (medium) - live Schwab chain Bear Call validated at 0.99 credit
- VZ: supportive (medium) - live Schwab chain Bear Put validated at 0.64 debit
- XLU: supportive (medium) - live Schwab chain Bear Put validated at 0.21 debit
- GOOG: supportive (medium) - live Schwab chain Bull Call validated at 0.56 debit
- SMH: avoid (high) - objective structure blocker: directional_bias_below_0.10
- BA: supportive (medium) - live Schwab chain Bull Call validated at 0.65 debit
- SLV: supportive (medium) - live Schwab chain Bear Put validated at 0.15 debit
- XLB: avoid (high) - objective structure blocker: signal_premium_below_1000000; live_quote_width_pct_above_40pct
- JNJ: supportive (medium) - live Schwab chain Bear Put validated at 2.97 debit
- ADBE: supportive (medium) - live Schwab chain Bear Put validated at 0.74 debit
- MU: supportive (medium) - live Schwab chain Bull Call validated at 1.49 debit
- AMAT: supportive (medium) - live Schwab chain Bull Call validated at 3.38 debit
- COP: caution (medium) - live Schwab chain found 4.54 debit above target 4.50
- SHOP: supportive (medium) - live Schwab chain Bull Call validated at 1.68 debit
- SNOW: supportive (medium) - live Schwab chain Bull Call validated at 1.22 debit
- VRT: avoid (high) - objective structure blocker: live_quote_width_pct_above_40pct
- XLI: supportive (medium) - live Schwab chain Bull Put validated at 0.40 credit
- MRVL: supportive (medium) - live Schwab chain Bull Call validated at 0.47 debit
- ORCL: supportive (medium) - live Schwab chain Bull Call validated at 0.59 debit
- QQQ: supportive (medium) - live Schwab chain Bull Call validated at 0.31 debit
- BX: avoid (high) - objective structure blocker: debit_width_ratio_above_65pct
- MSFT: supportive (medium) - live Schwab chain Bull Call validated at 0.51 debit
- CVS: supportive (medium) - live Schwab chain Bull Call validated at 2.43 debit
- IWM: supportive (medium) - live Schwab chain Bull Call validated at 0.28 debit
- INTC: supportive (medium) - live Schwab chain Bull Call validated at 0.28 debit
- WDC: avoid (high) - objective structure blocker: one_lot_max_loss_above_750; live_quote_width_pct_above_40pct
- AMD: supportive (medium) - live Schwab chain Bull Call validated at 0.58 debit
- CRM: supportive (medium) - live Schwab chain Bull Call validated at 0.65 debit
- LRCX: avoid (high) - objective structure blocker: directional_bias_below_0.10; one_lot_max_loss_above_750
- DELL: avoid (high) - objective structure blocker: directional_bias_below_0.10
- SNDK: avoid (high) - objective structure blocker: one_lot_max_loss_above_750
- AAPL: avoid (high) - objective structure blocker: directional_bias_below_0.10
- NOW: supportive (medium) - live Schwab chain Bull Call validated at 0.22 debit
