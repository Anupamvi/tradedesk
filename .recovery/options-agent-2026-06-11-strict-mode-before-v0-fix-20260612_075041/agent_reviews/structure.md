# structure

Constructed first-pass routed option structures from dated hot-chain bid/ask quotes and wrote `strategy_routing_audit.csv` plus `structure_attempts.csv` to audit dated and live-chain construction outcomes.

## Structured Reviews

- BRKB: supportive (medium) - dated UW EOD debit-spread quote; refresh Schwab chain before entry; live Schwab chain unavailable; keep as visible review row
- T: supportive (medium) - live Schwab chain Bull Call validated at 0.30 debit
- ABT: avoid (high) - objective structure blocker: live_quote_width_pct_above_40pct
- DDOG: caution (high) - missing debit-spread short leg in dated UW hot-chain source; live chain expansion required; no realistic debit spread with acceptable debit/delta/liquidity
- BKNG: avoid (high) - objective structure blocker: live_quote_width_pct_above_40pct
- UPS: supportive (medium) - live Schwab chain Bear Put validated at 1.46 debit
- XLV: avoid (high) - objective structure blocker: live_quote_width_pct_above_40pct; live_leg_liquidity_below_100
- PEP: supportive (medium) - live Schwab chain Bear Call validated at 0.97 credit
- VZ: supportive (medium) - live Schwab chain Bear Put validated at 0.66 debit
- XLU: supportive (medium) - live Schwab chain Bear Put validated at 0.19 debit
- GOOG: supportive (medium) - live Schwab chain Bull Call validated at 0.59 debit
- SMH: avoid (high) - objective structure blocker: directional_bias_below_0.10
- BA: supportive (medium) - live Schwab chain Bull Call validated at 0.54 debit
- SLV: supportive (medium) - live Schwab chain Bear Put validated at 0.15 debit
- XLB: avoid (high) - objective structure blocker: signal_premium_below_1000000
- JNJ: supportive (medium) - live Schwab chain Bear Put validated at 3.36 debit
- ADBE: supportive (medium) - live Schwab chain Bear Put validated at 0.70 debit
- MU: avoid (high) - objective structure blocker: directional_bias_below_0.10
- AMAT: supportive (medium) - live Schwab chain Bull Call validated at 3.27 debit
- COP: supportive (medium) - live Schwab chain Bear Put validated at 2.70 debit
- SHOP: supportive (medium) - live Schwab chain Bull Call validated at 1.79 debit
- SNOW: supportive (medium) - live Schwab chain Bull Call validated at 1.23 debit
- VRT: avoid (high) - objective structure blocker: live_quote_width_pct_above_40pct
- XLI: supportive (medium) - live Schwab chain Bull Put validated at 0.47 credit
- MRVL: avoid (high) - objective structure blocker: directional_bias_below_0.10
- ORCL: avoid (high) - objective structure blocker: directional_bias_below_0.10
- QQQ: avoid (high) - objective structure blocker: directional_bias_below_0.10
- BX: supportive (medium) - live Schwab chain Bull Call validated at 1.94 debit
- MSFT: avoid (high) - objective structure blocker: directional_bias_below_0.10
- CVS: supportive (medium) - live Schwab chain Bull Call validated at 2.20 debit
- IWM: avoid (high) - objective structure blocker: directional_bias_below_0.10
- INTC: avoid (high) - objective structure blocker: directional_bias_below_0.10
- WDC: avoid (high) - objective structure blocker: one_lot_max_loss_above_750; live_quote_width_pct_above_40pct
- AMD: avoid (high) - objective structure blocker: directional_bias_below_0.10
- CRM: avoid (high) - objective structure blocker: directional_bias_below_0.10
- LRCX: avoid (high) - objective structure blocker: directional_bias_below_0.10; one_lot_max_loss_above_750
- DELL: avoid (high) - objective structure blocker: directional_bias_below_0.10
- SNDK: avoid (high) - objective structure blocker: directional_bias_below_0.10; one_lot_max_loss_above_750
- AAPL: avoid (high) - objective structure blocker: directional_bias_below_0.10
- NOW: avoid (high) - objective structure blocker: directional_bias_below_0.10
