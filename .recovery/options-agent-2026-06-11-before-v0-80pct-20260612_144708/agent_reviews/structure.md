# structure

Constructed first-pass routed option structures from dated hot-chain bid/ask quotes and wrote `strategy_routing_audit.csv` plus `structure_attempts.csv` to audit dated and live-chain construction outcomes.

## Structured Reviews

- BRKB: supportive (medium) - dated UW EOD debit-spread quote; refresh Schwab chain before entry; preserve dated target credit/debit for the next fresh Schwab quote refresh; dated chain failed: Client error '400 Bad Request' for url 'https://api.schwabapi.com/marketdata/v1/chains?apikey=eBcIcuWWtLc9nSdyjoHuWiuMzbJASPPYpzRakIyI9jO0ciP7&symbol=BRKB&strikeCount=80&includeUnderlyingQuote=true&fromDate=2026-06-18&toDate=2026-06-18'
For more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/400; undated fallback failed: Client error '400 Bad Request' for url 'https://api.schwabapi.com/marketdata/v1/chains?apikey=eBcIcuWWtLc9nSdyjoHuWiuMzbJASPPYpzRakIyI9jO0ciP7&symbol=BRKB&strikeCount=80&includeUnderlyingQuote=true'
For more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/400
- T: supportive (medium) - live Schwab chain Bull Call validated at 0.30 debit
- ABT: avoid (high) - objective structure blocker: live_quote_width_pct_above_40pct
- DDOG: caution (high) - missing debit-spread short leg in dated UW hot-chain source; live chain expansion required; no realistic debit spread with acceptable debit/delta/liquidity
- BKNG: avoid (high) - objective structure blocker: live_quote_width_pct_above_40pct
- UPS: supportive (medium) - live Schwab chain Bear Put validated at 1.45 debit
- XLV: avoid (high) - objective structure blocker: live_quote_width_pct_above_40pct; live_leg_liquidity_below_100
- PEP: supportive (medium) - live Schwab chain Bear Call validated at 1.08 credit
- VZ: supportive (medium) - live Schwab chain Bear Put validated at 0.85 debit
- XLU: avoid (high) - objective structure blocker: live_quote_width_pct_above_40pct
- GOOG: supportive (medium) - live Schwab chain Bull Call validated at 0.63 debit
- SMH: avoid (high) - objective structure blocker: directional_bias_below_0.10
- BA: supportive (medium) - live Schwab chain Bull Call validated at 0.58 debit
- SLV: supportive (medium) - live Schwab chain Bear Put validated at 0.13 debit
- XLB: avoid (high) - objective structure blocker: signal_premium_below_1000000; live_quote_width_pct_above_40pct
- JNJ: avoid (high) - objective structure blocker: live_quote_width_pct_above_40pct
- ADBE: supportive (medium) - live Schwab chain Bear Put validated at 0.62 debit
- MU: supportive (medium) - live Schwab chain Bull Call validated at 0.91 debit
- AMAT: supportive (medium) - live Schwab chain Bull Call validated at 3.90 debit
- COP: supportive (medium) - live Schwab chain Bear Put validated at 1.58 debit
- SHOP: supportive (medium) - live Schwab chain Bull Call validated at 1.62 debit
- SNOW: supportive (medium) - live Schwab chain Bull Call validated at 1.07 debit
- VRT: supportive (medium) - live Schwab chain Bull Call validated at 5.00 debit
- XLI: avoid (high) - objective structure blocker: live_quote_width_pct_above_40pct
- MRVL: supportive (medium) - live Schwab chain Bull Call validated at 0.74 debit
- ORCL: supportive (medium) - live Schwab chain Bull Call validated at 0.61 debit
- QQQ: supportive (medium) - live Schwab chain Bull Call validated at 0.25 debit
- BX: avoid (high) - objective structure blocker: live_quote_width_pct_above_40pct
- MSFT: supportive (medium) - live Schwab chain Bear Put validated at 1.38 debit
- CVS: supportive (medium) - live Schwab chain Bull Call validated at 2.04 debit
- IWM: supportive (medium) - live Schwab chain Bull Call validated at 0.26 debit
- INTC: supportive (medium) - live Schwab chain Bull Call validated at 0.24 debit
- WDC: avoid (high) - objective structure blocker: one_lot_max_loss_above_750
- AMD: supportive (medium) - live Schwab chain Bull Call validated at 0.55 debit
- CRM: supportive (medium) - live Schwab chain Bull Call validated at 0.52 debit
- LRCX: avoid (high) - objective structure blocker: directional_bias_below_0.10; one_lot_max_loss_above_750
- DELL: avoid (high) - objective structure blocker: directional_bias_below_0.10
- SNDK: avoid (high) - objective structure blocker: one_lot_max_loss_above_750; live_leg_liquidity_below_100
- AAPL: avoid (high) - objective structure blocker: directional_bias_below_0.10
- NOW: supportive (medium) - live Schwab chain Bull Call validated at 0.22 debit
