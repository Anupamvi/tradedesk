# skeptic

Objective blockers remain blockers. Portfolio-only blockers are converted to annotations.

## Structured Reviews

- BRKB: caution (medium) - dated UW EOD debit-spread quote; refresh Schwab chain before entry; preserve dated target credit/debit for the next fresh Schwab quote refresh; dated chain failed: Client error '400 Bad Request' for url 'https://api.schwabapi.com/marketdata/v1/chains?apikey=eBcIcuWWtLc9nSdyjoHuWiuMzbJASPPYpzRakIyI9jO0ciP7&symbol=BRKB&strikeCount=80&includeUnderlyingQuote=true&fromDate=2026-06-18&toDate=2026-06-18'
For more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/400; undated fallback failed: Client error '400 Bad Request' for url 'https://api.schwabapi.com/marketdata/v1/chains?apikey=eBcIcuWWtLc9nSdyjoHuWiuMzbJASPPYpzRakIyI9jO0ciP7&symbol=BRKB&strikeCount=80&includeUnderlyingQuote=true'
For more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/400
- T: supportive (medium) - no objective blocker after structure review
- ABT: avoid (medium) - objective blocker remains: live_quote_width_pct_above_40pct
- DDOG: caution (medium) - missing debit-spread short leg in dated UW hot-chain source; live chain expansion required; no realistic debit spread with acceptable debit/delta/liquidity
- BKNG: avoid (medium) - objective blocker remains: live_quote_width_pct_above_40pct
- UPS: supportive (medium) - no objective blocker after structure review
- XLV: avoid (medium) - objective blocker remains: live_quote_width_pct_above_40pct; live_leg_liquidity_below_100
- PEP: supportive (medium) - no objective blocker after structure review
- VZ: supportive (medium) - no objective blocker after structure review
- XLU: avoid (medium) - objective blocker remains: live_quote_width_pct_above_40pct
- GOOG: supportive (medium) - no objective blocker after structure review
- SMH: avoid (medium) - objective blocker remains: directional_bias_below_0.10
- BA: supportive (medium) - no objective blocker after structure review
- SLV: supportive (medium) - no objective blocker after structure review
- XLB: avoid (medium) - objective blocker remains: signal_premium_below_1000000; live_quote_width_pct_above_40pct
- JNJ: avoid (medium) - objective blocker remains: live_quote_width_pct_above_40pct
- ADBE: supportive (medium) - no objective blocker after structure review
- MU: supportive (medium) - no objective blocker after structure review
- AMAT: supportive (medium) - no objective blocker after structure review
- COP: supportive (medium) - no objective blocker after structure review
- SHOP: supportive (medium) - no objective blocker after structure review
- SNOW: supportive (medium) - no objective blocker after structure review
- VRT: supportive (medium) - no objective blocker after structure review
- XLI: avoid (medium) - objective blocker remains: live_quote_width_pct_above_40pct
- MRVL: supportive (medium) - no objective blocker after structure review
- ORCL: supportive (medium) - no objective blocker after structure review
- QQQ: supportive (medium) - no objective blocker after structure review
- BX: avoid (medium) - objective blocker remains: live_quote_width_pct_above_40pct
- MSFT: supportive (medium) - no objective blocker after structure review
- CVS: supportive (medium) - no objective blocker after structure review
- IWM: supportive (medium) - no objective blocker after structure review
- INTC: supportive (medium) - no objective blocker after structure review
- WDC: avoid (medium) - objective blocker remains: one_lot_max_loss_above_750
- AMD: supportive (medium) - no objective blocker after structure review
- CRM: supportive (medium) - no objective blocker after structure review
- LRCX: avoid (medium) - objective blocker remains: directional_bias_below_0.10; one_lot_max_loss_above_750
- DELL: avoid (medium) - objective blocker remains: directional_bias_below_0.10
- SNDK: avoid (medium) - objective blocker remains: one_lot_max_loss_above_750; live_leg_liquidity_below_100
- AAPL: avoid (medium) - objective blocker remains: directional_bias_below_0.10
- NOW: supportive (medium) - no objective blocker after structure review
