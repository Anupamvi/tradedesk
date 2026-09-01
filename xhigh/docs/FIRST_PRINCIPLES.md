# xhigh first principles

Job: **every** legal ticket is listed; **CLICK** only if typical win > typical loss. Rank by EV then conf. No count cap on the catalog. Human clicks Schwab. Empty CLICK is success.

1. **Spot is Schwab `lastPrice`.** After hours, `closePrice` if last is missing or ≤0. Never mark, bid, ask, or ORATS `pxAtmIv`.
2. **Universe is movers** (or `analyze TICKER`). Never positions. No harvest. No covered calls. No OCC on the board.
3. **A strike that is not near that last is not a trade.**
   - CSP / put-credit short: **8–15% below last**.
   - Call debit long: **−2% to +4%**. Short: **+5% to +8%**.
   - Call-credit short: **+5% to +8%**.
   - Any spread/condor width ≤ **7% of last**.
4. **Catalog:** CSP, put credit, call debit, call credit, put debit, iron condor. Tape decides which are allowed. All that pass print.
5. **Fills:** credit = bid (net = short bid − long ask). Debit = long ask − short bid. No mids. No ORATS theoreticals.
6. **DTE 25–45.** Earnings date required and after expiry + 3 days. `wksNextErn` is **est**.
7. **CLICK is two sleeves.** Swing debit: defined-risk EV > 0. Wheel CSP: paid ≥ 8% annualized on cash, 8–15% OTM, |delta| ≤ 0.25. Do **not** score a CSP as crash-to-zero (strike × 100). Defined-risk credits: EV > 0 **or** credit ≥ 18% of width. Not P(win). Not fat-credit.
8. **Intel** (X, news, SEC, earnings content, macro) may KILL or cut conf. It never writes a strike or fill.
9. **No orders.**
