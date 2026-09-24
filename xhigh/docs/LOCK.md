# xhigh v1.1

Geometry frozen. **Scoring reopened 2026-08-31** so the wheel can CLICK without treating assignment as crash-to-zero.

Do not add harvest, CCs, new structures, or loosen strike bands. Fix only geometry invariants or CLICK/SKIP recommendation.

## Invariants

- Spot = Schwab lastPrice (close after hours). Never bid/mark/ORATS px.
- CSP OTM < 8% → None. Long call ≥ 10% above last → None. Width / last > 7% → None.
- Credit = bid. Debit = long ask − short bid.
- No positions API, no harvest, no `/orders`, no ticket-count cap.
- **Swing CLICK:** long at/ITM (|delta| ≥ 0.50), DTE ≥ 35, R/R ≥ 1.5, no ex-div before expiry. Put-debit max not below the 6-month low. Positive EV is not enough.
- **Wheel CLICK:** CSP annualized ≥ 8% on cash, |delta| ≤ 0.25, 8–15% OTM, 6-month low not already through the strike. Show 50%-off 6-month P&L in dollars. Not a growth forecast.
- **Credit CLICK:** credit ≥ 10% of width, POP ≥ 70%, one lot max-loss ≤ $500, and bid-ask ≤ 15% of the credit. N=0 or a wide market is SKIP.
- Every sleeve that passes its CLICK rule is listed. Rank CLICK by PD desc (nulls last). Keep conf.
- Recommendation at top of `board.md` / `recommendation.md`, wheel and swing separate.

## Known limits

- POP is delta, not a calibrated win rate. No profit promise.
- Earnings dates are often ORATS `wksNextErn` **est**.
- VIX may be blank. Intel/SEC/X is an agent overlay. Intel `kill` stamps **WATCH** and must survive restamp.
- Movers-only universe. Empty movers is **DATA UNAVAILABLE**, not empty CLICK. A credit whose one-lot loss exceeds $500, or whose bid-ask is wider than 15% of the credit, is SKIP.
- PD still ranks after the close. Option quote age > 120s is tagged `pd_stale` / `stale_quote`; the number is kept. DTE is `expiry − session asof`, not Schwab's calendar-today count.
