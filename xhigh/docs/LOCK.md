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
- **Credit CLICK:** credit ≥ 10% of width and POP ≥ 70%.
- Every sleeve that passes its CLICK rule is listed. Rank small dollars-at-risk first.
- Recommendation at top of `board.md` / `recommendation.md`, wheel and swing separate.

## Known limits

- POP is delta, not a calibrated win rate. No profit promise.
- Earnings dates are often ORATS `wksNextErn` **est**.
- VIX may be blank. Intel/SEC/X is an agent overlay.
- No OI/size fill gate. Movers-only universe.
