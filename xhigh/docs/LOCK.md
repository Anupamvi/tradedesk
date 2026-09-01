# xhigh v1 LOCKED

Frozen 2026-08-31. Do not add features, structures, scores, or harvest. Fix only if a **geometry invariant** fails (strike not next to last) or the CLICK/SKIP recommendation is missing from `board.md`.

## Invariants (must stay red-if-broken)

- Spot = Schwab lastPrice (close after hours). Never bid/mark/ORATS px.
- CSP OTM < 8% → None. Long call ≥ 10% above last → None. Width / last > 7% → None.
- Credit = bid. Debit = long ask − short bid.
- No positions API, no harvest, no `/orders`, no ticket-count cap.
- CLICK only if EV > 0 and conf ≥ 40. Everything else legal is SKIP or WATCH.
- Recommendation block is the top of `board.md` and `recommendation.md`.

## Known limits (not bugs; do not “fix” in v1)

- POP is delta, not a calibrated win rate. EV is a ranker. No profit promise.
- Earnings dates are often ORATS `wksNextErn` **est**.
- VIX may be blank. Intel/SEC/X is an agent overlay, not a full EDGAR crawl.
- No OI/size fill gate. Movers-only universe. `conf` is coarse.
- CSP EV uses full strike cash on purpose.

## Commands

```bash
python3 -m xhigh full --date YYYY-MM-DD
python3 -m xhigh analyze TICKER --date YYYY-MM-DD
python3 -m xhigh revalidate --date YYYY-MM-DD
python3 -m xhigh intel --date YYYY-MM-DD
python3 -m xhigh xhot --date YYYY-MM-DD
```
