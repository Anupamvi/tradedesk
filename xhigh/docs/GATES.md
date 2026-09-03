# xhigh gates

Units: ORATS vol fields are **percent**. Schwab last is dollars. Delta is Schwab chain delta.

| Gate | Value |
|---|---|
| CSP OTM | 8–15% below last |
| Put-credit short | 8–15% below last; width ≤ 7% of last |
| Call-debit long / short | −2% to +4% / +5% to +8%; width ≤ 7% |
| Call-credit short | +5% to +8%; width ≤ 7% |
| Put-debit long | −4% to +2% vs last; width ≤ 7% |
| Iron condor | put-credit + call-credit, same expiry |
| DTE | 25–45 |
| Earnings | expiry < earn − 3d; missing → no option |
| Cheap vol (skip short premium) | ivHvXernRatio < 0.90 **and** ivPctile1y < 40; ratio > 5 ignored |
| POP | `1 − \|short delta\|` (credit); `\|long delta\|` (debit); condor `1 − \|put short\| − \|call short\|` |
| EV debit / defined credit | Display only. `100 * [win * pop − max_loss * (1 − pop)]`. **Not** the CLICK rule. |
| Wheel CSP rank | annualized `(credit/strike)*(365/dte)` as percent. **Not** strike × 100. |
| Wheel naked CSP | annualized ≥ 8%, \|delta\| ≤ 0.25, and 6-month low is **not** already < 85% of strike |
| Defined-credit CLICK | credit ≥ **10% of width** and POP ≥ 70%. Put credit SKIP if a known ex-div sits before expiry (stock drop attacks the short put). 8–15% OTM is naturally ~1:7; 1:14 still SKIP. |
| Swing debit CLICK | long at/ITM (\|delta\| ≥ **0.50**), DTE ≥ **35**, R/R ≥ 1.5, no ex-div before expiry. Put-debit max must not sit below the 6-month low. 25-DTE 0.35-delta is SKIP (KO). Positive EV is not enough. |
| Board | Every sleeve that passes its CLICK rule is listed. Rank small dollars-at-risk first. |
| Wheel stress (display) | If last halves in 6 months, P&L vs (strike − credit) × 100. Scenario, not a forecast |
| Ticket cap | none |

Vintage: Schwab live last/bid/ask/delta. ORATS delayed cores only.
