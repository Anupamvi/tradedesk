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
| POP | `1 − \|short delta\|` (credit); `\|long\| − \|short\|` (debit); condor subtracts both shorts |
| EV debit / defined credit | `100 * [win * pop − max_loss * (1 − pop)]`; max_loss = debit or (width − credit) |
| Wheel CSP rank | annualized `(credit/strike)*(365/dte)` as percent. **Not** strike × 100. |
| Wheel naked CSP | annualized ≥ 8%, \|delta\| ≤ 0.25, and 6-month low is **not** already < 85% of strike |
| Wheel if 6m low through strike | **put credit** instead (defined-risk). CLICK if credit/width ≥ 6% and POP ≥ 70% |
| Wheel stress (display) | If last halves in 6 months, P&L vs (strike − credit) × 100. Scenario, not a forecast |
| Defined-credit CLICK | EV > 0 **or** (credit/width ≥ 18% and POP ≥ 70%) |
| Swing CLICK | defined-risk EV > 0 and conf ≥ 40 |
| Ticket cap | none |

Vintage: Schwab live last/bid/ask/delta. ORATS delayed cores only.
