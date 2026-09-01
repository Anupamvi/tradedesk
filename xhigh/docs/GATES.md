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
| EV credit | `100 * [credit * pop − max_loss * (1 − pop)]`; CSP max_loss = strike |
| EV debit | `100 * [max_gain * pop − debit * (1 − pop)]` |
| conf floor | 40 → WATCH; cap 85 |
| Ticket cap | none. CLICK = EV > 0 and conf ≥ 40; else SKIP/WATCH |

Vintage: Schwab live last/bid/ask/delta. ORATS delayed cores only.
