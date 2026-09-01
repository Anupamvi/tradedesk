# Groat evidence 2026-08-31

## Evidence — same setup, this ticker

Same ticker + same setup on cached tape — not this ticket's P(win). Options use hist/strikes when cached or under the daily cap. Option P&L is delta+theta clamped to defined risk, not a live exit mark. Not a system win rate. X-HOT is not tested. Does not change today's gates.

| ticker | setup | stock n | W/L/time | avg R | options n | opt P&L / risk | BE vs naive POP |
|---|---|---:|---|---:|---:|---:|---|
| **XLE** ⚠ | E Emerging Sector Rotation | 8 | 0/5/3 | -0.31 | 0 | — | — |
| **SHOP** | E Emerging Sector Rotation | 3 | 1/2/0 | 0.00 | 1 | -0.52 | 0% vs 45% |
| **CVX** | E Emerging Sector Rotation | 6 | 2/3/1 | 0.28 | 4 | -0.08 | 25% vs 39% |
| **ADBE** | E Emerging Sector Rotation | 1 | 0/1/0 | -1.00 | 0 | — | — |
| **NET** ⚠ | E Emerging Sector Rotation | 6 | 1/4/1 | -0.22 | 0 | — | — |
| **SNOW** | E Emerging Sector Rotation | 4 | 2/0/2 | 1.48 | 0 | — | — |
| **ANET** | E Emerging Sector Rotation | 6 | 1/3/2 | 0.02 | 0 | — | — |

| ticker | analog dates (entry → exit / result / R) |
|---|---|
| **XLE** | 2026-08-10→2026-08-31 time 1.42R; 2026-07-17→2026-08-07 time -0.08R; 2026-06-10→2026-06-15 loss -1.00R; 2026-05-18→2026-05-27 loss -1.00R; 2026-02-05→2026-02-27 time 1.17R; 2025-12-11→2025-12-16 loss -1.00R |
| **SHOP** | 2025-12-29→2026-01-02 loss -1.00R; 2025-10-14→2025-10-28 win 2.00R; 2025-09-22→2025-09-25 loss -1.00R |
| **CVX** | 2026-08-07→2026-08-18 win 2.00R; 2026-07-16→2026-08-06 time 0.69R; 2026-06-10→2026-06-15 loss -1.00R; 2026-05-18→2026-05-26 loss -1.00R; 2026-01-22→2026-02-04 win 2.00R; 2025-09-17→2025-09-30 loss -1.00R |
| **ADBE** | 2025-12-29→2026-01-02 loss -1.00R |
| **NET** | 2026-08-17→2026-08-25 loss -1.00R; 2026-07-24→2026-08-07 win 2.00R; 2026-06-11→2026-07-06 time 0.68R; 2026-05-07→2026-05-08 loss -1.00R; 2026-03-26→2026-04-09 loss -1.00R; 2025-10-02→2025-10-16 loss -1.00R |
| **SNOW** | 2026-07-24→2026-08-04 win 2.00R; 2026-06-11→2026-07-06 time 0.75R; 2025-10-14→2025-11-03 win 2.00R; 2025-09-22→2025-10-13 time 1.16R |
| **ANET** | 2026-04-30→2026-05-06 loss -1.00R; 2026-04-08→2026-04-22 win 2.00R; 2026-03-05→2026-03-26 loss -1.00R; 2026-01-28→2026-02-04 loss -1.00R; 2025-10-13→2025-11-03 time 0.71R; 2025-09-19→2025-10-10 time 0.41R |

HTTP this run: hist/strikes 0 · hist/earnings 0.

Empty analogs are valid. Do not invent missing chains.
