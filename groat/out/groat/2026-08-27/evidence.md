# Groat evidence 2026-08-27

## Evidence — same setup, this ticker

Same ticker + same setup on cached tape. Options use hist/strikes when cached or under the daily cap. Option P&L is delta+theta clamped to defined risk, not a live exit mark. Not a system win rate. X-HOT is not tested. Does not change today's gates.

| ticker | setup | stock n | W/L/time | avg R | options n | opt P&L / risk | BE vs naive POP |
|---|---|---:|---|---:|---:|---:|---|
| **NOW** | E Emerging Sector Rotation | 1 | 0/0/1 | -0.59 | 0 | — | — |
| **NVDA** | C Post-Earnings Drift | 4 | 0/2/2 | -0.09 | 1 | 0.71 | 100% vs 47% |
| **PLTR** | E Emerging Sector Rotation | 4 | 0/2/2 | -0.36 | 0 | — | — |
| **SMCI** | E Emerging Sector Rotation | 1 | 0/0/1 | -0.16 | 1 | -0.09 | 0% vs 46% |
| **CRWD** | C Post-Earnings Drift | 4 | 2/2/0 | 0.50 | 0 | — | — |
| **ADBE** | E Emerging Sector Rotation | 1 | 0/1/0 | -1.00 | 0 | — | — |
| **SHOP** | E Emerging Sector Rotation | 3 | 1/2/0 | 0.00 | 0 | — | — |
| **URNM** ⚠ | E Emerging Sector Rotation | 5 | 1/3/1 | -0.23 | 0 | — | — |
| **ANET** | E Emerging Sector Rotation | 6 | 1/3/2 | 0.02 | 0 | — | — |
| **PANW** | E Emerging Sector Rotation | 5 | 1/2/2 | 0.28 | 0 | — | — |

| ticker | analog dates (entry → exit / result / R) |
|---|---|
| **NOW** | 2025-09-24→2025-10-15 time -0.59R |
| **NVDA** | 2026-06-04→2026-06-09 loss -1.00R; 2026-03-11→2026-03-20 loss -1.00R; 2025-12-04→2025-12-26 time 0.49R; 2025-09-11→2025-10-02 time 1.16R |
| **PLTR** | 2026-03-25→2026-03-30 loss -1.00R; 2025-12-29→2026-01-02 loss -1.00R; 2025-10-14→2025-11-04 time 0.71R; 2025-09-22→2025-10-13 time -0.15R |
| **SMCI** | 2025-10-10→2025-10-31 time -0.16R |
| **CRWD** | 2026-06-15→2026-07-06 win 2.00R; 2026-03-18→2026-03-24 loss -1.00R; 2025-12-11→2025-12-17 loss -1.00R; 2025-09-11→2025-09-18 win 2.00R |
| **ADBE** | 2025-12-29→2026-01-02 loss -1.00R |
| **SHOP** | 2025-12-29→2026-01-02 loss -1.00R; 2025-10-14→2025-10-28 win 2.00R; 2025-09-22→2025-09-25 loss -1.00R |
| **URNM** | 2026-04-27→2026-05-14 loss -1.00R; 2026-01-30→2026-02-05 loss -1.00R; 2026-01-07→2026-01-23 win 2.00R; 2025-12-11→2025-12-16 loss -1.00R; 2025-09-26→2025-10-17 time -0.17R |
| **ANET** | 2026-04-30→2026-05-06 loss -1.00R; 2026-04-08→2026-04-22 win 2.00R; 2026-03-05→2026-03-26 loss -1.00R; 2026-01-28→2026-02-04 loss -1.00R; 2025-10-13→2025-11-03 time 0.71R; 2025-09-19→2025-10-10 time 0.41R |
| **PANW** | 2026-06-04→2026-06-26 time 0.78R; 2026-05-12→2026-05-29 win 2.00R; 2026-03-19→2026-03-24 loss -1.00R; 2025-10-23→2025-11-13 loss -1.00R; 2025-10-01→2025-10-22 time 0.61R |

HTTP this run: hist/strikes 5 · hist/earnings 0.

Empty analogs are valid. Do not invent missing chains.
