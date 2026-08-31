# Groat evidence 2026-08-31

## Evidence — same setup, this ticker

Same ticker + same setup on cached tape. Options use hist/strikes when cached or under the daily cap. Option P&L is delta+theta clamped to defined risk, not a live exit mark. Not a system win rate. X-HOT is not tested. Does not change today's gates.

| ticker | setup | stock n | W/L/time | avg R | options n | opt P&L / risk | BE vs naive POP |
|---|---|---:|---|---:|---:|---:|---|
| **CVX** | E Emerging Sector Rotation | 6 | 2/3/1 | 0.28 | 4 | -0.20 | 25% vs 32% |
| **PLTR** | E Emerging Sector Rotation | 4 | 0/2/2 | -0.36 | 3 | -0.55 | 0% vs 36% |
| **XLE** ⚠ | E Emerging Sector Rotation | 8 | 0/5/3 | -0.32 | 0 | — | — |
| **SHOP** | E Emerging Sector Rotation | 3 | 1/2/0 | 0.00 | 0 | — | — |
| **ADBE** | E Emerging Sector Rotation | 1 | 0/1/0 | -1.00 | 0 | — | — |
| **IGV** | E Emerging Sector Rotation | 3 | 0/1/2 | -0.62 | 0 | — | — |

| ticker | analog dates (entry → exit / result / R) |
|---|---|
| **CVX** | 2026-08-07→2026-08-18 win 2.00R; 2026-07-16→2026-08-06 time 0.69R; 2026-06-10→2026-06-15 loss -1.00R; 2026-05-18→2026-05-26 loss -1.00R; 2026-01-22→2026-02-04 win 2.00R; 2025-09-17→2025-09-30 loss -1.00R |
| **PLTR** | 2026-03-25→2026-03-30 loss -1.00R; 2025-12-29→2026-01-02 loss -1.00R; 2025-10-14→2025-11-04 time 0.71R; 2025-09-22→2025-10-13 time -0.15R |
| **XLE** | 2026-08-10→2026-08-31 time 1.32R; 2026-07-17→2026-08-07 time -0.08R; 2026-06-10→2026-06-15 loss -1.00R; 2026-05-18→2026-05-27 loss -1.00R; 2026-02-05→2026-02-27 time 1.17R; 2025-12-11→2025-12-16 loss -1.00R |
| **SHOP** | 2025-12-29→2026-01-02 loss -1.00R; 2025-10-14→2025-10-28 win 2.00R; 2025-09-22→2025-09-25 loss -1.00R |
| **ADBE** | 2025-12-29→2026-01-02 loss -1.00R |
| **IGV** | 2026-06-04→2026-06-09 loss -1.00R; 2025-10-14→2025-11-04 time -0.38R; 2025-09-22→2025-10-13 time -0.49R |

HTTP this run: hist/strikes 16 · hist/earnings 0.

Empty analogs are valid. Do not invent missing chains.
