# Cultra V2 Rebuild Record

## Disposition of the audited V1 surfaces

| V1 component | V2 decision |
|---|---|
| ORATS gateway, attempt ledger, cache, and secret boundary | Preserved |
| Exact historical chain database and exact-leg payoff generator | Preserved as development input |
| V1 untouched holdout claim | Invalidated permanently because outcomes were exposed |
| Ten-ETF current research orders | Disabled |
| Fixed 20-name finalist default | Removed |
| Current hand-set drift simulator | Removed from the production CLI path |
| Scalar hand-authored POP input | Replaced by multivariate learned features |
| Entry-date-only confidence clustering | Replaced by connected overlapping-exposure episodes |
| Reversed long-option volatility-value ratio | Corrected to IV divided by realized volatility |
| Missing universe rows | Replaced by exactly one disposition per source symbol |

## Evidence interpretation

The 450-session, ten-ETF database is now treated entirely as development data.
The new stock-relevant V2 campaign uses cohort-aligned expanding folds: block 0
trains, block 1 tunes, block 2 validates, and block 3 is the untouched 20
percent. Only each block's first 59 sessions admit signals; its 61-session
suffix contains T+1 plus the complete maximum holding path. Each hypothesis
must beat its unconditional POP Brier score, meet the ECE limit, and retain
positive selected expectancy before the holdout is opened.

The first V2 saved-data run failed those gates for all four historically
modeled families. A model that fails remains available for reproducibility but
cannot publish POP, edge, a ranked recommendation, or a ticket.

## Current-data boundary

The saved current run reconciles 503 source symbols. It has 79 ORATS Core rows
and 20 saved exact-chain symbols. V2 evaluates those saved chains without
claiming their legacy selection was reproducible. The other Core-resolved
symbols are explicitly unresolved pending a newly authorized read-only Schwab
chain collection.

The historical domain is ten ETFs while the current chain domain is broad
equities. No V2 equity model value is publishable as a probability or expected
profit until broad exact-leg historical evidence exists and a newly frozen,
unexposed holdout is evaluated.
