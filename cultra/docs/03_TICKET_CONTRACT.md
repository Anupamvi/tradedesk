# Manual Ticket Contract

A ticket is a normalized one-unit research handoff, never an order.

Every ticket must contain exact legs, actions and ratios; live Schwab bid/ask and timestamps; a proposed limit; finite maximum loss; calibrated `POP_net`, `P_target`, `P_stop`, and `P_max_loss`; net and conservative expected value; return on maximum loss; breakevens; expected shortfall; stress loss; untouched-holdout evidence and any available shadow monitoring; entry, target, stop, time exit, invalidation, assignment/exercise handling, and an ORATS snapshot identifier.

For one-expiration structures, Cultra derives natural executable price,
maximum loss, maximum profit, and breakevens directly from exact legs, Schwab
bid/ask sides, multiplier, and complete costs; caller-supplied values must
reproduce. A multi-expiration ticket instead requires a frozen pathwise
artifact covering assignment, early exercise, dividends, volatility shock,
liquidity collapse, adverse gaps, and partial fills.

POP target labels must reconcile to the saved point scenario distribution,
and every probability is bound to its hashed OOF artifact, regime bucket, 95
percent interval method, and calibration period. The ORATS snapshot ID,
provider trade date, schema, and entitlement-verified field profile must
resolve together. The ticket exit-policy version must equal the version frozen
with family evidence.

The machine ticket also stores the exact current model feature vector,
selection-model point/conservative returns, coherent five-category exit
distribution, category return profiles, projection distance, model artifact
identity, and a content hash of that calculation. Ticket POP must reproduce
from the stored joint distribution; the model and cost identities must match
the holdout registry evidence.

`Quantity` is always `USER DETERMINED`. Raw delta may be shown only as a separate heuristic and may never be labeled POP.

Candidates that lack mandatory data, fail evidence, or cannot be evaluated within the request budget remain visible in separate non-ticket sections.

A current research order may therefore show exact Schwab legs and complete edge
economics before ticket eligibility. It must not call an empirical historical
frequency `POP_net`; if calibration fails, `POP_net` is explicitly unavailable,
the calibration diagnostics remain attached, and manual-ticket enablement stays
false. Prospective shadow never delays a holdout-qualified manual handoff, but
a subsequent failed monitoring window revokes the family.
