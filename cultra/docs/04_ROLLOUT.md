# Rollout Gates

Implementation begins offline. `plan`, `doctor`, unit tests, fake-transport integration tests, and static isolation checks must pass before any entitlement discovery is considered.

Entitlement discovery is a distinct run type. Its dry-run is the default; execution requires an explicit command flag, confirmed remaining quota, an expected cap no greater than 12, and an absolute cap of 15.

Historical acquisition is split into immutable checkpointed slices with an absolute cap of 90 attempts per invocation. Daily runs cannot launch backfill or backtest work.

Before a slice can be planned, the point-in-time universe, exact XNYS
calendar, and historical event/adjustment data must be prepared from preserved
independent raw source bundles. The prerequisite receipt replays normalization
and cohort selection from those raw bytes. It is zero-network and is not ORATS
authorization.

Strategy families are frozen before validation. The final chronological holdout runs once. A family may produce a manual research handoff only after it passes that holdout and every POP/EV/current-data gate. Prospective shadow begins as continuous monitoring and may revoke the family; it is not a 90-day wait gate.

Research visibility does not wait for prospective shadow. Before broad-equity
historical passage, Cultra may show exploratory present-day setups with positive
point and reliability-shrunk scenario EV in an explicitly non-ticket section.
The human board shows plain-English legs and economics; the machine artifact
keeps exact identifiers and quotes. Scenario probability, delta, and empirical
frequency are never substituted for calibrated POP.

Until all required evidence exists, the daily board remains `UNPROVEN` and
emits no qualified manual handoffs. There is no fixed shadow calendar delay.
