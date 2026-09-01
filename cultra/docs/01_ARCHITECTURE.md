# Cultra Architecture

Cultra is a clean-room, options-only research system. It has no dependency on any other trading pipeline and no broker order surface.

The runtime is divided into tokenless research workers, a read-only Schwab market-data boundary, and one ORATS gateway daemon. The gateway executes only immutable preplanned request identifiers and charges a durable SQLite permit before every physical network attempt.

The daily flow is: versioned universe -> local Schwab screen -> bounded ORATS enrichment -> exact-contract construction -> calibrated POP and edge -> evidence gate -> manual-review board. A strategy family needs a one-time untouched historical holdout pass before it can emit an actionable manual handoff. Prospective shadow then monitors and may revoke it; shadow does not impose a fixed waiting period.

Portfolio value, positions, buying power, sector limits, diversification rules, and arbitrary output caps are not inputs to ticket eligibility or quantity. All economics are normalized to one structure unit and quantity remains user determined.
