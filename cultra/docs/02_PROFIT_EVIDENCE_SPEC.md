# Profit Evidence Specification

The initial and default confidence state is `UNPROVEN`.

Each strategy family advances independently through `UNPROVEN`, `RESEARCH_PASS`, `VALIDATION_PASS`, and `HOLDOUT_PASS`. A passing family may then be `MANUAL_TICKET_ENABLED`; `SHADOW_PASS` remains an optional monitoring assessment rather than a prerequisite or timer. No family may borrow evidence from another family.

Historical validation uses chronological partitions, a final untouched 20 percent test set, and a 60-session embargo. Model folds must align to the rotating cohorts' actual eligible signal windows; no-entry path suffixes cannot be mistaken for validation data. Exact historical legs and conservative executable-side pricing are mandatory. Missing legs are unavailable evidence, not reconstruction opportunities.

Development uses deterministic expanding walk-forward folds. Logistic and
isotonic calibration compete only on chronological out-of-fold validation
Brier score. Each of `POP_net`, `P_target`, `P_stop`, and `P_max_loss` has a
bucket-scoped, content-addressed artifact containing its folds, predictions,
selected frozen calibrator, base-rate comparison, ECE, 95 percent interval
provenance, development fingerprint, and pre-holdout freeze time.

The private SQLite evidence registry locks catalog, hypothesis, costs, exit
policy, POP artifacts, and exact partition membership. It consumes a final
holdout once across restarts. A failed holdout or shadow evaluation becomes a
terminal `REJECTED` record; neither parameter changes nor another process can
reuse that evidence.

Untouched-test promotion requires positive cost-adjusted expectancy in every preceding partition, at least 100 resolved test trades, 40 independent ticker/date clusters, a positive 95 percent clustered-bootstrap lower bound, Holm-adjusted significance, and no ticker or period contributing more than 20 percent of profit.

Prospective shadow runs continuously after historical enablement. It does not delay a manual handoff. A failed monitoring window revokes the family; a reportable shadow assessment uses resolved forward trades, a 90 percent clustered bound, and live-calibration diagnostics.
