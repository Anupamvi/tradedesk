# Acceptance Matrix

## Profit evidence

- POP is calibrated out of fold and is not option delta.
- Brier score beats the unconditional base rate and ECE is at most 0.05.
- Exact-leg train, validation, untouched-test, and shadow evidence remains separately identifiable.
- Untouched promotion requires the configured trade/cluster minimums, positive net expectancy, positive clustered lower bound, Holm control, and contribution concentration control.
- A holdout failure cannot be reset by changing a policy version in place.
- Expanding walk-forward folds are disjoint, have full 60-session embargoes, and never reach the sealed final holdout.
- A durable registry prevents holdout reuse across restarts and makes failed holdout/shadow evidence terminal.

## Manual action handoffs

- Actionable manual handoffs require untouched `HOLDOUT_PASS`; prospective
  shadow is nonblocking monitoring and can revoke a family after enablement.
- Every ticket has exact quotes, complete POP fields, complete edge fields, finite maximum loss, and one-unit economics.
- Quantity is user determined; portfolio state and arbitrary output caps are absent.
- Every fully evaluated passing candidate appears. Budget-unresolved and rejected candidates remain visible.
- Missing data, stale executable quotes, failed calibration, non-positive conservative EV, or undefined loss prevents ticket promotion.
- Same-expiration economics reproduce from exact legs; term structures require complete frozen pathwise stress evidence.
- POP metrics resolve to hashed OOF artifacts and reconcile to saved scenario outcomes; snapshot/profile/policy identities cannot be self-asserted.
- The current feature vector and complete coherent model-score calculation are
  content-addressed inside the ticket; POP is reproduced from that saved joint
  exit distribution.

## ORATS boundary

- A simulated 600-name universe has no per-symbol request path.
- The normal EOD target is 25 logical requests, the frozen logical ceiling is
  60, and the total charged-attempt ceiling is 99.
- No fixed 80-name or other arbitrary universe cap defines Core admission;
  every locally eligible symbol is planned or explicitly unresolved.
- Request sequence 100 cannot be reserved, including under process races and restart.
- Retries, redirects, recovery, split children, and status polls require separate irreversible permits.
- Identical and overlapping entity requests single-flight across workers.
- Live, one-minute, snapshot, legacy, undocumented, and unplanned endpoints fail before transport.
- Morning normally reuses the prior snapshot with zero ORATS calls.
- Non-idempotent POST work is never automatically retried; aggregate response bytes and a persisted provider circuit breaker are enforced pre-send.

## Clean-room and security

- No imports, subprocesses, symlinks, configuration references, cache references, or code references target another trading pipeline.
- Only the explicit secret bootstrap input and allowlisted Schwab token path may cross the Cultra directory boundary.
- Raw and URL-encoded token canaries are absent from logs, errors, command lines, reports, manifests, cache keys, and child environments.
- Workers have no ORATS token and cannot mint runs or arbitrary requests.
- Account, position, balance, transaction, and order paths are absent from the Schwab adapter.

## Artifact integrity

- Historical universe, session, event, and cohort manifests reproduce from
  preserved independent raw source bytes; a hand-authored digest or an
  ORATS-derived prerequisite source is rejected.
- The four universe snapshots match the exact cohort selection dates, current
  constituents cannot be projected backward, and empty earnings evidence
  cannot satisfy sampled-stock event coverage.
- Request plan, ledger, cache report, vintage manifest, evidence, tickets, rejections, data health, and run manifest agree on one run ID.
- Failed and zero-ticket runs still produce a complete artifact set.
- Raw blobs are immutable and normalized snapshots publish only after complete validation.
