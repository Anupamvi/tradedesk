# Cultra

Cultra is a clean-room, manual-only options research pipeline. It keeps four
things separate: a market pattern, an exact trade candidate, validated profit
evidence, and a current executable manual ticket. A candidate stays visible
when evidence fails, but it cannot become an entry instruction.

Current profit-confidence state: **`UNPROVEN`**. The legacy saved development
store is ten-ETF data, all saved POP models fail at least one calibration gate,
and the old holdout was exposed. The new public-source work has identified a
33-stock/7-ETF research cohort, but it has not produced complete event history,
exact option history, a valid POP model, or a holdout result. Neither dataset is
production validation.

## V0.7 offline correction

- The daily universe is an explicit broad stock-and-ETF manifest. There is no
  SPY-derived universe, fixed 20-name list, final top-N cap, NAV rule, or
  one-trade-per-ticker suppression of otherwise eligible tickets.
- Every admitted exact candidate remains visible. Failed or absent POP/EV is a
  source-level action blocker, not a reason to erase the row or relabel a raw
  classifier score as POP.
- A manual ticket requires `HOLDOUT_PASS`, four calibrated probabilities with
  intervals and provenance, positive point and conservative net EV, finite
  maximum loss, fresh Schwab quotes, exact legs, and a point-in-time event
  clearance covering the actual market-session holding window.
- Each ticket preserves the exact current feature vector, coherent joint-exit
  probabilities, category returns, model artifact identity, and calculation
  hash that generated its POP/edge. A displayed number is not accepted as its
  own evidence.
- Human tickets are action-first and color-coded. They show the exact structure
  in plain language, current leg markets, limit, POP interval/model/sample,
  edge, risk, evidence windows, event dates, exits, invalidation, and review
  date. OCC identifiers remain in the machine audit artifact.
- Signal time is session T after the close. Historical entry is exactly the
  next frozen market session, T+1. Same-session and skipped-session records
  fail closed.
- The finite hypothesis registry contains 30 structures at 20/40/60-session
  horizons: 90 hypotheses for Holm correction. The four-family ETF V1 cannot
  satisfy this registry.
- Point-in-time cohort selection requires an exact-date universe snapshot,
  rejects unknown/outcome fields, uses no future membership, requires broad
  stock-relevant coverage, and freezes 40 disjoint names across four blocks.

## Historical request estimate

The cold-cache base campaign is **474 ORATS attempts**, not 7,000+ and not a
per-run cost:

| Historical input | Attempts |
|---|---:|
| Full-history Core for 40 sampled names, two/request | 20 |
| One ten-name cohort chain for each of 450 sessions | 450 |
| Split history for 40 sampled names | 4 |
| **Total** | **474** |

The campaign is six independently authorized slices of at most 90 attempts
(`90 + 90 + 90 + 90 + 90 + 24`). The initial frozen graph cannot exceed 474
physical attempts; the unused 66 permits implied by six generic 90-attempt
caps are neither planned nor authorized. Request 100 is impossible inside
every slice, redirects and automatic retries are disabled, and completed
immutable partitions are not downloaded again. A provider failure adds no
automatic retry: any later recovery is a separately frozen exact count `R`,
making the completion cost `474 + R`.

Core is deliberately not batched at the provider's documented ten-name
maximum: Cultra's saved account ledger has one success in one two-name Core
attempt and zero successes in three ten-name attempts (all 502). Historical
chains remain ten names/request after 451 successes in 451 saved attempts.

The optimizer saves 183 calls by censoring new signals before each cohort
rotation. Every selected T+1 entry and full 60-session path remains inside the
same fetched block. A separately frozen continuous-entry extension would cost
183 additional chain calls (**657 total**) and is not part of the base campaign.

This estimate is not authorization. Before slice 1, Cultra still needs three
independent raw source bundles: four exact-date point-in-time universe
snapshots, a 450-session XNYS calendar, and point-in-time event/adjustment
history. The normalized files are not trusted merely because they contain a
hash: `prepare-history-inputs` binds every output to the preserved raw bytes,
rejects ORATS-derived prerequisite sources, rejects current-constituent
projection, and rejects empty earnings evidence for sampled stocks. No ORATS
or Schwab request is needed for this preparation.

### Public-source acquisition result (2026-08-31)

The first public-only acquisition is preserved under
`var/historical/public_sources/2026-08-31` and contains 23 raw artifacts. Its
offline audit found 3,344, 3,045, 3,495, and 3,474 security underlyings with
positive Cboe option volume on the four exact cohort-selection dates. This is
a broad stock-or-ETP discovery frame, not a fixed list and not an ETF cohort.
An execution-relevance rule of at least 1,000 same-day Cboe contracts across
at least two of the four venues leaves variable frames of 598, 469, 621, and
606 names. This is a disclosed liquidity threshold, not a fixed-name or top-N
output cap.

Unresolved stock-versus-ETP rows remain visible in the broad historical frame
but cannot enter a cohort. The cohort freeze can now use a small, deterministically
verified subset while preserving the full unresolved population, and it
enforces the 80% stock floor during selection. This removes the need to classify
all 3,000+ names or discover after sampling that an ETF-heavy cohort fails.

The same audit reconstructed all 450 XNYS sessions and reconciled 3,209 unique
OCC contract-adjustment memo index rows across eight non-overlapping exports.
It intentionally did **not** promote those files into the prerequisite freeze:
Cboe's security product type does not distinguish stocks from ETFs/ETPs, the
current SEC/Nasdaq references cannot be projected backward, and the OCC index
does not by itself provide complete earnings/dividend/delisting evidence or
exact adjusted-contract deliverables. Status remains
`PARTIAL_NOT_FREEZEABLE`; recommended ORATS attempts now remain zero.

A second public-only pass preserved 48 SEC submissions records with zero
retries and resolved every identity encountered before the four deterministic
cohort stopping points. It recovered GTLS despite its later disappearance from
the current ticker map and mapped pre-merger PARA to the historical Paramount
CIK rather than the reused current association. The frozen 40-name research
sample is disjoint and contains 33 stocks and 7 ETFs—not an ETF-only history.
Current ticker references are not accepted alone: classification combines the
exact-date Cboe row with the complete post-date OCC adjustment index and a
pre-date SEC filing where the issuer mapping is available. The audit separately
lists selected ETFs lacking a pre-date SEC fund filing so that this continuity
inference is visible rather than silently described as direct evidence.

The event pass preserves 80 provider responses plus a collection manifest that
explicitly labels them `TARGETED_CANDIDATE_DISCOVERY_NOT_COMPLETE`. It found 69
earnings candidates, 9 dividend candidates, and 5 affecting OCC memos, including
the `PARA → PSKY` successor chain. It does **not** convert candidate presence into
complete history: 77 event cells remain blocked (all 40 dividend cells, all 33
stock earnings cells, and the affected adjustment/transition/split cells). The
five exact OCC memo PDFs could be found in the official index, but curl,
Playwright, and the in-app browser produced no preservable provider bytes, so
the detail gate remains closed. The universe/cohort candidate is useful for
scoping acquisition; it is still not authorization for the 474 ORATS attempts.

Reproduce and verify the public-source audit without network access:

```bash
python3 -m cultra audit-public-history-sources \
  --source-root /Users/anuppamvi/tradedesk/cultra/var/historical/public_sources/2026-08-31 \
  --run-id <new-unique-public-source-audit-id>
python3 -m cultra verify-public-history-sources \
  /Users/anuppamvi/tradedesk/cultra/out/<public-source-audit-id>
python3 -m cultra classify-public-history-universe \
  --public-source-audit /Users/anuppamvi/tradedesk/cultra/out/<public-source-audit-id> \
  --sec-submission-root /Users/anuppamvi/tradedesk/cultra/var/historical/public_classification/2026-08-31/sec_submissions \
  --run-id <new-unique-classification-id>
python3 -m cultra verify-public-history-classification \
  /Users/anuppamvi/tradedesk/cultra/out/<classification-id>
python3 -m cultra audit-public-history-events \
  --classification-run /Users/anuppamvi/tradedesk/cultra/out/<classification-id> \
  --event-source-root /Users/anuppamvi/tradedesk/cultra/var/historical/public_events/2026-08-31 \
  --run-id <new-unique-public-event-audit-id>
python3 -m cultra verify-public-history-events \
  /Users/anuppamvi/tradedesk/cultra/out/<public-event-audit-id>
```

Reproduce the estimate without loading a token:

```bash
python3 -m cultra estimate-history --eligible-symbols 254
```

The eligible-symbol count proves the upstream universe is broad; it does not
change the 474-call base because ORATS Core is requested only after the external
point-in-time cohorts are frozen.

## Daily request boundary

Daily production is a separate budget: target 25, logical cap 60, protocol
ceiling 99, zero automatic retries, and normally zero same-vintage morning
calls. The saved 254-name local screen needs 26 Core batches at the documented
ten-name batch size. The all-max 83-call funnel is rejected; any unevaluated
names remain `NOT_FULLY_EVALUATED_BUDGET` rather than disappearing.

## Offline verification

```bash
python3 -m unittest discover -s tests -v
python3 -m cultra doctor --json
python3 -m cultra offline-audit --run-id <new-unique-offline-run-id>
```

The post-acquisition V2 path is also entirely offline and ordered:

```bash
python3 -m cultra prepare-history-inputs --input-set-id <id> --universe-source <raw-universe.json> --session-source <raw-sessions.json> --event-source <raw-events.json> --output-dir <prerequisite-dir>
python3 -m cultra freeze-history-campaign --campaign-id <id> --prerequisite-freeze <prerequisite_freeze.json> --output-dir <campaign-dir>
python3 -m cultra verify-history-campaign --campaign-freeze <freeze.json> --runs-root out --output-dir <completion-dir>
python3 -m cultra ingest-history-v2 --campaign-completion <completion.json> --database <cultra/var/historical_v2/normalized.sqlite3>
python3 -m cultra build-history-outcomes-v2 --normalized-database <normalized.sqlite3> --output-database <outcomes.sqlite3>
python3 -m cultra freeze-history-models-v2 --outcome-database <outcomes.sqlite3> --artifact <cultra/var/evidence/models.json> --evidence-registry <cultra/var/evidence/v2.sqlite3>
python3 -m cultra consume-history-holdout-v2 --model-artifact <models.json> --evidence-registry <v2.sqlite3> --output <cultra/var/evidence/holdout.json>
```

Model freezing reads only pre-holdout outcomes. Its expanding OOF folds use the
actual 59-session entry windows in cohort blocks 1 and 2; the 61-session
no-entry suffixes preserve each maximum-horizon path and the required embargo.
Logistic/isotonic calibration is selected chronologically and EV uses a
conservative clustered residual offset. The final 20 percent is then opened
once across the complete Holm-90 family; batch consumption is atomic.
The verified holdout result is bound to its registry receipt and exact frozen
model/cost identity before current scoring. Current ticket assembly is also
offline: supplied exact quotes and event evidence are converted into complete
one-unit economics and the final ticket gate is rerun. Legacy V1 validation,
research-order, opportunity, and V6/V7 build commands fail closed.

Cultra reads no account positions or buying power, chooses no quantity, and
has no broker order-creation or submission surface. Quantity is always
`USER DETERMINED`.
