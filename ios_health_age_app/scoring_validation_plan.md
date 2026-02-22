# Scoring and Validation Plan
Date: 2026-02-22

## 1. Scoring Principles
1. Scores are approximations for wellness insights, not diagnoses.
2. Personalized baseline is prioritized over population thresholding.
3. Every score includes confidence and top drivers.
4. Missing inputs reduce confidence before they change score heavily.

## 2. Heart Age (v1)
### Preferred method
- Framingham-style CVD risk age mapping when required inputs exist:
  - age, sex, total cholesterol, HDL, systolic BP, smoking, diabetes, BP treatment

### Fallback method
- If lipids are missing, use a proxy model with:
  - resting HR, HRV, systolic BP, sleep regularity, VO2 max (if available)

### Output
- `heart_age_years`
- `heart_confidence` (0 to 1)
- Driver list (for example: "higher resting HR than baseline")

## 3. Metabolic Age (v1)
### Composite score inputs
- Average glucose (or fasting glucose/HbA1c if available)
- Glucose variability proxy
- Sleep duration and regularity
- Resting HR and HRV
- Activity load (exercise minutes, active energy)

### Mapping
- Compute weighted normalized strain score.
- Convert strain score to age delta with clamped bounds.

### Output
- `metabolic_age_years`
- `metabolic_confidence`
- Driver list

## 4. Muscular Age (v1)
### Inputs
- Workout minutes and intensity pattern
- VO2 max/cardio fitness proxy
- Steps and exercise consistency
- Optional in-app functional tests:
  - Sit-to-stand
  - Push-up
  - Plank hold

### Mapping
- Build movement capacity index.
- Convert index to age delta from chronological age.

### Output
- `muscular_age_years`
- `muscular_confidence`
- Driver list

## 5. Confidence Framework
Confidence is a function of:
1. Input completeness
2. Recency
3. Data consistency
4. Baseline maturity

Suggested formula:

```text
confidence =
  0.40 * completeness +
  0.25 * recency +
  0.20 * consistency +
  0.15 * baseline_maturity
```

## 6. Smoothing and Stability
1. Use 7-day EMA for visible score line.
2. Cap day-over-day score changes to avoid noise spikes.
3. Recalculate baselines weekly, not every sample update.

## 7. Validation Protocol
### Phase 1: Internal validity
- Unit tests for feature and scoring functions.
- Sensitivity tests for missing/edge-case data.
- Monotonicity checks (worse inputs should not improve score).

### Phase 2: Pilot reliability (50-200 users)
- 4-8 week observational pilot.
- Collect repeated measures and user context updates.
- Evaluate:
  - Test-retest stability
  - Correlation with self-reported recovery/energy
  - User comprehension and trust

### Phase 3: Calibration
- Adjust feature weights.
- Tune age-delta bounds.
- Tighten confidence thresholds.

## 8. Reliability and Quality Gates
1. Test-retest CV below target for stable periods.
2. No systematic bias by sex or age band in score drift.
3. At least 80% of pilot users report score explanations as understandable.
4. Missing data states should never produce high-confidence outputs.

## 9. Guardrails
1. Show explicit disclaimer near all scores.
2. Do not show recommendations that imply diagnosis/treatment.
3. Escalate emergency-like values only as "seek medical attention" informational prompts, not definitive advice.

## 10. Instrumentation
Track:
1. Permission completion by data type.
2. Score compute success and failure reasons.
3. Confidence distributions by cohort.
4. Recommendation accept/reject interactions.
