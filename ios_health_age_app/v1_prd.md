# Product Requirements Document (v1)
Date: 2026-02-22
Platform: iPhone (iOS 17+ target)

## 1. Product Summary
Build a consumer wellness app that ingests Apple Health data and user-entered context to estimate:
- Heart age (risk-oriented cardiovascular age approximation)
- Muscular age (fitness and strength capacity approximation)
- Metabolic age (metabolic flexibility and glycemic regulation approximation)

The app should emphasize trend-based behavior change, not diagnosis.

## 2. Positioning and Claims
### Allowed language
- "Age approximation"
- "Wellness insight"
- "Recovery and readiness trend"
- "Behavioral coaching"

### Not allowed in v1
- Disease diagnosis claims
- Treatment recommendations
- Statements implying clinician replacement

## 3. Target Users
- Health-aware adults who already use Apple Watch and Apple Health
- Athletes using recovery signals (HRV, sleep, load)
- Users tracking blood glucose, blood pressure, and labs

## 4. Core User Problems
1. Data overload from Apple Health without integrated interpretation.
2. Lack of a single "status" lens across heart, fitness, and metabolism.
3. Difficulty linking daily behavior to long-term health trajectory.

## 5. v1 Product Goals
1. Generate daily heart/muscular/metabolic age approximations with confidence.
2. Explain top 3 drivers behind score changes.
3. Provide 1-3 actionable recommendations per day.
4. Show trends over 7/30/90 days.

## 6. Non-goals (v1)
1. Real-time coaching during workouts.
2. Provider-facing clinical workflows.
3. Full nutrition tracking platform.
4. Full diagnostic interpretation of lab panels.

## 7. Competitive Learnings to Apply
### WHOOP-style patterns
- Baseline-relative scoring over fixed thresholds.
- Recovery-oriented morning summary.
- Sleep + HRV + resting HR as primary readiness stack.

### Ultrahuman-style patterns
- Strong glucose and metabolic response framing.
- Tight feedback loops around routines.
- Blood report contextualization into a simplified score.

## 8. User Journey (v1)
1. Onboarding
- Consent and disclaimer
- HealthKit permissions (phased)
- Questionnaire: age, sex at birth, smoking status, diabetes status, BP medication
- Optional: upload blood reports (PDF/image)

2. Data ingestion
- Pull historical baseline (last 30-90 days)
- Set daily background refresh

3. Daily experience
- Dashboard card: heart/muscular/metabolic age and trend arrows
- Driver cards: what improved/worsened
- Daily actions: sleep, movement, recovery, metabolic targets

4. Weekly review
- Trend summary
- Habit adherence
- Best/worst drivers

## 9. Functional Requirements
### FR-1 Data Access
- Read from Apple Health:
  - Heart rate
  - Resting heart rate
  - HRV (SDNN)
  - Sleep analysis
  - Workouts
  - Activity signals (steps, exercise minutes, active energy)
  - VO2 max/cardio fitness (if available)
  - Blood glucose
  - Blood pressure (systolic/diastolic)
  - Optional clinical lab records

### FR-2 Data Processing
- Normalize units and timestamps.
- Compute daily aggregates and rolling windows (7/28/90 days).
- Build personalized baseline after at least 14 days of data.

### FR-3 Scoring
- Output three ages and confidence.
- Show missing-data warnings and confidence degradation.
- Store daily score history for trends.

### FR-4 Explainability
- For each score, show top positive and negative factors.
- Show "why confidence is low" when inputs are missing.

### FR-5 Coaching
- Generate actionable recommendations linked to drivers.
- Cap at 1-3 actions/day to avoid overload.

### FR-6 Privacy Controls
- Local-first data processing.
- Explicit opt-in for cloud sync and analytics.
- User can delete all app data at any time.

## 10. Non-functional Requirements
1. Daily refresh in under 2 seconds for cached data and under 8 seconds for full pull.
2. App launch under 1.5 seconds on recent iPhones.
3. No crash on partial/missing HealthKit permissions.
4. Encryption at rest for persisted health features.

## 11. UX Requirements
1. One-screen daily summary.
2. Clear confidence indicator next to each age score.
3. Always include "Not medical advice" copy near score outputs.
4. Trend charts at 7/30/90-day windows.

## 12. Metrics and Success Criteria
### Activation
- Permission completion rate > 65%
- First score generation rate > 55% of installs

### Engagement
- Day-7 retention > 30%
- Weekly active users who view scores >= 2 times/week

### Product quality
- Crash-free sessions > 99.5%
- Data refresh failures < 2%

### Insight quality
- User-rated usefulness >= 4/5 in pilot feedback

## 13. v1 Scope vs v1.1
### In scope (v1)
- iPhone app
- HealthKit ingestion
- Three age approximations
- Daily/weekly insights
- Manual blood-report upload pipeline placeholder

### Out of scope (v1)
- Apple Watch app UI
- Fully automated lab extraction for all report formats
- Clinical EHR integrations beyond Health Records availability

## 14. Risks and Mitigations
1. Incomplete data coverage
- Mitigation: confidence scoring + missing-data prompts.

2. User misunderstanding of "age" values
- Mitigation: transparent formula summary + trend emphasis.

3. Regulatory risk from claim language
- Mitigation: strict wellness positioning and legal copy review.

4. Model drift or unstable outputs
- Mitigation: smoothing, floor/ceiling constraints, pilot calibration.

## 15. Rollout Plan
1. Internal alpha (2 weeks): team devices, instrumentation and crash checks.
2. Private beta (4-6 weeks): 50-200 users, score calibration feedback.
3. Public launch after validation gates pass.
