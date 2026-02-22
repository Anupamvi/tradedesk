# HealthKit Data Dictionary
Date: 2026-02-22

## Core Signals
| Domain | HealthKit type | Identifier | Unit | Typical Window | Primary Use |
|---|---|---|---|---|---|
| Heart rate | Quantity | `.heartRate` | count/min | Latest + 7d avg | Context and trend |
| Resting HR | Quantity | `.restingHeartRate` | count/min | Latest + 7/28d avg | Heart and metabolic age |
| HRV | Quantity | `.heartRateVariabilitySDNN` | ms | 7/28d avg | Recovery and heart age |
| Sleep | Category | `.sleepAnalysis` | hours | Last night + 7d avg | Recovery/metabolic load |
| Workouts | Workout | `HKWorkoutType.workoutType()` | min/day | 7/28d avg | Muscular and metabolic age |
| Steps | Quantity | `.stepCount` | count/day | Daily + 7d avg | Activity context |
| Exercise minutes | Quantity | `.appleExerciseTime` | min/day | Daily + 7d avg | Activity load |
| Active energy | Quantity | `.activeEnergyBurned` | kcal/day | Daily + 7/28d avg | Metabolic/fitness proxy |
| VO2 max | Quantity | `.vo2Max` | ml/kg/min | Latest + 28d | Fitness and muscular age |
| Blood glucose | Quantity | `.bloodGlucose` | mg/dL | Latest + 7/28d avg | Metabolic age |
| Systolic BP | Quantity | `.bloodPressureSystolic` | mmHg | Latest + 28d avg | Heart age |
| Diastolic BP | Quantity | `.bloodPressureDiastolic` | mmHg | Latest + 28d avg | Heart age |

## Optional Signals
| Domain | HealthKit type | Identifier | Unit | Typical Window | Primary Use |
|---|---|---|---|---|---|
| Clinical labs | Clinical record | `.labResultRecord` | varies | Latest available | Heart/metabolic refinement |
| Respiratory rate | Quantity | `.respiratoryRate` | breaths/min | Sleep window | Recovery context |
| Resting energy | Quantity | `.basalEnergyBurned` | kcal/day | 7/28d avg | Metabolic refinement |

## Blood Report Upload Mapping (Manual/OCR)
| Biomarker | Unit | Used In |
|---|---|---|
| Total cholesterol | mg/dL | Heart age |
| HDL cholesterol | mg/dL | Heart age |
| LDL cholesterol | mg/dL | Secondary insight |
| Triglycerides | mg/dL | Metabolic age |
| HbA1c | % | Metabolic age |
| Fasting glucose | mg/dL | Metabolic age |
| hs-CRP | mg/L | Secondary inflammation context |

## Unit Standards
| Metric | Canonical Unit | Conversion Notes |
|---|---|---|
| Blood glucose | mg/dL | Convert from mmol/L using x18.0182 |
| BP | mmHg | Keep raw mmHg |
| Heart rate | bpm | count/min in HealthKit |
| HRV SDNN | ms | Use raw milliseconds |
| VO2 max | ml/kg/min | Already canonical from HealthKit |

## Data Quality Rules
1. Require at least 4 days of valid data in last 7 days for daily score confidence above 0.6.
2. Require at least 14 valid days in last 28 days for stable baseline scoring.
3. Mark outliers when a metric is beyond 4 SD from personal baseline and exclude from baseline updates.
4. Never silently backfill missing clinical biomarkers; show "not available" state.

## Permissions Strategy
1. Initial prompt: heart, sleep, workouts, activity only.
2. Secondary prompt after value discovery: glucose and blood pressure.
3. Optional advanced prompt: clinical records and uploaded labs.
