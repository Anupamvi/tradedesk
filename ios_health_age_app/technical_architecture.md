# Technical Architecture (iPhone v1)
Date: 2026-02-22

## 1. System Design
Local-first iOS architecture with optional backend sync.

```text
SwiftUI App
  -> ViewModels
    -> FeatureEngine
      -> ScoringEngine
    -> HealthKitService
      -> HKObserverQuery + HKAnchoredObjectQuery
    -> BloodReportParser (optional/manual upload path)
  -> Local Store (SwiftData/CoreData)
  -> Optional Sync API (explicit opt-in)
```

## 2. App Modules
### App/UI Layer
- `DashboardView`
- `TrendView`
- `InsightsView`
- `PermissionsOnboardingView`

### Data Access Layer
- `HealthKitService`
- `BloodReportParser`
- `ClinicalRecordAdapter` (if Health Records available)

### Domain Layer
- `FeatureEngine`
- `AgeScoringService`
- `RecommendationService`

### Persistence Layer
- `SnapshotStore`
- `ScoreHistoryStore`
- `UserProfileStore`

## 3. Data Flow
1. User grants permissions.
2. App runs bootstrap import (30-90 day lookback).
3. Daily updates use anchored incremental pulls.
4. Raw samples are normalized into daily feature snapshots.
5. Scoring service computes age approximations and confidence.
6. Recommendation service derives top actions from strongest drivers.
7. Results persisted and rendered in dashboard.

## 4. HealthKit Ingestion Pattern
### Initial sync
- Query historical windows by type.
- Persist normalized, de-duplicated records.

### Incremental sync
- Register `HKObserverQuery` for relevant types.
- On change event, run `HKAnchoredObjectQuery` from stored anchor.
- Update only affected day partitions and recompute scores.

### Background
- Enable HealthKit background delivery entitlement.
- Debounce recomputations to avoid battery spikes.

## 5. Storage Model (Local)
### `daily_feature_snapshot`
- `date`
- `resting_hr_bpm`
- `hrv_sdnn_ms`
- `sleep_hours`
- `sleep_efficiency`
- `exercise_minutes`
- `active_energy_kcal`
- `steps`
- `workout_minutes`
- `vo2max_ml_kg_min`
- `avg_glucose_mg_dl`
- `systolic_bp_mmhg`
- `diastolic_bp_mmhg`
- `total_cholesterol_mg_dl` (optional)
- `hdl_cholesterol_mg_dl` (optional)
- `strength_test_score` (optional)
- `data_completeness`

### `daily_age_scores`
- `date`
- `heart_age`
- `heart_confidence`
- `metabolic_age`
- `metabolic_confidence`
- `muscular_age`
- `muscular_confidence`
- `drivers_json`

### `user_profile`
- `chronological_age`
- `sex_at_birth`
- `smoker`
- `diabetes`
- `on_bp_medication`
- `units_preferences`
- `consent_version`

## 6. API Surface (if cloud sync enabled)
### `POST /v1/snapshots`
- Upload de-identified daily snapshots.

### `POST /v1/scores`
- Upload daily scores for cross-device sync.

### `GET /v1/insights`
- Return generated recommendation text variants.

Cloud should remain optional in v1 to reduce privacy and compliance load.

## 7. Security and Privacy Controls
1. Local encryption for persisted health-derived records.
2. TLS for all network traffic.
3. No advertising use of health data.
4. Explicit consent gate for cloud sync/analytics.
5. One-tap account and data deletion.

## 8. Performance Strategy
1. Precompute daily features in background.
2. Cache the last computed score bundle.
3. Recompute only impacted windows when new samples arrive.
4. Use lightweight charts and avoid heavy runtime transforms in UI thread.

## 9. Failure Handling
1. Partial permissions: compute with available data and degrade confidence.
2. Missing data windows: carry last known values where appropriate and mark stale.
3. Sync errors: local-first UI with retry queue.
4. Parsing errors for uploaded blood reports: preserve raw file and allow manual entry.
