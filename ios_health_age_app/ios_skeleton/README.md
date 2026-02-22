# iOS Skeleton Setup
This folder contains starter Swift files for a HealthKit-powered "health age" app.

## How to use
1. On macOS, install Xcode and `xcodegen`.
2. From `ios_health_age_app/`, run `xcodegen generate`.
3. Open `HealthAgeApp.xcodeproj`.
4. Enable capabilities:
- HealthKit
- Background delivery entitlement for HealthKit
5. Set signing team and bundle identifier.
6. Build and run on a physical iPhone (HealthKit data is limited on Simulator).

## What is implemented
- HealthKit authorization and basic sample queries
- Daily snapshot builder for heart/sleep/activity/glucose/blood pressure
- Observer + anchored incremental sync coordinator
- Feature vector builder (baseline-relative deltas)
- Age scoring service (heart/metabolic/muscular approximations)
- Onboarding flow and persistent profile storage
- Snapshot history persistence and baseline recomputation
- Starter dashboard/root views and view model

## What you should implement next
1. Replace JSON file persistence with SwiftData/CoreData if preferred.
2. Add full historical import for first-run 30-90 day backfill.
3. Add blood report OCR ingestion with Vision + domain parser rules.
4. Calibrate score weights and confidence model with pilot data.
5. Add tests for scoring monotonicity and data quality edge cases.
