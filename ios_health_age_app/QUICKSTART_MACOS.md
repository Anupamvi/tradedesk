# macOS Quickstart

## 1. Prerequisites
1. Xcode installed
2. Apple Developer signing configured in Xcode
3. Homebrew installed

## 2. Generate project
From `ios_health_age_app/`:

```bash
bash scripts/macos_bootstrap.sh
```

This generates `HealthAgeApp.xcodeproj`.

## 3. Open and configure
1. Open `HealthAgeApp.xcodeproj`.
2. In target `HealthAgeApp`, set:
- Team
- Unique bundle identifier
3. Confirm capabilities include HealthKit.

## 4. Run
1. Connect an iPhone (recommended).
2. Select device target in Xcode.
3. Build and run.

## 5. First app flow
1. Complete onboarding profile.
2. Grant HealthKit permissions.
3. Tap `Refresh` on dashboard.

## 6. Current data persistence
- User profile and daily snapshots are stored locally as JSON in Application Support.
- Baseline is recomputed from the latest 28 snapshots.
