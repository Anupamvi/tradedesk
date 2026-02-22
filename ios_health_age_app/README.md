# Apple Health Age App (Trackable Workspace)
Date: 2026-02-22

This folder is a git-trackable iPhone app workspace that includes:
- Product and technical planning docs
- A SwiftUI app skeleton
- Onboarding and local profile persistence
- Snapshot history persistence and baseline recomputation
- HealthKit observer + anchored incremental sync scaffolding
- Xcode project generation via `xcodegen`

## Key files
- `ios_health_age_app/v1_prd.md`
- `ios_health_age_app/technical_architecture.md`
- `ios_health_age_app/healthkit_data_dictionary.md`
- `ios_health_age_app/scoring_validation_plan.md`
- `ios_health_age_app/project.yml`
- `ios_health_age_app/QUICKSTART_MACOS.md`
- `ios_health_age_app/ios_skeleton/HealthAgeApp/...`

## macOS setup
1. Switch to a Mac with Xcode installed.
2. Run `bash scripts/macos_bootstrap.sh` from `ios_health_age_app/`.
3. Open `HealthAgeApp.xcodeproj`.
4. Set your signing team and bundle identifier.
5. Build on a physical iPhone for full HealthKit behavior.

## Notes
- This project is positioned as a wellness app, not a diagnostic app.
- Keep score outputs labeled as approximations with confidence indicators.
