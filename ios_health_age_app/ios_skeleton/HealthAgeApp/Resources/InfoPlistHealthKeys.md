# Info.plist and Entitlements Checklist

## Required Info.plist keys
Add these keys with user-facing copy:

```xml
<key>NSHealthShareUsageDescription</key>
<string>This app reads heart, sleep, activity, glucose, and blood pressure data to generate wellness insights.</string>
<key>NSHealthUpdateUsageDescription</key>
<string>This app may write selected wellness entries only when you explicitly opt in.</string>
```

If you allow file uploads for blood reports, also include:

```xml
<key>NSPhotoLibraryUsageDescription</key>
<string>This app uses photos to import lab report images you choose.</string>
```

## Capabilities
1. HealthKit
2. Background delivery entitlement (if using observer queries)

## App review reminders
1. Keep language as wellness/insight, not diagnosis.
2. Explain why each requested data type is needed.
3. Provide a visible way to revoke data usage and delete user data.
