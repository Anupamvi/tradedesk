import HealthKit

struct AnchoredSyncType {
    var key: String
    var sampleType: HKSampleType
}

enum HealthTypes {
    static var readTypes: Set<HKObjectType> {
        var types = Set<HKObjectType>()

        let quantityIds: [HKQuantityTypeIdentifier] = [
            .heartRate,
            .restingHeartRate,
            .heartRateVariabilitySDNN,
            .activeEnergyBurned,
            .appleExerciseTime,
            .stepCount,
            .vo2Max,
            .bloodGlucose,
            .bloodPressureSystolic,
            .bloodPressureDiastolic
        ]

        quantityIds.forEach { identifier in
            if let type = HKObjectType.quantityType(forIdentifier: identifier) {
                types.insert(type)
            }
        }

        if let sleep = HKObjectType.categoryType(forIdentifier: .sleepAnalysis) {
            types.insert(sleep)
        }

        types.insert(HKObjectType.workoutType())

        if #available(iOS 12.0, *) {
            if let labs = HKObjectType.clinicalType(forIdentifier: .labResultRecord) {
                types.insert(labs)
            }
        }

        return types
    }

    static var anchoredSyncTypes: [AnchoredSyncType] {
        var syncTypes: [AnchoredSyncType] = []

        let quantityIds: [HKQuantityTypeIdentifier] = [
            .heartRate,
            .restingHeartRate,
            .heartRateVariabilitySDNN,
            .activeEnergyBurned,
            .appleExerciseTime,
            .stepCount,
            .vo2Max,
            .bloodGlucose,
            .bloodPressureSystolic,
            .bloodPressureDiastolic
        ]

        quantityIds.forEach { identifier in
            if let type = HKObjectType.quantityType(forIdentifier: identifier) {
                syncTypes.append(
                    AnchoredSyncType(
                        key: identifier.rawValue,
                        sampleType: type
                    )
                )
            }
        }

        if let sleep = HKObjectType.categoryType(forIdentifier: .sleepAnalysis) {
            syncTypes.append(
                AnchoredSyncType(
                    key: HKCategoryTypeIdentifier.sleepAnalysis.rawValue,
                    sampleType: sleep
                )
            )
        }

        syncTypes.append(
            AnchoredSyncType(
                key: "workout",
                sampleType: HKObjectType.workoutType()
            )
        )

        return syncTypes
    }
}
