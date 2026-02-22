import Foundation
import HealthKit

enum HealthKitServiceError: Error {
    case healthDataUnavailable
    case authorizationDenied
    case missingReadTypes
}

protocol HealthKitServing {
    func requestAuthorization() async throws
    func loadLatestSnapshot(profile: UserProfile) async throws -> DailyFeatureSnapshot
    func makeBackgroundSyncCoordinator() -> HealthKitBackgroundSyncCoordinator
}

final class HealthKitService: HealthKitServing {
    private let store: HKHealthStore
    private let calendar: Calendar
    private let anchorStore: HealthKitAnchorStore

    private let heartRateUnit = HKUnit.count().unitDivided(by: HKUnit.minute())
    private let hrvUnit = HKUnit.secondUnit(with: .milli)
    private let glucoseUnit = HKUnit(from: "mg/dL")
    private let vo2Unit = HKUnit(from: "mL/(kg*min)")

    init(
        store: HKHealthStore = HKHealthStore(),
        calendar: Calendar = .current,
        anchorStore: HealthKitAnchorStore = HealthKitAnchorStore()
    ) {
        self.store = store
        self.calendar = calendar
        self.anchorStore = anchorStore
    }

    func requestAuthorization() async throws {
        guard HKHealthStore.isHealthDataAvailable() else {
            throw HealthKitServiceError.healthDataUnavailable
        }

        let readTypes = HealthTypes.readTypes
        guard !readTypes.isEmpty else {
            throw HealthKitServiceError.missingReadTypes
        }

        try await withCheckedThrowingContinuation { continuation in
            store.requestAuthorization(toShare: Set<HKSampleType>(), read: readTypes) { success, error in
                if let error {
                    continuation.resume(throwing: error)
                    return
                }
                guard success else {
                    continuation.resume(throwing: HealthKitServiceError.authorizationDenied)
                    return
                }
                continuation.resume(returning: ())
            }
        }
    }

    func loadLatestSnapshot(profile: UserProfile) async throws -> DailyFeatureSnapshot {
        async let restingHeartRate = mostRecentQuantity(
            typeIdentifier: .restingHeartRate,
            unit: heartRateUnit
        )
        async let averageHeartRate = averageDiscreteQuantity(
            typeIdentifier: .heartRate,
            unit: heartRateUnit,
            days: 7
        )
        async let hrv = averageDiscreteQuantity(
            typeIdentifier: .heartRateVariabilitySDNN,
            unit: hrvUnit,
            days: 7
        )
        async let sleepHours = sleepHoursForLastNight()
        async let activeEnergy = averageDailySumQuantity(
            typeIdentifier: .activeEnergyBurned,
            unit: .kilocalorie(),
            days: 7
        )
        async let exerciseMinutes = averageDailySumQuantity(
            typeIdentifier: .appleExerciseTime,
            unit: .minute(),
            days: 7
        )
        async let steps = averageDailySumQuantity(
            typeIdentifier: .stepCount,
            unit: .count(),
            days: 7
        )
        async let workoutMinutes = averageWorkoutMinutes(days: 7)
        async let vo2Max = mostRecentQuantity(
            typeIdentifier: .vo2Max,
            unit: vo2Unit
        )
        async let glucose = averageDiscreteQuantity(
            typeIdentifier: .bloodGlucose,
            unit: glucoseUnit,
            days: 7
        )
        async let systolic = mostRecentQuantity(
            typeIdentifier: .bloodPressureSystolic,
            unit: .millimeterOfMercury()
        )
        async let diastolic = mostRecentQuantity(
            typeIdentifier: .bloodPressureDiastolic,
            unit: .millimeterOfMercury()
        )

        let resolvedRestingHeartRate = try await restingHeartRate
        let resolvedAverageHeartRate = try await averageHeartRate
        let resolvedHRV = try await hrv
        let resolvedSleepHours = try await sleepHours
        let resolvedActiveEnergy = try await activeEnergy
        let resolvedExerciseMinutes = try await exerciseMinutes
        let resolvedSteps = try await steps
        let resolvedWorkoutMinutes = try await workoutMinutes
        let resolvedVO2Max = try await vo2Max
        let resolvedGlucose = try await glucose
        let resolvedSystolic = try await systolic
        let resolvedDiastolic = try await diastolic

        return DailyFeatureSnapshot(
            asOfDate: Date(),
            profile: profile,
            restingHeartRateBpm: resolvedRestingHeartRate,
            averageHeartRateBpm: resolvedAverageHeartRate,
            hrvSDNNMs: resolvedHRV,
            sleepHours: resolvedSleepHours,
            activeEnergyKcal: resolvedActiveEnergy,
            exerciseMinutes: resolvedExerciseMinutes,
            steps: resolvedSteps,
            workoutMinutes: resolvedWorkoutMinutes,
            vo2MaxMlKgMin: resolvedVO2Max,
            averageGlucoseMgDl: resolvedGlucose,
            systolicBPMmHg: resolvedSystolic,
            diastolicBPMmHg: resolvedDiastolic,
            totalCholesterolMgDl: nil,
            hdlCholesterolMgDl: nil,
            strengthTestScore: nil
        )
    }

    func makeBackgroundSyncCoordinator() -> HealthKitBackgroundSyncCoordinator {
        HealthKitBackgroundSyncCoordinator(
            store: store,
            anchorStore: anchorStore,
            syncTypes: HealthTypes.anchoredSyncTypes,
            calendar: calendar
        )
    }

    private func mostRecentQuantity(
        typeIdentifier: HKQuantityTypeIdentifier,
        unit: HKUnit
    ) async throws -> Double? {
        guard let type = HKObjectType.quantityType(forIdentifier: typeIdentifier) else {
            return nil
        }

        let sort = NSSortDescriptor(key: HKSampleSortIdentifierEndDate, ascending: false)
        return try await withCheckedThrowingContinuation { continuation in
            let query = HKSampleQuery(
                sampleType: type,
                predicate: nil,
                limit: 1,
                sortDescriptors: [sort]
            ) { _, samples, error in
                if let error {
                    continuation.resume(throwing: error)
                    return
                }
                let value = (samples?.first as? HKQuantitySample)?
                    .quantity
                    .doubleValue(for: unit)
                continuation.resume(returning: value)
            }
            store.execute(query)
        }
    }

    private func averageDailySumQuantity(
        typeIdentifier: HKQuantityTypeIdentifier,
        unit: HKUnit,
        days: Int
    ) async throws -> Double? {
        guard let type = HKObjectType.quantityType(forIdentifier: typeIdentifier) else {
            return nil
        }

        let now = Date()
        guard let start = calendar.date(byAdding: .day, value: -days, to: now) else {
            return nil
        }
        let predicate = HKQuery.predicateForSamples(withStart: start, end: now, options: .strictStartDate)

        return try await withCheckedThrowingContinuation { continuation in
            let query = HKStatisticsQuery(
                quantityType: type,
                quantitySamplePredicate: predicate,
                options: .cumulativeSum
            ) { _, statistics, error in
                if let error {
                    continuation.resume(throwing: error)
                    return
                }
                guard let sum = statistics?.sumQuantity() else {
                    continuation.resume(returning: nil)
                    return
                }
                let total = sum.doubleValue(for: unit)
                continuation.resume(returning: total / Double(days))
            }
            store.execute(query)
        }
    }

    private func averageDiscreteQuantity(
        typeIdentifier: HKQuantityTypeIdentifier,
        unit: HKUnit,
        days: Int
    ) async throws -> Double? {
        guard let type = HKObjectType.quantityType(forIdentifier: typeIdentifier) else {
            return nil
        }

        let now = Date()
        guard let start = calendar.date(byAdding: .day, value: -days, to: now) else {
            return nil
        }
        let predicate = HKQuery.predicateForSamples(withStart: start, end: now, options: .strictStartDate)

        return try await withCheckedThrowingContinuation { continuation in
            let query = HKStatisticsQuery(
                quantityType: type,
                quantitySamplePredicate: predicate,
                options: .discreteAverage
            ) { _, statistics, error in
                if let error {
                    continuation.resume(throwing: error)
                    return
                }
                let value = statistics?.averageQuantity()?.doubleValue(for: unit)
                continuation.resume(returning: value)
            }
            store.execute(query)
        }
    }

    private func sleepHoursForLastNight() async throws -> Double? {
        guard let sleepType = HKObjectType.categoryType(forIdentifier: .sleepAnalysis) else {
            return nil
        }

        let now = Date()
        guard let start = calendar.date(byAdding: .hour, value: -36, to: now) else {
            return nil
        }
        let predicate = HKQuery.predicateForSamples(withStart: start, end: now, options: .strictStartDate)
        let sort = NSSortDescriptor(key: HKSampleSortIdentifierStartDate, ascending: true)

        let samples: [HKCategorySample]? = try await withCheckedThrowingContinuation { continuation in
            let query = HKSampleQuery(
                sampleType: sleepType,
                predicate: predicate,
                limit: HKObjectQueryNoLimit,
                sortDescriptors: [sort]
            ) { _, samples, error in
                if let error {
                    continuation.resume(throwing: error)
                    return
                }
                continuation.resume(returning: samples as? [HKCategorySample])
            }
            store.execute(query)
        }

        let totalSeconds = (samples ?? []).reduce(0.0) { partial, sample in
            guard isAsleepCategory(sample.value) else {
                return partial
            }
            return partial + sample.endDate.timeIntervalSince(sample.startDate)
        }

        guard totalSeconds > 0 else {
            return nil
        }
        return totalSeconds / 3600.0
    }

    private func averageWorkoutMinutes(days: Int) async throws -> Double? {
        let now = Date()
        guard let start = calendar.date(byAdding: .day, value: -days, to: now) else {
            return nil
        }
        let predicate = HKQuery.predicateForSamples(withStart: start, end: now, options: .strictStartDate)

        let workouts: [HKWorkout]? = try await withCheckedThrowingContinuation { continuation in
            let query = HKSampleQuery(
                sampleType: HKObjectType.workoutType(),
                predicate: predicate,
                limit: HKObjectQueryNoLimit,
                sortDescriptors: nil
            ) { _, samples, error in
                if let error {
                    continuation.resume(throwing: error)
                    return
                }
                continuation.resume(returning: samples as? [HKWorkout])
            }
            store.execute(query)
        }

        let totalMinutes = (workouts ?? []).reduce(0.0) { partial, workout in
            partial + (workout.duration / 60.0)
        }
        return totalMinutes / Double(days)
    }

    private func isAsleepCategory(_ rawValue: Int) -> Bool {
        if rawValue == HKCategoryValueSleepAnalysis.asleep.rawValue {
            return true
        }

        if #available(iOS 16.0, *) {
            return rawValue == HKCategoryValueSleepAnalysis.asleepCore.rawValue
                || rawValue == HKCategoryValueSleepAnalysis.asleepDeep.rawValue
                || rawValue == HKCategoryValueSleepAnalysis.asleepREM.rawValue
                || rawValue == HKCategoryValueSleepAnalysis.asleepUnspecified.rawValue
        }

        return false
    }
}
