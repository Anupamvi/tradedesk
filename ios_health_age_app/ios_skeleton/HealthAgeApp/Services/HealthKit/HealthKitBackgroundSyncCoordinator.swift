import Foundation
import HealthKit

final class HealthKitBackgroundSyncCoordinator {
    private let store: HKHealthStore
    private let anchorStore: HealthKitAnchorStore
    private let syncTypes: [AnchoredSyncType]
    private let calendar: Calendar
    private let lookbackDays: Int
    private var observerQueries: [HKObserverQuery] = []

    init(
        store: HKHealthStore,
        anchorStore: HealthKitAnchorStore = HealthKitAnchorStore(),
        syncTypes: [AnchoredSyncType] = HealthTypes.anchoredSyncTypes,
        calendar: Calendar = .current,
        lookbackDays: Int = 120
    ) {
        self.store = store
        self.anchorStore = anchorStore
        self.syncTypes = syncTypes
        self.calendar = calendar
        self.lookbackDays = lookbackDays
    }

    func configureBackgroundDelivery() async {
        for syncType in syncTypes {
            _ = try? await enableBackgroundDelivery(for: syncType.sampleType)
        }
    }

    func startObservers(onUpdate: @escaping () -> Void) {
        stopObservers()

        for syncType in syncTypes {
            let query = HKObserverQuery(sampleType: syncType.sampleType, predicate: nil) { _, completion, error in
                defer {
                    completion()
                }

                guard error == nil else {
                    return
                }
                onUpdate()
            }
            observerQueries.append(query)
            store.execute(query)
        }
    }

    func stopObservers() {
        observerQueries.forEach { query in
            store.stop(query)
        }
        observerQueries.removeAll()
    }

    func performAnchoredSync() async throws -> Int {
        var totalChanges = 0

        for syncType in syncTypes {
            totalChanges += try await runAnchoredQuery(syncType: syncType)
        }

        return totalChanges
    }

    private func enableBackgroundDelivery(for sampleType: HKSampleType) async throws {
        try await withCheckedThrowingContinuation { continuation in
            store.enableBackgroundDelivery(for: sampleType, frequency: .hourly) { success, error in
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

    private func runAnchoredQuery(syncType: AnchoredSyncType) async throws -> Int {
        let existingAnchor = anchorStore.loadAnchor(forKey: syncType.key)
        let endDate = Date()
        let startDate = calendar.date(byAdding: .day, value: -lookbackDays, to: endDate)
        let predicate = startDate.map {
            HKQuery.predicateForSamples(withStart: $0, end: endDate, options: .strictStartDate)
        }

        return try await withCheckedThrowingContinuation { continuation in
            let query = HKAnchoredObjectQuery(
                type: syncType.sampleType,
                predicate: predicate,
                anchor: existingAnchor,
                limit: HKObjectQueryNoLimit
            ) { [weak self] _, samples, deletedObjects, newAnchor, error in
                if let error {
                    continuation.resume(throwing: error)
                    return
                }

                self?.anchorStore.saveAnchor(newAnchor, forKey: syncType.key)
                let sampleCount = samples?.count ?? 0
                let deletedCount = deletedObjects?.count ?? 0
                continuation.resume(returning: sampleCount + deletedCount)
            }
            store.execute(query)
        }
    }
}

final class HealthKitAnchorStore {
    private let defaults: UserDefaults
    private let keyPrefix: String

    init(defaults: UserDefaults = .standard, keyPrefix: String = "healthage.hk.anchor.") {
        self.defaults = defaults
        self.keyPrefix = keyPrefix
    }

    func loadAnchor(forKey key: String) -> HKQueryAnchor? {
        let scopedKey = keyPrefix + key
        guard let data = defaults.data(forKey: scopedKey) else {
            return nil
        }
        return try? NSKeyedUnarchiver.unarchivedObject(ofClass: HKQueryAnchor.self, from: data)
    }

    func saveAnchor(_ anchor: HKQueryAnchor?, forKey key: String) {
        let scopedKey = keyPrefix + key
        guard let anchor else {
            defaults.removeObject(forKey: scopedKey)
            return
        }

        if let data = try? NSKeyedArchiver.archivedData(withRootObject: anchor, requiringSecureCoding: true) {
            defaults.set(data, forKey: scopedKey)
        }
    }
}
