import Foundation
import Combine

@MainActor
final class DashboardViewModel: ObservableObject {
    @Published private(set) var isLoading = false
    @Published private(set) var hasCompletedOnboarding = false
    @Published private(set) var profile: UserProfile?
    @Published private(set) var snapshot: DailyFeatureSnapshot?
    @Published private(set) var scores: AgeScores?
    @Published private(set) var lastSyncSummary: String?
    @Published var errorMessage: String?

    private let healthKitService: HealthKitServing
    private let scoringService: AgeScoringServing
    private let featureEngine: FeatureEngine
    private let localStore: AppLocalStore
    private var baselineProfile: BaselineProfile?
    private var backgroundSyncCoordinator: HealthKitBackgroundSyncCoordinator?
    private var hasBootstrapped = false
    private var hasConfiguredBackgroundSync = false

    init(
        healthKitService: HealthKitServing,
        scoringService: AgeScoringServing,
        featureEngine: FeatureEngine = FeatureEngine(),
        localStore: AppLocalStore = AppLocalStore()
    ) {
        self.healthKitService = healthKitService
        self.scoringService = scoringService
        self.featureEngine = featureEngine
        self.localStore = localStore
    }

    var onboardingProfileDefaults: UserProfile {
        profile ?? UserProfile.placeholder
    }

    func bootstrapIfNeeded() async {
        guard !hasBootstrapped else {
            return
        }
        hasBootstrapped = true
        await bootstrap()
    }

    func completeOnboarding(profile: UserProfile) async {
        isLoading = true
        errorMessage = nil

        do {
            try await localStore.saveUserProfile(profile)
            self.profile = profile
            hasCompletedOnboarding = true
            lastSyncSummary = "Profile saved. Pulling HealthKit data..."
            isLoading = false
            await refresh()
        } catch {
            errorMessage = "Unable to save onboarding profile: \(error.localizedDescription)"
            isLoading = false
        }
    }

    func refresh() async {
        guard let profile else {
            hasCompletedOnboarding = false
            errorMessage = "Complete onboarding to start score generation."
            return
        }
        guard !isLoading else {
            return
        }

        isLoading = true
        errorMessage = nil

        do {
            try await healthKitService.requestAuthorization()
            try await configureBackgroundSyncIfNeeded()

            let incrementalChanges = try await backgroundSyncCoordinator?.performAnchoredSync() ?? 0
            let latestSnapshot = try await healthKitService.loadLatestSnapshot(profile: profile)
            try await localStore.appendSnapshot(latestSnapshot)

            let history = try await localStore.loadSnapshots()
            let baselineWindow = Array(history.suffix(28))
            baselineProfile = featureEngine.buildBaseline(from: baselineWindow)

            let featureVector = featureEngine.buildVector(
                snapshot: latestSnapshot,
                baseline: baselineProfile
            )
            let latestScores = scoringService.generateScores(
                snapshot: latestSnapshot,
                featureVector: featureVector
            )

            snapshot = latestSnapshot
            scores = latestScores
            lastSyncSummary = incrementalChanges > 0
                ? "Synced \(incrementalChanges) incremental HealthKit changes."
                : "No new incremental changes; snapshot refreshed."
        } catch {
            errorMessage = "Unable to refresh health insights: \(error.localizedDescription)"
        }

        isLoading = false
    }

    private func bootstrap() async {
        isLoading = true
        errorMessage = nil

        do {
            let savedProfile = try await localStore.loadUserProfile()
            profile = savedProfile
            hasCompletedOnboarding = savedProfile != nil

            let history = try await localStore.loadSnapshots()
            if let latest = history.last {
                let baselineWindow = Array(history.suffix(28))
                baselineProfile = featureEngine.buildBaseline(from: baselineWindow)
                let vector = featureEngine.buildVector(snapshot: latest, baseline: baselineProfile)
                scores = scoringService.generateScores(snapshot: latest, featureVector: vector)
                snapshot = latest
            }
        } catch {
            errorMessage = "Unable to load local app data: \(error.localizedDescription)"
        }

        isLoading = false
    }

    private func configureBackgroundSyncIfNeeded() async throws {
        guard !hasConfiguredBackgroundSync else {
            return
        }

        let coordinator = healthKitService.makeBackgroundSyncCoordinator()
        await coordinator.configureBackgroundDelivery()
        coordinator.startObservers { [weak self] in
            Task { @MainActor in
                guard let self else {
                    return
                }
                guard self.hasCompletedOnboarding, !self.isLoading else {
                    return
                }
                await self.refresh()
            }
        }

        backgroundSyncCoordinator = coordinator
        hasConfiguredBackgroundSync = true
    }
}
