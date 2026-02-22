import SwiftUI

@main
struct HealthAgeApp: App {
    @StateObject private var viewModel = DashboardViewModel(
        healthKitService: HealthKitService(),
        scoringService: AgeScoringService()
    )

    var body: some Scene {
        WindowGroup {
            RootView(viewModel: viewModel)
        }
    }
}
