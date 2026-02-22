import SwiftUI

struct RootView: View {
    @ObservedObject var viewModel: DashboardViewModel

    var body: some View {
        Group {
            if viewModel.hasCompletedOnboarding {
                DashboardView(viewModel: viewModel)
            } else {
                OnboardingView(viewModel: viewModel)
            }
        }
        .task {
            await viewModel.bootstrapIfNeeded()
        }
    }
}
