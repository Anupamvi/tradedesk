import SwiftUI

struct OnboardingView: View {
    @ObservedObject var viewModel: DashboardViewModel

    @State private var ageInput = ""
    @State private var sexAtBirth: BiologicalSex = .male
    @State private var isSmoker = false
    @State private var hasDiabetes = false
    @State private var onBPMedication = false
    @State private var hasLoadedDefaults = false
    @State private var validationError: String?

    var body: some View {
        NavigationStack {
            Form {
                Section("Profile") {
                    TextField("Age", text: $ageInput)
                        .keyboardType(.numberPad)
                    Picker("Sex at birth", selection: $sexAtBirth) {
                        ForEach(BiologicalSex.allCases, id: \.self) { option in
                            Text(option.rawValue.capitalized).tag(option)
                        }
                    }
                }

                Section("Health Context") {
                    Toggle("Smoker", isOn: $isSmoker)
                    Toggle("Diabetes", isOn: $hasDiabetes)
                    Toggle("On BP medication", isOn: $onBPMedication)
                }

                if let validationError {
                    Section {
                        Text(validationError)
                            .foregroundStyle(.red)
                    }
                }

                Section {
                    Button {
                        Task {
                            await saveProfile()
                        }
                    } label: {
                        if viewModel.isLoading {
                            ProgressView()
                        } else {
                            Text("Continue")
                                .fontWeight(.semibold)
                        }
                    }
                    .disabled(viewModel.isLoading)
                }

                Section {
                    Text("For wellness insights only. Not medical advice.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
            }
            .navigationTitle("Health Age Setup")
            .task {
                loadDefaultsIfNeeded()
            }
        }
    }

    private func loadDefaultsIfNeeded() {
        guard !hasLoadedDefaults else {
            return
        }
        let defaults = viewModel.onboardingProfileDefaults
        ageInput = String(Int(defaults.chronologicalAgeYears.rounded()))
        sexAtBirth = defaults.sexAtBirth
        isSmoker = defaults.isSmoker
        hasDiabetes = defaults.hasDiabetes
        onBPMedication = defaults.onBPMedication
        hasLoadedDefaults = true
    }

    private func saveProfile() async {
        validationError = nil
        guard let age = Double(ageInput), age >= 18, age <= 100 else {
            validationError = "Enter an age between 18 and 100."
            return
        }

        let profile = UserProfile(
            chronologicalAgeYears: age,
            sexAtBirth: sexAtBirth,
            isSmoker: isSmoker,
            hasDiabetes: hasDiabetes,
            onBPMedication: onBPMedication
        )
        await viewModel.completeOnboarding(profile: profile)
    }
}
