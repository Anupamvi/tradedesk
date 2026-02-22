import SwiftUI

struct DashboardView: View {
    @ObservedObject var viewModel: DashboardViewModel

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    if let error = viewModel.errorMessage {
                        Text(error)
                            .foregroundStyle(.red)
                            .font(.footnote)
                    }

                    if let lastSyncSummary = viewModel.lastSyncSummary {
                        Text(lastSyncSummary)
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }

                    scoreCards
                    metricsSection

                    Text("For wellness insights only. Not medical advice.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
                .padding()
            }
            .navigationTitle("Health Ages")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Refresh") {
                        Task {
                            await viewModel.refresh()
                        }
                    }
                    .disabled(viewModel.isLoading)
                }
            }
            .overlay {
                if viewModel.isLoading {
                    ProgressView("Refreshing...")
                }
            }
            .task {
                guard viewModel.snapshot == nil else {
                    return
                }
                await viewModel.refresh()
            }
        }
    }

    private var scoreCards: some View {
        VStack(spacing: 12) {
            ScoreCard(
                title: "Heart Age",
                ageValue: viewModel.scores?.heart.ageYears,
                confidence: viewModel.scores?.heart.confidence,
                drivers: viewModel.scores?.heart.drivers ?? []
            )
            ScoreCard(
                title: "Metabolic Age",
                ageValue: viewModel.scores?.metabolic.ageYears,
                confidence: viewModel.scores?.metabolic.confidence,
                drivers: viewModel.scores?.metabolic.drivers ?? []
            )
            ScoreCard(
                title: "Muscular Age",
                ageValue: viewModel.scores?.muscular.ageYears,
                confidence: viewModel.scores?.muscular.confidence,
                drivers: viewModel.scores?.muscular.drivers ?? []
            )
        }
    }

    private var metricsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Latest Inputs")
                .font(.headline)

            MetricRow(label: "Resting HR", value: valueText(viewModel.snapshot?.restingHeartRateBpm, suffix: "bpm"))
            MetricRow(label: "HRV (SDNN)", value: valueText(viewModel.snapshot?.hrvSDNNMs, suffix: "ms"))
            MetricRow(label: "Sleep", value: valueText(viewModel.snapshot?.sleepHours, suffix: "h"))
            MetricRow(label: "Exercise", value: valueText(viewModel.snapshot?.exerciseMinutes, suffix: "min"))
            MetricRow(label: "Glucose", value: valueText(viewModel.snapshot?.averageGlucoseMgDl, suffix: "mg/dL"))
            MetricRow(label: "BP", value: bloodPressureText())
            MetricRow(label: "VO2 max", value: valueText(viewModel.snapshot?.vo2MaxMlKgMin, suffix: "mL/kg/min"))
        }
    }

    private func valueText(_ value: Double?, suffix: String) -> String {
        guard let value else {
            return "N/A"
        }
        return "\(String(format: "%.1f", value)) \(suffix)"
    }

    private func bloodPressureText() -> String {
        guard
            let systolic = viewModel.snapshot?.systolicBPMmHg,
            let diastolic = viewModel.snapshot?.diastolicBPMmHg
        else {
            return "N/A"
        }
        return "\(Int(systolic.rounded())) / \(Int(diastolic.rounded())) mmHg"
    }
}

private struct ScoreCard: View {
    var title: String
    var ageValue: Double?
    var confidence: Double?
    var drivers: [String]

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(title)
                    .font(.headline)
                Spacer()
                Text(ageText)
                    .font(.title3)
                    .fontWeight(.semibold)
            }

            if let confidence {
                Text("Confidence: \(Int((confidence * 100).rounded()))%")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }

            if let firstDriver = drivers.first {
                Text(firstDriver)
                    .font(.footnote)
            }
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(.secondarySystemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    private var ageText: String {
        guard let ageValue else {
            return "N/A"
        }
        return "\(String(format: "%.1f", ageValue)) y"
    }
}

private struct MetricRow: View {
    var label: String
    var value: String

    var body: some View {
        HStack {
            Text(label)
            Spacer()
            Text(value)
                .foregroundStyle(.secondary)
        }
        .font(.subheadline)
    }
}
