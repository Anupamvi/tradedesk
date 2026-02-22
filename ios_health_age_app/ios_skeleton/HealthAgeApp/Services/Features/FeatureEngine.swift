import Foundation

struct FeatureVector: Equatable {
    var restingHeartRateDelta: Double?
    var hrvDelta: Double?
    var sleepDelta: Double?
    var exerciseDelta: Double?
    var glucoseDelta: Double?
    var systolicDelta: Double?
    var vo2Delta: Double?
}

final class FeatureEngine {
    func buildVector(snapshot: DailyFeatureSnapshot, baseline: BaselineProfile?) -> FeatureVector {
        FeatureVector(
            restingHeartRateDelta: delta(current: snapshot.restingHeartRateBpm, baseline: baseline?.restingHeartRateBpm),
            hrvDelta: delta(current: snapshot.hrvSDNNMs, baseline: baseline?.hrvSDNNMs),
            sleepDelta: delta(current: snapshot.sleepHours, baseline: baseline?.sleepHours),
            exerciseDelta: delta(current: snapshot.exerciseMinutes, baseline: baseline?.exerciseMinutes),
            glucoseDelta: delta(current: snapshot.averageGlucoseMgDl, baseline: baseline?.averageGlucoseMgDl),
            systolicDelta: delta(current: snapshot.systolicBPMmHg, baseline: baseline?.systolicBPMmHg),
            vo2Delta: delta(current: snapshot.vo2MaxMlKgMin, baseline: baseline?.vo2MaxMlKgMin)
        )
    }

    func buildBaseline(from snapshots: [DailyFeatureSnapshot]) -> BaselineProfile? {
        guard !snapshots.isEmpty else {
            return nil
        }

        return BaselineProfile(
            restingHeartRateBpm: average(snapshots.compactMap { $0.restingHeartRateBpm }),
            hrvSDNNMs: average(snapshots.compactMap { $0.hrvSDNNMs }),
            sleepHours: average(snapshots.compactMap { $0.sleepHours }),
            activeEnergyKcal: average(snapshots.compactMap { $0.activeEnergyKcal }),
            exerciseMinutes: average(snapshots.compactMap { $0.exerciseMinutes }),
            steps: average(snapshots.compactMap { $0.steps }),
            workoutMinutes: average(snapshots.compactMap { $0.workoutMinutes }),
            vo2MaxMlKgMin: average(snapshots.compactMap { $0.vo2MaxMlKgMin }),
            averageGlucoseMgDl: average(snapshots.compactMap { $0.averageGlucoseMgDl }),
            systolicBPMmHg: average(snapshots.compactMap { $0.systolicBPMmHg }),
            diastolicBPMmHg: average(snapshots.compactMap { $0.diastolicBPMmHg })
        )
    }

    private func delta(current: Double?, baseline: Double?) -> Double? {
        guard let current, let baseline else {
            return nil
        }
        return current - baseline
    }

    private func average(_ values: [Double]) -> Double? {
        guard !values.isEmpty else {
            return nil
        }
        return values.reduce(0.0, +) / Double(values.count)
    }
}
