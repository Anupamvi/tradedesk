import Foundation

struct BaselineProfile: Codable, Equatable {
    var restingHeartRateBpm: Double?
    var hrvSDNNMs: Double?
    var sleepHours: Double?
    var activeEnergyKcal: Double?
    var exerciseMinutes: Double?
    var steps: Double?
    var workoutMinutes: Double?
    var vo2MaxMlKgMin: Double?
    var averageGlucoseMgDl: Double?
    var systolicBPMmHg: Double?
    var diastolicBPMmHg: Double?
}
