import Foundation

struct DailyFeatureSnapshot: Codable, Equatable {
    var asOfDate: Date
    var profile: UserProfile

    var restingHeartRateBpm: Double?
    var averageHeartRateBpm: Double?
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

    var totalCholesterolMgDl: Double?
    var hdlCholesterolMgDl: Double?

    // 0..100 functional score from in-app tests (optional in v1).
    var strengthTestScore: Double?
}
