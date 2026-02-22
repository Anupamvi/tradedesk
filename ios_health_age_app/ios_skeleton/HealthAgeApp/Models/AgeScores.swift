import Foundation

struct AgeScoreComponent: Codable, Equatable {
    var ageYears: Double?
    var confidence: Double
    var drivers: [String]
}

struct AgeScores: Codable, Equatable {
    var generatedAt: Date
    var heart: AgeScoreComponent
    var metabolic: AgeScoreComponent
    var muscular: AgeScoreComponent
}
