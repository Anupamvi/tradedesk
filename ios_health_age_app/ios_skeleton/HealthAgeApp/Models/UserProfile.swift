import Foundation

enum BiologicalSex: String, Codable, CaseIterable {
    case female
    case male
    case other
}

struct UserProfile: Codable, Equatable {
    var chronologicalAgeYears: Double
    var sexAtBirth: BiologicalSex
    var isSmoker: Bool
    var hasDiabetes: Bool
    var onBPMedication: Bool

    static let placeholder = UserProfile(
        chronologicalAgeYears: 35,
        sexAtBirth: .male,
        isSmoker: false,
        hasDiabetes: false,
        onBPMedication: false
    )
}
