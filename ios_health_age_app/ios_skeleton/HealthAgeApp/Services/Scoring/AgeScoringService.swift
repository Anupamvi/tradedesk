import Foundation

protocol AgeScoringServing {
    func generateScores(snapshot: DailyFeatureSnapshot, featureVector: FeatureVector?) -> AgeScores
}

final class AgeScoringService: AgeScoringServing {
    func generateScores(snapshot: DailyFeatureSnapshot, featureVector: FeatureVector?) -> AgeScores {
        AgeScores(
            generatedAt: Date(),
            heart: heartComponent(snapshot: snapshot, featureVector: featureVector),
            metabolic: metabolicComponent(snapshot: snapshot, featureVector: featureVector),
            muscular: muscularComponent(snapshot: snapshot, featureVector: featureVector)
        )
    }

    private func heartComponent(snapshot: DailyFeatureSnapshot, featureVector: FeatureVector?) -> AgeScoreComponent {
        let profile = snapshot.profile
        let chronologicalAge = profile.chronologicalAgeYears
        var drivers: [String] = []

        if
            let totalCholesterol = snapshot.totalCholesterolMgDl,
            let hdl = snapshot.hdlCholesterolMgDl,
            let systolic = snapshot.systolicBPMmHg
        {
            let risk = framinghamRisk(
                age: chronologicalAge,
                sex: profile.sexAtBirth,
                totalCholesterol: totalCholesterol,
                hdl: hdl,
                systolicBP: systolic,
                onBPMedication: profile.onBPMedication,
                smoker: profile.isSmoker,
                diabetes: profile.hasDiabetes
            )
            let heartAge = equivalentHeartAge(for: risk, sex: profile.sexAtBirth)

            if let hrv = snapshot.hrvSDNNMs, hrv < 35 {
                drivers.append("Low HRV trend")
            }
            if let rhr = snapshot.restingHeartRateBpm, rhr > 70 {
                drivers.append("Elevated resting heart rate")
            }
            if systolic > 130 {
                drivers.append("Higher systolic blood pressure")
            }
            if drivers.isEmpty {
                drivers.append("Cardiovascular inputs are near baseline")
            }

            return AgeScoreComponent(
                ageYears: rounded(heartAge),
                confidence: 0.9,
                drivers: drivers
            )
        }

        // Proxy model for periods without lipid labs.
        var ageDelta = 0.0
        var availableInputs = 0

        if let rhr = snapshot.restingHeartRateBpm {
            ageDelta += normalizedHigherIsWorse(value: rhr, target: 60, scale: 10) * 5
            availableInputs += 1
            if rhr > 70 {
                drivers.append("Resting heart rate above target")
            }
        }
        if let hrv = snapshot.hrvSDNNMs {
            ageDelta += normalizedLowerIsWorse(value: hrv, target: 55, scale: 20) * 4
            availableInputs += 1
            if hrv < 40 {
                drivers.append("Low HRV for recovery")
            }
        }
        if let systolic = snapshot.systolicBPMmHg {
            ageDelta += normalizedHigherIsWorse(value: systolic, target: 120, scale: 12) * 4
            availableInputs += 1
            if systolic > 130 {
                drivers.append("Systolic blood pressure elevated")
            }
        }
        if let sleep = snapshot.sleepHours {
            ageDelta += normalizedLowerIsWorse(value: sleep, target: 7.5, scale: 1.5) * 2
            availableInputs += 1
            if sleep < 6.5 {
                drivers.append("Sleep duration below target")
            }
        }
        if let vo2 = snapshot.vo2MaxMlKgMin {
            ageDelta += normalizedLowerIsWorse(value: vo2, target: 42, scale: 9) * 3
            availableInputs += 1
            if vo2 < 35 {
                drivers.append("Lower cardio fitness")
            }
        }

        if drivers.isEmpty {
            drivers.append("Limited inputs available; confidence reduced")
        }

        let proxyAge = clamp(chronologicalAge + ageDelta, min: 18, max: 95)
        let confidence = clamp(Double(availableInputs) / 6.0, min: 0.25, max: 0.75)

        return AgeScoreComponent(
            ageYears: rounded(proxyAge),
            confidence: confidence,
            drivers: drivers
        )
    }

    private func metabolicComponent(snapshot: DailyFeatureSnapshot, featureVector: FeatureVector?) -> AgeScoreComponent {
        let chronologicalAge = snapshot.profile.chronologicalAgeYears
        var ageDelta = 0.0
        var availableInputs = 0
        var drivers: [String] = []

        if let glucose = snapshot.averageGlucoseMgDl {
            ageDelta += normalizedHigherIsWorse(value: glucose, target: 95, scale: 15) * 6
            availableInputs += 1
            if glucose > 105 {
                drivers.append("Higher average glucose")
            }
        }
        if let sleep = snapshot.sleepHours {
            ageDelta += normalizedLowerIsWorse(value: sleep, target: 7.5, scale: 1.5) * 3
            availableInputs += 1
            if sleep < 6.5 {
                drivers.append("Insufficient sleep")
            }
        }
        if let exercise = snapshot.exerciseMinutes {
            ageDelta += normalizedLowerIsWorse(value: exercise, target: 35, scale: 20) * 3
            availableInputs += 1
            if exercise < 20 {
                drivers.append("Low exercise minutes")
            }
        }
        if let rhr = snapshot.restingHeartRateBpm {
            ageDelta += normalizedHigherIsWorse(value: rhr, target: 60, scale: 10) * 2
            availableInputs += 1
            if rhr > 70 {
                drivers.append("High resting heart rate")
            }
        }
        if let hrv = snapshot.hrvSDNNMs {
            ageDelta += normalizedLowerIsWorse(value: hrv, target: 55, scale: 20) * 2
            availableInputs += 1
            if hrv < 40 {
                drivers.append("Low HRV recovery signal")
            }
        }
        if let activityDelta = featureVector?.exerciseDelta, activityDelta > 0 {
            ageDelta -= min(activityDelta / 20.0, 1.0)
            drivers.append("Exercise trend improving")
        }

        if drivers.isEmpty {
            drivers.append("Metabolic inputs are near baseline")
        }

        let metabolicAge = clamp(chronologicalAge + ageDelta, min: 18, max: 95)
        let confidence = clamp(Double(availableInputs) / 6.0, min: 0.25, max: 0.9)

        return AgeScoreComponent(
            ageYears: rounded(metabolicAge),
            confidence: confidence,
            drivers: drivers
        )
    }

    private func muscularComponent(snapshot: DailyFeatureSnapshot, featureVector: FeatureVector?) -> AgeScoreComponent {
        let chronologicalAge = snapshot.profile.chronologicalAgeYears
        var ageDelta = 0.0
        var availableInputs = 0
        var drivers: [String] = []

        if let vo2 = snapshot.vo2MaxMlKgMin {
            ageDelta += normalizedLowerIsWorse(value: vo2, target: 42, scale: 9) * 5
            availableInputs += 1
            if vo2 < 35 {
                drivers.append("Cardio fitness below target")
            }
        }
        if let workoutMinutes = snapshot.workoutMinutes {
            ageDelta += normalizedLowerIsWorse(value: workoutMinutes, target: 30, scale: 15) * 4
            availableInputs += 1
            if workoutMinutes < 20 {
                drivers.append("Low workout minutes")
            }
        }
        if let steps = snapshot.steps {
            ageDelta += normalizedLowerIsWorse(value: steps, target: 8000, scale: 3000) * 2
            availableInputs += 1
            if steps < 6000 {
                drivers.append("Low daily movement")
            }
        }
        if let strengthScore = snapshot.strengthTestScore {
            ageDelta += normalizedLowerIsWorse(value: strengthScore, target: 70, scale: 20) * 5
            availableInputs += 1
            if strengthScore < 60 {
                drivers.append("Functional strength below target")
            }
        }
        if let vo2Delta = featureVector?.vo2Delta, vo2Delta > 0 {
            ageDelta -= min(vo2Delta / 4.0, 1.5)
            drivers.append("VO2 max trend improving")
        }

        if drivers.isEmpty {
            drivers.append("Add a functional strength test to improve precision")
        }

        let muscularAge = clamp(chronologicalAge + ageDelta, min: 18, max: 95)
        let confidence = clamp(Double(availableInputs) / 5.0, min: 0.2, max: 0.9)

        return AgeScoreComponent(
            ageYears: rounded(muscularAge),
            confidence: confidence,
            drivers: drivers
        )
    }

    private func framinghamRisk(
        age: Double,
        sex: BiologicalSex,
        totalCholesterol: Double,
        hdl: Double,
        systolicBP: Double,
        onBPMedication: Bool,
        smoker: Bool,
        diabetes: Bool
    ) -> Double {
        let coefficients = coefficientsForSex(sex)
        let ageTerm = coefficients.age * log(Swift.max(age, 1.0))
        let tcTerm = coefficients.totalCholesterol * log(Swift.max(totalCholesterol, 1.0))
        let hdlTerm = coefficients.hdl * log(Swift.max(hdl, 1.0))
        let sbpCoeff = onBPMedication ? coefficients.sbpTreated : coefficients.sbpUntreated
        let sbpTerm = sbpCoeff * log(Swift.max(systolicBP, 1.0))
        let smokerTerm = smoker ? coefficients.smoker : 0
        let diabetesTerm = diabetes ? coefficients.diabetes : 0

        let sum = ageTerm + tcTerm + hdlTerm + sbpTerm + smokerTerm + diabetesTerm
        let exponent = exp(sum - coefficients.mean)
        let risk = 1.0 - pow(coefficients.baselineSurvival, exponent)
        return clamp(risk, min: 0.001, max: 0.8)
    }

    private func equivalentHeartAge(for risk: Double, sex: BiologicalSex) -> Double {
        var low = 20.0
        var high = 90.0

        for _ in 0..<40 {
            let mid = (low + high) / 2.0
            let midRisk = framinghamRisk(
                age: mid,
                sex: sex,
                totalCholesterol: 180,
                hdl: 50,
                systolicBP: 125,
                onBPMedication: false,
                smoker: false,
                diabetes: false
            )
            if midRisk < risk {
                low = mid
            } else {
                high = mid
            }
        }

        return (low + high) / 2.0
    }

    private func normalizedHigherIsWorse(value: Double, target: Double, scale: Double) -> Double {
        (value - target) / scale
    }

    private func normalizedLowerIsWorse(value: Double, target: Double, scale: Double) -> Double {
        (target - value) / scale
    }

    private func rounded(_ value: Double?) -> Double? {
        guard let value else {
            return nil
        }
        return (value * 10).rounded() / 10
    }

    private func clamp(_ value: Double, min lower: Double, max upper: Double) -> Double {
        Swift.max(lower, Swift.min(value, upper))
    }

    private func coefficientsForSex(_ sex: BiologicalSex) -> FraminghamCoefficients {
        switch sex {
        case .female:
            return FraminghamCoefficients(
                age: 2.32888,
                totalCholesterol: 1.20904,
                hdl: -0.70833,
                sbpTreated: 2.82263,
                sbpUntreated: 2.76157,
                smoker: 0.52873,
                diabetes: 0.69154,
                baselineSurvival: 0.95012,
                mean: 26.1931
            )
        case .male, .other:
            return FraminghamCoefficients(
                age: 3.06117,
                totalCholesterol: 1.1237,
                hdl: -0.93263,
                sbpTreated: 1.99881,
                sbpUntreated: 1.93303,
                smoker: 0.65451,
                diabetes: 0.57367,
                baselineSurvival: 0.88936,
                mean: 23.9802
            )
        }
    }
}

private struct FraminghamCoefficients {
    var age: Double
    var totalCholesterol: Double
    var hdl: Double
    var sbpTreated: Double
    var sbpUntreated: Double
    var smoker: Double
    var diabetes: Double
    var baselineSurvival: Double
    var mean: Double
}
