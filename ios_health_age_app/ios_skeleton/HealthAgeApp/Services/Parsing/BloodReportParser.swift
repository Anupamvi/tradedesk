import Foundation

struct ParsedBloodReport {
    var totalCholesterolMgDl: Double?
    var hdlCholesterolMgDl: Double?
    var ldlCholesterolMgDl: Double?
    var triglyceridesMgDl: Double?
    var hba1cPercent: Double?
    var fastingGlucoseMgDl: Double?
}

protocol BloodReportParsing {
    func parse(reportText: String) -> ParsedBloodReport
}

final class BloodReportParser: BloodReportParsing {
    func parse(reportText: String) -> ParsedBloodReport {
        ParsedBloodReport(
            totalCholesterolMgDl: parseValue(label: "total cholesterol", text: reportText),
            hdlCholesterolMgDl: parseValue(label: "hdl", text: reportText),
            ldlCholesterolMgDl: parseValue(label: "ldl", text: reportText),
            triglyceridesMgDl: parseValue(label: "triglycerides", text: reportText),
            hba1cPercent: parseValue(label: "hba1c", text: reportText),
            fastingGlucoseMgDl: parseValue(label: "fasting glucose", text: reportText)
        )
    }

    private func parseValue(label: String, text: String) -> Double? {
        let lower = text.lowercased()
        guard let labelRange = lower.range(of: label) else {
            return nil
        }

        let tail = String(lower[labelRange.upperBound...])
        let regex = try? NSRegularExpression(pattern: #"([0-9]+(\.[0-9]+)?)"#)
        let range = NSRange(tail.startIndex..<tail.endIndex, in: tail)
        guard
            let match = regex?.firstMatch(in: tail, options: [], range: range),
            let valueRange = Range(match.range(at: 1), in: tail)
        else {
            return nil
        }
        return Double(String(tail[valueRange]))
    }
}
