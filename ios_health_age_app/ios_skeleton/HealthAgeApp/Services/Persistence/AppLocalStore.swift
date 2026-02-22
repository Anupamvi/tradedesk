import Foundation

actor AppLocalStore {
    private let fileManager = FileManager.default
    private let encoder: JSONEncoder
    private let decoder: JSONDecoder
    private let baseURL: URL
    private let calendar: Calendar

    init(baseURL: URL? = nil, calendar: Calendar = .current) {
        self.calendar = calendar
        let fm = FileManager.default

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        self.encoder = encoder

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        self.decoder = decoder

        if let baseURL {
            self.baseURL = baseURL
        } else {
            let appSupport = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            self.baseURL = (appSupport ?? fm.temporaryDirectory)
                .appendingPathComponent("HealthAgeApp", isDirectory: true)
        }
    }

    func loadUserProfile() throws -> UserProfile? {
        let url = userProfileURL()
        guard fileManager.fileExists(atPath: url.path) else {
            return nil
        }
        let data = try Data(contentsOf: url)
        return try decoder.decode(UserProfile.self, from: data)
    }

    func saveUserProfile(_ profile: UserProfile) throws {
        try ensureBaseDirectory()
        let data = try encoder.encode(profile)
        try data.write(to: userProfileURL(), options: .atomic)
    }

    func loadSnapshots() throws -> [DailyFeatureSnapshot] {
        let url = snapshotsURL()
        guard fileManager.fileExists(atPath: url.path) else {
            return []
        }
        let data = try Data(contentsOf: url)
        return try decoder.decode([DailyFeatureSnapshot].self, from: data)
    }

    func appendSnapshot(_ snapshot: DailyFeatureSnapshot, maxHistory: Int = 180) throws {
        var snapshots = try loadSnapshots()

        if let existingIndex = snapshots.firstIndex(where: { calendar.isDate($0.asOfDate, inSameDayAs: snapshot.asOfDate) }) {
            snapshots[existingIndex] = snapshot
        } else {
            snapshots.append(snapshot)
        }

        snapshots.sort { $0.asOfDate < $1.asOfDate }
        if snapshots.count > maxHistory {
            snapshots = Array(snapshots.suffix(maxHistory))
        }

        try ensureBaseDirectory()
        let data = try encoder.encode(snapshots)
        try data.write(to: snapshotsURL(), options: .atomic)
    }

    func clearAll() throws {
        let profileURL = userProfileURL()
        let snapshotsURL = snapshotsURL()

        if fileManager.fileExists(atPath: profileURL.path) {
            try fileManager.removeItem(at: profileURL)
        }
        if fileManager.fileExists(atPath: snapshotsURL.path) {
            try fileManager.removeItem(at: snapshotsURL)
        }
    }

    private func ensureBaseDirectory() throws {
        if !fileManager.fileExists(atPath: baseURL.path) {
            try fileManager.createDirectory(at: baseURL, withIntermediateDirectories: true)
        }
    }

    private func userProfileURL() -> URL {
        baseURL.appendingPathComponent("user_profile.json", isDirectory: false)
    }

    private func snapshotsURL() -> URL {
        baseURL.appendingPathComponent("daily_snapshots.json", isDirectory: false)
    }
}
