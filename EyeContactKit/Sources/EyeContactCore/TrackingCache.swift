import Foundation
import simd

extension EyeMeasurement {
    static let fieldCount = 11

    /// Flat field list for averaging. Positions are divided by `scale`; the iris is stored
    /// relative to the eye so head motion doesn't drag it.
    func fields(scale k: Float) -> [Float] {
        let off = gazeOffset
        return [mid.x / k, mid.y / k, axis.x, axis.y, width / k,
                upperLid.a, upperLid.b, lowerLid.a, lowerLid.b, off.x, off.y]
    }

    init(fields f: ArraySlice<Float>, scale k: Float) {
        let f = Array(f)
        let ax = SIMD2(f[2], f[3])
        self.init(mid: SIMD2(f[0], f[1]) * k, axis: ax / max(simd_length(ax), 1e-6), width: f[4] * k,
                  upperLid: LidCurve(a: f[5], b: f[6]), lowerLid: LidCurve(a: f[7], b: f[8]), iris: .zero)
        iris = center + (axis * f[9] + normal * f[10]) * width
    }
}

extension FaceMeasurement {
    func fields(scale: Float) -> [Float] {
        eyes.flatMap { $0.fields(scale: scale) } + [(yawDegrees ?? 0) / 45]
    }

    init(fields f: [Float], eyeCount: Int, scale: Float, hasYaw: Bool) {
        let n = EyeMeasurement.fieldCount
        let eyes = (0..<eyeCount).map { EyeMeasurement(fields: f[($0 * n)..<($0 * n + n)], scale: scale) }
        self.init(eyes: eyes, yawDegrees: hasYaw ? f[eyeCount * n] * 45 : nil)
    }

    /// Multiplies all pixel coordinates by `s` (e.g. tracking buffer → full-size frame).
    public func scaled(_ s: Float) -> FaceMeasurement {
        var r = self
        for i in r.eyes.indices {
            r.eyes[i].mid *= s
            r.eyes[i].width *= s
            r.eyes[i].iris *= s
        }
        return r
    }

    /// Offsets all pixel coordinates (e.g. crop -> full frame).
    public func translated(by o: SIMD2<Float>) -> FaceMeasurement {
        var r = self
        for i in r.eyes.indices {
            r.eyes[i].mid += o
            r.eyes[i].iris += o
        }
        return r
    }

    /// Mirrors top-to-bottom in a frame `height` pixels tall (for bottom-left-origin images).
    public func flippedVertically(height: Float) -> FaceMeasurement {
        var r = self
        for i in r.eyes.indices {
            var e = r.eyes[i]
            e.mid.y = height - e.mid.y
            e.iris.y = height - e.iris.y
            e.axis.y = -e.axis.y
            // Local y flips sign, so the upper lid becomes the (negated) lower one and vice versa.
            let upper = e.upperLid
            e.upperLid = LidCurve(a: -e.lowerLid.a, b: -e.lowerLid.b)
            e.lowerLid = LidCurve(a: -upper.a, b: -upper.b)
            r.eyes[i] = e
        }
        return r
    }
}

/// Raw per-frame measurements keyed by time, smoothed on read with a centered Gaussian window
/// over whichever neighbors have been measured. Final Cut renders frames out of order and at
/// varying resolutions, so entries are stored relative to frame width and never depend on
/// render order. Thread-safe.
public final class TrackingCache {
    private struct Entry {
        var fields: [Float]?  // nil = measured, no face
        var eyeCount: Int
        var hasYaw: Bool
    }

    private var entries: [Int64: Entry] = [:]  // key: time in microseconds
    private let lock = NSLock()
    private let capacity: Int

    public init(capacity: Int = 2000) { self.capacity = capacity }

    private static func key(_ t: Double) -> Int64 { Int64((t * 1_000_000).rounded()) }

    public func contains(time: Double) -> Bool {
        lock.lock(); defer { lock.unlock() }
        return entries[Self.key(time)] != nil
    }

    /// Store a raw measurement measured on a frame `frameWidth` pixels wide.
    public func store(_ face: FaceMeasurement?, time: Double, frameWidth: Float) {
        let entry = face.map { Entry(fields: $0.fields(scale: frameWidth), eyeCount: $0.eyes.count,
                                     hasYaw: $0.yawDegrees != nil) }
            ?? Entry(fields: nil, eyeCount: 0, hasYaw: false)
        lock.lock(); defer { lock.unlock() }
        entries[Self.key(time)] = entry
        if entries.count > capacity {
            // Evict the entries farthest from the frame we just stored.
            let k = Self.key(time)
            let far = entries.keys.sorted { abs($0 - k) > abs($1 - k) }.prefix(entries.count - capacity)
            for f in far { entries.removeValue(forKey: f) }
        }
    }

    /// Smoothed measurement at `time` in pixels of a frame `frameWidth` wide, or nil if the
    /// frame at `time` has no face. `sigma` is the Gaussian width in seconds (0 = raw).
    public func smoothed(time: Double, sigma: Double, frameWidth: Float) -> FaceMeasurement? {
        lock.lock(); defer { lock.unlock() }
        let k0 = Self.key(time)
        guard let center = entries[k0], let centerFields = center.fields else { return nil }
        guard sigma > 0 else {
            return FaceMeasurement(fields: centerFields, eyeCount: center.eyeCount,
                                   scale: frameWidth, hasYaw: center.hasYaw)
        }

        let radius = Self.key(sigma * 2.5)
        var sum = [Float](repeating: 0, count: centerFields.count)
        var wsum: Float = 0
        for (k, e) in entries where abs(k - k0) <= radius {
            // Only average frames with the same face layout; a gap (no face) is skipped.
            guard let f = e.fields, e.eyeCount == center.eyeCount, e.hasYaw == center.hasYaw else { continue }
            let dt = Double(k - k0) / 1_000_000
            let w = Float(exp(-dt * dt / (2 * sigma * sigma)))
            for i in sum.indices { sum[i] += f[i] * w }
            wsum += w
        }
        return FaceMeasurement(fields: sum.map { $0 / wsum }, eyeCount: center.eyeCount,
                               scale: frameWidth, hasYaw: center.hasYaw)
    }
}
