import Foundation
import simd

/// One Euro filter (Casiez et al. 2012): heavy smoothing when still, low lag when moving.
public struct OneEuroFilter {
    public var minCutoff: Double
    public var beta: Double
    public var dCutoff: Double

    private var x: Double?
    private var dx: Double = 0
    private var lastTime: Double?

    public init(minCutoff: Double = 1.0, beta: Double = 0.3, dCutoff: Double = 1.0) {
        self.minCutoff = minCutoff
        self.beta = beta
        self.dCutoff = dCutoff
    }

    private static func alpha(_ cutoff: Double, _ dt: Double) -> Double {
        let tau = 1 / (2 * .pi * cutoff)
        return 1 / (1 + tau / dt)
    }

    public mutating func filter(_ value: Double, time: Double) -> Double {
        guard let prev = x, let t0 = lastTime, time > t0 else {
            x = value; lastTime = time; return value
        }
        let dt = time - t0
        let a_d = Self.alpha(dCutoff, dt)
        dx = a_d * ((value - prev) / dt) + (1 - a_d) * dx
        let a = Self.alpha(minCutoff + beta * abs(dx), dt)
        let out = a * value + (1 - a) * prev
        x = out; lastTime = time
        return out
    }

    public mutating func reset() { x = nil; lastTime = nil; dx = 0 }
}

/// Smooths every field of a FaceMeasurement independently.
public struct FaceSmoother {
    private var filters: [OneEuroFilter] = []
    private var scale: Float = 1
    private let minCutoff: Double, beta: Double

    public init(minCutoff: Double = 1.5, beta: Double = 0.5) {
        self.minCutoff = minCutoff
        self.beta = beta
    }

    public mutating func reset() { filters.removeAll() }

    public mutating func smooth(_ face: FaceMeasurement, time: Double) -> FaceMeasurement {
        // Pixel-valued fields are divided by the eye width seen when the filters
        // were created, so `beta` behaves the same at 720p and 4K.
        let fieldCount = face.eyes.count * EyeMeasurement.fieldCount + 1
        if filters.count != fieldCount {
            filters = (0..<fieldCount).map { _ in OneEuroFilter(minCutoff: minCutoff, beta: beta) }
            scale = max(face.eyes.map(\.width).reduce(0, +) / Float(face.eyes.count), 1)
        }
        let values = face.fields(scale: scale)
        let out = values.indices.map { Float(filters[$0].filter(Double(values[$0]), time: time)) }
        return FaceMeasurement(fields: out, eyeCount: face.eyes.count, scale: scale,
                               hasYaw: face.yawDegrees != nil)
    }
}
