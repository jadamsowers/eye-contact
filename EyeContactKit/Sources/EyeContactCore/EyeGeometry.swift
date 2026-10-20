import simd

/// A lid curve in eye-local coordinates (x along the corner axis, y down, both in
/// eye widths, origin midway between the corners): y = (x² − ¼)(a + b·x).
/// It passes through both corners; `a` sets the arch and `b` the tilt.
public struct LidCurve {
    public var a: Float
    public var b: Float

    public init(a: Float, b: Float) { self.a = a; self.b = b }

    public func y(_ x: Float) -> Float { (x * x - 0.25) * (a + b * x) }

    /// Least-squares fit through local points (corners excluded).
    public static func fit(_ pts: [SIMD2<Float>]) -> LidCurve? {
        // Minimize Σ (y − f·a − f·x·b)² with f = x² − ¼.
        var s11: Float = 0, s12: Float = 0, s22: Float = 0, t1: Float = 0, t2: Float = 0
        for p in pts {
            let f = p.x * p.x - 0.25, g = f * p.x
            s11 += f * f; s12 += f * g; s22 += g * g
            t1 += f * p.y; t2 += g * p.y
        }
        let det = s11 * s22 - s12 * s12
        if abs(det) > 1e-9 {
            return LidCurve(a: (t1 * s22 - t2 * s12) / det, b: (s11 * t2 - s12 * t1) / det)
        }
        // A single point only constrains the arch.
        return s11 > 1e-9 ? LidCurve(a: t1 / s11, b: 0) : nil
    }
}

/// One eye, measured in pixel coordinates with a top-left origin.
public struct EyeMeasurement {
    /// Midpoint between the two eye corners.
    public var mid: SIMD2<Float>
    /// Unit vector from corner to corner, pointing toward image-right.
    public var axis: SIMD2<Float>
    /// Corner-to-corner distance.
    public var width: Float
    public var upperLid: LidCurve
    public var lowerLid: LidCurve
    /// Iris/pupil center.
    public var iris: SIMD2<Float>

    /// Perpendicular to `axis`, pointing down in image space.
    public var normal: SIMD2<Float> { SIMD2(-axis.y, axis.x) }

    /// Opening height (lid to lid) at the middle of the eye.
    public var height: Float { (lowerLid.y(0) - upperLid.y(0)) * width }

    /// Middle of the opening, the reference point for gaze.
    public var center: SIMD2<Float> { mid + normal * ((upperLid.y(0) + lowerLid.y(0)) / 2 * width) }

    /// Height / width. Drops toward 0 during a blink.
    public var openness: Float { width > 0 ? height / width : 0 }

    /// Iris position relative to the eye center, in units of eye width.
    /// (0, 0) means the iris is centered in the opening.
    public var gazeOffset: SIMD2<Float> {
        let d = iris - center
        return SIMD2(simd_dot(d, axis), simd_dot(d, normal)) / max(width, 1)
    }
}

/// Both eyes plus head pose for one frame.
public struct FaceMeasurement {
    public var eyes: [EyeMeasurement]
    /// Head yaw in degrees, if Vision reported it.
    public var yawDegrees: Float?
    /// Head roll in degrees, if Vision reported it. About ±180 means the image was upside down.
    public var rollDegrees: Float? = nil
}

/// User-facing controls. These map 1:1 to plugin parameters later.
public struct CorrectionSettings {
    /// 0 = off, 1 = move the iris fully to the target.
    public var strength: Float = 1.0
    /// Where "looking at the camera" sits inside the eye, in eye widths.
    /// Positive y moves the target down.
    public var target: SIMD2<Float> = .zero
    /// Offsets larger than this (in eye widths) are treated as an intentional
    /// look-away and faded out instead of corrected.
    public var maxOffset: Float = 0.18
    /// Openness below this is treated as a blink and the correction fades out.
    public var blinkThreshold: Float = 0.18
    /// Head yaw (degrees) beyond which the correction fades out.
    public var maxYaw: Float = 30
    /// Iris radius as a fraction of eye width.
    public var irisRadius: Float = 0.21
    /// Draw tracking overlay instead of a clean result.
    public var debug: Bool = false

    public init() {}
}

/// GPU-side description of one eye's warp. Layout must match `EyeParams` in the shader.
public struct EyeWarp {
    public var mid: SIMD2<Float> = .zero
    public var axis: SIMD2<Float> = SIMD2(1, 0)
    public var iris: SIMD2<Float> = .zero
    public var shift: SIMD2<Float> = .zero
    public var width: Float = 0
    public var irisRadius: Float = 0
    public var upperA: Float = 0
    public var upperB: Float = 0
    public var lowerA: Float = 0
    public var lowerB: Float = 0
    /// 0...1 blend of the effect for this eye.
    public var amount: Float = 0
    private var pad: Float = 0

    public init() {}
}

private func smoothstep(_ e0: Float, _ e1: Float, _ x: Float) -> Float {
    let t = simd_clamp((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)
}

/// Turns a (smoothed) face measurement into per-eye warps.
public func makeWarps(for face: FaceMeasurement, settings: CorrectionSettings) -> [EyeWarp] {
    guard !face.eyes.isEmpty else { return [] }

    // Eyes move together, so drive both from the average offset. This also
    // hides per-eye tracking noise.
    let meanOffset = face.eyes.map(\.gazeOffset).reduce(.zero, +) / Float(face.eyes.count)
    let error = meanOffset - settings.target
    let errorLen = simd_length(error)

    // Fade out for look-aways, blinks and turned heads rather than forcing them.
    var amount = settings.strength
    amount *= 1 - smoothstep(settings.maxOffset, settings.maxOffset * 1.5, errorLen)
    let minOpen = face.eyes.map(\.openness).min() ?? 0
    amount *= smoothstep(settings.blinkThreshold * 0.7, settings.blinkThreshold, minOpen)
    if let yaw = face.yawDegrees {
        amount *= 1 - smoothstep(settings.maxYaw * 0.7, settings.maxYaw, abs(yaw))
    }

    return face.eyes.map { eye in
        var w = EyeWarp()
        w.mid = eye.mid
        w.axis = eye.axis
        w.width = eye.width
        w.upperA = eye.upperLid.a; w.upperB = eye.upperLid.b
        w.lowerA = eye.lowerLid.a; w.lowerB = eye.lowerLid.b
        w.iris = eye.iris
        // Same correction in eye-local units, converted back to this eye's pixels.
        let local = -error * eye.width
        w.shift = eye.axis * local.x + eye.normal * local.y
        w.irisRadius = eye.width * settings.irisRadius
        w.amount = amount
        return w
    }
}
