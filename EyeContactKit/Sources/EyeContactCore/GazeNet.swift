import CoreML
import Foundation
import simd

// Learned gaze redirection (gazenet/ in the repo): a small network predicts a warp field for each
// 144x96 eye patch, which is applied to the full-resolution frame. This file holds everything
// between Vision's measurements and the network: the eye frame the network was trained with
// (MediaPipe's), the patch geometry, gaze targets and the Core ML call.

public enum GazePatch {
    public static let width = 144
    public static let height = 96
    /// Corner-to-corner distance as a fraction of the patch width (gazenet/eyepatch.py).
    public static let eyeFraction: Float = 0.55
    /// 95th percentile of iris displacements in the training pairs, eye widths. Beyond this the
    /// network extrapolates (e.g. closes the lids), so requests are soft-limited to it.
    public static let trainedRange = SIMD2<Float>(0.14, 0.05)
}

/// How a particular trained model expects eyes to be framed and gaze to be measured.
public struct GazeProfile {
    public let name: String
    /// Eye corners used for the patch, in Vision's eye frame (eye widths; x mirrored for the
    /// image-left eye so "inner" is negative for both). Index = side (0 = image-left eye).
    let inner: [SIMD2<Float>]
    let outer: [SIMD2<Float>]
    /// gaze = scale * (Vision iris offset) + bias: puts Vision's noisier iris estimate on the
    /// MediaPipe scale the displacement labels were measured in.
    let gazeScale: SIMD2<Float>
    let gazeBias: SIMD2<Float>
    /// Gaze offset of Columbia Gaze subjects looking into the lens, in this profile's frame.
    public let eyeContactTarget: SIMD2<Float>

    /// v2: trained on MediaPipe-framed patches; Vision's corners are shifted to MediaPipe's
    /// (gazenet/calibrate_vision.py). Mismatches on turned heads.
    public static let v2 = GazeProfile(
        name: "v2", inner: [SIMD2(-0.509, 0.009), SIMD2(-0.498, 0.013)],
        outer: [SIMD2(0.530, 0.012), SIMD2(0.533, 0.004)],
        gazeScale: SIMD2(0.795, 0.639), gazeBias: SIMD2(0.0066, -0.0055), eyeContactTarget: SIMD2(0, -0.07))

    /// v3: trained on Vision-framed patches and masks (gazenet/prepare_vision.py), so the
    /// plug-in's measurements are used as-is.
    public static let v3 = GazeProfile(
        name: "v3", inner: [SIMD2(-0.5, 0), SIMD2(-0.5, 0)], outer: [SIMD2(0.5, 0), SIMD2(0.5, 0)],
        gazeScale: SIMD2(0.877, 0.760), gazeBias: SIMD2(0.0066, 0.0111), eyeContactTarget: SIMD2(0.0111, -0.0693))

    /// v3 with the iris from the iris locator (FaceMeasurer), which is trained on MediaPipe's
    /// iris, so offsets are already on the MediaPipe scale.
    public static let v3Iris = GazeProfile(
        name: "v3+iris", inner: v3.inner, outer: v3.outer,
        gazeScale: SIMD2(1, 1), gazeBias: .zero, eyeContactTarget: v3.eyeContactTarget)
}

/// One eye's patch: the affine from frame pixels to patch pixels and back.
public struct EyePatchFrame {
    public let side: Int               // 0 = image-left eye (mirrored in the patch)
    public let eye: EyeMeasurement     // Vision measurement, frame pixels
    public let inner: SIMD2<Float>     // MediaPipe-equivalent corners, frame pixels
    public let outer: SIMD2<Float>
    /// Frame -> patch: q = A * p + t (patch pixel j has its center at j).
    public let A: simd_float2x2
    public let t: SIMD2<Float>

    public init(eye: EyeMeasurement, side: Int, profile: GazeProfile) {
        self.side = side
        self.eye = eye
        let sx: Float = side == 0 ? -1 : 1
        let toFrame = { (q: SIMD2<Float>) in eye.mid + (eye.axis * (q.x * sx) + eye.normal * q.y) * eye.width }
        inner = toFrame(profile.inner[side])
        outer = toFrame(profile.outer[side])

        let width = max(simd_distance(inner, outer), 1e-3)
        let d = (outer - inner) / width
        let s = GazePatch.eyeFraction * Float(GazePatch.width) / width
        // Rows of the linear part. Side 1 rotates d onto +x; side 0 reflects (mirrors the eye)
        // while keeping image-down as patch-down.
        let r0 = SIMD2(d.x, d.y) * s
        let r1 = (side == 0 ? SIMD2(d.y, -d.x) : SIMD2(-d.y, d.x)) * s
        A = simd_float2x2(rows: [r0, r1])
        t = SIMD2(Float(GazePatch.width) / 2, Float(GazePatch.height) / 2) - A * ((inner + outer) / 2)
    }

    public func toPatch(_ p: SIMD2<Float>) -> SIMD2<Float> { A * p + t }
    public func toFrame(_ q: SIMD2<Float>) -> SIMD2<Float> { A.inverse * (q - t) }

    /// Iris offset from the corner midpoint (x toward image-right, y down, eye widths), before
    /// the profile's gaze regression.
    var rawOffset: SIMD2<Float> {
        let (a, b) = inner.x < outer.x ? (inner, outer) : (outer, inner)
        let w = max(simd_distance(a, b), 1e-3)
        let axis = (b - a) / w
        let normal = SIMD2(-axis.y, axis.x)
        let d = eye.iris - (a + b) / 2
        return SIMD2(simd_dot(d, axis), simd_dot(d, normal)) / w
    }

    /// Binary eye-opening mask in patch space, matching training (contour polygon grown by
    /// 3 patch pixels). Row-major, height x width.
    public func mask() -> [Float] {
        let W = GazePatch.width, H = GazePatch.height
        let Ainv = A.inverse
        let s = GazePatch.eyeFraction * Float(W) / max(simd_distance(inner, outer), 1e-3)
        var m = [Float](repeating: 0, count: W * H)
        for y in 0..<H {
            for x in 0..<W {
                let p = Ainv * (SIMD2(Float(x), Float(y)) - t)
                let d = p - eye.mid
                let lx = simd_dot(d, eye.axis) / eye.width
                let ly = simd_dot(d, eye.normal) / eye.width
                let inside = min(min(ly - eye.upperLid.y(lx), eye.lowerLid.y(lx) - ly), 0.5 - abs(lx))
                if inside * eye.width * s >= -3 { m[y * W + x] = 1 }
            }
        }
        return m
    }
}

public struct GazeSettings {
    public var strength: Float = 1
    /// Adjustments to the eye-contact target, eye widths (+y = lower).
    public var targetOffset: SIMD2<Float> = .zero
    /// Scale for the vertical part of the correction (vertical gaze is measured less reliably).
    public var vertical: Float = 1
    /// Gaze errors beyond this (eye widths) are treated as deliberate look-aways and left alone.
    public var lookAwayLimit: Float = 0.3
    public var blinkThreshold: Float = 0.18
    public var maxYaw: Float = 30
    public var debug = false

    public init() {}
}

private func smoothstep(_ e0: Float, _ e1: Float, _ x: Float) -> Float {
    let t = simd_clamp((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)
}

/// What to do with a (smoothed) face this frame.
public struct GazeRequest {
    public var patches: [EyePatchFrame]     // ordered by side
    /// Desired iris displacement per eye in patch orientation (eye widths) — the network's input.
    public var displacement: [SIMD2<Float>]
    /// Mean gaze offset in the MediaPipe convention, and the overall blend amount (for debugging).
    public var gaze: SIMD2<Float>
    public var amount: Float
}

public func makeGazeRequest(for face: FaceMeasurement, settings: GazeSettings,
                            profile: GazeProfile) -> GazeRequest? {
    guard face.eyes.count == 2 else { return nil }
    let ordered = face.eyes.sorted { $0.mid.x < $1.mid.x }
    let patches = ordered.enumerated().map { EyePatchFrame(eye: $0.element, side: $0.offset, profile: profile) }

    let raw = (patches[0].rawOffset + patches[1].rawOffset) / 2
    let gaze = profile.gazeScale * raw + profile.gazeBias
    let error = gaze - (profile.eyeContactTarget + settings.targetOffset)

    var amount = settings.strength
    amount *= 1 - smoothstep(settings.lookAwayLimit, settings.lookAwayLimit * 1.4, simd_length(error))
    let minOpen = ordered.map(\.openness).min() ?? 0
    amount *= smoothstep(settings.blinkThreshold * 0.7, settings.blinkThreshold, minOpen)
    if let yaw = face.yawDegrees {
        amount *= 1 - smoothstep(settings.maxYaw * 0.7, settings.maxYaw, abs(yaw))
    }

    var delta = -error * amount
    delta.y *= settings.vertical
    let r = GazePatch.trainedRange
    delta = SIMD2(r.x * tanh(delta.x / r.x), r.y * tanh(delta.y / r.y))

    // The image-left eye is mirrored in its patch, so its horizontal motion flips sign.
    let displacement = patches.map { $0.side == 0 ? SIMD2(-delta.x, delta.y) : delta }
    return GazeRequest(patches: patches, displacement: displacement, gaze: gaze, amount: amount)
}

/// Network output for both eyes at half patch resolution (72x48), channel-major per eye.
public struct GazeFields {
    public static let width = GazePatch.width / 2
    public static let height = GazePatch.height / 2
    public var flow: [Float]   // [2 eyes][2][48][72], normalized patch coordinates
    public var alpha: [Float]  // [2 eyes][1][48][72]
    public var color: [Float]  // [2 eyes][3][48][72]
}

/// Core ML wrapper for GazeWarp.mlpackage / .mlmodelc.
public final class GazeModel {
    private let model: MLModel

    /// Accepts a compiled .mlmodelc or an .mlpackage (compiled on first load).
    public init(url: URL) throws {
        let compiled = url.pathExtension == "mlmodelc" ? url : try MLModel.compileModel(at: url)
        let config = MLModelConfiguration()
        config.computeUnits = .all
        model = try MLModel(contentsOf: compiled, configuration: config)
    }

    /// `patches`: [2][3][96][144] RGB 0...1, `masks`: [2][96][144], `displacement`: per eye.
    public func predict(patches: [Float], masks: [Float], displacement: [SIMD2<Float>]) throws -> GazeFields {
        let W = GazePatch.width, H = GazePatch.height
        func array(_ values: [Float], _ shape: [Int]) throws -> MLMultiArray {
            let a = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
            values.withUnsafeBufferPointer { src in
                a.dataPointer.assumingMemoryBound(to: Float.self).update(from: src.baseAddress!, count: values.count)
            }
            return a
        }
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "image": try array(patches, [2, 3, H, W]),
            "mask": try array(masks, [2, 1, H, W]),
            "displacement": try array(displacement.flatMap { [$0.x, $0.y] }, [2, 2]),
        ])
        let out = try model.prediction(from: input)
        func floats(_ name: String) -> [Float] {
            guard let a = out.featureValue(for: name)?.multiArrayValue else { return [] }
            // Outputs can be strided or Float16 depending on the compute unit; MLShapedArray
            // gives contiguous logical-order Float32 either way.
            return MLShapedArray<Float>(a).scalars
        }
        return GazeFields(flow: floats("flow"), alpha: floats("alpha"), color: floats("color"))
    }
}
