import CoreVideo
import Vision
import simd

/// Finds the largest face in a frame and measures both eyes.
public final class EyeTracker {
    private let request: VNDetectFaceLandmarksRequest

    public init() {
        request = VNDetectFaceLandmarksRequest()
        request.revision = VNDetectFaceLandmarksRequestRevision3
        request.constellation = .constellation76Points
    }

    /// `pixelBuffer` must be 32BGRA if `refinePupils` is true.
    public func measure(_ pixelBuffer: CVPixelBuffer, refinePupils: Bool = true) -> FaceMeasurement? {
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up)
        do { try handler.perform([request]) } catch { return nil }

        guard let face = request.results?.max(by: {
            $0.boundingBox.width * $0.boundingBox.height < $1.boundingBox.width * $1.boundingBox.height
        }), let lm = face.landmarks else { return nil }

        let w = CVPixelBufferGetWidth(pixelBuffer)
        let h = CVPixelBufferGetHeight(pixelBuffer)
        let size = CGSize(width: w, height: h)

        // Vision uses a bottom-left origin; flip to top-left to match Metal.
        func pts(_ r: VNFaceLandmarkRegion2D?) -> [SIMD2<Float>] {
            guard let r else { return [] }
            return r.pointsInImage(imageSize: size).map { SIMD2(Float($0.x), Float(h) - Float($0.y)) }
        }

        var eyes: [EyeMeasurement] = []
        for (contour, pupil) in [(pts(lm.leftEye), pts(lm.leftPupil)), (pts(lm.rightEye), pts(lm.rightPupil))] {
            guard var eye = Self.eye(contour: contour, pupil: pupil.first) else { continue }
            if refinePupils, let refined = Self.darkCentroid(in: pixelBuffer, eye: eye) {
                // Vision's pupil point is often biased toward the eye center;
                // lean on the image-based estimate but keep Vision as an anchor.
                eye.iris = simd_mix(eye.iris, refined, SIMD2(repeating: 0.7))
            }
            eyes.append(eye)
        }
        guard !eyes.isEmpty else { return nil }

        let yaw = face.yaw.map { Float(truncating: $0) * 180 / .pi }
        let roll = face.roll.map { Float(truncating: $0) * 180 / .pi }
        return FaceMeasurement(eyes: eyes, yawDegrees: yaw, rollDegrees: roll)
    }

    static func eye(contour: [SIMD2<Float>], pupil: SIMD2<Float>?) -> EyeMeasurement? {
        guard contour.count >= 4 else { return nil }
        // Corners = the two contour points farthest apart.
        var a = contour[0], b = contour[1], best: Float = 0
        for i in 0..<contour.count {
            for j in (i + 1)..<contour.count {
                let d = simd_distance_squared(contour[i], contour[j])
                if d > best { best = d; a = contour[i]; b = contour[j] }
            }
        }
        if a.x > b.x { swap(&a, &b) }
        let width = simd_distance(a, b)
        guard width > 2 else { return nil }
        let axis = (b - a) / width
        let normal = SIMD2(-axis.y, axis.x)
        let mid = (a + b) / 2

        // The non-corner points, upper half by height = upper lid. Splitting by rank rather
        // than by side of the corner line survives squints and tilted eyes.
        let local = contour.map { SIMD2(simd_dot($0 - mid, axis), simd_dot($0 - mid, normal)) / width }
            .filter { abs($0.x) < 0.45 }
            .sorted { $0.y < $1.y }
        guard local.count >= 2 else { return nil }
        let half = local.count / 2
        guard var upper = LidCurve.fit(Array(local[..<half])),
              var lower = LidCurve.fit(Array(local[half...])) else { return nil }
        // Keep the lids from crossing.
        if upper.y(0) > lower.y(0) { swap(&upper, &lower) }

        var eye = EyeMeasurement(mid: mid, axis: axis, width: width,
                                 upperLid: upper, lowerLid: lower, iris: .zero)
        eye.iris = pupil ?? eye.center
        return eye
    }

    /// Luminance-weighted centroid of the darkest pixels inside the eye opening.
    static func darkCentroid(in pb: CVPixelBuffer, eye: EyeMeasurement) -> SIMD2<Float>? {
        guard CVPixelBufferGetPixelFormatType(pb) == kCVPixelFormatType_32BGRA else { return nil }
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pb)?.assumingMemoryBound(to: UInt8.self) else { return nil }
        let rowBytes = CVPixelBufferGetBytesPerRow(pb)
        let W = CVPixelBufferGetWidth(pb), H = CVPixelBufferGetHeight(pb)

        let n = eye.normal
        let r = Int((eye.width / 2).rounded(.up))
        let cx = Int(eye.center.x), cy = Int(eye.center.y)

        var samples: [(SIMD2<Float>, Float)] = []
        for y in max(0, cy - r)...min(H - 1, cy + r) {
            for x in max(0, cx - r)...min(W - 1, cx + r) {
                let p = SIMD2(Float(x) + 0.5, Float(y) + 0.5)
                let d = p - eye.mid
                let lx = simd_dot(d, eye.axis) / eye.width, ly = simd_dot(d, n) / eye.width
                // Stay well inside the opening: lids, lashes and corners are also dark.
                let up = eye.upperLid.y(lx), lo = eye.lowerLid.y(lx)
                let margin = (lo - up) * 0.2
                guard abs(lx) < 0.38, ly > up + margin, ly < lo - margin else { continue }
                let px = base + y * rowBytes + x * 4
                let lum = 0.114 * Float(px[0]) + 0.587 * Float(px[1]) + 0.299 * Float(px[2])
                samples.append((p, lum))
            }
        }
        guard samples.count >= 12 else { return nil }

        let sorted = samples.map(\.1).sorted()
        let threshold = sorted[sorted.count / 4]
        var sum = SIMD2<Float>.zero, wsum: Float = 0
        for (p, lum) in samples where lum <= threshold {
            let wgt = threshold - lum + 1
            sum += p * wgt; wsum += wgt
        }
        return wsum > 0 ? sum / wsum : nil
    }
}
