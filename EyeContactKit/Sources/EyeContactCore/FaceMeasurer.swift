import CoreML
import CoreVideo
import Foundation
import Metal
import simd

/// Core ML wrapper for IrisLocator.mlpackage (gazenet/iris_net.py): iris centers for the two
/// Vision-framed eye patches, trained on MediaPipe's iris landmark.
public final class IrisModel {
    private let model: MLModel
    private let lock = NSLock()

    public init(url: URL) throws {
        let compiled = url.pathExtension == "mlmodelc" ? url : try MLModel.compileModel(at: url)
        let config = MLModelConfiguration()
        config.computeUnits = .all
        model = try MLModel(contentsOf: compiled, configuration: config)
    }

    /// `patches` [2][3][96][144], `masks` [2][96][144] -> iris center per eye, patch pixels.
    public func predict(patches: [Float], masks: [Float]) throws -> [SIMD2<Float>] {
        func array(_ v: [Float], _ shape: [Int]) throws -> MLMultiArray {
            let a = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
            v.withUnsafeBufferPointer {
                a.dataPointer.assumingMemoryBound(to: Float.self).update(from: $0.baseAddress!, count: v.count)
            }
            return a
        }
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "image": try array(patches, [2, 3, GazePatch.height, GazePatch.width]),
            "mask": try array(masks, [2, 1, GazePatch.height, GazePatch.width]),
        ])
        lock.lock(); defer { lock.unlock() }
        guard let a = try model.prediction(from: input).featureValue(for: "iris")?.multiArrayValue else { return [] }
        let v = MLShapedArray<Float>(a).scalars
        return [SIMD2(v[0], v[1]), SIMD2(v[2], v[3])]
    }
}

/// Measures the face in a frame for the learned pipeline:
///  1. Vision on the whole frame downscaled to `coarseSize` finds the eyes;
///  2. Vision again on a crop around the eyes at `fineSize`, so small faces in 4K frames get
///     enough pixels for stable landmarks;
///  3. the iris locator replaces Vision's pupil estimate.
/// Results are in upright frame pixels (top-left origin), whatever the texture's row order.
public final class FaceMeasurer {
    public let device: MTLDevice
    private let renderer: EyeWarpRenderer
    private let gazeRenderer: GazeRenderer
    private let iris: IrisModel?
    private let textureCache: CVMetalTextureCache
    public var coarseSize = 1280
    public var fineSize = 768

    public init(device: MTLDevice, renderer: EyeWarpRenderer, gazeRenderer: GazeRenderer, iris: IrisModel?) {
        self.device = device
        self.renderer = renderer
        self.gazeRenderer = gazeRenderer
        self.iris = iris
        var cache: CVMetalTextureCache?
        CVMetalTextureCacheCreate(nil, nil, device, nil, &cache)
        textureCache = cache!
    }

    /// Renders (a region of) `src` into a new BGRA buffer and runs Vision on it. Returns the
    /// measurement in that buffer's pixels and the buffer's size.
    private func track(_ src: MTLTexture, queue: MTLCommandQueue, bottomUp: Bool, width: Int, height: Int,
                       region: (origin: SIMD2<Float>, size: SIMD2<Float>)?) throws -> FaceMeasurement? {
        var pb: CVPixelBuffer?
        let attrs: [String: Any] = [kCVPixelBufferMetalCompatibilityKey as String: true,
                                    kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any]]
        CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb)
        guard let pb else { return nil }
        var cvTex: CVMetalTexture?
        CVMetalTextureCacheCreateTextureFromImage(nil, textureCache, pb, nil, .bgra8Unorm, width, height, 0, &cvTex)
        guard let cvTex, let tex = CVMetalTextureGetTexture(cvTex), let cb = queue.makeCommandBuffer() else { return nil }
        try renderer.encodeDownsample(into: cb, source: src, destination: tex, flipY: bottomUp, region: region)
        cb.commit()
        cb.waitUntilCompleted()
        // Vision request objects aren't thread-safe, and hosts render on several threads.
        return EyeTracker().measure(pb)
    }

    /// Single pass on the whole frame at `coarseSize` (what the plug-in originally did).
    public func measureCoarse(_ src: MTLTexture, queue: MTLCommandQueue, bottomUp: Bool = false) throws -> FaceMeasurement? {
        let W = Float(src.width), H = Float(src.height)
        let scale = min(1, Float(coarseSize) / max(W, H))
        let cw = max(1, Int((W * scale).rounded())), ch = max(1, Int((H * scale).rounded()))
        return try track(src, queue: queue, bottomUp: bottomUp, width: cw, height: ch, region: nil)?.scaled(W / Float(cw))
    }

    public func measure(_ src: MTLTexture, queue: MTLCommandQueue, bottomUp: Bool = false) throws -> FaceMeasurement? {
        let W = Float(src.width), H = Float(src.height)

        // 1. Coarse: whole frame.
        let scale = min(1, Float(coarseSize) / max(W, H))
        let cw = max(1, Int((W * scale).rounded())), ch = max(1, Int((H * scale).rounded()))
        guard let coarse = try track(src, queue: queue, bottomUp: bottomUp, width: cw, height: ch, region: nil)?
            .scaled(W / Float(cw)), coarse.eyes.count == 2 else { return nil }

        // 2. Fine: a square around the eyes, 4x the eye spacing (covers lids, brows and some margin).
        var face = coarse
        let c = (coarse.eyes[0].mid + coarse.eyes[1].mid) / 2
        let side = min(max(simd_distance(coarse.eyes[0].mid, coarse.eyes[1].mid) * 4, 64), min(W, H))
        let origin = simd_clamp(c - side / 2, .zero, SIMD2(W, H) - side)
        if let fine = try track(src, queue: queue, bottomUp: bottomUp, width: fineSize, height: fineSize,
                                region: (origin / SIMD2(W, H), SIMD2(side, side) / SIMD2(W, H))),
           fine.eyes.count == 2 {
            face = fine.scaled(side / Float(fineSize)).translated(by: origin)
        }

        // 3. Iris locator on the Vision-framed patches (v3 framing).
        if let iris, face.eyes.count == 2 {
            let ordered = face.eyes.sorted { $0.mid.x < $1.mid.x }
            let frames = ordered.enumerated().map { EyePatchFrame(eye: $0.element, side: $0.offset, profile: .v3) }
            let request = GazeRequest(patches: frames, displacement: [.zero, .zero], gaze: .zero, amount: 1)
            let patchTex = gazeRenderer.makePatchTexture()
            if let cb = queue.makeCommandBuffer() {
                try gazeRenderer.encodePatches(into: cb, source: src, patchTexture: patchTex, request: request,
                                               flipY: bottomUp)
                cb.commit()
                cb.waitUntilCompleted()
                let centers = try iris.predict(patches: gazeRenderer.readPatches(patchTex),
                                               masks: frames.flatMap { $0.mask() })
                if centers.count == 2 {
                    face.eyes = zip(frames, centers).map { f, q in
                        var e = f.eye
                        e.iris = f.toFrame(q)
                        return e
                    }
                }
            }
        }
        return face
    }
}
