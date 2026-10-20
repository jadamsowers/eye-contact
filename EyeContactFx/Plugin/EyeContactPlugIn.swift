import CoreVideo
import EyeContactCore
import Foundation
import Metal
import os

private let log = Logger(subsystem: "local.eyecontact", category: "plugin")

private enum Param: UInt32 {
    // IDs are stable because the Motion template refers to them. irisSize (6) belonged to the
    // geometric warp and is now hidden; vertical (8) was added with the learned model.
    case strength = 1, targetX, targetY, maxOffset, smoothing, irisSize, showTracking, vertical, model
}

/// Entries of the Model popup, in menu order.
private enum ModelChoice: Int32, CaseIterable {
    case learnedV3 = 0, learnedV2, geometric

    var title: String {
        switch self {
        case .learnedV3: return "Learned v3"
        case .learnedV2: return "Learned v2"
        case .geometric: return "Geometric"
        }
    }
}

/// Parameter values captured in pluginState and handed back at render time. Plain-old-data
/// so it can be copied straight into and out of NSData.
private struct RenderState {
    var strength: Float = 1
    var targetX: Float = 0
    var targetY: Float = 0
    var maxOffset: Float = 0.18
    var smoothing: Float = 0.5
    var irisSize: Float = 0.21
    var showTracking: Int32 = 0
    var vertical: Float = 1
    var model: Int32 = ModelChoice.learnedV3.rawValue
}

/// Per-GPU objects shared by every instance of the effect in this XPC process.
private final class GPUContext {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let renderer: EyeWarpRenderer
    let gazeRenderer: GazeRenderer
    let textureCache: CVMetalTextureCache

    private static var contexts: [UInt64: GPUContext] = [:]
    private static let lock = NSLock()

    static func forRegistryID(_ id: UInt64) throws -> GPUContext {
        lock.lock(); defer { lock.unlock() }
        if let c = contexts[id] { return c }
        guard let device = MTLCopyAllDevices().first(where: { $0.registryID == id }) else {
            throw NSError(domain: FxPlugErrorDomain, code: kFxError_InvalidParameter,
                          userInfo: [NSLocalizedDescriptionKey: "No Metal device with registry ID \(id)"])
        }
        let c = try GPUContext(device: device)
        contexts[id] = c
        return c
    }

    private init(device: MTLDevice) throws {
        self.device = device
        queue = device.makeCommandQueue()!
        renderer = try EyeWarpRenderer(device: device)
        gazeRenderer = try GazeRenderer(device: device)
        var cache: CVMetalTextureCache?
        CVMetalTextureCacheCreate(nil, nil, device, nil, &cache)
        textureCache = cache!
    }
}

/// The learned gaze models (GazeWarpV2/V3.mlpackage, compiled into the bundle by Xcode), loaded
/// on first use and shared by all instances. A model that fails to load falls back to the
/// geometric warp.
private enum SharedGazeModels {
    static let lock = NSLock()
    private static var models: [String: GazeModel?] = [:]

    private static func model(_ name: String) -> GazeModel? {
        if let m = models[name] { return m }
        var m: GazeModel?
        if let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc") {
            do { m = try GazeModel(url: url) } catch {
                log.error("couldn't load \(name, privacy: .public): \(error.localizedDescription, privacy: .public)")
            }
        } else {
            log.error("\(name, privacy: .public).mlmodelc missing from the bundle")
        }
        models[name] = m
        return m
    }

    static func available(_ name: String) -> Bool {
        lock.lock(); defer { lock.unlock() }
        return model(name) != nil
    }

    static func predict(_ name: String, patches: [Float], masks: [Float],
                        displacement: [SIMD2<Float>]) throws -> GazeFields? {
        lock.lock(); defer { lock.unlock() }
        return try model(name)?.predict(patches: patches, masks: masks, displacement: displacement)
    }
}

@objc(EyeContactPlugIn) class EyeContactPlugIn: NSObject, FxTileableEffect {
    private let apiManager: PROAPIAccessing
    /// Raw eye measurements for frames this effect instance has seen, keyed by time.
    private let cache = TrackingCache()

    /// Longest side of the copy that face tracking runs on. Full resolution up to UHD: on a 4K
    /// talking head, a 1280 px copy leaves the eyes ~37 px wide and doubles landmark jitter.
    private static let trackingSize = 3840

    required init?(apiManager: PROAPIAccessing) {
        self.apiManager = apiManager
        super.init()
        log.notice("EyeContactPlugIn init")
    }

    func addParameters() throws {
        log.notice("addParameters")
        guard let p = apiManager.api(for: FxParameterCreationAPI_v5.self) as? FxParameterCreationAPI_v5 else {
            throw NSError(domain: FxPlugErrorDomain, code: kFxError_APIUnavailable, userInfo: nil)
        }
        let flags = FxParameterFlags(kFxParameterFlag_DEFAULT)
        p.addPercentSlider(withName: "Strength", parameterID: Param.strength.rawValue, defaultValue: 1,
                           parameterMin: 0, parameterMax: 1, sliderMin: 0, sliderMax: 1, delta: 0.01,
                           parameterFlags: flags)
        // Target X/Y adjust the eye-contact position measured on the Columbia Gaze data (0 = lens).
        p.addFloatSlider(withName: "Target X", parameterID: Param.targetX.rawValue, defaultValue: 0,
                         parameterMin: -0.3, parameterMax: 0.3, sliderMin: -0.15, sliderMax: 0.15, delta: 0.005,
                         parameterFlags: flags)
        p.addFloatSlider(withName: "Target Y", parameterID: Param.targetY.rawValue, defaultValue: 0,
                         parameterMin: -0.3, parameterMax: 0.3, sliderMin: -0.15, sliderMax: 0.15, delta: 0.005,
                         parameterFlags: flags)
        // Gaze errors beyond this are treated as deliberate look-aways and left alone.
        p.addFloatSlider(withName: "Max Correction", parameterID: Param.maxOffset.rawValue, defaultValue: 0.3,
                         parameterMin: 0.02, parameterMax: 0.6, sliderMin: 0.05, sliderMax: 0.5, delta: 0.01,
                         parameterFlags: flags)
        p.addPercentSlider(withName: "Vertical Correction", parameterID: Param.vertical.rawValue, defaultValue: 1,
                           parameterMin: 0, parameterMax: 1, sliderMin: 0, sliderMax: 1, delta: 0.01,
                           parameterFlags: flags)
        p.addPercentSlider(withName: "Smoothing", parameterID: Param.smoothing.rawValue, defaultValue: 0.5,
                           parameterMin: 0, parameterMax: 1, sliderMin: 0, sliderMax: 1, delta: 0.01,
                           parameterFlags: flags)
        p.addFloatSlider(withName: "Iris Size", parameterID: Param.irisSize.rawValue, defaultValue: 0.21,
                         parameterMin: 0.1, parameterMax: 0.35, sliderMin: 0.15, sliderMax: 0.3, delta: 0.005,
                         parameterFlags: FxParameterFlags(kFxParameterFlag_HIDDEN))
        p.addPopupMenu(withName: "Model", parameterID: Param.model.rawValue,
                       defaultValue: UInt32(ModelChoice.learnedV3.rawValue),
                       menuEntries: ModelChoice.allCases.map(\.title), parameterFlags: flags)
        p.addToggleButton(withName: "Show Tracking", parameterID: Param.showTracking.rawValue, defaultValue: false,
                          parameterFlags: flags)
    }

    func properties(_ properties: AutoreleasingUnsafeMutablePointer<NSDictionary>?) throws {
        log.notice("properties")
        properties?.pointee = [
            kFxPropertyKey_MayRemapTime: false,
            kFxPropertyKey_PixelTransformSupport: NSNumber(value: kFxPixelTransform_ScaleTranslate),
            // Output depends on neighboring frames through smoothing, not just this frame's input.
            kFxPropertyKey_VariesWhenParamsAreStatic: true,
            // Face tracking needs the whole frame.
            kFxPropertyKey_NeedsFullBuffer: true,
            kFxPropertyKey_DesiredProcessingColorInfo: NSNumber(value: kFxImageColorInfo_RGB_GAMMA_VIDEO),
        ] as NSDictionary
    }

    func pluginState(_ pluginState: AutoreleasingUnsafeMutablePointer<NSData>?, at renderTime: CMTime,
                     quality qualityLevel: UInt) throws {
        guard let p = apiManager.api(for: FxParameterRetrievalAPI_v6.self) as? FxParameterRetrievalAPI_v6 else {
            throw NSError(domain: FxPlugErrorDomain, code: kFxError_APIUnavailable, userInfo: nil)
        }
        func float(_ id: Param) -> Float {
            var v = 0.0
            p.getFloatValue(&v, fromParameter: id.rawValue, at: renderTime)
            return Float(v)
        }
        var modelIndex: Int32 = 0
        p.getIntValue(&modelIndex, fromParameter: Param.model.rawValue, at: renderTime)
        var show: ObjCBool = false
        p.getBoolValue(&show, fromParameter: Param.showTracking.rawValue, at: renderTime)

        var state = RenderState(strength: float(.strength), targetX: float(.targetX), targetY: float(.targetY),
                                maxOffset: float(.maxOffset), smoothing: float(.smoothing),
                                irisSize: float(.irisSize), showTracking: show.boolValue ? 1 : 0,
                                vertical: float(.vertical), model: modelIndex)
        pluginState?.pointee = NSData(bytes: &state, length: MemoryLayout<RenderState>.size)
    }

    func destinationImageRect(_ destinationImageRect: UnsafeMutablePointer<FxRect>, sourceImages: [FxImageTile],
                              destinationImage: FxImageTile, pluginState: Data?, at renderTime: CMTime) throws {
        destinationImageRect.pointee = sourceImages[0].imagePixelBounds
    }

    func sourceTileRect(_ sourceTileRect: UnsafeMutablePointer<FxRect>, sourceImageIndex: UInt,
                        sourceImages: [FxImageTile], destinationTileRect: FxRect, destinationImage: FxImageTile,
                        pluginState: Data?, at renderTime: CMTime) throws {
        sourceTileRect.pointee = destinationTileRect
    }

    func renderDestinationImage(_ destinationImage: FxImageTile, sourceImages: [FxImageTile], pluginState: Data?,
                                at renderTime: CMTime) throws {
        var state = RenderState()
        if let pluginState, pluginState.count == MemoryLayout<RenderState>.size {
            state = pluginState.withUnsafeBytes { $0.loadUnaligned(as: RenderState.self) }
        }

        let source = sourceImages[0]
        let gpu = try GPUContext.forRegistryID(source.deviceRegistryID)
        guard let srcTex = source.metalTexture(for: gpu.device),
              let dstTex = destinationImage.metalTexture(for: gpu.device) else {
            throw NSError(domain: FxPlugErrorDomain, code: kFxError_InvalidParameter,
                          userInfo: [NSLocalizedDescriptionKey: "Couldn't get Metal textures"])
        }

        let time = renderTime.seconds
        let frameWidth = Float(srcTex.width)
        let reportedBottomUp = source.imageOrigin == FxImageOrigin(kFxImageOrigin_BOTTOM_LEFT)

        if !cache.contains(time: time) {
            let face = try measure(srcTex, gpu: gpu, reportedBottomUp: reportedBottomUp)
            cache.store(face, time: time, frameWidth: frameWidth)
        }

        // Up to a 0.25 s Gaussian (default 0.125 s, about ±3 frames at 24 fps): enough to keep
        // landmark noise from making the eyes shimmer.
        let sigma = Double(state.smoothing) * 0.25
        let face = cache.smoothed(time: time, sigma: sigma, frameWidth: frameWidth)
        let debug = state.showTracking != 0
        // Geometry is upright; the shaders flip when reading and writing bottom-up textures.
        let bottomUp = Self.isBottomUp(reported: reportedBottomUp)

        guard let cb = gpu.queue.makeCommandBuffer() else { return }
        cb.label = "Eye Contact"
        let choice = ModelChoice(rawValue: state.model) ?? .learnedV3
        let learned: (name: String, profile: GazeProfile)? = switch choice {
        case .learnedV3: ("GazeWarpV3", .v3)
        case .learnedV2: ("GazeWarpV2", .v2)
        case .geometric: nil
        }
        if let learned, SharedGazeModels.available(learned.name) {
            var gaze = GazeSettings()
            gaze.strength = state.strength
            gaze.targetOffset = SIMD2(state.targetX, state.targetY)
            gaze.lookAwayLimit = state.maxOffset
            gaze.vertical = state.vertical
            let request = face.flatMap { makeGazeRequest(for: $0, settings: gaze, profile: learned.profile) }
            var fields: [(MTLTexture, MTLTexture)]?
            if let request, request.amount > 0.001 {
                fields = try redirect(srcTex, request: request, model: learned.name, gpu: gpu, bottomUp: bottomUp)
            }
            try gpu.gazeRenderer.encodeWarp(into: cb, source: srcTex, destination: dstTex, request: request,
                                            fields: fields, debug: debug, flipY: bottomUp)
        } else {
            var settings = CorrectionSettings()
            settings.strength = state.strength
            settings.target = SIMD2(state.targetX, state.targetY)
            settings.maxOffset = state.maxOffset
            settings.irisRadius = state.irisSize
            let warps = face.map { makeWarps(for: $0, settings: settings) } ?? []
            try gpu.renderer.encode(into: cb, source: srcTex, destination: dstTex, warps: warps, debug: debug,
                                    flipY: bottomUp)
        }
        cb.commit()
        cb.waitUntilCompleted()
    }

    /// Cuts out both eye patches, runs the network and returns its fields as textures.
    private func redirect(_ src: MTLTexture, request: GazeRequest, model: String, gpu: GPUContext,
                          bottomUp: Bool) throws -> [(MTLTexture, MTLTexture)]? {
        let patchTex = gpu.gazeRenderer.makePatchTexture()
        guard let cb = gpu.queue.makeCommandBuffer() else { return nil }
        try gpu.gazeRenderer.encodePatches(into: cb, source: src, patchTexture: patchTex, request: request,
                                           flipY: bottomUp)
        cb.commit()
        cb.waitUntilCompleted()
        guard let fields = try SharedGazeModels.predict(model, patches: gpu.gazeRenderer.readPatches(patchTex),
                                                       masks: request.patches.flatMap { $0.mask() },
                                                       displacement: request.displacement) else { return nil }
        return gpu.gazeRenderer.makeFieldTextures(fields)
    }

    /// Runs face tracking on a downscaled 8-bit copy and returns the result in the
    /// source texture's pixel coordinates (top-left origin of the texture).
    /// Whether the host's textures are stored bottom-up. Starts from FxImageTile.imageOrigin;
    /// if Vision then finds the face upside down, the opposite is used from then on.
    private static var bottomUpOverride: Bool?
    private static let orientationLock = NSLock()

    private static func isBottomUp(reported: Bool) -> Bool {
        orientationLock.lock(); defer { orientationLock.unlock() }
        return bottomUpOverride ?? reported
    }

    /// Face measurement in upright frame pixels (top-left origin), whatever the row order.
    private func measure(_ src: MTLTexture, gpu: GPUContext, reportedBottomUp: Bool) throws -> FaceMeasurement? {
        let flip = Self.isBottomUp(reported: reportedBottomUp)
        let face = try track(src, gpu: gpu, flip: flip)
        if let roll = face?.rollDegrees, abs(roll) > 120 {
            // Vision saw the face upside down: the row order is the other way round.
            if let other = try track(src, gpu: gpu, flip: !flip), abs(other.rollDegrees ?? 0) < 60 {
                Self.orientationLock.lock()
                Self.bottomUpOverride = !flip
                Self.orientationLock.unlock()
                log.notice("image origin reported \(reportedBottomUp ? "bottom-left" : "top-left", privacy: .public); using the opposite")
                return other
            }
        }
        return face
    }

    private func track(_ src: MTLTexture, gpu: GPUContext, flip: Bool) throws -> FaceMeasurement? {
        let scale = min(1, Float(Self.trackingSize) / Float(max(src.width, src.height)))
        let w = max(1, Int((Float(src.width) * scale).rounded()))
        let h = max(1, Int((Float(src.height) * scale).rounded()))

        var pb: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferMetalCompatibilityKey as String: true,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any],
        ]
        CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb)
        guard let pb else { return nil }

        var cvTex: CVMetalTexture?
        CVMetalTextureCacheCreateTextureFromImage(nil, gpu.textureCache, pb, nil, .bgra8Unorm, w, h, 0, &cvTex)
        guard let cvTex, let small = CVMetalTextureGetTexture(cvTex),
              let cb = gpu.queue.makeCommandBuffer() else { return nil }
        try gpu.renderer.encodeDownsample(into: cb, source: src, destination: small, flipY: flip)
        cb.commit()
        cb.waitUntilCompleted()

        // Vision request objects aren't thread-safe and FCP renders on several threads.
        return EyeTracker().measure(pb)?.scaled(Float(src.width) / Float(w))
    }
}
