import AVFoundation
import CoreVideo
import EyeContactCore
import Foundation
import Metal

let usage = """
usage: eyecontact <input> <output.mov> [options]
  --strength <0..1>     correction amount (default 1.0)
  --target-x <f>        horizontal "eye contact" point, in eye widths (default 0)
  --target-y <f>        vertical point, + is down (default 0)
  --max <f>             look-aways larger than this are left alone (default 0.18)
  --debug               draw tracking overlay
  --no-smooth           disable temporal smoothing
  --no-refine           use Vision's pupil points only
  --frames <n>          stop after n frames
  --log <file.csv>      write per-frame gaze measurements
  --model <GazeWarp.mlpackage>  use the learned gaze model instead of the geometric warp;
                        then --target-x/--target-y adjust the measured eye-contact position
  --vertical <0..1>     (model) scale of vertical correction (default 1)
  --profile v2|v3       (model) framing the model was trained with (default v3)
  --iris <IrisLocator.mlpackage>  (model) two-stage tracking + learned iris locator, as the plug-in
  --track full|1280|2stage  (model) tracking resolution: full frame, plug-in's old 1280 px copy, or
                        coarse 1280 px + fine crop around the eyes (default full; 2stage with --iris)
  --sigma <seconds>     (model) temporal smoothing like the plug-in: Gaussian over past frames (default 0.125)
"""

var args = Array(CommandLine.arguments.dropFirst())
guard args.count >= 2 else { print(usage); exit(1) }
let inputURL = URL(fileURLWithPath: args.removeFirst())
let outputURL = URL(fileURLWithPath: args.removeFirst())

var settings = CorrectionSettings()
var smooth = true, refine = true
var maxFrames = Int.max
var logURL: URL?
var modelURL: URL?
var gaze = GazeSettings()
var profile = GazeProfile.v3
var irisURL: URL?
var sigma = 0.125
var trackMode = "full"
var bottomUp = false
while !args.isEmpty {
    let a = args.removeFirst()
    func value() -> Float {
        guard !args.isEmpty, let v = Float(args.removeFirst()) else { print("missing value for \(a)"); exit(1) }
        return v
    }
    switch a {
    case "--strength": settings.strength = value(); gaze.strength = settings.strength
    case "--target-x": settings.target.x = value(); gaze.targetOffset.x = settings.target.x
    case "--target-y": settings.target.y = value(); gaze.targetOffset.y = settings.target.y
    case "--max": settings.maxOffset = value(); gaze.lookAwayLimit = settings.maxOffset
    case "--vertical": gaze.vertical = value()
    case "--iris":
        guard !args.isEmpty else { print("missing value for --iris"); exit(1) }
        irisURL = URL(fileURLWithPath: args.removeFirst())
    case "--sigma": sigma = Double(value())
    case "--bottom-up": bottomUp = true  // treat frames as stored bottom-up, like Final Cut's (testing)
    case "--track":
        guard !args.isEmpty else { print("missing value for --track"); exit(1) }
        trackMode = args.removeFirst()
    case "--profile":
        guard !args.isEmpty else { print("missing value for --profile"); exit(1) }
        switch args.removeFirst() {
        case "v2": profile = .v2
        case "v3": profile = .v3
        default: print("--profile must be v2 or v3"); exit(1)
        }
    case "--model":
        guard !args.isEmpty else { print("missing value for --model"); exit(1) }
        modelURL = URL(fileURLWithPath: args.removeFirst())
    case "--debug": settings.debug = true; gaze.debug = true
    case "--no-smooth": smooth = false
    case "--no-refine": refine = false
    case "--frames": maxFrames = Int(value())
    case "--log":
        guard !args.isEmpty else { print("missing value for --log"); exit(1) }
        logURL = URL(fileURLWithPath: args.removeFirst())
    default: print("unknown option \(a)\n\(usage)"); exit(1)
    }
}

// MARK: - Setup

let asset = AVURLAsset(url: inputURL)
let semaphore = DispatchSemaphore(value: 0)
var videoTrack: AVAssetTrack?, audioTrack: AVAssetTrack?
var naturalSize = CGSize.zero, frameRate: Float = 30, transform = CGAffineTransform.identity
var audioHint: CMFormatDescription?
Task {
    videoTrack = try? await asset.loadTracks(withMediaType: .video).first
    audioTrack = try? await asset.loadTracks(withMediaType: .audio).first
    if let a = audioTrack { audioHint = (try? await a.load(.formatDescriptions))?.first }
    if let v = videoTrack {
        naturalSize = (try? await v.load(.naturalSize)) ?? .zero
        frameRate = (try? await v.load(.nominalFrameRate)) ?? 30
        transform = (try? await v.load(.preferredTransform)) ?? .identity
    }
    semaphore.signal()
}
semaphore.wait()
guard let videoTrack else { print("no video track in \(inputURL.path)"); exit(1) }
if !transform.isIdentity { print("warning: input has a rotation transform; tracking assumes upright frames") }

let reader = try AVAssetReader(asset: asset)
let videoOut = AVAssetReaderTrackOutput(track: videoTrack, outputSettings: [
    kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
    kCVPixelBufferMetalCompatibilityKey as String: true,
])
videoOut.alwaysCopiesSampleData = false
reader.add(videoOut)
var audioOut: AVAssetReaderTrackOutput?
if let audioTrack {
    let o = AVAssetReaderTrackOutput(track: audioTrack, outputSettings: nil)  // passthrough
    reader.add(o); audioOut = o
}

try? FileManager.default.removeItem(at: outputURL)
let writer = try AVAssetWriter(outputURL: outputURL, fileType: .mov)
let videoIn = AVAssetWriterInput(mediaType: .video, outputSettings: [
    AVVideoCodecKey: AVVideoCodecType.proRes422,
    AVVideoWidthKey: Int(naturalSize.width),
    AVVideoHeightKey: Int(naturalSize.height),
])
videoIn.transform = transform
let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: videoIn, sourcePixelBufferAttributes: [
    kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
    kCVPixelBufferWidthKey as String: Int(naturalSize.width),
    kCVPixelBufferHeightKey as String: Int(naturalSize.height),
    kCVPixelBufferMetalCompatibilityKey as String: true,
    kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any],
])
writer.add(videoIn)
var audioIn: AVAssetWriterInput?
if audioOut != nil {
    let i = AVAssetWriterInput(mediaType: .audio, outputSettings: nil, sourceFormatHint: audioHint)
    writer.add(i); audioIn = i
}

guard reader.startReading() else { print("reader failed: \(String(describing: reader.error))"); exit(1) }
guard writer.startWriting() else { print("writer failed: \(String(describing: writer.error))"); exit(1) }
writer.startSession(atSourceTime: .zero)

let device = MTLCreateSystemDefaultDevice()!
let queue = device.makeCommandQueue()!
let renderer = try EyeWarpRenderer(device: device)
var textureCache: CVMetalTextureCache?
CVMetalTextureCacheCreate(nil, nil, device, nil, &textureCache)

func texture(_ pb: CVPixelBuffer) -> (MTLTexture, CVMetalTexture)? {
    var cvTex: CVMetalTexture?
    CVMetalTextureCacheCreateTextureFromImage(nil, textureCache!, pb, nil, .bgra8Unorm,
                                              CVPixelBufferGetWidth(pb), CVPixelBufferGetHeight(pb), 0, &cvTex)
    guard let cvTex, let t = CVMetalTextureGetTexture(cvTex) else { return nil }
    return (t, cvTex)
}

let gazeModel = try modelURL.map { try GazeModel(url: $0) }
let gazeRenderer = try GazeRenderer(device: device)
let patchTexture = gazeRenderer.makePatchTexture()
let irisModel = try irisURL.map { try IrisModel(url: $0) }
let measurer = FaceMeasurer(device: device, renderer: renderer, gazeRenderer: gazeRenderer, iris: irisModel)
let trackingCache = TrackingCache()
if irisModel != nil { profile = .v3Iris; trackMode = "2stage" }

// MARK: - Processing

let tracker = EyeTracker()
var smoother = FaceSmoother()
var frameCount = 0, trackedCount = 0
var trackTime = 0.0, renderTime = 0.0
var logLines = ["frame,time,offset_x,offset_y,openness,yaw,amount"]
let group = DispatchGroup()
let started = Date()

group.enter()
videoIn.requestMediaDataWhenReady(on: DispatchQueue(label: "video")) {
    while videoIn.isReadyForMoreMediaData {
        guard frameCount < maxFrames, let sample = videoOut.copyNextSampleBuffer(),
              let srcPB = CMSampleBufferGetImageBuffer(sample) else {
            videoIn.markAsFinished(); group.leave(); return
        }
        let pts = CMSampleBufferGetPresentationTimeStamp(sample)

        let t0 = Date()
        var warps: [EyeWarp] = []
        var request: GazeRequest?
        var measured: FaceMeasurement?
        if gazeModel != nil {
            // Plug-in pipeline: measure, then smooth through the time-keyed cache.
            let (srcTex, srcRef) = texture(srcPB)!
            var raw: FaceMeasurement?
            switch trackMode {
            case "2stage": raw = try! measurer.measure(srcTex, queue: queue, bottomUp: bottomUp)
            case "1280": raw = try! measurer.measureCoarse(srcTex, queue: queue, bottomUp: bottomUp)
            default: raw = tracker.measure(srcPB, refinePupils: refine)
            }
            withExtendedLifetime(srcRef) {}
            trackingCache.store(raw, time: pts.seconds, frameWidth: Float(CVPixelBufferGetWidth(srcPB)))
            measured = trackingCache.smoothed(time: pts.seconds, sigma: smooth ? sigma : 0,
                                              frameWidth: Float(CVPixelBufferGetWidth(srcPB)))
        } else {
            measured = tracker.measure(srcPB, refinePupils: refine)
            if smooth, let m = measured { measured = smoother.smooth(m, time: pts.seconds) }
        }
        if let face = measured {
            if gazeModel != nil {
                request = makeGazeRequest(for: face, settings: gaze, profile: profile)
                if logURL != nil, let r = request {
                    logLines.append(String(format: "%d,%.4f,%.4f,%.4f,%.3f,%.1f,%.3f", frameCount, pts.seconds,
                                           r.gaze.x, r.gaze.y, face.eyes.map(\.openness).min() ?? 0,
                                           face.yawDegrees ?? .nan, r.amount))
                }
            } else {
                warps = makeWarps(for: face, settings: settings)
            }
            trackedCount += 1
            if logURL != nil, gazeModel == nil {
                let o = face.eyes.map(\.gazeOffset).reduce(.zero, +) / Float(face.eyes.count)
                let open = face.eyes.map(\.openness).min() ?? 0
                logLines.append(String(format: "%d,%.4f,%.4f,%.4f,%.3f,%.1f,%.3f", frameCount, pts.seconds,
                                       o.x, o.y, open, face.yawDegrees ?? .nan, warps.first?.amount ?? 0))
            }
        } else {
            smoother.reset()
        }
        let t1 = Date()

        var dstPB: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(nil, adaptor.pixelBufferPool!, &dstPB)
        guard let dstPB, let (src, srcRef) = texture(srcPB), let (dst, dstRef) = texture(dstPB) else {
            print("texture creation failed"); exit(1)
        }
        let cb = queue.makeCommandBuffer()!
        if let gazeModel {
            var fields: [(MTLTexture, MTLTexture)]?
            if let r = request, r.amount > 0.001 {
                let pcb = queue.makeCommandBuffer()!
                try! gazeRenderer.encodePatches(into: pcb, source: src, patchTexture: patchTexture, request: r,
                                                flipY: bottomUp)
                pcb.commit()
                pcb.waitUntilCompleted()
                let patches = gazeRenderer.readPatches(patchTexture)
                let masks = r.patches.flatMap { $0.mask() }
                if let dump = ProcessInfo.processInfo.environment["EYECONTACT_DUMP_FRAME"], Int(dump) == frameCount {
                    // Debug aid: raw network inputs for one frame.
                    let blob = (patches + masks + r.displacement.flatMap { [$0.x, $0.y] }).withUnsafeBytes { Data($0) }
                    try? blob.write(to: URL(fileURLWithPath: "gaze_inputs_\(frameCount).f32"))
                }
                let out = try! gazeModel.predict(patches: patches, masks: masks, displacement: r.displacement)
                fields = gazeRenderer.makeFieldTextures(out)
            }
            try! gazeRenderer.encodeWarp(into: cb, source: src, destination: dst, request: request,
                                         fields: fields, debug: gaze.debug, flipY: bottomUp)
        } else {
            try! renderer.encode(into: cb, source: src, destination: dst, warps: warps, debug: settings.debug)
        }
        cb.commit()
        cb.waitUntilCompleted()
        withExtendedLifetime((srcRef, dstRef)) {}
        renderTime += Date().timeIntervalSince(t1)
        trackTime += t1.timeIntervalSince(t0)

        adaptor.append(dstPB, withPresentationTime: pts)
        frameCount += 1
        if frameCount % 60 == 0 { print("frame \(frameCount)") }
    }
}

if let audioIn, let audioOut {
    group.enter()
    audioIn.requestMediaDataWhenReady(on: DispatchQueue(label: "audio")) {
        while audioIn.isReadyForMoreMediaData {
            guard let s = audioOut.copyNextSampleBuffer() else {
                audioIn.markAsFinished(); group.leave(); return
            }
            audioIn.append(s)
        }
    }
}

group.wait()
if reader.status == .reading { reader.cancelReading() }
writer.finishWriting { semaphore.signal() }
semaphore.wait()

if let logURL { try logLines.joined(separator: "\n").write(to: logURL, atomically: true, encoding: .utf8) }
if writer.status != .completed { print("write failed: \(String(describing: writer.error))"); exit(1) }
let n = Double(max(frameCount, 1))
print(String(format: "%d frames (%d with a face) in %.1fs — tracking %.1f ms/frame, warp %.1f ms/frame",
             frameCount, trackedCount, Date().timeIntervalSince(started), trackTime / n * 1000, renderTime / n * 1000))
print("wrote \(outputURL.path)")
