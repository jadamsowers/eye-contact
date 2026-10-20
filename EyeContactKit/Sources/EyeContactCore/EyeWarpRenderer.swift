import Foundation
import Metal

/// Applies the eye warps on the GPU. Draws a full-screen triangle into the destination as a
/// render target, which works for Final Cut's textures (render-target only) and CVPixelBuffers.
public final class EyeWarpRenderer {
    public let device: MTLDevice
    private let library: MTLLibrary
    private let sampler: MTLSamplerState
    private var pipelines: [String: MTLRenderPipelineState] = [:]
    private let lock = NSLock()

    struct Uniforms {
        var size: SIMD2<Float>
        var eyeCount: Int32
        var debug: Int32
        var flipY: Int32 = 0
        var pad: Int32 = 0
    }

    struct DownsampleUniforms {
        var dstSize: SIMD2<Float>
        var flipY: Int32
        var pad: Int32 = 0
        var rectOrigin = SIMD2<Float>(0, 0)   // source region, normalized upright coordinates
        var rectSize = SIMD2<Float>(1, 1)
    }

    public init(device: MTLDevice) throws {
        self.device = device
        // Compiled at runtime so the package builds without Xcode's Metal toolchain.
        library = try device.makeLibrary(source: Self.shaderSource, options: nil)

        let sd = MTLSamplerDescriptor()
        sd.minFilter = .linear
        sd.magFilter = .linear
        sd.sAddressMode = .clampToEdge
        sd.tAddressMode = .clampToEdge
        sampler = device.makeSamplerState(descriptor: sd)!
    }

    private func pipeline(_ fragment: String, format: MTLPixelFormat) throws -> MTLRenderPipelineState {
        let key = "\(fragment)/\(format.rawValue)"
        lock.lock(); defer { lock.unlock() }
        if let p = pipelines[key] { return p }
        let d = MTLRenderPipelineDescriptor()
        d.vertexFunction = library.makeFunction(name: "fullscreen")
        d.fragmentFunction = library.makeFunction(name: fragment)
        d.colorAttachments[0].pixelFormat = format
        let p = try device.makeRenderPipelineState(descriptor: d)
        pipelines[key] = p
        return p
    }

    private func encoder(_ cb: MTLCommandBuffer, target: MTLTexture) -> MTLRenderCommandEncoder? {
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = target
        pass.colorAttachments[0].loadAction = .dontCare
        pass.colorAttachments[0].storeAction = .store
        return cb.makeRenderCommandEncoder(descriptor: pass)
    }

    /// Encodes the warp from `source` into `destination` (same size, distinct textures).
    /// `flipY`: the textures are stored bottom-up; `warps` geometry is always upright.
    public func encode(into commandBuffer: MTLCommandBuffer, source: MTLTexture, destination: MTLTexture,
                       warps: [EyeWarp], debug: Bool, flipY: Bool = false) throws {
        let ps = try pipeline("eyeWarp", format: destination.pixelFormat)
        guard let enc = encoder(commandBuffer, target: destination) else { return }
        var eyes = warps.isEmpty ? [EyeWarp()] : warps
        var u = Uniforms(size: SIMD2(Float(source.width), Float(source.height)),
                         eyeCount: Int32(warps.count), debug: debug ? 1 : 0, flipY: flipY ? 1 : 0)
        enc.setRenderPipelineState(ps)
        enc.setFragmentTexture(source, index: 0)
        enc.setFragmentBytes(&eyes, length: MemoryLayout<EyeWarp>.stride * eyes.count, index: 0)
        enc.setFragmentBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 1)
        enc.setFragmentSamplerState(sampler, index: 0)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        enc.endEncoding()
    }

    /// Resamples `source` (or the normalized, upright `region` of it) into `destination`,
    /// typically a small BGRA buffer for Vision, clamping to 0...1. `flipY` turns a
    /// bottom-left-origin image upright.
    public func encodeDownsample(into commandBuffer: MTLCommandBuffer, source: MTLTexture,
                                 destination: MTLTexture, flipY: Bool,
                                 region: (origin: SIMD2<Float>, size: SIMD2<Float>)? = nil) throws {
        let ps = try pipeline("downsample", format: destination.pixelFormat)
        guard let enc = encoder(commandBuffer, target: destination) else { return }
        var u = DownsampleUniforms(dstSize: SIMD2(Float(destination.width), Float(destination.height)),
                                   flipY: flipY ? 1 : 0)
        if let region { u.rectOrigin = region.origin; u.rectSize = region.size }
        enc.setRenderPipelineState(ps)
        enc.setFragmentTexture(source, index: 0)
        enc.setFragmentBytes(&u, length: MemoryLayout<DownsampleUniforms>.stride, index: 0)
        enc.setFragmentSamplerState(sampler, index: 0)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        enc.endEncoding()
    }

    static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    struct EyeParams {
        float2 mid;
        float2 axis;
        float2 iris;
        float2 shift;
        float width;
        float irisRadius;
        float upperA;
        float upperB;
        float lowerA;
        float lowerB;
        float amount;
        float pad;
    };

    struct Uniforms {
        float2 size;
        int eyeCount;
        int debug;
        int flipY;
        int pad;
    };

    struct VOut { float4 pos [[position]]; };

    vertex VOut fullscreen(uint vid [[vertex_id]]) {
        float2 uv = float2((vid << 1) & 2, vid & 2);
        VOut o;
        o.pos = float4(uv * 2.0 - 1.0, 0.0, 1.0);
        return o;
    }

    // Signed distance (pixels, positive inside) to the eye opening bounded by the lid curves.
    static float openingDistance(float2 p, constant EyeParams &e) {
        float2 n = float2(-e.axis.y, e.axis.x);
        float2 d = p - e.mid;
        float lx = dot(d, e.axis) / e.width;
        float ly = dot(d, n) / e.width;
        float f = lx * lx - 0.25;
        float up = f * (e.upperA + e.upperB * lx);
        float lo = f * (e.lowerA + e.lowerB * lx);
        return min(min(ly - up, lo - ly), 0.5 - abs(lx)) * e.width;
    }

    static float segmentDistance(float2 p, float2 a, float2 b) {
        float2 ab = b - a;
        float t = clamp(dot(p - a, ab) / max(dot(ab, ab), 1e-6), 0.0, 1.0);
        return length(p - (a + ab * t));
    }

    fragment float4 eyeWarp(VOut in [[stage_in]],
                            texture2d<float, access::sample> src [[texture(0)]],
                            constant EyeParams *eyes [[buffer(0)]],
                            constant Uniforms &u [[buffer(1)]],
                            sampler s [[sampler(0)]])
    {
        // Pixel center in upright, top-left-origin pixel coordinates.
        float2 p = u.flipY ? float2(in.pos.x, u.size.y - in.pos.y) : in.pos.xy;
        float2 q = p;
        float4 overlay = 0;

        for (int i = 0; i < u.eyeCount; i++) {
            constant EyeParams &e = eyes[i];
            float2 shift = e.shift * e.amount;
            float feather = max(1.0, e.width * 0.025);

            // The eyeball slides behind the lids: everything inside the opening moves,
            // nothing outside does, and the lid edge acts as an occluding boundary.
            float opening = smoothstep(0.0, feather, openingDistance(p, e) - 0.5);

            // The iris moves rigidly along its path (a capsule from where it was to where
            // it goes). Sclera around it compresses ahead and stretches behind; the ramp
            // widens with the shift so the mapping never folds over.
            float core = e.irisRadius * 1.15;
            float ramp = max(e.irisRadius * 0.6, length(shift) * 1.6);
            float k = 1.0 - smoothstep(core, core + ramp, segmentDistance(p, e.iris, e.iris + shift));

            float2 disp = shift * k * opening;
            if (opening > 0.0 && dot(disp, disp) > 0.0) {
                // Never pull skin into the eye: shorten the displacement until the
                // sample point is inside the opening.
                float t = 1.0;
                for (int j = 0; j < 4 && openingDistance(p - disp * t, e) < 0.5; j++) t *= 0.6;
                disp *= t;
            }
            q -= disp;

            if (u.debug != 0) {
                float od = openingDistance(p, e);
                float px = max(0.8, e.width * 0.01);
                if (abs(od) < px) overlay = float4(0, 1, 0, 1);
                if (length(p - e.iris) < px * 3.0) overlay = float4(1, 0, 0, 1);
                if (length(p - (e.iris + e.shift)) < px * 3.0) overlay = float4(0, 0.6, 1, 1);
                if (abs(length(p - (e.iris + shift)) - e.irisRadius) < px * 0.75) overlay = float4(1, 1, 0, 1);
            }
        }

        if (u.flipY) q.y = u.size.y - q.y;
        float4 c = src.sample(s, q / u.size);
        if (overlay.a > 0) c = float4(overlay.rgb * c.a, c.a);
        return c;
    }

    struct DownsampleUniforms {
        float2 dstSize;
        int flipY;
        int pad;
        float2 rectOrigin;
        float2 rectSize;
    };

    fragment float4 downsample(VOut in [[stage_in]],
                               texture2d<float, access::sample> src [[texture(0)]],
                               constant DownsampleUniforms &u [[buffer(0)]],
                               sampler s [[sampler(0)]])
    {
        float2 uv = u.rectOrigin + in.pos.xy / u.dstSize * u.rectSize;
        if (u.flipY != 0) uv.y = 1.0 - uv.y;
        // 4-tap box filter to cut aliasing when shrinking 4K to tracking size.
        float2 o = 0.25 / u.dstSize * u.rectSize;
        float4 c = src.sample(s, uv + float2(-o.x, -o.y)) + src.sample(s, uv + float2(o.x, -o.y))
                 + src.sample(s, uv + float2(-o.x, o.y)) + src.sample(s, uv + float2(o.x, o.y));
        return float4(clamp(c.rgb * 0.25, 0.0, 1.0), 1.0);
    }
    """
}
