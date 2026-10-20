import Foundation
import Metal
import simd

/// GPU side of the learned gaze redirection: cuts the two eye patches out of the frame for the
/// network, then applies the network's fields to the full-resolution frame.
public final class GazeRenderer {
    public let device: MTLDevice
    private let library: MTLLibrary
    private let sampler: MTLSamplerState
    private var pipelines: [String: MTLRenderPipelineState] = [:]
    private let lock = NSLock()

    /// Layout must match `GazeEye` in the shader.
    struct EyeParams {
        var a0: SIMD2<Float>, a1: SIMD2<Float>, t: SIMD2<Float>      // frame -> patch
        var i0: SIMD2<Float>, i1: SIMD2<Float>, ti: SIMD2<Float>     // patch -> frame
        var iris: SIMD2<Float>, target: SIMD2<Float>                 // debug markers, frame px
        var active: Float = 1
        var pad: Float = 0
    }

    struct Uniforms {
        var srcSize: SIMD2<Float>
        var patchSize = SIMD2<Float>(Float(GazePatch.width), Float(GazePatch.height))
        var eyeCount: Int32
        var debug: Int32
        var flipY: Int32 = 0
        var pad: Int32 = 0
    }

    public init(device: MTLDevice) throws {
        self.device = device
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
        d.vertexFunction = library.makeFunction(name: "gazeFullscreen")
        d.fragmentFunction = library.makeFunction(name: fragment)
        d.colorAttachments[0].pixelFormat = format
        let p = try device.makeRenderPipelineState(descriptor: d)
        pipelines[key] = p
        return p
    }

    private func params(_ request: GazeRequest) -> [EyeParams] {
        request.patches.enumerated().map { k, f in
            let ai = f.A.inverse
            // Debug marker: where the iris is being sent (frame px), from the patch-space request.
            let s = GazePatch.eyeFraction * Float(GazePatch.width)
            let irisQ = f.toPatch(f.eye.iris)
            let targetQ = irisQ + request.displacement[k] * s
            return EyeParams(a0: SIMD2(f.A[0][0], f.A[1][0]), a1: SIMD2(f.A[0][1], f.A[1][1]), t: f.t,
                             i0: SIMD2(ai[0][0], ai[1][0]), i1: SIMD2(ai[0][1], ai[1][1]), ti: -(ai * f.t),
                             iris: f.eye.iris, target: f.toFrame(targetQ))
        }
    }

    private func encoder(_ cb: MTLCommandBuffer, target: MTLTexture) -> MTLRenderCommandEncoder? {
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = target
        pass.colorAttachments[0].loadAction = .dontCare
        pass.colorAttachments[0].storeAction = .store
        return cb.makeRenderCommandEncoder(descriptor: pass)
    }

    // MARK: Patches

    /// A 144x192 float texture holding both eye patches (eye k in rows k*96 ..< (k+1)*96).
    public func makePatchTexture() -> MTLTexture {
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Float, width: GazePatch.width,
                                                         height: GazePatch.height * 2, mipmapped: false)
        d.usage = [.renderTarget, .shaderRead]
        d.storageMode = .shared
        return device.makeTexture(descriptor: d)!
    }

    /// `flipY`: the source texture is stored bottom-up (Final Cut / Motion); geometry in
    /// `request` is always upright.
    public func encodePatches(into cb: MTLCommandBuffer, source: MTLTexture, patchTexture: MTLTexture,
                              request: GazeRequest, flipY: Bool = false) throws {
        let ps = try pipeline("gazePatch", format: patchTexture.pixelFormat)
        guard let enc = encoder(cb, target: patchTexture) else { return }
        var eyes = params(request)
        var u = Uniforms(srcSize: SIMD2(Float(source.width), Float(source.height)), eyeCount: 2, debug: 0,
                         flipY: flipY ? 1 : 0)
        enc.setRenderPipelineState(ps)
        enc.setFragmentTexture(source, index: 0)
        enc.setFragmentBytes(&eyes, length: MemoryLayout<EyeParams>.stride * eyes.count, index: 0)
        enc.setFragmentBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 1)
        enc.setFragmentSamplerState(sampler, index: 0)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        enc.endEncoding()
    }

    /// Reads the patch texture (after its command buffer completed) as [2][3][96][144].
    public func readPatches(_ tex: MTLTexture) -> [Float] {
        let W = GazePatch.width, H = GazePatch.height * 2
        var rgba = [Float](repeating: 0, count: W * H * 4)
        tex.getBytes(&rgba, bytesPerRow: W * 16, from: MTLRegionMake2D(0, 0, W, H), mipmapLevel: 0)
        var out = [Float](repeating: 0, count: 2 * 3 * GazePatch.height * W)
        let plane = GazePatch.height * W
        for k in 0..<2 {
            for y in 0..<GazePatch.height {
                for x in 0..<W {
                    let src = ((k * GazePatch.height + y) * W + x) * 4
                    let dst = k * 3 * plane + y * W + x
                    out[dst] = rgba[src]
                    out[dst + plane] = rgba[src + 1]
                    out[dst + 2 * plane] = rgba[src + 2]
                }
            }
        }
        return out
    }

    // MARK: Warp

    /// Textures for one eye's fields: (flow.x, flow.y, alpha, 0) and (r, g, b, 1), 72x48.
    public func makeFieldTextures(_ fields: GazeFields) -> [(MTLTexture, MTLTexture)] {
        let W = GazeFields.width, H = GazeFields.height, n = W * H
        func texture(_ data: [Float]) -> MTLTexture {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Float, width: W, height: H, mipmapped: false)
            d.usage = .shaderRead
            d.storageMode = .shared
            let t = device.makeTexture(descriptor: d)!
            data.withUnsafeBytes { t.replace(region: MTLRegionMake2D(0, 0, W, H), mipmapLevel: 0,
                                             withBytes: $0.baseAddress!, bytesPerRow: W * 16) }
            return t
        }
        return (0..<2).map { k in
            var f = [Float](repeating: 0, count: n * 4), c = [Float](repeating: 1, count: n * 4)
            for i in 0..<n {
                f[i * 4] = fields.flow[(k * 2) * n + i]
                f[i * 4 + 1] = fields.flow[(k * 2 + 1) * n + i]
                f[i * 4 + 2] = fields.alpha[k * n + i]
                for ch in 0..<3 { c[i * 4 + ch] = fields.color[(k * 3 + ch) * n + i] }
            }
            return (texture(f), texture(c))
        }
    }

    /// Writes `source` with the eyes redirected into `destination`. With no request, copies.
    public func encodeWarp(into cb: MTLCommandBuffer, source: MTLTexture, destination: MTLTexture,
                           request: GazeRequest?, fields: [(MTLTexture, MTLTexture)]?, debug: Bool,
                           flipY: Bool = false) throws {
        let ps = try pipeline("gazeWarp", format: destination.pixelFormat)
        guard let enc = encoder(cb, target: destination) else { return }
        var eyes = request.map(params) ?? []
        let active = request != nil && fields != nil
        if !active { eyes = [EyeParams(a0: .zero, a1: .zero, t: .zero, i0: .zero, i1: .zero, ti: .zero,
                                       iris: .zero, target: .zero, active: 0)] }
        var u = Uniforms(srcSize: SIMD2(Float(source.width), Float(source.height)),
                         eyeCount: active ? 2 : 0, debug: debug ? 1 : 0, flipY: flipY ? 1 : 0)
        enc.setRenderPipelineState(ps)
        enc.setFragmentTexture(source, index: 0)
        if let fields, active {
            enc.setFragmentTexture(fields[0].0, index: 1)
            enc.setFragmentTexture(fields[0].1, index: 2)
            enc.setFragmentTexture(fields[1].0, index: 3)
            enc.setFragmentTexture(fields[1].1, index: 4)
        }
        enc.setFragmentBytes(&eyes, length: MemoryLayout<EyeParams>.stride * eyes.count, index: 0)
        enc.setFragmentBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 1)
        enc.setFragmentSamplerState(sampler, index: 0)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        enc.endEncoding()
    }

    static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    struct GazeEye {
        float2 a0, a1, t;    // frame -> patch
        float2 i0, i1, ti;   // patch -> frame
        float2 iris, target;
        float active, pad;
    };

    struct GazeUniforms {
        float2 srcSize;
        float2 patchSize;
        int eyeCount;
        int debug;
        int flipY;   // texture rows run bottom-up; all geometry is in upright coordinates
        int pad;
    };

    // Upright frame coordinates <-> texture coordinates (a vertical flip when stored bottom-up).
    static float2 toTex(constant GazeUniforms &u, float2 c) { return u.flipY ? float2(c.x, u.srcSize.y - c.y) : c; }

    struct VOut { float4 pos [[position]]; };

    vertex VOut gazeFullscreen(uint vid [[vertex_id]]) {
        float2 uv = float2((vid << 1) & 2, vid & 2);
        VOut o;
        o.pos = float4(uv * 2.0 - 1.0, 0.0, 1.0);
        return o;
    }

    static float2 toPatch(constant GazeEye &e, float2 p) { return float2(dot(e.a0, p), dot(e.a1, p)) + e.t; }
    static float2 toFrame(constant GazeEye &e, float2 q) { return float2(dot(e.i0, q), dot(e.i1, q)) + e.ti; }

    // Patch pixel j (fragment position j + 0.5) samples the frame at toFrame(j), 3x3 supersampled
    // like the training data (warped at 2x, then area-downsampled).
    fragment float4 gazePatch(VOut in [[stage_in]],
                              texture2d<float, access::sample> src [[texture(0)]],
                              constant GazeEye *eyes [[buffer(0)]],
                              constant GazeUniforms &u [[buffer(1)]],
                              sampler s [[sampler(0)]])
    {
        int k = in.pos.y >= u.patchSize.y ? 1 : 0;
        float2 q = float2(in.pos.x, in.pos.y - k * u.patchSize.y) - 0.5;
        float3 c = 0;
        for (int j = -1; j <= 1; j++)
            for (int i = -1; i <= 1; i++)
                c += src.sample(s, toTex(u, toFrame(eyes[k], q + float2(i, j) / 3.0)) / u.srcSize).rgb;
        return float4(clamp(c / 9.0, 0.0, 1.0), 1.0);
    }

    // Catmull-Rom bicubic at continuous pixel coordinates (texel i centered at i + 0.5), so the
    // redirected eye keeps the source's sharpness.
    static float4 bicubic(texture2d<float, access::sample> t, float2 c) {
        float2 x = c - 0.5;
        float2 i = floor(x);
        float2 f = x - i;
        float2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
        float2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
        float2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
        float2 w3 = f * f * (-0.5 + 0.5 * f);
        float wx[4] = {w0.x, w1.x, w2.x, w3.x};
        float wy[4] = {w0.y, w1.y, w2.y, w3.y};
        int2 maxc = int2(t.get_width() - 1, t.get_height() - 1);
        float4 sum = 0, lo = float4(1e9), hi = float4(-1e9);
        for (int y = 0; y < 4; y++) {
            for (int xx = 0; xx < 4; xx++) {
                int2 p = clamp(int2(i) + int2(xx - 1, y - 1), int2(0), maxc);
                float4 v = t.read(uint2(p));
                sum += v * wx[xx] * wy[y];
                if (xx >= 1 && xx <= 2 && y >= 1 && y <= 2) { lo = min(lo, v); hi = max(hi, v); }
            }
        }
        // Clamp to the 2x2 neighborhood: no ringing halos around the iris.
        return clamp(sum, lo, hi);
    }

    fragment float4 gazeWarp(VOut in [[stage_in]],
                             texture2d<float, access::sample> src [[texture(0)]],
                             texture2d<float, access::sample> f0 [[texture(1)]],
                             texture2d<float, access::sample> c0 [[texture(2)]],
                             texture2d<float, access::sample> f1 [[texture(3)]],
                             texture2d<float, access::sample> c1 [[texture(4)]],
                             constant GazeEye *eyes [[buffer(0)]],
                             constant GazeUniforms &u [[buffer(1)]],
                             sampler s [[sampler(0)]])
    {
        float4 result = src.read(uint2(in.pos.xy));
        float2 p = toTex(u, in.pos.xy);  // upright frame coordinates of this pixel
        float2 P = u.patchSize;

        for (int k = 0; k < u.eyeCount; k++) {
            constant GazeEye &e = eyes[k];
            if (e.active == 0) continue;
            float2 q = toPatch(e, p);
            if (any(q < -0.5) || any(q > P - 0.5)) continue;

            // Fields are at half resolution; uv follows PyTorch's align_corners=False upsampling.
            float2 uv = (q + 0.5) / P;
            float4 F = k == 0 ? f0.sample(s, uv) : f1.sample(s, uv);
            float3 C = (k == 0 ? c0.sample(s, uv) : c1.sample(s, uv)).rgb;
            // Flow is in align_corners=True normalized coordinates: [-1, 1] spans 0 .. P-1.
            float2 q2 = q + F.xy * (P - 1.0) * 0.5;
            float4 warped = bicubic(src, toTex(u, toFrame(e, q2)));
            float4 outc = float4(mix(warped.rgb, C * warped.a, F.z), warped.a);

            // Fade to the original near the patch border (the network's gate already keeps
            // everything away from the eye unchanged).
            float edge = min(min(q.x, P.x - 1.0 - q.x), min(q.y, P.y - 1.0 - q.y));
            result = mix(result, outc, clamp(edge / (0.12 * P.y), 0.0, 1.0));

            if (u.debug != 0) {
                // Patch outline about 1.5 frame pixels wide (|a0| = patch px per frame px).
                if (abs(edge) < 1.5 * length(e.a0)) result = float4(0, 1, 0, 1);
                if (length(p - e.iris) < 3.0) result = float4(1, 0, 0, 1);
                if (length(p - e.target) < 3.0) result = float4(0, 0.6, 1, 1);
            }
        }
        return result;
    }
    """
}
