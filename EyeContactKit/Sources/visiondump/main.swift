import CoreGraphics
import CoreVideo
import EyeContactCore
import Foundation
import ImageIO

// Dumps EyeTracker measurements for images (one JSON object per line), measured the way the
// plug-in does: on a copy whose longest side is 1280 px, scaled back to full-image pixels.
//   visiondump image1.jpg image2.jpg ... > vision.jsonl

let trackingSize = 1280
let tracker = EyeTracker()

func pixelBuffer(_ url: URL) -> (CVPixelBuffer, Float)? {
    guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
          let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else { return nil }
    let scale = min(1, Float(trackingSize) / Float(max(img.width, img.height)))
    let w = Int((Float(img.width) * scale).rounded()), h = Int((Float(img.height) * scale).rounded())
    var pb: CVPixelBuffer?
    CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_32BGRA, nil, &pb)
    guard let pb else { return nil }
    CVPixelBufferLockBaseAddress(pb, [])
    defer { CVPixelBufferUnlockBaseAddress(pb, []) }
    guard let ctx = CGContext(data: CVPixelBufferGetBaseAddress(pb), width: w, height: h, bitsPerComponent: 8,
                              bytesPerRow: CVPixelBufferGetBytesPerRow(pb), space: CGColorSpaceCreateDeviceRGB(),
                              bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)
    else { return nil }
    ctx.interpolationQuality = .high
    ctx.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
    return (pb, Float(img.width) / Float(w))
}

for path in CommandLine.arguments.dropFirst() {
    autoreleasepool {
        var out: [String: Any] = ["path": path]
        if let (pb, s) = pixelBuffer(URL(fileURLWithPath: path)), let face = tracker.measure(pb) {
            out["eyes"] = face.scaled(s).eyes.map { e -> [String: Any] in
                ["mid": [e.mid.x, e.mid.y], "axis": [e.axis.x, e.axis.y], "width": e.width,
                 "upper": [e.upperLid.a, e.upperLid.b], "lower": [e.lowerLid.a, e.lowerLid.b],
                 "iris": [e.iris.x, e.iris.y]]
            }
            out["yaw"] = face.yawDegrees ?? NSNull()
        }
        let data = try! JSONSerialization.data(withJSONObject: out)
        print(String(data: data, encoding: .utf8)!)
    }
}
