import CoreVideo
import Foundation
import Metal

/// GPU pre-processing straight from the camera buffer (BGRA or bi-planar 4:2:0 YCbCr, no full-frame RGB copy):
///  - the detector letterbox (OpenCV's INTER_LINEAR resize of the upright image + gray-114 padding) into the detector's
///    fp16 NCHW input,
///  - the pose crops (OpenCV's warpAffine: float source coordinates, bilinear, zero border, uint8 rounding) into the
///    pose network's fp32 NCHW input.
/// Both emulate the uint8 OpenCV path of FAH's Python pipeline, so the networks see the same pixels as in Python.
final class ImageWarp {
    let device: MTLDevice
    private var textureCache: CVMetalTextureCache?
    private let pipes: [String: MTLComputePipelineState]

    struct Source {
        let tex0: MTLTexture
        let tex1: MTLTexture?
        let isYUV: Bool
        let width: Int, height: Int
        let refs: [CVMetalTexture]  // keep the CV textures alive while the GPU reads them
        let yuv: YUVParams
    }

    struct YUVParams {
        var m0: SIMD4<Float>   // R = Y' + m0.z * Cr'                      (rows: coefficients of (Y', Cb', Cr'))
        var m1: SIMD4<Float>   // G
        var m2: SIMD4<Float>   // B
        var off: SIMD4<Float>  // (y offset, y scale, c offset, c scale)
    }

    init(device: MTLDevice) throws {
        self.device = device
        CVMetalTextureCacheCreate(nil, nil, device, nil, &textureCache)
        let lib: MTLLibrary
        do { lib = try device.makeLibrary(source: try shaderSource("image"), options: nil) } catch { throw FAHError.metal("image.metal: \(error)") }
        var p: [String: MTLComputePipelineState] = [:]
        for name in ["crop_bgra", "crop_yuv", "letterbox_bgra", "letterbox_yuv"] {
            guard let f = lib.makeFunction(name: name) else { throw FAHError.metal("no function \(name)") }
            p[name] = try device.makeComputePipelineState(function: f)
        }
        pipes = p
    }

    /// Drops the texture cache's unused textures (once per frame).
    func flush() { if let c = textureCache { CVMetalTextureCacheFlush(c, 0) } }

    func source(_ pb: CVPixelBuffer) throws -> Source {
        guard let cache = textureCache else { throw FAHError.metal("no texture cache") }
        let fmt = CVPixelBufferGetPixelFormatType(pb)
        let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
        func tex(_ plane: Int, _ pf: MTLPixelFormat, _ tw: Int, _ th: Int) throws -> (MTLTexture, CVMetalTexture) {
            var out: CVMetalTexture?
            let r = CVMetalTextureCacheCreateTextureFromImage(nil, cache, pb, nil, pf, tw, th, plane, &out)
            guard r == kCVReturnSuccess, let o = out, let t = CVMetalTextureGetTexture(o) else {
                throw FAHError.metal("texture from pixel buffer failed (\(r)); the buffer must be IOSurface / Metal compatible")
            }
            return (t, o)
        }
        switch fmt {
        case kCVPixelFormatType_32BGRA:
            let (t, r) = try tex(0, .bgra8Unorm, w, h)
            return Source(tex0: t, tex1: nil, isYUV: false, width: w, height: h, refs: [r], yuv: ImageWarp.yuvParams(pb, full: true))
        case kCVPixelFormatType_420YpCbCr8BiPlanarFullRange, kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange:
            let (t0, r0) = try tex(0, .r8Unorm, CVPixelBufferGetWidthOfPlane(pb, 0), CVPixelBufferGetHeightOfPlane(pb, 0))
            let (t1, r1) = try tex(1, .rg8Unorm, CVPixelBufferGetWidthOfPlane(pb, 1), CVPixelBufferGetHeightOfPlane(pb, 1))
            return Source(tex0: t0, tex1: t1, isYUV: true, width: w, height: h, refs: [r0, r1],
                          yuv: ImageWarp.yuvParams(pb, full: fmt == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange))
        default:
            throw FAHError.unsupportedPixelFormat(fmt)
        }
    }

    /// BT.601 unless the buffer says BT.709 (ARKit's capturedImage: 601 full range).
    static func yuvParams(_ pb: CVPixelBuffer, full: Bool) -> YUVParams {
        var is709 = false
        if let m = CVBufferCopyAttachment(pb, kCVImageBufferYCbCrMatrixKey, nil) as? String {
            is709 = (m == (kCVImageBufferYCbCrMatrix_ITU_R_709_2 as String))
        }
        let (kr, kb): (Float, Float) = is709 ? (0.2126, 0.0722) : (0.299, 0.114)
        let kg = 1 - kr - kb
        let m0 = SIMD4<Float>(1, 0, 2 * (1 - kr), 0)
        let m1 = SIMD4<Float>(1, -2 * (1 - kb) * kb / kg, -2 * (1 - kr) * kr / kg, 0)
        let m2 = SIMD4<Float>(1, 2 * (1 - kb), 0, 0)
        let off = full ? SIMD4<Float>(0, 1, 128.0 / 255, 1) : SIMD4<Float>(16.0 / 255, 255.0 / 219, 128.0 / 255, 255.0 / 224)
        return YUVParams(m0: m0, m1: m1, m2: m2, off: off)
    }

    /// OpenCV (5.x) warpAffine source coordinates: the frame -> crop affine M inverted as cv::invertAffineTransform (double),
    /// then evaluated per pixel in float32 (dst -> canvas), bilinear in float, rounded to uint8 (half to even).
    static func cvInverse(_ m: [Double]) -> [Float] {
        var D = m[0] * m[4] - m[1] * m[3]
        D = D != 0 ? 1 / D : 0
        let a11 = m[4] * D, a22 = m[0] * D, a12 = -m[1] * D, a21 = -m[3] * D
        let b1 = -a11 * m[2] - a12 * m[5], b2 = -a21 * m[2] - a22 * m[5]
        return [Float(a11), Float(a12), Float(b1), Float(a21), Float(a22), Float(b2), 0, 0]
    }

    /// Encodes the crops: per crop the frame -> crop affine in the upright canvas; out (n, 3, S, S) fp32 RGB [0, 1].
    /// canvasToSource: the upright canvas -> buffer pixel map.
    func encodeCrops(_ cb: MTLCommandBuffer, src: Source, affines: [[Double]], out: MTLBuffer,
                     canvasToSource a: (Double, Double, Double, Double, Double, Double), size S: Int) throws {
        var params = CropParams(dims: SIMD4(Int32(S), Int32(src.width), Int32(src.height), 0),
                                f0: SIMD4(Float(a.0), Float(a.1), Float(a.2), 0), f1: SIMD4(Float(a.3), Float(a.4), Float(a.5), 0))
        var y = src.yuv
        var m = affines.flatMap { ImageWarp.cvInverse($0) }
        guard let enc = cb.makeComputeCommandEncoder() else { throw FAHError.metal("encoder") }
        enc.setComputePipelineState(pipes[src.isYUV ? "crop_yuv" : "crop_bgra"]!)
        enc.setTexture(src.tex0, index: 0)
        enc.setTexture(src.tex1 ?? src.tex0, index: 1)
        enc.setBuffer(out, offset: 0, index: 0)
        enc.setBytes(&m, length: m.count * 4, index: 1)
        enc.setBytes(&params, length: MemoryLayout<CropParams>.stride, index: 2)
        enc.setBytes(&y, length: MemoryLayout<YUVParams>.stride, index: 3)
        enc.dispatchThreads(MTLSize(width: S, height: S, depth: affines.count), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        enc.endEncoding()
    }

    struct CropParams {
        var dims: SIMD4<Int32>   // S, source w, source h, 0
        var f0: SIMD4<Float>     // canvas -> source affine rows
        var f1: SIMD4<Float>
    }

    struct LetterboxParams {
        var outSize: SIMD2<Int32>      // (W, H) of the network input
        var resized: SIMD2<Int32>      // (nw, nh) of the resized upright image
        var pad: SIMD2<Int32>          // (px, py) total offset of the resized image in the input
        var canvas: SIMD2<Int32>       // upright canvas (w, h)
        var scale: SIMD2<Float>        // canvas / resized per axis (cv2 resize scale_x, scale_y)
        var srcSize: SIMD2<Int32>      // buffer (w, h)
        var a0: SIMD4<Float>           // canvas -> buffer affine row 0 (a, b, c, 0)
        var a1: SIMD4<Float>           // row 1
        var quarter: Int32             // canvas pixels are buffer pixels (quarter turn): no resampling
    }

    /// Encodes the letterbox: out (1, 3, H, W) fp16 RGB [0, 1].
    func encodeLetterbox(_ cb: MTLCommandBuffer, src: Source, params: LetterboxParams, out: MTLBuffer) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pipes[src.isYUV ? "letterbox_yuv" : "letterbox_bgra"]!)
        enc.setTexture(src.tex0, index: 0)
        enc.setTexture(src.tex1 ?? src.tex0, index: 1)
        var p = params
        enc.setBytes(&p, length: MemoryLayout<LetterboxParams>.stride, index: 1)
        enc.setBuffer(out, offset: 0, index: 0)
        var y = src.yuv
        enc.setBytes(&y, length: MemoryLayout<YUVParams>.stride, index: 3)
        enc.dispatchThreads(MTLSize(width: Int(params.outSize.x), height: Int(params.outSize.y), depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        enc.endEncoding()
    }
}

/// The source of one of the package's shaders (Shaders/<name>.metal). They are compiled at run time: pose.metal is
/// specialised for the GPU (tensor ops or simdgroup matrices) with preprocessor macros.
func shaderSource(_ name: String) throws -> String {
    guard let url = Bundle.module.url(forResource: name, withExtension: "metal", subdirectory: "Shaders") else {
        throw FAHError.missingResource("Shaders/\(name).metal")
    }
    return try String(contentsOf: url, encoding: .utf8)
}
