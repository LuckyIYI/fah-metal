import CoreGraphics
import Foundation
import simd

/// The upright frame: the buffer rotated clockwise by `rotation` into a canvas (W, H); intrinsics and the pixel maps
/// both ways. Quarter turns are exact pixel permutations.
struct Geometry {
    struct Intr { var fx: Double, fy: Double, cx: Double, cy: Double }
    let srcW: Int, srcH: Int
    let W: Int, H: Int
    let quarter: Bool
    let A_us: (Double, Double, Double, Double, Double, Double)  // buffer -> upright
    let A_su: (Double, Double, Double, Double, Double, Double)  // upright -> buffer
    let K: Intr, Kflip: Intr
    let camRot: simd_double3x3  // X_upright = camRot X_buffer

    init(width: Int, height: Int, rotation: FAHRotation, K k: simd_float3x3) {
        srcW = width; srcH = height
        let (c, s) = rotation.cosSin
        quarter = (c == 0 || s == 0)
        if quarter {
            (W, H) = c == 0 ? (height, width) : (width, height)
        } else {
            W = Int((abs(c) * Double(width) + abs(s) * Double(height) - 1e-9).rounded(.up))
            H = Int((abs(s) * Double(width) + abs(c) * Double(height) - 1e-9).rounded(.up))
        }
        let cs = SIMD2(Double(width - 1) / 2, Double(height - 1) / 2), cu = SIMD2(Double(W - 1) / 2, Double(H - 1) / 2)
        // p_u = R (p_s - cs) + cu, R = [[c, -s], [s, c]]
        A_us = (c, -s, cu.x - (c * cs.x - s * cs.y), s, c, cu.y - (s * cs.x + c * cs.y))
        A_su = (c, s, cs.x - (c * cu.x + s * cu.y), -s, c, cs.y - (-s * cu.x + c * cu.y))
        camRot = rotation.cameraRotation
        let fx = Double(k[0][0]), fy = Double(k[1][1]), cx = Double(k[2][0]), cy = Double(k[2][1])
        let cxu = A_us.0 * cx + A_us.1 * cy + A_us.2, cyu = A_us.3 * cx + A_us.4 * cy + A_us.5
        let (fxu, fyu): (Double, Double) = quarter ? (c == 0 ? (fy, fx) : (fx, fy)) : ((fx + fy) / 2, (fx + fy) / 2)
        K = Intr(fx: fxu, fy: fyu, cx: cxu, cy: cyu)
        Kflip = Intr(fx: fxu, fy: fyu, cx: Double(W - 1) - cxu, cy: cyu)
    }

    func toSensor(_ p: SIMD2<Double>) -> SIMD2<Double> {
        SIMD2(A_su.0 * p.x + A_su.1 * p.y + A_su.2, A_su.3 * p.x + A_su.4 * p.y + A_su.5)
    }
    func toUpright(_ p: SIMD2<Double>) -> SIMD2<Double> {
        SIMD2(A_us.0 * p.x + A_us.1 * p.y + A_us.2, A_us.3 * p.x + A_us.4 * p.y + A_us.5)
    }
    private func mapBox(_ b: SIMD4<Double>, _ f: (SIMD2<Double>) -> SIMD2<Double>) -> SIMD4<Double> {
        let ps = [f(SIMD2(b.x, b.y)), f(SIMD2(b.z, b.y)), f(SIMD2(b.x, b.w)), f(SIMD2(b.z, b.w))]
        var lo = ps[0], hi = ps[0]
        for p in ps { lo = simd_min(lo, p); hi = simd_max(hi, p) }
        return SIMD4(lo.x, lo.y, hi.x, hi.y)
    }
    func boxToSensor(_ b: SIMD4<Double>) -> SIMD4<Double> { mapBox(b, toSensor) }
    func boxToUpright(_ b: SIMD4<Double>) -> SIMD4<Double> { mapBox(b, toUpright) }
}

@inline(__always) func flipBox(_ b: SIMD4<Double>, _ width: Int) -> SIMD4<Double> {
    SIMD4(Double(width) - b.z - 1, b.y, Double(width) - b.x - 1, b.w)
}

@inline(__always) func rect(_ b: SIMD4<Double>) -> CGRect { CGRect(x: b.x, y: b.y, width: b.z - b.x, height: b.w - b.y) }

@inline(__always) func now() -> Double { ProcessInfo.processInfo.systemUptime }

/// Python's list.sort (stable).
func stableSorted<T>(_ a: [T], by less: (T, T) -> Bool) -> [T] {
    a.enumerated().sorted { less($0.element, $1.element) || (!less($1.element, $0.element) && $0.offset < $1.offset) }.map { $0.element }
}

@inline(__always) func f3(_ v: SIMD3<Double>) -> SIMD3<Float> { SIMD3(Float(v.x), Float(v.y), Float(v.z)) }
@inline(__always) func d3(_ v: SIMD3<Float>) -> SIMD3<Double> { SIMD3(Double(v.x), Double(v.y), Double(v.z)) }
@inline(__always) func f33(_ m: simd_double3x3) -> simd_float3x3 { simd_float3x3(columns: (f3(m[0]), f3(m[1]), f3(m[2]))) }
