import CoreGraphics
import Foundation
import simd

/// One Euro filter (Casiez et al. 2012) per hand side, a light default for `FAHPipeline.smoother`: positions (wrist,
/// joints, dense surface) per coordinate, rotations on sign-aligned quaternions (renormalised), 2D joints with their own
/// parameters. Defaults follow FAH's video-mode One Euro settings (3D: min cutoff 12 Hz, beta 1; 2D: 30 Hz, beta
/// 0.05 per px/s), i.e. jitter removal without visible lag. Hands are matched to the previous frame's by side; a side
/// unseen for `resetAfter` seconds starts fresh.
public final class FAHOneEuroSmoother: FAHSmoother {
    public var minCutoff: Double
    public var beta: Double
    public var minCutoff2D: Double
    public var beta2D: Double
    public var derivativeCutoff: Double = 1.0
    public var resetAfter: Double = 0.25

    private final class State {
        var t: Double
        var x: [Double], dx: [Double]
        init(t: Double, x: [Double]) { self.t = t; self.x = x; dx = [Double](repeating: 0, count: x.count) }
    }
    private var states: [String: State] = [:]

    public init(minCutoff: Double = 12, beta: Double = 1, minCutoff2D: Double = 30, beta2D: Double = 0.05) {
        self.minCutoff = minCutoff
        self.beta = beta
        self.minCutoff2D = minCutoff2D
        self.beta2D = beta2D
    }

    public func reset() { states = [:] }

    private func alpha(_ cutoff: Double, _ dt: Double) -> Double {
        let tau = 1 / (2 * Double.pi * cutoff)
        return 1 / (1 + tau / dt)
    }

    private func filter(_ key: String, _ x: [Double], _ t: Double, _ mc: Double, _ b: Double) -> [Double] {
        guard let s = states[key], s.x.count == x.count, t > s.t, t - s.t < resetAfter else {
            states[key] = State(t: t, x: x)
            return x
        }
        let dt = t - s.t
        let ad = alpha(derivativeCutoff, dt)
        var out = x
        for i in x.indices {
            let d = (x[i] - s.x[i]) / dt
            s.dx[i] += ad * (d - s.dx[i])
            let a = alpha(mc + b * abs(s.dx[i]), dt)
            out[i] = s.x[i] + a * (x[i] - s.x[i])
        }
        s.x = out
        s.t = t
        return out
    }

    private func quats(_ ms: [simd_float3x3], _ key: String) -> [Double] {
        let prev: [Double]? = states[key]?.x
        var out: [Double] = []
        for (i, m) in ms.enumerated() {
            var q = simd_quatf(m).vector
            if let p = prev, p.count == 4 * ms.count {
                let pq = SIMD4<Float>(Float(p[4 * i]), Float(p[4 * i + 1]), Float(p[4 * i + 2]), Float(p[4 * i + 3]))
                if simd_dot(q, pq) < 0 { q = -q }
            }
            out += [Double(q.x), Double(q.y), Double(q.z), Double(q.w)]
        }
        return out
    }

    private func mats(_ v: [Double]) -> [simd_float3x3] {
        var out: [simd_float3x3] = []
        for i in stride(from: 0, to: v.count, by: 4) {
            let q = SIMD4<Float>(Float(v[i]), Float(v[i + 1]), Float(v[i + 2]), Float(v[i + 3]))
            out.append(simd_float3x3(simd_quatf(vector: simd_normalize(q))))
        }
        return out
    }

    public func smooth(_ hands: [FAHHand], timestamp t: Double) -> [FAHHand] {
        var seen = Set<String>()
        return hands.map { h0 in
            var h = h0
            let side = h.isRight ? "R" : "L"
            if seen.contains(side) { return h }  // a second hand of one side: unfiltered
            seen.insert(side)
            let p3 = filter(side + "j3", h.joints3d.flatMap { [Double($0.x), Double($0.y), Double($0.z)] }, t, minCutoff, beta)
            h.joints3d = stride(from: 0, to: p3.count, by: 3).map { SIMD3(Float(p3[$0]), Float(p3[$0 + 1]), Float(p3[$0 + 2])) }
            let shift = h.joints3d[0] - h0.joints3d[0]  // the dense surface follows the filtered wrist rigidly
            h.denseVertices = h.denseVertices.map { $0 + shift }
            h.meshVertices = h.meshVertices.map { $0 + shift }
            h.wristPosition = h.joints3d[0]
            let p2 = filter(side + "j2", h.joints2d.flatMap { [Double($0.x), Double($0.y)] }, t, minCutoff2D, beta2D)
            h.joints2d = stride(from: 0, to: p2.count, by: 2).map { SIMD2(Float(p2[$0]), Float(p2[$0 + 1])) }
            let r = filter(side + "rot", quats([h.rootRotation] + h.localRotations, side + "rot"), t, minCutoff, beta)
            let ms = mats(r)
            h.rootRotation = ms[0]
            h.localRotations = Array(ms[1...])
            h.localAxisAngles = h.localRotations.map { m in
                let q = simd_quatf(m)
                let a = q.angle
                return a > 1e-6 ? q.axis * (a > .pi ? a - 2 * .pi : a) : .zero
            }
            return h
        }
    }
}
