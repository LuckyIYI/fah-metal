import Accelerate
import Foundation
import simd

/// FAH's MANO mesh fit (fah.fast_mano.MeshFitter, solver "gn" with central-difference Jacobians, as `fah-live
/// --mesh_fit`): damped Gauss-Newton on MANO's 15 joint rotations, 10 shape coefficients, a global rotation (relative
/// to the Procrustes root rotation) and a translation, so that the mesh matches the predicted 778-point surface
/// (weight 0.3) and its 21 joints the predicted joints, with Huber (IRLS) weights at 1 cm. Then the mesh is placed
/// rigidly so that its joints project onto the 2D keypoints (bench.methods.fah._align_2d, a PnP).
///
/// One Gauss-Newton iteration evaluates MANO for the parameters and their 122 central-difference perturbations as one
/// batch: the shape and pose-corrective blend shapes and the skinning are a few large matrix products (Accelerate),
/// ~70 M multiply-adds. Right-hand frame throughout (callers mirror left hands).
final class ManoFitter {
    static let parameters = 61  // pose 45 | betas 10 | global rotation 3 | translation 3
    /// MANO's 16 joints + the 5 fingertip vertices -> FAH's 21 keypoints (OpenPose order)
    static let manoToOpenPose = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
    static let tipVertices = [744, 320, 443, 554, 671]

    private let m: ManoData
    private let nv: Int, nj: Int
    private let vt: [Float]        // (nv * 3) template
    private let jt: [Float]        // (nj * 3) rest joints
    private let wv: Float, wj: Float
    private let eps: Float = 1e-3

    init(_ mano: ManoData, surfaceWeight: Float = 0.3, jointWeight: Float = 1) {
        m = mano
        nv = mano.nv
        nj = mano.nparts
        vt = mano.template.flatMap { [Float($0.x), Float($0.y), Float($0.z)] }
        jt = mano.joints.flatMap { [Float($0.x), Float($0.y), Float($0.z)] }
        wv = surfaceWeight
        wj = jointWeight
    }

    struct Result {
        var vertices: [SIMD3<Double>]   // fitted mesh (right-hand frame)
        var rotation: simd_double3x3    // global rotation
        var pose: [Float]               // 45, flat-hand axis-angle
        var betas: [Float]              // 10
    }

    /// tV: the predicted surface (nv), tJ: the predicted 21 joints (OpenPose order), both right-hand frame, metres.
    /// R0 / pose0 / betas0: the start (the Procrustes root rotation and local axis-angles, mean shape).
    func fit(surface tV: [SIMD3<Double>], joints tJ: [SIMD3<Double>], R0: simd_double3x3, pose0: [Float], betas0: [Float],
             iterations: Int) -> Result {
        let P = Self.parameters
        var x = [Float](repeating: 0, count: P)
        for i in 0..<45 { x[i] = pose0[i] }
        for i in 0..<10 { x[45 + i] = betas0[i] }
        x[58] = Float(tJ[0].x); x[59] = Float(tJ[0].y); x[60] = Float(tJ[0].z)
        let R0f = simd_float3x3(SIMD3<Float>(R0.columns.0), SIMD3<Float>(R0.columns.1), SIMD3<Float>(R0.columns.2))
        var target = [Float](repeating: 0, count: 3 * nv + 63)
        for v in 0..<nv { for c in 0..<3 { target[3 * v + c] = Float(tV[v][c]) } }
        for j in 0..<21 { for c in 0..<3 { target[3 * nv + 3 * j + c] = Float(tJ[j][c]) } }
        let sv = (wv / Float(nv)).squareRoot(), sj = (wj / 21).squareRoot()
        let nr = 3 * nv + 63
        for _ in 0..<iterations {
            // the parameters and their +- eps perturbations, one MANO batch
            let N = 1 + 2 * P
            var X = [Float](repeating: 0, count: N * P)
            for n in 0..<N { for i in 0..<P { X[n * P + i] = x[i] } }
            for i in 0..<P { X[(1 + i) * P + i] += eps; X[(1 + P + i) * P + i] -= eps }
            let out = forward(X, N, R0f)  // (N, nr): vertices then joints
            // weighted residuals of the base point, central-difference Jacobian (nr, P)
            var r = [Float](repeating: 0, count: nr)
            var Jac = [Float](repeating: 0, count: nr * P)
            for k in 0..<nr {
                let s = k < 3 * nv ? sv : sj
                r[k] = (out[k] - target[k]) * s
                for i in 0..<P { Jac[k * P + i] = (out[(1 + i) * nr + k] - out[(1 + P + i) * nr + k]) * s / (2 * eps) }
            }
            // Huber IRLS weights (delta 1 cm on the unweighted residual)
            var Jw = Jac
            var rw = r
            for k in 0..<nr {
                let s = k < 3 * nv ? sv : sj
                let w = min(0.01 / max(abs(r[k]) / s, 1e-9), 1)
                rw[k] *= w
                for i in 0..<P { Jw[k * P + i] *= w }
            }
            // H = J^T W J, g = J^T W r (double for the solve)
            var H = [Float](repeating: 0, count: P * P), g = [Float](repeating: 0, count: P)
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, Int32(P), Int32(P), Int32(nr), 1, Jac, Int32(P), Jw, Int32(P), 0, &H, Int32(P))
            cblas_sgemv(CblasRowMajor, CblasTrans, Int32(nr), Int32(P), 1, Jac, Int32(P), rw, 1, 0, &g, 1)
            var Hd = H.map(Double.init)
            for i in 0..<P { Hd[i * P + i] += 1e-3 * Hd[i * P + i] + 1e-6 }
            guard let dx = Self.cholesky(Hd, g.map(Double.init), P) else { break }
            for i in 0..<P { x[i] -= Float(dx[i]) }
        }
        let out = forward(x, 1, R0f)
        let verts = (0..<nv).map { SIMD3<Double>(Double(out[3 * $0]), Double(out[3 * $0 + 1]), Double(out[3 * $0 + 2])) }
        let Rg = R0f * Self.rodrigues(SIMD3(x[55], x[56], x[57]))
        let R = simd_double3x3(SIMD3<Double>(Rg.columns.0), SIMD3<Double>(Rg.columns.1), SIMD3<Double>(Rg.columns.2))
        return Result(vertices: verts, rotation: R, pose: Array(x[0..<45]), betas: Array(x[45..<55]))
    }

    /// MANO for N parameter sets (rows of X) -> (N, nv * 3 + 63): vertices then the 21 joints (OpenPose order), root
    /// (wrist) relative, rotated by R0 rodrigues(x[55..<58]) and translated by x[58..<61] (fah.fast_mano.FastMano).
    private func forward(_ X: [Float], _ N: Int, _ R0: simd_float3x3) -> [Float] {
        let P = Self.parameters, V3 = nv * 3, J3 = nj * 3
        // shape: v = v_t + betas . shapedirs, J = J_t + betas . joint_dirs
        var betas = [Float](repeating: 0, count: N * 10)
        for n in 0..<N { for i in 0..<10 { betas[n * 10 + i] = X[n * P + 45 + i] } }
        var v = [Float](repeating: 0, count: N * V3), J = [Float](repeating: 0, count: N * J3)
        for n in 0..<N { for k in 0..<V3 { v[n * V3 + k] = vt[k] }; for k in 0..<J3 { J[n * J3 + k] = jt[k] } }
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(N), Int32(V3), 10, 1, betas, 10, m.shapedirs, Int32(V3), 1, &v, Int32(V3))
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(N), Int32(J3), 10, 1, betas, 10, m.jointDirs, Int32(J3), 1, &J, Int32(J3))
        // joint rotations and the pose-corrective blend shapes ((R - I) of joints 1..15 . posedirs)
        var Rj = [simd_float3x3](repeating: matrix_identity_float3x3, count: N * nj)
        var feat = [Float](repeating: 0, count: N * 135)
        for n in 0..<N {
            for k in 1..<nj {
                let R = Self.rodrigues(SIMD3(X[n * P + 3 * (k - 1)], X[n * P + 3 * (k - 1) + 1], X[n * P + 3 * (k - 1) + 2]))
                Rj[n * nj + k] = R
                for a in 0..<3 { for b in 0..<3 { feat[n * 135 + 9 * (k - 1) + 3 * a + b] = R[b][a] - (a == b ? 1 : 0) } }  // row-major R
            }
        }
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(N), Int32(V3), 135, 1, feat, 135, m.posedirs, Int32(V3), 1, &v, Int32(V3))
        // kinematic chain -> per-joint skinning transforms A_k = [G_k | T_k - G_k J_k], (N, 16 * 12) row-major 3x4
        var A = [Float](repeating: 0, count: N * nj * 12)
        var T = [SIMD3<Float>](repeating: .zero, count: N * nj)
        for n in 0..<N {
            var G = [simd_float3x3](repeating: matrix_identity_float3x3, count: nj)
            let Jn = (0..<nj).map { SIMD3(J[n * J3 + 3 * $0], J[n * J3 + 3 * $0 + 1], J[n * J3 + 3 * $0 + 2]) }
            T[n * nj] = Jn[0]
            for k in 1..<nj {
                let p = m.parents[k]
                G[k] = G[p] * Rj[n * nj + k]
                T[n * nj + k] = T[n * nj + p] + G[p] * (Jn[k] - Jn[p])
            }
            for k in 0..<nj {
                let t = T[n * nj + k] - G[k] * Jn[k]
                for a in 0..<3 {
                    for b in 0..<3 { A[(n * nj + k) * 12 + 4 * a + b] = G[k][b][a] }
                    A[(n * nj + k) * 12 + 4 * a + 3] = t[a]
                }
            }
        }
        // linear blend skinning: M_v = sum_k W_vk A_k for every set at once: (nv, 16) x (16, N * 12)
        var At = [Float](repeating: 0, count: nj * N * 12)  // (16, N * 12)
        for n in 0..<N { for k in 0..<nj { for e in 0..<12 { At[k * N * 12 + n * 12 + e] = A[(n * nj + k) * 12 + e] } } }
        var Mv = [Float](repeating: 0, count: nv * N * 12)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(nv), Int32(N * 12), Int32(nj), 1, m.weights, Int32(nj), At,
                    Int32(N * 12), 0, &Mv, Int32(N * 12))
        let nr = V3 + 63
        var out = [Float](repeating: 0, count: N * nr)
        for n in 0..<N {
            let Rg = R0 * Self.rodrigues(SIMD3(X[n * P + 55], X[n * P + 56], X[n * P + 57]))
            let tr = SIMD3(X[n * P + 58], X[n * P + 59], X[n * P + 60])
            var verts = [SIMD3<Float>](repeating: .zero, count: nv)
            for i in 0..<nv {
                let o = (i * N + n) * 12
                let p = SIMD3(v[n * V3 + 3 * i], v[n * V3 + 3 * i + 1], v[n * V3 + 3 * i + 2])
                verts[i] = SIMD3(Mv[o] * p.x + Mv[o + 1] * p.y + Mv[o + 2] * p.z + Mv[o + 3],
                                 Mv[o + 4] * p.x + Mv[o + 5] * p.y + Mv[o + 6] * p.z + Mv[o + 7],
                                 Mv[o + 8] * p.x + Mv[o + 9] * p.y + Mv[o + 10] * p.z + Mv[o + 11])
            }
            var joints21 = [SIMD3<Float>](repeating: .zero, count: 21)
            for (j, src) in Self.manoToOpenPose.enumerated() {
                joints21[j] = src < nj ? T[n * nj + src] : verts[Self.tipVertices[src - nj]]
            }
            let root = joints21[0]
            for i in 0..<nv {
                let q = Rg * (verts[i] - root) + tr
                out[n * nr + 3 * i] = q.x; out[n * nr + 3 * i + 1] = q.y; out[n * nr + 3 * i + 2] = q.z
            }
            for j in 0..<21 {
                let q = Rg * (joints21[j] - root) + tr
                out[n * nr + V3 + 3 * j] = q.x; out[n * nr + V3 + 3 * j + 1] = q.y; out[n * nr + V3 + 3 * j + 2] = q.z
            }
        }
        return out
    }

    static func rodrigues(_ w: SIMD3<Float>) -> simd_float3x3 {
        let th = (simd_length_squared(w) + 1e-12).squareRoot()
        let k = w / th
        let K = simd_float3x3(rows: [SIMD3(0, -k.z, k.y), SIMD3(k.z, 0, -k.x), SIMD3(-k.y, k.x, 0)])
        return matrix_identity_float3x3 + sin(th) * K + (1 - cos(th)) * (K * K)
    }

    /// Solves H x = g for a symmetric positive definite H (n x n, row-major); nil when it is not.
    static func cholesky(_ H: [Double], _ g: [Double], _ n: Int) -> [Double]? {
        var L = [Double](repeating: 0, count: n * n)
        for i in 0..<n {
            for j in 0...i {
                var s = H[i * n + j]
                for k in 0..<j { s -= L[i * n + k] * L[j * n + k] }
                if i == j {
                    guard s > 0 else { return nil }
                    L[i * n + i] = s.squareRoot()
                } else {
                    L[i * n + j] = s / L[j * n + j]
                }
            }
        }
        var y = [Double](repeating: 0, count: n)
        for i in 0..<n {
            var s = g[i]
            for k in 0..<i { s -= L[i * n + k] * y[k] }
            y[i] = s / L[i * n + i]
        }
        var x = [Double](repeating: 0, count: n)
        for i in stride(from: n - 1, through: 0, by: -1) {
            var s = y[i]
            for k in (i + 1)..<n { s -= L[k * n + i] * x[k] }
            x[i] = s / L[i * n + i]
        }
        return x
    }

    /// bench.methods.fah._align_2d: the rigid motion (about the mesh's wrist) that makes its 21 joints project onto the
    /// 2D keypoints uv (pixels of the camera K, mesh in that camera's frame): Gauss-Newton on the reprojection error
    /// from the 3D placement. Returns the moved mesh, or the mesh unchanged when the solve fails.
    func alignTo2D(_ mesh: [SIMD3<Double>], uv: [SIMD2<Double>], fx: Double, fy: Double, cx: Double, cy: Double) -> [SIMD3<Double>] {
        // the mesh's joints: J_regressor (16) + fingertips, OpenPose order
        var joints = [SIMD3<Double>](repeating: .zero, count: 21)
        for (j, src) in Self.manoToOpenPose.enumerated() {
            if src < nj {
                var s = SIMD3<Double>.zero
                for v in 0..<nv { let w = Double(m.jointRegressor[src * nv + v]); if w != 0 { s += w * mesh[v] } }
                joints[j] = s
            } else {
                joints[j] = mesh[Self.tipVertices[src - nj]]
            }
        }
        let c = joints[0]
        let local = joints.map { $0 - c }
        var p = [Double](repeating: 0, count: 6)  // rotation vector, translation (from c)
        p[3] = c.x; p[4] = c.y; p[5] = c.z
        func project(_ p: [Double]) -> [Double]? {
            let R = Self.rodriguesD(SIMD3(p[0], p[1], p[2])), t = SIMD3(p[3], p[4], p[5])
            var r = [Double](repeating: 0, count: 42)
            for j in 0..<21 {
                let q = R * local[j] + t
                guard q.z > 1e-3 else { return nil }
                r[2 * j] = fx * q.x / q.z + cx - uv[j].x
                r[2 * j + 1] = fy * q.y / q.z + cy - uv[j].y
            }
            return r
        }
        for _ in 0..<10 {
            guard let r = project(p) else { return mesh }
            var Jac = [Double](repeating: 0, count: 42 * 6)
            for i in 0..<6 {
                var a = p, b = p
                a[i] += 1e-6; b[i] -= 1e-6
                guard let ra = project(a), let rb = project(b) else { return mesh }
                for k in 0..<42 { Jac[k * 6 + i] = (ra[k] - rb[k]) / 2e-6 }
            }
            var H = [Double](repeating: 0, count: 36), g = [Double](repeating: 0, count: 6)
            for k in 0..<42 {
                for i in 0..<6 {
                    g[i] += Jac[k * 6 + i] * r[k]
                    for j in 0..<6 { H[i * 6 + j] += Jac[k * 6 + i] * Jac[k * 6 + j] }
                }
            }
            for i in 0..<6 { H[i * 6 + i] *= 1 + 1e-6; H[i * 6 + i] += 1e-12 }
            guard let dx = Self.cholesky(H, g, 6) else { return mesh }
            for i in 0..<6 { p[i] -= dx[i] }
            if dx.map(abs).max()! < 1e-9 { break }
        }
        let R = Self.rodriguesD(SIMD3(p[0], p[1], p[2])), t = SIMD3(p[3], p[4], p[5])
        guard t.z > 0.05, t.x.isFinite, t.y.isFinite else { return mesh }
        return mesh.map { R * ($0 - c) + t }
    }

    static func rodriguesD(_ w: SIMD3<Double>) -> simd_double3x3 {
        let th = simd_length(w)
        guard th > 1e-12 else { return matrix_identity_double3x3 }
        let k = w / th
        let K = simd_double3x3(rows: [SIMD3(0, -k.z, k.y), SIMD3(k.z, 0, -k.x), SIMD3(-k.y, k.x, 0)])
        return matrix_identity_double3x3 + sin(th) * K + (1 - cos(th)) * (K * K)
    }
}
