import Foundation
import simd

/// fah_mano.bin (tools/export.py): the MANO template data the per-part Procrustes needs, FAH-S's dense point -> 778-vertex
/// surface map and the mesh faces. A flat little-endian file, so no pickle at run time.
final class ManoData {
    let nv: Int, nparts: Int, ndense: Int, nbone: Int
    let template: [SIMD3<Double>]     // (nv)
    let part: [Int]                   // (nv) skinning argmax
    let parents: [Int]                // (nparts), root -1
    let denseIndex: [Int]             // (ndense) template vertex of each dense point
    let subR: [Float]                 // (nv, ndense) row-major
    let faces: [SIMD3<UInt32>]        // right-hand winding
    let joints: [SIMD3<Double>]       // (nparts) rest joints
    let weights: [Float]              // (nv, nparts) skinning weights, row-major
    let shapedirs: [Float]            // (10, nv * 3)
    let posedirs: [Float]             // (135, nv * 3)
    let jointDirs: [Float]            // (10, nparts * 3)
    let jointRegressor: [Float]       // (nparts, nv)

    init(url: URL) throws {
        let d = try Data(contentsOf: url, options: .mappedIfSafe)
        let name = url.lastPathComponent
        guard d.count > 28, String(data: d.prefix(8), encoding: .ascii) == "FAHMANO4" else {
            throw FAHError.badResource("\(name): not a FAHMANO4 file (re-run tools/export.py)")
        }
        var o = 8
        func i32s(_ n: Int) -> [Int] {
            defer { o += 4 * n }
            return d.withUnsafeBytes { b in (0..<n).map { Int(Int32(littleEndian: b.loadUnaligned(fromByteOffset: o + 4 * $0, as: Int32.self))) } }
        }
        func f64s(_ n: Int) -> [Double] {
            defer { o += 8 * n }
            return d.withUnsafeBytes { b in (0..<n).map { Double(bitPattern: UInt64(littleEndian: b.loadUnaligned(fromByteOffset: o + 8 * $0, as: UInt64.self))) } }
        }
        func f32s(_ n: Int) -> [Float] {
            defer { o += 4 * n }
            return d.withUnsafeBytes { b in (0..<n).map { Float(bitPattern: UInt32(littleEndian: b.loadUnaligned(fromByteOffset: o + 4 * $0, as: UInt32.self))) } }
        }
        func vec3(_ a: [Double]) -> [SIMD3<Double>] { stride(from: 0, to: a.count, by: 3).map { SIMD3(a[$0], a[$0 + 1], a[$0 + 2]) } }
        let h = i32s(5)
        let (nv, nparts, ndense, nbone, nfaces) = (h[0], h[1], h[2], h[3], h[4])
        (self.nv, self.nparts, self.ndense, self.nbone) = (nv, nparts, ndense, nbone)
        let need: Int = 28 + 28 * nv + 4 * nparts + 4 * ndense + 4 * nv * ndense + 12 * nfaces + 24 * nparts + 4 * nv * nparts
            + 4 * (10 * nv * 3 + 135 * nv * 3 + 10 * nparts * 3 + nparts * nv)
        guard d.count == need else { throw FAHError.badResource("\(name): size \(d.count) != \(need)") }
        template = vec3(f64s(nv * 3))
        part = i32s(nv)
        parents = i32s(nparts)
        denseIndex = i32s(ndense)
        subR = f32s(nv * ndense)
        let f = i32s(nfaces * 3).map { UInt32($0) }
        faces = stride(from: 0, to: f.count, by: 3).map { SIMD3<UInt32>(f[$0], f[$0 + 1], f[$0 + 2]) }
        joints = vec3(f64s(nparts * 3))
        weights = f32s(nv * nparts)
        shapedirs = f32s(10 * nv * 3)
        posedirs = f32s(135 * nv * 3)
        jointDirs = f32s(10 * nparts * 3)
        jointRegressor = f32s(nparts * nv)
    }
}

/// fah.mano_procrustes.ManoProcrustes.rotations / local_pose: per MANO part the rotation that best maps its template
/// vertices (largest skinning weight) onto the predicted surface (Kabsch, here via Horn's quaternion method: the same
/// optimal proper rotation), the hand scale, and the root + 15 local rotations.
final class ManoProcrustes {
    let mano: ManoData
    private let groups: [[Int]]
    private let centred: [SIMD3<Double>]   // template centred per part
    private let big: [Bool]
    private let spread: [Double]
    private let templateMean: [SIMD3<Double>]  // per part

    init(_ m: ManoData) {
        mano = m
        var g = [[Int]](repeating: [], count: m.nparts)
        for (v, k) in m.part.enumerated() { g[k].append(v) }
        groups = g
        var c = [SIMD3<Double>](repeating: .zero, count: m.nv)
        var sp = [Double](repeating: 0, count: m.nparts)
        for k in 0..<m.nparts {
            var mean = SIMD3<Double>.zero
            for v in g[k] { mean += m.template[v] }
            mean /= Double(max(g[k].count, 1))
            for v in g[k] { c[v] = m.template[v] - mean; sp[k] += simd_length_squared(c[v]) }
        }
        centred = c
        spread = sp
        templateMean = (0..<m.nparts).map { k in g[k].reduce(SIMD3<Double>.zero) { $0 + m.template[$1] } / Double(max(g[k].count, 1)) }
        big = g.map { $0.count >= 20 }
    }

    /// P: (nv) surface in the right-hand frame. Returns per-part global rotations and the scale.
    func rotations(_ P: [SIMD3<Double>]) -> ([simd_double3x3], Double) {
        var R = [simd_double3x3](repeating: matrix_identity_double3x3, count: mano.nparts)
        var ratios: [Double] = []
        for k in 0..<mano.nparts {
            let g = groups[k]
            var mean = SIMD3<Double>.zero
            for v in g { mean += P[v] }
            mean /= Double(max(g.count, 1))
            // S[i][j] = sum_v a_i b_j, a = template (centred), b = prediction (centred)
            var S = simd_double3x3()
            var ss = 0.0
            for v in g {
                let a = centred[v], b = P[v] - mean
                ss += simd_length_squared(b)
                S.columns.0 += a.x * b
                S.columns.1 += a.y * b
                S.columns.2 += a.z * b
            }
            // S as stored: columns.i = sum a_i b  ->  S[col i][row j] = sum a_i b_j
            R[k] = ManoProcrustes.horn(S)
            if big[k] { ratios.append((ss / spread[k]).squareRoot()) }
        }
        ratios.sort()
        let n = ratios.count
        let s = n == 0 ? 1 : (n % 2 == 1 ? ratios[n / 2] : 0.5 * (ratios[n / 2 - 1] + ratios[n / 2]))
        return (R, s)
    }

    /// The MANO template posed by the per-part rotations and scale of `rotations(P)` (fah.mano_procrustes
    /// ManoProcrustes.__call__): the wrist where the root part maps the template's, every joint from its parent along the
    /// parent's rotated rest bone, then linear blend skinning. A clean, watertight hand following the predicted surface
    /// (the raw dense surface can fold and spike), right-hand frame like P.
    func posedMesh(_ P: [SIMD3<Double>], _ R: [simd_double3x3], _ s: Double) -> [SIMD3<Double>] {
        let m = mano, J = m.joints
        let g0 = groups[0]
        let mean0 = g0.reduce(SIMD3<Double>.zero) { $0 + P[$1] } / Double(max(g0.count, 1))
        var Jp = [SIMD3<Double>](repeating: .zero, count: m.nparts)
        Jp[0] = mean0 + s * (R[0] * (J[0] - templateMean[0]))
        for k in 1..<m.nparts { Jp[k] = Jp[m.parents[k]] + s * (R[m.parents[k]] * (J[k] - J[m.parents[k]])) }
        let Rs = R.map { $0 * s }
        return (0..<m.nv).map { v in
            var out = SIMD3<Double>.zero
            for k in 0..<m.nparts {
                let w = Double(m.weights[v * m.nparts + k])
                if w > 1e-6 { out += w * (Rs[k] * (m.template[v] - J[k]) + Jp[k]) }
            }
            return out
        }
    }

    /// MANO joint -> FAH keypoint (OpenPose order), and for the finger ends the fingertip keypoint and template vertex.
    private static let keypointOf = [0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3]
    private static let tipOf: [Int: (keypoint: Int, vertex: Int)] = [3: (8, 320), 6: (12, 443), 9: (20, 671), 12: (16, 554),
                                                                      15: (4, 744)]

    /// The MANO template fitted to the 21 predicted keypoints (right-hand frame, like P): every joint at its keypoint,
    /// every finger bone turned (the smallest rotation on top of its Procrustes rotation, which keeps the twist) and
    /// stretched along its axis to reach the next keypoint; thickness and the palm follow the Procrustes rotation and
    /// scale. Then linear blend skinning. The keypoints are the network's most accurate output, so the mesh lands on them.
    func keypointMesh(_ K: [SIMD3<Double>], _ R: [simd_double3x3], _ s: Double) -> [SIMD3<Double>] {
        let m = mano, J = m.joints
        var child = [Int?](repeating: nil, count: m.nparts)
        for k in 1..<m.nparts where child[m.parents[k]] == nil { child[m.parents[k]] = k }
        var Rf = R, axis = [SIMD3<Double>](repeating: .zero, count: m.nparts), stretch = [Double](repeating: 1, count: m.nparts)
        let Jp = (0..<m.nparts).map { K[Self.keypointOf[$0]] }
        for k in 1..<m.nparts {
            let (tEnd, pEnd): (SIMD3<Double>, SIMD3<Double>)
            if let c = child[k] { (tEnd, pEnd) = (J[c], Jp[c]) } else if let t = Self.tipOf[k] { (tEnd, pEnd) = (m.template[t.vertex], K[t.keypoint]) } else { continue }
            let bt = tEnd - J[k], bp = pEnd - Jp[k]
            let lt = simd_length(bt), lp = simd_length(bp)
            guard lt > 1e-6, lp > 1e-6 else { continue }
            axis[k] = bt / lt
            let d = simd_normalize(R[k] * axis[k])
            Rf[k] = simd_double3x3(simd_quatd(from: d, to: bp / lp)) * R[k]
            stretch[k] = min(max(lp / lt, 0.3 * s), 3 * s)  // implausible lengths (a bad keypoint) clamped
        }
        return (0..<m.nv).map { v in
            var out = SIMD3<Double>.zero
            for k in 0..<m.nparts {
                let w = Double(m.weights[v * m.nparts + k])
                guard w > 1e-6 else { continue }
                let local = m.template[v] - J[k]
                let a = simd_dot(local, axis[k])  // along the bone (zero axis: the palm, scaled uniformly)
                out += w * (Rf[k] * (s * (local - a * axis[k]) + stretch[k] * a * axis[k]) + Jp[k])
            }
            return out
        }
    }

    /// Optimal proper rotation R maximising sum_v b_v . (R a_v), from Sab[col i][row j] = sum a_i b_j (Horn 1987).
    static func horn(_ S: simd_double3x3) -> simd_double3x3 {
        func s(_ i: Int, _ j: Int) -> Double { S[i][j] }  // sum a_i b_j
        let sxx = s(0, 0), sxy = s(0, 1), sxz = s(0, 2)
        let syx = s(1, 0), syy = s(1, 1), syz = s(1, 2)
        let szx = s(2, 0), szy = s(2, 1), szz = s(2, 2)
        var N: [[Double]] = [
            [sxx + syy + szz, syz - szy, szx - sxz, sxy - syx],
            [syz - szy, sxx - syy - szz, sxy + syx, szx + sxz],
            [szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy],
            [sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz],
        ]
        let (vals, vecs) = jacobiEigen4(&N)
        var best = 0
        for i in 1..<4 where vals[i] > vals[best] { best = i }
        let q = simd_quatd(ix: vecs[1][best], iy: vecs[2][best], iz: vecs[3][best], r: vecs[0][best]).normalized
        return simd_double3x3(q)
    }

    /// Symmetric 4x4 eigen decomposition (cyclic Jacobi). Returns eigenvalues and eigenvectors as columns V[row][col].
    static func jacobiEigen4(_ A: inout [[Double]]) -> ([Double], [[Double]]) {
        var V: [[Double]] = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        for _ in 0..<50 {
            var off = 0.0
            for p in 0..<4 { for q in (p + 1)..<4 { off += A[p][q] * A[p][q] } }
            var scale = 0.0
            for p in 0..<4 { scale += A[p][p] * A[p][p] }
            if off <= 1e-30 * max(scale, 1e-300) { break }
            for p in 0..<4 {
                for q in (p + 1)..<4 where A[p][q] != 0 {
                    let theta = (A[q][q] - A[p][p]) / (2 * A[p][q])
                    let t = (theta >= 0 ? 1.0 : -1.0) / (abs(theta) + (theta * theta + 1).squareRoot())
                    let c = 1 / (t * t + 1).squareRoot(), sn = t * c
                    for k in 0..<4 {  // A <- J^T A J
                        let akp = A[k][p], akq = A[k][q]
                        A[k][p] = c * akp - sn * akq
                        A[k][q] = sn * akp + c * akq
                    }
                    for k in 0..<4 {
                        let apk = A[p][k], aqk = A[q][k]
                        A[p][k] = c * apk - sn * aqk
                        A[q][k] = sn * apk + c * aqk
                    }
                    for k in 0..<4 {
                        let vkp = V[k][p], vkq = V[k][q]
                        V[k][p] = c * vkp - sn * vkq
                        V[k][q] = sn * vkp + c * vkq
                    }
                }
            }
        }
        return ([A[0][0], A[1][1], A[2][2], A[3][3]], V)
    }

    /// Global part rotations -> root rotation and the 15 local rotations (parent^T child) with their axis-angles
    /// (fah.mano_procrustes local_pose: log map, angles < pi).
    func localPose(_ R: [simd_double3x3]) -> (simd_double3x3, [simd_double3x3], [SIMD3<Double>]) {
        var L: [simd_double3x3] = [], W: [SIMD3<Double>] = []
        for k in 1..<mano.nparts {
            let l = R[mano.parents[k]].transpose * R[k]
            L.append(l)
            W.append(ManoProcrustes.logMap(l))
        }
        return (R[0], L, W)
    }

    static func logMap(_ l: simd_double3x3) -> SIMD3<Double> {
        // l[col][row]
        let tr = l[0][0] + l[1][1] + l[2][2]
        let ang = acos(min(max((tr - 1) / 2, -1), 1))
        let w = SIMD3(l[1][2] - l[2][1], l[2][0] - l[0][2], l[0][1] - l[1][0])  // (L21 - L12, L02 - L20, L10 - L01)
        let sn = sin(ang)
        let f = sn > 1e-6 ? ang / (2 * max(sn, 1e-12)) : 0.5
        return w * f
    }
}
