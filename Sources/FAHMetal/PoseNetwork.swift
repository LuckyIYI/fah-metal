import Foundation
import Metal

/// What the pipeline needs from a pose network: an input image buffer the crop kernel writes into, the per-crop camera
/// inputs, GPU work it can append to its own command buffer, and the outputs once that has completed.
protocol PoseNetwork: AnyObject {
    var maxBatch: Int { get }
    /// (maxBatch, 3, S, S) fp32 RGB in [0, 1], shared storage.
    var imageBuffer: MTLBuffer { get }
    /// Per crop: ray = [sx, ox, sy, oy] (the crop's pixel -> normalised camera ray map), cond = the CLIFF condition.
    func setCamera(rays: [[Double]], conds: [[Double]])
    func encode(into cb: MTLCommandBuffer, batch: Int) throws
    func outputs(batch: Int) -> FAHPoseOutputs
    /// The contact head's tolerances (mm; empty without it) and the index of the 10 mm one.
    var contactDistances: [Float] { get }
    var contactMain: Int { get }
    /// Run the contact head on the next inferences (only when the network has it).
    var contactActive: Bool { get set }
}

/// Which matrix-multiply hardware path the pose network's kernels use.
public enum FAHPoseBackend: Sendable {
    /// Tensor ops where they are fast (Apple GPU family 10, e.g. M5 / A19, on macOS / iOS 26.2+), else simdgroup matrices.
    case automatic
    /// Metal 4 tensor ops (MetalPerformancePrimitives matmul2d), which run on the GPU's neural accelerators.
    case tensorOps
    /// simdgroup_matrix 8x8 multiply-accumulate (any Apple GPU family 7+, e.g. A17 Pro / M1-M4).
    case simdgroupMatrix
}

/// Raw outputs of the pose network for a batch of crops (row-major, float32).
public struct FAHPoseOutputs: Sendable {
    public var uv: [Float]       // (B, 461, 2) crop coordinates in [0, 1]
    public var xyz: [Float]      // (B, 461, 3) root-relative camera-frame points (m)
    public var conf: [Float]     // (B, 461) heatmap peak probability
    public var vis: [Float]      // (B, 461) visibility logit
    public var depth: [Float]    // (B) log root depth (in crop-normalised units)
    public var token: [Float]    // (B, 1024) pooled pose token
    public var verify: [Float]   // (B, 2) presence and side logits
    /// Hand-object contact logits, empty without the contact head: per dense surface point (B, 400, K) and per hand (B, K),
    /// at the K tolerances of `FAHPoseNetwork.contactDistances`. The calibration offset is folded in: logit >= 0 is contact.
    public var contact: [Float]
    public var contactHand: [Float]
}

// Parameter blocks of the kernels; the layouts mirror the structs in pose.metal.
private struct GP { var M: Int32 = 0, K: Int32 = 0, lda: Int32 = 0, ldb: Int32 = 0, ldo: Int32 = 0, N: Int32 = 0
    var eps: Float = 0, eps2: Float = 0
    var a0: Int32 = 0, a1: Int32 = 0, a2: Int32 = 0, a3: Int32 = 0 }
private struct AP { var NQ: Int32 = 0, NK: Int32 = 0, ldq: Int32 = 0, ldk: Int32 = 0, ldv: Int32 = 0, ldo: Int32 = 0
    var qbs: Int32 = 0, kbs: Int32 = 0, vbs: Int32 = 0, heads: Int32 = 0, kvPerSplit: Int32 = 0, NQP: Int32 = 0
    var scale: Float = 0 }
private struct PoseP { var B: Int32, KIN: Int32, KP: Int32, NOUT: Int32; var eps: Float, invTokens: Float }
private struct PQP { var NP: Int32, NPP: Int32; var eps: Float }
private struct PHP { var NP: Int32, NPP: Int32, NS: Int32; var eps: Float, invW: Float }
private struct CP { var B: Int32 = 0, rows: Int32 = 0, a0: Int32 = 0, a1: Int32 = 0; var eps: Float = 0, eps2: Float = 0 }
private struct A32 { var NQ: Int32, NK: Int32, ldq: Int32, ldk: Int32, ldv: Int32, ldo: Int32; var scale: Float }

/// FAH-S's pose network (DINOv3 ViT-S/16 student + keypoint, depth and verification heads) in plain Metal compute.
///
/// Inputs and outputs are those of FAH's Core ML pose export: img (B, 3, 320, 320) RGB in [0, 1], ray (B, 4) and
/// cond (B, 3) -> pt_uv, pt_xyz, pt_conf, pt_vis, pt_depth, pt_token, verify_logits, plus (when exported) FAH's
/// hand-object contact head. The whole network is one compute encoder of ~80 dispatches (~140 with contact), recorded
/// once per batch size; see Shaders/pose.metal for the kernels.
///
/// Not thread-safe: the intermediate buffers are persistent, so run one inference at a time per instance.
public final class FAHPoseNetwork: PoseNetwork {
    public static let weightsName = "fah-s-pose"
    public static let inputSize = 320
    public static let points = 461

    public let device: MTLDevice
    public let maxBatch: Int
    /// True when the kernels use Metal 4 tensor ops, false for simdgroup matrices.
    public let usesTensorOps: Bool
    /// True when the hand-object contact head runs (it is in the exported weights and was not switched off).
    public let usesContact: Bool
    /// The contact head's tolerances (mm) for its K outputs per point and hand (FAH-S: 5, 10, 20, 35), and the index of
    /// the 10 mm one FAH reports as "contact".
    public let contactDistances: [Float]
    public let contactMain: Int
    /// Switch the contact head on / off between inferences without rebuilding anything (its dispatches are skipped).
    public var contactActive: Bool

    private let queue: MTLCommandQueue
    private let config: [String: Any]
    private let eps: [String: Any]
    private let weights: MTLBuffer
    private var offsets: [String: Int] = [:]
    private var library: MTLLibrary!
    private var pipelines: [String: MTLComputePipelineState] = [:]
    private var bufs: [String: MTLBuffer] = [:]
    private var schedules: [Int: [Step]] = [:]

    private let NS = 10    // heatmap key/value splits (6400 / NS keys each, a multiple of 64)
    private let NPP = 480  // point-query rows, padded
    private let NKP = 448  // ViT tokens per head, padded

    /// One encoded dispatch: pipeline, buffer bindings, inline parameters, grid.
    private struct Step {
        let name: String
        let pso: MTLComputePipelineState
        var bufs: [(Int, MTLBuffer, Int)]
        var bytes: (Int, [UInt8])
        let groups: MTLSize
        let threads: MTLSize
        var concurrent = false  // may overlap the previous step (no barrier in between)
    }

    /// - Parameters:
    ///   - modelDirectory: holds `fah-s-pose.bin` / `.json` (tools/export.py).
    ///   - maxBatch: the largest number of crops per call (buffers are allocated for it).
    ///   - contact: run the hand-object contact head when the export has it (about +3 ms per crop on an M5).
    public init(modelDirectory: URL, maxBatch: Int = 4, backend: FAHPoseBackend = .automatic, contact: Bool = true,
                device: MTLDevice? = nil) throws {
        guard let dev = device ?? MTLCreateSystemDefaultDevice(), let q = dev.makeCommandQueue() else { throw FAHError.metal("no Metal device") }
        self.device = dev
        self.queue = q
        self.maxBatch = maxBatch
        usesTensorOps = try Self.resolve(backend, dev)
        let json = modelDirectory.appendingPathComponent(Self.weightsName + ".json")
        let bin = modelDirectory.appendingPathComponent(Self.weightsName + ".bin")
        for u in [json, bin] where !FileManager.default.fileExists(atPath: u.path) { throw FAHError.missingResource(u.path) }
        guard let meta = try JSONSerialization.jsonObject(with: Data(contentsOf: json)) as? [String: Any],
              let cfg = meta["config"] as? [String: Any], let e = cfg["eps"] as? [String: Any],
              let tensors = meta["tensors"] as? [String: [String: Any]] else { throw FAHError.badResource(json.path) }
        config = cfg
        eps = e
        let cc = cfg["contact"] as? [String: Any]
        usesContact = contact && cc != nil
        contactDistances = (cc?["dists_mm"] as? [Double] ?? []).map(Float.init)
        contactMain = cc?["main"] as? Int ?? 0
        contactActive = usesContact
        for (k, v) in tensors { offsets[k] = v["offset"] as? Int }
        let data = try Data(contentsOf: bin, options: .alwaysMapped)
        guard let wb = data.withUnsafeBytes({ dev.makeBuffer(bytes: $0.baseAddress!, length: data.count, options: .storageModeShared) }) else {
            throw FAHError.metal("weights buffer")
        }
        weights = wb
        try compile()
        try allocate()
        for b in 1...maxBatch { schedules[b] = try build(batch: b) }
    }

    private static func resolve(_ backend: FAHPoseBackend, _ dev: MTLDevice) throws -> Bool {
        var available = false
        if #available(macOS 26.2, iOS 26.2, *) { available = dev.supportsFamily(.apple10) }
        switch backend {
        case .automatic: return available
        case .simdgroupMatrix: return false
        case .tensorOps:
            guard #available(macOS 26.0, iOS 26.0, *), dev.supportsFamily(.metal4) else {
                throw FAHError.metal("tensor ops need Metal 4 (macOS / iOS 26)")
            }
            return true
        }
    }

    // MARK: setup

    private func compile() throws {
        let opts = MTLCompileOptions()
        opts.preprocessorMacros = ["FAH_NAX": NSNumber(value: usesTensorOps ? 1 : 0)]
        opts.mathMode = .relaxed
        if usesTensorOps, #available(macOS 26.0, iOS 26.0, *) { opts.languageVersion = .version4_0 } else { opts.languageVersion = .version3_2 }
        do { library = try device.makeLibrary(source: try shaderSource("pose"), options: opts) } catch {
            throw FAHError.metal("pose.metal: \(error)")
        }
    }

    private func pso(_ name: String) throws -> MTLComputePipelineState {
        if let p = pipelines[name] { return p }
        guard let f = library.makeFunction(name: name) else { throw FAHError.metal("no kernel \(name)") }
        let p = try device.makeComputePipelineState(function: f)
        pipelines[name] = p
        return p
    }

    private func allocate() throws {
        let B = maxBatch, h = 2, f = 4, sl = 64  // h / f: fp16 / fp32 bytes; sl: slack rows (tiles may read past the last row)
        let P = Self.points
        var sizes: [(String, Int)] = [
            ("img", B * 3 * 320 * 320 * f), ("ray", B * 4 * f), ("cond", B * 3 * f),
            ("patches", (B * 400 + sl) * 768 * h), ("hh", (B * 400 + sl) * 1408 * h),
            ("X", (B * 405 + sl) * 384 * f), ("Y", (B * 405 + sl) * 384 * h),
            ("q", B * 6 * NKP * 128 * h), ("k", B * 6 * NKP * 128 * h), ("v", B * 6 * NKP * 64 * h),
            ("o", (B * 405 + sl) * 384 * h), ("H", (B * 405 + sl) * 1536 * h), ("ya", (B * 400 + sl) * 384 * h),
            ("scratch", (B * 400 + sl) * 1280 * f), ("tok", (B * 400 + sl) * 1280 * h), ("colsum", B * 1280 * f),
            ("pose", B * 1024 * f), ("ph", B * 1280 * f),
            ("f0", (B * 400 + sl) * 256 * h), ("f1", (B * 1600 + sl) * 128 * h), ("s0", B * 162 * 162 * 32 * h),
            ("F", B * 82 * 82 * 96 * h), ("g0", B * 82 * 82 * 64 * h), ("ff", B * 6400 * 96 * h), ("qp", B * NPP * 64 * h),
            ("OP", B * NS * NPP * 96 * f), ("ML", B * NS * NPP * 2 * f),
            ("uv", B * P * 2 * f), ("xyz", B * P * 3 * f), ("conf", B * P * f), ("vis", B * P * f), ("depth", B * f),
            ("verify", B * 2 * f), ("geo", B * P * 132 * f),
        ]
        if usesContact {
            let K = contactDistances.count, layers = contactConfig["layers"] as! Int
            sizes += [
                ("Xb", (B * 405 + sl) * 384 * f), ("Yb", (B * 405 + sl) * 384 * h), ("ybr", (B * 400 + sl) * 384 * h),
                ("bm", (B * 400 + sl) * 256 * h), ("mem", (B * 800 + sl) * 256 * h), ("kv", (B * 800 + sl) * 512 * layers * h),
                ("ctok", B * 256 * f), ("cx", (B * 400 + sl) * 256 * f), ("cy", (B * 400 + sl) * 256 * h),
                ("dqkv", (B * 400 + sl) * 768 * h), ("dq", (B * 400 + sl) * 256 * h), ("dao", (B * 400 + sl) * 256 * h),
                ("dh", (B * 400 + sl) * 512 * h), ("contact", B * 400 * K * f), ("contact_hand", B * K * f),
            ]
        }
        for (name, bytes) in sizes {
            guard let b = device.makeBuffer(length: bytes, options: .storageModeShared) else { throw FAHError.metal("buffer \(name)") }
            b.label = name
            memset(b.contents(), 0, b.length)  // zero borders and padding rows are never written
            bufs[name] = b
        }
        // constant columns of the heatmap value matrix: [features (64) | x + 0.5 | y + 0.5 | 1 | 0 ...]
        let ff = bufs["ff"]!.contents().bindMemory(to: Float16.self, capacity: B * 6400 * 96)
        for b in 0..<B { for i in 0..<80 { for j in 0..<80 {
            let r = ((b * 80 + i) * 80 + j) * 96
            ff[r + 64] = Float16(Float(j) + 0.5); ff[r + 65] = Float16(Float(i) + 0.5); ff[r + 66] = 1
        } } }
    }

    private func w(_ name: String) throws -> (MTLBuffer, Int) {
        guard let o = offsets[name] else { throw FAHError.badResource("\(Self.weightsName).json: no tensor \(name)") }
        return (weights, o)
    }

    private var contactConfig: [String: Any] { config["contact"] as? [String: Any] ?? [:] }

    private func e(_ k: String, _ i: Int? = nil) -> Float {
        if let i { return Float((eps[k] as! [Double])[i]) }
        return Float(eps[k] as! Double)
    }

    // MARK: graph

    /// The dispatches of one inference of `B` crops, in schedule order.
    private func build(batch B: Int) throws -> [Step] {
        var out: [Step] = []
        func b(_ n: String) -> (MTLBuffer, Int) { (bufs[n]!, 0) }
        func add<T>(_ name: String, _ kernel: String, _ bindings: [(MTLBuffer, Int)], _ params: T, _ groups: MTLSize,
                    _ threads: Int) throws {
            var p = params
            let bytes = withUnsafeBytes(of: &p) { Array($0) }
            out.append(Step(name: name, pso: try pso(kernel), bufs: bindings.enumerated().map { ($0, $1.0, $1.1) },
                            bytes: (bindings.count, bytes), groups: groups, threads: MTLSize(width: threads, height: 1, depth: 1)))
        }
        let M = B * 405, MT = B * 400  // ViT rows (prefix + patch tokens), patch-token rows
        let blocks = config["blocks"] as! Int
        let unused = b("colsum")  // bound to slots a kernel variant does not read

        // input prep (im2col, ray embedding, prefix rows) + patch embedding (+ block 0's LN1)
        try add("prep", "prep", [b("img"), b("ray"), w("mean"), w("std"), w("uv_ds"), w("freq"), w("prefix_x"), w("prefix_y"),
                                 b("patches"), b("hh"), b("X"), b("Y"), b("colsum")], Int32(B),
                MTLSize(width: (B * 102400 + 255) / 256, height: 1, depth: 1), 256)
        // GEMM + row LayerNorm epilogue (the threadgroup owns whole rows)
        func rowln(_ name: String, tsn: Int, wn: Int, mode: Int, A: (MTLBuffer, Int), lda: Int, Bw: (MTLBuffer, Int), N: Int,
                   K: Int, rows: Int, bias: (MTLBuffer, Int), gam: (MTLBuffer, Int)? = nil, X: (MTLBuffer, Int)? = nil,
                   Y: (MTLBuffer, Int), ldo: Int, lnw: (MTLBuffer, Int), lnb: (MTLBuffer, Int), lnw2: (MTLBuffer, Int)? = nil,
                   lnb2: (MTLBuffer, Int)? = nil, eps: Float, eps2: Float = 0, a0: Int = 0, ldx: Int = 0) throws {
            let p = GP(M: Int32(rows), K: Int32(K), lda: Int32(lda), ldb: Int32(N), ldo: Int32(ldo), N: Int32(N),
                       eps: eps, eps2: eps2, a0: Int32(a0), a1: Int32(ldx))
            try add(name, "mm_rowln_\(tsn)_\(wn)_\(mode)", [A, Bw, bias, gam ?? bias, X ?? unused, Y, lnw, lnb, lnw2 ?? lnw, lnb2 ?? lnb], p,
                    MTLSize(width: (rows + 15) / 16, height: N / (tsn * wn), depth: 1), wn * 32)
        }
        // GEMM + bias (act 0 none, 1 GELU, 2 ReLU)
        func mact(_ name: String, act: Int, outMode: Int = 0, A: (MTLBuffer, Int), lda: Int, Bw: (MTLBuffer, Int), N: Int,
                  K: Int, rows: Int, bias: (MTLBuffer, Int), out: (MTLBuffer, Int), ldo: Int, a0: Int = 0, a1: Int = 0) throws {
            let (tsm, tsn, wm, wn) = (32, 32, 1, 2)
            let p = GP(M: Int32(rows), K: Int32(K), lda: Int32(lda), ldb: Int32(N), ldo: Int32(ldo), N: Int32(N),
                       a0: Int32(a0), a1: Int32(a1))
            try add(name, "mm_act_\(tsm)_\(tsn)_\(wm)_\(wn)_\(act)_\(outMode)", [A, Bw, bias, out], p,
                    MTLSize(width: (rows + tsm * wm - 1) / (tsm * wm), height: N / (tsn * wn), depth: 1), wm * wn * 32)
        }
        let (rtsn, rwn) = (64, 6)  // ViT row-LayerNorm GEMMs: 16 rows x 384 columns per threadgroup (6 simdgroups x 64)
        try rowln("embed", tsn: rtsn, wn: rwn, mode: 2, A: b("patches"), lda: 768, Bw: w("patch_w"), N: 384, K: 768, rows: MT,
                  bias: w("patch_b"), X: b("X"), Y: b("Y"), ldo: 384, lnw: w("b0.n1w"), lnb: w("b0.n1b"), eps: e("ln1", 0))

        // ViT blocks on residual X (fp32) / LayerNormed Y (fp16): QKV (+ RoPE) -> attention -> projection (+ residual + LN2)
        // -> fc1 -> fc2 (+ residual + next LN1). Weights "<pre><i>.*", LayerNorm eps of encoder block e0 + i; the last
        // block's fc2 adds the final norm and a second LayerNorm and writes the patch tokens to `final`.
        var ap = AP()
        ap.NQ = 405; ap.NK = 405; ap.ldq = 128; ap.ldk = Int32(NKP); ap.ldv = 64; ap.ldo = 384
        ap.qbs = Int32(NKP * 128); ap.kbs = Int32(NKP * 128); ap.vbs = Int32(NKP * 64); ap.heads = 6; ap.scale = 1.4426950408889634
        let branchBlocks = usesContact ? contactConfig["branch_blocks"] as! Int : 0
        func vit(_ pre: String, _ n: Int, e0: Int, X: String, Y: String, final: String, norm: (String, String),
                 norm2: (String, String), eps2: Float, tag: String = "") throws {
            for i in 0..<n {
                let qp = GP(M: Int32(M), K: 384, lda: 384, ldb: 1152, ldo: 0, N: 1152, a0: 405, a1: 5, a2: Int32(NKP))
                try add(tag + "qkv", "mm_qkv_16_2_2", [b(Y), w("\(pre)\(i).qkv"), w("rope_cos"), w("rope_sin"), b("q"), b("k"), b("v")],
                        qp, MTLSize(width: (M + 31) / 32, height: 1152 / 128, depth: 1), 128)
                // 2 simdgroups x 16 query rows per threadgroup; tensor ops stage S / P / O in threadgroup memory, the
                // simdgroup-matrix variant keeps them in registers
                try add(tag + "attn", usesTensorOps ? "attn_vit" : "attn_sg_vit", [b("q"), b("k"), b("v"), b("o"), unused, unused], ap,
                        MTLSize(width: (405 + 31) / 32, height: 6, depth: B), 64)
                try rowln(tag + "proj", tsn: rtsn, wn: rwn, mode: 0, A: b("o"), lda: 384, Bw: w("\(pre)\(i).proj"), N: 384, K: 384,
                          rows: M, bias: w("\(pre)\(i).projb"), gam: w("\(pre)\(i).g1"), X: b(X), Y: b(Y), ldo: 384,
                          lnw: w("\(pre)\(i).n2w"), lnb: w("\(pre)\(i).n2b"), eps: e("ln2", e0 + i))
                try mact(tag + "fc1", act: 1, A: b(Y), lda: 384, Bw: w("\(pre)\(i).fc1"), N: 1536, K: 384, rows: M,
                         bias: w("\(pre)\(i).fc1b"), out: b("H"), ldo: 1536)
                if i + 1 < n {
                    try rowln(tag + "fc2", tsn: rtsn, wn: rwn, mode: 0, A: b("H"), lda: 1536, Bw: w("\(pre)\(i).fc2"), N: 384,
                              K: 1536, rows: M, bias: w("\(pre)\(i).fc2b"), gam: w("\(pre)\(i).g2"), X: b(X), Y: b(Y), ldo: 384,
                              lnw: w("\(pre)\(i + 1).n1w"), lnb: w("\(pre)\(i + 1).n1b"), eps: e("ln1", e0 + i + 1))
                } else {
                    try rowln(tag + "fc2", tsn: rtsn, wn: rwn, mode: 1, A: b("H"), lda: 1536, Bw: w("\(pre)\(i).fc2"), N: 384,
                              K: 1536, rows: M, bias: w("\(pre)\(i).fc2b"), gam: w("\(pre)\(i).g2"), X: b(X), Y: b(final),
                              ldo: 384, lnw: w(norm.0), lnb: w(norm.1), lnw2: w(norm2.0), lnb2: w(norm2.1), eps: e("norm"), eps2: eps2)
                }
                // the contact branch starts from the encoder's hidden state before its last `branchBlocks` blocks
                if pre == "b" && branchBlocks > 0 && i + 1 == n - branchBlocks {
                    try add("c_branch_in", "ln_copy", [b(X), w("cb0.n1w"), w("cb0.n1b"), b("Xb"), b("Yb")],
                            CP(rows: Int32(M), eps: e("ln1", i + 1)), MTLSize(width: (M + 7) / 8, height: 1, depth: 1), 256)
                }
            }
        }
        // the encoder; its final norm and the adapter's LayerNorm on the patch tokens
        try vit("b", blocks, e0: 0, X: "X", Y: "Y", final: "ya", norm: ("norm_w", "norm_b"), norm2: ("ad_lnw", "ad_lnb"),
                eps2: e("ad"))
        // adapter (GELU) -> hh[:, :1280]; the folded projection + ray embedding (fp32 out) -> LayerNorm -> tokens, and
        // their per-image column sums for the pooled pose token
        try mact("adapter", act: 1, A: b("ya"), lda: 384, Bw: w("ad_w"), N: 1280, K: 384, rows: MT, bias: w("ad_b"),
                 out: b("hh"), ldo: 1408)
        try mact("tokens", act: 0, outMode: 2, A: b("hh"), lda: 1408, Bw: w("w_cat"), N: 1280, K: 1408, rows: MT,
                 bias: w("b_fold"), out: b("scratch"), ldo: 1280)
        try add("ln_tok", "ln_tok", [b("scratch"), w("ce_w"), w("ce_b"), b("tok"), b("colsum")], GP(eps: e("ce")),
                MTLSize(width: 13, height: B, depth: 1), 256)
        // pose token and the first layers of its MLPs; the point queries (+ depth and verification outputs)
        try add("pose", "pose_gemv", [b("colsum"), b("cond"), w("pool_lnw"), w("pool_lnb"), w("pool_wt"), w("pool_b"), b("pose")],
                PoseP(B: Int32(B), KIN: 1283, KP: 1312, NOUT: 1024, eps: e("pool"), invTokens: 1.0 / 400),
                MTLSize(width: 1024 / 8, height: 1, depth: 1), 256)
        try add("pose_heads", "pose_heads", [b("pose"), w("depth_lnw"), w("depth_lnb"), w("ph_wt"), w("ph_b"), b("ph")],
                PoseP(B: Int32(B), KIN: 1024, KP: 1024, NOUT: 1280, eps: e("depth"), invTokens: 0),
                MTLSize(width: 1280 / 8, height: 1, depth: 1), 256)
        try add("point_query", "point_query", [b("ph"), w("q1_wt"), w("q1_b"), w("point"), w("qmlp_lnw"), w("qmlp_lnb"),
                                                w("qmlp_m0"), w("qmlp_m0b"), w("qmlp_m1"), w("qmlp_m1b"), w("d1_w"), w("d1_b"),
                                                w("a1_wt"), w("a1_b"), b("qp"), b("depth"), b("verify")],
                PQP(NP: Int32(Self.points), NPP: Int32(NPP), eps: e("qmlp")),
                MTLSize(width: (Self.points + 31) / 32, height: B, depth: 1), 256)
        // keypoint head: image projection, two transposed convs, RGB stem, fuse convs (zero-bordered NHWC maps)
        try rowln("img_proj", tsn: 64, wn: 4, mode: 3, A: b("tok"), lda: 1280, Bw: w("ip_w"), N: 256, K: 1280, rows: MT,
                  bias: w("ip_b"), Y: b("f0"), ldo: 256, lnw: w("ip_lnw"), lnb: w("ip_lnb"), eps: e("ip"))
        try rowln("up0", tsn: 64, wn: 2, mode: 4, A: b("f0"), lda: 256, Bw: w("up0_w"), N: 512, K: 256, rows: MT,
                  bias: w("up0_b"), Y: b("f1"), ldo: 128, lnw: w("up0_lnw"), lnb: w("up0_lnb"), eps: e("up0"), a0: 20)
        try mact("up1", act: 1, outMode: 1, A: b("f1"), lda: 128, Bw: w("up1_w"), N: 256, K: 128, rows: B * 1600,
                 bias: w("up1_b"), out: b("F"), ldo: 96, a0: 40, a1: 64)
        try add("stem0", "stem0", [b("img"), w("mean"), w("std"), w("stem0_w"), w("stem0_b"), b("s0")], Int32(B),
                MTLSize(width: (B * 25600 + 255) / 256, height: 1, depth: 1), 256)
        func conv(_ name: String, _ kernel: String, X: (MTLBuffer, Int), Bw: (MTLBuffer, Int), bias: (MTLBuffer, Int),
                  lnw: (MTLBuffer, Int)? = nil, lnb: (MTLBuffer, Int)? = nil, W2: (MTLBuffer, Int)? = nil,
                  b2: (MTLBuffer, Int)? = nil, out: (MTLBuffer, Int), ldo: Int, hin: Int, choff: Int = 0, eps: Float = 0) throws {
            let wm = 4, tiles = B * 80 * 5  // 16-pixel row segments of the 80 x 80 output
            let p = GP(M: Int32(B * 6400), ldo: Int32(ldo), eps: eps, a0: Int32(hin), a1: Int32(B), a2: 80, a3: Int32(choff))
            try add(name, kernel, [X, Bw, bias, lnw ?? bias, lnb ?? bias, W2 ?? Bw, b2 ?? bias, out], p,
                    MTLSize(width: (tiles + wm - 1) / wm, height: 1, depth: 1), wm * 32)
        }
        try conv("stem1", "conv3_32_32_2_4_0", X: b("s0"), Bw: w("stem1_w"), bias: w("stem1_b"), out: b("F"), ldo: 96,
                 hin: 160, choff: 64)
        try conv("fuse0", "conv3_64_96_1_4_1", X: b("F"), Bw: w("fuse0_w"), bias: w("fuse0_b"), lnw: w("fuse0_lnw"),
                 lnb: w("fuse0_lnb"), out: b("g0"), ldo: 64, hin: 80, eps: e("fuse0"))
        try conv("fuse1+2", "conv3_64_64_1_4_2", X: b("g0"), Bw: w("fuse1_w"), bias: w("fuse1_b"), lnw: w("fuse1_lnw"),
                 lnb: w("fuse1_lnb"), W2: w("fuse2_w"), b2: w("fuse2_b"), out: b("ff"), ldo: 96, hin: 80, eps: e("fuse1"))
        // heatmap: split-kv attention of the 461 point queries over the 6400 pixels; then uv / conf and the xyz / vis MLPs
        var hp = AP()
        hp.NQ = Int32(Self.points); hp.NK = 6400; hp.ldq = 64; hp.ldk = 96; hp.ldv = 96; hp.ldo = 96
        hp.qbs = Int32(NPP * 64); hp.kbs = 6400 * 96; hp.vbs = 6400 * 96; hp.heads = 1
        hp.kvPerSplit = Int32(6400 / NS); hp.NQP = Int32(NPP)
        hp.scale = Float(config["temp"] as! Double) * 1.4426950408889634
        try add("heatmap", usesTensorOps ? "attn_heat" : "attn_sg_heat", [b("qp"), b("ff"), b("ff"), b("o"), b("OP"), b("ML")], hp,
                MTLSize(width: (Self.points + 31) / 32, height: NS, depth: B), 64)
        try add("point_heads", "point_heads", [b("OP"), b("ML"), b("qp"), b("ph"), w("xyz_a"), w("xyz_w2t"), w("xyz_b2"),
                                                w("vis_lnw"), w("vis_lnb"), w("vis_w0"), w("vis_b0"), w("vis_w1"), w("vis_b1"),
                                                b("uv"), b("xyz"), b("conf"), b("vis"), b("geo")],
                PHP(NP: Int32(Self.points), NPP: Int32(NPP), NS: Int32(NS), eps: e("vis"), invW: 1.0 / 80),
                MTLSize(width: (Self.points + 7) / 8, height: B, depth: 1), 256)

        // FAH's hand-object contact head (keypoint_head.contact_from), after the pose outputs: it reuses the encoder's
        // scratch buffers and needs point_heads' per-point inputs ("geo"). A branch of the encoder's last blocks (own
        // weights, from the snapshot c_branch_in took) and the heatmap features pooled 4x4 form an 800-token memory; 400
        // dense-point tokens (pooled features, query, 3D point and visibility of each point + the pose token + learned
        // positions) go through pre-norm transformer decoder layers (self-attention, cross-attention to the memory,
        // ReLU MLP); then a logit per point and per hand (mean and max over the points) at each tolerance.
        if usesContact {
            let K = contactDistances.count, layers = contactConfig["layers"] as! Int, f16 = MemoryLayout<Float16>.size
            try vit("cb", branchBlocks, e0: blocks - branchBlocks, X: "Xb", Y: "Yb", final: "ybr", norm: ("cb.norm_w", "cb.norm_b"),
                    norm2: ("cb.proj_lnw", "cb.proj_lnb"), eps2: 1e-5, tag: "c_")
            try mact("c_proj", act: 0, A: b("ybr"), lda: 384, Bw: w("cb.proj_w"), N: 256, K: 384, rows: MT, bias: w("cb.proj_b"),
                     out: b("bm"), ldo: 256)
            try add("c_mem", "contact_mem", [b("ff"), b("bm"), w("c.img_w"), w("c.img_b"), w("c.img_lnw"), w("c.img_lnb"),
                                             w("c.pos_mem"), b("mem")], CP(eps: 1e-6), MTLSize(width: 100, height: B, depth: 1), 256)
            // every decoder layer's cross-attention keys / values of the memory in one GEMM
            try mact("c_kv", act: 0, A: b("mem"), lda: 256, Bw: w("c.kv_w"), N: 512 * layers, K: 256, rows: B * 800,
                     bias: w("c.kv_b"), out: b("kv"), ldo: 512 * layers)
            try add("c_tok", "contact_tok", [b("pose"), w("c.tok_w"), w("c.tok_b"), b("ctok")], Int32(0),
                    MTLSize(width: 1, height: B, depth: 1), 256)
            try add("c_points", "contact_points", [b("geo"), w("c.in_lnw"), w("c.in_lnb"), w("c.in_w1"), w("c.in_b1"), w("c.in_w2"),
                                                   w("c.in_b2"), b("ctok"), w("c.pos"), w("c0.n1w"), w("c0.n1b"), b("cx"), b("cy")],
                    CP(a0: Int32(Self.points - 400), a1: Int32(Self.points), eps: 1e-5, eps2: 1e-5),
                    MTLSize(width: 50, height: B, depth: 1), 256)
            func attention(_ name: String, q: (MTLBuffer, Int), k: (MTLBuffer, Int), v: (MTLBuffer, Int), NK: Int, ldq: Int,
                           ldk: Int) throws {
                try add(name, "attn32_mma", [q, k, v, b("dao")],
                        A32(NQ: 400, NK: Int32(NK), ldq: Int32(ldq), ldk: Int32(ldk), ldv: Int32(ldk), ldo: 256,
                            scale: 1.4426950408889634 / Float(32).squareRoot()),
                        MTLSize(width: (400 + 31) / 32, height: 8, depth: B), 128)
            }
            for l in 0..<layers {
                let c = "c\(l)", last = l + 1 == layers
                try mact("c_sa_in", act: 0, A: b("cy"), lda: 256, Bw: w("\(c).sa_w"), N: 768, K: 256, rows: MT, bias: w("\(c).sa_b"),
                         out: b("dqkv"), ldo: 768)
                try attention("c_sa", q: b("dqkv"), k: (bufs["dqkv"]!, 256 * f16), v: (bufs["dqkv"]!, 512 * f16), NK: 400,
                              ldq: 768, ldk: 768)
                try rowln("c_sa_out", tsn: 64, wn: 4, mode: 0, A: b("dao"), lda: 256, Bw: w("\(c).sa_ow"), N: 256, K: 256, rows: MT,
                          bias: w("\(c).sa_ob"), gam: w("c.ones"), X: b("cx"), Y: b("cy"), ldo: 256, lnw: w("\(c).n2w"),
                          lnb: w("\(c).n2b"), eps: 1e-5, ldx: 256)
                try mact("c_ca_q", act: 0, A: b("cy"), lda: 256, Bw: w("\(c).ca_qw"), N: 256, K: 256, rows: MT, bias: w("\(c).ca_qb"),
                         out: b("dq"), ldo: 256)
                try attention("c_ca", q: b("dq"), k: (bufs["kv"]!, l * 512 * f16), v: (bufs["kv"]!, (l * 512 + 256) * f16),
                              NK: 800, ldq: 256, ldk: 512 * layers)
                try rowln("c_ca_out", tsn: 64, wn: 4, mode: 0, A: b("dao"), lda: 256, Bw: w("\(c).ca_ow"), N: 256, K: 256, rows: MT,
                          bias: w("\(c).ca_ob"), gam: w("c.ones"), X: b("cx"), Y: b("cy"), ldo: 256, lnw: w("\(c).n3w"),
                          lnb: w("\(c).n3b"), eps: 1e-5, ldx: 256)
                try mact("c_ff1", act: 2, A: b("cy"), lda: 256, Bw: w("\(c).l1w"), N: 512, K: 256, rows: MT, bias: w("\(c).l1b"),
                         out: b("dh"), ldo: 512)
                // the last layer's output LayerNorm is the contact output's
                try rowln("c_ff2", tsn: 64, wn: 4, mode: 0, A: b("dh"), lda: 512, Bw: w("\(c).l2w"), N: 256, K: 512, rows: MT,
                          bias: w("\(c).l2b"), gam: w("c.ones"), X: b("cx"), Y: b("cy"), ldo: 256,
                          lnw: w(last ? "c.out_lnw" : "c\(l + 1).n1w"), lnb: w(last ? "c.out_lnb" : "c\(l + 1).n1b"), eps: 1e-5,
                          ldx: 256)
            }
            try add("c_out", "contact_out", [b("cx"), b("cy"), w("c.out_w"), w("c.out_b"), w("c.hand_lnw"), w("c.hand_lnb"),
                                             w("c.hand_w1"), w("c.hand_b1"), w("c.hand_w2"), w("c.hand_b2"), b("contact"),
                                             b("contact_hand")], CP(a0: Int32(K), eps: 1e-5), MTLSize(width: 1, height: B, depth: 1), 256)
        }

        // Schedule on one concurrent encoder: independent branches overlap (the RGB stem with the patch embedding, the
        // pose-token MLPs with the keypoint-head convolutions); everything else is a chain with a barrier in between.
        var sched: [Step] = []
        func take(_ n: String, concurrent: Bool = false) {
            var st = out.remove(at: out.firstIndex { $0.name == n }!)
            st.concurrent = concurrent
            sched.append(st)
        }
        take("prep"); take("stem0", concurrent: true); take("embed"); take("stem1", concurrent: true)
        let tail = ["adapter", "tokens", "ln_tok", "pose", "img_proj", "pose_heads", "up0", "point_query", "up1",
                    "fuse0", "fuse1+2", "heatmap", "point_heads"]
        // the ViT blocks in order, with the contact branch's input snapshot in place; the contact head runs last
        let encoderChain = { (s: Step) in !tail.contains(s.name) && (!s.name.hasPrefix("c_") || s.name == "c_branch_in") }
        sched += out.filter(encoderChain)
        out.removeAll(where: encoderChain)
        take("adapter"); take("tokens"); take("ln_tok")
        take("pose"); take("img_proj", concurrent: true)
        take("pose_heads"); take("up0", concurrent: true)
        take("point_query"); take("up1", concurrent: true)
        take("fuse0"); take("fuse1+2"); take("heatmap"); take("point_heads")
        sched += out
        out.removeAll()
        precondition(out.isEmpty, "unscheduled steps \(out.map(\.name))")
        return sched
    }

    // MARK: run

    /// (maxBatch, 3, 320, 320) fp32 RGB in [0, 1].
    public var imageBuffer: MTLBuffer { bufs["img"]! }

    func setCamera(rays: [[Double]], conds: [[Double]]) {
        let r = bufs["ray"]!.contents().bindMemory(to: Float.self, capacity: maxBatch * 4)
        let c = bufs["cond"]!.contents().bindMemory(to: Float.self, capacity: maxBatch * 3)
        for (i, v) in rays.enumerated() { for k in 0..<4 { r[4 * i + k] = Float(v[k]) } }
        for (i, v) in conds.enumerated() { for k in 0..<3 { c[3 * i + k] = Float(v[k]) } }
    }

    /// Encodes the network for `batch` crops already in the input buffers.
    public func encode(into cb: MTLCommandBuffer, batch: Int) throws {
        guard let steps = schedules[batch], let enc = cb.makeComputeCommandEncoder(dispatchType: .concurrent) else {
            throw FAHError.metal("batch \(batch) out of 1...\(maxBatch)")
        }
        let contact = usesContact && contactActive
        for (i, s) in steps.enumerated() {
            if !contact && s.name.hasPrefix("c_") { continue }
            if i > 0 && !s.concurrent { enc.memoryBarrier(scope: .buffers) }
            enc.setComputePipelineState(s.pso)
            for (slot, buf, off) in s.bufs { enc.setBuffer(buf, offset: off, index: slot) }
            s.bytes.1.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: $0.count, index: s.bytes.0) }
            enc.dispatchThreadgroups(s.groups, threadsPerThreadgroup: s.threads)
        }
        enc.endEncoding()
    }

    /// The outputs of the last completed inference.
    public func outputs(batch B: Int) -> FAHPoseOutputs {
        let P = Self.points
        func arr(_ n: String, _ c: Int) -> [Float] { Array(UnsafeBufferPointer(start: bufs[n]!.contents().assumingMemoryBound(to: Float.self), count: c)) }
        let K = contactDistances.count
        return FAHPoseOutputs(uv: arr("uv", B * P * 2), xyz: arr("xyz", B * P * 3), conf: arr("conf", B * P), vis: arr("vis", B * P),
                              depth: arr("depth", B), token: arr("pose", B * 1024), verify: arr("verify", B * 2),
                              contact: usesContact && contactActive ? arr("contact", B * 400 * K) : [],
                              contactHand: usesContact && contactActive ? arr("contact_hand", B * K) : [])
    }

    /// Synchronous inference on `batch` crops: img (B, 3, 320, 320) RGB in [0, 1], ray (B, 4), cond (B, 3).
    /// Returns the outputs and the GPU time (s).
    public func predict(img: UnsafePointer<Float>, ray: UnsafePointer<Float>, cond: UnsafePointer<Float>,
                        batch B: Int) throws -> (FAHPoseOutputs, gpuSeconds: Double) {
        guard (1...maxBatch).contains(B) else { throw FAHError.metal("batch \(B) out of 1...\(maxBatch)") }
        bufs["img"]!.contents().copyMemory(from: img, byteCount: B * 3 * 320 * 320 * 4)
        bufs["ray"]!.contents().copyMemory(from: ray, byteCount: B * 4 * 4)
        bufs["cond"]!.contents().copyMemory(from: cond, byteCount: B * 3 * 4)
        guard let cb = queue.makeCommandBufferWithUnretainedReferences() else { throw FAHError.metal("command buffer") }
        try encode(into: cb, batch: B)
        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error { throw FAHError.metal("\(err)") }
        return (outputs(batch: B), cb.gpuEndTime - cb.gpuStartTime)
    }
}
