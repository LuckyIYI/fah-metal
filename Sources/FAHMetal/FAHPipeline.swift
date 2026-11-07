import Accelerate
import CoreGraphics
import CoreVideo
import Foundation
import Metal
import simd

/// FAH-S hand pose on Apple silicon: FAH's hand detector (Core ML), crops on the GPU, the pose network in Metal
/// (`FAHPoseNetwork`), and FAH's decoding, verification and tracking ported from Python (fah.estimator,
/// bench.methods.fah), plus the MANO per-part Procrustes of the dense surface (fah.mano_procrustes).
///
/// Thread safety: calls are serialised by an internal lock. `process` blocks for the GPU and Core ML, so call it from
/// one background queue, not the main thread.
@available(iOS 18.0, macOS 15.0, *)
public final class FAHPipeline: @unchecked Sendable {
    public static let detectorModelName = "detector-640"
    public static let manoResourceName = "fah_mano.bin"

    private let lock = NSLock()
    private var _options: FAHOptions
    private var _smoother: FAHSmoother?
    private var _lastTimings = FAHTimings()

    private let queue: MTLCommandQueue
    private let warp: ImageWarp
    private let pose: PoseNetwork
    private let detector: Detector
    private let mano: ManoData
    private let procrustes: ManoProcrustes
    private let fitter: ManoFitter
    /// the last fitted hand shape per side (a hand's shape does not change: each frame's fit starts from it)
    private var shape: [Bool: [Float]] = [:]
    private var meshTime = 0.0  // ms, this frame's meshes

    // tracking state (bench.methods.fah: self.prev, DetectionSchedule.t)
    private var prev: [Tracked] = []
    private var frameCount = 0
    private var forceDetect = false

    public var options: FAHOptions {
        get { lock.lock(); defer { lock.unlock() }; return _options }
        set { lock.lock(); _options = newValue; lock.unlock() }
    }
    /// Optional smoothing of the returned hands (nil, the default: none).
    public var smoother: FAHSmoother? {
        get { lock.lock(); defer { lock.unlock() }; return _smoother }
        set { lock.lock(); _smoother = newValue; lock.unlock() }
    }
    /// Per-stage times of the last `process` call.
    public var lastTimings: FAHTimings { lock.lock(); defer { lock.unlock() }; return _lastTimings }
    /// True when the pose network runs on Metal 4 tensor ops, false for simdgroup matrices.
    public let usesTensorOps: Bool
    /// True when the export has FAH's hand-object contact head (`FAHOptions.contact` switches it per frame).
    public let usesContact: Bool
    /// The 1538 triangles of `FAHHand.denseVertices` (MANO's topology, right-hand winding; for left hands, whose
    /// surface is mirrored, the winding is reversed).
    public let meshFaces: [SIMD3<UInt32>]

    /// - Parameters:
    ///   - modelDirectory: the folder tools/export.py wrote: `fah-s-pose.bin` / `.json`, `detector-640.mlpackage`
    ///     (or a compiled `.mlmodelc`) and `fah_mano.bin`.
    ///   - poseBackend: the pose network's matrix-multiply path (default: the fastest for this GPU).
    public init(modelDirectory: URL, options: FAHOptions = FAHOptions(), poseBackend: FAHPoseBackend = .automatic,
                device: MTLDevice? = nil) throws {
        guard let dev = device ?? MTLCreateSystemDefaultDevice(), let q = dev.makeCommandQueue() else { throw FAHError.metal("no Metal device") }
        _options = options
        queue = q
        warp = try ImageWarp(device: dev)
        let manoURL = modelDirectory.appendingPathComponent(FAHPipeline.manoResourceName)
        guard FileManager.default.fileExists(atPath: manoURL.path) else { throw FAHError.missingResource(manoURL.path) }
        mano = try ManoData(url: manoURL)
        procrustes = ManoProcrustes(mano)
        fitter = ManoFitter(mano)
        meshFaces = mano.faces
        let net = try FAHPoseNetwork(modelDirectory: modelDirectory, maxBatch: 4, backend: poseBackend, device: dev)
        usesTensorOps = net.usesTensorOps
        usesContact = net.usesContact
        pose = net
        detector = try Detector(url: compiledModelURL(directory: modelDirectory, name: FAHPipeline.detectorModelName),
                                units: options.detectorComputeUnits.ml, device: dev)
    }

    /// Forget the tracked hands (next frame runs the detector).
    public func reset() {
        lock.lock(); defer { lock.unlock() }
        prev = []
        shape = [:]
        frameCount = 0
        forceDetect = false
        _smoother?.reset()
    }

    /// Loads (and compiles for the GPU / ANE) the detector function for this frame size and rotation, so the first
    /// `process` call does not pay for it (~0.1-1 s).
    public func warmUp(width: Int, height: Int, rotation: FAHRotation) throws {
        lock.lock(); defer { lock.unlock() }
        let g = Geometry(width: width, height: height, rotation: rotation, K: matrix_identity_float3x3)
        let lb = try detector.letterbox(width: g.W, height: g.H)
        _ = try detector.run(lb)
    }

    /// One frame.
    /// - Parameters:
    ///   - pixelBuffer: BGRA (`kCVPixelFormatType_32BGRA`) or bi-planar 4:2:0 YCbCr (`420f` / `420v`, ARKit's
    ///     `capturedImage`), IOSurface backed.
    ///   - intrinsics: pinhole intrinsics of this buffer in its own pixels, column-major as ARKit's `camera.intrinsics`
    ///     (columns.0.x = fx, columns.1.y = fy, columns.2 = (cx, cy, 1)).
    ///   - rotation: clockwise rotation that makes the image upright (portrait iPhone + ARKit: `.clockwise90`).
    ///   - timestamp: frame time in seconds (for the smoother; default: now).
    /// - Returns: hands in the buffer's own pixels / camera frame, best first (at most `options.maxHands`).
    public func process(pixelBuffer: CVPixelBuffer, intrinsics: simd_float3x3, rotation: FAHRotation = .none,
                        timestamp: Double? = nil) throws -> [FAHHand] {
        lock.lock(); defer { lock.unlock() }
        var tm = FAHTimings()
        let t0 = now()
        let g = Geometry(width: CVPixelBufferGetWidth(pixelBuffer), height: CVPixelBufferGetHeight(pixelBuffer),
                         rotation: rotation, K: intrinsics)
        let src = try warp.source(pixelBuffer)
        let o = _options
        var ctx = Context(g: g, src: src, o: o)
        pose.contactActive = o.contact

        // detector schedule (fah.video.DetectionSchedule): every N frames, or when nothing is tracked
        let every = o.detectorInterval
        let run = prev.isEmpty || forceDetect || (every > 0 ? frameCount % every == 0 : false)
        frameCount += 1
        forceDetect = false
        var boxes: [SIMD4<Double>] = [], scores: [Float] = [], rights: [Bool] = []
        if run {
            tm.ranDetector = true
            for d in try detect(&ctx, &tm) { boxes.append(d.box); scores.append(d.score); rights.append(d.isRight) }
        } else {
            for p in prev {
                var b = g.boxToUpright(p.predBoxSensor)
                b.x = min(max(b.x, 0), Double(g.W - 1)); b.z = min(max(b.z, 0), Double(g.W - 1))
                b.y = min(max(b.y, 0), Double(g.H - 1)); b.w = min(max(b.w, 0), Double(g.H - 1))
                if b.z - b.x > 4 && b.w - b.y > 4 { boxes.append(b); scores.append(p.score); rights.append(p.isRight) }
            }
        }
        var kept = try verified(&ctx, boxes: boxes, scores: scores, isRight: rights)
        if o.maxHands > 0 && kept.count > o.maxHands { kept = Array(kept.prefix(o.maxHands)) }
        tm.crop = ctx.tCrop
        tm.pose = ctx.tPose
        tm.poseCrops = ctx.nCrops

        let tp = now()
        meshTime = 0
        var hands = kept.map { makeHand($0, g, fromDetector: run) }
        tm.mesh = meshTime
        tm.procrustes = (now() - tp) * 1e3 - meshTime
        prev = kept.map { Tracked(predBoxSensor: g.boxToSensor($0.predBox), score: $0.score, isRight: $0.isRight) }
        if o.redetectBelowPresence > 0, kept.contains(where: { $0.presence < o.redetectBelowPresence }) { forceDetect = true }
        if let s = _smoother { hands = s.smooth(hands, timestamp: timestamp ?? ProcessInfo.processInfo.systemUptime) }
        tm.total = (now() - t0) * 1e3
        tm.decode = max(tm.total - tm.detectorPrep - tm.detector - tm.detectorDecode - tm.crop - tm.pose - tm.procrustes - tm.mesh, 0)
        _lastTimings = tm
        warp.flush()
        return hands
    }

    private let workQueue = DispatchQueue(label: "FAHPipeline", qos: .userInitiated)

    /// `process` on the pipeline's own serial queue, for Swift concurrency callers. The buffer must stay valid (and
    /// unmodified) until this returns.
    public func processAsync(pixelBuffer: CVPixelBuffer, intrinsics: simd_float3x3, rotation: FAHRotation = .none,
                             timestamp: Double? = nil) async throws -> [FAHHand] {
        nonisolated(unsafe) let pb = pixelBuffer
        return try await withCheckedThrowingContinuation { c in
            workQueue.async {
                c.resume(with: Result { try self.process(pixelBuffer: pb, intrinsics: intrinsics, rotation: rotation, timestamp: timestamp) })
            }
        }
    }

    // MARK: - internals

    private struct Tracked { var predBoxSensor: SIMD4<Double>; var score: Float; var isRight: Bool }

    private struct Context {
        let g: Geometry
        let src: ImageWarp.Source
        let o: FAHOptions
        var tCrop = 0.0, tPose = 0.0
        var nCrops = 0
    }

    /// A hand as fah.estimator._kp_only_results makes it, in the upright frame.
    struct RawHand {
        var isRight: Bool
        var score: Float            // detector score in; fused score after verification
        var detScore: Float = 0
        var presence: Float
        var sideScore: Float
        var bbox: SIMD4<Double>     // the detector / tracking box
        var cropBox: SIMD4<Double>
        var uv: [SIMD2<Double>]     // 461 points, upright px (hidden surface points: projected 3D)
        var xyz: [SIMD3<Double>]    // 461 points, root-relative (camera axes)
        var t: SIMD3<Double>
        var verts: [SIMD3<Float>]   // 778 dense surface (camera frame)
        var predBox: SIMD4<Double>
        var contact: [Float] = []   // 778 vertex probabilities (10 mm)
        var contactScore: Float?
    }

    /// Per crop: the affine (frame -> crop, in the crop's (mirrored) frame) and the network outputs.
    private struct CropOut {
        var A: (Double, Double, Double, Double, Double, Double)  // s, 0, tx; 0, s, ty
        var uv: [Float], xyz: [Float], vis: [Float]
        var depth: Float
        var verify: SIMD2<Float>
        var contact: [Float]        // (400) point logits at the 10 mm tolerance, empty without the contact head
        var contactHand: Float?
    }

    private func detect(_ ctx: inout Context, _ tm: inout FAHTimings) throws -> [Detector.Detection] {
        let g = ctx.g
        var t = now()
        let lb = try detector.letterbox(width: g.W, height: g.H)
        let (buf, _) = try detector.input(lb)
        let a = g.A_su
        let p = ImageWarp.LetterboxParams(
            outSize: SIMD2(Int32(lb.W), Int32(lb.H)), resized: SIMD2(Int32(lb.nw), Int32(lb.nh)), pad: SIMD2(Int32(lb.px), Int32(lb.py)),
            canvas: SIMD2(Int32(g.W), Int32(g.H)), scale: SIMD2(Float(Double(g.W) / Double(lb.nw)), Float(Double(g.H) / Double(lb.nh))),
            srcSize: SIMD2(Int32(g.srcW), Int32(g.srcH)),
            a0: SIMD4(Float(a.0), Float(a.1), Float(a.2), 0), a1: SIMD4(Float(a.3), Float(a.4), Float(a.5), 0), quarter: g.quarter ? 1 : 0)
        guard let cb = queue.makeCommandBuffer() else { throw FAHError.metal("command buffer") }
        warp.encodeLetterbox(cb, src: ctx.src, params: p, out: buf)
        cb.commit()
        cb.waitUntilCompleted()
        tm.detectorPrep = (now() - t) * 1e3
        t = now()
        let (bx, sc) = try detector.run(lb)
        tm.detector = (now() - t) * 1e3
        t = now()
        let d = detector.decode(boxes: bx, scores: sc, lb: lb, canvasW: g.W, canvasH: g.H, conf: ctx.o.detectorConfidence, iou: ctx.o.detectorIoU)
        tm.detectorDecode = (now() - t) * 1e3
        return d
    }

    /// bench.methods.fah.FAHMethod._verified (image mode): pose on every box, the other side for doubtful boxes,
    /// fused scores, presence / score filters, identity NMS by wrist distance (pair_wrist).
    private func verified(_ ctx: inout Context, boxes: [SIMD4<Double>], scores: [Float], isRight: [Bool]) throws -> [RawHand] {
        if boxes.isEmpty { return [] }
        let o = ctx.o
        var hands = try processImage(&ctx, boxes: boxes, isRight: isRight, scores: scores)
        let low = hands.indices.filter { hands[$0].sideScore < o.sideThreshold }
        var other: [RawHand] = []
        if !low.isEmpty {
            let alt = try processImage(&ctx, boxes: low.map { boxes[$0] }, isRight: low.map { !isRight[$0] }, scores: low.map { scores[$0] })
            for (k, i) in low.enumerated() {
                let first = hands[i], h = alt[k]
                if h.sideScore > first.sideScore { hands[i] = h; other.append(first) } else { other.append(h) }
            }
        }
        func fuse(_ h: inout RawHand) { h.detScore = h.score; h.score = (h.detScore * h.presence).squareRoot() }
        for i in hands.indices { fuse(&hands[i]) }
        for i in other.indices { fuse(&other[i]) }
        hands = hands.filter { $0.presence >= o.presenceThreshold && $0.score >= o.fuseThreshold }
        hands = stableSorted(hands) { $0.score > $1.score }
        guard o.pairWrist > 0 else { return hands }
        var cand = hands + other.filter { $0.presence >= o.presenceThreshold }
        cand = stableSorted(cand) { $0.score * $0.sideScore > $1.score * $1.sideScore }
        func same(_ h: RawHand, _ k: RawHand) -> Bool {
            let pa = simd_length(h.uv[0] - h.uv[9]), pb = simd_length(k.uv[0] - k.uv[9])
            return simd_length(h.uv[0] - k.uv[0]) < Double(o.pairWrist) * 0.5 * (pa + pb)
        }
        var keep: [RawHand] = []
        for h in cand where h.score >= o.fuseThreshold && !keep.contains(where: { same(h, $0) }) { keep.append(h) }
        return keep
    }

    /// fah.estimator.HandEstimator.process_image with given boxes (upright frame): left hands mirrored into right ones,
    /// crops, the joint-at-the-border re-crop (recrop_border), decoding.
    private func processImage(_ ctx: inout Context, boxes: [SIMD4<Double>], isRight: [Bool], scores: [Float]) throws -> [RawHand] {
        let g = ctx.g
        let W = Double(g.W), H = Double(g.H)
        var boxesIn = zip(boxes, isRight).map { $1 ? $0 : flipBox($0, g.W) }
        var out = try runCrops(&ctx, boxesIn: boxesIn, flips: isRight.map { !$0 })
        let m = Double(ctx.o.recropBorder)
        if m > 0 {
            var redo: [Int] = []
            let S = Double(FAHPoseNetwork.inputSize)
            for i in boxes.indices {
                let c = out[i]
                var anyEdge = false
                for j in 0..<21 { for a in 0..<2 { let u = Double(c.uv[2 * j + a]); if u < m || u > 1 - m { anyEdge = true } } }
                if !anyEdge { continue }
                let s = c.A.0
                let lo = SIMD2(-c.A.2 / s, -c.A.5 / s), hi = SIMD2((S - c.A.2) / s, (S - c.A.5) / s)
                var loT = [false, false], hiT = [false, false]
                var pmin = SIMD2(Double.infinity, Double.infinity), pmax = -pmin
                for j in 0..<21 {
                    let u = SIMD2(Double(c.uv[2 * j]), Double(c.uv[2 * j + 1]))
                    let p = SIMD2((u.x * S - c.A.2) / s, (u.y * S - c.A.5) / s)
                    pmin = simd_min(pmin, p); pmax = simd_max(pmax, p)
                    for a in 0..<2 {
                        if u[a] < m && lo[a] > 0 { loT[a] = true }
                        if u[a] > 1 - m && hi[a] < (a == 0 ? W : H) { hiT[a] = true }
                    }
                }
                if !(loT[0] || loT[1] || hiT[0] || hiT[1]) { continue }
                let ext = 0.25 * S / s
                let b = boxesIn[i]
                boxesIn[i] = SIMD4(min(b.x, pmin.x - (loT[0] ? ext : 0)), min(b.y, pmin.y - (loT[1] ? ext : 0)),
                                   max(b.z, pmax.x + (hiT[0] ? ext : 0)), max(b.w, pmax.y + (hiT[1] ? ext : 0)))
                redo.append(i)
            }
            if !redo.isEmpty {
                let o2 = try runCrops(&ctx, boxesIn: redo.map { boxesIn[$0] }, flips: redo.map { !isRight[$0] })
                for (k, i) in redo.enumerated() {
                    var n = o2[k]
                    if !(abs(n.verify.y) > abs(out[i].verify.y)) { n.verify = out[i].verify }  // side / presence: the more confident pass
                    out[i] = n
                }
            }
        }
        return boxes.indices.map { decodeHand(out[$0], g: g, box: boxes[$0], boxIn: boxesIn[$0], isRight: isRight[$0], score: scores[$0]) }
    }

    /// Crops (GPU) + pose network for boxes in the crop frame (mirrored for flips).
    private func runCrops(_ ctx: inout Context, boxesIn: [SIMD4<Double>], flips: [Bool]) throws -> [CropOut] {
        let g = ctx.g, o = ctx.o
        let S = Double(FAHPoseNetwork.inputSize)
        var affines: [(Double, Double, Double, Double, Double, Double)] = [], rays: [[Double]] = [], conds: [[Double]] = [], srcMaps: [[Double]] = []
        for (b0, flip) in zip(boxesIn, flips) {
            var b = b0
            let bw = b.z - b.x, bh = b.w - b.y
            if o.minCrop > 0 {  // tall boxes widened to min_crop x their long side
                let need = o.minCrop * bh / (4.0 / 3.0 * o.bboxPadding)
                if bh > bw && bw < need { let d = (need - bw) / 2; b.x -= d; b.z += d }
            }
            let cx = (b.x + b.z) * 0.5, cy = (b.y + b.w) * 0.5
            var sw = (b.z - b.x) * o.bboxPadding, sh = (b.w - b.y) * o.bboxPadding
            for ar in [0.75, 1.0] { if sw > sh * ar { sh = sw / ar } else { sw = sh * ar } }
            let s = S / sw
            let A = (s, 0.0, S * 0.5 - s * cx, 0.0, s, S * 0.5 - s * cy)
            affines.append(A)
            let K = flip ? g.Kflip : g.K
            rays.append([1 / (A.0 * K.fx), (-A.2 / A.0 - K.cx) / K.fx, 1 / (A.4 * K.fy), (-A.5 / A.4 - K.cy) / K.fy])
            conds.append([(cx - K.cx) / K.fx, (cy - K.cy) / K.fx, sw / K.fx])
            // the frame -> crop affine OpenCV warps with (A F for mirrored crops: x -> W - 1 - x first)
            srcMaps.append(flip ? [-A.0, A.1, A.0 * Double(g.W - 1) + A.2, A.3, A.4, A.5] : [A.0, A.1, A.2, A.3, A.4, A.5])
        }
        // Crops and the pose network go to the GPU back to back and are waited for once. They are two command
        // buffers on one queue only so that the crop kernels' GPU time can be reported on its own.
        var outs: [CropOut] = []
        let P = FAHPoseNetwork.points
        var start = 0
        while start < boxesIn.count {
            let b = min(pose.maxBatch, boxesIn.count - start)
            let r = start..<(start + b)
            let t = now()
            guard let cropCB = queue.makeCommandBuffer(), let poseCB = queue.makeCommandBuffer() else { throw FAHError.metal("command buffer") }
            try warp.encodeCrops(cropCB, src: ctx.src, affines: Array(srcMaps[r]), out: pose.imageBuffer, canvasToSource: g.A_su,
                                 size: FAHPoseNetwork.inputSize)
            cropCB.commit()
            pose.setCamera(rays: Array(rays[r]), conds: Array(conds[r]))
            try pose.encode(into: poseCB, batch: b)
            poseCB.commit()
            poseCB.waitUntilCompleted()
            if let e = cropCB.error ?? poseCB.error { throw FAHError.metal("\(e)") }
            let cropMs = max(cropCB.gpuEndTime - cropCB.gpuStartTime, 0) * 1e3
            ctx.tCrop += cropMs
            ctx.tPose += (now() - t) * 1e3 - cropMs
            let o = pose.outputs(batch: b)
            let K = pose.contactDistances.count, mi = pose.contactMain
            for k in 0..<b {
                let contact = o.contact.isEmpty ? [] : (0..<400).map { o.contact[(k * 400 + $0) * K + mi] }
                outs.append(CropOut(A: affines[start + k], uv: Array(o.uv[(k * P * 2)..<((k + 1) * P * 2)]),
                                    xyz: Array(o.xyz[(k * P * 3)..<((k + 1) * P * 3)]), vis: Array(o.vis[(k * P)..<((k + 1) * P)]),
                                    depth: o.depth[k], verify: SIMD2(o.verify[2 * k], o.verify[2 * k + 1]), contact: contact,
                                    contactHand: o.contactHand.isEmpty ? nil : o.contactHand[k * K + mi]))
            }
            start += b
        }
        ctx.nCrops += boxesIn.count
        return outs
    }

    /// fah.estimator.HandEstimator._kp_only_results for one crop (dense layout: 21 joints, 40 bone points, 400 surface
    /// points; root depth from pt_depth, x / y from all points; hidden surface points' 2D from the 3D).
    private func decodeHand(_ c: CropOut, g: Geometry, box: SIMD4<Double>, boxIn: SIMD4<Double>, isRight right: Bool, score: Float) -> RawHand {
        let P = FAHPoseNetwork.points, S = Double(FAHPoseNetwork.inputSize)
        let K = g.K
        let s = c.A.0
        var uv = [SIMD2<Double>](repeating: .zero, count: P)
        var xyz = [SIMD3<Double>](repeating: .zero, count: P)
        let W1 = Double(g.W - 1)
        for i in 0..<P {
            var u = SIMD2((Double(c.uv[2 * i]) * S - c.A.2) / s, (Double(c.uv[2 * i + 1]) * S - c.A.5) / c.A.4)
            var x = SIMD3(Double(c.xyz[3 * i]), Double(c.xyz[3 * i + 1]), Double(c.xyz[3 * i + 2]))
            if !right { u.x = W1 - u.x; x.x = -x.x }
            uv[i] = u; xyz[i] = x
        }
        let tz = Double(exp(c.depth)) * K.fx * s / S
        var xy = SIMD2<Double>.zero
        for i in 0..<P {
            let u = uv[i] - SIMD2(K.cx, K.cy)
            xy += u * (xyz[i].z + tz) / SIMD2(K.fx, K.fy) - SIMD2(xyz[i].x, xyz[i].y)
        }
        xy /= Double(P)
        let t = SIMD3(xy.x, xy.y, tz)
        // dense surface: 400 points -> 778 vertices (sub_R, float32 as FAH)
        let nb = mano.nbone, nd = mano.ndense, first = 21 + nb
        var x400 = [SIMD3<Float>](repeating: .zero, count: nd)
        var c3 = SIMD3<Float>.zero
        for k in 0..<nd { x400[k] = f3(xyz[first + k] + t); c3 += x400[k] }
        c3 /= Float(nd)
        for k in 0..<nd { x400[k] -= c3 }
        var flat = [Float](repeating: 0, count: nd * 3)
        for k in 0..<nd { flat[3 * k] = x400[k].x; flat[3 * k + 1] = x400[k].y; flat[3 * k + 2] = x400[k].z }
        var prod = [Float](repeating: 0, count: mano.nv * 3)
        vDSP_mmul(mano.subR, 1, flat, 1, &prod, 1, vDSP_Length(mano.nv), 3, vDSP_Length(nd))  // (nv, nd) @ (nd, 3)
        let verts = (0..<mano.nv).map { SIMD3(prod[3 * $0], prod[3 * $0 + 1], prod[3 * $0 + 2]) + c3 }
        // hidden points (pt_vis < 0): 2D from the projected 3D (bone points: raw 3D; surface: the 778 surface at its vertex)
        for i in 21..<P where c.vis[i] < 0 {
            let pv: SIMD3<Double> = i < first ? xyz[i] + t : d3(verts[mano.denseIndex[i - first]])
            uv[i] = SIMD2(K.fx * pv.x / pv.z + K.cx, K.fy * pv.y / pv.z + K.cy)
        }
        var lo = SIMD2(Double.infinity, Double.infinity), hi = -lo
        for p in uv { lo = simd_min(lo, p); hi = simd_max(hi, p) }
        let ctr = (lo + hi) / 2, side = max(hi.x - lo.x, hi.y - lo.y)
        let sig = { (x: Float) -> Float in 1 / (1 + exp(-x)) }
        // contact: point probabilities -> the 778 vertices through the same surface map, clipped (fah.estimator)
        var contact: [Float] = []
        if c.contact.count == nd {
            let pc = c.contact.map(sig)
            var pv = [Float](repeating: 0, count: mano.nv)
            vDSP_mmul(mano.subR, 1, pc, 1, &pv, 1, vDSP_Length(mano.nv), 1, vDSP_Length(nd))
            contact = pv.map { min(max($0, 0), 1) }
        }
        return RawHand(isRight: right, score: score, presence: sig(c.verify.x), sideScore: sig(c.verify.y), bbox: box,
                       cropBox: right ? boxIn : flipBox(boxIn, g.W), uv: uv, xyz: xyz, t: t, verts: verts,
                       predBox: SIMD4(ctr.x - side / 2, ctr.y - side / 2, ctr.x + side / 2, ctr.y + side / 2),
                       contact: contact, contactScore: c.contactHand.map(sig))
    }

    /// MANO part -> contact region (palm, thumb, index, middle, ring, pinky), as fah.estimator's contact_groups.
    private static let contactRegion = [0, 2, 2, 2, 3, 3, 3, 5, 5, 5, 4, 4, 4, 1, 1, 1]

    /// The public hand: MANO rotations from the Procrustes, everything mapped back to the buffer's frame.
    private func makeHand(_ h: RawHand, _ g: Geometry, fromDetector: Bool) -> FAHHand {
        let mx: Double = h.isRight ? 1 : -1
        let V = h.verts.map { SIMD3<Double>(Double($0.x) * mx, Double($0.y), Double($0.z)) }  // right-hand frame
        let (R, s) = procrustes.rotations(V)
        let (R0, L, Wl) = procrustes.localPose(R)
        let M = simd_double3x3(diagonal: SIMD3(mx, 1, 1))
        let Rt = g.camRot.transpose  // upright camera -> buffer camera
        let root = Rt * (M * R0 * M)
        let local = L.map { f33(M * $0 * M) }
        let aa = Wl.map { SIMD3<Float>(Float($0.x), Float(mx * $0.y), Float(mx * $0.z)) }
        let j3 = (0..<21).map { f3(Rt * (h.xyz[$0] + h.t)) }
        let dense = h.verts.map { f3(Rt * d3($0)) }
        let K21 = (0..<21).map { p -> SIMD3<Double> in let q = h.xyz[p] + h.t; return SIMD3(q.x * mx, q.y, q.z) }  // right-hand frame
        let tm0 = now()
        defer { meshTime += (now() - tm0) * 1e3 }
        let posed: [SIMD3<Float>]
        if !_options.mesh {
            posed = []
        } else if _options.meshFitIterations > 0 {
            // FAH's MANO fit (from the Procrustes pose and this hand's last shape), then onto the 2D keypoints
            let fit = fitter.fit(surface: V, joints: K21, R0: R0, pose0: Wl.flatMap { [Float($0.x), Float($0.y), Float($0.z)] },
                                 betas0: shape[h.isRight] ?? [Float](repeating: 0, count: 10), iterations: _options.meshFitIterations)
            shape[h.isRight] = fit.betas
            let mesh = fitter.alignTo2D(fit.vertices.map { SIMD3($0.x * mx, $0.y, $0.z) }, uv: Array(h.uv[0..<21]),
                                        fx: g.K.fx, fy: g.K.fy, cx: g.K.cx, cy: g.K.cy)
            posed = mesh.map { f3(Rt * $0) }
        } else {
            posed = procrustes.keypointMesh(K21, R, s).map { f3(Rt * SIMD3($0.x * mx, $0.y, $0.z)) }
        }
        let j2 = (0..<21).map { p -> SIMD2<Float> in let q = g.toSensor(h.uv[p]); return SIMD2(Float(q.x), Float(q.y)) }
        var hand = FAHHand(isRight: h.isRight, score: h.score, presence: h.presence, sideScore: h.sideScore,
                           bbox: rect(g.boxToSensor(h.bbox)), joints2d: j2, joints3d: j3, denseVertices: dense, meshVertices: posed,
                           wristPosition: j3[0], rootRotation: f33(root), localRotations: local, localAxisAngles: aa,
                           handScale: Float(s), predictedBox: rect(g.boxToSensor(h.predBox)), fromDetector: fromDetector)
        if !h.contact.isEmpty {
            hand.contact = h.contact
            var fingers = [Float](repeating: 0, count: 6)
            for (v, part) in mano.part.enumerated() { fingers[Self.contactRegion[part]] = max(fingers[Self.contactRegion[part]], h.contact[v]) }
            hand.contactFingers = fingers
            hand.contactScore = h.contactScore
        }
        return hand
    }
}
