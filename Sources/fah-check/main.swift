// fah-check: the Swift side of tools/check.py.
//
//   fah-check pipeline --frames <dir> --models <dir> [--out swift.json] [--det-every N] [--rotate D] [--yuv]
//                      [--timing-frames 60] [--backend auto|tensor|simdgroup]
//       Runs FAHPipeline on the frames check.py saved (frame_XXXXX.png + python.json: the same pixels and intrinsics
//       the Python run used) and writes the hands as JSON for check.py to compare, then times the pipeline on the
//       first --timing-frames frames.
//       --rotate D: each frame is first turned D degrees counter-clockwise (a landscape "sensor" image, intrinsics
//       turned with it) and the pipeline is told to turn it back (rotation = D degrees clockwise); the results come
//       back in the turned frame and check.py maps them back with its own code, an independent test of the rotation
//       path. --yuv: the frames are converted to 420f (bi-planar full-range BT.601, like camera buffers) first.
//
//   fah-check pose --models <dir> --ref <dir> [--iters 200] [--backend ...]
//       The pose network alone against PyTorch fp32 (ref.bin / ref.json from `check.py pose`), then its speed.
//
//   fah-check speed --models <dir> --video <file> [--frames 120] [--hfov 70] [--backend ...]
//       The whole pipeline on a video's frames (decoded into memory first, then run back to back as from a fast
//       camera, tracking with the default options) with the mesh and the contact head each on and off.
import AVFoundation
import CoreGraphics
import CoreVideo
import FAHMetal
import Foundation
import ImageIO
import simd

func arg(_ name: String) -> String? {
    let a = CommandLine.arguments
    guard let i = a.firstIndex(of: name), i + 1 < a.count else { return nil }
    return a[i + 1]
}
func flag(_ name: String) -> Bool { CommandLine.arguments.contains(name) }
func fail(_ s: String) -> Never { FileHandle.standardError.write((s + "\n").data(using: .utf8)!); exit(1) }
func path(_ s: String) -> URL { URL(fileURLWithPath: (s as NSString).expandingTildeInPath) }

func median(_ x: [Double]) -> Double {
    if x.isEmpty { return .nan }
    let s = x.sorted()
    return s.count % 2 == 1 ? s[s.count / 2] : 0.5 * (s[s.count / 2 - 1] + s[s.count / 2])
}
func percentile(_ x: [Double], _ p: Double) -> Double {
    let s = x.sorted()
    return s.isEmpty ? .nan : s[min(s.count - 1, Int(Double(s.count - 1) * p / 100 + 0.5))]
}

let backend: FAHPoseBackend = switch arg("--backend") ?? "auto" {
    case "tensor": .tensorOps
    case "simdgroup": .simdgroupMatrix
    default: .automatic
}
guard let modelsArg = arg("--models") else {
    fail("usage: fah-check pipeline --frames <dir> --models <dir> [--out file] [--det-every N] [--rotate D] [--yuv]\n" +
         "       fah-check pose --models <dir> --ref <dir> [--iters N]")
}
let models = path(modelsArg)

// MARK: - pipeline parity

func loadBGRA(_ url: URL) -> (w: Int, h: Int, px: [UInt8]) {
    guard let src = CGImageSourceCreateWithURL(url as CFURL, nil), let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
        fail("cannot read \(url.path)")
    }
    let w = img.width, h = img.height
    var px = [UInt8](repeating: 0, count: w * h * 4)
    // drawn into its own colour space: the PNG bytes as written by OpenCV, no colour conversion
    let cs = img.colorSpace ?? CGColorSpace(name: CGColorSpace.sRGB)!
    let ctx = CGContext(data: &px, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4, space: cs,
                        bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)!
    ctx.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
    return (w, h, px)
}

func makeBuffer(_ w: Int, _ h: Int, _ fmt: OSType) -> CVPixelBuffer {
    var pb: CVPixelBuffer?
    let attrs: [String: Any] = [kCVPixelBufferIOSurfacePropertiesKey as String: [:], kCVPixelBufferMetalCompatibilityKey as String: true]
    CVPixelBufferCreate(nil, w, h, fmt, attrs as CFDictionary, &pb)
    return pb!
}

/// The upright image turned k quarter turns counter-clockwise, as a BGRA or 420f buffer.
func sensorBuffer(_ img: (w: Int, h: Int, px: [UInt8]), k: Int, yuv: Bool) -> CVPixelBuffer {
    let (Ws, Hs) = k % 2 == 1 ? (img.h, img.w) : (img.w, img.h)
    // sensor (xs, ys) -> upright (xu, yu): turning the sensor image clockwise by k quarter turns gives the upright one
    func up(_ xs: Int, _ ys: Int) -> (Int, Int) {
        switch k {
        case 1: return (Hs - 1 - ys, xs)
        case 2: return (Ws - 1 - xs, Hs - 1 - ys)
        case 3: return (ys, Ws - 1 - xs)
        default: return (xs, ys)
        }
    }
    var bgra = [UInt8](repeating: 0, count: Ws * Hs * 4)
    for ys in 0..<Hs { for xs in 0..<Ws {
        let (xu, yu) = up(xs, ys)
        let s = (yu * img.w + xu) * 4, d = (ys * Ws + xs) * 4
        bgra[d] = img.px[s]; bgra[d + 1] = img.px[s + 1]; bgra[d + 2] = img.px[s + 2]; bgra[d + 3] = 255
    } }
    if !yuv {
        let pb = makeBuffer(Ws, Hs, kCVPixelFormatType_32BGRA)
        CVPixelBufferLockBaseAddress(pb, [])
        let base = CVPixelBufferGetBaseAddress(pb)!.assumingMemoryBound(to: UInt8.self), bpr = CVPixelBufferGetBytesPerRow(pb)
        bgra.withUnsafeBufferPointer { p in for y in 0..<Hs { memcpy(base + y * bpr, p.baseAddress! + y * Ws * 4, Ws * 4) } }
        CVPixelBufferUnlockBaseAddress(pb, [])
        return pb
    }
    let pb = makeBuffer(Ws, Hs, kCVPixelFormatType_420YpCbCr8BiPlanarFullRange)
    CVBufferSetAttachment(pb, kCVImageBufferYCbCrMatrixKey, kCVImageBufferYCbCrMatrix_ITU_R_601_4, .shouldPropagate)
    CVPixelBufferLockBaseAddress(pb, [])
    let yb = CVPixelBufferGetBaseAddressOfPlane(pb, 0)!.assumingMemoryBound(to: UInt8.self), ybpr = CVPixelBufferGetBytesPerRowOfPlane(pb, 0)
    let cb = CVPixelBufferGetBaseAddressOfPlane(pb, 1)!.assumingMemoryBound(to: UInt8.self), cbpr = CVPixelBufferGetBytesPerRowOfPlane(pb, 1)
    func rgb(_ x: Int, _ y: Int) -> (Double, Double, Double) {
        let i = (y * Ws + x) * 4
        return (Double(bgra[i + 2]), Double(bgra[i + 1]), Double(bgra[i]))
    }
    func u8(_ v: Double) -> UInt8 { UInt8(max(0, min(255, v.rounded()))) }
    for y in 0..<Hs { for x in 0..<Ws {
        let (r, g, b) = rgb(x, y)
        yb[y * ybpr + x] = u8(0.299 * r + 0.587 * g + 0.114 * b)
    } }
    for y in 0..<(Hs / 2) { for x in 0..<(Ws / 2) {
        var cbs = 0.0, crs = 0.0
        for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            let (r, g, b) = rgb(2 * x + dx, 2 * y + dy)
            let yy = 0.299 * r + 0.587 * g + 0.114 * b
            cbs += (b - yy) / 1.772 + 128
            crs += (r - yy) / 1.402 + 128
        }
        cb[y * cbpr + 2 * x] = u8(cbs / 4)
        cb[y * cbpr + 2 * x + 1] = u8(crs / 4)
    } }
    CVPixelBufferUnlockBaseAddress(pb, [])
    return pb
}

func checkPipeline() {
    guard let framesArg = arg("--frames") else { fail("pipeline: --frames <dir> needed") }
    let frames = path(framesArg)
    let ref = frames.appendingPathComponent("python.json")
    let out = arg("--out").map(path) ?? frames.appendingPathComponent("swift.json")
    let detEvery = Int(arg("--det-every") ?? "1")!
    let k = (Int((Double(arg("--rotate") ?? "0")! / 90).rounded()) % 4 + 4) % 4
    let yuv = flag("--yuv")
    guard let data = try? Data(contentsOf: ref), let meta = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
          let list = meta["frames"] as? [[String: Any]], let Kref = meta["K"] as? [[Double]] else { fail("cannot read \(ref.path)") }

    var opts = FAHOptions.pythonImageMode
    opts.detectorInterval = detEvery
    let tl = Date()
    let pipe: FAHPipeline
    do { pipe = try FAHPipeline(modelDirectory: models, options: opts, poseBackend: backend) } catch { fail("\(error)") }
    print(String(format: "loaded in %.2f s; pose network on %@", Date().timeIntervalSince(tl),
                 pipe.usesTensorOps ? "tensor ops" : "simdgroup matrices"))

    let fu = Kref[0][0], fv = Kref[1][1], cxu = Kref[0][2], cyu = Kref[1][2]
    /// The intrinsics turned with the image: the upright principal point in sensor pixels.
    func intrinsics(_ pb: CVPixelBuffer) -> simd_float3x3 {
        let Ws = Double(CVPixelBufferGetWidth(pb)), Hs = Double(CVPixelBufferGetHeight(pb))
        let (cxs, cys): (Double, Double) = switch k {
            case 1: (cyu, Hs - 1 - cxu)
            case 2: (Ws - 1 - cxu, Hs - 1 - cyu)
            case 3: (Ws - 1 - cyu, cxu)
            default: (cxu, cyu)
        }
        let (fxs, fys) = k % 2 == 1 ? (fv, fu) : (fu, fv)
        return simd_float3x3(columns: (SIMD3(Float(fxs), 0, 0), SIMD3(0, Float(fys), 0), SIMD3(Float(cxs), Float(cys), 1)))
    }
    func buffer(_ idx: Int) -> CVPixelBuffer {
        sensorBuffer(loadBGRA(frames.appendingPathComponent(String(format: "frame_%05d.png", idx))), k: k, yuv: yuv)
    }
    let rotation = FAHRotation(quarterTurns: k)
    var results: [[String: Any]] = []
    for f in list {
        let idx = f["index"] as! Int
        let pb = buffer(idx)
        let hands: [FAHHand]
        do { hands = try pipe.process(pixelBuffer: pb, intrinsics: intrinsics(pb), rotation: rotation) } catch { fail("frame \(idx): \(error)") }
        func m3(_ m: simd_float3x3) -> [[Float]] { (0..<3).map { r in (0..<3).map { c in m[c][r] } } }
        results.append(["index": idx, "hands": hands.map { h -> [String: Any] in [
            "is_right": h.isRight, "score": h.score, "presence": h.presence, "side_score": h.sideScore,
            "bbox": [h.bbox.minX, h.bbox.minY, h.bbox.maxX, h.bbox.maxY],
            "joints2d": h.joints2d.map { [$0.x, $0.y] }, "joints3d": h.joints3d.map { [$0.x, $0.y, $0.z] },
            "verts3d_dense": h.denseVertices.map { [$0.x, $0.y, $0.z] },
            "root_rotation": m3(h.rootRotation), "local_rotations": h.localRotations.map(m3), "scale": h.handScale,
        ] }])
    }
    print("\(results.count) frames -> \(out.path)")

    // Timing. The loop above spends most of its time loading PNGs, so the GPU idles (and clocks down) between frames.
    // Here the pipeline runs on frames already in memory, back to back as from a fast camera: with the detector on
    // every frame, and tracking with the default options (detector every 10th frame).
    let nTiming = min(Int(arg("--timing-frames") ?? "60")!, list.count)
    let bufs = list.prefix(nTiming).map { buffer($0["index"] as! Int) }
    var timing: [String: [String: Double]] = [:]
    for (name, o) in [("detector_every_frame", FAHOptions.pythonImageMode), ("tracking", FAHOptions())] where nTiming > 0 {
        pipe.options = o
        pipe.reset()
        var ts: [FAHTimings] = []
        for rep in 0..<6 {
            for pb in bufs {
                do { _ = try pipe.process(pixelBuffer: pb, intrinsics: intrinsics(pb), rotation: rotation) } catch { fail("\(error)") }
                if rep > 0 { ts.append(pipe.lastTimings) }  // the first pass warms up
            }
        }
        let det = ts.filter(\.ranDetector)
        timing[name] = [
            "total": median(ts.map(\.total)), "detector": median(det.map { $0.detectorPrep + $0.detector + $0.detectorDecode }),
            "crop": median(ts.map(\.crop)), "pose": median(ts.map(\.pose)), "decode": median(ts.map { $0.decode + $0.procrustes }),
            "crops": median(ts.map { Double($0.poseCrops) }), "detector_frames": Double(det.count), "frames": Double(ts.count),
        ]
        let t = timing[name]!
        print(String(format: "%@ (%d frames): median total %.2f ms | detector %.2f (on %.0f%% of frames), crop %.2f, pose %.2f (%.0f crops), decode %.2f",
                     name, ts.count, t["total"]!, t["detector"]!, 100 * Double(det.count) / Double(max(ts.count, 1)), t["crop"]!, t["pose"]!,
                     t["crops"]!, t["decode"]!))
    }
    let doc: [String: Any] = ["rotation_deg": k * 90, "yuv": yuv, "det_every": detEvery, "tensor_ops": pipe.usesTensorOps,
                              "frames": results, "timings": timing]
    do { try JSONSerialization.data(withJSONObject: doc).write(to: out) } catch { fail("\(error)") }
}

// MARK: - speed

func checkSpeed() {
    guard let videoArg = arg("--video") else { fail("speed: --video <file> needed") }
    let n = Int(arg("--frames") ?? "120")!, hfov = Double(arg("--hfov") ?? "70")!
    let asset = AVURLAsset(url: path(videoArg))
    guard let track = asset.tracks(withMediaType: .video).first, let reader = try? AVAssetReader(asset: asset) else { fail("cannot read \(videoArg)") }
    let output = AVAssetReaderTrackOutput(track: track, outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                                                                         kCVPixelBufferIOSurfacePropertiesKey as String: [:]])
    reader.add(output)
    reader.startReading()
    var bufs: [CVPixelBuffer] = []
    while bufs.count < n, let sb = output.copyNextSampleBuffer() { if let pb = CMSampleBufferGetImageBuffer(sb) { bufs.append(pb) } }
    guard let first = bufs.first else { fail("no frames in \(videoArg)") }
    let W = CVPixelBufferGetWidth(first), H = CVPixelBufferGetHeight(first)
    let fx = Float(Double(W) / 2 / tan(hfov * .pi / 360))
    let K = simd_float3x3(columns: (SIMD3(fx, 0, 0), SIMD3(0, fx, 0), SIMD3(Float(W) / 2, Float(H) / 2, 1)))
    let pipe: FAHPipeline
    do { pipe = try FAHPipeline(modelDirectory: models, poseBackend: backend) } catch { fail("\(error)") }
    print("\(bufs.count) frames \(W)x\(H); pose network on \(pipe.usesTensorOps ? "tensor ops" : "simdgroup matrices"); tracking (detector every 10th frame)")
    for (mesh, contact) in [(false, false), (false, true), (true, false), (true, true)] where contact == false || pipe.usesContact {
        var o = FAHOptions()
        o.mesh = mesh
        o.contact = contact
        pipe.options = o
        pipe.reset()
        var ts: [FAHTimings] = []
        for rep in 0..<4 {
            for pb in bufs {
                do { _ = try pipe.process(pixelBuffer: pb, intrinsics: K) } catch { fail("\(error)") }
                if rep > 0 { ts.append(pipe.lastTimings) }  // the first pass warms up
            }
        }
        let total = median(ts.map(\.total))
        print(String(format: "  mesh %@ contact %@: %.2f ms per frame (%.0f fps; p90 %.2f) | pose %.2f, decode %.2f, mesh %.2f, detector frames %.1f",
                     mesh ? "on " : "off", contact ? "on " : "off", total, 1000 / total, percentile(ts.map(\.total), 90),
                     median(ts.map(\.pose)), median(ts.map { $0.decode + $0.procrustes }), median(ts.map(\.mesh)),
                     median(ts.filter(\.ranDetector).map(\.total))))
    }
}

// MARK: - pose network vs PyTorch fp32

func checkPose() {
    guard let refArg = arg("--ref") else { fail("pose: --ref <dir> needed") }
    let refDir = path(refArg)
    guard let meta = try? JSONSerialization.jsonObject(with: Data(contentsOf: refDir.appendingPathComponent("ref.json"))) as? [String: Any],
          let tensors = meta["tensors"] as? [String: [String: Any]], let n = meta["n"] as? Int,
          let raw = try? Data(contentsOf: refDir.appendingPathComponent("ref.bin")) else { fail("cannot read \(refDir.path)/ref.json, ref.bin") }
    func tensor(_ k: String) -> [Float] {
        let t = tensors[k]!, off = t["offset"] as! Int, c = (t["shape"] as! [Int]).reduce(1, *)
        return raw.withUnsafeBytes { Array(UnsafeBufferPointer(start: $0.baseAddress!.advanced(by: off).assumingMemoryBound(to: Float.self), count: c)) }
    }
    var net: FAHPoseNetwork
    do { net = try FAHPoseNetwork(modelDirectory: models, maxBatch: 2, backend: backend) } catch { fail("\(error)") }
    print("pose network on \(net.device.name), \(net.usesTensorOps ? "tensor ops" : "simdgroup matrices")")
    let img = tensor("img"), ray = tensor("ray"), cond = tensor("cond")
    let S = FAHPoseNetwork.inputSize, P = FAHPoseNetwork.points
    func predict(_ i: Int, _ B: Int) throws -> (FAHPoseOutputs, gpuSeconds: Double) {
        try img.withUnsafeBufferPointer { im in try ray.withUnsafeBufferPointer { r in try cond.withUnsafeBufferPointer { c in
            try net.predict(img: im.baseAddress! + i * 3 * S * S, ray: r.baseAddress! + 4 * i, cond: c.baseAddress! + 3 * i, batch: B)
        } } }
    }

    // accuracy, one crop at a time
    let ref = (uv: tensor("pt_uv"), xyz: tensor("pt_xyz"), conf: tensor("pt_conf"), vis: tensor("pt_vis"),
               depth: tensor("pt_depth"), verify: tensor("verify_logits"))
    var du: [Double] = [], dx: [Double] = [], dconf = 0.0, dvis = 0.0, ddepth = 0.0, dverify = 0.0
    for i in 0..<n {
        let o: FAHPoseOutputs
        do { o = try predict(i, 1).0 } catch { fail("\(error)") }
        for p in 0..<P {
            let q = i * P + p
            du.append(Double(simd_length(SIMD2(o.uv[2 * p] - ref.uv[2 * q], o.uv[2 * p + 1] - ref.uv[2 * q + 1]))) * Double(S))
            dx.append(Double(simd_length(SIMD3(o.xyz[3 * p] - ref.xyz[3 * q], o.xyz[3 * p + 1] - ref.xyz[3 * q + 1],
                                               o.xyz[3 * p + 2] - ref.xyz[3 * q + 2]))) * 1e3)
            dconf = max(dconf, Double(abs(o.conf[p] - ref.conf[q])))
            dvis = max(dvis, Double(abs(o.vis[p] - ref.vis[q])))
        }
        ddepth = max(ddepth, Double(abs(o.depth[0] - ref.depth[i])))
        dverify = max(dverify, Double(max(abs(o.verify[0] - ref.verify[2 * i]), abs(o.verify[1] - ref.verify[2 * i + 1]))))
    }
    print("\(n) crops vs PyTorch fp32, all \(P) points (uv in 320-px crop pixels):")
    print(String(format: "  uv   px  median %.3f  p95 %.3f  p99 %.3f  max %.3f", median(du), percentile(du, 95), percentile(du, 99), du.max()!))
    print(String(format: "  xyz  mm  median %.3f  p95 %.3f  p99 %.3f  max %.3f", median(dx), percentile(dx, 95), percentile(dx, 99), dx.max()!))
    print(String(format: "  max |diff|: conf %.1e, vis logit %.1e, depth %.1e, verify logits %.1e", dconf, dvis, ddepth, dverify))
    if net.usesContact, tensors["contact"] != nil {  // the logits, and FAH's contact decision (10 mm, logit >= 0)
        let rc = tensor("contact"), rh = tensor("contact_hand"), K = net.contactDistances.count, mi = net.contactMain
        var dc: [Double] = [], dh = 0.0, agree = 0, hagree = 0
        for i in 0..<n {
            let o: FAHPoseOutputs
            do { o = try predict(i, 1).0 } catch { fail("\(error)") }
            for j in 0..<(400 * K) { dc.append(Double(abs(o.contact[j] - rc[i * 400 * K + j]))) }
            for p in 0..<400 { agree += (o.contact[p * K + mi] >= 0) == (rc[(i * 400 + p) * K + mi] >= 0) ? 1 : 0 }
            for k in 0..<K { dh = max(dh, Double(abs(o.contactHand[k] - rh[i * K + k]))) }
            hagree += (o.contactHand[mi] >= 0) == (rh[i * K + mi] >= 0) ? 1 : 0
        }
        print(String(format: "  contact logits: point median %.4f p99 %.4f max %.4f, hand max %.4f; contact at 10 mm agrees on "
                     + "%d / %d points, %d / %d hands", median(dc), percentile(dc, 99), dc.max()!, dh, agree, n * 400, hagree, n))
    }

    // speed: wall time includes the input copy, encoding, submission and the output read-back
    let iters = Int(arg("--iters") ?? "200")!
    func bench(_ label: String) {
    for B in [1, 2] where B <= n {
        var wall: [Double] = [], gpu: [Double] = []
        for it in 0..<(iters + 20) {
            let t = DispatchTime.now().uptimeNanoseconds
            let g: Double
            do { g = try predict(0, B).gpuSeconds } catch { fail("\(error)") }
            if it >= 20 {  // warm-up excluded
                wall.append(Double(DispatchTime.now().uptimeNanoseconds - t) / 1e6)
                gpu.append(g * 1e3)
            }
        }
        print(String(format: "%@batch %d: wall median %.2f ms (p10 %.2f, p90 %.2f), GPU median %.2f ms, %d runs", label, B,
                     median(wall), percentile(wall, 10), percentile(wall, 90), median(gpu), iters))
    }
    }
    bench(net.usesContact ? "with contact, " : "")
    if net.usesContact {  // the same without the contact head
        do { net = try FAHPoseNetwork(modelDirectory: models, maxBatch: 2, backend: backend, contact: false) } catch { fail("\(error)") }
        bench("without contact, ")
    }
}

switch CommandLine.arguments.dropFirst().first {
case "pipeline": checkPipeline()
case "pose": checkPose()
case "speed": checkSpeed()
default: fail("fah-check pipeline | pose (see the comment at the top of Sources/fah-check/main.swift)")
}
