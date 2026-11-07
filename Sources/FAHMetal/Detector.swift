import CoreML
import Foundation
import Metal

extension FAHOptions.ComputeUnits {
    var ml: MLComputeUnits {
        switch self {
        case .all: return .all
        case .cpuAndGPU: return .cpuAndGPU
        case .cpuAndNeuralEngine: return .cpuAndNeuralEngine
        case .cpuOnly: return .cpuOnly
        }
    }
}

/// A compiled model (.mlmodelc) in the model directory, or the .mlpackage compiled once into Caches/FAHMetal (reused while
/// the package is unchanged; loading a package directly recompiles it on every launch).
func compiledModelURL(directory: URL, name: String) throws -> URL {
    let fm = FileManager.default
    let c = directory.appendingPathComponent(name + ".mlmodelc")
    if fm.fileExists(atPath: c.path) { return c }
    let pkg = directory.appendingPathComponent(name + ".mlpackage")
    guard fm.fileExists(atPath: pkg.path) else {
        throw FAHError.missingResource("\(name).mlmodelc / \(name).mlpackage in \(directory.path)")
    }
    let caches = try fm.url(for: .cachesDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
        .appendingPathComponent("FAHMetal", isDirectory: true)
    try fm.createDirectory(at: caches, withIntermediateDirectories: true)
    let dst = caches.appendingPathComponent(name + ".mlmodelc")
    let pkgDate = newestModification(pkg)
    if fm.fileExists(atPath: dst.path), newestModification(dst) >= pkgDate { return dst }
    let tmp = try MLModel.compileModel(at: pkg)
    try? fm.removeItem(at: dst)
    try fm.moveItem(at: tmp, to: dst)
    return dst
}

private func newestModification(_ url: URL) -> Date {
    var newest = Date.distantPast
    if let e = FileManager.default.enumerator(at: url, includingPropertiesForKeys: [.contentModificationDateKey]) {
        for case let f as URL in e {
            if let d = try? f.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate, d > newest { newest = d }
        }
    }
    return newest
}

/// The hand detector (detector-640: YOLOv8-seg, one function per letterbox shape "s<H>x<W>"; input x (1, 3, H, W) fp16 RGB
/// [0, 1]; outputs boxes (1, 64, A) DFL bin logits, scores (1, 2, A) class logits (0 left, 1 right), mc, proto).
@available(iOS 18.0, macOS 15.0, *)
final class Detector {
    static let imgsz = 640, stride = 32, bins = 16
    let url: URL
    let units: MLComputeUnits
    let shapes: [(h: Int, w: Int)]
    private var models: [String: MLModel] = [:]
    private var inputs: [String: (MTLBuffer, MLMultiArray)] = [:]
    private let device: MTLDevice

    init(url: URL, units: MLComputeUnits, device: MTLDevice) throws {
        self.url = url
        self.units = units
        self.device = device
        // the function names are the built letterbox shapes
        let names = Detector.functionNames(url)
        shapes = names.compactMap { n in
            let p = n.dropFirst().split(separator: "x")
            guard n.hasPrefix("s"), p.count == 2, let h = Int(p[0]), let w = Int(p[1]) else { return nil }
            return (h, w)
        }
        if shapes.isEmpty { throw FAHError.badResource("detector has no s<H>x<W> functions") }
    }

    /// Function names of a multifunction .mlmodelc (from its model.mil).
    static func functionNames(_ url: URL) -> [String] {
        if let d = try? Data(contentsOf: url.appendingPathComponent("model.mil")), let s = String(data: d, encoding: .utf8) {
            var out: [String] = []
            for line in s.split(separator: "\n") {
                let t = line.trimmingCharacters(in: .whitespaces)
                if t.hasPrefix("func s"), let r = t.dropFirst(5).split(separator: "<").first ?? t.dropFirst(5).split(separator: "(").first {
                    let name = String(r).trimmingCharacters(in: .whitespaces)
                    if name.contains("x") { out.append(name) }
                }
            }
            if !out.isEmpty { return out }
        }
        // fallback: the shapes fah-build-coreml builds by default
        return ["s320x640", "s352x640", "s384x640", "s416x640", "s448x640", "s480x640", "s640x320", "s640x384", "s640x448", "s640x480", "s640x640"]
    }

    /// The letterbox geometry of FAH (fah.detector._letterbox + fah.coreml.fit_letterbox) for an upright canvas (w, h).
    struct Letterbox { var r: Double; var nw: Int; var nh: Int; var px: Int; var py: Int; var H: Int; var W: Int }

    func letterbox(width w: Int, height h: Int) throws -> Letterbox {
        let r = min(Double(Detector.imgsz) / Double(h), Double(Detector.imgsz) / Double(w))
        let nw = Int((Double(w) * r).rounded(.toNearestOrEven)), nh = Int((Double(h) * r).rounded(.toNearestOrEven))
        let pw = (Detector.imgsz - nw) % Detector.stride, ph = (Detector.imgsz - nh) % Detector.stride
        var px = pw / 2, py = ph / 2
        let lw = nw + pw, lh = nh + ph
        if shapes.contains(where: { $0.h == lh && $0.w == lw }) { return Letterbox(r: r, nw: nw, nh: nh, px: px, py: py, H: lh, W: lw) }
        let fit = shapes.filter { $0.h >= lh && $0.w >= lw }.min { $0.h * $0.w < $1.h * $1.w }
        guard let f = fit else { throw FAHError.badResource("detector has no letterbox shape \(lh)x\(lw) or larger") }
        px += (f.w - lw) / 2
        py += (f.h - lh) / 2
        return Letterbox(r: r, nw: nw, nh: nh, px: px, py: py, H: f.h, W: f.w)
    }

    func input(_ lb: Letterbox) throws -> (MTLBuffer, MLMultiArray) {
        let key = "s\(lb.H)x\(lb.W)"
        if let i = inputs[key] { return i }
        guard let buf = device.makeBuffer(length: 3 * lb.H * lb.W * 2, options: .storageModeShared) else { throw FAHError.metal("buffer") }
        let arr = try MLMultiArray(dataPointer: buf.contents(), shape: [1, 3, lb.H, lb.W].map { NSNumber(value: $0) }, dataType: .float16,
                                   strides: [3 * lb.H * lb.W, lb.H * lb.W, lb.W, 1].map { NSNumber(value: $0) }, deallocator: nil)
        inputs[key] = (buf, arr)
        return (buf, arr)
    }

    func model(_ lb: Letterbox) throws -> MLModel {
        let key = "s\(lb.H)x\(lb.W)"
        if let m = models[key] { return m }
        let cfg = MLModelConfiguration()
        cfg.computeUnits = units
        cfg.functionName = key
        let m = try MLModel(contentsOf: url, configuration: cfg)
        models[key] = m
        return m
    }

    struct Detection { var box: SIMD4<Double>; var score: Float; var isRight: Bool }

    /// Raw outputs of the letterboxed input already in input(lb).
    func run(_ lb: Letterbox) throws -> (boxes: MLMultiArray, scores: MLMultiArray) {
        let (_, arr) = try input(lb)
        let o = try model(lb).prediction(from: try MLDictionaryFeatureProvider(dictionary: ["x": arr]))
        guard let b = o.featureValue(for: "boxes")?.multiArrayValue, let s = o.featureValue(for: "scores")?.multiArrayValue else {
            throw FAHError.coreML("detector outputs missing")
        }
        return (b, s)
    }

    /// fah.coreml.CoreMLDet + fah.detector._fast_call: sigmoid class scores, anchors above conf, DFL decode, per-class NMS
    /// (torchvision batched_nms), at most 300, boxes back to the canvas (clamped to [0, w] x [0, h]).
    func decode(boxes: MLMultiArray, scores: MLMultiArray, lb: Letterbox, canvasW: Int, canvasH: Int, conf: Float, iou: Float) -> [Detection] {
        // anchors (row-major per level, strides 8, 16, 32)
        var levels: [(n: Int, w: Int, s: Double)] = []
        for s in [8, 16, 32] { levels.append(((lb.H / s) * (lb.W / s), lb.W / s, Double(s))) }
        let A = levels.reduce(0) { $0 + $1.n }
        // sigmoid(x) > conf <=> x > logit(conf): only the candidates are decoded (fah.coreml.CoreMLDet)
        let thr = log(conf / (1 - conf))
        var cand: [(Int, Float, Int)] = []
        let sst = scores.strides.map { $0.intValue }
        func sigmoid(_ x: Float) -> Float { 1 / (1 + exp(-x)) }
        func scan<T: BinaryFloatingPoint>(_ p: UnsafeBufferPointer<T>) {
            let c0 = 0, c1 = sst[1]
            for a in 0..<A {
                let l0 = Float(p[c0 + a * sst[2]]), l1 = Float(p[c1 + a * sst[2]])
                if l0 > thr || l1 > thr {
                    let s0 = sigmoid(l0), s1 = sigmoid(l1)
                    let (best, cls) = s1 > s0 ? (s1, 1) : (s0, 0)
                    if best > conf { cand.append((a, best, cls)) }
                }
            }
        }
        if scores.dataType == .float16 { scores.withUnsafeBufferPointer(ofType: Float16.self) { scan($0) } }
        else { scores.withUnsafeBufferPointer(ofType: Float.self) { scan($0) } }
        if cand.isEmpty { return [] }
        let bst = boxes.strides.map { $0.intValue }
        var logits = [Float](repeating: 0, count: cand.count * 4 * Detector.bins)
        func gather<T: BinaryFloatingPoint>(_ p: UnsafeBufferPointer<T>) {
            for (ci, c) in cand.enumerated() { for ch in 0..<(4 * Detector.bins) { logits[ci * 4 * Detector.bins + ch] = Float(p[ch * bst[1] + c.0 * bst[2]]) } }
        }
        if boxes.dataType == .float16 { boxes.withUnsafeBufferPointer(ofType: Float16.self) { gather($0) } }
        else { boxes.withUnsafeBufferPointer(ofType: Float.self) { gather($0) } }
        var dets: [(box: SIMD4<Double>, score: Float, cls: Int)] = []
        for (ci, (a, score, cls)) in cand.enumerated() {
            var lvl = 0, idx = a
            while idx >= levels[lvl].n { idx -= levels[lvl].n; lvl += 1 }
            let ax = Double(idx % levels[lvl].w) + 0.5, ay = Double(idx / levels[lvl].w) + 0.5, st = levels[lvl].s
            var dist = [Double](repeating: 0, count: 4)
            for side in 0..<4 {  // DFL: softmax over the bins, expected bin
                let o = ci * 4 * Detector.bins + side * Detector.bins
                var m = -Float.infinity
                for k in 0..<Detector.bins { m = max(m, logits[o + k]) }
                var sum: Float = 0, ex: Float = 0
                for k in 0..<Detector.bins {
                    let e = exp(logits[o + k] - m)
                    sum += e
                    ex += e * Float(k)
                }
                dist[side] = Double(ex / sum)
            }
            let x1 = ax - dist[0], y1 = ay - dist[1], x2 = ax + dist[2], y2 = ay + dist[3]
            let cx = (x1 + x2) / 2 * st, cy = (y1 + y2) / 2 * st, w = (x2 - x1) * st, h = (y2 - y1) * st
            dets.append((SIMD4(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), score, cls))
        }
        // batched NMS: per class, by decreasing score; output sorted by score
        let order = dets.indices.sorted { dets[$0].score > dets[$1].score }
        var keep: [Int] = []
        var removed = [Bool](repeating: false, count: dets.count)
        for (oi, i) in order.enumerated() where !removed[i] {
            keep.append(i)
            if keep.count >= 300 { break }
            for j in order[(oi + 1)...] where !removed[j] && dets[j].cls == dets[i].cls {
                if boxIoU(dets[i].box, dets[j].box) > Double(iou) { removed[j] = true }
            }
        }
        return keep.map { i in
            var b = dets[i].box
            b.x = min(max((b.x - Double(lb.px)) / lb.r, 0), Double(canvasW))
            b.z = min(max((b.z - Double(lb.px)) / lb.r, 0), Double(canvasW))
            b.y = min(max((b.y - Double(lb.py)) / lb.r, 0), Double(canvasH))
            b.w = min(max((b.w - Double(lb.py)) / lb.r, 0), Double(canvasH))
            return Detection(box: b, score: dets[i].score, isRight: dets[i].cls == 1)
        }
    }
}

func boxIoU(_ a: SIMD4<Double>, _ b: SIMD4<Double>) -> Double {
    let iw = max(min(a.z, b.z) - max(a.x, b.x), 0), ih = max(min(a.w, b.w) - max(a.y, b.y), 0)
    let inter = iw * ih
    let ua = (a.z - a.x) * (a.w - a.y) + (b.z - b.x) * (b.w - b.y) - inter
    return ua > 0 ? inter / ua : 0
}
