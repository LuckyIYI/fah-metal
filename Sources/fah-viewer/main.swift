// fah-viewer: live FAH-S hand pose on a camera or a video file.
//
//   swift run -c release fah-viewer [--models <dir>] [--video <file>] [--hfov 70] [--mesh] [--rotate 90]
//                                   [--no-contact] [--snapshot <out.png> [--snapshot-frame 60]]
//
// Keys: m toggles the hand mesh, c the hand-object contact, r turns the image a quarter turn clockwise (for a camera
// held sideways; --rotate sets the initial turn, added to a video file's own orientation). Switching the mesh or contact
// off also stops computing it (the overlay's timings show the difference); --no-contact starts with contact off.
// --snapshot writes the given frame, with its overlay, at the video's resolution and quits.
import AVFoundation
import AppKit
import FAHMetal
import VideoToolbox
import simd

func arg(_ name: String) -> String? {
    let a = CommandLine.arguments
    guard let i = a.firstIndex(of: name), i + 1 < a.count else { return nil }
    return a[i + 1]
}

final class Viewer: NSObject, NSApplicationDelegate {
    let pipeline: FAHPipeline
    let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1100, height: 760), styleMask: [.titled, .closable, .resizable, .miniaturizable],
                          backing: .buffered, defer: false)
    let view: OverlayView
    let cameraMenu = NSPopUpButton()
    let fovSlider = NSSlider(value: 70, minValue: 30, maxValue: 120, target: nil, action: nil)
    let fovLabel = NSTextField(labelWithString: "")
    let meshBox = NSButton(checkboxWithTitle: "Mesh (m)", target: nil, action: nil)
    let contactBox = NSButton(checkboxWithTitle: "Contact (c)", target: nil, action: nil)

    var source: FrameSource?
    var videoURL: URL?
    var snapshot: (url: URL, frame: Int)?

    // read on the processing queue, written on the main thread
    private let lock = NSLock()
    private var _hfov: Double = 70
    private var _quarterTurns = 0
    var hfov: Double { get { lock.withLock { _hfov } } set { lock.withLock { _hfov = newValue } } }
    var quarterTurns: Int { get { lock.withLock { _quarterTurns } } set { lock.withLock { _quarterTurns = newValue } } }

    private var frameCount = 0
    private var lastFrameTime: Double?
    private var fps = 0.0

    init(pipeline: FAHPipeline) {
        self.pipeline = pipeline
        view = OverlayView(renderer: OverlayRenderer(faces: pipeline.meshFaces))
        super.init()
    }

    func applicationDidFinishLaunching(_ note: Notification) {
        window.title = "FAH-S on Metal"
        let bar = NSStackView(views: [cameraMenu, NSTextField(labelWithString: "Horizontal FOV"), fovSlider, fovLabel, meshBox,
                                      contactBox])
        bar.edgeInsets = NSEdgeInsets(top: 6, left: 10, bottom: 6, right: 10)
        fovSlider.widthAnchor.constraint(equalToConstant: 160).isActive = true
        fovSlider.target = self; fovSlider.action = #selector(fovChanged)
        fovSlider.doubleValue = hfov
        meshBox.target = self; meshBox.action = #selector(meshChanged)
        meshBox.state = view.renderer.drawMesh ? .on : .off
        contactBox.target = self; contactBox.action = #selector(contactChanged)
        contactBox.state = view.renderer.drawContact ? .on : .off
        contactBox.isEnabled = pipeline.usesContact
        cameraMenu.target = self; cameraMenu.action = #selector(cameraChanged)
        updateFOVLabel()
        let root = NSStackView(views: [bar, view])
        root.orientation = .vertical
        root.spacing = 0
        root.setHuggingPriority(.defaultHigh, for: .vertical)
        view.setContentHuggingPriority(.defaultLow, for: .vertical)
        window.contentView = root
        window.center()
        window.makeKeyAndOrderFront(nil)
        NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] e in self?.key(e) == true ? nil : e }
        NSApp.activate(ignoringOtherApps: true)

        if let url = videoURL {
            cameraMenu.isHidden = true
            Task { @MainActor in
                do {
                    let v = try await VideoSource(url: url)
                    quarterTurns = (quarterTurns + v.quarterTurns) % 4
                    start(v)
                } catch { fatal("\(error)") }
            }
        } else {
            let devices = CameraSource.devices()
            if devices.isEmpty { fatal("no camera found (use --video <file>)") }
            cameraMenu.addItems(withTitles: devices.map(\.localizedName))
            cameraChanged()
        }
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ app: NSApplication) -> Bool { true }

    func fatal(_ s: String) -> Never {
        FileHandle.standardError.write((s + "\n").data(using: .utf8)!)
        exit(1)
    }

    // MARK: controls

    @objc func cameraChanged() {
        let devices = CameraSource.devices()
        guard cameraMenu.indexOfSelectedItem >= 0, cameraMenu.indexOfSelectedItem < devices.count else { return }
        do { start(try CameraSource(device: devices[cameraMenu.indexOfSelectedItem])) } catch {
            NSAlert(error: error).runModal()
        }
    }

    @objc func fovChanged() {
        hfov = fovSlider.doubleValue.rounded()
        updateFOVLabel()
    }

    @objc func meshChanged() {
        let on = meshBox.state == .on
        view.renderer.drawMesh = on
        var o = pipeline.options
        o.mesh = on
        pipeline.options = o
        view.needsDisplay = true
    }

    @objc func contactChanged() {
        let on = contactBox.state == .on
        view.renderer.drawContact = on
        var o = pipeline.options
        o.contact = on
        pipeline.options = o
        view.needsDisplay = true
    }

    func updateFOVLabel() { fovLabel.stringValue = String(format: "%.0f°", hfov) }

    func key(_ e: NSEvent) -> Bool {
        switch e.charactersIgnoringModifiers {
        case "m":
            meshBox.state = meshBox.state == .on ? .off : .on
            meshChanged()
        case "c" where pipeline.usesContact:
            contactBox.state = contactBox.state == .on ? .off : .on
            contactChanged()
        case "r":
            quarterTurns = (quarterTurns + 1) % 4
            pipeline.reset()
        default: return false
        }
        return true
    }

    static let regions = ["palm", "thumb", "index", "middle", "ring", "pinky"]

    /// "left hand: contact 0.87 · thumb, index" (the regions with contact >= 0.5)
    static func contactLine(_ h: FAHHand) -> String? {
        guard let s = h.contactScore else { return nil }
        let touching = zip(regions, h.contactFingers).filter { $0.1 >= 0.5 }.map(\.0)
        return String(format: "%@ hand: contact %.2f", h.isRight ? "right" : "left", s)
            + (touching.isEmpty ? "" : " · " + touching.joined(separator: ", "))
    }

    // MARK: frames

    func start(_ s: FrameSource) {
        source?.stop()
        pipeline.reset()
        source = s
        s.start { [weak self] f in self?.process(f) }
    }

    /// On the source's queue: run the pipeline, then hand the scene to the main thread.
    func process(_ f: Frame) {
        let W = CVPixelBufferGetWidth(f.pixelBuffer), H = CVPixelBufferGetHeight(f.pixelBuffer)
        let k = quarterTurns
        let K: simd_float3x3
        let intrinsicsNote: String
        if let reported = f.intrinsics {
            K = reported
            intrinsicsNote = String(format: "reported intrinsics, fx %.0f px", reported[0][0])
        } else {
            // a pinhole camera with the set horizontal FOV (horizontal as displayed) and its centre in the middle
            let across = Float(k % 2 == 1 ? H : W)
            let fx = across / 2 / tan(Float(hfov) * .pi / 360)
            K = simd_float3x3(columns: (SIMD3(fx, 0, 0), SIMD3(0, fx, 0), SIMD3(Float(W) / 2, Float(H) / 2, 1)))
            intrinsicsNote = String(format: "FOV %.0f° (fx %.0f px)", hfov, fx)
        }
        let hands: [FAHHand]
        do { hands = try pipeline.process(pixelBuffer: f.pixelBuffer, intrinsics: K, rotation: FAHRotation(quarterTurns: k), timestamp: f.time) } catch {
            print("process: \(error)")
            return
        }
        let tm = pipeline.lastTimings
        var image: CGImage?
        VTCreateCGImageFromCVPixelBuffer(f.pixelBuffer, options: nil, imageOut: &image)
        guard let image else { return }
        let now = CACurrentMediaTime()
        DispatchQueue.main.async { [self] in
            if let last = lastFrameTime { fps = fps == 0 ? 1 / (now - last) : 0.9 * fps + 0.1 / (now - last) }
            lastFrameTime = now
            frameCount += 1
            let det = tm.detectorPrep + tm.detector + tm.detectorDecode
            view.scene = Scene(image: image, hands: hands, K: K, quarterTurns: k, hud: [
                String(format: "%.1f fps, %d×%d, %@", fps, W, H, intrinsicsNote),
                String(format: "pipeline %.1f ms (%.0f fps alone): detector %@, crop %.2f, pose%@ %.2f (%d crops), decode %.2f, mesh %@",
                       tm.total, 1000 / max(tm.total, 0.1), tm.ranDetector ? String(format: "%.1f", det) : "–", tm.crop,
                       pipeline.options.contact && pipeline.usesContact ? "+contact" : "", tm.pose, tm.poseCrops,
                       tm.decode + tm.procrustes, pipeline.options.mesh ? String(format: "%.2f", tm.mesh) : "off"),
                "\(hands.count) hand\(hands.count == 1 ? "" : "s") · pose network on \(pipeline.usesTensorOps ? "tensor ops" : "simdgroup matrices")"
                    + (pipeline.options.contact && pipeline.usesContact ? " + contact" : ""),
            ] + hands.compactMap(Self.contactLine))
            if let s = snapshot, frameCount == s.frame {
                view.display()
                do { try view.snapshotPNG()?.write(to: s.url) } catch { fatal("\(error)") }
                print("snapshot -> \(s.url.path)")
                NSApp.terminate(nil)
            }
        }
    }
}

// MARK: - main

let modelDir = URL(fileURLWithPath: ((arg("--models") ?? "models") as NSString).expandingTildeInPath)
let pipeline: FAHPipeline
var options = FAHOptions()
options.contact = !CommandLine.arguments.contains("--no-contact")
options.mesh = CommandLine.arguments.contains("--mesh")
do { pipeline = try FAHPipeline(modelDirectory: modelDir, options: options) } catch {
    FileHandle.standardError.write("\(error)\nRun tools/export.py first, or pass --models <dir>.\n".data(using: .utf8)!)
    exit(1)
}
let viewer = Viewer(pipeline: pipeline)
viewer.videoURL = arg("--video").map { URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath) }
if let h = arg("--hfov").flatMap(Double.init) { viewer.hfov = h }
if let r = arg("--rotate").flatMap(Double.init) { viewer.quarterTurns = (Int((r / 90).rounded()) % 4 + 4) % 4 }
viewer.view.renderer.drawMesh = options.mesh
viewer.view.renderer.drawContact = options.contact
if let s = arg("--snapshot") { viewer.snapshot = (URL(fileURLWithPath: (s as NSString).expandingTildeInPath), Int(arg("--snapshot-frame") ?? "60")!) }

let app = NSApplication.shared
app.setActivationPolicy(.regular)
app.delegate = viewer
app.run()
