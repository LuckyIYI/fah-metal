import AVFoundation
import CoreMedia
import simd

/// A frame for the pipeline: a BGRA buffer, the intrinsics the source reports (nil: unknown) and its time (s).
struct Frame {
    let pixelBuffer: CVPixelBuffer
    let intrinsics: simd_float3x3?
    let time: Double
}

protocol FrameSource: AnyObject {
    /// Delivers frames on a background queue, one at a time: the next frame comes after `handler` returns (a camera
    /// drops the frames that arrive meanwhile).
    func start(_ handler: @escaping (Frame) -> Void)
    func stop()
}

private let bgra: [String: Any] = [
    kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
    kCVPixelBufferMetalCompatibilityKey as String: true,
    kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any](),  // the pipeline reads buffers as Metal textures
]

// MARK: - camera

final class CameraSource: NSObject, FrameSource, AVCaptureVideoDataOutputSampleBufferDelegate {
    /// Built-in, USB and Continuity Camera devices.
    static func devices() -> [AVCaptureDevice] {
        AVCaptureDevice.DiscoverySession(deviceTypes: [.builtInWideAngleCamera, .external, .continuityCamera],
                                         mediaType: .video, position: .unspecified).devices
    }

    private let session = AVCaptureSession()
    private let queue = DispatchQueue(label: "camera", qos: .userInteractive)
    private var handler: ((Frame) -> Void)?

    init(device: AVCaptureDevice) throws {
        super.init()
        session.beginConfiguration()
        session.sessionPreset = .high
        let input = try AVCaptureDeviceInput(device: device)
        guard session.canAddInput(input) else { throw NSError(domain: "fah-viewer", code: 1, userInfo: [NSLocalizedDescriptionKey: "cannot use \(device.localizedName)"]) }
        session.addInput(input)
        let output = AVCaptureVideoDataOutput()
        output.videoSettings = bgra
        output.alwaysDiscardsLateVideoFrames = true  // keep latency low: frames that arrive while one is processed are dropped
        output.setSampleBufferDelegate(self, queue: queue)
        session.addOutput(output)
        session.commitConfiguration()
    }

    func start(_ handler: @escaping (Frame) -> Void) {
        self.handler = handler
        queue.async { self.session.startRunning() }
    }

    func stop() {
        queue.sync { handler = nil }
        session.stopRunning()
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sb: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pb = CMSampleBufferGetImageBuffer(sb), let handler else { return }
        // Some devices attach their intrinsics to each buffer; most webcams do not.
        var K: simd_float3x3?
        if let d = CMGetAttachment(sb, key: kCMSampleBufferAttachmentKey_CameraIntrinsicMatrix, attachmentModeOut: nil) as? Data,
           d.count == MemoryLayout<simd_float3x3>.size {
            K = d.withUnsafeBytes { $0.loadUnaligned(as: simd_float3x3.self) }
        }
        handler(Frame(pixelBuffer: pb, intrinsics: K, time: CMSampleBufferGetPresentationTimeStamp(sb).seconds))
    }
}

// MARK: - video file

/// Decodes a video file in a loop at its own frame rate (slower when processing cannot keep up).
final class VideoSource: FrameSource {
    let url: URL
    /// The clockwise rotation the file's display transform asks for, in quarter turns.
    let quarterTurns: Int
    private let asset: AVURLAsset
    private let track: AVAssetTrack
    private var running = false
    private let queue = DispatchQueue(label: "video", qos: .userInteractive)

    init(url: URL) async throws {
        self.url = url
        asset = AVURLAsset(url: url)
        guard let t = try await asset.loadTracks(withMediaType: .video).first else {
            throw NSError(domain: "fah-viewer", code: 2, userInfo: [NSLocalizedDescriptionKey: "no video track in \(url.path)"])
        }
        track = t
        let tf = try await t.load(.preferredTransform)
        quarterTurns = (Int((atan2(Double(tf.b), Double(tf.a)) * 180 / .pi / 90).rounded()) % 4 + 4) % 4
    }

    func start(_ handler: @escaping (Frame) -> Void) {
        running = true
        queue.async { [self] in
            while running {
                guard let reader = try? AVAssetReader(asset: asset) else { return }
                let out = AVAssetReaderTrackOutput(track: track, outputSettings: bgra)
                out.alwaysCopiesSampleData = false
                reader.add(out)
                reader.startReading()
                // pace at the file's frame rate: each frame waits for its interval after the previous one, so a
                // slow frame delays the rest instead of being followed by a burst
                var last: (pts: Double, wall: Double)?
                while running, let sb = out.copyNextSampleBuffer() {
                    guard let pb = CMSampleBufferGetImageBuffer(sb) else { continue }
                    let t = CMSampleBufferGetPresentationTimeStamp(sb).seconds
                    if let last, case let wait = last.wall + (t - last.pts) - CACurrentMediaTime(), wait > 0 {
                        Thread.sleep(forTimeInterval: wait)
                    }
                    last = (t, CACurrentMediaTime())
                    handler(Frame(pixelBuffer: pb, intrinsics: nil, time: CACurrentMediaTime()))
                }
                reader.cancelReading()
            }
        }
    }

    func stop() { running = false }
}
