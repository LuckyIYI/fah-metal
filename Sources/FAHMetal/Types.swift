import CoreGraphics
import Foundation
import simd

/// How to rotate the input image (as displayed: x right, y down) clockwise, in degrees, to make the hands upright.
///
/// FAH is trained on upright hands, so the detector and the crops see the image rotated by this angle. Every result
/// comes back in the original, unrotated buffer's pixels and camera frame. Quarter turns are exact pixel permutations;
/// other angles (a continuous roll from gravity, for example) resample the image.
///
/// ARKit's `capturedImage` is in the sensor's landscape orientation: phone held portrait -> `.clockwise90`, portrait
/// upside down -> `.counterClockwise90`, landscape left -> `.rotate180`, landscape right -> `.none`.
public struct FAHRotation: Sendable, Equatable {
    public var degrees: Double
    public init(degrees: Double) { self.degrees = degrees }
    public init(quarterTurns k: Int) { self.degrees = Double(((k % 4) + 4) % 4) * 90 }
    public static let none = FAHRotation(degrees: 0)
    public static let clockwise90 = FAHRotation(degrees: 90)
    public static let rotate180 = FAHRotation(degrees: 180)
    public static let counterClockwise90 = FAHRotation(degrees: 270)

    /// (cos, sin) of the angle, exact for quarter turns.
    var cosSin: (Double, Double) {
        var d = degrees.truncatingRemainder(dividingBy: 360)
        if d < 0 { d += 360 }
        switch d {
        case 0: return (1, 0)
        case 90: return (0, 1)
        case 180: return (-1, 0)
        case 270: return (0, -1)
        default:
            let r = d * .pi / 180
            return (cos(r), sin(r))
        }
    }

    /// The camera-frame rotation that goes with the image rotation, about the optical axis:
    /// X_upright = cameraRotation * X_original (x right, y down, z forward).
    public var cameraRotation: simd_double3x3 {
        let (c, s) = cosSin
        return simd_double3x3(rows: [SIMD3(c, -s, 0), SIMD3(s, c, 0), SIMD3(0, 0, 1)])
    }
}

/// One hand of one frame. Everything is in the original image passed to `process`, not the rotated one: 2D in its
/// pixels, 3D in its camera frame (OpenCV convention: x right along the buffer's rows, y down, z forward, metres).
public struct FAHHand: Sendable {
    /// Handedness (FAH's detector class, confirmed or flipped by the pose network's side check).
    public var isRight: Bool
    /// FAH's fused score, sqrt(detector score x presence).
    public var score: Float
    /// The pose network's hand-presence probability.
    public var presence: Float
    /// The pose network's probability that `isRight` is correct.
    public var sideScore: Float
    /// The detector (or tracking) box the hand was cropped from.
    public var bbox: CGRect
    /// 21 joints in OpenPose order (wrist, thumb 1-4, index 1-4, middle 1-4, ring 1-4, pinky 1-4), pixels.
    public var joints2d: [SIMD2<Float>]
    /// The same 21 joints in the camera frame (m).
    public var joints3d: [SIMD3<Float>]
    /// The dense surface: 778 vertices with MANO's topology (FAH's `verts3d_dense`), camera frame (m). Draw it with
    /// `FAHPipeline.meshFaces`.
    public var denseVertices: [SIMD3<Float>]
    /// A clean, watertight MANO hand, for drawing; same 778-vertex topology as `denseVertices`, camera frame (m).
    /// With `FAHOptions.meshFitIterations` > 0: FAH's MANO fit (pose, shape, global rotation and translation by
    /// Gauss-Newton to the dense surface and the joints) placed onto the 2D keypoints. With 0: the template posed
    /// onto the keypoints in closed form (joints at the keypoints, bones turned and stretched to them).
    public var meshVertices: [SIMD3<Float>]

    // MANO pose of the dense surface (per-part Procrustes onto the MANO template, fah.mano_procrustes)
    /// The wrist, = joints3d[0].
    public var wristPosition: SIMD3<Float>
    /// Global rotation of the MANO root (wrist) in the camera frame. Right hands use MANO's right-hand convention, left
    /// hands the mirrored (MANO left) one: R_left = M R M with M = diag(-1, 1, 1).
    public var rootRotation: simd_float3x3
    /// 15 joint rotations relative to their parents, MANO joint order (index 1-3, middle 1-3, pinky 1-3, ring 1-3,
    /// thumb 1-3), in the hand's own convention (left = M L M).
    public var localRotations: [simd_float3x3]
    /// The same as axis-angle vectors (MANO's `hand_pose` layout, 15 x 3).
    public var localAxisAngles: [SIMD3<Float>]
    /// Hand size relative to the MANO mean template (median over the rigid parts).
    public var handScale: Float

    /// Square box around all predicted 2D points: the next frame's tracking crop.
    public var predictedBox: CGRect
    /// True when this hand came from a detector run this frame, false when tracked from the previous frame.
    public var fromDetector: Bool

    // hand-object contact (FAH's contact head; empty / nil when the export has none or `FAHOptions.contact` is off)
    /// Per vertex of `denseVertices`: the probability that it touches an object (within 10 mm). >= 0.5 is contact.
    public var contact: [Float] = []
    /// Per region: palm, thumb, index, middle, ring, pinky; the maximum of `contact` over the region's vertices
    /// (MANO skinning parts).
    public var contactFingers: [Float] = []
    /// The probability that the hand touches an object at all (within 10 mm).
    public var contactScore: Float?
}

/// Wall-clock times of the last `process` call, milliseconds.
public struct FAHTimings: Sendable, CustomStringConvertible {
    /// Detector letterbox on the GPU.
    public var detectorPrep: Double = 0
    /// Detector (Core ML).
    public var detector: Double = 0
    /// Detector box decoding and NMS.
    public var detectorDecode: Double = 0
    /// GPU time of the crop kernels (all pose passes of the frame).
    public var crop: Double = 0
    /// The pose network: the rest of the crop + pose GPU pass, from commit to completion (all passes).
    public var pose: Double = 0
    /// CPU decoding: 2D / 3D / dense surface, verification, identity NMS.
    public var decode: Double = 0
    /// Per-part Procrustes (MANO rotations).
    public var procrustes: Double = 0
    /// The hand mesh (`FAHHand.meshVertices`): the MANO fit and its 2D alignment, or the closed-form mesh.
    public var mesh: Double = 0
    /// The whole call.
    public var total: Double = 0
    /// Number of crops through the pose network this frame.
    public var poseCrops: Int = 0
    public var ranDetector: Bool = false

    public var description: String {
        String(format: "total %.2f ms (det prep %.2f, det %.2f, det decode %.2f, crop %.2f, pose %.2f [%d crops], decode %.2f, procrustes %.2f, mesh %.2f)%@",
               total, detectorPrep, detector, detectorDecode, crop, pose, poseCrops, decode, procrustes, mesh, ranDetector ? "" : " tracked")
    }
}

/// Pipeline settings. The defaults follow FAH-S's image mode (bench.methods.fah.FAHMethod with FAH-S's config.json),
/// except `detectorInterval`, `redetectBelowPresence` and `maxHands`, which make it a tracker for live video.
public struct FAHOptions: Sendable {
    /// Run the detector every N frames (FAH's DetectionSchedule `det_every`). In between, each hand is cropped from its
    /// previous predicted box. 1: every frame (Python's image mode); 0: only when nothing is tracked. The detector also
    /// runs whenever no hand is tracked.
    public var detectorInterval: Int = 10
    /// Also run the detector on the next frame when a tracked hand's presence drops below this (0: off, as Python).
    public var redetectBelowPresence: Float = 0.8
    /// Detector confidence (det_conf) and NMS IoU.
    public var detectorConfidence: Float = 0.05
    public var detectorIoU: Float = 0.5
    /// Hands below this presence are dropped (verify).
    public var presenceThreshold: Float = 0.5
    /// Boxes whose side score is below this are re-run with the other handedness.
    public var sideThreshold: Float = 0.5
    /// Fused-score threshold (fuse_thresh).
    public var fuseThreshold: Float = 0.25
    /// Identity NMS: one hand per wrist within this x the mean palm length (pair_wrist).
    public var pairWrist: Float = 0.5
    /// Re-crop hands whose joints come within this fraction of a crop edge (recrop_border).
    public var recropBorder: Float = 0.03
    /// Crop padding and minimum crop width (bbox_padding, min_crop).
    public var bboxPadding: Double = 0.9
    public var minCrop: Double = 1.1
    /// At most this many hands are returned (highest score first) and tracked; 0: no limit.
    public var maxHands: Int = 2
    /// Run FAH's hand-object contact head when the export has it (FAHHand.contact / contactFingers / contactScore;
    /// about +3 ms per hand on an M5). Can be switched per frame.
    public var contact: Bool = true
    /// Compute `FAHHand.meshVertices` (empty when off). Can be switched per frame.
    public var mesh: Bool = true
    /// Gauss-Newton iterations of the MANO mesh fit behind `FAHHand.meshVertices` (FAH's `--mesh_fit`; ~2 ms each on an
    /// M5 CPU). The hand's shape carries over between frames, so one per frame converges on a live video (FAH's live
    /// default is one, too). 0: the closed-form mesh.
    public var meshFitIterations: Int = 1
    /// Core ML compute units of the detector (FAH uses all, so the Neural Engine where there is one).
    public var detectorComputeUnits: ComputeUnits = .all

    public enum ComputeUnits: Sendable { case all, cpuAndGPU, cpuAndNeuralEngine, cpuOnly }

    public init() {}

    /// Python's image mode exactly: detector every frame, no presence re-detect, no hand limit.
    public static var pythonImageMode: FAHOptions {
        var o = FAHOptions()
        o.detectorInterval = 1
        o.redetectBelowPresence = 0
        o.maxHands = 0
        return o
    }
}

/// Smoothing hook: called with each frame's hands and the frame time (s); returns the hands to publish.
public protocol FAHSmoother: AnyObject {
    func smooth(_ hands: [FAHHand], timestamp: Double) -> [FAHHand]
    func reset()
}

public enum FAHError: Error, CustomStringConvertible {
    case missingResource(String)
    case badResource(String)
    case metal(String)
    case unsupportedPixelFormat(OSType)
    case coreML(String)

    public var description: String {
        switch self {
        case .missingResource(let s): return "FAHMetal: missing resource \(s)"
        case .badResource(let s): return "FAHMetal: bad resource \(s)"
        case .metal(let s): return "FAHMetal: Metal: \(s)"
        case .unsupportedPixelFormat(let f): return "FAHMetal: unsupported pixel format \(String(format: "0x%08x", f))"
        case .coreML(let s): return "FAHMetal: Core ML: \(s)"
        }
    }
}
