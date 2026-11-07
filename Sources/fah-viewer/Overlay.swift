import AppKit
import FAHMetal
import simd

/// Everything one displayed frame needs.
struct Scene {
    var image: CGImage
    var hands: [FAHHand]
    var K: simd_float3x3
    var quarterTurns: Int      // clockwise display rotation
    var hud: [String]
}

/// Draws a scene into a y-down CGContext: the frame (rotated by the display rotation, fitted into `rect`), the hands'
/// keypoints and bones, optionally the shaded mesh and the hand-object contact, and the text lines of the HUD.
struct OverlayRenderer {
    let faces: [SIMD3<UInt32>]
    var drawMesh = false
    /// Contact: the mesh turns red where it touches; without the mesh, the touching vertices are drawn as red dots.
    var drawContact = true

    static let rightColor = NSColor(calibratedRed: 1.0, green: 0.55, blue: 0.1, alpha: 1)
    static let leftColor = NSColor(calibratedRed: 0.2, green: 0.65, blue: 1.0, alpha: 1)
    /// OpenPose order: the wrist, then 4 joints per finger from the base to the tip.
    static let bones: [(Int, Int)] = (0..<5).flatMap { (f: Int) -> [(Int, Int)] in
        let b = 1 + 4 * f
        return [(0, b), (b, b + 1), (b + 1, b + 2), (b + 2, b + 3)]
    }

    /// Buffer pixels -> `rect`: the clockwise display rotation, then a uniform scale that fits.
    static func transform(width: Int, height: Int, quarterTurns k: Int, into rect: CGRect) -> CGAffineTransform {
        let (w, h) = k % 2 == 1 ? (CGFloat(height), CGFloat(width)) : (CGFloat(width), CGFloat(height))
        let s = min(rect.width / w, rect.height / h)
        return CGAffineTransform(translationX: -CGFloat(width) / 2, y: -CGFloat(height) / 2)
            .concatenating(CGAffineTransform(rotationAngle: CGFloat(k) * .pi / 2))  // clockwise on screen in y-down
            .concatenating(CGAffineTransform(scaleX: s, y: s))
            .concatenating(CGAffineTransform(translationX: rect.midX, y: rect.midY))
    }

    /// pointSize: one UI point in the context's units (1 on screen; larger for a full-resolution snapshot).
    func draw(_ scene: Scene, in ctx: CGContext, rect: CGRect, pointSize: CGFloat = 1) {
        let t = Self.transform(width: scene.image.width, height: scene.image.height, quarterTurns: scene.quarterTurns, into: rect)
        let scale = sqrt(abs(t.a * t.d - t.b * t.c))
        ctx.saveGState()
        ctx.concatenate(t)
        // CGContext.draw puts the image's first row at the bottom; flip it into the y-down space
        ctx.saveGState()
        ctx.translateBy(x: 0, y: CGFloat(scene.image.height))
        ctx.scaleBy(x: 1, y: -1)
        ctx.interpolationQuality = .medium
        ctx.draw(scene.image, in: CGRect(x: 0, y: 0, width: scene.image.width, height: scene.image.height))
        ctx.restoreGState()
        let px = pointSize / scale  // one UI point in buffer pixels
        if drawMesh { for h in scene.hands { mesh(h, scene.K, ctx) } }
        for h in scene.hands { skeleton(h, ctx, px) }
        if drawContact && !drawMesh { for h in scene.hands { contactDots(h, scene.K, ctx, px) } }
        ctx.restoreGState()
        hud(scene.hud, ctx, rect, pointSize)
    }

    private func skeleton(_ h: FAHHand, _ ctx: CGContext, _ px: CGFloat) {
        let c = h.isRight ? Self.rightColor : Self.leftColor
        let p = h.joints2d.map { CGPoint(x: CGFloat($0.x), y: CGFloat($0.y)) }
        ctx.setLineCap(.round)
        ctx.setLineWidth(3 * px)
        ctx.setStrokeColor(c.cgColor)
        for (a, b) in Self.bones {
            ctx.move(to: p[a])
            ctx.addLine(to: p[b])
        }
        ctx.strokePath()
        let r = 4 * px
        ctx.setFillColor(NSColor.white.cgColor)
        ctx.setStrokeColor(c.cgColor)
        ctx.setLineWidth(1.5 * px)
        for q in p { ctx.addEllipse(in: CGRect(x: q.x - r, y: q.y - r, width: 2 * r, height: 2 * r)) }
        ctx.drawPath(using: .fillStroke)
    }

    static let contactColor = SIMD3<Double>(0.95, 0.12, 0.15)

    /// The touching vertices (contact probability >= 0.5) as dots.
    private func contactDots(_ h: FAHHand, _ K: simd_float3x3, _ ctx: CGContext, _ px: CGFloat) {
        let verts = h.meshVertices.isEmpty ? h.denseVertices : h.meshVertices
        guard h.contact.count == verts.count else { return }
        let fx = K[0][0], fy = K[1][1], cx = K[2][0], cy = K[2][1], r = 2.5 * px
        ctx.setFillColor(red: Self.contactColor.x, green: Self.contactColor.y, blue: Self.contactColor.z, alpha: 0.9)
        for (v, c) in zip(verts, h.contact) where c >= 0.5 {
            let p = CGPoint(x: CGFloat(fx * v.x / v.z + cx), y: CGFloat(fy * v.y / v.z + cy))
            ctx.addEllipse(in: CGRect(x: p.x - r, y: p.y - r, width: 2 * r, height: 2 * r))
        }
        ctx.fillPath()
    }

    /// The hand mesh (the posed MANO template; 778 vertices) projected with the intrinsics, triangles back to front, Lambert-shaded by a light at the
    /// camera; with contact, each triangle blends towards red by its vertices' largest contact probability.
    private func mesh(_ h: FAHHand, _ K: simd_float3x3, _ ctx: CGContext) {
        let v = h.meshVertices.isEmpty ? h.denseVertices : h.meshVertices
        guard v.count == 778 else { return }
        let fx = K[0][0], fy = K[1][1], cx = K[2][0], cy = K[2][1]
        let p = v.map { CGPoint(x: CGFloat(fx * $0.x / $0.z + cx), y: CGFloat(fy * $0.y / $0.z + cy)) }
        let base = (h.isRight ? Self.rightColor : Self.leftColor).usingColorSpace(.deviceRGB)!
        let rgb = SIMD3(Double(base.redComponent), Double(base.greenComponent), Double(base.blueComponent))
        let contact = drawContact && h.contact.count == v.count ? h.contact : nil
        let order = faces.indices.sorted { i, j in
            let a = faces[i], b = faces[j]
            return v[Int(a.x)].z + v[Int(a.y)].z + v[Int(a.z)].z > v[Int(b.x)].z + v[Int(b.y)].z + v[Int(b.z)].z
        }
        for i in order {
            let f = faces[i]
            let a = v[Int(f.x)], b = v[Int(f.y)], c = v[Int(f.z)]
            let n = simd_normalize(simd_cross(b - a, c - a))
            let view = simd_normalize(-(a + b + c))  // towards the camera
            let shade = Double(0.35 + 0.65 * abs(simd_dot(n, view)))
            var col = rgb
            if let contact {
                let t = Double(max(contact[Int(f.x)], contact[Int(f.y)], contact[Int(f.z)]))
                col = rgb + (Self.contactColor - rgb) * t
            }
            ctx.setFillColor(red: col.x * shade, green: col.y * shade, blue: col.z * shade, alpha: 0.75)
            ctx.move(to: p[Int(f.x)])
            ctx.addLine(to: p[Int(f.y)])
            ctx.addLine(to: p[Int(f.z)])
            ctx.closePath()
            ctx.fillPath()
        }
    }

    private func hud(_ lines: [String], _ ctx: CGContext, _ rect: CGRect, _ pt: CGFloat) {
        guard !lines.isEmpty else { return }
        let font = NSFont.monospacedSystemFont(ofSize: 12 * pt, weight: .medium)
        let attrs: [NSAttributedString.Key: Any] = [.font: font, .foregroundColor: NSColor.white]
        let lineH = 16 * pt
        let width = lines.map { ($0 as NSString).size(withAttributes: attrs).width }.max()! + 16 * pt
        let box = CGRect(x: rect.minX + 8 * pt, y: rect.minY + 8 * pt, width: width, height: CGFloat(lines.count) * lineH + 8 * pt)
        ctx.setFillColor(NSColor.black.withAlphaComponent(0.55).cgColor)
        ctx.fill(box)
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = NSGraphicsContext(cgContext: ctx, flipped: true)
        for (i, l) in lines.enumerated() {
            (l as NSString).draw(at: CGPoint(x: box.minX + 8 * pt, y: box.minY + 4 * pt + CGFloat(i) * lineH), withAttributes: attrs)
        }
        NSGraphicsContext.restoreGraphicsState()
    }
}

/// The video view: draws the latest scene on the main thread.
final class OverlayView: NSView {
    var renderer: OverlayRenderer
    var scene: Scene? { didSet { needsDisplay = true } }

    init(renderer: OverlayRenderer) {
        self.renderer = renderer
        super.init(frame: .zero)
    }
    required init?(coder: NSCoder) { fatalError() }

    override var isFlipped: Bool { true }

    override func draw(_ dirty: NSRect) {
        guard let ctx = NSGraphicsContext.current?.cgContext else { return }
        ctx.setFillColor(NSColor.black.cgColor)
        ctx.fill(bounds)
        if let scene { renderer.draw(scene, in: ctx, rect: bounds) }
    }

    /// The scene at the frame's own resolution, as PNG data.
    func snapshotPNG() -> Data? {
        guard let scene else { return nil }
        let k = scene.quarterTurns
        let (w, h) = k % 2 == 1 ? (scene.image.height, scene.image.width) : (scene.image.width, scene.image.height)
        guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0,
                                  space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        ctx.translateBy(x: 0, y: CGFloat(h))  // y down, as in the view
        ctx.scaleBy(x: 1, y: -1)
        renderer.draw(scene, in: ctx, rect: CGRect(x: 0, y: 0, width: w, height: h), pointSize: CGFloat(max(w, h)) / 900)
        guard let img = ctx.makeImage() else { return nil }
        return NSBitmapImageRep(cgImage: img).representation(using: .png, properties: [:])
    }
}
