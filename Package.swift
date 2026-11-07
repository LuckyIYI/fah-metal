// swift-tools-version:6.0
import PackageDescription

let package = Package(
    name: "fah-metal",
    platforms: [.macOS(.v15), .iOS(.v18)],
    products: [
        .library(name: "FAHMetal", targets: ["FAHMetal"]),
        .executable(name: "fah-viewer", targets: ["fah-viewer"]),
        .executable(name: "fah-check", targets: ["fah-check"]),
    ],
    targets: [
        // The shaders are compiled at run time (Shaders/pose.metal is specialised per GPU), so they ship as sources.
        .target(name: "FAHMetal", resources: [.copy("Shaders")]),
        .executableTarget(name: "fah-viewer", dependencies: ["FAHMetal"]),
        .executableTarget(name: "fah-check", dependencies: ["FAHMetal"]),
    ],
    swiftLanguageModes: [.v5]
)
