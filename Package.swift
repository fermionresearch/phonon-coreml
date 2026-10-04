// swift-tools-version:6.0
import PackageDescription

let package = Package(
    name: "phonon-coreml",
    platforms: [.macOS(.v15), .iOS(.v18)],
    products: [
        .library(name: "PhononCoreML", targets: ["PhononCoreML"]),
        .executable(name: "phonon-coreml-cli", targets: ["phonon-coreml-cli"]),
        .executable(name: "phonon-coreml-example", targets: ["phonon-coreml-example"]),
    ],
    targets: [
        .binaryTarget(name: "PhononTDT", path: "Binaries/PhononTDT.xcframework"),
        .target(name: "PhononCoreML", dependencies: ["PhononTDT"], path: "Sources/PhononCoreML"),
        .executableTarget(name: "phonon-coreml-cli", dependencies: ["PhononCoreML"], path: "Sources/phonon-coreml-cli"),
        .executableTarget(name: "phonon-coreml-example", dependencies: ["PhononCoreML"], path: "Sources/phonon-coreml-example"),
        .testTarget(name: "PhononCoreMLTests", dependencies: ["PhononCoreML"], path: "Tests/PhononCoreMLTests"),
    ]
)
