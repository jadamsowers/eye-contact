// swift-tools-version:5.10
import PackageDescription

let package = Package(
    name: "EyeContactKit",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "EyeContactCore", targets: ["EyeContactCore"]),
        .executable(name: "eyecontact", targets: ["eyecontact"]),
        .executable(name: "visiondump", targets: ["visiondump"]),
    ],
    targets: [
        .target(name: "EyeContactCore"),
        .executableTarget(name: "eyecontact", dependencies: ["EyeContactCore"]),
        .executableTarget(name: "visiondump", dependencies: ["EyeContactCore"]),
    ]
)
