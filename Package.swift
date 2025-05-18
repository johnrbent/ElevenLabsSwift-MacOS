// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "ElevenLabsSDK",
    platforms: [
        .iOS(.v16), .macOS(.v11)
    ],
    products: [
        .library(
            name: "ElevenLabsSDK",
            targets: ["ElevenLabsSDK"]
        ),
    ],
    targets: [
        .target(
            name: "ElevenLabsSDK"
        ),
    ]
)
