// swift-tools-version: 5.10
// The swift-tools-version declares the minimum version of Swift required to build this package.

import CompilerPluginSupport
import PackageDescription

let package = Package(
  name: "Honeycrisp",
  platforms: [
    .macOS(.v13),
    .iOS(.v16),
  ],
  products: [
    .library(
      name: "HCBacktrace",
      targets: ["HCBacktrace"]),
    .library(
      name: "Honeycrisp",
      targets: ["Honeycrisp"]),
    .library(
      name: "HCTestUtils",
      targets: ["HCTestUtils"]),
  ],
  dependencies: [
    .package(
      url: "https://github.com/swiftlang/swift-syntax.git", from: "600.0.0"),
    .package(url: "https://github.com/1024jp/GzipSwift", "6.0.0"..<"6.1.0"),
    .package(url: "https://github.com/unixpickle/coreml-builder.git", from: "0.2.0"),
    .package(url: "https://github.com/swiftlang/swift-docc-plugin", from: "1.1.0"),
  ],
  targets: [
    .macro(
      name: "HCBacktraceMacros",
      dependencies: [
        .product(name: "SwiftSyntaxMacros", package: "swift-syntax"),
        .product(name: "SwiftCompilerPlugin", package: "swift-syntax"),
      ],
      path: "Sources/HCBacktrace/Macros"),
    .target(
      name: "HCBacktrace",
      dependencies: ["HCBacktraceMacros"],
      path: "Sources/HCBacktrace/Interface"),
    .target(
      name: "Honeycrisp",
      dependencies: [
        .product(name: "CoreMLBuilder", package: "coreml-builder"),
        "HCBacktrace",
      ],
      resources: [
        .process("Resources")
      ],
      cSettings: [
        .define("ACCELERATE_NEW_LAPACK")
      ]),
    .target(
      name: "HCTestUtils",
      dependencies: [
        "Honeycrisp", .product(name: "Gzip", package: "GzipSwift"),
      ],
      resources: [
        .process("Resources")
      ]),
    .testTarget(
      name: "HoneycrispTests",
      dependencies: ["Honeycrisp", "HCTestUtils"]),
    .executableTarget(
      name: "HCMatrixBench",
      dependencies: ["Honeycrisp"]),
  ],
  swiftLanguageVersions: [.v5, .version("6")]
)
