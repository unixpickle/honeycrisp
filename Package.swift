// swift-tools-version: 5.10
// The swift-tools-version declares the minimum version of Swift required to build this package.

import CompilerPluginSupport
import PackageDescription

let package = Package(
  name: "Honeycrisp",
  platforms: [
    .macOS(.v13)
  ],
  products: [
    .library(
      name: "HCBacktrace",
      targets: ["HCBacktrace"]),
    .library(
      name: "Honeycrisp",
      targets: ["Honeycrisp"]),
    .library(
      name: "MNIST",
      targets: ["MNIST"]),
  ],
  dependencies: [
    .package(
      url: "https://github.com/swiftlang/swift-syntax.git", "509.0.0"..<"601.0.0-prerelease"),
    .package(url: "https://github.com/1024jp/GzipSwift", "6.0.0"..<"6.1.0"),
    .package(url: "https://github.com/swift-server/async-http-client.git", from: "1.9.0"),
    .package(url: "https://github.com/apple/swift-crypto.git", "1.0.0"..<"4.0.0"),
    .package(url: "https://github.com/unixpickle/coreml-builder.git", "0.1.0"..<"0.2.0"),
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
      ]),
    .testTarget(
      name: "HoneycrispTests",
      dependencies: ["Honeycrisp", .product(name: "Gzip", package: "GzipSwift")],
      resources: [
        .process("Resources")
      ]),
    .executableTarget(
      name: "HCMatrixBench",
      dependencies: ["Honeycrisp"]),
    .target(
      name: "MNIST",
      dependencies: [
        .product(name: "AsyncHTTPClient", package: "async-http-client"),
        .product(name: "Gzip", package: "GzipSwift"),
        .product(name: "Crypto", package: "swift-crypto"),
      ]),
    .executableTarget(
      name: "MNISTExample",
      dependencies: ["MNIST", "Honeycrisp"]),
    .executableTarget(
      name: "MNISTGenExample",
      dependencies: ["MNIST", "Honeycrisp"]),
    .executableTarget(
      name: "Text2Image",
      dependencies: ["Honeycrisp"]),
  ]
)
