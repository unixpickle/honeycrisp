// swift-tools-version: 5.10
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
  name: "Honeycrisp",
  platforms: [
    .macOS(.v13)
  ],
  products: [
    // Products define the executables and libraries a package produces, making them visible to other packages.
    .library(
      name: "Honeycrisp",
      targets: ["Honeycrisp"]),
    .library(
      name: "MNIST",
      targets: ["MNIST"]),
  ],
  dependencies: [
    .package(url: "https://github.com/1024jp/GzipSwift", "6.0.0"..<"6.1.0"),
    .package(url: "https://github.com/swift-server/async-http-client.git", from: "1.9.0"),
    .package(url: "https://github.com/apple/swift-crypto.git", "1.0.0"..<"4.0.0"),
  ],
  targets: [
    .target(
      name: "Honeycrisp"),
    .testTarget(
      name: "HoneycrispTests",
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
  ]
)
