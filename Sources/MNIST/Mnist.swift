import AsyncHTTPClient
import Crypto
import Foundation
import Gzip

public struct MNISTDataset {

  public enum MNISTError: Error {
    case bodyError
    case unexpectedResolution
    case unexpectedDatasetSize
    case unexpectedEOF
    case incorrectHash(String)
  }

  public struct Image {
    public let pixels: [UInt8]
    public let label: Int
  }

  public let train: [Image]
  public let test: [Image]

  public static let downloadURL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
  public static let resources = [
    ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
    ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
    ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
    ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c"),
  ]

  init(fromDir dirPath: String) throws {
    let baseInURL = URL.init(fileURLWithPath: dirPath)
    var mapping = [String: Data]()
    for (filename, hash) in MNISTDataset.resources {
      let path = baseInURL.appendingPathComponent(filename)
      let data = try Data(contentsOf: path)
      let hexDigest = MNISTDataset.checksum(data)
      if hexDigest != hash {
        throw MNISTError.incorrectHash(
          "file \(filename) should have hash \(hash) but got \(hexDigest)")
      }
      let decompressed = try data.gunzipped()
      mapping[filename] = decompressed
    }
    train = try MNISTDataset.decodeDataset(
      intensities: mapping["train-images-idx3-ubyte.gz"]!,
      labels: mapping["train-labels-idx1-ubyte.gz"]!
    )
    test = try MNISTDataset.decodeDataset(
      intensities: mapping["t10k-images-idx3-ubyte.gz"]!,
      labels: mapping["t10k-labels-idx1-ubyte.gz"]!
    )
  }

  public static func download(toDir: String) async throws -> MNISTDataset {
    let baseOutURL = URL.init(fileURLWithPath: toDir).standardizedFileURL

    try FileManager.default.createDirectory(atPath: toDir, withIntermediateDirectories: true)
    for (filename, hash) in resources {
      let path = baseOutURL.appendingPathComponent(filename)
      if FileManager.default.fileExists(atPath: path.path) {
        continue
      }
      let request = HTTPClientRequest(url: "\(downloadURL)\(filename)")
      let response = try await HTTPClient.shared.execute(request, timeout: .seconds(30))
      let body = try await response.body.collect(upTo: 1 << 27)
      guard let data = body.getBytes(at: 0, length: body.readableBytes) else {
        throw MNISTError.bodyError
      }
      let hexDigest = MNISTDataset.checksum(Data(data))
      if hexDigest != hash {
        throw MNISTError.incorrectHash(
          "file \(filename) should have hash \(hash) but got \(hexDigest)")
      }
      try Data(data).write(to: path)
    }
    return try MNISTDataset(fromDir: toDir)
  }

  private static func decodeDataset(intensities: Data, labels: Data) throws -> [Image] {
    let (width, height, pixels) = try decodeIntensities(intensities)
    if width != 28 || height != 28 {
      throw MNISTError.unexpectedResolution
    }
    let labels = try decodeLabels(labels)
    if pixels.count != labels.count {
      throw MNISTError.unexpectedDatasetSize
    }
    return Array(
      zip(pixels, labels).map { (imgData, label) in
        Image(pixels: imgData, label: label)
      })
  }

  private static func decodeIntensities(_ data: Data) throws -> (Int, Int, [[UInt8]]) {
    let reader = SequenceDecoder(buffer: data)
    let _ = try reader.read(4)
    let count = Int(try reader.readUInt32())
    let width = Int(try reader.readUInt32())
    let height = Int(try reader.readUInt32())

    var results = [[UInt8]]()
    for _ in 0..<count {
      let chunk = try reader.read(width * height)
      results.append(Array(chunk))
    }
    return (width, height, results)
  }

  private static func decodeLabels(_ data: Data) throws -> [Int] {
    let reader = SequenceDecoder(buffer: data)
    let _ = try reader.read(4)
    let count = Int(try reader.readUInt32())
    return Array(try reader.read(count).map { Int($0) })
  }

  private static func checksum(_ data: Data) -> String {
    return Crypto.Insecure.MD5.hash(data: data).map { String(format: "%02hhx", $0) }.joined()
  }

  private class SequenceDecoder {
    let buffer: Data
    var offset: Int = 0

    init(buffer: Data) {
      self.buffer = buffer
    }

    func readUInt32() throws -> UInt32 {
      let x = try read(4)
      return (UInt32(x[0]) << 24) | (UInt32(x[1]) << 16) | (UInt32(x[2]) << 8) | UInt32(x[3])
    }

    func read(_ size: Int) throws -> Data {
      if offset + size > buffer.count {
        throw MNISTError.unexpectedEOF
      }
      let result = buffer[offset..<(offset + size)]
      offset += size
      // Copy the data to make indices behave normally.
      return Data(result)
    }
  }

}
