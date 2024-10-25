import CryptoKit
import Foundation
import Honeycrisp

class CommandTokenize: Command {

  public struct Shard: Codable {
    public struct Record: Codable {
      public struct Metadata: Codable {
        let url: String
        let width: Int
        let height: Int
        let caption: String
      }

      public let id: String
      public let metadata: Metadata
      public let tokens: [UInt16]
    }

    public var records: [Record]
  }

  let batchSize = 8

  let imageDir: String
  let vqPath: String
  let outputDir: String
  let metaDir: URL
  let vqvae: VQVAE

  init(_ args: [String]) throws {
    if args.count != 3 {
      print("Usage: ... tokenize <image_dir> <vq_path> <output_dir>")
      throw ArgumentError.invalidArgs
    }
    imageDir = args[0]
    vqPath = args[1]
    outputDir = args[2]

    metaDir = URL(filePath: imageDir).deletingLastPathComponent().appending(
      component: "success")

    if !FileManager.default.fileExists(atPath: outputDir) {
      try FileManager.default.createDirectory(
        at: URL(filePath: outputDir), withIntermediateDirectories: false)
    }

    vqvae = VQVAE(channels: 4, vocab: 16384, latentChannels: 4, downsamples: 4)
  }

  override public func run() async throws {
    try loadVQ()
    let shards = try listShards()
    startFLOPCounter()
    for (shard, pathsAndMeta) in shards {
      try await tokenizeShard(shard: shard, pathsAndMeta: pathsAndMeta)
    }
  }

  func loadVQ() throws {
    print("loading VQVAE from checkpoint: \(vqPath) ...")
    let decoder = PropertyListDecoder()
    let state = try decoder.decode(
      CommandVQVAE.State.self, from: try Data(contentsOf: URL(fileURLWithPath: vqPath)))
    try vqvae.loadState(state.model)
  }

  func listShards() throws -> [Int: [(URL, URL)]] {
    print("listing image filenames...")
    var shards: [Int: [(URL, URL)]] = [:]
    let fileManager = FileManager.default
    let directoryURL = URL(filePath: imageDir)
    let contents = try fileManager.contentsOfDirectory(
      at: directoryURL, includingPropertiesForKeys: nil, options: [])
    var count = 0
    for imageURL in contents {
      let metaURL = metaDir.appending(component: imageURL.lastPathComponent + ".json")
      let shard = shardForImageName(imageURL.lastPathComponent)
      if shards[shard] == nil {
        shards[shard] = []
      }
      shards[shard]!.append((imageURL, metaURL))
      count += 1
    }
    print(" => listed total of \(count) images")
    return shards
  }

  func tokenizeShard(shard: Int, pathsAndMeta: [(URL, URL)]) async throws {
    let fileManager = FileManager.default

    print("working on shard \(shard) ...")
    print(" => found \(pathsAndMeta.count) images in this shard")

    let shardURL = URL(filePath: outputDir).appending(component: "\(shard).plist")
    var shard: Shard = Shard(records: [])
    if fileManager.fileExists(atPath: shardURL.path()) {
      shard = try PropertyListDecoder().decode(Shard.self, from: try Data(contentsOf: shardURL))
      print(" => shard exists with \(shard.records.count) records")
    }

    let existingIDs = Set(shard.records.map { $0.id })
    var numFailed = 0
    var numSucceeded = 0

    var currentBatch: [Tensor] = []
    var currentIDs: [String] = []
    var currentMeta: [Shard.Record.Metadata] = []

    func flushBatch() async throws {
      let vqs = Tensor.withGrad(enabled: false) {
        vqvae.bottleneck(vqvae.encoder(Tensor(stack: currentBatch).move(axis: -1, to: 1))).codes
      }
      for (i, (id, meta)) in zip(currentIDs, currentMeta).enumerated() {
        let tokens = try await vqs[i].ints().map { UInt16($0) }
        shard.records.append(
          Shard.Record(id: id, metadata: meta, tokens: tokens))
      }
      currentBatch = []
      currentIDs = []
      currentMeta = []
    }

    for (imagePath, metaPath) in pathsAndMeta {
      let imageID = imagePath.lastPathComponent
      if existingIDs.contains(imageID) {
        continue
      }
      do {
        let metadata = try JSONDecoder().decode(
          Shard.Record.Metadata.self, from: try Data(contentsOf: metaPath))
        guard let image = loadImage(path: imagePath.path(), imageSize: 256) else {
          numFailed += 1
          continue
        }
        currentBatch.append(image)
        currentIDs.append(imageID)
        currentMeta.append(metadata)
        if currentBatch.count == batchSize {
          try await flushBatch()
        }
        numSucceeded += 1
      } catch {
        print("\(error)")
        numFailed += 1
      }
    }
    if !currentBatch.isEmpty {
      try await flushBatch()
    }
    let data = try PropertyListEncoder().encode(shard)
    try data.write(to: shardURL, options: .atomic)
    print(" => added \(numSucceeded) records successfully with \(numFailed) errors")
  }

  func shardForImageName(_ imageName: String) -> Int {
    let hash = Insecure.MD5.hash(data: imageName.data(using: .utf8)!)
    return Int(Array(hash)[0])
  }

}
