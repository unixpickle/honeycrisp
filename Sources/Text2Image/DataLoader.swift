import Foundation
import Honeycrisp

enum DataError: Error {
  case datasetIsEmpty
}

class ImageIterator: Sequence, IteratorProtocol {
  struct State: Codable {
    let imageSize: Int
    var imagePaths: [String]
    var offset: Int = 0
  }

  public var state: State

  init(imageDir dirPath: String, imageSize: Int) throws {
    var paths = [String]()
    let fileManager = FileManager.default
    let directoryURL = URL(fileURLWithPath: dirPath, isDirectory: true)
    let contents = try fileManager.contentsOfDirectory(
      at: directoryURL, includingPropertiesForKeys: nil, options: [])
    for fileURL in contents {
      paths.append(fileURL.path())
    }
    self.state = State(imageSize: imageSize, imagePaths: paths)
  }

  func next() -> (String, Tensor, State)? {
    while state.imagePaths.count > 0 {
      state.offset = state.offset % state.imagePaths.count
      guard let image = loadImage(path: state.imagePaths[state.offset], imageSize: state.imageSize)
      else {
        state.imagePaths.remove(at: state.offset)
        continue
      }
      let path = state.imagePaths[state.offset]
      state.offset += 1
      return (path, image, state)
    }
    return nil
  }
}

class ImageDataLoader: Sequence, IteratorProtocol {
  typealias State = ImageIterator.State

  let batchSize: Int
  var images: ImageIterator

  var state: State {
    get { images.state }
    set { images.state = newValue }
  }

  init(batchSize: Int, images: ImageIterator) {
    self.batchSize = batchSize
    self.images = images
  }

  func next() -> Result<(Tensor, State), Error>? {
    var batch = [Tensor]()
    var state: State?
    for (_, x, s) in images {
      batch.append(x)
      state = s
      if batch.count == batchSize {
        break
      }
    }
    if batch.count == 0 {
      return .failure(DataError.datasetIsEmpty)
    }
    return .success((Tensor(stack: batch).move(axis: -1, to: 1), state!))
  }
}

class CaptionedSequenceDataLoader: Sequence, IteratorProtocol {

  typealias Shard = CommandTokenize.Shard

  struct State: Codable {
    var shardPaths: [String]
    var currentShard: Int
    var offsetInShard: Int
  }

  let batchSize: Int
  let dropProb: Float
  let isEval: Bool
  let captionLength: Int
  let captionTokenOffset: Int
  private var _state: State
  private var shardData: Shard?

  var state: State {
    get {
      _state
    }
    set {
      if newValue.currentShard != _state.currentShard {
        shardData = nil
      }
      _state = newValue
    }
  }

  init(
    batchSize: Int, dropProb: Float, captionLength: Int, captionTokenOffset: Int, shardDir: String,
    isEval: Bool = false
  ) throws {
    self.batchSize = batchSize
    self.dropProb = dropProb
    self.isEval = isEval
    self.captionLength = captionLength
    self.captionTokenOffset = captionTokenOffset
    var paths = [String]()
    let fileManager = FileManager.default
    let directoryURL = URL(fileURLWithPath: shardDir, isDirectory: true)
    let contents = try fileManager.contentsOfDirectory(
      at: directoryURL, includingPropertiesForKeys: nil, options: [])
    for fileURL in contents {
      if fileURL.pathExtension != "plist" {
        continue
      }
      paths.append(fileURL.path())
    }
    paths.sort()
    if isEval {
      paths = [paths[0]]
    } else {
      paths.remove(at: 0)
    }
    _state = State(shardPaths: paths, currentShard: 0, offsetInShard: 0)
  }

  func captionTensor(_ caption: String) -> Tensor {
    var textTokens = [Int](repeating: 0, count: captionLength)
    for (j, char) in caption.utf8.enumerated() {
      if j >= captionLength {
        break
      }
      textTokens[j] = Int(char) + captionTokenOffset
    }
    return Tensor(data: textTokens, shape: [captionLength])
  }

  func next() -> Result<(Tensor, State), Error>? {
    do {
      let tensors = try (0..<batchSize).map { _ in
        let result = try self.nextSingle()
        return result
      }
      return .success((Tensor(stack: tensors), state))
    } catch {
      return .failure(error)
    }
  }

  func nextSingle() throws -> Tensor {
    for _ in 0..<(state.shardPaths.count + 1) {
      if self.shardData == nil {
        let data = try Data(
          contentsOf: URL(fileURLWithPath: state.shardPaths[state.currentShard]))
        let decoder = PropertyListDecoder()
        self.shardData = try decoder.decode(CommandTokenize.Shard.self, from: data)
      }
      if state.offsetInShard >= shardData!.records.count {
        state.currentShard = (state.currentShard + 1) % state.shardPaths.count
        state.offsetInShard = 0
        shardData = nil
        continue
      }
      let record = shardData!.records[state.offsetInShard]
      state.offsetInShard += 1
      let vqTokens = Tensor(data: record.tokens.map { Int64($0) }, shape: [record.tokens.count])
      let textTokens = captionTensor(record.metadata.caption)

      let mask = (Tensor(rand: [1]) > dropProb).expand(as: textTokens)
      let maskedText = mask.when(isTrue: textTokens, isFalse: Tensor(zerosLike: textTokens))
      let zeros = Tensor(zeros: [1], dtype: .int64)

      return Tensor(concat: [zeros, maskedText, vqTokens])
    }
    throw DataError.datasetIsEmpty
  }

}
