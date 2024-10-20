import Foundation
import Honeycrisp

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

class DataLoader: Sequence, IteratorProtocol {
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

  func next() -> (Tensor, State)? {
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
      fatalError("failed to load data")
    }
    return (Tensor(stack: batch).move(axis: -1, to: 1), state!)
  }
}

class CaptionedDataLoader: Sequence, IteratorProtocol {
  typealias State = ImageIterator.State

  struct Metadata: Codable {
    let url: String
    let width: Int
    let height: Int
    let caption: String
  }

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

  func next() -> (Tensor, [String], State)? {
    var batch = [Tensor]()
    var captions = [String]()
    var state: State?
    for (imagePath, x, s) in images {
      state = s
      batch.append(x)
      let metaPath = URL(filePath: imagePath).deletingLastPathComponent()
        .deletingLastPathComponent().appending(component: "success").appending(
          component: URL(filePath: imagePath).lastPathComponent + ".json")
      let metadata: Metadata
      do {
        let data = try Data(contentsOf: metaPath)
        let decoder = JSONDecoder()
        metadata = try decoder.decode(Metadata.self, from: data)
      } catch {
        fatalError("failed to read metadata at \(metaPath): \(error)")
      }
      captions.append(metadata.caption)
      if batch.count == batchSize {
        break
      }
    }
    if batch.count == 0 {
      fatalError("failed to load data")
    }
    return (Tensor(stack: batch).move(axis: -1, to: 1), captions, state!)
  }
}
