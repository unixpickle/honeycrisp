import Cocoa
import Foundation
import Honeycrisp
import MNIST

struct ImageIterator: Sequence, IteratorProtocol {
  let imageSize: Int
  var imagePaths: [String]
  var offset = 0

  init(imageDir dirPath: String, imageSize: Int) throws {
    var paths = [String]()
    let fileManager = FileManager.default
    let directoryURL = URL(fileURLWithPath: dirPath, isDirectory: true)
    let contents = try fileManager.contentsOfDirectory(
      at: directoryURL, includingPropertiesForKeys: nil, options: [])
    for fileURL in contents {
      paths.append(fileURL.path())
    }
    self.imageSize = imageSize
    self.imagePaths = paths
  }

  mutating func next() -> (String, Tensor)? {
    while imagePaths.count > 0 {
      offset = offset % imagePaths.count
      guard let image = loadImage(path: imagePaths[offset], imageSize: imageSize) else {
        imagePaths.remove(at: offset)
        continue
      }
      let path = imagePaths[offset]
      offset += 1
      return (path, image)
    }
    return nil
  }
}

struct DataLoader: Sequence, IteratorProtocol {
  let batchSize: Int
  var images: ImageIterator

  mutating func next() -> Tensor? {
    var batch = [Tensor]()
    for (_, x) in images {
      batch.append(x)
      if batch.count == batchSize {
        break
      }
    }
    if batch.count == 0 {
      fatalError("failed to load data")
    }
    return Tensor(stack: batch).move(axis: -1, to: 1)
  }
}

@main
struct Main {
  static func main() async {
    let bs = 1
    let clusterInterval = 1000
    let clusterBatches = 1
    let commitCoeff = 0.5
    let lr: Float = 0.001

    if CommandLine.arguments.count != 2 {
      print("Usage: Text2Image <image_dir>")
      return
    }
    let imageDir = CommandLine.arguments[1]

    do {
      Backend.defaultBackend = try MPSBackend()
    } catch {
      print("failed to init MPS backend: \(error)")
    }

    let model = VQVAE(channels: 4, vocab: 16384, latentChannels: 4, downsamples: 5)

    do {
      let dataset = DataLoader(
        batchSize: bs, images: try ImageIterator(imageDir: imageDir, imageSize: 256))
      let opt = Adam(model.parameters, lr: lr)
      var step = 0
      while true {
        print("recomputing VQ centers...")
        Tensor.withGrad(enabled: false) {
          print(" => collecting features...")
          let features = Tensor(concat: dataset.prefix(clusterBatches).map(model.features))
          print(" => fitting features...")
          model.bottleneck.fitFeatures(features)
        }
        print("training...")
        for batch in dataset.prefix(clusterInterval) {
          step += 1
          let (output, vqLosses) = model(batch)
          let loss = (output - batch).abs().mean()
          (loss + vqLosses.codebookLoss + commitCoeff * vqLosses.commitmentLoss).backward()
          opt.step()
          opt.clearGrads()
          print(
            "step \(step):"
              + " loss=\(try await loss.item())"
              + " commitment=\(try await vqLosses.commitmentLoss.item())")
        }
      }
      /*
          print("writing tiff for \(path) ...")
          let img = try await tensorToImage(tensor: img)
          try img.write(to: URL(filePath: "output.tiff"))
          break
      */
    } catch {
      print("error while training: \(error)")
    }
  }
}

func formatFloat(_ x: Float) -> String {
  String(format: "%.5f", x)
}
