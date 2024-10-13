import Cocoa
import Foundation
import Honeycrisp
import MNIST

class ImageIterator: Sequence, IteratorProtocol {
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

  func next() -> (String, Tensor)? {
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

class DataLoader: Sequence, IteratorProtocol {
  let batchSize: Int
  var images: ImageIterator

  init(batchSize: Int, images: ImageIterator) {
    self.batchSize = batchSize
    self.images = images
  }

  func next() -> Tensor? {
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
    let bs = 8
    let reviveInterval = 100
    let reviveBatches = 2
    let commitCoeff = 0.5
    let lr: Float = 0.0001

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

    let model = VQVAE(channels: 4, vocab: 16384, latentChannels: 4, downsamples: 4)

    do {
      let dataset = DataLoader(
        batchSize: bs, images: try ImageIterator(imageDir: imageDir, imageSize: 256))

      func takeDataset(_ n: Int) -> some Collection<Tensor> {
        return (0..<n).lazy.map { _ in dataset.next()! }
      }

      let opt = Adam(model.parameters, lr: lr)
      var step = 0
      while true {
        print("revining unused dictionary entries...")
        let revivedCount = Tensor.withGrad(enabled: false) {
          print(" => collecting features...")
          let features = model.withMode(.inference) {
            Tensor(concat: takeDataset(reviveBatches).map(model.features))
          }
          print(" => reviving with \(features.shape[0]) features...")
          return model.bottleneck.revive(features)
        }
        print(" => revived \(try await revivedCount.ints()[0]) entries")
        let _ = try await model.bottleneck.dictionary.sum().item()

        print("training...")
        for batch in takeDataset(reviveInterval) {
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

        print("dumping samples to: samples.png ...")
        let input = dataset.next()!
        let (output, _) = Tensor.withGrad(enabled: false) {
          model.withMode(.inference) {
            model(input)
          }
        }
        let images = Tensor(concat: [input, output], axis: -1)
        let img = try await tensorToImage(tensor: images.move(axis: 1, to: -1).flatten(endAxis: 1))
        try img.write(to: URL(filePath: "samples.png"))
      }
    } catch {
      print("error while training: \(error)")
    }
  }
}

func formatFloat(_ x: Float) -> String {
  String(format: "%.5f", x)
}
