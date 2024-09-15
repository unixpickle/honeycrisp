import Foundation
import Honeycrisp
import MNIST

class Model: Trainable {
  @Child var layer1: Linear
  @Child var layer2: Linear
  @Child var layer3: Linear

  override init() {
    super.init()
    layer1 = Linear(inCount: 28 * 28, outCount: 256)
    layer2 = Linear(inCount: 256, outCount: 256)
    layer3 = Linear(inCount: 256, outCount: 10)
  }

  func callAsFunction(_ x: Tensor) -> Tensor {
    var h = x
    h = layer1(h)
    h = h.gelu()
    h = layer2(h)
    h = h.gelu()
    h = layer3(h)
    return h.logSoftmax(axis: -1)
  }
}

struct DataIterator: Sequence, IteratorProtocol {
  let images: [MNISTDataset.Image]
  let batchSize: Int
  var offset = 0

  mutating func next() -> (Tensor, Tensor)? {
    var inputData = [Float]()
    var outputLabels = [Int]()
    for _ in 0..<batchSize {
      let img = images[offset % images.count]
      for pixel in img.pixels {
        inputData.append(Float(pixel) / 255)
      }
      outputLabels.append(img.label)
      offset += 1
    }
    return (
      Tensor(data: inputData, shape: [batchSize, 28 * 28], dtype: .float32),
      Tensor(data: outputLabels, shape: [batchSize, 1], dtype: .int64)
    )
  }
}

@main
struct Main {
  static func main() async {
    let bs = 8

    print("creating model and optimizer...")
    let model = Model()
    let opt = Adam(model.parameters, lr: 0.001)

    print("creating dataset...")
    let dataset: MNISTDataset
    do {
      dataset = try await MNISTDataset.download(toDir: "mnist_data")
    } catch {
      print("Error downloading dataset: \(error)")
      return
    }
    let train = DataIterator(images: dataset.train, batchSize: bs)
    let test = DataIterator(images: dataset.test, batchSize: bs)

    func computeLossAndAcc(_ inputsAndTargets: (Tensor, Tensor)) -> (Tensor, Tensor) {
      let (inputs, targets) = inputsAndTargets
      let output = model(inputs)

      // Compute accuracy where we evenly distribute out ties.
      let acc = (output.argmax(axis: -1, keepdims: true) == targets).cast(.float32).mean()

      return (-(output.gather(axis: 1, indices: targets)).mean(), acc)
    }

    for (i, (batch, testBatch)) in zip(train, test).enumerated() {
      let (loss, acc) = computeLossAndAcc(batch)
      loss.backward()
      opt.step()
      opt.clearGrads()
      let (testLoss, testAcc) = computeLossAndAcc(testBatch)
      do {
        print(
          "step \(i): loss=\(try await loss.item()) testLoss=\(try await testLoss.item()) acc=\(try await acc.item()) testAcc=\(try await testAcc.item())"
        )
      } catch {
        print("fatal error: \(error)")
        return
      }
    }
  }
}
