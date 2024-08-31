import Foundation
import Honeycrisp
import MNIST

class Linear: Trainable {
  @Parameter var weight: Tensor
  @Parameter var bias: Tensor

  init(inSize: Int, outSize: Int) {
    super.init()
    weight = Tensor(gaussian: [inSize, outSize]) / sqrt(Float(inSize))
    bias = Tensor(zeros: [outSize])
  }

  func callAsFunction(_ x: Tensor) -> Tensor {
    let h = (x &* weight)
    return h + bias.expand(as: h)
  }
}

class Model: Trainable {
  @Child var layer1: Linear
  @Child var layer2: Linear
  @Child var layer3: Linear

  override init() {
    super.init()
    layer1 = Linear(inSize: 28 * 28, outSize: 256)
    layer2 = Linear(inSize: 256, outSize: 256)
    layer3 = Linear(inSize: 256, outSize: 10)
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
      Tensor(data: inputData, shape: [batchSize, 28 * 28]),
      Tensor(oneHot: outputLabels, count: 10)
    )
  }
}

@main
struct Main {
  static func main() async {
    let bs = 256

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

    func computeLossAndAcc(_ inputsAndTargets: (Tensor, Tensor)) -> (Tensor, Float) {
      let (inputs, targets) = inputsAndTargets
      let output = model(inputs)

      // Compute accuracy where we evenly distribute out ties.
      var maskMax = output == (output.max(axis: -1, keepdims: true).expand(as: output))
      maskMax = maskMax / maskMax.sum(axis: -1, keepdims: true).expand(as: maskMax)
      let acc = (maskMax * targets).sum(axis: -1).mean().item()

      return (-(output * targets).sum(axis: 1).mean(), acc)
    }

    for (i, (batch, testBatch)) in zip(train, test).enumerated() {
      let (loss, acc) = computeLossAndAcc(batch)
      loss.backward()
      opt.step()
      opt.clearGrads()

      let (testLoss, testAcc) = computeLossAndAcc(testBatch)
      print(
        "step \(i): loss=\(loss.item()) testLoss=\(testLoss.item()) acc=\(acc) testAcc=\(testAcc)")
    }
  }
}
