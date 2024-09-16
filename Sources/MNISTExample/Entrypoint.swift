import Foundation
import Honeycrisp
import MNIST

class Model: Trainable {
  @Child var layer1: Linear
  @Child var layer2: Linear
  @Child var layer3: Linear

  override init() {
    super.init()
    layer1 = Linear(inCount: 28 * 28, outCount: 1024)
    layer2 = Linear(inCount: 1024, outCount: 1024)
    layer3 = Linear(inCount: 1024, outCount: 10)
  }

  func callAsFunction(_ x: Tensor) -> Tensor {
    var h = x
    h = layer1(h)
    h = h.relu()
    h = layer2(h)
    h = h.relu()
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
    let bs = 1024

    do {
      Backend.defaultBackend = try MPSBackend()
    } catch {
      print("failed to init MPS backend: \(error)")
    }

    print("creating model and optimizer...")
    let model = Model()
    let opt = Adam(model.parameters, lr: 0.001)

    do {
      let paramNorm = try await model.parameters.map { (_, param) in param.data!.pow(2).sum() }
        .reduce(
          Tensor(zeros: []), { $0 + $1 }
        ).sqrt().item()
      print(" => initial param norm: \(paramNorm)")
    } catch {
      print("error getting param norm: \(error)")
      return
    }

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

    print("training...")
    var seenExamples = 0
    for (i, (batch, testBatch)) in zip(train, test).enumerated() {
      let (loss, acc) = computeLossAndAcc(batch)
      loss.backward()
      opt.step()
      opt.clearGrads()
      let (testLoss, testAcc) = Tensor.withGrad(enabled: false) {
        computeLossAndAcc(testBatch)
      }
      seenExamples += batch.0.shape[0]
      let epochs = Float(seenExamples) / Float(train.images.count)
      do {
        print(
          "step \(i): loss=\(try await loss.item()) testLoss=\(try await testLoss.item()) acc=\(try await acc.item()) testAcc=\(try await testAcc.item()) epochs=\(epochs)"
        )
      } catch {
        print("fatal error: \(error)")
        return
      }
    }
  }
}
