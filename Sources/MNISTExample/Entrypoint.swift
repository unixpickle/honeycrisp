import Foundation
import Honeycrisp
import MNIST

class Model: Trainable {
  @Child var conv1: Conv2D
  @Child var conv2: Conv2D
  @Child var dropout1: Dropout
  @Child var dropout2: Dropout
  @Child var linear1: Linear
  @Child var linear2: Linear

  override init() {
    super.init()
    conv1 = Conv2D(inChannels: 1, outChannels: 32, kernelSize: .square(3))
    conv2 = Conv2D(inChannels: 32, outChannels: 64, kernelSize: .square(3))
    self.dropout1 = Dropout(dropProb: 0.25)
    self.dropout2 = Dropout(dropProb: 0.5)
    self.linear1 = Linear(inCount: 9216, outCount: 128)
    self.linear2 = Linear(inCount: 128, outCount: 10)
  }

  func callAsFunction(_ x: Tensor) -> Tensor {
    var h = x
    h = conv1(h)
    h = h.relu()
    h = conv2(h)
    h = h.relu()
    h = h.maxPool2D(width: 2, height: 2)
    h = dropout1(h)
    h = h.flatten(startAxis: 1)
    h = linear1(h)
    h = h.relu()
    h = linear2(h)
    h = h.logSoftmax(axis: -1)
    return h
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
      Tensor(data: inputData, shape: [batchSize, 1, 28, 28], dtype: .float32),
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
        model.withMode(.inference) {
          computeLossAndAcc(testBatch)
        }
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
