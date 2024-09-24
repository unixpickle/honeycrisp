import Foundation
import Honeycrisp
import MNIST

let PatchSize: Int = 2
let TokenCount: Int = (28 / PatchSize) * (28 / PatchSize)
let ModelDim: Int = 128
let HeadDim: Int = 64
let LayerCount: Int = 4
let NumLabels: Int = 10
let VocabSize: Int = NumLabels + (1 << (PatchSize * PatchSize))

class Attention: Trainable {
  let causalMask: Tensor

  @Child var qProj: Linear
  @Child var kProj: Linear
  @Child var vProj: Linear
  @Child var outProj: Linear

  override init() {
    causalMask = Tensor(constant: 1e8, shape: [TokenCount, TokenCount]).tril() - 1e8
    assert(!causalMask.needsGrad)
    super.init()
    self.qProj = Linear(inCount: ModelDim, outCount: ModelDim)
    self.kProj = Linear(inCount: ModelDim, outCount: ModelDim)
    self.vProj = Linear(inCount: ModelDim, outCount: ModelDim)
    self.outProj = Linear(inCount: ModelDim, outCount: ModelDim)
  }

  func callAsFunction(_ x: Tensor) -> Tensor {
    // Go from [B x T x C] -> [B x H x T x C/H]
    func moveHeadsToOuter(_ x: Tensor) -> Tensor {
      x.reshape([x.shape[0], x.shape[1], ModelDim / HeadDim, HeadDim])[
        FullRange(), PermuteAxes(1, 0)]
    }

    // Go from [B x H x T x C/H] -> [B x T x C]
    func moveHeadsToInner(_ x: Tensor) -> Tensor {
      x[FullRange(), PermuteAxes(1, 0)].reshape([x.shape[0], x.shape[2], x.shape[1] * x.shape[3]])
    }

    let q = moveHeadsToOuter(qProj(x)) / sqrt(sqrt(Float(HeadDim)))
    let k = moveHeadsToOuter(kProj(x)) / sqrt(sqrt(Float(HeadDim)))
    let v = moveHeadsToOuter(vProj(x))

    let energy = Tensor.batchedMatmul(a: q, transA: false, b: k, transB: true, transOut: false)

    let probs = (energy + causalMask.expand(as: energy)).softmax()
    let reducedValues = Tensor.batchedMatmul(
      a: probs, transA: false, b: v, transB: false, transOut: false)
    return outProj(moveHeadsToInner(reducedValues))
  }
}

class Block: Trainable {
  @Child var attn: Attention
  @Child var norm1: LayerNorm
  @Child var norm2: LayerNorm
  @Child var lin1: Linear
  @Child var lin2: Linear

  override init() {
    super.init()
    self.attn = Attention()
    self.norm1 = LayerNorm(shape: [ModelDim])
    self.norm2 = LayerNorm(shape: [ModelDim])
    self.lin1 = Linear(inCount: ModelDim, outCount: ModelDim * 2)
    self.lin2 = Linear(inCount: ModelDim * 2, outCount: ModelDim)
  }

  func callAsFunction(_ x: Tensor) -> Tensor {
    var h = x
    h = h + attn(norm1(h))
    h = h + lin2(lin1(norm2(h)).gelu())
    return h
  }
}

class Transformer: Trainable {
  @Param var embed: Tensor
  @Param var posEmbed: Tensor
  @Child var layers: TrainableArray<Block>
  @Child var normOut: LayerNorm
  @Child var unembed: Linear

  override init() {
    super.init()
    embed = Tensor(randn: [VocabSize, ModelDim])
    posEmbed = Tensor(randn: [TokenCount, ModelDim])
    layers = TrainableArray((0..<LayerCount).map { _ in Block() })
    normOut = LayerNorm(shape: [ModelDim])

    unembed = Linear(inCount: ModelDim, outCount: VocabSize - NumLabels)

    // Uniform initial probability
    unembed.weight = unembed.weight.noGrad() * 0
  }

  func callAsFunction(_ x: Tensor) -> Tensor {
    // Input should be a [N x T] tensor of indices
    var h = embed.expand(shape: [x.shape[0], x.shape[1], VocabSize, ModelDim]).gather(
      axis: 2,
      indices: x.reshape([x.shape[0], x.shape[1], 1, 1]).repeating(axis: -1, count: ModelDim)
    ).reshape([x.shape[0], x.shape[1], ModelDim])
    h = h + posEmbed.expand(as: h)
    for layer in layers.children {
      h = layer(h)
    }
    h = normOut(h)
    h = unembed(h)
    return h
  }

  func paramNorm() async throws -> Float {
    try await parameters.map { (_, param) in param.data!.pow(2).sum() }
      .reduce(
        Tensor(zeros: []), { $0 + $1 }
      ).sqrt().item()
  }

  func gradNorm() async throws -> Float {
    var sum = Tensor(zeros: [])
    for (name, param) in parameters {
      if let grad = param.grad {
        sum = sum + grad.pow(2).sum()
      } else {
        print("WARNING: param \(name) has no gradient!")
      }
    }
    return try await sum.sqrt().item()
  }
}

struct DataIterator: Sequence, IteratorProtocol {
  let images: [MNISTDataset.Image]
  let batchSize: Int
  var offset = 0

  mutating func next() -> Tensor? {
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
    let probs = Tensor(data: inputData, shape: [batchSize, 28, 28], dtype: .float32)
    let pixels = (probs > Tensor(randLike: probs)).cast(.int64)
    var patches: Tensor?
    for i in 0..<PatchSize {
      for j in 0..<PatchSize {
        let subPatch = pixels[
          ..., stride(from: i, to: 28, by: PatchSize), stride(from: j, to: 28, by: PatchSize)]
        let scale = 1 << (i * PatchSize + j)
        if let p = patches {
          patches = p + subPatch * scale
        } else {
          patches = subPatch * scale
        }
      }
    }
    let flatPatches = patches!.reshape([patches!.shape[0], patches!.shape[1] * patches!.shape[2]])
    let label = Tensor(data: outputLabels, shape: [batchSize, 1], dtype: .int64) + 2
    return Tensor(concat: [label, flatPatches], axis: 1)
  }
}

@main
struct Main {
  static func main() async {
    let bs = 8

    do {
      Backend.defaultBackend = try MPSBackend()
    } catch {
      print("failed to init MPS backend: \(error)")
    }

    print("creating model and optimizer...")
    let model = Transformer()
    let opt = Adam(model.parameters, lr: 0.0001, eps: 1e-5)

    do {
      print(" => initial param norm: \(try await model.paramNorm())")
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
    var trainShuffle = dataset.train
    trainShuffle.shuffle()
    var testShuffle = dataset.test
    testShuffle.shuffle()
    let train = DataIterator(images: trainShuffle, batchSize: bs)
    let test = DataIterator(images: testShuffle, batchSize: bs)

    func computeLoss(_ seq: Tensor) -> Tensor {
      let inSeq = seq[..., ..<(-1)]
      let outSeq = seq[..., 1...]
      let output = model(inSeq).logSoftmax()
      return
        -(output.gather(axis: -1, indices: outSeq.reshape([outSeq.shape[0], outSeq.shape[1], 1])))
        .mean() / log(2.0) * Float(TokenCount)
    }

    print("training...")
    var seenExamples = 0
    for (i, (batch, testBatch)) in zip(train, test).enumerated() {
      let t1 = DispatchTime.now().uptimeNanoseconds
      let trainLoss = computeLoss(batch)
      opt.clearGrads()
      trainLoss.backward()
      opt.step()
      let testLoss = Tensor.withGrad(enabled: false) {
        computeLoss(testBatch)
      }
      seenExamples += batch.shape[0]
      let epochs = Float(seenExamples) / Float(train.images.count)
      do {
        let paramNorm = try await model.paramNorm()
        let gradNorm = try await model.gradNorm()
        let t2 = DispatchTime.now().uptimeNanoseconds
        print(
          "step \(i): loss=\(formatFloat(try await trainLoss.item())) "
            + "testLoss=\(formatFloat(try await testLoss.item())) "
            + "epochs=\(formatFloat(epochs)) "
            + "param_norm=\(paramNorm) grad_norm=\(gradNorm) "
            + "time=\(formatFloat(Float(t2 - t1) / 1_000_000_000))"
        )
      } catch {
        print("fatal error: \(error)")
        return
      }
    }
  }
}

func formatFloat(_ x: Float) -> String {
  String(format: "%.5f", x)
}
