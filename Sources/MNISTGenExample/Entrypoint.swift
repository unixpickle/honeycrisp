import Cocoa
import Foundation
import Honeycrisp
import MNIST

// Dataset configuration
let PatchSize: Int = 2
let TokenCount: Int = (28 / PatchSize) * (28 / PatchSize)
let NumLabels: Int = 10
let VocabSize: Int = NumLabels + (1 << (PatchSize * PatchSize))

// Model architecture settings
let ModelDim: Int = 128
let HeadDim: Int = 64  // must divide ModelDim
let LayerCount: Int = 4

// Training hyperparameters
let BatchSize: Int = 8
let LearningRate: Float = 0.0001

// Sampling hyperparameters
let SampleCount = 10
let SampleInterval = 1000

class KVCache {
  class Layer {
    var k: Tensor
    var v: Tensor

    var tokenCount: Int {
      k.shape[1]
    }

    init(batchSize: Int) {
      k = Tensor(zeros: [batchSize, 0, ModelDim])
      v = Tensor(zeros: [batchSize, 0, ModelDim])
    }
  }

  var layers: [Layer]

  var tokenCount: Int {
    layers[0].tokenCount
  }

  init(batchSize: Int, layerCount: Int) {
    layers = []
    for _ in 0..<layerCount {
      layers.append(Layer(batchSize: batchSize))
    }
  }
}

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

  func callAsFunction(_ x: Tensor, kvCache: KVCache.Layer? = nil) -> Tensor {
    // Go from [B x T x C] -> [B x H x T x C/H]
    func moveHeadsToOuter(_ x: Tensor) -> Tensor {
      x.reshape([x.shape[0], x.shape[1], ModelDim / HeadDim, HeadDim])[
        FullRange(), PermuteAxes(1, 0)]
    }

    // Go from [B x H x T x C/H] -> [B x T x C]
    func moveHeadsToInner(_ x: Tensor) -> Tensor {
      x[FullRange(), PermuteAxes(1, 0)].reshape([x.shape[0], x.shape[2], x.shape[1] * x.shape[3]])
    }

    let tokenOffset = kvCache?.tokenCount ?? 0

    let (k, v) =
      if let kvCache = kvCache {
        {
          let innerK = Tensor(concat: [kvCache.k, kProj(x)], axis: 1)
          let innerV = Tensor(concat: [kvCache.v, vProj(x)], axis: 1)
          let k = moveHeadsToOuter(innerK) / sqrt(sqrt(Float(HeadDim)))
          let v = moveHeadsToOuter(innerV)
          kvCache.k = innerK
          kvCache.v = innerV
          return (k, v)
        }()
      } else {
        (moveHeadsToOuter(kProj(x)) / sqrt(sqrt(Float(HeadDim))), moveHeadsToOuter(vProj(x)))
      }
    let q = moveHeadsToOuter(qProj(x)) / sqrt(sqrt(Float(HeadDim)))

    let energy = Tensor.batchedMatmul(a: q, transA: false, b: k, transB: true, transOut: false)
    let probs = (energy + causalMask[tokenOffset..<k.shape[2], 0..<k.shape[2]].expand(as: energy))
      .softmax()
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

  func callAsFunction(_ x: Tensor, kvCache: KVCache.Layer? = nil) -> Tensor {
    var h = x
    h = h + attn(norm1(h), kvCache: kvCache)
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

  func callAsFunction(_ x: Tensor, kvCache: KVCache? = nil) -> Tensor {
    // Input should be a [N x T] tensor of indices
    var h = embed.expand(shape: [x.shape[0], x.shape[1], VocabSize, ModelDim]).gather(
      axis: 2,
      indices: x.reshape([x.shape[0], x.shape[1], 1, 1]).repeating(axis: -1, count: ModelDim)
    ).reshape([x.shape[0], x.shape[1], ModelDim])

    let posOffset = kvCache?.tokenCount ?? 0
    h = h + posEmbed[posOffset..<(posOffset + x.shape[1])].expand(as: h)

    for (i, layer) in layers.children.enumerated() {
      let cacheLayer: KVCache.Layer? =
        if let kvCache = kvCache {
          kvCache.layers[i]
        } else {
          nil
        }
      h = layer(h, kvCache: cacheLayer)
    }
    h = normOut(h)
    h = unembed(h)
    return h
  }

  func sample(firstTokens: Tensor) async throws -> Tensor {
    assert(firstTokens.shape.count == 2, "\(firstTokens.shape)")
    assert(firstTokens.shape[1] == 1, "\(firstTokens.shape)")
    let kvCache = KVCache(batchSize: firstTokens.shape[0], layerCount: LayerCount)
    var outputs: [Tensor] = []
    var prevToken = firstTokens
    for _ in 0..<TokenCount {
      let logits = self(prevToken, kvCache: kvCache)
      let gumbels = -(-Tensor(randLike: logits).log()).log()
      prevToken = (logits + gumbels).argmax(axis: -1)
      outputs.append(prevToken)
    }
    return Tensor(concat: outputs, axis: 1)
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
    let patches = packPixelsIntoTokens(pixels: pixels, patchSize: PatchSize)
    let flatPatches = patches.flatten(startAxis: 1)
    let label =
      Tensor(data: outputLabels, shape: [batchSize, 1], dtype: .int64) + (VocabSize - NumLabels)
    return Tensor(concat: [label, flatPatches], axis: 1)
  }
}

@main
struct Main {
  static func main() async {
    do {
      Backend.defaultBackend = try MPSBackend()
    } catch {
      print("failed to init MPS backend: \(error)")
    }

    print("creating model and optimizer...")
    let model = Transformer()
    let opt = Adam(model.parameters, lr: LearningRate, eps: 1e-5)

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
    let train = DataIterator(images: trainShuffle, batchSize: BatchSize)
    let test = DataIterator(images: testShuffle, batchSize: BatchSize)

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
            + "test_loss=\(formatFloat(try await testLoss.item())) "
            + "epochs=\(formatFloat(epochs)) "
            + "param_norm=\(paramNorm) grad_norm=\(gradNorm) "
            + "time=\(formatFloat(Float(t2 - t1) / 1_000_000_000))"
        )
      } catch {
        print("fatal error: \(error)")
        return
      }
      if (i + 1) % SampleInterval == 0 {
        print("sampling...")
        do {
          let firstTokens =
            Tensor(range: 0..<(NumLabels * SampleCount), dtype: .int64).reshape([
              NumLabels * SampleCount, 1,
            ])
            / SampleCount + (VocabSize - NumLabels)
          let samples = try await model.sample(firstTokens: firstTokens)
          let pixelBatch = try await unpackPixelsInTokens(bitmaps: samples, patchSize: PatchSize)
          let pixels =
            pixelBatch.reshape([10, SampleCount, 28, 28])[
              PermuteAxes(0, 2, 1, 3)
            ].reshape([10 * 28, SampleCount * 28, 1]) * 255
          try await saveGrayscaleBitmap(pixels: pixels, to: URL(filePath: "samples.png"))
        } catch {
          print("failed to create samples: \(error)")
          return
        }
      }
    }
  }
}

func formatFloat(_ x: Float) -> String {
  String(format: "%.5f", x)
}
