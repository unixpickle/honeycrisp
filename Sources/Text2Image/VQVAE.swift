import Honeycrisp

class DownsampleBlock: Trainable {
  @Child var norm: GroupNorm
  @Child var conv1: Conv2D
  @Child var conv2: Conv2D
  @Child var skip: Conv2D

  init(inChannels: Int, outChannels: Int) {
    super.init()
    self.norm = GroupNorm(groupCount: 32, channelCount: inChannels)
    self.conv1 = Conv2D(
      inChannels: inChannels, outChannels: outChannels, kernelSize: .square(3), padding: .same)
    self.conv2 = Conv2D(
      inChannels: outChannels, outChannels: outChannels, kernelSize: .square(3), stride: .square(2),
      padding: .allSides(1))
    self.skip = Conv2D(
      inChannels: inChannels, outChannels: outChannels, kernelSize: .square(1), stride: .square(2))
  }

  func callAsFunction(_ x: Tensor) -> Tensor {
    var h = norm(x)
    h = conv1(h)
    h = h.gelu()
    h = conv2(h)
    h = h + skip(x)
    return h
  }
}

class UpsampleBlock: Trainable {
  @Child var norm: GroupNorm
  @Child var conv1: Conv2D
  @Child var conv2: Conv2D
  @Child var skip: Conv2D

  init(inChannels: Int, outChannels: Int) {
    super.init()
    self.norm = GroupNorm(groupCount: 32, channelCount: inChannels)
    self.conv1 = Conv2D(
      inChannels: inChannels, outChannels: outChannels, kernelSize: .square(3), padding: .same)
    self.conv2 = Conv2D(
      inChannels: outChannels, outChannels: outChannels, kernelSize: .square(3), padding: .same)
    self.skip = Conv2D(
      inChannels: inChannels, outChannels: outChannels, kernelSize: .square(1))
  }

  func callAsFunction(_ x: Tensor) -> Tensor {
    // Neighbor upsampling for x.
    let x = x.unsqueeze(axis: -2).unsqueeze(axis: -1).repeating(axis: -1, count: 2).repeating(
      axis: -3, count: 2
    ).reshape([x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3] * 2])

    var h = norm(x)
    h = conv1(h)
    h = h.gelu()
    h = conv2(h)
    h = h + skip(x)
    return h
  }
}

class VQEncoder: Trainable {
  @Child var inProj: Conv2D
  @Child var blocks: TrainableArray<DownsampleBlock>
  @Child var outProj: Conv2D

  init(inChannels: Int, outChannels: Int, downsamples: Int) {
    super.init()
    self.inProj = Conv2D(
      inChannels: inChannels, outChannels: 64, kernelSize: .square(3), padding: .same)

    var blockArr = [DownsampleBlock]()
    var ch = 64
    for _ in 0..<downsamples {
      blockArr.append(DownsampleBlock(inChannels: ch, outChannels: ch * 2))
      ch *= 2
    }
    self.blocks = TrainableArray(blockArr)

    self.outProj = Conv2D(inChannels: ch, outChannels: outChannels, kernelSize: .square(1))
  }

  func callAsFunction(_ x: Tensor) -> Tensor {
    var h = self.inProj(x)
    for layer in blocks.children {
      h = layer(h)
    }
    h = self.outProj(h)
    return h
  }
}

class VQDecoder: Trainable {
  @Child var inProj: Conv2D
  @Child var blocks: TrainableArray<UpsampleBlock>
  @Child var outProj: Conv2D

  init(inChannels: Int, outChannels: Int, upsamples: Int) {
    super.init()
    self.inProj = Conv2D(
      inChannels: inChannels, outChannels: 64 * (1 << upsamples), kernelSize: .square(3),
      padding: .same)

    var blockArr = [UpsampleBlock]()
    var ch = 64 * (1 << upsamples)
    for _ in 0..<upsamples {
      blockArr.append(UpsampleBlock(inChannels: ch, outChannels: ch / 2))
      ch /= 2
    }
    self.blocks = TrainableArray(blockArr)

    self.outProj = Conv2D(inChannels: ch, outChannels: outChannels, kernelSize: .square(1))
  }

  func callAsFunction(_ x: Tensor) -> Tensor {
    var h = self.inProj(x)
    for layer in blocks.children {
      h = layer(h)
    }
    h = self.outProj(h)
    return h
  }
}

class VQBottleneck: Trainable {
  struct Losses {
    let commitmentLoss: Tensor
    let codebookLoss: Tensor
  }

  struct Output {
    let straightThrough: Tensor
    let codes: Tensor
    let losses: Losses
  }

  let vocab: Int
  let channels: Int

  @Param var dictionary: Tensor

  init(vocab: Int, channels: Int) {
    self.vocab = vocab
    self.channels = channels
    super.init()
    self.dictionary = Tensor(randn: [vocab, channels])
  }

  func callAsFunction(_ x: Tensor) -> Output {
    let batch = x.shape[0]
    let channels = x.shape[1]
    let spatialShape = Array(x.shape[2...])

    let vecs = x.move(axis: 1, to: -1).flatten(endAxis: -2)
    let codes = nearestIndices(vecs, dictionary)
    let selection = self.dictionary.gather(axis: 0, indices: codes)
    let out = selection.reshape([batch] + spatialShape + [channels]).move(axis: -1, to: 1)
    return Output(
      straightThrough: out.noGrad() + (x - x.noGrad()),
      codes: codes.reshape([batch] + spatialShape),
      losses: Losses(
        commitmentLoss: (out.noGrad() - x).pow(2).mean(),
        codebookLoss: (out - x.noGrad()).pow(2).mean()
      )
    )
  }

  func fitFeatures(_ x: Tensor) {
    // Make sure there's enough centers.
    var x = x
    while x.shape[0] < vocab {
      x = Tensor(concat: [x, x + Tensor(randnLike: x) * 0.001])
    }

    let indices = Tensor(data: Array((0..<x.shape[0]).shuffled()[..<vocab]), shape: [vocab])
    var centers = x.gather(axis: 0, indices: indices)
    for _ in 0..<10 {
      let indices = nearestIndices(x, centers)
      let sums = x.scatter(axis: 0, count: vocab, indices: indices)
      let counts = Tensor(onesLike: indices).scatter(axis: 0, count: vocab, indices: indices)
      centers = sums / (counts.cast(.float32) + 1e-5).unsqueeze(axis: 1).expand(as: sums)
    }
    dictionary = centers
  }
}

func nearestIndices(_ vecs: Tensor, _ centers: Tensor) -> Tensor {
  let dots = Tensor.matmul(a: vecs, transA: false, b: centers, transB: true, transOut: false)
  let vecsNorm = vecs.pow(2).sum(axis: 1, keepdims: true).expand(as: dots)
  let dictNorm = centers.pow(2).sum(axis: 1).unsqueeze(axis: 0).expand(as: dots)
  let dists = vecsNorm + dictNorm - 2 * dots
  return dists.argmin(axis: 1)
}

class VQVAE: Trainable {
  @Child var encoder: VQEncoder
  @Child var bottleneck: VQBottleneck
  @Child var decoder: VQDecoder

  init(channels: Int, vocab: Int, latentChannels: Int, downsamples: Int) {
    super.init()
    self.encoder = VQEncoder(
      inChannels: channels, outChannels: latentChannels, downsamples: downsamples)
    self.bottleneck = VQBottleneck(vocab: vocab, channels: latentChannels)
    self.decoder = VQDecoder(
      inChannels: latentChannels, outChannels: channels, upsamples: downsamples)
  }

  func callAsFunction(_ x: Tensor) -> (Tensor, VQBottleneck.Losses) {
    var h = x
    h = encoder(h)
    let vqOut = bottleneck(h)
    h = vqOut.straightThrough
    h = decoder(h)
    return (h, vqOut.losses)
  }

  func features(_ x: Tensor) -> Tensor {
    print("computing features...")
    return encoder(x).move(axis: 1, to: -1).flatten(endAxis: -2)
  }
}