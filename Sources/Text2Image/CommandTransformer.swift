import Foundation
import Honeycrisp

struct TrainAndEval<T> {
  let train: T
  let eval: T
}

extension TrainAndEval: Codable where T: Codable {}

class CommandTransformer: Command {

  typealias DataStream = AsyncThrowingStream<
    TrainAndEval<(Tensor, CaptionedSequenceDataLoader.State)>, Error
  >

  public struct State: Codable {
    let step: Int
    let model: Trainable.State
    let dataset: TrainAndEval<CaptionedSequenceDataLoader.State>?
    let opt: Adam.State?
    let gradScale: Float?
  }

  let testCaptions = [
    "a red heart icon, red heart vector graphic",
    "a green tree, a tree with green leaves",
    "A blue square. A simple blue square icon.",
    "a cute corgi vector graphic. corgi dog graphic",
  ]

  let lr: Float = 0.00001
  let bs = 8
  let microbatch = 4
  let captionBytes: Int = 128
  let saveInterval: Int = 1000
  let cfgProb: Float = 0.1
  let cfgScales: [Float] = [1.0, 2.0, 4.0]

  let savePath: String
  let vqPath: String
  let dataDir: String
  let vqvae: VQVAE
  let model: Transformer
  let opt: Adam
  let weightGradBackend: BackendFLOPCounter
  var gradScale: Float = 65536.0
  var step: Int = 0

  let dataLoader: TrainAndEval<CaptionedSequenceDataLoader>
  var dataStream: DataStream?
  var lastDataState: TrainAndEval<CaptionedSequenceDataLoader.State>?

  override internal var flopCount: Int64 {
    return super.flopCount + weightGradBackend.flopCount
  }

  init(_ args: [String]) throws {
    if args.count != 3 {
      print("Usage: Text2Image transformer <data_dir> <vq_path> <save_path>")
      throw ArgumentError.invalidArgs
    }
    dataDir = args[0]
    vqPath = args[1]
    savePath = args[2]

    weightGradBackend = BackendFLOPCounter(
      wrapping: CoreMLBackend(wrapping: Backend.defaultBackend))

    vqvae = VQVAE(channels: 4, vocab: 16384, latentChannels: 4, downsamples: 4)
    model = Transformer(
      config: TransformerConfig(
        VocabSize: vqvae.bottleneck.vocab + 256, TokenCount: captionBytes + 16 * 16,
        WeightGradBackend: weightGradBackend))
    opt = Adam(model.parameters, lr: lr)
    dataLoader = TrainAndEval(
      train: try CaptionedSequenceDataLoader(
        batchSize: bs, dropProb: cfgProb, captionLength: captionBytes,
        captionTokenOffset: vqvae.bottleneck.vocab, shardDir: dataDir),
      eval: try CaptionedSequenceDataLoader(
        batchSize: bs, dropProb: cfgProb, captionLength: captionBytes,
        captionTokenOffset: vqvae.bottleneck.vocab, shardDir: dataDir, isEval: true)
    )
  }

  override public func run() async throws {
    try await prepare()

    while true {
      try await trainInnerLoop()
      try await sampleAndSave()
    }
  }

  private func prepare() async throws {
    print("loading VQVAE from checkpoint: \(vqPath) ...")
    let data = try Data(contentsOf: URL(fileURLWithPath: vqPath))
    let decoder = PropertyListDecoder()
    let state = try decoder.decode(CommandVQVAE.State.self, from: data)
    try vqvae.loadState(state.model)

    if FileManager.default.fileExists(atPath: savePath) {
      print("loading from checkpoint: \(savePath) ...")
      let data = try Data(contentsOf: URL(fileURLWithPath: savePath))
      let decoder = PropertyListDecoder()
      let state = try decoder.decode(State.self, from: data)
      try model.loadState(state.model)
      if let optState = state.opt {
        try opt.loadState(optState)
      }
      if let dataState = state.dataset {
        dataLoader.train.state = dataState.train
        dataLoader.eval.state = dataState.eval
      }
      if let gs = state.gradScale {
        gradScale = gs
      }
      step = state.step
    }

    let it = zip(dataLoader.train, dataLoader.eval).lazy.map {
      x, y -> Result<TrainAndEval<(Tensor, CaptionedSequenceDataLoader.State)>, Error> in
      switch x {
      case .failure(let e):
        return .failure(e)
      case .success(let x):
        switch y {
        case .failure(let e):
          return .failure(e)
        case .success(let y):
          return .success(TrainAndEval(train: x, eval: y))
        }
      }
    }
    dataStream = loadDataInBackground(it)
  }

  private func captionTensor(_ captions: [String]) -> Tensor {
    Tensor(stack: captions.map { dataLoader.train.captionTensor($0) })
  }

  private func takeDataset(_ n: Int) -> AsyncPrefixSequence<DataStream> {
    dataStream!.prefix(n)
  }

  private func trainInnerLoop() async throws {
    func loss(_ batch: Tensor) -> Tensor {
      // We do not model the caption prefix, only the VQ tokens.
      // If we truncate the input, the dimensions aren't aligned well and slow
      // down training significantly.
      //     let outputs = model(batch[..., ..<(-1)])[..., (captionBytes - 1)...]
      let outputs = model(batch)[..., (captionBytes - 1)..<(batch.shape[1] - 1)]
      let targets = batch[..., captionBytes...]

      let logProbs = outputs.logSoftmax(axis: -1)
      let losses = -logProbs.gather(axis: -1, indices: targets.unsqueeze(axis: -1))
      return losses.mean()
    }

    print("training...")
    for try await batch in takeDataset(saveInterval) {
      lastDataState = TrainAndEval(train: batch.train.1, eval: batch.eval.1)
      step += 1

      let evalLoss = Tensor.withGrad(enabled: false) { loss(batch.eval.0) }
      try await evalLoss.wait()

      var trainLoss: Tensor?
      var gradNorm: Float?
      while true {
        for i in stride(from: 0, to: bs, by: microbatch) {
          let smallBatch = min(bs - i, microbatch)
          trainLoss = loss(batch.train.0[i..<(i + smallBatch)])
          (trainLoss! * gradScale * Float(smallBatch) / Float(bs)).backward()

          // Ensure that the memory from the step is done being used.
          for (_, p) in model.parameters {
            try await p.grad?.wait()
          }
        }
        for (_, var p) in model.parameters {
          if let g = p.grad {
            p.grad = g / gradScale
          }
        }
        gradNorm = try await model.gradNorm()
        if gradNorm!.isFinite {
          gradScale *= 1.01
          break
        }
        gradScale *= 0.5
        opt.clearGrads()
        if !(try await trainLoss!.item()).isFinite {
          print("got NaN in forward pass!")
          return
        }
        trainLoss = nil
        print("got NaN in backward pass, reducing gradScale to \(gradScale)")
      }
      opt.step()
      opt.clearGrads()
      print(
        "step \(step):"
          + " loss=\(try await trainLoss!.item())"
          + " valid_loss=\(try await evalLoss.item())"
          + " grad_norm=\(gradNorm!)"
          + " grad_scale=\(gradScale)"
          + " gflops=\(gflops)")
    }
  }

  private func sampleAndSave() async throws {
    let filename = "text2im_samples.png"
    print("sampling to \(filename) ...")
    var images = [Tensor]()
    for scale in cfgScales {
      let gen = try await Backend.current.createRandom()
      try await gen.seed(step)

      let captions = captionTensor(testCaptions)
      let samples = try await model.sample(prefixes: captions, generator: gen, cfgScale: scale)
      Tensor.withGrad(enabled: false) {
        let embs = vqvae.bottleneck.embed(samples.reshape([-1, 16, 16]))
        images.append(
          vqvae.decoder(embs.move(axis: -1, to: 1)).move(axis: 1, to: -1).flatten(endAxis: 1))
      }
    }
    let img = try await tensorToImage(tensor: Tensor(concat: images, axis: 1))
    try img.write(to: URL(filePath: filename))

    print("saving to \(savePath) ...")
    let state = State(
      step: step,
      model: try await model.state(),
      dataset: lastDataState!,
      opt: try await opt.state(),
      gradScale: gradScale
    )
    let stateData = try PropertyListEncoder().encode(state)
    try stateData.write(to: URL(filePath: savePath), options: .atomic)
  }

}
