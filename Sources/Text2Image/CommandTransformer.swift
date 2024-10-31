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
  }

  let testCaptions = [
    "a red heart icon, red heart vector graphic",
    "a green tree, a tree with green leaves",
    "A blue square. A simple blue square icon.",
    "a cute corgi vector graphic. corgi dog graphic",
  ]

  let lr: Float = 0.0001
  let bs = 8
  let captionBytes: Int = 128
  let saveInterval: Int = 500
  let cfgProb: Float = 0.1
  let cfgScale: Float = 1.1

  let savePath: String
  let vqPath: String
  let dataDir: String
  let vqvae: VQVAE
  let model: Transformer
  let opt: Adam
  var step: Int = 0

  let dataLoader: TrainAndEval<CaptionedSequenceDataLoader>
  var dataStream: DataStream?
  var lastDataState: TrainAndEval<CaptionedSequenceDataLoader.State>?

  init(_ args: [String]) throws {
    if args.count != 3 {
      print("Usage: Text2Image transformer <data_dir> <vq_path> <save_path>")
      throw ArgumentError.invalidArgs
    }
    dataDir = args[0]
    vqPath = args[1]
    savePath = args[2]

    vqvae = VQVAE(channels: 4, vocab: 16384, latentChannels: 4, downsamples: 4)
    model = Transformer(
      config: TransformerConfig(
        VocabSize: vqvae.bottleneck.vocab + 256, TokenCount: captionBytes + 16 * 16))
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

      let trainLoss = loss(batch.train.0)
      let evalLoss = Tensor.withGrad(enabled: false) { loss(batch.eval.0) }

      trainLoss.backward()
      opt.step()
      opt.clearGrads()
      print(
        "step \(step):"
          + " loss=\(try await trainLoss.item())"
          + " valid_loss=\(try await evalLoss.item())"
          + " gflops=\(gflops)")
    }
  }

  private func sampleAndSave() async throws {
    let filename = "text2im_samples.png"
    print("sampling to \(filename) ...")
    let captions = captionTensor(testCaptions)
    let samples = try await model.sample(prefixes: captions, cfgScale: cfgScale)
    let embs = vqvae.bottleneck.embed(samples.reshape([-1, 16, 16]))
    let sampleImages = vqvae.decoder(embs.move(axis: -1, to: 1)).move(axis: 1, to: -1)
    let img = try await tensorToImage(tensor: sampleImages.flatten(endAxis: 1))
    try img.write(to: URL(filePath: filename))

    print("saving to \(savePath) ...")
    let state = State(
      step: step,
      model: try await model.state(),
      dataset: lastDataState!,
      opt: try await opt.state()
    )
    let stateData = try PropertyListEncoder().encode(state)
    try stateData.write(to: URL(filePath: savePath), options: .atomic)
  }

}
