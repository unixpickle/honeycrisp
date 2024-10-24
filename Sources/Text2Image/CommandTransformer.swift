import Foundation
import Honeycrisp

class CommandTransformer: Command {

  public struct State: Codable {
    let step: Int
    let model: Trainable.State
    let dataset: DataLoader.State?
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
  let saveInterval: Int = 2
  let cfgProb: Float = 0.1
  let cfgScale: Float = 1.1

  let savePath: String
  let vqPath: String
  let imageDir: String
  let vqvae: VQVAE
  let model: Transformer
  let opt: Adam
  var step: Int = 0
  let dataLoader: CaptionedDataLoader
  var dataStream: AsyncStream<(Tensor, [String], DataLoader.State)>?
  var lastDataState: CaptionedDataLoader.State?

  init(_ args: [String]) throws {
    if args.count != 3 {
      print("Usage: Text2Image transformer <image_dir> <vq_path> <save_path>")
      throw ArgumentError.invalidArgs
    }
    imageDir = args[0]
    vqPath = args[1]
    savePath = args[2]

    vqvae = VQVAE(channels: 4, vocab: 16384, latentChannels: 4, downsamples: 4)
    model = Transformer(
      config: TransformerConfig(
        VocabSize: vqvae.bottleneck.vocab + 256, TokenCount: captionBytes + 16 * 16))
    opt = Adam(model.parameters, lr: lr)
    dataLoader = CaptionedDataLoader(
      batchSize: bs, images: try ImageIterator(imageDir: imageDir, imageSize: 256))
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
        dataLoader.state = dataState
      }
      step = state.step
    }

    dataStream = loadDataInBackground(dataLoader)
  }

  private func captionTensor(_ captions: [String]) -> Tensor {
    var textTokens = [Int](repeating: 0, count: captions.count * captionBytes)
    for (i, caption) in captions.enumerated() {
      for (j, char) in caption.utf8.enumerated() {
        if j < captionBytes {
          textTokens[i * captionBytes + j] = Int(char) + vqvae.bottleneck.vocab
        }
      }
    }
    return Tensor(data: textTokens, shape: [captions.count, captionBytes])
  }

  private func takeDataset(_ n: Int) -> AsyncMapSequence<
    AsyncPrefixSequence<AsyncStream<(Tensor, [String], DataLoader.State)>>,
    (Tensor, DataLoader.State)
  > {
    return dataStream!.prefix(n).map { [self] (images, captions, state) in
      let tokens = vqvae.bottleneck(vqvae.encoder(images)).codes.flatten(startAxis: 1)
      let textTensor = captionTensor(captions)
      let mask = (Tensor(rand: [textTensor.shape[0]]) > cfgProb).unsqueeze(axis: -1).expand(
        as: textTensor)
      let maskedText = mask.when(isTrue: textTensor, isFalse: Tensor(zerosLike: textTensor))
      let zeros = Tensor(zeros: [textTensor.shape[0], 1], dtype: textTensor.dtype)
      return (Tensor(concat: [zeros, maskedText, tokens], axis: 1), state)
    }
  }

  private func trainInnerLoop() async throws {
    print("training...")
    for await (batch, state) in takeDataset(saveInterval) {
      lastDataState = state
      step += 1

      // We do not model the caption prefix, only the VQ tokens.
      let outputs = model(batch[..., ..<(-1)])[..., captionBytes...]
      let targets = batch[..., (captionBytes + 1)...]

      let logProbs = outputs.logSoftmax(axis: -1)
      let losses = -logProbs.gather(axis: -1, indices: targets.unsqueeze(axis: -1))
      let loss = losses.mean()
      loss.backward()
      opt.step()
      opt.clearGrads()
      print("step \(step):" + " loss=\(try await loss.item()) gflops=\(gflops)")
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
