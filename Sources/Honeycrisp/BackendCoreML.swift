import CoreML
import CoreMLBuilder

open class CoreMLBackend: BackendWrapper {

  internal struct MatmulKey: Hashable {
    public let sizeA0: Int
    public let sizeA1: Int
    public let sizeB0: Int
    public let sizeB1: Int
    public let transA: Bool
    public let transB: Bool
    public let isFloat32: Bool
  }

  internal class ModelCache {
    private let computeUnits: MLComputeUnits
    private var matmuls: [MatmulKey: MLModel] = [:]

    public init(computeUnits: MLComputeUnits) {
      self.computeUnits = computeUnits
    }

    public func matmul(_ key: MatmulKey) async throws -> MLModel {
      if let model = matmuls[key] {
        return model
      }

      let matmul = Matmul(
        xShape: (Int64(key.sizeA0), Int64(key.sizeA1)),
        yShape: (Int64(key.sizeB0), Int64(key.sizeB1)),
        transposeX: key.transA,
        transposeY: key.transB,
        dtype: key.isFloat32 ? .float32 : .float16
      )
      let model = try await matmul.model(computeUnits: computeUnits)
      matmuls[key] = model
      return model
    }
  }

  internal typealias Job = ((ModelCache) async throws -> MLModel, (Result<MLModel, Error>) -> Void)
  internal let queue: Queue<Job> = Queue()
  public let computeUnits: MLComputeUnits

  deinit {
    queue.close()
  }

  public init(
    wrapping: Backend, threads: Int = 8, computeUnits: MLComputeUnits = .cpuAndNeuralEngine
  ) {
    self.computeUnits = computeUnits
    super.init(wrapping: wrapping)
    for i in 0..<threads {
      let thread = Thread { [queue = self.queue] in
        let cache = ModelCache(computeUnits: computeUnits)
        while let (buildModel, useModel) = queue.get() {
          let q1 = Queue<Result<MLModel, Error>>()
          Task.detached {
            do {
              let model = try await buildModel(cache)
              q1.put(.success(model))
            } catch {
              q1.put(.failure(error))
            }
          }
          autoreleasepool {
            useModel(q1.get()!)
          }
        }
      }
      thread.name = "BackendCoreML\(i)"
      thread.start()
    }
  }

  internal func runModel<T: Sendable>(
    buildModel: @escaping (ModelCache) async throws -> MLModel,
    useModel: @escaping (MLModel) throws -> T
  ) async throws -> T {
    try await withCheckedThrowingContinuation { continuation in
      queue.put(
        (
          buildModel,
          { result in
            switch result {
            case .success(let model):
              var result: Result<T, Error>?
              do {
                result = Result.success(try useModel(model))
              } catch {
                result = Result.failure(error)
              }
              let constResult = result!
              continuation.resume(with: constResult)
            case .failure(let error):
              continuation.resume(throwing: error)
            }
          }
        ))
    }
  }

  internal func waitForData(_ xs: Tensor.Data...) async throws {
    for x in xs {
      if let waiter = x.completeOnAllDevices {
        try await waiter.value
      }
    }
  }

  override public func matmul(
    a: Tensor.Data, transA: Bool, b: Tensor.Data, transB: Bool, transOut: Bool, rows: Int,
    inner: Int, cols: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    return try await batchedMatmul(
      matrixCount: 1, a: a, transA: transA, b: b, transB: transB, transOut: transOut, rows: rows,
      inner: inner, cols: cols, dtype: dtype)
  }

  override public func batchedMatmul(
    matrixCount: Int, a: Tensor.Data, transA: Bool, b: Tensor.Data, transB: Bool, transOut: Bool,
    rows: Int, inner: Int, cols: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    if dtype != .float32 && dtype != .float16 {
      return try await wrapped.batchedMatmul(
        matrixCount: matrixCount, a: a, transA: transA, b: b, transB: transB, transOut: transOut,
        rows: rows, inner: inner, cols: cols, dtype: dtype)
    } else if transOut {
      return try await batchedMatmul(
        matrixCount: matrixCount, a: b, transA: !transB, b: a, transB: !transA, transOut: false,
        rows: cols, inner: inner,
        cols: rows, dtype: dtype)
    }

    try await waitForData(a, b)

    let aCount = rows * inner
    let bCount = inner * cols
    let outCount = rows * cols
    let buffer = try await wrapped.allocate(length: matrixCount * outCount * dtype.byteSize)

    let matmulKey = MatmulKey(
      sizeA0: (transA ? inner : rows), sizeA1: (transA ? rows : inner),
      sizeB0: transB ? cols : inner, sizeB1: transB ? inner : cols, transA: transA,
      transB: transB, isFloat32: dtype == .float32)

    var tasks = [Task<(), Error>]()
    for i in 0..<matrixCount {
      let i = i
      // Chunking over the batch dimension to multi-thread the ANE calls
      // tends to improve performance.
      let aChunk = transA ? matmulKey.sizeA0 : min(1024, matmulKey.sizeA0)
      for aIdx in stride(from: 0, to: matmulKey.sizeA0, by: aChunk) {
        let aSize = min(matmulKey.sizeA0 - aIdx, aChunk)
        tasks.append(
          Task.detached { [self] in
            let mmKey = MatmulKey(
              sizeA0: aSize, sizeA1: (transA ? rows : inner),
              sizeB0: transB ? cols : inner, sizeB1: transB ? inner : cols, transA: transA,
              transB: transB, isFloat32: dtype == .float32)
            try await runModel(buildModel: { mc in try await mc.matmul(mmKey) }) { model in
              let x = try MLMultiArray(
                dataPointer: a.buffer.contents().advanced(
                  by: (i * aCount + aIdx * matmulKey.sizeA1) * dtype.byteSize),
                shape: mpsShape([aSize, matmulKey.sizeA1]),
                dataType: matmulKey.isFloat32 ? .float32 : .float16,
                strides: mpsShape([matmulKey.sizeA1, 1]),
                deallocator: { [a = a.buffer] _ in _ = a }
              )
              let y = try MLMultiArray(
                dataPointer: b.buffer.contents().advanced(by: i * bCount * dtype.byteSize),
                shape: mpsShape([matmulKey.sizeB0, matmulKey.sizeB1]),
                dataType: x.dataType,
                strides: mpsShape([matmulKey.sizeB1, 1]),
                deallocator: { [b = b.buffer] _ in _ = b }
              )
              let featureProvider: MLFeatureProvider = try MLDictionaryFeatureProvider(
                dictionary: [
                  "x": MLFeatureValue(multiArray: x),
                  "y": MLFeatureValue(multiArray: y),
                ])
              let options = MLPredictionOptions()
              options.outputBackings = [
                "output": try MLMultiArray(
                  dataPointer: buffer.contents().advanced(
                    by: (i * outCount + aIdx * cols) * dtype.byteSize),
                  shape: mpsShape([transA ? rows : min(aChunk, rows - aIdx), cols]),
                  dataType: x.dataType,
                  strides: mpsShape([cols, 1]),
                  deallocator: { [b = buffer] _ in _ = b }
                )
              ]
              try model.prediction(from: featureProvider, options: options)
            }
          })
      }
    }
    for task in tasks {
      let _ = try await task.value
    }
    return Tensor.Data(backend: self, buffer: buffer)
  }

}
