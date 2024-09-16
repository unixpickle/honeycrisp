import Accelerate
import Foundation
import Metal
import MetalPerformanceShaders

open class MPSBackend: CPUBackend {

  private struct MatmulKey: Hashable {
    let transA: Bool
    let transB: Bool
    let rows: Int
    let cols: Int
    let inner: Int
  }

  public class MPSRandomGenerator: NativeRandomGenerator {
    public let mpsBackend: MPSBackend
    private var seed: Int

    init(mpsBackend: MPSBackend, seed: Int) {
      self.mpsBackend = mpsBackend
      self.seed = seed
      super.init(cpuBackend: mpsBackend)
    }

    override public func save() async throws -> Data {
      let seed = try await mpsBackend.serialize {
        return UInt64(self.seed)
      }
      return Data((0..<8).map { UInt8((seed >> (8 * $0)) & 0xff) })
    }

    override public func restore(_ x: Data) async throws {
      var result: UInt64 = 0
      for (i, x) in x.enumerated() {
        result |= UInt64(x) << (8 * i)
      }
      try await mpsBackend.serialize {
        seed = Int(result)
      }
    }

    override public func seed(_ seed: Int) async throws {
      try await mpsBackend.serialize {
        self.seed = seed
      }
    }

    override public func sample(count: Int, dist: RandomDist, dtype: Tensor.DType) async throws
      -> Tensor.Data
    {
      if dtype != .float32 && dtype != .float16 {
        return try await super.sample(count: count, dist: dist, dtype: dtype)
      }

      let buffer = try await backend.allocate(length: count * dtype.byteSize)
      return try await mpsBackend.serialize {
        let completion = mpsBackend.completionBuffer { buf in
          let rng = MPSMatrixRandomPhilox(
            device: try! mpsBackend.device,
            destinationDataType: dtype == .float32 ? .float32 : .float16, seed: seed,
            distributionDescriptor: dist == .normal
              ? .normalDistributionDescriptor(withMean: 0, standardDeviation: 1)
              : .uniformDistributionDescriptor(withMinimum: 0, maximum: 1))
          seed += 1
          rng.encode(
            commandBuffer: buf,
            destinationVector: MPSVector(
              buffer: buffer,
              descriptor: MPSVectorDescriptor(
                length: count, dataType: (dtype == .float32 ? .float32 : .float16))))
        }
        return Tensor.Data(backend: mpsBackend, buffer: buffer, completeOnAllDevices: completion)
      }
    }
  }

  private var queue = DispatchQueue(label: "mps-backend-worker")
  private var commandQueue: MTLCommandQueue? = nil
  private var matmuls: [MatmulKey: MPSMatrixMultiplication] = [:]
  private var defaultRNG: MPSRandomGenerator? = nil

  public init(device: MTLDevice? = nil, commandQueue: MTLCommandQueue? = nil) throws {
    super.init()
    if let device = device {
      self._device = device
      if let commandQueue = commandQueue {
        self.commandQueue = commandQueue
      } else {
        if let q = device.makeCommandQueue() {
          self.commandQueue = q
        } else {
          throw BackendError.failedToCreateCommandQueue
        }
      }
    } else {
      guard let d = MTLCreateSystemDefaultDevice() else {
        throw BackendError.failedToCreateMTLDevice
      }
      self._device = d
      guard let q = d.makeCommandQueue() else {
        throw BackendError.failedToCreateCommandQueue
      }
      self.commandQueue = q
    }
  }

  internal func waitForGPUData(_ xs: Tensor.Data...) async throws {
    for x in xs {
      if x.backend === self {
        continue
      }
      try await waitForData(x)
    }
  }

  override public func matmul(
    a: Tensor.Data, transA: Bool, b: Tensor.Data, transB: Bool, transOut: Bool, rows: Int,
    inner: Int, cols: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    if dtype != .float32 && dtype != .float16 {
      return try await super.matmul(
        a: a, transA: transA, b: b, transB: transB, transOut: transOut, rows: rows, inner: inner,
        cols: cols, dtype: dtype)
    } else if transOut {
      return try await matmul(
        a: b, transA: !transB, b: a, transB: !transA, transOut: false, rows: cols, inner: inner,
        cols: rows, dtype: dtype)
    }

    let aShape = transA ? (inner, rows) : (rows, inner)
    let bShape = transB ? (cols, inner) : (inner, cols)
    let mpsDtype = dtype == .float32 ? MPSDataType.float32 : MPSDataType.float16

    let output = try await allocate(length: rows * cols * dtype.byteSize)
    try await waitForGPUData(a, b)
    return try await serialize {
      let mm = self.createMatmul(
        transA: transA, transB: transB, rows: rows, inner: inner, cols: cols)
      let completion = completionBuffer { buf in
        mm.encode(
          commandBuffer: buf,
          leftMatrix: MPSMatrix(
            buffer: a.buffer,
            descriptor: MPSMatrixDescriptor(
              rows: aShape.0, columns: aShape.1, rowBytes: aShape.1 * dtype.byteSize,
              dataType: mpsDtype)),
          rightMatrix: MPSMatrix(
            buffer: b.buffer,
            descriptor: MPSMatrixDescriptor(
              rows: bShape.0, columns: bShape.1, rowBytes: bShape.1 * dtype.byteSize,
              dataType: mpsDtype)),
          resultMatrix: MPSMatrix(
            buffer: output,
            descriptor: MPSMatrixDescriptor(
              rows: rows, columns: cols, rowBytes: cols * dtype.byteSize, dataType: mpsDtype))
        )
      }
      return Tensor.Data(backend: self, buffer: output, completeOnAllDevices: completion)
    }
  }

  override public func defaultRandom() async throws -> RandomGenerator {
    try await serialize {
      if let d = defaultRNG {
        return d
      } else {
        defaultRNG = MPSRandomGenerator(mpsBackend: self, seed: Int.random(in: 0..<1_000_000_000))
        return defaultRNG!
      }
    }
  }

  override public func createRandom() async throws -> RandomGenerator {
    try await serialize {
      MPSRandomGenerator(mpsBackend: self, seed: Int.random(in: 0..<1_000_000_000))
    }
  }

  internal func completionBuffer(_ action: (MTLCommandBuffer) throws -> Void) rethrows -> Task<
    (), Error
  > {
    let buf = commandQueue!.makeCommandBuffer()!
    try action(buf)
    return Task {
      return try await withCheckedThrowingContinuation { continuation in
        buf.addCompletedHandler { _ in
          if let e = buf.error {
            continuation.resume(throwing: e)
          } else {
            continuation.resume(returning: ())
          }
        }
        // I'm not sure we need to commit() from the main queue,
        // but let's do it to be safe.
        Task {
          try await serialize { buf.commit() }
        }
      }
    }
  }

  private func createMatmul(transA: Bool, transB: Bool, rows: Int, inner: Int, cols: Int)
    -> MPSMatrixMultiplication
  {
    let key = MatmulKey(transA: transA, transB: transB, rows: rows, cols: cols, inner: inner)
    if let matmul = matmuls[key] {
      return matmul
    } else {
      let mm = MPSMatrixMultiplication(
        device: try! device, transposeLeft: transA, transposeRight: transB, resultRows: rows,
        resultColumns: cols, interiorColumns: inner, alpha: 1, beta: 0)
      matmuls[key] = mm
      return mm
    }
  }

}
