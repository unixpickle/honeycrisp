import Accelerate
import Foundation
import HCBacktrace
import Metal

public enum BackendError: Error {
  case notImplemented(String)
  case failedToCreateMTLDevice
  case failedToCreateCommandQueue
  case allocationFailed(Int)
  case kernelFailed(String)
  case failedToLoadKernels(String)
}

/// An base class for backends which can perform operations on tensor data.
///
/// This is a base class and does not implement any actual functionality.
/// Instead, it is inherited by implementations such as ``CPUBackend`` or ``MPSBackend``.
/// Additionally, a ``BackendWrapper`` can be used to patch certain functions on a wrapped
/// backend implementation.
open class Backend {

  public enum TensorOrScalar<T: TensorElement> {
    case tensor(BroadcastData)
    case scalar(T, Int)

    internal var strides: BroadcastStrides {
      switch self {
      case .tensor(let t):
        t.strides
      case .scalar(_, let s):
        BroadcastStrides(dataCount: 1, outerRepeats: s, innerRepeats: 1)
      }
    }

    internal func makeOrGetBuffer(_ b: Backend, _ dtype: Tensor.DType) async throws -> MTLBuffer {
      switch self {
      case .tensor(let t):
        try await t.data.cpuBuffer
      case .scalar(let s, _):
        try await {
          let buf = try await b.allocate(length: dtype.byteSize)
          try arrayToPointer([s], output: buf.contents(), dtype: dtype)
          return buf
        }()
      }
    }
  }

  private static let ThreadKey = "HONEYCRISP_CURRENT_BACKEND"

  public static var defaultBackend: Backend = CPUBackend()

  public static var current: Backend {
    if let backend = Thread.current.threadDictionary[ThreadKey] {
      (backend as? Backend)!
    } else {
      defaultBackend
    }
  }

  public func use<T>(_ fn: () throws -> T) rethrows -> T {
    if let backend = Thread.current.threadDictionary[Backend.ThreadKey] {
      let old = (backend as? Backend)!
      defer {
        Thread.current.threadDictionary[Backend.ThreadKey] = old
      }
      Thread.current.threadDictionary[Backend.ThreadKey] = self
      return try fn()
    } else {
      defer {
        Thread.current.threadDictionary.removeObject(forKey: Backend.ThreadKey)
      }
      Thread.current.threadDictionary[Backend.ThreadKey] = self
      return try fn()
    }
  }

  /// A lightweight thread-safe FIFO queue.
  public class Queue<T: Sendable> {
    private var items: [T] = []
    private var lock: NSLock = NSLock()
    private var closed: Bool = false
    private var sem: DispatchSemaphore = DispatchSemaphore(value: 0)

    /// Push an item to the queue.
    /// This should never be called after close().
    public func put(_ x: T) {
      lock.lock()
      alwaysAssert(!closed, "cannot put() on queue after closing")
      items.insert(x, at: 0)
      lock.unlock()
      sem.signal()
    }

    /// Wait for the next item on the queue.
    ///
    /// Returns nil forever once the queue has been closed and depleted.
    public func get() -> T? {
      sem.wait()
      lock.lock()
      let item = items.popLast()
      if closed {
        // We don't want to block future get() calls.
        sem.signal()
      }
      lock.unlock()
      return item
    }

    /// Finish writing to the queue and unblock all future get() calls.
    public func close() {
      lock.lock()
      closed = true
      lock.unlock()
      sem.signal()
    }
  }

  /// A background thread which executes blocks serially.
  ///
  /// Once this is deinitialized, the background thread will complete its
  /// remaining work and then exit.
  public class WorkerThread {
    public typealias Job = () -> Void

    private let queue = Queue<Job>()
    private var thread: Thread? = nil

    deinit {
      queue.close()
    }

    public init() {
      thread = Thread { [queue = self.queue] in
        while true {
          guard let job = queue.get() else {
            return
          }
          autoreleasepool {
            job()
          }
        }
      }
      thread!.name = "Backend.WorkerThread"
      thread!.start()
    }

    public func schedule(_ job: @escaping Job) {
      queue.put(job)
    }
  }

  public func allocate(length: Int) async throws -> MTLBuffer {
    throw BackendError.notImplemented("allocate")
  }

  public func binaryOp(
    _ a: BroadcastData, _ b: BroadcastData, op: NumericBinaryOp, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("binaryOp")
  }

  public func binaryOp<T: NumericTensorElement>(
    _ a: Tensor.Data, _ b: T, op: NumericBinaryOp, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("binaryOp")
  }

  public func binaryOp<T: NumericTensorElement>(
    _ a: T, _ b: Tensor.Data, op: NumericBinaryOp, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("binaryOp")
  }

  public func bitwiseOp(
    _ a: BroadcastData, _ b: BroadcastData, op: BitwiseOp, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("bitwiseOp")
  }

  public func bitwiseOp<T: TensorElementBitPattern>(
    _ a: Tensor.Data, _ b: T, op: BitwiseOp, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("bitwiseOp")
  }

  public func bitwiseOp<T: TensorElementBitPattern>(
    _ a: T, _ b: Tensor.Data, op: BitwiseOp, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("bitwiseOp")
  }

  public func mulAdd(
    input: BroadcastData, coeff: BroadcastData, bias: BroadcastData, count: Int, dtype: Tensor.DType
  ) async throws -> Tensor.Data {
    throw BackendError.notImplemented("mulAdd")
  }

  public func addMul(
    input: BroadcastData, bias: BroadcastData, coeff: BroadcastData, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("addMul")
  }

  public func normalize<T: TensorElement>(
    input: BroadcastData, mean: BroadcastData, variance: BroadcastData, epsilon: T, count: Int,
    dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("normalize")
  }

  public func normalizeXGrad<T: TensorElement>(
    variance: BroadcastData, outGrad: BroadcastData, epsilon: T, sign: Float, count: Int,
    dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("normalizeXGrad")
  }

  public func normalizeVarianceGrad<T: TensorElement>(
    input: BroadcastData, mean: BroadcastData, variance: BroadcastData, outGrad: BroadcastData,
    epsilon: T, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("normalizeVarianceGrad")
  }

  public func compare(
    _ a: BroadcastData, _ b: BroadcastData, op: ComparisonOp, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("compare")
  }

  public func compare<T: TensorElement>(
    _ a: Tensor.Data, _ b: T, op: ComparisonOp, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("compare")
  }

  public func compare<T: TensorElement>(
    _ a: T, _ b: Tensor.Data, op: ComparisonOp, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("compare")
  }

  public func cast(_ a: Tensor.Data, count: Int, inType: Tensor.DType, outType: Tensor.DType)
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("cast")
  }

  public func pow<T: NumericTensorElement>(
    _ a: Tensor.Data, _ b: T, scale: T, scales: Tensor.Data?, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("pow")
  }

  public func clamp<T: NumericTensorElement>(
    _ a: Tensor.Data, min: T?, max: T?, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("clamp")
  }

  public func reduce(
    _ a: Tensor.Data, op: ReduceOp, dims: ReduceDims, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("reduce")
  }

  public func logSoftmax(
    _ a: Tensor.Data, outerCount: Int, middleCount: Int, innerCount: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("logSoftmax")
  }

  public func logSoftmaxGrad(
    _ a: Tensor.Data, _ outGrad: Tensor.Data, outerCount: Int, middleCount: Int, innerCount: Int,
    dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("logSoftmaxGrad")
  }

  public func repeated(
    _ a: Tensor.Data, dims: RepeatDims, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("repeated")
  }

  public func gather(
    _ a: Tensor.Data, _ s: ScatterGatherIndices, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("gather")
  }

  public func scatter(
    _ a: Tensor.Data, _ s: ScatterGatherIndices, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("scatter")
  }

  public func when<T>(
    _ mask: BroadcastData, _ a: TensorOrScalar<T>, _ b: TensorOrScalar<T>, _: T.Type, count: Int,
    dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("when")
  }

  public func matmul(
    a: Tensor.Data, transA: Bool, b: Tensor.Data, transB: Bool, transOut: Bool, rows: Int,
    inner: Int, cols: Int,
    dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("matmul")
  }

  public func batchedMatmul(
    matrixCount: Int, a: Tensor.Data, transA: Bool, b: Tensor.Data, transB: Bool, transOut: Bool,
    rows: Int, inner: Int, cols: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("batchedMatmul")
  }

  public func tril(_ a: Tensor.Data, batch: Int, rows: Int, cols: Int, dtype: Tensor.DType)
    async throws -> Tensor.Data
  {
    throw BackendError.notImplemented("tril")
  }

  public func conv1D(
    _ config: Conv1DConfig, batch: Int, image: Tensor.Data, kernel: Tensor.Data, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("conv1D")
  }

  public func conv1DTranspose(
    _ config: Conv1DConfig, batch: Int, image: Tensor.Data, kernel: Tensor.Data, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("conv1DTranspose")
  }

  public func conv1DKernelGrad(
    _ config: Conv1DConfig, batch: Int, image: Tensor.Data, outGrad: Tensor.Data,
    dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("conv1DKernelGrad")
  }

  public func conv2D(
    _ config: Conv2DConfig, batch: Int, image: Tensor.Data, kernel: Tensor.Data, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("conv2D")
  }

  public func conv2DTranspose(
    _ config: Conv2DConfig, batch: Int, image: Tensor.Data, kernel: Tensor.Data, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("conv2DTranspose")
  }

  public func conv2DKernelGrad(
    _ config: Conv2DConfig, batch: Int, image: Tensor.Data, outGrad: Tensor.Data,
    dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("conv2DKernelGrad")
  }

  public func elemwise(
    _ a: Tensor.Data, op: ElemwiseOp, scales: Tensor.Data?, count: Int, dtype: Tensor.DType
  ) async throws -> Tensor.Data {
    throw BackendError.notImplemented("elemwise")
  }

  public func concat(
    _ inputs: [Tensor.Data], outerCount: Int, innerCounts: [Int], dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("concat")
  }

  public func constant<T: TensorElement>(_ value: T, count: Int, dtype: Tensor.DType) async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("constant")
  }

  public func collection<T: TensorElement>(
    _ collection: some Collection<T>, reverse: Bool, dtype: Tensor.DType
  )
    async throws -> Tensor.Data
  {
    throw BackendError.notImplemented("collection")
  }

  public func axisPermutation(permutation: [Int], shape: [Int]) async throws -> Tensor.Data {
    throw BackendError.notImplemented("axisPermutation")
  }

  public func defaultRandom() async throws -> RandomGenerator {
    throw BackendError.notImplemented("defaultRandom")
  }

  public func createRandom() async throws -> RandomGenerator {
    throw BackendError.notImplemented("createRandom")
  }

}
