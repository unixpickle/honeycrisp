import Accelerate
import Foundation
import HCBacktrace

/// An error produced by a ``Backend`` implementation.
public enum BackendError: Error {
  case notImplemented(String)
  case failedToCreateMTLDevice
  case failedToCreateMTLBuffer
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
///
/// On a given thread, there is a current default backend, accessible via ``Backend/current``.
/// You can call ``Backend/use(_:)`` to override the backend on the current thread, or else
/// the global ``Backend/defaultBackend`` will be used.
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
  }

  private static let ThreadKey = "HONEYCRISP_CURRENT_BACKEND"

  /// The ``Backend`` that is used by default but can be overriden on a
  /// per-thread basis by ``Backend/use(_:)``.
  public static var defaultBackend: Backend = CPUBackend()

  /// The current backend in use on this thread.
  ///
  /// See ``Backend/use(_:)`` and ``Backend/defaultBackend``.
  public static var current: Backend {
    if let backend = Thread.current.threadDictionary[ThreadKey] {
      (backend as? Backend)!
    } else {
      defaultBackend
    }
  }

  /// Call the function with the current backend set as the thread-local default
  /// backend.
  ///
  /// Inside of this function, ``Backend/current`` will be `self`, unless the backend
  /// is overridden again by a nested call to this method.
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

  public init() {
  }

  /// Perform a broadcasted binary operator between two tensors.
  ///
  /// Both inputs and the output should be of type `dtype`.
  /// The count argument specifies the number of elements in the output.
  public func binaryOp(
    _ a: BroadcastData, _ b: BroadcastData, op: NumericBinaryOp, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("binaryOp")
  }

  /// Perform a binary operator between a tensor and a scalar.
  ///
  /// The `count` is the number of elements in the input and the output.
  public func binaryOp<T: NumericTensorElement>(
    _ a: Tensor.Data, _ b: T, op: NumericBinaryOp, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("binaryOp")
  }

  /// Perform a binary operator between a scalar and a tensor.
  ///
  /// The `count` is the number of elements in the input and the output.
  public func binaryOp<T: NumericTensorElement>(
    _ a: T, _ b: Tensor.Data, op: NumericBinaryOp, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("binaryOp")
  }

  /// Perform a broadcasted bitwise operator between two tensors.
  ///
  /// Both inputs and the output should be of type `dtype`.
  /// The count argument specifies the number of elements in the output.
  public func bitwiseOp(
    _ a: BroadcastData, _ b: BroadcastData, op: BitwiseOp, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("bitwiseOp")
  }

  /// Perform a bitwise operator between a tensor and a scalar.
  ///
  /// The `count` is the number of elements in the input and the output.
  public func bitwiseOp<T: TensorElementBitPattern>(
    _ a: Tensor.Data, _ b: T, op: BitwiseOp, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("bitwiseOp")
  }

  /// Perform a bitwise operator between a scalar and a tensor.
  ///
  /// The `count` is the number of elements in the input and the output.
  public func bitwiseOp<T: TensorElementBitPattern>(
    _ a: T, _ b: Tensor.Data, op: BitwiseOp, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("bitwiseOp")
  }

  /// Perform a broadcasted, fused multiply-then-add operation.
  ///
  /// All inputs and the output should be of type `dtype`.
  /// The `count` is the number of elements in the output.
  public func mulAdd(
    input: BroadcastData, coeff: BroadcastData, bias: BroadcastData, count: Int, dtype: Tensor.DType
  ) async throws -> Tensor.Data {
    throw BackendError.notImplemented("mulAdd")
  }

  /// Perform a broadcasted, fused add-then-multiply operation.
  ///
  /// All inputs and the output should be of type `dtype`.
  /// The `count` is the number of elements in the output.
  public func addMul(
    input: BroadcastData, bias: BroadcastData, coeff: BroadcastData, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("addMul")
  }

  /// Normalize an input given a (broadcasted) mean and variance.
  ///
  /// Computes
  ///
  /// ```swift
  /// (input - mean) / (variance + epsilson).sqrt()
  /// ````
  public func normalize<T: TensorElement>(
    input: BroadcastData, mean: BroadcastData, variance: BroadcastData, epsilon: T, count: Int,
    dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("normalize")
  }

  /// Compute the gradient of ``Backend/normalize(input:mean:variance:epsilon:count:dtype:)``
  /// with respect to the input or the mean (depending on `sign`).
  ///
  /// The result is the same shape as the output of the operation, not necessarily the shape
  /// of the (pre-broadcasted) input.
  public func normalizeXGrad<T: TensorElement>(
    variance: BroadcastData, outGrad: BroadcastData, epsilon: T, sign: Float, count: Int,
    dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("normalizeXGrad")
  }

  /// Compute the gradient of ``Backend/normalize(input:mean:variance:epsilon:count:dtype:)``
  /// with respect to the variance.
  ///
  /// The result is the same shape as the output of the operation, not necessarily the shape
  /// of the (pre-broadcasted) variance.
  public func normalizeVarianceGrad<T: TensorElement>(
    input: BroadcastData, mean: BroadcastData, variance: BroadcastData, outGrad: BroadcastData,
    epsilon: T, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("normalizeVarianceGrad")
  }

  /// Perform a broadcasted, elementwise comparison between two tensors, computing an
  /// output of booleans.
  public func compare(
    _ a: BroadcastData, _ b: BroadcastData, op: ComparisonOp, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("compare")
  }

  /// Perform an element-wise comparison between a tensor and a scalar, computing an
  /// output of booleans.
  public func compare<T: TensorElement>(
    _ a: Tensor.Data, _ b: T, op: ComparisonOp, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("compare")
  }

  /// Perform an element-wise comparison between a scalar and a tensor, computing an
  /// output of booleans.
  public func compare<T: TensorElement>(
    _ a: T, _ b: Tensor.Data, op: ComparisonOp, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("compare")
  }

  /// Convert the input elements into a different data-type.
  public func cast(_ a: Tensor.Data, count: Int, inType: Tensor.DType, outType: Tensor.DType)
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("cast")
  }

  /// Raise the input elements to a given scalar power, optionally scaling the result.
  ///
  /// The `scale` is applied to all output elements, while the `scales` are multiplied
  /// to the result element-wise and must have the same count as the input (if provided).
  public func pow<T: NumericTensorElement>(
    _ a: Tensor.Data, _ b: T, scale: T, scales: Tensor.Data?, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("pow")
  }

  /// Constrain the elements in the tensor between a minimum and maximum value.
  ///
  /// If `min` or `max` are nil, then this serves as a pure max or min operation.
  public func clamp<T: NumericTensorElement>(
    _ a: Tensor.Data, min: T?, max: T?, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("clamp")
  }

  /// Perform a reduction of an input tensor given the combined size of the leading, reduction,
  /// and trailing dimensions.
  public func reduce(
    _ a: Tensor.Data, op: ReduceOp, dims: ReduceDims, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("reduce")
  }

  /// Compute the log of the softmax operator along an axis of the tensor.
  public func logSoftmax(_ a: Tensor.Data, dims: ReduceDims, dtype: Tensor.DType) async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("logSoftmax")
  }

  /// Compute the gradient of the log of the softmax operator.
  public func logSoftmaxGrad(
    _ a: Tensor.Data, _ outGrad: Tensor.Data, dims: ReduceDims, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("logSoftmaxGrad")
  }

  /// Repeat (chunks of) elements of a tensor.
  public func repeated(
    _ a: Tensor.Data, dims: RepeatDims, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("repeated")
  }

  /// Select values of a tensor from indices that are specified by another tensor.
  public func gather(
    _ a: Tensor.Data, _ s: ScatterGatherIndices, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("gather")
  }

  /// Invert the operation of ``Backend/gather(_:_:dtype:)`` operation.
  ///
  /// When indices are not unique, sum the values for that index.
  public func scatter(
    _ a: Tensor.Data, _ s: ScatterGatherIndices, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("scatter")
  }

  /// Use a boolean tensor to select between values from one tensor or another.
  ///
  /// Both the true and false values can be broadcasted or may be scalars.
  public func when<T>(
    _ mask: BroadcastData, _ a: TensorOrScalar<T>, _ b: TensorOrScalar<T>, _: T.Type, count: Int,
    dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("when")
  }

  /// Compute a matrix product, possibly transposing operands and the output.
  ///
  /// The output can be defined as `transOut(transA(a) * transB(b))`, where `transX` is a function
  /// that potentially transposes the matrix.
  ///
  /// The arguments `rows` and `cols` define the shape of `trans?(a) * trans?(b)`.
  /// The `inner` argument is the number of columns in `transA(a)` or rows in `transB(b)`.
  public func matmul(
    a: Tensor.Data, transA: Bool, b: Tensor.Data, transB: Bool, transOut: Bool, rows: Int,
    inner: Int, cols: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("matmul")
  }

  /// A batched version of ``Backend/matmul(a:transA:b:transB:transOut:rows:inner:cols:dtype:)``.
  public func batchedMatmul(
    matrixCount: Int, a: Tensor.Data, transA: Bool, b: Tensor.Data, transB: Bool, transOut: Bool,
    rows: Int, inner: Int, cols: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("batchedMatmul")
  }

  /// Create a low-triangular version of a (batch of) matrices.
  public func tril(_ a: Tensor.Data, batch: Int, rows: Int, cols: Int, dtype: Tensor.DType)
    async throws -> Tensor.Data
  {
    throw BackendError.notImplemented("tril")
  }

  /// Compute a 1-dimensional convolution.
  public func conv1D(
    _ config: Conv1DConfig, batch: Int, image: Tensor.Data, kernel: Tensor.Data, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("conv1D")
  }

  /// Compute a 1-dimensional transposed convolution, or equivalently, a gradient through a
  /// 1-dimensional convolution with respect to the input.
  public func conv1DTranspose(
    _ config: Conv1DConfig, batch: Int, image: Tensor.Data, kernel: Tensor.Data, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("conv1DTranspose")
  }

  /// Compute the gradient of a 1-dimensional convolution with respect to the kernel.
  public func conv1DKernelGrad(
    _ config: Conv1DConfig, batch: Int, image: Tensor.Data, outGrad: Tensor.Data,
    dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("conv1DKernelGrad")
  }

  /// Compute a 2-dimensional convolution.
  public func conv2D(
    _ config: Conv2DConfig, batch: Int, image: Tensor.Data, kernel: Tensor.Data, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("conv2D")
  }

  /// Compute a 2-dimensional transposed convolution, or equivalently, a gradient through a
  /// 2-dimensional convolution with respect to the input.
  public func conv2DTranspose(
    _ config: Conv2DConfig, batch: Int, image: Tensor.Data, kernel: Tensor.Data, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("conv2DTranspose")
  }

  /// Compute the gradient of a 2-dimensional convolution with respect to the kernel.
  public func conv2DKernelGrad(
    _ config: Conv2DConfig, batch: Int, image: Tensor.Data, outGrad: Tensor.Data,
    dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("conv2DKernelGrad")
  }

  /// Apply an element-wise operation to a tensor, optionally multiplying the output
  /// element-wise by a tensor of scales (of the same dtype).
  public func elemwise(
    _ a: Tensor.Data, op: ElemwiseOp, scales: Tensor.Data?, count: Int, dtype: Tensor.DType
  ) async throws -> Tensor.Data {
    throw BackendError.notImplemented("elemwise")
  }

  /// Concatenate multiple tensors with per-tensor inner strides.
  public func concat(
    _ inputs: [Tensor.Data], outerCount: Int, innerCounts: [Int], dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("concat")
  }

  /// Create a tensor filled with a constant.
  public func constant<T: TensorElement>(_ value: T, count: Int, dtype: Tensor.DType) async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("constant")
  }

  /// Create a tensor from the contents of a collection.
  public func collection<T: TensorElement>(
    _ collection: some Collection<T>, reverse: Bool, dtype: Tensor.DType
  )
    async throws -> Tensor.Data
  {
    throw BackendError.notImplemented("collection")
  }

  /// Compute a permutation of integers for permuting the axes of a tensor of the given shape.
  public func axisPermutation(permutation: [Int], shape: [Int]) async throws -> Tensor.Data {
    throw BackendError.notImplemented("axisPermutation")
  }

  /// Get the default random number generator for this backend.
  public func defaultRandom() async throws -> RandomGenerator {
    throw BackendError.notImplemented("defaultRandom")
  }

  /// Create a new random number generator for this backend.
  public func createRandom() async throws -> RandomGenerator {
    throw BackendError.notImplemented("createRandom")
  }

}
