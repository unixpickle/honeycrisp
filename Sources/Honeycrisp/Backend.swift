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
  case allocationFailedForArgument(Int)
  case kernelFailed(String)
  case failedToLoadKernels(String)
  case lapackError(String)
}

/// An base class for backends which can perform operations on tensor data.
///
/// This is a base class and does not implement any actual functionality.
/// Instead, it is inherited by implementations such as ``CPUBackend`` or ``MPSBackend``.
/// Additionally, a ``BackendWrapper`` can be used to patch certain functions on a wrapped
/// backend implementation.
///
/// On a given thread, there is a current default backend, accessible via ``Backend/current``.
/// You can call ``Backend/use(_:)-79uzk`` to override the backend on the current thread, or else
/// the global ``Backend/defaultBackend`` will be used.
open class Backend: @unchecked Sendable {

  public enum TensorOrScalar<T: TensorElement>: Sendable {
    case tensor(BroadcastData)
    case scalar(T, [Int])

    internal var strides: BroadcastStrides {
      switch self {
      case .tensor(let t):
        t.strides
      case .scalar(_, let shape):
        BroadcastStrides(shape: shape, strides: [Int](repeating: 0, count: shape.count))
      }
    }
  }

  @TaskLocal
  private static var taskCurrentBackend: Backend? = nil

  nonisolated(unsafe) private static var _defaultBackend: Backend = CPUBackend()
  private static let _defaultBackendLock = NSLock()

  /// The ``Backend`` that is used by default but can be overriden on a
  /// per-thread basis by ``Backend/use(_:)-79uzk``.
  public static var defaultBackend: Backend {
    get {
      _defaultBackendLock.withLock { _defaultBackend }
    }
    set {
      _defaultBackendLock.withLock { _defaultBackend = newValue }
    }
  }

  /// The current backend in use on this thread.
  ///
  /// See ``Backend/use(_:)-79uzk`` and ``Backend/defaultBackend``.
  public static var current: Backend {
    return taskCurrentBackend ?? defaultBackend
  }

  /// Call the function with the current backend set as the thread-local default
  /// backend.
  ///
  /// Inside of this function, ``Backend/current`` will be `self`, unless the backend
  /// is overridden again by a nested call to this method.
  public func use<T>(_ fn: () throws -> T) rethrows -> T {
    try Backend.$taskCurrentBackend.withValue(self) {
      try fn()
    }
  }

  /// Call the function with the current backend set as the thread-local default
  /// backend.
  ///
  /// Inside of this function, ``Backend/current`` will be `self`, unless the backend
  /// is overridden again by a nested call to this method.
  public func use<T>(_ fn: () async throws -> T) async rethrows -> T {
    try await Backend.$taskCurrentBackend.withValue(self) {
      try await fn()
    }
  }

  /// A lightweight thread-safe FIFO queue.
  public final class Queue<T: Sendable>: @unchecked Sendable {
    private var items: [T] = []
    private var lock: NSLock = NSLock()
    private var closed: Bool = false
    private var sem: DispatchSemaphore = DispatchSemaphore(value: 0)

    /// Push an item to the queue.
    /// This should never be called after close().
    public func put(_ x: T) {
      lock.lock()
      #alwaysAssert(!closed, "cannot put() on queue after closing")
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
    public typealias Job = @Sendable () -> Void

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

  /// Convert broadcasted data to a plain Tensor.Data object to be used in
  /// operations that do not support broadcasting.
  open func broadcast(_ a: BroadcastData, dtype: Tensor.DType) async throws -> Tensor.Data {
    throw BackendError.notImplemented("broadcast")
  }

  /// Perform a broadcasted binary operator between two tensors.
  ///
  /// Both inputs and the output should be of type `dtype`.
  /// The count argument specifies the number of elements in the output.
  open func binaryOp(
    _ a: BroadcastData, _ b: BroadcastData, op: NumericBinaryOp, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("binaryOp")
  }

  /// Perform a binary operator between a tensor and a scalar.
  ///
  /// The `count` is the number of elements in the input and the output.
  open func binaryOp<T: NumericTensorElement>(
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
  open func binaryOp<T: NumericTensorElement>(
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
  open func bitwiseOp(
    _ a: BroadcastData, _ b: BroadcastData, op: BitwiseOp, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("bitwiseOp")
  }

  /// Perform a bitwise operator between a tensor and a scalar.
  ///
  /// The `count` is the number of elements in the input and the output.
  open func bitwiseOp<T: TensorElementBitPattern>(
    _ a: Tensor.Data, _ b: T, op: BitwiseOp, count: Int, dtype: Tensor.DType
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
  open func mulAdd(
    input: BroadcastData, coeff: BroadcastData, bias: BroadcastData, dtype: Tensor.DType
  ) async throws -> Tensor.Data {
    throw BackendError.notImplemented("mulAdd")
  }

  /// Perform a broadcasted, fused add-then-multiply operation.
  ///
  /// All inputs and the output should be of type `dtype`.
  /// The `count` is the number of elements in the output.
  open func addMul(
    input: BroadcastData, bias: BroadcastData, coeff: BroadcastData, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("addMul")
  }

  /// Normalize an input along a given axis.
  ///
  /// Uses the formula `(x - mu) / sqrt(sigma^2 + eps)`
  /// where sigma^2 is estimated without bias correction.
  ///
  /// Defaults to using other primitives to implement this operation.
  open func normalize<T: NumericTensorElement>(
    input: Tensor.Data, dims: ReduceDims, eps: T, dtype: Tensor.DType
  )
    async throws -> Tensor.Data
  {
    let fullStrides = BroadcastStrides(contiguousForShape: [
      dims.outerCount, dims.reduceCount, dims.innerCount,
    ])
    let reducedStrides = BroadcastStrides(
      shape: fullStrides.shape, strides: [dims.innerCount, 0, 1])
    let mean = try await binaryOp(
      try await reduce(input, op: .sum, dims: dims, dtype: dtype),
      1.0 / Float(dims.reduceCount),
      op: .mul,
      count: reducedStrides.dataCount,
      dtype: dtype
    )
    let centered = try await binaryOp(
      BroadcastData(strides: fullStrides, data: input),
      BroadcastData(strides: reducedStrides, data: mean),
      op: .sub,
      dtype: dtype
    )
    let variance = try await binaryOp(
      try await reduce(
        try await pow(
          centered,
          2.0,
          scale: 1.0,
          scales: nil,
          count: fullStrides.dataCount,
          dtype: dtype
        ),
        op: .sum,
        dims: dims,
        dtype: dtype
      ),
      1.0 / Float(dims.reduceCount),
      op: .mul,
      count: reducedStrides.dataCount,
      dtype: dtype
    )
    let normalizer = try await pow(
      try await binaryOp(
        variance,
        eps,
        op: .add,
        count: reducedStrides.dataCount,
        dtype: dtype
      ),
      -0.5,
      scale: 1.0,
      scales: nil,
      count: reducedStrides.dataCount,
      dtype: dtype
    )
    return try await binaryOp(
      BroadcastData(strides: fullStrides, data: centered),
      BroadcastData(strides: reducedStrides, data: normalizer),
      op: .mul,
      dtype: dtype
    )
  }

  /// Compute the gradient of the normalize() operation.
  ///
  /// Defaults to using other primitives to implement this operation.
  open func normalizeGrad<T: NumericTensorElement>(
    input: Tensor.Data, outGrad: Tensor.Data, dims: ReduceDims, eps: T, dtype: Tensor.DType
  )
    async throws -> Tensor.Data
  {
    let fullStrides = BroadcastStrides(contiguousForShape: [
      dims.outerCount, dims.reduceCount, dims.innerCount,
    ])
    let reducedStrides = BroadcastStrides(
      shape: fullStrides.shape, strides: [dims.innerCount, 0, 1])
    let mean = try await binaryOp(
      try await reduce(input, op: .sum, dims: dims, dtype: dtype),
      1.0 / Float(dims.reduceCount),
      op: .mul,
      count: reducedStrides.dataCount,
      dtype: dtype
    )
    let centered = try await binaryOp(
      BroadcastData(strides: fullStrides, data: input),
      BroadcastData(strides: reducedStrides, data: mean),
      op: .sub,
      dtype: dtype
    )
    let variance = try await binaryOp(
      try await reduce(
        try await pow(
          centered,
          2.0,
          scale: 1.0,
          scales: nil,
          count: fullStrides.dataCount,
          dtype: dtype
        ),
        op: .sum,
        dims: dims,
        dtype: dtype
      ),
      1.0 / Float(dims.reduceCount),
      op: .mul,
      count: reducedStrides.dataCount,
      dtype: dtype
    )
    let stableVariance = try await binaryOp(
      variance,
      eps,
      op: .add,
      count: reducedStrides.dataCount,
      dtype: dtype
    )
    let normalizer1 = try await pow(
      stableVariance,
      -0.5,
      scale: 1.0,
      scales: nil,
      count: reducedStrides.dataCount,
      dtype: dtype
    )
    let normalizer2 = try await pow(
      stableVariance,
      -1.5,
      scale: 1 / Float(dims.reduceCount),
      scales: nil,
      count: reducedStrides.dataCount,
      dtype: dtype
    )

    let meanGrad = try await binaryOp(
      try await reduce(
        outGrad,
        op: .sum,
        dims: dims,
        dtype: dtype
      ),
      1 / Float(dims.reduceCount),
      op: .mul,
      count: reducedStrides.dataCount,
      dtype: dtype
    )
    let centeredGrad = try await binaryOp(
      BroadcastData(strides: fullStrides, data: outGrad),
      BroadcastData(strides: reducedStrides, data: meanGrad),
      op: .sub,
      dtype: dtype
    )

    let covTerm = try await reduce(
      try await binaryOp(
        BroadcastData(strides: fullStrides, data: centered),
        BroadcastData(strides: fullStrides, data: outGrad),
        op: .mul,
        dtype: dtype
      ),
      op: .sum,
      dims: dims,
      dtype: dtype
    )

    let term1 = try await binaryOp(
      BroadcastData(strides: fullStrides, data: centeredGrad),
      BroadcastData(strides: reducedStrides, data: normalizer1),
      op: .mul,
      dtype: dtype
    )
    let covTermTimesCentered = try await binaryOp(
      BroadcastData(strides: fullStrides, data: centered),
      BroadcastData(strides: reducedStrides, data: covTerm),
      op: .mul,
      dtype: dtype
    )
    let term2 = try await binaryOp(
      BroadcastData(strides: fullStrides, data: covTermTimesCentered),
      BroadcastData(strides: reducedStrides, data: normalizer2),
      op: .mul,
      dtype: dtype
    )
    return try await binaryOp(
      BroadcastData(strides: fullStrides, data: term1),
      BroadcastData(strides: fullStrides, data: term2),
      op: .sub,
      dtype: dtype
    )
  }

  /// Perform a broadcasted, elementwise comparison between two tensors, computing an
  /// output of booleans.
  open func compare(
    _ a: BroadcastData, _ b: BroadcastData, op: ComparisonOp, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("compare")
  }

  /// Perform an element-wise comparison between a tensor and a scalar, computing an
  /// output of booleans.
  open func compare<T: TensorElement>(
    _ a: Tensor.Data, _ b: T, op: ComparisonOp, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("compare")
  }

  /// Perform an element-wise comparison between a scalar and a tensor, computing an
  /// output of booleans.
  open func compare<T: TensorElement>(
    _ a: T, _ b: Tensor.Data, op: ComparisonOp, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("compare")
  }

  /// Convert the input elements into a different data-type.
  open func cast(_ a: Tensor.Data, count: Int, inType: Tensor.DType, outType: Tensor.DType)
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("cast")
  }

  /// Raise the input elements to a given scalar power, optionally scaling the result.
  ///
  /// The `scale` is applied to all output elements, while the `scales` are multiplied
  /// to the result element-wise and must have the same count as the input (if provided).
  open func pow<T: NumericTensorElement>(
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
  open func clamp<T: NumericTensorElement>(
    _ a: BroadcastData, _: T.Type, min: TensorOrScalar<T>?, max: TensorOrScalar<T>?,
    dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("clamp")
  }

  /// Perform a reduction of an input tensor given the combined size of the leading, reduction,
  /// and trailing dimensions.
  open func reduce(
    _ a: Tensor.Data, op: ReduceOp, dims: ReduceDims, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("reduce")
  }

  /// Sort the array along an axis, returning the indices of the permutation
  /// to sort the array in the given order.
  open func argsort(
    _ a: Tensor.Data, dims: ReduceDims, descending: Bool, stable: Bool, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("argsort")
  }

  // Perform a cumulative sum along an axis.
  //
  // If exclusive is true, then the sum is effectively offset by 1.
  // If reverse is true, then the axis is effectively reversed before and after the operation.
  open func cumulativeSum(
    _ a: Tensor.Data, dims: ReduceDims, exclusive: Bool, reverse: Bool, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("cumulativeSum")
  }

  // Perform a cumulative product along an axis.
  //
  // If exclusive is true, then the product is effectively offset by 1.
  // If reverse is true, then the axis is effectively reversed before and after the operation.
  open func cumulativeProd(
    _ a: Tensor.Data, dims: ReduceDims, exclusive: Bool, reverse: Bool, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("cumulativeProd")
  }

  /// Compute the log of the softmax operator along an axis of the tensor.
  open func logSoftmax(_ a: Tensor.Data, dims: ReduceDims, dtype: Tensor.DType) async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("logSoftmax")
  }

  /// Compute the gradient of the log of the softmax operator.
  open func logSoftmaxGrad(
    _ a: Tensor.Data, _ outGrad: Tensor.Data, dims: ReduceDims, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("logSoftmaxGrad")
  }

  /// Repeat (chunks of) elements of a tensor.
  ///
  /// The default implementation uses broadcasting.
  open func repeated(
    _ a: Tensor.Data, dims: RepeatDims, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await broadcast(
      BroadcastData(
        strides: BroadcastStrides(
          shape: [dims.outerCount, dims.repeatCount, dims.innerCount],
          strides: [dims.innerCount, 0, 1]
        ),
        data: a
      ),
      dtype: dtype
    )
  }

  /// Select values of a tensor from indices that are specified by another tensor.
  open func gather(
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
  open func scatter(
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
  open func when<T>(
    _ mask: BroadcastData, _ a: TensorOrScalar<T>, _ b: TensorOrScalar<T>, _: T.Type,
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
  open func matmul(
    a: Tensor.Data, transA: Bool, b: Tensor.Data, transB: Bool, transOut: Bool, rows: Int,
    inner: Int, cols: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("matmul")
  }

  /// A batched version of ``Backend/matmul(a:transA:b:transB:transOut:rows:inner:cols:dtype:)``.
  open func batchedMatmul(
    matrixCount: Int, a: Tensor.Data, transA: Bool, b: Tensor.Data, transB: Bool, transOut: Bool,
    rows: Int, inner: Int, cols: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("batchedMatmul")
  }

  /// Create a low-triangular or upper-triangular version of a
  /// (batch of) matrices.
  /// Offsets the diagonal by the given number of rows/columns.
  open func triangular(
    _ a: Tensor.Data, batch: Int, rows: Int, cols: Int, upper: Bool, offset: Int,
    dtype: Tensor.DType
  )
    async throws -> Tensor.Data
  {
    throw BackendError.notImplemented("triangular")
  }

  /// Compute the QR decomposition of the (batch of) matrices.
  ///
  /// If full is true, then a Q will be square, and some of the elements in Q and R
  /// may be redundant.
  open func qrDecomposition(
    _ a: Tensor.Data, batch: Int, rows: Int, cols: Int, full: Bool, dtype: Tensor.DType
  ) async throws -> (q: Tensor.Data, r: Tensor.Data) {
    throw BackendError.notImplemented("qrDecomposition")
  }

  /// Compute the SVD of the (batch of) matrices.
  ///
  /// If full is true, then both U and Vt will be square.
  /// Otherwise, if rows > cols, then U and Vt will be of shape
  /// [rows, k] and [k, cols], where k = min(rows, cols).
  open func svd(
    _ a: Tensor.Data, batch: Int, rows: Int, cols: Int, full: Bool, dtype: Tensor.DType
  ) async throws -> (u: Tensor.Data, s: Tensor.Data, vt: Tensor.Data) {
    throw BackendError.notImplemented("svd")
  }

  /// Compute a 1-dimensional convolution.
  open func conv1D(
    _ config: Conv1DConfig, batch: Int, image: Tensor.Data, kernel: Tensor.Data, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("conv1D")
  }

  /// Compute a 1-dimensional transposed convolution, or equivalently, a gradient through a
  /// 1-dimensional convolution with respect to the input.
  open func conv1DTranspose(
    _ config: Conv1DConfig, batch: Int, image: Tensor.Data, kernel: Tensor.Data, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("conv1DTranspose")
  }

  /// Compute the gradient of a 1-dimensional convolution with respect to the kernel.
  open func conv1DKernelGrad(
    _ config: Conv1DConfig, batch: Int, image: Tensor.Data, outGrad: Tensor.Data,
    dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("conv1DKernelGrad")
  }

  /// Compute a 2-dimensional convolution.
  open func conv2D(
    _ config: Conv2DConfig, batch: Int, image: Tensor.Data, kernel: Tensor.Data, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("conv2D")
  }

  /// Compute a 2-dimensional transposed convolution, or equivalently, a gradient through a
  /// 2-dimensional convolution with respect to the input.
  open func conv2DTranspose(
    _ config: Conv2DConfig, batch: Int, image: Tensor.Data, kernel: Tensor.Data, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("conv2DTranspose")
  }

  /// Compute the gradient of a 2-dimensional convolution with respect to the kernel.
  open func conv2DKernelGrad(
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
  open func elemwise(
    _ a: Tensor.Data, op: ElemwiseOp, scales: Tensor.Data?, count: Int, dtype: Tensor.DType
  ) async throws -> Tensor.Data {
    throw BackendError.notImplemented("elemwise")
  }

  /// Concatenate multiple tensors with per-tensor inner strides.
  open func concat(
    _ inputs: [Tensor.Data], outerCount: Int, innerCounts: [Int], dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("concat")
  }

  /// Create a tensor filled with a constant.
  open func constant<T: TensorElement>(_ value: T, count: Int, dtype: Tensor.DType) async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("constant")
  }

  /// Create a tensor from the contents of a collection.
  open func collection<T: TensorElement>(
    _ collection: some Collection<T>, reverse: Bool, dtype: Tensor.DType
  )
    async throws -> Tensor.Data
  {
    throw BackendError.notImplemented("collection")
  }

  /// Compute a permutation of integers for permuting the axes of a tensor of the given shape.
  open func axisPermutation(permutation: [Int], shape: [Int]) async throws -> Tensor.Data {
    throw BackendError.notImplemented("axisPermutation")
  }

  /// Get the default random number generator for this backend.
  open func defaultRandom() -> RandomGenerator {
    tracedFatalError("defaultRandom() not implemented")
  }

  /// Create a new random number generator for this backend.
  open func createRandom() -> RandomGenerator {
    tracedFatalError("createRandom() not implemented")
  }

  /// Perform an AdamW update.
  open func adamW(
    param: Tensor.Data,
    grad: Tensor.Data,
    moment1: Tensor.Data,
    moment2: Tensor.Data,
    beta1: Float,
    beta2: Float,
    eps: Float,
    weightDecay: Float,
    lr: Float,
    step: Float,
    count: Int,
    dtype: Tensor.DType
  )
    async throws
    -> (param: Tensor.Data, moment1: Tensor.Data, moment2: Tensor.Data)
  {
    func bd(_ data: Tensor.Data) -> BroadcastData {
      BroadcastData.simple(data: data, shape: [count])
    }

    let newMt = try await binaryOp(
      bd(try await binaryOp(beta1, moment1, op: .mul, count: count, dtype: dtype)),
      bd(try await binaryOp(1 - beta1, grad, op: .mul, count: count, dtype: dtype)),
      op: .add,
      dtype: dtype
    )
    let gradSq = try await pow(grad, 2, scale: 1 - beta2, scales: nil, count: count, dtype: dtype)
    let newVt = try await binaryOp(
      bd(try await binaryOp(beta2, moment2, op: .mul, count: count, dtype: dtype)),
      bd(gradSq),
      op: .add,
      dtype: dtype
    )
    let scaledMt = try await binaryOp(
      newMt, 1.0 / (1 - Foundation.pow(beta1, Float(step))), op: .mul, count: count, dtype: dtype)
    let scaledVt = try await binaryOp(
      newVt, 1.0 / (1 - Foundation.pow(beta2, Float(step))), op: .mul, count: count, dtype: dtype)
    let stepDir = try await binaryOp(
      try await binaryOp(
        bd(scaledMt),
        bd(
          try await binaryOp(
            try await pow(scaledVt, 0.5, scale: 1.0, scales: nil, count: count, dtype: dtype),
            eps,
            op: .add,
            count: count,
            dtype: dtype
          )
        ),
        op: .div,
        dtype: dtype
      ),
      lr,
      op: .mul,
      count: count,
      dtype: dtype
    )
    let decayed = try await binaryOp(
      param,
      (1 - lr * weightDecay),
      op: .mul,
      count: count,
      dtype: dtype
    )
    let newParam = try await binaryOp(
      bd(decayed),
      bd(stepDir),
      op: .sub,
      dtype: dtype
    )
    return (param: newParam, moment1: newMt, moment2: newVt)
  }

}

public protocol DataAllocator {
  func allocate(_ byteCount: Int) async throws -> Tensor.Data
}
