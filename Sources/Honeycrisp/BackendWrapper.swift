import Accelerate
import Foundation
import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

/// A ``Backend`` which wraps another `Backend`, intended to be sub-classed to override
/// a selection of methods from a wrapped implementation.
open class BackendWrapper: Backend, @unchecked Sendable {

  public class WrappedRandomGenerator<B: BackendWrapper>: RandomGenerator, @unchecked Sendable {
    public let wrappedBackend: B
    public let wrapped: RandomGenerator

    override open var stateCount: Int {
      wrapped.stateCount
    }

    override open var stateDType: Tensor.DType {
      wrapped.stateDType
    }

    internal init(wrappedBackend: B, wrapped: RandomGenerator) {
      self.wrappedBackend = wrappedBackend
      self.wrapped = wrapped
      super.init(backend: wrappedBackend, state: wrapped.state)
    }

    override open var state: Tensor {
      get {
        wrapped.state
      }
      set {
        wrapped.state = newValue
      }
    }

    /// Update the state of the generator given the seed.
    override public func seed(_ x: Int) {
      wrapped.seed(x)
    }

    /// Sample a numeric tensor from a given continuous distribution.
    override public func sample(count: Int, dist: RandomDist, dtype: Tensor.DType) -> Task<
      Tensor.Data, Error
    > {
      wrapped.sample(count: count, dist: dist, dtype: dtype)
    }

    /// Sample a tensor of int64 values uniformly in the given range.
    override public func sample(count: Int, in range: Range<Int64>) -> Task<Tensor.Data, Error> {
      wrapped.sample(count: count, in: range)
    }
  }

  public let wrapped: Backend

  public init(wrapping: Backend) {
    wrapped = wrapping
  }

  override open func broadcast(_ a: BroadcastData, dtype: Tensor.DType) async throws -> Tensor.Data
  {
    try await wrapped.broadcast(a, dtype: dtype)
  }

  override public func binaryOp(
    _ a: BroadcastData, _ b: BroadcastData, op: NumericBinaryOp, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await wrapped.binaryOp(a, b, op: op, dtype: dtype)
  }

  override public func binaryOp<T: NumericTensorElement>(
    _ a: Tensor.Data, _ b: T, op: NumericBinaryOp, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await wrapped.binaryOp(a, b, op: op, count: count, dtype: dtype)
  }

  override public func binaryOp<T: NumericTensorElement>(
    _ a: T, _ b: Tensor.Data, op: NumericBinaryOp, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await wrapped.binaryOp(a, b, op: op, count: count, dtype: dtype)
  }

  override open func bitwiseOp(
    _ a: BroadcastData, _ b: BroadcastData, op: BitwiseOp, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await wrapped.bitwiseOp(a, b, op: op, dtype: dtype)
  }

  override open func bitwiseOp<T: TensorElementBitPattern>(
    _ a: Tensor.Data, _ b: T, op: BitwiseOp, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await wrapped.bitwiseOp(a, b, op: op, count: count, dtype: dtype)
  }

  override public func mulAdd(
    input: BroadcastData, coeff: BroadcastData, bias: BroadcastData, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await wrapped.mulAdd(input: input, coeff: coeff, bias: bias, dtype: dtype)
  }

  override public func addMul(
    input: BroadcastData, bias: BroadcastData, coeff: BroadcastData, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await wrapped.addMul(input: input, bias: bias, coeff: coeff, dtype: dtype)
  }

  override open func normalize<T: NumericTensorElement>(
    input: Tensor.Data, dims: ReduceDims, eps: T, dtype: Tensor.DType
  )
    async throws -> Tensor.Data
  {
    try await wrapped.normalize(input: input, dims: dims, eps: eps, dtype: dtype)
  }

  override open func normalizeGrad<T: NumericTensorElement>(
    input: Tensor.Data, outGrad: Tensor.Data, dims: ReduceDims, eps: T, dtype: Tensor.DType
  )
    async throws -> Tensor.Data
  {
    try await wrapped.normalizeGrad(
      input: input, outGrad: outGrad, dims: dims, eps: eps, dtype: dtype)
  }

  override public func compare(
    _ a: BroadcastData, _ b: BroadcastData, op: ComparisonOp, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await wrapped.compare(a, b, op: op, dtype: dtype)
  }

  override public func compare<T: TensorElement>(
    _ a: Tensor.Data, _ b: T, op: ComparisonOp, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await wrapped.compare(a, b, op: op, count: count, dtype: dtype)
  }

  override public func compare<T: TensorElement>(
    _ a: T, _ b: Tensor.Data, op: ComparisonOp, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await wrapped.compare(a, b, op: op, count: count, dtype: dtype)
  }

  override public func cast(
    _ a: Tensor.Data, count: Int, inType: Tensor.DType, outType: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await wrapped.cast(a, count: count, inType: inType, outType: outType)
  }

  override public func pow<T: NumericTensorElement>(
    _ a: Tensor.Data, _ b: T, scale: T, scales: Tensor.Data?, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await wrapped.pow(a, b, scale: scale, scales: scales, count: count, dtype: dtype)
  }

  override public func clamp<T: NumericTensorElement>(
    _ a: Tensor.Data, min: T?, max: T?, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await wrapped.clamp(a, min: min, max: max, count: count, dtype: dtype)
  }

  override public func reduce(
    _ a: Tensor.Data, op: ReduceOp, dims: ReduceDims, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await wrapped.reduce(a, op: op, dims: dims, dtype: dtype)
  }

  override public func logSoftmax(
    _ a: Tensor.Data, dims: ReduceDims, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await wrapped.logSoftmax(a, dims: dims, dtype: dtype)
  }

  override public func logSoftmaxGrad(
    _ a: Tensor.Data, _ outGrad: Tensor.Data, dims: ReduceDims, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await wrapped.logSoftmaxGrad(a, outGrad, dims: dims, dtype: dtype)

  }

  override public func repeated(
    _ a: Tensor.Data, dims: RepeatDims, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await wrapped.repeated(a, dims: dims, dtype: dtype)
  }

  override public func gather(
    _ a: Tensor.Data, _ s: ScatterGatherIndices, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await wrapped.gather(a, s, dtype: dtype)
  }

  override public func scatter(
    _ a: Tensor.Data, _ s: ScatterGatherIndices, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await wrapped.scatter(a, s, dtype: dtype)
  }

  override public func when<T>(
    _ mask: BroadcastData, _ a: TensorOrScalar<T>, _ b: TensorOrScalar<T>, _ x: T.Type,
    dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await wrapped.when(mask, a, b, x, dtype: dtype)
  }

  override public func matmul(
    a: Tensor.Data, transA: Bool, b: Tensor.Data, transB: Bool, transOut: Bool, rows: Int,
    inner: Int, cols: Int,
    dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await wrapped.matmul(
      a: a, transA: transA, b: b, transB: transB, transOut: transOut, rows: rows, inner: inner,
      cols: cols, dtype: dtype)
  }

  override public func batchedMatmul(
    matrixCount: Int, a: Tensor.Data, transA: Bool, b: Tensor.Data, transB: Bool, transOut: Bool,
    rows: Int, inner: Int, cols: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await wrapped.batchedMatmul(
      matrixCount: matrixCount, a: a, transA: transA, b: b, transB: transB, transOut: transOut,
      rows: rows, inner: inner, cols: cols, dtype: dtype)
  }

  override public func triangular(
    _ a: Tensor.Data, batch: Int, rows: Int, cols: Int, upper: Bool, offset: Int,
    dtype: Tensor.DType
  )
    async throws -> Tensor.Data
  {
    try await wrapped.triangular(
      a, batch: batch, rows: rows, cols: cols, upper: upper, offset: offset, dtype: dtype)
  }

  override public func conv1D(
    _ config: Conv1DConfig, batch: Int, image: Tensor.Data, kernel: Tensor.Data, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await wrapped.conv1D(config, batch: batch, image: image, kernel: kernel, dtype: dtype)
  }

  override public func conv1DTranspose(
    _ config: Conv1DConfig, batch: Int, image: Tensor.Data, kernel: Tensor.Data, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await wrapped.conv1DTranspose(
      config, batch: batch, image: image, kernel: kernel, dtype: dtype)
  }

  override public func conv1DKernelGrad(
    _ config: Conv1DConfig, batch: Int, image: Tensor.Data, outGrad: Tensor.Data,
    dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await wrapped.conv1DKernelGrad(
      config, batch: batch, image: image, outGrad: outGrad, dtype: dtype)
  }

  override public func conv2D(
    _ config: Conv2DConfig, batch: Int, image: Tensor.Data, kernel: Tensor.Data, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await wrapped.conv2D(config, batch: batch, image: image, kernel: kernel, dtype: dtype)
  }

  override public func conv2DTranspose(
    _ config: Conv2DConfig, batch: Int, image: Tensor.Data, kernel: Tensor.Data, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await wrapped.conv2DTranspose(
      config, batch: batch, image: image, kernel: kernel, dtype: dtype)
  }

  override public func conv2DKernelGrad(
    _ config: Conv2DConfig, batch: Int, image: Tensor.Data, outGrad: Tensor.Data,
    dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await wrapped.conv2DKernelGrad(
      config, batch: batch, image: image, outGrad: outGrad, dtype: dtype)
  }

  override public func elemwise(
    _ a: Tensor.Data, op: ElemwiseOp, scales: Tensor.Data?, count: Int, dtype: Tensor.DType
  ) async throws -> Tensor.Data {
    try await wrapped.elemwise(a, op: op, scales: scales, count: count, dtype: dtype)
  }

  override public func concat(
    _ inputs: [Tensor.Data], outerCount: Int, innerCounts: [Int], dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await wrapped.concat(inputs, outerCount: outerCount, innerCounts: innerCounts, dtype: dtype)
  }

  override public func constant<T: TensorElement>(_ value: T, count: Int, dtype: Tensor.DType)
    async throws -> Tensor.Data
  {
    try await wrapped.constant(value, count: count, dtype: dtype)
  }

  override public func collection<T: TensorElement>(
    _ collection: some Collection<T>, reverse: Bool, dtype: Tensor.DType
  )
    async throws -> Tensor.Data
  {
    try await wrapped.collection(collection, reverse: reverse, dtype: dtype)
  }

  override public func axisPermutation(permutation: [Int], shape: [Int]) async throws -> Tensor.Data
  {
    try await wrapped.axisPermutation(permutation: permutation, shape: shape)
  }

  override public func defaultRandom() -> RandomGenerator {
    WrappedRandomGenerator(wrappedBackend: self, wrapped: wrapped.defaultRandom())
  }

  override public func createRandom() -> RandomGenerator {
    WrappedRandomGenerator(wrappedBackend: self, wrapped: wrapped.createRandom())
  }

}

/// A backend wrapper that counts floating-point operations in an atomic counter.
open class BackendFLOPCounter: BackendWrapper, @unchecked Sendable {
  internal var lock = NSLock()
  internal var counter: Int64 = 0

  public var flopCount: Int64 {
    get {
      lock.lock()
      defer { lock.unlock() }
      return counter
    }
    set {
      lock.lock()
      counter = newValue
      lock.unlock()
    }
  }

  public func addFLOPs(_ c: Int64) {
    lock.lock()
    defer { lock.unlock() }
    counter += c
  }

  override public init(wrapping: Backend) {
    super.init(wrapping: wrapping)
  }

  override public func matmul(
    a: Tensor.Data, transA: Bool, b: Tensor.Data, transB: Bool, transOut: Bool, rows: Int,
    inner: Int, cols: Int,
    dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    let result = try await super.matmul(
      a: a, transA: transA, b: b, transB: transB, transOut: transOut, rows: rows, inner: inner,
      cols: cols, dtype: dtype)
    addFLOPs(Int64(2 * rows * inner * cols))
    return result
  }

  override public func batchedMatmul(
    matrixCount: Int, a: Tensor.Data, transA: Bool, b: Tensor.Data, transB: Bool, transOut: Bool,
    rows: Int, inner: Int, cols: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    let result = try await super.batchedMatmul(
      matrixCount: matrixCount, a: a, transA: transA, b: b, transB: transB, transOut: transOut,
      rows: rows, inner: inner, cols: cols, dtype: dtype)
    addFLOPs(Int64(2 * matrixCount * rows * inner * cols))
    return result
  }

  override public func conv1D(
    _ config: Conv1DConfig, batch: Int, image: Tensor.Data, kernel: Tensor.Data, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    let result = try await super.conv1D(
      config, batch: batch, image: image, kernel: kernel, dtype: dtype)
    self.addFLOPs(Int64(config.forwardFLOPs(batch: batch)))
    return result
  }

  override public func conv1DTranspose(
    _ config: Conv1DConfig, batch: Int, image: Tensor.Data, kernel: Tensor.Data, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    let result = try await super.conv1DTranspose(
      config, batch: batch, image: image, kernel: kernel, dtype: dtype)
    self.addFLOPs(Int64(config.transposeFLOPs(batch: batch)))
    return result
  }

  override public func conv1DKernelGrad(
    _ config: Conv1DConfig, batch: Int, image: Tensor.Data, outGrad: Tensor.Data,
    dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    let result = try await super.conv1DKernelGrad(
      config, batch: batch, image: image, outGrad: outGrad, dtype: dtype)
    self.addFLOPs(Int64(config.kernelGradFLOPs(batch: batch)))
    return result
  }

  override public func conv2D(
    _ config: Conv2DConfig, batch: Int, image: Tensor.Data, kernel: Tensor.Data, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    let result = try await super.conv2D(
      config, batch: batch, image: image, kernel: kernel, dtype: dtype)
    self.addFLOPs(Int64(config.forwardFLOPs(batch: batch)))
    return result
  }

  override public func conv2DTranspose(
    _ config: Conv2DConfig, batch: Int, image: Tensor.Data, kernel: Tensor.Data, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    let result = try await super.conv2DTranspose(
      config, batch: batch, image: image, kernel: kernel, dtype: dtype)
    self.addFLOPs(Int64(config.transposeFLOPs(batch: batch)))
    return result
  }

  override public func conv2DKernelGrad(
    _ config: Conv2DConfig, batch: Int, image: Tensor.Data, outGrad: Tensor.Data,
    dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    let result = try await super.conv2DKernelGrad(
      config, batch: batch, image: image, outGrad: outGrad, dtype: dtype)
    self.addFLOPs(Int64(config.kernelGradFLOPs(batch: batch)))
    return result
  }
}
