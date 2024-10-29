import Accelerate
import Atomics
import Foundation
import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

open class BackendWrapper: Backend {

  public class WrappedRandomGenerator<B: BackendWrapper>: RandomGenerator {
    public let wrappedBackend: B
    public let wrapped: RandomGenerator

    public var backend: Backend {
      wrappedBackend
    }

    internal init(wrappedBackend: B, wrapped: RandomGenerator) {
      self.wrappedBackend = wrappedBackend
      self.wrapped = wrapped
    }

    public func save() async throws -> Data {
      try await wrapped.save()
    }

    public func restore(_ x: Data) async throws {
      try await wrapped.restore(x)
    }

    public func seed(_ x: Int) async throws {
      try await wrapped.seed(x)
    }

    public func sample(count: Int, dist: RandomDist, dtype: Tensor.DType) async throws
      -> Tensor.Data
    {
      try await wrapped.sample(count: count, dist: dist, dtype: dtype)
    }

    public func sample(count: Int, in range: Range<Int64>) async throws -> Tensor.Data {
      try await wrapped.sample(count: count, in: range)
    }
  }

  public let wrapped: Backend

  public init(wrapping: Backend) {
    wrapped = wrapping
  }

  override public func allocate(length: Int) async throws -> MTLBuffer {
    try await wrapped.allocate(length: length)
  }

  override public func binaryOp(
    a: Tensor.Data, aCount: Int, b: Tensor.Data, bCount: Int, op: NumericBinaryOp, count: Int,
    dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await wrapped.binaryOp(
      a: a, aCount: aCount, b: b, bCount: bCount, op: op, count: count, dtype: dtype)
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

  override public func mulAdd(
    input: Tensor.Data, inputCount: Int, coeff: Tensor.Data, coeffCount: Int, bias: Tensor.Data,
    biasCount: Int, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await wrapped.mulAdd(
      input: input, inputCount: inputCount, coeff: coeff, coeffCount: coeffCount, bias: bias,
      biasCount: biasCount, count: count, dtype: dtype)
  }

  override public func addMul(
    input: Tensor.Data, inputCount: Int, bias: Tensor.Data, biasCount: Int, coeff: Tensor.Data,
    coeffCount: Int, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await wrapped.addMul(
      input: input, inputCount: inputCount, bias: bias, biasCount: biasCount, coeff: coeff,
      coeffCount: coeffCount, count: count, dtype: dtype)
  }

  override public func compare(
    _ a: Tensor.Data, _ b: Tensor.Data, op: ComparisonOp, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await wrapped.compare(a, b, op: op, count: count, dtype: dtype)
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
    _ a: Tensor.Data, _ b: T, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await wrapped.pow(a, b, count: count, dtype: dtype)
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
    _ a: Tensor.Data, outerCount: Int, middleCount: Int, innerCount: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await wrapped.logSoftmax(
      a, outerCount: outerCount, middleCount: middleCount, innerCount: innerCount, dtype: dtype)
  }

  override public func logSoftmaxGrad(
    _ a: Tensor.Data, _ outGrad: Tensor.Data, outerCount: Int, middleCount: Int, innerCount: Int,
    dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await wrapped.logSoftmaxGrad(
      a, outGrad, outerCount: outerCount, middleCount: middleCount, innerCount: innerCount,
      dtype: dtype)

  }

  override public func repeated(
    _ a: Tensor.Data, outerCount: Int, innerCount: Int, repeats: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await wrapped.repeated(
      a, outerCount: outerCount, innerCount: innerCount, repeats: repeats, dtype: dtype)
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
    _ mask: Tensor.Data, _ a: TensorOrScalar<T>, _ b: TensorOrScalar<T>, _ x: T.Type, count: Int,
    dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await wrapped.when(mask, a, b, x, count: count, dtype: dtype)
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

  override public func tril(_ a: Tensor.Data, batch: Int, rows: Int, cols: Int, dtype: Tensor.DType)
    async throws -> Tensor.Data
  {
    try await wrapped.tril(a, batch: batch, rows: rows, cols: cols, dtype: dtype)
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

  override public func elemwise(_ a: Tensor.Data, op: ElemwiseOp, count: Int, dtype: Tensor.DType)
    async throws
    -> Tensor.Data
  {
    try await wrapped.elemwise(a, op: op, count: count, dtype: dtype)
  }

  override public func concat(
    _ inputs: [Tensor.Data], outerCount: Int, innerCounts: [Int], dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await wrapped.concat(inputs, outerCount: outerCount, innerCounts: innerCounts, dtype: dtype)
  }

  override public func axisPermutation(permutation: [Int], shape: [Int]) async throws -> Tensor.Data
  {
    try await wrapped.axisPermutation(permutation: permutation, shape: shape)
  }

  override public func defaultRandom() async throws -> RandomGenerator {
    WrappedRandomGenerator(wrappedBackend: self, wrapped: try await wrapped.defaultRandom())
  }

  override public func createRandom() async throws -> RandomGenerator {
    WrappedRandomGenerator(wrappedBackend: self, wrapped: try await wrapped.createRandom())
  }

}

open class BackendFLOPCounter: BackendWrapper {
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
