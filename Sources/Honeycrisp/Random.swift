import Foundation
import HCBacktrace

public enum RandomDist {
  case uniform
  case normal
}

/// A random number generator associated with a ``Backend``.
///
/// Methods can be called from any thread, but the caller may want to synchronize
/// operations to make sure results are produced in a deterministic order.
/// For example, it may be desirable to serialize calls to ``RandomGenerator/sample(count:dist:dtype:)``
/// and ``RandomGenerator/seed(_:)`` to ensure sequences of tensors are generated in a
/// deterministic order.
///
/// Often times, generation methods will be called indirectly and asynchronously via ``Tensor``
/// initializers like ``Tensor/init(rand:dtype:generator:function:file:line:)``.
/// To ensure that these use cases are serialized with methods like ``RandomGenerator/seed(_:)``,
/// you can call ``Tensor/wait(function:file:line:)`` on the random `Tensor`s before calling further
/// methods that use the ``RandomGenerator``.
public protocol RandomGenerator {
  /// The ``Backend`` which created this generator.
  var backend: Backend { get }

  /// Encode the current state of the generator.
  func save() async throws -> Data

  /// Restore the state of the generator from a previous ``RandomGenerator/save()`` call.
  func restore(_ x: Data) async throws

  /// Seed the generator with the integer.
  func seed(_ x: Int) async throws

  /// Sample a tensor of numeric values from the continuous distribution.
  func sample(count: Int, dist: RandomDist, dtype: Tensor.DType) async throws -> Tensor.Data

  /// Sample a tensor of int64 values uniformly in the given range.
  func sample(count: Int, in range: Range<Int64>) async throws -> Tensor.Data
}

extension Tensor {
  /// Sample values in the range [0, 1).
  public convenience init(
    rand shape: [Int],
    dtype: DType = .float32,
    generator: RandomGenerator? = nil,
    function: StaticString = #function,
    file: StaticString = #file,
    line: UInt = #line
  ) {
    self.init(
      rand: shape, dist: .uniform, dtype: dtype, generator: generator, function: function,
      file: file, line: line)
  }

  /// Sample values from the Normal distribution.
  public convenience init(
    randn shape: [Int],
    dtype: DType = .float32,
    generator: RandomGenerator? = nil,
    function: StaticString = #function,
    file: StaticString = #file,
    line: UInt = #line
  ) {
    self.init(
      rand: shape, dist: .normal, dtype: dtype, generator: generator, function: function,
      file: file, line: line)
  }

  private convenience init(
    rand shape: [Int],
    dist: RandomDist,
    dtype: DType = .float32,
    generator: RandomGenerator? = nil,
    function: StaticString = #function,
    file: StaticString = #file,
    line: UInt = #line
  ) {
    let backend = Backend.current
    let dataTask = Backtrace.record(function: function, file: file, line: line) {
      alwaysAssert(
        generator == nil || generator!.backend === backend,
        "backend for provided generator is not the current backend")
      return Tensor.createDataTask {
        let generator =
          if let generator = generator {
            generator
          } else {
            try await backend.defaultRandom()
          }
        return try await generator.sample(count: shape.product(), dist: dist, dtype: dtype)
      }
    }
    self.init(dataTask: dataTask, shape: shape, dtype: dtype)
  }

  /// Sample values from the Normal distribution with the shape and dtype of a given `Tensor`.
  public convenience init(
    randnLike other: Tensor,
    generator: RandomGenerator? = nil,
    function: StaticString = #function,
    file: StaticString = #file,
    line: UInt = #line
  ) {
    self.init(
      rand: other.shape, dist: .normal, dtype: other.dtype, generator: generator,
      function: function, file: file, line: line)
  }

  /// Sample values uniformly in [0, 1) with the shape and dtype of a given `Tensor`.
  public convenience init(
    randLike other: Tensor,
    generator: RandomGenerator? = nil,
    function: StaticString = #function,
    file: StaticString = #file,
    line: UInt = #line
  ) {
    self.init(
      rand: other.shape, dist: .uniform, dtype: other.dtype, generator: generator,
      function: function, file: file, line: line)
  }

  /// Sample random integers in the given range.
  public convenience init(
    randInt shape: [Int],
    in range: Range<Int64>,
    generator: RandomGenerator? = nil,
    function: StaticString = #function,
    file: StaticString = #file,
    line: UInt = #line
  ) {
    let dataTask = Backtrace.record(function: function, file: file, line: line) {
      let backend = Backend.current
      alwaysAssert(
        generator == nil || generator!.backend === backend,
        "backend for provided generator is not the current backend")
      return Tensor.createDataTask {
        let generator =
          if let generator = generator {
            generator
          } else {
            try await backend.defaultRandom()
          }
        return try await generator.sample(count: shape.product(), in: range)
      }
    }
    self.init(
      dataTask: dataTask, shape: shape, dtype: .int64, function: function, file: file, line: line)
  }
}
