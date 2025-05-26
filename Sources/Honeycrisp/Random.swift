import Foundation
import HCBacktrace

/// A probability distribution over continuous values.
public enum RandomDist: Sendable {
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
/// These calls with synchronously update ``RandomGenerator/state``, ensuring that the generator
/// is used in a deterministic order.
open class RandomGenerator: @unchecked Sendable {
  public let backend: Backend
  private var _state: Tensor
  private let _opLock: NSLock = NSLock()

  open var stateCount: Int {
    tracedFatalError("must override stateCount")
  }

  open var stateDType: Tensor.DType {
    tracedFatalError("must override stateDType")
  }

  /// All the information necessary to determine how the generator will behave.
  ///
  /// This can be accessed and restored to ensure deterministic reproducibility of a random
  /// operation or sequence of random operations.
  public var state: Tensor {
    get {
      _opLock.withLock { _state }
    }
    set {
      _opLock.withLock {
        #alwaysAssert(newValue.shape == [stateCount])
        #alwaysAssert(newValue.dtype == stateDType)
        _state = newValue
      }
    }
  }

  public init(backend: Backend, state: Tensor) {
    self.backend = backend
    self._state = state
  }

  /// Update the state of the generator given the seed.
  public func seed(_ x: Int) {
    _opLock.withLock {
      _state = Tensor(
        dataTask: Tensor.createDataTask {
          try await self._seed(x)
        }, shape: [stateCount], dtype: stateDType)
    }
  }

  open func _seed(_ x: Int) async throws -> Tensor.Data {
    tracedFatalError("_seed() is not implemented")
  }

  /// Sample a numeric tensor from a given continuous distribution.
  ///
  /// This will synchronously update ``state`` to the state after the operation.
  public func sample(count: Int, dist: RandomDist, dtype: Tensor.DType) -> Task<Tensor.Data, Error>
  {
    _opLock.withLock {
      let s = _state.noGrad()
      let task = Tensor.createDataTask {
        try await self._sample(state: try await s.data, count: count, dist: dist, dtype: dtype)
      }
      _state = Tensor(
        dataTask: Task {
          try await task.value.state
        }, shape: [stateCount], dtype: stateDType)
      return Task {
        try await task.value.sample
      }
    }
  }

  open func _sample(state: Tensor.Data, count: Int, dist: RandomDist, dtype: Tensor.DType)
    async throws -> (
      sample: Tensor.Data, state: Tensor.Data
    )
  {
    tracedFatalError("_sample(state:count:dist:dtype:) is not implemented")
  }

  /// Sample a tensor of int64 values uniformly in the given range.
  ///
  /// This will synchronously update ``state`` to the state after the operation.
  public func sample(count: Int, in range: Range<Int64>) -> Task<Tensor.Data, Error> {
    _opLock.withLock {
      let s = _state.noGrad()
      let task = Tensor.createDataTask {
        try await self._sample(state: try await s.data, count: count, in: range)
      }
      _state = Tensor(
        dataTask: Task {
          try await task.value.state
        }, shape: [stateCount], dtype: stateDType)
      return Task {
        try await task.value.sample
      }
    }
  }

  open func _sample(state: Tensor.Data, count: Int, in range: Range<Int64>) async throws -> (
    sample: Tensor.Data, state: Tensor.Data
  ) {
    tracedFatalError("_sample(state:count:in:) is not implemented")
  }
}

extension Tensor {
  /// Sample values in the range [0, 1).
  public convenience init(
    rand shape: [Int],
    dtype: DType = .float32,
    generator: RandomGenerator? = nil,
    function: StaticString = #function,
    file: StaticString = #filePath,
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
    file: StaticString = #filePath,
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
    file: StaticString = #filePath,
    line: UInt = #line
  ) {
    let backend = Backend.current
    let dataTask = Backtrace.record(function: function, file: file, line: line) {
      #alwaysAssert(
        generator == nil || generator!.backend === backend,
        "backend for provided generator is not the current backend")
      let generator = generator ?? backend.defaultRandom()
      return generator.sample(count: shape.product(), dist: dist, dtype: dtype)
    }
    self.init(dataTask: dataTask, shape: shape, dtype: dtype)
  }

  public convenience init(
    randPerm shape: [Int],
    axis: Int = -1,
    generator: RandomGenerator? = nil,
    function: StaticString = #function,
    file: StaticString = #filePath,
    line: UInt = #line
  ) {
    let dataTask = Backtrace.record(function: function, file: file, line: line) {
      let values = Tensor(
        randInt: shape, in: -0x8000_0000_0000_0000..<0x7fff_ffff_ffff_ffff, generator: generator)
      return values.argsort(axis: axis, stable: true).dataTask
    }
    self.init(dataTask: dataTask, shape: shape, dtype: .int64)
  }

  /// Sample values from the Normal distribution with the shape and dtype of a given `Tensor`.
  public convenience init(
    randnLike other: Tensor,
    generator: RandomGenerator? = nil,
    function: StaticString = #function,
    file: StaticString = #filePath,
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
    file: StaticString = #filePath,
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
    file: StaticString = #filePath,
    line: UInt = #line
  ) {
    let dataTask = Backtrace.record(function: function, file: file, line: line) {
      let backend = Backend.current
      #alwaysAssert(
        generator == nil || generator!.backend === backend,
        "backend for provided generator is not the current backend")
      let generator = generator ?? backend.defaultRandom()
      return generator.sample(count: shape.product(), in: range)
    }
    self.init(
      dataTask: dataTask, shape: shape, dtype: .int64, function: function, file: file, line: line)
  }

  @recordCaller
  private func _multinomial(
    sampleCount: Int, replacement: Bool = false, generator: RandomGenerator? = nil
  ) -> Tensor {
    #alwaysAssert(
      (shape.count == 1 || shape.count == 2) && shape.last! > 0,
      "cannot use tensor of shape \(shape) as multinomial weights")
    #alwaysAssert(
      replacement || sampleCount <= shape.last!,
      "cannot sample \(sampleCount) indices from only \(shape.last!) possible values without replacement"
    )
    #alwaysAssert(dtype.isFloat, "cannot use dtype \(dtype) as multinomial weights")
    if shape.count == 1 {
      return unsqueeze(axis: 0).multinomial(
        sampleCount: sampleCount, replacement: replacement, generator: generator
      ).squeeze(axis: 0)
    }
    if !replacement {
      #alwaysAssert(
        shape[1] >= sampleCount,
        "cannot sample \(sampleCount) elements from only \(shape[1]) options without replacement"
      )
    }
    let ng = noGrad()
    let probs = (ng / ng.sum(axis: 1, keepdims: true)).cast(.float32)
    if replacement || sampleCount == 1 {
      let cumProbs = probs.cast(.float32).cumulativeSum(axis: -1).unsqueeze(axis: 1)
      let noise = Tensor(rand: [shape[0], sampleCount, 1])
      return (cumProbs >= noise).argmax(axis: -1)
    } else {
      let logits = probs.clamp(min: 1e-8).log()
      let gumbels = -(-Tensor(rand: [shape[0], shape[1]]).clamp(min: 1e-8).log()).log()
      let indices = (logits + gumbels).argsort(axis: -1, descending: true)
      return indices[..., ..<sampleCount]
    }
  }
}
