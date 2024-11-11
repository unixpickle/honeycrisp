import Foundation
import HCBacktrace

public enum RandomDist {
  case uniform
  case normal
}

public protocol RandomGenerator {
  var backend: Backend { get }

  func save() async throws -> Data
  func restore(_ x: Data) async throws
  func seed(_ x: Int) async throws

  func sample(count: Int, dist: RandomDist, dtype: Tensor.DType) async throws
    -> Tensor.Data
  func sample(count: Int, in range: Range<Int64>) async throws -> Tensor.Data
}

extension Tensor {
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
