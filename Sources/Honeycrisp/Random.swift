import Foundation

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
    rand shape: [Int], dtype: DType = .float32, generator: RandomGenerator? = nil
  ) {
    self.init(rand: shape, dist: .uniform, dtype: dtype, generator: generator)
  }

  public convenience init(
    randn shape: [Int], dtype: DType = .float32, generator: RandomGenerator? = nil
  ) {
    self.init(rand: shape, dist: .normal, dtype: dtype, generator: generator)
  }

  private convenience init(
    rand shape: [Int], dist: RandomDist, dtype: DType = .float32, generator: RandomGenerator? = nil
  ) {
    let backend = Backend.current
    assert(
      generator == nil || generator!.backend === backend,
      "backend for provided generator is not the current backend")
    let dataTask = Task {
      let generator =
        if let generator = generator {
          generator
        } else {
          try await backend.defaultRandom()
        }
      return try await generator.sample(count: shape.product(), dist: dist, dtype: dtype)
    }
    self.init(dataTask: dataTask, shape: shape, dtype: dtype)
  }

  public convenience init(
    randnLike other: Tensor, generator: RandomGenerator? = nil
  ) {
    self.init(rand: other.shape, dist: .normal, dtype: other.dtype, generator: generator)
  }

  public convenience init(
    randLike other: Tensor, generator: RandomGenerator? = nil
  ) {
    self.init(rand: other.shape, dist: .uniform, dtype: other.dtype, generator: generator)
  }

  private convenience init(
    randInt shape: [Int], in range: Range<Int64>, generator: RandomGenerator? = nil
  ) {
    let backend = Backend.current
    assert(
      generator == nil || generator!.backend === backend,
      "backend for provided generator is not the current backend")
    let dataTask = Task {
      let generator =
        if let generator = generator {
          generator
        } else {
          try await backend.defaultRandom()
        }
      return try await generator.sample(count: shape.product(), in: range)
    }
    self.init(dataTask: dataTask, shape: shape, dtype: .int64)
  }
}
