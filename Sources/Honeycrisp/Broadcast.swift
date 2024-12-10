import HCBacktrace
import Metal

/// A shape and corresponding strides, allowing a mapping from a tensor
/// to its underlying data.
///
/// The strides must correspond to contiguous data, i.e. each stride that
/// is non-zero must be larger than the following strides.
public struct BroadcastStrides: Hashable, Equatable, Sendable {
  public let shape: [Int]
  public let strides: [Int]
  public let dataCount: Int
  public let isNoOp: Bool

  public init(shape: [Int], strides: [Int]) {
    assert(shape.count == strides.count)
    self.shape = shape
    self.strides = strides

    dataCount = zip(strides, shape).filter({ $0.0 != 0 }).map({ $0.1 }).product()
    isNoOp = zip(strides, shape).allSatisfy { stride, size in stride != 0 || size == 1 }
  }

  public init(contiguousForShape shape: [Int]) {
    self.init(shape: shape, strides: stridesForShape(shape))
  }

  public var dataShape: [Int] {
    return zip(shape, strides).map { size, stride in stride == 0 ? 1 : size }
  }

  /// Get the `RepeatDims` which could be applied to the raw array to get the
  /// broadcasted array.
  public func repeats() -> [RepeatDims] {
    var result = [RepeatDims]()
    for (i, stride) in strides.enumerated().reversed() {
      if stride == 0 && shape[i] != 1 {
        var newDims = RepeatDims(
          outerCount: zip(strides[..<i], shape[..<i]).filter({ $0.0 != 0 }).map({ $0.1 })
            .product(),
          repeatCount: shape[i],
          innerCount: shape[(i + 1)...].product()
        )
        if let last = result.last {
          if last.outerCount == newDims.outerCount
            && last.innerCount * last.repeatCount == newDims.innerCount
          {
            // Coalesce consecutive repetitions.
            // This is useful when broadcasting multiple consecutive dimensions,
            // or dimensions with size 1 between them.
            result.removeLast()
            newDims = RepeatDims(
              outerCount: newDims.outerCount,
              repeatCount: last.repeatCount * newDims.repeatCount,
              innerCount: last.innerCount
            )
          }
        }
        result.append(newDims)
      }
    }
    return result
  }

  /// Get the `ReduceDims` that we could apply to the virtual gradient array to
  /// get a gradient for the raw data array.
  public func repeatsInverse() -> [ReduceDims] {
    return repeats().reversed().map { $0.inverse() }
  }

  /// Translate an index in the virtual array to the data array.
  public func callAsFunction(_ i: Int) -> Int {
    assert(i >= 0 && i < shape.product())

    if isNoOp {
      return i
    }

    var curIdx = i
    var result = 0
    for (stride, size) in zip(strides, shape).reversed() {
      result += stride * (curIdx % size)
      curIdx /= size
    }

    assert(
      result >= 0 && result < dataCount,
      "shape=\(shape) strides=\(strides) index=\(i) result=\(result)")
    return result
  }
}

/// A structure annotating an actual instance of raw data with a
/// `BroadcastStrides` indicating how to interpret it as a virtual array.
///
/// Note that this structure does not include the dtype of the underlying
/// data, which is crucial for actually performing operations on it.
public struct BroadcastData: Sendable {
  public let strides: BroadcastStrides
  public let data: Tensor.Data

  /// Get the size of the data array, in elements (not bytes).
  public var dataCount: Int { strides.dataCount }

  /// If true, the data is actually not broadcasted and the raw array is
  /// equal to the virtual array.
  public var isSimple: Bool { strides.isNoOp }

  /// Wrap an unbroadcasted tensor in a ``BroadcastData``.
  public static func simple(data: Tensor.Data, shape: [Int]) -> BroadcastData {
    Self(strides: BroadcastStrides(contiguousForShape: shape), data: data)
  }
}

extension Tensor {

  @recordCaller
  private func _expand(as asTensor: Tensor) -> Tensor {
    expand(shape: asTensor.shape)
  }

  @recordCaller
  private func _expand(shape newShape: [Int]) -> Tensor {
    if self.shape == newShape {
      return self
    }

    let bcastStrides = expandStrides(shape: newShape)
    let backend = Backend.current
    let newData = createDataTask { t in
      try await backend.broadcast(
        BroadcastData(strides: bcastStrides, data: try await t.data),
        dtype: t.dtype
      )
    }

    if !Tensor.isGradEnabled || !needsGrad {
      return Tensor(dataTask: newData, shape: newShape, dtype: dtype)
    } else {
      let handle = saveForBackward()
      return Tensor(dataTask: newData, shape: newShape, dtype: dtype) { grad in
        handle.backward(backend) { grad.reduceBroadcast(bcastStrides, as: self) }
      }
    }
  }

  @recordCaller
  internal func _expandStrides(shape newShape: [Int]) -> BroadcastStrides {
    #alwaysAssert(
      newShape.count >= shape.count,
      "cannot broadcast shape \(shape) to shorter shape \(newShape)"
    )

    let extraAxes = newShape.count - shape.count
    var strides = Array(repeating: 0, count: extraAxes)
    for (i, (oldSize, oldStride)) in zip(shape, stridesForShape(shape)).enumerated() {
      let newSize = newShape[i + extraAxes]
      if newSize != oldSize {
        #alwaysAssert(
          oldSize == 1,
          "axis \(i) cannot expand from size \(oldSize) to \(newSize): old shape \(shape), new shape \(newShape)"
        )
        strides.append(0)
      } else {
        strides.append(oldStride)
      }
    }

    return BroadcastStrides(shape: newShape, strides: strides)
  }

  @recordCaller
  private static func _broadcast(_ xs: [Tensor]) -> [Tensor] {
    let shape = Tensor.broadcastShape(xs.map { $0.shape })
    return xs.map { $0.expand(shape: shape) }
  }

  @recordCaller
  private static func _broadcast(_ x: Tensor, _ y: Tensor) -> (Tensor, Tensor) {
    let results = broadcast([x, y])
    return (results[0], results[1])
  }

  @recordCaller
  internal static func _lazyBroadcast(_ t: [Tensor]) -> ([Int], [BroadcastStrides]) {
    let newShape = broadcastShape(t.map { $0.shape })
    let results = t.map { $0.expandStrides(shape: newShape) }
    return (newShape, results)
  }

  @recordCaller
  internal static func _lazyBroadcast(_ t1: Tensor, _ t2: Tensor) -> (
    [Int], (BroadcastStrides, BroadcastStrides)
  ) {
    let newShape = broadcastShape([t1.shape, t2.shape])
    let r1 = t1.expandStrides(shape: newShape)
    let r2 = t2.expandStrides(shape: newShape)
    return (newShape, (r1, r2))
  }

  @recordCaller
  internal func _reduceBroadcast(_ bcast: BroadcastStrides, as tensor: Tensor) -> Tensor {
    return flatApplySums(bcast.repeatsInverse()).reshape(tensor.shape)
  }

  @recordCaller
  internal static func _broadcastShape(_ s: [[Int]]) -> [Int] {
    if s.count == 1 {
      return s[0]
    } else if s.count == 2 {
      var shape0 = s[0]
      var shape1 = s[1]
      while shape0.count < shape1.count {
        shape0.insert(1, at: 0)
      }
      while shape1.count < shape0.count {
        shape1.insert(1, at: 0)
      }
      return zip(shape0, shape1).map { x, y in
        if x == y {
          return x
        } else if x == 0 || y == 0 {
          tracedFatalError("shapes \(s[0]) and \(s[1]) do not support broadcasting")
        } else if x == 1 {
          return y
        } else if y == 1 {
          return x
        } else {
          tracedFatalError("shapes \(s[0]) and \(s[1]) do not support broadcasting")
        }
      }
    } else {
      return broadcastShape([broadcastShape([s[0], s[1]])] + Array(s[2...]))
    }
  }

}

func stridesForShape(_ shape: [Int]) -> [Int] {
  var strides = [Int](repeating: 0, count: shape.count)
  for i in 0..<shape.count {
    strides[i] = shape[(i + 1)...].product()
  }
  return strides
}
