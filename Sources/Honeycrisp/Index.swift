import HCBacktrace

/// A description of how to gather a subset of a Tensor
/// according to some indexing operation.
///
/// When multiple TensorIndex objects are combined in a subscript, we must
/// chain multiple TensorIndexResults into a single larger result. To do this,
/// each TensorIndexResult "consumes" axes through gatherAxis, and later
/// TensorIndex's will only see a subset of axes (via the shape argument to
/// tensorIndex()) after this index.
public struct TensorIndexResult {
  /// The Tensor is reshaped to this before a gather() is called.
  public let reshape: [Int]

  /// A one-dimensional Tensor of int64 indices.
  ///
  /// If nil, then no gather is performed and only a reshape is performed.
  public let indices: Tensor?

  /// If this is true, then no values may be repeated inside `indices`.
  public let indicesAreUnique: Bool

  /// The axis index (in the reshape) on which gather
  /// is called. The size on this axis must match the size of the
  /// `indices` Tensor.
  ///
  /// If indices is unspecified, then it is assumed that we would
  /// like to gather all of the indices on this axis in the existing
  /// order, making the gather a no-op.
  public let gatherAxis: Int

  // Optionally reshape the gathered axis after the gather.
  public let gatherReshape: [Int]?
}

/// A type which can be used within a subscript of a ``Tensor``.
public protocol TensorIndex {
  /// The minimum number of dimensions in a Tensor shape that supports this index.
  var minTensorIndexDims: Int { get }

  func tensorIndex(forShape inShape: [Int]) -> TensorIndexResult
}

extension Int: TensorIndex {
  public var minTensorIndexDims: Int { 1 }

  public func tensorIndex(forShape inShape: [Int]) -> TensorIndexResult {
    alwaysAssert(inShape.count > 0)
    let idx = self < 0 ? inShape[0] + self : self
    alwaysAssert(idx >= 0 && idx < inShape[0], "index \(self) out of range for size \(inShape[0])")
    return TensorIndexResult(
      reshape: inShape,
      indices: Tensor(data: [idx], shape: [1], dtype: .int64),
      indicesAreUnique: true,
      gatherAxis: 0,
      gatherReshape: []
    )
  }
}

extension Range<Int>: TensorIndex {
  public var minTensorIndexDims: Int { 1 }

  public func tensorIndex(forShape inShape: [Int]) -> TensorIndexResult {
    alwaysAssert(inShape.count > 0)
    let start = self.lowerBound < 0 ? inShape[0] + self.lowerBound : self.lowerBound
    let end = self.upperBound < 0 ? inShape[0] + self.upperBound : self.upperBound
    alwaysAssert(end >= start, "end (\(end)) must be >= start (\(start))")
    alwaysAssert(
      start >= 0 && start <= inShape[0],
      "index \(self.lowerBound) out of range for size \(inShape[0])"
    )
    alwaysAssert(
      end >= 0 && end <= inShape[0],
      "index \(self.upperBound) out of range for size \(inShape[0])"
    )
    if start == 0 && end == inShape[0] {
      return FullRange().tensorIndex(forShape: inShape)
    }
    return TensorIndexResult(
      reshape: inShape,
      indices: Tensor(data: start..<end, dtype: .int64),
      indicesAreUnique: true,
      gatherAxis: 0,
      gatherReshape: nil)
  }
}

extension StrideTo<Int>: TensorIndex {
  public var minTensorIndexDims: Int { 1 }

  public func tensorIndex(forShape inShape: [Int]) -> TensorIndexResult {
    alwaysAssert(inShape.count > 0)
    let indices = Array(self)
    alwaysAssert(
      indices.allSatisfy({ $0 < inShape[0] && $0 >= 0 }),
      "stride \(indices) out of bounds for shape \(inShape)")
    return TensorIndexResult(
      reshape: inShape,
      indices: Tensor(data: indices, dtype: .int64),
      indicesAreUnique: true,
      gatherAxis: 0,
      gatherReshape: nil)
  }
}

extension ClosedRange<Int>: TensorIndex {
  public var minTensorIndexDims: Int { 1 }

  public func tensorIndex(forShape inShape: [Int]) -> TensorIndexResult {
    alwaysAssert(inShape.count > 0)
    let end = 1 + (self.upperBound < 0 ? inShape[0] + self.upperBound : self.upperBound)
    return (self.lowerBound..<end).tensorIndex(forShape: inShape)
  }
}

/// A ``TensorIndex`` which leaves one or more axes unchanged during indexing.
public struct FullRange: TensorIndex {
  public let count: Int

  public init(count: Int = 1) {
    self.count = count
  }

  public var minTensorIndexDims: Int { count }

  public func tensorIndex(forShape inShape: [Int]) -> TensorIndexResult {
    alwaysAssert(inShape.count >= count)
    return TensorIndexResult(
      reshape: inShape,
      indices: nil,
      indicesAreUnique: true,
      gatherAxis: count - 1,
      gatherReshape: nil)
  }
}

/// A ``TensorIndex`` which creates a new axis at this position.
///
/// For example, if `tensor` has shape `[3, 4, 5]`, then you can create a new tensor
/// of shape `[3, 1, 4, 5]` using:
///
///     tensor[..., NewAxis()]
///
public struct NewAxis: TensorIndex {
  public let count: Int

  public init(count: Int = 1) {
    self.count = count
  }

  public var minTensorIndexDims: Int { 0 }

  public func tensorIndex(forShape inShape: [Int]) -> TensorIndexResult {
    return TensorIndexResult(
      reshape: Array(repeating: 1, count: count) + inShape,
      indices: nil,
      indicesAreUnique: true,
      gatherAxis: count - 1,
      gatherReshape: nil)
  }
}

extension PartialRangeFrom<Int>: TensorIndex {
  public var minTensorIndexDims: Int { 1 }

  public func tensorIndex(forShape inShape: [Int]) -> TensorIndexResult {
    alwaysAssert(inShape.count > 0)
    return (self.lowerBound..<inShape[0]).tensorIndex(forShape: inShape)
  }
}

extension PartialRangeUpTo<Int>: TensorIndex {
  public var minTensorIndexDims: Int { 1 }

  public func tensorIndex(forShape inShape: [Int]) -> TensorIndexResult {
    alwaysAssert(inShape.count > 0)
    let end = (self.upperBound > 0 ? self.upperBound : self.upperBound + inShape[0])
    alwaysAssert(
      end >= 0 && end <= inShape[0],
      "index \(self.upperBound) out of range for size \(inShape[0])"
    )
    return (0..<end).tensorIndex(forShape: inShape)
  }
}

extension PartialRangeThrough<Int>: TensorIndex {
  public var minTensorIndexDims: Int { 1 }

  public func tensorIndex(forShape inShape: [Int]) -> TensorIndexResult {
    alwaysAssert(inShape.count > 0)
    let end = (self.upperBound > 0 ? self.upperBound : self.upperBound + inShape[0])
    alwaysAssert(
      end >= 0 && end < inShape[0],
      "index \(self.upperBound) out of range for size \(inShape[0])"
    )
    return (0...end).tensorIndex(forShape: inShape)
  }
}

/// A ``TensorIndex`` which reverses the order of an axis.
public struct FlipAxis: TensorIndex {
  public var minTensorIndexDims: Int { 1 }

  public func tensorIndex(forShape inShape: [Int]) -> TensorIndexResult {
    return TensorIndexResult(
      reshape: inShape,
      indices: Tensor(data: 0..<inShape[0], dtype: .int64, reverse: true),
      indicesAreUnique: true,
      gatherAxis: 0,
      gatherReshape: nil)
  }
}

/// A ``TensorIndex`` which permutes axes in a ``Tensor``.
///
/// For example, `tensor[PermuteAxes(0, 2, 1)]` will swap the second and third axes of `tensor`.
public struct PermuteAxes: TensorIndex {
  public let perm: [Int]

  public var minTensorIndexDims: Int { perm.count }

  public init(_ perm: Int...) {
    self.init(perm)
  }

  public init(_ perm: [Int]) {
    for x in perm {
      alwaysAssert(x >= 0 && x < perm.count, "invalid permutation indices \(perm)")
    }
    alwaysAssert(perm.count == Set(perm).count, "permutation has repeated axis \(perm)")
    self.perm = perm
  }

  public func tensorIndex(forShape inShape: [Int]) -> TensorIndexResult {
    var newPerm = self.perm
    var removed = 0
    while newPerm.count > 0 && newPerm[0] == 0 {
      newPerm.remove(at: 0)
      newPerm = newPerm.map { $0 - 1 }
      removed += 1
    }
    if removed > 0 {
      return [FullRange(count: removed), PermuteAxes(newPerm)].tensorIndex(forShape: inShape)
    }

    alwaysAssert(
      perm.count <= inShape.count,
      "incompatible permutation \(perm) for shape \(inShape)")
    let reshape = [inShape[..<perm.count].product()] + inShape[perm.count...]
    let outShape = perm.map { inShape[$0] }
    let backend = Backend.current
    let indices = Tensor(
      dataTask: Task {
        try await backend.axisPermutation(permutation: perm, shape: Array(inShape[..<perm.count]))
      }, shape: [reshape[0]], dtype: .int64)
    return TensorIndexResult(
      reshape: reshape,
      indices: indices,
      indicesAreUnique: true,
      gatherAxis: 0,
      gatherReshape: outShape)
  }
}

extension Array: TensorIndex where Element == any TensorIndex {
  public var minTensorIndexDims: Int { self.map({ $0.minTensorIndexDims }).sum() }

  public func tensorIndex(forShape inShape: [Int]) -> TensorIndexResult {
    alwaysAssert(inShape.count >= self.minTensorIndexDims)
    switch self.count {
    case 0:
      return TensorIndexResult(
        reshape: inShape, indices: nil, indicesAreUnique: true, gatherAxis: -1, gatherReshape: nil)
    case 1:
      return self[0].tensorIndex(forShape: inShape)
    default:
      var outerShape: [Int] = []
      var gatherInSize: Int = 1
      var gatherShape: [Int] = []
      var indices: Tensor? = nil
      var unique: Bool = true
      var innerShape: [Int] = inShape

      func addFullAxes(_ axesShape: some Collection<Int>) {
        if indices == nil {
          // We can cheaply skip this gather dimension.
          outerShape += axesShape
        } else {
          let innerCount = axesShape.product()
          // We must explicitly represent the gather dimension.
          indices = innerCount * indices!.unsqueeze(axis: 1)
          indices = (indices! + Tensor(data: 0..<innerCount)).flatten()
          gatherShape += axesShape
          gatherInSize *= innerCount
        }
      }

      for x in self {
        let result = x.tensorIndex(forShape: innerShape)
        if !result.indicesAreUnique {
          unique = false
        }
        alwaysAssert(
          result.reshape.product() == innerShape.product(),
          "indexing operation cannot change size of tensor in reshape")
        if result.gatherAxis > 0 {
          addFullAxes(result.reshape[..<result.gatherAxis])
        }
        if let newIndices = result.indices {
          if indices == nil {
            indices = newIndices
          } else {
            let axisSize = result.reshape[result.gatherAxis]
            indices = axisSize * indices!.unsqueeze(axis: 1)
            indices = (indices! + newIndices).flatten()
          }
          gatherShape += result.gatherReshape ?? [newIndices.shape[0]]
          gatherInSize *= result.reshape[result.gatherAxis]
        } else if result.gatherAxis >= 0 {
          let axisSize = result.reshape[result.gatherAxis]
          let newShape = result.gatherReshape ?? [axisSize]
          addFullAxes(newShape)
        }
        innerShape = [Int](result.reshape[(result.gatherAxis + 1)...])
      }
      let reshape = outerShape + [gatherInSize] + innerShape

      // Sanity check to make sure we did accounting correctly.
      if indices != nil {
        alwaysAssert(gatherShape.product() == indices!.shape[0])
      }
      alwaysAssert(
        reshape.product() == inShape.product(),
        "cannot reshape to \(outerShape) + [\(gatherInSize)] + \(innerShape) from \(inShape)")

      return TensorIndexResult(
        reshape: reshape,
        indices: indices,
        indicesAreUnique: unique,
        gatherAxis: outerShape.count,
        gatherReshape: gatherShape)
    }
  }
}

extension Tensor {
  @recordCaller
  private func _t() -> Tensor {
    return self[FullRange(count: shape.count - 2), PermuteAxes(1, 0)]
  }

  @recordCaller
  private func _swap(axis: Int, with: Int) -> Tensor {
    let axis = positiveAxis(axis)
    let with = positiveAxis(with)
    if axis == with {
      return self
    }

    var perm = Array(0..<(Swift.max(axis, with) + 1))
    perm[axis] = with
    perm[with] = axis
    return self[PermuteAxes(perm)]
  }

  @recordCaller
  private func _move(axis: Int, to: Int) -> Tensor {
    let axis = positiveAxis(axis)
    let to = positiveAxis(to)
    if axis == to {
      return self
    }

    var perm = Array(0..<(Swift.max(axis, to) + 1))
    perm.remove(at: axis)
    perm.insert(axis, at: to)
    return self[PermuteAxes(perm)]
  }

  public subscript(index: any TensorIndex..., function function: StaticString = #function,
    file: StaticString = #file, line: UInt = #line
  ) -> Tensor {
    Backtrace.record(function: function, file: file, line: line) {
      let result = Array(index).tensorIndex(forShape: shape)
      var out = reshape(result.reshape)
      if let indices = result.indices {
        out = out.gather(
          axis: result.gatherAxis, indices: indices, indicesAreUnique: result.indicesAreUnique)
      }
      if let newShape = result.gatherReshape {
        out = out.reshape(
          result.reshape[..<result.gatherAxis] + newShape
            + result.reshape[(result.gatherAxis + 1)...]
        )
      }
      return out
    }
  }

  // Support UnboundedRange up to the first few indices

  /* python code:
    for num_args in range(1, 5):
        for pattern in range(0, 2 ** (num_args - 1)):
            args = []
            use_args = []
            for i in range(num_args):
                if not pattern & (1 << i):
                    args.append("_: UnboundedRange")
                    use_args.append("FullRange()")
                else:
                    args.append(f"arg{i}: any TensorIndex")
                    use_args.append(f"arg{i}")
            print(
                f"""
        public subscript({', '.join(args)}, other: any TensorIndex..., function function: StaticString = #function, file: StaticString = #file, line: UInt = #line) -> Tensor {{
            Backtrace.record(function: function, file: file, line: line) {{
                let idx = [{', '.join(use_args)}] + other
                return self[idx]
            }}
        }}
                """.rstrip()
            )
      */

  public subscript(_: UnboundedRange, other: any TensorIndex...,
    function function: StaticString = #function, file: StaticString = #file, line: UInt = #line
  ) -> Tensor {
    Backtrace.record(function: function, file: file, line: line) {
      let idx = [FullRange()] + other
      return self[idx]
    }
  }

  public subscript(_: UnboundedRange, _: UnboundedRange, other: any TensorIndex...,
    function function: StaticString = #function, file: StaticString = #file, line: UInt = #line
  ) -> Tensor {
    Backtrace.record(function: function, file: file, line: line) {
      let idx = [FullRange(), FullRange()] + other
      return self[idx]
    }
  }

  public subscript(arg0: any TensorIndex, _: UnboundedRange, other: any TensorIndex...,
    function function: StaticString = #function, file: StaticString = #file, line: UInt = #line
  ) -> Tensor {
    Backtrace.record(function: function, file: file, line: line) {
      let idx = [arg0, FullRange()] + other
      return self[idx]
    }
  }

  public subscript(_: UnboundedRange, _: UnboundedRange, _: UnboundedRange,
    other: any TensorIndex..., function function: StaticString = #function,
    file: StaticString = #file, line: UInt = #line
  ) -> Tensor {
    Backtrace.record(function: function, file: file, line: line) {
      let idx = [FullRange(), FullRange(), FullRange()] + other
      return self[idx]
    }
  }

  public subscript(arg0: any TensorIndex, _: UnboundedRange, _: UnboundedRange,
    other: any TensorIndex..., function function: StaticString = #function,
    file: StaticString = #file, line: UInt = #line
  ) -> Tensor {
    Backtrace.record(function: function, file: file, line: line) {
      let idx = [arg0, FullRange(), FullRange()] + other
      return self[idx]
    }
  }

  public subscript(_: UnboundedRange, arg1: any TensorIndex, _: UnboundedRange,
    other: any TensorIndex..., function function: StaticString = #function,
    file: StaticString = #file, line: UInt = #line
  ) -> Tensor {
    Backtrace.record(function: function, file: file, line: line) {
      let idx = [FullRange(), arg1, FullRange()] + other
      return self[idx]
    }
  }

  public subscript(arg0: any TensorIndex, arg1: any TensorIndex, _: UnboundedRange,
    other: any TensorIndex..., function function: StaticString = #function,
    file: StaticString = #file, line: UInt = #line
  ) -> Tensor {
    Backtrace.record(function: function, file: file, line: line) {
      let idx = [arg0, arg1, FullRange()] + other
      return self[idx]
    }
  }

  public subscript(_: UnboundedRange, _: UnboundedRange, _: UnboundedRange, _: UnboundedRange,
    other: any TensorIndex..., function function: StaticString = #function,
    file: StaticString = #file, line: UInt = #line
  ) -> Tensor {
    Backtrace.record(function: function, file: file, line: line) {
      let idx = [FullRange(), FullRange(), FullRange(), FullRange()] + other
      return self[idx]
    }
  }

  public subscript(arg0: any TensorIndex, _: UnboundedRange, _: UnboundedRange, _: UnboundedRange,
    other: any TensorIndex..., function function: StaticString = #function,
    file: StaticString = #file, line: UInt = #line
  ) -> Tensor {
    Backtrace.record(function: function, file: file, line: line) {
      let idx = [arg0, FullRange(), FullRange(), FullRange()] + other
      return self[idx]
    }
  }

  public subscript(_: UnboundedRange, arg1: any TensorIndex, _: UnboundedRange, _: UnboundedRange,
    other: any TensorIndex..., function function: StaticString = #function,
    file: StaticString = #file, line: UInt = #line
  ) -> Tensor {
    Backtrace.record(function: function, file: file, line: line) {
      let idx = [FullRange(), arg1, FullRange(), FullRange()] + other
      return self[idx]
    }
  }

  public subscript(arg0: any TensorIndex, arg1: any TensorIndex, _: UnboundedRange,
    _: UnboundedRange, other: any TensorIndex..., function function: StaticString = #function,
    file: StaticString = #file, line: UInt = #line
  ) -> Tensor {
    Backtrace.record(function: function, file: file, line: line) {
      let idx = [arg0, arg1, FullRange(), FullRange()] + other
      return self[idx]
    }
  }

  public subscript(_: UnboundedRange, _: UnboundedRange, arg2: any TensorIndex, _: UnboundedRange,
    other: any TensorIndex..., function function: StaticString = #function,
    file: StaticString = #file, line: UInt = #line
  ) -> Tensor {
    Backtrace.record(function: function, file: file, line: line) {
      let idx = [FullRange(), FullRange(), arg2, FullRange()] + other
      return self[idx]
    }
  }

  public subscript(arg0: any TensorIndex, _: UnboundedRange, arg2: any TensorIndex,
    _: UnboundedRange, other: any TensorIndex..., function function: StaticString = #function,
    file: StaticString = #file, line: UInt = #line
  ) -> Tensor {
    Backtrace.record(function: function, file: file, line: line) {
      let idx = [arg0, FullRange(), arg2, FullRange()] + other
      return self[idx]
    }
  }

  public subscript(_: UnboundedRange, arg1: any TensorIndex, arg2: any TensorIndex,
    _: UnboundedRange, other: any TensorIndex..., function function: StaticString = #function,
    file: StaticString = #file, line: UInt = #line
  ) -> Tensor {
    Backtrace.record(function: function, file: file, line: line) {
      let idx = [FullRange(), arg1, arg2, FullRange()] + other
      return self[idx]
    }
  }

  public subscript(arg0: any TensorIndex, arg1: any TensorIndex, arg2: any TensorIndex,
    _: UnboundedRange, other: any TensorIndex..., function function: StaticString = #function,
    file: StaticString = #file, line: UInt = #line
  ) -> Tensor {
    Backtrace.record(function: function, file: file, line: line) {
      let idx = [arg0, arg1, arg2, FullRange()] + other
      return self[idx]
    }
  }

}
