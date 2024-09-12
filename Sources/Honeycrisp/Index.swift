/// A description of how to gather a subset of a Tensor
/// according to some indexing operation.
public struct TensorIndexResult {
  /// The Tensor is reshaped to this before a gather() is called.
  public let reshape: [Int]

  /// A one-dimensional Tensor of int64 indices.
  ///
  /// If nil, then no gather is performed and only a reshape is performed.
  public let indices: Tensor?

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

public protocol TensorIndex {
  /// The minimum number of dimensions in a Tensor shape that supports this index.
  var minTensorIndexDims: Int { get }

  func tensorIndex(forShape inShape: [Int]) -> TensorIndexResult
}

extension Int: TensorIndex {
  public var minTensorIndexDims: Int { 1 }

  public func tensorIndex(forShape inShape: [Int]) -> TensorIndexResult {
    assert(inShape.count > 0)
    let idx = self < 0 ? inShape[0] + self : self
    assert(idx >= 0 && idx < inShape[0], "index \(self) out of range for size \(inShape[0])")
    return TensorIndexResult(
      reshape: inShape,
      indices: Tensor(data: [self], shape: [1], dtype: .int64),
      gatherAxis: 0,
      gatherReshape: []
    )
  }
}

extension Range<Int>: TensorIndex {
  public var minTensorIndexDims: Int { 1 }

  public func tensorIndex(forShape inShape: [Int]) -> TensorIndexResult {
    assert(inShape.count > 0)
    let start = self.lowerBound < 0 ? inShape[0] + self.lowerBound : self.lowerBound
    let end = self.upperBound < 0 ? inShape[0] + self.upperBound : self.upperBound
    assert(end >= start, "end (\(end)) must be >= start (\(start))")
    assert(
      start >= 0 && start <= inShape[0],
      "index \(self.lowerBound) out of range for size \(inShape[0])"
    )
    assert(
      end >= 0 && end <= inShape[0],
      "index \(self.upperBound) out of range for size \(inShape[0])"
    )
    return TensorIndexResult(
      reshape: inShape,
      indices: Tensor(range: start..<end, dtype: .int64),
      gatherAxis: 0,
      gatherReshape: nil)
  }
}

extension ClosedRange<Int>: TensorIndex {
  public var minTensorIndexDims: Int { 1 }

  public func tensorIndex(forShape inShape: [Int]) -> TensorIndexResult {
    assert(inShape.count > 0)
    let end = 1 + (self.upperBound < 0 ? inShape[0] + self.upperBound : self.upperBound)
    return (self.lowerBound..<end).tensorIndex(forShape: inShape)
  }
}

public struct FullRange: TensorIndex {
  public let dims: Int

  public init(dims: Int = 1) {
    self.dims = dims
  }

  public var minTensorIndexDims: Int { dims }

  public func tensorIndex(forShape inShape: [Int]) -> TensorIndexResult {
    assert(inShape.count >= dims)
    return TensorIndexResult(
      reshape: inShape, indices: nil, gatherAxis: dims - 1, gatherReshape: nil)
  }
}

public struct NewAxis: TensorIndex {
  public let count: Int

  public init(count: Int = 1) {
    self.count = count
  }

  public var minTensorIndexDims: Int { 0 }

  public func tensorIndex(forShape inShape: [Int]) -> TensorIndexResult {
    return TensorIndexResult(
      reshape: Array(repeating: 1, count: count) + inShape, indices: nil, gatherAxis: count - 1,
      gatherReshape: nil)
  }
}

extension PartialRangeFrom<Int>: TensorIndex {
  public var minTensorIndexDims: Int { 1 }

  public func tensorIndex(forShape inShape: [Int]) -> TensorIndexResult {
    assert(inShape.count > 0)
    return (self.lowerBound..<inShape[0]).tensorIndex(forShape: inShape)
  }
}

extension PartialRangeUpTo<Int>: TensorIndex {
  public var minTensorIndexDims: Int { 1 }

  public func tensorIndex(forShape inShape: [Int]) -> TensorIndexResult {
    assert(inShape.count > 0)
    let end = (self.upperBound > 0 ? self.upperBound : self.upperBound + inShape[0])
    assert(
      end >= 0 && end <= inShape[0],
      "index \(self.upperBound) out of range for size \(inShape[0])"
    )
    return (0..<end).tensorIndex(forShape: inShape)
  }
}

extension PartialRangeThrough<Int>: TensorIndex {
  public var minTensorIndexDims: Int { 1 }

  public func tensorIndex(forShape inShape: [Int]) -> TensorIndexResult {
    assert(inShape.count > 0)
    let end = (self.upperBound > 0 ? self.upperBound : self.upperBound + inShape[0])
    assert(
      end >= 0 && end < inShape[0],
      "index \(self.upperBound) out of range for size \(inShape[0])"
    )
    return (0...end).tensorIndex(forShape: inShape)
  }
}

public struct PermuteAxes: TensorIndex {
  public let perm: [Int]

  public var minTensorIndexDims: Int { perm.count }

  public init(_ perm: Int...) {
    for x in perm {
      assert(x >= 0 && x < perm.count, "invalid permutation indices \(perm)")
    }
    assert(perm.count == Set(perm).count, "permutation has repeated axis \(perm)")
    self.perm = perm
  }

  public func tensorIndex(forShape inShape: [Int]) -> TensorIndexResult {
    assert(
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
      reshape: reshape, indices: indices, gatherAxis: 0, gatherReshape: outShape)
  }
}

extension Array: TensorIndex where Element == any TensorIndex {
  public var minTensorIndexDims: Int { self.map({ $0.minTensorIndexDims }).sum() }

  public func tensorIndex(forShape inShape: [Int]) -> TensorIndexResult {
    assert(inShape.count >= self.minTensorIndexDims)
    switch self.count {
    case 0:
      return TensorIndexResult(reshape: inShape, indices: nil, gatherAxis: -1, gatherReshape: nil)
    case 1:
      return self[0].tensorIndex(forShape: inShape)
    default:
      var outerShape: [Int] = []
      var gatherInSize: Int = 1
      var gatherShape: [Int] = []
      var indices: Tensor? = nil
      var innerShape: [Int] = inShape

      for x in self {
        let result = x.tensorIndex(forShape: innerShape)
        assert(
          result.reshape.product() == innerShape.product(),
          "indexing operation cannot change size of tensor in reshape")
        if result.gatherAxis > 0 {
          let prefixShape = result.reshape[..<result.gatherAxis]
          if indices == nil {
            outerShape += prefixShape
          } else {
            // We cannot cheaply skip the dimension(s) with broadcasted gather.
            let prefixSize = prefixShape.product()
            indices =
              prefixSize
              * indices!.reshape([indices!.shape[0], 1]).repeating(axis: 1, count: prefixSize)
            indices = (indices! + Tensor(range: 0..<prefixSize).expand(as: indices!)).flatten()
            gatherShape += prefixShape
            gatherInSize *= prefixSize
          }
        }
        if let newIndices = result.indices {
          if indices == nil {
            indices = newIndices
          } else {
            let axisSize = result.reshape[result.gatherAxis]
            indices =
              axisSize
              * indices!.reshape([indices!.shape[0], 1]).repeating(
                axis: 1, count: newIndices.shape[0])
            indices = (indices! + newIndices.expand(as: indices!)).flatten()
          }
          gatherShape += result.gatherReshape ?? [newIndices.shape[0]]
          gatherInSize *= result.reshape[result.gatherAxis]
        } else if result.gatherAxis >= 0 {
          let axisSize = result.reshape[result.gatherAxis]
          let newShape = result.gatherReshape ?? [axisSize]
          if indices == nil {
            // We can cheaply skip this gather dimension.
            outerShape += newShape
          } else {
            // We must explicitly represent the gather dimension.
            indices =
              axisSize
              * indices!.reshape([indices!.shape[0], 1]).repeating(
                axis: 1, count: axisSize)
            indices = (indices! + Tensor(range: 0..<axisSize).expand(as: indices!)).flatten()
            gatherShape += newShape
            gatherInSize *= axisSize
          }
        }
        innerShape = [Int](result.reshape[(result.gatherAxis + 1)...])
      }
      let reshape = outerShape + [gatherInSize] + innerShape

      // Sanity check to make sure we did accounting correctly.
      if indices != nil {
        assert(gatherShape.product() == indices!.shape[0])
      }
      assert(
        reshape.product() == inShape.product(),
        "cannot reshape to \(outerShape) + [\(gatherInSize)] + \(innerShape) from \(inShape)")

      return TensorIndexResult(
        reshape: reshape, indices: indices,
        gatherAxis: outerShape.count, gatherReshape: gatherShape)
    }
  }
}

extension Tensor {
  public subscript(index: any TensorIndex...) -> Tensor {
    let result = Array(index).tensorIndex(forShape: shape)
    var out = reshape(result.reshape)
    if let indices = result.indices {
      out = out.gather(axis: result.gatherAxis, indices: indices)
    }
    if let newShape = result.gatherReshape {
      out = out.reshape(
        result.reshape[..<result.gatherAxis] + newShape + result.reshape[(result.gatherAxis + 1)...]
      )
    }
    return out
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
      public subscript({', '.join(args)}, other: any TensorIndex...) -> Tensor {{
        let idx = [{', '.join(use_args)}] + other
        return self[idx]
      }}
              """.rstrip()
          )
      */

  public subscript(_: UnboundedRange, other: any TensorIndex...)
    -> Tensor
  {
    let idx = [FullRange()] + other
    return self[idx]
  }

  public subscript(_: UnboundedRange, _: UnboundedRange, other: any TensorIndex...,
    backend backend: Backend? = nil
  ) -> Tensor {
    let idx = [FullRange(), FullRange()] + other
    return self[idx]
  }

  public subscript(arg0: any TensorIndex, _: UnboundedRange, other: any TensorIndex...,
    backend backend: Backend? = nil
  ) -> Tensor {
    let idx = [arg0, FullRange()] + other
    return self[idx]
  }

  public subscript(_: UnboundedRange, _: UnboundedRange, _: UnboundedRange,
    other: any TensorIndex..., backend backend: Backend? = nil
  ) -> Tensor {
    let idx = [FullRange(), FullRange(), FullRange()] + other
    return self[idx]
  }

  public subscript(arg0: any TensorIndex, _: UnboundedRange, _: UnboundedRange,
    other: any TensorIndex..., backend backend: Backend? = nil
  ) -> Tensor {
    let idx = [arg0, FullRange(), FullRange()] + other
    return self[idx]
  }

  public subscript(_: UnboundedRange, arg1: any TensorIndex, _: UnboundedRange,
    other: any TensorIndex..., backend backend: Backend? = nil
  ) -> Tensor {
    let idx = [FullRange(), arg1, FullRange()] + other
    return self[idx]
  }

  public subscript(arg0: any TensorIndex, arg1: any TensorIndex, _: UnboundedRange,
    other: any TensorIndex..., backend backend: Backend? = nil
  ) -> Tensor {
    let idx = [arg0, arg1, FullRange()] + other
    return self[idx]
  }

  public subscript(_: UnboundedRange, _: UnboundedRange, _: UnboundedRange, _: UnboundedRange,
    other: any TensorIndex..., backend backend: Backend? = nil
  ) -> Tensor {
    let idx = [FullRange(), FullRange(), FullRange(), FullRange()] + other
    return self[idx]
  }

  public subscript(arg0: any TensorIndex, _: UnboundedRange, _: UnboundedRange, _: UnboundedRange,
    other: any TensorIndex..., backend backend: Backend? = nil
  ) -> Tensor {
    let idx = [arg0, FullRange(), FullRange(), FullRange()] + other
    return self[idx]
  }

  public subscript(_: UnboundedRange, arg1: any TensorIndex, _: UnboundedRange, _: UnboundedRange,
    other: any TensorIndex..., backend backend: Backend? = nil
  ) -> Tensor {
    let idx = [FullRange(), arg1, FullRange(), FullRange()] + other
    return self[idx]
  }

  public subscript(arg0: any TensorIndex, arg1: any TensorIndex, _: UnboundedRange,
    _: UnboundedRange, other: any TensorIndex..., backend backend: Backend? = nil
  ) -> Tensor {
    let idx = [arg0, arg1, FullRange(), FullRange()] + other
    return self[idx]
  }

  public subscript(_: UnboundedRange, _: UnboundedRange, arg2: any TensorIndex, _: UnboundedRange,
    other: any TensorIndex..., backend backend: Backend? = nil
  ) -> Tensor {
    let idx = [FullRange(), FullRange(), arg2, FullRange()] + other
    return self[idx]
  }

  public subscript(arg0: any TensorIndex, _: UnboundedRange, arg2: any TensorIndex,
    _: UnboundedRange, other: any TensorIndex..., backend backend: Backend? = nil
  ) -> Tensor {
    let idx = [arg0, FullRange(), arg2, FullRange()] + other
    return self[idx]
  }

  public subscript(_: UnboundedRange, arg1: any TensorIndex, arg2: any TensorIndex,
    _: UnboundedRange, other: any TensorIndex..., backend backend: Backend? = nil
  ) -> Tensor {
    let idx = [FullRange(), arg1, arg2, FullRange()] + other
    return self[idx]
  }

  public subscript(arg0: any TensorIndex, arg1: any TensorIndex, arg2: any TensorIndex,
    _: UnboundedRange, other: any TensorIndex..., backend backend: Backend? = nil
  ) -> Tensor {
    let idx = [arg0, arg1, arg2, FullRange()] + other
    return self[idx]
  }
}
