import HCBacktrace

/// A low-level description of indexing for a gather or scatter operation.
public struct ScatterGatherIndices {
  /// The shape of the tensor of input values before the gather.
  public let valueShape: [Int]

  /// The axis to perform the gather along.
  public let axis: Int

  /// The int64 tensor data containing the indices.
  /// Must have the same shape as valueShape.
  public let indices: BroadcastData

  /// If `true`, then no index is repeated inside of ``ScatterGatherIndices/indices``
  /// within the gathered axis.
  ///
  /// This is used to determine if reduction is necessary during a scatter.
  public let indicesAreUnique: Bool

  var outerCount: Int {
    valueShape[..<axis].product()
  }

  var middleCount: Int {
    valueShape[axis]
  }

  var innerCount: Int {
    valueShape[(axis + 1)...].product()
  }

  var outCount: Int {
    indices.strides.shape[axis]
  }

  /// The number of input elements for a gather, or output elements for a scatter.
  public var gatherInCount: Int {
    valueShape.product()
  }

  /// The number of output elements for a gather, or input elements for a scatter.
  public var gatherOutCount: Int {
    return indices.strides.shape.product()
  }
}

extension Tensor {

  @recordCaller
  internal func _expandIndices(axis: Int, indices: Tensor) -> (Tensor, BroadcastStrides) {
    let indices =
      if indices.shape.count == 1 {
        {
          var bcastShape = Array(repeating: 1, count: shape.count)
          bcastShape[axis] = indices.shape[0]
          return indices.reshape(bcastShape)
        }()
      } else {
        indices
      }

    var expandShape = shape
    expandShape[axis] = indices.shape[axis]
    let indicesStrides = indices.expandStrides(shape: expandShape)

    return (indices, indicesStrides)
  }

  @recordCaller
  private func _gather(axis: Int, indices: Tensor, indicesAreUnique: Bool = false) -> Tensor {
    alwaysAssert(shape.count > 0, "cannot gather() on a zero-dimensional tensor")
    alwaysAssert(
      indices.dtype == .int64,
      "can only gather with indices of dtype int64, but got \(indices.dtype)")

    let axis = positiveAxis(axis)
    let (indices, indicesStrides) = expandIndices(axis: axis, indices: indices)

    let newShape = indicesStrides.shape
    let backend = Backend.current
    let newData = Tensor.createDataTask(self, indices) { t, indices in
      try await backend.gather(
        try await t.data,
        ScatterGatherIndices(
          valueShape: t.shape,
          axis: axis,
          indices: BroadcastData(strides: indicesStrides, data: try await indices.data),
          indicesAreUnique: indicesAreUnique
        ),
        dtype: t.dtype
      )
    }
    if !needsGrad || !Tensor.isGradEnabled {
      return Tensor(dataTask: newData, shape: newShape, dtype: dtype)
    } else {
      let handle = self.saveForBackward()
      return Tensor(dataTask: newData, shape: newShape, dtype: dtype) { [self] grad in
        let shape = shape
        handle.backward(backend) {
          grad.scatter(
            axis: axis,
            count: shape[axis],
            indices: indices,
            indicesAreUnique: indicesAreUnique
          )
        }
      }
    }
  }

  @recordCaller
  private func _scatter(axis: Int, count: Int, indices: Tensor, indicesAreUnique: Bool = false)
    -> Tensor
  {
    alwaysAssert(shape.count > 0, "cannot scatter() on a zero-dimensional tensor")
    alwaysAssert(
      indices.dtype == .int64,
      "can only scatter with indices of dtype int64, but got \(indices.dtype)")

    let axis = positiveAxis(axis)
    let (indices, indicesStrides) = expandIndices(axis: axis, indices: indices)

    var newShape = shape
    newShape[axis] = count

    let backend = Backend.current
    let newData = Tensor.createDataTask(self, indices) { t, indices in
      try await backend.scatter(
        try await t.data,
        ScatterGatherIndices(
          valueShape: newShape,
          axis: axis,
          indices: BroadcastData(strides: indicesStrides, data: try await indices.data),
          indicesAreUnique: indicesAreUnique
        ), dtype: t.dtype)
    }
    if !needsGrad || !Tensor.isGradEnabled {
      return Tensor(dataTask: newData, shape: newShape, dtype: dtype)
    } else {
      let handle = self.saveForBackward()
      return Tensor(dataTask: newData, shape: newShape, dtype: dtype) { grad in
        handle.backward(backend) {
          grad.gather(axis: axis, indices: indices, indicesAreUnique: indicesAreUnique)
        }
      }
    }
  }

}
