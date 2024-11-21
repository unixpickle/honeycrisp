import HCBacktrace

/// A low-level description of indexing for a gather or scatter operation.
public struct ScatterGatherIndices {
  /// If true, then the indices are broadcasted along all non-indexed dimensions.
  /// Otherwise, there is a unique index for every output position in the tensor.
  public let broadcasted: Bool

  /// The int64 tensor data containing the indices.
  public let indices: Tensor.Data

  /// If `true`, then no index is repeated inside of ``ScatterGatherIndices/indices``.
  ///
  /// This is used to determine if reduction is necessary during a scatter.
  public let indicesAreUnique: Bool

  /// The new size of the gathered dimension.
  public let outCount: Int

  /// The product of all leading dimensions before the affected axis.
  public let outerCount: Int

  /// The size of the gathered dimension before the gather.
  public let middleCount: Int

  /// The product of all trailing dimensions after the affected axis.
  public let innerCount: Int

  /// The number of indices in the ``ScatterGatherIndices/indices`` data.
  ///
  /// This depends on whether this operation is broadcasted.
  public var indicesCount: Int {
    if broadcasted {
      outCount
    } else {
      gatherOutCount
    }
  }

  /// The number of input elements for a gather, or output elements for a scatter.
  public var gatherInCount: Int {
    outerCount * middleCount * innerCount
  }

  /// The number of output elements for a gather, or input elements for a scatter.
  public var gatherOutCount: Int {
    outerCount * outCount * innerCount
  }
}

extension Tensor {

  @recordCaller
  private func _gather(axis: Int, indices: Tensor, indicesAreUnique: Bool = false) -> Tensor {
    let axis = positiveAxis(axis)
    alwaysAssert(indices.dtype == .int64, "can only gather with indices of dtype \(indices.dtype)")
    alwaysAssert(
      shape.count == indices.shape.count || indices.shape.count == 1,
      "incompatible indices shape \(indices.shape) for tensor shape \(shape)")

    var newShape = shape
    newShape[axis] = indices.shape.count == 1 ? indices.shape[0] : indices.shape[axis]
    if indices.shape.count != 1 {
      alwaysAssert(
        newShape == indices.shape,
        "tensor shape \(shape) must match indices shape \(indices) except at axis \(axis)")
    }

    let backend = Backend.current
    let newData = Tensor.createDataTask(self, indices) { t, indices in
      try await backend.gather(
        try await t.data,
        ScatterGatherIndices(
          broadcasted: indices.shape.count != t.shape.count,
          indices: try await indices.data,
          indicesAreUnique: indicesAreUnique,
          outCount: indices.shape.count == 1 ? indices.shape[0] : indices.shape[axis],
          outerCount: t.shape[..<axis].product(),
          middleCount: t.shape[axis],
          innerCount: t.shape[(axis + 1)...].product()
        ),
        dtype: t.dtype)
    }
    if !needsGrad || !Tensor.isGradEnabled {
      return Tensor(dataTask: newData, shape: newShape, dtype: dtype)
    } else {
      let handle = self.saveForBackward()
      return Tensor(dataTask: newData, shape: newShape, dtype: dtype) { [self] grad in
        let shape = shape
        handle.backward(backend) {
          grad.scatter(
            axis: axis, count: shape[axis], indices: indices, indicesAreUnique: indicesAreUnique)
        }
      }
    }
  }

  @recordCaller
  private func _scatter(axis: Int, count: Int, indices: Tensor, indicesAreUnique: Bool = false)
    -> Tensor
  {
    let axis = positiveAxis(axis)
    alwaysAssert(indices.dtype == .int64, "can only scatter with indices of dtype \(indices.dtype)")
    alwaysAssert(
      shape == indices.shape || (indices.shape.count == 1 && shape[axis] == indices.shape[0]),
      "incompatible indices shape \(indices.shape) for tensor shape \(shape)")

    var newShape = shape
    newShape[axis] = count

    let backend = Backend.current
    let newData = Tensor.createDataTask(self, indices) { t, indices in
      try await backend.scatter(
        try await t.data,
        ScatterGatherIndices(
          broadcasted: indices.shape.count != t.shape.count,
          indices: try await indices.data,
          indicesAreUnique: indicesAreUnique,
          outCount: indices.shape.count == 1 ? indices.shape[0] : indices.shape[axis],
          outerCount: t.shape[..<axis].product(),
          middleCount: count,
          innerCount: t.shape[(axis + 1)...].product()
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
