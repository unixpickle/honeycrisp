public struct ScatterGatherIndices {
  public let broadcasted: Bool
  public let indices: Tensor.Data
  public let outCount: Int

  public let outerCount: Int
  public let middleCount: Int
  public let innerCount: Int

  public var indicesCount: Int {
    if broadcasted {
      outCount
    } else {
      gatherOutCount
    }
  }

  public var gatherInCount: Int {
    outerCount * middleCount * innerCount
  }

  public var gatherOutCount: Int {
    outerCount * outCount * innerCount
  }
}

extension Tensor {
  public func gather(axis: Int, indices: Tensor) -> Tensor {
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
    let newData = Task {
      try await backend.gather(
        try await self.data,
        ScatterGatherIndices(
          broadcasted: indices.shape.count != shape.count,
          indices: try await indices.data,
          outCount: indices.shape.count == 1 ? indices.shape[0] : indices.shape[axis],
          outerCount: shape[..<axis].product(),
          middleCount: shape[axis],
          innerCount: shape[(axis + 1)...].product()
        ),
        dtype: dtype)
    }
    if !needsGrad || !Tensor.isGradEnabled {
      return Tensor(dataTask: newData, shape: newShape, dtype: dtype)
    } else {
      let handle = self.saveForBackward()
      return Tensor(dataTask: newData, shape: newShape, dtype: dtype) { [self] grad in
        handle.backward(
          backend.use {
            grad.scatter(axis: axis, count: shape[axis], indices: indices)
          })
      }
    }
  }

  public func scatter(axis: Int, count: Int, indices: Tensor) -> Tensor {
    let axis = positiveAxis(axis)
    alwaysAssert(indices.dtype == .int64, "can only scatter with indices of dtype \(indices.dtype)")
    alwaysAssert(
      shape == indices.shape || (indices.shape.count == 1 && shape[axis] == indices.shape[0]),
      "incompatible indices shape \(indices.shape) for tensor shape \(shape)")

    var newShape = shape
    newShape[axis] = count

    let backend = Backend.current
    let newData = Task { [self] in
      try await backend.scatter(
        try await self.data,
        ScatterGatherIndices(
          broadcasted: indices.shape.count != shape.count,
          indices: try await indices.data,
          outCount: indices.shape.count == 1 ? indices.shape[0] : indices.shape[axis],
          outerCount: shape[..<axis].product(),
          middleCount: count,
          innerCount: shape[(axis + 1)...].product()
        ), dtype: dtype)
    }
    if !needsGrad || !Tensor.isGradEnabled {
      return Tensor(dataTask: newData, shape: newShape, dtype: dtype)
    } else {
      let handle = self.saveForBackward()
      return Tensor(dataTask: newData, shape: newShape, dtype: dtype) { grad in
        handle.backward(
          backend.use { grad.gather(axis: axis, indices: indices) })
      }
    }
  }
}
