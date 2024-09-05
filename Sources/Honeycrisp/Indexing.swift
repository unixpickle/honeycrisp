public struct ScatterGatherIndices {
  public let broadcasted: Bool
  public let indices: Tensor.Data
  public let outCount: Int

  public let outerCount: Int
  public let middleCount: Int
  public let innerCount: Int

  var indicesCount: Int {
    if broadcasted {
      outCount
    } else {
      scatterOutCount
    }
  }

  var scatterInCount: Int {
    outerCount * middleCount * innerCount
  }

  var scatterOutCount: Int {
    outerCount * outCount * innerCount
  }
}

extension Tensor {
  public func gather(axis: Int, indices: Tensor, backend: Backend? = nil) -> Tensor {
    let axis = positiveAxis(axis)!
    assert(indices.dtype == .int64, "can only gather with indices of dtype \(indices.dtype)")
    assert(
      shape.count == indices.shape.count || indices.shape.count == 1,
      "incompatible indices shape \(indices.shape) for tensor shape \(shape)")

    var newShape = shape
    newShape[axis] = indices.shape.count == 1 ? indices.shape[0] : indices.shape[axis]
    if indices.shape.count != 1 {
      assert(
        newShape == indices.shape,
        "tensor shape \(shape) must match indices shape \(indices) except at axis \(axis)")
    }

    let backend = backend ?? Backend.defaultBackend
    let newData = Task {
      let (innerData, indexData) = try await backend.waitForData(await data, await indices.data)
      return try await backend.execute { handle in
        let info = ScatterGatherIndices(
          broadcasted: indices.shape.count != shape.count,
          indices: indexData,
          outCount: indices.shape.count == 1 ? indices.shape[0] : indices.shape[axis],
          outerCount: shape[..<axis].product(),
          middleCount: shape[axis],
          innerCount: shape[(axis + 1)...].product()
        )
        return try handle.gather(innerData, info, dtype: dtype)
      }
    }
    if !needsGrad {
      return Tensor(dataTask: newData, shape: newShape, dtype: dtype)
    } else {
      let handle = self.saveForBackward()
      return Tensor(dataTask: newData, shape: newShape, dtype: dtype) { [self] grad in
        try handle.backward(
          grad.scatter(axis: axis, count: shape[axis], indices: indices, backend: backend))
      }
    }
  }

  public func scatter(axis: Int, count: Int, indices: Tensor, backend: Backend? = nil) -> Tensor {
    let axis = positiveAxis(axis)!
    assert(indices.dtype == .int64, "can only scatter with indices of dtype \(indices.dtype)")
    assert(
      shape == indices.shape || (indices.shape.count == 1 && shape[axis] == indices.shape[0]),
      "incompatible indices shape \(indices.shape) for tensor shape \(shape)")

    var newShape = shape
    newShape[axis] = count

    let backend = backend ?? Backend.defaultBackend
    let newData = Task { [self] in
      let (innerData, indexData) = try await backend.waitForData(await data, await indices.data)
      return try await backend.execute { handle in
        let selection = ScatterGatherIndices(
          broadcasted: indices.shape.count != shape.count,
          indices: indexData,
          outCount: indices.shape.count == 1 ? indices.shape[0] : indices.shape[axis],
          outerCount: shape[..<axis].product(),
          middleCount: count,
          innerCount: shape[(axis + 1)...].product()
        )
        return try handle.scatter(innerData, selection, dtype: dtype)
      }
    }
    if !needsGrad {
      return Tensor(dataTask: newData, shape: newShape, dtype: dtype)
    } else {
      let handle = self.saveForBackward()
      return Tensor(dataTask: newData, shape: newShape, dtype: dtype) { grad in
        try handle.backward(grad.gather(axis: axis, indices: indices, backend: backend))
      }
    }
  }
}
