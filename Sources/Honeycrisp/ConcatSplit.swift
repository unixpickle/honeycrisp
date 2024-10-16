extension Tensor {
  public convenience init(concat tensors: [Tensor], axis: Int = 0) {
    let backend = Backend.current

    alwaysAssert(tensors.count > 0, "cannot concatenate zero tensors")

    let axis = tensors[0].positiveAxis(axis)
    alwaysAssert(
      axis >= 0 && axis < tensors[0].shape.count,
      "axis \(axis) out of bounds for shape \(tensors[0].shape)")

    for (i, t) in tensors.enumerated() {
      alwaysAssert(
        t.dtype == tensors[0].dtype,
        "tensor at index \(i) has different dtype \(t.dtype) than tensor 0 \(tensors[0].dtype)")
      alwaysAssert(
        t.shape.count == tensors[0].shape.count && t.shape[..<axis] == tensors[0].shape[..<axis]
          && t.shape[(axis + 1)...] == tensors[0].shape[(axis + 1)...],
        "tensor at index \(i) has shape \(t.shape) which is incompatible with shape at index 0 \(tensors[0].shape)"
      )
    }

    let middleCounts: [Int] = tensors.map { $0.shape[axis] }
    let innerCounts: [Int] = tensors.map { $0.shape[axis...].product() }
    let outerCount = tensors[0].shape[..<axis].product()
    let dtype = tensors[0].dtype

    // We cannot use createDataTask(), so we explicitly detach all arguments.
    let capturedTensors = tensors.map { $0.noGrad() }
    let newData = Task {
      var datas: [Tensor.Data] = []
      for tensor in capturedTensors {
        datas.append(try await tensor.data)
      }
      return try await backend.concat(
        datas, outerCount: outerCount, innerCounts: innerCounts, dtype: dtype)
    }
    var newShape = tensors[0].shape
    newShape[axis] = middleCounts.sum()
    if tensors.map({ $0.needsGrad }).allSatisfy({ $0 == false }) {
      self.init(dataTask: newData, shape: newShape, dtype: dtype)
    } else {
      let handles = tensors.map { $0.saveForBackward() }
      self.init(dataTask: newData, shape: newShape, dtype: dtype) { grad in
        let pieces = backend.use { grad.split(axis: axis, counts: middleCounts) }
        for (handle, piece) in zip(handles, pieces) {
          handle.backward(backend) { piece }
        }
      }
    }
  }

  public convenience init(stack tensors: [Tensor], axis: Int = 0) {
    self.init(concat: tensors.map { x in x.unsqueeze(axis: axis) }, axis: axis)
  }

  public func split(axis: Int, counts: [Int]) -> [Tensor] {
    let axis = positiveAxis(axis)
    alwaysAssert(axis >= 0 && axis < shape.count, "axis \(axis) out of bounds for shape \(shape)")
    alwaysAssert(
      shape[axis] == counts.sum(),
      "split counts \(counts) do not sum to axis \(axis) of shape \(shape)")
    var results: [Tensor] = []
    var start = 0
    for count in counts {
      results.append(self[FullRange(dims: axis), start..<(start + count)])
      start += count
    }
    return results
  }
}
