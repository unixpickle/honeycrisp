import HCBacktrace

extension Tensor {
  @recordCaller
  private func _when(isTrue: Tensor, isFalse: Tensor) -> Tensor {
    alwaysAssert(dtype == .bool, "can only call when() on boolean Tensor")
    alwaysAssert(
      isTrue.dtype == isFalse.dtype,
      "when() argument dtypes differ: \(isTrue.dtype) vs \(isFalse.dtype)")
    alwaysAssert(
      isTrue.shape == isFalse.shape,
      "when() argument shapes differ: \(isTrue.shape) vs \(isFalse.shape)")
    alwaysAssert(
      self.shape == isTrue.shape,
      "when() mask shape \(self.shape) does not match argument \(isTrue.shape)")

    let backend = Backend.current
    let newData = Tensor.createDataTask(self, isTrue, isFalse) { t, isTrue, isFalse in
      try await backend.when(
        try await t.data, .tensor(try await isTrue.data), .tensor(try await isFalse.data),
        Float.self, count: t.shape.product(), dtype: isTrue.dtype)
    }
    if !Tensor.isGradEnabled || (!isTrue.needsGrad && !isFalse.needsGrad) {
      return Tensor(dataTask: newData, shape: shape, dtype: isTrue.dtype)
    } else {
      let lhsHandle = isTrue.saveForBackward()
      let rhsHandle = isFalse.saveForBackward()
      return Tensor(dataTask: newData, shape: shape, dtype: isTrue.dtype) { grad in
        lhsHandle.backward(backend) { self.when(isTrue: grad, isFalse: 0) }
        rhsHandle.backward(backend) { self.when(isTrue: 0, isFalse: grad) }
      }
    }
  }

  @recordCaller
  private func _when<T: TensorElement>(isTrue: Tensor, isFalse: T) -> Tensor {
    alwaysAssert(dtype == .bool, "can only call when() on boolean Tensor")
    alwaysAssert(
      self.shape == isTrue.shape,
      "when() mask shape \(self.shape) does not match argument \(isTrue.shape)")

    let backend = Backend.current
    let newData = Tensor.createDataTask(self, isTrue) { t, isTrue in
      try await backend.when(
        try await t.data, .tensor(try await isTrue.data), .scalar(isFalse), T.self,
        count: t.shape.product(), dtype: isTrue.dtype)
    }
    if !Tensor.isGradEnabled || (!isTrue.needsGrad) {
      return Tensor(dataTask: newData, shape: shape, dtype: isTrue.dtype)
    } else {
      let handle = isTrue.saveForBackward()
      return Tensor(dataTask: newData, shape: shape, dtype: isTrue.dtype) { grad in
        handle.backward(backend) { self.when(isTrue: grad, isFalse: 0) }
      }
    }
  }

  @recordCaller
  private func _when<T: TensorElement>(isTrue: T, isFalse: Tensor) -> Tensor {
    alwaysAssert(dtype == .bool, "can only call when() on boolean Tensor")
    alwaysAssert(
      self.shape == isFalse.shape,
      "when() mask shape \(self.shape) does not match argument \(isFalse.shape)")

    let backend = Backend.current
    let newData = Tensor.createDataTask(self, isFalse) { t, isFalse in
      try await backend.when(
        try await t.data, .scalar(isTrue), .tensor(try await isFalse.data), T.self,
        count: t.shape.product(), dtype: isFalse.dtype)
    }
    if !Tensor.isGradEnabled || (!isFalse.needsGrad) {
      return Tensor(dataTask: newData, shape: shape, dtype: isFalse.dtype)
    } else {
      let handle = isFalse.saveForBackward()
      return Tensor(dataTask: newData, shape: shape, dtype: isFalse.dtype) { grad in
        handle.backward(backend) { self.when(isTrue: 0, isFalse: grad) }
      }
    }
  }

  @recordCaller
  private func _when<T: TensorElement>(isTrue: T, isFalse: T, dtype: DType) -> Tensor {
    alwaysAssert(dtype == .bool, "can only call when() on boolean Tensor")
    let backend = Backend.current
    let newData = createDataTask { t in
      try await backend.when(
        try await t.data, .scalar(isTrue), .scalar(isFalse), T.self, count: t.shape.product(),
        dtype: dtype)
    }
    return Tensor(dataTask: newData, shape: shape, dtype: dtype)
  }
}
