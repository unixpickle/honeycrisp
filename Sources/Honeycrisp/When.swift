import HCBacktrace

extension Tensor {
  @recordCaller
  private func _when(isTrue: Tensor, isFalse: Tensor) -> Tensor {
    alwaysAssert(dtype == .bool, "can only call when() on boolean Tensor")
    alwaysAssert(
      isTrue.dtype == isFalse.dtype,
      "when() argument dtypes differ: \(isTrue.dtype) vs \(isFalse.dtype)")

    let (newShape, allStrides) = Tensor.lazyBroadcast([self, isTrue, isFalse])
    let tStrides = allStrides[0]
    let isTrueStrides = allStrides[1]
    let isFalseStrides = allStrides[2]

    let backend = Backend.current
    let newData = Tensor.createDataTask(self, isTrue, isFalse) { t, isTrue, isFalse in
      try await backend.when(
        BroadcastData(strides: tStrides, data: try await t.data),
        .tensor(BroadcastData(strides: isTrueStrides, data: try await isTrue.data)),
        .tensor(BroadcastData(strides: isFalseStrides, data: try await isFalse.data)),
        Float.self,
        count: newShape.product(),
        dtype: isTrue.dtype)
    }
    if !Tensor.isGradEnabled || (!isTrue.needsGrad && !isFalse.needsGrad) {
      return Tensor(dataTask: newData, shape: newShape, dtype: isTrue.dtype)
    } else {
      let lhsHandle = isTrue.saveForBackward()
      let rhsHandle = isFalse.saveForBackward()
      return Tensor(dataTask: newData, shape: newShape, dtype: isTrue.dtype) { grad in
        lhsHandle.backward(backend) {
          self.when(isTrue: grad, isFalse: 0).reduceBroadcast(isTrueStrides, as: isTrue)
        }
        rhsHandle.backward(backend) {
          self.when(isTrue: 0, isFalse: grad).reduceBroadcast(isFalseStrides, as: isFalse)
        }
      }
    }
  }

  @recordCaller
  private func _when<T: TensorElement>(isTrue: Tensor, isFalse: T) -> Tensor {
    alwaysAssert(dtype == .bool, "can only call when() on boolean Tensor")

    let (newShape, (tStrides, isTrueStrides)) = Tensor.lazyBroadcast(self, isTrue)

    let backend = Backend.current
    let newData = Tensor.createDataTask(self, isTrue) { t, isTrue in
      try await backend.when(
        BroadcastData(strides: tStrides, data: try await t.data),
        .tensor(BroadcastData(strides: isTrueStrides, data: try await isTrue.data)),
        .scalar(isFalse, newShape),
        T.self,
        count: newShape.product(),
        dtype: isTrue.dtype)
    }
    if !Tensor.isGradEnabled || (!isTrue.needsGrad) {
      return Tensor(dataTask: newData, shape: newShape, dtype: isTrue.dtype)
    } else {
      let handle = isTrue.saveForBackward()
      return Tensor(dataTask: newData, shape: newShape, dtype: isTrue.dtype) { grad in
        handle.backward(backend) {
          self.when(isTrue: grad, isFalse: 0).reduceBroadcast(isTrueStrides, as: isTrue)
        }
      }
    }
  }

  @recordCaller
  private func _when<T: TensorElement>(isTrue: T, isFalse: Tensor) -> Tensor {
    alwaysAssert(dtype == .bool, "can only call when() on boolean Tensor")

    let (newShape, (tStrides, isFalseStrides)) = Tensor.lazyBroadcast(self, isFalse)

    let backend = Backend.current
    let newData = Tensor.createDataTask(self, isFalse) { t, isFalse in
      try await backend.when(
        BroadcastData(strides: tStrides, data: try await t.data),
        .scalar(isTrue, newShape),
        .tensor(BroadcastData(strides: isFalseStrides, data: try await isFalse.data)),
        T.self,
        count: newShape.product(),
        dtype: isFalse.dtype)
    }
    if !Tensor.isGradEnabled || (!isFalse.needsGrad) {
      return Tensor(dataTask: newData, shape: newShape, dtype: isFalse.dtype)
    } else {
      let handle = isFalse.saveForBackward()
      return Tensor(dataTask: newData, shape: newShape, dtype: isFalse.dtype) { grad in
        handle.backward(backend) {
          self.when(isTrue: 0, isFalse: grad).reduceBroadcast(isFalseStrides, as: isFalse)
        }
      }
    }
  }

  @recordCaller
  private func _when<T: TensorElement>(isTrue: T, isFalse: T, dtype outDType: DType) -> Tensor {
    alwaysAssert(dtype == .bool, "can only call when() on boolean Tensor")
    let backend = Backend.current
    let newData = createDataTask { t in
      try await backend.when(
        BroadcastData.simple(data: try await t.data, shape: t.shape),
        .scalar(isTrue, t.shape),
        .scalar(isFalse, t.shape),
        T.self,
        count: t.shape.product(),
        dtype: outDType)
    }
    return Tensor(dataTask: newData, shape: shape, dtype: outDType)
  }
}
