extension Tensor {

  public func logSoftmax(axis: Int = -1) -> Tensor {
    alwaysAssert(dtype.isNumeric, "dtype \(dtype) not supported for logSoftmax")
    let posAxis = positiveAxis(axis)
    alwaysAssert(posAxis >= 0 && posAxis < shape.count, "invalid axis \(axis) for shape \(shape)")

    let backend = Backend.current
    let newData = createDataTask { t in
      try await backend.logSoftmax(
        try await t.data,
        outerCount: t.shape[..<posAxis].product(),
        middleCount: t.shape[posAxis],
        innerCount: t.shape[(posAxis + 1)...].product(),
        dtype: t.dtype)
    }
    if !needsGrad || !Tensor.isGradEnabled {
      return Tensor(dataTask: newData, shape: shape, dtype: dtype)
    } else {
      let handle = self.saveForBackward()
      return Tensor(dataTask: newData, shape: shape, dtype: dtype) { [self] grad in
        handle.backward(backend) {
          Tensor.logSoftmaxGrad(inputs: self.noGrad(), grads: grad, axis: axis)
        }
      }
    }
  }

  static public func logSoftmaxGrad(inputs: Tensor, grads: Tensor, axis: Int = -1) -> Tensor {
    alwaysAssert(inputs.dtype.isNumeric)
    alwaysAssert(inputs.dtype == grads.dtype)
    alwaysAssert(inputs.shape == grads.shape)
    let posAxis = inputs.positiveAxis(axis)
    alwaysAssert(
      posAxis >= 0 && posAxis < inputs.shape.count, "invalid axis \(axis) for shape \(inputs.shape)"
    )

    let backend = Backend.current
    let newData = createDataTask(inputs, grads) { inputs, grads in
      try await backend.logSoftmaxGrad(
        try await inputs.data,
        try await grads.data,
        outerCount: inputs.shape[..<posAxis].product(),
        middleCount: inputs.shape[posAxis],
        innerCount: inputs.shape[(posAxis + 1)...].product(),
        dtype: inputs.dtype)
    }
    if (!inputs.needsGrad && !grads.needsGrad) || !Tensor.isGradEnabled {
      return Tensor(dataTask: newData, shape: inputs.shape, dtype: inputs.dtype)
    } else {
      fatalError("gradients not supported for logSoftmaxGrad operation")
    }
  }

  public func softmax(axis: Int = -1) -> Tensor {
    logSoftmax().exp()
  }

}
