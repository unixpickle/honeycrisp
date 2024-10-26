extension Tensor {
  public func mul(_ coeff: Tensor, thenAdd bias: Tensor) -> Tensor {
    alwaysAssert(shape == coeff.shape, "shape \(shape) does not match coefficients \(coeff.shape)")
    alwaysAssert(dtype == coeff.dtype, "dtype \(dtype) does not match coefficients \(coeff.dtype)")
    alwaysAssert(shape == bias.shape, "shape \(shape) does not match bias \(bias.shape)")
    alwaysAssert(dtype == bias.dtype, "dtype \(dtype) does not match bias \(bias.dtype)")

    let backend = Backend.current
    let newData = Tensor.createDataTask(self, coeff, bias) { t, coeff, bias in
      try await backend.mulAdd(
        try await t.data, coeff: coeff.data, bias: bias.data, count: t.shape.product(),
        dtype: t.dtype)
    }
    if (needsGrad || coeff.needsGrad || bias.needsGrad) && Tensor.isGradEnabled {
      let handle = self.saveForBackward()
      let coeffHandle = coeff.saveForBackward()
      let biasHandle = bias.saveForBackward()
      return Tensor(dataTask: newData, shape: shape, dtype: dtype) { grad in
        handle.backward(backend) { grad * coeff.noGrad() }
        coeffHandle.backward(backend) { grad * self.noGrad() }
        biasHandle.backward(backend) { grad }
      }
    } else {
      return Tensor(dataTask: newData, shape: shape, dtype: dtype)
    }
  }

  public func add(_ bias: Tensor, thenMul coeff: Tensor) -> Tensor {
    alwaysAssert(shape == coeff.shape, "shape \(shape) does not match coefficients \(coeff.shape)")
    alwaysAssert(dtype == coeff.dtype, "dtype \(dtype) does not match coefficients \(coeff.dtype)")
    alwaysAssert(shape == bias.shape, "shape \(shape) does not match bias \(bias.shape)")
    alwaysAssert(dtype == bias.dtype, "dtype \(dtype) does not match bias \(bias.dtype)")

    let backend = Backend.current
    let newData = Tensor.createDataTask(self, coeff, bias) { t, coeff, bias in
      try await backend.addMul(
        try await t.data, bias: bias.data, coeff: coeff.data, count: t.shape.product(),
        dtype: t.dtype)
    }
    if (needsGrad || coeff.needsGrad || bias.needsGrad) && Tensor.isGradEnabled {
      let handle = self.saveForBackward()
      let coeffHandle = coeff.saveForBackward()
      let biasHandle = bias.saveForBackward()
      return Tensor(dataTask: newData, shape: shape, dtype: dtype) { grad in
        handle.backward(backend) { grad * coeff.noGrad() }
        coeffHandle.backward(backend) { grad * (self.noGrad() + bias.noGrad()) }
        biasHandle.backward(backend) { grad * coeff.noGrad() }
      }
    } else {
      return Tensor(dataTask: newData, shape: shape, dtype: dtype)
    }
  }
}
