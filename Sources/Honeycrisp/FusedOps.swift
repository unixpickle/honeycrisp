extension Tensor {
  public func mul(_ coeff: Tensor, thenAdd bias: Tensor) -> Tensor {
    alwaysAssert(dtype == coeff.dtype, "dtype \(dtype) does not match coefficients \(coeff.dtype)")
    alwaysAssert(dtype == bias.dtype, "dtype \(dtype) does not match bias \(bias.dtype)")

    let broadcasted = Tensor.lazyBroadcast([self, coeff, bias])
    let outputShape = broadcasted.0
    let (t, tStrides) = broadcasted.1[0]
    let (coeff, coeffStrides) = broadcasted.1[1]
    let (bias, biasStrides) = broadcasted.1[2]

    let backend = Backend.current
    let newData = Tensor.createDataTask(t, coeff, bias) { t, coeff, bias in
      try await backend.mulAdd(
        input: BroadcastData(strides: tStrides, data: try await t.data),
        coeff: BroadcastData(strides: coeffStrides, data: try await coeff.data),
        bias: BroadcastData(strides: biasStrides, data: try await bias.data),
        count: outputShape.product(), dtype: t.dtype)
    }
    if (needsGrad || coeff.needsGrad || bias.needsGrad) && Tensor.isGradEnabled {
      let handle = t.saveForBackward()
      let coeffHandle = coeff.saveForBackward()
      let biasHandle = bias.saveForBackward()
      return Tensor(dataTask: newData, shape: outputShape, dtype: dtype) { grad in
        handle.backward(backend) { (grad * coeff.noGrad()).reduceBroadcast(tStrides, as: t) }
        coeffHandle.backward(backend) {
          (grad * t.noGrad()).reduceBroadcast(coeffStrides, as: coeff)
        }
        biasHandle.backward(backend) { grad.reduceBroadcast(biasStrides, as: bias) }
      }
    } else {
      return Tensor(dataTask: newData, shape: outputShape, dtype: dtype)
    }
  }

  public func add(_ bias: Tensor, thenMul coeff: Tensor) -> Tensor {
    alwaysAssert(dtype == coeff.dtype, "dtype \(dtype) does not match coefficients \(coeff.dtype)")
    alwaysAssert(dtype == bias.dtype, "dtype \(dtype) does not match bias \(bias.dtype)")

    let broadcasted = Tensor.lazyBroadcast([self, coeff, bias])
    let outputShape = broadcasted.0
    let (t, tStrides) = broadcasted.1[0]
    let (coeff, coeffStrides) = broadcasted.1[1]
    let (bias, biasStrides) = broadcasted.1[2]

    let backend = Backend.current
    let newData = Tensor.createDataTask(t, coeff, bias) { t, coeff, bias in
      try await backend.addMul(
        input: BroadcastData(strides: tStrides, data: try await t.data),
        bias: BroadcastData(strides: biasStrides, data: try await bias.data),
        coeff: BroadcastData(strides: coeffStrides, data: try await coeff.data),
        count: outputShape.product(), dtype: t.dtype)
    }
    if (needsGrad || coeff.needsGrad || bias.needsGrad) && Tensor.isGradEnabled {
      let handle = t.saveForBackward()
      let coeffHandle = coeff.saveForBackward()
      let biasHandle = bias.saveForBackward()
      return Tensor(dataTask: newData, shape: outputShape, dtype: dtype) { grad in
        handle.backward(backend) { (grad * coeff.noGrad()).reduceBroadcast(tStrides, as: t) }
        coeffHandle.backward(backend) {
          (grad * (t.noGrad() + bias.noGrad())).reduceBroadcast(coeffStrides, as: coeff)
        }
        biasHandle.backward(backend) {
          (grad * coeff.noGrad()).reduceBroadcast(biasStrides, as: bias)
        }
      }
    } else {
      return Tensor(dataTask: newData, shape: outputShape, dtype: dtype)
    }
  }
}
