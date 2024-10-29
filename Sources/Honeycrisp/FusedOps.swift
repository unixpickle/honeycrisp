extension Tensor {
  public func mul(_ coeff: Tensor, thenAdd bias: Tensor) -> Tensor {
    alwaysAssert(dtype == coeff.dtype, "dtype \(dtype) does not match coefficients \(coeff.dtype)")
    alwaysAssert(dtype == bias.dtype, "dtype \(dtype) does not match bias \(bias.dtype)")

    let broadcasted = Tensor.lazyBroadcast([self, coeff, bias])
    let outputShape = broadcasted.shape
    let (t, coeff, bias) = (broadcasted.tails[0], broadcasted.tails[1], broadcasted.tails[2])

    let backend = Backend.current
    let newData = Tensor.createDataTask(t, coeff, bias) { t, coeff, bias in
      try await backend.mulAdd(
        input: try await t.data, inputCount: t.shape.product(), coeff: coeff.data,
        coeffCount: coeff.shape.product(), bias: bias.data, biasCount: bias.shape.product(),
        count: outputShape.product(), dtype: t.dtype)
    }
    if (needsGrad || coeff.needsGrad || bias.needsGrad) && Tensor.isGradEnabled {
      let handle = t.saveForBackward()
      let coeffHandle = coeff.saveForBackward()
      let biasHandle = bias.saveForBackward()
      return Tensor(dataTask: newData, shape: outputShape, dtype: dtype) { grad in
        handle.backward(backend) { (grad * coeff.noGrad()).reduceOuter(as: t) }
        coeffHandle.backward(backend) { (grad * self.noGrad()).reduceOuter(as: coeff) }
        biasHandle.backward(backend) { grad.reduceOuter(as: bias) }
      }
    } else {
      return Tensor(dataTask: newData, shape: outputShape, dtype: dtype)
    }
  }

  public func add(_ bias: Tensor, thenMul coeff: Tensor) -> Tensor {
    alwaysAssert(dtype == coeff.dtype, "dtype \(dtype) does not match coefficients \(coeff.dtype)")
    alwaysAssert(dtype == bias.dtype, "dtype \(dtype) does not match bias \(bias.dtype)")

    let broadcasted = Tensor.lazyBroadcast([self, coeff, bias])
    let outputShape = broadcasted.shape
    let (t, coeff, bias) = (broadcasted.tails[0], broadcasted.tails[1], broadcasted.tails[2])

    let backend = Backend.current
    let newData = Tensor.createDataTask(t, coeff, bias) { t, coeff, bias in
      try await backend.addMul(
        input: try await t.data, inputCount: t.shape.product(), bias: bias.data,
        biasCount: bias.shape.product(), coeff: coeff.data, coeffCount: coeff.shape.product(),
        count: outputShape.product(), dtype: t.dtype)
    }
    if (needsGrad || coeff.needsGrad || bias.needsGrad) && Tensor.isGradEnabled {
      let handle = t.saveForBackward()
      let coeffHandle = coeff.saveForBackward()
      let biasHandle = bias.saveForBackward()
      return Tensor(dataTask: newData, shape: outputShape, dtype: dtype) { grad in
        handle.backward(backend) { (grad * coeff.noGrad()).reduceOuter(as: t) }
        coeffHandle.backward(backend) {
          (grad * (self.noGrad() + bias.noGrad())).reduceOuter(as: coeff)
        }
        biasHandle.backward(backend) { (grad * coeff.noGrad()).reduceOuter(as: bias) }
      }
    } else {
      return Tensor(dataTask: newData, shape: outputShape, dtype: dtype)
    }
  }
}
