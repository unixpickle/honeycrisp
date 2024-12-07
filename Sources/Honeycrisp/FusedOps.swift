import HCBacktrace

extension Tensor {
  @recordCaller
  private func _mul(_ coeff: Tensor, thenAdd bias: Tensor) -> Tensor {
    #alwaysAssert(dtype == coeff.dtype, "dtype \(dtype) does not match coefficients \(coeff.dtype)")
    #alwaysAssert(dtype == bias.dtype, "dtype \(dtype) does not match bias \(bias.dtype)")

    let (outputShape, allStrides) = Tensor.lazyBroadcast([self, coeff, bias])
    let tStrides = allStrides[0]
    let coeffStrides = allStrides[1]
    let biasStrides = allStrides[2]

    let backend = Backend.current
    let newData = Tensor.createDataTask(self, coeff, bias) { t, coeff, bias in
      try await backend.mulAdd(
        input: BroadcastData(strides: tStrides, data: try await t.data),
        coeff: BroadcastData(strides: coeffStrides, data: try await coeff.data),
        bias: BroadcastData(strides: biasStrides, data: try await bias.data),
        dtype: t.dtype
      )
    }
    if (needsGrad || coeff.needsGrad || bias.needsGrad) && Tensor.isGradEnabled {
      let handle = self.saveForBackward()
      let coeffHandle = coeff.saveForBackward()
      let biasHandle = bias.saveForBackward()
      return Tensor(dataTask: newData, shape: outputShape, dtype: dtype) { grad in
        handle.backward(backend) { (grad * coeff.noGrad()).reduceBroadcast(tStrides, as: self) }
        coeffHandle.backward(backend) {
          (grad * self.noGrad()).reduceBroadcast(coeffStrides, as: coeff)
        }
        biasHandle.backward(backend) { grad.reduceBroadcast(biasStrides, as: bias) }
      }
    } else {
      return Tensor(dataTask: newData, shape: outputShape, dtype: dtype)
    }
  }

  @recordCaller
  private func _add(_ bias: Tensor, thenMul coeff: Tensor) -> Tensor {
    #alwaysAssert(dtype == coeff.dtype, "dtype \(dtype) does not match coefficients \(coeff.dtype)")
    #alwaysAssert(dtype == bias.dtype, "dtype \(dtype) does not match bias \(bias.dtype)")

    let (outputShape, allStrides) = Tensor.lazyBroadcast([self, coeff, bias])
    let tStrides = allStrides[0]
    let coeffStrides = allStrides[1]
    let biasStrides = allStrides[2]

    let backend = Backend.current
    let newData = Tensor.createDataTask(self, coeff, bias) { t, coeff, bias in
      try await backend.addMul(
        input: BroadcastData(strides: tStrides, data: try await t.data),
        bias: BroadcastData(strides: biasStrides, data: try await bias.data),
        coeff: BroadcastData(strides: coeffStrides, data: try await coeff.data),
        dtype: t.dtype
      )
    }
    if (needsGrad || coeff.needsGrad || bias.needsGrad) && Tensor.isGradEnabled {
      let handle = self.saveForBackward()
      let coeffHandle = coeff.saveForBackward()
      let biasHandle = bias.saveForBackward()
      return Tensor(dataTask: newData, shape: outputShape, dtype: dtype) { grad in
        handle.backward(backend) { (grad * coeff.noGrad()).reduceBroadcast(tStrides, as: self) }
        coeffHandle.backward(backend) {
          (grad * (self.noGrad() + bias.noGrad())).reduceBroadcast(coeffStrides, as: coeff)
        }
        biasHandle.backward(backend) {
          (grad * coeff.noGrad()).reduceBroadcast(biasStrides, as: bias)
        }
      }
    } else {
      return Tensor(dataTask: newData, shape: outputShape, dtype: dtype)
    }
  }

  @recordCaller
  private func _normalize<T: NumericTensorElement>(axis: Int, eps: T) -> Tensor {
    #alwaysAssert(dtype.isFloat, "cannot apply normalize() to dtype \(dtype)")

    let backend = Backend.current
    let newData = createDataTask { t in
      try await backend.normalize(
        input: try await t.data,
        dims: t.reduceDims(axis),
        eps: eps,
        dtype: t.dtype
      )
    }
    if needsGrad && Tensor.isGradEnabled {
      let handle = self.saveForBackward()
      return Tensor(dataTask: newData, shape: shape, dtype: dtype) { grad in
        handle.backward(backend) {
          self.noGrad().normalizeGrad(axis: axis, outGrad: grad, eps: eps)
        }
      }
    } else {
      return Tensor(dataTask: newData, shape: shape, dtype: dtype)
    }
  }

  @recordCaller
  private func _normalizeGrad<T: NumericTensorElement>(axis: Int, outGrad: Tensor, eps: T)
    -> Tensor
  {
    #alwaysAssert(dtype.isFloat, "cannot apply normalizeGrad() to dtype \(dtype)")
    #alwaysAssert(dtype == outGrad.dtype, "gradient dtype \(outGrad.dtype) does not match \(dtype)")
    #alwaysAssert(shape == outGrad.shape, "gradient shape \(outGrad.shape) does not match \(shape)")
    #alwaysAssert(
      !Tensor.isGradEnabled || (!needsGrad && !outGrad.needsGrad),
      "gradients of normalizeGrad() are not supported")

    let backend = Backend.current
    let newData = Tensor.createDataTask(self, outGrad) { t, outGrad in
      try await backend.normalizeGrad(
        input: try await t.data,
        outGrad: try await outGrad.data,
        dims: t.reduceDims(axis),
        eps: eps,
        dtype: t.dtype
      )
    }
    return Tensor(dataTask: newData, shape: shape, dtype: dtype)
  }
}
