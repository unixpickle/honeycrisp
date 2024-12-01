import HCBacktrace

extension Tensor {
  @recordCaller
  private func _mul(_ coeff: Tensor, thenAdd bias: Tensor) -> Tensor {
    alwaysAssert(dtype == coeff.dtype, "dtype \(dtype) does not match coefficients \(coeff.dtype)")
    alwaysAssert(dtype == bias.dtype, "dtype \(dtype) does not match bias \(bias.dtype)")

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
        count: outputShape.product(), dtype: t.dtype)
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
    alwaysAssert(dtype == coeff.dtype, "dtype \(dtype) does not match coefficients \(coeff.dtype)")
    alwaysAssert(dtype == bias.dtype, "dtype \(dtype) does not match bias \(bias.dtype)")

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
        count: outputShape.product(), dtype: t.dtype)
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
  private func _normalize<T: TensorElement>(mean: Tensor, variance: Tensor, epsilon: T) -> Tensor {
    alwaysAssert(dtype.isFloat, "cannot apply normalize() to dtype \(dtype)")
    alwaysAssert(dtype == mean.dtype, "dtype \(dtype) does not match mean \(mean.dtype)")
    alwaysAssert(
      dtype == variance.dtype, "dtype \(dtype) does not match variance \(variance.dtype)")

    let (outputShape, allStrides) = Tensor.lazyBroadcast([self, mean, variance])
    let tStrides = allStrides[0]
    let meanStrides = allStrides[1]
    let varianceStrides = allStrides[2]

    let backend = Backend.current
    let newData = Tensor.createDataTask(self, mean, variance) { t, mean, variance in
      try await backend.normalize(
        input: BroadcastData(strides: tStrides, data: try await t.data),
        mean: BroadcastData(strides: meanStrides, data: try await mean.data),
        variance: BroadcastData(strides: varianceStrides, data: try await variance.data),
        epsilon: epsilon,
        count: outputShape.product(),
        dtype: t.dtype)
    }
    if (needsGrad || mean.needsGrad || variance.needsGrad) && Tensor.isGradEnabled {
      let handle = self.saveForBackward()
      let meanHandle = mean.saveForBackward()
      let varianceHandle = variance.saveForBackward()
      return Tensor(dataTask: newData, shape: outputShape, dtype: dtype) { grad in
        handle.backward(backend) {
          Tensor.normalizeXGrad(
            variance: variance.noGrad(), outGrad: grad, epsilon: epsilon, sign: 1.0
          )
          .reduceBroadcast(tStrides, as: self)
        }
        meanHandle.backward(backend) {
          Tensor.normalizeXGrad(
            variance: variance.noGrad(), outGrad: grad, epsilon: epsilon, sign: -1.0
          )
          .reduceBroadcast(meanStrides, as: mean)
        }
        varianceHandle.backward(backend) {
          Tensor.normalizeVarianceGrad(
            input: self.noGrad(), mean: mean.noGrad(), variance: variance.noGrad(), outGrad: grad,
            epsilon: epsilon
          ).reduceBroadcast(varianceStrides, as: variance)
        }
      }
    } else {
      return Tensor(dataTask: newData, shape: outputShape, dtype: dtype)
    }
  }

  @recordCaller
  private static func _normalizeXGrad<T: TensorElement>(
    variance: Tensor, outGrad: Tensor, epsilon: T, sign: Float
  ) -> Tensor {
    alwaysAssert(!variance.needsGrad && !outGrad.needsGrad)
    alwaysAssert(variance.dtype.isFloat, "cannot apply normalizeXGrad() to dtype \(variance.dtype)")
    alwaysAssert(
      variance.dtype == outGrad.dtype, "dtype \(variance.dtype) does not match \(outGrad.dtype)")

    let (outputShape, (varianceStrides, outGradStrides)) = Tensor.lazyBroadcast(variance, outGrad)

    let backend = Backend.current
    let newData = Tensor.createDataTask(variance, outGrad) { variance, outGrad in
      try await backend.normalizeXGrad(
        variance: BroadcastData(strides: varianceStrides, data: try await variance.data),
        outGrad: BroadcastData(strides: outGradStrides, data: try await outGrad.data),
        epsilon: epsilon,
        sign: sign,
        count: outputShape.product(),
        dtype: variance.dtype)
    }
    return Tensor(dataTask: newData, shape: outputShape, dtype: variance.dtype)
  }

  @recordCaller
  private static func _normalizeVarianceGrad<T: TensorElement>(
    input: Tensor, mean: Tensor, variance: Tensor, outGrad: Tensor, epsilon: T
  ) -> Tensor {
    alwaysAssert(!input.needsGrad && !mean.needsGrad && !variance.needsGrad && !outGrad.needsGrad)
    alwaysAssert(input.dtype.isFloat, "cannot apply normalizeXGrad() to dtype \(variance.dtype)")
    alwaysAssert(input.dtype == mean.dtype)
    alwaysAssert(input.dtype == variance.dtype)
    alwaysAssert(input.dtype == outGrad.dtype)

    let (outputShape, allStrides) = Tensor.lazyBroadcast([input, mean, variance, outGrad])
    let inputStrides = allStrides[0]
    let meanStrides = allStrides[1]
    let varianceStrides = allStrides[2]
    let outGradStrides = allStrides[3]

    let backend = Backend.current
    let newData = Tensor.createDataTask(input, mean, variance, outGrad) {
      input, mean, variance, outGrad in
      try await backend.normalizeVarianceGrad(
        input: BroadcastData(strides: inputStrides, data: try await input.data),
        mean: BroadcastData(strides: meanStrides, data: try await mean.data),
        variance: BroadcastData(strides: varianceStrides, data: try await variance.data),
        outGrad: BroadcastData(strides: outGradStrides, data: try await outGrad.data),
        epsilon: epsilon,
        count: outputShape.product(),
        dtype: input.dtype)
    }
    return Tensor(dataTask: newData, shape: outputShape, dtype: variance.dtype)
  }
}
