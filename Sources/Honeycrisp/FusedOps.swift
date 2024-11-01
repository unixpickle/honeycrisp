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

  public func normalize<T: TensorElement>(mean: Tensor, variance: Tensor, epsilon: T) -> Tensor {
    alwaysAssert(dtype.isFloat, "cannot apply normalize() to dtype \(dtype)")
    alwaysAssert(dtype == mean.dtype, "dtype \(dtype) does not match mean \(mean.dtype)")
    alwaysAssert(
      dtype == variance.dtype, "dtype \(dtype) does not match variance \(variance.dtype)")

    let broadcasted = Tensor.lazyBroadcast([self, mean, variance])
    let outputShape = broadcasted.0
    let (t, tStrides) = broadcasted.1[0]
    let (mean, meanStrides) = broadcasted.1[1]
    let (variance, varianceStrides) = broadcasted.1[2]

    let backend = Backend.current
    let newData = Tensor.createDataTask(t, mean, variance) { t, mean, variance in
      try await backend.normalize(
        input: BroadcastData(strides: tStrides, data: try await t.data),
        mean: BroadcastData(strides: meanStrides, data: try await mean.data),
        variance: BroadcastData(strides: varianceStrides, data: try await variance.data),
        epsilon: epsilon,
        count: outputShape.product(),
        dtype: t.dtype)
    }
    if (needsGrad || mean.needsGrad || variance.needsGrad) && Tensor.isGradEnabled {
      let handle = t.saveForBackward()
      let meanHandle = mean.saveForBackward()
      let varianceHandle = variance.saveForBackward()
      return Tensor(dataTask: newData, shape: outputShape, dtype: dtype) { grad in
        handle.backward(backend) {
          normalizeXGrad(variance: variance.noGrad(), outGrad: grad, epsilon: epsilon, sign: 1.0)
            .reduceBroadcast(tStrides, as: t)
        }
        meanHandle.backward(backend) {
          normalizeXGrad(variance: variance.noGrad(), outGrad: grad, epsilon: epsilon, sign: -1.0)
            .reduceBroadcast(meanStrides, as: mean)
        }
        varianceHandle.backward(backend) {
          normalizeVarianceGrad(
            input: t.noGrad(), mean: mean.noGrad(), variance: variance.noGrad(), outGrad: grad,
            epsilon: epsilon
          ).reduceBroadcast(varianceStrides, as: variance)
        }
      }
    } else {
      return Tensor(dataTask: newData, shape: outputShape, dtype: dtype)
    }
  }
}

private func normalizeXGrad<T: TensorElement>(
  variance: Tensor, outGrad: Tensor, epsilon: T, sign: Float
) -> Tensor {
  alwaysAssert(!variance.needsGrad && !outGrad.needsGrad)
  alwaysAssert(variance.dtype.isFloat, "cannot apply normalizeXGrad() to dtype \(variance.dtype)")
  alwaysAssert(
    variance.dtype == outGrad.dtype, "dtype \(variance.dtype) does not match \(outGrad.dtype)")

  let (outputShape, ((variance, varianceStrides), (outGrad, outGradStrides))) =
    Tensor.lazyBroadcast(variance, outGrad)

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

private func normalizeVarianceGrad<T: TensorElement>(
  input: Tensor, mean: Tensor, variance: Tensor, outGrad: Tensor, epsilon: T
) -> Tensor {
  alwaysAssert(!input.needsGrad && !mean.needsGrad && !variance.needsGrad && !outGrad.needsGrad)
  alwaysAssert(input.dtype.isFloat, "cannot apply normalizeXGrad() to dtype \(variance.dtype)")
  alwaysAssert(input.dtype == mean.dtype)
  alwaysAssert(input.dtype == variance.dtype)
  alwaysAssert(input.dtype == outGrad.dtype)

  let broadcasted = Tensor.lazyBroadcast([input, mean, variance, outGrad])
  let outputShape = broadcasted.0
  let (input, inputStrides) = broadcasted.1[0]
  let (mean, meanStrides) = broadcasted.1[1]
  let (variance, varianceStrides) = broadcasted.1[2]
  let (outGrad, outGradStrides) = broadcasted.1[3]

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
