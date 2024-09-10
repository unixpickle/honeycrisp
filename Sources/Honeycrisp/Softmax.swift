extension Tensor {

  public func logSoftmax(axis: Int = -1) -> Tensor {
    let maxes = max(axis: axis, keepdims: true).expand(as: self)
    let safeValues = self - maxes
    let logSumExp = safeValues.exp().sum(axis: axis, keepdims: true).log()
    return self - logSumExp.expand(as: self) - maxes
  }

  public func softmax(axis: Int = -1) -> Tensor {
    logSoftmax().exp()
  }

}
