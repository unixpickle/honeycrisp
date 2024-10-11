extension Tensor {

  public func expand(as asTensor: Tensor) -> Tensor {
    expand(shape: asTensor.shape)
  }

  public func expand(shape newShape: [Int]) -> Tensor {
    alwaysAssert(
      newShape.count >= shape.count,
      "cannot broadcast shape \(shape) to shorter shape \(newShape)"
    )
    var result = self
    for i in 0..<shape.count {
      let axis = shape.count - (i + 1)
      let oldValue = shape[axis]
      let newValue = newShape[newShape.count - (i + 1)]
      if newValue != oldValue {
        alwaysAssert(
          oldValue == 1,
          "axis \(axis) cannot expand from size \(oldValue) to \(newValue): old shape \(shape), new shape \(newShape)"
        )
        result = result.repeating(axis: axis, count: newValue)
      }
    }
    if newShape.count > shape.count {
      if result.shape.isEmpty {
        result = result.reshape([1])
      }
      let repeats = newShape[..<(newShape.count - shape.count)].product()
      result = result.repeating(axis: 0, count: repeats)
    }
    return result.reshape(newShape)
  }

}
