extension Tensor {

  public convenience init(concat tensors: [Tensor], axis: Int = 0) {
    assert(!tensors.isEmpty, "cannot concatenate zero tensors")
    let firstShape = tensors[0].shape
    let trueAxis = axis < 0 ? firstShape.count + axis : axis

    assert(
      trueAxis >= 0 && trueAxis < firstShape.count,
      "axis \(axis) out of bounds for shape \(firstShape)"
    )
    for x in tensors {
      assert(
        x.shape.count == firstShape.count,
        "all tensor arguments to concat must have same number of dimensions; got \(firstShape) and \(x.shape)"
      )
      for (i, (shape1, shape2)) in zip(firstShape, x.shape).enumerated() {
        assert(
          i == axis || shape1 == shape2,
          "tensors must match at non-concatenation dimension \(i) but got \(shape1) != \(shape2)"
        )
      }
    }

    let outerDim = firstShape[..<trueAxis].product()
    var newData = [Float]()
    var newShape = firstShape
    newShape[trueAxis] = tensors.map({ $0.shape[trueAxis] }).sum()
    for i in 0..<outerDim {
      for tensor in tensors {
        let innerShape = tensor.shape[trueAxis...].product()
        newData += tensor.data[(i * innerShape)..<((i + 1) * innerShape)]
      }
    }
    if !tensors.map({ x in x.needsGrad }).reduce(false, { x, y in x || y }) {
      self.init(data: newData, shape: newShape)
    } else {
      let handles = Array(tensors.map({ $0.saveForBackward() }))
      self.init(data: newData, shape: newShape) { grad in
        var offset = 0
        for (tensor, handle) in zip(tensors, handles) {
          let subSize = tensor.shape[trueAxis]
          handle.backward(
            grad: grad[FullRange(dims: trueAxis), offset..<(offset + subSize)]
          )
          offset += subSize
        }
        assert(offset == grad.shape[trueAxis], "gradient size was inconsistent")
      }
    }
  }

}
