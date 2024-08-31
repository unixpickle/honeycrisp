extension Tensor {

  public func sum(axis: Int? = nil, keepdims: Bool = false) -> Tensor {
    if let axis = axis {
      let trueAxis = (axis < 0 ? axis + shape.count : axis)
      assert(
        trueAxis >= 0 && trueAxis < shape.count,
        "axis \(axis) out of bounds for tensor shape \(shape)"
      )
      let outerCount = shape[..<trueAxis].product()
      let reduceCount = shape[trueAxis]
      let innerCount = shape[(trueAxis + 1)...].product()
      var newData = Array(repeating: Float(0), count: outerCount * innerCount)
      for i in 0..<outerCount {
        let batchOffset = i * reduceCount * innerCount
        for j in 0..<innerCount {
          var sum = Float(0)
          for k in 0..<reduceCount {
            sum += data[batchOffset + k * innerCount + j]
          }
          newData[i * innerCount + j] = sum
        }
      }
      let keepdimShape = Array(shape[..<trueAxis] + [1] + shape[(trueAxis + 1)...])
      let newShape = Array(shape[..<trueAxis] + (keepdims ? [1] : []) + shape[(trueAxis + 1)...])
      if !needsGrad {
        return Tensor(data: newData, shape: newShape)
      } else {
        let handle = self.saveForBackward()
        return Tensor(data: newData, shape: newShape) { grad in
          handle.backward(
            grad: grad.reshape(keepdimShape).repeating(axis: trueAxis, count: reduceCount)
          )
        }
      }
    } else {
      let newData = [self.data.sum()]
      let newShape = (keepdims ? Array(repeating: 1, count: shape.count) : [])
      if !needsGrad {
        return Tensor(data: newData, shape: newShape)
      } else {
        let handle = self.saveForBackward()
        return Tensor(data: newData, shape: newShape) { grad in
          handle.backward(grad: grad.expand(as: self))
        }
      }
    }
  }

  public func mean(axis: Int? = nil, keepdims: Bool = false) -> Tensor {
    let unscaled = sum(axis: axis, keepdims: keepdims)
    let ratio = Float(unscaled.shape.product()) / Float(shape.product())
    return unscaled * ratio
  }

  public func min(axis: Int? = nil, keepdims: Bool = false) -> Tensor {
    return maxOrMin(isMax: false, axis: axis, keepdims: keepdims)
  }

  public func max(axis: Int? = nil, keepdims: Bool = false) -> Tensor {
    return maxOrMin(isMax: true, axis: axis, keepdims: keepdims)
  }

  private func maxOrMin(isMax: Bool, axis: Int?, keepdims: Bool) -> Tensor {
    if let axis = axis {
      let trueAxis = (axis < 0 ? axis + shape.count : axis)
      assert(
        trueAxis >= 0 && trueAxis < shape.count,
        "axis \(axis) out of bounds for tensor shape \(shape)"
      )
      assert(shape[trueAxis] > 0, "cannot perform this reduction along empty axis")
      let outerCount = shape[..<trueAxis].product()
      let reduceCount = shape[trueAxis]
      let innerCount = shape[(trueAxis + 1)...].product()
      var newData = Array(repeating: Float(0), count: outerCount * innerCount)
      var indices = Array(repeating: 0, count: outerCount * innerCount)
      for i in 0..<outerCount {
        let batchOffset = i * reduceCount * innerCount
        for j in 0..<innerCount {
          var reducedValue = Float(0)
          var chosenIndex = 0
          for k in 0..<reduceCount {
            let v = data[batchOffset + k * innerCount + j]
            if k == 0 || ((isMax && v > reducedValue) || (!isMax && v < reducedValue)) {
              reducedValue = v
              chosenIndex = k
            }
          }
          newData[i * innerCount + j] = reducedValue
          indices[i * innerCount + j] = batchOffset + chosenIndex * innerCount + j
        }
      }
      let newShape = Array(shape[..<trueAxis] + (keepdims ? [1] : []) + shape[(trueAxis + 1)...])
      if !needsGrad {
        return Tensor(data: newData, shape: newShape)
      } else {
        let handle = self.saveForBackward()
        return Tensor(data: newData, shape: newShape) { grad in
          handle.backward(grad: grad.scatter(outShape: self.shape, dstIndices: indices))
        }
      }
    } else {
      let out = self.reshape([shape.product()]).maxOrMin(isMax: isMax, axis: 0, keepdims: false)
      if keepdims {
        return out.reshape(Array(repeating: 1, count: shape.count))
      } else {
        return out
      }
    }
  }

}
