import Foundation
import HCBacktrace

extension Tensor {

  @recordCaller
  private func _cumulativeSum(axis: Int, exclusive: Bool = false, reverse: Bool = false) -> Tensor {
    let axis = positiveAxis(axis)
    #alwaysAssert(axis >= 0 && axis < shape.count, "axis \(axis) out of bounds for shape \(shape)")
    let backend = Backend.current
    let newData = createDataTask { t in
      try await backend.cumulativeSum(
        try await t.data, dims: t.reduceDims(axis), exclusive: exclusive, reverse: reverse,
        dtype: t.dtype)
    }
    if !Tensor.isGradEnabled || !needsGrad {
      return Tensor(dataTask: newData, shape: shape, dtype: dtype)
    } else {
      let handle = self.saveForBackward()
      return Tensor(dataTask: newData, shape: shape, dtype: dtype) { grad in
        handle.backward(backend) {
          grad.cumulativeSum(axis: axis, exclusive: exclusive, reverse: !reverse)
        }
      }
    }
  }

  @recordCaller
  private func _cumulativeProd(axis: Int, exclusive: Bool = false, reverse: Bool = false) -> Tensor
  {
    #alwaysAssert(
      !Tensor.isGradEnabled || !needsGrad,
      "gradients are currently not supported for cumulative product")
    let axis = positiveAxis(axis)
    #alwaysAssert(axis >= 0 && axis < shape.count, "axis \(axis) out of bounds for shape \(shape)")
    let backend = Backend.current
    let newData = createDataTask { t in
      try await backend.cumulativeProd(
        try await t.data, dims: t.reduceDims(axis), exclusive: exclusive, reverse: reverse,
        dtype: t.dtype)
    }
    return Tensor(dataTask: newData, shape: shape, dtype: dtype)
  }

}
