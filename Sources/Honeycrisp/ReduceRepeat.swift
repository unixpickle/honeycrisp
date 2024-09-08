import Foundation

public struct ReduceDims {
  let outerCount: Int
  let reduceCount: Int
  let innerCount: Int

  var inCount: Int {
    outerCount * reduceCount * innerCount
  }

  var outCount: Int {
    outerCount * innerCount
  }
}

public enum ReduceOp {
  case sum
  case argmax
  case argmin
}

extension Tensor {
  public func sum(axis: Int? = nil, keepdims: Bool = false) -> Tensor {
    reduce(op: .sum, axis: axis, keepdims: keepdims)
  }

  public func argmax(axis: Int? = nil, keepdims: Bool = false) -> Tensor {
    reduce(op: .argmax, axis: axis, keepdims: keepdims)
  }

  public func argmin(axis: Int? = nil, keepdims: Bool = false) -> Tensor {
    reduce(op: .argmin, axis: axis, keepdims: keepdims)
  }

  public func min(axis: Int? = nil, keepdims: Bool = false) -> Tensor {
    gatherFromReduce(op: .argmin, axis: axis, keepdims: keepdims)
  }

  public func max(axis: Int? = nil, keepdims: Bool = false) -> Tensor {
    gatherFromReduce(op: .argmax, axis: axis, keepdims: keepdims)
  }

  internal func gatherFromReduce(op: ReduceOp, axis: Int?, keepdims: Bool) -> Tensor {
    guard let axis = positiveAxis(axis) else {
      let result = flatten().gatherFromReduce(op: op, axis: 0, keepdims: false)
      return keepdims ? result.reshape(Array(repeating: 1, count: shape.count)) : result
    }
    let indices = reduce(op: op, axis: axis, keepdims: true)
    let selection = gather(axis: axis, indices: indices)
    return keepdims ? selection : selection.squeeze(axis: axis)
  }

  internal func reduce(
    op: ReduceOp, axis: Int? = nil, keepdims: Bool = false
  ) -> Tensor {
    guard let axis = positiveAxis(axis) else {
      let outShape = keepdims ? Array(repeating: 1, count: shape.count) : []
      return reshape([shape.product()]).reduce(op: op, axis: 0).reshape(outShape)
    }
    assert(axis >= 0 && axis < shape.count, "axis \(axis) out of bounds for shape \(shape)")
    let backend = Backend.current
    let newData = Task {
      try await backend.reduce(try await self.data, op: op, dims: reduceDims(axis), dtype: dtype)
    }
    let newShape = Array(shape[..<axis]) + (keepdims ? [1] : []) + Array(shape[(axis + 1)...])
    if !needsGrad || (op == .argmin || op == .argmax) {
      return Tensor(
        dataTask: newData, shape: newShape, dtype: op == .argmin || op == .argmax ? .int64 : dtype)
    } else {
      let handle = self.saveForBackward()
      return Tensor(dataTask: newData, shape: newShape, dtype: dtype) { grad in
        handle.backward(
          backend.use { grad.repeating(axis: axis, count: self.shape[axis]).reshape(self.shape) }
        )
      }
    }
  }

  // Repeat the tensor along a given axis for zero or more times.
  //
  // The axis may be equal to shape.count, in which case a new trailing
  // dimension is added with the value `count`.
  public func repeating(axis: Int, count: Int) -> Tensor {
    let axis = positiveAxis(axis)
    assert(axis >= 0 && axis <= shape.count, "axis \(axis) out of bounds for shape \(shape)")
    let outerCount = shape[..<axis].product()
    let innerCount = shape[axis...].product()
    let backend = Backend.current
    let newData = Task {
      try await backend.repeated(
        try await self.data, outerCount: outerCount, innerCount: innerCount, repeats: count,
        dtype: dtype)
    }
    let newShape =
      if axis == shape.count {
        shape + [count]
      } else {
        Array(shape[..<axis]) + [shape[axis] * count] + Array(shape[(axis + 1)...])
      }
    if !needsGrad {
      return Tensor(
        dataTask: newData, shape: newShape, dtype: dtype)
    } else {
      let handle = self.saveForBackward()
      return Tensor(dataTask: newData, shape: newShape, dtype: dtype) { grad in
        let grad = backend.use { grad.reshape([outerCount, count, innerCount]).sum(axis: 1) }
        handle.backward(grad.reshape(self.shape))
      }
    }
  }

  internal func reduceDims(_ axis: Int? = nil) -> ReduceDims {
    if let axis = axis {
      let axis = (axis < 0 ? axis + shape.count : axis)
      assert(axis >= 0 && axis < shape.count, "axis \(axis) out of bounds for shape \(shape)")
      return ReduceDims(
        outerCount: shape[..<axis].product(),
        reduceCount: shape[axis],
        innerCount: shape[(axis + 1)...].product()
      )
    } else {
      return ReduceDims(outerCount: 1, reduceCount: shape.product(), innerCount: 1)
    }
  }
}
