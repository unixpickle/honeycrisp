import Foundation
import HCBacktrace

public struct ReduceDims: Hashable {
  let outerCount: Int
  let reduceCount: Int
  let innerCount: Int

  var inCount: Int {
    outerCount * reduceCount * innerCount
  }

  var outCount: Int {
    outerCount * innerCount
  }

  var shape: [Int] {
    [outerCount, reduceCount, innerCount]
  }

  public func inverse() -> RepeatDims {
    RepeatDims(outerCount: outerCount, repeatCount: reduceCount, innerCount: innerCount)
  }
}

public struct RepeatDims: Hashable {
  public let outerCount: Int
  public let repeatCount: Int
  public let innerCount: Int

  var inCount: Int {
    return outerCount * innerCount
  }

  var outCount: Int {
    return outerCount * repeatCount * innerCount
  }

  public func inverse() -> ReduceDims {
    ReduceDims(outerCount: outerCount, reduceCount: repeatCount, innerCount: innerCount)
  }
}

public enum ReduceOp {
  case sum
  case argmax
  case argmin
}

extension Tensor {
  @recordCaller
  private func _sum(axis: Int? = nil, keepdims: Bool = false) -> Tensor {
    reduce(op: .sum, axis: axis, keepdims: keepdims)
  }

  @recordCaller
  private func _argmax(axis: Int? = nil, keepdims: Bool = false) -> Tensor {
    reduce(op: .argmax, axis: axis, keepdims: keepdims)
  }

  @recordCaller
  private func _argmin(axis: Int? = nil, keepdims: Bool = false) -> Tensor {
    reduce(op: .argmin, axis: axis, keepdims: keepdims)
  }

  @recordCaller
  private func _min(axis: Int? = nil, keepdims: Bool = false) -> Tensor {
    gatherFromReduce(op: .argmin, axis: axis, keepdims: keepdims)
  }

  @recordCaller
  private func _max(axis: Int? = nil, keepdims: Bool = false) -> Tensor {
    gatherFromReduce(op: .argmax, axis: axis, keepdims: keepdims)
  }

  @recordCaller
  private func _mean(axis: Int? = nil, keepdims: Bool = false) -> Tensor {
    let axis = positiveAxis(axis)
    return sum(axis: axis, keepdims: keepdims) / Float(axis != nil ? shape[axis!] : shape.product())
  }

  @recordCaller
  private func _variance(axis: Int? = nil, keepdims: Bool = false) -> Tensor {
    let mean = self.mean(axis: axis, keepdims: true)
    let result = (self - mean.expand(as: self)).pow(2).mean(axis: axis, keepdims: true)
    if keepdims {
      return result
    } else if let axis = axis {
      return result.reshape(Array(result.shape[..<axis]) + Array(result.shape[(axis + 1)...]))
    } else {
      return result.reshape([])
    }
  }

  @recordCaller
  private func _meanAndVariance(axis: Int? = nil, keepdims: Bool = false) -> (Tensor, Tensor) {
    let moment1 = self.mean(axis: axis, keepdims: true)
    let moment2 = self.pow(2).mean(axis: axis, keepdims: true)
    return (moment1, (moment2 - moment1.pow(2)).clamp(min: 0))
  }

  @recordCaller
  private func _maxPool2D(width kw: Int, height kh: Int, channelsLast: Bool = false) -> Tensor {
    alwaysAssert(shape.count == 4, "invalid shape for maxPool2D \(shape)")
    let b = shape[0]
    let (h, w, c) = channelsLast ? (shape[1], shape[2], shape[3]) : (shape[2], shape[3], shape[1])
    alwaysAssert(h % kh == 0, "pool height \(kh) does not divide image height \(h)")
    alwaysAssert(w % kw == 0, "pool width \(kw) does not divide image width \(w)")
    if channelsLast {
      return reshape([b, h / kh, kh, w / kw, kw, c])[FullRange(dims: 2), PermuteAxes(1, 0)]
        .reshape([b, h / kh, kh * kw, w / kw, c]).max(axis: 2)
    } else {
      return reshape([b, c, h / kh, kh, w / kw, kw])[FullRange(dims: 3), PermuteAxes(1, 0)]
        .reshape([b, c, h / kh, kh * kw, w / kw]).max(axis: 3)
    }
  }

  @recordCaller
  internal func _gatherFromReduce(op: ReduceOp, axis: Int?, keepdims: Bool) -> Tensor {
    guard let axis = positiveAxis(axis) else {
      let result = flatten().gatherFromReduce(op: op, axis: 0, keepdims: false)
      return keepdims ? result.reshape(Array(repeating: 1, count: shape.count)) : result
    }
    let indices = reduce(op: op, axis: axis, keepdims: true)
    let selection = gather(axis: axis, indices: indices)
    return keepdims ? selection : selection.squeeze(axis: axis)
  }

  @recordCaller
  internal func _reduce(
    op: ReduceOp, axis: Int? = nil, keepdims: Bool = false
  ) -> Tensor {
    guard let axis = positiveAxis(axis) else {
      let outShape = keepdims ? Array(repeating: 1, count: shape.count) : []
      return reshape([shape.product()]).reduce(op: op, axis: 0).reshape(outShape)
    }
    alwaysAssert(axis >= 0 && axis < shape.count, "axis \(axis) out of bounds for shape \(shape)")
    let backend = Backend.current
    let newData = createDataTask { t in
      try await backend.reduce(try await t.data, op: op, dims: t.reduceDims(axis), dtype: t.dtype)
    }
    let newShape = Array(shape[..<axis]) + (keepdims ? [1] : []) + Array(shape[(axis + 1)...])
    if !Tensor.isGradEnabled || !needsGrad || (op != .sum) {
      return Tensor(
        dataTask: newData, shape: newShape, dtype: op == .argmin || op == .argmax ? .int64 : dtype)
    } else {
      let handle = self.saveForBackward()
      return Tensor(dataTask: newData, shape: newShape, dtype: dtype) { grad in
        handle.backward(backend) {
          grad.repeating(axis: axis, count: self.shape[axis]).reshape(self.shape)
        }
      }
    }
  }

  @recordCaller
  private func _repeating(axis: Int, count: Int) -> Tensor {
    let axis = positiveAxis(axis)
    alwaysAssert(axis >= 0 && axis <= shape.count, "axis \(axis) out of bounds for shape \(shape)")
    let outerCount = shape[..<axis].product()
    let innerCount = shape[axis...].product()
    let newShape =
      if axis == shape.count {
        shape + [count]
      } else {
        Array(shape[..<axis]) + [shape[axis] * count] + Array(shape[(axis + 1)...])
      }
    return flatApplyRepeats([
      RepeatDims(outerCount: outerCount, repeatCount: count, innerCount: innerCount)
    ]).reshape(newShape)
  }

  @recordCaller
  internal func _reduceDims(_ axis: Int? = nil) -> ReduceDims {
    if let axis = axis {
      let axis = (axis < 0 ? axis + shape.count : axis)
      alwaysAssert(axis >= 0 && axis < shape.count, "axis \(axis) out of bounds for shape \(shape)")
      return ReduceDims(
        outerCount: shape[..<axis].product(),
        reduceCount: shape[axis],
        innerCount: shape[(axis + 1)...].product()
      )
    } else {
      return ReduceDims(outerCount: 1, reduceCount: shape.product(), innerCount: 1)
    }
  }

  @recordCaller
  internal func _flatApplySums(_ allDims: [ReduceDims]) -> Tensor {
    let backend = Backend.current
    let newData = createDataTask { t in
      var count = t.shape.product()
      var result = try await t.data
      for dims in allDims {
        assert(dims.inCount == count)
        result = try await backend.reduce(result, op: .sum, dims: dims, dtype: t.dtype)
        count = dims.outCount
      }
      return result
    }
    let newShape = [allDims.last?.outCount ?? shape.product()]
    if !Tensor.isGradEnabled || !needsGrad {
      return Tensor(dataTask: newData, shape: newShape, dtype: dtype)
    } else {
      let repeats = allDims.reversed().map { $0.inverse() }
      let handle = self.saveForBackward()
      return Tensor(dataTask: newData, shape: newShape, dtype: dtype) { grad in
        handle.backward(backend) { grad.flatApplyRepeats(repeats).reshape(self.shape) }
      }
    }
  }

  @recordCaller
  internal func _flatApplyRepeats(_ allDims: [RepeatDims]) -> Tensor {
    let backend = Backend.current
    let newData = createDataTask { t in
      var count = t.shape.product()
      var result = try await t.data
      for dims in allDims {
        assert(dims.inCount == count)
        result = try await backend.repeated(result, dims: dims, dtype: t.dtype)
        count = dims.outCount
      }
      return result
    }
    let newShape = [allDims.last?.outCount ?? shape.product()]
    if !Tensor.isGradEnabled || !needsGrad {
      return Tensor(dataTask: newData, shape: newShape, dtype: dtype)
    } else {
      let sums = allDims.reversed().map { $0.inverse() }
      let handle = self.saveForBackward()
      return Tensor(dataTask: newData, shape: newShape, dtype: dtype) { grad in
        handle.backward(backend) { grad.flatApplySums(sums).reshape(self.shape) }
      }
    }
  }
}
