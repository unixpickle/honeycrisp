import HCBacktrace
import Metal

/// A mapping from a broadcasted array to the underlying data array.
///
/// We can imagine we have some data with `dataCount` elements, but broadcast
/// it to a virtual array of `dataCount * outerRepeats * innerRepeats`
/// elements, where each element of the data is repeated sequentially
/// `innerRepeats` times, and the entire result is repeated `outerRepeats`
/// times.
///
/// An instance of this structure can be called as a function, and will map an
/// index in the virtual array to an index in the data array.
public struct BroadcastStrides {
  public let dataCount: Int
  public let outerRepeats: Int
  public let innerRepeats: Int

  public var isNoOp: Bool {
    outerRepeats == 1 && innerRepeats == 1
  }

  /// Get the `RepeatDims` which could be applied to the raw array to get the
  /// virtual array.
  public func repeats() -> [RepeatDims] {
    var result = [RepeatDims]()
    if innerRepeats > 1 {
      result.append(RepeatDims(outerCount: dataCount, repeatCount: innerRepeats, innerCount: 1))
    }
    if outerRepeats > 1 {
      result.append(
        RepeatDims(outerCount: 1, repeatCount: outerRepeats, innerCount: dataCount * innerRepeats))
    }
    return result
  }

  /// Get the `ReduceDims` that we could apply to the virtual gradient array to
  /// get a gradient for the raw data array.
  public func repeatsInverse() -> [ReduceDims] {
    return repeats().reversed().map { $0.inverse() }
  }

  /// May an index in the virtual array to the data array.
  public func callAsFunction(_ i: Int) -> Int {
    assert(i >= 0 && i < dataCount * outerRepeats * innerRepeats)
    return (i / innerRepeats) % dataCount
  }
}

/// A structure annotating an actual instance of raw data with a
/// `BroadcastStrides` indicating how to interpret it as a virtual array.
///
/// Note that this structure does not include the dtype of the underlying
/// data, which is crucial for actually performing operations on it.
public struct BroadcastData {
  public let strides: BroadcastStrides
  public let data: Tensor.Data

  /// Get the size of the data array, in elements (not bytes).
  public var dataCount: Int { strides.dataCount }

  /// If true, the data is actually not broadcasted and the raw array is
  /// equal to the virtual array.
  public var isSimple: Bool { strides.isNoOp }

  /// Wrap an unbroadcasted tensor in a ``BroadcastData``.
  public static func simple(data: Tensor.Data, count: Int) -> BroadcastData {
    BroadcastData(
      strides: BroadcastStrides(dataCount: count, outerRepeats: 1, innerRepeats: 1), data: data)
  }
}

/// An ordered sequence of `RepeatDims` which, when applied, expands the
/// dimensions of one tensor to another shape.
///
/// Each step includes the (axes) which it affects in the source tensor,
/// and this may be empty if new (outer) axes are being created.
private class RepeatSequence {
  public struct Step {
    let dims: RepeatDims
    let axes: [Int]
  }

  private var _steps: [Step] = []
  public let inCount: Int
  public var steps: [Step] { _steps }

  public var outCount: Int {
    steps.count == 0 ? inCount : steps[steps.count - 1].dims.outCount
  }

  public init(inCount: Int) {
    self.inCount = inCount
  }

  private init(inCount: Int, steps: [Step]) {
    self.inCount = inCount
    _steps = steps
  }

  public func add(_ d: RepeatDims, axes: [Int]) {
    if _steps.isEmpty {
      assert(d.innerCount * d.outerCount == inCount)
      _steps = [Step(dims: d, axes: axes)]
    } else {
      let last = _steps[_steps.count - 1]
      assert(
        d.innerCount >= last.dims.innerCount * last.dims.repeatCount,
        "repeats must be strided along increasing dimensions")
      assert(d.outerCount * d.innerCount == last.dims.outCount)
      if last.dims.innerCount * last.dims.repeatCount == d.innerCount {
        // We can modify the existing step
        _steps[_steps.count - 1] = Step(
          dims: RepeatDims(
            outerCount: d.outerCount, repeatCount: last.dims.repeatCount * d.repeatCount,
            innerCount: last.dims.innerCount
          ),
          axes: last.axes + axes)
      } else {
        _steps.append(Step(dims: d, axes: axes))
      }
    }
  }

  /// Compress the repeat sequence such that we only repeat dimensions that are
  /// not supported by broadcasting.
  /// We then return a combination of a `RepeatSequence` which must be applied
  /// to the original data, and then a `BroadcastStrides` to be used with the
  /// result to get the full expansion.
  public func lazify() -> (RepeatSequence, BroadcastStrides) {
    var newSteps = steps
    var result = BroadcastStrides(dataCount: outCount, outerRepeats: 1, innerRepeats: 1)
    if let first = newSteps.first, first.dims.innerCount == 1 {
      newSteps.removeFirst()
      result = BroadcastStrides(
        dataCount: result.dataCount / first.dims.repeatCount, outerRepeats: 1,
        innerRepeats: first.dims.repeatCount)

      // We must modify all later repetitions to reflect the new
      // smaller inner dimension.
      for (i, x) in newSteps.enumerated() {
        newSteps[i] = Step(
          dims: RepeatDims(
            outerCount: x.dims.outerCount,
            repeatCount: x.dims.repeatCount,
            innerCount: x.dims.innerCount / first.dims.repeatCount
          ),
          axes: x.axes
        )
      }
    }
    if let last = newSteps.last, last.dims.outerCount == 1 {
      newSteps.removeLast()
      result = BroadcastStrides(
        dataCount: result.dataCount / last.dims.repeatCount,
        outerRepeats: last.dims.repeatCount,
        innerRepeats: result.innerRepeats
      )
    }
    return (RepeatSequence(inCount: inCount, steps: newSteps), result)
  }

  public func inverse() -> [ReduceDims] {
    var result = [ReduceDims]()
    for step in steps.reversed() {
      result.append(
        ReduceDims(
          outerCount: step.dims.outerCount,
          reduceCount: step.dims.repeatCount,
          innerCount: step.dims.innerCount
        )
      )
    }
    return result
  }
}

extension Tensor {

  @recordCaller
  private func _expand(as asTensor: Tensor) -> Tensor {
    expand(shape: asTensor.shape)
  }

  @recordCaller
  private func _expand(shape newShape: [Int]) -> Tensor {
    let plan = planExpansion(shape: newShape)
    return flatApplyRepeats(plan.steps.map { $0.dims }).reshape(newShape)
  }

  @recordCaller
  private static func _broadcast(_ xs: [Tensor]) -> [Tensor] {
    let shape = Tensor.broadcastShape(xs.map { $0.shape })
    return xs.map { $0.expand(shape: shape) }
  }

  @recordCaller
  private static func _broadcast(_ x: Tensor, _ y: Tensor) -> (Tensor, Tensor) {
    let results = broadcast([x, y])
    return (results[0], results[1])
  }

  @recordCaller
  internal static func _lazyBroadcast(_ t: [Tensor]) -> ([Int], [(Tensor, BroadcastStrides)]) {
    let newShape = broadcastShape(t.map { $0.shape })
    let results = t.map { $0.lazyExpand(shape: newShape) }
    return (newShape, results)
  }

  @recordCaller
  internal static func _lazyBroadcast(_ t1: Tensor, _ t2: Tensor) -> (
    [Int], ((Tensor, BroadcastStrides), (Tensor, BroadcastStrides))
  ) {
    let newShape = broadcastShape([t1.shape, t2.shape])
    let r1 = t1.lazyExpand(shape: newShape)
    let r2 = t2.lazyExpand(shape: newShape)
    return (newShape, (r1, r2))
  }

  @recordCaller
  internal func _lazyExpand(shape newShape: [Int]) -> (Tensor, BroadcastStrides) {
    alwaysAssert(
      newShape.count >= shape.count,
      "cannot broadcast shape \(shape) to shorter shape \(newShape)"
    )

    let plan = planExpansion(shape: newShape)
    let (lazyPlan, bcast) = plan.lazify()
    let expanded = flatApplyRepeats(lazyPlan.steps.map { $0.dims })
    var lazyShape = shape
    for axis in lazyPlan.steps.flatMap({ $0.axes }) {
      lazyShape[axis] = newShape[axis + (newShape.count - shape.count)]
    }
    assert(lazyShape.product() == expanded.shape.product())
    return (expanded.reshape(lazyShape), bcast)
  }

  @recordCaller
  internal func _reduceBroadcast(_ bcast: BroadcastStrides, as tensor: Tensor) -> Tensor {
    return flatApplySums(bcast.repeatsInverse()).reshape(tensor.shape)
  }

  private func planExpansion(shape newShape: [Int]) -> RepeatSequence {
    alwaysAssert(
      newShape.count >= shape.count,
      "cannot broadcast shape \(shape) to shorter shape \(newShape)"
    )

    let result = RepeatSequence(inCount: shape.product())
    var innerCount = 1
    var outerCount = shape.product()
    for i in 0..<shape.count {
      let axis = shape.count - (i + 1)
      let oldValue = shape[axis]
      let newValue = newShape[newShape.count - (i + 1)]
      if newValue != oldValue {
        alwaysAssert(
          oldValue == 1,
          "axis \(axis) cannot expand from size \(oldValue) to \(newValue): old shape \(shape), new shape \(newShape)"
        )
        result.add(
          RepeatDims(outerCount: outerCount, repeatCount: newValue, innerCount: innerCount),
          axes: [axis])
        innerCount *= newValue
      } else {
        innerCount *= oldValue
        outerCount /= oldValue
      }
    }
    if newShape.count > shape.count {
      let repeats = newShape[..<(newShape.count - shape.count)].product()
      result.add(RepeatDims(outerCount: 1, repeatCount: repeats, innerCount: innerCount), axes: [])
    }
    return result
  }

  @recordCaller
  internal static func _broadcastShape(_ s: [[Int]]) -> [Int] {
    if s.count == 1 {
      return s[0]
    } else if s.count == 2 {
      var shape0 = s[0]
      var shape1 = s[1]
      while shape0.count < shape1.count {
        shape0.insert(1, at: 0)
      }
      while shape1.count < shape0.count {
        shape1.insert(1, at: 0)
      }
      return zip(shape0, shape1).map { x, y in
        if x == y {
          return x
        } else if x == 0 || y == 0 {
          tracedFatalError("shapes \(s[0]) and \(s[1]) do not support broadcasting")
        } else if x == 1 {
          return y
        } else if y == 1 {
          return x
        } else {
          tracedFatalError("shapes \(s[0]) and \(s[1]) do not support broadcasting")
        }
      }
    } else {
      return broadcastShape([broadcastShape([s[0], s[1]])] + Array(s[2...]))
    }
  }

}
