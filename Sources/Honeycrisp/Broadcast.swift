import Metal

public struct BroadcastStrides {
  public let dataCount: Int
  public let outerRepeats: Int
  public let innerRepeats: Int

  public var isNoOp: Bool {
    outerRepeats == 1 && innerRepeats == 1
  }

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

  public func repeatsInverse() -> [ReduceDims] {
    return repeats().reversed().map { $0.inverse() }
  }

  public func callAsFunction(_ i: Int) -> Int {
    assert(i >= 0 && i < dataCount * outerRepeats * innerRepeats)
    return (i / innerRepeats) % dataCount
  }
}

public struct BroadcastData {
  public let strides: BroadcastStrides
  public let data: Tensor.Data

  public var dataCount: Int { strides.dataCount }
  public var isSimple: Bool { strides.isNoOp }
}

private class RepeatSequence {
  private var _steps: [RepeatDims] = []
  private var _axes: [[Int]] = []
  public let inCount: Int
  public var steps: [RepeatDims] { _steps }
  public var axes: [[Int]] { _axes }

  public var outCount: Int {
    steps.count == 0 ? inCount : steps[steps.count - 1].outCount
  }

  public init(inCount: Int) {
    self.inCount = inCount
  }

  private init(inCount: Int, steps: [RepeatDims], axes: [[Int]]) {
    self.inCount = inCount
    _steps = steps
    _axes = axes
  }

  public func add(_ d: RepeatDims, axes: [Int]) {
    if _steps.isEmpty {
      assert(d.innerCount * d.outerCount == inCount)
      _steps = [d]
      _axes = [axes]
    } else {
      let last = _steps[_steps.count - 1]
      assert(
        d.innerCount >= last.innerCount * last.repeatCount,
        "repeats must be strided along increasing dimensions")
      assert(d.outerCount * d.innerCount == last.outCount)
      if last.innerCount * last.repeatCount == d.innerCount {
        _steps[_steps.count - 1] = RepeatDims(
          outerCount: d.outerCount, repeatCount: last.repeatCount * d.repeatCount,
          innerCount: last.innerCount)
        _axes[_axes.count - 1].append(contentsOf: axes)
      } else {
        _steps.append(d)
        _axes.append(axes)
      }
    }
  }

  public func lazify() -> (RepeatSequence, BroadcastStrides) {
    var s = _steps
    var a = _axes
    var result = BroadcastStrides(dataCount: outCount, outerRepeats: 1, innerRepeats: 1)
    if let first = s.first, first.innerCount == 1 {
      s.removeFirst()
      a.removeFirst()
      result = BroadcastStrides(
        dataCount: result.dataCount / first.repeatCount, outerRepeats: 1,
        innerRepeats: first.repeatCount)

      // We must modify all later repetitions to reflect the new
      // smaller inner dimension.
      for (i, x) in s.enumerated() {
        s[i] = RepeatDims(
          outerCount: x.outerCount,
          repeatCount: x.repeatCount,
          innerCount: x.innerCount / first.repeatCount
        )
      }
    }
    if let last = s.last, last.outerCount == 1 {
      s.removeLast()
      a.removeLast()
      result = BroadcastStrides(
        dataCount: result.dataCount / last.repeatCount, outerRepeats: last.repeatCount,
        innerRepeats: result.innerRepeats)
    }
    return (RepeatSequence(inCount: inCount, steps: s, axes: a), result)
  }

  public func inverse() -> [ReduceDims] {
    var result = [ReduceDims]()
    for step in steps.reversed() {
      result.append(
        ReduceDims(
          outerCount: step.outerCount, reduceCount: step.repeatCount, innerCount: step.innerCount))
    }
    return result
  }
}

extension Tensor {

  public func expand(as asTensor: Tensor) -> Tensor {
    expand(shape: asTensor.shape)
  }

  public func expand(shape newShape: [Int]) -> Tensor {
    let plan = planExpansion(shape: newShape)
    return flatApplyRepeats(plan.steps).reshape(newShape)
  }

  public static func broadcast(_ xs: [Tensor]) -> [Tensor] {
    let shape = Tensor.broadcastShape(xs.map { $0.shape })
    return xs.map { $0.expand(shape: shape) }
  }

  public static func broadcast(_ x: Tensor, _ y: Tensor) -> (Tensor, Tensor) {
    let results = broadcast([x, y])
    return (results[0], results[1])
  }

  internal static func lazyBroadcast(_ t: [Tensor]) -> ([Int], [(Tensor, BroadcastStrides)]) {
    let newShape = broadcastShape(t.map { $0.shape })
    let results = t.map { $0.lazyExpand(shape: newShape) }
    return (newShape, results)
  }

  internal static func lazyBroadcast(_ t1: Tensor, _ t2: Tensor) -> (
    [Int], ((Tensor, BroadcastStrides), (Tensor, BroadcastStrides))
  ) {
    let newShape = broadcastShape([t1.shape, t2.shape])
    let r1 = t1.lazyExpand(shape: newShape)
    let r2 = t2.lazyExpand(shape: newShape)
    return (newShape, (r1, r2))
  }

  internal func lazyExpand(shape newShape: [Int]) -> (Tensor, BroadcastStrides) {
    alwaysAssert(
      newShape.count >= shape.count,
      "cannot broadcast shape \(shape) to shorter shape \(newShape)"
    )

    let plan = planExpansion(shape: newShape)
    let (lazyPlan, bcast) = plan.lazify()
    let expanded = flatApplyRepeats(lazyPlan.steps)
    var lazyShape = shape
    for axis in lazyPlan.axes.flatMap({ $0 }) {
      lazyShape[axis] = newShape[axis + (newShape.count - shape.count)]
    }
    assert(lazyShape.product() == expanded.shape.product())
    return (expanded.reshape(lazyShape), bcast)
  }

  internal func reduceBroadcast(_ bcast: BroadcastStrides, as tensor: Tensor) -> Tensor {
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

  internal static func broadcastShape(_ s: [[Int]]) -> [Int] {
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
          fatalError("shapes \(s[0]) and \(s[1]) do not support broadcasting")
        } else if x == 1 {
          return y
        } else if y == 1 {
          return x
        } else {
          fatalError("shapes \(s[0]) and \(s[1]) do not support broadcasting")
        }
      }
    } else {
      return broadcastShape([broadcastShape([s[0], s[1]])] + Array(s[2...]))
    }
  }

}
