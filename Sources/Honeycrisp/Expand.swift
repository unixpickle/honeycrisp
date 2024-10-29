extension Tensor {

  public func expand(as asTensor: Tensor) -> Tensor {
    expand(shape: asTensor.shape)
  }

  public func expand(shape newShape: [Int]) -> Tensor {
    let (tail, repeats) = lazyExpand(shape: newShape)
    if repeats == 1 {
      return tail.reshape(newShape)
    }
    return tail.flatten().repeating(axis: 0, count: repeats).reshape(newShape)
  }

  public static func broadcast(_ xs: [Tensor]) -> [Tensor] {
    let shape = Tensor.broadcastShape(xs.map { $0.shape })
    return xs.map { $0.expand(shape: shape) }
  }

  public static func broadcast(_ x: Tensor, _ y: Tensor) -> (Tensor, Tensor) {
    let results = broadcast([x, y])
    return (results[0], results[1])
  }

  internal func lazyExpand(shape newShape: [Int]) -> (tail: Tensor, repeats: Int) {
    alwaysAssert(
      newShape.count >= shape.count,
      "cannot broadcast shape \(shape) to shorter shape \(newShape)"
    )

    let oneCount = shape.prefix { $0 == 1 }.count
    if oneCount > 0 {
      return reshape(Array(shape[oneCount...])).lazyExpand(shape: newShape)
    }

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
      let repeats = newShape[..<(newShape.count - shape.count)].product()
      return (tail: result, repeats: repeats)
    } else {
      return (tail: result, repeats: 1)
    }
  }

  internal static func lazyBroadcast(_ t: [Tensor]) -> (shape: [Int], tails: [Tensor]) {
    let newShape = broadcastShape(t.map { $0.shape })
    let results = t.map { $0.lazyExpand(shape: newShape) }
    return (shape: newShape, tails: results.map { $0.tail })
  }

  internal func reduceOuter(as asTensor: Tensor) -> Tensor {
    assert(shape.product() % asTensor.shape.product() == 0)
    let count = shape.product() / asTensor.shape.product()
    if count == 1 {
      return self.reshape(asTensor.shape)
    }
    return reshape([count] + [asTensor.shape.product()]).sum(axis: 0).reshape(asTensor.shape)
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
        } else if x < y {
          alwaysAssert(
            y % x == 0, "shapes \(s[0]) and \(s[1]) do not support broadcasting")
          return y
        } else {
          alwaysAssert(
            x % y == 0, "shapes \(s[0]) and \(s[1]) do not support broadcasting")
          return x
        }
      }
    } else {
      return broadcastShape([broadcastShape([s[0], s[1]])] + Array(s[2...]))
    }
  }

}
