public protocol TensorIndex {
  var minTensorSliceDims: Int { get }

  func tensorSliceIndices(forShape inShape: [Int]) -> ([Int], [Int])
}

extension Sequence where Element: Numeric {
  func product() -> Element {
    reduce(Element(exactly: 1)!, *)
  }

  func sum() -> Element {
    reduce(Element(exactly: 0)!, +)
  }
}

private func allIndices<S: Sequence>(forShape inShape: S) -> [Int] where S.Element == Int {
  Array(0..<inShape.product())
}

private func allIndices(forShape inShape: [Int]) -> [Int] {
  Array(0..<inShape.product())
}

extension Int: TensorIndex {
  public var minTensorSliceDims: Int { 1 }

  public func tensorSliceIndices(forShape inShape: [Int]) -> ([Int], [Int]) {
    assert(inShape.count > 0)
    let idx = self < 0 ? inShape[0] + self : self
    assert(idx >= 0 && idx < inShape[0], "index \(self) out of range for size \(inShape[0])")
    let innerIndices = allIndices(forShape: inShape[1...])
    let offset = idx * innerIndices.count
    return (Array(innerIndices.map({ $0 + offset })), Array(inShape[1...]))
  }
}

extension Range<Int>: TensorIndex {
  public var minTensorSliceDims: Int { 1 }

  public func tensorSliceIndices(forShape inShape: [Int]) -> ([Int], [Int]) {
    assert(inShape.count > 0)
    let start = self.lowerBound < 0 ? inShape[0] + self.lowerBound : self.lowerBound
    let end = self.upperBound < 0 ? inShape[0] + self.upperBound : self.upperBound
    assert(end >= start, "end (\(end)) must be >= start (\(start))")
    assert(
      start >= 0 && start <= inShape[0],
      "index \(self.lowerBound) out of range for size \(inShape[0])"
    )
    assert(
      end >= 0 && end <= inShape[0],
      "index \(self.upperBound) out of range for size \(inShape[0])"
    )
    var result: [Int] = Array()
    for i in start..<end {
      result += i.tensorSliceIndices(forShape: inShape).0
    }
    return (result, [end - start] + inShape[1...])
  }
}

extension ClosedRange<Int>: TensorIndex {
  public var minTensorSliceDims: Int { 1 }

  public func tensorSliceIndices(forShape inShape: [Int]) -> ([Int], [Int]) {
    assert(inShape.count > 0)
    let end = 1 + (self.upperBound < 0 ? inShape[0] + self.upperBound : self.upperBound)
    return (self.lowerBound..<end).tensorSliceIndices(forShape: inShape)
  }
}

public struct FullRange: TensorIndex {
  let dims: Int

  init(dims: Int = 1) {
    self.dims = dims
  }

  public var minTensorSliceDims: Int { dims }

  public func tensorSliceIndices(forShape inShape: [Int]) -> ([Int], [Int]) {
    return (allIndices(forShape: inShape), inShape)
  }
}

public struct NewAxis: TensorIndex {
  let count: Int

  init(count: Int = 1) {
    self.count = count
  }

  public var minTensorSliceDims: Int { 0 }

  public func tensorSliceIndices(forShape inShape: [Int]) -> ([Int], [Int]) {
    return (allIndices(forShape: inShape), Array(repeating: 1, count: count) + inShape)
  }
}

extension PartialRangeFrom<Int>: TensorIndex {
  public var minTensorSliceDims: Int { 1 }

  public func tensorSliceIndices(forShape inShape: [Int]) -> ([Int], [Int]) {
    assert(inShape.count > 0)
    return (self.lowerBound..<inShape[0]).tensorSliceIndices(forShape: inShape)
  }
}

extension PartialRangeUpTo<Int>: TensorIndex {
  public var minTensorSliceDims: Int { 1 }

  public func tensorSliceIndices(forShape inShape: [Int]) -> ([Int], [Int]) {
    assert(inShape.count > 0)
    let end = (self.upperBound > 0 ? self.upperBound : self.upperBound + inShape[0])
    assert(
      end >= 0 && end <= inShape[0],
      "index \(self.upperBound) out of range for size \(inShape[0])"
    )
    return (0..<end).tensorSliceIndices(forShape: inShape)
  }
}

extension PartialRangeThrough<Int>: TensorIndex {
  public var minTensorSliceDims: Int { 1 }

  public func tensorSliceIndices(forShape inShape: [Int]) -> ([Int], [Int]) {
    assert(inShape.count > 0)
    let end = (self.upperBound > 0 ? self.upperBound : self.upperBound + inShape[0])
    assert(
      end >= 0 && end < inShape[0],
      "index \(self.upperBound) out of range for size \(inShape[0])"
    )
    return (0...end).tensorSliceIndices(forShape: inShape)
  }
}

extension Array: TensorIndex where Element == any TensorIndex {
  public var minTensorSliceDims: Int { self.map({ $0.minTensorSliceDims }).sum() }

  public func tensorSliceIndices(forShape inShape: [Int]) -> ([Int], [Int]) {
    assert(inShape.count >= self.minTensorSliceDims)
    let a = Array(self)
    switch a.count {
    case 0:
      return (allIndices(forShape: inShape), inShape)
    case 1:
      return a[0].tensorSliceIndices(forShape: inShape)
    default:
      let currentShape: [Int] = [Int](inShape[..<a[0].minTensorSliceDims])
      let laterShape: [Int] = [Int](inShape[a[0].minTensorSliceDims...])
      let (subIndices, subShape) = Array(a[1...]).tensorSliceIndices(forShape: laterShape)
      let (firstIndices, firstShape) = a[0].tensorSliceIndices(forShape: currentShape)
      var result: [Int] = []
      let innerSize = laterShape.product()
      for firstIdx in firstIndices {
        let offset = firstIdx * innerSize
        result += subIndices.map({ $0 + offset })
      }
      return (result, firstShape + subShape)
    }
  }
}

extension Tensor {
  public func scatter(outShape: [Int], dstIndices: [Int]) -> Tensor {
    assert(dstIndices.count == data.count)
    var newData: [Float] = Array(repeating: 0, count: outShape.product())
    for (src, dstIdx) in zip(data, dstIndices) {
      newData[dstIdx] = src
    }
    if !needsGrad {
      return Tensor(data: newData, shape: outShape)
    } else {
      let handle = self.saveForBackward()
      return Tensor(data: newData, shape: outShape) { grad in
        handle.backward(grad: grad.gather(outShape: self.shape, srcIndices: dstIndices))
      }
    }
  }

  public func gather(outShape: [Int], srcIndices: [Int]) -> Tensor {
    var newData: [Float] = Array(repeating: 0, count: outShape.product())
    assert(
      srcIndices.count == newData.count,
      "expected indices count \(srcIndices.count) to equal to size of shape \(outShape)"
    )
    for (i, srcIdx) in srcIndices.enumerated() {
      newData[i] = data[srcIdx]
    }
    if !needsGrad {
      return Tensor(data: newData, shape: outShape)
    } else {
      let handle = self.saveForBackward()
      return Tensor(data: newData, shape: outShape) { grad in
        handle.backward(grad: grad.scatter(outShape: self.shape, dstIndices: srcIndices))
      }
    }
  }

  public subscript(index: any TensorIndex...) -> Tensor {
    let (srcIndices, outShape) = Array(index).tensorSliceIndices(forShape: shape)
    return self.gather(outShape: outShape, srcIndices: srcIndices)
  }

  // Support UnboundedRange up to the first few indices

  /* python code:
for num_args in range(1, 5):
    for pattern in range(0, 2 ** (num_args - 1)):
        args = []
        use_args = []
        for i in range(num_args):
            if not pattern & (1 << i):
                args.append("_: UnboundedRange")
                use_args.append("FullRange()")
            else:
                args.append(f"arg{i}: any TensorIndex")
                use_args.append(f"arg{i}")
        print(
            f"""
    subscript({', '.join(args)}, other: any TensorIndex...) -> Tensor {{
        let (srcIndices, outShape) = ([{', '.join(use_args)}] + other).tensorSliceIndices(forShape: shape)
        return self.gather(outShape: outShape, srcIndices: srcIndices)
    }}
                """.rstrip()
        )
    */

  public subscript(_: UnboundedRange, other: any TensorIndex...) -> Tensor {
    let (srcIndices, outShape) = ([FullRange()] + other).tensorSliceIndices(forShape: shape)
    return self.gather(outShape: outShape, srcIndices: srcIndices)
  }

  public subscript(_: UnboundedRange, _: UnboundedRange, other: any TensorIndex...) -> Tensor {
    let (srcIndices, outShape) = ([FullRange(), FullRange()] + other).tensorSliceIndices(
      forShape: shape)
    return self.gather(outShape: outShape, srcIndices: srcIndices)
  }

  public subscript(arg0: any TensorIndex, _: UnboundedRange, other: any TensorIndex...) -> Tensor {
    let (srcIndices, outShape) = ([arg0, FullRange()] + other).tensorSliceIndices(forShape: shape)
    return self.gather(outShape: outShape, srcIndices: srcIndices)
  }

  public subscript(_: UnboundedRange, _: UnboundedRange, _: UnboundedRange,
    other: any TensorIndex...
  ) -> Tensor {
    let (srcIndices, outShape) = ([FullRange(), FullRange(), FullRange()] + other)
      .tensorSliceIndices(forShape: shape)
    return self.gather(outShape: outShape, srcIndices: srcIndices)
  }

  public subscript(arg0: any TensorIndex, _: UnboundedRange, _: UnboundedRange,
    other: any TensorIndex...
  ) -> Tensor {
    let (srcIndices, outShape) = ([arg0, FullRange(), FullRange()] + other).tensorSliceIndices(
      forShape: shape)
    return self.gather(outShape: outShape, srcIndices: srcIndices)
  }

  public subscript(_: UnboundedRange, arg1: any TensorIndex, _: UnboundedRange,
    other: any TensorIndex...
  ) -> Tensor {
    let (srcIndices, outShape) = ([FullRange(), arg1, FullRange()] + other).tensorSliceIndices(
      forShape: shape)
    return self.gather(outShape: outShape, srcIndices: srcIndices)
  }

  public subscript(arg0: any TensorIndex, arg1: any TensorIndex, _: UnboundedRange,
    other: any TensorIndex...
  ) -> Tensor {
    let (srcIndices, outShape) = ([arg0, arg1, FullRange()] + other).tensorSliceIndices(
      forShape: shape)
    return self.gather(outShape: outShape, srcIndices: srcIndices)
  }

  public subscript(_: UnboundedRange, _: UnboundedRange, _: UnboundedRange, _: UnboundedRange,
    other: any TensorIndex...
  ) -> Tensor {
    let (srcIndices, outShape) = ([FullRange(), FullRange(), FullRange(), FullRange()] + other)
      .tensorSliceIndices(forShape: shape)
    return self.gather(outShape: outShape, srcIndices: srcIndices)
  }

  public subscript(arg0: any TensorIndex, _: UnboundedRange, _: UnboundedRange, _: UnboundedRange,
    other: any TensorIndex...
  ) -> Tensor {
    let (srcIndices, outShape) = ([arg0, FullRange(), FullRange(), FullRange()] + other)
      .tensorSliceIndices(forShape: shape)
    return self.gather(outShape: outShape, srcIndices: srcIndices)
  }

  public subscript(_: UnboundedRange, arg1: any TensorIndex, _: UnboundedRange, _: UnboundedRange,
    other: any TensorIndex...
  ) -> Tensor {
    let (srcIndices, outShape) = ([FullRange(), arg1, FullRange(), FullRange()] + other)
      .tensorSliceIndices(forShape: shape)
    return self.gather(outShape: outShape, srcIndices: srcIndices)
  }

  public subscript(arg0: any TensorIndex, arg1: any TensorIndex, _: UnboundedRange,
    _: UnboundedRange, other: any TensorIndex...
  ) -> Tensor {
    let (srcIndices, outShape) = ([arg0, arg1, FullRange(), FullRange()] + other)
      .tensorSliceIndices(forShape: shape)
    return self.gather(outShape: outShape, srcIndices: srcIndices)
  }

  public subscript(_: UnboundedRange, _: UnboundedRange, arg2: any TensorIndex, _: UnboundedRange,
    other: any TensorIndex...
  ) -> Tensor {
    let (srcIndices, outShape) = ([FullRange(), FullRange(), arg2, FullRange()] + other)
      .tensorSliceIndices(forShape: shape)
    return self.gather(outShape: outShape, srcIndices: srcIndices)
  }

  public subscript(arg0: any TensorIndex, _: UnboundedRange, arg2: any TensorIndex,
    _: UnboundedRange, other: any TensorIndex...
  ) -> Tensor {
    let (srcIndices, outShape) = ([arg0, FullRange(), arg2, FullRange()] + other)
      .tensorSliceIndices(forShape: shape)
    return self.gather(outShape: outShape, srcIndices: srcIndices)
  }

  public subscript(_: UnboundedRange, arg1: any TensorIndex, arg2: any TensorIndex,
    _: UnboundedRange, other: any TensorIndex...
  ) -> Tensor {
    let (srcIndices, outShape) = ([FullRange(), arg1, arg2, FullRange()] + other)
      .tensorSliceIndices(forShape: shape)
    return self.gather(outShape: outShape, srcIndices: srcIndices)
  }

  public subscript(arg0: any TensorIndex, arg1: any TensorIndex, arg2: any TensorIndex,
    _: UnboundedRange, other: any TensorIndex...
  ) -> Tensor {
    let (srcIndices, outShape) = ([arg0, arg1, arg2, FullRange()] + other).tensorSliceIndices(
      forShape: shape)
    return self.gather(outShape: outShape, srcIndices: srcIndices)
  }
}
