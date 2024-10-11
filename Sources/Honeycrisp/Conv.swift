public enum ConvConfigError: Error {
  case invalidGroups(String)
}

/// A size or coordinate in N-dimensional space.
///
///  * For 1D, this can be considered (length) or (x).
///  * For 2D, this can be treated as (height, width) or (y, x).
///  * For 3D, this can be (depth, height, width) or (z, y, x).
///
/// All multidimensional comparisons are only true if the comparison is true
/// for all sub-elements. For example, (x == y) is only true if all elements
/// of x equal corresponding ones in y. The same goes for (x != y), where it is
/// only true if no elements are equal. As a result, (x != y) may be different
/// than !(x == y).
public protocol SpatialDim: Hashable, Comparable {
  static var dimCount: Int { get }
  var dims: [Int] { get }
  init(constant: Int)
  init(dims: [Int])

  static func + (lhs: Self, rhs: Self) -> Self
  static func - (lhs: Self, rhs: Self) -> Self
  static func * (lhs: Self, rhs: Self) -> Self
  static func / (lhs: Self, rhs: Self) -> Self
  static func % (lhs: Self, rhs: Self) -> Self

  /// This implements the dot product.
  static func &* (lhs: Self, rhs: Self) -> Int

  /// Treat self as a size, and return the coordinates within a rectangle
  /// of this size, in first-major (e.g. row-major) order.
  func coordsInRect() -> [Self]
}

public struct SpatialDim1D: SpatialDim {
  public static var dimCount: Int { 1 }

  let x: Int

  public var dims: [Int] { [x] }

  public init(x: Int) {
    self.x = x
  }

  public init(constant: Int) {
    x = constant
  }

  public init(dims: [Int]) {
    assert(dims.count == 1)
    x = dims[0]
  }

  public static func + (lhs: Self, rhs: Self) -> Self {
    Self(x: lhs.x + rhs.x)
  }

  public static func - (lhs: Self, rhs: Self) -> Self {
    Self(x: lhs.x - rhs.x)
  }

  public static func * (lhs: Self, rhs: Self) -> Self {
    Self(x: lhs.x * rhs.x)
  }

  public static func / (lhs: Self, rhs: Self) -> Self {
    Self(x: lhs.x / rhs.x)
  }

  public static func % (lhs: Self, rhs: Self) -> Self {
    Self(x: lhs.x % rhs.x)
  }

  public static func &* (lhs: Self, rhs: Self) -> Int {
    return lhs.x * rhs.x
  }

  public static func == (lhs: Self, rhs: Self) -> Bool {
    lhs.x == rhs.x
  }

  public static func >= (lhs: Self, rhs: Self) -> Bool {
    lhs.x >= rhs.x
  }

  public static func <= (lhs: Self, rhs: Self) -> Bool {
    lhs.x <= rhs.x
  }

  public static func < (lhs: Self, rhs: Self) -> Bool {
    lhs.x < rhs.x
  }

  public static func > (lhs: Self, rhs: Self) -> Bool {
    lhs.x > rhs.x
  }

  public func coordsInRect() -> [Self] {
    (0..<x).map { Self(x: $0) }
  }
}

public struct SpatialDim2D: SpatialDim {
  public static var dimCount: Int { 2 }

  let x: Int
  let y: Int

  public var dims: [Int] { [y, x] }

  public init(x: Int, y: Int) {
    self.x = x
    self.y = y
  }

  public init(constant: Int) {
    x = constant
    y = constant
  }

  public init(dims: [Int]) {
    assert(dims.count == 2)
    y = dims[0]
    x = dims[1]
  }

  public static func + (lhs: Self, rhs: Self) -> Self {
    Self(x: lhs.x + rhs.x, y: lhs.y + rhs.y)
  }

  public static func - (lhs: Self, rhs: Self) -> Self {
    Self(x: lhs.x - rhs.x, y: lhs.y - rhs.y)
  }

  public static func * (lhs: Self, rhs: Self) -> Self {
    Self(x: lhs.x * rhs.x, y: lhs.y * rhs.y)
  }

  public static func / (lhs: Self, rhs: Self) -> Self {
    Self(x: lhs.x / rhs.x, y: lhs.y / rhs.y)
  }

  public static func % (lhs: Self, rhs: Self) -> Self {
    Self(x: lhs.x % rhs.x, y: lhs.y % rhs.y)
  }

  public static func &* (lhs: Self, rhs: Self) -> Int {
    return lhs.x * rhs.x + lhs.y * rhs.y
  }

  public static func == (lhs: Self, rhs: Self) -> Bool {
    lhs.x == rhs.x && lhs.y == rhs.y
  }

  public static func >= (lhs: Self, rhs: Self) -> Bool {
    lhs.x >= rhs.x && lhs.y >= rhs.y
  }

  public static func <= (lhs: Self, rhs: Self) -> Bool {
    lhs.x <= rhs.x && lhs.y <= rhs.y
  }

  public static func < (lhs: Self, rhs: Self) -> Bool {
    lhs.x < rhs.x && lhs.y < rhs.y
  }

  public static func > (lhs: Self, rhs: Self) -> Bool {
    lhs.x > rhs.x && lhs.y > rhs.y
  }

  public func coordsInRect() -> [Self] {
    (0..<y).flatMap { i in (0..<x).map { j in Self(x: j, y: i) } }
  }
}

public struct ConvConfig<Dim: SpatialDim>: Hashable {
  public typealias Dim = Dim

  struct LazyTensor<T: NumericTensorElement> {
    let shape: (Int, Int, Dim)
    let fn: (Int, Int, Dim) -> T

    static func strides(shape: (Int, Int, Dim), channelsLast: Bool) -> (Int, Int, Dim) {
      let dimMult = channelsLast ? shape.1 : 1
      let spatialStride = Array(
        (1...shape.2.dims.count).map({ shape.2.dims[$0...].product() * dimMult }))
      let dimStride = Dim(dims: spatialStride)
      let channelStride = channelsLast ? 1 : shape.2.dims.product()
      let batchStride = shape.1 * shape.2.dims.product()
      return (batchStride, channelStride, dimStride)
    }

    public init(shape: (Int, Int, Dim), _ fn: @escaping (Int, Int, Dim) -> T) {
      self.shape = shape
      self.fn = fn
    }

    public init<C: Collection<T>>(from arr: C, shape: [Int], channelsLast: Bool)
    where C.Index == Int {
      assert(arr.count == shape.product())
      let s =
        channelsLast
        ? (shape[0], shape[shape.count - 1], Dim(dims: Array(shape[1..<(shape.count - 1)])))
        : (shape[0], shape[1], Dim(dims: Array(shape[2...])))
      let (batchStride, channelStride, dimStride) = LazyTensor<T>.strides(
        shape: s, channelsLast: channelsLast)
      self.init(shape: s) { (i: Int, j: Int, k: Dim) in
        assert(i >= 0 && i < s.0)
        assert(j >= 0 && j < s.1)
        assert(k >= Dim(constant: 0) && k < s.2, "\(k) out of bounds with size \(s.2)")
        return arr[i * batchStride + j * channelStride + (k &* dimStride)]
      }
    }

    public func callAsFunction(_ i: Int, _ j: Int, _ k: Dim) -> T {
      fn(i, j, k)
    }

    public func unlazify<C: MutableCollection<T>>(to result: inout C, channelsLast: Bool)
    where C.Index == Int {
      let (batchStride, channelStride, dimStride) = LazyTensor<T>.strides(
        shape: shape, channelsLast: channelsLast)
      let coords = shape.2.coordsInRect()
      for batchIdx in 0..<shape.0 {
        for ch in 0..<shape.1 {
          for dim in coords {
            result[batchIdx * batchStride + ch * channelStride + (dim &* dimStride)] = fn(
              batchIdx, ch, dim)
          }
        }
      }
    }
  }

  public struct Padding: Hashable {
    public let before: Dim
    public let after: Dim
  }

  public let inChannels: Int
  public let outChannels: Int
  public let kernelSize: Dim
  public let imageSize: Dim
  public let stride: Dim
  public let dilation: Dim
  public let padding: Padding
  public let groups: Int
  public let channelsLast: Bool

  public var outSize: Dim {
    let one: Dim = Dim(constant: 1)
    // Parentheses seem to make this expression compile faster.
    return
      ((((imageSize + padding.before) + padding.after) - ((kernelSize - one) * dilation))
      + (stride - one)) / stride
  }

  public init(
    inChannels: Int,
    outChannels: Int,
    kernelSize: Dim,
    imageSize: Dim,
    stride: Dim,
    dilation: Dim,
    padding: Padding,
    groups: Int,
    channelsLast: Bool
  ) throws {
    if inChannels % groups != 0 {
      throw ConvConfigError.invalidGroups(
        "groups \(groups) must divide input channels \(inChannels)")
    } else if outChannels % groups != 0 {
      throw ConvConfigError.invalidGroups(
        "groups \(groups) must divide output channels \(outChannels)")
    }
    self.inChannels = inChannels
    self.outChannels = outChannels
    self.kernelSize = kernelSize
    self.imageSize = imageSize
    self.stride = stride
    self.dilation = dilation
    self.padding = padding
    self.groups = groups
    self.channelsLast = channelsLast
  }

  public static func samePadding(kernelSize: Dim, dilation: Dim = Dim(constant: 1))
    -> Padding
  {
    let effectiveKernel =
      kernelSize + (kernelSize - Dim(constant: 1)) * (dilation - Dim(constant: 1))
    return Padding(
      before: (effectiveKernel - Dim(constant: 1)) / Dim(constant: 2),
      after: effectiveKernel / Dim(constant: 2))
  }

  internal func imageShape(batch: Int) -> (Int, Int, Dim) {
    (batch, inChannels, imageSize)
  }

  internal func outShape(batch: Int) -> (Int, Int, Dim) {
    (batch, outChannels, outSize)
  }

  public func kernelTensorShape() -> [Int] {
    return [outChannels, inChannels / groups] + kernelSize.dims
  }

  public func imageTensorShape(batch: Int) -> [Int] {
    if channelsLast {
      [batch] + imageSize.dims + [inChannels]
    } else {
      [batch, inChannels] + imageSize.dims
    }
  }

  public func outputTensorShape(batch: Int) -> [Int] {
    if channelsLast {
      return [batch] + outSize.dims + [outChannels]
    } else {
      return [batch, outChannels] + outSize.dims
    }
  }

  func lazy<T: TensorElement, C: Collection<T>>(from: C, shape: [Int]) -> LazyTensor<T>
  where C.Index == Int {
    return LazyTensor(from: from, shape: shape, channelsLast: channelsLast)
  }

  func unlazify<T, C: MutableCollection<T>>(from: LazyTensor<T>, to: inout C) where C.Index == Int {
    from.unlazify(to: &to, channelsLast: channelsLast)
  }

  private func paddedInput<T>(_ rawInput: LazyTensor<T>) -> LazyTensor<T> {
    LazyTensor(
      shape: (rawInput.shape.0, rawInput.shape.1, rawInput.shape.2 + padding.before + padding.after)
    ) { b, c, x in
      let newX = x - padding.before
      if !(newX >= Dim(constant: 0)) || !(newX < imageSize) {
        return T(0.0)
      } else {
        return rawInput(b, c, newX)
      }
    }
  }

  func lazyForward<T: NumericTensorElement>(image: LazyTensor<T>, kernel: LazyTensor<T>)
    -> LazyTensor<T>
  {
    let paddedImage = paddedInput(image)
    let coords = kernelSize.coordsInRect()
    return LazyTensor(shape: outShape(batch: image.shape.0)) { (b: Int, c: Int, x: Dim) in
      let inGroupSize = inChannels / groups
      let outGroupSize = outChannels / groups
      let groupIdx = c / outGroupSize

      var sum = T(0.0)
      for kernelX in coords {
        let imageX = x * stride + kernelX * dilation
        for inC in 0..<inGroupSize {
          let imageVal = paddedImage(b, inC + inGroupSize * groupIdx, imageX)
          let kernelVal = kernel(c, inC, kernelX)
          sum = sum + imageVal * kernelVal
        }
      }
      return sum
    }
  }

  func lazyTranspose<T: NumericTensorElement>(image: LazyTensor<T>, kernel: LazyTensor<T>)
    -> LazyTensor<T>
  {
    let o = outShape(batch: image.shape.0)
    let coords = kernelSize.coordsInRect()
    let imgShape = imageShape(batch: image.shape.0)
    return LazyTensor(shape: imgShape) { (b: Int, c: Int, x: Dim) in
      let inGroupSize = inChannels / groups
      let outGroupSize = outChannels / groups
      let groupIdx = c / inGroupSize

      var sum = T(0.0)
      for kernelX in coords {
        let multOffset: Dim = (x + padding.before) - (kernelX * dilation)
        if !((multOffset % stride) == Dim(constant: 0)) {
          continue
        }
        let sourceX = multOffset / stride
        if !(sourceX >= Dim(constant: 0)) || !(sourceX < o.2) {
          continue
        }
        for sourceC in 0..<outGroupSize {
          let imageVal = image(b, sourceC + outGroupSize * groupIdx, sourceX)
          let kernelVal = kernel(
            sourceC + outGroupSize * groupIdx, c % inGroupSize, kernelX)
          sum = sum + imageVal * kernelVal
        }
      }
      return sum
    }
  }

  func lazyKernelGrad<T: NumericTensorElement>(image: LazyTensor<T>, outGrad: LazyTensor<T>)
    -> LazyTensor<T>
  {
    let paddedImage = paddedInput(image)

    let kernelShape = (outChannels, inChannels / groups, kernelSize)
    let outShape = outShape(batch: image.shape.0)
    let coords = outShape.2.coordsInRect()
    return LazyTensor(shape: kernelShape) { (cOut: Int, cIn: Int, kernelX: Dim) in
      let outGroupSize = outChannels / groups
      let inGroupSize = inChannels / groups
      let groupIdx = cOut / outGroupSize

      var sum = T(0.0)
      for batchIdx in 0..<image.shape.0 {
        for outX in coords {
          let imageX = outX * stride + kernelX * dilation
          sum =
            sum + paddedImage(batchIdx, cIn + groupIdx * inGroupSize, imageX)
            * outGrad(batchIdx, cOut, outX)
        }
      }
      return sum
    }
  }
}

public typealias Conv1DConfig = ConvConfig<SpatialDim1D>
public typealias Conv2DConfig = ConvConfig<SpatialDim2D>

extension Tensor {
  public static func conv1D(_ conv: Conv1DConfig, image: Tensor, kernel: Tensor) -> Tensor {
    convND(conv, image: image, kernel: kernel)
  }

  public static func conv1DTranspose(_ conv: Conv1DConfig, image: Tensor, kernel: Tensor) -> Tensor
  {
    convNDTranspose(conv, image: image, kernel: kernel)
  }

  public static func conv1DKernelGrad(_ conv: Conv1DConfig, image: Tensor, outGrad: Tensor)
    -> Tensor
  {
    convNDKernelGrad(conv, image: image, outGrad: outGrad)
  }

  public static func conv2D(_ conv: Conv2DConfig, image: Tensor, kernel: Tensor) -> Tensor {
    convND(conv, image: image, kernel: kernel)
  }

  public static func conv2DTranspose(_ conv: Conv2DConfig, image: Tensor, kernel: Tensor) -> Tensor
  {
    convNDTranspose(conv, image: image, kernel: kernel)
  }

  public static func conv2DKernelGrad(_ conv: Conv2DConfig, image: Tensor, outGrad: Tensor)
    -> Tensor
  {
    convNDKernelGrad(conv, image: image, outGrad: outGrad)
  }

  internal static func convND<Dim: SpatialDim>(
    _ conv: ConvConfig<Dim>, image: Tensor, kernel: Tensor
  ) -> Tensor {
    alwaysAssert(
      image.dtype == kernel.dtype,
      "image and kernel dtypes differ: \(image.dtype) vs \(kernel.dtype)")
    alwaysAssert(image.shape.count == 2 + Dim.dimCount, "invalid image shape: \(image.shape)")
    alwaysAssert(kernel.shape.count == 2 + Dim.dimCount, "invalid kernel shape: \(kernel.shape)")

    let expectedImageShape = conv.imageTensorShape(batch: image.shape[0])
    alwaysAssert(
      image.shape == expectedImageShape,
      "invalid image shape \(image.shape), expected \(expectedImageShape)")

    let outShape = conv.outputTensorShape(batch: image.shape[0])
    let kernelShape = conv.kernelTensorShape()
    alwaysAssert(kernel.shape == kernelShape)

    let backend = Backend.current
    let newData = Task {
      switch Dim.dimCount {
      case 1:
        try await backend.conv1D(
          conv as! Conv1DConfig, batch: image.shape[0], image: try await image.data,
          kernel: try await kernel.data,
          dtype: image.dtype)
      case 2:
        try await backend.conv2D(
          conv as! Conv2DConfig, batch: image.shape[0], image: try await image.data,
          kernel: try await kernel.data,
          dtype: image.dtype)
      default:
        fatalError()
      }
    }
    if !Tensor.isGradEnabled || (!image.needsGrad && !kernel.needsGrad) {
      return Tensor(dataTask: newData, shape: outShape, dtype: image.dtype)
    } else {
      let imageHandle = image.saveForBackward()
      let kernelHandle = kernel.saveForBackward()
      return Tensor(dataTask: newData, shape: outShape, dtype: image.dtype) { grad in
        imageHandle.backward(backend) {
          Tensor.convNDTranspose(conv, image: grad, kernel: kernel.noGrad())
        }
        kernelHandle.backward(backend) {
          Tensor.convNDKernelGrad(conv, image: image.noGrad(), outGrad: grad)
        }
      }
    }
  }

  internal static func convNDTranspose<Dim: SpatialDim>(
    _ conv: ConvConfig<Dim>, image: Tensor, kernel: Tensor
  ) -> Tensor {
    alwaysAssert(
      image.dtype == kernel.dtype,
      "image and kernel dtypes differ: \(image.dtype) vs \(kernel.dtype)")
    alwaysAssert(image.shape.count == 2 + Dim.dimCount, "invalid image shape: \(image.shape)")
    alwaysAssert(kernel.shape.count == 2 + Dim.dimCount, "invalid image shape: \(kernel.shape)")
    alwaysAssert(
      kernel.shape == conv.kernelTensorShape(),
      "invalid kernel shape \(kernel.shape) for conv \(conv)")

    let outShape: [Int]
    let inShape: [Int]
    inShape = conv.outputTensorShape(batch: image.shape[0])
    outShape = conv.imageTensorShape(batch: image.shape[0])
    alwaysAssert(
      image.shape == inShape,
      "invalid input shape for transposed conv2D: \(image.shape) (expected \(inShape))")

    let backend = Backend.current
    let newData = Task {
      switch Dim.dimCount {
      case 1:
        try await backend.conv1DTranspose(
          conv as! Conv1DConfig, batch: image.shape[0], image: try await image.data,
          kernel: try await kernel.data,
          dtype: image.dtype)
      case 2:
        try await backend.conv2DTranspose(
          conv as! Conv2DConfig, batch: image.shape[0], image: try await image.data,
          kernel: try await kernel.data,
          dtype: image.dtype)
      default:
        fatalError()
      }
    }
    if !Tensor.isGradEnabled || (!image.needsGrad && !kernel.needsGrad) {
      return Tensor(dataTask: newData, shape: outShape, dtype: image.dtype)
    } else {
      let imageHandle = image.saveForBackward()
      let kernelHandle = kernel.saveForBackward()
      return Tensor(dataTask: newData, shape: outShape, dtype: image.dtype) { grad in
        imageHandle.backward(backend) { convND(conv, image: grad, kernel: kernel.noGrad()) }
        kernelHandle.backward(backend) {
          convNDKernelGrad(conv, image: grad, outGrad: image.noGrad())
        }
      }
    }
  }

  internal static func convNDKernelGrad<Dim: SpatialDim>(
    _ conv: ConvConfig<Dim>, image: Tensor, outGrad: Tensor
  )
    -> Tensor
  {
    alwaysAssert(
      image.dtype == outGrad.dtype,
      "image and outGrad dtypes differ: \(image.dtype) vs \(outGrad.dtype)")
    alwaysAssert(image.shape.count == 2 + Dim.dimCount, "invalid image shape: \(image.shape)")
    alwaysAssert(outGrad.shape.count == 2 + Dim.dimCount, "invalid outGrad shape: \(outGrad.shape)")
    alwaysAssert(
      image.shape == conv.imageTensorShape(batch: image.shape[0]),
      "invalid image shape \(image.shape) for conv \(conv)")

    let kernelShape = conv.kernelTensorShape()
    let outShape = conv.outputTensorShape(batch: image.shape[0])
    alwaysAssert(
      outGrad.shape == outShape, "unexpected outGrad shape \(outGrad.shape), expected \(outShape)"
    )

    let backend = Backend.current
    let newData = Task {
      switch Dim.dimCount {
      case 1:
        try await backend.conv1DKernelGrad(
          conv as! Conv1DConfig, batch: image.shape[0], image: try await image.data,
          outGrad: try await outGrad.data, dtype: image.dtype)
      case 2:
        try await backend.conv2DKernelGrad(
          conv as! Conv2DConfig, batch: image.shape[0], image: try await image.data,
          outGrad: try await outGrad.data, dtype: image.dtype)
      default:
        fatalError()
      }
    }
    if !Tensor.isGradEnabled || (!image.needsGrad && !outGrad.needsGrad) {
      return Tensor(dataTask: newData, shape: kernelShape, dtype: image.dtype)
    } else {
      fatalError("convNDKernelGrad gradient is not yet implemented")
    }
  }
}

func exactDiv(_ x: Int, _ y: Int) -> Int? {
  if x % y != 0 {
    nil
  } else {
    x / y
  }
}
